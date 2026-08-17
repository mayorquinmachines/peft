# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND

from .layer import SuperTuningLayer, dispatch_default


class SuperTuningModel(BaseTuner):
    """A tuner implementing Super-Tuning: sparse fine-tuning on a fixed, saliency-selected support."""

    prefix: str = "super_tuning_"
    tuner_layer_cls = SuperTuningLayer

    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        # Pass state_dict through for compatibility with BaseTuner
        super().__init__(
            model,
            config,
            adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped base model.

        This mirrors the behavior of other tuners (e.g., LoRA), ensuring attributes like `device` resolve to the
        underlying transformers model.
        """
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # avoid infinite recursion during init
                raise
            return getattr(self.model, name)

    def _prepare_adapter_config(self, peft_config, model_config):
        # If target_modules is unspecified, fall back to all linear layers
        if peft_config.target_modules is None:
            peft_config.target_modules = INCLUDE_LINEAR_LAYERS_SHORTHAND
        return peft_config

    def _create_and_replace(
        self,
        config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: str | None = None,
    ) -> None:
        # Super-Tuning only works on 2D weight matrices
        if not hasattr(target, "weight") or len(target.weight.shape) != 2:
            return

        if isinstance(target, SuperTuningLayer):
            target.update_layer(adapter_name, config=config)
        else:
            new_module = dispatch_default(target, adapter_name, config=config)
            if new_module is None:
                return
            # If adding an additional adapter, keep it frozen initially
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            # Only the sparse delta parameters should be trainable
            if "super_tuning_delta" not in n:
                p.requires_grad = False

    @torch.no_grad()
    def calibrate(self, inputs, adapter_name: str | None = None) -> None:
        """
        Refine the sparse supports with Wanda-style activation-weighted saliency.

        Runs the wrapped model on the given calibration input(s) while recording, per targeted linear layer, the L2
        norm of the activations of each input feature. Each adapter's support is then recomputed with the score
        `|W| * ||X_j||_2`. This realizes the "Super" support selection of the Super-Tuning paper; call it once before
        training when the config uses `saliency="wanda"`.

        Args:
            inputs:
                A calibration batch (a tensor passed positionally to the model, or a dict of keyword arguments), or a
                list of such batches.
            adapter_name (`str`, *optional*):
                The adapter whose support should be recalibrated. Defaults to the active adapter.
        """
        batches = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

        stats = {}
        handles = []

        def make_hook(name):
            def hook(module, args):
                x = args[0]
                if not isinstance(x, torch.Tensor):
                    return
                flat = x.detach().float().reshape(-1, x.shape[-1])
                if name in stats:
                    stats[name] += flat.pow(2).sum(0)
                else:
                    stats[name] = flat.pow(2).sum(0)

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, SuperTuningLayer):
                handles.append(module.register_forward_pre_hook(make_hook(name)))
        try:
            for batch in batches:
                if isinstance(batch, dict):
                    self.model(**batch)
                else:
                    self.model(batch)
        finally:
            for handle in handles:
                handle.remove()

        if not stats:
            raise ValueError("Calibration did not reach any Super-Tuning layer; check the calibration inputs.")

        for name, module in self.model.named_modules():
            if isinstance(module, SuperTuningLayer) and name in stats:
                if adapter_name is None:
                    adapter_name = module.active_adapters[0]
                module.update_support(adapter_name, activation_norms=stats[name].sqrt())
