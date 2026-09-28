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

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer

from .layer import RLoraLayer, RLoraLinear


class RLoraModel(BaseTuner):
    """
    Creates an R-LoRA (Randomized Multi-Head LoRA) model from a pretrained model.

    R-LoRA splits LoRA's up-projection into `num_heads` head matrices `B_i` that share a single down-projection
    `A`, combines them with a learned router `omega = softmax(W_r x)`, and randomizes the heads via Multi-Head
    Dropout (per-head masking of the low-rank intermediate during training) and Multi-Head Random Initialization
    (non-zero init of `A` and the heads, with the initial delta subtracted from the base weight). See
    [`RLoraConfig`] for the available options and the paper reference.

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`RLoraConfig`]): The configuration of the R-LoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The R-LoRA model.

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`RLoraConfig`]): The configuration of the R-LoRA model.
    """

    prefix: str = "rlora_"
    tuner_layer_cls = RLoraLayer
    # no default target modules per architecture; `target_modules` must be specified in the config
    target_module_mapping: dict[str, list[str]] = {}

    def _create_and_replace(
        self,
        rlora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # If it is not an RLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, RLoraLayer):
            new_module = self._create_new_module(rlora_config, adapter_name, target, r=rlora_config.r)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                r=rlora_config.r,
                config=rlora_config,
            )

    @staticmethod
    def _create_new_module(rlora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = RLoraLinear(target, adapter_name, config=rlora_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module
