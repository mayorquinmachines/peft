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
from peft.utils import TRANSFORMERS_MODELS_TO_MIXLORA_TARGET_MODULES_MAPPING

from .layer import MixLoraLayer, MixLoraLinear


class MixLoraModel(BaseTuner):
    """
    Creates a MixLoRA (LoRA-based Mixture of Experts) model from a pretrained model.

    MixLoRA (https://arxiv.org/abs/2404.15159) places a small mixture of LoRA experts on each targeted (feed-forward)
    linear layer. A per-layer router, trained jointly with the experts, routes every token to its `top_k` highest
    scoring experts and combines their outputs with the re-normalized router probabilities. See [`MixLoraConfig`] for
    the available options.

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`MixLoraConfig`]): The configuration of the MixLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The MixLoRA model.

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`MixLoraConfig`]): The configuration of the MixLoRA model.
    """

    prefix: str = "mixlora_"
    tuner_layer_cls = MixLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_MIXLORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        mixlora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # If it is not a MixLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, MixLoraLayer):
            new_module = self._create_new_module(mixlora_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                config=mixlora_config,
            )

    @staticmethod
    def _create_new_module(mixlora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = MixLoraLinear(target, adapter_name, config=mixlora_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module

    def get_router_aux_loss(self) -> torch.Tensor:
        """
        Return the load-balancing auxiliary loss recorded during the most recent training forward pass, summed over
        all MixLoRA layers and the active adapters. Add it to the task loss to encourage balanced expert usage.

        The loss is only recorded when the model is in training mode and `router_aux_loss_coef > 0`.
        """
        aux_losses = [
            module._router_aux_loss
            for module in self.model.modules()
            if isinstance(module, MixLoraLayer) and module._router_aux_loss is not None
        ]
        if not aux_losses:
            raise ValueError(
                "No router auxiliary loss was recorded. Make sure the model is in training mode, that at least one "
                "forward pass has run, and that `router_aux_loss_coef > 0`."
            )
        return torch.stack(aux_losses).sum()
