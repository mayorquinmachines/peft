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
from peft.utils import (
    TRANSFORMERS_MODELS_TO_RLORA_TARGET_MODULES_MAPPING,
)

from .layer import RLoraLayer, RLoraLinear


class RLoraModel(BaseTuner):
    """
    Creates a Randomized Multi-Head LoRA (R-LoRA) model from a pretrained model. The method is described in
    https://huggingface.co/papers/2502.15455

    R-LoRA splits the LoRA update into a shared down-projection A and multiple head matrices B_i that are combined
    by an input-dependent router, and applies multi-head randomization (random mean-centered head initialization and
    multi-head dropout) to diversify the heads for multi-task learning.

    Note that merging the adapter into the base weights is not supported, since the router makes the update
    input-dependent.

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`RLoraConfig`]): The configuration of the R-LoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The R-LoRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import RLoraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = RLoraConfig(r=8, rlora_num_heads=4, rlora_head_dropout=0.2, target_modules=["q_proj", "v_proj"])
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`RLoraConfig`]): The configuration of the R-LoRA model.
    """

    prefix: str = "rlora_"
    tuner_layer_cls = RLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_RLORA_TARGET_MODULES_MAPPING

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

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": rlora_config.r,
            "bias": bias,
        }

        # If it is not an RLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, RLoraLayer):
            new_module = self._create_new_module(rlora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                config=rlora_config,
                **kwargs,
            )

    @staticmethod
    def _create_new_module(rlora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, torch.nn.Linear):
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        new_module = RLoraLinear(target, adapter_name, config=rlora_config, **kwargs)
        return new_module
