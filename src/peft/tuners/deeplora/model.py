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

from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .layer import DeepLoraLinear


class DeepLoraModel(LoraModel):
    """
    Creates a Deep LoRA model from a pretrained model, representing the low-rank update of each targeted layer
    as a chain of `num_factors >= 2` low-rank factors. The method is described in
    https://huggingface.co/papers/2406.04112

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`DeepLoraConfig`]): The configuration of the Deep LoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Deep LoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import DeepLoraConfig, DeepLoraModel

        >>> config = DeepLoraConfig(
        ...     r=8,
        ...     num_factors=3,
        ...     target_modules=["q_proj", "v_proj"],
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model = DeepLoraModel(model, config, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`DeepLoraConfig`]): The configuration of the Deep LoRA model.
    """

    prefix: str = "deeplora_"
    tuner_layer_cls = DeepLoraLinear
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        # Deep LoRA stores the outer factors under LoRA's standard `lora_A`/`lora_B` names, which don't contain
        # the "deeplora_" prefix used by `BaseTuner._mark_only_adapters_as_trainable`; the "lora_" substring
        # covers those as well as `deeplora_factors`.
        for n, p in model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias.endswith("_only"):  # e.g. "lora_only"
                for m in model.modules():
                    if isinstance(m, self.tuner_layer_cls) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

        for module in model.modules():
            if isinstance(module, self.tuner_layer_cls):
                module._freeze_non_trainable_peft_weights()

    @staticmethod
    def _create_new_module(deeplora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, torch.nn.Linear):
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        new_module = DeepLoraLinear(target, adapter_name, config=deeplora_config, **kwargs)
        return new_module
