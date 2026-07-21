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

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer

from .layer import MosLoraLinear


class MosLoraModel(LoraModel):
    """
    Creates a Mixture-of-Subspaces LoRA (MoSLoRA) model from a pretrained model.

    The method is described in detail in https://huggingface.co/papers/2406.11909. It behaves exactly like
    [`LoraModel`], except that each targeted linear layer gets an additional learnable `r x r` mixer matrix between
    the LoRA `A` and `B` matrices, i.e. the update is `Delta W = B @ M @ A` (see [`MosLoraLinear`]).

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`MosLoraConfig`]): The configuration of the MoSLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Example:

        ```py
        >>> import torch
        >>> from peft import get_peft_model
        >>> from peft.tuners.moslora import MosLoraConfig

        >>> base_model = torch.nn.Sequential(torch.nn.Linear(10, 100), torch.nn.ReLU(), torch.nn.Linear(100, 2))
        >>> config = MosLoraConfig(r=8, lora_alpha=16, target_modules=["0", "2"])
        >>> model = get_peft_model(base_model, config)
        ```
    """

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # MoSLoRA currently supports unquantized `torch.nn.Linear` and transformers `Conv1D` targets. Quantized
        # layers, embeddings and convolutions are not supported (yet).
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        new_module = None
        if kwargs.get("parameter_name") is not None:
            raise ValueError("MoSLoRA does not support `target_parameters`, please use `target_modules` instead.")
        if isinstance(target_base_layer, torch.nn.Linear):
            if lora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                lora_config.fan_in_fan_out = False
            new_module = MosLoraLinear(target, adapter_name, config=lora_config, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            if not lora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                lora_config.fan_in_fan_out = True
            new_module = MosLoraLinear(
                target, adapter_name, is_target_conv_1d_layer=True, config=lora_config, **kwargs
            )

        if new_module is None:
            supported = (torch.nn.Linear, Conv1D)
            raise ValueError(
                f"Target module {target} is not supported by MoSLoRA. "
                f"Currently, only the following module types are supported: {supported}."
            )
        return new_module
