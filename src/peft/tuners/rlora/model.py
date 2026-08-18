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

from peft.tuners.lora import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer

from .layer import RLoraLinear


class RLoraModel(LoraModel):
    """
    Creates a R-LoRA (Randomized Multi-Head LoRA) model from a pretrained model.

    R-LoRA (https://arxiv.org/abs/2502.15455) keeps the LoRA down-projection shared and learns `num_heads` distinct
    up-projection head matrices, diversified through Multi-Head Dropout on the shared intermediate representation and
    Multi-Head Random Initialization. See [`RLoraConfig`] for the available options.

    Since R-LoRA reuses the LoRA machinery, all LoRA model features (merging, unloading, multiple adapters, ...) are
    available. Only `torch.nn.Linear` and `transformers.pytorch_utils.Conv1D` target modules are supported.

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

    # Note: don't redefine prefix or tuner_layer_cls here, it should be inherited from LoraModel

    @staticmethod
    def _create_new_module(rlora_config, adapter_name, target, **kwargs):
        parameter_name = kwargs.pop("parameter_name", None)
        if parameter_name is not None:
            raise TypeError("RLoraModel does not support `target_parameters` yet.")
        if kwargs.get("loaded_in_8bit") or kwargs.get("loaded_in_4bit"):
            raise TypeError("RLoraModel does not support quantized base layers yet.")

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if rlora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                rlora_config.fan_in_fan_out = False
            new_module = RLoraLinear(target, adapter_name, config=rlora_config, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            if not rlora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                rlora_config.fan_in_fan_out = True
            new_module = RLoraLinear(target, adapter_name, is_target_conv_1d_layer=True, config=rlora_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` and "
                "`transformers.pytorch_utils.Conv1D` are supported."
            )

        return new_module
