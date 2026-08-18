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

from peft.utils import register_peft_method

from .config import RLoraConfig
from .layer import RLoraLinear, RLoraLinearVariant
from .model import RLoraModel


__all__ = ["RLoraConfig", "RLoraLinear", "RLoraLinearVariant", "RLoraModel"]

# The prefix is shared with LoRA ("lora_") on purpose: an R-LoRA adapter stores standard `lora_A`/`lora_B` weights, so
# the LoRA state dict handling applies.
register_peft_method(name="rlora", config_cls=RLoraConfig, model_cls=RLoraModel, prefix="lora_")
