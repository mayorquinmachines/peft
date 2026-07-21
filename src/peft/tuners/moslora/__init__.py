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

from peft.utils import PeftType, register_peft_method

from .config import MosLoraConfig
from .layer import MosLoraLinear
from .model import MosLoraModel


__all__ = ["MosLoraConfig", "MosLoraLinear", "MosLoraModel"]


def _ensure_moslora_peft_type() -> None:
    # `register_peft_method` requires a matching entry in the `PeftType` enum. New built-in tuners add that entry
    # directly in `peft/utils/peft_types.py`; since this package is self-contained, the entry is added dynamically
    # here instead. Once MoSLoRA is moved into the core tuner registry, this helper can be replaced by a regular
    # `MOSLORA = "MOSLORA"` entry in `peft_types.py`.
    if "MOSLORA" in PeftType.__members__:
        return
    member = str.__new__(PeftType, "MOSLORA")
    member._name_ = "MOSLORA"
    member._value_ = "MOSLORA"
    PeftType._member_map_["MOSLORA"] = member
    PeftType._value2member_map_["MOSLORA"] = member
    PeftType._member_names_.append("MOSLORA")


_ensure_moslora_peft_type()

# The prefix stays "lora_": MoSLoRA adapters are a superset of the LoRA state dict format (the mixer weights are
# stored under "lora_mixer"), which keeps adapter saving/loading working through the generic prefix-based path.
register_peft_method(name="moslora", config_cls=MosLoraConfig, model_cls=MosLoraModel, prefix="lora_")
