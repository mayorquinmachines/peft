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

from .config import TERNARY_ADAPT_PEFT_TYPE, TernaryAdaptConfig
from .layer import TernaryAdaptLayer, TernaryAdaptLinear
from .model import TernaryAdaptModel


__all__ = ["TernaryAdaptConfig", "TernaryAdaptLayer", "TernaryAdaptLinear", "TernaryAdaptModel"]

# `register_peft_method` requires a `PeftType` enum entry, which lives in `peft/utils/peft_types.py`. To keep this
# change confined to new files, the enum member is added dynamically here; since `PeftType` is a `str` enum, the
# dynamically added member is indistinguishable from a statically defined one in all mapping lookups, comparisons,
# and serialization. A proper static entry in `peft.utils.peft_types.PeftType` should replace this in a follow-up.
if TERNARY_ADAPT_PEFT_TYPE not in PeftType._value2member_map_:
    _member = str.__new__(PeftType, TERNARY_ADAPT_PEFT_TYPE)
    _member._name_ = TERNARY_ADAPT_PEFT_TYPE
    _member._value_ = TERNARY_ADAPT_PEFT_TYPE
    PeftType._value2member_map_[TERNARY_ADAPT_PEFT_TYPE] = _member
    PeftType._member_map_[TERNARY_ADAPT_PEFT_TYPE] = _member
    PeftType._member_names_.append(TERNARY_ADAPT_PEFT_TYPE)

register_peft_method(name="ternary_adapt", config_cls=TernaryAdaptConfig, model_cls=TernaryAdaptModel)
