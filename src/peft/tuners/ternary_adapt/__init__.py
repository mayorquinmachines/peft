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

# `register_peft_method` resolves the method name against the `PeftType` enum, so `TERNARY_ADAPT` is defined as a
# static member of `peft.utils.peft_types.PeftType`. Do not inject enum members at runtime: poking
# `_member_map_`/`_member_names_` is not sufficient for plain attribute lookup (`getattr(PeftType, ...)`), which is
# what `register_peft_method` uses.
assert TERNARY_ADAPT_PEFT_TYPE == PeftType.TERNARY_ADAPT.value

register_peft_method(name="ternary_adapt", config_cls=TernaryAdaptConfig, model_cls=TernaryAdaptModel)
