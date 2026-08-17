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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SuperTuningConfig(PeftConfig):
    """
    Configuration for Super-Tuning (Super), a sparse PEFT method with a fixed trainable support.

    Super selects a small support (a sparse subset of entries) of each targeted weight matrix using a pruning-style
    saliency score and trains a sparse delta restricted to that support while the base weights stay frozen. This is
    an implementation of "Super" from the paper "Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning"
    (https://arxiv.org/abs/2607.09287).

    Args:
        density (`float`, *optional*, defaults to `0.05`):
            Fraction of weight entries included in the trainable support of each targeted matrix. Must be in (0, 1].
        saliency (`str`, *optional*, defaults to `"magnitude"`):
            Scoring rule used to order weight entries for support selection. `"magnitude"` uses `|W|` (PaFi-style,
            training-free). `"wanda"` uses the Wanda-style activation-weighted score `|W| * ||X_j||_2`; with this
            setting, call `SuperTuningModel.calibrate(...)` on a calibration batch before training to refine the
            support (until then, the magnitude score is used as a fallback).
        select_lowest (`bool`, *optional*, defaults to `False`):
            If `True`, select the *lowest*-scoring entries for the support instead of the highest-scoring ones. The
            paper found that low-score supports can be effective under both orderings.
        target_modules (`Union[list[str], str]`, *optional*):
            The names of the modules to apply Super-Tuning to. Can be a list of module names or `"all-linear"`. If
            `None`, defaults to all linear layers.
    """

    density: float = field(
        default=0.05,
        metadata={
            "help": (
                "Fraction of weight entries included in the trainable support of each targeted matrix. "
                "Must be in (0, 1]."
            )
        },
    )
    saliency: str = field(
        default="magnitude",
        metadata={
            "help": (
                "Scoring rule for support selection: 'magnitude' uses |W| (training-free), 'wanda' uses the "
                "activation-weighted score |W| * ||X_j||_2 and requires calling `calibrate(...)` on the tuner model "
                "with a calibration batch before training."
            )
        },
    )
    select_lowest: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, select the lowest-scoring entries for the support instead of the highest-scoring ones. "
                "Low-score supports can be effective under both orderings."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The names of the modules to apply Super-Tuning to. Can be a list of module names or 'all-linear'."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SUPER_TUNING

        if not 0 < self.density <= 1:
            raise ValueError(f"`density` must be a float in (0, 1], got {self.density}.")
        if self.saliency not in ("magnitude", "wanda"):
            raise ValueError(f"`saliency` must be 'magnitude' or 'wanda', got {self.saliency!r}.")
