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

"""Condition-number based selection of LoRA target modules (spectral targeting).

This module implements the targeting rule of κ-LoRA ("κ-LoRA: Condition Numbers Reveal Which LoRA Matrices Worth
Updating", https://arxiv.org/abs/2607.22489): the candidate target modules are ranked by the condition number of
their base weight matrix (the ratio of the largest to the smallest singular value) and only the top fraction is
adapted. The paper shows that matrices with large condition numbers contain underdeveloped directions that drive
most of the adaptation gains, whereas well-balanced matrices (small condition number) contribute only marginally.
Restricting LoRA updates to the top ~50% of weight matrices therefore halves the trainable parameter count while
matching the accuracy of standard LoRA.
"""

import math
from typing import Optional

import torch
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import check_target_module_exists


def weight_condition_number(weight: torch.Tensor) -> Optional[float]:
    """Compute the condition number (largest / smallest singular value) of a 2D weight matrix.

    Returns `None` for weights that cannot be ranked (non-2D, non-floating point, or meta-device tensors) so that
    callers can skip them. Rank-deficient matrices have an infinite condition number, which is consistent with the
    κ-LoRA framing: they contain the most underdeveloped directions.
    """
    if weight.ndim != 2 or not weight.is_floating_point() or weight.device.type == "meta":
        return None
    singular_values = torch.linalg.svdvals(weight.detach().to(torch.float32))
    if singular_values[-1].item() == 0.0:
        return math.inf
    return (singular_values[0] / singular_values[-1]).item()


def compute_target_condition_numbers(model: nn.Module, config: PeftConfig) -> dict[str, float]:
    """Compute the condition numbers of all modules of `model` matched by `config.target_modules`.

    Returns a mapping from full module name to condition number. Matched modules whose weight cannot be ranked
    (e.g. quantized or non-2D weights) are omitted.
    """
    condition_numbers = {}
    for key, module in model.named_modules():
        if not key or not check_target_module_exists(config, key):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        condition_number = weight_condition_number(weight)
        if condition_number is not None:
            condition_numbers[key] = condition_number
    return condition_numbers


def select_top_condition_number_modules(model: nn.Module, config: PeftConfig, top_fraction: float) -> set[str]:
    """Select the matched modules whose weight matrices have the largest condition numbers.

    Args:
        model (`nn.Module`):
            The base model whose modules are inspected.
        config (`PeftConfig`):
            A PEFT config with `target_modules` resolved, e.g. a [`LoraConfig`].
        top_fraction (`float`):
            Fraction in the interval (0, 1] of the matched modules to keep, ranked by condition number.

    Returns:
        `set[str]`: The full names of the selected modules.
    """
    condition_numbers = compute_target_condition_numbers(model, config)
    if not condition_numbers:
        raise ValueError(
            "`condition_number_top_fraction` was set but no targeted modules with a rankable 2D floating point "
            "weight matrix were found. Check `target_modules` and make sure the model is not on the meta device."
        )
    num_selected = max(1, math.ceil(len(condition_numbers) * top_fraction))
    ranked_keys = sorted(condition_numbers, key=lambda key: (-condition_numbers[key], key))
    return set(ranked_keys[:num_selected])
