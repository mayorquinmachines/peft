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

"""
Bi-level magnitude/direction optimization for DoRA adapters.

Adapted from "BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation"
(https://arxiv.org/abs/2410.09758).

DoRA decomposes each adapted weight matrix into a magnitude vector m and a direction component V = W0 + BA, and
optimizes both simultaneously. BiDoRA argues that this joint optimization is over-expressive and prone to overfitting,
and instead splits the optimization into a bi-level scheme:

- lower level: the direction parameters (the LoRA weights A and B) are optimized on the training data while the
  magnitude vectors are kept fixed.
- upper level: the magnitude vectors are optimized on held-out validation data while the direction is kept fixed.

This module implements that alternating scheme for the training loop in `run.py`. It does not require a dedicated
optimizer: the phase is encoded by toggling `requires_grad` on the respective parameter group, so frozen parameters
receive no gradients and are skipped by the regular optimizer step.
"""

import enum
from collections.abc import Callable, Iterator
from typing import Any

from torch import nn


# name of the DoRA magnitude vector parameter inside peft's LoraLayer (active when LoraConfig.use_dora=True)
MAGNITUDE_VECTOR_MARKER = "lora_magnitude_vector"
# BiDoRA performs one lower-level (direction) update followed by one upper-level (magnitude) update per bi-level
# iteration, i.e. a 1:1 ratio, which corresponds to a magnitude update on every 2nd step of the training loop.
MAGNITUDE_UPDATE_EVERY_N_STEPS = 2


class BilevelPhase(enum.Enum):
    DIRECTION = "direction"
    MAGNITUDE = "magnitude"


def get_magnitude_and_direction_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split the trainable parameters of the model into DoRA magnitude vectors and direction parameters.

    Args:
        model: The model with a DoRA adapter (i.e. a `LoraConfig` with `use_dora=True`) applied.

    Returns:
        A tuple of two lists: the magnitude vector parameters and all other (direction) parameters.

    Raises:
        ValueError: If no DoRA magnitude vectors are found among the trainable parameters.
    """
    magnitude_params = []
    direction_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if MAGNITUDE_VECTOR_MARKER in name:
            magnitude_params.append(param)
        else:
            direction_params.append(param)
    if not magnitude_params:
        raise ValueError(
            "Bi-level optimization requires a DoRA adapter (LoraConfig with use_dora=True), but no parameter "
            f"containing '{MAGNITUDE_VECTOR_MARKER}' was found among the trainable parameters."
        )
    return magnitude_params, direction_params


def _set_requires_grad(params: list[nn.Parameter], value: bool) -> None:
    for param in params:
        param.requires_grad_(value)


class BilevelUpdateScheduler:
    """
    Alternates which component of a DoRA adapter is optimized on a given training step.

    In the direction phase (lower level), the LoRA direction parameters are trainable and the magnitude vectors are
    frozen; the step should be taken on a training batch. In the magnitude phase (upper level), only the magnitude
    vectors are trainable; the step should be taken on a validation batch. The optimizer itself is untouched, as
    frozen parameters receive no gradients and are thus skipped during the optimizer step.

    Args:
        model: The model with a DoRA adapter applied.
        magnitude_every_n_steps: Every n-th step (1-indexed) is a magnitude step, all other steps are direction
            steps. Must be >= 2 so that both levels are optimized.
    """

    def __init__(self, model: nn.Module, *, magnitude_every_n_steps: int = MAGNITUDE_UPDATE_EVERY_N_STEPS) -> None:
        if magnitude_every_n_steps < 2:
            raise ValueError(f"magnitude_every_n_steps must be >= 2, got {magnitude_every_n_steps}")
        self.magnitude_every_n_steps = magnitude_every_n_steps
        self.magnitude_params, self.direction_params = get_magnitude_and_direction_params(model)
        self.phase = None

    def set_phase_for_step(self, step: int) -> BilevelPhase:
        """
        Activate the parameter group for the given 1-indexed training step and return the active phase.

        The caller is expected to take the subsequent optimizer step on a training batch when the returned phase is
        `BilevelPhase.DIRECTION` and on a validation batch when it is `BilevelPhase.MAGNITUDE`.
        """
        if step % self.magnitude_every_n_steps == 0:
            phase = BilevelPhase.MAGNITUDE
        else:
            phase = BilevelPhase.DIRECTION
        _set_requires_grad(self.magnitude_params, phase is BilevelPhase.MAGNITUDE)
        _set_requires_grad(self.direction_params, phase is BilevelPhase.DIRECTION)
        self.phase = phase
        return phase


def infinite_batches(batch_iterator_factory: Callable[[], Iterator[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """
    Endlessly yield batches by re-creating the given batch iterator whenever it is exhausted.

    Used for the upper-level (magnitude) steps, which draw from the much smaller validation set and thus cycle
    through it multiple times over the course of training.
    """
    while True:
        yield from batch_iterator_factory()
