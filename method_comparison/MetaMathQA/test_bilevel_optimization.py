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
Tests for the BiDoRA-style bi-level optimization (`bilevel_optimization.py`) and its wiring into the MetaMathQA
training harness (`utils.py`, `run.py`).
"""

import importlib
import importlib.util
import json
import os
import sys
import types

import pytest


try:
    import torch
    from torch import nn

    import peft.utils
    from peft import LoraConfig, PeftConfig, get_peft_model

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    IMPORT_ERROR = str(exc)

from bilevel_optimization import (
    BilevelPhase,
    BilevelUpdateScheduler,
    get_magnitude_and_direction_params,
    infinite_batches,
)


pytestmark = pytest.mark.skipif(IMPORT_ERROR is not None, reason=f"requires torch and peft: {IMPORT_ERROR}")


@pytest.fixture()
def utils_module(monkeypatch):
    """Import the existing `utils` module of the MetaMathQA harness (the call site of the bi-level wiring)."""
    # utils.py raises at import time when no CUDA/XPU accelerator is present; patch the device detection so that the
    # harness utilities can be imported on CPU-only machines as well.
    if peft.utils.infer_device() not in ("cuda", "xpu"):
        monkeypatch.setattr(peft.utils, "infer_device", lambda: "cuda")
    if "bitsandbytes" not in sys.modules and importlib.util.find_spec("bitsandbytes") is None:
        # utils.py imports bitsandbytes but the tests below don't exercise it
        sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")
    try:
        return importlib.import_module("utils")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"could not import utils.py from the MetaMathQA harness: {exc}")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(16, 16)
        self.lin2 = nn.Linear(16, 16)

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


def get_tiny_dora_model(*, use_dora):
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["lin1", "lin2"], use_dora=use_dora)
    return get_peft_model(TinyModel(), config)


def test_get_magnitude_and_direction_params():
    model = get_tiny_dora_model(use_dora=True)
    magnitude_params, direction_params = get_magnitude_and_direction_params(model)
    # one magnitude vector per targeted layer, plus the LoRA A and B weights of both layers as direction params
    assert len(magnitude_params) == 2
    assert len(direction_params) == 4


def test_get_magnitude_and_direction_params_requires_dora():
    model = get_tiny_dora_model(use_dora=False)
    with pytest.raises(ValueError, match="use_dora=True"):
        get_magnitude_and_direction_params(model)


def test_invalid_magnitude_every_n_steps():
    model = get_tiny_dora_model(use_dora=True)
    with pytest.raises(ValueError, match="magnitude_every_n_steps"):
        BilevelUpdateScheduler(model, magnitude_every_n_steps=1)


def test_phase_alternation():
    model = get_tiny_dora_model(use_dora=True)
    scheduler = BilevelUpdateScheduler(model, magnitude_every_n_steps=2)
    expected_phases = [BilevelPhase.DIRECTION, BilevelPhase.MAGNITUDE, BilevelPhase.DIRECTION, BilevelPhase.MAGNITUDE]
    for step, expected_phase in enumerate(expected_phases, start=1):
        phase = scheduler.set_phase_for_step(step)
        assert phase is expected_phase
        is_magnitude_phase = phase is BilevelPhase.MAGNITUDE
        assert all(param.requires_grad == is_magnitude_phase for param in scheduler.magnitude_params)
        assert all(param.requires_grad != is_magnitude_phase for param in scheduler.direction_params)


def test_gradients_only_flow_to_active_phase():
    model = get_tiny_dora_model(use_dora=True)
    scheduler = BilevelUpdateScheduler(model)
    inputs = torch.randn(3, 16)

    scheduler.set_phase_for_step(1)
    model(inputs).sum().backward()
    assert all(param.grad is None for param in scheduler.magnitude_params)
    assert all(param.grad is not None for param in scheduler.direction_params)

    model.zero_grad()
    scheduler.set_phase_for_step(2)
    model(inputs).sum().backward()
    assert all(param.grad is not None for param in scheduler.magnitude_params)
    assert all(param.grad is None for param in scheduler.direction_params)


def test_optimizer_step_only_updates_active_phase(utils_module):
    # exercises the "bidora" branch of get_optimizer_and_scheduler in utils.py the same way run.py does
    model = get_tiny_dora_model(use_dora=True)
    scheduler = BilevelUpdateScheduler(model)
    optimizer, _ = utils_module.get_optimizer_and_scheduler(
        model, optimizer_type="bidora", max_steps=10, lr_scheduler_arg=None, lr=1e-3
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    inputs = torch.randn(3, 16)

    # direction step: the magnitude vectors must not be updated
    magnitude_before = [param.detach().clone() for param in scheduler.magnitude_params]
    scheduler.set_phase_for_step(1)
    optimizer.zero_grad()
    model(inputs).sum().backward()
    optimizer.step()
    assert all(torch.equal(param, before) for param, before in zip(scheduler.magnitude_params, magnitude_before))

    # magnitude step: the direction weights must not be updated
    direction_before = [param.detach().clone() for param in scheduler.direction_params]
    scheduler.set_phase_for_step(2)
    optimizer.zero_grad()
    model(inputs).sum().backward()
    optimizer.step()
    assert all(torch.equal(param, before) for param, before in zip(scheduler.direction_params, direction_before))


def test_train_config_accepts_bidora(utils_module):
    with open(utils_module.FILE_NAME_DEFAULT_TRAIN_PARAMS) as f:
        config_kwargs = json.load(f)
    config_kwargs["optimizer_type"] = "bidora"
    train_config = utils_module.TrainConfig(**config_kwargs)
    assert train_config.optimizer_type == "bidora"


def test_bidora_experiment_config(utils_module):
    # the experiment mirrors the llama-3.2-3B-rank32-dora baseline but with bi-level optimization enabled
    experiment_path = os.path.join(os.path.dirname(__file__), "experiments", "lora", "llama-3.2-3B-rank32-bidora")
    peft_config = PeftConfig.from_pretrained(experiment_path)
    assert peft_config.use_dora
    train_config = utils_module.get_train_config(os.path.join(experiment_path, "training_params.json"))
    assert train_config.optimizer_type == "bidora"


def test_infinite_batches():
    def factory():
        return iter([{"batch": 1}, {"batch": 2}])

    batches = infinite_batches(factory)
    assert [next(batches)["batch"] for _ in range(5)] == [1, 2, 1, 2, 1]
