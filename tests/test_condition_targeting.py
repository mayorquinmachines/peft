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

import math

import pytest
import torch
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.spectral_targeting import compute_target_condition_numbers, weight_condition_number


TARGETS = ["lin0", "lin1", "lin2", "lin3"]


class TinyModel(nn.Module):
    """A tiny MLP whose linear layers have distinct, well-controlled condition numbers (1, 10, 100, 1000)."""

    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(8, 8, bias=False)
        self.lin1 = nn.Linear(8, 8, bias=False)
        self.lin2 = nn.Linear(8, 8, bias=False)
        self.lin3 = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.lin0.weight.copy_(torch.eye(8))
            self.lin1.weight.copy_(torch.diag(torch.linspace(1, 10, 8)))
            self.lin2.weight.copy_(torch.diag(torch.linspace(1, 100, 8)))
            self.lin3.weight.copy_(torch.diag(torch.linspace(1, 1000, 8)))

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.lin0(x))))


def test_weight_condition_number():
    assert weight_condition_number(torch.eye(8)) == pytest.approx(1.0)
    assert weight_condition_number(torch.diag(torch.tensor([1.0, 100.0]))) == pytest.approx(100.0)
    # rank-deficient matrices have an infinite condition number
    assert weight_condition_number(torch.zeros(4, 4)) == math.inf
    # non-2D weights cannot be ranked
    assert weight_condition_number(torch.randn(4, 4, 4)) is None


def test_compute_target_condition_numbers():
    model = TinyModel()
    config = LoraConfig(target_modules=TARGETS, r=4)
    condition_numbers = compute_target_condition_numbers(model, config)
    assert set(condition_numbers) == set(TARGETS)
    assert condition_numbers["lin0"] == pytest.approx(1.0)
    assert condition_numbers["lin1"] == pytest.approx(10.0)
    assert condition_numbers["lin3"] > condition_numbers["lin2"] > condition_numbers["lin1"]


def test_top_fraction_targets_highest_condition_numbers():
    model = TinyModel()
    config = LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=0.5)
    get_peft_model(model, config)
    # only the two layers with the largest condition numbers are adapted
    assert not isinstance(model.lin0, LoraLayer)
    assert not isinstance(model.lin1, LoraLayer)
    assert isinstance(model.lin2, LoraLayer)
    assert isinstance(model.lin3, LoraLayer)


def test_top_fraction_one_targets_all_modules():
    model = TinyModel()
    config = LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=1.0)
    get_peft_model(model, config)
    for name in TARGETS:
        assert isinstance(getattr(model, name), LoraLayer)


def test_top_fraction_halves_trainable_parameters():
    def num_trainable(config):
        model = get_peft_model(TinyModel(), config)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    full = num_trainable(LoraConfig(target_modules=TARGETS, r=4))
    halved = num_trainable(LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=0.5))
    assert halved * 2 == full


def test_invalid_top_fraction_raises():
    with pytest.raises(ValueError, match="condition_number_top_fraction"):
        LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=0.0)
    with pytest.raises(ValueError, match="condition_number_top_fraction"):
        LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=1.5)


def test_top_fraction_without_matching_modules_raises():
    model = TinyModel()
    config = LoraConfig(target_modules=["does_not_exist"], r=4, condition_number_top_fraction=0.5)
    with pytest.raises(ValueError, match="no targeted modules"):
        get_peft_model(model, config)


def test_save_load_roundtrip_keeps_selection(tmp_path):
    model = TinyModel()
    peft_model = get_peft_model(model, LoraConfig(target_modules=TARGETS, r=4, condition_number_top_fraction=0.5))
    peft_model.save_pretrained(tmp_path)

    reloaded_model = TinyModel()
    PeftModel.from_pretrained(reloaded_model, tmp_path)
    assert not isinstance(reloaded_model.lin0, LoraLayer)
    assert not isinstance(reloaded_model.lin1, LoraLayer)
    assert isinstance(reloaded_model.lin2, LoraLayer)
    assert isinstance(reloaded_model.lin3, LoraLayer)
