# Copyright 2023-present the HuggingFace Inc. team.
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
Unit tests for the SNR-based importance criterion of AdaLoRA's RankAllocator.

Adapted from "A Bayesian Interpretation of Adaptive Low-Rank Adaptation" (https://arxiv.org/abs/2409.10673): the rank
budget is allocated by a signal-to-noise ratio score (weight^2 / EMA of squared gradients) instead of the default
sensitivity-based score.
"""

import pytest
import torch
from torch import nn

from peft import AdaLoraConfig, get_peft_model


class SimpleMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, in_features=20, hidden=40, out_features=5):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


def _make_adalora_model(target_modules=("lin0", "lin1"), **extra):
    """Create a simple AdaLoRA model; config defaults can be overridden via **extra."""
    base = SimpleMLP()
    config_kwargs = {
        "target_modules": list(target_modules),
        "init_r": 12,
        "target_r": 4,
        "init_lora_weights": False,
        "tinit": 0,
        "tfinal": 10,
        "deltaT": 1,
        "total_step": 100,
    }
    config_kwargs.update(extra)
    config = AdaLoraConfig(**config_kwargs)
    return get_peft_model(base, config)


def _run_train_steps(model, num_steps, in_features=20):
    """Run training steps, calling update_and_allocate after each optimizer step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for step in range(num_steps):
        x = torch.randn(4, in_features)
        model(x).sum().backward()
        optimizer.step()
        model.base_model.update_and_allocate(step)
        optimizer.zero_grad()


class TestAdaLoraSnrImportance:
    def test_invalid_importance_criterion_raises(self):
        with pytest.raises(ValueError, match="importance_criterion"):
            AdaLoraConfig(target_modules=["lin0"], total_step=10, tfinal=2, importance_criterion="unknown")

    def test_snr_scores_populated_after_update(self):
        """The snr criterion should populate the gradient second moment and the smoothed SNR scores."""
        torch.manual_seed(0)
        model = _make_adalora_model(importance_criterion="snr")

        _run_train_steps(model, num_steps=2)

        rankallocator = model.base_model.rankallocator
        assert len(rankallocator.exp_avg_grad_sq) > 0
        assert any(v.abs().sum() > 0 for v in rankallocator.exp_avg_grad_sq.values())
        assert any(v.abs().sum() > 0 for v in rankallocator.exp_avg_ipt.values())
        # uncertainty quantification is not part of the SNR criterion
        assert all(v.abs().sum() == 0 for v in rankallocator.exp_avg_unc.values())

    def test_snr_masking_respects_budget(self):
        """Masking with the snr criterion should keep exactly the scheduled number of rank triplets."""
        torch.manual_seed(0)
        model = _make_adalora_model(importance_criterion="snr")
        last_step = 19

        _run_train_steps(model, num_steps=last_step + 1)

        rankallocator = model.base_model.rankallocator
        budget, mask_ind = rankallocator.budget_schedule(last_step)
        assert mask_ind
        rank_pattern = model.base_model.peft_config["default"].rank_pattern
        kept = sum(sum(pattern) for pattern in rank_pattern.values())
        assert kept == budget

    def test_default_sensitivity_criterion_unchanged(self):
        """The default sensitivity criterion should not populate the SNR-specific state."""
        torch.manual_seed(0)
        model = _make_adalora_model()

        _run_train_steps(model, num_steps=2)

        rankallocator = model.base_model.rankallocator
        assert rankallocator.exp_avg_grad_sq == {}
        assert any(v.abs().sum() > 0 for v in rankallocator.exp_avg_unc.values())
