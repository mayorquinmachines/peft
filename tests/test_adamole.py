# Copyright 2024-present the HuggingFace Inc. team.
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

# This test file is for tests specific to AdaMoLE (Adaptive Mixture of LoRA Experts,
# https://arxiv.org/abs/2405.00361).

import math

import pytest
import torch
from torch import nn

from peft import AdaMoLEConfig, AdaMoLEModel, get_peft_model, get_peft_model_state_dict
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
from peft.tuners.adamole.layer import AdaMoLELinear
from peft.utils import PeftType


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 40, bias=bias)
        self.lin2 = nn.Linear(40, 30, bias=bias)
        self.lin3 = nn.Linear(30, 10, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestAdaMoLE:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return MLP()

    def test_adamole_creates_n_experts(self, mlp):
        # Wrapping an nn.Linear with num_experts=4 creates exactly 4 (lora_A, lora_B) pairs, one router and one
        # threshold network.
        config = AdaMoLEConfig(r=4, num_experts=4, target_modules=["lin1"])
        peft_model = get_peft_model(mlp, config)

        layer = peft_model.base_model.model.lin1
        assert isinstance(layer, AdaMoLELinear)
        assert len(layer.adamole_A["default"]) == 4
        assert len(layer.adamole_B["default"]) == 4
        # single router producing num_experts logits
        router = layer.adamole_router["default"]
        assert router.out_features == 4
        # threshold network mapping the layer input to a single threshold logit
        threshold_net = layer.adamole_threshold["default"]
        assert isinstance(threshold_net, nn.Linear)
        assert threshold_net.in_features == layer.in_features
        assert threshold_net.out_features == 1
        # default threshold_max = 1 / num_experts guarantees at least one active expert per token
        assert layer.adamole_threshold_max["default"] == pytest.approx(0.25)

    def test_adamole_activates_only_experts_above_threshold(self):
        # Per-expert activation: only experts whose softmax weight reaches the input-adaptive threshold contribute,
        # with renormalized, threshold-subtracted weights alpha_i = (p_i - tau) / sum_j (p_j - tau).
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, lora_alpha=2, num_experts=4, use_threshold=True, threshold_max=0.5)
        layer = AdaMoLELinear(base, "default", config)

        # Make expert e's delta distinguishable: B_e(A_e(x)) = (e + 1) * x[0] in the first output coordinate.
        for e in range(4):
            layer.adamole_A["default"][e].weight.data.zero_()
            layer.adamole_A["default"][e].weight.data[0, 0] = 1.0
            layer.adamole_B["default"][e].weight.data.zero_()
            layer.adamole_B["default"][e].weight.data[0, 0] = float(e + 1)

        # Constant gates [0.7, 0.1, 0.1, 0.1] for x = ones: the router logits are the row sums of its weight.
        logits = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        layer.adamole_router["default"].weight.data = logits.unsqueeze(1).expand(4, 4) / 4

        # Threshold network with zero weights: tau = 0.5 * sigmoid(bias).
        threshold_net = layer.adamole_threshold["default"]
        threshold_net.weight.data.zero_()

        x = torch.ones(1, 4)

        # tau = 0.5 * 0.6 = 0.3 -> only expert 0 (p = 0.7) is active, with renormalized weight 1.
        threshold_net.bias.data.fill_(math.log(1.5))
        delta = layer.compute_delta("default", x)
        assert torch.allclose(delta, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)

        # tau = 0.5 * 0.1 = 0.05 -> all four experts activate with alpha_i proportional to (p_i - tau).
        threshold_net.bias.data.fill_(math.log(0.1 / 0.9))
        delta = layer.compute_delta("default", x)
        weights = torch.tensor([0.7, 0.1, 0.1, 0.1]) - 0.05
        alpha = weights / weights.sum()
        expected = sum(alpha[e].item() * (e + 1) for e in range(4))
        assert torch.allclose(delta[0, 0], torch.tensor(expected), atol=1e-6)

    def test_adamole_without_threshold_is_dense_mixture(self):
        # use_threshold=False recovers a plain dense softmax mixture: delta = sum_i p_i * B_i A_i x.
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, lora_alpha=2, num_experts=4, use_threshold=False)
        layer = AdaMoLELinear(base, "default", config)

        for e in range(4):
            layer.adamole_A["default"][e].weight.data.zero_()
            layer.adamole_A["default"][e].weight.data[0, 0] = 1.0
            layer.adamole_B["default"][e].weight.data.zero_()
            layer.adamole_B["default"][e].weight.data[0, 0] = float(e + 1)

        logits = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        layer.adamole_router["default"].weight.data = logits.unsqueeze(1).expand(4, 4) / 4

        delta = layer.compute_delta("default", torch.ones(1, 4))
        expected = 0.7 * 1 + 0.1 * (2 + 3 + 4)
        assert torch.allclose(delta[0, 0], torch.tensor(expected), atol=1e-6)

    def test_adamole_threshold_is_input_adaptive(self):
        # The threshold network maps the input to the threshold: two samples with identical (uniform) routing are
        # gated differently because their thresholds differ.
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, lora_alpha=2, num_experts=4, use_threshold=True, threshold_max=0.9)
        layer = AdaMoLELinear(base, "default", config)

        # Give every expert a non-zero delta so a surviving token is clearly non-zero.
        for e in range(4):
            layer.adamole_A["default"][e].weight.data.fill_(1.0)
            layer.adamole_B["default"][e].weight.data.fill_(1.0)

        # Uniform routing (all gates 0.25) for any input.
        layer.adamole_router["default"].weight.data.zero_()

        # tau = 0.9 * sigmoid(10 * x[0]): x[0] = 1 -> tau ~ 0.9 masks every expert; x[0] = -1 -> tau ~ 0 keeps all.
        threshold_net = layer.adamole_threshold["default"]
        threshold_net.weight.data.zero_()
        threshold_net.weight.data[0, 0] = 10.0
        threshold_net.bias.data.zero_()

        x = torch.zeros(2, 4)
        x[0, 0] = 1.0
        x[1, 0] = -1.0

        delta = layer.compute_delta("default", x)
        assert torch.all(delta[0] == 0.0)  # high input-adaptive threshold masks every expert
        assert torch.any(delta[1] != 0.0)  # low threshold keeps the mixture active

    def test_adamole_backward_updates_router_and_threshold(self):
        # A forward + backward pass must produce gradients on the router weights and on the threshold network, whose
        # parameters are trainable because tau enters the mixture weights differentiably via (p_i - tau).
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=False)
        config = AdaMoLEConfig(r=2, num_experts=4, use_threshold=True)
        layer = AdaMoLELinear(base, "default", config)

        # Make the experts contribute a clear signal.
        for e in range(4):
            layer.adamole_B["default"][e].weight.data.fill_(1.0)

        x = torch.randn(3, 4)
        layer(x).sum().backward()

        router_weight = layer.adamole_router["default"].weight
        assert router_weight.grad is not None
        assert torch.any(router_weight.grad != 0)

        threshold_weight = layer.adamole_threshold["default"].weight
        assert threshold_weight.grad is not None
        assert torch.any(threshold_weight.grad != 0)

    def test_adamole_registered_peft_type(self):
        # AdaMoLE is registered and get_peft_model dispatches to AdaMoLEModel, preserving the base output shape.
        assert PeftType.ADAMOLE in PEFT_TYPE_TO_TUNER_MAPPING
        assert PEFT_TYPE_TO_TUNER_MAPPING[PeftType.ADAMOLE] is AdaMoLEModel

        torch.manual_seed(0)
        base = MLP()
        torch.manual_seed(0)
        model_for_peft = MLP()

        config = AdaMoLEConfig(r=4, num_experts=2, target_modules=["lin1", "lin2"])
        peft_model = get_peft_model(model_for_peft, config)
        assert isinstance(peft_model.base_model, AdaMoLEModel)

        x = torch.randn(5, 10)
        with torch.no_grad():
            base_out = base(x)
            peft_out = peft_model(x)
        assert base_out.shape == peft_out.shape

    def test_adamole_state_dict_roundtrip(self):
        # The adapter state dict must carry the experts, router and threshold network, and a fresh AdaMoLE model must
        # reproduce the same output after loading it.
        torch.manual_seed(0)
        config = AdaMoLEConfig(r=4, num_experts=3, target_modules=["lin1"])
        model = get_peft_model(MLP(), config)

        # Move the adapter off its (partly zero) initialization.
        with torch.no_grad():
            for param in model.base_model.model.lin1.adamole_B["default"].parameters():
                param.add_(torch.randn_like(param))

        state_dict = get_peft_model_state_dict(model)
        assert any("adamole_A" in key for key in state_dict)
        assert any("adamole_B" in key for key in state_dict)
        assert any("adamole_router" in key for key in state_dict)
        assert any("adamole_threshold" in key for key in state_dict)

        torch.manual_seed(0)
        fresh = get_peft_model(MLP(), config)
        fresh.load_state_dict(state_dict, strict=False)

        x = torch.randn(3, 10)
        with torch.no_grad():
            assert torch.allclose(model(x), fresh(x))
