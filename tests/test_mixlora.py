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

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from peft import MixLoraConfig, PeftModel, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)
        self.dtype = torch.float

    def forward(self, X):
        X = X.to(self.dtype)
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


class TestMixLoraConfig:
    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="`top_k` must be between 1 and `num_experts`"):
            MixLoraConfig(target_modules=["lin0"], num_experts=2, top_k=3)

    def test_invalid_num_experts_raises(self):
        with pytest.raises(ValueError, match="`num_experts` should be a positive integer"):
            MixLoraConfig(target_modules=["lin0"], num_experts=0)


class TestMixLora:
    def get_config(self, **kwargs):
        kwargs.setdefault("target_modules", ["lin0", "lin1"])
        kwargs.setdefault("r", 4)
        kwargs.setdefault("num_experts", 4)
        kwargs.setdefault("top_k", 2)
        kwargs.setdefault("lora_alpha", 8)
        return MixLoraConfig(**kwargs)

    def test_identity_at_init(self):
        # with the default initialization (experts' B matrices are zero), the adapter is an exact no-op
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()

        peft_model = get_peft_model(model, self.get_config())
        peft_model.eval()
        assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_only_adapter_weights_are_trainable(self):
        torch.manual_seed(0)
        model = MLP()
        peft_model = get_peft_model(model, self.get_config())
        trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        # 2 targeted layers x (experts_A + experts_B + router)
        assert len(trainable) == 6
        assert all("mixlora_" in n for n in trainable)

    def test_output_changes_when_experts_are_nontrivial(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()

        peft_model = get_peft_model(model, self.get_config())
        peft_model.eval()
        layer = peft_model.base_model.model.lin0
        with torch.no_grad():
            layer.mixlora_experts_B["default"].normal_(std=0.1)
        assert not torch.allclose(base_out, peft_model(x), atol=1e-4)

    def test_disable_adapter_restores_base_output(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()

        peft_model = get_peft_model(model, self.get_config())
        with torch.no_grad():
            peft_model.base_model.model.lin0.mixlora_experts_B["default"].normal_(std=0.1)
        with peft_model.disable_adapter():
            assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_routing_matches_manual_topk_computation(self):
        # the forward must combine the top-k experts with the re-normalized router probabilities
        torch.manual_seed(0)
        model = MLP()
        config = self.get_config(target_modules=["lin0"], num_experts=3, top_k=2)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0
        with torch.no_grad():
            layer.mixlora_experts_B["default"].normal_(std=0.1)
            layer.mixlora_router["default"].weight.normal_(std=0.5)

        x = torch.rand(5, 10)
        layer_out = layer(x)

        experts_A = layer.mixlora_experts_A["default"]
        experts_B = layer.mixlora_experts_B["default"]
        router_weight = layer.mixlora_router["default"].weight
        probs = F.softmax((x @ router_weight.T).float(), dim=-1)
        topk_weights, topk_indices = torch.topk(probs, 2, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        gates = torch.zeros_like(probs).scatter(-1, topk_indices, topk_weights)
        hidden = torch.einsum("ti,eri->ter", x, experts_A)
        expert_outputs = torch.einsum("ter,eor->teo", hidden, experts_B)
        manual_delta = torch.einsum("te,teo->to", gates, expert_outputs) * (config.lora_alpha / config.r)
        manual_out = layer.base_layer(x) + manual_delta
        assert torch.allclose(layer_out, manual_out, atol=1e-5)

    def test_router_aux_loss_recorded_in_training_mode(self):
        torch.manual_seed(0)
        model = MLP()
        config = self.get_config(router_aux_loss_coef=0.01)
        peft_model = get_peft_model(model, config)
        peft_model.train()
        _ = peft_model(torch.rand(5, 10))
        aux_loss = peft_model.base_model.get_router_aux_loss()
        assert aux_loss.ndim == 0
        assert aux_loss.item() > 0
        assert aux_loss.requires_grad

    def test_no_router_aux_loss_in_eval_mode(self):
        torch.manual_seed(0)
        model = MLP()
        config = self.get_config(router_aux_loss_coef=0.01)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        _ = peft_model(torch.rand(5, 10))
        with pytest.raises(ValueError, match="No router auxiliary loss was recorded"):
            peft_model.base_model.get_router_aux_loss()

    def test_gradients_flow_to_router_and_experts(self):
        torch.manual_seed(0)
        model = MLP()
        peft_model = get_peft_model(model, self.get_config())
        peft_model.train()
        peft_model(torch.rand(5, 10)).sum().backward()
        layer = peft_model.base_model.model.lin0
        assert layer.mixlora_router["default"].weight.grad is not None
        assert layer.mixlora_experts_A["default"].grad is not None
        assert layer.mixlora_experts_B["default"].grad is not None

    def test_merge_raises(self):
        torch.manual_seed(0)
        model = MLP()
        peft_model = get_peft_model(model, self.get_config())
        with pytest.raises(ValueError, match="cannot be merged"):
            peft_model.merge_adapter()

    def test_save_and_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        peft_model = get_peft_model(model, self.get_config(target_modules=["lin0"]))
        peft_model.eval()
        with torch.no_grad():
            peft_model.base_model.model.lin0.mixlora_experts_B["default"].normal_(std=0.1)

        x = torch.rand(5, 10)
        out = peft_model(x).detach().clone()

        peft_model.save_pretrained(tmp_path)
        fresh_model = MLP()
        fresh_model.load_state_dict(base_state_dict)
        fresh_model.eval()
        loaded_model = PeftModel.from_pretrained(fresh_model, tmp_path)
        loaded_model.eval()
        assert torch.allclose(out, loaded_model(x), atol=1e-6)
