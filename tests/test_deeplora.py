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
from torch import nn

from peft import DeepLoraConfig, LoraConfig, PeftModel, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)
        self.dtype = torch.float

    def forward(self, X):
        X = X.to(self.dtype)
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


def get_peft_mlp(**config_kwargs):
    torch.manual_seed(0)
    model = MLP()
    model.eval()  # disable dropout so the forward is deterministic
    config = DeepLoraConfig(target_modules=["lin0"], **config_kwargs)
    peft_model = get_peft_model(model, config)
    peft_model.eval()
    return peft_model


class TestDeepLoraConfig:
    def test_invalid_num_factors(self):
        with pytest.raises(ValueError, match="`num_factors` must be at least 2"):
            DeepLoraConfig(target_modules=["lin0"], num_factors=1)

    def test_incompatible_lora_options(self):
        with pytest.raises(ValueError, match="`use_dora` is not supported"):
            DeepLoraConfig(target_modules=["lin0"], use_dora=True)
        with pytest.raises(ValueError, match="`use_rslora` is not supported"):
            DeepLoraConfig(target_modules=["lin0"], use_rslora=True)
        with pytest.raises(ValueError, match="only supports `init_lora_weights"):
            DeepLoraConfig(target_modules=["lin0"], init_lora_weights="pissa")


class TestDeepLoraForward:
    def test_zero_delta_at_init(self):
        # like LoRA, Deep LoRA is an identity at initialization: B is zero and the intermediate factors are
        # identities, so the chain contributes nothing
        torch.manual_seed(0)
        base_model = MLP().eval()
        x = torch.rand(5, 10)
        base_out = base_model(x).detach()

        peft_model = get_peft_mlp(r=4, num_factors=4)
        assert torch.allclose(peft_model(x), base_out, atol=1e-6)

    def test_forward_matches_manual_chain(self):
        peft_model = get_peft_mlp(r=4, lora_alpha=8, num_factors=4)
        layer = peft_model.base_model.model.lin0

        # perturb all factors so the chain is non-trivial
        with torch.no_grad():
            layer.lora_A["default"].weight.normal_(std=0.1)
            layer.lora_B["default"].weight.normal_(std=0.1)
            layer.deeplora_factors["default"].normal_(std=0.1)

        x = torch.rand(5, 10)
        out = peft_model(x)

        # manually contract the chain: scaling * B @ M2 @ M1 @ A
        chain = layer.lora_A["default"].weight
        for factor in layer.deeplora_factors["default"]:
            chain = factor @ chain
        delta_weight = layer.lora_B["default"].weight @ chain * layer.scaling["default"]
        assert layer.scaling["default"] == pytest.approx(8 / 4**3)

        lin0 = layer.base_layer
        expected = torch.log_softmax(
            peft_model.base_model.model.lin1(torch.relu(x @ (lin0.weight + delta_weight).t() + lin0.bias)),
            dim=-1,
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_depth_two_matches_lora(self):
        # with num_factors=2, Deep LoRA must reduce exactly to LoRA
        torch.manual_seed(0)
        deeplora_model = get_peft_mlp(r=4, lora_alpha=8, num_factors=2)
        deeplora_layer = deeplora_model.base_model.model.lin0
        assert "default" not in deeplora_layer.deeplora_factors

        torch.manual_seed(0)
        lora_model = get_peft_model(MLP().eval(), LoraConfig(target_modules=["lin0"], r=4, lora_alpha=8))
        lora_model.eval()
        lora_layer = lora_model.base_model.model.lin0

        # align the randomly initialized weights, then perturb them identically
        with torch.no_grad():
            lora_layer.lora_A["default"].weight.copy_(deeplora_layer.lora_A["default"].weight)
            lora_layer.lora_B["default"].weight.copy_(deeplora_layer.lora_B["default"].weight)
            deeplora_layer.lora_A["default"].weight.normal_(std=0.1)
            deeplora_layer.lora_B["default"].weight.normal_(std=0.1)
            lora_layer.lora_A["default"].weight.copy_(deeplora_layer.lora_A["default"].weight)
            lora_layer.lora_B["default"].weight.copy_(deeplora_layer.lora_B["default"].weight)

        x = torch.rand(5, 10)
        assert torch.allclose(deeplora_model(x), lora_model(x), atol=1e-6)

    def test_gradients_flow_through_whole_chain(self):
        # the paper's core claim: the overparameterized chain is jointly optimized, so every factor must
        # receive gradients
        peft_model = get_peft_mlp(r=4, num_factors=3)
        layer = peft_model.base_model.model.lin0
        with torch.no_grad():
            layer.lora_B["default"].weight.normal_(std=0.1)

        x = torch.rand(5, 10)
        peft_model(x).sum().backward()

        assert layer.lora_A["default"].weight.grad.abs().sum() > 0
        assert layer.lora_B["default"].weight.grad.abs().sum() > 0
        assert layer.deeplora_factors["default"].grad.abs().sum() > 0

    def test_only_adapter_weights_are_trainable(self):
        peft_model = get_peft_mlp(r=4, num_factors=3)
        layer = peft_model.base_model.model.lin0

        assert layer.lora_A["default"].weight.requires_grad
        assert layer.lora_B["default"].weight.requires_grad
        assert layer.deeplora_factors["default"].requires_grad
        assert not layer.base_layer.weight.requires_grad
        assert not layer.base_layer.bias.requires_grad


class TestDeepLoraMerge:
    def test_merge_unmerge_exact(self):
        peft_model = get_peft_mlp(r=4, lora_alpha=8, num_factors=3)
        layer = peft_model.base_model.model.lin0
        with torch.no_grad():
            layer.lora_B["default"].weight.normal_(std=0.1)
            layer.deeplora_factors["default"].normal_(std=0.1)

        x = torch.rand(5, 10)
        unmerged_out = peft_model(x).detach().clone()
        w0 = layer.base_layer.weight.detach().clone()

        peft_model.merge_adapter(safe_merge=True)
        assert torch.allclose(peft_model(x), unmerged_out, atol=1e-4)

        peft_model.unmerge_adapter()
        assert torch.allclose(layer.base_layer.weight, w0, atol=1e-5)


class TestDeepLoraSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        model = MLP().eval()
        config = DeepLoraConfig(target_modules=["lin0", "lin1"], r=4, lora_alpha=8, num_factors=3)
        peft_model = get_peft_model(model, config)
        peft_model.eval()

        for layer in (peft_model.base_model.model.lin0, peft_model.base_model.model.lin1):
            with torch.no_grad():
                layer.lora_A["default"].weight.normal_(std=0.1)
                layer.lora_B["default"].weight.normal_(std=0.1)
                layer.deeplora_factors["default"].normal_(std=0.1)

        x = torch.rand(5, 10)
        expected = peft_model(x).detach()

        peft_model.save_pretrained(tmp_path)

        torch.manual_seed(0)
        base_model = MLP().eval()
        loaded = PeftModel.from_pretrained(base_model, tmp_path)
        loaded.eval()

        assert torch.allclose(loaded(x), expected, atol=1e-5)

        # the intermediate factors must have survived the roundtrip
        for layer in (loaded.base_model.model.lin0, loaded.base_model.model.lin1):
            assert "default" in layer.deeplora_factors
