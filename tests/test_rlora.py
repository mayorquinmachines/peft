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

import tempfile

import pytest
import torch
from torch import nn

from peft import PeftModel, RLoraConfig, get_peft_model


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


class TestRLoraInit:
    """Multi-Head Random Initialization: non-zero `A` and heads with base-weight compensation, so the adapted layer
    is an exact identity at initialization."""

    def test_identity_at_init(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()  # disable dropout so the forward is deterministic
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        w0 = model.lin0.weight.detach().clone()

        config = RLoraConfig(target_modules=["lin0"], r=4, num_heads=4)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        # the adapter weights are non-zero at init (that is the point of the random init) ...
        assert not torch.allclose(layer.rlora_A["default"], torch.zeros_like(layer.rlora_A["default"]))
        assert not torch.allclose(layer.rlora_B["default"], torch.zeros_like(layer.rlora_B["default"]))
        # ... the base weight was compensated with the initial delta ...
        assert not torch.allclose(layer.base_layer.weight, w0)
        # ... and the compensation exactly preserves the forward output
        assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_init_is_deterministic_for_same_seed(self):
        # the compensation is not part of the saved state dict; it is reproduced when the adapter is re-created,
        # so the init must be independent of the global RNG state and only depend on `init_seed`
        torch.manual_seed(0)
        layer0 = get_peft_model(MLP(), RLoraConfig(target_modules=["lin0"])).base_model.model.lin0
        torch.manual_seed(1234)
        layer1 = get_peft_model(MLP(), RLoraConfig(target_modules=["lin0"])).base_model.model.lin0
        assert torch.allclose(layer0.rlora_A["default"], layer1.rlora_A["default"])
        assert torch.allclose(layer0.rlora_B["default"], layer1.rlora_B["default"])

    def test_init_weights_false_is_standard_identity_init(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        w0 = model.lin0.weight.detach().clone()

        config = RLoraConfig(target_modules=["lin0"], init_weights=False)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0

        # standard LoRA identity init: zero heads, untouched base weight
        assert torch.allclose(layer.rlora_B["default"], torch.zeros_like(layer.rlora_B["default"]))
        assert torch.allclose(layer.base_layer.weight, w0)
        assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="`num_heads` should be a positive integer"):
            RLoraConfig(target_modules=["lin0"], num_heads=0)
        with pytest.raises(ValueError, match="only supports `bias='none'`"):
            RLoraConfig(target_modules=["lin0"], bias="all")


class TestRLoraForward:
    def test_multi_head_structure(self):
        torch.manual_seed(0)
        config = RLoraConfig(target_modules=["lin0"], r=4, num_heads=3)
        peft_model = get_peft_model(MLP(), config)
        layer = peft_model.base_model.model.lin0

        # one shared down-projection A, num_heads stacked head matrices, and a router from input to head logits
        assert layer.rlora_A["default"].shape == (4, 10)
        assert layer.rlora_B["default"].shape == (3, 20, 4)
        assert layer.rlora_router["default"].weight.shape == (3, 10)

    def test_no_router_means_uniform_head_combination(self):
        torch.manual_seed(0)
        config = RLoraConfig(target_modules=["lin0"], r=4, use_router=False)
        peft_model = get_peft_model(MLP(), config)
        layer = peft_model.base_model.model.lin0
        assert "default" not in layer.rlora_router

    def test_head_dropout_stochastic_in_train_deterministic_in_eval(self):
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0"], head_dropout=0.5)
        peft_model = get_peft_model(model, config)
        x = torch.rand(5, 10)

        peft_model.train()
        out1 = peft_model(x).detach().clone()
        out2 = peft_model(x).detach().clone()
        assert not torch.allclose(out1, out2)

        peft_model.eval()
        out3 = peft_model(x).detach().clone()
        out4 = peft_model(x).detach().clone()
        assert torch.allclose(out3, out4)

    def test_gradient_flows_to_shared_a_all_heads_and_router(self):
        torch.manual_seed(0)
        config = RLoraConfig(target_modules=["lin0"], r=4, num_heads=4)
        peft_model = get_peft_model(MLP(), config)
        peft_model.train()
        peft_model(torch.rand(5, 10)).sum().backward()
        layer = peft_model.base_model.model.lin0

        assert layer.rlora_A["default"].grad is not None
        assert layer.rlora_B["default"].grad is not None
        assert layer.rlora_router["default"].weight.grad is not None


class TestRLoraMerge:
    def test_merge_unmerge_exact_without_router(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()  # disable dropout for a deterministic forward
        x = torch.rand(5, 10)

        config = RLoraConfig(target_modules=["lin0"], r=4, use_router=False)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        # make the adapter update non-trivial so the merge delta is non-zero
        with torch.no_grad():
            layer.rlora_B["default"] += 0.1

        unmerged_out = peft_model(x).detach().clone()
        w0 = layer.base_layer.weight.detach().clone()

        peft_model.merge_adapter(safe_merge=True)
        assert torch.allclose(peft_model(x), unmerged_out, atol=1e-5)
        peft_model.unmerge_adapter()
        assert torch.allclose(layer.base_layer.weight, w0, atol=1e-6)

    def test_merge_with_router_warns(self):
        torch.manual_seed(0)
        config = RLoraConfig(target_modules=["lin0"], use_router=True)
        peft_model = get_peft_model(MLP(), config)
        with pytest.warns(UserWarning, match="router"):
            peft_model.merge_adapter()

    def test_save_load_roundtrip_after_training(self):
        # the base-weight compensation is re-applied deterministically when the adapter is re-created on load, so a
        # trained adapter reproduces its outputs after a save/load roundtrip
        torch.manual_seed(0)
        x = torch.rand(5, 10)
        torch.manual_seed(7)
        peft_model = get_peft_model(MLP(), RLoraConfig(target_modules=["lin0"], r=4, num_heads=3))
        optimizer = torch.optim.SGD(peft_model.parameters(), lr=0.1)
        peft_model.train()
        for _ in range(3):
            loss = peft_model(x).sum()
            peft_model.zero_grad()
            loss.backward()
            optimizer.step()
        peft_model.eval()
        out_before = peft_model(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            peft_model.save_pretrained(tmp_dirname)
            torch.manual_seed(7)  # same base model weights as before saving
            loaded = PeftModel.from_pretrained(MLP(), tmp_dirname)
            loaded.eval()
            assert torch.allclose(out_before, loaded(x), atol=1e-6)
