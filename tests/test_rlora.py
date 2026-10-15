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

from peft import PeftModel, RLoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 2, bias=bias)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        return X


class TestRLoraInit:
    """Tests for the multi-head random initialization of R-LoRA."""

    def test_random_init_preserves_base_output(self):
        # The head matrices are randomly initialized and mean-centered, which keeps the adapter's initial
        # contribution at exactly zero without mutating the base weights.
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        w0 = model.lin0.weight.detach().clone()

        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        rlora_B = layer.rlora_B["default"]
        # heads are randomly initialized (non-zero) and mean-centered
        assert rlora_B.abs().sum() > 0
        assert torch.allclose(rlora_B.mean(dim=0), torch.zeros_like(rlora_B[0]), atol=1e-7)
        # the base weight is not mutated by the initialization ...
        assert torch.allclose(layer.base_layer.weight, w0)
        # ... yet the output is preserved
        assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_zero_init_variant(self):
        # With init_rlora_weights=False the heads are zero-initialized, as in standard LoRA.
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()

        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3, init_rlora_weights=False)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        assert layer.rlora_B["default"].abs().sum() == 0
        assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_head_diversity(self):
        # The paper motivates the random initialization by the increased diversity of the head matrices:
        # heads must not be (near-)duplicates of each other.
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0"], r=8, rlora_num_heads=4)
        peft_model = get_peft_model(model, config)
        rlora_B = peft_model.base_model.model.lin0.rlora_B["default"]

        flat = rlora_B.flatten(start_dim=1)
        cosine = nn.functional.cosine_similarity(flat.unsqueeze(0), flat.unsqueeze(1), dim=-1)
        off_diagonal = cosine - torch.eye(len(flat))
        assert off_diagonal.abs().max() < 0.9

    def test_invalid_num_heads_raises(self):
        with pytest.raises(ValueError, match="`rlora_num_heads` must be at least 2"):
            RLoraConfig(target_modules=["lin0"], rlora_num_heads=1)


class TestRLoraForward:
    def test_head_dropout_only_active_in_training(self):
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=4, rlora_head_dropout=0.5)
        peft_model = get_peft_model(model, config)
        x = torch.rand(5, 10)

        peft_model.train()
        out1 = peft_model(x).detach()
        out2 = peft_model(x).detach()
        assert not torch.allclose(out1, out2)

        peft_model.eval()
        out3 = peft_model(x).detach()
        out4 = peft_model(x).detach()
        assert torch.allclose(out3, out4)

    def test_gradient_flows_to_all_adapter_parameters(self):
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0

        peft_model(torch.rand(5, 10)).sum().backward()

        assert layer.rlora_A["default"].weight.grad is not None
        assert layer.rlora_B["default"].grad is not None
        assert layer.rlora_route["default"].weight.grad is not None
        # base weights stay frozen
        assert layer.base_layer.weight.grad is None

    def test_disable_adapter_recovers_base_model(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        # make the adapter contribution non-trivial via the router
        with torch.no_grad():
            peft_model.base_model.model.lin0.rlora_route["default"].weight.normal_(std=0.5)

        assert not torch.allclose(base_out, peft_model(x), atol=1e-4)
        with peft_model.disable_adapter():
            assert torch.allclose(base_out, peft_model(x), atol=1e-6)

    def test_merge_is_not_supported(self):
        # the head router is input-dependent, so there is no static delta weight to merge
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        with pytest.raises(NotImplementedError, match="router is input-dependent"):
            peft_model.merge_adapter()

    def test_multiple_adapters(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        config = RLoraConfig(target_modules=["lin0"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        config2 = RLoraConfig(target_modules=["lin0"], r=2, rlora_num_heads=2)
        peft_model.add_adapter("other", config2)

        peft_model.set_adapter("other")
        out_other = peft_model(x)
        peft_model.set_adapter("default")
        out_default = peft_model(x)
        # both adapters are identity at init
        assert torch.allclose(out_other, base_out, atol=1e-6)
        assert torch.allclose(out_default, base_out, atol=1e-6)


class TestRLoraSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        model = MLP()
        config = RLoraConfig(target_modules=["lin0", "lin1"], r=4, rlora_num_heads=3)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        # make the adapter contribution non-trivial
        with torch.no_grad():
            peft_model.base_model.model.lin0.rlora_route["default"].weight.normal_(std=0.5)

        x = torch.rand(5, 10)
        out_before = peft_model(x).detach().clone()

        peft_model.save_pretrained(tmp_path)

        # same seed -> identical base model weights; scrambled RNG -> different fresh adapter init
        torch.manual_seed(0)
        fresh_model = MLP()
        torch.manual_seed(999)
        loaded_model = PeftModel.from_pretrained(fresh_model, tmp_path)
        loaded_model.eval()

        out_after = loaded_model(x).detach().clone()
        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_non_linear_target_raises(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(4, 4, 3)

            def forward(self, x):
                return self.conv(x)

        torch.manual_seed(0)
        model = ConvModel()
        config = RLoraConfig(target_modules=["conv"], r=2, rlora_num_heads=2)
        with pytest.raises(TypeError, match="only `torch.nn.Linear` is supported"):
            get_peft_model(model, config)
