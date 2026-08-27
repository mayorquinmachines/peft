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

import torch
from torch import nn

from peft import PeftModel, get_peft_model
from peft.tuners.ternary_adapt import TernaryAdaptConfig


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
    config = TernaryAdaptConfig(target_modules=["lin0"], **config_kwargs)
    return get_peft_model(model, config)


def assert_ternary_up_to_row_scale(weight):
    # every row of a ternary weight (stored as floats) has at most one nonzero magnitude: the per-row scale
    for row in weight:
        nonzero_magnitudes = row[row != 0].abs().unique()
        assert len(nonzero_magnitudes) <= 1


class TestTernaryAdapt:
    def test_base_weight_ternarized_at_injection(self):
        peft_model = get_peft_mlp()
        layer = peft_model.base_model.model.lin0

        assert layer._is_base_ternarized
        # the base weight is ternarized in-place: ternary values up to a per-output-channel scale
        assert_ternary_up_to_row_scale(layer.base_layer.weight.data)

    def test_identity_at_init_and_gradient_flows_through_ste(self):
        peft_model = get_peft_mlp()
        layer = peft_model.base_model.model.lin0
        x = torch.rand(5, 10)

        # identity at init: the all-ones latent factors ternarize to an all-ones mask
        peft_model.eval()  # disable dropout so the forward is deterministic
        with peft_model.disable_adapter():
            ternary_base_out = peft_model(x)
        adapted_out = peft_model(x)
        assert torch.allclose(ternary_base_out, adapted_out, atol=1e-6)

        # gradients flow to the latent factors through the straight-through estimator; the base weight stays frozen
        peft_model.train()
        loss = torch.nn.functional.nll_loss(peft_model(x), torch.randint(0, 2, (5,)))
        loss.backward()
        assert layer.ternary_adapt_A["default"].grad is not None
        assert layer.ternary_adapt_A["default"].grad.abs().sum() > 0
        assert layer.ternary_adapt_B["default"].grad is not None
        assert layer.ternary_adapt_B["default"].grad.abs().sum() > 0
        assert not layer.base_layer.weight.requires_grad

    def test_merge_stays_ternary_and_unmerge_restores(self):
        peft_model = get_peft_mlp()
        peft_model.eval()  # disable dropout so the forward is deterministic
        layer = peft_model.base_model.model.lin0
        x = torch.rand(5, 10)

        # make the mask non-trivial so the merge actually changes the weight (identity-init gives mask == 1)
        with torch.no_grad():
            layer.ternary_adapt_A["default"].normal_(std=0.1)
            layer.ternary_adapt_B["default"].normal_(std=0.1)

        unmerged_out = peft_model(x).detach().clone()
        ternary_w0 = layer.base_layer.weight.detach().clone()

        # merging is a plain element-wise multiplication of ternary weights: no dequantization, the merged weight
        # stays ternary (up to the per-row scale)
        peft_model.merge_adapter(safe_merge=True)
        assert_ternary_up_to_row_scale(layer.base_layer.weight.data)
        assert torch.allclose(peft_model(x), unmerged_out, atol=1e-6)

        # unmerge restores the exact pre-merge (ternarized) base weight from the cache
        peft_model.unmerge_adapter()
        assert torch.allclose(layer.base_layer.weight, ternary_w0, atol=1e-6)

    def test_save_and_load_roundtrip(self, tmp_path):
        peft_model = get_peft_mlp()
        peft_model.eval()
        layer = peft_model.base_model.model.lin0
        with torch.no_grad():
            layer.ternary_adapt_A["default"].normal_(std=0.1)
            layer.ternary_adapt_B["default"].normal_(std=0.1)
        x = torch.rand(5, 10)
        out_before = peft_model(x).detach().clone()

        peft_model.save_pretrained(tmp_path)

        # a fresh model with the same seed has the same base weights; loading ternarizes them identically at
        # injection and restores the adapter factors from the checkpoint
        torch.manual_seed(0)
        fresh_model = MLP()
        loaded_model = PeftModel.from_pretrained(fresh_model, tmp_path)
        loaded_model.eval()

        loaded_layer = loaded_model.base_model.model.lin0
        assert torch.allclose(loaded_layer.ternary_adapt_A["default"], layer.ternary_adapt_A["default"])
        assert torch.allclose(loaded_layer.ternary_adapt_B["default"], layer.ternary_adapt_B["default"])
        assert torch.allclose(loaded_model(x), out_before, atol=1e-6)
