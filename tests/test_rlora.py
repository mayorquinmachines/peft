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

from peft import LoraConfig, PeftModel, RLoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 20)
        self.lin1 = nn.Linear(20, 2)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


def get_peft_mlp(seed=0, **config_kwargs):
    torch.manual_seed(seed)
    model = MLP().eval()
    torch.manual_seed(seed + 1)
    config = RLoraConfig(target_modules=["lin0", "lin1"], **config_kwargs)
    return get_peft_model(model, config)


class TestRLora:
    def test_identity_at_init(self):
        # Multi-Head Random Initialization is centered, so the adapter is an exact identity at init
        torch.manual_seed(0)
        model = MLP().eval()
        x = torch.rand(5, 10)
        base_out = model(x)

        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=4, head_dropout=0.5)
        peft_model.eval()
        assert torch.allclose(peft_model(x), base_out, atol=1e-6)

    def test_heads_are_diverse_and_zero_sum_at_init(self):
        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=4)
        layer = peft_model.base_model.model.lin0
        weight_B = layer.lora_B["default"].weight  # (out_features, num_heads * r)
        heads = weight_B.view(weight_B.shape[0], 4, -1)

        # Multi-Head Random Initialization: distinct, randomly initialized heads that sum to zero
        assert torch.allclose(heads.sum(dim=1), torch.zeros_like(heads[:, 0]), atol=1e-6)
        assert not torch.allclose(heads[:, 0], heads[:, 1])
        assert not torch.allclose(heads[:, 2], heads[:, 3])

    def test_multi_head_dropout_stochastic_in_training_deterministic_in_eval(self):
        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=4, head_dropout=0.5)
        x = torch.rand(5, 10)

        peft_model.train()
        assert not torch.allclose(peft_model(x), peft_model(x))

        peft_model.eval()
        assert torch.allclose(peft_model(x), peft_model(x))

    def test_merge_unload_preserves_eval_output(self):
        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=3, head_dropout=0.2)
        x = torch.rand(5, 10)
        with torch.no_grad():
            # move the adapter away from its identity initialization
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.add_(torch.rand_like(param) * 0.01)

        peft_model.eval()
        expected = peft_model(x)
        merged = peft_model.merge_and_unload()
        assert torch.allclose(merged(x), expected, atol=1e-5)

    def test_save_and_load(self):
        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=3, head_dropout=0.2)
        x = torch.rand(5, 10)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.add_(torch.rand_like(param) * 0.01)
        peft_model.eval()
        expected = peft_model(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_model.save_pretrained(tmp_dir)
            torch.manual_seed(0)
            fresh_model = MLP().eval()
            loaded = PeftModel.from_pretrained(fresh_model, tmp_dir)
            loaded.eval()
            assert torch.allclose(loaded(x), expected, atol=1e-6)

    def test_single_head_equals_vanilla_lora(self):
        x = torch.rand(5, 10)

        torch.manual_seed(0)
        rlora_model = MLP().eval()
        torch.manual_seed(1)
        rlora_model = get_peft_model(
            rlora_model, RLoraConfig(r=8, lora_alpha=16, target_modules=["lin0", "lin1"], num_heads=1)
        )

        torch.manual_seed(0)
        lora_model = MLP().eval()
        torch.manual_seed(1)
        lora_model = get_peft_model(lora_model, LoraConfig(r=8, lora_alpha=16, target_modules=["lin0", "lin1"]))

        rlora_params = dict(rlora_model.named_parameters())
        lora_params = dict(lora_model.named_parameters())
        assert all(torch.equal(rlora_params[name], lora_params[name]) for name in lora_params)

        with torch.no_grad():
            for params in (rlora_params, lora_params):
                torch.manual_seed(7)
                for name, param in params.items():
                    if "lora_B" in name:
                        param.add_(torch.rand_like(param))

        rlora_model.eval()
        lora_model.eval()
        assert torch.allclose(rlora_model(x), lora_model(x), atol=1e-6)

    def test_gradient_flows_to_shared_projection_and_heads(self):
        peft_model = get_peft_mlp(r=8, lora_alpha=16, num_heads=4, head_dropout=0.5)
        x = torch.rand(32, 10)
        peft_model.train()
        peft_model(x).sum().backward()

        layer = peft_model.base_model.model.lin0
        assert layer.lora_A["default"].weight.grad is not None
        grad_B = layer.lora_B["default"].weight.grad
        assert grad_B is not None
        # the independent per-head masks produce different gradients for different heads
        head_grads = grad_B.view(grad_B.shape[0], 4, -1)
        assert not torch.allclose(head_grads[:, 0], head_grads[:, 1])

    def test_conv1d_target_merge(self):
        from transformers.pytorch_utils import Conv1D

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv1D(20, 10)  # stores weight as (fan_in, fan_out)

            def forward(self, x):
                return self.conv(x)

        torch.manual_seed(0)
        model = ConvModel().eval()
        x = torch.rand(5, 10)
        config = RLoraConfig(r=8, lora_alpha=16, target_modules=["conv"], num_heads=4)
        peft_model = get_peft_model(model, config)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.add_(torch.rand_like(param) * 0.01)

        peft_model.eval()
        expected = peft_model(x)
        merged = peft_model.merge_and_unload()
        assert torch.allclose(merged(x), expected, atol=1e-5)

    def test_invalid_configs_raise(self):
        with pytest.raises(ValueError, match="`num_heads` should be a positive integer"):
            RLoraConfig(target_modules=["lin0"], num_heads=0)
        with pytest.raises(ValueError, match="`head_dropout` should be a value in"):
            RLoraConfig(target_modules=["lin0"], head_dropout=1.0)
        with pytest.raises(ValueError, match="does not support `lora_bias=True`"):
            RLoraConfig(target_modules=["lin0"], num_heads=2, lora_bias=True)
        with pytest.raises(ValueError, match="does not support DoRA"):
            RLoraConfig(target_modules=["lin0"], num_heads=2, use_dora=True)
        with pytest.raises(ValueError, match="does not support `init_lora_weights"):
            RLoraConfig(target_modules=["lin0"], num_heads=2, init_lora_weights="pissa")

    def test_unsupported_target_module_raises(self):
        class EmbModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 10)

            def forward(self, x):
                return self.emb(x)

        config = RLoraConfig(target_modules=["emb"], num_heads=2)
        with pytest.raises(TypeError, match="not supported"):
            get_peft_model(EmbModel(), config)
