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

import copy

import pytest
import torch
from torch import nn

from peft import HyperLoraConfig, PeftModel, PeftType, get_peft_model
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.tuners.hyperlora import HyperLoraLinear


CONTEXT_DIM = 6


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 20)
        self.lin1 = nn.Linear(20, 2)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


def get_config(**kwargs):
    config_kwargs = {"target_modules": ["lin0"], "r": 4, "context_dim": CONTEXT_DIM, "hypernet_hidden_size": 8}
    config_kwargs.update(kwargs)
    return HyperLoraConfig(**config_kwargs)


def make_nonzero_generator(peft_model, adapter_name="default"):
    # move the generator away from its zero initialization so that the generated LoRA is non-trivial
    layer = peft_model.base_model.model.lin0
    with torch.no_grad():
        for param in layer.hyperlora_generator[adapter_name][-1].parameters():
            param.normal_(0.0, 0.05)


class TestHyperLora:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)

    def test_registered_and_injected_via_get_peft_model(self):
        # exercises the register_peft_method wiring in peft.tuners / peft.mapping
        assert PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.HYPERLORA] is HyperLoraConfig

        model = get_peft_model(TinyMLP(), get_config())
        assert isinstance(model.base_model.model.lin0, HyperLoraLinear)
        # the untargeted layer is left untouched
        assert isinstance(model.base_model.model.lin1, nn.Linear)

    def test_forward_requires_context(self):
        model = get_peft_model(TinyMLP(), get_config())
        x = torch.randn(3, 10)
        with pytest.raises(RuntimeError, match="No context set"):
            model(x)

    def test_zero_init_matches_base_model_output(self):
        base_model = TinyMLP()
        model = get_peft_model(copy.deepcopy(base_model), get_config())
        model.base_model.set_context(torch.randn(CONTEXT_DIM))

        x = torch.randn(3, 10)
        assert torch.allclose(model(x), base_model(x))

    def test_context_controls_generated_lora(self):
        model = get_peft_model(TinyMLP(), get_config())
        make_nonzero_generator(model)
        x = torch.randn(3, 10)

        context_a = torch.randn(CONTEXT_DIM)
        context_b = torch.randn(CONTEXT_DIM)

        model.base_model.set_context(context_a)
        out_a1 = model(x)
        model.base_model.set_context(context_b)
        out_b = model(x)
        model.base_model.set_context(context_a)
        out_a2 = model(x)

        assert not torch.allclose(out_a1, out_b)
        assert torch.allclose(out_a1, out_a2)

    def test_gradients_flow_to_generator_not_base(self):
        model = get_peft_model(TinyMLP(), get_config())
        model.base_model.set_context(torch.randn(CONTEXT_DIM))

        x = torch.randn(3, 10)
        model(x).sum().backward()

        layer = model.base_model.model.lin0
        for param in layer.hyperlora_generator["default"].parameters():
            assert param.grad is not None
        assert layer.get_base_layer().weight.grad is None

    def test_save_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        base_model = TinyMLP()
        model = get_peft_model(base_model, get_config())
        make_nonzero_generator(model)
        context = torch.randn(CONTEXT_DIM)
        model.base_model.set_context(context)
        x = torch.randn(3, 10)
        expected = model(x)

        model.save_pretrained(tmp_path)

        del model
        torch.manual_seed(0)
        reloaded = PeftModel.from_pretrained(TinyMLP(), tmp_path)
        assert isinstance(reloaded.peft_config["default"], HyperLoraConfig)

        reloaded.base_model.set_context(context)
        assert torch.allclose(reloaded(x), expected)

    def test_merge_and_unload_bakes_in_generated_lora(self):
        model = get_peft_model(TinyMLP(), get_config())
        make_nonzero_generator(model)
        context = torch.randn(CONTEXT_DIM)
        model.base_model.set_context(context)
        x = torch.randn(3, 10)
        expected = model(x)

        merged = model.merge_and_unload()
        # after merging, the generated LoRA for the set context is part of the base weights
        assert torch.allclose(merged(x), expected, atol=1e-6)
