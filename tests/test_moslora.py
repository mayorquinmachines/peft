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

"""
Unit tests for MoSLoRA (Mixture-of-Subspaces LoRA, https://huggingface.co/papers/2406.11909).

Tests cover:
- registration with the PEFT mapping and `get_peft_model`
- the learnable r x r mixer between the LoRA A and B matrices (delta W = B @ M @ A)
- the mixer initializations (kaiming, orthogonal, identity, butterfly)
- the fixed-mixer variant (trainable_mixer=False)
- merge/unmerge and save/load roundtrips through the regular PeftModel API
"""

import copy

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.moslora import MosLoraConfig, MosLoraLinear, MosLoraModel


class SimpleMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, in_features=20, hidden=40, out_features=5):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


def _make_moslora_model(r=8, **config_kwargs):
    base = SimpleMLP()
    config_kwargs.setdefault("lora_alpha", 2 * r)
    config = MosLoraConfig(target_modules=["lin0", "lin1"], r=r, **config_kwargs)
    return get_peft_model(base, config)


def _get_moslora_layers(model):
    return [m for m in model.modules() if isinstance(m, MosLoraLinear)]


class TestMosLoraIntegration:
    def test_get_peft_model_uses_moslora_tuner(self):
        model = _make_moslora_model()
        assert isinstance(model, PeftModel)
        assert isinstance(model.base_model, MosLoraModel)
        layers = _get_moslora_layers(model)
        assert len(layers) == 2
        assert isinstance(layers[0], LoraLinear)  # MoSLoRA layers build on the LoRA layer machinery

    def test_peft_type_registered(self):
        config = MosLoraConfig(target_modules=["lin0"])
        assert config.peft_type == "MOSLORA"

    def test_output_matches_base_model_at_init(self):
        # like LoRA, B is initialized to zero, so the initial adapter update is zero regardless of the mixer init
        torch.manual_seed(0)
        base = SimpleMLP()
        x = torch.randn(4, 20)
        expected = base(x)
        model = get_peft_model(base, MosLoraConfig(target_modules=["lin0", "lin1"], r=8))
        assert torch.allclose(model(x), expected)

    def test_delta_weight_is_b_times_mixer_times_a(self):
        model = _make_moslora_model(r=4, lora_alpha=8)
        layer = _get_moslora_layers(model)[0]
        weight_A = layer.lora_A["default"].weight
        weight_B = layer.lora_B["default"].weight
        weight_mixer = layer.lora_mixer["default"].weight
        scaling = layer.scaling["default"]
        expected = (weight_B @ weight_mixer @ weight_A) * scaling
        assert torch.allclose(layer.get_delta_weight("default"), expected)

    def test_forward_uses_mixer(self):
        # check the full forward pass against the analytic delta W = B @ M @ A
        model = _make_moslora_model(r=4)
        layer = _get_moslora_layers(model)[0]
        base_layer = layer.get_base_layer()
        x = torch.randn(4, 20)
        delta_weight = layer.get_delta_weight("default")
        expected = base_layer(x) + x @ delta_weight.T
        assert torch.allclose(layer(x), expected, atol=1e-6)

    def test_identity_mixer_recovers_lora(self):
        # the paper shows that vanilla LoRA is MoSLoRA with the mixer fixed to the identity matrix
        lora_model = get_peft_model(SimpleMLP(), LoraConfig(target_modules=["lin0", "lin1"], r=8, lora_alpha=16))
        moslora_model = _make_moslora_model(r=8, lora_alpha=16, mixer_init="identity")

        lora_layer = next(m for m in lora_model.modules() if isinstance(m, LoraLinear))
        moslora_layer = _get_moslora_layers(moslora_model)[0]
        with torch.no_grad():
            lora_layer.lora_A["default"].weight.normal_()
            lora_layer.lora_B["default"].weight.normal_()
            moslora_layer.lora_A["default"].weight.copy_(lora_layer.lora_A["default"].weight)
            moslora_layer.lora_B["default"].weight.copy_(lora_layer.lora_B["default"].weight)
        assert torch.allclose(lora_layer.get_delta_weight("default"), moslora_layer.get_delta_weight("default"))

    def test_gradients_flow_to_mixer_and_b(self):
        # at initialization B = 0, so the gradients w.r.t. A and the mixer are zero, but B receives a non-zero
        # gradient because the mixer is non-zero -- this is why the paper initializes the mixer to non-zero values
        # (with a zero mixer, the gradient w.r.t. B would vanish as well and training would not converge)
        model = _make_moslora_model()
        x = torch.randn(4, 20)
        model(x).sum().backward()
        layer = _get_moslora_layers(model)[0]
        assert layer.lora_mixer["default"].weight.grad is not None
        assert layer.lora_A["default"].weight.grad is not None
        assert layer.lora_B["default"].weight.grad.abs().sum() > 0

    def test_merge_unmerge(self):
        model = _make_moslora_model()
        layer = _get_moslora_layers(model)[0]
        with torch.no_grad():
            layer.lora_B["default"].weight.normal_()
        x = torch.randn(4, 20)
        expected = model(x)

        base_weight_before = copy.deepcopy(layer.get_base_layer().weight)
        model.merge_adapter()
        assert torch.allclose(model(x), expected, atol=1e-6)
        model.unmerge_adapter()
        assert torch.allclose(layer.get_base_layer().weight, base_weight_before)
        assert torch.allclose(model(x), expected, atol=1e-6)

    def test_save_and_load_roundtrip(self, tmp_path):
        base = SimpleMLP()
        base_copy = copy.deepcopy(base)
        model = get_peft_model(base, MosLoraConfig(target_modules=["lin0", "lin1"], r=4, mixer_init="orthogonal"))
        layer = _get_moslora_layers(model)[0]
        with torch.no_grad():
            layer.lora_B["default"].weight.normal_()
        x = torch.randn(4, 20)
        expected = model(x)

        model.save_pretrained(tmp_path)
        # the mixer must be part of the saved adapter
        adapter_weights = load_file(tmp_path / "adapter_model.safetensors")
        assert any("lora_mixer" in key for key in adapter_weights)

        loaded = PeftModel.from_pretrained(base_copy, tmp_path)
        loaded_layer = _get_moslora_layers(loaded)[0]
        assert torch.allclose(loaded_layer.lora_mixer["default"].weight, layer.lora_mixer["default"].weight)
        assert torch.allclose(loaded(x), expected)
        assert loaded.peft_config["default"].peft_type == "MOSLORA"
        assert isinstance(loaded.peft_config["default"], MosLoraConfig)


class TestMosLoraMixerInit:
    @pytest.mark.parametrize("mixer_init", ["kaiming", "orthogonal"])
    def test_mixer_is_nonzero_at_init(self, mixer_init):
        # a zero mixer would prevent convergence, see the paper
        model = _make_moslora_model(mixer_init=mixer_init)
        layer = _get_moslora_layers(model)[0]
        assert layer.lora_mixer["default"].weight.abs().sum() > 0

    def test_identity_init(self):
        model = _make_moslora_model(r=4, mixer_init="identity")
        layer = _get_moslora_layers(model)[0]
        assert torch.allclose(layer.lora_mixer["default"].weight, torch.eye(4))

    def test_orthogonal_init(self):
        model = _make_moslora_model(r=4, mixer_init="orthogonal")
        layer = _get_moslora_layers(model)[0]
        weight = layer.lora_mixer["default"].weight
        assert torch.allclose(weight @ weight.T, torch.eye(4), atol=1e-6)

    def test_butterfly_init(self):
        # the paper's fixed mixer: [[I, I], [I, I]] with r // 2 sized identity blocks
        model = _make_moslora_model(r=4, mixer_init="butterfly")
        layer = _get_moslora_layers(model)[0]
        expected = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]] * 2)
        assert torch.allclose(layer.lora_mixer["default"].weight, expected)

    def test_butterfly_requires_even_rank(self):
        with pytest.raises(ValueError, match="even rank"):
            _make_moslora_model(r=5, mixer_init="butterfly")

    def test_invalid_mixer_init_raises(self):
        with pytest.raises(ValueError, match="mixer_init"):
            MosLoraConfig(target_modules=["lin0"], mixer_init="zeros")


class TestMosLoraFixedMixer:
    def test_fixed_mixer_is_frozen(self):
        model = _make_moslora_model(mixer_init="butterfly", trainable_mixer=False)
        layer = _get_moslora_layers(model)[0]
        assert not layer.lora_mixer["default"].weight.requires_grad
        # A and B stay trainable
        assert layer.lora_A["default"].weight.requires_grad
        assert layer.lora_B["default"].weight.requires_grad

    def test_fixed_mixer_stays_frozen_after_set_adapter(self):
        model = _make_moslora_model(mixer_init="butterfly", trainable_mixer=False)
        model.set_adapter("default")
        layer = _get_moslora_layers(model)[0]
        assert not layer.lora_mixer["default"].weight.requires_grad

    def test_fixed_mixer_does_not_change_during_training(self):
        model = _make_moslora_model(mixer_init="orthogonal", trainable_mixer=False)
        layer = _get_moslora_layers(model)[0]
        mixer_before = layer.lora_mixer["default"].weight.clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        x = torch.randn(4, 20)
        model(x).sum().backward()
        optimizer.step()
        assert torch.allclose(layer.lora_mixer["default"].weight, mixer_before)


class TestMosLoraConfigValidation:
    def test_use_dora_raises(self):
        with pytest.raises(ValueError, match="use_dora"):
            MosLoraConfig(target_modules=["lin0"], use_dora=True)

    def test_unsupported_init_lora_weights_raises(self):
        with pytest.raises(ValueError, match="init_lora_weights"):
            MosLoraConfig(target_modules=["lin0"], init_lora_weights="pissa")

    def test_unsupported_target_module_type_raises(self):
        base = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8))
        config = MosLoraConfig(target_modules=["0"], r=4)
        with pytest.raises(ValueError, match="not supported by MoSLoRA"):
            get_peft_model(base, config)
