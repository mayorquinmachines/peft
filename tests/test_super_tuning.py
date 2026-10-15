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
from torch.testing import assert_close

from peft import SuperTuningConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.super_tuning.layer import Linear, compute_support_mask


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(16, 32)
        self.lin1 = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


def get_wrapped_model(**config_kwargs):
    torch.manual_seed(0)
    model = DummyModel()
    config_kwargs.setdefault("target_modules", ["lin0"])
    config_kwargs.setdefault("density", 0.25)
    return get_peft_model(model, SuperTuningConfig(**config_kwargs))


def get_super_layer(wrapped, name="lin0"):
    return getattr(wrapped.base_model.model, name)


def test_support_mask_density():
    weight = torch.randn(32, 16)
    for density in (0.05, 0.25, 1.0):
        mask = compute_support_mask(weight, density)
        expected = min(weight.numel(), max(1, round(density * weight.numel())))
        assert mask.sum().item() == expected


def test_support_mask_selects_highest_and_lowest_scores():
    weight = torch.rand(2, 4)  # all positive, so the extremes below are unique
    weight[0, 0] = 10.0  # highest magnitude
    weight[1, 3] = 1e-4  # lowest magnitude
    mask_top = compute_support_mask(weight, density=0.125)
    assert mask_top[0, 0] and not mask_top[1, 3]
    mask_bottom = compute_support_mask(weight, density=0.125, select_lowest=True)
    assert mask_bottom[1, 3] and not mask_bottom[0, 0]


def test_support_mask_wanda_uses_activation_norms():
    weight = torch.ones(2, 2)
    # input feature 1 sees much larger activations than feature 0
    activation_norms = torch.tensor([0.01, 100.0])
    mask = compute_support_mask(weight, density=0.5, activation_norms=activation_norms)
    assert mask[:, 1].all() and not mask[:, 0].any()


def test_injection_wraps_target_linear():
    wrapped = get_wrapped_model()
    layer = get_super_layer(wrapped)
    assert isinstance(layer, Linear)
    assert isinstance(wrapped.base_model.model.lin1, torch.nn.Linear)
    mask = layer.super_tuning_mask["default"]
    assert mask.sum().item() == round(0.25 * 32 * 16)


def test_default_target_modules_targets_all_linear():
    wrapped = get_wrapped_model(target_modules=None)
    assert isinstance(get_super_layer(wrapped, "lin0"), Linear)
    assert isinstance(get_super_layer(wrapped, "lin1"), Linear)


def test_only_sparse_delta_is_trainable():
    wrapped = get_wrapped_model()
    trainable = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable
    assert all("super_tuning_delta" in n for n in trainable)


def test_gradient_is_zero_outside_support():
    wrapped = get_wrapped_model()
    x = torch.randn(4, 16)
    wrapped(x).sum().backward()
    layer = get_super_layer(wrapped)
    delta = layer.super_tuning_delta["default"]
    mask = layer.super_tuning_mask["default"]
    assert delta.grad is not None
    assert (delta.grad[~mask] == 0).all()
    assert (delta.grad[mask] != 0).any()


def test_forward_matches_base_plus_sparse_update():
    wrapped = get_wrapped_model()
    layer = get_super_layer(wrapped)
    x = torch.randn(4, 16)
    with torch.no_grad():
        layer.super_tuning_delta["default"].add_(torch.randn_like(layer.super_tuning_delta["default"]))
    base = wrapped.base_model.model.lin0.base_layer
    expected = torch.nn.functional.linear(x, base.weight + layer.get_delta_weight("default"), base.bias)
    assert_close(layer(x), expected)


def test_disable_adapters_restores_base_output():
    wrapped = get_wrapped_model()
    layer = get_super_layer(wrapped)
    with torch.no_grad():
        layer.super_tuning_delta["default"].normal_()
    x = torch.randn(4, 16)
    with wrapped.disable_adapter():
        disabled = wrapped(x)
    assert_close(disabled, get_wrapped_model()(x))


def test_merge_unmerge_roundtrip():
    wrapped = get_wrapped_model()
    layer = get_super_layer(wrapped)
    with torch.no_grad():
        layer.super_tuning_delta["default"].normal_()
    x = torch.randn(4, 16)
    orig_weight = layer.get_base_layer().weight.data.clone()
    before = layer(x)
    layer.merge()
    assert layer.merged
    # merged forward goes through the base layer but must match the adapter-path output
    assert_close(layer(x), before, atol=1e-6, rtol=1e-5)
    layer.unmerge()
    assert not layer.merged
    # unmerge restores the original base weight, and the layer is back in adapter mode
    assert_close(layer.get_base_layer().weight.data, orig_weight, atol=1e-6, rtol=1e-5)
    assert_close(layer(x), before, atol=1e-6, rtol=1e-5)


def test_state_dict_roundtrip_preserves_mask_and_delta():
    wrapped = get_wrapped_model(saliency="wanda")
    layer = get_super_layer(wrapped)
    with torch.no_grad():
        layer.super_tuning_delta["default"].normal_()
    # wanda masks depend on calibration, so they must survive save/load
    wrapped.base_model.calibrate(torch.randn(8, 16))
    state_dict = get_peft_model_state_dict(wrapped)
    assert any("super_tuning_mask" in k for k in state_dict)
    assert any("super_tuning_delta" in k for k in state_dict)

    fresh = get_wrapped_model(saliency="wanda")
    set_peft_model_state_dict(fresh, state_dict)
    fresh_layer = get_super_layer(fresh)
    assert_close(fresh_layer.super_tuning_mask["default"].float(), layer.super_tuning_mask["default"].float())
    assert_close(fresh_layer.super_tuning_delta["default"], layer.super_tuning_delta["default"])


def test_calibrate_recomputes_wanda_support():
    wrapped = get_wrapped_model(saliency="wanda")
    layer = get_super_layer(wrapped)
    mask_before = layer.super_tuning_mask["default"].clone()

    # calibrate with inputs concentrated on a few input features
    x = torch.zeros(64, 16)
    x[:, :4] = torch.randn(64, 4) * 5
    wrapped.base_model.calibrate(x)

    mask_after = layer.super_tuning_mask["default"]
    assert mask_after.sum() == mask_before.sum()  # density is preserved
    assert not torch.equal(mask_after, mask_before)  # support is recomputed from activations
    # most of the support should lie on the active input features
    assert mask_after[:, :4].sum() > mask_after[:, 4:].sum()


def test_calibrate_raises_without_super_layer_inputs():
    wrapped = get_wrapped_model()
    with pytest.raises(ValueError, match="Calibration did not reach any Super-Tuning layer"):
        wrapped.base_model.calibrate([])
