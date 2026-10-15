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
from transformers.pytorch_utils import Conv1D

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.tara import get_target_lora_layers, initialize_tara_weights


RANK = 4
IN_FEATURES = 16
OUT_FEATURES = 32


@pytest.fixture
def synthetic_data():
    """Structured data with a low-rank teacher signal, so that the full-rank gradient has a clear top subspace."""
    torch.manual_seed(0)
    teacher = torch.randn(IN_FEATURES, OUT_FEATURES) @ torch.randn(OUT_FEATURES, OUT_FEATURES) / IN_FEATURES**0.5
    mixing = torch.randn(RANK, IN_FEATURES)

    def sample_batch(n_samples=64):
        z = torch.randn(n_samples, RANK)
        x = z @ mixing + 0.05 * torch.randn(n_samples, IN_FEATURES)
        y = x @ teacher + 0.01 * torch.randn(n_samples, OUT_FEATURES)
        return x, y

    return sample_batch


@pytest.fixture
def simple_model():
    torch.manual_seed(42)
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES)


def make_peft_model(model):
    # lora_alpha = r (scaling 1), the setting used by TaRA
    lora_config = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["0"])
    return get_peft_model(torch.nn.Sequential(model), lora_config)


class TestTaraInitialization:
    def test_no_target_modules_raises(self, synthetic_data):
        model = torch.nn.Sequential(torch.nn.Linear(IN_FEATURES, OUT_FEATURES))
        with pytest.raises(ValueError):
            initialize_tara_weights(model, lambda: None)

    def test_nonzero_lora_b_raises(self, simple_model, synthetic_data):
        peft_model = make_peft_model(simple_model)
        _, lora_layer = next(get_target_lora_layers(peft_model))
        torch.nn.init.normal_(lora_layer.lora_B["default"].weight)

        def calibrate_step():
            x, y = synthetic_data()
            torch.nn.functional.mse_loss(peft_model(x), y).backward()

        with pytest.raises(ValueError, match="lora_B == 0"):
            initialize_tara_weights(peft_model, calibrate_step)

    def test_output_preservation(self, simple_model, synthetic_data):
        """Residual compensation (W_res = W0 - BA) must leave the initial model output unchanged."""
        torch.manual_seed(42)
        peft_model = make_peft_model(simple_model)
        x, _ = synthetic_data()
        peft_model.eval()
        with torch.no_grad():
            output_before = peft_model(x)

        def calibrate_step():
            for _ in range(4):
                x_cal, y_cal = synthetic_data()
                torch.nn.functional.mse_loss(peft_model(x_cal), y_cal).backward()

        initialize_tara_weights(peft_model, calibrate_step)

        peft_model.eval()
        with torch.no_grad():
            output_after = peft_model(x)
        assert torch.allclose(output_before, output_after, atol=1e-4)

    def test_output_preservation_conv1d(self, synthetic_data):
        """The transposed weight layout of transformers Conv1D must be handled correctly."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(Conv1D(OUT_FEATURES, IN_FEATURES))
        peft_model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["0"]))
        x, _ = synthetic_data()
        peft_model.eval()
        with torch.no_grad():
            output_before = peft_model(x)

        def calibrate_step():
            for _ in range(4):
                x_cal, y_cal = synthetic_data()
                torch.nn.functional.mse_loss(peft_model(x_cal), y_cal).backward()

        initialize_tara_weights(peft_model, calibrate_step)

        _, lora_layer = next(get_target_lora_layers(peft_model))
        assert lora_layer.lora_B["default"].weight.abs().sum() > 0
        peft_model.eval()
        with torch.no_grad():
            output_after = peft_model(x)
        assert torch.allclose(output_before, output_after, atol=1e-4)

    def test_weights_are_set(self, simple_model, synthetic_data):
        peft_model = make_peft_model(simple_model)

        def calibrate_step():
            x_cal, y_cal = synthetic_data()
            torch.nn.functional.mse_loss(peft_model(x_cal), y_cal).backward()

        initialize_tara_weights(peft_model, calibrate_step)

        _, lora_layer = next(get_target_lora_layers(peft_model))
        lora_a = lora_layer.lora_A["default"].weight
        lora_b = lora_layer.lora_B["default"].weight
        assert lora_a.shape == (RANK, IN_FEATURES)
        assert lora_b.shape == (OUT_FEATURES, RANK)
        assert lora_b.abs().sum() > 0  # B is no longer the zero initialization
        assert torch.isfinite(lora_a).all() and torch.isfinite(lora_b).all()

    def test_training_aware_weight_reconstruction(self, simple_model, synthetic_data):
        """
        Core TaRA claim: BA retains the components of the base weight W0 that contribute most to the full-rank
        gradient, so that the gradient at the initialized point closely approximates the full-rank gradient.

        The paper approximates the gradient change induced by replacing W0 with BA as
        `Σ_G (BA - W0) Σ_X`, where `Σ_X = E[x xᵀ]` and `Σ_G = E[δ δᵀ]` are the activation and output-gradient
        second moments. We measure the relative error of BA in this covariance-weighted metric; the default
        initialization (BA = 0) has a relative error of 1 by definition.
        """
        torch.manual_seed(42)
        peft_model = make_peft_model(copy.deepcopy(simple_model))

        def calibrate_step():
            for _ in range(4):
                x_cal, y_cal = synthetic_data()
                torch.nn.functional.mse_loss(peft_model(x_cal), y_cal).backward()

        initialize_tara_weights(peft_model, calibrate_step)

        _, lora_layer = next(get_target_lora_layers(peft_model))
        a = lora_layer.lora_A["default"].weight.detach()
        b = lora_layer.lora_B["default"].weight.detach()
        scaling = lora_layer.scaling["default"]
        ba = scaling * (b @ a)
        w0 = simple_model.weight.detach()

        # independently recomputed calibration statistics (mean second moments)
        sigma_x = torch.zeros(IN_FEATURES, IN_FEATURES)
        sigma_g = torch.zeros(OUT_FEATURES, OUT_FEATURES)
        n_tokens = 0
        for _ in range(4):
            x_cal, y_cal = synthetic_data()
            delta = 2 * (simple_model(x_cal) - y_cal) / y_cal.numel()  # d(mse_loss)/d(output), per token
            sigma_x += x_cal.T @ x_cal
            sigma_g += delta.T @ delta
            n_tokens += x_cal.shape[0]
        sigma_x /= n_tokens
        sigma_g /= n_tokens

        reference = torch.linalg.norm(sigma_g @ w0 @ sigma_x)
        error = torch.linalg.norm(sigma_g @ (ba - w0) @ sigma_x) / reference
        assert error < 0.1
