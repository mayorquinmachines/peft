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

from __future__ import annotations

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lora_tsd_optimizer
from peft.optimizers.lora_tsd import (
    _ridged_inverse,
    _tangent_grad_factors,
    _tangent_project_factored,
)

from .testing_utils import torch_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        X = self.lin1(X)
        return X


def _get_lora_model(bias="none"):
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["lin0", "lin1"], bias=bias)
    return get_peft_model(SimpleNet(), config)


def _one_step(model, optimizer, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    loss_value = torch.nn.CrossEntropyLoss()(output, label)
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_value


def test_lora_tsd_helper_success():
    """
    Test if the optimizer is correctly created and the LoRA factor pairs are identified.
    """
    model = _get_lora_model()
    optimizer = create_lora_tsd_optimizer(model=model, lr=1e-2)

    assert optimizer is not None
    group = optimizer.param_groups[0]
    # lin0 and lin1 each contribute one (lora_A, lora_B) pair
    assert len(group["pairs"]) == 2
    for index_A, index_B in group["pairs"]:
        assert group["params"][index_A].shape[0] == 8  # lora_A is (r, in_features)
        assert group["params"][index_B].shape[1] == 8  # lora_B is (out_features, r)


def test_lora_tsd_invalid_hyperparams():
    model = _get_lora_model()
    with pytest.raises(ValueError, match="Invalid learning rate"):
        create_lora_tsd_optimizer(model=model, lr=-1.0)
    with pytest.raises(ValueError, match="Invalid momentum"):
        create_lora_tsd_optimizer(model=model, momentum=1.5)


def test_tangent_grad_factors_match_dense_projection():
    """
    Test that the factored tangent-projected gradient L @ R equals the dense tangent projection P_T(G_W) =
    Pi_B G_W + G_W Pi_A - Pi_B G_W Pi_A of a full gradient G_W.
    """
    torch.manual_seed(0)
    m, r, n = 12, 4, 9
    A = torch.randn(r, n)
    B = torch.randn(m, r)
    G_W = torch.randn(m, n)
    g_A = B.T @ G_W
    g_B = G_W @ A.T

    N_A = torch.linalg.inv(A @ A.T)
    N_B = torch.linalg.inv(B.T @ B)
    proj_B = B @ N_B @ B.T
    proj_A = A.T @ N_A @ A
    dense = proj_B @ G_W + G_W @ proj_A - proj_B @ G_W @ proj_A

    L, R = _tangent_grad_factors(A, B, g_A, g_B, N_A, N_B)
    assert torch.allclose(L @ R, dense, atol=1e-4)


def test_tangent_project_factored_matches_dense_projection():
    """
    Test that the factored tangent projection of X = L0 @ R0 equals the dense tangent projection.
    """
    torch.manual_seed(0)
    m, r, n, k = 12, 4, 9, 5
    A = torch.randn(r, n)
    B = torch.randn(m, r)
    L0 = torch.randn(m, k)
    R0 = torch.randn(k, n)
    X = L0 @ R0

    N_A = torch.linalg.inv(A @ A.T)
    N_B = torch.linalg.inv(B.T @ B)
    proj_B = B @ N_B @ B.T
    proj_A = A.T @ N_A @ A
    dense = proj_B @ X + X @ proj_A - proj_B @ X @ proj_A

    L, R = _tangent_project_factored(L0, R0, A, B, N_A, N_B)
    assert torch.allclose(L @ R, dense, atol=1e-4)


def test_lora_tsd_step_updates_both_factors():
    """
    Test that the step function runs without exceptions, that the first step only updates lora_B (lora_A gets a zero
    update at the rank-deficient initialization B = 0, where the tangent space has no lora_A component), and that
    both factors are updated from the second step on.
    """
    model = _get_lora_model().to(torch_device)
    optimizer = create_lora_tsd_optimizer(model=model, lr=1e-2)

    initial_A = {name: param.clone() for name, param in model.named_parameters() if "lora_A" in name}
    initial_B = {name: param.clone() for name, param in model.named_parameters() if "lora_B" in name}
    for name, param in initial_B.items():
        assert torch.all(param == 0), f"lora_B weights not initialized to zero for {name}"

    _one_step(model, optimizer, seed=0)

    params = dict(model.named_parameters())
    for name, param in params.items():
        assert torch.all(torch.isfinite(param)), f"Non-finite weights for {name}"
        if "lora_A" in name:
            assert torch.equal(param, initial_A[name]), f"lora_A weights changed at B = 0 for {name}"
        elif "lora_B" in name:
            assert torch.any(param != 0), f"lora_B weights are still zero for {name}"

    _one_step(model, optimizer, seed=1)

    for name, param in params.items():
        if "lora_A" in name or "lora_B" in name:
            assert torch.all(torch.isfinite(param)), f"Non-finite weights for {name}"
    assert any(not torch.equal(params[name], initial_A[name]) for name in initial_A), (
        "lora_A weights never changed after lora_B became non-zero"
    )


def test_lora_tsd_step_matches_tangent_projected_gradient():
    """
    Test that without momentum and without ball-projection iterations (tau=0), one step induces the low-rank weight
    change prescribed by the retraction: the linearized update dB @ A + B @ dA reproduces -lr * P_T(G_W), the
    tangent-projected gradient, which is the paper's stationarity measure and the core of the method.
    """
    lr = 1e-4
    eps = 1e-6
    model = _get_lora_model().to(torch_device)
    optimizer = create_lora_tsd_optimizer(model=model, lr=lr, momentum=0.0, nesterov=False, tau=0, eps=eps)

    # Warm-up step so that lora_B is non-zero and the Gram matrices are invertible.
    _one_step(model, optimizer, seed=0)

    params = dict(model.named_parameters())
    name_A = next(name for name in params if "lora_A" in name)
    name_B = name_A.replace("lora_A", "lora_B")
    A0 = params[name_A].detach().clone()
    B0 = params[name_B].detach().clone()

    torch.manual_seed(1)
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    loss_value = torch.nn.CrossEntropyLoss()(output, label)
    loss_value.backward()
    g_A = params[name_A].grad.clone()
    g_B = params[name_B].grad.clone()
    optimizer.step()

    # Recompute the retraction solve that the optimizer should have applied.
    N_A = _ridged_inverse(A0 @ A0.T, eps)
    N_B = _ridged_inverse(B0.T @ B0, eps)
    L, R = _tangent_grad_factors(A0, B0, g_A, g_B, N_A, N_B)
    tangent_grad = L @ R
    d_B = -lr * (L @ (R @ A0.T) @ N_A)
    d_A = N_B @ (-lr * (B0.T @ L) @ R - (B0.T @ d_B) @ A0)

    # First order: the linearized update equals -lr times the tangent-projected gradient.
    assert torch.allclose(d_B @ A0 + B0 @ d_A, -lr * tangent_grad, atol=1e-8)

    # The induced weight change matches the retraction, up to the second-order term dB @ dA.
    delta_W = params[name_B].detach() @ params[name_A].detach() - B0 @ A0
    assert torch.allclose(delta_W, d_B @ A0 + B0 @ d_A + d_B @ d_A, atol=1e-8)


def test_lora_tsd_step_with_lora_bias():
    """
    Test that the optimizer also handles trainable parameters outside the LoRA factor pairs (here: LoRA biases) via
    the momentum-SGD fallback.
    """
    model = _get_lora_model(bias="lora_only").to(torch_device)
    optimizer = create_lora_tsd_optimizer(model=model, lr=1e-2)

    bias_names = [name for name, param in model.named_parameters() if param.requires_grad and "lora" not in name]
    assert bias_names, "Expected trainable non-LoRA parameters with bias='lora_only'"
    initial_biases = {name: param.clone() for name, param in model.named_parameters() if name in bias_names}

    for seed in range(3):
        _one_step(model, optimizer, seed=seed)

    params = dict(model.named_parameters())
    for name, param in params.items():
        if param.requires_grad:
            assert torch.all(torch.isfinite(param)), f"Non-finite weights for {name}"
    assert any(not torch.equal(params[name], initial_biases[name]) for name in bias_names), (
        "Non-LoRA trainable parameters were never updated"
    )
