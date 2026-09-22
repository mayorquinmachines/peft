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

# Reference paper: "TaRA: Training-Aware Low-Rank Adaptation Initialization"
# https://arxiv.org/abs/2609.02639

from collections.abc import Callable

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora.layer import LoraLayer


def get_target_lora_layers(model: nn.Module, adapter_name: str = "default"):
    """
    Iterate over the LoRA adapter layers of a PEFT model whose base layer is `nn.Linear` or `Conv1D`.
    """
    for name, module in model.named_modules():
        if (
            isinstance(module, LoraLayer)
            and adapter_name in module.lora_A
            and isinstance(module.base_layer, (nn.Linear, Conv1D))
        ):
            yield name, module


def _get_base_weight(module: LoraLayer) -> torch.Tensor:
    """Return the base weight of a LoRA layer as an `(out_features, in_features)` matrix."""
    if isinstance(module.base_layer, Conv1D):  # Conv1D stores the weight transposed
        return module.base_layer.weight.T
    return module.base_layer.weight


def _collect_covariances(model: nn.Module, target_modules: list, calibrate_step: Callable[[], None]):
    """
    Run `calibrate_step` while accumulating, for each target LoRA layer, the activation second moments `Σ_X = Σ x
    xᵀ` and the output-gradient second moments `Σ_G = Σ δ δᵀ` in float32.
    """
    sigma_x = {name: None for name, _ in target_modules}
    sigma_g = {name: None for name, _ in target_modules}

    def make_forward_hook(name):
        def forward_hook(module, args, output):
            x = args[0].detach()
            with torch.no_grad():
                x = x.reshape(-1, x.shape[-1]).float()
                cov = x.T @ x
                sigma_x[name] = cov if sigma_x[name] is None else sigma_x[name] + cov

        return forward_hook

    def make_backward_hook(name):
        def backward_hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            with torch.no_grad():
                g = g.reshape(-1, g.shape[-1]).float()
                cov = g.T @ g
                sigma_g[name] = cov if sigma_g[name] is None else sigma_g[name] + cov

        return backward_hook

    hooks = []
    for name, module in target_modules:
        hooks.append(module.register_forward_hook(make_forward_hook(name)))
        hooks.append(module.register_full_backward_hook(make_backward_hook(name)))

    try:
        with torch.enable_grad():
            calibrate_step()
    finally:
        for hook in hooks:
            hook.remove()

    missing = [name for name, _ in target_modules if sigma_g[name] is None]
    if missing:
        raise ValueError(
            "No gradients were observed for the following target modules during calibration: "
            f"{missing}. Ensure that `calibrate_step` performs backward passes through the PEFT model."
        )
    return sigma_x, sigma_g


@torch.no_grad()
def _compute_tara_factors(sigma_x, sigma_g, base_weight, rank, damping_factor):
    """
    Compute the TaRA factors from the damped covariances and the base weight.

    With `Ũ S̃ Ṽᵀ` the truncated SVD of `Σ_G W₀ Σ_X`, returns `B = Σ_G⁻¹ Ũ[:, :r] S̃[:r]^{1/2}` and `A = S̃[:r]^{1/2}
    Ṽ[:, :r]ᵀ Σ_X⁻¹` in float32.
    """
    device = base_weight.device
    sigma_x = sigma_x.to(device)
    sigma_g = sigma_g.to(device)
    weight = base_weight.float()

    # relative diagonal damping: Σ ← Σ + c · mean_eig(Σ) · I
    sigma_x = sigma_x + damping_factor * (torch.trace(sigma_x) / sigma_x.shape[0]) * torch.eye(
        sigma_x.shape[0], device=device
    )
    sigma_g = sigma_g + damping_factor * (torch.trace(sigma_g) / sigma_g.shape[0]) * torch.eye(
        sigma_g.shape[0], device=device
    )

    u, s, vh = torch.linalg.svd(sigma_g @ weight @ sigma_x, full_matrices=False)
    k = min(rank, s.numel())
    sqrt_s = s[:k].clamp_min(0).sqrt()

    # solve against the symmetric covariances instead of forming explicit inverses
    a = torch.linalg.solve(sigma_x, (sqrt_s.unsqueeze(1) * vh[:k]).T).T  # (k, in_features)
    b = torch.linalg.solve(sigma_g, u[:, :k] * sqrt_s.unsqueeze(0))  # (out_features, k)

    # pad with zeros if the spectrum was shorter than the requested rank
    a_full = torch.zeros(rank, weight.shape[1], device=device)
    b_full = torch.zeros(weight.shape[0], rank, device=device)
    a_full[:k] = a
    b_full[:, :k] = b
    return a_full, b_full


def initialize_tara_weights(
    model: nn.Module,
    calibrate_step: Callable[[], None],
    adapter_name: str = "default",
    damping_factor: float = 1e-2,
):
    """
    Initialize the LoRA weights of a PEFT model with training-aware initialization (TaRA).

    TaRA initializes the low-rank factors such that the gradients they induce at the start of training closely
    approximate the gradient of the corresponding full-rank weight matrix. From one calibration pass over the
    frozen base model, the activation second moments `Σ_X = E[x xᵀ]` and the output-gradient second moments `Σ_G =
    E[δ δᵀ]` are collected for every targeted linear layer. With the truncated SVD `Ũ S̃ Ṽᵀ` of the
    covariance-weighted base weight `Σ_G W₀ Σ_X`, the factors are set to `B = Σ_G⁻¹ Ũ[:, :r] S̃[:r]^{1/2}` and `A =
    S̃[:r]^{1/2} Ṽ[:, :r]ᵀ Σ_X⁻¹`, and the frozen base weight is replaced by the residual `W₀ - BA`. The model
    output is therefore unchanged at initialization while BA carries the components of W₀ that matter most for
    training.

    Args:
        model (`nn.Module`):
            PEFT model with a freshly initialized LoRA adapter, i.e. with `lora_B` still zero as produced by the
            default `init_lora_weights=True`.
        calibrate_step (`Callable[[], None]`):
            Callback that runs forward and backward passes on calibration data (the paper uses ~256 samples).
            Statistics are accumulated over all backward passes issued inside this callback.
        adapter_name (`str`):
            Name of the LoRA adapter to initialize.
        damping_factor (`float`):
            Relative diagonal damping `c` of the covariances, `Σ ← Σ + c · mean_eig(Σ) · I`, for numerically
            stable covariance solves.

    Example:
        ```python
        peft_model = get_peft_model(base_model, lora_config)  # default init: lora_B == 0


        def calibrate_step():
            for batch in calibration_dataloader:
                loss = peft_model(**batch).loss
                loss.backward()


        initialize_tara_weights(peft_model, calibrate_step)
        # proceed with regular fine-tuning
        ```
    """
    target_modules = list(get_target_lora_layers(model, adapter_name))
    if not target_modules:
        raise ValueError(
            "No LoRA layers wrapping nn.Linear or Conv1D found for adapter "
            f"'{adapter_name}'. Call `initialize_tara_weights` on a PEFT model created with `get_peft_model`."
        )

    for name, module in target_modules:
        if hasattr(module.base_layer, "quant_state"):
            raise ValueError(
                f"TaRA does not support quantized base layers, found quantized module: '{name}'. "
                "TaRA requires full-precision weights and gradients during calibration."
            )
        if module.use_dora.get(adapter_name, False):
            raise ValueError("TaRA initialization is not compatible with DoRA.")
        if module.lora_B[adapter_name].weight.detach().abs().sum() > 0:
            raise ValueError(
                "TaRA expects a freshly initialized LoRA adapter with `lora_B == 0` (the default "
                "`init_lora_weights=True`), otherwise the calibration statistics do not reflect the base model."
            )

    was_training = model.training
    model.train()

    sigma_x, sigma_g = _collect_covariances(model, target_modules, calibrate_step)

    with torch.no_grad():
        for name, module in target_modules:
            rank = module.r[adapter_name]
            a, b = _compute_tara_factors(sigma_x[name], sigma_g[name], _get_base_weight(module), rank, damping_factor)

            # store A / scaling so that the effective update scaling * (B @ A) equals B @ (A_computed)
            scaling = float(module.scaling[adapter_name])
            lora_a_weight = module.lora_A[adapter_name].weight
            lora_b_weight = module.lora_B[adapter_name].weight
            lora_a_weight.copy_((a / scaling).to(lora_a_weight.dtype))
            lora_b_weight.copy_(b.to(lora_b_weight.dtype))

            # residual compensation: freeze W_res = W₀ - BA so the initial model output is preserved
            delta = (b @ a).to(module.base_layer.weight.dtype)
            if isinstance(module.base_layer, Conv1D):
                module.base_layer.weight.data -= delta.T
            else:
                module.base_layer.weight.data -= delta

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
