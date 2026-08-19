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
This module contains the implementation of the sMuon optimizer for low-rank adapters.
"""

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from ..peft_model import PeftModel


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize G via the quintic Newton-Schulz iteration used by Muon.

    Uses matmul operations only, no SVD or other decomposition routines. Returns a matrix whose singular values are
    pushed towards 1, computed in float32 for numerical stability.
    """
    # coefficients from the Muon reference implementation, selected to maximize the slope at 0
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class SMuonOptimizer(Optimizer):
    """
    sMuon optimizer: approximate Muon for low-rank adapters.

    Approximate Muon with low-rank adapters: https://arxiv.org/abs/2608.14492

    Muon orthogonalizes the full weight update, which is not mathematically possible for the rank-constrained update
    of a low-rank adapter. sMuon relaxes the Muon objective: it linearizes the adapter update `B @ A` in the factors
    `(B, A)`, orthogonalizes the resulting full-size update direction with Newton-Schulz iterations (matmul only), and
    pulls the orthogonalized direction back onto the factors with a least-squares step.

    Use `create_smuon_optimizer` to construct an instance from a `PeftModel` instead of instantiating this class
    directly.

    Args:
        params: Parameter groups, as produced by `create_smuon_optimizer`.
        lr (`float`): Learning rate for the low-rank factor updates.
        adam_lr (`float`): Learning rate for the remaining parameters, which are updated with AdamW.
        momentum (`float`): Momentum coefficient for the factor gradients.
        nesterov (`bool`): Whether to use Nesterov-style momentum for the factor gradients.
        ns_steps (`int`): Number of Newton-Schulz iterations used to orthogonalize the update direction.
        split (`float`):
            Fraction of the orthogonalized update applied to the A factor; the remaining fraction is applied to the B
            factor. 0.5 distributes the update evenly.
        weight_decay (`float`): Decoupled weight decay applied to the AdamW-updated parameters.
        betas (`tuple[float, float]`): AdamW betas for the non-factor parameters.
        eps (`float`): AdamW epsilon for the non-factor parameters.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        adam_lr: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        split: float = 0.5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "adam_lr": adam_lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "split": split,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_smuon", False):
                self._smuon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _smuon_step(self, group) -> None:
        # group["params"] holds exactly one (B, A) low-rank factor pair
        param_b, param_a = group["params"]
        if param_b.grad is None or param_a.grad is None:
            return

        momentum = group["momentum"]
        grad_a = self._momentum_buffer(param_a, momentum, group["nesterov"])
        grad_b = self._momentum_buffer(param_b, momentum, group["nesterov"])

        # Linearize the adapter update B @ A in the factors: updating the factors along their gradients induces the
        # full-size direction dB @ A + B @ dA on the merged weight.
        direction = grad_b @ param_a + param_b @ grad_a
        # Orthogonalize the relaxed full-size direction, matmul-only. Scale as in Muon to match the update RMS norm
        # of an AdamW-style update.
        update = zeropower_via_newtonschulz5(direction, steps=group["ns_steps"])
        update = update * max(1.0, param_b.size(0) / param_a.size(1)) ** 0.5

        # Least-squares pull-back onto the factors: solve min ||dB @ A + B @ dA - update||_F approximately, using the
        # transposes in place of the pseudo-inverses (matmul-only). The `split` fraction distributes the update over
        # the two factors so that the induced full-size step stays close to `lr * update`.
        lr, split = group["lr"], group["split"]
        param_a.add_(param_b.mT @ update, alpha=-lr * split)
        param_b.add_(update @ param_a.mT, alpha=-lr * (1.0 - split))

    def _momentum_buffer(self, param: torch.Tensor, momentum: float, nesterov: bool) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param.grad)
        buffer = state["momentum_buffer"]
        buffer.mul_(momentum).add_(param.grad)
        if nesterov:
            return param.grad.add(buffer, alpha=momentum)
        return buffer

    def _adamw_step(self, group) -> None:
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = (exp_avg_sq / bias_correction2).sqrt_().add_(group["eps"])
            if group["weight_decay"] != 0.0:
                param.mul_(1.0 - group["adam_lr"] * group["weight_decay"])
            param.addcdiv_(exp_avg, denom, value=-group["adam_lr"] / bias_correction1)


def create_smuon_optimizer(
    model: PeftModel,
    *,
    lr: float = 1e-3,
    adam_lr: float = 1e-4,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    split: float = 0.5,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    factor_names: tuple[str, str] = ("lora_A", "lora_B"),
) -> SMuonOptimizer:
    """
    Creates a sMuon optimizer for a PEFT model with low-rank adapters.

    Approximate Muon with low-rank adapters: https://arxiv.org/abs/2608.14492

    Trainable parameters that form a low-rank factor pair (e.g. `lora_A`/`lora_B`) are updated with the sMuon rule
    (orthogonalized update direction, pulled back onto the factors); all other trainable parameters (embeddings,
    biases, norms, non-paired adapter weights) are updated with AdamW.

    Note: with LoRA's default zero initialization of the B factor, the very first update only moves the B factor — the
    A factor's update is pulled back through B and therefore vanishes until B becomes non-zero.

    Args:
        model (`PeftModel`): The model to be optimized.
        lr (`float`): Learning rate for the low-rank factor updates.
        adam_lr (`float`): Learning rate for the remaining (AdamW-updated) parameters.
        momentum (`float`): Momentum coefficient for the factor gradients.
        nesterov (`bool`): Whether to use Nesterov-style momentum for the factor gradients.
        ns_steps (`int`): Number of Newton-Schulz iterations used to orthogonalize the update direction.
        split (`float`): Fraction of the orthogonalized update applied to the A factor.
        weight_decay (`float`): Decoupled weight decay for the AdamW-updated parameters.
        betas (`tuple[float, float]`): AdamW betas for the non-factor parameters.
        eps (`float`): AdamW epsilon for the non-factor parameters.
        factor_names (`tuple[str, str]`):
            Substrings identifying the A and B factors of a low-rank pair in the parameter names. Defaults to the LoRA
            naming convention; other low-rank tuners can be supported by passing their factor names.

    Returns:
        `SMuonOptimizer`: An optimizer instance with one sMuon parameter group per low-rank factor pair and one AdamW
        group for the remaining trainable parameters.
    """
    name_a, name_b = factor_names
    named_params = {name: param for name, param in model.named_parameters() if param.requires_grad}

    paired_names: set[str] = set()
    param_groups = []
    for name, param in named_params.items():
        if name_a not in name or param.ndim != 2:
            continue
        partner_name = name.replace(name_a, name_b)
        partner = named_params.get(partner_name)
        if partner is None or partner.ndim != 2:
            continue
        # sanity check: the factors must multiply to a full-size update, A: (r, in), B: (out, r)
        if partner.shape[1] != param.shape[0]:
            continue
        param_groups.append({"params": [partner, param], "use_smuon": True})
        paired_names.update({name, partner_name})

    remaining = [param for name, param in named_params.items() if name not in paired_names]
    if remaining:
        param_groups.append({"params": remaining, "use_smuon": False})

    if not any(group["use_smuon"] for group in param_groups):
        raise ValueError(
            f"No low-rank factor pairs matching {factor_names} found among the trainable parameters. sMuon requires "
            "a PEFT model with low-rank adapters, e.g. LoRA."
        )

    return SMuonOptimizer(
        param_groups,
        lr=lr,
        adam_lr=adam_lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        split=split,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
