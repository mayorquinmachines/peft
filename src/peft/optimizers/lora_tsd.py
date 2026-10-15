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
This module contains the implementation of the LoRA-TSD optimizer.
"""

from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from ..peft_model import PeftModel


def _ridged_inverse(gram: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Inverse of a small symmetric positive definite Gram matrix, stabilized with a ridge proportional to the average
    diagonal entry so that it stays well-defined at the rank-deficient LoRA initialization (lora_B = 0).
    """
    rank = gram.shape[0]
    ridge = eps * gram.diagonal().mean().clamp(min=1e-12)
    return torch.linalg.inv(gram + ridge * torch.eye(rank, device=gram.device, dtype=gram.dtype))


def _newton_schulz_orth(core: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Approximate the orthogonal (polar) factor msign(core) = U V^T of a small square matrix with the quintic
    Newton-Schulz iteration used by Muon.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = core / (core.norm() + 1e-7)
    for _ in range(steps):
        S = X @ X.T
        X = a * X + (b * S + c * (S @ S)) @ X
    return X


def _tangent_grad_factors(
    A: torch.Tensor, B: torch.Tensor, g_A: torch.Tensor, g_B: torch.Tensor, N_A: torch.Tensor, N_B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Factor the tangent-projected gradient P_T(G_W) = L @ R of the low-rank weight W = B @ A, using only the factor
    gradients g_A = B^T G_W and g_B = G_W A^T. With N_A = (A A^T)^-1, N_B = (B^T B)^-1 and the projectors
    Pi_B = B N_B B^T, Pi_A = A^T N_A A, this is P_T(G_W) = Pi_B G_W + G_W Pi_A - Pi_B G_W Pi_A in factored form.
    """
    cross = N_B @ g_A @ A.T @ N_A
    L = torch.cat([B, g_B @ N_A], dim=1)
    R = torch.cat([N_B @ g_A - cross @ A, A], dim=0)
    return L, R


def _tangent_project_factored(
    L: torch.Tensor, R: torch.Tensor, A: torch.Tensor, B: torch.Tensor, N_A: torch.Tensor, N_B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project X = L @ R onto the tangent space at W = B @ A, keeping the factored form: returns factors (of rank at
    most 2r) whose product is Pi_B X + X Pi_A - Pi_B X Pi_A, without forming the full m x n matrix.
    """
    proj_L = N_B @ B.T @ L
    proj_R = R @ A.T @ N_A
    new_L = torch.cat([B, L @ proj_R], dim=1)
    new_R = torch.cat([proj_L @ R - (proj_L @ proj_R) @ A, A], dim=0)
    return new_L, new_R


def _spectral_tangent_iterate(
    L: torch.Tensor,
    R: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    N_A: torch.Tensor,
    N_B: torch.Tensor,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One ball-projection iteration X <- P_T(msign(X)) in factored form. The orthogonalization acts on the small
    square core of X obtained from QR decompositions of the factors, so no full-size matrix is ever formed.
    """
    Q_L, R_L = torch.linalg.qr(L)
    Q_R, R_R = torch.linalg.qr(R.T)
    core = _newton_schulz_orth(R_L @ R_R.T, ns_steps)
    return _tangent_project_factored(Q_L @ core, Q_R.T, A, B, N_A, N_B)


class LoraTSDOptimizer(Optimizer):
    """
    Implements the LoRA-TSD optimizer, which treats every LoRA step as a tangent vector of the fixed-rank matrix
    manifold and takes the spectral-norm steepest-descent step of Muon inside that tangent space.

    For each LoRA factor pair (lora_A, lora_B) inducing the low-rank weight change W = B @ A, one step is:

    1. Reconstruct the tangent-projected gradient P_T(G_W) (the Riemannian gradient of the fixed-rank manifold) in
       factored form, from the momentum-smoothed factor gradients alone.
    2. Apply `tau` ball-projection iterations X <- P_T(msign(X)), where msign is the Muon orthogonalization
       computed with a Newton-Schulz iteration on a small 2r x 2r core matrix.
    3. Map the result back to the factors through a retraction native to the LoRA parametrization: solve for dA and
       dB such that dB @ A + B @ dA = -lr * X, using only small r x r Gram matrix inverses.

    The step avoids expensive operations on full weight matrices. Trainable parameters that are not part of a LoRA
    A/B factor pair (e.g. biases or modules_to_save) are updated with momentum SGD as a fallback.

    Args:
        params (Iterable[nn.parameter.Parameter]): Parameters to optimize.
        lr (float, optional): Learning rate, i.e. the spectral-norm step size (default: 1e-2).
        momentum (float, optional): Momentum factor for the factor gradients (default: 0.95).
        nesterov (bool, optional): Whether to use Nesterov momentum (default: True).
        tau (int, optional): Number of ball-projection iterations X <- P_T(msign(X)) per step (default: 3).
        ns_steps (int, optional): Number of Newton-Schulz iterations per orthogonalization (default: 5).
        eps (float, optional): Relative ridge for the small Gram matrix inverses (default: 1e-3).
        weight_decay (float, optional): Decoupled weight decay (default: 0.0).

    Args in sub-function step:
        closure (Callable, optional): A closure that reevaluates the model and returns the loss.

    Reference:
        - LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates:
          https://arxiv.org/abs/2609.02734v1
        - Reference implementation: https://github.com/brain-lab-research/LoRA-TSD

    Differences to the reference implementation: the tangent projection uses Gram-matrix projectors instead of QR
    orthonormal bases (algebraically identical at full rank, but well-defined at the rank-deficient LoRA
    initialization lora_B = 0), and auxiliary features of the reference (learning-rate schedulers, diagnostics,
    factor rebalancing, trust-region capping) are intentionally left out in favor of the standard PEFT/Trainer
    tooling.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        tau: int = 3,
        ns_steps: int = 5,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum} - should be in [0.0, 1.0)")
        if tau < 0:
            raise ValueError(f"Invalid tau value: {tau} - should be >= 0")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps} - should be >= 1")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be > 0.0")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "tau": tau,
            "ns_steps": ns_steps,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def _smoothed_grad(self, p: nn.parameter.Parameter, momentum: float, nesterov: bool) -> torch.Tensor:
        """Momentum-smoothed gradient of a single parameter, Muon-style."""
        grad = p.grad
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(grad)
        if nesterov:
            return grad.add(buf, alpha=momentum)
        return buf

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            params = group["params"]

            for index_A, index_B in group["pairs"]:
                p_A, p_B = params[index_A], params[index_B]
                if p_A.grad is None or p_B.grad is None:
                    continue
                g_A = self._smoothed_grad(p_A, momentum, nesterov)
                g_B = self._smoothed_grad(p_B, momentum, nesterov)

                # The geometry computations run in float32 for numerical stability.
                A = p_A.float()
                B = p_B.float()
                N_A = _ridged_inverse(A @ A.T, group["eps"])
                N_B = _ridged_inverse(B.T @ B, group["eps"])

                L, R = _tangent_grad_factors(A, B, g_A.float(), g_B.float(), N_A, N_B)
                for _ in range(group["tau"]):
                    L, R = _spectral_tangent_iterate(L, R, A, B, N_A, N_B, group["ns_steps"])

                # Retraction native to the LoRA parametrization: solve dB @ A + B @ dA = -lr * L @ R for dA, dB
                # without forming the m x n matrix L @ R.
                d_B = -lr * (L @ (R @ A.T) @ N_A)
                d_A = N_B @ (-lr * (B.T @ L) @ R - (B.T @ d_B) @ A)

                p_A.add_(d_A.to(p_A.dtype))
                p_B.add_(d_B.to(p_B.dtype))
                if weight_decay > 0.0:
                    p_A.add_(p_A, alpha=(-lr * weight_decay))
                    p_B.add_(p_B, alpha=(-lr * weight_decay))

            # Fallback for trainable parameters that are not part of a LoRA A/B factor pair.
            for index in group["others"]:
                p = params[index]
                if p.grad is None:
                    continue
                g = self._smoothed_grad(p, momentum, nesterov)
                p.add_(g, alpha=-lr)
                if weight_decay > 0.0:
                    p.add_(p, alpha=(-lr * weight_decay))

        return loss


def create_lora_tsd_optimizer(
    model: PeftModel,
    lr: float = 1e-2,
    momentum: float = 0.95,
    nesterov: bool = True,
    tau: int = 3,
    ns_steps: int = 5,
    eps: float = 1e-3,
    weight_decay: float = 0.0,
) -> Optimizer:
    """
    Helper function to instantiate a LoRA-TSD optimizer specifically configured for a given model using the LoRA
    method.

    This function will:
    - Collect all trainable parameters of the model.
    - Pair up "lora_A"/"lora_B" (and "lora_embedding_A"/"lora_embedding_B") parameters of the same adapter layer;
      each pair is updated with the tangent-space spectral descent step described in `LoraTSDOptimizer`.
    - Assign every other trainable parameter (e.g. biases, modules_to_save) to a momentum-SGD fallback group.

    In contrast to AdamW-style LoRA training, the update magnitude is normalized through the spectral
    orthogonalization, so the learning rate plays the role of a spectral-norm step size (as in Muon) rather than an
    AdamW learning rate. Learning-rate scheduling is left to the trainer, as usual in PEFT.

    Args:
        model (PeftModel): The model containing LoRA-adapted parameters.
        lr (float): Learning rate, i.e. the spectral-norm step size.
        momentum (float): Momentum factor for the factor gradients.
        nesterov (bool): Whether to use Nesterov momentum.
        tau (int): Number of ball-projection iterations X <- P_T(msign(X)) per step.
        ns_steps (int): Number of Newton-Schulz iterations per orthogonalization.
        eps (float): Relative ridge for the small Gram matrix inverses.
        weight_decay (float): Decoupled weight decay.

    Returns:
        Optimizer: Configured LoRA-TSD optimizer instance ready for training.
    """
    params = []
    slots_A = {}
    slots_B = {}
    others = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name or "lora_embedding_A" in name:
            key = name.replace("lora_embedding_A", "lora").replace("lora_A", "lora")
            slots_A[key] = len(params)
        elif "lora_B" in name or "lora_embedding_B" in name:
            key = name.replace("lora_embedding_B", "lora").replace("lora_B", "lora")
            slots_B[key] = len(params)
        else:
            others.append(len(params))
        params.append(param)

    pairs = []
    for key, index_A in slots_A.items():
        if key in slots_B:
            pairs.append((index_A, slots_B[key]))
        else:
            others.append(index_A)
    for key, index_B in slots_B.items():
        if key not in slots_A:
            others.append(index_B)

    param_groups = [{"params": params, "pairs": pairs, "others": others}]
    return LoraTSDOptimizer(
        param_groups,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        tau=tau,
        ns_steps=ns_steps,
        eps=eps,
        weight_decay=weight_decay,
    )
