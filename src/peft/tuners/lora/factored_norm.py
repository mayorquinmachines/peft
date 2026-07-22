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

"""Factored computation of the DoRA weight norm.

DoRA (https://huggingface.co/papers/2402.09353) requires the row-wise L2 norm of `W + scaling * BA` on every forward
pass, which is conventionally computed by materializing the dense `[out_features, in_features]` product `BA`. For
high-rank adapters on wide layers, this transient tensor dominates memory usage (e.g. ~512MB in bf16 for
`d_in = 8192, r = 384`), which can make high-rank DoRA costly or infeasible on single-GPU setups.

This module computes the identical norm in factored form, without ever materializing `BA`:

    ||W + s·BA||²_i = ||W_i||² + 2s·⟨W_i, (BA)_i⟩ + s²·||(BA)_i||²
    ⟨W_i, (BA)_i⟩   = ((W @ Aᵀ) * B).sum(-1)
    ||(BA)_i||²     = ((B @ (A @ Aᵀ)) * B).sum(-1)

reducing the transient memory from O(out_features * in_features) to O((out_features + in_features) * r).

Adapted from "Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels"
(https://arxiv.org/abs/2603.22276v1). Unlike the paper, no custom fused kernels are used — the factorization alone
already removes the memory bottleneck. The factored path is only used for large layers (see
`FACTORED_NORM_MIN_NUMEL`); smaller layers keep the original dense computation, which is bitwise-identical to
previous PEFT releases.
"""

import torch


FACTORED_NORM_MIN_NUMEL = 2**24
"""Minimum number of elements of the dense `lora_B @ lora_A` product for the factored weight norm to kick in.

Below this threshold (~16.7M elements, i.e. 64MB in fp32 / 32MB in bf16), materializing the dense product is cheap
enough and the original computation is kept for exact backwards compatibility.
"""


def use_factored_weight_norm(lora_A_weight: torch.Tensor, lora_B_weight: torch.Tensor) -> bool:
    """Whether to compute the DoRA weight norm in factored form instead of materializing `lora_B @ lora_A`."""
    dense_numel = lora_B_weight.shape[0] * lora_A_weight.shape[1]
    return dense_numel >= FACTORED_NORM_MIN_NUMEL


def factored_weight_norm(
    *, weight: torch.Tensor, lora_A_weight: torch.Tensor, lora_B_weight: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Row-wise L2 norm of `weight + scaling * (lora_B_weight @ lora_A_weight)` without dense materialization.

    Equivalent to `torch.linalg.norm(weight + scaling * lora_B_weight @ lora_A_weight, dim=1)` up to floating point
    associativity. The LoRA weights are detached, mirroring how the dense path detaches the delta weight before the
    norm computation (see section 4.3 of the DoRA paper).

    Args:
        weight (`torch.Tensor`): Base weight of shape `[out_features, in_features]`.
        lora_A_weight (`torch.Tensor`): LoRA A weight of shape `[r, in_features]`.
        lora_B_weight (`torch.Tensor`): LoRA B weight of shape `[out_features, r]`.
        scaling (`float`): LoRA scaling factor.

    Returns:
        `torch.Tensor`: The weight norm of shape `[out_features]`, same as the dense computation.
    """
    lora_A_weight = lora_A_weight.detach()
    lora_B_weight = lora_B_weight.detach()
    # ⟨W_i, (BA)_i⟩ = Σ_k B[i,k]·⟨W_i, A_k⟩
    inner = ((weight @ lora_A_weight.T) * lora_B_weight).sum(dim=1)
    # ||(BA)_i||² = Σ_kl B[i,k]·B[i,l]·⟨A_k, A_l⟩
    gram = lora_A_weight @ lora_A_weight.T
    lora_sq = ((lora_B_weight @ gram) * lora_B_weight).sum(dim=1)
    row_sq = weight.pow(2).sum(dim=1) + 2 * scaling * inner + scaling**2 * lora_sq
    # clamp against small negative values caused by floating point cancellation
    return row_sq.clamp_min(0).sqrt().to(weight.dtype)
