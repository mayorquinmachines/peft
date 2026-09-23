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

# Reference paper: https://huggingface.co/papers/2608.31036

"""Column-normalized initialization of low-rank down-projections.

Implements the initialization-only variant of Normalized Low-Rank Adaptation (NoRA): after the standard
Kaiming-uniform initialization, each column of the LoRA down-projection A (the per-input-coordinate projection
vector along the rank dimension) is normalized to unit L2 norm. Since the up-projection B is initialized to zero,
the adapter remains an identity transform at step 0, while its early optimization dynamics -- which are largely
governed by the down-projection -- are regularized. This adds no trainable parameters and no inference-time cost.
"""

import torch


@torch.no_grad()
def normalize_lora_down_projection(weight: torch.Tensor, eps: float = 1e-12) -> None:
    """Normalize the columns of a LoRA down-projection weight to unit L2 norm, in place.

    Args:
        weight (`torch.Tensor`):
            The down-projection (LoRA A) weight, of shape `(r, in_features)` for linear layers or
            `(r, in_features, *kernel_size)` for convolutional layers. Normalization is applied along the rank
            dimension (dim 0), i.e. each per-input-coordinate projection vector is scaled to unit norm.
        eps (`float`):
            Lower bound on the column norm, for numerical stability.
    """
    weight /= weight.norm(dim=0, keepdim=True).clamp_min(eps)
