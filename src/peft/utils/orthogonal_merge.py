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

"""Orthogonal-subspace merging of task vectors.

Adapted from "Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging"
(https://arxiv.org/abs/2505.22934). The paper attributes the performance drop observed when merging
LoRA adapters to interference between the subspaces that the per-task LoRA updates act on. Its remedy
is to project each task's update onto the orthogonal complement of the other tasks' subspaces before
combining them, so that directions used by one task are not disturbed by the others.
"""

import torch


def _subspace_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal bases for the column space and row space of a 2D matrix."""
    # SVD is computed in float32 for numerical stability, also for low precision inputs.
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    if s.numel() == 0 or s.max() == 0:
        empty_cols = torch.zeros(matrix.shape[0], 0, dtype=matrix.dtype, device=matrix.device)
        empty_rows = torch.zeros(matrix.shape[1], 0, dtype=matrix.dtype, device=matrix.device)
        return empty_cols, empty_rows
    tol = s.max() * max(matrix.shape) * torch.finfo(s.dtype).eps
    keep = s > tol
    column_basis = u[:, keep].to(matrix.dtype)
    row_basis = vh[keep, :].T.to(matrix.dtype)
    return column_basis, row_basis


def _projection_onto_span(basis: torch.Tensor) -> torch.Tensor:
    """Return the orthogonal projection matrix onto the span of the columns of `basis`."""
    if basis.shape[1] == 0:
        return torch.zeros(basis.shape[0], basis.shape[0], dtype=basis.dtype, device=basis.device)
    # orthonormalize with QR, then P = Q @ Q.T
    q, _ = torch.linalg.qr(basis.to(torch.float32))
    return (q @ q.T).to(basis.dtype)


def orthogonal_subspace_merge(task_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    Merge task tensors after removing, from each task tensor, the components that lie in the subspaces spanned
    by the other task tensors.

    Each task tensor is first flattened to a 2D matrix. For every task, the union of the column spaces and the
    union of the row spaces of all *other* task tensors are computed via SVD, and the task tensor is projected
    onto the orthogonal complement of both unions:

        W_i' = (I - P_out_i) @ W_i @ (I - P_in_i)

    where P_out_i / P_in_i are the projections onto the other tasks' output / input subspaces. The projected
    tensors are then combined with weighted summation, as in task arithmetic.

    Args:
        task_tensors (`List[torch.Tensor]`): The task tensors (e.g. LoRA delta weights) to merge.
        weights (`torch.Tensor`): The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    if len(task_tensors) == 1:
        # with a single task there is no other subspace to orthogonalize against
        weights = weights.to(task_tensors[0].device)
        return task_tensors[0] * weights[0]

    original_shapes = [tensor.shape for tensor in task_tensors]
    matrices = [tensor.reshape(tensor.shape[0], -1) for tensor in task_tensors]

    column_bases = []
    row_bases = []
    for matrix in matrices:
        column_basis, row_basis = _subspace_basis(matrix)
        column_bases.append(column_basis)
        row_bases.append(row_basis)

    orthogonalized = []
    for i, matrix in enumerate(matrices):
        other_column_basis = torch.cat([b for j, b in enumerate(column_bases) if j != i], dim=1)
        other_row_basis = torch.cat([b for j, b in enumerate(row_bases) if j != i], dim=1)
        proj_out = _projection_onto_span(other_column_basis)
        proj_in = _projection_onto_span(other_row_basis)
        # (I - P_out) @ W @ (I - P_in), computed in the dtype of the inputs
        orthogonalized.append(matrix - proj_out @ matrix - matrix @ proj_in + proj_out @ matrix @ proj_in)

    merged = torch.stack(orthogonalized, dim=0)
    weights = weights.view(-1, 1, 1).to(merged.device)
    merged = (merged * weights).sum(dim=0)
    return merged.reshape(original_shapes[0])
