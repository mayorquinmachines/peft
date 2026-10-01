# Copyright 2025-present the HuggingFace Inc. team.
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

import torch

from peft.utils.merge_utils import prune


def _shared_subspace_basis(task_tensors: list[torch.Tensor], basis_size: int) -> torch.Tensor:
    """
    Compute an orthonormal basis for the subspace shared by all task tensors.

    Each task tensor is decomposed with an SVD and its top `basis_size` left singular vectors are kept. The
    concatenated per-task bases are decomposed once more to deduplicate overlapping directions, and the top
    `basis_size` left singular vectors of the concatenation form the shared basis.
    """
    per_task_bases = []
    for tensor in task_tensors:
        u, _, _ = torch.linalg.svd(tensor, full_matrices=False)
        per_task_bases.append(u[:, :basis_size])
    stacked_basis = torch.cat(per_task_bases, dim=1)
    u, _, _ = torch.linalg.svd(stacked_basis, full_matrices=False)
    return u[:, :basis_size]


def projective_merge(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float | None = None,
    num_steps: int = 400,
    lr: float = 1e-4,
    basis_size: int | None = None,
) -> torch.Tensor:
    """
    Merge the task tensors using adaptive projective gradient descent, as proposed in [Modeling Multi-Task Model
    Merging as Adaptive Projective Gradient Descent](https://arxiv.org/abs/2501.01230).

    Instead of sparsifying task vectors or enforcing orthogonality between them, which discards task-specific
    information, this method directly optimizes the merged task vector to stay close to every individual task
    vector, while constraining the update to the complement of the subspace shared by all tasks so that shared
    knowledge is preserved:

    1. Each task vector gets an adaptive coefficient `lambda_i = weights_i / ||task_tensors_i||`, where `weights`
       plays the role of the paper's global factor. This equalizes the influence of tasks whose task vectors have
       very different magnitudes.
    2. A shared subspace is built from the top left singular vectors of all task vectors.
    3. A modification vector, initialized at zero, is optimized for `num_steps` steps to minimize the squared gap
       `sum_j <task_tensors_j, sum_i lambda_i * (task_tensors_i + modification) - task_tensors_j>^2`. Before each
       optimizer step, the gradient is projected onto the orthogonal complement of the shared subspace.
    4. The merged task tensor is `sum_i lambda_i * (task_tensors_i + modification)`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors, acting as the global factor of the adaptive
            per-task coefficients.
        density (`float`, *optional*):
            The fraction of values to preserve per task tensor before merging, following the TIES-style trimming
            used in the paper. Should be in [0,1]. No pruning is performed when `None`.
        num_steps (`int`, *optional*, defaults to 400):
            The number of optimization steps performed on the modification vector.
        lr (`float`, *optional*, defaults to 1e-4):
            The learning rate of the Adam optimizer used on the modification vector.
        basis_size (`int`, *optional*):
            The number of singular vectors kept per task tensor and for the shared subspace. Defaults to the
            smallest task tensor dimension divided by the number of tasks (at least 1), following the paper's
            heuristic of dividing the task vector rank by the number of tasks.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    dtype = task_tensors[0].dtype
    original_shape = task_tensors[0].shape
    flat_tensors = [tensor.detach().reshape(original_shape[0], -1).float() for tensor in task_tensors]
    if density is not None:
        flat_tensors = [prune(tensor, density, method="magnitude") for tensor in flat_tensors]

    num_tasks = len(flat_tensors)
    if basis_size is None:
        basis_size = max(1, min(flat_tensors[0].shape) // num_tasks)
    basis = _shared_subspace_basis(flat_tensors, basis_size)

    weights = weights.to(device=flat_tensors[0].device, dtype=torch.float32)
    norms = torch.stack([tensor.norm() for tensor in flat_tensors])
    # a task vector with zero norm carries no information, its coefficient is irrelevant: set it to zero
    adaptive_coeffs = weights / torch.where(norms > 0, norms, torch.ones_like(norms))
    adaptive_coeffs = torch.where(norms > 0, adaptive_coeffs, torch.zeros_like(adaptive_coeffs))

    stacked = torch.stack(flat_tensors)
    modification = torch.zeros_like(flat_tensors[0], requires_grad=True)
    optimizer = torch.optim.Adam([modification], lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad()
        merged = (adaptive_coeffs.view(-1, 1, 1) * (stacked + modification)).sum(dim=0)
        # squared gap between the merged vector and each task vector, measured along that task vector
        gaps = ((stacked * (merged.unsqueeze(0) - stacked)).flatten(1).sum(dim=1) ** 2).sum()
        gaps.backward()
        with torch.no_grad():
            # project out the shared-subspace component of the gradient so shared knowledge is preserved
            modification.grad -= basis @ (basis.T @ modification.grad)
        optimizer.step()

    with torch.no_grad():
        merged = (adaptive_coeffs.view(-1, 1, 1) * (stacked + modification)).sum(dim=0)
    return merged.reshape(original_shape).to(dtype)
