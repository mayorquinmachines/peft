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

"""Utilities for measuring and preserving weight sparsity when fine-tuning pruned base models.

Adapted from "SPP: Sparsity-Preserved Parameter-Efficient Fine-Tuning for Large Language Models"
(https://arxiv.org/abs/2405.16057). SPP's core observation is that merging a dense adapter update (e.g. LoRA's
low-rank BA product) into a pruned base weight re-densifies the pruned entries, silently discarding the memory
and latency benefits that pruning was supposed to deliver. Masking the update to the base weight's nonzero
support keeps the merged weight exactly as sparse as the pruned base.

This module provides the target-native pieces for the method_comparison harness:

- measurement of base weight sparsity and of "merge densification" (the fraction of pruned base entries that the
  active adapter update would fill back in when merged), which MetaMathQA/run.py logs as experiment metrics;
- `merge_preserving_sparsity`, a dense-tensor version of SPP's sparsity-preserving merge.

Compared to the paper, SPP's learnable column/row factor matrices are substituted with whichever PEFT method the
experiment already runs (the densification metric reads the adapter's delta weight through the tuner API), and
the sparse-CUDA-kernel training integration is out of scope -- all operations here are plain PyTorch mask ops.
"""

import torch
from torch import nn


# keys of the experiment metrics produced by `get_model_sparsity_metrics`; they follow the naming convention of
# the other MetaMathQA train metrics (see "train loss", "test accuracy", ...) and are consumed by
# method_comparison/processing.py
METRIC_KEY_BASE_WEIGHT_SPARSITY = "base weight sparsity"
METRIC_KEY_MERGE_DENSIFICATION = "merge densification"


def weight_sparsity(weight: torch.Tensor) -> float:
    """Return the fraction of exactly-zero entries of a weight tensor."""
    if weight.numel() == 0:
        return 0.0
    return (weight == 0).sum().item() / weight.numel()


def merge_preserving_sparsity(base_weight: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Merge an adapter update into a pruned base weight while preserving the base sparsity pattern.

    This is the SPP-style merge: entries of `update` that fall on pruned (zero) positions of `base_weight` are
    masked out, so the merged weight has exactly the same zero pattern as the pruned base.
    """
    if base_weight.shape != update.shape:
        raise ValueError(f"Shape mismatch: base weight {base_weight.shape} vs update {update.shape}")
    return base_weight + update * (base_weight != 0)


def get_merge_densification(model: nn.Module) -> float:
    """Return the fraction of pruned base entries that merging the active adapters would re-densify.

    Iterates over the tuner layers of a PEFT model (identified by their `get_delta_weight` API) and compares the
    nonzero support of each adapter's delta weight with the zero pattern of the corresponding base weight.
    Returns 0.0 for models without adapter layers or without pruned entries.
    """
    num_pruned = 0
    num_densified = 0
    for module in model.modules():
        get_delta_weight = getattr(module, "get_delta_weight", None)
        if get_delta_weight is None:
            continue
        base_weight = module.get_base_layer().weight
        pruned_mask = base_weight == 0
        if not pruned_mask.any():
            continue
        with torch.no_grad():
            densified_mask = torch.zeros_like(pruned_mask)
            for adapter in module.active_adapters:
                delta = get_delta_weight(adapter)
                if delta.shape == base_weight.shape:
                    densified_mask |= delta != 0
        num_pruned += pruned_mask.sum().item()
        num_densified += (pruned_mask & densified_mask).sum().item()
    if num_pruned == 0:
        return 0.0
    return num_densified / num_pruned


def get_model_sparsity_metrics(model: nn.Module) -> dict[str, float]:
    """Compute the sparsity metrics of a (possibly adapter-wrapped) model for the experiment result log.

    Returns a dict with the `METRIC_KEY_BASE_WEIGHT_SPARSITY` and `METRIC_KEY_MERGE_DENSIFICATION` entries.
    Sparsity is measured over all 2D weight matrices of the model; for PEFT models the adapter weights are
    included, which is negligible since they are a tiny fraction of the total parameters. Merge densification is
    only computed when the base model is actually pruned, since it requires materializing per-layer delta
    weights; for dense base models it is trivially 0.0.
    """
    num_zeros = 0
    num_elements = 0
    for param in model.parameters():
        if param.ndim != 2:
            continue
        num_elements += param.numel()
        num_zeros += (param == 0).sum().item()
    base_sparsity = num_zeros / num_elements if num_elements else 0.0
    if base_sparsity > 0:
        densification = get_merge_densification(model)
    else:
        densification = 0.0
    return {
        METRIC_KEY_BASE_WEIGHT_SPARSITY: base_sparsity,
        METRIC_KEY_MERGE_DENSIFICATION: densification,
    }
