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

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import SuperTuningConfig


def compute_support_mask(
    weight: torch.Tensor,
    density: float,
    activation_norms: Optional[torch.Tensor] = None,
    select_lowest: bool = False,
) -> torch.Tensor:
    """
    Compute the fixed sparse support of a weight matrix from a pruning-style saliency score.

    The score is `|W|` (magnitude-only, PaFi-style) or `|W| * ||X_j||_2` (Wanda-style, activation-weighted) when
    `activation_norms` (one L2 norm per input feature) is provided. The support contains the `density` fraction of
    entries with the highest scores, or the lowest scores when `select_lowest` is True.
    """
    score = weight.detach().abs().float()
    if activation_norms is not None:
        score = score * activation_norms.detach().float().to(score.device)

    num_selected = max(1, round(density * score.numel()))
    num_selected = min(num_selected, score.numel())

    flat_mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    indices = torch.topk(score.flatten(), k=num_selected, largest=not select_lowest).indices
    flat_mask[indices] = True
    return flat_mask.reshape(score.shape)


class SuperTuningLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("super_tuning_delta",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("super_tuning_mask",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Map adapter_name -> sparse delta (full shape, but only the support receives gradient)
        self.super_tuning_delta = nn.ParameterDict({})
        # Fixed support masks; persistent so that calibration-dependent ("wanda") masks round-trip via save/load
        self.super_tuning_mask = BufferDict({}, persistent=True)
        # Per-adapter selection settings, needed to recompute masks after calibration
        self.density = {}
        self.select_lowest = {}

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if (
            hasattr(base_layer, "weight")
            and isinstance(base_layer.weight, torch.Tensor)
            and base_layer.weight.ndim == 2
        ):
            # For Linear-like modules, weight is [out_features, in_features]
            out_features, in_features = base_layer.weight.shape
        else:
            in_features, out_features = None, None

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name: str, config: SuperTuningConfig, **kwargs):
        """Add a new Super-Tuning adapter: fix the sparse support and create the trainable delta."""
        weight = self.get_base_layer().weight.data
        self.density[adapter_name] = config.density
        self.select_lowest[adapter_name] = config.select_lowest
        self.super_tuning_mask[adapter_name] = compute_support_mask(
            weight, config.density, select_lowest=config.select_lowest
        ).to(weight.device)
        self.super_tuning_delta[adapter_name] = nn.Parameter(torch.zeros_like(weight))
        self.set_adapter(self.active_adapters)

    def update_support(self, adapter_name: str, activation_norms: torch.Tensor) -> None:
        """Recompute the support of an adapter with Wanda-style activation-weighted saliency."""
        if adapter_name not in self.super_tuning_mask:
            raise ValueError(f"Adapter '{adapter_name}' not found on this layer.")
        weight = self.get_base_layer().weight.data
        self.super_tuning_mask[adapter_name] = compute_support_mask(
            weight,
            self.density[adapter_name],
            activation_norms=activation_norms,
            select_lowest=self.select_lowest[adapter_name],
        ).to(weight.device)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return the sparse update of the given adapter, zero outside the fixed support."""
        return self.super_tuning_mask[adapter_name] * self.super_tuning_delta[adapter_name]

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge the active adapter's sparse update into the base weights."""
        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter not in self.super_tuning_delta.keys():
                continue
            base_layer = self.get_base_layer()
            delta = self.get_delta_weight(active_adapter)
            if safe_merge:
                orig_weight = base_layer.weight.data.clone()
                new_weight = orig_weight + delta.to(orig_weight.dtype)
                if not torch.isfinite(new_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = new_weight
            else:
                base_layer.weight.data += delta.to(base_layer.weight.dtype)
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Remove all merged sparse updates from the base weights."""
        if not self.merged:
            return
        for active_adapter in list(self.merged_adapters):
            if active_adapter in self.super_tuning_delta.keys():
                base_layer = self.get_base_layer()
                base_layer.weight.data -= self.get_delta_weight(active_adapter).to(base_layer.weight.dtype)
                self.merged_adapters.remove(active_adapter)

    def check_adapters_to_merge(self, adapter_names: Optional[list[str]] = None) -> list[str]:
        # Kept for API parity with other tuners; merging is always allowed here.
        if adapter_names is None:
            adapter_names = self.active_adapters
        return [adapter_name for adapter_name in adapter_names if adapter_name in self.super_tuning_delta.keys()]


class Linear(nn.Module, SuperTuningLayer):
    # Super-Tuning implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: SuperTuningConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        SuperTuningLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            return self.base_layer(x, *args, **kwargs)

        active_deltas = [
            self.get_delta_weight(adapter) for adapter in self.active_adapters if adapter in self.super_tuning_delta
        ]
        if not active_deltas:
            return self.base_layer(x, *args, **kwargs)

        base_layer = self.get_base_layer()
        orig_dtype = x.dtype
        x = self._cast_input_dtype(x, active_deltas[0].dtype)
        weight = base_layer.weight.to(active_deltas[0].dtype) + torch.stack(active_deltas).sum(0)
        bias = base_layer.bias
        if bias is not None:
            bias = bias.to(weight.dtype)
        result = F.linear(x, weight, bias)
        return result.to(orig_dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "super_tuning." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    config: SuperTuningConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, config=config, **kwargs)

    return new_module
