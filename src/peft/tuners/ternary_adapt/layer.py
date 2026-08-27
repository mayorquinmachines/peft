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

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import TernaryAdaptConfig


def ternarize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean ternarization per output channel (row) of a logical `(out_features, in_features)` weight.

    Returns `(ternary, scale)` such that `weight ~= ternary * scale` with `ternary` in `{-1, 0, +1}` and `scale` of
    shape `(out_features, 1)`.
    """
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)
    ternary = torch.clamp(torch.round(weight / scale), -1.0, 1.0)
    return ternary, scale


def ste_ternary(param: torch.Tensor) -> torch.Tensor:
    """Ternary quantization of `param` with a straight-through estimator, so gradients flow to the latent values."""
    scale = param.detach().abs().mean().clamp_min(torch.finfo(param.dtype).eps)
    quantized = torch.clamp(torch.round(param.detach() / scale), -1.0, 1.0)
    return param + (quantized - param.detach())


def default_block_shape(out_features: int, in_features: int) -> tuple[int, int]:
    """Near-square default Kronecker block shape `(rows, cols)` dividing `(out_features, in_features)`.

    Picks the greatest divisor of each dimension not exceeding its square root, which keeps the adapter parameter
    count close to the minimum for a Kronecker factorization of an `(out_features, in_features)` mask.
    """

    def greatest_divisor_at_most(n: int, limit: int) -> int:
        for d in range(min(limit, n), 0, -1):
            if n % d == 0:
                return d

    return greatest_divisor_at_most(out_features, math.isqrt(out_features)), greatest_divisor_at_most(
        in_features, math.isqrt(in_features)
    )


class TernaryAdaptLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("ternary_adapt_A", "ternary_adapt_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("ternary_adapt_block_shape",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.ternary_adapt_A = nn.ParameterDict({})
        self.ternary_adapt_B = nn.ParameterDict({})
        self.ternary_adapt_block_shape = {}
        # whether the base weight is stored transposed, i.e. (in_features, out_features) as in `Conv1D` (gpt-2)
        self.fan_in_fan_out = False
        # The multiplicative mask can zero out weight entries, so merging is not invertible. The (already
        # ternarized) pre-merge base weight is cached per merged adapter and restored on unmerge.
        self._cached_base_weight = {}
        self._is_base_ternarized = False
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            # Conv1D (e.g. gpt-2) stores its weight transposed as (in_features, out_features)
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name: str, config: TernaryAdaptConfig, **kwargs) -> None:
        """Internal function to create the ternary adaptation adapter.

        Args:
            adapter_name (`str`): Name for the adapter to add.
            config (`TernaryAdaptConfig`): The adapter configuration for this layer.
        """
        block_shape = config.block_shape or default_block_shape(self.out_features, self.in_features)
        block_rows, block_cols = block_shape
        if (self.out_features % block_rows != 0) or (self.in_features % block_cols != 0):
            raise ValueError(
                f"`block_shape` {block_shape} must divide the layer dimensions "
                f"({self.out_features}, {self.in_features})."
            )
        self.ternary_adapt_block_shape[adapter_name] = (block_rows, block_cols)
        self.fan_in_fan_out = config.fan_in_fan_out

        if config.ternarize_base:
            self._ternarize_base_weight()

        # A: (out_features // block_rows, in_features // block_cols), B: block_shape; kron(A, B) is the full mask.
        self.ternary_adapt_A[adapter_name] = nn.Parameter(
            torch.empty(self.out_features // block_rows, self.in_features // block_cols)
        )
        self.ternary_adapt_B[adapter_name] = nn.Parameter(torch.empty(block_rows, block_cols))
        self.reset_ternary_adapt_parameters(adapter_name, init_weights=config.init_weights)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def _ternarize_base_weight(self) -> None:
        """Ternarize the base weight in-place (absmean per output channel), turning the base layer into a ternary
        layer. Done once per layer, when the first adapter is added."""
        if self._is_base_ternarized:
            return

        base_weight = self.get_base_layer().weight
        if base_weight.is_meta:
            # tensors are on meta (e.g. low_cpu_mem_usage loading); ternarization needs real values, so it is
            # skipped here and the base weight is expected to be ternarized before the real weights are loaded.
            return

        with torch.no_grad():
            weight = transpose(base_weight.data.to(torch.float32), self.fan_in_fan_out)
            ternary, scale = ternarize_rows(weight)
            base_weight.data = transpose(ternary * scale, self.fan_in_fan_out).to(
                dtype=base_weight.dtype, device=base_weight.device
            )
        self._is_base_ternarized = True

    def reset_ternary_adapt_parameters(self, adapter_name: str, init_weights: bool = True) -> None:
        if adapter_name not in self.ternary_adapt_A.keys():
            return

        if init_weights:
            # all-ones latents ternarize to an all-ones mask: the adapter is an exact identity at init, so training
            # starts from the (ternarized) pretrained weights.
            nn.init.ones_(self.ternary_adapt_A[adapter_name])
            nn.init.ones_(self.ternary_adapt_B[adapter_name])
        else:
            # small random latents ternarize to a random mix of {-1, 0, +1} (non-identity at init)
            nn.init.normal_(self.ternary_adapt_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.ternary_adapt_B[adapter_name], mean=0.0, std=0.02)

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.ternary_adapt_A.keys():
                continue
            warnings.warn("Scaling operation for ternary adaptation not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.ternary_adapt_A.keys():
                continue
            warnings.warn("Unscaling operation for ternary adaptation not supported! Keeping scale at 1.")


class TernaryAdaptLinear(nn.Module, TernaryAdaptLayer):
    """Ternary multiplicative adaptation implemented in a dense layer."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: TernaryAdaptConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        TernaryAdaptLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config, **kwargs)

    def get_mask(self, adapter_name: str) -> torch.Tensor:
        """The multiplicative ternary mask `M = kron(A, B)` in logical `(out_features, in_features)` layout.

        Both factors pass through the straight-through ternary quantization, so `M` is ternary (`{-1, 0, +1}`) in
        the forward pass while gradients flow to the latent factor values.
        """
        A = ste_ternary(self.ternary_adapt_A[adapter_name])
        B = ste_ternary(self.ternary_adapt_B[adapter_name])
        return torch.kron(A, B)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return the additive delta such that `W + delta` equals the adapted weight `W (.) M`, i.e. `W (.) (M - 1)`.

        Used for compatibility with code paths that expect an additive delta; the forward pass and `merge` apply
        the mask multiplicatively instead, which is what keeps the adapted weight in the ternary domain.
        """
        weight = transpose(self.get_base_layer().weight.to(torch.float32), self.fan_in_fan_out)
        delta = weight * (self.get_mask(adapter_name).to(torch.float32) - 1.0)
        # delta is logical (out_features, in_features); transpose back to the base layer's storage order (Conv1D)
        return transpose(delta, self.fan_in_fan_out).to(self.get_base_layer().weight.dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        The merge is a plain element-wise multiplication of the (ternarized) base weight with the ternary mask, so
        the merged weight stays in the ternary domain -- no dequantization is involved at any point.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.ternary_adapt_A.keys():
                continue
            # mask entries are exactly -1/0/+1, so casting to the weight dtype is lossless and the merged weight
            # keeps the base layer's dtype
            mask = transpose(self.get_mask(active_adapter), self.fan_in_fan_out).to(base_layer.weight.dtype)
            # the multiplicative merge is not invertible (the mask can zero out entries), so cache the pre-merge
            # weight for an exact unmerge
            self._cached_base_weight[active_adapter] = base_layer.weight.data.clone()
            if safe_merge:
                new_weight = base_layer.weight.data.clone() * mask
                if not torch.isfinite(new_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = new_weight
            else:
                base_layer.weight.data = base_layer.weight.data * mask
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.ternary_adapt_A.keys():
                continue
            cached = self._cached_base_weight.pop(active_adapter, None)
            if cached is None:
                raise ValueError(
                    f"Cannot unmerge adapter {active_adapter}: the pre-merge base weight was not cached."
                )
            base_layer = self.get_base_layer()
            base_layer.weight.data = cached

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif not any(active_adapter in self.ternary_adapt_A.keys() for active_adapter in self.active_adapters):
            # no active ternary adapter on this layer
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Apply the mask on the weight: x @ (W (.) M).T. Multiple active adapters compose multiplicatively,
            # which keeps the combined mask (and hence the adapted weight) ternary as well.
            base_layer = self.get_base_layer()
            weight = transpose(base_layer.weight, self.fan_in_fan_out).to(torch.float32)
            mask = None
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ternary_adapt_A.keys():
                    continue
                adapter_mask = self.get_mask(active_adapter).to(torch.float32)
                mask = adapter_mask if mask is None else mask * adapter_mask
            weight = weight * mask

            x = self._cast_input_dtype(x, weight.dtype)
            result = F.linear(x, weight, bias=base_layer.bias)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ternary_adapt." + rep
