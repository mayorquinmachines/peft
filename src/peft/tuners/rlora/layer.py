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

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import RLoraConfig


class RLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("rlora_A", "rlora_B", "rlora_route")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "rlora_num_heads", "scaling", "rlora_dropout", "rlora_head_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.rlora_num_heads = {}
        self.scaling = {}
        self.rlora_dropout = nn.ModuleDict({})
        # multi-head dropout probabilities, one per adapter (masks are sampled per forward pass)
        self.rlora_head_dropout = {}

        # Actual trainable parameters: a shared down-projection A, a stack of head matrices B and a router
        self.rlora_A = nn.ModuleDict({})
        self.rlora_B = nn.ParameterDict({})
        self.rlora_route = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Unsupported layer type {type(base_layer)}, only `torch.nn.Linear` is supported.")
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: RLoraConfig,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if config.rlora_num_heads < 2:
            raise ValueError(f"`rlora_num_heads` must be at least 2 but the value passed is {config.rlora_num_heads}")

        self.r[adapter_name] = r
        self.rlora_num_heads[adapter_name] = config.rlora_num_heads
        self.rlora_head_dropout[adapter_name] = config.rlora_head_dropout
        if config.rlora_dropout > 0.0:
            rlora_dropout_layer = nn.Dropout(p=config.rlora_dropout)
        else:
            rlora_dropout_layer = nn.Identity()
        self.rlora_dropout[adapter_name] = rlora_dropout_layer

        self.rlora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # all head matrices are stored in a single parameter of shape (num_heads, out_features, r)
        self.rlora_B[adapter_name] = nn.Parameter(
            torch.empty(config.rlora_num_heads, self.out_features, r), requires_grad=True
        )
        self.rlora_route[adapter_name] = nn.Linear(self.in_features, config.rlora_num_heads, bias=False)

        self.scaling[adapter_name] = config.rlora_alpha / r

        self.reset_rlora_parameters(
            adapter_name, init_rlora_weights=config.init_rlora_weights, gamma=config.init_gamma
        )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_rlora_parameters(self, adapter_name: str, init_rlora_weights: bool, gamma: float) -> None:
        if adapter_name not in self.rlora_A:
            return
        nn.init.kaiming_uniform_(self.rlora_A[adapter_name].weight, a=math.sqrt(5))
        # zero-init the router so that the head weights are exactly uniform at initialization
        nn.init.zeros_(self.rlora_route[adapter_name].weight)
        if init_rlora_weights:
            # multi-head random initialization from the paper: head weights are sampled from
            # (out_features ** 0.25 / sqrt(gamma)) * N(0, 1 / out_features)
            std = self.out_features ** (-0.25) / math.sqrt(gamma)
            nn.init.normal_(self.rlora_B[adapter_name], std=std)
            with torch.no_grad():
                # Mean-center the heads. Together with the uniform router at init this keeps the adapter's
                # initial contribution at exactly zero, just like the base-weight offset correction from the
                # paper, but without mutating the base weights (which keeps adapters portable across save/load).
                rlora_B = self.rlora_B[adapter_name]
                rlora_B.sub_(rlora_B.mean(dim=0, keepdim=True))
        else:
            nn.init.zeros_(self.rlora_B[adapter_name])


class RLoraLinear(nn.Module, RLoraLayer):
    """R-LoRA implemented in a dense layer."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: RLoraConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        RLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # The head combination weights are produced by an input-dependent router, so there is no static delta
        # weight that could be merged into the base layer.
        raise NotImplementedError("Merging is not supported for R-LoRA because the head router is input-dependent.")

    def unmerge(self) -> None:
        raise NotImplementedError("Merging is not supported for R-LoRA because the head router is input-dependent.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError(
            "R-LoRA does not have a static delta weight because the head router is input-dependent."
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.rlora_A.keys():
                    continue
                rlora_A = self.rlora_A[active_adapter]
                rlora_B = self.rlora_B[active_adapter]
                dropout = self.rlora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = self._cast_input_dtype(x, rlora_A.weight.dtype)

                # router weights over the heads, computed in float32 like the reference implementation
                route_weight = F.softmax(self.rlora_route[active_adapter](x), dim=-1, dtype=torch.float32)
                route_weight = route_weight.to(x.dtype)

                # shared low-rank representation
                hidden = rlora_A(dropout(x))

                head_dropout_p = self.rlora_head_dropout[active_adapter]
                if self.training and head_dropout_p > 0.0:
                    # multi-head dropout: sample an independent dropout mask per head on the low-rank
                    # representation (with inverted scaling), so that each head sees a diversified input
                    mask = F.dropout(
                        torch.ones(
                            *hidden.shape[:-1],
                            rlora_B.shape[0],
                            hidden.shape[-1],
                            dtype=hidden.dtype,
                            device=hidden.device,
                        ),
                        p=head_dropout_p,
                    )
                    head_out = torch.einsum("...nr,nor->...no", hidden.unsqueeze(-2) * mask, rlora_B)
                else:
                    head_out = torch.einsum("...r,nor->...no", hidden, rlora_B)

                result = result + (head_out * route_weight.unsqueeze(-1)).sum(dim=-2) * scaling

        result = result.to(previous_dtype)
        return result

    def scale_layer(self, scale: float) -> None:
        if scale != 1:
            warnings.warn("Scaling operation for R-LoRA not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        warnings.warn("Unscaling operation for R-LoRA not supported! Keeping scale at 1.")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rlora." + rep
