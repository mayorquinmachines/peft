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

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import RLoraConfig


class RLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("rlora_A", "rlora_B", "rlora_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("rlora_r", "rlora_num_heads", "rlora_scaling", "rlora_dropout", "rlora_head_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.rlora_r = {}
        self.rlora_num_heads = {}
        self.rlora_scaling = {}
        self.rlora_A = nn.ParameterDict({})
        self.rlora_B = nn.ParameterDict({})
        self.rlora_router = nn.ModuleDict({})
        self.rlora_dropout = nn.ModuleDict({})
        self.rlora_head_dropout = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: RLoraConfig,
        **kwargs,
    ) -> None:
        """Internal function to create the R-LoRA adapter.

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            config (`RLoraConfig`): The adapter configuration for this layer.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.rlora_r[adapter_name] = r
        self.rlora_num_heads[adapter_name] = config.num_heads
        self.rlora_scaling[adapter_name] = config.rlora_alpha / r

        if config.rlora_dropout > 0.0:
            self.rlora_dropout[adapter_name] = nn.Dropout(p=config.rlora_dropout)
        else:
            self.rlora_dropout[adapter_name] = nn.Identity()

        # Multi-Head Dropout: element-wise dropout on the low-rank intermediate `A x`, applied to the
        # per-head-expanded tensor so that each head receives an independently masked input (training only).
        if config.head_dropout > 0.0:
            self.rlora_head_dropout[adapter_name] = nn.Dropout(p=config.head_dropout)
        else:
            self.rlora_head_dropout[adapter_name] = nn.Identity()

        # one shared down-projection A (r x in_features); num_heads head matrices B_i stacked as (num_heads,
        # out_features, r); the optional router maps the input to per-head combination logits
        self.rlora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.rlora_B[adapter_name] = nn.Parameter(torch.empty(config.num_heads, self.out_features, r))
        if config.use_router:
            self.rlora_router[adapter_name] = nn.Linear(self.in_features, config.num_heads, bias=False)

        self.reset_rlora_parameters(adapter_name, config=config)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def reset_rlora_parameters(self, adapter_name: str, config: RLoraConfig) -> None:
        if adapter_name not in self.rlora_A.keys():
            return

        A = self.rlora_A[adapter_name]
        B = self.rlora_B[adapter_name]
        if adapter_name in self.rlora_router:
            # zero router logits -> uniform head combination at init
            nn.init.zeros_(self.rlora_router[adapter_name].weight)

        if not config.init_weights:
            # standard LoRA identity initialization: the adapter update is exactly zero at init
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            return

        base_weight = self.get_base_layer().weight
        if A.is_meta or B.is_meta or base_weight.is_meta:
            # tensors are on meta (e.g. low_cpu_mem_usage loading); the real values are loaded afterwards, so the
            # initialization and the base-weight compensation are skipped here.
            return

        # Multi-Head Random Initialization: draw A and all heads non-zero from scaled normal distributions (std
        # out_features**0.25 / (sqrt(init_gamma) * sqrt(fan)), following LoRA-GA). The generator is seeded from the
        # config so that the init — and thus the base-weight compensation below, which is not part of the saved
        # adapter state dict — is exactly reproduced when a saved adapter is re-created during loading.
        generator = torch.Generator(device=A.device)
        generator.manual_seed(config.init_seed)
        scale = self.out_features**0.25 / math.sqrt(config.init_gamma)
        nn.init.normal_(A, mean=0.0, std=scale / math.sqrt(self.in_features), generator=generator)
        nn.init.normal_(B, mean=0.0, std=scale / math.sqrt(self.out_features), generator=generator)

        # the non-zero init would otherwise change the output at init; subtract the initial delta from the base
        # weight (uniform head combination, matching the zero-initialized router) so that the adapted layer is an
        # exact identity at initialization
        delta0 = self.rlora_scaling[adapter_name] * (B.detach().mean(dim=0) @ A.detach())
        base_weight.data = base_weight.data - delta0.to(dtype=base_weight.dtype)


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
        self.update_layer(adapter_name, r, config=config, **kwargs)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return the additive delta of the adapter with uniform head combination: `scaling * mean_i(B_i) @ A`.

        The learned router's head weights are input-dependent and cannot be folded into a merged weight, so merging
        uses the uniform combination (equal to the router's initialization).
        """
        A = self.rlora_A[adapter_name]
        B = self.rlora_B[adapter_name]
        delta = self.rlora_scaling[adapter_name] * (B.mean(dim=0) @ A)
        return delta.to(self.get_base_layer().weight.dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

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
            if active_adapter not in self.rlora_A.keys():
                continue
            if active_adapter in self.rlora_router:
                warnings.warn(
                    f"Merging R-LoRA adapter {active_adapter!r} which uses a router: the router's input-dependent "
                    "head weights cannot be merged, the merged weight uses the uniform head combination instead."
                )
            delta_weight = self.get_delta_weight(active_adapter)
            if safe_merge:
                # Note that safe_merge will be slower than the normal merge because of the copy operation.
                orig_weights = base_layer.weight.data.clone()
                delta_weight = delta_weight.to(orig_weights.dtype)
                orig_weights += delta_weight
                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = orig_weights
            else:
                base_layer.weight.data = base_layer.weight.data + delta_weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.rlora_A.keys():
                continue
            delta_weight = self.get_delta_weight(active_adapter)
            base_layer = self.get_base_layer()
            base_layer.weight.data = base_layer.weight.data - delta_weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif not any(active_adapter in self.rlora_A.keys() for active_adapter in self.active_adapters):
            # no active R-LoRA adapter on this layer
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.rlora_A.keys():
                    continue
                A = self.rlora_A[active_adapter]
                B = self.rlora_B[active_adapter]
                num_heads = self.rlora_num_heads[active_adapter]

                x = self._cast_input_dtype(x, A.dtype)
                x_drop = self.rlora_dropout[active_adapter](x)
                hidden = F.linear(x_drop, A)  # (..., r)
                # expand the low-rank intermediate per head; the element-wise dropout then masks each head's copy
                # independently (Multi-Head Dropout). In eval mode this is an identity.
                hidden = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], num_heads, hidden.shape[-1])
                hidden = self.rlora_head_dropout[active_adapter](hidden)
                head_out = torch.einsum("...nr,nor->...no", hidden, B)  # (..., num_heads, out_features)

                if active_adapter in self.rlora_router:
                    # router head combination: omega = softmax(W_r x); softmax in float32 for stability
                    logits = self.rlora_router[active_adapter](x_drop)
                    omega = torch.softmax(logits.float(), dim=-1).to(head_out.dtype)
                    update = (head_out * omega.unsqueeze(-1)).sum(dim=-2)
                else:
                    update = head_out.mean(dim=-2)
                result = result + update.to(result.dtype) * self.rlora_scaling[active_adapter]

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rlora." + rep
