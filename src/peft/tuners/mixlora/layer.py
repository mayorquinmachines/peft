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
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import MixLoraConfig


class MixLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("mixlora_experts_A", "mixlora_experts_B", "mixlora_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "mixlora_r",
        "mixlora_alpha",
        "mixlora_scaling",
        "mixlora_top_k",
        "mixlora_aux_loss_coef",
        "mixlora_dropout",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.mixlora_r = {}
        self.mixlora_alpha = {}
        self.mixlora_scaling = {}
        self.mixlora_top_k = {}
        self.mixlora_aux_loss_coef = {}
        self.mixlora_experts_A = nn.ParameterDict({})
        self.mixlora_experts_B = nn.ParameterDict({})
        self.mixlora_router = nn.ModuleDict({})
        self.mixlora_dropout = nn.ModuleDict({})
        # the load-balancing auxiliary loss of the most recent training forward pass, summed over the active adapters
        self._router_aux_loss = None
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
        config: MixLoraConfig,
        **kwargs,
    ) -> None:
        """Internal function to create the MixLoRA adapter.

        Args:
            adapter_name (`str`): Name for the adapter to add.
            config (`MixLoraConfig`): The adapter configuration for this layer.
        """
        # The experts of one adapter are stored as single batched parameters of shape
        # (num_experts, r, in_features) and (num_experts, out_features, r), so all experts are applied with two
        # einsums instead of a Python loop over per-expert low-rank matrices.
        self.mixlora_r[adapter_name] = config.r
        self.mixlora_alpha[adapter_name] = config.lora_alpha
        self.mixlora_scaling[adapter_name] = config.lora_alpha / config.r
        self.mixlora_top_k[adapter_name] = config.top_k
        self.mixlora_aux_loss_coef[adapter_name] = config.router_aux_loss_coef

        if config.mixlora_dropout > 0.0:
            self.mixlora_dropout[adapter_name] = nn.Dropout(p=config.mixlora_dropout)
        else:
            self.mixlora_dropout[adapter_name] = nn.Identity()

        self.mixlora_experts_A[adapter_name] = nn.Parameter(
            torch.empty(config.num_experts, config.r, self.in_features)
        )
        self.mixlora_experts_B[adapter_name] = nn.Parameter(
            torch.empty(config.num_experts, self.out_features, config.r)
        )
        # one router per adapter and layer, trained jointly with the experts
        self.mixlora_router[adapter_name] = nn.Linear(self.in_features, config.num_experts, bias=False)

        self.reset_mixlora_parameters(adapter_name, init_weights=config.init_mixlora_weights)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def reset_mixlora_parameters(self, adapter_name: str, init_weights: bool = True) -> None:
        if adapter_name not in self.mixlora_experts_A.keys():
            return

        experts_A = self.mixlora_experts_A[adapter_name]
        experts_B = self.mixlora_experts_B[adapter_name]
        if experts_A.is_meta or experts_B.is_meta:
            # tensors are on meta (e.g. low_cpu_mem_usage loading); the real values are loaded afterwards, so any
            # initialization is skipped here.
            return

        # same initialization scheme as LoRA, applied per expert
        for expert_A in experts_A:
            nn.init.kaiming_uniform_(expert_A, a=math.sqrt(5))
        if init_weights:
            # identity initialization: zero B matrices make the adapter an exact no-op at the start of training
            nn.init.zeros_(experts_B)
        else:
            for expert_B in experts_B:
                nn.init.normal_(expert_B, mean=0.0, std=0.02)
        nn.init.normal_(self.mixlora_router[adapter_name].weight, mean=0.0, std=0.02)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # The MixLoRA update is input-dependent (per-token routing), so it cannot be folded into the base weight.
        raise ValueError(
            "MixLoRA layers cannot be merged into the base model: the mixture-of-experts update is input-dependent "
            "(per-token routing) and has no equivalent static delta weight."
        )

    def unmerge(self) -> None:
        raise ValueError("MixLoRA layers cannot be merged, hence there is nothing to unmerge.")

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.mixlora_experts_A.keys():
                continue
            self.mixlora_scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.mixlora_experts_A.keys():
                continue
            self.mixlora_scaling[active_adapter] /= scale


class MixLoraLinear(nn.Module, MixLoraLayer):
    """MixLoRA implemented in a dense layer: a mixture of LoRA experts with sparse per-token routing."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: MixLoraConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        MixLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config, **kwargs)

    def _route_experts(self, x: torch.Tensor, adapter_name: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the routed mixture-of-experts update of one adapter for the (flattened) input `x` of shape
        `(num_tokens, in_features)`.

        Returns the expert update of shape `(num_tokens, out_features)` and, when in training mode with
        `router_aux_loss_coef > 0`, the load-balancing auxiliary loss.
        """
        router = self.mixlora_router[adapter_name]
        experts_A = self.mixlora_experts_A[adapter_name]  # (num_experts, r, in_features)
        experts_B = self.mixlora_experts_B[adapter_name]  # (num_experts, out_features, r)
        top_k = self.mixlora_top_k[adapter_name]
        num_experts = experts_A.shape[0]

        router_logits = router(x.to(router.weight.dtype))  # (num_tokens, num_experts)
        # float32 softmax for numerically stable gates in half-precision training
        router_probs = F.softmax(router_logits.to(torch.float32), dim=-1)
        # sparse per-token routing: keep the top-k experts and re-normalize their probabilities to sum to 1
        topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        gates = torch.zeros_like(router_probs).scatter(dim=-1, index=topk_indices, src=topk_weights)

        # All experts are evaluated densely and their outputs masked by the (sparse) gates. This computes the same
        # result as the paper's grouped-GEMM dispatch of only the selected experts, trading compute for simplicity.
        x_drop = self.mixlora_dropout[adapter_name](x).to(experts_A.dtype)
        hidden = torch.einsum("ti,eri->ter", x_drop, experts_A)
        expert_outputs = torch.einsum("ter,eor->teo", hidden, experts_B)
        moe_output = torch.einsum("te,teo->to", gates.to(expert_outputs.dtype), expert_outputs)
        moe_output = moe_output * self.mixlora_scaling[adapter_name]

        aux_loss = None
        if self.training and self.mixlora_aux_loss_coef[adapter_name] > 0.0:
            # Switch-style load-balancing loss, generalized to top-k routing: f_e is the fraction of expert slots
            # dispatched to expert e and P_e is the mean router probability assigned to it. Minimizing
            # num_experts * sum_e f_e * P_e prevents expert collapse.
            dispatched = torch.zeros_like(router_probs).scatter(dim=-1, index=topk_indices, value=1.0 / top_k)
            tokens_fraction = dispatched.mean(dim=0)
            probs_mean = router_probs.mean(dim=0)
            aux_loss = num_experts * (tokens_fraction * probs_mean).sum() * self.mixlora_aux_loss_coef[adapter_name]

        return moe_output, aux_loss

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        self._router_aux_loss = None

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        elif not any(active_adapter in self.mixlora_experts_A.keys() for active_adapter in self.active_adapters):
            # no active MixLoRA adapter on this layer
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            x_flat = x.reshape(-1, x.shape[-1])
            for active_adapter in self.active_adapters:
                if active_adapter not in self.mixlora_experts_A.keys():
                    continue
                moe_output, aux_loss = self._route_experts(x_flat, active_adapter)
                result = result + moe_output.reshape(result.shape)
                if aux_loss is not None:
                    if self._router_aux_loss is None:
                        self._router_aux_loss = aux_loss
                    else:
                        self._router_aux_loss = self._router_aux_loss + aux_loss

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mixlora." + rep
