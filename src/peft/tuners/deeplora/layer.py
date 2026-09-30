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

from typing import Any

import torch
from torch import nn

from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import LoraLayer
from peft.utils.other import transpose

from .config import DeepLoraConfig


class DeepLoraLinear(LoraLinear):
    """
    Deep LoRA implemented in a dense layer.

    The outer factors of the chain reuse LoRA's `lora_A`/`lora_B` (so that merging, adapter switching, and
    state-dict handling behave like LoRA); the `num_factors - 2` intermediate (r, r) factors are stacked into a
    single parameter per adapter stored in `deeplora_factors`.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = LoraLayer.adapter_layer_names + ("deeplora_factors",)

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: DeepLoraConfig,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ) -> None:
        # replicate LoraLinear.__init__, but create `deeplora_factors` before `update_layer` is called
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        # intermediate (r, r) factors of the chain, stacked into one parameter of shape
        # (num_factors - 2, r, r) per adapter; stays empty for chains of depth 2 (i.e. plain LoRA)
        self.deeplora_factors = nn.ParameterDict({})
        self.fan_in_fan_out = config.fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, config=config, **kwargs)
        self.is_target_conv_1d_layer = False

    @property
    def lora_variants(self):
        # Deep LoRA does not compose with other LoRA variants (DoRA, Arrow, ...); restricting the mapping to
        # the empty key makes `resolve_lora_variant` reject any such combination.
        return {(): None}

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        config: DeepLoraConfig,
        **kwargs,
    ) -> None:
        super().update_layer(adapter_name, r, lora_alpha=lora_alpha, config=config, **kwargs)

        num_factors = config.num_factors
        if num_factors > 2:
            # near-identity initialization: the intermediate factors start as exact identities, so the chain
            # collapses to the standard (zero-initialized) LoRA update at initialization
            factors = torch.eye(r).repeat(num_factors - 2, 1, 1)
            self.deeplora_factors[adapter_name] = nn.Parameter(factors)

        # the product of num_factors factors is scaled by r ** (num_factors - 1), which matches the LoRA
        # scaling of lora_alpha / r for num_factors=2
        self.scaling[adapter_name] = lora_alpha / (r ** (num_factors - 1))

        # move the newly created intermediate factors to the device of the base layer
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then
        # cast back to (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # contract the chain: lora_B @ deeplora_factors[-1] @ ... @ deeplora_factors[0] @ lora_A
        chain = weight_A
        if adapter in self.deeplora_factors:
            factors = self.deeplora_factors[adapter]
            if cast_to_fp32:
                factors = factors.float()
            for factor in factors:
                chain = factor @ chain

        output_tensor = transpose(weight_B @ chain, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if adapter_names is not None:
            raise ValueError("Deep LoRA does not support mixed batch inference via the `adapter_names` argument.")
        elif self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                out = lora_A(dropout(x))
                if active_adapter in self.deeplora_factors:
                    for factor in self.deeplora_factors[active_adapter]:
                        out = out @ factor.t()
                result = result + lora_B(out) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "deeplora." + rep
