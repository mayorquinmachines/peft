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
from typing import Any

import torch
from torch import nn

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS, Linear, LoraLayer
from peft.utils.other import transpose


class MosLoraLinear(Linear):
    """
    LoRA linear layer with a learnable `r x r` mixer matrix between `lora_A` and `lora_B`.

    The weight update is `Delta W = B @ M @ A * scaling` (MoSLoRA, https://huggingface.co/papers/2406.11909) instead
    of LoRA's `Delta W = B @ A * scaling`. The mixer is stored per adapter in `self.lora_mixer`, mirroring how
    `lora_A`/`lora_B` are stored, so all LoRA machinery (device placement, adapter switching, state dict handling)
    applies to it unchanged.
    """

    # "lora_mixer" deliberately contains the "lora_" prefix so that the mixer weights are covered by the same state
    # dict filtering and trainable-parameter marking as the other LoRA weights.
    adapter_layer_names = LoraLayer.adapter_layer_names + ("lora_mixer",)

    @property
    def lora_variants(self):
        # MoSLoRA does not support LoRA variants that change the update structure (DoRA, Arrow, aLoRA, ...). Since
        # only the empty combination is registered, requesting any variant raises a clear error.
        return {(): None}

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        config: LoraConfig,
        **kwargs,
    ) -> None:
        if "lora_mixer" not in self._modules:
            self.lora_mixer = nn.ModuleDict({})
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        # add the mixer *before* calling the parent method, so that the device/dtype placement and the
        # requires_grad handling at the end of `update_layer` also apply to the mixer.
        mixer = nn.Linear(r, r, bias=False)
        self._init_mixer_weight(mixer.weight, r, config.mixer_init)
        self.lora_mixer[adapter_name] = mixer
        super().update_layer(adapter_name, r, lora_alpha=lora_alpha, config=config, **kwargs)
        if not config.trainable_mixer:
            # declare the mixer as non-trainable so that it stays frozen also after `set_adapter` calls
            self.frozen_peft_weight_names = {**self.frozen_peft_weight_names, adapter_name: ("lora_mixer",)}
            self._freeze_non_trainable_peft_weights(adapter_name)

    @staticmethod
    def _init_mixer_weight(weight: torch.Tensor, r: int, mixer_init: str) -> None:
        # Kaiming uniform and orthogonal are the two best performing mixer initializations reported in the paper;
        # a zero-initialized mixer is not offered since it does not converge (the gradients w.r.t. the mixer and B
        # vanish at initialization).
        if mixer_init == "kaiming":
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        elif mixer_init == "orthogonal":
            nn.init.orthogonal_(weight)
        elif mixer_init == "identity":
            nn.init.eye_(weight)
        elif mixer_init == "butterfly":
            if r % 2 != 0:
                raise ValueError(f"`mixer_init='butterfly'` requires an even rank, got r={r}.")
            half = r // 2
            with torch.no_grad():
                weight.zero_()
                weight[:half, :half] = torch.eye(half)
                weight[:half, half:] = torch.eye(half)
                weight[half:, :half] = torch.eye(half)
                weight[half:, half:] = torch.eye(half)
        else:
            raise ValueError(f"Unknown `mixer_init` value: {mixer_init!r}.")

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
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        weight_mixer = self.lora_mixer[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            weight_mixer = weight_mixer.float()

        output_tensor = transpose(weight_B @ weight_mixer @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
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
                mixer = self.lora_mixer[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                result = result + lora_B(mixer(lora_A(dropout(x)))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            mixer = self.lora_mixer[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to MoSLoRA layers and updating the corresponding indices of the
            # linear layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            moslora_output = lora_B(mixer(lora_A(dropout(sub_batch)))) * scaling
            result[sub_batch_indices_list[i]] += moslora_output.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = nn.Module.__repr__(self)
        return "moslora." + rep
