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

import torch
from torch import nn

from peft.tuners.lora.layer import Linear, LoraVariant
from peft.utils.other import transpose


def _init_head_weights(weight: torch.Tensor, num_heads: int) -> None:
    """
    Multi-Head Random Initialization for the packed head matrices.

    `weight` is the `(out_features, num_heads * r)` weight of the packed up-projection; head `i` occupies columns
    `[i * r, (i + 1) * r)`. Each head is initialized with a scaled Gaussian, following the paper's scale of
    `d_out ** 0.25 / sqrt(gamma) * N(0, 1 / d_out)` with `gamma = 64`, and the heads are then centered so that they
    sum to zero. The centering makes the adapter an exact identity at initialization (the average head is zero), so
    the pretrained model output is preserved without having to modify the base weights.
    """
    out_features = weight.shape[0]
    std = out_features**-0.25 / 8.0
    with torch.no_grad():
        weight.normal_(0.0, std)
        heads = weight.view(out_features, num_heads, -1)
        heads -= heads.mean(dim=1, keepdim=True)


class RLoraLinearVariant(LoraVariant):
    """
    R-LoRA variant for `torch.nn.Linear` layers.

    During training, each head receives an independently dropout-masked view of the shared intermediate `lora_A(x)`
    (Multi-Head Dropout). At inference and for merging, all heads are active and uniformly averaged, which reduces to
    the vanilla LoRA computation with the average head matrix.
    """

    @staticmethod
    def init(module: Linear, adapter_name: str, config, **kwargs) -> None:
        if not hasattr(module, "rlora_num_heads"):
            module.rlora_num_heads = {}
            module.rlora_head_dropout = {}

        num_heads = config.num_heads
        module.rlora_num_heads[adapter_name] = num_heads
        module.rlora_head_dropout[adapter_name] = config.head_dropout

        if num_heads == 1:
            # vanilla LoRA: keep the single up-projection created by `update_layer` as is
            return

        # Replace the single up-projection created by `update_layer` with `num_heads` head matrices packed into one
        # Linear of shape (out_features, num_heads * r); head i occupies the columns [i * r, (i + 1) * r).
        r = module.r[adapter_name]
        prev_lora_B = module.lora_B[adapter_name]
        lora_B = nn.Linear(num_heads * r, prev_lora_B.out_features, bias=False)
        lora_B = lora_B.to(device=prev_lora_B.weight.device, dtype=prev_lora_B.weight.dtype)
        if lora_B.weight.device.type != "meta":
            _init_head_weights(lora_B.weight, num_heads)
        module.lora_B[adapter_name] = lora_B

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        # Merging uses the average head matrix (see `RLoraLinear.get_delta_weight`), which is the exact update applied
        # at inference time.
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        return orig_weight + delta_weight.to(orig_dtype)

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        delta_weight = module.get_delta_weight(active_adapter)
        orig_weight.data += delta_weight

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        return orig_weight - delta_weight.to(orig_dtype)

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor, **kwargs) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        num_heads = module.rlora_num_heads[active_adapter]
        head_dropout = module.rlora_head_dropout[active_adapter]

        hidden = lora_A(dropout(x))  # (..., r), shared by all heads

        # give each head its own slot: (..., num_heads, r)
        stacked = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], num_heads, hidden.shape[-1])
        if module.training and num_heads > 1 and head_dropout > 0.0:
            # Multi-Head Dropout: an independent Bernoulli mask per head, applied to the shared low-dimensional
            # intermediate (cheaper than masking the layer input), with the standard inverted rescaling.
            keep_prob = 1.0 - head_dropout
            mask = torch.rand(stacked.shape[:-1], device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            stacked = stacked * (mask < keep_prob).to(hidden.dtype) / keep_prob

        update = lora_B(stacked.reshape(*hidden.shape[:-1], num_heads * hidden.shape[-1]))
        return result + update * (scaling / num_heads)


class RLoraLinear(Linear):
    """
    R-LoRA adapter layer for `torch.nn.Linear` (and `transformers.pytorch_utils.Conv1D`) modules.

    Behaves exactly like a vanilla LoRA layer when the adapter is configured with `num_heads=1`; otherwise the
    up-projection packs `num_heads` head matrices and the forward pass is handled by [`RLoraLinearVariant`].
    """

    @property
    def lora_variants(self):
        return {**super().lora_variants, ("num_heads",): RLoraLinearVariant}

    def get_delta_weight(self, adapter) -> torch.Tensor:
        rlora_num_heads = getattr(self, "rlora_num_heads", {})
        if adapter not in rlora_num_heads:
            # adapter configured with num_heads=1, i.e. vanilla LoRA
            return super().get_delta_weight(adapter)

        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        # average the packed head matrices: (out_features, num_heads * r) -> (out_features, r)
        weight_B = self.lora_B[adapter].weight
        weight_B = weight_B.view(weight_B.shape[0], rlora_num_heads[adapter], -1).mean(dim=1)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor
