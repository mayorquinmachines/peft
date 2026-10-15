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

import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend

from .config import HyperLoraConfig


class HyperLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("hyperlora_generator",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("hyperlora_r", "hyperlora_alpha", "hyperlora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )
        self.hyperlora_r = {}
        self.hyperlora_alpha = {}
        self.hyperlora_dropout = nn.ModuleDict({})
        self.hyperlora_generator = nn.ModuleDict({})
        # Runtime context vectors conditioning the generator. These are inputs, not checkpoint state, so they are
        # deliberately kept outside of adapter_layer_names / other_param_names.
        self.hyperlora_context: dict[str, torch.Tensor] = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        if (in_features is None) or (out_features is None):
            raise TypeError(f"Unsupported layer type {type(base_layer)}")
        self.in_features, self.out_features = in_features, out_features

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: HyperLoraConfig,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create a HyperLoRA adapter (a LoRA-generating hypernetwork)."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if config.context_dim <= 0:
            raise ValueError(f"`context_dim` should be a positive integer, got {config.context_dim}")

        self.hyperlora_r[adapter_name] = r
        self.hyperlora_alpha[adapter_name] = config.hyperlora_alpha
        if config.hyperlora_dropout > 0.0:
            hyperlora_dropout_layer = nn.Dropout(p=config.hyperlora_dropout)
        else:
            hyperlora_dropout_layer = nn.Identity()
        self.hyperlora_dropout[adapter_name] = hyperlora_dropout_layer

        num_generated = r * (self.in_features + self.out_features)
        generator = nn.Sequential(
            nn.Linear(config.context_dim, config.hypernet_hidden_size),
            nn.SiLU(),
            nn.Linear(config.hypernet_hidden_size, num_generated),
        )
        if config.init_weights:
            # Zero-init the output projection so that the generated LoRA starts as an exact zero update.
            nn.init.zeros_(generator[-1].weight)
            nn.init.zeros_(generator[-1].bias)
        self.hyperlora_generator[adapter_name] = generator

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def set_context(self, context: torch.Tensor, adapter_name: str) -> None:
        """
        Set the context vector that conditions the LoRA generation for the given adapter.

        Args:
            context (`torch.Tensor`):
                A 1D tensor of shape `(context_dim,)` (a batch dimension of 1 is also accepted).
            adapter_name (`str`):
                The name of the adapter whose generator should be conditioned on this context.
        """
        if adapter_name not in self.hyperlora_generator:
            raise ValueError(f"Adapter {adapter_name} not found on this layer.")
        context = context.reshape(-1)
        expected_dim = self.hyperlora_generator[adapter_name][0].in_features
        if context.numel() != expected_dim:
            raise ValueError(f"Expected a context vector of dimension {expected_dim}, got {context.numel()}.")
        self.hyperlora_context[adapter_name] = context

    def _get_context(self, adapter_name: str) -> torch.Tensor:
        if adapter_name not in self.hyperlora_context:
            raise RuntimeError(
                f"No context set for adapter '{adapter_name}'. HyperLoRA generates its LoRA weights from a context "
                "vector; call `HyperLoraModel.set_context(context)` (or `layer.set_context(context, adapter_name)`) "
                "before the forward pass."
            )
        return self.hyperlora_context[adapter_name]

    def generate_lora_weights(self, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the LoRA matrices (A, B) for the given adapter from the currently set context.

        Returns a tuple `(lora_A, lora_B)` of shapes `(r, in_features)` and `(out_features, r)`.
        """
        context = self._get_context(adapter_name)
        generator = self.hyperlora_generator[adapter_name]
        gen_param = generator[-1].weight
        flat = generator(context.to(device=gen_param.device, dtype=gen_param.dtype))
        r = self.hyperlora_r[adapter_name]
        num_a = r * self.in_features
        lora_A = flat[:num_a].reshape(r, self.in_features)
        lora_B = flat[num_a:].reshape(self.out_features, r)
        return lora_A, lora_B

    def scaling(self, adapter_name: str) -> float:
        return self.hyperlora_alpha[adapter_name] / self.hyperlora_r[adapter_name]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.hyperlora_generator.keys():
                continue
            self.hyperlora_alpha[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.hyperlora_generator.keys():
                continue
            self.hyperlora_alpha[active_adapter] /= scale


class HyperLoraLinear(nn.Module, HyperLoraLayer):
    """
    HyperLoRA implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: HyperLoraConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        HyperLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Note that the generated LoRA depends on the currently set context: merging bakes in the LoRA generated for
        the context that is set at merge time.

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

        for active_adapter in adapter_names:
            if active_adapter in self.hyperlora_generator.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    weight = self.get_base_weight().clone()
                    orig_dtype = weight.dtype
                    weight += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.set_base_weight(weight.to(orig_dtype))
                else:
                    weight = self.get_base_weight()
                    orig_dtype = weight.dtype
                    weight += self.get_delta_weight(active_adapter)
                    self.set_base_weight(weight.to(orig_dtype))
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hyperlora_generator.keys():
                weight = self.get_base_weight()
                orig_dtype = weight.dtype
                weight -= self.get_delta_weight(active_adapter)
                self.set_base_weight(weight.to(orig_dtype))

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter, i.e. the scaled generated LoRA update for the currently set
        context.
        """
        lora_A, lora_B = self.generate_lora_weights(adapter_name)
        device = lora_A.device
        dtype = lora_A.dtype
        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back
        # to (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        output_tensor = (lora_B @ lora_A) * self.scaling(adapter_name)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

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
            if self.quantization_backend is not None:
                result = self.quantization_backend.maybe_clone_base_result(result)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hyperlora_generator.keys():
                    continue
                lora_A, lora_B = self.generate_lora_weights(active_adapter)
                dropout = self.hyperlora_dropout[active_adapter]
                x = self._cast_input_dtype(x, lora_A.dtype)
                result = result + F.linear(F.linear(dropout(x), lora_A), lora_B) * self.scaling(active_adapter)

        result = result.to(previous_dtype)
        return result

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return True

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hyperlora." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)
