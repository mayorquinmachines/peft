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

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_HYPERLORA_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import HyperLoraLayer, HyperLoraLinear


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[HyperLoraLayer] | None:
    layer_cls: type[HyperLoraLayer] | None = None
    if isinstance(target_base_layer, torch.nn.Linear):
        layer_cls = HyperLoraLinear
    elif (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
        layer_cls = {"linear": HyperLoraLinear}.get(quant_backend.layer_type)

    return layer_cls


class HyperLoraModel(BaseTuner):
    """
    Creates a HyperLoRA model from a pretrained model. Instead of optimizing LoRA weight matrices directly, a small
    per-layer hypernetwork *generates* the LoRA matrices from a context vector that is supplied at runtime via
    [`HyperLoraModel.set_context`]. The method is described in
    https://huggingface.co/papers/2609.24979

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`HyperLoraConfig`]): The configuration of the HyperLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The HyperLoRA model.

    Example:
        ```py
        >>> import torch
        >>> from transformers import AutoModelForCausalLM

        >>> from peft import HyperLoraConfig, HyperLoraModel

        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> config = HyperLoraConfig(r=8, context_dim=model.config.n_embd, target_modules=["c_attn"])
        >>> peft_model = HyperLoraModel(model, config, "default")

        >>> # condition the LoRA generation on a context vector, e.g. pooled hidden states of the user's context
        >>> context = torch.randn(model.config.n_embd)
        >>> peft_model.set_context(context)
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`HyperLoraConfig`]): The configuration of the HyperLoRA model.
    """

    prefix: str = "hyperlora_"
    tuner_layer_cls = HyperLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_HYPERLORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        hyperlora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "r": hyperlora_config.r,
        }
        kwargs.update(get_quantization_kwargs(self))

        # If it is not a HyperLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, HyperLoraLayer):
            new_module = self._create_new_module(hyperlora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                config=hyperlora_config,
                **kwargs,
            )

    @staticmethod
    def _create_new_module(hyperlora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)
        if layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` (optionally quantized) "
                "is supported."
            )

        new_module = layer_cls(target, adapter_name, config=hyperlora_config, **kwargs)
        return new_module

    def set_context(self, context: torch.Tensor, adapter_name: str = "default") -> None:
        """
        Set the context vector that conditions the LoRA generation for the given adapter on all HyperLoRA layers.

        The context is a runtime input (e.g. pooled hidden states of the base model over the user's context tokens)
        and is intentionally not stored in the adapter checkpoint: the same trained hypernetwork synthesizes a
        personalized LoRA per supplied context with forward passes only.

        Args:
            context (`torch.Tensor`):
                A 1D tensor of shape `(context_dim,)` (a batch dimension of 1 is also accepted).
            adapter_name (`str`):
                The name of the adapter to condition, defaults to `"default"`.
        """
        found = False
        for module in self.model.modules():
            if isinstance(module, HyperLoraLayer) and adapter_name in module.hyperlora_generator:
                module.set_context(context, adapter_name)
                found = True
        if not found:
            raise ValueError(f"Adapter {adapter_name} not found on any HyperLoRA layer of this model.")
