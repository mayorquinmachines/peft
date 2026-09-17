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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MixLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`MixLoraModel`].

    MixLoRA ("MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts",
    https://arxiv.org/abs/2404.15159) replaces the single low-rank update of LoRA with a small mixture of `num_experts`
    LoRA experts. A lightweight router, trained jointly with the experts, assigns each token to its `top_k` highest
    scoring experts, and the expert outputs are combined with the (re-normalized) router probabilities:

        delta(x) = scaling * sum_e g_e(x) * (x @ A_e.T @ B_e.T)

    where `g(x)` is the top-k gated softmax over the router logits. With the default initialization (`B == 0`), the
    adapter is an exact identity at the start of training, like LoRA. An optional Switch-style load-balancing loss
    (`router_aux_loss_coef`) can be added to the training loss to avoid expert collapse; retrieve it after a training
    forward pass via `MixLoraModel.get_router_aux_loss()`.

    Because the update is input-dependent (per-token routing), MixLoRA layers cannot be merged into the base weights.

    Args:
        r (`int`):
            The rank of each LoRA expert.
        num_experts (`int`):
            The number of LoRA experts per targeted layer.
        top_k (`int`):
            The number of experts each token is routed to. Must be between 1 and `num_experts`.
        lora_alpha (`int`):
            The scaling factor for the expert updates, which are scaled by `lora_alpha / r` (as in LoRA).
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
            the output layer. If this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised -- in this case, you should specify the target
            modules manually. MixLoRA is designed for the feed-forward blocks (e.g. `gate_proj`, `up_proj`,
            `down_proj`), which is where the paper places the experts.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        mixlora_dropout (`float`):
            The dropout probability applied to the expert inputs (the router sees the undropped input). Defaults to
            `0.0`.
        router_aux_loss_coef (`float`):
            Coefficient of the load-balancing auxiliary loss. When greater than 0 and the model is in training mode,
            each MixLoRA layer records its auxiliary loss, retrievable via `MixLoraModel.get_router_aux_loss()`, to be
            added to the task loss. Defaults to `0.0` (no auxiliary loss).
        init_mixlora_weights (`bool`):
            Whether to use the default (identity) initialization for the adapter weights, so the adapter is a no-op at
            the start of training (`A` kaiming-initialized, `B` zero). Don't change this setting, except if you know
            exactly what you're doing. Defaults to `True`.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "The rank of each LoRA expert."})
    num_experts: int = field(default=4, metadata={"help": "The number of LoRA experts per targeted layer."})
    top_k: int = field(
        default=2,
        metadata={"help": "The number of experts each token is routed to. Must be between 1 and `num_experts`."},
    )
    lora_alpha: int = field(
        default=8, metadata={"help": "The scaling factor for the expert updates, scaled by `lora_alpha / r`."}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with MixLoRA.",
            "example": "For example, ['gate_proj', 'up_proj', 'down_proj'] or '.*mlp.*(gate|up|down)_proj$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from MixLoRA."},
    )
    mixlora_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability applied to the expert inputs."},
    )
    router_aux_loss_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient of the load-balancing auxiliary loss. When > 0 in training mode, the per-layer auxiliary "
                "losses are recorded and can be retrieved via `MixLoraModel.get_router_aux_loss()`."
            )
        },
    )
    init_mixlora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the MixLoRA layers with their default (identity) initialization. "
                "Don't change this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, if this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from MixLoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MIXLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {self.r}")
        if self.num_experts < 1:
            raise ValueError(
                f"`num_experts` should be a positive integer value but the value passed is {self.num_experts}"
            )
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(f"`top_k` must be between 1 and `num_experts` ({self.num_experts}), got {self.top_k}.")

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
