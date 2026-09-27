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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class HyperLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`HyperLoraModel`].

    HyperLoRA replaces the directly-optimized LoRA weight matrices with a small hypernetwork that *generates* the
    LoRA matrices from a context vector. The context vector is provided at runtime via
    `HyperLoraModel.set_context(...)` and is not part of the saved checkpoint: the same trained hypernetwork can
    synthesize a different LoRA per context (e.g. per user) with forward passes only.

    Args:
        r (`int`):
            The rank of the generated LoRA matrices.
        context_dim (`int`):
            Dimensionality of the context vector that conditions the hypernetwork. The caller is responsible for
            producing this vector (e.g. by mean-pooling hidden states of the base model over the user's context
            tokens).
        hypernet_hidden_size (`int`):
            Hidden size of the per-layer generator MLP that maps the context vector to the flattened LoRA matrices.
        hyperlora_alpha (`int`):
            The alpha parameter for HyperLoRA scaling; the generated delta weight is scaled by
            `hyperlora_alpha / r`.
        hyperlora_dropout (`float`):
            The dropout probability applied to the input of HyperLoRA layers.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the
            specified names will be replaced. When passing a string, a regex match will be performed. When passing a
            list of strings, either an exact match will be performed or it is checked if the name of the module ends
            with any of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should
            specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be
            performed. When passing a list of strings, either an exact match will be performed or it is checked if
            the name of the module ends with any of the passed strings.
        init_weights (`bool`):
            Whether to zero-initialize the output projection of each generator so that the generated LoRA starts as
            an exact zero update (the model initially behaves like the base model). Don't change this setting,
            except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer
            indices that are specified in this list. If a single integer is passed, it will apply the transformations
            on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        bias (`str`):
            Bias type for HyperLoRA. Can be `'none'`, `'all'` or `'hyperlora_only'`. Currently only `'none'` is
            supported.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "The rank of the generated LoRA matrices."})
    context_dim: int = field(
        default=768,
        metadata={"help": "Dimensionality of the context vector that conditions the LoRA-generating hypernetwork."},
    )
    hypernet_hidden_size: int = field(default=64, metadata={"help": "Hidden size of the per-layer generator MLP."})
    hyperlora_alpha: int = field(default=16, metadata={"help": "HyperLoRA alpha scaling parameter."})
    hyperlora_dropout: float = field(default=0.0, metadata={"help": "HyperLoRA dropout"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with HyperLoRA.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from HyperLoRA."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to zero-initialize the generator output projection so the generated LoRA starts as a zero "
                "update. Don't change this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for HyperLoRA. Can be 'none', 'all' or 'hyperlora_only'"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from HyperLoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HYPERLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.bias != "none":
            raise ValueError(f"HyperLoRA currently only supports bias='none', got bias='{self.bias}'.")
