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


# The PEFT type name for this method. It is registered as a `PeftType` member dynamically in
# `peft.tuners.ternary_adapt.__init__` (see there for why), so it behaves like any other type.
TERNARY_ADAPT_PEFT_TYPE = "TERNARY_ADAPT"


@dataclass
class TernaryAdaptConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`TernaryAdaptModel`].

    Ternary multiplicative adaptation fine-tunes ternary transformers without dequantization, following "Low-Rank
    Ternary Adaptation for Fine-Tuning Transformers" (https://arxiv.org/abs/2608.24469v1). The base weight is
    ternarized on the fly at injection time (absmean quantization per output channel, analogous to how PEFT handles
    quantized bases for other methods). Each adapted layer then learns a multiplicative mask `M` with a low-rank
    Kronecker structure: `M = A (x) B` is the Kronecker product of two small matrices whose forward values are
    ternary (`{-1, 0, +1}`, trained with a straight-through estimator). The adapted weight `W' = W (.) M` is an
    element-wise product of ternary values, so it stays in the ternary domain and can be merged back into the base
    weight directly, without dequantization.

    Args:
        block_shape (`Optional[tuple[int, int]]`):
            The shape `(rows, cols)` of the Kronecker block `B`. The other factor `A` is derived as
            `(out_features // rows, in_features // cols)`, so `rows` must divide `out_features` and `cols` must divide
            `in_features`. The total adapter parameter count per layer is
            `(out_features / rows) * (in_features / cols) + rows * cols`, far below the `out_features * in_features`
            of a full mask. If `None` (default), a near-square block shape is derived automatically from the layer
            dimensions.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the
            specified names will be replaced. When passing a string, a regex match will be performed. When passing a
            list of strings, either an exact match will be performed or it is checked if the name of the module ends
            with any of the passed strings. If this is not specified, an error is raised -- there are no architecture
            defaults for this method yet, so you should specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be
            performed. When passing a list of strings, either an exact match will be performed or it is checked if
            the name of the module ends with any of the passed strings.
        ternarize_base (`bool`):
            Whether to ternarize the base weight in-place at injection time (absmean quantization per output
            channel). This is the setting of the paper: the base model is a ternary transformer. Set to `False` to
            apply the multiplicative ternary mask to a full-precision base weight instead (an ablation; merging then
            no longer yields a ternary weight). Defaults to `True`.
        fan_in_fan_out (`bool`):
            Set this to `True` if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        init_weights (`bool`):
            Whether to initialize the adapter factors to all ones, which ternarizes to an all-ones mask, so the
            adapter is an exact identity at the start of training. Don't change this setting, except if you know
            exactly what you're doing. Defaults to `True`.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer
            indices that are specified in this list. If a single integer is passed, it will apply the transformations
            on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target
            the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    block_shape: Optional[tuple[int, int]] = field(
        default=None,
        metadata={
            "help": (
                "Shape (rows, cols) of the Kronecker block B; the other factor A is derived as "
                "(out_features // rows, in_features // cols). If None, a near-square block shape is derived "
                "automatically from the layer dimensions."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ternary adaptation.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from the adapter."},
    )
    ternarize_base: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to ternarize the base weight in-place at injection time (absmean quantization per output "
                "channel). Set to False to adapt a full-precision base weight instead (an ablation)."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": (
                "Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 "
                "uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to True."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the adapter factors to all ones (an exact identity mask at init). Don't "
                "change this setting, except if you know exactly what you're doing."
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
            "help": "List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # plain str; the `PeftType` member with this value is added dynamically on package import
        self.peft_type = TERNARY_ADAPT_PEFT_TYPE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if self.block_shape is not None:
            if len(self.block_shape) != 2:
                raise ValueError(f"`block_shape` must be a (rows, cols) pair, got {self.block_shape}.")
            rows, cols = self.block_shape
            if (rows < 1) or (cols < 1):
                raise ValueError(f"`block_shape` entries must be positive integers, got {self.block_shape}.")
            # normalize to a tuple so config round-trips (JSON loads lists) compare equal
            self.block_shape = (int(rows), int(cols))

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
