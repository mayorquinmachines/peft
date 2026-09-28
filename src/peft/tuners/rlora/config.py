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
class RLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RLoraModel`].

    R-LoRA ("R-LoRA: Randomized Multi-Head LoRA for Efficient Multi-Task Learning", https://arxiv.org/abs/2502.15455)
    extends LoRA with a multi-head structure and Multi-Head Randomization: the up-projection is split into
    `num_heads` head matrices `B_i` that share a single down-projection `A`, and the update is the router-weighted
    sum `sum_i omega_i B_i A`. Multi-Head Dropout masks the low-rank intermediate `A x` independently per head
    during training, and Multi-Head Random Initialization initializes both `A` and the heads non-zero (compensating
    the base weight so that the adapted layer is an exact identity at initialization).

    Args:
        r (`int`):
            The rank of the shared down-projection `A` and of each head matrix `B_i`.
        num_heads (`int`):
            The number of head matrices `B_i` per adapted layer.
        rlora_alpha (`int`):
            The alpha parameter for R-LoRA scaling; the update is scaled by `rlora_alpha / r`.
        rlora_dropout (`float`):
            The dropout probability applied to the layer input, before the down-projection.
        head_dropout (`float`):
            The Multi-Head Dropout probability: dropout applied element-wise to the low-rank intermediate `A x`,
            with an independent mask per head, so that each head sees a differently masked input. Only active
            during training.
        use_router (`bool`):
            Whether to combine the heads with the learned router `omega = softmax(W_r x)` (paper default). If
            `False`, the heads are averaged uniformly, which also makes merging exact for arbitrary (trained)
            weights.
        init_weights (`bool`):
            Whether to use Multi-Head Random Initialization: `A` and the heads are drawn from scaled normal
            distributions (std `out_features**0.25 / (sqrt(init_gamma) * sqrt(fan))`) and the initial delta
            `scaling * mean_i(B_i) @ A` is subtracted from the base weight, so the adapted layer is an exact
            identity at initialization. If `False`, falls back to the standard LoRA identity initialization
            (kaiming `A`, zero heads) without touching the base weight.
        init_gamma (`float`):
            The gamma hyperparameter of Multi-Head Random Initialization (paper: 64, following LoRA-GA).
        init_seed (`int`):
            Seed for the generator used by Multi-Head Random Initialization. The init must be deterministic
            because the base-weight compensation is not part of the saved adapter state dict: when a checkpoint is
            loaded, the adapter is re-created (re-drawing the init and re-applying the compensation) before the
            saved weights are loaded, so the same seed must reproduce the same compensation.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the
            specified names will be replaced. When passing a string, a regex match will be performed. When passing
            a list of strings, either an exact match will be performed or it is checked if the name of the module
            ends with any of the passed strings. If this is specified as 'all-linear', then all linear modules are
            chosen, excluding the output layer.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be
            performed. When passing a list of strings, either an exact match will be performed or it is checked if
            the name of the module ends with any of the passed strings.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for R-LoRA. Currently only `'none'` is supported.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer
            indices that are specified in this list. If a single integer is passed, it will apply the
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
    """

    r: int = field(default=8, metadata={"help": "R-LoRA rank of the shared down-projection and each head."})
    num_heads: int = field(default=4, metadata={"help": "Number of R-LoRA head matrices per adapted layer."})
    rlora_alpha: int = field(default=16, metadata={"help": "R-LoRA alpha; the update is scaled by rlora_alpha / r."})
    rlora_dropout: float = field(
        default=0.0, metadata={"help": "Dropout probability applied to the layer input before the down-projection."}
    )
    head_dropout: float = field(
        default=0.2,
        metadata={
            "help": "Multi-Head Dropout probability on the low-rank intermediate (independent mask per head, training only)."
        },
    )
    use_router: bool = field(
        default=True,
        metadata={"help": "Combine heads with the learned router softmax(W_r x); if False, average heads uniformly."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": "Multi-Head Random Initialization with base-weight compensation (exact identity at init). "
            "False -> standard LoRA identity init (kaiming A, zero heads)."
        },
    )
    init_gamma: float = field(
        default=64.0, metadata={"help": "Gamma of Multi-Head Random Initialization (paper: 64, following LoRA-GA)."}
    )
    init_seed: int = field(
        default=42,
        metadata={
            "help": "Seed for Multi-Head Random Initialization. Must stay fixed so that the base-weight "
            "compensation is reproduced when a saved adapter is re-created during loading."
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with R-LoRA.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None, metadata={"help": "List of module names or regex expression of the module names to exclude from R-LoRA."}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for R-LoRA. Currently only 'none' is supported."})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from R-LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.RLORA
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

        if self.num_heads < 1:
            raise ValueError(f"`num_heads` should be a positive integer value but the value passed is {self.num_heads}")

        if self.bias != "none":
            raise ValueError(f"R-LoRA currently only supports `bias='none'`, got {self.bias!r}.")
