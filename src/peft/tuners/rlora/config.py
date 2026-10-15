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
class RLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RLoraModel`].

    Paper: https://huggingface.co/papers/2502.15455.

    R-LoRA is a multi-head LoRA variant for multi-task learning. A shared down-projection A feeds several head
    matrices B_i whose outputs are combined by a learned router. On top of this architecture, R-LoRA applies
    "multi-head randomization": the head matrices are randomly initialized (with a mean-centering correction that
    keeps the adapter's initial contribution at zero) and multi-head dropout diversifies the low-rank representation
    that each head receives during training.

    Args:
        r (`int`, *optional*, defaults to `8`):
            The rank of the shared down-projection matrix A.
        rlora_num_heads (`int`, *optional*, defaults to `4`):
            The number of head matrices B_i. Must be at least 2.
        target_modules (`Union[list[str], str]`):
            The names of the modules to apply R-LoRA to. Only linear layers are supported. If this is specified as
            'all-linear', then all linear modules are chosen, excluding the output layer. If this is not specified,
            modules will be chosen according to the model architecture.
        exclude_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to not apply the adapter to. When passing a string, a regex match will be
            performed.
        rlora_alpha (`int`, *optional*, defaults to `16`):
            The scaling coefficient for R-LoRA layers. The scaling is `rlora_alpha / r`.
        rlora_dropout (`float`, *optional*, defaults to `0.0`):
            The dropout probability applied to the input of the R-LoRA layers.
        rlora_head_dropout (`float`, *optional*, defaults to `0.0`):
            The multi-head dropout probability. During training, an independent dropout mask is sampled per head on
            the low-rank representation `A @ x` (inverted dropout scaling), diversifying the input each head sees.
            The paper uses a value of 0.2.
        init_rlora_weights (`bool`, *optional*, defaults to `True`):
            Whether to use multi-head random initialization: the head matrices are sampled from a scaled normal
            distribution and then mean-centered, which breaks the symmetry between heads while keeping the adapter's
            initial contribution exactly zero (equivalent to the base-weight offset correction from the paper, but
            without mutating the base weights). If `False`, the head matrices are zero-initialized as in standard
            LoRA / HydraLoRA.
        init_gamma (`float`, *optional*, defaults to `64.0`):
            The gamma hyperparameter of the multi-head random initialization. Head weights are sampled with standard
            deviation `1 / (sqrt(init_gamma) * out_features ** 0.25)`.
        bias (`str`, *optional*, defaults to `"none"`):
            Bias type for R-LoRA. Can be 'none', 'all' or 'rlora_only'.
        modules_to_save (`Optional[list[str]]`):
            List of modules apart from R-LoRA layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Union[list[int], int]`):
            The layer indexes to transform, if this argument is specified, it will apply the R-LoRA transformations
            on the layer indexes that are specified in this list. If a single integer is passed, it will apply the
            R-LoRA transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=8, metadata={"help": "R-LoRA rank of the shared down-projection matrix"})
    rlora_num_heads: int = field(default=4, metadata={"help": "Number of R-LoRA head matrices, must be at least 2"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "list of module names or regex expression of the module names to replace with R-LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from R-LoRA."},
    )
    rlora_alpha: int = field(default=16, metadata={"help": "Scaling coefficient in the adapter layers"})
    rlora_dropout: float = field(default=0.0, metadata={"help": "Dropout on the input of the adapter layers"})
    rlora_head_dropout: float = field(
        default=0.0,
        metadata={
            "help": (
                "Multi-head dropout probability: during training, an independent dropout mask is sampled per head on "
                "the low-rank representation. The paper uses a value of 0.2."
            )
        },
    )
    init_rlora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use multi-head random initialization (random, mean-centered head matrices). If False, "
                "head matrices are zero-initialized as in standard LoRA."
            ),
        },
    )
    init_gamma: float = field(
        default=64.0,
        metadata={"help": "Gamma of the multi-head random initialization, controls the head weight scale."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for R-LoRA. Can be 'none', 'all' or 'rlora_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "list of modules apart from R-LoRA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
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

        if self.rlora_num_heads < 2:
            raise ValueError(f"`rlora_num_heads` must be at least 2 but the value passed is {self.rlora_num_heads}")

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
