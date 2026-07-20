# Copyright 2024-present the HuggingFace Inc. team.
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
class AdaMoLEConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`AdaMoLEModel`].

    AdaMoLE (Adaptive Mixture of LoRA Experts, https://arxiv.org/abs/2405.00361) replaces the single low-rank update of
    a targeted layer with a small set of LoRA *experts*. A learned softmax router produces per-token expert weights.
    Instead of a static top-k strategy, a dedicated **threshold network** (a linear layer followed by a sigmoid,
    scaled by `threshold_max`) maps each input to an adaptive activation threshold. Every expert whose routing weight
    reaches the threshold is activated, so the number of active experts varies with the input. The activated experts
    are combined with renormalized, threshold-subtracted weights, `alpha_i = m_i * (p_i - tau) / sum_j m_j * (p_j -
    tau)`, which also keeps the threshold network trainable.

    Args:
        r (`int`):
            LoRA rank used by every expert.
        lora_alpha (`int`):
            Scaling numerator; the expert delta is multiplied by `lora_alpha / r`, as in LoRA.
        num_experts (`int`):
            Number of LoRA experts placed on each targeted layer.
        router_hidden_dim (`Optional[int]`):
            Hidden size of the (two-layer) router MLP. When `None` (the default) the router is a single linear map from
            the input features to `num_experts` logits.
        router_temperature (`float`):
            Temperature applied to the router logits before the softmax.
        threshold_max (`Optional[float]`):
            Upper bound of the input-adaptive activation threshold produced by the threshold network. When `None` (the
            default) it is set to `1 / num_experts`, which guarantees that at least one expert is activated per token
            since the largest softmax weight is always `>= 1 / num_experts`.
        use_threshold (`bool`):
            Whether to apply the adaptive threshold at all. Set to `False` to recover plain dense mixture-of-experts
            routing.
        target_modules (`Union[list[str], str]`):
            List of module names or regex of module names to replace with AdaMoLE. Only `nn.Linear` layers are
            supported.
        exclude_modules (`Optional[Union[list[str], str]]`):
            Names of modules to skip; matched exactly or as a suffix when a list is passed, or as a regex when a string
            is passed.
        modules_to_save (`list[str]`):
            Modules apart from AdaMoLE layers to keep trainable and save in the final checkpoint.
        layers_to_transform (`Union[list[int], int]`):
            Indexes of layers to transform; other layers are left untouched.
        layers_pattern (`Optional[Union[list[str], str]]`):
            Layer pattern name, used only together with `layers_to_transform`.
        fan_in_fan_out (`bool`):
            Set to `True` when the layer to replace stores weights as `(fan_in, fan_out)` (e.g. transformers `Conv1D`).
    """

    r: int = field(default=8, metadata={"help": "LoRA rank used by every expert."})
    lora_alpha: int = field(
        default=8, metadata={"help": "Scaling numerator; the expert delta is scaled by lora_alpha / r."}
    )
    num_experts: int = field(default=4, metadata={"help": "Number of LoRA experts placed on each targeted layer."})
    router_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size of the two-layer router MLP; None uses a single linear router."},
    )
    router_temperature: float = field(default=1.0, metadata={"help": "Temperature of the router softmax."})
    threshold_max: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper bound of the adaptive activation threshold; None uses 1 / num_experts, which guarantees "
            "at least one active expert per token."
        },
    )
    use_threshold: bool = field(
        default=True,
        metadata={"help": "Whether to apply the adaptive activation threshold gate."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex of module names to replace with AdaMoLE. Only nn.Linear supported."
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex of module names to exclude from AdaMoLE."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Modules apart from AdaMoLE layers to keep trainable and save in the final checkpoint."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "The layer indexes to transform with AdaMoLE."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The layer pattern name, used together with `layers_to_transform`."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the layer to replace stores weight like (fan_in, fan_out)."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ADAMOLE
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

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {self.r}")

        if self.num_experts <= 0:
            raise ValueError(
                f"`num_experts` should be a positive integer value but the value passed is {self.num_experts}"
            )

        if self.threshold_max is not None and self.threshold_max <= 0:
            raise ValueError(
                f"`threshold_max` should be a positive value but the value passed is {self.threshold_max}"
            )
