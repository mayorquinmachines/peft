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

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class RLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`RLoraModel`].

    R-LoRA (Randomized Multi-Head LoRA, https://arxiv.org/abs/2502.15455) improves LoRA in multi-task learning
    scenarios by diversifying the up-projection. Instead of a single `lora_B`, the adapter keeps `num_heads` distinct
    up-projection "head" matrices that all share the same down-projection `lora_A`. Two mechanisms diversify the
    heads:

      - Multi-Head Dropout: during training, each head receives an independently dropout-masked version of the shared
        low-dimensional intermediate `lora_A(x)` (Bernoulli mask with probability `head_dropout`, rescaled by
        `1 / (1 - head_dropout)`). Masking the intermediate is cheaper than masking the layer input. At inference,
        all heads are active and no mask is applied.
      - Multi-Head Random Initialization: the head matrices are randomly initialized (scaled Gaussian) instead of
        zeros to break symmetry. The initialization is centered so that the heads sum to zero, meaning the adapter is
        still an exact identity at initialization and the base model output is preserved.

    The head outputs are uniformly averaged, i.e. `delta = (1 / num_heads) * sum_i B_i @ A`, so the adapter can be
    merged exactly like a vanilla LoRA adapter. With `num_heads=1`, R-LoRA reduces exactly to vanilla LoRA.

    This class inherits all arguments from [`LoraConfig`]; only the R-LoRA specific arguments are documented below.

    Args:
        num_heads (`int`):
            The number of up-projection head matrices sharing the same down-projection. Must be a positive integer.
            With `num_heads=1` (and any `head_dropout`), the adapter behaves exactly like vanilla LoRA. The paper
            uses 3 heads in its main experiments. Defaults to 4.
        head_dropout (`float`):
            The probability of masking out an entry of the per-head intermediate representation during training
            (Multi-Head Dropout). Must be in `[0, 1)`. Only used during training and only when `num_heads > 1`.
            Defaults to 0.1 (the paper reports a dropout rate of 0.2).

    Note:
        When `num_heads > 1`, the following LoRA options are not supported and raise an error: `lora_bias=True`,
        `use_dora=True`, and non-default `init_lora_weights` (e.g. "pissa", "olora", "eva", ...). Only
        `torch.nn.Linear` and `transformers.pytorch_utils.Conv1D` target modules are supported.
    """

    num_heads: int = field(
        default=4,
        metadata={
            "help": (
                "The number of up-projection head matrices sharing the same down-projection. With `num_heads=1`, "
                "R-LoRA reduces exactly to vanilla LoRA."
            ),
            "is_lora_variant": True,
        },
    )
    head_dropout: float = field(
        default=0.1,
        metadata={
            "help": (
                "The probability of masking out an entry of the per-head intermediate representation during training "
                "(Multi-Head Dropout). Only used during training and only when `num_heads > 1`."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.RLORA

        if not isinstance(self.num_heads, int) or self.num_heads < 1:
            raise ValueError(
                f"`num_heads` should be a positive integer value but the value passed is {self.num_heads}"
            )
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError(f"`head_dropout` should be a value in [0, 1) but the value passed is {self.head_dropout}")

        if self.num_heads > 1:
            if self.lora_bias:
                raise ValueError(f"{self.peft_type} with `num_heads > 1` does not support `lora_bias=True`.")
            if self.use_dora:
                raise ValueError(f"{self.peft_type} with `num_heads > 1` does not support DoRA.")
            if self.init_lora_weights not in (True, "gaussian"):
                raise ValueError(
                    f"{self.peft_type} with `num_heads > 1` does not support "
                    f"`init_lora_weights={self.init_lora_weights}`, use `True` or 'gaussian' instead."
                )
