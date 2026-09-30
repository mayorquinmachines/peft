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
class DeepLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`DeepLoraModel`].

    Deep LoRA ("Compressible Dynamics in Deep Overparameterized Low-Rank Learning & Adaptation",
    https://huggingface.co/papers/2406.04112) represents the low-rank update as a chain of `num_factors >= 2`
    low-rank factors instead of the usual two. The intermediate factors are square (r, r) matrices initialized
    close to the identity, so the update is zero at initialization exactly like LoRA, while the
    overparameterized chain improves the optimization dynamics of the adaptation.

    Args:
        num_factors (`int`):
            The number of low-rank factors in the chain. The weight update is the product of an
            `(r, in_features)` input factor, `num_factors - 2` intermediate `(r, r)` factors initialized to the
            identity, and an `(out_features, r)` output factor, scaled by `lora_alpha / r ** (num_factors - 1)`.
            `num_factors=2` recovers standard LoRA.
        r (`int`):
            The rank of the Deep LoRA factors.
        lora_alpha (`int`):
            The alpha parameter for the Deep LoRA scaling.
        lora_dropout (`float`):
            The dropout probability of the Deep LoRA layers.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the
            specified names will be replaced. When passing a string, a regex match will be performed. When
            passing a list of strings, either an exact match will be performed or it is checked if the name of
            the module ends with any of the passed strings. If this is specified as 'all-linear', then all
            linear modules are chosen, excluding the output layer. If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be
            raised -- in this case, you should specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be
            performed. When passing a list of strings, either an exact match will be performed or it is checked
            if the name of the module ends with any of the passed strings.
        init_lora_weights (`bool` | `Literal["gaussian"]`):
            How to initialize the outer factors of the chain. Passing `True` (default) initializes the input
            factor with the default `nn.Linear` initialization and the output factor with zeros, passing
            `"gaussian"` initializes the input factor with a Gaussian and the output factor with zeros, and
            passing `False` leaves both factors randomly initialized. Initializations that decompose the base
            weights (e.g. `'pissa'`, `'olora'`, `'corda'`, `'eva'`, `'loftq'`) are not supported. The
            intermediate factors are always initialized to the identity.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer
            indices that are specified in this list. If a single integer is passed, it will apply the
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        bias (`str`):
            Bias type for Deep LoRA. Can be `'none'`, `'all'` or `'lora_only'`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.

    Note: Deep LoRA currently supports `torch.nn.Linear` layers and does not compose with other LoRA variants
    (`use_dora`, `use_rslora`).
    """

    num_factors: int = field(
        default=3,
        metadata={
            "help": (
                "The number of low-rank factors in the Deep LoRA chain; the update is the product of an "
                "(r, in_features) factor, `num_factors - 2` intermediate (r, r) factors initialized to the "
                "identity, and an (out_features, r) factor. `num_factors=2` recovers standard LoRA."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.DEEPLORA
        if self.num_factors < 2:
            raise ValueError(f"`num_factors` must be at least 2, got {self.num_factors}.")
        if self.use_dora:
            raise ValueError("`use_dora` is not supported for Deep LoRA.")
        if self.use_rslora:
            raise ValueError(
                "`use_rslora` is not supported for Deep LoRA; the update is scaled by "
                "`lora_alpha / r ** (num_factors - 1)`."
            )
        if not (isinstance(self.init_lora_weights, bool) or self.init_lora_weights == "gaussian"):
            raise ValueError(
                "Deep LoRA only supports `init_lora_weights=True`, `False`, or `'gaussian'`; initializations "
                f"that decompose the base weights are not supported, got {self.init_lora_weights!r}."
            )
