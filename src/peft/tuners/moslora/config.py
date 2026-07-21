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


@dataclass
class MosLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`MosLoraModel`].

    MoSLoRA (Mixture-of-Subspaces LoRA, https://huggingface.co/papers/2406.11909) inserts a learnable `r x r` mixer
    matrix `M` between the LoRA matrices, so that the weight update becomes `Delta W = B @ M @ A` instead of
    `Delta W = B @ A`. Vanilla LoRA is the special case where `M` is the fixed identity matrix. The mixer is
    initialized with Kaiming uniform or orthogonal weights (the two best performing variants reported in the paper);
    note that a zero-initialized mixer does not converge because the gradients w.r.t. both the mixer and `B` vanish.

    This implementation is written from the paper's equations, as the authors' reference code is published without a
    license. It reuses the LoRA tuner machinery and supports all `LoraConfig` arguments except the LoRA variants that
    change the update structure (DoRA, aLoRA, VeLoRA, etc.) and string-valued `init_lora_weights` other than
    `"gaussian"`.

    Args:
        mixer_init (`str`):
            How to initialize the `r x r` mixer matrix. One of:
            - `"kaiming"` (default): Kaiming uniform init, the best performing variant in the paper.
            - `"orthogonal"`: orthogonal init, the second best variant in the paper.
            - `"identity"`: identity init, exactly recovering vanilla LoRA at initialization.
            - `"butterfly"`: the fixed butterfly mixer `[[I, I], [I, I]]` (with `r // 2` sized identity blocks) from
              the paper's fixed-mixer analysis, which fuses `2r` rank-1 subspaces. Requires an even rank `r`.
        trainable_mixer (`bool`):
            Whether the mixer matrix is learned jointly with the LoRA weights (default, the MoSLoRA method proper).
            When `False`, the mixer stays fixed at its initialization, e.g. `"butterfly"` gives the paper's fixed
            mixer variant.
        r (`int`):
            The rank of the adapter (same as in `LoraConfig`).
        target_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to apply the adapter to (same as in `LoraConfig`). Only `torch.nn.Linear` and
            `transformers.pytorch_utils.Conv1D` targets are supported.

    Note:
        MoSLoRA is registered dynamically when `peft.tuners.moslora` is imported. To load a saved MoSLoRA adapter
        with `PeftModel.from_pretrained` or `AutoPeftModel`, import `peft.tuners.moslora` first so that the
        `peft_type` `"MOSLORA"` resolves to this config class.
    """

    mixer_init: str = field(
        default="kaiming",
        metadata={
            "help": (
                "Initialization of the r x r mixer matrix, one of 'kaiming', 'orthogonal', 'identity', 'butterfly'."
            )
        },
    )
    trainable_mixer: bool = field(
        default=True,
        metadata={"help": "Whether the mixer matrix is trained jointly with the LoRA weights."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "MOSLORA"
        if self.mixer_init not in ("kaiming", "orthogonal", "identity", "butterfly"):
            raise ValueError(
                f"`mixer_init` must be one of 'kaiming', 'orthogonal', 'identity', 'butterfly', got "
                f"'{self.mixer_init}'."
            )
        if self.use_dora:
            raise ValueError("MoSLoRA does not support `use_dora=True`, please pass `use_dora=False`.")
        if isinstance(self.init_lora_weights, str) and self.init_lora_weights != "gaussian":
            raise ValueError(
                "MoSLoRA only supports `init_lora_weights=True`, `False` or 'gaussian', got "
                f"{self.init_lora_weights!r}."
            )
