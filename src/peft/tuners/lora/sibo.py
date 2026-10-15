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

"""Initial-residual injection for LoRA, adapted from SIBO (https://arxiv.org/abs/2402.11896).

To counteract over-smoothing in Transformer-based LLMs, the input to LoRA's low-rank branch is replaced by a
convex mix of the layer input `h` and the model's initial token representation `h0` (the output of the input
embedding layer):

    h~ = (1 - lambda) * h + lambda * h0

The frozen base path is left untouched and no additional trainable parameters are introduced; `lambda` is a
fixed hyperparameter (see `LoraConfig.sibo_lambda`). The initial residual is captured by a forward hook on the
model's input embedding layer and injected into each LoRA layer through the PEFT forward hooks, so it also
tracks the current tokens during cached generation.
"""

import torch


def mix_initial_residual(x: torch.Tensor, initial_residual: torch.Tensor, sibo_lambda: float) -> torch.Tensor:
    """
    Blend the initial residual into the branch input: `(1 - lambda) * x + lambda * h0`.

    Shape mismatches that occur during generation are aligned: with a KV cache, `x` only holds the newly
    processed tokens, and with beam search, the batch of `x` is repeated relative to the prompt's residual.
    """
    if initial_residual.dim() == x.dim() and initial_residual.shape[-2] != x.shape[-2]:
        initial_residual = initial_residual[..., -x.shape[-2] :, :]
    if initial_residual.shape[0] != x.shape[0] and x.shape[0] % initial_residual.shape[0] == 0:
        initial_residual = initial_residual.repeat_interleave(x.shape[0] // initial_residual.shape[0], dim=0)
    initial_residual = initial_residual.to(device=x.device, dtype=x.dtype)
    return (1.0 - sibo_lambda) * x + sibo_lambda * initial_residual


def make_residual_capture_hook(sibo_state: dict):
    """
    Forward hook for the input embedding layer that stores its latest output as the initial residual `h0`.

    Since the embedding layer is re-invoked on every forward pass (including each cached generation step), the
    captured residual always corresponds to the tokens currently being processed.
    """

    def _capture_hook(module, args, output):
        sibo_state["initial_residual"] = output

    return _capture_hook


def make_residual_injection_hook(sibo_state: dict):
    """
    Pre-forward hook for LoRA layers that injects the captured initial residual into the layer's kwargs, from
    where it is forwarded to `SiboLinearVariant.forward`.
    """

    def _injection_hook(target, args, kwargs):
        residual = sibo_state.get("initial_residual")
        if residual is not None:
            kwargs["sibo_initial_residual"] = residual
        return args, kwargs

    return _injection_hook
