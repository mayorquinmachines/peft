# Copyright 2023-present the HuggingFace Inc. team.
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

"""Signal-to-noise ratio (SNR) importance scoring for adaptive rank allocation.

Adapted from "A Bayesian Interpretation of Adaptive Low-Rank Adaptation" (https://arxiv.org/abs/2409.10673), which
motivates SNR = mean^2 / variance of a Bayesian posterior over the weights as a theoretically grounded alternative to
AdaLoRA's sensitivity-based importance. Here the posterior variance is approximated by an Adam-style exponential moving
average of the squared gradients, the parameter-free proxy the paper validates as a drop-in replacement when training
with Adam-like optimizers instead of IVON.
"""

import torch


def update_grad_sq_ema(exp_avg_grad_sq: torch.Tensor, grad: torch.Tensor, beta: float) -> torch.Tensor:
    """In-place EMA update of the gradient second moment (Adam-style variance proxy).

    No bias correction is applied: at a given step the correction factor is identical for all parameters, so it does
    not change the ranking of the resulting importance scores that budget allocation is based on.
    """
    exp_avg_grad_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)
    return exp_avg_grad_sq


def snr_importance_score(weight: torch.Tensor, exp_avg_grad_sq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SNR-based importance score: weight^2 / (E[grad^2] + eps)."""
    return weight.detach().pow(2) / (exp_avg_grad_sq + eps)
