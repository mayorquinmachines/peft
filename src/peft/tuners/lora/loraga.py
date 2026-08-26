# Copyright 2025-present the HuggingFace Inc. team.
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

# Reference code: https://github.com/Outsider565/LoRA-GA
# Reference paper: https://arxiv.org/abs/2407.05000

import os
from collections.abc import Callable
from typing import Any, Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel


def get_target_modules(model: nn.Module, config: LoraConfig):
    """
    Iterate over LoRA-GA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear` or `Conv1D`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, (nn.Linear, Conv1D)):
            yield name, module


def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device


@torch.no_grad()
def preprocess_loraga(
    model: nn.Module,
    lora_config: LoraConfig,
    train_step: Callable[[], None],
    cache_file: Optional[str] = None,
):
    """
    Build necessary LoRA-GA fields for a model by estimating gradients.

    For each linear layer, gradients will be estimated by running the provided train_step callback. These gradients are
    then attached to the modules and used during initialization.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `lora_config.lora_ga_config` should be set.
        train_step (`Callable[[], None]`):
            Callback to run gradient estimation. Typically you should run model forward and backward passes in this
            callback. The gradients will be accumulated across all calls within this callback.
        cache_file (`Optional[str]`):
            Optional path to cache file for saving/loading gradients. If provided and the file exists, gradients will
            be loaded from cache. Otherwise, gradients will be estimated and saved to this path.

    Upon completion, the following fields are set for each target module:
        _peft_loraga_grad (`torch.Tensor`):
            Accumulated gradient for the weight matrix.
        _peft_loraga_grad_left_cov / _peft_loraga_grad_right_cov (`torch.Tensor`, only when
        `lora_config.lora_ga_config.n_steps > 1`):
            Gradient second moments (in canonical `(out_features, in_features)` orientation) accumulated across the
            `n_steps` probes, used by LoRA-GA² multi-step initialization and spectrum-aware rank allocation.
    """
    if lora_config.lora_ga_config is None:
        raise ValueError(
            "If you want to use LoRA-GA, please initialize the LoraConfig with "
            "init_lora_weights='lora_ga' and lora_ga_config=LoraGAConfig(...)."
        )

    # Populate target_modules from defaults if empty
    # This logic mirrors BaseTuner._prepare_adapter_config which runs after get_peft_model.
    # Since preprocess_loraga is called before get_peft_model, we need to handle this ourselves.
    if lora_config.target_modules is None:
        model_config = LoraModel.get_model_config(model)
        target_modules = LoraModel.target_module_mapping.get(model_config["model_type"])
        if target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        lora_config.target_modules = set(target_modules)

    # Check for quantized models - LoRA-GA requires full-precision gradients
    for name, module in get_target_modules(model, lora_config):
        if hasattr(module, "quant_state"):
            raise ValueError(
                f"LoRA-GA does not support quantized models. Found quantized module: '{name}'. "
                "LoRA-GA requires full-precision gradients during preprocessing."
            )

    # If cache exists, load from cache
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        for name, module in get_target_modules(model, lora_config):
            module._peft_loraga_grad = cache[f"{name}._peft_loraga_grad"]
            left_cov_key = f"{name}._peft_loraga_grad_left_cov"
            if left_cov_key in cache:
                module._peft_loraga_grad_left_cov = cache[left_cov_key]
                module._peft_loraga_grad_right_cov = cache[f"{name}._peft_loraga_grad_right_cov"]
    else:
        # Estimate gradients by running train_step
        estimate_gradients(model, lora_config, train_step)

        # Save cache to disk if specified
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in get_target_modules(model, lora_config):
                cache[f"{name}._peft_loraga_grad"] = module._peft_loraga_grad
                if hasattr(module, "_peft_loraga_grad_left_cov"):
                    cache[f"{name}._peft_loraga_grad_left_cov"] = module._peft_loraga_grad_left_cov
                    cache[f"{name}._peft_loraga_grad_right_cov"] = module._peft_loraga_grad_right_cov

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)


def estimate_gradients(
    model: nn.Module,
    lora_config: LoraConfig,
    train_step: Callable[[], None],
):
    """
    Estimate gradients for LoRA-GA initialization.

    This function enables gradient computation ONLY on target module weights and runs the train_step callback. This is
    more memory-efficient than enabling gradients globally.

    When `lora_config.lora_ga_config.n_steps > 1`, the callback is run once per probe step and gradient second moments
    are accumulated across steps (the LoRA-GA² multi-step gradient probe). Gradients are cleared between steps so that
    each probe captures a distinct snapshot of the gradient trajectory. With `n_steps == 1` (default) this reduces to
    the standard single-probe LoRA-GA estimation.
    """
    lora_ga_config = lora_config.lora_ga_config
    n_steps = lora_ga_config.n_steps if lora_ga_config is not None else 1
    if n_steps < 1:
        raise ValueError(f"lora_ga_config.n_steps must be a positive integer, got {n_steps}.")

    # Remember original training state
    was_training = model.training
    model.train()

    # Get target modules list once for efficiency
    target_module_list = list(get_target_modules(model, lora_config))

    # Check if any supported layers were found
    if not target_module_list:
        raise ValueError(
            "No supported layers found for LoRA-GA initialization. "
            "LoRA-GA only supports nn.Linear and Conv1D layers. "
            "Please ensure your model contains at least one of these layer types in target_modules."
        )

    # Initialize gradient storage and count for each target module
    for name, module in target_module_list:
        module._peft_loraga_grad_count = 0

    # Memory-efficient gradient computation: disable gradients for all parameters first,
    # then enable only for target module weights
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = False

    # Enable gradients ONLY for target module weights
    for name, module in target_module_list:
        module.weight.requires_grad = True

    # Register backward hooks to count gradient computations
    hooks = []

    def backward_hook(module, grad_input, grad_output):
        module._peft_loraga_grad_count += 1

    for name, module in target_module_list:
        hook = module.register_full_backward_hook(backward_hook)
        hooks.append(hook)

    # Enable gradient computation and run train_step once per probe step
    with torch.enable_grad():
        for _ in range(n_steps):
            counts_before = {name: module._peft_loraga_grad_count for name, module in target_module_list}
            train_step()
            for name, module in target_module_list:
                step_count = module._peft_loraga_grad_count - counts_before[name]
                grad = module.weight.grad
                if step_count == 0 or grad is None:
                    continue
                grad = grad.detach()
                # Running sum for the mean gradient; divided by the total count below, which
                # reproduces the single-probe estimate exactly when n_steps == 1.
                module._peft_loraga_grad_sum = grad + getattr(module, "_peft_loraga_grad_sum", 0)
                if n_steps > 1:
                    # LoRA-GA² multi-step probe: accumulate second moments of the per-step average
                    # gradient in canonical (out_features, in_features) orientation.
                    g = grad.to(torch.float32) / step_count
                    if isinstance(module, Conv1D):
                        g = g.t()
                    if not hasattr(module, "_peft_loraga_grad_left_cov"):
                        module._peft_loraga_grad_left_cov = torch.zeros(
                            g.shape[0], g.shape[0], dtype=torch.float32, device=g.device
                        )
                        module._peft_loraga_grad_right_cov = torch.zeros(
                            g.shape[1], g.shape[1], dtype=torch.float32, device=g.device
                        )
                        module._peft_loraga_grad_steps = 0
                    module._peft_loraga_grad_left_cov += g @ g.t()
                    module._peft_loraga_grad_right_cov += g.t() @ g
                    module._peft_loraga_grad_steps += 1
                # Clear gradients so that the next probe captures a fresh snapshot
                module.weight.grad = None

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Restore original requires_grad state for all parameters
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad = original_requires_grad[name]

    # Average gradients and clean up temporary fields
    for name, module in target_module_list:
        if module._peft_loraga_grad_count > 0:
            module._peft_loraga_grad = module._peft_loraga_grad_sum / module._peft_loraga_grad_count
            del module._peft_loraga_grad_sum
        module.weight.grad = None
        del module._peft_loraga_grad_count
        if hasattr(module, "_peft_loraga_grad_steps"):
            module._peft_loraga_grad_left_cov /= module._peft_loraga_grad_steps
            module._peft_loraga_grad_right_cov /= module._peft_loraga_grad_steps
            del module._peft_loraga_grad_steps

    # Restore original training state
    if not was_training:
        model.eval()


def get_spectrum_aware_rank_pattern(model: nn.Module, lora_config: LoraConfig, min_rank: int = 1) -> dict[str, int]:
    """
    Allocate per-layer LoRA ranks from the multi-step gradient spectrum (LoRA-GA²).

    Uses the gradient second moments attached by `preprocess_loraga` when run with
    `lora_config.lora_ga_config.n_steps > 1`. Each layer's importance is the spectral mass of its multi-step gradients
    (the trace of its gradient second moment, i.e. the mean squared Frobenius norm of the per-step gradients), and
    ranks are assigned proportionally to importance while preserving the average rank `lora_config.r` across target
    layers. Ranks are clipped to `[min_rank, min(out_features, in_features)]` per layer.

    Args:
        model (`nn.Module`):
            Model that was preprocessed with `preprocess_loraga` using `LoraGAConfig(n_steps>1)`.
        lora_config (`LoraConfig`):
            Lora configuration of the model. The average rank is taken from `lora_config.r`.
        min_rank (`int`):
            Minimum rank assigned to any target layer. Default: 1.

    Returns:
        `dict[str, int]`: Mapping from module name to rank, suitable for `LoraConfig.rank_pattern`.
    """
    importances: dict[str, float] = {}
    max_ranks: dict[str, int] = {}
    for name, module in get_target_modules(model, lora_config):
        if not hasattr(module, "_peft_loraga_grad_left_cov"):
            raise ValueError(
                f"Multi-step gradient statistics not found on module '{name}'. "
                "Spectrum-aware rank allocation requires running preprocess_loraga with "
                "LoraGAConfig(n_steps>1) first."
            )
        left_cov = module._peft_loraga_grad_left_cov
        importances[name] = left_cov.diagonal().sum().item()
        max_ranks[name] = min(left_cov.shape[0], module._peft_loraga_grad_right_cov.shape[0])

    total_importance = sum(importances.values())
    n_layers = len(importances)
    rank_pattern: dict[str, int] = {}
    for name, importance in importances.items():
        if total_importance > 0:
            rank = round(lora_config.r * n_layers * importance / total_importance)
        else:
            rank = lora_config.r
        rank_pattern[name] = int(max(min_rank, min(max_ranks[name], rank)))
    return rank_pattern
