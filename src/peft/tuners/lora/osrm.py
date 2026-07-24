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

import warnings
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import check_target_module_exists
from peft.utils.other import get_pattern_key

from .config import LoraConfig
from .eva import (
    UNSUPPORTED_LORA_MODULES,
    _Hook,
    forward_fn_dict,
    get_device_with_meta_params,
    move_inputs_to_device,
    prepare_layer_inputs_fn_language_modeling,
    prepare_model_inputs_fn_language_modeling,
)
from .layer import LoraLayer


class CovarianceHook(_Hook):
    """
    A forward hook that accumulates the (uncentered) covariance of layer inputs. The hook is designed to be registered
    to a PyTorch module using the `register_forward_hook` method.

    For OSRM, the eigenvectors of the input covariance associated with the smallest eigenvalues define the
    low-interference subspace that LoRA's A matrix is initialized into.

    Args:
        name (str): Name of the layer to which this hook is attached.
        prepare_layer_inputs_fn (Optional[Callable]): Function to prepare layer inputs for the covariance computation.
        gather_distributed_inputs (bool): Whether to gather the layer inputs from all ranks.
    """

    def __init__(self, **base_class_kwargs):
        super().__init__(**base_class_kwargs)
        self.covariance = None
        self.n_samples = 0

    @torch.no_grad()
    def __call__(self, model, input, output):
        states = self.prepare_layer_inputs(input)
        states = self.gather_layer_inputs(states)
        if states.size(0) == 0:
            return
        states = states.to(torch.float32)
        covariance = states.T @ states
        if self.covariance is None:
            self.covariance = covariance
        else:
            self.covariance += covariance
        self.n_samples += states.size(0)


@torch.no_grad()
def get_osrm_state_dict(
    model: torch.nn.Module,
    dataloader: Iterable,
    peft_config: Optional[LoraConfig] = None,
    forward_fn: Optional[Callable] = forward_fn_dict,
    prepare_model_inputs_fn: Optional[Callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[Callable, dict[str, Callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
) -> dict:
    """
    Compute the OSRM initialization for each targeted layer in the model.

    This function accumulates the covariance of the inputs of each adapter layer over the dataloader and computes its
    eigendecomposition. The eigenvectors associated with the `r` smallest eigenvalues form the low-interference
    subspace into which LoRA's A matrix is initialized, as proposed in
    [OSRM](https://huggingface.co/papers/2505.22934) (Orthogonal Subspaces for Robust Model Merging). The dataloader
    should contain out-of-task data, i.e. data from tasks other than the one the adapter will be fine-tuned on, so
    that the resulting adapter minimally interferes with those tasks when merged.

    Args:
        model (torch.nn.Module): The model to compute the OSRM initialization for. Does not need to be a PeftModel.
        dataloader (Iterable): The dataloader with out-of-task data to use for the forward passes.
        peft_config (Optional[LoraConfig]):
            The configuration for the LoRA layers. Only required if `model` is not a PeftModel.
        forward_fn (Callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[Callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the covariance computation based on the
            original model inputs. For example for language modeling the attention mask is used to determine which
            indices are padding tokens and should not be used. Any function defined here expects two arguments:
            `model_input` and `peft_config`. `peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling` is used
            by default.
        prepare_layer_inputs_fn (Union[Callable, Dict[str, Callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for the
            covariance computation for that particular layer. Any custom function defined here expects three arguments:
            `layer_input`, `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be
            found in peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling and works for language modeling.
        adapter_name (str): The name of the adapter to compute the OSRM initialization for.
        gather_distributed_inputs (bool):
            Whether to gather the layer inputs from all ranks. Default is True meaning in a distributed setting the
            layer inputs will be gathered from all ranks for the covariance computation. For non-distributed settings
            this argument is ignored. Set to False if you are using a non-distributed dataloader in a distributed
            setting.
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        osrm_state_dict (dict): The state dictionary containing the OSRM initialization for each layer.
    """

    def target_module_check_fn_peft_model(name, module, unsupported_lora_modules):
        "check if a module is an adapter module via base_layer attribute"
        return hasattr(module, "base_layer") and not isinstance(module, unsupported_lora_modules)

    def target_module_check_fn_default(name, module, peft_config):
        "check if a module is an adapter module via target_modules"
        is_target_module = True
        if peft_config.target_modules is not None:
            is_target_module = check_target_module_exists(peft_config, name)
        # Conv1D for GPT2 support
        return isinstance(module, (torch.nn.Linear, Conv1D)) and is_target_module

    # dataloader is not empty
    if len(dataloader) == 0:
        raise ValueError("dataloader is empty")

    is_peft_model = hasattr(model, "peft_config")

    # get peft_config
    if is_peft_model and peft_config is None:
        peft_config = model.peft_config[adapter_name]
    elif peft_config is None:
        raise ValueError("peft_config is required if model is not a PeftModel")

    # setup context and target module check function
    if is_peft_model:
        ctx = model.disable_adapter()
        target_module_check_fn = partial(
            target_module_check_fn_peft_model, unsupported_lora_modules=UNSUPPORTED_LORA_MODULES
        )
    else:
        ctx = nullcontext()
        target_module_check_fn = partial(target_module_check_fn_default, peft_config=peft_config)

    training = model.training
    device = get_device_with_meta_params(model)
    model.eval()

    with ctx:
        hooks = {}
        for name, module in model.named_modules():
            if not target_module_check_fn(name, module):
                continue
            if isinstance(prepare_layer_inputs_fn, Mapping):
                fn = prepare_layer_inputs_fn.pop(name, None)
            else:
                fn = prepare_layer_inputs_fn
            hook = CovarianceHook(
                name=name, prepare_layer_inputs_fn=fn, gather_distributed_inputs=gather_distributed_inputs
            )
            handle = module.register_forward_hook(hook)
            hooks[name] = (hook, handle)
        if isinstance(prepare_layer_inputs_fn, Mapping) and len(prepare_layer_inputs_fn) > 0:
            raise ValueError(
                "prepare_layer_inputs_fn is a mapping but the following module names were not found in the model: "
                f"{prepare_layer_inputs_fn.keys()}"
            )

        if show_progress_bar and (not dist.is_initialized() or dist.get_rank() == 0):
            iterable = tqdm(dataloader, position=0, leave=False)
        else:
            iterable = dataloader

        # accumulate the input covariance of all targeted layers in a single pass over the dataloader
        for inputs in iterable:
            if device is not None:
                inputs = move_inputs_to_device(inputs, device)
            if prepare_model_inputs_fn is not None:
                model_inputs_for_hooks = prepare_model_inputs_fn(inputs, peft_config)
            else:
                model_inputs_for_hooks = deepcopy(inputs)
            for hook, _ in hooks.values():
                hook.model_input = model_inputs_for_hooks
            forward_fn(model, inputs)

        for _, handle in hooks.values():
            handle.remove()

    osrm_state_dict = {}
    for name, (hook, _) in hooks.items():
        # layers that never received inputs fall back to the default initialization
        if hook.covariance is None:
            continue
        r = peft_config.rank_pattern.get(get_pattern_key(peft_config.rank_pattern.keys(), name), peft_config.r)
        r = min(r, hook.covariance.size(0))
        # eigenvalues are returned in ascending order, so the first r eigenvectors span the subspace where the
        # out-of-task data has the least variance, i.e. the low-interference subspace (Eq. 3 in the paper)
        eigenvectors = torch.linalg.eigh(hook.covariance).eigenvectors
        osrm_state_dict[name] = eigenvectors[:, :r].T.contiguous()

    # restore model state
    model.train(training)

    # move tensors to device
    if device is not None:
        osrm_state_dict = {k: v.to(device) for k, v in osrm_state_dict.items()}

    return osrm_state_dict


def _load_osrm_state_dict(
    model: torch.nn.Module,
    osrm_state_dict: dict,
    adapter_name: str,
):
    missing_osrm_inits = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if name in osrm_state_dict:
            w = osrm_state_dict[name]
            lora_A_weight = module.lora_A[adapter_name].weight
            if w.shape != lora_A_weight.shape:
                raise ValueError(
                    f"Shape mismatch for layer {name}: the osrm_state_dict entry has shape {tuple(w.shape)} but "
                    f"lora_A has shape {tuple(lora_A_weight.shape)}. Ensure that the osrm_state_dict was computed "
                    "with the same model, rank, and rank_pattern."
                )
            lora_A_weight.copy_(w)
        else:
            missing_osrm_inits.append(name)

    if missing_osrm_inits:
        warnings.warn(
            "the following layers were initialized with the default initialization because they "
            f"were not found in the osrm state_dict: {missing_osrm_inits}\ncurrently the "
            f"following lora modules are not supported by OSRM: {UNSUPPORTED_LORA_MODULES}"
        )


@torch.no_grad()
def initialize_lora_osrm_weights(
    model: torch.nn.Module,
    dataloader: Optional[Iterable] = None,
    osrm_state_dict: Optional[dict] = None,
    forward_fn: Optional[Callable] = forward_fn_dict,
    prepare_model_inputs_fn: Optional[Callable] = prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: Union[Callable, dict[str, Callable], None] = prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
):
    """
    Initialize the weights of the LoRA layers using the OSRM method.

    This function initializes LoRA's A matrix with the eigenvectors associated with the smallest eigenvalues of the
    covariance of out-of-task layer inputs, as proposed in [OSRM](https://huggingface.co/papers/2505.22934)
    (Orthogonal Subspaces for Robust Model Merging). LoRA's B matrix remains zero-initialized, so the adapter is a
    no-op before training. Both A and B stay trainable; constraining A to this low-interference subspace at
    initialization reduces cross-task interference when multiple task-specific adapters are merged.

    Args:
        model (PeftModel): The peft model to initialize.
        dataloader (Optional[Iterable]):
            The dataloader with out-of-task data to use for the forward passes. If None, osrm_state_dict needs to be
            provided.
        osrm_state_dict (Optional[dict]):
            The state_dict to load into the model. If None, a dataloader needs to be provided and the state_dict will
            be computed using `get_osrm_state_dict`.
        forward_fn (Callable):
            The forward function to use for the forward pass. Takes two arguments: `model` and `inputs`. Default
            behavior is `return model(**inputs)`
        prepare_model_inputs_fn (Optional[Callable]):
            This function receives the model inputs and the peft_config and passes the output to
            `prepare_layer_inputs_fn`. Can be used to modify the input to the covariance computation based on the
            original model inputs. Any function defined here expects two arguments: `model_input` and `peft_config`.
            `peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling` is used by default.
        prepare_layer_inputs_fn (Union[Callable, Dict[str, Callable], None]):
            This function receives the layer inputs, the model inputs (potentially modified by
            `prepare_model_inputs_fn`) and the name of the layer and returns the inputs that should be used for the
            covariance computation for that particular layer. Any custom function defined here expects three arguments:
            `layer_input`, `model_input`, and `layer_name` and should return a 2d tensor. The default logic can be
            found in peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling and works for language modeling.
        adapter_name (str): The name of the adapter to initialize the weights for.
        gather_distributed_inputs (bool):
            Whether to gather the layer inputs from all ranks. Default is True meaning in a distributed setting the
            layer inputs will be gathered from all ranks for the covariance computation. For non-distributed settings
            this argument is ignored. Set to False if you are using a non-distributed dataloader in a distributed
            setting.
        show_progress_bar (bool): Whether to show a progress bar. Default is True.

    Returns:
        model (torch.nn.Module): The model with the initialized LoRA weights.
    """
    if not hasattr(model, "peft_config"):
        raise ValueError("model must be a PeftModel")

    # osrm currently only works with a single active adapter
    if len(model.active_adapters) > 1:
        raise ValueError("`initialize_lora_osrm_weights` currently only works with a single active adapter")

    # initialize_lora_osrm_weights only works with `init_lora_weights='osrm'`
    if model.peft_config[adapter_name].init_lora_weights != "osrm":
        raise ValueError("`initialize_lora_osrm_weights` can only be used with `init_lora_weights='osrm'`")

    # compute covariance and eigendecomposition
    if osrm_state_dict is None:
        if dataloader is None:
            raise ValueError("dataloader is required if osrm_state_dict is not provided")
        osrm_state_dict = get_osrm_state_dict(
            model=model,
            dataloader=dataloader,
            forward_fn=forward_fn,
            prepare_model_inputs_fn=prepare_model_inputs_fn,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            adapter_name=adapter_name,
            gather_distributed_inputs=gather_distributed_inputs,
            show_progress_bar=show_progress_bar,
        )

    _load_osrm_state_dict(model, osrm_state_dict, adapter_name)
