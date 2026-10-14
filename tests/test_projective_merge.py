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

import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.tuners import lora
from peft.utils.merge_utils import task_arithmetic
from peft.utils.projective_merge import projective_merge


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)
        self.dtype = torch.float

    def forward(self, X):
        X = X.to(self.dtype)
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


def _gap_objective(task_tensors, merged):
    # the data-free objective of the paper: the merged vector should be close to each task vector, measured along
    # that task vector
    stacked = torch.stack(task_tensors)
    gaps = (stacked * (merged.unsqueeze(0) - stacked)).flatten(1).sum(dim=1)
    return (gaps**2).sum()


class TestProjectiveMerge:
    def test_output_shape_and_dtype_preserved(self):
        torch.manual_seed(0)
        task_tensors = [torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)]
        weights = torch.tensor([1.0, 1.0, 1.0])
        merged = projective_merge(task_tensors, weights, num_steps=10)
        assert merged.shape == task_tensors[0].shape
        assert merged.dtype == task_tensors[0].dtype

    def test_conv_shaped_input_shape_preserved(self):
        torch.manual_seed(0)
        task_tensors = [torch.randn(4, 3, 3, 3), torch.randn(4, 3, 3, 3)]
        weights = torch.tensor([1.0, 1.0])
        merged = projective_merge(task_tensors, weights, num_steps=10)
        assert merged.shape == task_tensors[0].shape
        assert torch.isfinite(merged).all()

    def test_adaptive_coefficients_normalize_task_magnitude(self):
        # with no optimization steps, the merged vector is sum_i weights_i * task_tensor_i / ||task_tensor_i||, i.e.
        # the result does not depend on the magnitudes of the task vectors, unlike task arithmetic
        torch.manual_seed(0)
        task_tensors = [torch.randn(8, 6), torch.randn(8, 6)]
        weights = torch.tensor([1.0, 1.0])
        merged = projective_merge(task_tensors, weights, num_steps=0)
        expected = sum(tensor / tensor.norm() for tensor in task_tensors)
        assert torch.allclose(merged, expected)

        scaled_tensors = [task_tensors[0], 5.0 * task_tensors[1]]
        merged_scaled = projective_merge(scaled_tensors, weights, num_steps=0)
        assert torch.allclose(merged, merged_scaled)

        linear_scaled = task_arithmetic(scaled_tensors, weights)
        assert not torch.allclose(task_arithmetic(task_tensors, weights), linear_scaled)

    def test_optimization_reduces_gap_objective(self):
        torch.manual_seed(0)
        task_tensors = [torch.randn(12, 10), torch.randn(12, 10), torch.randn(12, 10)]
        weights = torch.tensor([1.0, 1.0, 1.0])
        merged_init = projective_merge(task_tensors, weights, num_steps=0)
        merged_final = projective_merge(task_tensors, weights, num_steps=100)
        objective_init = _gap_objective(task_tensors, merged_init)
        objective_final = _gap_objective(task_tensors, merged_final)
        assert objective_final < objective_init

    def test_zero_task_vector_is_ignored(self):
        torch.manual_seed(0)
        task_tensors = [torch.randn(8, 6), torch.zeros(8, 6)]
        weights = torch.tensor([1.0, 1.0])
        merged = projective_merge(task_tensors, weights, num_steps=10)
        assert torch.isfinite(merged).all()

    def test_density_pruning(self):
        torch.manual_seed(0)
        task_tensors = [torch.randn(8, 6), torch.randn(8, 6)]
        weights = torch.tensor([1.0, 1.0])
        merged = projective_merge(task_tensors, weights, density=0.5, num_steps=10)
        assert merged.shape == task_tensors[0].shape
        assert torch.isfinite(merged).all()


class TestDogeSvdWeightedAdapter:
    def _get_model(self):
        config = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        torch.manual_seed(0)
        model = get_peft_model(MLP(), config, adapter_name="adapter1")
        torch.manual_seed(1)
        model.add_adapter("adapter2", config)
        return model

    def test_add_weighted_adapter_doge_svd(self):
        model = self._get_model()
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[1.0, 1.0],
            adapter_name="doge_merged",
            combination_type="doge_svd",
        )
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[1.0, 1.0],
            adapter_name="linear_merged",
            combination_type="linear",
        )

        found_lora_layer = False
        for module in model.modules():
            if isinstance(module, lora.LoraLayer):
                found_lora_layer = True
                delta_doge = module.get_delta_weight("doge_merged")
                delta_linear = module.get_delta_weight("linear_merged")
                assert torch.isfinite(delta_doge).all()
                assert not torch.allclose(delta_doge, torch.zeros_like(delta_doge))
                # the projective merge must not degenerate to the plain linear combination
                assert not torch.allclose(delta_doge, delta_linear)
        assert found_lora_layer

        # check that using the merged adapter does not raise
        model.set_adapter("doge_merged")
        model(torch.randn(3, 10))

    def test_add_weighted_adapter_doge_svd_with_density(self):
        model = self._get_model()
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[1.0, 1.0],
            adapter_name="doge_merged",
            combination_type="doge_svd",
            density=0.5,
        )
        for module in model.modules():
            if isinstance(module, lora.LoraLayer):
                delta_doge = module.get_delta_weight("doge_merged")
                assert torch.isfinite(delta_doge).all()
