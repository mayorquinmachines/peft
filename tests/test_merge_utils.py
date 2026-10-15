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

import pytest
import torch

from peft.utils.merge_utils import breadcrumbs, breadcrumbs_prune, magnitude_based_pruning, ties


class TestBreadcrumbsPrune:
    def test_prunes_smallest_and_largest_values(self):
        tensor = torch.tensor([10.0, 2.0, 1.0, -1.5, 0.5, -0.25, 4.0, -8.0])
        # density=0.5 keeps the 4 largest magnitudes (10, 8, 4, 2), gamma=0.25 prunes the 2 largest (10, 8)
        pruned = breadcrumbs_prune(tensor, density=0.5, gamma=0.25)
        expected = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
        assert torch.equal(pruned, expected)

    def test_zero_gamma_matches_magnitude_based_pruning(self):
        torch.manual_seed(0)
        tensor = torch.randn(10, 10)
        pruned = breadcrumbs_prune(tensor, density=0.5, gamma=0.0)
        assert torch.equal(pruned, magnitude_based_pruning(tensor, density=0.5))


class TestBreadcrumbs:
    task_tensors = [
        torch.tensor([10.0, 2.0, 1.0, -1.5, 3.0, -4.0, 0.5, -0.25]),
        torch.tensor([1.0, 2.5, -8.0, -1.5, 3.5, -4.5, 0.5, 0.25]),
    ]
    weights = torch.tensor([1.0, 1.0])

    def test_merge_prunes_outliers_and_follows_majority_sign(self):
        # density=0.75 keeps 6 of 8 values, gamma=0.125 prunes the largest (10.0 and -8.0)
        merged = breadcrumbs(self.task_tensors, self.weights, density=0.75, gamma=0.125)
        # after pruning: [0, 2, 1, -1.5, 3, -4, 0, 0] and [1, 2.5, 0, -1.5, 3.5, -4.5, 0, 0];
        # disjoint merge over the majority sign
        expected = torch.tensor([1.0, 2.25, 1.0, -1.5, 3.25, -4.25, 0.0, 0.0])
        assert torch.allclose(merged, expected)

    def test_outlier_values_do_not_leak_into_merge(self):
        merged = breadcrumbs(self.task_tensors, self.weights, density=0.75, gamma=0.125)
        # with ties, the outliers 10.0 and -8.0 would dominate the first and third entries
        merged_ties = ties(self.task_tensors, self.weights, density=0.75)
        assert not torch.allclose(merged, merged_ties)

    def test_gamma_smaller_than_density_is_required(self):
        task_tensors = [torch.ones(4), torch.ones(4)]
        weights = torch.tensor([1.0, 1.0])
        with pytest.raises(ValueError, match="gamma should be smaller than density"):
            breadcrumbs(task_tensors, weights, density=0.5, gamma=0.5)
