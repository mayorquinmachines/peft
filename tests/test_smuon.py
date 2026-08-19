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
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.optimizers import create_smuon_optimizer
from peft.optimizers.smuon import zeropower_via_newtonschulz5

from .testing_utils import torch_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        return self.lin1(X)


def get_peft_simplenet(**config_kwargs):
    model = SimpleNet()
    config = LoraConfig(target_modules=["lin0", "lin1"], **config_kwargs)
    return get_peft_model(model, config)


def test_newtonschulz_orthogonalizes():
    # the iteration should push the singular values towards 1
    torch.manual_seed(0)
    G = torch.randn(32, 24)  # full-rank input: Newton-Schulz cannot revive exactly-zero singular values
    X = zeropower_via_newtonschulz5(G, steps=5)
    singular_values = torch.linalg.svdvals(X)
    assert torch.all(singular_values > 0.5) and torch.all(singular_values < 1.3)


def test_newtonschulz_tall_matrix():
    # covers the internal transpose path for matrices with more rows than columns
    torch.manual_seed(0)
    G = torch.randn(64, 16)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape
    singular_values = torch.linalg.svdvals(X)
    assert torch.all(singular_values > 0.5) and torch.all(singular_values < 1.3)


def test_smuon_helper_groups_parameters():
    model = get_peft_simplenet(r=4, modules_to_save=["layer_norm"])
    optim = create_smuon_optimizer(model=model, lr=1e-3, adam_lr=1e-4)

    smuon_groups = [group for group in optim.param_groups if group["use_smuon"]]
    adamw_groups = [group for group in optim.param_groups if not group["use_smuon"]]

    # one sMuon group per adapted module, each holding exactly the (B, A) factor pair
    assert len(smuon_groups) == 2
    for group in smuon_groups:
        param_b, param_a = group["params"]
        assert param_b.shape[1] == param_a.shape[0] == 4

    # the copied layer_norm lands in the AdamW fallback group
    assert len(adamw_groups) == 1
    assert all(param.ndim <= 1 for param in adamw_groups[0]["params"])


def test_smuon_helper_raises_without_low_rank_pairs():
    model = SimpleNet()
    with pytest.raises(ValueError, match="No low-rank factor pairs"):
        create_smuon_optimizer(model=model, lr=1e-3)


def test_smuon_optimizer_step_success():
    """
    Test if the optimizer is correctly created and the step function runs without any exception
    """
    torch.manual_seed(0)
    model = get_peft_simplenet(r=4).to(torch_device)
    optim = create_smuon_optimizer(model=model, lr=1e-2)

    loss_fct = torch.nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    loss_value = loss_fct(output, label)
    loss_value.backward()
    optim.step()

    trainable = [param for param in model.parameters() if param.requires_grad]
    assert all(param.grad is not None for param in trainable)


def test_smuon_optimizer_reduces_loss():
    torch.manual_seed(0)
    model = get_peft_simplenet(r=4).to(torch_device)
    optim = create_smuon_optimizer(model=model, lr=0.1)

    loss_fct = torch.nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)

    losses = []
    for _ in range(20):
        output = model(x).permute(0, 3, 1, 2)
        loss_value = loss_fct(output, label)
        losses.append(loss_value.item())
        loss_value.backward()
        optim.step()
        optim.zero_grad()

    assert losses[-1] < losses[0]


@pytest.mark.parametrize("split,changed,frozen", [(1.0, "lora_A", "lora_B"), (0.0, "lora_B", "lora_A")])
def test_smuon_split_controls_which_factor_moves(split, changed, frozen):
    torch.manual_seed(0)
    # non-zero init for both factors: with the default zero-initialized lora_B, the A-update is pulled back through
    # lora_B and can only start moving once lora_B becomes non-zero
    model = get_peft_simplenet(r=4, init_lora_weights="orthogonal").to(torch_device)
    optim = create_smuon_optimizer(model=model, lr=1e-2, split=split)

    before = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    model(x).sum().backward()
    optim.step()

    for name, param in model.named_parameters():
        if name not in before:
            continue
        if changed in name:
            assert not torch.allclose(param, before[name])
        elif frozen in name:
            assert torch.allclose(param, before[name])
