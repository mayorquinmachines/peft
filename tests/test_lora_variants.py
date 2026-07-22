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

import dataclasses
from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.factored_norm import factored_weight_norm
from peft.tuners.lora.layer import Conv1d as LoraConv1d
from peft.tuners.lora.layer import Conv2d as LoraConv2d
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.variants import (
    ALoraLinearVariant,
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
    calculate_alora_offsets,
    get_alora_offsets_for_forward,
    get_alora_offsets_for_generate,
)

from .testing_common import hub_online_once


# Custom model featuring embeddings and a 'visual stack'
class CustomModel(nn.Module):
    """pytorch module that contains common targetable layers (linear, embedding, conv, ...)"""

    def __init__(self, num_embeddings=100, embedding_dim=16, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.dummy_conv1d_output_dim = 32 * 10
        self.dummy_conv2d_output_dim = 16 * 10 * 10
        self.linear1 = nn.Linear(self.dummy_conv1d_output_dim + self.dummy_conv2d_output_dim, 64)
        self.linear2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, dummy_image_input):
        # Path 1: Embedding -> Conv1d
        x1 = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        x1 = x1.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        x1 = self.relu(self.conv1d(x1))  # (batch_size, 32, seq_len)
        x1_flat = self.flatten(x1)
        # Path 2: Conv2d -> Linear
        x2 = self.relu(self.conv2d(dummy_image_input))  # (batch_size, 16, H, W)
        x2_flat = self.flatten(x2)  # (batch_size, 16*H*W)
        # Combine or select paths if making a functional model.
        # For this test, we mainly care about layer types, so forward might not be fully executed.
        # Let's use x2_flat for subsequent linear layers.
        output = self.relu(self.linear1(torch.concat([x1_flat, x2_flat], dim=1)))
        output = self.linear2(output)
        return output


# Used for testing alora_offsets for aLoRA
class DummyLM(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs

    def forward(self, X=None, embeds=None, num_beams=None, alora_offsets=None):
        if X is not None:
            embeds = self.embed(X)
        return self.linear(embeds)


class MockTransformerWrapper:
    """Mock class to behave like a transformers model.

    This is needed because the tests initialize the model by calling transformers_class.from_pretrained.

    """

    @classmethod
    def from_pretrained(cls):
        # set the seed so that from_pretrained always returns the same model
        torch.manual_seed(0)

        dtype = torch.float32

        return DummyLM().to(dtype)


VARIANT_MAP = {
    "dora": {
        LoraLinear: DoraLinearVariant,
        LoraEmbedding: DoraEmbeddingVariant,
        LoraConv1d: DoraConv1dVariant,
        LoraConv2d: DoraConv2dVariant,
    },
    "alora": {
        LoraLinear: ALoraLinearVariant,
    },
}


TEST_CASES = [
    (
        "dora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2", "conv1d", "conv2d", "embedding"], "use_dora": True},
    ),
    (
        "alora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2"], "alora_invocation_tokens": [1]},
    ),
]


class TestLoraVariants:
    @pytest.mark.parametrize("variant_name, config_cls, config_kwargs", TEST_CASES)
    def test_variant_is_applied_to_layers(self, variant_name, config_cls, config_kwargs):
        # This test assumes that targeting and replacing layers works and that after `get_peft_model` we
        # have a model with LoRA layers. We just make sure that each LoRA layer has its variant set and
        # it is also the correct variant for that layer.
        base_model = CustomModel()
        peft_config = config_cls(**config_kwargs)
        peft_model = get_peft_model(base_model, peft_config)

        layer_type_map = VARIANT_MAP[variant_name]

        for _, module in peft_model.named_modules():
            if not hasattr(module, "lora_variant"):
                continue

            # Note that not every variant supports every layer. If it is not mapped it is deemed unsupported and
            # will not be tested.
            expected_variant_type = layer_type_map.get(type(module), None)
            if not expected_variant_type:
                continue

            assert isinstance(module.lora_variant["default"], expected_variant_type)

    def custom_model_with_loss_backpropagated(self, peft_config):
        """Returns the CustomModel + PEFT model instance with a dummy loss that was backpropagated once."""
        base_model = CustomModel()
        peft_model = get_peft_model(base_model, peft_config)

        x, y = torch.ones(10, 10).long(), torch.ones(10, 1, 10, 10)
        out = peft_model(x, y)
        loss = out.sum()
        loss.backward()

        return base_model, peft_model

    def test_dora_params_have_gradients(self):
        """Ensure that the parameters added by the DoRA variant are participating in the output computation."""
        layer_names = ["linear1", "linear2", "conv1d", "conv2d", "embedding"]
        peft_config = LoraConfig(target_modules=layer_names, use_dora=True)
        _, peft_model = self.custom_model_with_loss_backpropagated(peft_config)

        for layer in layer_names:
            assert getattr(peft_model.base_model.model, layer).lora_magnitude_vector["default"].weight.grad is not None

    def test_unregistered_variant_raises_error(self):
        # 1. Create a config and dummy linear layer
        config = LoraConfig()
        base_layer = nn.Linear(10, 10)
        layer = LoraLinear(base_layer, "default", config, r=8, lora_alpha=8)

        # 2. Monkey-patch the lora_variants property to include a fake variant
        with patch("peft.tuners.lora.layer.Linear.lora_variants", new_callable=PropertyMock) as mock_variants:
            mock_variants.return_value = {("fake_unregistered_variant",): None}

            # 3. Assert that the sanity check catches it and throws the right error
            with pytest.raises(
                ValueError,
                match=".*found in lora_variant.*",
            ):
                layer.resolve_lora_variant(config=config)

    def test_invalid_variant_combination_raises_error(self):
        # 1. Create a config with no variants active
        config = LoraConfig()
        base_layer = nn.Linear(10, 10)
        layer = LoraLinear(base_layer, "default", config, r=8, lora_alpha=8)

        # 2. Monkey-patch lora_variants to include a valid tagged combo that isn't active
        with patch("peft.tuners.lora.layer.Linear.lora_variants", new_callable=PropertyMock) as mock_variants:
            mock_variants.return_value = {
                ("use_dora",): None,  # only use_dora is valid, empty combo not listed
            }
            # 3. Assert invalid combination error is raised
            with pytest.raises(ValueError, match="Invalid or unsupported variant combination"):
                layer.resolve_lora_variant(config=config)

    def test_unsorted_variant_keys_raises_error(self):
        config = LoraConfig()
        base_layer = nn.Linear(10, 10)
        layer = LoraLinear(base_layer, "default", config, r=8, lora_alpha=8)

        with patch("peft.tuners.lora.layer.Linear.lora_variants", new_callable=PropertyMock) as mock_variants:
            mock_variants.return_value = {
                ("use_dora", "use_bdlora"): None,
            }
            with pytest.raises(ValueError, match="must be sorted tuples"):
                layer.resolve_lora_variant(config=config)

    def test_multiple_string_variants_in_init_lora_weights(self):
        """
        Verify that multiple variant names originating from the same configuration field (init_lora_weights) resolve to
        different LoraVariant implementations.
        """

        @dataclasses.dataclass
        class MockConfig:
            init_lora_weights: str = dataclasses.field(
                default="foobar", metadata={"lora_variants": ["mica", "foobar"]}
            )

        class MockMiCAVariant:
            pass

        class MockFoobarVariant:
            pass

        class MockLayer(LoraLayer):
            @property
            def lora_variants(self):
                return {
                    ("mica",): MockMiCAVariant,
                    ("foobar",): MockFoobarVariant,
                }

        layer = MockLayer(base_layer=nn.Linear(10, 10))

        # Resolve and verify the correct variants
        for value, expected_class in [
            ("mica", MockMiCAVariant),
            ("foobar", MockFoobarVariant),
        ]:
            config = MockConfig(init_lora_weights=value)
            resolved_instance = layer.resolve_lora_variant(config=config)

            assert isinstance(resolved_instance, expected_class)


class TestFactoredDoraNorm:
    """Tests for the factored DoRA weight norm, which avoids materializing the dense delta weight."""

    def get_peft_model_with_dora(self):
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
        config = LoraConfig(target_modules=["0", "1"], use_dora=True, r=8, lora_alpha=16)
        return get_peft_model(base_model, config)

    def test_factored_weight_norm_matches_dense(self):
        torch.manual_seed(0)
        out_features, in_features, rank = 24, 32, 4
        scaling = 2.0
        weight = torch.randn(out_features, in_features)
        lora_A = torch.randn(rank, in_features)
        lora_B = torch.randn(out_features, rank)

        dense_norm = torch.linalg.norm(weight + scaling * (lora_B @ lora_A), dim=1)
        factored_norm = factored_weight_norm(
            weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=scaling
        )
        assert torch.allclose(factored_norm, dense_norm, atol=1e-5)

    def test_factored_weight_norm_detaches_lora_weights(self):
        # the dense path detaches the delta weight before computing the norm (DoRA paper, section 4.3); the factored
        # path must do the same
        weight = torch.randn(6, 8)
        lora_A = torch.randn(4, 8, requires_grad=True)
        lora_B = torch.randn(6, 4, requires_grad=True)

        weight_norm = factored_weight_norm(weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=1.0)
        assert not weight_norm.requires_grad

    def test_factored_weight_norm_fp32_accumulation_for_half_precision(self):
        # for bf16/fp16 weights the norm is accumulated in fp32: construct a cancellation-prone case with
        # weight ~= -scaling * (lora_B @ lora_A), where a half-precision accumulation deviates noticeably from
        # the fp32 reference
        torch.manual_seed(0)
        out_features, in_features, rank = 24, 512, 4
        scaling = 2.0
        lora_A = torch.randn(rank, in_features, dtype=torch.bfloat16)
        lora_B = torch.randn(out_features, rank, dtype=torch.bfloat16)
        delta = scaling * (lora_B.float() @ lora_A.float())
        weight = (-delta + 0.01 * torch.randn(out_features, in_features)).to(torch.bfloat16)

        reference = torch.linalg.norm(weight.float() + scaling * (lora_B.float() @ lora_A.float()), dim=1)
        dense_bf16 = torch.linalg.norm(weight + scaling * (lora_B @ lora_A), dim=1)
        factored_norm = factored_weight_norm(
            weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=scaling
        )

        assert factored_norm.dtype == torch.bfloat16
        factored_err = (factored_norm.float() - reference).abs().max()
        bf16_err = (dense_bf16.float() - reference).abs().max()
        assert factored_err < bf16_err

        # autocast must not downcast the norm computation: the result is identical with and without autocast
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            factored_norm_autocast = factored_weight_norm(
                weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=scaling
            )
        assert torch.equal(factored_norm_autocast, factored_norm)

    def test_dora_forward_always_uses_factored_norm(self, monkeypatch):
        # the factored weight norm is used unconditionally, regardless of layer size: the dense weight-norm path
        # must not be hit during the forward pass
        peft_model = self.get_peft_model_with_dora()
        monkeypatch.setattr(
            DoraLinearLayer, "get_weight_norm", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError)
        )
        peft_model(torch.randn(4, 64))

    def test_dora_forward_factored_matches_dense(self, monkeypatch):
        # emulate the dense norm computation (materializing the delta weight) and check that the DoRA output and
        # the magnitude gradients match the factored path
        peft_model = self.get_peft_model_with_dora()
        dora_layers = [module for module in peft_model.modules() if isinstance(module, DoraLinearLayer)]
        assert len(dora_layers) > 0

        x = torch.randn(4, 64)
        factored_out = peft_model(x)
        factored_out.sum().backward()
        factored_grads = [dora_layer.weight.grad.clone() for dora_layer in dora_layers]
        peft_model.zero_grad()

        def dense_weight_norm(*, weight, lora_A_weight, lora_B_weight, scaling):
            lora_weight = (lora_B_weight @ lora_A_weight).detach()
            return torch.linalg.norm(weight + scaling * lora_weight, dim=1).to(weight.dtype)

        monkeypatch.setattr("peft.tuners.lora.dora.factored_weight_norm", dense_weight_norm)
        dense_out = peft_model(x)
        dense_out.sum().backward()
        dense_grads = [dora_layer.weight.grad.clone() for dora_layer in dora_layers]

        assert torch.allclose(factored_out, dense_out, atol=1e-5)
        for factored_grad, dense_grad in zip(factored_grads, dense_grads):
            assert torch.allclose(factored_grad, dense_grad, atol=1e-5)


class TestActivatedLora:
    @pytest.mark.parametrize(
        "input_ids, alora_invocation_tokens, expected_offsets",
        [
            ([[0, 1, 2, 3], [0, 4, 5, 6]], [1, 2], [3, None]),
            ([[1, 2, 1, 2], [0, 4, 1, 2]], [1, 2], [2, 2]),
            ([[1, 2, 3, 4], [0, 4, 1, 4]], [1, 2], [4, None]),
            ([[1, 2, 3, 4]], None, [None]),
        ],
    )
    # Verify alora_offsets are calculated correctly
    def test_calculate_alora_offsets(self, input_ids, alora_invocation_tokens, expected_offsets):
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, alora_invocation_tokens=alora_invocation_tokens)
        peft_config = {"default": config}

        # compute offsets
        offsets = calculate_alora_offsets(peft_config, "default", torch.tensor(input_ids))

        assert offsets == expected_offsets

    @pytest.mark.parametrize(
        "input_ids, alora_invocations, expected_offsets",
        [
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": [2]}, [1, 1]),
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": None}, [1, None]),
        ],
    )
    # Verify alora_offsets are correct with adapter names
    def test_calculate_alora_offsets_with_adapter_names(self, input_ids, alora_invocations, expected_offsets):
        peft_config = {}
        for alora_name in alora_invocations.keys():
            peft_config[alora_name] = LoraConfig(alora_invocation_tokens=alora_invocations[alora_name])

        adapter_names = list(alora_invocations.keys())
        offsets = calculate_alora_offsets(
            peft_config, adapter_names[0], torch.tensor(input_ids), adapter_names=adapter_names
        )

        assert offsets == expected_offsets

    # Verify that the adapter does not modify outputs prior to invocation point
    def test_alora_activation_matches_base_until_invocation(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        start = 2
        with lora_model.disable_adapter():
            with torch.no_grad():
                base_out = lora_model(X=input_ids)

        kwargs = get_alora_offsets_for_forward(lora_model, input_ids)
        with torch.no_grad():
            lora_out = lora_model(X=input_ids, **kwargs)
        assert torch.allclose(lora_out[:, :start], base_out[:, :start])
        assert not torch.allclose(lora_out[:, start:], base_out[:, start:])

    # Verify that warning is given for alora when providing embeddings only
    def test_input_embeds_warning(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["linear"],
            alora_invocation_tokens=[2],
            init_lora_weights=False,
        )
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        input_embeds = base_model.embed(input_ids)
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets when only inputs_embeds are provided. Disabling aLoRA for this forward pass.",
        ):
            kwargs = get_alora_offsets_for_forward(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets during generate as input_ids are not available. Disabling aLoRA.",
        ):
            kwargs = get_alora_offsets_for_generate(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None

    # Verify that error is raised when requesting num_beams > 1 for alora
    def test_num_beams_error(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        with pytest.raises(ValueError) as e:
            with torch.no_grad():
                lora_out = lora_model(X=input_ids, num_beams=2, alora_offsets=[3])
        assert "Beam search not yet supported for aLoRA." in str(e.value)

    def test_gradient_checkpointing_double_forward_raises(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear", alora_invocation_tokens=[0])
            lora_model = get_peft_model(base_model, cfg)
            lora_model.train()

            lora_model.prepare_model_for_gradient_checkpointing(lora_model)
            lora_model.gradient_checkpointing_enable()

            inputs = {"input_ids": torch.tensor([[0, 1, 2, 3]])}

            lora_model.forward(**inputs)

            with pytest.raises(ValueError, match="Multiple invocations of PEFT forward hooks.*"):
                lora_model.forward(**inputs)

    def test_gradient_checkpointing_dpo_doesnt_raise(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear", alora_invocation_tokens=[0])
            lora_model = get_peft_model(base_model, cfg)
            lora_model.train()

            lora_model.prepare_model_for_gradient_checkpointing(lora_model)
            lora_model.gradient_checkpointing_enable()

            inputs = {"input_ids": torch.tensor([[0, 1, 2, 3]])}

            with lora_model.disable_adapter():
                lora_model.forward(**inputs)

            lora_model.forward(**inputs)
