import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# the MetaMathQA experiment scripts and the Gradio app use flat sibling imports (see app.py), so make the parent
# method_comparison directory importable the same way for the `processing` module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import processing  # noqa: E402
from sparsity_preservation import (  # noqa: E402
    METRIC_KEY_BASE_WEIGHT_SPARSITY,
    METRIC_KEY_MERGE_DENSIFICATION,
    get_merge_densification,
    get_model_sparsity_metrics,
    merge_preserving_sparsity,
    weight_sparsity,
)

from peft import LoraConfig, get_peft_model  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(8, 8)
        self.lin1 = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin1(self.lin0(x))


def get_tiny_peft_model():
    torch.manual_seed(0)
    model = TinyModel()
    # simulate a pruned base model: zero out half of the entries of the targeted layer
    with torch.no_grad():
        model.lin0.weight[::2] = 0.0
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["lin0"])
    model = get_peft_model(model, config)
    # simulate a trained adapter (LoRA initializes lora_B to zero, which would give a zero delta weight)
    for module in model.modules():
        if hasattr(module, "lora_A"):
            nn.init.ones_(module.lora_A["default"].weight)
            nn.init.ones_(module.lora_B["default"].weight)
    return model


def get_result_row(train_metrics):
    """Build a minimal result row with the structure that processing.preprocess expects (see log_results)."""
    return {
        "run_info": {
            "created_at": "2025-01-01T00:00:00+00:00",
            "total_time": 10.0,
            "experiment_name": "test-experiment",
            "peft_branch": "main",
            "train_config": {"model_id": "tiny-model"},
            "peft_config": {"peft_type": "LORA"},
            "error_msg": "",
        },
        "train_info": {
            "accelerator_memory_reserved_avg": 0,
            "accelerator_memory_max": 0,
            "accelerator_memory_reserved_99th": 0,
            "train_time": 5.0,
            "file_size": 1000,
            "num_trainable_params": 10,
            "num_total_params": 100,
            "status": "success",
            "metrics": [{"step": 1, "train loss": 1.0}, train_metrics],
        },
        "meta_info": {
            "package_info": {
                "peft-version": "0.0.0",
                "transformers-version": "0.0.0",
                "datasets-version": "0.0.0",
                "torch-version": "0.0.0",
                "bitsandbytes-version": "0.0.0",
            },
            "system_info": {},
        },
    }


def test_weight_sparsity():
    weight = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    assert weight_sparsity(weight) == 0.5
    assert weight_sparsity(torch.ones(4)) == 0.0


def test_merge_preserving_sparsity():
    base_weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    update = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    merged = merge_preserving_sparsity(base_weight, update)
    # the zero pattern of the pruned base is preserved, pruned entries are not filled in
    assert torch.equal(merged == 0, base_weight == 0)
    assert torch.equal(merged, torch.tensor([[1.5, 0.0], [0.0, 2.5]]))
    assert weight_sparsity(merged) == weight_sparsity(base_weight)


def test_merge_preserving_sparsity_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_preserving_sparsity(torch.zeros(2, 2), torch.zeros(3, 3))


def test_get_merge_densification_dense_update():
    model = get_tiny_peft_model()
    # a dense (all-ones) delta weight fills in every pruned entry of the targeted layer
    assert get_merge_densification(model) == 1.0


def test_get_merge_densification_dense_base_model():
    model = TinyModel()  # no pruned entries, no adapters
    assert get_merge_densification(model) == 0.0


def test_get_model_sparsity_metrics_keys_and_values():
    model = get_tiny_peft_model()
    metrics = get_model_sparsity_metrics(model)
    assert set(metrics) == {METRIC_KEY_BASE_WEIGHT_SPARSITY, METRIC_KEY_MERGE_DENSIFICATION}
    # half of lin0's weights are pruned, lin1 and the adapter weights are dense
    assert 0.0 < metrics[METRIC_KEY_BASE_WEIGHT_SPARSITY] < 0.5
    assert metrics[METRIC_KEY_MERGE_DENSIFICATION] == 1.0


def test_get_model_sparsity_metrics_dense_model():
    metrics = get_model_sparsity_metrics(TinyModel())
    assert metrics[METRIC_KEY_BASE_WEIGHT_SPARSITY] == 0.0
    # densification computation is skipped for dense base models and trivially reported as 0.0
    assert metrics[METRIC_KEY_MERGE_DENSIFICATION] == 0.0


def test_sparsity_metrics_surface_in_processing():
    # exercise the same producer function that MetaMathQA/run.py calls and check that processing.preprocess
    # surfaces the metrics under the expected column names
    model = get_tiny_peft_model()
    sparsity_metrics = get_model_sparsity_metrics(model)
    train_metrics = {
        "step": 1,
        "test accuracy": 0.5,
        "train loss": 1.0,
        "train samples": 100,
        "train total tokens": 1000,
        "forgetting": 0.1,
        **sparsity_metrics,
    }
    results = processing.preprocess([get_result_row(train_metrics)], task_name="MetaMathQA")
    assert len(results) == 1
    row = results[0]
    assert row["base_weight_sparsity"] == sparsity_metrics[METRIC_KEY_BASE_WEIGHT_SPARSITY]
    assert row["merge_densification*"] == sparsity_metrics[METRIC_KEY_MERGE_DENSIFICATION]
    # the new columns are registered for display and comparison
    assert "base_weight_sparsity" in processing.get_task_columns("MetaMathQA")
    assert "merge_densification*" in processing.get_task_columns("MetaMathQA")
    assert processing.get_metric_preferences("MetaMathQA")["merge_densification*"] == "lower"


def test_sparsity_metrics_default_for_legacy_rows():
    # result rows from before the sparsity metrics were logged must still preprocess cleanly
    train_metrics = {
        "step": 1,
        "test accuracy": 0.5,
        "train loss": 1.0,
        "train samples": 100,
        "train total tokens": 1000,
        "forgetting": 0.1,
    }
    results = processing.preprocess([get_result_row(train_metrics)], task_name="MetaMathQA")
    assert results[0]["base_weight_sparsity"] == 0.0
    assert results[0]["merge_densification*"] == 0.0
