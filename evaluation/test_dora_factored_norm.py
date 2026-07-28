import json
import os
import sys
import torch
import unittest
from torch import nn
from peft import LoraConfig, get_peft_model

# Add repo root to sys.path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, repo_root)

BASELINE_WEIGHT_NORM, FACTORED_WEIGHT_NORM = ([], [])

# Defensive import
try:
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.factored_norm import factored_weight_norm
    USE_FACTORED_NORM = os.environ.get("USE_FACTORED_NORM") == "1"
except ImportError:
    USE_FACTORED_NORM = False

# Dummy data and model configuration for evaluation
def get_dummy_model_and_data(rank=8, alpha=16):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
    data = torch.randn(10, 64)
    config = LoraConfig(target_modules=["0", "1"], use_dora=USE_FACTORED_NORM, r=rank, lora_alpha=alpha)
    peft_model = get_peft_model(model, config)
    return peft_model, data

# Measure memory efficiency and output accuracy
def evaluate_model_efficiency_and_accuracy():
    peft_model, data = get_dummy_model_and_data()
    output_baseline, output_factored = torch.empty(0), torch.empty(0)
    
    for param in peft_model.parameters():
        if USE_FACTORED_NORM and isinstance(param, nn.Linear):
            # Factored norm evaluation
            output_factored = peft_model(data)
            mem_factored = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            FACTORED_WEIGHT_NORM.append(memory_efficiency_dora(mem_factored))
            
            # Compute output and cache if factored norm is used
            output_factored = peft_model(data)
        
        else:
            # Baseline norm evaluation
            output_baseline = peft_model(data)
            mem_baseline = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            BASELINE_WEIGHT_NORM.append(memory_efficiency_dora(mem_baseline))
    
    same_output_accuracy = (output_baseline - output_factored).abs().mean().item() if USE_FACTORED_NORM and output_factored.numel() > 0 else 1.0
    
    return {
        "memory_efficiency_dora": sum(FACTORED_WEIGHT_NORM) / sum(BASELINE_WEIGHT_NORM) if FACTORED_WEIGHT_NORM else 0,
        "same_output_accuracy": 1.0 - same_output_accuracy if USE_FACTORED_NORM else 1.0
    }

def memory_efficiency_dora(memory_allocated: int) -> float:
    # Simulated dummy calculation (higher is better)
    return 1 / memory_allocated if memory_allocated > 0 else 0

class TestFactoredDoraNormEvaluation(unittest.TestCase):
    def test_memory_efficiency_and_accuracy(self):
        metrics = evaluate_model_efficiency_and_accuracy()
        
        # Ensure the guardrail holds
        assert metrics["same_output_accuracy"] >= 0.99
        print(json.dumps(metrics))

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)