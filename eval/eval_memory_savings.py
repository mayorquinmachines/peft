import os
import sys
import torch
import json
from torch import nn
from peft import LoraConfig, get_peft_model

# Constants
OUT_FEATURES = 128
IN_FEATURES = 128
RANK = 8
SCALING = 1.0

# Attempt to import the changed modules
try:
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.factored_norm import factored_weight_norm
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False

def evaluate_dora_memory_savings():
    try:
        if FEATURE_AVAILABLE:
            # Setup model with DoRA variant enabled
            base_model = nn.Sequential(nn.Linear(IN_FEATURES, OUT_FEATURES))
            config = LoraConfig(target_modules=["0"], use_dora=True, r=RANK, lora_alpha=16)
            peft_model = get_peft_model(base_model, config)
            dora_layer: DoraLinearLayer = peft_model.base_model[0].lora_magnitude_vector["default"]

            # Create inputs
            weight = torch.randn(OUT_FEATURES, IN_FEATURES)
            lora_A = nn.Linear(RANK, IN_FEATURES)
            lora_B = nn.Linear(OUT_FEATURES, RANK)

            # Measure memory savings
            dense_delta_bytes = (weight + SCALING * torch.matmul(lora_B.weight, lora_A.weight)).element_size() * OUT_FEATURES * IN_FEATURES
            factored_bytes = weight.element_size() * OUT_FEATURES * IN_FEATURES
            memory_savings_bytes = dense_delta_bytes - factored_bytes
            
            # Guardrail: Ensure function computes correct norms
            original_norm = torch.linalg.norm(weight + SCALING * torch.matmul(lora_B.weight, lora_A.weight), dim=1)
            factored_norm = factored_weight_norm(
                weight=weight, lora_A_weight=lora_A.weight, lora_B_weight=lora_B.weight, scaling=SCALING
            )
            dora_correctness = int(torch.allclose(original_norm, factored_norm, atol=1e-5))
        else:
            raise ImportError("DoRA features not available.")
            
    except Exception:
        memory_savings_bytes = 0
        dora_correctness = 0

    # Output results
    results = {
        "memory_savings_bytes": memory_savings_bytes,
        "dora_correctness": dora_correctness
    }
    print(json.dumps(results))

def main():
    evaluate_dora_memory_savings()

if __name__ == "__main__":
    main()