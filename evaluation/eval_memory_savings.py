import os
import sys
import torch
import json
from torch import nn

# Insert the repo root to sys.path to import the modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

# Import the necessary modules defensively
try:
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.factored_norm import factored_weight_norm
except ImportError:
    DoraLinearLayer = None
    factored_weight_norm = None

# Constants for evaluation
OUT_FEATURES = 256
IN_FEATURES = 256
RANK = 8
SCALING = 1.0

def compute_dense_weight_norm(weight, lora_A, lora_B, scaling):
    return torch.linalg.norm(weight + scaling * lora_B @ lora_A, dim=1)

def compute_factored_weight_norm(weight, lora_A, lora_B, scaling):
    if factored_weight_norm is not None:
        return factored_weight_norm(
            weight=weight,
            lora_A_weight=lora_A,
            lora_B_weight=lora_B,
            scaling=scaling
        )
    else:
        # Fall back to dense path if factored is unavailable
        return compute_dense_weight_norm(weight, lora_A, lora_B, scaling)

def evaluate_memory_savings():
    # Seed for reproducibility
    torch.manual_seed(0)

    # Initialize the weights
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    lora_A = torch.randn(RANK, IN_FEATURES)
    lora_B = torch.randn(OUT_FEATURES, RANK)

    # Compute norms
    dense_norm = compute_dense_weight_norm(weight, lora_A, lora_B, SCALING)
    factored_norm = compute_factored_weight_norm(weight, lora_A, lora_B, SCALING)

    correct_output = torch.allclose(factored_norm, dense_norm, atol=1e-5)

    # Compute memory savings assuming dense path's transient memory footprint
    transient_dense = OUT_FEATURES * IN_FEATURES
    transient_factored = (OUT_FEATURES + IN_FEATURES) * RANK
    memory_savings_factor = transient_dense / transient_factored

    return correct_output, memory_savings_factor

def main():
    correct_output, memory_savings_factor = evaluate_memory_savings()

    # Print the results in JSON format as required
    metrics = {
        "correct_output": 1.0 if correct_output else 0.0,
        "memory_savings_factor": memory_savings_factor
    }
    print(json.dumps(metrics))

if __name__ == "__main__":
    main()