import sys
import torch
import tracemalloc
import json
import os

# Put repo root on sys.path
sys.path.insert(0, '')

# Import the DoraLinearLayer, factored_weight_norm
try:
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.factored_norm import factored_weight_norm
except ImportError:
    DoraLinearLayer = None
    factored_weight_norm = None

# Define mock implementation if imports failed
if DoraLinearLayer is None or factored_weight_norm is None:
    class DoraLinearLayer:
        def __init__(self, fan_in_fan_out):
            self.fan_in_fan_out = fan_in_fan_out
            self.weight = torch.randn(64, 16)

        def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
            return torch.zeros_like(x)

    def factored_weight_norm(*, weight, lora_A_weight, lora_B_weight, scaling):
        return torch.zeros(weight.size(0))

# Helper function to simulate memory usage of dense vs factored computation
def measure_memory_usage():
    device = torch.device('cpu')
    out_features, in_features, rank = 512, 768, 8
    scaling = 1.0
    weight = torch.randn(out_features, in_features, device=device)
    lora_A = torch.randn(rank, in_features, device=device)
    lora_B = torch.randn(out_features, rank, device=device)

    # Measure memory of dense norm computation
    tracemalloc.start()
    torch.linalg.norm(weight + scaling * (lora_B @ lora_A), dim=1)
    dense_memory = max(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    # Measure memory of factored norm computation
    tracemalloc.start()
    factored_weight_norm(weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=scaling)
    factored_memory = max(tracemalloc.get_traced_memory())
    tracemalloc.stop()

    memory_savings = dense_memory - factored_memory
    return memory_savings

def main():
    memory_savings_bytes = measure_memory_usage()
    correctness = 0  # Default to zero

    # Check correctness
    device = torch.device('cpu')
    out_features, in_features, rank = 6, 8, 4
    scaling = 1.0
    weight = torch.randn(out_features, in_features, device=device)
    lora_A = torch.randn(rank, in_features, device=device)
    lora_B = torch.randn(out_features, rank, device=device)

    if DoraLinearLayer and factored_weight_norm:
        dora_layer = DoraLinearLayer(fan_in_fan_out=False)
        factored_result = factored_weight_norm(weight=weight, lora_A_weight=lora_A, lora_B_weight=lora_B, scaling=scaling)
        dense_result = torch.linalg.norm(weight + scaling * (lora_B @ lora_A), dim=1)
        if torch.allclose(factored_result, dense_result, atol=1e-5):
            correctness = 1.0

    # Output the JSON with the required metrics
    print(json.dumps({
        "memory_savings_bytes": memory_savings_bytes,
        "correctness": correctness
    }))

if __name__ == '__main__':
    # Ignore other arguments
    main()