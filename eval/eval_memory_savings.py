import os
import torch
import json
import sys
from contextlib import contextmanager
from unittest.mock import patch
from peft.tuners.lora.dora import DoraLinearLayer
try:
    from peft.tuners.lora.factored_norm import factored_weight_norm
    USING_FEATURE = os.environ.get("ENABLE_FACTORED_NORM", "false") == "true"
except ImportError:
    # Fall back in baseline if factored norm is not available
    factored_weight_norm = None
    USING_FEATURE = False

# Set a base seed for reproducibility
torch.manual_seed(0)

@contextmanager
def temp_environment(env_vars):
    original_env = os.environ.copy()
    os.environ.update(env_vars)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)

# Mock `infer_device` as done in the repo tests
sys.modules["transformers.integrations"] = type("mock", (object,), {})()
torch.backends.cudnn.allow_tf32 = False  # Ensure reproducibility by disabling TF32

def measure_parameter_size():
    model = DoraLinearLayer(64, 64)
    param_size = sum(p.numel() for p in model.parameters()) * 4  # Assuming float32 (4 bytes per element)
    return param_size / (1024 ** 2)  # Convert to MB for consistency

def check_consistency():
    model = DoraLinearLayer(64, 64)
    input_data = torch.randn(10, 64)
    initial_output = model(input_data).detach()
    
    # Simulate the backward pass to ensure stability of results
    loss = initial_output.sum()
    loss.backward()
    
    final_output = model(input_data).detach()
    similarity = torch.cosine_similarity(initial_output.flatten(), final_output.flatten(), dim=0)
    return similarity.item()

def main():
    parameter_size_mb = measure_parameter_size()
    consistency = check_consistency()
    
    print(json.dumps({
        "parameter_size_mb": parameter_size_mb,
        "consistency": consistency
    }))

if __name__ == "__main__":
    main()