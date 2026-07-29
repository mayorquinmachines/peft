import sys
import json
import torch
from functools import wraps

def defensively_import_dora():
    """Import DoraLinearLayer class safely, fallback if it fails."""
    try:
        from peft.tuners.lora.dora import DoraLinearLayer
        from peft.tuners.lora.factored_norm import factored_weight_norm
        return DoraLinearLayer, factored_weight_norm
    except ImportError:
        return None, None

def get_memory_consumption(weight, lora_A, lora_B, scaling, factored_function=None):
    """Measure memory consumption for norm calculation using CPU."""
    if factored_function is not None:
        func = factored_function
    else:
        def func(weight, lora_A_weight, lora_B_weight, scaling):
            weight = weight.detach()
            lora_A_weight = lora_A_weight.detach()
            lora_B_weight = lora_B_weight.detach()
            dense_delta = lora_B_weight @ lora_A_weight
            result = weight + scaling * dense_delta
            norm_result = torch.linalg.norm(result, dim=1)
            return norm_result

    with torch.no_grad():
        result = func(weight, lora_A, lora_B, scaling)  # Execute the operation
    # Memory used in terms of materialized intermediate result bytes
    memory_used_MB = result.element_size() * result.nelement() / (1024 ** 2)
    return memory_used_MB

def calculate_actual_memory_savings(dense_memory_MB, factored_memory_MB):
    """Calculate memory savings in MB."""
    return dense_memory_MB - factored_memory_MB

def construct_dora_model():
    """Construct a sample DoraLinearLayer model for testing."""
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dora_layer = DoraLinearLayer(fan_in_fan_out=False)

        def forward(self, x, lora_A, lora_B, scaling):
            return self.dora_layer.forward(
                x, lora_A=lora_A, lora_B=lora_B, scaling=scaling,
                base_layer=torch.nn.Linear(32, 32), base_result=None
            )
    return CustomModel()

def run_eval():
    """Run the evaluation script."""
    DoraLinearLayer, factored_weight_norm = defensively_import_dora()
    feature_enabled = bool(DoraLinearLayer and factored_weight_norm)

    torch.manual_seed(0)
    out_features, in_features, rank = 32, 64, 8
    weight = torch.randn(out_features, in_features)
    lora_A = torch.randn(rank, in_features)
    lora_B = torch.randn(out_features, rank)

    dense_memory_MB = get_memory_consumption(weight, lora_A, lora_B, scaling=2.0)
    
    factored_memory_MB = dense_memory_MB  # Default, if the feature isn't available
    if feature_enabled:
        factored_memory_MB = get_memory_consumption(
            weight, lora_A, lora_B, scaling=2.0, factored_function=factored_weight_norm
        )
    
    memory_savings_MB = calculate_actual_memory_savings(dense_memory_MB, factored_memory_MB)
    correct_functionality = int(feature_enabled)

    result = {"memory_savings_MB": memory_savings_MB, "correct_functionality": correct_functionality}
    print(json.dumps(result))  # Ensure JSON output as last line

if __name__ == "__main__":
    sys.argv = sys.argv[:1]  # Ignore any additional args
    run_eval()