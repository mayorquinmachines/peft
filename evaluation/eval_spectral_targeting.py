import os
import sys
import json
import torch
from torch import nn
from peft import LoraConfig, get_peft_model

try:
    from peft.tuners.lora.spectral_targeting import select_top_condition_number_modules
    # Assume that the presence of select_top_condition_number_modules allows us to modify the config
    HAS_SPECTRAL_TARGETING = True
except ImportError:
    HAS_SPECTRAL_TARGETING = False

# Define a small model to work with
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin2(self.lin1(x))

# Measure trainable parameters for standard and spectral LoRA configurations
def evaluate_lora_trainable_params():
    model = SimpleModel()
    lora_config = LoraConfig(
        target_modules=["lin1", "lin2"],
        r=4,
        lora_alpha=8,
        condition_number_top_fraction=None  # default: adapt all modules
    )
    
    if HAS_SPECTRAL_TARGETING and "LORA_CONDITION_NUMBER_TOP_FRACTION" in os.environ:
        fraction = float(os.environ["LORA_CONDITION_NUMBER_TOP_FRACTION"])
        lora_config.condition_number_top_fraction = fraction

    model = get_peft_model(model, lora_config)
    
    # Measure number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return trainable_params

# Measure fit quality (MSE) with potential spectral LoRA configurations
def evaluate_lora_fit():
    torch.manual_seed(42)
    
    # Create a simple dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 10)

    model = SimpleModel()
    lora_config = LoraConfig(
        target_modules=["lin1", "lin2"],
        r=4,
        lora_alpha=8
    )

    if HAS_SPECTRAL_TARGETING and "LORA_CONDITION_NUMBER_TOP_FRACTION" in os.environ:
        fraction = float(os.environ["LORA_CONDITION_NUMBER_TOP_FRACTION"])
        lora_config.condition_number_top_fraction = fraction

    model = get_peft_model(model, lora_config)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a few training steps
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Compute the final loss (MSE) as a metric of fit
    with torch.no_grad():
        final_output = model(x)
        fit_mse = criterion(final_output, y).item()

    return fit_mse

def main():
    trainable_params = evaluate_lora_trainable_params()
    fit_mse = evaluate_lora_fit()
    
    # Output the results
    print(json.dumps({
        "spectral_lora_trainable_params": trainable_params,
        "spectral_lora_fit_mse": fit_mse
    }))

if __name__ == "__main__":
    main()