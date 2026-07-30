"""Validation benchmark for kappa-LoRA spectral targeting (`condition_number_top_fraction`).

Measures both halves of the claim ("halves trainable parameters while matching accuracy"):

- `spectral_lora_trainable_params` (target): exact trainable-parameter count of the
  adapted model. 8 rank-4 adapters on 32-dim layers = 256 params each -> 2048 standard;
  spectral targeting at top_fraction=0.5 keeps 4 adapters -> 1024. Baseline (pre-change
  code, no spectral flag) counts the standard 2048.
- `spectral_lora_fit_mse` (guardrail): the RATIO of spectral-LoRA fit MSE to standard-LoRA
  fit MSE on a small deterministic regression task (seeded, full-batch, fixed steps).
  1.0 means parity. On the baseline arm both configurations are standard LoRA, so the
  ratio is exactly 1.0 by construction — the guardrail is calibrated on both arms.

The base model alternates well-conditioned (kappa ~= 2) and ill-conditioned (kappa ~= 1e4)
layer spectra so the condition-number selection is deterministic.
"""
import json
import os
import sys

import torch
from torch import nn

try:
    from peft.tuners.lora.spectral_targeting import select_top_condition_number_modules  # noqa: F401
    HAS_SPECTRAL_TARGETING = True
except ImportError:
    HAS_SPECTRAL_TARGETING = False

N_LAYERS = 8
DIM = 32
RANK = 4
FIT_STEPS = 60


def build_base_model() -> nn.Sequential:
    """8 bias-free linear layers with alternating conditioning of the weight spectra."""
    torch.manual_seed(0)
    layers = []
    for i in range(N_LAYERS):
        linear = nn.Linear(DIM, DIM, bias=False)
        u, _ = torch.linalg.qr(torch.randn(DIM, DIM))
        v, _ = torch.linalg.qr(torch.randn(DIM, DIM))
        if i % 2 == 0:
            spectrum = torch.linspace(1.0, 2.0, DIM)   # kappa ~= 2
        else:
            spectrum = torch.logspace(0.0, 4.0, DIM)   # kappa ~= 1e4
        with torch.no_grad():
            linear.weight.copy_(u @ torch.diag(spectrum) @ v.T)
        layers.append(linear)
    return nn.Sequential(*layers)


def make_config(spectral: bool):
    """Standard LoRA config; the spectral flag is applied as an attribute only when
    the changed code is importable, so the same script runs on the baseline arm."""
    from peft import LoraConfig
    config = LoraConfig(target_modules=[str(i) for i in range(N_LAYERS)],
                        r=RANK, lora_alpha=8)
    if spectral:
        config.condition_number_top_fraction = float(
            os.environ.get("LORA_CONDITION_NUMBER_TOP_FRACTION", "0.5"))
    return config


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_mse(spectral: bool) -> float:
    """Deterministic toy regression: fixed seed, full-batch, fixed step count."""
    from peft import get_peft_model
    torch.manual_seed(42)
    x = torch.randn(64, DIM)
    target_map = torch.randn(DIM, DIM) * 0.1
    y = x @ target_map
    model = get_peft_model(build_base_model(), make_config(spectral))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(FIT_STEPS):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return criterion(model(x), y).item()


def main() -> None:
    from peft import get_peft_model

    # Spectral targeting is exercised only where the changed code exists AND the
    # feature arm's declared env flag is set — the baseline arm runs standard LoRA.
    spectral = HAS_SPECTRAL_TARGETING and "LORA_CONDITION_NUMBER_TOP_FRACTION" in os.environ

    counted_model = get_peft_model(build_base_model(), make_config(spectral))
    spectral_lora_trainable_params = trainable_parameter_count(counted_model)

    standard_mse = fit_mse(False)
    candidate_mse = fit_mse(spectral)
    spectral_lora_fit_mse = round(candidate_mse / standard_mse, 4) if standard_mse > 0 else 1.0

    print(json.dumps({
        "spectral_lora_trainable_params": spectral_lora_trainable_params,
        "spectral_lora_fit_mse": spectral_lora_fit_mse,
    }))


if __name__ == "__main__":
    sys.argv = sys.argv[:1]  # accept and ignore --variant/--ref/--seed
    main()
