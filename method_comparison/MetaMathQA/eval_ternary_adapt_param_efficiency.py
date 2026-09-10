# Copyright 2026-present the HuggingFace Inc. team.
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

"""Param-efficiency + fit-retention evidence for `peft.tuners.ternary_adapt` vs standard LoRA.

Deterministic CPU surrogate for the MetaMathQA method-comparison protocol (llama-3.2-3B shapes, rank32 corpus
runs, per method_comparison/README.md): identical seeded attention projections (q: 3072->3072, v: 3072->1024)
and an identical synthetic additive rank-32 fine-tune teacher for BOTH adapters; both are injected through the
real `get_peft_model` entry point and trained with the same 150-step Adam loop. No network, no Hub, no GPU.

Metrics (single JSON line as the LAST stdout line):
  param_ratio_lora_over_ternary  TARGET threshold 8.0 in validation.yaml.
      Derivation: LoRA r=32 -> q 32*(3072+3072)=196,608 + v 32*(3072+1024)=131,072 = 327,680 trainable.
      TernaryAdapt default near-square blocks: q (48,48) -> A(64,64)+B(48,48)=6,400;
      v (32,48) -> A(32,64)+B(32,48)=3,584; total 9,984. Expected ratio 327,680/9,984 ~= 32.8,
      so 8.0 sits ~4x under the derivation and 8x above parity.
  lora_fit_loss_reduction_pct    GUARDRAIL (>= 50.0): the untouched LoRA path must still close most of the MSE
      gap (teacher is exactly a rank-32 additive delta, i.e. a classic LoRA fine-tune that r=32 represents
      exactly; 150 full-batch Adam steps at lr 1e-2). Baseline `main` runs the identical LoRA arm and clears
      this, so it catches regressions (e.g. the dynamic PeftType mutation breaking existing tuners).
  ternary_fit_loss_reduction_pct REPORTED (>= 20.0): the "without losing fit" half of the claim — TernaryAdapt
      retention on the SAME task. Report-only: baseline `main` has no ternary tuner and reads 0.0 there.

Baseline symmetry: on `main` the package `peft.tuners.ternary_adapt` does not exist, so ternary metrics degrade
to 0.0 (ratio 0.0 < 8.0) while the LoRA guardrail is still measured. The feature is default-on (ternarize_base=True,
init_weights=True), so no env gating is needed on the PR head.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from peft import LoraConfig, get_peft_model  # noqa: E402

try:  # defensive: absent on baseline (main), present on the PR head
    from peft.tuners.ternary_adapt import TernaryAdaptConfig

    TERNARY_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    TernaryAdaptConfig = None
    TERNARY_AVAILABLE = False
    print(f"peft.tuners.ternary_adapt unavailable ({type(exc).__name__}); ternary metrics degraded to 0.0", file=sys.stderr)

HID, Q_OUT, V_OUT = 3072, 3072, 1024  # llama-3.2-3B attention projection shapes
RANK, BATCH, STEPS, LR = 32, 16, 150, 1e-2  # rank32 corpus protocol; batch 16; 150 Adam steps
WEIGHT_STD, DELTA_STD, SEED = 0.02, 0.005, 0  # teacher delta std = 25% of base weight std


class Projections(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, Q_OUT, bias=False)
        self.v_proj = nn.Linear(HID, V_OUT, bias=False)

    def forward(self, x):
        return self.q_proj(x), self.v_proj(x)


def make_base() -> Projections:
    torch.manual_seed(SEED)  # identical fp32 base for every arm
    model = Projections()
    with torch.no_grad():
        nn.init.normal_(model.q_proj.weight, std=WEIGHT_STD)
        nn.init.normal_(model.v_proj.weight, std=WEIGHT_STD)
    return model


def make_fixture():
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(BATCH, HID, generator=g)
    scale = DELTA_STD / (RANK**0.5)  # rank-32 product entries have std sqrt(32)
    delta_q = (torch.randn(Q_OUT, RANK, generator=g) @ torch.randn(RANK, HID, generator=g)) * scale
    delta_v = (torch.randn(V_OUT, RANK, generator=g) @ torch.randn(RANK, HID, generator=g)) * scale
    ref = copy.deepcopy(make_base())
    with torch.no_grad():  # teacher outputs = fine-tuned (W + delta) forward, identical for both adapters
        target_q = F.linear(x, ref.q_proj.weight + delta_q)
        target_v = F.linear(x, ref.v_proj.weight + delta_v)
    return x, target_q, target_v


def fit(peft_model, x, target_q, target_v):
    """Train one adapter against the shared teacher; return (trainable params, % MSE-gap closed)."""
    params = [p for p in peft_model.parameters() if p.requires_grad]
    if not params:
        return 0, 0.0
    opt = torch.optim.Adam(params, lr=LR)

    def loss_fn():
        q, v = peft_model(x)
        return F.mse_loss(q, target_q) + F.mse_loss(v, target_v)

    with torch.no_grad():
        init = loss_fn().item()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()
    with torch.no_grad():
        final = loss_fn().item()
    reduction = 100.0 * (1.0 - final / init) if init > 0 else 0.0
    return sum(p.numel() for p in params), reduction


def main():
    torch.set_num_threads(1)  # deterministic CPU reductions
    x, target_q, target_v = make_fixture()

    # Guardrail arm: standard LoRA through the shared get_peft_model path (identical on both refs).
    lora_params, lora_reduction = 0, 0.0
    try:
        lora_params, lora_reduction = fit(
            get_peft_model(make_base(), LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["q_proj", "v_proj"])),
            x,
            target_q,
            target_v,
        )
    except Exception as exc:  # noqa: BLE001  failure -> guardrail reads 0.0, never a crash
        print(f"lora measurement failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Feature arm: TernaryAdapt with its default-on mechanism (in-place base ternarization + identity init).
    ternary_params, ternary_reduction = 0, 0.0
    if TERNARY_AVAILABLE:
        try:
            ternary_params, ternary_reduction = fit(
                get_peft_model(make_base(), TernaryAdaptConfig(target_modules=["q_proj", "v_proj"])),
                x,
                target_q,
                target_v,
            )
        except Exception as exc:  # noqa: BLE001  failure -> ternary metrics read 0.0 (ratio 0.0 < 8.0)
            print(f"ternary_adapt measurement failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    ratio = lora_params / ternary_params if ternary_params > 0 else 0.0
    print(
        json.dumps(
            {
                "param_ratio_lora_over_ternary": round(ratio, 4),
                "lora_fit_loss_reduction_pct": round(lora_reduction, 4),
                "ternary_fit_loss_reduction_pct": round(ternary_reduction, 4),
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)  # accepted and ignored (symmetric baseline/feature arms)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.parse_known_args()
    main()