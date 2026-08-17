<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Super-Tuning (Super)

Super-Tuning ([Super](https://arxiv.org/abs/2607.09287)) is a sparse PEFT method that reuses saliency signals originally developed for pruning to choose *where* a model should adapt. Instead of a low-rank update, Super trains a sparse delta restricted to a small fixed support of each targeted weight matrix, while the base weights stay frozen.

The abstract from the paper is:

*Large language models (LLMs) remain expensive to fine-tune because full-parameter updates require substantial memory, compute, and per-task storage. We study whether saliency signals originally developed for pruning can be reused to choose where a model should adapt. We propose Super, a sparse parameter-efficient fine-tuning (PEFT) method that fixes a small trainable support using a Wanda-style activation-weighted magnitude score computed from a calibration pass. We then introduce Supra, a hybrid adapter that combines this sparse update with LoRA while preserving a matched trainable-parameter budget through a simple budget-splitting rule. In single-seed Math17K arithmetic experiments on Llama-3.2-1B and Meta-Llama-3-8B, the best Super/Supra variants achieve the highest average accuracy among the tested schedule-selected adapter configurations. We also include a PaFi-style magnitude-only support as a closest training-free sparse baseline and find that low-score supports under both magnitude and Wanda-style orderings can be effective. These results suggest that simple pruning-inspired orderings can provide useful fixed sparse supports for PEFT, especially when combined with low-rank adapters.*

## How Super-Tuning works

For each targeted linear layer with weight `W`, Super computes a saliency score per weight entry and keeps only the top-`density` fraction of entries trainable:

- `saliency="magnitude"` (default): the score is `|W|` (PaFi-style, training-free — no calibration needed).
- `saliency="wanda"`: the score is the Wanda-style `|W| * ||X_j||_2`, where `||X_j||_2` is the L2 norm of the activations of input feature `j`, collected from a calibration pass via `SuperTuningModel.calibrate(...)`.

The trainable delta is initialized to zero and gradients are automatically masked to the fixed support, so only the selected entries are ever updated. Setting `select_lowest=True` selects the lowest-scoring entries instead, which the paper found can also be effective.

## Basic usage

```python
import torch
from transformers import AutoModelForCausalLM
from peft import SuperTuningConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")

config = SuperTuningConfig(
    target_modules=["c_attn", "c_proj"],  # target attention layers
    density=0.05,                         # 5% of weight entries are trainable
    saliency="magnitude",                 # training-free support selection
)
model = get_peft_model(model, config)
```

With Wanda-style activation-aware support selection, run one calibration pass before training:

```python
config = SuperTuningConfig(
    target_modules=["c_attn", "c_proj"],
    density=0.05,
    saliency="wanda",
)
model = get_peft_model(model, config)

# Calibration: a small batch of representative inputs
inputs = tokenizer("The quick brown fox", return_tensors="pt")
model.base_model.calibrate(dict(inputs))  # recomputes the supports with |W| * ||X_j||_2
```

## SuperTuningConfig

[[autodoc]] tuners.super_tuning.config.SuperTuningConfig
