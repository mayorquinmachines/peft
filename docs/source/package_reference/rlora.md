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

# R-LoRA: Randomized Multi-Head LoRA for Efficient Multi-Task Learning

[R-LoRA](https://arxiv.org/abs/2502.15455) (Randomized Multi-Head LoRA) is a LoRA variant designed for multi-task
learning, where a model is fine-tuned on data from multiple domains and plain LoRA tends to underperform. R-LoRA
replaces LoRA's single up-projection with a multi-head structure and randomizes the heads so that they diversify:
each head can specialize on task-specific features while the shared down-projection preserves common knowledge.

Concretely, per target layer R-LoRA learns one shared down-projection `A` (shape `r x in_features`) and `num_heads`
head matrices `B_i` (stacked, shape `num_heads x out_features x r`). The update is the router-weighted sum of the
per-head low-rank updates:

```
W' = W + scaling * sum_i omega_i B_i A,   omega = softmax(W_r x)
```

with `scaling = rlora_alpha / r` and a learned router `W_r` (disable it with `use_router=False` to average the
heads uniformly). On top of this structure, R-LoRA applies **Multi-Head Randomization**:

- **Multi-Head Dropout** (`head_dropout`, paper default 0.2): dropout on the low-rank intermediate `A x` with an
  independent mask per head, so each head sees a differently masked input. Applied during training only.
- **Multi-Head Random Initialization** (`init_weights=True`, default): `A` and the heads are drawn non-zero from
  scaled normal distributions (std `out_features**0.25 / (sqrt(init_gamma) * sqrt(fan))`, with `init_gamma=64`
  following LoRA-GA), breaking the symmetry of zero-initialized heads. The initial delta is subtracted from the
  base weight, so the adapted layer is an exact identity at initialization. Note that, like other methods that
  modify the base weight at init (e.g. PiSSA), the compensation is not part of the saved adapter; it is reproduced
  deterministically from `init_seed` when the adapter is re-created during loading, so checkpoints round-trip
  exactly. With `init_weights=False`, R-LoRA falls back to the standard LoRA identity initialization (kaiming `A`,
  zero heads).

The head outputs are input-dependent through the router, so merging (`merge_adapter`, `merge_and_unload`) folds in
the uniform head combination `scaling * mean_i(B_i) @ A` and warns when a router is present.

R-LoRA is currently implemented for `torch.nn.Linear` layers.

If you use R-LoRA in your work, please cite the paper:

```bibtex
@article{liu2025rlora,
  title={R-LoRA: Randomized Multi-Head LoRA for Efficient Multi-Task Learning},
  author={Liu, Jinda and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2502.15455},
  year={2025}
}
```

## RLoraConfig

[[autodoc]] tuners.rlora.config.RLoraConfig

## RLoraModel

[[autodoc]] tuners.rlora.model.RLoraModel
