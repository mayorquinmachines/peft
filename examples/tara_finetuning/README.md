# Fine-tuning with training-aware LoRA initialization (TaRA)

This example demonstrates how to initialize LoRA adapters with training-aware initialization (TaRA, ["Training-Aware Low-Rank Adaptation Initialization"](https://arxiv.org/abs/2609.02639)) before fine-tuning a causal language model.

Standard LoRA initializes `lora_A` randomly and `lora_B` to zero, so the initial adapter carries no information about the task. TaRA instead runs one cheap calibration pass over a few hundred samples, collecting the activation second moments `Σ_X = E[x xᵀ]` and the output-gradient second moments `Σ_G = E[δ δᵀ]` of every targeted linear layer. From the truncated SVD `Ũ S̃ Ṽᵀ` of the covariance-weighted base weight `Σ_G W₀ Σ_X`, it sets

- `B ← Σ_G⁻¹ Ũ[:, :r] S̃[:r]^{1/2}`
- `A ← S̃[:r]^{1/2} Ṽ[:, :r]ᵀ Σ_X⁻¹`

and replaces the frozen base weight with the residual `W₀ - BA`. The model output is therefore unchanged at initialization, while the trainable `BA` already holds the components of `W₀` that matter most for training, so the initial low-rank gradients closely approximate the full-rank gradients.

## Usage

```python
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.tara import initialize_tara_weights

lora_config = LoraConfig(r=8, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
peft_model = get_peft_model(base_model, lora_config)  # default init: lora_B == 0


def calibrate_step():
    for batch in calibration_dataloader:
        loss = peft_model(**batch).loss
        loss.backward()


initialize_tara_weights(peft_model, calibrate_step)
# ... continue with regular fine-tuning
```

Run the full example with:

```bash
python tara_finetuning.py --base_model gpt2 --r 8 --lora_alpha 8
```

## Notes

- `calibrate_step` must perform backward passes through the PEFT model; statistics are accumulated over all passes issued inside the callback. The paper uses ~256 calibration samples.
- TaRA expects a freshly initialized adapter (`lora_B == 0`, the default `init_lora_weights=True`) so that the calibration statistics reflect the base model.
- Because the base weight is replaced by the frozen residual `W₀ - BA`, an adapter-only checkpoint does not describe the fine-tuned model. Save the merged model (`merge_and_unload()`), as shown in the example script.
- Quantized base layers and DoRA are not supported.
