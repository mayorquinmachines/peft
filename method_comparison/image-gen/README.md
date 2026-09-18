# PEFT method comparison on a DreamBooth-style image generation task

## Goal

This benchmark mirrors the structure of [`method_comparison/MetaMathQA`](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) but targets DreamBooth-style fine-tuning for image generation. It is designed to compare PEFT methods along multiple dimensions like:

- objective quality ([`DINOv2`](https://huggingface.co/facebook/dinov2-base) cosine similarity)
- runtime
- memory usage
- checkpoint size

Note that for max memory reserved, this benchmark measures the memory only for the training part, not the evaluation. This is because evaluation requires extra memory (for running the DINO model) which should not be attributed to the corresponding PEFT method.

## Setup choices

- Base model: [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- Dataset (default): [`cat pillow`](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset)

## Running

### Experiment settings

Create an experiment under `experiments/<peft-method>/<experiment-name>/` or use one of the experiments there.

Each experiment directory may contain:

- `adapter_config.json` (optional; if missing, full fine-tuning is used)
- `training_params.json` (optional; overrides `default_training_params.json`)

### Running a single experiment

Run one experiment:

```sh
python run.py -v experiments/lora/flux2-klein-rank16/
```

By default, the adapter will be saved in a temporary file for further inspection if needed. To prevent this, add the `--clean` flag to the call. To upload the model checkpoint and sample images to a Hugging Face Hub Bucket, pass the `--bucket_name your_user/my_bucket_name` argument.

### Evaluating an existing checkpoint

To run the evaluation (DINOv2 similarity and drift on the test set, sample images) on an already trained checkpoint without retraining, pass the directory containing the trained PEFT checkpoint to `evaluate.py`:

```sh
python evaluate.py -v /path/to/checkpoint/
```

The adapter is loaded on top of the same base model and the evaluation runs under the same conditions (seeds, settings) as at the end of a training run. By default, the training parameters are taken from `default_training_params.json`; if the checkpoint was trained with different parameters, place the corresponding `training_params.json` into the checkpoint directory.

The results and sample images of such an evaluation run are always treated as temporary results, i.e. they are stored in `temporary_results/` and `sample-images/temporary_results/`, respectively. Note that evaluating full fine-tuning checkpoints is not supported.

### Running all pending experiments

The Makefile checks which experiments are missing a corresponding results file and runs those experiments. Note that running a whole sweep can easily take many hours.

```sh
make
```

If you set `UPLOAD_BUCKET_IMAGEGEN="your_user/bucket_name"` as an environment variable prior to starting experiments via `make`, all experiments will be called with the `--bucket_name $UPLOAD_BUCKET_IMAGEGEN` parameter and therefore store the checkpoints and sample images in that bucket. _For maintainers_: The default bucket name should be `"peft-internal-testing/image-gen-benchmark"`.

List experiments to run:

```sh
make list
```

## Training configs

### `adapter_config.json`

This must be a valid PEFT configuration. It is easiest to create it programmatically, e.g.:

```python
from peft import LoraConfig

config = LoraConfig(...)
config.save_pretrained(<path-to-experiment>)
```

### `training_params.json`

There is a default file for the non-PEFT parameters: `default_training_params.json`. This contains all the other parameters that are relevant for training, e.g. the base model id, number of steps, batch size, learning rate, etc. If parameters that differ from the defaults are needed for a specific experiment, place a `training_params.json` into the experiment directory and adjust the parameters that need changing. The other parameters are taken from the aforementioned default config.

For an overview of all possible arguments, you can also check the `TrainConfig` `dataclass` in `utils.py`.

## Dependencies

Install additional dependencies from:

```sh
python -m pip install -r requirements.txt
```

Python 3.12+ is required.

## TODO

- Add further experiments (more PEFT methods) and explore better hyper-parameters.
- Test images are already created but they're not uploaded anywhere.

## Optional: Tier-2 prompt-fidelity scoring

In addition to the deterministic Tier-1 metrics (DINOv2 similarity, drift), `evaluate.py` can score the prompt fidelity of the test generations with a judge vision-language model — a semantic-quality tier adapted from the two-tier evaluation methodology of [MetroLLM-Bench](https://arxiv.org/abs/2609.10016):

```sh
python evaluate.py -v /path/to/checkpoint/ --prompt-fidelity-judge Qwen/Qwen2.5-VL-3B-Instruct
```

The judge scores each generated image against the prompt it was generated from on a structured rubric (subject, attributes, composition). The mean score, normalized to [0, 1], is logged as `test prompt_fidelity` alongside the Tier-1 metrics. Any instruction-tuned VLM that supports image+text chat templates via `transformers` can be passed as the judge model id; see `prompt_fidelity.py` for details. Note that the judge model requires additional memory, which — like the DINO model — is not attributed to the PEFT method's training memory footprint.
