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

"""Tier-2 semantic prompt-fidelity scoring for the image generation benchmark.

Adapted from the two-tier evaluation methodology of MetroLLM-Bench (https://arxiv.org/abs/2609.10016): the
deterministic Tier-1 metrics (DINOv2 similarity, drift) are complemented by a judged tier that scores semantic
quality along a structured rubric with a language-model judge. Here, the judged dimension is the fidelity of a
generated image to the prompt it was generated from, scored by an instruction-tuned vision-language model.

The module is import-safe without heavy dependencies: the rubric construction, judge-response parsing, and score
aggregation are pure Python. Only `VLMJudge` lazily imports torch/transformers when instantiated.
"""

import json
import re
from collections.abc import Callable
from typing import Any, Optional


# judge model used when the user does not pass an explicit model id; any instruction-tuned VLM that supports
# image+text chat templates via transformers works
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

RUBRIC_DIMENSIONS = ("subject", "attributes", "composition")
MAX_DIM_SCORE = 2

_JUDGE_PROMPT_TEMPLATE = """You are judging an AI-generated image against the prompt it was generated from.

Prompt: "{prompt}"

Score the image on each of the following dimensions:
- subject: 2 = the main subject of the prompt is clearly depicted; 1 = the subject is only partially depicted or \
ambiguous; 0 = the subject is missing or wrong.
- attributes: 2 = the attributes mentioned in the prompt (color, count, material, style, etc.) all match; 1 = some \
match; 0 = most do not match.
- composition: 2 = the spatial arrangement and overall composition follow the prompt; 1 = they partially follow it; \
0 = they do not follow it at all.

Answer with a single JSON object and nothing else, e.g. {{"subject": 2, "attributes": 1, "composition": 2}}."""


def build_judge_prompt(prompt: str) -> str:
    """Build the structured-rubric prompt that the judge model scores a generated image against."""
    return _JUDGE_PROMPT_TEMPLATE.format(prompt=prompt)


def parse_judge_response(text: str) -> dict[str, int]:
    """Extract the rubric scores from the raw judge output.

    Judges frequently wrap the JSON object in prose or markdown, so all JSON-like objects in the output are tried,
    starting from the last one (the final answer, if the judge reasons before answering). Scores are coerced to int
    and clamped to the valid range.
    """
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(dim in parsed for dim in RUBRIC_DIMENSIONS):
            return {dim: min(max(int(parsed[dim]), 0), MAX_DIM_SCORE) for dim in RUBRIC_DIMENSIONS}
    raise ValueError(f"Could not parse rubric scores for dimensions {RUBRIC_DIMENSIONS} from judge output: {text!r}")


def fidelity_from_scores(scores: dict[str, int]) -> float:
    """Aggregate rubric dimension scores into a single fidelity score normalized to [0, 1]."""
    return sum(scores[dim] for dim in RUBRIC_DIMENSIONS) / (MAX_DIM_SCORE * len(RUBRIC_DIMENSIONS))


def score_prompt_fidelity(
    images: list[Any], prompts: list[str], judge_fn: Callable[[Any, str], str]
) -> dict[str, Any]:
    """Score the prompt fidelity of generated images with a judge model.

    Args:
        images: The generated images, aligned with `prompts`.
        prompts: The prompts the images were generated from.
        judge_fn: Callable taking an image and the prompt it was generated from, returning the judge's raw text
            output, which must contain the rubric scores as a JSON object (see `build_judge_prompt`).

    Returns:
        A dict with the mean fidelity score in [0, 1], the number of scored samples, and the per-dimension means.
    """
    if len(images) != len(prompts):
        raise ValueError(f"Need 1 prompt per image, found {len(prompts)} and {len(images)} instead.")

    dim_totals = dict.fromkeys(RUBRIC_DIMENSIONS, 0)
    fidelity_total = 0.0
    for image, prompt in zip(images, prompts):
        scores = parse_judge_response(judge_fn(image, prompt))
        fidelity_total += fidelity_from_scores(scores)
        for dim in RUBRIC_DIMENSIONS:
            dim_totals[dim] += scores[dim]

    num_samples = len(images)
    return {
        "fidelity": fidelity_total / num_samples,
        "num_samples": num_samples,
        "dimensions": {dim: dim_totals[dim] / num_samples for dim in RUBRIC_DIMENSIONS},
    }


class VLMJudge:
    """Prompt-fidelity judge backed by an instruction-tuned vision-language model loaded via transformers."""

    def __init__(self, model_id: str = DEFAULT_JUDGE_MODEL_ID, max_new_tokens: int = 128, **model_kwargs) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from peft.utils import infer_device

        self.max_new_tokens = max_new_tokens
        device = infer_device()
        model_kwargs.setdefault("dtype", torch.bfloat16 if device != "cpu" else torch.float32)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, image, prompt: str) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": build_judge_prompt(prompt)}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            # greedy decoding keeps the judged scores reproducible across evaluation runs
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def get_judge_fn(model_id: Optional[str] = None, **model_kwargs) -> Callable[[Any, str], str]:
    """Build the default judge callable from a VLM model id (uses `DEFAULT_JUDGE_MODEL_ID` when None)."""
    return VLMJudge(model_id=model_id or DEFAULT_JUDGE_MODEL_ID, **model_kwargs)
