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

"""Tests for the Tier-2 prompt-fidelity scoring (prompt_fidelity.py) and its wiring into run.evaluate."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from prompt_fidelity import (
    RUBRIC_DIMENSIONS,
    build_judge_prompt,
    fidelity_from_scores,
    parse_judge_response,
    score_prompt_fidelity,
)


def _import_run():
    # the benchmark scripts require the full image-gen stack (diffusers, PIL, torchvision), which is not part of the
    # core PEFT test environment; skip the integration tests when it is unavailable
    for module in ("diffusers", "PIL", "torchvision"):
        pytest.importorskip(module)
    import run

    return run


class TestBuildJudgePrompt:
    def test_contains_prompt_and_all_rubric_dimensions(self):
        judge_prompt = build_judge_prompt("a photo of a red cat on a pillow")
        assert "a photo of a red cat on a pillow" in judge_prompt
        for dim in RUBRIC_DIMENSIONS:
            assert dim in judge_prompt


class TestParseJudgeResponse:
    def test_parses_plain_json(self):
        scores = parse_judge_response('{"subject": 2, "attributes": 1, "composition": 0}')
        assert scores == {"subject": 2, "attributes": 1, "composition": 0}

    def test_parses_json_wrapped_in_prose_and_prefers_the_last_object(self):
        text = (
            'Let me think. {"subject": 0, "attributes": 0, "composition": 0} is wrong, '
            'the actual answer is {"subject": 2, "attributes": 2, "composition": 1}.'
        )
        scores = parse_judge_response(text)
        assert scores == {"subject": 2, "attributes": 2, "composition": 1}

    def test_scores_are_clamped_to_valid_range(self):
        scores = parse_judge_response('{"subject": 5, "attributes": -1, "composition": 2}')
        assert scores == {"subject": 2, "attributes": 0, "composition": 2}

    def test_raises_when_no_valid_scores_found(self):
        with pytest.raises(ValueError, match="Could not parse rubric scores"):
            parse_judge_response("I cannot judge this image.")


class TestFidelityFromScores:
    def test_perfect_scores_give_one(self):
        assert fidelity_from_scores(dict.fromkeys(RUBRIC_DIMENSIONS, 2)) == 1.0

    def test_zero_scores_give_zero(self):
        assert fidelity_from_scores(dict.fromkeys(RUBRIC_DIMENSIONS, 0)) == 0.0


class TestScorePromptFidelity:
    def test_aggregates_over_samples_and_dimensions(self):
        images = [object(), object()]
        prompts = ["prompt a", "prompt b"]
        responses = iter(
            [
                json.dumps({"subject": 2, "attributes": 2, "composition": 2}),
                json.dumps({"subject": 0, "attributes": 1, "composition": 2}),
            ]
        )
        result = score_prompt_fidelity(images, prompts, lambda image, prompt: next(responses))
        assert result["num_samples"] == 2
        assert result["fidelity"] == pytest.approx((6 / 6 + 3 / 6) / 2)
        assert result["dimensions"]["subject"] == pytest.approx(1.0)
        assert result["dimensions"]["attributes"] == pytest.approx(1.5)
        assert result["dimensions"]["composition"] == pytest.approx(2.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Need 1 prompt per image"):
            score_prompt_fidelity([object()], [], lambda image, prompt: "")


class TestEvaluateTier2Integration:
    """Integration tests for the judge_fn/tier2_log wiring in run.evaluate (the existing Tier-1 entry point)."""

    def _make_fakes(self, monkeypatch, run, num_samples):
        import torch
        from PIL import Image

        class FakePipeline:
            text_encoder = None
            vae = None
            transformer = SimpleNamespace(device=torch.device("cpu"))

            def __call__(self, *, prompt, **kwargs):
                return SimpleNamespace(images=[Image.new("RGB", (8, 8)) for _ in prompt])

        def fake_get_dino_embeddings(images, processor, dino_model, batch_size):
            embeddings = torch.ones(len(images), 4)
            return embeddings / embeddings.norm(dim=-1, keepdim=True)

        monkeypatch.setattr(run, "offload_models", lambda *args, **kwargs: nullcontext())
        monkeypatch.setattr(run, "get_dino_embeddings", fake_get_dino_embeddings)

        config = SimpleNamespace(
            seed=0,
            batch_size_eval=2,
            num_inference_steps=1,
            guidance_scale=1.0,
            resolution=8,
            max_sequence_length=8,
            text_encoder_out_layers=[0],
        )
        ds_eval = [
            {"prompt": f"a photo of a cat, variation {i}", "raw_image": Image.new("RGB", (8, 8))}
            for i in range(num_samples)
        ]
        return FakePipeline(), ds_eval, config

    def test_judge_fn_populates_tier2_log_and_keeps_tier1(self, monkeypatch):
        run = _import_run()
        pipeline, ds_eval, config = self._make_fakes(monkeypatch, run, num_samples=3)

        def judge_fn(image, prompt):
            return json.dumps({"subject": 2, "attributes": 1, "composition": 2})

        tier2_log = []

        similarity = run.evaluate(
            pipeline=pipeline,
            ds_eval=ds_eval,
            processor=None,
            dino_model=None,
            config=config,
            judge_fn=judge_fn,
            tier2_log=tier2_log,
        )

        # Tier 1 is unchanged: identical (fake) embeddings give a cosine similarity of 1
        assert similarity == pytest.approx(1.0)
        # Tier 2 scored every generated image against its prompt
        assert len(tier2_log) == 1
        assert tier2_log[0]["num_samples"] == 3
        assert tier2_log[0]["fidelity"] == pytest.approx(5 / 6)

    def test_without_judge_fn_tier2_log_stays_empty(self, monkeypatch):
        run = _import_run()
        pipeline, ds_eval, config = self._make_fakes(monkeypatch, run, num_samples=2)
        tier2_log = []

        similarity = run.evaluate(
            pipeline=pipeline,
            ds_eval=ds_eval,
            processor=None,
            dino_model=None,
            config=config,
            tier2_log=tier2_log,
        )

        assert similarity == pytest.approx(1.0)
        assert tier2_log == []
