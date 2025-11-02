"""Skeleton integration test for Gemini image editing."""

from __future__ import annotations

from pathlib import Path

import pytest

from my_project.gemini_config import build_gemini_edit_config


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "images"


@pytest.mark.ai
@pytest.mark.slow
def test_prepare_single_edit_configuration() -> None:
    reference_path = (FIXTURE_DIR / "reference.png").resolve()
    target_path = (FIXTURE_DIR / "target.png").resolve()

    config = build_gemini_edit_config(
        reference_images=[str(reference_path)],
        target_image=str(target_path),
        output_base_name="integration-edit",
        prompt="Blend the reference colors into the target outfit while keeping everything else identical.",
    )

    # TODO: Uncomment once ready to hit the real Gemini API.
    # from google import genai
    # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # response = client.models.generate_content(
    #     model="models/gemini-2.0-nano-banana",
    #     contents=[...],
    # )
    # assert response.candidates

    assert Path(config["files"]["reference_images"][0]) == reference_path
    assert Path(config["files"]["target_image"]) == target_path
    assert config["payloadOrderHint"]["images"][-1] == str(target_path)
    assert 0.20 <= config["sampling"]["temperature"] <= 0.35
    assert 0.70 <= config["sampling"]["topP"] <= 0.85

    assert 1 == 2
