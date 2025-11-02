"""Unit tests for ``buildGeminiEditConfig``."""

from __future__ import annotations

from my_project.gemini_config import build_gemini_edit_config


def test_randomized_sampling_within_expected_bounds() -> None:
    config = build_gemini_edit_config(
        reference_images=["ref1.png"],
        target_image="target.png",
    )

    temperature = config["sampling"]["temperature"]
    top_p = config["sampling"]["topP"]

    assert 0.20 <= temperature <= 0.35
    assert 0.70 <= top_p <= 0.85


def test_sampling_overrides_are_respected() -> None:
    config = build_gemini_edit_config(
        reference_images=["ref1.png"],
        target_image="target.png",
        temperature=0.25,
        top_p=0.8,
    )

    assert config["sampling"]["temperature"] == 0.25
    assert config["sampling"]["topP"] == 0.8


def test_payload_order_places_target_last() -> None:
    config = build_gemini_edit_config(
        reference_images=["ref1.png", "ref2.png"],
        target_image="target.png",
    )

    assert config["payloadOrderHint"]["images"] == [
        "ref1.png",
        "ref2.png",
        "target.png",
    ]

