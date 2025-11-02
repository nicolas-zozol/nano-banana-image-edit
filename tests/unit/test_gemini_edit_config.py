"""Unit tests for ``buildGeminiEditConfig``."""

from __future__ import annotations

from yourpkg.gemini_edit_config import buildGeminiEditConfig


def test_randomized_sampling_within_expected_bounds() -> None:
    config = buildGeminiEditConfig(
        referenceImages=["ref1.png"],
        targetImage="target.png",
    )

    temperature = config["sampling"]["temperature"]
    top_p = config["sampling"]["topP"]

    assert 0.20 <= temperature <= 0.35
    assert 0.70 <= top_p <= 0.85


def test_sampling_overrides_are_respected() -> None:
    config = buildGeminiEditConfig(
        referenceImages=["ref1.png"],
        targetImage="target.png",
        temperature=0.25,
        topP=0.8,
    )

    assert config["sampling"]["temperature"] == 0.25
    assert config["sampling"]["topP"] == 0.8


def test_payload_order_places_target_last() -> None:
    config = buildGeminiEditConfig(
        referenceImages=["ref1.png", "ref2.png"],
        targetImage="target.png",
    )

    assert config["payloadOrderHint"]["images"] == [
        "ref1.png",
        "ref2.png",
        "target.png",
    ]

