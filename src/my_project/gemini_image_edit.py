"""Utility for editing images with the Gemini Nano Banana model.

This module wires together a minimal workflow to:

1. Load your Gemini API key from a ``.env`` file.
2. Select source images from ``data/raw``.
3. Send an edit request to the Gemini Nano Banana model.
4. Persist the returned image assets under ``data/processed``.

Update the configuration block just below to experiment with prompts and
input imagery.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from my_project.edit_configuration import prepare_edit_configuration
from my_project.gemini_config import build_gemini_edit_config
from my_project.shared import build_user_content, generate_temperature_schedule, request_image_edit, save_images

# ---------------------------------------------------------------------------
# Configuration section – tweak these values before each run.

# Human-friendly instructions describing the edit you want Gemini to perform.
# Provide the filename of a markdown prompt located under ``data/prompts``.
PROMPT_FILE_NAME: str = "two-references.md"

# Reference imagery (dress details etc.), relative to REFERENCE_IMAGE_DIR.
REFERENCE_IMAGE_NAMES: List[str] = [
    "bord-de-plage.png",
    "bord-de-plage-zoom-4.jpeg"
]

# Target image (the photo to edit), relative to TARGET_IMAGE_DIR.
#  "asian-girl-supermarket.jpg"
TARGET_IMAGE_NAME: str ="asian-girl-supermarket.jpg"

# Output filename base (timestamp appended automatically).
OUTPUT_BASE_NAME: str = "asian"

# System instruction passed to Gemini.
SYSTEM_PROMPT: str = (
    "Perform a targeted wardrobe overwrite on the target canvas. You may fully replace clothing inside the editable region. Preserve the target woman’s identity, facial geometry, hairstyle, skin tone, hands, accessories, pose, framing, and background. Keep scene lighting and color grade. No logos or text. Do not crop or recompose"
)

# Gemini model to invoke. Adjust this if Google changes the model identifier.
MODEL_NAME: str = "models/gemini-2.5-flash-image"


VARIATIONS: int = 3
BASE_TEMPERATURE: float = 0.23
TEMPERATURE_SPREAD: float = 0.05

# ---------------------------------------------------------------------------
# Derived paths.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROMPT_DIR: Path = PROJECT_ROOT / "data" / "prompts"
REFERENCE_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "raw"
TARGET_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "model"
PROCESSED_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "processed"  # reserved for manual curation
SAMPLE_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "samples"


def run_image_edit() -> List[Path]:
    """Top-level helper that executes the end-to-end edit workflow."""

    print("📝 Prompt source:")
    print(f"  - {(PROMPT_DIR / PROMPT_FILE_NAME).relative_to(PROJECT_ROOT)}")

    bundle = prepare_edit_configuration(
        prompt_dir=PROMPT_DIR,
        prompt_file_name=PROMPT_FILE_NAME,
        reference_dir=REFERENCE_IMAGE_DIR,
        reference_names=REFERENCE_IMAGE_NAMES,
        target_dir=TARGET_IMAGE_DIR,
        target_name=TARGET_IMAGE_NAME,
        output_base_name=OUTPUT_BASE_NAME,
        system_prompt=SYSTEM_PROMPT,
        temperature=BASE_TEMPERATURE,
    )

    prompt_text = bundle.prompt_text
    reference_paths = bundle.reference_paths
    target_path = bundle.target_path
    base_config = bundle.config

    print("📚 Reference images:")
    for path in reference_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")
    print("🎯 Target image:")
    print(f"  - {target_path.relative_to(PROJECT_ROOT)}")

    base_top_p = base_config["sampling"]["topP"]
    min_temp = max(0.20, BASE_TEMPERATURE - TEMPERATURE_SPREAD)
    max_temp = min(0.35, BASE_TEMPERATURE + TEMPERATURE_SPREAD)
    temperatures = generate_temperature_schedule(
        base=BASE_TEMPERATURE,
        count=VARIATIONS,
        minimum=min_temp,
        maximum=max_temp,
    )
    print("🎛️ Temperature schedule:")
    for idx, temp in enumerate(temperatures, start=1):
        print(f"  - variation {idx}: temperature={temp}")

    saved_paths: List[Path] = []

    for index, temperature in enumerate(temperatures, start=1):
        variant_base = f"{OUTPUT_BASE_NAME}_v{index}"
        print(f"🚀 Running variation {index}/{len(temperatures)} at temperature {temperature}")

        config = build_gemini_edit_config(
            reference_images=[str(path) for path in reference_paths],
            target_image=str(target_path),
            output_base_name=variant_base,
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt_text,
            temperature=temperature,
            top_p=base_top_p,
        )

        sampling = config["sampling"]
        print("💾 Planned output filename:")
        print(f"  - {config['outputFile']}")

        user_content = build_user_content(
            system_text=config["system"],
            prompt_text=prompt_text,
            reference_paths=reference_paths,
            target_path=target_path,
        )
        response = request_image_edit(
            model_name=MODEL_NAME,
            user_content=user_content,
            temperature=sampling["temperature"],
            top_p=sampling["topP"],
        )
        output_paths = save_images(response, SAMPLE_IMAGE_DIR, config["outputFile"])

        print("✅ Gemini returned the following edits:")
        for path in output_paths:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")
        saved_paths.extend(output_paths)

    return saved_paths


def main() -> None:
    """CLI entry-point used when running this module directly."""

    try:
        run_image_edit()
    except Exception as exc:  # noqa: BLE001 - surface helpful message to newcomers.
        print(f"❌ {exc}")


if __name__ == "__main__":
    main()

