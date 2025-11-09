"""Quick utility to extract garments from reference imagery using Gemini."""

from __future__ import annotations

from pathlib import Path
from typing import List

from my_project.gemini_config import build_gemini_edit_config
from my_project.shared import (
    build_user_content,
    generate_temperature_schedule,
    load_prompt,
    request_image_edit,
    save_images,
)

# ---------------------------------------------------------------------------
# Configuration section – tweak these values before each run.

# Human-friendly instructions describing the extraction task.
PROMPT_FILE_NAME: str = "extract-garment.md"
SYSTEM_PROMPT_FILE_NAME: str = "extract-garment-system.md"

# Reference imagery (dress details etc.), relative to REFERENCE_IMAGE_DIR.
REFERENCE_IMAGE_NAMES: List[str] = [
    "model-louvres"
]

# Output filename base (timestamp appended automatically).
OUTPUT_BASE_NAME: str = "asian-extract"

# Gemini model to invoke. Adjust this if Google changes the model identifier.
MODEL_NAME: str = "models/gemini-2.5-flash-image"

VARIATIONS: int = 3
BASE_TEMPERATURE: float = 0.23
TEMPERATURE_SPREAD: float = 0.05
BASE_TOP_P: float = 0.75

# ---------------------------------------------------------------------------
# Derived paths.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROMPT_DIR: Path = PROJECT_ROOT / "data" / "prompts"
SYSTEM_PROMPT_DIR: Path = PROJECT_ROOT / "data" / "prompts-system"
REFERENCE_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "raw"
EXTRACTED_SAMPLE_DIR: Path = PROJECT_ROOT / "data" / "extracted-sample"


def run_extraction() -> List[Path]:
    """Run multiple Gemini variations to isolate the garment."""

    print("📝 Prompt source:")
    print(f"  - {(PROMPT_DIR / PROMPT_FILE_NAME).relative_to(PROJECT_ROOT)}")
    print("🧾 System prompt source:")
    print(f"  - {(SYSTEM_PROMPT_DIR / SYSTEM_PROMPT_FILE_NAME).relative_to(PROJECT_ROOT)}")

    system_prompt_text = load_prompt(SYSTEM_PROMPT_DIR, SYSTEM_PROMPT_FILE_NAME)
    prompt_text = load_prompt(PROMPT_DIR, PROMPT_FILE_NAME)

    if not REFERENCE_IMAGE_NAMES:
        raise ValueError("Configure at least one reference image for extraction.")

    resolved_references: List[Path] = []
    for name in REFERENCE_IMAGE_NAMES:
        candidate = REFERENCE_IMAGE_DIR / name
        if not candidate.exists():
            raise FileNotFoundError(f"Reference image '{name}' was not found in '{REFERENCE_IMAGE_DIR}'.")
        resolved_references.append(candidate)

    target_path = resolved_references[0]
    supporting_paths = resolved_references[1:]
    base_top_p = BASE_TOP_P

    print("📚 Reference images:")
    for path in resolved_references:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")
    print("🎯 Extraction canvas:")
    print(f"  - {target_path.relative_to(PROJECT_ROOT)} (primary reference)")

    temperatures = generate_temperature_schedule(
        base=BASE_TEMPERATURE,
        count=VARIATIONS,
        minimum=max(0.20, BASE_TEMPERATURE - TEMPERATURE_SPREAD),
        maximum=min(0.35, BASE_TEMPERATURE + TEMPERATURE_SPREAD),
    )
    print("🎛️ Temperature schedule:")
    for idx, temp in enumerate(temperatures, start=1):
        print(f"  - variation {idx}: temperature={temp}")

    saved_paths: List[Path] = []

    for index, temperature in enumerate(temperatures, start=1):
        variant_base = f"{OUTPUT_BASE_NAME}_v{index}"
        print(f"🚀 Running extraction variation {index}/{len(temperatures)} at temperature {temperature}")

        reference_images_for_config = [str(path) for path in supporting_paths]
        include_fallback_reference = False
        if not reference_images_for_config:
            reference_images_for_config = [str(target_path)]
            include_fallback_reference = True

        config = build_gemini_edit_config(
            reference_images=reference_images_for_config,
            target_image=str(target_path),
            output_base_name=variant_base,
            system_prompt=system_prompt_text,
            prompt=prompt_text,
            temperature=temperature,
            top_p=base_top_p,
        )

        if include_fallback_reference:
            config["files"]["referenceImages"] = []
            config["payloadOrderHint"]["images"] = [str(target_path)]

        sampling = config["sampling"]
        print("💾 Planned output filename:")
        print(f"  - {config['outputFile']}")

        user_content = build_user_content(
            system_text=config["system"],
            prompt_text=prompt_text,
            reference_paths=supporting_paths,
            target_path=target_path,
        )
        response = request_image_edit(
            model_name=MODEL_NAME,
            user_content=user_content,
            temperature=sampling["temperature"],
            top_p=sampling["topP"],
        )
        output_paths = save_images(response, EXTRACTED_SAMPLE_DIR, config["outputFile"])

        print("✅ Gemini returned the following extractions:")
        for path in output_paths:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")
        saved_paths.extend(output_paths)

    return saved_paths


def main() -> None:
    try:
        run_extraction()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {exc}")


if __name__ == "__main__":
    main()
