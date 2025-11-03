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

import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from my_project.edit_configuration import (
    EditConfigurationBundle,
    prepare_edit_configuration,
)

# ---------------------------------------------------------------------------
# Configuration section – tweak these values before each run.

# Human-friendly instructions describing the edit you want Gemini to perform.
# Provide the filename of a markdown prompt located under ``data/prompts``.
PROMPT_FILE_NAME: str = "swap-dress.md"

# Reference imagery (dress details etc.), relative to REFERENCE_IMAGE_DIR.
REFERENCE_IMAGE_NAMES: List[str] = [
    "robe-orange.jpeg"
]

# Target image (the photo to edit), relative to TARGET_IMAGE_DIR.
#  "asian-girl-supermarket.jpg"
TARGET_IMAGE_NAME: str ="street-graph.png"

# Output filename base (timestamp appended automatically).
OUTPUT_BASE_NAME: str = "street-graph"

# System instruction passed to Gemini.
SYSTEM_PROMPT: str = (
    "Perform a surgical wardrobe swap. Preserve the target woman's identity, pose, framing, "
    "hairstyle, skin tone, accessories, and background. Keep lighting direction and color grade. "
    "No cropping, no recomposition, no text or logos. Only modify clothing as requested."
)

# Gemini model to invoke. Adjust this if Google changes the model identifier.
MODEL_NAME: str = "models/gemini-2.5-flash-image"


# ---------------------------------------------------------------------------
# Derived paths.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PROMPT_DIR: Path = PROJECT_ROOT / "data" / "prompts"
REFERENCE_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "raw"
TARGET_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "model"
PROCESSED_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "processed"


def load_api_key() -> str:
    """Fetch the Gemini API key from the environment (via .env)."""

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Set it in your .env file before running."
        )
    return api_key


def build_user_content(
    *,
    system_text: str,
    prompt_text: str,
    reference_paths: Iterable[Path],
    target_path: Path,
) -> genai_types.Content:
    """Assemble a single ``user`` content block with system and prompt text."""

    stripped_prompt = prompt_text.strip()
    if not stripped_prompt:
        raise ValueError("The prompt could not be empty. Check the markdown file content.")

    parts: List[genai_types.Part] = []

    stripped_system = system_text.strip()
    if stripped_system:
        parts.append(genai_types.Part(text=f"[SYSTEM]\n{stripped_system}"))

    parts.append(genai_types.Part(text=stripped_prompt))

    ordered_paths = [*reference_paths, target_path]
    for path in ordered_paths:
        mime_type, _ = mimetypes.guess_type(path.as_posix())
        if not mime_type:
            raise ValueError(
                f"Could not infer a MIME type for '{path.name}'. Rename it with a known extension."
            )

        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type=mime_type,
                    data=path.read_bytes(),
                )
            )
        )

    return genai_types.Content(role="user", parts=parts)


def request_image_edit(
    *,
    user_content: genai_types.Content,
    temperature: float,
    top_p: float,
) -> genai_types.GenerateContentResponse:
    """Send the edit request to Gemini and return the raw response."""

    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[user_content],
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            candidate_count=1,
            temperature=temperature,
            top_p=top_p,
        ),
    )

    if not response.candidates:
        raise RuntimeError("Gemini response did not include any candidates. Check the request.")

    return response


def save_images(
    response: genai_types.GenerateContentResponse,
    output_dir: Path,
    preferred_filename: str,
) -> List[Path]:
    """Persist inline image content to disk and return the created paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    base_name, ext = os.path.splitext(preferred_filename)
    if not base_name:
        base_name = f"gemini_edit_{timestamp}"
    if not ext:
        ext = ".png"

    for candidate_index, candidate in enumerate(response.candidates):
        if not candidate.content or not candidate.content.parts:
            continue

        for part_index, part in enumerate(candidate.content.parts):
            inline = getattr(part, "inline_data", None)
            if not inline or not getattr(inline, "data", None):
                continue

            mime_type = getattr(inline, "mime_type", None) or "image/png"
            guessed_ext = mimetypes.guess_extension(mime_type) or ext
            if candidate_index == 0 and part_index == 0:
                filename = f"{base_name}{guessed_ext}"
            else:
                filename = f"{base_name}_{timestamp}_{candidate_index:02d}_{part_index:02d}{guessed_ext}"

            target_path = output_dir / filename
            target_path.write_bytes(inline.data)
            saved_paths.append(target_path)

    if not saved_paths:
        raise RuntimeError("Gemini response did not include any inline image data to save.")

    return saved_paths


def run_image_edit() -> List[Path]:
    """Top-level helper that executes the end-to-end edit workflow."""

    print("📝 Prompt source:")
    print(f"  - {(PROMPT_DIR / PROMPT_FILE_NAME).relative_to(PROJECT_ROOT)}")

    bundle: EditConfigurationBundle = prepare_edit_configuration(
        prompt_dir=PROMPT_DIR,
        prompt_file_name=PROMPT_FILE_NAME,
        reference_dir=REFERENCE_IMAGE_DIR,
        reference_names=REFERENCE_IMAGE_NAMES,
        target_dir=TARGET_IMAGE_DIR,
        target_name=TARGET_IMAGE_NAME,
        output_base_name=OUTPUT_BASE_NAME,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.28,
    )

    prompt_text = bundle.prompt_text
    reference_paths = bundle.reference_paths
    target_path = bundle.target_path
    config = bundle.config

    print("📚 Reference images:")
    for path in reference_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")
    print("🎯 Target image:")
    print(f"  - {target_path.relative_to(PROJECT_ROOT)}")

    sampling = config["sampling"]
    print("🎛️ Sampling params:")
    print(f"  - temperature={sampling['temperature']}")
    print(f"  - topP={sampling['topP']}")
    print("💾 Planned output filename:")
    print(f"  - {config['outputFile']}")

    user_content = build_user_content(
        system_text=config["system"],
        prompt_text=prompt_text,
        reference_paths=reference_paths,
        target_path=target_path,
    )
    response = request_image_edit(
        user_content=user_content,
        temperature=sampling["temperature"],
        top_p=sampling["topP"],
    )
    output_paths = save_images(response, PROCESSED_IMAGE_DIR, config["outputFile"])

    print("✅ Gemini returned the following edits:")
    for path in output_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")

    return output_paths


def main() -> None:
    """CLI entry-point used when running this module directly."""

    try:
        run_image_edit()
    except Exception as exc:  # noqa: BLE001 - surface helpful message to newcomers.
        print(f"❌ {exc}")


if __name__ == "__main__":
    main()

