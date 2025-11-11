from __future__ import annotations

import mimetypes
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Basic environment helpers


def load_api_key() -> str:
    """Fetch the Gemini API key from the environment (via .env)."""

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Set it in your .env file before running."
        )
    return api_key


# ---------------------------------------------------------------------------
# Prompt & file resolution helpers


def load_prompt(prompt_dir: Path, prompt_file_name: str) -> str:
    if not prompt_file_name:
        raise ValueError("PROMPT_FILE_NAME is empty. Select a markdown file from data/prompts.")

    if not prompt_dir.exists():
        raise FileNotFoundError(
            f"Prompt directory '{prompt_dir}' does not exist. Create it or update PROMPT_DIR."
        )

    prompt_path = prompt_dir / prompt_file_name
    if not prompt_path.exists():
        available = ", ".join(path.name for path in prompt_dir.glob("*.md")) or "<none>"
        raise FileNotFoundError(
            f"Prompt file '{prompt_file_name}' was not found in '{prompt_dir}'. Available prompts: {available}"
        )

    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file '{prompt_file_name}' is empty. Populate it before running.")

    return text


def resolve_reference_and_target_paths(
    reference_dir: Path,
    target_dir: Path,
    reference_names: Iterable[str],
    target_name: str,
) -> Tuple[List[Path], Path]:
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference images directory '{reference_dir}' does not exist.")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target images directory '{target_dir}' does not exist.")

    if not target_name:
        raise ValueError("TARGET_IMAGE_NAME is empty. Set it to the filename of the photo you want to edit.")

    target_path = target_dir / target_name
    if not target_path.exists():
        available = ", ".join(path.name for path in target_dir.iterdir() if path.is_file()) or "<none>"
        raise FileNotFoundError(
            f"Target image '{target_name}' was not found in '{target_dir}'. Available files: {available}"
        )

    references: List[Path] = []
    if reference_names:
        for name in reference_names:
            candidate = reference_dir / name
            if not candidate.exists():
                raise FileNotFoundError(f"Reference image '{name}' was not found in '{reference_dir}'.")
            if candidate.resolve() == target_path.resolve():
                raise ValueError(
                    "A reference image duplicates the target image. Remove it from REFERENCE_IMAGE_NAMES."
                )
            references.append(candidate)
    else:
        references = [
            path
            for path in sorted(reference_dir.iterdir())
            if path.is_file() and path.resolve() != target_path.resolve()
        ]

    if not references:
        raise ValueError(
            "No reference images were resolved. Add at least one file to REFERENCE_IMAGE_NAMES or keep other "
            "files in the reference directory."
        )

    if len(references) > 2:
        raise ValueError("Nano Banana works best with at most two reference images. Reduce REFERENCE_IMAGE_NAMES to <=2.")

    return references, target_path


# ---------------------------------------------------------------------------
# Gemini request construction helpers


def build_user_content(
    *,
    system_text: str,
    prompt_text: str,
    reference_paths: Iterable[Path],
    target_path: Path,
) -> genai_types.Content:
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
    model_name: str,
    user_content: genai_types.Content,
    temperature: float,
    top_p: float,
) -> genai_types.GenerateContentResponse:
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

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


def generate_temperature_schedule(base: float, count: int, minimum: float = 0.20, maximum: float = 0.35) -> List[float]:
    if count <= 1:
        clamped = max(min(base, maximum), minimum)
        return [round(clamped, 4)]

    lower = max(0.20, minimum)
    upper = min(0.35, maximum)

    if lower > upper:
        lower = upper = max(min(base, 0.35), 0.20)

    if count == 2:
        return [round(lower, 4), round(upper, 4)]

    step = (upper - lower) / (count - 1)
    return [round(lower + step * idx, 4) for idx in range(count)]


def extract_text_responses(response: genai_types.GenerateContentResponse) -> List[str]:
    """Collect any textual explanations returned by Gemini."""

    texts: List[str] = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return texts
