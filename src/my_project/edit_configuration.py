"""Utilities for preparing Gemini image-edit configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from my_project.gemini_config import build_gemini_edit_config


@dataclass(slots=True)
class EditConfigurationBundle:
    """Container describing the assets and settings for an edit run."""

    prompt_text: str
    reference_paths: List[Path]
    target_path: Path
    config: Dict[str, object]


def load_prompt(prompt_dir: Path, prompt_file_name: str) -> str:
    """Read the selected prompt markdown file."""

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
    """Locate reference images and the target image on disk."""

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


def prepare_edit_configuration(
    *,
    prompt_dir: Path,
    prompt_file_name: str,
    reference_dir: Path,
    reference_names: Iterable[str],
    target_dir: Path,
    target_name: str,
    output_base_name: str,
    system_prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
) -> EditConfigurationBundle:
    """Load assets and build the Gemini configuration for an edit run."""

    prompt_text = load_prompt(prompt_dir, prompt_file_name)
    reference_paths, target_path = resolve_reference_and_target_paths(
        reference_dir=reference_dir,
        target_dir=target_dir,
        reference_names=reference_names,
        target_name=target_name,
    )

    config = build_gemini_edit_config(
        reference_images=[str(path) for path in reference_paths],
        target_image=str(target_path),
        output_base_name=output_base_name,
        system_prompt=system_prompt,
        prompt=prompt_text,
        temperature=temperature,
        top_p=top_p,
    )

    return EditConfigurationBundle(
        prompt_text=prompt_text,
        reference_paths=reference_paths,
        target_path=target_path,
        config=config,
    )

