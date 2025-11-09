"""Utilities for preparing Gemini image-edit configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from my_project.gemini_config import build_gemini_edit_config
from my_project.shared import load_prompt, resolve_reference_and_target_paths


@dataclass(slots=True)
class EditConfigurationBundle:
    """Container describing the assets and settings for an edit run."""

    prompt_text: str
    reference_paths: List[Path]
    target_path: Path
    config: Dict[str, object]


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

