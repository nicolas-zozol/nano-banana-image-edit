"""Helpers for constructing Gemini Nano Banana edit configurations."""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional


def build_gemini_edit_config(
    *,
    reference_images: List[str],
    target_image: str,
    output_base_name: str = "edit-result",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    system_prompt: Optional[str] = None,
    prompt: Optional[str] = None,
    output_ext: str = "png",
) -> Dict[str, Any]:
    """Return a dict describing a Gemini Nano Banana image-edit request.

    The returned mapping includes:

    - ``files``: the reference assets and the target image (target comes last).
    - ``outputFile``: a timestamped filename for saving Nano Banana's reply.
    - ``sampling``: temperature and top-p values tuned for edit reliability.
    - ``system`` / ``prompt``: the message pair to send alongside the images.
    - ``payloadOrderHint``: handy guidance to keep image order stable.
    """

    if not reference_images:
        raise ValueError("Provide at least one reference image for the edit.")
    if len(reference_images) > 2:
        raise ValueError("Nano Banana works best with <=2 reference images (<=3 total inputs).")
    if not target_image:
        raise ValueError("Specify the target image to edit.")

    # Randomise sampling parameters (closed intervals) when not supplied.
    if temperature is None:
        temperature = round(random.uniform(0.20, 0.35), 4)
    if top_p is None:
        top_p = round(random.uniform(0.70, 0.85), 4)

    timestamp = int(time.time())
    safe_base = os.path.splitext(os.path.basename(output_base_name))[0] or "edit-result"
    ext = output_ext.lstrip(".") or "png"
    output_file = f"{safe_base}_{timestamp}.{ext}"

    default_system = (
        "Perform a surgical wardrobe swap. Preserve the target woman's identity, pose, "
        "framing, hairstyle, skin tone, accessories, and background. Keep lighting "
        "direction and color grade. No cropping, no recomposition, no text or logos. "
        "Only modify clothing as requested."
    )
    default_prompt = (
        "Put the dress from reference images onto the woman in the target image. Match "
        "the dress cut, neckline, sleeve length, hem, color, and fabric texture. Conform "
        "the cloth naturally to her pose with realistic drape and contact shadows. Adjust "
        "shading to match the target's light; do not alter face, hair, body shape, "
        "jewelry, or background."
    )

    system_text = system_prompt or default_system
    user_prompt = prompt or default_prompt

    config: Dict[str, Any] = {
        "files": {
            "referenceImages": reference_images,
            "targetImage": target_image,
        },
        "outputFile": output_file,
        "sampling": {
            "temperature": temperature,
            "topP": top_p,
        },
        "system": system_text,
        "prompt": user_prompt,
        "payloadOrderHint": {
            "images": [*reference_images, target_image],
        },
    }

    return config

