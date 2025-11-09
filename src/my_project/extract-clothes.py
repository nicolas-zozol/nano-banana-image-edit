from typing import Iterable, List
from pathlib import Path

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
SAMPLE_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "extracted-sample"
PROCESSED_IMAGE_DIR: Path = PROJECT_ROOT / "data" / "extracted"  # reserved for manual curation

