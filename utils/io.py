"""File I/O utilities."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Load an image file as H x W x 3 uint8 RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def crop_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 32,
    fill_value: int = 128,
) -> np.ndarray:
    """Crop the tightest bounding box around mask, padded, background filled.

    Background pixels outside the mask are filled with fill_value (neutral grey
    by default — consistent with SAM-3D's expected input conditioning).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask — cannot crop.")

    H, W = image.shape[:2]
    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(W, xs.max() + padding + 1)
    y2 = min(H, ys.max() + padding + 1)

    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[~crop_mask] = fill_value
    return crop
