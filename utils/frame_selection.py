"""Anchor frame selection heuristics.

The anchor frame is used for:
  - MoGe depth estimation (camera intrinsics come from here)
  - SAM-2 seed mask
  - SAM-3D object mesh generation

A good anchor frame maximises:
  1. Hand detection confidence (hand is clearly visible)
  2. Object visibility (object not fully occluded by hand)
  3. Image sharpness (not blurry)
"""

from __future__ import annotations

import numpy as np

from pipeline.data import Frame


def select_anchor_frame(frames: list[Frame]) -> int:
    """Return the index into `frames` of the best anchor frame.

    Scoring combines hand detection confidence and a Laplacian sharpness
    estimate.  Frames with no detected hand are excluded.
    """
    best_idx = 0
    best_score = -1.0

    for frame in frames:
        if frame.hand_bbox is None:
            continue

        sharpness = _laplacian_variance(frame.image)
        score = frame.hand_score * 0.6 + _normalise(sharpness) * 0.4

        if score > best_score:
            best_score = score
            best_idx = frame.index

    return best_idx


def _laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian — a standard blur measure."""
    gray = image.mean(axis=2).astype(np.float32)
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _normalise(value: float, eps: float = 1e-6) -> float:
    """Sigmoid-normalise a positive value to [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-value / (value + eps))))
