"""Stage 1 — Frame loading and hand detection.

Loads a sorted sequence of image frames, runs a lightweight hand detector
(WiLoR's built-in YOLO-based detector) to localise hand bounding boxes,
and selects the anchor frame used by downstream stages.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from pipeline.data import Frame, PipelineData
from models.wilor import WiLoRDetector
from utils.frame_selection import select_anchor_frame
from utils.io import load_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PreprocessingStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.detector = WiLoRDetector(
            confidence_threshold=cfg.get("hand_detection_confidence", 0.3),
            device=cfg.get("device", "cuda"),
        )

    def run(self, frames_dir: Path, data: PipelineData) -> PipelineData:
        paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No image frames found in {frames_dir}")

        frames: list[Frame] = []
        for idx, path in enumerate(paths):
            image = load_image(path)
            bbox, score = self.detector.detect(image)
            frames.append(Frame(
                index=idx,
                path=path,
                image=image,
                hand_bbox=bbox,
                hand_score=score,
            ))

        data.frames = frames
        data.anchor_index = select_anchor_frame(frames)
        return data
