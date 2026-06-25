"""Stage 3 — Object segmentation across all frames with SAM-2.

On the anchor frame we obtain a seed mask by:
  1. Running SAM-2 automatic segmentation.
  2. Removing any segment that overlaps significantly with the hand mask.
  3. Selecting the largest remaining segment as the object.

SAM-2's video predictor then propagates this seed mask forward and backward
through the full frame sequence.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectSegmentation, PipelineData
from models.sam2_wrapper import SAM2Model
from utils.geometry import mano_vertices_to_mask


class ObjectSegmentationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sam2 = SAM2Model(
            checkpoint=cfg["sam2_checkpoint"],
            config=cfg.get("sam2_config", "sam2_hiera_large.yaml"),
            device=cfg.get("device", "cuda"),
        )

    def run(self, data: PipelineData) -> PipelineData:
        anchor_frame = data.frames[data.anchor_index]
        anchor_hand = next(
            (r for r in data.hand_results if r.frame_index == data.anchor_index),
            None,
        )

        hand_mask = (
            mano_vertices_to_mask(
                anchor_hand.vertices,
                data.camera_intrinsics,
                anchor_frame.image.shape[:2],
            )
            if anchor_hand is not None
            else np.zeros(anchor_frame.image.shape[:2], dtype=bool)
        )

        seed_mask = self._seed_mask(anchor_frame.image, hand_mask)

        all_images = [f.image for f in data.frames]
        masks = self.sam2.propagate(
            images=all_images,
            anchor_index=data.anchor_index,
            seed_mask=seed_mask,
        )

        data.object_seg = ObjectSegmentation(
            masks=masks,
            anchor_frame_index=data.anchor_index,
        )
        return data

    def _seed_mask(self, image: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
        """Auto-segment anchor frame, exclude hand region, pick largest segment."""
        segments = self.sam2.auto_segment(image)

        best_mask: np.ndarray | None = None
        best_area = 0
        hand_overlap_threshold = self.cfg.get("hand_overlap_threshold", 0.3)

        for seg in segments:
            overlap = (seg & hand_mask).sum() / max(seg.sum(), 1)
            if overlap > hand_overlap_threshold:
                continue
            area = int(seg.sum())
            if area > best_area:
                best_area = area
                best_mask = seg

        if best_mask is None:
            raise RuntimeError(
                "Could not find an object segment on the anchor frame. "
                "Check that the object is visible and not fully occluded by the hand."
            )

        return best_mask
