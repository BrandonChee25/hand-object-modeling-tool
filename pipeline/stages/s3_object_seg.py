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
from scipy.ndimage import binary_dilation

from pipeline.data import ObjectSegmentation, PipelineData
from models.sam2_wrapper import SAM2Model


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

        # Use the YOLO detection bbox — reliable pixel coords, no projection needed.
        H, W = anchor_frame.image.shape[:2]
        hand_mask = np.zeros((H, W), dtype=bool)
        if anchor_frame.hand_bbox is not None:
            x1, y1, x2, y2 = anchor_frame.hand_bbox.astype(int)
            hand_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True

        object_point = self.cfg.get("object_point")
        if object_point is not None:
            seed_mask = self.sam2.segment_with_point(anchor_frame.image, tuple(object_point))
            _save_debug_mask(anchor_frame.image, hand_mask, seed_mask)
        else:
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
        """Auto-segment anchor frame, exclude hand region, pick segment touching hand."""
        segments = self.sam2.auto_segment(image)
        hand_overlap_threshold = self.cfg.get("hand_overlap_threshold", 0.3)

        # Ring of pixels just outside the hand mask — the held object touches this.
        hand_border = binary_dilation(hand_mask, iterations=15) & ~hand_mask

        # Object can't be larger than 4× the hand bbox area (filters person/background).
        hand_area = max(int(hand_mask.sum()), 1)
        max_area = hand_area * 4

        candidates = []
        for seg in segments:
            area = int(seg.sum())
            if area == 0 or area > max_area:
                continue
            overlap = (seg & hand_mask).sum() / area
            if overlap > hand_overlap_threshold:
                continue
            contact = int((seg & hand_border).sum())
            # Score by contact density — penalises large segments that barely touch.
            score = contact / (area ** 0.5)
            candidates.append((score, contact, seg))

        if not candidates:
            raise RuntimeError(
                "Could not find an object segment on the anchor frame. "
                "Check that the object is visible and not fully occluded by the hand."
            )

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        print(f"[s3] top-3 segments by contact density: "
              + ", ".join(f"score={sc:.3f} contact={c} area={s.sum()}" for sc, c, s in candidates[:3]))

        selected = candidates[0][2]
        _save_debug_mask(image, hand_mask, selected, hand_border)
        return selected


def _save_debug_mask(
    image: np.ndarray,
    hand_mask: np.ndarray,
    object_mask: np.ndarray,
    hand_border: np.ndarray | None = None,
) -> None:
    import os
    from PIL import Image as PILImage
    os.makedirs("output", exist_ok=True)
    overlay = image.copy()
    overlay[hand_mask] = (overlay[hand_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    if hand_border is not None:
        overlay[hand_border] = (overlay[hand_border] * 0.5 + np.array([255, 165, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    overlay[object_mask] = (overlay[object_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(overlay).save("output/debug_segmentation.png")
    print("[s3] saved output/debug_segmentation.png  (red=hand bbox, orange=border zone, green=selected object)")
