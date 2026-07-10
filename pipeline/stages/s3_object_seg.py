"""Stage 3 — Object segmentation across all frames.

Model selection (automatic, no config needed):
  • If SAM3 is installed (compute cluster) → uses SAM3SegModel.
    SAM3 is jointly-trained for detection + segmentation + tracking, which
    handles partially occluded held objects better than SAM-2 alone.
  • Otherwise falls back to SAM-2 (local dev machine).

Override with config key  use_sam3: false  to force SAM-2 even when SAM3 is present.

Seed mask selection priority:
  1. object_point in config  →  point prompt (manual, most reliable).
  2. Otherwise               →  fingertip-directed box prompt across N candidate frames.
  3. Fallback                →  SAM-2 auto-segment filtered by hand contact + depth.

The seed mask is propagated through all frames using the loaded model's video predictor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from pipeline.data import ObjectSegmentation, PipelineData

# MANO vertex indices for fingertips (same set used in Stage 5).
_FINGERTIP_VERT_IDX = [745, 317, 444, 556, 673]


def _load_seg_model(cfg: dict):
    """Return a SAM3SegModel if available, else fall back to SAM2Model."""
    device = cfg.get("device", "cuda")
    use_sam3 = cfg.get("use_sam3", True)

    if use_sam3:
        try:
            from models.sam3_seg_wrapper import SAM3SegModel
            model = SAM3SegModel(device=device, version=cfg.get("sam3_version", "sam3.1"))
            print("[s3] using SAM3 for object segmentation")
            return model, "sam3"
        except ImportError:
            print("[s3] SAM3 not installed — falling back to SAM-2")

    from models.sam2_wrapper import SAM2Model
    model = SAM2Model(
        checkpoint=cfg["sam2_checkpoint"],
        config=cfg.get("sam2_config", "sam2_hiera_large.yaml"),
        device=device,
    )
    print("[s3] using SAM-2 for object segmentation")
    return model, "sam2"


def _grip_direction_points(
    hand_result,
    hand_bbox_px: np.ndarray,
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return candidate (x, y) pixel coordinates beyond each end of the hand.

    Returns BOTH the computed grip direction AND its opposite.  The caller
    picks whichever lands at a depth similar to the hand (the arm extends
    away from camera = deeper; the held object stays at hand depth).
    """
    vertices = hand_result.vertices   # (778, 3) MANO local space
    R        = hand_result.global_rot # (3, 3)  local → camera

    hand_center_local  = vertices.mean(0)
    tip_centroid_local = vertices[_FINGERTIP_VERT_IDX].mean(0)
    grip_local = tip_centroid_local - hand_center_local

    grip_cam = R @ grip_local
    direction_2d = np.array([grip_cam[0], grip_cam[1]])
    d_norm = float(np.linalg.norm(direction_2d))
    if d_norm < 1e-6:
        return []
    direction_2d /= d_norm

    hx1, hy1, hx2, hy2 = hand_bbox_px.astype(float)
    hand_cx   = (hx1 + hx2) / 2
    hand_cy   = (hy1 + hy2) / 2
    hand_size = max(hx2 - hx1, hy2 - hy1)
    H, W = image_shape

    pts = []
    for sign in (+1, -1):
        pt = np.array([hand_cx, hand_cy]) + sign * direction_2d * hand_size * 0.9
        px = int(np.clip(pt[0], 0, W - 1))
        py = int(np.clip(pt[1], 0, H - 1))
        pts.append((px, py))
    return pts


class ObjectSegmentationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seg_model, self._model_type = _load_seg_model(cfg)
        # Keep a direct reference to SAM2Model for the contact-heuristic fallback
        # (auto_segment is SAM-2 only; not needed when SAM3 finds the object).
        self.sam2 = self.seg_model if self._model_type == "sam2" else None
        if self._model_type == "sam3":
            from models.sam2_wrapper import SAM2Model
            self._fallback_sam2 = SAM2Model(
                checkpoint=cfg["sam2_checkpoint"],
                config=cfg.get("sam2_config", "sam2_hiera_large.yaml"),
                device=cfg.get("device", "cuda"),
            )
        else:
            self._fallback_sam2 = self.seg_model

    def run(self, data: PipelineData) -> PipelineData:
        output_dir = data.output_dir or Path("output")
        object_point = self.cfg.get("object_point")

        if object_point is not None:
            anchor_frame = data.frames[data.anchor_index]
            # Point prompt is SAM-2 only; use the fallback SAM2 regardless of primary model.
            seed_mask = self._fallback_sam2.segment_with_point(
                anchor_frame.image, tuple(object_point)
            )
            seed_index = data.anchor_index
            print(f"[s3] used manual object_point {object_point}")
        else:
            seed_index, seed_mask = self._find_held_object(data)

        seed_frame = data.frames[seed_index]
        hand_mask = self._hand_mask(seed_frame)
        _save_debug_mask(seed_frame.image, hand_mask, seed_mask, output_dir)

        all_images = [f.image for f in data.frames]
        masks = self.seg_model.propagate(
            images=all_images,
            anchor_index=seed_index,
            seed_mask=seed_mask,
        )

        data.object_seg = ObjectSegmentation(
            masks=masks,
            anchor_frame_index=seed_index,
        )
        return data

    def _candidate_indices(self, data: PipelineData) -> list[int]:
        """Return N evenly-spaced frame indices, always including the anchor."""
        frames = data.frames
        n = self.cfg.get("detection_candidate_frames", 5)
        step = max(1, (len(frames) - 1) // (n - 1)) if len(frames) > 1 else 1
        indices = list(range(0, len(frames), step))[:n]
        return sorted(set(indices + [data.anchor_index]))

    def _hand_mask(self, frame) -> np.ndarray:
        H, W = frame.image.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        if frame.hand_bbox is not None:
            x1, y1, x2, y2 = frame.hand_bbox.astype(int)
            mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
        return mask

    def _find_held_object(self, data: PipelineData) -> tuple[int, np.ndarray]:
        """Find the held object seed mask.

        Priority order per candidate frame:
          1. SAM-2 point prompt at the fingertip-projected location (most targeted).
          2. SAM3 text + box prompt at the same location (fallback if point hits background).
          3. SAM-2 auto-segment contact heuristic (final fallback across all frames).
        """
        frames = data.frames
        indices = self._candidate_indices(data)
        H, W = frames[0].image.shape[:2]
        max_pixels = 0.30 * H * W

        hand_results = {r.frame_index: r for r in data.hand_results}

        for fidx in indices:
            frame = frames[fidx]
            if frame.hand_bbox is None:
                continue
            hand_mask = self._hand_mask(frame)
            depth     = data.depth_maps.get(fidx, data.depth_map)
            hand_depth = _median_depth(depth, hand_mask)

            hr = hand_results.get(fidx)
            tip_points: list[tuple[int, int]] = []
            if hr is not None:
                try:
                    tip_points = _grip_direction_points(hr, frame.hand_bbox, (H, W))
                except Exception:
                    pass

            hx1, hy1, hx2, hy2 = frame.hand_bbox.astype(float)
            hand_size = max(hx2 - hx1, hy2 - hy1)
            r = int(hand_size * 0.55)

            # --- attempt 1: SAM-2 point prompt at each candidate location ---
            for tip_point in tip_points:
                print(f"[s3] frame {fidx}: trying SAM-2 point prompt at {tip_point}")
                mask = self._fallback_sam2.segment_with_point(frame.image, tip_point)
                if mask is not None and mask.any():
                    mask = mask & ~hand_mask
                    if self._valid_object_mask(mask, max_pixels, depth, hand_depth, fidx, "SAM-2 point"):
                        return fidx, mask

            # --- attempt 2: SAM3 text + box prompt at each candidate location ---
            bbox_hints = []
            for tip_point in tip_points:
                bbox_hints.append((
                    max(0, tip_point[0] - r), max(0, tip_point[1] - r),
                    min(W - 1, tip_point[0] + r), min(H - 1, tip_point[1] + r),
                ))
            if not bbox_hints:
                bbox_hints = [None]  # fall back to expanded hand bbox

            for bbox_hint in bbox_hints:
                print(f"[s3] frame {fidx}: trying SAM3 box prompt {bbox_hint or '(expanded hand bbox)'}")
                mask = self.seg_model.segment_held_object(
                    frame.image,
                    tuple(frame.hand_bbox.astype(int)),
                    hand_mask,
                    object_bbox_hint=bbox_hint,
                )
                if mask is not None and mask.any():
                    if self._valid_object_mask(mask, max_pixels, depth, hand_depth, fidx, "SAM3 box"):
                        return fidx, mask

        print("[s3] all prompted attempts failed — using contact heuristic")
        anchor_frame = frames[data.anchor_index]
        hand_mask    = self._hand_mask(anchor_frame)
        depth        = data.depth_maps.get(data.anchor_index, data.depth_map)
        return data.anchor_index, self._seed_mask(anchor_frame.image, hand_mask, depth_map=depth)

    def _valid_object_mask(
        self,
        mask: np.ndarray,
        max_pixels: float,
        depth: np.ndarray | None,
        hand_depth: float | None,
        fidx: int,
        label: str,
    ) -> bool:
        if not mask.any():
            return False
        n = int(mask.sum())
        if n < 200:
            print(f"[s3] frame {fidx} {label}: mask too small ({n} px)")
            return False
        if n > max_pixels:
            print(f"[s3] frame {fidx} {label}: mask too large ({n} px)")
            return False
        if depth is not None and hand_depth is not None:
            md = _median_depth(depth, mask)
            if md is not None and md > hand_depth * 1.2:
                print(f"[s3] frame {fidx} {label}: mask depth {md:.2f}m > hand {hand_depth:.2f}m, likely arm")
                return False
        print(f"[s3] frame {fidx} {label}: accepted ({n} px)")
        return True

    def _seed_mask(
        self,
        image: np.ndarray,
        hand_mask: np.ndarray,
        depth_map: np.ndarray | None = None,
    ) -> np.ndarray:
        """SAM-2 auto-segment; pick segment touching the hand at the same depth."""
        segments = self._fallback_sam2.auto_segment(image)
        hand_overlap_threshold = self.cfg.get("hand_overlap_threshold", 0.3)
        hand_border = binary_dilation(hand_mask, iterations=20) & ~hand_mask
        hand_depth = _median_depth(depth_map, hand_mask)

        if hand_depth is not None:
            print(f"[s3] heuristic hand depth: {hand_depth:.3f}m")

        candidates = []
        for seg in segments:
            area = int(seg.sum())
            if area == 0:
                continue
            if (seg & hand_mask).sum() / area > hand_overlap_threshold:
                continue
            contact = int((seg & hand_border).sum())
            if contact == 0:
                continue

            contact_score = contact / (area ** 0.5)

            depth_weight = 1.0
            if hand_depth is not None and depth_map is not None:
                seg_depth = _median_depth(depth_map, seg)
                if seg_depth is not None:
                    rel_diff = abs(seg_depth - hand_depth) / max(hand_depth, 1e-6)
                    depth_weight = 1.0 / (1.0 + 5.0 * rel_diff)
                else:
                    depth_weight = 0.5

            candidates.append((contact_score * depth_weight, contact, seg))

        if not candidates:
            raise RuntimeError(
                "Could not find an object segment touching the hand. "
                "Set object_point: [x, y] in config as a manual pixel-coordinate fallback."
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        print(
            "[s3] top-3 heuristic segments: "
            + ", ".join(
                f"score={sc:.3f} contact={c} area={int(s.sum())}"
                for sc, c, s in candidates[:3]
            )
        )
        return candidates[0][2]


def _median_depth(
    depth_map: np.ndarray | None, mask: np.ndarray
) -> float | None:
    if depth_map is None or not mask.any():
        return None
    vals = depth_map[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) > 0 else None


def _save_debug_mask(
    image: np.ndarray,
    hand_mask: np.ndarray,
    object_mask: np.ndarray,
    output_dir: Path,
) -> None:
    from PIL import Image as PILImage
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    overlay[hand_mask] = (overlay[hand_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    overlay[object_mask] = (overlay[object_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(overlay).save(output_dir / "debug_segmentation.png")
    print(f"[s3] saved {output_dir}/debug_segmentation.png  (red=hand, green=object)")
