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

# MANO 21-joint convention: joints 4,8,12,16,20 are fingertips; 0 is wrist.
_FINGERTIP_JOINTS = [4, 8, 12, 16, 20]


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


def _fingertip_object_bbox(
    hand_result,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a pixel bbox for the expected object location using fingertip projections.

    Projects the MANO fingertip joints (camera space) to 2D, then places a
    search box just BEYOND the fingertips in the palm→fingertip direction.
    This avoids including the arm/wrist in the SAM3 prompt.
    """
    kp = hand_result.keypoints_3d  # (21, 3) camera space
    if kp is None or len(kp) < 21:
        return None

    K = np.array(K, dtype=np.float64)

    def project(p3d: np.ndarray) -> np.ndarray:
        p = K @ p3d.astype(np.float64)
        return np.array([p[0] / p[2], p[1] / p[2]])

    wrist_2d      = project(kp[0])
    fingertips_2d = np.array([project(kp[i]) for i in _FINGERTIP_JOINTS])  # (5, 2)
    tip_centroid  = fingertips_2d.mean(0)  # (2,)

    direction = tip_centroid - wrist_2d
    dist = float(np.linalg.norm(direction))
    if dist < 1e-3:
        return None
    direction /= dist

    # Place the search box just beyond the fingertip centroid.
    # dist ≈ distance from wrist to fingertips in pixels ≈ half the hand length.
    hand_len_px = dist
    object_center = tip_centroid + direction * hand_len_px * 0.35
    radius = hand_len_px * 0.55  # generous radius to capture the held object

    H, W = image_shape
    x1 = int(np.clip(object_center[0] - radius, 0, W - 1))
    y1 = int(np.clip(object_center[1] - radius, 0, H - 1))
    x2 = int(np.clip(object_center[0] + radius, 0, W - 1))
    y2 = int(np.clip(object_center[1] + radius, 0, H - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


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
        """Try SAM3 fingertip-directed box prompt across candidate frames; fall back to heuristic."""
        frames = data.frames
        indices = self._candidate_indices(data)
        H, W = frames[0].image.shape[:2]
        max_pixels = 0.30 * H * W

        # Build a quick lookup: frame_index → HandResult
        hand_results = {r.frame_index: r for r in data.hand_results}

        for fidx in indices:
            frame = frames[fidx]
            if frame.hand_bbox is None:
                continue
            hand_mask = self._hand_mask(frame)
            depth = data.depth_maps.get(fidx, data.depth_map)
            hand_depth = _median_depth(depth, hand_mask)

            # Compute a fingertip-directed search box (avoids the arm/wrist).
            object_bbox_hint = None
            hr = hand_results.get(fidx)
            if hr is not None and data.camera_intrinsics is not None:
                try:
                    object_bbox_hint = _fingertip_object_bbox(
                        hr, data.camera_intrinsics, (H, W)
                    )
                except Exception:
                    pass
            if object_bbox_hint is not None:
                print(f"[s3] frame {fidx}: using fingertip-directed bbox {object_bbox_hint}")
            else:
                print(f"[s3] frame {fidx}: fingertip projection unavailable, using expanded hand bbox")

            mask = self.seg_model.segment_held_object(
                frame.image,
                tuple(frame.hand_bbox.astype(int)),
                hand_mask,
                object_bbox_hint=object_bbox_hint,
            )
            if mask is None or not mask.any():
                print(f"[s3] frame {fidx}: hand-box prompt returned empty")
                continue

            # Reject if mask is implausibly large (grabbed whole background).
            if mask.sum() > max_pixels:
                print(f"[s3] frame {fidx}: mask too large ({mask.sum()} px), skipping")
                continue

            # Reject if mask is tiny (noise / degenerate).
            if mask.sum() < 200:
                print(f"[s3] frame {fidx}: mask too small ({mask.sum()} px), skipping")
                continue

            # Reject if mask depth is farther than hand depth (arm extends away).
            if depth is not None and hand_depth is not None:
                mask_depth = _median_depth(depth, mask)
                if mask_depth is not None and mask_depth > hand_depth * 1.2:
                    print(f"[s3] frame {fidx}: mask depth {mask_depth:.2f}m "
                          f"farther than hand {hand_depth:.2f}m, likely arm")
                    continue

            print(f"[s3] hand-box prompt succeeded on frame {fidx} ({mask.sum()} px)")
            return fidx, mask

        print("[s3] hand-box prompt failed on all frames — using contact heuristic")
        anchor_frame = frames[data.anchor_index]
        hand_mask = self._hand_mask(anchor_frame)
        depth = data.depth_maps.get(data.anchor_index, data.depth_map)
        return data.anchor_index, self._seed_mask(anchor_frame.image, hand_mask, depth_map=depth)

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
