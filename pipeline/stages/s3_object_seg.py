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


def _mano_silhouette_mask(
    hand_result,
    hand_bbox_px: np.ndarray,
    H: int,
    W: int,
    dilation: int = 10,
) -> np.ndarray | None:
    """Orthographic projection of MANO mesh vertices → filled convex-hull mask.

    Uses the detected hand bbox as the scale reference so the result is
    correctly positioned and sized regardless of camera K or Y-axis convention.
    The orthographic approximation is accurate enough for hands (all vertices
    at roughly the same depth).
    """
    import cv2

    verts = hand_result.vertices  # (778, 3)
    center = verts.mean(0)
    c = verts - center  # centred at 3D centroid

    hx1, hy1, hx2, hy2 = hand_bbox_px.astype(float)
    bbox_cx = (hx1 + hx2) / 2.0
    bbox_cy = (hy1 + hy2) / 2.0
    bbox_w  = hx2 - hx1
    bbox_h  = hy2 - hy1

    x_range = float(c[:, 0].max() - c[:, 0].min())
    y_range = float(c[:, 1].max() - c[:, 1].min())
    if x_range < 1e-6 or y_range < 1e-6:
        return None

    # Scale 3D X/Y extents to match 2D bbox dimensions.
    sx = bbox_w / x_range
    sy = bbox_h / y_range

    us = np.clip((c[:, 0] * sx + bbox_cx).astype(int), 0, W - 1)
    vs = np.clip((c[:, 1] * sy + bbox_cy).astype(int), 0, H - 1)

    pts = np.stack([us, vs], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)

    silhouette = mask.astype(bool)
    if dilation > 0:
        silhouette = binary_dilation(silhouette, iterations=dilation)
    return silhouette


def _projected_fingertip_probe(
    hand_result,
    K: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Project MANO fingertip joints (camera space) to a single 2D pixel probe.

    When a hand grips an object, the fingertips cluster around it, so their
    projected centroid lands on or very near the object — much more accurate
    than the geometric hand_center + direction heuristic.
    """
    tips_3d = hand_result.keypoints_3d[[4, 8, 12, 16, 20]]  # (5, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts = []
    for pt in tips_3d:
        X, Y, Z = float(pt[0]), float(pt[1]), float(pt[2])
        if Z <= 0:
            continue
        u = int(np.clip(fx * X / Z + cx, 0, W - 1))
        v = int(np.clip(fy * Y / Z + cy, 0, H - 1))
        pts.append((u, v))
    if not pts:
        return None
    mu = int(np.mean([p[0] for p in pts]))
    mv = int(np.mean([p[1] for p in pts]))
    print(f"[s3] projected fingertip centroid: ({mu}, {mv})  "
          f"(individual: {pts})")
    return (mu, mv)


def _grip_direction_points(
    hand_result,
    hand_bbox_px: np.ndarray,
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Fallback: wrist→fingertip direction, probing both sides of the hand."""
    kp = hand_result.keypoints_3d  # (21, 3) camera space
    wrist_cam = kp[0]
    tips_cam  = kp[[4, 8, 12, 16, 20]].mean(0)
    grip_cam  = tips_cam - wrist_cam

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
        hand_results_map = {r.frame_index: r for r in data.hand_results}

        if object_point is not None:
            anchor_frame = data.frames[data.anchor_index]
            seed_mask = self._fallback_sam2.segment_with_point(
                anchor_frame.image, tuple(object_point)
            )
            seed_index = data.anchor_index
            print(f"[s3] used manual object_point {object_point}")
        else:
            seed_index, seed_mask = self._find_held_object(data)

        seed_frame = data.frames[seed_index]
        hand_mask = self._hand_mask_for(seed_frame, hand_results_map.get(seed_index))
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
        """Bbox-rectangle hand mask (fast fallback)."""
        H, W = frame.image.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        if frame.hand_bbox is not None:
            x1, y1, x2, y2 = frame.hand_bbox.astype(int)
            mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
        return mask

    def _hand_mask_for(self, frame, hand_result=None) -> np.ndarray:
        """MANO silhouette mask when available, otherwise bbox rectangle."""
        if hand_result is not None and frame.hand_bbox is not None:
            H, W = frame.image.shape[:2]
            try:
                m = _mano_silhouette_mask(hand_result, frame.hand_bbox, H, W)
                if m is not None:
                    return m
            except Exception as e:
                print(f"[s3] MANO silhouette failed ({e}), falling back to bbox")
        return self._hand_mask(frame)

    def _find_held_object(self, data: PipelineData) -> tuple[int, np.ndarray]:
        """Find the held object seed mask.

        Priority order per candidate frame:
          1. SAM-2 box+point prompt around fingertip-projected location.
          2. SAM3 text + box prompt at the same location (fallback).
          3. SAM-2 auto-segment contact heuristic (final fallback).

        All steps use a tight MANO silhouette mask (not the bbox rectangle)
        to exclude the hand, so object pixels inside the bbox are preserved.
        """
        frames = data.frames
        indices = self._candidate_indices(data)
        H, W = frames[0].image.shape[:2]
        max_pixels = 0.30 * H * W

        hand_results = {r.frame_index: r for r in data.hand_results}
        K = data.camera_intrinsics

        for fidx in indices:
            frame = frames[fidx]
            if frame.hand_bbox is None:
                continue

            hr        = hand_results.get(fidx)
            hand_mask = self._hand_mask_for(frame, hr)
            depth     = data.depth_maps.get(fidx, data.depth_map)
            hand_depth = _median_depth(depth, hand_mask)

            tip_points: list[tuple[int, int]] = []
            if hr is not None:
                if K is not None:
                    try:
                        p = _projected_fingertip_probe(hr, K, H, W)
                        if p is not None:
                            tip_points = [p]
                    except Exception as e:
                        print(f"[s3] K-projection failed: {e}")
                if not tip_points:
                    try:
                        tip_points = _grip_direction_points(hr, frame.hand_bbox, (H, W))
                    except Exception:
                        pass

            hx1, hy1, hx2, hy2 = frame.hand_bbox.astype(float)
            hand_size = max(hx2 - hx1, hy2 - hy1)
            r    = int(hand_size * 0.55)
            half = int(hand_size * 0.7)

            # --- attempt 1: depth-band isolation ---
            # The hand is in the foreground; the body is farther away.
            # Pixels at hand_depth ± 25% are foreground (hand + object).
            # After removing the MANO silhouette, what remains should be
            # the object — no SAM call needed, no body leakage.
            if depth is not None and hand_depth is not None and tip_points:
                depth_lo = hand_depth * 0.75
                depth_hi = hand_depth * 1.25
                foreground = (
                    np.isfinite(depth)
                    & (depth >= depth_lo)
                    & (depth <= depth_hi)
                )
                candidates = foreground & ~hand_mask
                if candidates.any():
                    obj = _nearest_component(candidates, tip_points[0])
                    print(f"[s3] frame {fidx}: depth-band foreground "
                          f"[{depth_lo:.2f}, {depth_hi:.2f}]m → "
                          f"{int(candidates.sum())} px candidates, "
                          f"{int(obj.sum())} px nearest component")
                    if self._valid_object_mask(obj, max_pixels, depth, hand_depth, fidx, "depth-band"):
                        return fidx, obj

            # --- attempt 2: SAM3 text+box (SAM fallback) ---
            bbox_hints = [
                (
                    max(0, tip_point[0] - r), max(0, tip_point[1] - r),
                    min(W - 1, tip_point[0] + r), min(H - 1, tip_point[1] + r),
                )
                for tip_point in tip_points
            ] or [None]

            for bbox_hint in bbox_hints:
                print(f"[s3] frame {fidx}: trying SAM3 text+box {bbox_hint or '(expanded hand bbox)'}")
                mask = self.seg_model.segment_held_object(
                    frame.image,
                    tuple(frame.hand_bbox.astype(int)),
                    hand_mask,
                    object_bbox_hint=bbox_hint,
                )
                if mask is not None and mask.any():
                    if self._valid_object_mask(mask, max_pixels, depth, hand_depth, fidx, "SAM3"):
                        return fidx, mask

            # --- attempt 3: SAM-2 box+point prompt (final SAM fallback) ---
            for tip_point in tip_points:
                if depth is not None and hand_depth is not None:
                    px, py = tip_point
                    if 0 <= py < depth.shape[0] and 0 <= px < depth.shape[1]:
                        pt_depth = float(depth[py, px])
                        if pt_depth > hand_depth * 1.5:
                            print(f"[s3] frame {fidx}: point {tip_point} depth "
                                  f"{pt_depth:.2f}m > hand {hand_depth:.2f}m, skipping")
                            continue
                box = (
                    max(0, tip_point[0] - half),
                    max(0, tip_point[1] - half),
                    min(W - 1, tip_point[0] + half),
                    min(H - 1, tip_point[1] + half),
                )
                print(f"[s3] frame {fidx}: trying SAM-2 box+point {box} "
                      f"({box[2]-box[0]}×{box[3]-box[1]}px) anchored at {tip_point}")
                mask = self._fallback_sam2.segment_with_box_and_point(
                    frame.image, box, tip_point
                )
                if mask is not None and mask.any():
                    mask = mask & ~hand_mask
                    mask = _nearest_component(mask, tip_point)
                    if self._valid_object_mask(mask, max_pixels, depth, hand_depth, fidx, "SAM-2 box+point"):
                        return fidx, mask

        print("[s3] all prompted attempts failed — using contact heuristic")
        anchor_frame = frames[data.anchor_index]
        anchor_hr    = hand_results.get(data.anchor_index)
        hand_mask    = self._hand_mask_for(anchor_frame, anchor_hr)
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
            if md is not None and md > hand_depth * 1.5:
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


def _nearest_component(mask: np.ndarray, probe: tuple[int, int]) -> np.ndarray:
    """Keep only the connected component whose closest pixel is nearest probe (x, y)."""
    from scipy.ndimage import label as _label
    labeled, n = _label(mask)
    if n <= 1:
        return mask
    px, py = probe
    best_id, best_dist = 1, float("inf")
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        d = float(np.sqrt((xs - px) ** 2 + (ys - py) ** 2).min())
        if d < best_dist:
            best_dist = d
            best_id = cid
    return (labeled == best_id).astype(bool)


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
