"""Stage 2 — Per-frame hand reconstruction with WiLoR.

WiLoR (DINOv2-L backbone) estimates MANO pose (theta) and shape (beta)
parameters for each frame that contains a detected hand.  For the static
output we use the anchor frame mesh as the canonical hand geometry, but we
reconstruct all frames so alignment (Stage 6) can use the best available
translation estimate.

MANO face topology is fixed (1538 triangles, 778 vertices) so we store only
vertices per frame; faces come from the MANO layer itself.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import HandResult, PipelineData
from models.wilor import WiLoRModel
from models.moge_wrapper import MoGeModel


class HandReconstructionStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.wilor = WiLoRModel(
            checkpoint=cfg["wilor_checkpoint"],
            device=cfg.get("device", "cuda"),
        )
        self.moge = MoGeModel(
            checkpoint=cfg["moge_checkpoint"],
            device=cfg.get("device", "cuda"),
        )

    def run(self, data: PipelineData) -> PipelineData:
        anchor = data.frames[data.anchor_index]

        # Metric depth + camera intrinsics from MoGe on the anchor frame.
        depth, K = self.moge.estimate(anchor.image)
        data.depth_map = depth
        data.camera_intrinsics = K
        data.depth_maps[anchor.index] = depth

        results: list[HandResult] = []
        for frame in data.frames:
            if frame.hand_bbox is None:
                continue

            # Estimate depth for every frame — needed by FoundationPose in Stage 5.
            if frame.index not in data.depth_maps:
                frame_depth, _ = self.moge.estimate(frame.image)
                data.depth_maps[frame.index] = frame_depth

            frame_depth = data.depth_maps[frame.index]

            out = self.wilor.reconstruct(frame.image, frame.hand_bbox)

            metric_trans = _rescale_translation(
                out["translation"],
                out["vertices"],
                frame_depth if frame.index == anchor.index else None,
                K,
            )

            results.append(HandResult(
                frame_index=frame.index,
                mano_pose=out["pose"],
                mano_shape=out["shape"],
                global_rot=out["global_rot"],
                translation=metric_trans,
                vertices=out["vertices"],
                keypoints_3d=out["keypoints_3d"],
            ))

        data.hand_results = results
        return data


def _rescale_translation(
    raw_trans: np.ndarray,
    vertices: np.ndarray,
    depth_map: np.ndarray | None,
    K: np.ndarray,
) -> np.ndarray:
    """Align MANO root translation to MoGe metric depth via fingertip reprojection.

    WiLoR outputs translation up to an unknown scale. We solve for the scale
    factor by minimising reprojection error of fingertip vertices against the
    corresponding depth-lifted 3D points from MoGe.  If depth_map is None
    (non-anchor frame) we return raw_trans unchanged as a fallback.
    """
    if depth_map is None:
        return raw_trans

    # Fingertip vertex indices in the MANO topology (thumb to pinky).
    FINGERTIP_IDX = [745, 317, 444, 556, 673]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth_map.shape

    scales = []
    for vi in FINGERTIP_IDX:
        x3d, y3d, z3d = vertices[vi]
        # Project vertex to pixel
        u = int(round(x3d * fx / z3d + cx))
        v = int(round(y3d * fy / z3d + cy))
        if 0 <= u < W and 0 <= v < H and depth_map[v, u] > 0:
            scales.append(depth_map[v, u] / z3d)

    if not scales:
        return raw_trans

    scale = float(np.median(scales))
    return raw_trans * scale
