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

            out = self.wilor.reconstruct(frame.image, frame.hand_bbox)

            metric_trans = out["translation"]

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
