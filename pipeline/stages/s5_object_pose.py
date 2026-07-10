"""Stage 5 — Object pose estimation (STUBBED).

FoundationPose will replace this stage. For now the stage passes the
TripoSR canonical pose (identity rotation, zero translation) through for
every frame so the rest of the pipeline can be tested end-to-end.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectPoseSequence, PipelineData


class ObjectPoseEstimationStage:
    def __init__(self, cfg: dict):
        pass

    def run(self, data: PipelineData) -> PipelineData:
        print("[5/7] Object pose estimation: STUBBED — using identity pose for all frames.")
        n = len(data.frames)
        data.object_poses = ObjectPoseSequence(
            rots=[np.eye(3, dtype=np.float32) for _ in range(n)],
            trans=[np.zeros(3, dtype=np.float32) for _ in range(n)],
            alpha_p_values=[0.0] * n,
        )
        return data
