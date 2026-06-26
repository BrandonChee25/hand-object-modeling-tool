"""Stage 4 — Blind 3D object mesh generation (SAM-3D).

Takes the anchor frame crop (masked to the object) and feeds it into SAM-3D,
an image-conditioned generative 3D foundation model that produces a mesh
without any template or category prior — i.e. fully blind reconstruction.

SAM-3D jointly outputs:
  x_s — object shape (mesh in a canonical coordinate frame)
  x_p — canonical pose (R, t) in the anchor camera frame

We record both; the pose is used as the starting point for guided diffusion
in Stage 5.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectMesh, PipelineData
from models.sam3d_wrapper import SAM3DModel
from utils.io import crop_with_mask


class ObjectMeshGenerationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sam3d = SAM3DModel(
            checkpoint=cfg.get("sam3d_checkpoint", "stabilityai/TripoSR"),
            device=cfg.get("device", "cuda"),
        )

    def run(self, data: PipelineData) -> PipelineData:
        anchor_frame = data.frames[data.anchor_index]
        anchor_mask = data.object_seg.masks[data.anchor_index]

        # Crop the object region; background filled with the model's expected fill.
        cropped_image = crop_with_mask(
            anchor_frame.image,
            anchor_mask,
            padding=self.cfg.get("crop_padding_px", 32),
        )

        result = self.sam3d.generate(
            image=cropped_image,
            camera_intrinsics=data.camera_intrinsics,
        )
        # result keys: "vertices", "faces", "canonical_rot", "canonical_trans"

        data.object_mesh = ObjectMesh(
            vertices=result["vertices"],
            faces=result["faces"],
            canonical_rot=result["canonical_rot"],
            canonical_trans=result["canonical_trans"],
        )
        return data
