"""3D mesh generation wrapper — TripoSR (public substitute for SAM-3D).

SAM-3D (as used in "Do as I Do", arXiv 2606.19333) is not yet publicly
released.  TripoSR is the closest available model: it is an image-conditioned
feed-forward 3D reconstruction model that produces a mesh from a single masked
RGB crop with no category prior (fully blind).

The key difference from the paper's SAM-3D:
  - TripoSR is FEED-FORWARD, not diffusion-based.
  - This means the guided flow-matching tracker in Stage 5 (which calls
    model.step() per denoising step) cannot be used as described.
  - Instead, Stage 5 uses FoundationPose for per-frame 6-DoF pose tracking
    given the TripoSR-reconstructed mesh.  See guided_diffusion.py.

Install: pip install git+https://github.com/VAST-AI-Research/TripoSR
Weights:  downloaded automatically from HuggingFace (stabilityai/TripoSR).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class SAM3DModel:
    def __init__(
        self,
        checkpoint: str = "stabilityai/TripoSR",
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from tsr.system import TSR

        self._model = TSR.from_pretrained(
            self.checkpoint,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(131072)
        self._model.to(self.device)

    def generate(
        self,
        image: np.ndarray,
        camera_intrinsics: np.ndarray,
        n_steps: int = 50,  # unused for TripoSR (feed-forward); kept for API compat
    ) -> dict:
        """
        Full blind 3D mesh generation from the anchor frame crop.

        Parameters
        ----------
        image             : H x W x 3 uint8 (object masked, background grey)
        camera_intrinsics : (3, 3) K matrix  [not used by TripoSR; kept for compat]
        n_steps           : ignored for TripoSR

        Returns
        -------
        dict with keys:
            vertices        : (V, 3) float32 in object-canonical space
            faces           : (F, 3) int32
            canonical_rot   : (3, 3)  identity (TripoSR outputs in canonical frame)
            canonical_trans : (3,)    zeros
        """
        self._load()

        pil_image = Image.fromarray(image).resize((512, 512))

        with torch.no_grad():
            scene_codes = self._model([pil_image], device=self.device)
            meshes = self._model.extract_mesh(scene_codes, resolution=256)

        mesh = meshes[0]
        # mesh is a trimesh.Trimesh object
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        # TripoSR outputs in a canonical object frame centred at the origin.
        return {
            "vertices": vertices,
            "faces": faces,
            "canonical_rot": np.eye(3, dtype=np.float32),
            "canonical_trans": np.zeros(3, dtype=np.float32),
        }

    def step(self, x_s, x_p, t, dt, image, camera_intrinsics):
        """Not available for TripoSR — use FoundationPose tracker instead."""
        raise NotImplementedError(
            "TripoSR is feed-forward and does not support iterative denoising steps. "
            "Stage 5 uses FoundationPose for pose tracking instead."
        )
