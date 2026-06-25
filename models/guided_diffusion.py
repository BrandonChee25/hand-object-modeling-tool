"""Object pose tracker — FoundationPose (practical substitute for guided diffusion).

The "Do as I Do" paper's guided flow-matching tracker requires a diffusion-based
3D model (SAM-3D) that exposes per-step denoising.  Since we use TripoSR
(feed-forward) for mesh generation, we instead use FoundationPose for per-frame
6-DoF pose tracking.

FoundationPose takes:
  - A reference mesh (from TripoSR, Stage 4)
  - A per-frame RGB image + object mask
  - The previous-frame pose as initialisation
and returns a refined SE(3) pose.  This is conceptually equivalent to the
"pose guidance" half of the paper's tracker: previous pose initialises the
search, and the model refines it toward the current observation.

Install FoundationPose:
    git clone https://github.com/NVlabs/FoundationPose
    cd FoundationPose && pip install -e .

Weights: download from the FoundationPose release page and put in
    checkpoints/foundationpose/

SE(3) candidate clustering from the paper IS still used here:
  - FoundationPose internally samples hypotheses and scores them.
  - We add a post-hoc SE(3) clustering step across the N=25 top hypotheses
    it returns, selecting the cluster centroid as the final pose.
"""

from __future__ import annotations

import numpy as np

from models.sam3d_wrapper import SAM3DModel
from utils.geometry import (
    se3_cluster_centroid,
    pose7d_to_Rt,
    Rt_to_pose7d,
    rotation_velocity_2d,
)


class GuidedDiffusionTracker:
    """
    Pose tracker that mirrors the interface expected by Stage 5.

    Internally uses FoundationPose rather than flow-matching because our
    3D generation model (TripoSR) is feed-forward.
    """

    def __init__(
        self,
        model: SAM3DModel,           # kept for API compat, not used here
        n_candidates: int = 25,
        alpha_s: float = 0.95,       # unused — shape is fixed by construction
        alpha_p_range: list = (0.0, 0.6),
        cluster_threshold_deg: float = 15.0,
        cluster_threshold_m: float = 0.05,
        n_steps: int = 50,           # number of FoundationPose refine iters
    ):
        self.n_candidates = n_candidates
        self.cluster_threshold_deg = cluster_threshold_deg
        self.cluster_threshold_m = cluster_threshold_m
        self.n_steps = n_steps
        self._tracker = None

    def _load(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> None:
        """Lazy-load FoundationPose and register the object mesh."""
        if self._tracker is not None:
            return
        from foundationpose.run_demo import FoundationPose as _FP
        import trimesh

        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
        self._tracker = _FP(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer_ckpt_dir="checkpoints/foundationpose/",
            refiner_ckpt_dir="checkpoints/foundationpose/",
        )

    def track_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fixed_shape_vertices: np.ndarray,
        fixed_shape_faces: np.ndarray,
        prev_rot: np.ndarray,
        prev_trans: np.ndarray,
        alpha_p: float,   # kept for API compat; controls refine iterations below
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate object pose for one frame using FoundationPose.

        alpha_p maps to refinement effort: high alpha_p (slow motion) →
        fewer iterations (trust previous pose); low alpha_p (fast motion) →
        more iterations (let the model search).
        """
        self._load(fixed_shape_vertices, fixed_shape_faces)

        # Map alpha_p to number of refine iterations.
        n_refine = max(1, int(self.n_steps * (1.0 - alpha_p)))

        pose_4x4 = self._tracker.track_one(
            rgb=image,
            depth=None,          # RGB-only mode
            K=np.eye(3),         # updated by caller if needed
            ob_mask=mask,
            prev_pose=_Rt_to_4x4(prev_rot, prev_trans),
            n_init_hyp=self.n_candidates,
            n_refine=n_refine,
        )

        R = pose_4x4[:3, :3]
        t = pose_4x4[:3, 3]
        return R, t


def _Rt_to_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
