"""Stage 5 — Per-frame object pose estimation via guided diffusion.

Implements the guided flow-matching tracker from "Do as I Do" (arXiv 2606.19333).

The key idea:
  - SAM-3D learns a joint distribution p_θ(x_s, x_p | c) over shape and pose.
  - We FIX the shape x̄_s at the anchor frame (blind reconstruction from Stage 4).
  - For each subsequent frame k we sample the pose from:
        p_θ(x_k_p | x_k_s = x̄_s, c_k)
    guided toward the previous-frame pose to enforce temporal continuity.

Guidance blends the denoising velocity with a reference interpolant:
    x_t_s = (1 - α_s)(x_{t-Δ}_s + Δ v_θ_s) + α_s · z_ref_s(t)
    x_t_p = (1 - α_p)(x_{t-Δ}_p + Δ v_θ_p) + α_p · z_ref_p(t)

α_s ∈ [0.9, 1.0]  — near-rigid shape anchoring.
α_p               — derived per-frame from 2D rotational velocity (BootsTAPIR),
                    higher when the object moves fast (more trust in denoiser),
                    lower when static (more trust in previous pose).

From N=25 candidate poses we select the best via SE(3) clustering (30× faster
than likelihood scoring): find the cluster with the most members, return its
centroid.

For a STATIC output (our goal) we use all per-frame poses to compute a single
consensus pose — the SE(3) mean of the inlier cluster.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectPoseSequence, PipelineData
from models.sam3d_wrapper import SAM3DModel
from models.guided_diffusion import GuidedDiffusionTracker
from utils.geometry import se3_cluster_mean, rotation_velocity_2d


class ObjectPoseEstimationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sam3d = SAM3DModel(
            checkpoint=cfg.get("sam3d_checkpoint", "stabilityai/TripoSR"),
            device=cfg.get("device", "cuda"),
        )
        self.tracker = GuidedDiffusionTracker(
            model=self.sam3d,
            n_candidates=cfg.get("n_pose_candidates", 25),
            alpha_s=cfg.get("alpha_s", 0.95),
            alpha_p_range=cfg.get("alpha_p_range", [0.0, 0.6]),
            cluster_threshold_deg=cfg.get("cluster_threshold_deg", 15.0),
            cluster_threshold_m=cfg.get("cluster_threshold_m", 0.05),
        )

    def run(self, data: PipelineData) -> PipelineData:
        shape = data.object_mesh  # fixed x̄_s from Stage 4
        anchor_rot = shape.canonical_rot
        anchor_trans = shape.canonical_trans

        rots: list[np.ndarray] = []
        trans: list[np.ndarray] = []
        alpha_ps: list[float] = []

        prev_rot = anchor_rot.copy()
        prev_trans = anchor_trans.copy()

        frames = data.frames
        masks = data.object_seg.masks

        for frame in frames:
            if frame.index == data.anchor_index:
                rots.append(anchor_rot.copy())
                trans.append(anchor_trans.copy())
                alpha_ps.append(0.0)
                continue

            # Adaptive α_p from 2D rotational velocity of tracked object points.
            alpha_p = _compute_alpha_p(
                masks[frame.index - 1] if frame.index > 0 else masks[frame.index],
                masks[frame.index],
                frame.image,
                frames[frame.index - 1].image if frame.index > 0 else frame.image,
                self.cfg.get("alpha_p_range", [0.0, 0.6]),
            )

            R, t = self.tracker.track_frame(
                image=frame.image,
                mask=masks[frame.index],
                fixed_shape_vertices=shape.vertices,
                fixed_shape_faces=shape.faces,
                prev_rot=prev_rot,
                prev_trans=prev_trans,
                alpha_p=alpha_p,
            )

            rots.append(R)
            trans.append(t)
            alpha_ps.append(alpha_p)
            prev_rot = R
            prev_trans = t

        data.object_poses = ObjectPoseSequence(
            rots=rots,
            trans=trans,
            alpha_p_values=alpha_ps,
        )
        return data


def _compute_alpha_p(
    prev_mask: np.ndarray,
    curr_mask: np.ndarray,
    curr_image: np.ndarray,
    prev_image: np.ndarray,
    alpha_range: list[float],
) -> float:
    """Map 2D rotational velocity to a guidance weight α_p.

    Higher velocity → object is moving fast → denoiser gets more freedom (low α_p).
    Near-static → previous pose is reliable → high α_p anchors to it.
    """
    vel = rotation_velocity_2d(prev_mask, curr_mask, prev_image, curr_image)
    alpha_min, alpha_max = alpha_range
    # Sigmoid-like mapping: vel=0 → alpha_max, vel large → alpha_min
    alpha_p = alpha_max - (alpha_max - alpha_min) * np.tanh(vel * 2.0)
    return float(np.clip(alpha_p, alpha_min, alpha_max))
