"""Stage 5 — Per-frame object pose estimation via ICP 3D registration.

For each frame, aligns the fixed TripoSR mesh (from Stage 4) to the MoGe
depth point cloud of the segmented object using point-to-plane ICP, trying
10 canonical initial orientations and keeping the best-fit result.  This
gives accurate rotation AND metric position from 3D geometry directly,
generalising across object shapes without image-specific tuning.

For a STATIC output (our goal) we use all per-frame poses to compute a single
consensus pose — the geodesic mean rotation + arithmetic mean translation
across frames (see Stage 6's _consensus_pose).
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectPoseSequence, PipelineData
from models.guided_diffusion import GuidedDiffusionTracker
from utils.geometry import rotation_velocity_2d


class ObjectPoseEstimationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tracker = GuidedDiffusionTracker()

    def run(self, data: PipelineData) -> PipelineData:
        shape = data.object_mesh  # fixed x̄_s from Stage 4
        anchor_rot = shape.canonical_rot
        anchor_trans = shape.canonical_trans

        # Compute hand grip axis from MANO vertices.
        # normalize(fingertip_centroid - hand_centroid) gives the wrist→fingertips
        # direction, which is the axis any elongated held object is aligned with.
        # Direction is scale-invariant, so MANO local-space vertices work fine.
        FINGERTIP_IDX = [745, 317, 444, 556, 673]
        anchor_hand = next(
            r for r in data.hand_results if r.frame_index == data.anchor_index
        )
        hand_center = anchor_hand.vertices.mean(0)
        fingertip_centroid = anchor_hand.vertices[FINGERTIP_IDX].mean(0)
        _grip = fingertip_centroid - hand_center
        grip_norm = np.linalg.norm(_grip)
        hand_grip_axis = (_grip / grip_norm).astype(np.float32) if grip_norm > 1e-8 else None

        rots: list[np.ndarray] = []
        trans: list[np.ndarray] = []
        alpha_ps: list[float] = []

        prev_rot = anchor_rot.copy()
        prev_trans = anchor_trans.copy()

        frames = data.frames
        masks = data.object_seg.masks

        for frame in frames:
            if frame.index == data.anchor_index:
                R, t = self.tracker.track_frame(
                    image=frame.image,
                    mask=masks[frame.index],
                    fixed_shape_vertices=shape.vertices,
                    fixed_shape_faces=shape.faces,
                    prev_rot=anchor_rot,
                    prev_trans=anchor_trans,
                    alpha_p=0.0,
                    depth=data.depth_maps.get(frame.index),
                    K=data.camera_intrinsics,
                    hand_grip_axis=hand_grip_axis,
                )
                rots.append(R)
                trans.append(t)
                alpha_ps.append(0.0)
                prev_rot, prev_trans = R, t
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
                depth=data.depth_maps.get(frame.index),
                K=data.camera_intrinsics,
                hand_grip_axis=hand_grip_axis,
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
