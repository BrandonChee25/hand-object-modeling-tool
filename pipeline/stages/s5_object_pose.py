"""Stage 5 — 6DoF object pose estimation.

Primary path: FoundationPose (if `foundation_pose_dir` is set in config).
  Scales the canonical mesh to metric units using the object mask and MoGe
  depth, then runs FoundationPose's register + track_one pipeline to produce
  a 4×4 ob_in_cam pose for every frame.

Fallback path: DINOv2-based rotation search (GuidedDiffusionTracker).
  Runs a hand-constrained render-and-compare search on the anchor frame and
  broadcasts that single rotation to all frames.  Translation is filled from
  the depth-lifted object mask centroid.

Config keys (all optional):
  foundation_pose_dir  : path to FoundationPose clone (enables primary path)
  fp_est_refine_iter   : pose-hypothesis refinement iterations (default 5)
  fp_track_refine_iter : per-frame tracking refinement iterations (default 2)
  fp_debug             : FoundationPose debug level (default 0)
"""
from __future__ import annotations

import numpy as np

from pipeline.data import ObjectPoseSequence, PipelineData


class ObjectPoseEstimationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, data: PipelineData) -> PipelineData:
        fp_dir = self.cfg.get("foundation_pose_dir")
        if fp_dir:
            try:
                return self._run_foundation_pose(data, fp_dir)
            except ImportError as e:
                print(f"[s5] FoundationPose not importable ({e}) — falling back to DINOv2 search")
        return self._run_dinov2(data)

    # ------------------------------------------------------------------
    # Primary path — FoundationPose
    # ------------------------------------------------------------------

    def _run_foundation_pose(self, data: PipelineData, fp_dir: str) -> PipelineData:
        from models.foundation_pose_wrapper import FoundationPoseModel

        seed_idx = data.object_seg.anchor_frame_index
        K = data.camera_intrinsics

        mesh_verts, mesh_faces = _scale_mesh_to_metric(
            data.object_mesh.vertices,
            data.object_mesh.faces,
            data.object_seg.masks[seed_idx],
            data.depth_maps.get(seed_idx, data.depth_map),
            K,
        )
        span = (mesh_verts.max(0) - mesh_verts.min(0)).tolist()
        print(f"[s5] metric mesh span (m): {[f'{v:.3f}' for v in span]}")

        n = len(data.frames)
        images = [f.image for f in data.frames]
        depths = [data.depth_maps.get(f.index, data.depth_map) for f in data.frames]
        masks  = list(data.object_seg.masks)

        fp = FoundationPoseModel(
            foundation_pose_dir=fp_dir,
            est_refine_iter=self.cfg.get("fp_est_refine_iter", 5),
            track_refine_iter=self.cfg.get("fp_track_refine_iter", 2),
            debug=self.cfg.get("fp_debug", 0),
        )
        poses_4x4 = fp.estimate_poses(
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            K=K,
            images=images,
            depths=depths,
            masks=masks,
            anchor_index=seed_idx,
        )

        data.object_poses = ObjectPoseSequence(
            rots=[p[:3, :3].astype(np.float32) for p in poses_4x4],
            trans=[p[:3, 3].astype(np.float32) for p in poses_4x4],
            # Mark as FP output: Stage 6 reads this flag to use FP translation.
            alpha_p_values=[1.0] * n,
        )
        print(f"[s5] FoundationPose complete ({n} frames)")
        return data

    # ------------------------------------------------------------------
    # Fallback path — DINOv2 render-and-compare (anchor frame only)
    # ------------------------------------------------------------------

    def _run_dinov2(self, data: PipelineData) -> PipelineData:
        from models.guided_diffusion import GuidedDiffusionTracker

        seed_idx   = data.object_seg.anchor_frame_index
        seed_frame = data.frames[seed_idx]
        seed_mask  = data.object_seg.masks[seed_idx]
        depth      = data.depth_maps.get(seed_idx, data.depth_map)
        K          = data.camera_intrinsics

        hand_results_map = {r.frame_index: r for r in data.hand_results}
        hr = hand_results_map.get(seed_idx)
        hand_grip_axis: np.ndarray | None = None
        if hr is not None:
            kp = hr.keypoints_3d
            hand_grip_axis = kp[[4, 8, 12, 16, 20]].mean(0) - kp[0]

        tracker = GuidedDiffusionTracker(device=self.cfg.get("device", "cuda"))
        R, t = tracker.track_frame(
            image=seed_frame.image,
            mask=seed_mask,
            fixed_shape_vertices=data.object_mesh.vertices,
            fixed_shape_faces=data.object_mesh.faces,
            prev_rot=np.eye(3, dtype=np.float32),
            prev_trans=np.zeros(3, dtype=np.float32),
            alpha_p=0.0,
            depth=depth,
            K=K,
            hand_grip_axis=hand_grip_axis,
        )

        n = len(data.frames)
        data.object_poses = ObjectPoseSequence(
            rots=[R.copy() for _ in range(n)],
            trans=[t.copy() for _ in range(n)],
            alpha_p_values=[0.0] * n,
        )
        print("[s5] DINOv2 search complete (rotation broadcast to all frames)")
        return data


# ------------------------------------------------------------------
# Shared helper
# ------------------------------------------------------------------


def _scale_mesh_to_metric(
    canon_verts: np.ndarray,
    canon_faces: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray | None,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of the canonical mesh scaled to metric units (metres).

    The mesh is centred at the origin so that FoundationPose's output
    translation directly gives the object centroid in camera space.

    Scale = object metric diagonal / canonical mesh diagonal.
    Object metric diagonal = mask pixel span × depth / fx.
    This matches the same estimate used in Stage 6, keeping the two stages
    consistent.
    """
    from utils.geometry import depth_lift_mask

    centred = canon_verts - canon_verts.mean(0)

    if depth is None or not mask.any():
        return centred.astype(np.float32), canon_faces

    pts = depth_lift_mask(depth, mask, K)
    if len(pts) == 0:
        return centred.astype(np.float32), canon_faces

    depth_at_obj = float(pts[:, 2].mean())

    ys, xs = np.where(mask)
    mask_diag_px = float(
        np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
    )
    fx = float(K[0, 0])
    obj_metric_diag = mask_diag_px * depth_at_obj / fx

    canon_diag = float(np.linalg.norm(canon_verts.max(0) - canon_verts.min(0)))
    scale = obj_metric_diag / max(canon_diag, 1e-6)

    return (centred * scale).astype(np.float32), canon_faces
