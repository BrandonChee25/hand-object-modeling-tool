"""FoundationPose wrapper — 6DoF object pose estimation and tracking.

FoundationPose estimates the SE(3) object-in-camera pose from RGBD frames
and a known 3D mesh without requiring any category-specific training.

Install:
  git clone https://github.com/NVlabs/FoundationPose
  Follow the README (conda env, build C++ extensions, download weights).
  Set  foundation_pose_dir: /path/to/FoundationPose  in config.

Weights live under foundation_pose_dir/weights/ (auto-discovered by
ScorePredictor / PoseRefinePredictor via their own relative-path logic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


class FoundationPoseModel:
    """Wraps FoundationPose for per-frame 6DoF pose estimation + tracking."""

    def __init__(
        self,
        foundation_pose_dir: str | Path,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        debug_dir: str | Path | None = None,
    ):
        self.fp_dir = Path(foundation_pose_dir)
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.debug_dir = str(debug_dir) if debug_dir else str(self.fp_dir / "debug")

    def _ensure_on_path(self) -> None:
        fp_str = str(self.fp_dir)
        if fp_str not in sys.path:
            sys.path.insert(0, fp_str)

    def _build_estimator(self, mesh: trimesh.Trimesh):
        """Import FoundationPose and build an estimator for the given mesh."""
        self._ensure_on_path()
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        return FoundationPose(
            model_pts=mesh.vertices.astype(np.float64),
            model_normals=mesh.vertex_normals.astype(np.float64),
            symmetry_tfs=None,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug=self.debug,
            debug_dir=self.debug_dir,
        )

    def estimate_poses(
        self,
        mesh_verts: np.ndarray,
        mesh_faces: np.ndarray,
        K: np.ndarray,
        images: list[np.ndarray],
        depths: list[np.ndarray | None],
        masks: list[np.ndarray],
        anchor_index: int,
    ) -> list[np.ndarray]:
        """Estimate per-frame 4×4 ob_in_cam homogeneous transforms.

        Registers on the anchor frame, tracks forward to the last frame, then
        re-registers and tracks backward to frame 0.  Frames without a depth
        map inherit the nearest frame's pose.

        Returns a list of (4, 4) float64 arrays, one per input frame.
        """
        if depths[anchor_index] is None:
            raise RuntimeError("[fp] No depth available for anchor frame.")

        mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces, process=False)
        est = self._build_estimator(mesh)
        n = len(images)
        poses: list[np.ndarray | None] = [None] * n
        K64 = K.astype(np.float64)

        # FoundationPose expects ob_mask as uint8 0/255, not 0/1.
        def _u8(m: np.ndarray) -> np.ndarray:
            return (m.astype(np.uint8) * 255)

        # --- 1. Initial pose estimation on anchor frame ---
        print(f"[fp] registering on anchor frame {anchor_index}")
        anchor_pose = est.register(
            K=K64,
            rgb=images[anchor_index],
            depth=depths[anchor_index].astype(np.float64),
            ob_mask=_u8(masks[anchor_index]),
            iteration=self.est_refine_iter,
        )
        poses[anchor_index] = np.array(anchor_pose, dtype=np.float64)

        # --- 2. Forward pass ---
        for i in range(anchor_index + 1, n):
            d = depths[i]
            if d is None:
                poses[i] = poses[i - 1]
                continue
            pose_i = est.track_one(
                rgb=images[i],
                depth=d.astype(np.float64),
                K=K64,
                iteration=self.track_refine_iter,
            )
            poses[i] = np.array(pose_i, dtype=np.float64)
            print(f"[fp] tracked fwd frame {i}")

        # --- 3. Backward pass: re-register at anchor, then track in reverse ---
        if anchor_index > 0:
            est.register(
                K=K64,
                rgb=images[anchor_index],
                depth=depths[anchor_index].astype(np.float64),
                ob_mask=_u8(masks[anchor_index]),
                iteration=self.est_refine_iter,
            )
            for i in range(anchor_index - 1, -1, -1):
                d = depths[i]
                if d is None:
                    poses[i] = poses[i + 1]
                    continue
                pose_i = est.track_one(
                    rgb=images[i],
                    depth=d.astype(np.float64),
                    K=K64,
                    iteration=self.track_refine_iter,
                )
                poses[i] = np.array(pose_i, dtype=np.float64)
                print(f"[fp] tracked bwd frame {i}")

        # Fill any remaining None slots (no-depth frames at boundaries).
        anchor_pose_arr = poses[anchor_index]
        for i in range(n):
            if poses[i] is None:
                poses[i] = anchor_pose_arr

        return [p for p in poses]
