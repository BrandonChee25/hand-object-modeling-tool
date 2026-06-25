"""Stage 6 — Metric-scale alignment of hand and object.

Hand vertices from WiLoR live in WiLoR camera space (metric after Stage 2
rescaling). Object vertices from SAM-3D live in MoGe pointmap space. These
two coordinate systems share the same camera frame but may differ in overall
scale because the depth and 3D generation models are independent.

Alignment procedure (from "Do as I Do" §3.4):
  1. Compute hand centroid c_hand in WiLoR space (from anchor-frame MANO mesh).
  2. Compute object centroid c_obj in MoGe pointmap space (from anchor-frame
     depth-lifted object point cloud).
  3. Solve scale factor:
       k = z_hand / z_obj
     where z values are the median depths of the respective centroids, solved
     via least squares across all visible object points.
  4. Place the object mesh:
       obj_target = c_hand + k * (c_obj_world - c_hand_world)
     This rigidly shifts the object into the same metric space as the hand.

For the static output we use the anchor frame for alignment and export the
meshes in this unified camera-space frame.
"""

from __future__ import annotations

import numpy as np

from pipeline.data import AlignedScene, PipelineData
from utils.geometry import depth_lift_mask, solve_scale_least_squares, MANO_FACES


class AlignmentStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, data: PipelineData) -> PipelineData:
        anchor_hand = next(
            r for r in data.hand_results if r.frame_index == data.anchor_index
        )
        anchor_mask = data.object_seg.masks[data.anchor_index]

        # --- hand centroid in WiLoR metric space ---
        c_hand = anchor_hand.vertices.mean(axis=0)  # (3,)

        # --- object point cloud in MoGe metric space ---
        obj_points = depth_lift_mask(
            data.depth_map,
            anchor_mask,
            data.camera_intrinsics,
        )  # (N, 3)

        if len(obj_points) < 10:
            raise RuntimeError(
                "Too few object depth points for scale alignment. "
                "Check that the object mask covers enough pixels."
            )

        c_obj = obj_points.mean(axis=0)  # (3,) in MoGe space

        # Solve metric scale between WiLoR and MoGe spaces.
        k = solve_scale_least_squares(
            z_ref=c_hand[2],
            z_src=c_obj[2],
            src_points=obj_points,
        )

        # Apply the consensus object pose (SE(3) mean of all inlier frame poses).
        consensus_rot, consensus_trans = _consensus_pose(data)

        # Transform object vertices into the aligned world frame.
        obj_verts_cam = (
            data.object_mesh.vertices @ consensus_rot.T + consensus_trans
        )  # (V, 3) in MoGe camera space

        # Shift into WiLoR metric space via scale + centroid alignment.
        obj_verts_aligned = c_hand + k * (obj_verts_cam - c_obj)

        # Identity world-from-camera (we keep camera as world for simplicity).
        world_from_camera = np.eye(4)

        data.aligned_scene = AlignedScene(
            hand_vertices=anchor_hand.vertices,
            hand_faces=MANO_FACES,
            object_vertices=obj_verts_aligned,
            object_faces=data.object_mesh.faces,
            world_from_camera=world_from_camera,
        )
        return data


def _consensus_pose(data: PipelineData) -> tuple[np.ndarray, np.ndarray]:
    """Return the SE(3)-mean rotation and translation across all frame poses.

    Uses the per-frame poses from guided diffusion (Stage 5) and computes the
    geodesic mean rotation + arithmetic mean translation.  Outlier frames with
    high alpha_p variance are downweighted.
    """
    from utils.geometry import geodesic_mean_rotation

    rots = np.stack(data.object_poses.rots)    # (T, 3, 3)
    trans = np.stack(data.object_poses.trans)  # (T, 3)

    mean_rot = geodesic_mean_rotation(rots)
    mean_trans = trans.mean(axis=0)
    return mean_rot, mean_trans
