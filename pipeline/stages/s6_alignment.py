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
from utils.geometry import depth_lift_mask, solve_scale_least_squares
import utils.geometry as _geom


class AlignmentStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, data: PipelineData) -> PipelineData:
        anchor_hand = next(
            r for r in data.hand_results if r.frame_index == data.anchor_index
        )
        anchor_mask = data.object_seg.masks[data.anchor_index]

        # --- hand position and scale from MoGe depth + YOLO bbox ---
        # WiLoR's translation is over-scaled; use MoGe depth at the hand bbox centre
        # instead so that hand and object live in the same metric coordinate system.
        anchor_frame = data.frames[data.anchor_index]
        K = data.camera_intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H_img, W_img = data.depth_map.shape

        if anchor_frame.hand_bbox is not None:
            x1, y1, x2, y2 = anchor_frame.hand_bbox.astype(int)
            hx = int(np.clip((x1 + x2) / 2, 0, W_img - 1))
            hy = int(np.clip((y1 + y2) / 2, 0, H_img - 1))
        else:
            hx, hy = W_img // 2, H_img // 2

        # Sample depth in a small patch around the bbox centre.
        py0, py1 = max(0, hy - 20), min(H_img, hy + 20)
        px0, px1 = max(0, hx - 20), min(W_img, hx + 20)
        patch = data.depth_map[py0:py1, px0:px1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        hand_depth = float(np.median(valid)) if len(valid) > 0 else float(data.depth_map[hy, hx])

        # Backproject bbox centre to 3D.
        c_hand = np.array([
            (hx - cx) * hand_depth / fx,
            (hy - cy) * hand_depth / fy,
            hand_depth,
        ], dtype=np.float32)

        # Scale MANO vertices to metric using bbox apparent size at that depth.
        if anchor_frame.hand_bbox is not None:
            bbox_diag_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            hand_metric_size = bbox_diag_px * hand_depth / fx
        else:
            hand_metric_size = 0.2  # fallback: 20 cm

        mano_span = np.linalg.norm(
            anchor_hand.vertices.max(0) - anchor_hand.vertices.min(0)
        )
        hand_scale = hand_metric_size / max(mano_span, 1e-6)
        mano_center = anchor_hand.vertices.mean(0)
        hand_verts_cam = c_hand + hand_scale * (anchor_hand.vertices - mano_center)

        # --- object point cloud in MoGe metric space ---
        obj_points = depth_lift_mask(
            data.depth_map,
            anchor_mask,
            data.camera_intrinsics,
        )  # (N, 3)

        if len(obj_points) < 10 or not np.all(np.isfinite(obj_points)):
            raise RuntimeError(
                "Too few or invalid object depth points for scale alignment. "
                "Check that the object mask covers enough pixels and the depth map is valid."
            )

        c_obj = obj_points.mean(axis=0)  # (3,) in MoGe space

        # --- scale TripoSR canonical mesh to metric size ---
        # Estimate object metric diameter from its mask pixel extent and MoGe depth.
        ys_mask, xs_mask = np.where(anchor_mask)
        mask_diag_px = np.sqrt((xs_mask.max() - xs_mask.min()) ** 2 +
                               (ys_mask.max() - ys_mask.min()) ** 2)
        obj_metric_diag = mask_diag_px * c_obj[2] / K[0, 0]  # fx used as reference

        canon_verts = data.object_mesh.vertices
        canon_diag = np.linalg.norm(canon_verts.max(0) - canon_verts.min(0))
        obj_scale = obj_metric_diag / max(canon_diag, 1e-6)

        # Grip centre = blend between palm (c_hand) and fingertip centroid.
        # grip_position=0 → palm centre, 1 → fingertips. Default 0.6 puts the
        # grip at the proximal-phalanx region where the hand closes around the object.
        FINGERTIP_IDX = [745, 317, 444, 556, 673]  # thumb to pinky in MANO topology
        finger_center_local = anchor_hand.vertices[FINGERTIP_IDX].mean(axis=0)
        finger_center_cam = c_hand + hand_scale * (finger_center_local - mano_center)
        grip_pos = self.cfg.get("grip_position", 0.6)
        grip_center_cam = c_hand + grip_pos * (finger_center_cam - c_hand)

        # Align the object's longest axis with the wrist→fingertip direction.
        hand_axis = finger_center_cam - c_hand
        hand_axis = hand_axis / (np.linalg.norm(hand_axis) + 1e-8)
        R_aligned = _align_primary_axis(canon_verts, hand_axis, obj_points)

        # Optional flip (180° around long axis) and roll correction from config.
        if self.cfg.get("flip_object", False):
            # Rotate 180° around the hand axis.
            K = np.array([[0, -hand_axis[2], hand_axis[1]],
                          [hand_axis[2], 0, -hand_axis[0]],
                          [-hand_axis[1], hand_axis[0], 0]])
            R_flip = np.eye(3) + 2 * K @ K  # Rodrigues for angle=π
            R_aligned = R_flip @ R_aligned

        roll_deg = self.cfg.get("object_roll_degrees", 0.0)
        if abs(roll_deg) > 1e-3:
            angle = np.deg2rad(roll_deg)
            K = np.array([[0, -hand_axis[2], hand_axis[1]],
                          [hand_axis[2], 0, -hand_axis[0]],
                          [-hand_axis[1], hand_axis[0], 0]])
            R_roll = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R_aligned = R_roll @ R_aligned

        # Apply rotation then scale and place at grip centre.
        obj_verts_posed = canon_verts @ R_aligned.T
        canon_center = obj_verts_posed.mean(axis=0)
        obj_verts_aligned = grip_center_cam + obj_scale * (obj_verts_posed - canon_center)


        # Identity world-from-camera (we keep camera as world for simplicity).
        world_from_camera = np.eye(4)

        data.aligned_scene = AlignedScene(
            hand_vertices=hand_verts_cam,
            hand_faces=_geom.MANO_FACES,
            object_vertices=obj_verts_aligned,
            object_faces=data.object_mesh.faces,
            world_from_camera=world_from_camera,
        )
        return data


def _align_primary_axis(
    canon_verts: np.ndarray,
    target_axis: np.ndarray,
    obj_points: np.ndarray,
) -> np.ndarray:
    """Rotate the canonical mesh so its longest axis aligns with target_axis.

    The secondary axis is chosen to minimise deviation from the depth point
    cloud's secondary PCA direction (controls roll around the long axis).
    """
    # Primary axis of canonical mesh.
    c = canon_verts - canon_verts.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    canon_primary = Vt[0]

    # Rodrigues rotation: canon_primary → target_axis.
    v = np.cross(canon_primary, target_axis)
    cos_a = float(np.dot(canon_primary, target_axis))
    sin_a = np.linalg.norm(v)
    if sin_a > 1e-6:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + K + K @ K * ((1 - cos_a) / sin_a ** 2)
    else:
        R1 = np.eye(3) if cos_a > 0 else -np.eye(3)

    # Use depth cloud secondary PCA axis to constrain roll around target_axis.
    op = obj_points - obj_points.mean(0)
    _, _, Vt_d = np.linalg.svd(op, full_matrices=False)
    depth_secondary = Vt_d[1]
    depth_secondary -= np.dot(depth_secondary, target_axis) * target_axis
    depth_secondary /= np.linalg.norm(depth_secondary) + 1e-8

    # Secondary axis of canon mesh after R1.
    canon_secondary = R1 @ Vt[1]
    canon_secondary -= np.dot(canon_secondary, target_axis) * target_axis
    canon_secondary /= np.linalg.norm(canon_secondary) + 1e-8

    # Rodrigues rotation around target_axis to align secondary axes.
    v2 = np.cross(canon_secondary, depth_secondary)
    cos_a2 = float(np.clip(np.dot(canon_secondary, depth_secondary), -1, 1))
    sin_a2 = np.dot(v2, target_axis)
    angle2 = np.arctan2(sin_a2, cos_a2)
    K2 = np.array([[0, -target_axis[2], target_axis[1]],
                   [target_axis[2], 0, -target_axis[0]],
                   [-target_axis[1], target_axis[0], 0]])
    R2 = np.eye(3) + np.sin(angle2) * K2 + (1 - np.cos(angle2)) * K2 @ K2

    R = R2 @ R1

    # Enforce proper rotation.
    U, _, Vt_r = np.linalg.svd(R)
    R = U @ Vt_r
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt_r

    return R.astype(np.float32)


def _pca_rotation(obj_points: np.ndarray, canon_verts: np.ndarray) -> np.ndarray:
    """Estimate a rotation that aligns TripoSR canonical axes to the depth point cloud.

    Runs PCA on both the depth-lifted object point cloud and the canonical mesh,
    then solves for the rotation mapping one set of principal axes to the other.
    The camera-facing axis is forced to point toward the camera (+z in camera space).
    """
    def pca_axes(pts: np.ndarray) -> np.ndarray:
        c = pts - pts.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        return Vt  # rows are principal axes, descending variance

    depth_axes = pca_axes(obj_points)   # (3, 3) rows = axes in depth space
    canon_axes = pca_axes(canon_verts)  # (3, 3) rows = axes in canonical space

    # R maps canon frame to depth frame: depth_axes = R @ canon_axes
    R = depth_axes.T @ canon_axes

    # Enforce proper rotation (det = +1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Force the smallest-variance axis (surface normal) to face the camera (+z).
    if R[2, 2] < 0:
        R[:, 2] *= -1
        R[:, 0] *= -1  # keep det = +1

    return R.astype(np.float32)


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
