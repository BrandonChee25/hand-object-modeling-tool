"""Stage 6 — Metric-scale alignment of hand and object.

Both hand and object are placed in MoGe's metric camera space:
  - Hand: WiLoR outputs vertices and keypoints already in camera space (global
    rotation + translation applied), but at WiLoR's own depth scale.  A metric
    scale factor is recovered by matching WiLoR's fingertip depths to MoGe
    depth values, then the mesh is shifted so its wrist→fingertip grip point
    coincides with FP's object centroid estimate.
  - Object: scaled to match its mask's apparent size at the depth-lifted
    object centroid, oriented by Stage 5's pose estimate, placed at t_fp.
  - Contact: any residual hand-mesh penetration is resolved by a small
    outward push (see _resolve_penetration).
"""

from __future__ import annotations

import numpy as np

from pipeline.data import AlignedScene, PipelineData
from utils.geometry import depth_lift_mask
import utils.geometry as _geom


class AlignmentStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, data: PipelineData) -> PipelineData:
        # Use the segmentation seed frame as the single reference for everything:
        # FP is registered (most accurate) there, the object mask is from there,
        # and Stage 7 anchors the viewer's hand delta at that frame.
        seed_idx   = data.object_seg.anchor_frame_index
        anchor_mask = data.object_seg.masks[seed_idx]
        seed_depth  = data.depth_maps.get(seed_idx, data.depth_map)

        # WiLoR result at the seed frame; fall back to the hand anchor if missing.
        hand_results_by_frame = {r.frame_index: r for r in data.hand_results}
        anchor_hand = (
            hand_results_by_frame.get(seed_idx)
            or hand_results_by_frame.get(data.anchor_index)
        )
        if anchor_hand is None:
            raise RuntimeError("No WiLoR hand result found at seed or anchor frame.")

        K = data.camera_intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H_img, W_img = seed_depth.shape

        # --- hand in metric camera space ---
        # WiLoR outputs pred_vertices / pred_keypoints_3d in MANO local space:
        # global orientation is already applied, but the camera-space translation
        # is NOT included.  pred_cam_t_full (stored as anchor_hand.translation) is
        # the metric camera-space position of the MANO root joint — adding it gives
        # the full camera-space positions.
        FINGERTIP_KP_IDX = [4, 8, 12, 16, 20]
        kps        = anchor_hand.keypoints_3d               # (21, 3) MANO local
        verts      = anchor_hand.vertices                   # (778, 3) MANO local
        mano_trans = anchor_hand.translation.astype(np.float32)  # (3,) metric camera

        kps_metric   = (kps   + mano_trans).astype(np.float32)  # (21, 3)  camera space
        verts_metric = (verts + mano_trans).astype(np.float32)  # (778, 3) camera space

        wrist_metric         = kps_metric[0]
        finger_center_metric = kps_metric[FINGERTIP_KP_IDX].mean(axis=0)
        finger_dist          = float(np.linalg.norm(finger_center_metric - wrist_metric))
        print(f"[s6] mano_trans={mano_trans.tolist()}")
        print(f"[s6] wrist_cam={wrist_metric.tolist()}")
        print(f"[s6] finger_center_cam={finger_center_metric.tolist()}  finger_dist={finger_dist:.3f}m")

        grip_pos           = self.cfg.get("grip_position", 0.6)
        grip_center_metric = wrist_metric + grip_pos * (finger_center_metric - wrist_metric)

        # --- object point cloud in MoGe metric space ---
        obj_points = depth_lift_mask(seed_depth, anchor_mask, data.camera_intrinsics)
        if len(obj_points) < 10 or not np.all(np.isfinite(obj_points)):
            raise RuntimeError(
                "Too few or invalid object depth points for scale alignment. "
                "Check that the object mask covers enough pixels and the depth map is valid."
            )

        c_obj = obj_points.mean(axis=0)

        # --- scale canonical mesh to metric size ---
        ys_mask, xs_mask = np.where(anchor_mask)
        mask_diag_px    = np.sqrt((xs_mask.max() - xs_mask.min()) ** 2 +
                                  (ys_mask.max() - ys_mask.min()) ** 2)
        obj_metric_diag = mask_diag_px * c_obj[2] / K[0, 0]
        canon_verts     = data.object_mesh.vertices
        canon_diag      = np.linalg.norm(canon_verts.max(0) - canon_verts.min(0))
        obj_scale       = obj_metric_diag / max(canon_diag, 1e-6)

        fp_path = (
            data.object_poses is not None
            and all(a == 1.0 for a in data.object_poses.alpha_p_values)
        )

        if fp_path:
            R_aligned = np.array(data.object_poses.rots[seed_idx], dtype=np.float64)
            t_fp      = np.array(data.object_poses.trans[seed_idx], dtype=np.float64)
        else:
            R_aligned, t_fp = _consensus_pose(data)

        fp_trans_valid = fp_path and float(np.linalg.norm(t_fp)) > 0.05

        print(f"[s6] c_obj (depth-lift)={c_obj.tolist()}")
        if fp_trans_valid:
            print(f"[s6] t_fp (FP)={t_fp.tolist()}  |c_obj-t_fp|="
                  f"{float(np.linalg.norm(c_obj - t_fp.astype(np.float32))):.3f}m")

        # Shift hand so the grip centre (grip_pos fraction from wrist to fingertips)
        # coincides with FP's object centroid.
        if fp_trans_valid:
            grip_shift        = t_fp.astype(np.float32) - grip_center_metric
            hand_verts_cam    = verts_metric + grip_shift
            finger_center_cam = finger_center_metric + grip_shift
            print(f"[s6] grip_centre={grip_center_metric.tolist()}")
            print(f"[s6] grip_shift={grip_shift.tolist()}  |shift|={float(np.linalg.norm(grip_shift)):.3f}m")
        else:
            hand_verts_cam    = verts_metric
            finger_center_cam = finger_center_metric
            print(f"[s6] FP translation invalid, using WiLoR position directly")

        from scipy.spatial.transform import Rotation as _Rot
        euler = _Rot.from_matrix(R_aligned).as_euler("xyz", degrees=True)
        print(f"[s6] FP R_seed Euler (xyz deg): {euler.tolist()}")

        obj_verts_posed   = canon_verts @ R_aligned.T
        canon_center      = obj_verts_posed.mean(axis=0)
        obj_center_cam    = t_fp if fp_trans_valid else grip_center_metric
        obj_verts_aligned = obj_center_cam + obj_scale * (obj_verts_posed - canon_center)

        print(f"[s6] obj_scale={obj_scale:.4f}  obj_center={obj_center_cam.tolist()}")

        obj_verts_pre     = obj_verts_aligned.copy()
        obj_verts_aligned = _resolve_penetration(
            hand_verts_cam,
            _geom.MANO_FACES,
            obj_verts_aligned,
            fallback_dir=finger_center_cam - hand_verts_cam.mean(0),
            max_push=self.cfg.get("max_contact_push_m", 0.05),
        )
        push_delta = obj_verts_aligned.mean(0) - obj_verts_pre.mean(0)
        print(f"[s6] penetration_push |push|={float(np.linalg.norm(push_delta)):.3f}m")

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


def _resolve_penetration(
    hand_verts: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts: np.ndarray,
    fallback_dir: np.ndarray,
    max_push: float = 0.05,
    max_iters: int = 30,
) -> np.ndarray:
    """Rigidly translate obj_verts out of the hand mesh.

    A single fixed push axis can't resolve penetration for an elongated object
    lying diagonally across a curved hand: translating it can pull one end
    clear while driving the other end deeper in. Instead, each iteration finds
    the currently-penetrating vertices, computes the average vector from each
    to its nearest point on the hand surface (the locally shortest way out),
    and takes a small step in that direction — so the push direction adapts
    as different parts of the object clear the hand mesh.
    """
    import trimesh

    try:
        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces, process=False)
    except Exception:
        return obj_verts

    fb_norm = np.linalg.norm(fallback_dir)
    fallback_dir = fallback_dir / fb_norm if fb_norm > 1e-8 else np.array([0.0, 0.0, -1.0])

    verts = obj_verts.copy()
    step = max_push / max_iters

    try:
        for _ in range(max_iters):
            # Use proximity + surface normals instead of mesh.contains() so this
            # works on the MANO mesh, which is open at the wrist (not watertight).
            closest, _, tri_ids = trimesh.proximity.closest_point(hand_mesh, verts)
            face_normals = hand_mesh.face_normals[tri_ids]
            # Vector from nearest surface point to each object vertex.
            surf_to_vert = verts - closest
            # A vertex is inside the hand when it points opposite to the outward normal.
            inside = (surf_to_vert * face_normals).sum(axis=1) < 0

            if not inside.any():
                return verts

            # Push direction: average of (surface_point - inside_vertex), i.e. toward exit.
            exit_vecs = closest[inside] - verts[inside]
            direction = exit_vecs.mean(axis=0)
            dir_norm = np.linalg.norm(direction)
            direction = direction / dir_norm if dir_norm > 1e-8 else fallback_dir

            verts = verts + step * direction
        return verts
    except Exception:
        # Geometry-query failure: keep original placement.
        return obj_verts


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
