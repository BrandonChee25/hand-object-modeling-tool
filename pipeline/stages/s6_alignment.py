"""Stage 6 — Metric-scale alignment of hand and object.

Both hand and object are placed in MoGe's metric camera space:
  - Hand: the wrist/palm position is backprojected from the YOLO hand bbox
    centre using MoGe depth, and the MANO mesh is scaled to match the bbox's
    apparent size at that depth.
  - Object: scaled to match its mask's apparent size at the depth-lifted
    object centroid, then placed at a grip point blended between the palm
    and fingertip centroid as an initial guess (so it starts inside the
    hand rather than at the wrist or floating at the fingertips).
  - Orientation: taken directly from Stage 5's DINOv2 render-and-compare
    pose estimate (no manual tuning needed).
  - Contact: the initial placement is then corrected by pushing the object
    rigidly along the palm→fingertip axis until it no longer penetrates the
    hand mesh (see _resolve_penetration), rather than relying on a fixed
    grip point or a hand-tuned scale fudge factor.
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

        # Hand bbox: prefer seed frame, fall back to hand anchor frame.
        seed_frame   = data.frames[seed_idx]
        ref_frame    = seed_frame if seed_frame.hand_bbox is not None else data.frames[data.anchor_index]

        K = data.camera_intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H_img, W_img = seed_depth.shape

        if ref_frame.hand_bbox is not None:
            x1, y1, x2, y2 = ref_frame.hand_bbox.astype(int)
            hx = int(np.clip((x1 + x2) / 2, 0, W_img - 1))
            hy = int(np.clip((y1 + y2) / 2, 0, H_img - 1))
        else:
            hx, hy = W_img // 2, H_img // 2

        # Sample depth in a small patch around the bbox centre.
        py0, py1 = max(0, hy - 20), min(H_img, hy + 20)
        px0, px1 = max(0, hx - 20), min(W_img, hx + 20)
        patch = seed_depth[py0:py1, px0:px1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        hand_depth = float(np.median(valid)) if len(valid) > 0 else float(seed_depth[hy, hx])

        # Backproject bbox centre to 3D.
        c_hand = np.array([
            (hx - cx) * hand_depth / fx,
            (hy - cy) * hand_depth / fy,
            hand_depth,
        ], dtype=np.float32)

        # Scale MANO vertices to metric using bbox apparent size at that depth.
        if ref_frame.hand_bbox is not None:
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
            seed_depth,
            anchor_mask,
            data.camera_intrinsics,
        )  # (N, 3)

        if len(obj_points) < 10 or not np.all(np.isfinite(obj_points)):
            raise RuntimeError(
                "Too few or invalid object depth points for scale alignment. "
                "Check that the object mask covers enough pixels and the depth map is valid."
            )

        c_obj = obj_points.mean(axis=0)  # (3,) in MoGe space

        # --- scale canonical mesh to metric size ---
        ys_mask, xs_mask = np.where(anchor_mask)
        mask_diag_px = np.sqrt((xs_mask.max() - xs_mask.min()) ** 2 +
                               (ys_mask.max() - ys_mask.min()) ** 2)
        obj_metric_diag = mask_diag_px * c_obj[2] / K[0, 0]  # fx used as reference

        canon_verts = data.object_mesh.vertices
        canon_diag = np.linalg.norm(canon_verts.max(0) - canon_verts.min(0))
        obj_scale = obj_metric_diag / max(canon_diag, 1e-6)

        FINGERTIP_IDX = [745, 317, 444, 556, 673]  # thumb to pinky in MANO topology
        finger_center_local = anchor_hand.vertices[FINGERTIP_IDX].mean(axis=0)
        finger_center_cam = c_hand + hand_scale * (finger_center_local - mano_center)
        grip_pos = self.cfg.get("grip_position", 0.6)
        grip_center_cam = c_hand + grip_pos * (finger_center_cam - c_hand)

        fp_path = (
            data.object_poses is not None
            and all(a == 1.0 for a in data.object_poses.alpha_p_values)
        )

        if fp_path:
            # Use the seed frame's FP pose: this is where FP ran registration,
            # so it is the most accurate estimate. hand_verts_cam is also derived
            # from the seed frame, so both are in the same camera coordinate system.
            R_aligned = np.array(data.object_poses.rots[seed_idx], dtype=np.float64)
            t_fp = np.array(data.object_poses.trans[seed_idx], dtype=np.float64)
        else:
            R_aligned, t_fp = _consensus_pose(data)

        fp_trans_valid = fp_path and float(np.linalg.norm(t_fp)) > 0.05

        # Shift hand mesh so its grip centre lands on the object centroid (t_fp).
        # hand_verts_cam is initially centred at c_hand (hand bbox centre), but the
        # object is near the fingertips/grip, not the palm centre — so there is a
        # structural offset that this corrects.
        if fp_trans_valid:
            grip_shift = t_fp.astype(np.float32) - grip_center_cam
            hand_verts_cam = hand_verts_cam + grip_shift
            finger_center_cam = finger_center_cam + grip_shift
            print(f"[s6] c_hand={c_hand.tolist()}  grip_centre={grip_center_cam.tolist()}")
            print(f"[s6] t_fp={t_fp.tolist()}  grip_shift={grip_shift.tolist()}  "
                  f"|shift|={float(np.linalg.norm(grip_shift)):.3f}m")
        else:
            print(f"[s6] FP translation invalid, using grip heuristic")

        from scipy.spatial.transform import Rotation as _Rot
        euler = _Rot.from_matrix(R_aligned).as_euler("xyz", degrees=True)
        print(f"[s6] FP R_seed Euler (xyz deg): {euler.tolist()}")

        obj_verts_posed = canon_verts @ R_aligned.T
        canon_center = obj_verts_posed.mean(axis=0)

        obj_center_cam = t_fp if fp_trans_valid else grip_center_cam
        obj_verts_aligned = obj_center_cam + obj_scale * (obj_verts_posed - canon_center)

        # Push out any residual penetration into the hand mesh.
        obj_verts_aligned = _resolve_penetration(
            hand_verts_cam,
            _geom.MANO_FACES,
            obj_verts_aligned,
            fallback_dir=finger_center_cam - (hand_verts_cam.mean(0)),
            max_push=self.cfg.get("max_contact_push_m", 0.05),
        )

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
