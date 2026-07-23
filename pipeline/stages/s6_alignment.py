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
        anchor_hand = next(
            r for r in data.hand_results if r.frame_index == data.anchor_index
        )
        # Object mask comes from the SAM-2 seed frame, not necessarily the hand anchor.
        seed_idx = data.object_seg.anchor_frame_index
        anchor_mask = data.object_seg.masks[seed_idx]

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
            # FoundationPose: use anchor frame pose directly.
            # Each frame's pose is in that frame's camera space, so averaging
            # across frames gives a meaningless translation when the camera moves.
            anchor_idx = data.object_seg.anchor_frame_index
            R_aligned = np.array(data.object_poses.rots[anchor_idx], dtype=np.float64)
            t_fp = np.array(data.object_poses.trans[anchor_idx], dtype=np.float64)
        else:
            R_aligned, t_fp = _consensus_pose(data)

        obj_verts_posed = canon_verts @ R_aligned.T
        canon_center = obj_verts_posed.mean(axis=0)

        fp_trans_valid = fp_path and float(np.linalg.norm(t_fp)) > 0.05
        obj_center_cam = grip_center_cam  # DEBUG: force grip centre
        print(f"[s6 debug] c_hand      = {c_hand.tolist()}")
        print(f"[s6 debug] grip_centre = {grip_center_cam.tolist()}")
        print(f"[s6 debug] c_obj(moge) = {c_obj.tolist()}")
        if fp_path:
            print(f"[s6 debug] t_fp(FP)    = {t_fp.tolist()}")
            print(f"[s6 debug] fp_valid={fp_trans_valid}  chosen={obj_center_cam.tolist()}")

        obj_verts_aligned = obj_center_cam + obj_scale * (obj_verts_posed - canon_center)

        # Push out any residual penetration into the hand mesh.
        obj_verts_aligned = _resolve_penetration(
            hand_verts_cam,
            _geom.MANO_FACES,
            obj_verts_aligned,
            fallback_dir=finger_center_cam - c_hand,
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
