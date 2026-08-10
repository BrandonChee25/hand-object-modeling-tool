"""Stage 7 — Export static 3D meshes + per-frame trajectory.

Output layout:
    <out_dir>/
        hand.obj
        hand.mtl
        object.obj
        object.mtl
        scene.glb          # anchor-frame hand + object combined
        trajectory.npz     # full per-frame trajectory for the whole video

trajectory.npz arrays
---------------------
    frame_indices         (T,)        int32   — original frame index in source video
    object_rots           (T, 3, 3)   float32 — object rotation (WiLoR-bound; FP fallback)
    object_trans          (T, 3)      float32 — object translation in camera space (metres)
    hand_global_rot       (T, 3, 3)   float32 — MANO global rotation in camera space
    hand_translation      (T, 3)      float32 — MANO root translation in camera space
    hand_mano_pose        (T, 48)     float32 — MANO axis-angle pose params (theta)
    hand_mano_shape       (10,)       float32 — MANO shape betas (fixed across frames)
    hand_vertices         (T, 778, 3) float32 — MANO mesh vertices (WiLoR local space)
    hand_keypoints_3d     (T, 21, 3)  float32 — hand joints (WiLoR local space)
    camera_intrinsics     (3, 3)      float32 — K matrix used for depth/pose stages
    object_mesh_vertices  (V, 3)      float32 — canonical TripoSR mesh (object space)
    object_mesh_faces     (F, 3)      int32   — canonical TripoSR mesh faces
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import trimesh

from pipeline.data import PipelineData


class ExportStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, data: PipelineData, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = data.aligned_scene

        hand_mesh = trimesh.Trimesh(
            vertices=np.nan_to_num(scene.hand_vertices),
            faces=scene.hand_faces,
            process=False,
        )
        object_mesh = trimesh.Trimesh(
            vertices=np.nan_to_num(scene.object_vertices),
            faces=scene.object_faces,
            process=False,
        )

        _assign_vertex_colors(hand_mesh, color=(210, 180, 140, 255))   # tan
        _assign_vertex_colors(object_mesh, color=(100, 149, 237, 255)) # cornflower blue

        hand_mesh.export(output_dir / "hand.obj")
        object_mesh.export(output_dir / "object.obj")

        combined = trimesh.Scene({
            "hand": hand_mesh,
            "object": object_mesh,
        })
        combined.export(output_dir / "scene.glb")

        _save_trajectory(data, output_dir)

        return output_dir


def _save_trajectory(data: PipelineData, output_dir: Path) -> None:
    """Pack all per-frame hand and object data into trajectory.npz."""

    # --- frame indices ---
    frame_indices = np.array(
        [f.index for f in data.frames], dtype=np.int32
    )
    T = len(frame_indices)

    # --- object pose trajectory from Stage 5 (FP) ---
    object_rots  = np.stack(data.object_poses.rots).astype(np.float32)   # (T, 3, 3)
    object_trans = np.stack(data.object_poses.trans).astype(np.float32)  # (T, 3)


    # --- hand trajectory from Stage 2, ordered by frame index ---
    hand_by_frame = {r.frame_index: r for r in data.hand_results}

    hand_global_rot   = np.zeros((T, 3, 3), dtype=np.float32)
    hand_translation  = np.zeros((T, 3),    dtype=np.float32)
    hand_mano_pose    = np.zeros((T, 48),   dtype=np.float32)
    hand_vertices     = np.zeros((T, 778, 3), dtype=np.float32)
    hand_keypoints_3d = np.zeros((T, 21, 3),  dtype=np.float32)

    for i, fidx in enumerate(frame_indices):
        if fidx in hand_by_frame:
            r = hand_by_frame[fidx]
            hand_global_rot[i]   = r.global_rot
            hand_translation[i]  = r.translation
            hand_mano_pose[i]    = r.mano_pose
            hand_vertices[i]     = r.vertices
            hand_keypoints_3d[i] = r.keypoints_3d

    # --- WiLoR-bound object rotation ---
    # For grasped objects the hand's wrist rotation (WiLoR) is more stable than
    # FP's per-frame rotation estimate.  Compute a fixed canonical offset at the
    # seed frame: R_cano = R_wilor_seed.T @ R_fp_seed.  At every subsequent frame:
    # R_object = R_wilor_t @ R_cano.  For gaps where WiLoR has no detection,
    # SLERP between the nearest WiLoR-bound rotations rather than dropping in the
    # raw FP rotation (which is unrelated and causes systematic orientation jumps).
    from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp

    seed_idx  = data.object_seg.anchor_frame_index
    seed_fidx = int(frame_indices[seed_idx])
    seed_hand = hand_by_frame.get(seed_fidx)

    if seed_hand is not None:
        R_fp_seed    = np.array(data.object_poses.rots[seed_idx], dtype=np.float64)
        R_wilor_seed = seed_hand.global_rot.astype(np.float64)
        R_cano       = R_wilor_seed.T @ R_fp_seed

        wilor_mask = np.zeros(T, dtype=bool)
        for i, fidx in enumerate(frame_indices):
            hr = hand_by_frame.get(int(fidx))
            if hr is not None:
                object_rots[i] = (hr.global_rot.astype(np.float64) @ R_cano).astype(np.float32)
                wilor_mask[i]  = True

        n_wilor = int(wilor_mask.sum())
        print(f"[s7] WiLoR-bound rotation: {n_wilor}/{T} frames from WiLoR")

        # Fill FP-fallback gaps by interpolating between surrounding WiLoR frames.
        if n_wilor > 0 and n_wilor < T:
            wilor_idxs = np.where(wilor_mask)[0]
            for i in range(T):
                if wilor_mask[i]:
                    continue
                before = wilor_idxs[wilor_idxs < i]
                after  = wilor_idxs[wilor_idxs > i]
                if len(before) == 0:
                    object_rots[i] = object_rots[after[0]]
                elif len(after) == 0:
                    object_rots[i] = object_rots[before[-1]]
                else:
                    i0, i1 = int(before[-1]), int(after[0])
                    t = (i - i0) / (i1 - i0)
                    r0 = _Rot.from_matrix(object_rots[i0].astype(np.float64))
                    r1 = _Rot.from_matrix(object_rots[i1].astype(np.float64))
                    slerp = _Slerp([0.0, 1.0], _Rot.concatenate([r0, r1]))
                    object_rots[i] = slerp([t]).as_matrix()[0].astype(np.float32)
            print(f"[s7] SLERP-filled {T - n_wilor} FP-fallback frames")
    else:
        print(f"[s7] no WiLoR at seed frame {seed_fidx}; keeping FP rotations")

    # MANO shape is person-specific and fixed across frames; take from anchor.
    anchor_hand = hand_by_frame.get(data.anchor_index)
    hand_mano_shape = (
        anchor_hand.mano_shape.astype(np.float32)
        if anchor_hand is not None
        else np.zeros(10, dtype=np.float32)
    )

    # --- camera intrinsics ---
    K = (data.camera_intrinsics.astype(np.float32)
         if data.camera_intrinsics is not None
         else np.eye(3, dtype=np.float32))

    # --- canonical object mesh ---
    obj_verts = data.object_mesh.vertices.astype(np.float32)
    obj_faces = data.object_mesh.faces.astype(np.int32)

    # --- metric-scaled, origin-centred object mesh for correct viewer placement ---
    # FP poses assume the mesh is centred at origin and scaled to metres, so applying
    # (R, t) to this mesh places the object correctly in camera space.
    seed_idx = data.object_seg.anchor_frame_index
    from pipeline.stages.s5_object_pose import _scale_mesh_to_metric
    depth_seed = data.depth_maps.get(seed_idx, data.depth_map)
    obj_verts_metric, _ = _scale_mesh_to_metric(
        data.object_mesh.vertices,
        data.object_mesh.faces,
        data.object_seg.masks[seed_idx],
        depth_seed,
        data.camera_intrinsics,
    )

    # --- Stage 6 aligned hand vertices (anchor frame, metric camera space) ---
    aligned_hand_verts = (
        data.aligned_scene.hand_vertices.astype(np.float32)
        if data.aligned_scene is not None
        else np.zeros((778, 3), dtype=np.float32)
    )

    import utils.geometry as _geom

    out_path = output_dir / "trajectory.npz"
    np.savez_compressed(
        out_path,
        frame_indices=frame_indices,
        object_rots=object_rots,
        object_trans=object_trans,
        hand_global_rot=hand_global_rot,
        hand_translation=hand_translation,
        hand_mano_pose=hand_mano_pose,
        hand_mano_shape=hand_mano_shape,
        hand_vertices=hand_vertices,
        hand_keypoints_3d=hand_keypoints_3d,
        camera_intrinsics=K,
        object_mesh_vertices=obj_verts,
        object_mesh_faces=obj_faces,
        object_mesh_verts_metric=obj_verts_metric,
        aligned_hand_verts=aligned_hand_verts,
        hand_mesh_faces=_geom.MANO_FACES.astype(np.int32),
        # Must match the seed frame used for Stage 6 hand alignment — the viewer
        # zeroes the FP delta at this frame so hand and object are co-located there.
        anchor_frame_idx=np.int32(seed_idx),
    )
    print(f"[s7] saved trajectory.npz  ({T} frames, {out_path.stat().st_size // 1024} KB)")


def _assign_vertex_colors(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> None:
    mesh.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
