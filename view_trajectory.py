"""Interactive 3D viewer for trajectory.npz files.

Usage
-----
    python view_trajectory.py out/trajectory.npz
    python view_trajectory.py out/trajectory.npz --fps 10

Controls
--------
    N / Right arrow  — next frame
    P / Left arrow   — previous frame
    Space            — play / pause auto-advance
    Q / Escape       — quit
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def load_npz(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def hand_verts_in_camera(data: dict, i: int) -> np.ndarray:
    """Transform frame i hand vertices from WiLoR local space into camera space."""
    verts = data["hand_vertices"][i]          # (778, 3) local
    R     = data["hand_global_rot"][i]        # (3, 3)
    t     = data["hand_translation"][i]       # (3,)
    center = verts.mean(axis=0)
    return (R @ (verts - center).T).T + t     # (778, 3) camera space


def object_verts_in_camera(data: dict, i: int) -> np.ndarray:
    """Apply frame i object pose to the canonical mesh."""
    verts = data["object_mesh_vertices"]      # (V, 3) canonical
    R     = data["object_rots"][i]            # (3, 3)
    t     = data["object_trans"][i]           # (3,)
    return (R @ verts.T).T + t               # (V, 3) camera space


def make_hand_mesh(data: dict, i: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(hand_verts_in_camera(data, i))
    # MANO has a fixed face topology of 1538 triangles stored in object_mesh_faces? No —
    # hand faces aren't in the npz yet; use a point cloud fallback.
    mesh.paint_uniform_color([0.82, 0.71, 0.55])  # tan
    mesh.compute_vertex_normals()
    return mesh


def make_hand_pcd(data: dict, i: int) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hand_verts_in_camera(data, i))
    pcd.paint_uniform_color([0.82, 0.71, 0.55])
    return pcd


def make_object_mesh(data: dict, i: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(object_verts_in_camera(data, i))
    mesh.triangles = o3d.utility.Vector3iVector(data["object_mesh_faces"])
    mesh.paint_uniform_color([0.39, 0.58, 0.93])  # cornflower blue
    mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Path to trajectory.npz")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="Playback speed in frames per second (default: 6)")
    args = parser.parse_args()

    data = load_npz(args.npz)
    T = len(data["frame_indices"])
    print(f"Loaded {T} frames from {args.npz}")

    state = {"frame": 0, "playing": False, "last_t": 0.0}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Trajectory Viewer", width=1280, height=720)

    hand_geo   = make_hand_pcd(data, 0)
    object_geo = make_object_mesh(data, 0)
    vis.add_geometry(hand_geo)
    vis.add_geometry(object_geo)

    # Coordinate frame at origin for reference.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(axes)

    def update_geometries(i: int) -> None:
        # Hand — update points in place.
        hand_geo.points = o3d.utility.Vector3dVector(hand_verts_in_camera(data, i))
        hand_geo.paint_uniform_color([0.82, 0.71, 0.55])
        vis.update_geometry(hand_geo)

        # Object — update vertices in place.
        object_geo.vertices = o3d.utility.Vector3dVector(object_verts_in_camera(data, i))
        object_geo.compute_vertex_normals()
        vis.update_geometry(object_geo)

        fidx = data["frame_indices"][i]
        vis.get_render_option().point_size = 3.0
        print(f"\r  frame {i+1}/{T}  (source frame {fidx})", end="", flush=True)

    def go_next(vis):
        state["frame"] = min(state["frame"] + 1, T - 1)
        update_geometries(state["frame"])

    def go_prev(vis):
        state["frame"] = max(state["frame"] - 1, 0)
        update_geometries(state["frame"])

    def toggle_play(vis):
        state["playing"] = not state["playing"]
        print(f"\n{'▶ Playing' if state['playing'] else '⏸ Paused'}")

    # Key bindings (Open3D uses GLFW key codes).
    GLFW_KEY_RIGHT = 262
    GLFW_KEY_LEFT  = 263
    GLFW_KEY_SPACE = 32
    GLFW_KEY_N     = ord("N")
    GLFW_KEY_P     = ord("P")
    GLFW_KEY_Q     = ord("Q")

    vis.register_key_callback(GLFW_KEY_RIGHT, go_next)
    vis.register_key_callback(GLFW_KEY_N,     go_next)
    vis.register_key_callback(GLFW_KEY_LEFT,  go_prev)
    vis.register_key_callback(GLFW_KEY_P,     go_prev)
    vis.register_key_callback(GLFW_KEY_SPACE, toggle_play)
    vis.register_key_callback(GLFW_KEY_Q,     lambda v: v.close())

    print("Controls: N/→ next  P/← prev  Space play/pause  Q quit")
    update_geometries(0)

    frame_dt = 1.0 / args.fps
    while vis.poll_events():
        vis.update_renderer()
        if state["playing"]:
            now = time.time()
            if now - state["last_t"] >= frame_dt:
                state["last_t"] = now
                if state["frame"] < T - 1:
                    state["frame"] += 1
                    update_geometries(state["frame"])
                else:
                    state["playing"] = False
                    print("\n[end of sequence]")

    vis.destroy_window()
    print()


if __name__ == "__main__":
    main()
