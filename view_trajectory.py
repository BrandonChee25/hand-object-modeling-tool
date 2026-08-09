"""Trajectory viewer — generates an interactive HTML file viewable in any browser.

Usage
-----
    python view_trajectory.py out/trajectory.npz
    python view_trajectory.py out/trajectory.npz --out viewer.html --every 2

Controls (in browser)
---------------------
    Slider at the bottom — scrub to any frame
    Play / Pause button  — animate at chosen FPS
    Mouse drag / scroll  — rotate / zoom the 3D scene
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def hand_verts_in_camera(data: dict, i: int) -> np.ndarray:
    # Global motion comes from FP's object translation delta (hand and object move
    # rigidly together). Local articulation (finger shape) comes from WiLoR's
    # centroid-relative vertex change, scaled to metric units.
    if "aligned_hand_verts" in data:
        aligned    = data["aligned_hand_verts"]              # metric, anchor frame
        anchor_idx = int(data["anchor_frame_idx"])
        anchor_raw = data["hand_vertices"][anchor_idx]       # WiLoR at anchor

        raw_span = float(np.linalg.norm(anchor_raw.max(0) - anchor_raw.min(0)))
        met_span = float(np.linalg.norm(aligned.max(0) - aligned.min(0)))
        scale    = met_span / max(raw_span, 1e-6)

        # Global translation from FP tracking
        fp_delta = data["object_trans"][i] - data["object_trans"][anchor_idx]

        # Articulation: centroid-relative finger shape change in WiLoR space
        anchor_c  = anchor_raw.mean(0)
        verts_i   = data["hand_vertices"][i]
        verts_i_c = verts_i.mean(0)
        articulation = scale * ((verts_i - verts_i_c) - (anchor_raw - anchor_c))

        return aligned + fp_delta + articulation

    # Legacy fallback (WiLoR coordinate space — only correct if hand_translation
    # happens to be in the same metric camera space as the object poses).
    verts  = data["hand_vertices"][i]
    R      = data["hand_global_rot"][i]
    t      = data["hand_translation"][i]
    center = verts.mean(axis=0)
    return (R @ (verts - center).T).T + t


def object_verts_in_camera(data: dict, i: int, extra_flip: str = "") -> np.ndarray:
    # Use the metric-scaled, origin-centred mesh when available so that FP's
    # (R, t) pose places the object at the correct size and depth.
    verts = (data["object_mesh_verts_metric"]
             if "object_mesh_verts_metric" in data
             else data["object_mesh_vertices"]).copy()

    # Optional diagnostic pre-flip applied before FP's rotation.
    # Useful for identifying the correct flip without re-running the pipeline.
    axes = set(extra_flip.lower())
    if "x" in axes:
        verts[:, 0] *= -1
    if "y" in axes:
        verts[:, 1] *= -1
    if "z" in axes:
        verts[:, 2] *= -1

    R = data["object_rots"][i]
    t = data["object_trans"][i]

    return (R @ verts.T).T + t


def _hand_trace(hv: np.ndarray, hand_faces: np.ndarray | None, **kwargs) -> object:
    import plotly.graph_objects as go
    if hand_faces is not None:
        return go.Mesh3d(
            x=hv[:, 0], y=hv[:, 1], z=hv[:, 2],
            i=hand_faces[:, 0], j=hand_faces[:, 1], k=hand_faces[:, 2],
            color="rgb(210, 180, 140)",
            opacity=0.6,
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.1),
            name="Hand",
            **kwargs,
        )
    return go.Scatter3d(
        x=hv[:, 0], y=hv[:, 1], z=hv[:, 2],
        mode="markers",
        marker=dict(size=2, color="rgb(210, 180, 140)", opacity=0.8),
        name="Hand",
        **kwargs,
    )


def build_figure(data: dict, every: int, fps: float, obj_flip: str = ""):
    import plotly.graph_objects as go

    T           = len(data["frame_indices"])
    obj_faces   = data["object_mesh_faces"]
    hand_faces  = data.get("hand_mesh_faces")  # None for old NPZ files
    frame_idxs  = data["frame_indices"]

    frames = []
    steps  = []

    for i in range(0, T, every):
        hv = hand_verts_in_camera(data, i)
        ov = object_verts_in_camera(data, i, extra_flip=obj_flip)

        hand_trace = _hand_trace(hv, hand_faces)
        obj_trace = go.Mesh3d(
            x=ov[:, 0], y=ov[:, 1], z=ov[:, 2],
            i=obj_faces[:, 0], j=obj_faces[:, 1], k=obj_faces[:, 2],
            color="cornflowerblue",
            opacity=0.85,
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
            name="Object",
        )

        label = f"frame {i+1}/{T}  (src {frame_idxs[i]})"
        frames.append(go.Frame(data=[hand_trace, obj_trace], name=str(i), layout=go.Layout(title_text=label)))
        steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=str(frame_idxs[i]),
            method="animate",
        ))

    # Initial traces
    hv0 = hand_verts_in_camera(data, 0)
    ov0 = object_verts_in_camera(data, 0, extra_flip=obj_flip)

    fig = go.Figure(
        data=[
            _hand_trace(hv0, hand_faces),
            go.Mesh3d(
                x=ov0[:, 0], y=ov0[:, 1], z=ov0[:, 2],
                i=obj_faces[:, 0], j=obj_faces[:, 1], k=obj_faces[:, 2],
                color="cornflowerblue",
                opacity=0.85,
                flatshading=True,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
                name="Object",
            ),
        ],
        frames=frames,
        layout=go.Layout(
            title=f"frame 1/{T}  (src {frame_idxs[0]})",
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                yanchor="top",
                pad=dict(t=45),
                buttons=[
                    dict(label="▶ Play",  method="animate",
                         args=[None, dict(frame=dict(duration=int(1000 / fps), redraw=True),
                                         fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                pad=dict(t=50),
                steps=steps,
                currentvalue=dict(prefix="source frame: ", visible=True, xanchor="center"),
                transition=dict(duration=0),
            )],
        ),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz",   type=Path, help="Path to trajectory.npz")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: <npz>.html)")
    parser.add_argument("--every", type=int, default=1,
                        help="Only include every Nth frame (default: 1 = all frames)")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="Playback speed in frames per second (default: 6)")
    parser.add_argument("--obj-flip", default="",
                        help="Diagnostic: axes to negate on canonical object mesh before FP rotation "
                             "(e.g. 'x', 'xz', 'y'). Does not re-run FP — useful for identifying "
                             "the right flip to set in config/default.yaml object_mesh_flip_axes.")
    args = parser.parse_args()

    data = {k: v for k, v in np.load(args.npz).items()}
    T    = len(data["frame_indices"])
    print(f"Loaded {T} frames from {args.npz}")
    if args.obj_flip:
        print(f"[viewer] applying diagnostic obj-flip={args.obj_flip!r} to canonical mesh")

    fig = build_figure(data, every=args.every, fps=args.fps, obj_flip=args.obj_flip)

    out = args.out or args.npz.with_suffix(".html")
    fig.write_html(str(out), auto_play=False)
    print(f"Viewer written to: {out}")
    print("Open it in any browser.")


if __name__ == "__main__":
    main()
