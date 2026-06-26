"""Stage 7 — Export static 3D meshes.

Writes the aligned hand + object meshes to the output directory in both OBJ
(universal) and GLB (compact, web-compatible) formats.

Output layout:
    <out_dir>/
        hand.obj
        hand.mtl
        object.obj
        object.mtl
        scene.glb          # hand + object combined
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

        return output_dir


def _assign_vertex_colors(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> None:
    mesh.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
