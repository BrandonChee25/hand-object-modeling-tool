"""Stage 4 — 3D object mesh generation with SAM-3D Objects.

Takes the anchor frame crop (masked to the object) and feeds it into SAM-3D,
Meta's diffusion-based 3D reconstruction model trained on hand-object interaction
data.  Because it was trained specifically on occluded held objects, it can
hallucinate plausible geometry for the parts of the object hidden under the hand —
which TripoSR (the previous feed-forward substitute) could not do.

SAM-3D jointly outputs:
  mesh (via GLB post-processing)  — object shape in canonical frame
  rotation / translation          — estimated canonical pose

Required config keys:
    sam3d_dir             : path to cloned facebookresearch/sam-3d-objects repo
    sam3d_checkpoint_tag  : checkpoint tag inside checkpoints/ (default: "hf")
"""

from __future__ import annotations

import numpy as np

from pipeline.data import ObjectMesh, PipelineData
from models.sam3d_wrapper import SAM3DModel
from utils.io import crop_with_mask


class ObjectMeshGenerationStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sam3d = SAM3DModel(
            sam3d_dir=cfg["sam3d_dir"],
            checkpoint_tag=cfg.get("sam3d_checkpoint_tag", "hf"),
            device=cfg.get("device", "cuda"),
        )

    def run(self, data: PipelineData) -> PipelineData:
        # Use the SAM-2 seed frame — this is where the object mask is most reliable,
        # and may differ from data.anchor_index when multi-frame detection was used.
        seed_idx = data.object_seg.anchor_frame_index
        anchor_frame = data.frames[seed_idx]
        anchor_mask = data.object_seg.masks[seed_idx]

        # Crop the object region; background filled with neutral grey.
        # crop_with_mask returns both the RGB crop and the matching boolean mask
        # crop, which SAM-3D uses as an alpha channel to know what is object vs
        # background.
        cropped_image, cropped_mask = crop_with_mask(
            anchor_frame.image,
            anchor_mask,
            padding=self.cfg.get("crop_padding_px", 32),
        )

        result = self.sam3d.generate(
            image=cropped_image,
            mask=cropped_mask,
            camera_intrinsics=data.camera_intrinsics,
            flip_axes=self.cfg.get("object_mesh_flip_axes", "yz"),
        )
        # result keys: "vertices", "faces", "canonical_rot", "canonical_trans"

        vertices = result["vertices"]
        faces    = result["faces"]
        print(f"[s4] SAM-3D mesh: {len(vertices)} verts, {len(faces)} faces")

        max_faces = int(self.cfg.get("max_object_mesh_faces", 5000))
        if max_faces > 0 and len(faces) > max_faces:
            vertices, faces = _decimate(vertices, faces, max_faces)
            print(f"[s4] decimated to {len(vertices)} verts, {len(faces)} faces")

        data.object_mesh = ObjectMesh(
            vertices=vertices,
            faces=faces,
            canonical_rot=result["canonical_rot"],
            canonical_trans=result["canonical_trans"],
        )
        return data


def _decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce face count, trying available backends in order."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 1. fast_simplification — call directly; trimesh's wrapper passes target_count
    #    which older versions don't support (they only accept target_reduction 0-1).
    try:
        from fast_simplification import simplify as _fs_simplify
        target_reduction = 1.0 - max_faces / len(faces)
        target_reduction = float(np.clip(target_reduction, 0.0, 0.99))
        vf, ff = _fs_simplify(
            np.array(mesh.vertices, dtype=np.float32),
            np.array(mesh.faces,    dtype=np.int32),
            target_reduction=target_reduction,
        )
        return vf.astype(np.float32), ff.astype(np.int32)
    except (ImportError, ModuleNotFoundError):
        pass

    # 2. Random face subsampling — no quality guarantees but never crashes
    print(f"[s4] WARNING: no decimation library found; using random face subsampling")
    rng  = np.random.default_rng(0)
    keep = rng.choice(len(faces), max_faces, replace=False)
    kept_faces = faces[keep]
    used = np.unique(kept_faces)
    vmap = np.full(len(vertices), -1, dtype=np.int32)
    vmap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], vmap[kept_faces]
