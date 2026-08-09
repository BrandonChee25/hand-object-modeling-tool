"""3D mesh generation wrapper — SAM-3D Objects (Meta, facebookresearch/sam-3d-objects).

SAM-3D is a diffusion-based, image-conditioned 3D reconstruction model trained
specifically on hand-object interaction data.  Unlike TripoSR (previous substitute),
it understands occlusion by hands and can hallucinate plausible geometry for the
portions of the object hidden under the fingers.

Setup (run once on the cluster):
    git clone https://github.com/facebookresearch/sam-3d-objects
    cd sam-3d-objects
    mamba env create -f environments/default.yml && mamba activate sam3d-objects
    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e '.[dev]' -e '.[p3d]' -e '.[inference]'
    ./patching/hydra
    # Request HuggingFace access then:
    hf auth login
    TAG=hf
    hf download --repo-type model --local-dir checkpoints/${TAG}-download \\
        --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
    rm -rf checkpoints/${TAG}-download

Config keys (pipeline config):
    sam3d_dir             : path to the cloned sam-3d-objects repo (required)
    sam3d_checkpoint_tag  : checkpoint tag inside checkpoints/ (default: "hf")
"""

from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np


class SAM3DModel:
    def __init__(
        self,
        sam3d_dir: str,
        checkpoint_tag: str = "hf",
        device: str = "cuda",
    ):
        self.sam3d_dir = Path(sam3d_dir)
        self.checkpoint_tag = checkpoint_tag
        self.device = device
        self._inference = None

    def _load(self) -> None:
        if self._inference is not None:
            return

        # Add the repo root and notebook dir so `from inference import Inference` works.
        repo = str(self.sam3d_dir)
        notebook = str(self.sam3d_dir / "notebook")
        for p in (repo, notebook):
            if p not in sys.path:
                sys.path.insert(0, p)

        from inference import Inference  # sam-3d-objects/notebook/inference.py

        config_path = self.sam3d_dir / "checkpoints" / self.checkpoint_tag / "pipeline.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"SAM-3D checkpoint config not found at {config_path}. "
                "Check sam3d_dir and sam3d_checkpoint_tag in your pipeline config."
            )
        self._inference = Inference(str(config_path), compile=False)

    def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        seed: int = 42,
    ) -> dict:
        """Reconstruct a 3D mesh from a single masked object crop.

        SAM-3D merges the mask into the image's alpha channel so it can reason
        about which pixels are object vs. background, then runs diffusion-based
        3D generation with explicit occlusion awareness.

        Parameters
        ----------
        image             : (H, W, 3) uint8 — object crop, background grey
        mask              : (H, W) bool     — object pixels True
        camera_intrinsics : (3, 3) float    — K matrix (not used by SAM-3D; kept for API compat)
        seed              : int             — random seed for reproducibility

        Returns
        -------
        dict with keys:
            vertices        : (V, 3) float32 — mesh in object-canonical space, centred at origin
            faces           : (F, 3) int32
            canonical_rot   : (3, 3) float32 — estimated canonical orientation (rotation matrix)
            canonical_trans : (3,)   float32 — estimated canonical translation
        """
        self._load()

        from PIL import Image as PILImage

        # Resize to SAM-3D's expected input resolution.
        pil_rgb = PILImage.fromarray(image).resize((512, 512))
        pil_mask = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
            (512, 512), PILImage.NEAREST
        )

        # Build RGBA: SAM-3D uses the alpha channel as the object mask.
        rgba = np.array(pil_rgb.convert("RGBA"))
        rgba[:, :, 3] = np.array(pil_mask)
        pil_rgba = PILImage.fromarray(rgba)

        # Call the internal pipeline directly so we can request mesh post-processing
        # (the public Inference.__call__ hard-codes with_mesh_postprocess=False).
        result = self._inference._pipeline.run(
            pil_rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )

        vertices, faces = self._extract_mesh(result)
        canonical_rot, canonical_trans = self._extract_pose(result)

        # SAM-3D generates geometry in OpenGL convention (Y-up, Z-backward).
        # FP and MoGe use OpenCV convention (Y-down, Z-forward).
        # Convert: negate Y and Z, equivalent to a 180° rotation around X.
        vertices[:, 1] *= -1
        vertices[:, 2] *= -1

        # Apply SAM-3D's canonical rotation so FP receives the mesh in
        # approximately the correct camera-space orientation.
        vertices = (canonical_rot @ vertices.T).T
        vertices -= vertices.mean(0)

        return {
            "vertices": vertices,
            "faces": faces,
            "canonical_rot": canonical_rot,
            "canonical_trans": canonical_trans,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_mesh(self, result: dict) -> tuple[np.ndarray, np.ndarray]:
        import trimesh

        glb = result.get("glb")
        if glb is not None:
            if isinstance(glb, (str, Path)):
                scene = trimesh.load(str(glb))
            elif isinstance(glb, bytes):
                scene = trimesh.load(io.BytesIO(glb), file_type="glb")
            elif hasattr(glb, "geometry"):
                scene = glb
            elif hasattr(glb, "vertices"):
                return (
                    np.array(glb.vertices, dtype=np.float32),
                    np.array(glb.faces, dtype=np.int32),
                )
            else:
                raise RuntimeError(f"Unrecognised GLB type: {type(glb)}")

            if hasattr(scene, "geometry") and scene.geometry:
                mesh = trimesh.util.concatenate(list(scene.geometry.values()))
            elif hasattr(scene, "vertices"):
                mesh = scene
            else:
                raise RuntimeError("Could not extract mesh geometry from SAM-3D GLB output.")

            return (
                np.array(mesh.vertices, dtype=np.float32),
                np.array(mesh.faces, dtype=np.int32),
            )

        # Fallback: reconstruct mesh from Gaussian splat centres via Poisson reconstruction.
        gaussian = result.get("gaussian")
        if gaussian is not None:
            if isinstance(gaussian, list):
                gaussian = gaussian[0]
            return self._gaussians_to_mesh(gaussian)

        raise RuntimeError(
            "SAM-3D output contains neither 'glb' nor 'gaussian'. "
            "Check that with_mesh_postprocess=True is supported by the installed version."
        )

    def _gaussians_to_mesh(self, gaussian) -> tuple[np.ndarray, np.ndarray]:
        """Poisson surface reconstruction from high-opacity Gaussian centres."""
        import torch
        import open3d as o3d

        xyz = gaussian.get_xyz().detach().cpu().numpy()           # (N, 3)
        opacity = torch.sigmoid(
            gaussian.get_opacity()
        ).detach().cpu().numpy().squeeze()                         # (N,)

        # Keep only surface-level Gaussians; relax threshold if too few survive.
        for threshold in (0.5, 0.2, 0.05):
            pts = xyz[opacity > threshold]
            if len(pts) >= 500:
                break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        return (
            np.asarray(mesh_o3d.vertices, dtype=np.float32),
            np.asarray(mesh_o3d.triangles, dtype=np.int32),
        )

    def _extract_pose(self, result: dict) -> tuple[np.ndarray, np.ndarray]:
        """Convert SAM-3D's quaternion + translation output to (R 3×3, t 3)."""
        from scipy.spatial.transform import Rotation

        rot_q = result.get("rotation")
        trans = result.get("translation")

        if rot_q is not None:
            if hasattr(rot_q, "detach"):
                rot_q = rot_q.detach().float().cpu().numpy()
            q = np.array(rot_q, dtype=np.float32).flatten()
            # SAM-3D uses [w, x, y, z]; scipy expects [x, y, z, w].
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        else:
            R = np.eye(3, dtype=np.float64)

        if trans is not None:
            if hasattr(trans, "detach"):
                trans = trans.detach().float().cpu().numpy()
            t = np.array(trans, dtype=np.float32).flatten()[:3]
        else:
            t = np.zeros(3, dtype=np.float64)

        return R.astype(np.float32), t.astype(np.float32)

    def step(self, x_s, x_p, t, dt, image, camera_intrinsics):
        """Iterative denoising step — reserved for guided flow-matching tracker (Stage 5)."""
        raise NotImplementedError(
            "step() is not yet wired up. Stage 5 uses FoundationPose for pose tracking."
        )
