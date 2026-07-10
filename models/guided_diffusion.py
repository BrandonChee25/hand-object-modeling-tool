"""Object pose estimator — hand-constrained DINOv2 search.

Rather than searching all 162 sphere directions uniformly (slow, and the wrong
prior for a held object), this module uses the hand's MANO geometry to pin the
object's long axis to the grip direction (wrist → fingertips), then scores
only the remaining rotations around that axis with DINOv2 feature similarity.

Two steps:
  1. Align the TripoSR mesh's longest PCA axis to the hand grip axis.
  2. Sample 8 rotations around that axis × 2 end-flips = 16 candidates.
  3. Score each with a DINOv2-small CLS feature vs the real object crop.
  4. Refine around the winner with small random perturbations.

This is O(16) renders instead of O(162), and the hand geometry prior makes
the search accurate for any elongated held object (spoon, fork, pen, etc.)
without per-object tuning.  Falls back to a full Fibonacci sphere when no
hand axis is provided.

Install: open3d (rendering), DINOv2 auto-downloaded via torch.hub on first use.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_dinov2_cache: dict[str, object] = {}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_dinov2(device: str):
    if device not in _dinov2_cache:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        )
        _dinov2_cache[device] = model.to(device).eval()
    return _dinov2_cache[device]


def _preprocess(image: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(image).convert("RGB").resize((224, 224))
    arr = (np.array(pil, dtype=np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def _extract_feature(image: np.ndarray, device: str) -> torch.Tensor:
    model = _get_dinov2(device)
    with torch.no_grad():
        feat = model(_preprocess(image).to(device))
    return torch.nn.functional.normalize(feat, dim=-1).squeeze(0)


def _fibonacci_sphere(n: int) -> np.ndarray:
    indices = np.arange(n)
    phi   = np.arccos(1 - 2 * (indices + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _look_at_rotation(view_dir: np.ndarray) -> np.ndarray:
    z = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _align_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates unit vector a to point along b."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    c = float(np.dot(a, b))
    if c < -0.9999:
        # antiparallel — rotate 180° around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(a, perp); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return (-np.eye(3) + 2 * np.outer(ax, ax)).astype(np.float32)
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K / (1 + c)).astype(np.float32)


def _rotate_around_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around a unit vector."""
    ax = axis / (np.linalg.norm(axis) + 1e-8)
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (c * np.eye(3) + s * K + (1 - c) * np.outer(ax, ax)).astype(np.float32)


def _hand_constrained_candidates(
    vertices: np.ndarray,
    hand_grip_axis: np.ndarray,
    n_around: int = 8,
) -> list[np.ndarray]:
    """Return rotations that place the mesh's long axis along the grip direction.

    The object's primary PCA axis (longest dimension = handle for utensils) is
    aligned to hand_grip_axis, then rotated in n_around steps around that axis.
    Both orientations (which end is head/handle) are included, giving 2×n_around
    candidates in total.
    """
    verts_c = vertices - vertices.mean(0)
    _, _, Vt = np.linalg.svd(verts_c, full_matrices=False)
    mesh_long_axis = Vt[0]  # direction of maximum variance

    grip = hand_grip_axis / (np.linalg.norm(hand_grip_axis) + 1e-8)
    angles = np.linspace(0, 2 * np.pi, n_around, endpoint=False)

    candidates = []
    for flip in (+1.0, -1.0):
        R_align = _align_vector(flip * mesh_long_axis, grip)
        for ang in angles:
            R_spin = _rotate_around_axis(grip, ang)
            candidates.append((R_spin @ R_align).astype(np.float32))
    return candidates


class GuidedDiffusionTracker:
    """Pose tracker: hand-constrained DINOv2 search."""

    def __init__(
        self,
        model=None,
        n_candidates: int = 162,
        alpha_s: float = 0.95,
        alpha_p_range: list = (0.0, 0.6),
        cluster_threshold_deg: float = 15.0,
        cluster_threshold_m: float = 0.05,
        n_steps: int = 50,
        device: str = "cuda",
        render_size: int = 224,
        n_refine_steps: int = 12,
    ):
        self.n_candidates   = n_candidates
        self.device         = device
        self.render_size    = render_size
        self.n_refine_steps = n_refine_steps
        self._mesh       = None
        self._pr_scene   = None
        self._pr_mesh_node = None
        self._pr_renderer  = None

    # ------------------------------------------------------------------
    # Rendering (pyrender EGL — headless GPU rendering, no display needed)
    # ------------------------------------------------------------------

    def _load(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> None:
        if self._mesh is not None:
            return
        import trimesh
        self._mesh = trimesh.Trimesh(
            vertices=np.array(mesh_vertices, dtype=np.float32),
            faces=np.array(mesh_faces, dtype=np.int32),
            process=False,
        )

    def _ensure_renderer(self) -> None:
        if self._pr_renderer is not None:
            return
        import os, pyrender
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        extent = float(np.abs(self._mesh.vertices).max()) * 1.2 + 1e-6
        self._pr_scene = pyrender.Scene(
            bg_color=[0.5, 0.5, 0.5, 1.0],
            ambient_light=[0.4, 0.4, 0.4],
        )
        cam = pyrender.OrthographicCamera(xmag=extent, ymag=extent)
        cam_pose = np.eye(4, dtype=np.float64)
        cam_pose[2, 3] = extent * 3
        self._pr_scene.add(cam, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._pr_scene.add(light, pose=cam_pose)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0], roughnessFactor=0.8
        )
        pr_mesh = pyrender.Mesh.from_trimesh(self._mesh, material=mat, smooth=False)
        self._pr_mesh_node = self._pr_scene.add(pr_mesh)
        self._pr_renderer  = pyrender.OffscreenRenderer(self.render_size, self.render_size)

    def _render(self, R: np.ndarray) -> np.ndarray:
        self._ensure_renderer()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        self._pr_scene.set_pose(self._pr_mesh_node, pose=pose)
        color, _ = self._pr_renderer.render(self._pr_scene)
        return color

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def track_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fixed_shape_vertices: np.ndarray,
        fixed_shape_faces: np.ndarray,
        prev_rot: np.ndarray,
        prev_trans: np.ndarray,
        alpha_p: float,
        depth: np.ndarray | None = None,
        K: np.ndarray | None = None,
        hand_grip_axis: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._load(fixed_shape_vertices, fixed_shape_faces)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return prev_rot, prev_trans
        pad = 10
        y0 = max(0, ys.min() - pad); y1 = min(image.shape[0], ys.max() + pad)
        x0 = max(0, xs.min() - pad); x1 = min(image.shape[1], xs.max() + pad)
        real_feat = _extract_feature(image[y0:y1, x0:x1], self.device)

        # Choose candidate rotations: hand-constrained (16) or full sphere (n_candidates).
        if hand_grip_axis is not None:
            candidates = _hand_constrained_candidates(
                fixed_shape_vertices, hand_grip_axis, n_around=8
            )
        else:
            candidates = [_look_at_rotation(v) for v in _fibonacci_sphere(self.n_candidates)]

        best_score, best_R = -1.0, np.eye(3, dtype=np.float32)
        for R in candidates:
            score = float(torch.dot(real_feat, _extract_feature(self._render(R), self.device)))
            if score > best_score:
                best_score, best_R = score, R

        # Fine refinement: small random perturbations around the winner.
        from scipy.spatial.transform import Rotation as _Rot
        base_rv = _Rot.from_matrix(best_R).as_rotvec()
        for _ in range(self.n_refine_steps):
            delta = np.random.normal(scale=0.1, size=3)
            R_try = _Rot.from_rotvec(base_rv + delta).as_matrix().astype(np.float32)
            score = float(torch.dot(real_feat, _extract_feature(self._render(R_try), self.device)))
            if score > best_score:
                best_score, best_R = score, R_try

        # Translation: depth-lifted centroid of the object mask.
        t = prev_trans
        if depth is not None and K is not None:
            from utils.geometry import depth_lift_mask
            pts = depth_lift_mask(depth, mask, K)
            if len(pts) > 0:
                t = pts.mean(axis=0).astype(np.float32)

        return best_R, t
