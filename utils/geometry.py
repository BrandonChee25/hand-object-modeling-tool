"""Geometry utilities: SE(3) ops, MANO helpers, depth lifting."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Fixed MANO face topology (1538 triangles).  Loaded once at import time.
# Replace with actual MANO face array from the MANO model package.
MANO_FACES: np.ndarray = np.zeros((1538, 3), dtype=np.int32)  # placeholder


def mano_vertices_to_mask(
    vertices: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project MANO vertices to image space and rasterise a rough hand mask."""
    H, W = image_shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    mask = np.zeros((H, W), dtype=bool)
    for x3d, y3d, z3d in vertices:
        if z3d <= 0:
            continue
        u = int(x3d * fx / z3d + cx)
        v = int(y3d * fy / z3d + cy)
        if 0 <= u < W and 0 <= v < H:
            mask[v, u] = True

    # Dilate slightly to cover gaps between projected points.
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=8)
    return mask


def depth_lift_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Unproject masked depth pixels to 3D points in camera space.

    Returns (N, 3) float32 array.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.where(mask & (depth > 0))
    zs = depth[ys, xs]
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    return np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)


def solve_scale_least_squares(
    z_ref: float,
    z_src: float,
    src_points: np.ndarray,
) -> float:
    """Solve for scale k such that k * z_src ≈ z_ref via least squares.

    Minimises sum_i (k * src_points[i,2] - z_ref)^2 over all object points.
    """
    zs = src_points[:, 2]
    k = np.dot(zs, np.full_like(zs, z_ref)) / np.dot(zs, zs)
    return float(k)


def rotation_velocity_2d(
    prev_mask: np.ndarray,
    curr_mask: np.ndarray,
    prev_image: np.ndarray,
    curr_image: np.ndarray,
) -> float:
    """Estimate 2D angular velocity of the object between two frames.

    Uses optical flow centroids of tracked mask points as a proxy.
    Returns angular velocity in radians/frame.
    """
    # Compute centroid shift as a simple proxy for rotation velocity.
    # A full implementation would use BootsTAPIR point tracks.
    def centroid(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.array([0.0, 0.0])
        return np.array([xs.mean(), ys.mean()])

    c_prev = centroid(prev_mask)
    c_curr = centroid(curr_mask)
    displacement = np.linalg.norm(c_curr - c_prev)
    # Normalise by mask diagonal to get a scale-invariant velocity.
    ys, xs = np.where(prev_mask)
    if len(xs) > 1:
        diag = np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
        diag = max(diag, 1.0)
    else:
        diag = 1.0
    return float(displacement / diag)


def rotation_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    R_rel = R1.T @ R2
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def se3_cluster_centroid(
    candidates: list[np.ndarray],
    threshold_deg: float,
    threshold_m: float,
) -> np.ndarray:
    """Find the largest SE(3) cluster and return its centroid pose (7-vector).

    Each candidate is a 7-vector [R_6d(6) | t(3)] — see pose7d_to_Rt.
    Pairwise distance = geodesic(R1, R2) + ||t1 - t2||_2.
    """
    n = len(candidates)
    threshold_rad = np.deg2rad(threshold_deg)

    rots = [pose7d_to_Rt(c)[0] for c in candidates]
    trans = np.array([pose7d_to_Rt(c)[1] for c in candidates])

    cluster_sizes = []
    cluster_members: list[list[int]] = []

    for i in range(n):
        members = []
        for j in range(n):
            d_rot = rotation_geodesic_distance(rots[i], rots[j])
            d_trans = np.linalg.norm(trans[i] - trans[j])
            if d_rot < threshold_rad and d_trans < threshold_m:
                members.append(j)
        cluster_sizes.append(len(members))
        cluster_members.append(members)

    best_i = int(np.argmax(cluster_sizes))
    inliers = cluster_members[best_i]

    # Arithmetic mean of translation.
    mean_trans = trans[inliers].mean(axis=0)
    # Geodesic mean of rotations (Fréchet mean via iterative averaging).
    mean_rot = geodesic_mean_rotation(np.stack([rots[j] for j in inliers]))

    return Rt_to_pose7d(mean_rot, mean_trans)


def geodesic_mean_rotation(rots: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Fréchet mean of rotation matrices via iterative geodesic averaging."""
    mean = rots[0].copy()
    for _ in range(max_iter):
        tangents = []
        for R in rots:
            R_rel = mean.T @ R
            r = Rotation.from_matrix(R_rel)
            tangents.append(r.as_rotvec())
        delta = np.mean(tangents, axis=0)
        if np.linalg.norm(delta) < 1e-6:
            break
        mean = mean @ Rotation.from_rotvec(delta).as_matrix()
    return mean


def Rt_to_pose7d(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack (3,3) rotation + (3,) translation into a 9-vector [R.ravel() | t]."""
    return np.concatenate([R.ravel(), t])


def pose7d_to_Rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 9-vector back to (R, t).  Re-orthogonalise R via SVD."""
    R_raw = pose[:9].reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = pose[9:12] if len(pose) >= 12 else pose[6:9]
    return R, t


def se3_cluster_mean(data: "PipelineData") -> tuple[np.ndarray, np.ndarray]:  # noqa: F821
    """Convenience wrapper used by Stage 6."""
    from pipeline.data import PipelineData
    rots = np.stack(data.object_poses.rots)
    mean_rot = geodesic_mean_rotation(rots)
    mean_trans = np.stack(data.object_poses.trans).mean(axis=0)
    return mean_rot, mean_trans
