"""Shared dataclasses that flow through every pipeline stage."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """One input image with pre-computed detection results."""
    index: int
    path: Path
    image: np.ndarray                   # H x W x 3, uint8 RGB
    hand_bbox: Optional[np.ndarray]     # (x1, y1, x2, y2) or None if no hand detected
    hand_score: float = 0.0


@dataclass
class HandResult:
    """MANO reconstruction for a single frame."""
    frame_index: int
    mano_pose: np.ndarray       # (48,) axis-angle pose params (theta)
    mano_shape: np.ndarray      # (10,) shape betas
    global_rot: np.ndarray      # (3, 3) rotation matrix in camera space
    translation: np.ndarray     # (3,) translation in camera space (metric)
    vertices: np.ndarray        # (778, 3) MANO mesh vertices in camera space
    keypoints_3d: np.ndarray    # (21, 3) 3D hand joints in camera space


@dataclass
class ObjectSegmentation:
    """Per-frame object masks from SAM-2."""
    # masks[i] is a bool H x W array for frame i
    masks: list[np.ndarray]
    anchor_frame_index: int


@dataclass
class ObjectMesh:
    """Blind 3D reconstruction of the object from the anchor frame."""
    vertices: np.ndarray    # (V, 3) in object-canonical space
    faces: np.ndarray       # (F, 3) triangle indices
    # canonical pose from SAM-3D at the anchor frame
    canonical_rot: np.ndarray       # (3, 3)
    canonical_trans: np.ndarray     # (3,)


@dataclass
class ObjectPoseSequence:
    """Per-frame object poses produced by guided diffusion."""
    # rots[i], trans[i] give SE(3) pose for frame i in camera space
    rots: list[np.ndarray]      # each (3, 3)
    trans: list[np.ndarray]     # each (3,)
    # guidance alpha values used (diagnostic)
    alpha_p_values: list[float]


@dataclass
class AlignedScene:
    """Hand mesh + object mesh in a shared metric coordinate frame."""
    hand_vertices: np.ndarray   # (778, 3)
    hand_faces: np.ndarray      # (1538, 3)  — fixed MANO topology
    object_vertices: np.ndarray # (V, 3)
    object_faces: np.ndarray    # (F, 3)
    # 4x4 transform that maps raw camera-space coords to the shared frame
    world_from_camera: np.ndarray


@dataclass
class PipelineData:
    """Accumulates outputs across all stages."""
    frames: list[Frame] = field(default_factory=list)
    anchor_index: int = 0
    output_dir: Optional[Path] = None              # set by runner; used for debug outputs
    depth_map: Optional[np.ndarray] = None         # (H, W) metric depth at anchor
    depth_maps: dict = field(default_factory=dict) # frame_index → (H, W) depth
    camera_intrinsics: Optional[np.ndarray] = None # (3, 3) K matrix at anchor
    hand_results: list[HandResult] = field(default_factory=list)
    object_seg: Optional[ObjectSegmentation] = None
    object_mesh: Optional[ObjectMesh] = None
    object_poses: Optional[ObjectPoseSequence] = None
    aligned_scene: Optional[AlignedScene] = None
