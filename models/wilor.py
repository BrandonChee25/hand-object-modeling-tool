"""WiLoR-mini model wrappers.

WiLoR-mini is a lightweight reimplementation of WiLoR that provides a single
unified pipeline for hand detection + MANO reconstruction.

Install: pip install git+https://github.com/warmshao/WiLoR-mini
Weights:  downloaded automatically from HuggingFace on first use.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Module-level cache so both wrapper classes share one loaded pipeline per device.
_pipeline_cache: dict[str, object] = {}


def _get_pipeline(device: str):
    if device not in _pipeline_cache:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
        import utils.geometry as _geom
        dtype = torch.float16 if "cuda" in device else torch.float32
        pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
        _pipeline_cache[device] = pipe
        # Populate the real MANO face topology now that the model is loaded.
        _geom.MANO_FACES = pipe.wilor_model.mano.faces.astype("int32")
    return _pipeline_cache[device]


class WiLoRDetector:
    """Hand bounding-box detector backed by WiLoR-mini's unified pipeline."""

    def __init__(self, confidence_threshold: float = 0.3, device: str = "cuda"):
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return (bbox_xyxy, confidence) for the most prominent hand, or (None, 0)."""
        outputs = _get_pipeline(self.device).predict(image)
        if not outputs:
            return None, 0.0
        # WiLoR-mini doesn't expose per-detection confidence scores; use 1.0.
        bbox = np.array(outputs[0]["hand_bbox"], dtype=np.float32)
        return bbox, 1.0


class WiLoRModel:
    """Runs WiLoR-mini inference to produce MANO parameters + mesh vertices."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint  # unused — WiLoR-mini auto-downloads weights
        self.device = device

    def reconstruct(self, image: np.ndarray, hand_bbox: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image     : H x W x 3 uint8 RGB
        hand_bbox : (x1, y1, x2, y2) float32

        Returns
        -------
        dict with keys:
            pose        : (48,)  MANO axis-angle pose (global_orient + hand_pose)
            shape       : (10,)  MANO betas (zeros — not exposed by WiLoR-mini)
            global_rot  : (3, 3)
            translation : (3,)   camera-space translation
            vertices    : (778, 3)
            keypoints_3d: (21, 3)
        """
        outputs = _get_pipeline(self.device).predict(image)
        if not outputs:
            raise RuntimeError("WiLoR-mini returned no detections for this frame.")

        out = _best_matching_detection(outputs, hand_bbox)
        preds = out["wilor_preds"]

        vertices = np.array(preds["pred_vertices"][0], dtype=np.float32)        # (778, 3)
        keypoints_3d = np.array(preds["pred_keypoints_3d"][0], dtype=np.float32)  # (21, 3)
        translation = np.array(preds["pred_cam_t_full"][0], dtype=np.float32)   # (3,)

        global_orient_aa = np.array(preds["global_orient"][0], dtype=np.float32).flatten()[:3]  # (3,)
        hand_pose = np.array(preds["hand_pose"][0], dtype=np.float32).flatten()[:45]            # (45,)
        pose = np.concatenate([global_orient_aa, hand_pose])                     # (48,)
        global_rot = Rotation.from_rotvec(global_orient_aa).as_matrix()          # (3, 3)

        return {
            "pose": pose,
            "shape": np.zeros(10, dtype=np.float32),
            "global_rot": global_rot,
            "translation": translation,
            "vertices": vertices,
            "keypoints_3d": keypoints_3d,
        }


def _best_matching_detection(outputs: list[dict], ref_bbox: np.ndarray) -> dict:
    """Return the detection whose bbox centroid is closest to ref_bbox centroid."""
    ref_cx = (ref_bbox[0] + ref_bbox[2]) / 2
    ref_cy = (ref_bbox[1] + ref_bbox[3]) / 2
    best, best_dist = outputs[0], float("inf")
    for out in outputs:
        b = out["hand_bbox"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dist = (cx - ref_cx) ** 2 + (cy - ref_cy) ** 2
        if dist < best_dist:
            best, best_dist = out, dist
    return best
