"""WiLoR model wrappers.

WiLoR (Wild Loco Reconstruction) uses a DINOv2-L backbone to estimate MANO
parameters from a single image crop.

Install: pip install git+https://github.com/rolpotamias/WiLoR
Weights:  downloaded automatically from HuggingFace on first use.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class WiLoRDetector:
    """Lightweight YOLO-based hand bounding-box detector bundled with WiLoR."""

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from wilor.models import WiLoR as _WiLoR
        # WiLoR ships its own ViTDet hand detector; load it here.
        # The detector is separate from the MANO regressor.
        from wilor.utils.hand_detector import HandDetector
        self._model = HandDetector(confidence_threshold=self.confidence_threshold)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return (bbox_xyxy, confidence) for the most prominent hand, or (None, 0)."""
        self._load()
        detections = self._model.detect(image)  # list of (bbox, score)
        if not detections:
            return None, 0.0
        # Pick the detection with the highest score.
        best_bbox, best_score = max(detections, key=lambda x: x[1])
        return np.array(best_bbox, dtype=np.float32), float(best_score)


class WiLoRModel:
    """Runs WiLoR inference to produce MANO parameters + mesh vertices."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint  # ignored — WiLoR auto-downloads from HF
        self.device = device
        self._model = None
        self._mano_layer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from wilor.models import WiLoR as _WiLoR
        from smplx import build_layer as build_mano

        self._model = _WiLoR.from_pretrained("rolpotamias/WiLoR").to(self.device).eval()

        # MANO layer gives us the face topology and forward kinematics.
        self._mano_layer = build_mano(
            model_type="mano",
            model_path="checkpoints/mano",  # download MANO weights separately
            is_rhand=True,
            num_pca_comps=45,
            flat_hand_mean=False,
        ).to(self.device)

    def reconstruct(self, image: np.ndarray, hand_bbox: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image     : H x W x 3 uint8 RGB
        hand_bbox : (x1, y1, x2, y2) float32

        Returns
        -------
        dict with keys:
            pose        : (48,)  MANO axis-angle pose
            shape       : (10,)  MANO betas
            global_rot  : (3, 3)
            translation : (3,)   raw un-rescaled translation
            vertices    : (778, 3)
            keypoints_3d: (21, 3)
        """
        self._load()

        x1, y1, x2, y2 = hand_bbox.astype(int)
        crop = image[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop).resize((256, 256))
        inp = np.array(pil_crop).astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._model(inp)

        pose = out["pose"].squeeze(0).cpu().numpy()         # (48,)
        shape = out["betas"].squeeze(0).cpu().numpy()       # (10,)
        global_rot_aa = out["global_orient"].squeeze(0).cpu().numpy()  # (3,)
        trans = out["transl"].squeeze(0).cpu().numpy()      # (3,)

        from scipy.spatial.transform import Rotation
        global_rot = Rotation.from_rotvec(global_rot_aa).as_matrix()

        vertices = out["vertices"].squeeze(0).cpu().numpy()         # (778, 3)
        keypoints = out["joints"].squeeze(0).cpu().numpy()          # (21, 3)

        return {
            "pose": pose,
            "shape": shape,
            "global_rot": global_rot,
            "translation": trans,
            "vertices": vertices,
            "keypoints_3d": keypoints,
        }
