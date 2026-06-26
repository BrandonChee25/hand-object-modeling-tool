"""MoGe wrapper — metric monocular depth + camera intrinsics.

Install: pip install git+https://github.com/microsoft/MoGe
Weights:  downloaded automatically via torch.hub on first inference.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch


class MoGeModel:
    def __init__(self, checkpoint: str = "Ruicheng/moge-vitl", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from moge.model import import_model_class_by_version
        _MoGe = import_model_class_by_version("v1")
        self._model = _MoGe.from_pretrained(self.checkpoint).to(self.device).eval()

    def estimate(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        image : H x W x 3 uint8 RGB

        Returns
        -------
        depth : (H, W) float32  metric depth in metres
        K     : (3, 3) float32  camera intrinsic matrix
        """
        self._load()

        H, W = image.shape[:2]
        inp = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        inp = inp.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model.infer(inp, apply_mask=False, use_fp16=False)

        # Use the depth key directly; points[:,:,2] can be inf when apply_mask=True.
        depth = output["depth"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)

        intrinsics = output["intrinsics"].squeeze(0).cpu().numpy()  # (3, 3) normalised
        # Denormalise to pixel units.
        K = intrinsics.copy()
        K[0, 0] *= W   # fx
        K[1, 1] *= H   # fy
        K[0, 2] *= W   # cx
        K[1, 2] *= H   # cy

        return depth, K.astype(np.float32)
