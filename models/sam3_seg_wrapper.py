"""SAM3 wrapper — text/visual-prompted object segmentation + video tracking.

Requires: Python 3.12+, PyTorch 2.7+, CUDA 12.6+
Install:  git clone https://github.com/facebookresearch/sam3 && pip install -e .
Weights:  gated on HuggingFace — agree to terms at facebook/sam3 or facebook/sam3.1,
          then: huggingface-cli login

This wrapper is the Stage 3 replacement for SAM-2's segment_held_object + propagate.
SAM3 does detection, segmentation, and video tracking in a single jointly-trained model,
which handles partially occluded held objects better than GroundingDINO + SAM2 separately.
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np


class SAM3SegModel:
    def __init__(self, device: str = "cuda", version: str = "sam3.1"):
        self.device = device
        self.version = version
        self._processor = None
        self._video_predictor = None

    def _load(self) -> None:
        if self._processor is not None:
            return
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
        from sam3.model.sam3_image_processor import Sam3Processor

        hf_repo = f"facebook/{self.version}"
        model = build_sam3_image_model(
            bpe_path=None, device=self.device, load_from_HF=True
        )
        self._processor = Sam3Processor(model, confidence_threshold=0.3)
        self._video_predictor = build_sam3_video_predictor(gpus_to_use=None)
        print(f"[sam3] loaded {self.version} ({hf_repo})")

    def segment_held_object(
        self,
        image: np.ndarray,
        hand_bbox: tuple[int, int, int, int],
        hand_mask: np.ndarray,
        expand: float = 0.6,
    ) -> np.ndarray | None:
        """Single-frame: find held object via expanded hand bbox visual prompt.

        Converts the expanded hand bbox to SAM3's normalized cxcywh format,
        then post-processes with largest-component + erosion (same as SAM2 path).
        """
        self._load()
        from PIL import Image as PILImage
        from scipy.ndimage import label as _label, binary_erosion

        H, W = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in hand_bbox)
        bw, bh = x2 - x1, y2 - y1

        pad_x = int(bw * expand)
        pad_y = int(bh * expand)
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(W, x2 + pad_x)
        ey2 = min(H, y2 + pad_y)

        # SAM3 box prompt: normalized cx, cy, w, h in [0, 1]
        cx  = ((ex1 + ex2) / 2) / W
        cy  = ((ey1 + ey2) / 2) / H
        bw_n = (ex2 - ex1) / W
        bh_n = (ey2 - ey1) / H

        pil = PILImage.fromarray(image)
        state = self._processor.set_image(pil)
        state = self._processor.add_geometric_prompt(
            box=[cx, cy, bw_n, bh_n], label=True, state=state
        )

        masks  = state["masks"]
        scores = state["scores"]
        if len(scores) == 0:
            return None

        if hasattr(scores, "cpu"):
            scores = scores.cpu().float()
        best = int(np.array(scores).argmax())
        mask = masks[best]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.array(mask).astype(bool)
        while mask.ndim > 2:
            mask = mask[0]

        # Strip hand pixels that leaked in.
        mask = mask & ~hand_mask

        # Keep only the largest connected component.
        labeled, n = _label(mask)
        if n > 1:
            sizes = np.array([(labeled == i).sum() for i in range(1, n + 1)])
            mask = (labeled == (np.argmax(sizes) + 1)).astype(bool)

        # Erode boundary noise.
        eroded = binary_erosion(mask, iterations=3)
        if eroded.any():
            mask = eroded

        return mask if mask.any() else None

    def propagate(
        self,
        images: list[np.ndarray],
        anchor_index: int,
        seed_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Propagate seed mask through all frames using SAM3 video predictor.

        The seed mask is converted to a spread of positive sample points +
        negative points outside, which SAM3 uses to initialise tracking.
        """
        self._load()
        import os
        import torch

        tmp_dir = tempfile.mkdtemp()
        try:
            # Write frames as individual JPEGs — SAM3 accepts a directory of images.
            from PIL import Image as PILImage
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir)
            for i, img in enumerate(images):
                PILImage.fromarray(img).save(
                    os.path.join(frames_dir, f"{i:06d}.jpg"), quality=95
                )

            resp = self._video_predictor.handle_request({
                "type": "start_session",
                "resource_path": frames_dir,
            })
            session_id = resp["session_id"]

            # Convert seed mask → bounding box for the semantic prompt path.
            # SAM3 video predictor uses text+box (not points) for propagation.
            ys, xs = np.where(seed_mask)
            cx = float((xs.min() + xs.max()) / 2)
            cy = float((ys.min() + ys.max()) / 2)
            w  = float(xs.max() - xs.min())
            h  = float(ys.max() - ys.min())
            H, W = images[0].shape[:2]
            # Normalize to [0, 1] — model's add_prompt at line 839 expects relative coords.
            box = torch.tensor(
                [[cx / W, cy / H, w / W, h / H]], dtype=torch.float32
            )
            self._video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor_index,
                "text": "held object",
                "bounding_boxes": box,
                "bounding_box_labels": torch.tensor([1], dtype=torch.int32),
            })

            masks_dict: dict[int, np.ndarray] = {}
            for resp in self._video_predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
            }):
                fidx = resp["frame_index"]
                m = _extract_mask(resp["outputs"])
                if m is not None:
                    masks_dict[fidx] = m

            self._video_predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })

            # Always keep the seed mask for the anchor frame — propagation may miss it.
            masks_dict[anchor_index] = seed_mask
            return [masks_dict.get(i, np.zeros_like(seed_mask)) for i in range(len(images))]

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_mask(outputs) -> np.ndarray | None:
    """Pull a bool H×W mask out of a SAM3 video predictor output dict."""
    if isinstance(outputs, dict):
        m = outputs.get("masks") or outputs.get("mask")
    else:
        m = outputs
    if m is None:
        return None
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.array(m)
    if m.ndim > 2:
        m = m[0]   # take first object
    return m.astype(bool)
