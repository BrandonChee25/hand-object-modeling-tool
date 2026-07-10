"""SAM-2 wrapper — automatic segmentation + video mask propagation.

Install: pip install git+https://github.com/facebookresearch/segment-anything-2
Weights:  download sam2_hiera_large.pt from Meta and put in checkpoints/.
"""

from __future__ import annotations
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class SAM2Model:
    def __init__(
        self,
        checkpoint: str = "checkpoints/sam2_hiera_large.pt",
        config: str = "sam2_hiera_large.yaml",
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self._predictor = None
        self._auto_gen = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2 = build_sam2(self.config, self.checkpoint, device=self.device)
        self._predictor = SAM2ImagePredictor(sam2)
        self._auto_gen = SAM2AutomaticMaskGenerator(sam2)

        # Separate instance for video propagation (shares weights, different state).
        self._video_predictor = build_sam2_video_predictor(
            self.config, self.checkpoint, device=self.device
        )

    def segment_with_point(self, image: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
        """Return a single bool H×W mask for the object at the given pixel (x, y)."""
        self._load()
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[point_xy[0], point_xy[1]]]),
            point_labels=np.array([1]),  # foreground
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    def segment_with_box(
        self, image: np.ndarray, box_xyxy: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Return a single bool H×W mask for the object inside the given box (x1,y1,x2,y2)."""
        self._load()
        self._predictor.set_image(image)
        x1, y1, x2, y2 = box_xyxy
        masks, scores, _ = self._predictor.predict(
            box=np.array([[x1, y1, x2, y2]], dtype=float),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    def segment_held_object(
        self,
        image: np.ndarray,
        hand_bbox: tuple[int, int, int, int],
        hand_mask: np.ndarray,
        expand: float = 0.6,
    ) -> np.ndarray | None:
        """Find the object held in the hand using box + negative-point prompts.

        Expands the hand bbox to cover the held object, then places negative
        (background) points across the palm so SAM-2 segments what is inside
        the expanded region but is NOT the hand.

        Returns a bool H×W mask, or None if the result is empty.
        """
        self._load()
        self._predictor.set_image(image)

        H, W = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in hand_bbox)
        bw, bh = x2 - x1, y2 - y1

        pad_x = int(bw * expand)
        pad_y = int(bh * expand)
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(W, x2 + pad_x)
        ey2 = min(H, y2 + pad_y)

        hcx, hcy = (x1 + x2) // 2, (y1 + y2) // 2
        neg_points = np.array([
            [hcx, hcy],
            [hcx, y1 + bh // 4],
            [hcx, y2 - bh // 4],
            [x1 + bw // 4, hcy],
            [x2 - bw // 4, hcy],
        ], dtype=float)
        neg_labels = np.zeros(len(neg_points), dtype=int)

        masks, scores, _ = self._predictor.predict(
            point_coords=neg_points,
            point_labels=neg_labels,
            box=np.array([ex1, ey1, ex2, ey2], dtype=float),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask = masks[best].astype(bool) & ~hand_mask

        # Keep only the largest connected component — spurious stray regions are small.
        from scipy.ndimage import label as _label, binary_erosion
        labeled, n = _label(mask)
        if n > 1:
            sizes = np.array([(labeled == i).sum() for i in range(1, n + 1)])
            mask = (labeled == (np.argmax(sizes) + 1)).astype(bool)

        # Erode slightly to drop noisy boundary pixels SAM-2 bleeds into.
        eroded = binary_erosion(mask, iterations=2)
        if eroded.any():
            mask = eroded

        return mask if mask.any() else None

    def auto_segment(self, image: np.ndarray) -> list[np.ndarray]:
        """Return list of bool H x W masks from automatic everything-segmentation."""
        self._load()
        results = self._auto_gen.generate(image)
        return [r["segmentation"].astype(bool) for r in results]

    def propagate(
        self,
        images: list[np.ndarray],
        anchor_index: int,
        seed_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Propagate the seed mask from the anchor frame to all other frames.

        SAM-2 video predictor works on a list of frames held in memory.
        We feed it one frame at a time via its inference state.
        """
        self._load()

        # SAM-2 video predictor requires frames as JPEGs in a directory.
        tmp_dir = tempfile.mkdtemp()
        try:
            for i, img in enumerate(images):
                Image.fromarray(img).save(f"{tmp_dir}/{i:06d}.jpg")
            inference_state = self._video_predictor.init_state(tmp_dir)

            # Provide the seed mask on the anchor frame.
            _, _, _ = self._video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=anchor_index,
                obj_id=1,
                mask=seed_mask,
            )

            # Propagate forward then backward.
            masks_dict: dict[int, np.ndarray] = {}
            for frame_idx, obj_ids, mask_logits in self._video_predictor.propagate_in_video(
                inference_state
            ):
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze(0).astype(bool)
                masks_dict[frame_idx] = mask
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Return in frame order.
        return [masks_dict.get(i, np.zeros_like(seed_mask)) for i in range(len(images))]
