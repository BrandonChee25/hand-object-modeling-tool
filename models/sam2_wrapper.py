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
