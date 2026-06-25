"""SAM-2 wrapper — automatic segmentation + video mask propagation.

Install: pip install git+https://github.com/facebookresearch/segment-anything-2
Weights:  download sam2_hiera_large.pt from Meta and put in checkpoints/.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch


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

        # SAM-2 video predictor expects frames as a directory path or a list
        # of numpy arrays.  We use the in-memory path via inference_state.
        inference_state = self._video_predictor.init_state_from_frames(images)

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

        # Return in frame order.
        return [masks_dict.get(i, np.zeros_like(seed_mask)) for i in range(len(images))]
