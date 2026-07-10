"""Quick test for Stage 3 object detection without running the full pipeline.

Usage
-----
    # Auto-detect hand with YOLO, then try all detection methods:
    python test_detection.py path/to/frame.jpg

    # Provide hand bbox manually if YOLO misses it (x1 y1 x2 y2):
    python test_detection.py path/to/frame.jpg --hand-bbox 100 200 400 600

    # Override the object description:
    python test_detection.py path/to/frame.jpg --description "cup"

Outputs (written next to the input image):
    <name>_detections.png   — GroundingDINO boxes
    <name>_handsam.png      — SAM-2 hand-box prompt result
    <name>_heuristic.png    — depth-weighted contact heuristic result
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path, color=(0, 255, 0)) -> None:
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)
    print(f"  saved {path}")


def detect_hand(image: np.ndarray, cfg: dict) -> np.ndarray | None:
    from models.wilor import WiLoRDetector
    det = WiLoRDetector(
        confidence_threshold=cfg.get("hand_detection_confidence", 0.3),
        device=cfg.get("device", "cuda"),
    )
    bbox, score = det.detect(image)
    if bbox is not None:
        print(f"  hand bbox: {bbox.astype(int).tolist()}  score={score:.2f}")
    else:
        print("  no hand detected")
    return bbox


def hand_mask_from_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = bbox.astype(int)
    mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--hand-bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--description", default="spoon")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open("config/default.yaml"))
    cfg["device"] = args.device

    image = load_image(args.image)
    out_stem = args.image.parent / args.image.stem
    print(f"Image: {args.image}  shape={image.shape}")

    # --- Hand bbox ---
    if args.hand_bbox:
        hand_bbox = np.array(args.hand_bbox, dtype=float)
        print(f"  using manual hand bbox: {args.hand_bbox}")
    else:
        print("Detecting hand...")
        hand_bbox = detect_hand(image, cfg)
        if hand_bbox is None:
            print("ERROR: no hand found. Pass --hand-bbox X1 Y1 X2 Y2 manually.")
            return

    hand_mask = hand_mask_from_bbox(image, hand_bbox)

    # --- Method 1: GroundingDINO ---
    print(f"\nMethod 1: GroundingDINO ('{args.description}')")
    from pipeline.stages.s3_object_seg import _detect_with_grounding_dino
    box, score = _detect_with_grounding_dino(
        image, args.description, hand_mask, args.device,
        output_dir=args.image.parent,
        return_score=True,
    )
    if box is not None:
        print(f"  selected box={box}  combined={score:.3f}")
        from models.sam2_wrapper import SAM2Model
        sam2 = SAM2Model(
            checkpoint=cfg["sam2_checkpoint"],
            config=cfg.get("sam2_config", "sam2_hiera_large.yaml"),
            device=args.device,
        )
        gdino_mask = sam2.segment_with_box(image, box)
        save_overlay(image, hand_mask | gdino_mask, Path(f"{out_stem}_detections.png"),
                     color=(100, 149, 237))
    else:
        print("  no detection")

    # --- Method 2: SAM-2 hand-box prompt ---
    print("\nMethod 2: SAM-2 hand-box prompt")
    from models.sam2_wrapper import SAM2Model
    sam2 = SAM2Model(
        checkpoint=cfg["sam2_checkpoint"],
        config=cfg.get("sam2_config", "sam2_hiera_large.yaml"),
        device=args.device,
    )
    handsam_mask = sam2.segment_held_object(
        image, tuple(hand_bbox.astype(int)), hand_mask
    )
    if handsam_mask is not None:
        print(f"  mask pixels: {handsam_mask.sum()}")
        save_overlay(image, hand_mask | handsam_mask, Path(f"{out_stem}_handsam.png"),
                     color=(0, 220, 80))
    else:
        print("  returned empty mask")

    # --- Method 3: depth-weighted contact heuristic ---
    print("\nMethod 3: depth-weighted contact heuristic")
    try:
        from models.moge_wrapper import MoGeModel
        moge = MoGeModel(checkpoint=cfg["moge_checkpoint"], device=args.device)
        depth_map, _ = moge.estimate(image)
        print(f"  depth map: min={depth_map.min():.2f} max={depth_map.max():.2f}")
    except Exception as e:
        print(f"  MoGe failed ({e}), running without depth")
        depth_map = None

    from pipeline.stages.s3_object_seg import ObjectSegmentationStage
    stage = ObjectSegmentationStage(cfg)
    try:
        heuristic_mask = stage._seed_mask(image, hand_mask, depth_map=depth_map)
        print(f"  mask pixels: {heuristic_mask.sum()}")
        save_overlay(image, hand_mask | heuristic_mask, Path(f"{out_stem}_heuristic.png"),
                     color=(255, 165, 0))
    except RuntimeError as e:
        print(f"  heuristic failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
