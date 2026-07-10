"""Command-line entry point.

Usage
-----
    # Video input (extracts frames automatically):
    python cli.py --input video.mp4 --output out/
    python cli.py --input video.mp4 --output out/ --fps 6

    # Pre-extracted frames directory (original behaviour):
    python cli.py --input frames/ --output out/

    # Override config or object detection:
    python cli.py --input video.mp4 --output out/ --config config/custom.yaml
    python cli.py --input video.mp4 --output out/ --object-point 512 340
"""

from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a hand-object video (or frame directory) to a 3D model."
    )
    p.add_argument(
        "--input", required=True, type=Path,
        help="Path to a video file (.mp4/.mov/.avi/…) or a directory of sorted "
             "image frames (JPG/PNG). Videos are automatically subsampled to --fps.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Directory where hand.obj, object.obj, and scene.glb are written.",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="Frames per second to extract from video input (default: all frames). "
             "Ignored when --input is already a directory.",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/default.yaml"),
        help="YAML config file (default: config/default.yaml).",
    )
    p.add_argument(
        "--object-point", type=int, nargs=2, metavar=("X", "Y"), default=None,
        help="Pixel coordinate of a point on the held object in the anchor frame "
             "(x y). Use when the automatic SAM-2 hand-box prompt picks the wrong object.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.object_point:
        cfg["object_point"] = args.object_point

    args.output.mkdir(parents=True, exist_ok=True)

    # Resolve input: video file → extract frames; directory → use directly.
    input_path = args.input
    if input_path.is_file():
        from utils.video import extract_frames
        frames_dir = extract_frames(input_path, args.output, fps=args.fps)
    elif input_path.is_dir():
        if args.fps is not None:
            print("[cli] --fps is ignored when --input is a directory")
        frames_dir = input_path
    else:
        raise FileNotFoundError(f"--input path does not exist: {input_path}")

    runner = PipelineRunner(cfg)
    runner.run(frames_dir, args.output)


if __name__ == "__main__":
    main()
