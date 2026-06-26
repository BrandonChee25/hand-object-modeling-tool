"""Command-line entry point.

Usage
-----
    python cli.py --frames path/to/frames/ --output path/to/output/
    python cli.py --frames frames/ --output out/ --config config/custom.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert exocentric hand-object video frames to a static 3D model."
    )
    p.add_argument(
        "--frames", required=True, type=Path,
        help="Directory containing sorted image frames (JPG/PNG).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Directory where hand.obj, object.obj, and scene.glb are written.",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/default.yaml"),
        help="YAML config file (default: config/default.yaml).",
    )
    p.add_argument(
        "--object-point", type=int, nargs=2, metavar=("X", "Y"), default=None,
        help="Pixel coordinate of a point on the held object in the anchor frame. "
             "Overrides automatic object detection. Use when auto-detection picks the wrong object.",
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
    runner = PipelineRunner(cfg)
    runner.run(args.frames, args.output)


if __name__ == "__main__":
    main()
