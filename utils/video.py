"""Video frame extraction utilities."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float | None = None,
) -> Path:
    """Extract frames from a video file to output_dir/frames/.

    Parameters
    ----------
    video_path : path to the input video (.mp4, .mov, .avi, etc.)
    output_dir : root output directory; frames are written to output_dir/frames/
    fps        : target frames-per-second to extract. None = extract every frame.
                 If fps exceeds the source fps, every frame is extracted.

    Returns
    -------
    Path to the frames directory (output_dir/frames/).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video input. "
            "Install it with: pip install opencv-python"
        )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute how many source frames to skip between each saved frame.
    if fps is None or fps >= source_fps:
        frame_interval = 1
        effective_fps = source_fps
    else:
        frame_interval = max(1, round(source_fps / fps))
        effective_fps = source_fps / frame_interval

    print(
        f"[video] {video_path.name}: {source_fps:.1f} fps, {total_frames} frames  →  "
        f"extracting every {frame_interval} frame(s) ({effective_fps:.2f} fps effective)"
    )

    frame_idx = 0
    saved = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(frames_dir / f"{saved:06d}.jpg", quality=95)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[video] saved {saved} frames to {frames_dir}")
    return frames_dir
