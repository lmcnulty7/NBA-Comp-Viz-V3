"""Shared fixtures: repo-root imports + a synthetic 60fps long-GOP test video."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def open_video(path) -> cv2.VideoCapture:
    """Same backend fallback as build_trajectories: FFMPEG first, then default
    (this Mac's cv2 has no FFMPEG backend and opens via AVFoundation)."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    return cap

# Synthetic video contract: frame i is solid gray with value (i * 13) % 251 —
# any misread frame is off by a multiple of 13, far above codec round-trip
# noise, so frame identity is checkable from pixels alone.
N_FRAMES, FPS, SIZE, GOP = 900, 60, 64, 300


def frame_value(i: int) -> int:
    return (i * 13) % 251


@pytest.fixture(scope="session")
def video_60fps(tmp_path_factory) -> Path:
    """900-frame 720p60-like stream in miniature: 60fps h264 with 300-frame
    GOPs (scene-cut keyframes disabled). Near-lossless qp — NOT qp 0, which
    flips x264 into the High 4:4:4 profile that cv2 builds can refuse to open;
    ±few luma error vs the 13-apart frame values still identifies frames."""
    out = tmp_path_factory.mktemp("videoseq") / "seq60.mp4"
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-qp", "12", "-g", str(GOP),
         "-sc_threshold", "0", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        p.stdin.write(np.full((SIZE, SIZE), frame_value(i), np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed to build the test video"
    return out
