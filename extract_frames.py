#!/usr/bin/env python
"""
extract_frames.py — Phase 1, step 1.

Sample frames from the source .mp4 clips and write them to
data/visibility/_unsorted/ for labeling. Filenames encode the source clip and
the absolute frame index so every frame is traceable back to its video.

Adapted from the old project's VideoReader pattern (FFmpeg backend forced to
avoid the macOS VideoToolbox/Metal segfault; stride-based seeking). Frames are
JPEG-encoded; no model is loaded here.

Examples
────────
  python extract_frames.py
  python extract_frames.py --interval-sec 1.0 --max-per-clip 200
  python extract_frames.py --clips "/path/a.mp4" "/path/b.mp4"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_frames")


def extract_from_clip(clip: Path, interval_sec: float, max_frames: int, out_dir: Path) -> int:
    # Force FFmpeg backend (macOS VideoToolbox holds Metal resources that clash
    # with torch/Metal later — see old src/ingestion/video_reader.py).
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        log.warning("Cannot open %s — skipping", clip.name)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(interval_sec * fps))
    log.info("%s | %.2f fps | %d frames | stride=%d (~%.2fs)", clip.name, fps, total, stride, stride / fps)

    written = 0
    idx = 0
    while idx < total and written < max_frames:
        # Seek to the exact frame we want, then read it (avoids decoding skips).
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        name = f"{clip.stem}__f{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
        idx += stride

    cap.release()
    log.info("%s → %d frames", clip.name, written)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frames from .mp4 clips into _unsorted/.")
    ap.add_argument("--video-dir", type=Path, default=config.VIDEO_DIR,
                    help=f"Directory of .mp4 clips (default: {config.VIDEO_DIR})")
    ap.add_argument("--clips", nargs="*", type=Path, default=None,
                    help="Specific clip paths (overrides --video-dir).")
    ap.add_argument("--interval-sec", type=float, default=1.5,
                    help="Seconds between sampled frames (default 1.5).")
    ap.add_argument("--max-per-clip", type=int, default=150,
                    help="Max frames to keep per clip (default 150).")
    ap.add_argument("--out", type=Path, default=config.UNSORTED_DIR)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = args.clips if args.clips else sorted(Path(args.video_dir).glob("*.mp4"))
    if not clips:
        log.error("No .mp4 files found in %s", args.video_dir)
        sys.exit(1)

    log.info("Extracting from %d clip(s) → %s", len(clips), args.out)
    total = sum(extract_from_clip(c, args.interval_sec, args.max_per_clip, args.out) for c in clips)
    n_files = len(list(args.out.glob("*.jpg")))
    log.info("Done. Wrote %d frames this run; %d total in %s", total, n_files, args.out)
    log.info("Next: python presort_frames.py")


if __name__ == "__main__":
    main()
