#!/usr/bin/env python
"""
run_tracking.py — run player detection + tracking over a video clip.

Produces three things:
  • an annotated .mp4 (boxes + persistent IDs, color-coded per track)
  • tracks.json (every track on every processed frame)
  • diagnostics.json + console summary (the automatic, label-free tracking health
    metrics: players/frame, track-length distribution, camera-cut resets, etc.)

Tracking needs CONTIGUOUS frames (IDs are propagated frame-to-frame), so this reads
the video in order at a fixed stride — unlike the gate's 1.5s-spaced sampling.

Optionally gates first (--use-gate): dead-ball frames are skipped (not tracked),
exactly as in the real pipeline, so you can see the gate + tracker working together.

Examples
────────
  python run_tracking.py --source ".../curry_q1_clip.mp4" --max-frames 300
  python run_tracking.py --source ".../curry_q1_clip.mp4" --use-gate --stride 3
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics as stats
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from detect import PlayerTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_tracking")


def id_color(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def draw(frame, tracks, frame_idx, gated_dead=False):
    out = frame.copy()
    if gated_dead:
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 90), -1)
        cv2.putText(out, "DEAD BALL — gate skipped (not tracked)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        c = id_color(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"#{t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(out, f"frame {frame_idx}  players={len(tracks)}", (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run player detection + tracking over a clip.")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--stride", type=int, default=3, help="Process every Nth frame (default 3 ≈ 10fps).")
    ap.add_argument("--max-frames", type=int, default=300, help="Max frames to process (default 300).")
    ap.add_argument("--use-gate", action="store_true", help="Skip dead-ball frames via the trained gate.")
    ap.add_argument("--out", type=Path, default=None, help="Output mp4 path.")
    ap.add_argument("--no-video", action="store_true", help="Skip writing the overlay mp4.")
    args = ap.parse_args()

    config.set_seed()
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    device = config.get_device()

    if not args.source.exists():
        raise SystemExit(f"Source video not found: {args.source}")

    # Optional gate
    gate = None
    if args.use_gate:
        from gate.backbones import get_backbone
        from gate.trained_head import TrainedHeadGate
        if not config.HEAD_PATH.exists():
            raise SystemExit("--use-gate needs a trained gate; run train_gate.py first.")
        thr = json.loads(config.THRESHOLDS_PATH.read_text())["trained"]
        log.info("Loading trained gate (threshold=%.3f) + CLIP backbone …", thr)
        gate = TrainedHeadGate.load(config.HEAD_PATH, backbone=get_backbone("clip", device), threshold=thr)

    log.info("Loading tracker …")
    tracker = PlayerTracker(device=device)

    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    out_path = args.out or (config.TRACKING_DIR / f"{args.source.stem}_tracked.mp4")

    per_frame = []                       # for tracks.json
    track_lengths = defaultdict(int)     # track_id -> frames seen
    players_per_frame = []
    n_processed = n_dead = 0

    idx = 0
    stem_msg = "gate+track" if args.use_gate else "track"
    log.info("Running (%s) on %s | stride=%d | up to %d frames …", stem_msg, args.source.name, args.stride, args.max_frames)
    while idx < total and n_processed < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        ts = idx / fps

        gated_dead = False
        if gate is not None and not gate.is_court_visible(frame):
            gated_dead = True
            n_dead += 1
            tracks = []
        else:
            tracks = tracker.update(frame, idx, ts)
            players_per_frame.append(len(tracks))
            for t in tracks:
                track_lengths[t.track_id] += 1
            per_frame.append({"frame_idx": idx, "timestamp_sec": round(ts, 3),
                              "tracks": [t.to_dict() for t in tracks]})
            n_processed += 1

        if not args.no_video:
            vis = draw(frame, tracks, idx, gated_dead=gated_dead)
            if writer is None:
                h, w = vis.shape[:2]
                # avc1 (H.264 via AVFoundation) is the codec that actually writes
                # readable .mp4 on this macOS OpenCV build (mp4v/XVID fail to open).
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps / args.stride, (w, h))
                if not writer.isOpened():
                    raise SystemExit("VideoWriter failed to open (codec issue). Try --no-video.")
            writer.write(vis)
        idx += args.stride

    cap.release()
    if writer is not None:
        writer.release()

    # ── Diagnostics (label-free tracking health) ──────────────────────────────
    lengths = list(track_lengths.values())
    diagnostics = {
        "source": str(args.source),
        "used_gate": bool(args.use_gate),
        "frames_processed": n_processed,
        "frames_skipped_dead": n_dead,
        "avg_players_per_frame": round(float(np.mean(players_per_frame)), 2) if players_per_frame else 0.0,
        "median_players_per_frame": int(np.median(players_per_frame)) if players_per_frame else 0,
        "max_players_in_a_frame": int(np.max(players_per_frame)) if players_per_frame else 0,
        "unique_track_ids": len(track_lengths),
        "track_length_median_frames": int(stats.median(lengths)) if lengths else 0,
        "track_length_mean_frames": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "short_tracks_lt5_frames": int(sum(1 for v in lengths if v < 5)),
        "camera_cut_resets": tracker.n_resets,
    }
    # Fragmentation proxy: many more unique IDs than typical on-court players ⇒ ID churn.
    typ = diagnostics["median_players_per_frame"] or 1
    diagnostics["id_churn_ratio"] = round(diagnostics["unique_track_ids"] / typ, 2)

    tracks_path = config.TRACKING_DIR / f"{args.source.stem}_tracks.json"
    diag_path = config.TRACKING_DIR / f"{args.source.stem}_diagnostics.json"
    tracks_path.write_text(json.dumps(per_frame, indent=2))
    diag_path.write_text(json.dumps(diagnostics, indent=2))

    log.info("── TRACKING DIAGNOSTICS ──")
    for k, v in diagnostics.items():
        log.info("  %-28s %s", k, v)
    if not args.no_video:
        log.info("overlay video → %s", out_path)
    log.info("tracks → %s", tracks_path)
    log.info("diagnostics → %s", diag_path)
    log.info("Sanity: avg players/frame should be ~8–12 on live NBA wide shots; "
             "id_churn_ratio near 1 = stable IDs, high = churn/ID switches.")


if __name__ == "__main__":
    main()
