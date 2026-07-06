#!/usr/bin/env python
"""
visualize_masking.py — render court-masking onto a clip for YOU to review.

For each frame it overlays everything you need to judge masking accuracy by eye:
  • the projected court template      (GREEN = confident homography, ORANGE = low-conf)
  • the mask boundary polygon         (CYAN) — the exact line used to keep/drop
  • kept player boxes                 (player-colored, with #id and court (x,y) ft)
  • dropped detections                (RED, labeled "off-court") — drawn, not hidden,
                                        so you can see every call masking makes
  • an H-status line + a legend

Outputs a scrubable .mp4 AND full-res annotated PNGs you can flip through.

Usage
─────
  python visualize_masking.py --source ".../curry_q1_clip.mp4"
  python visualize_masking.py --source ".../clip.mp4" --start 9000 --max-frames 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from court.court33 import court33_segments


def id_color(tid: int):
    rng = np.random.default_rng(tid * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def draw_legend(img):
    items = [("court (confident H)", (60, 230, 60)), ("court (low-conf, mask off)", (40, 170, 235)),
             ("mask boundary", (0, 200, 255)), ("DROPPED off-court", (40, 40, 235)),
             ("kept player", (230, 230, 230))]
    x0, y0 = img.shape[1] - 250, img.shape[0] - 18 * len(items) - 10
    cv2.rectangle(img, (x0 - 8, y0 - 16), (img.shape[1] - 4, img.shape[0] - 4), (0, 0, 0), -1)
    for i, (label, col) in enumerate(items):
        y = y0 + i * 18
        cv2.line(img, (x0, y - 4), (x0 + 22, y - 4), col, 3)
        cv2.putText(img, label, (x0 + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1)


def render(frame, idx, on_tracks, off_tracks, mapper):
    out = frame.copy()
    if mapper.has_homography:
        line_col = (60, 230, 60) if mapper.confident else (40, 170, 235)
        for P1, P2 in court33_segments():
            seg = mapper.last_hom.to_pixel_batch(np.array([P1, P2], np.float32))
            if np.isfinite(seg).all():
                cv2.polylines(out, [seg.astype(np.int32)], False, line_col, 1, cv2.LINE_AA)
        mp = mapper.court_polygon_px()
        if mp is not None:
            cv2.polylines(out, [mp.astype(np.int32)], True, (0, 200, 255), 2, cv2.LINE_AA)
        h = mapper.last_hom
        status = (f"H {'CONFIDENT (masking ON)' if mapper.confident else 'low-conf (masking OFF)'}"
                  f"  inliers={h.n_inliers}  reproj={h.quality:.2f}ft")
    else:
        status = "H: none (too few keypoints) — masking OFF"
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, f"frame {idx}   {status}   kept={len(on_tracks)} dropped={len(off_tracks)}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)

    for t in off_tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 235), 2)
        cv2.putText(out, "off-court", (x1, y2 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 235), 1)
    for t in on_tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        c = id_color(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"#{t.track_id}", (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cp = getattr(t, "court_pos", None)
        if cp is not None and np.isfinite(cp).all():
            cv2.putText(out, f"({cp[0]:.0f},{cp[1]:.0f})", (x1, y2 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
    draw_legend(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Render court-masking onto a clip for visual review.")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--start", type=int, default=0, help="First source-frame index.")
    ap.add_argument("--max-frames", type=int, default=200, help="Max frames to render.")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--margin", type=float, default=None, help="Court margin (ft) override.")
    ap.add_argument("--use-gate", action="store_true",
                    help="Only render live court frames (skip intro/replay/crowd via the trained gate).")
    ap.add_argument("--no-refine", action="store_true",
                    help="Disable line-based homography refinement (keypoints only), for A/B comparison.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    config.set_seed()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    from detect import PlayerTracker
    from court import CourtMapper

    device = config.get_device()
    tracker = PlayerTracker(device=device)
    mapper = CourtMapper(margin_ft=args.margin)   # 33-pt scheme; --no-refine kept as no-op for now

    gate = None
    if args.use_gate:
        import json
        from gate.backbones import get_backbone
        from gate.trained_head import TrainedHeadGate
        thr = json.loads(config.THRESHOLDS_PATH.read_text())["trained"]
        gate = TrainedHeadGate.load(config.HEAD_PATH, backbone=get_backbone("clip", device), threshold=thr)

    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_norefine" if args.no_refine else ""   # so A/B runs don't overwrite each other
    out_mp4 = args.out or (config.TRACKING_DIR / f"{args.source.stem}_masking{suffix}.mp4")
    if out_mp4.exists():
        out_mp4.unlink()        # AVFoundation refuses to overwrite → delete first
    frames_dir = config.TRACKING_DIR / f"{args.source.stem}_masking{suffix}_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.png"):
            old.unlink()        # clear stale frames so the folder reflects this run
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    idx, n = args.start, 0
    n_confident = n_dropped = 0
    while idx < total and n < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        if gate is not None and not gate.is_court_visible(frame):
            idx += args.stride
            continue            # skip non-court frames (intro/replay/crowd)
        tracks = tracker.update(frame, idx, idx / fps)
        mapper.update(frame)
        on, off = mapper.split_tracks(tracks)
        n_confident += int(mapper.confident)
        n_dropped += len(off)
        vis = render(frame, idx, on, off, mapper)
        cv2.imwrite(str(frames_dir / f"f{idx:06d}.png"), vis)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps / args.stride, (w, h))
        writer.write(vis)
        idx += args.stride
        n += 1

    cap.release()
    if writer:
        writer.release()
    print(f"Rendered {n} frames  |  {n_confident} had a confident homography  |  {n_dropped} detections dropped")
    print(f"  video  → {out_mp4}")
    print(f"  frames → {frames_dir}/   (full-res PNGs to flip through)")


if __name__ == "__main__":
    main()
