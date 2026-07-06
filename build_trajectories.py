#!/usr/bin/env python
"""
build_trajectories.py — Component A: per-player court trajectories + correction.

Runs a clip through tracking (BoT-SORT) + homography (33-pt court), projects each
tracked player's foot-point to court feet per frame, then cleans the per-player
paths (jump rejection + smoothing — court/trajectories.py). This is the bridge from
"perception" to "analytics": clean court-space trajectories per player.

Outputs: data/tracking/<clip>_trajectories.json  and a top-down court image
(raw | cleaned) for visual review.

Examples
────────
  python build_trajectories.py --source ".../curry_q1_clip.mp4" --start 6000 --max-frames 300
  python build_trajectories.py --source ".../clip.mp4" --use-gate
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from court.court33 import COURT_LENGTH_FT, COURT_WIDTH_FT, court33_segments, court_ft_to_px, draw_court_topdown
from court.trajectories import clean_trajectories

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_trajectories")

# generous court bound (ft) — drop horizon blow-ups; clean_paths handles the rest
X_LO, X_HI = -25.0, COURT_LENGTH_FT + 25
Y_LO, Y_HI = -25.0, COURT_WIDTH_FT + 25


def id_color(tid):
    rng = np.random.default_rng(tid * 9973 + 1)
    return tuple(int(c) for c in rng.integers(70, 256, size=3))


def draw_paths(per_track_series, title):
    """per_track_series: {tid: [(frame, x, y, ...)]} → top-down court image with paths."""
    img, scale, margin = draw_court_topdown()
    for tid, pts in per_track_series.items():
        xy = np.array([(p[1], p[2]) for p in pts], np.float32)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 2:
            continue
        px = court_ft_to_px(xy, scale, margin)
        cv2.polylines(img, [px], False, id_color(tid), 1, cv2.LINE_AA)
    cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build + clean per-player court trajectories.")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--use-gate", action="store_true", help="Skip non-court frames via the gate.")
    # clean_paths params (defaults = the external project's basketball settings)
    ap.add_argument("--jump-sigma", type=float, default=3.5)
    ap.add_argument("--min-jump-dist", type=float, default=0.6)
    ap.add_argument("--max-jump-run", type=int, default=18)
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--smooth-poly", type=int, default=2)
    args = ap.parse_args()

    config.set_seed()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    from detect import PlayerTracker
    from court import CourtMapper

    device = config.get_device()
    gate = None
    if args.use_gate:
        from gate.backbones import get_backbone
        from gate.trained_head import TrainedHeadGate
        thr = json.loads(config.THRESHOLDS_PATH.read_text())["trained"]
        gate = TrainedHeadGate.load(config.HEAD_PATH, backbone=get_backbone("clip", device), threshold=thr)
    tracker = PlayerTracker(device=device)
    mapper = CourtMapper()

    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw = defaultdict(dict)       # {track_id: {frame_idx: (x, y)}}
    boxes_by_frame = {}           # {frame_idx: [(track_id, bbox)]}
    extrap = {}                   # {(track_id, frame_idx): bool}  — projection outside keypoint hull
    hull_by_frame = {}            # {frame_idx: keypoint convex hull (pixel)}
    hom_by_frame = {}             # {frame_idx: CourtHomography}  — for reprojecting the court model
    frame_order = []
    idx, n, n_H = args.start, 0, 0
    log.info("Tracking + projecting %s from frame %d …", args.source.name, args.start)
    while idx < total and n < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        if gate is not None and not gate.is_court_visible(frame):
            idx += args.stride
            continue
        tracks = tracker.update(frame, idx, idx / fps)
        mapper.update(frame)
        frame_order.append(idx)
        boxes_by_frame[idx] = [(t.track_id, [int(v) for v in t.bbox]) for t in tracks]
        hull_by_frame[idx] = mapper.keypoint_hull
        hom_by_frame[idx] = mapper.last_hom
        if mapper.has_homography:
            n_H += 1
            for t in tracks:
                c = mapper.court_pos(t.foot_point)
                if np.isfinite(c).all() and X_LO <= c[0] <= X_HI and Y_LO <= c[1] <= Y_HI:
                    raw[t.track_id][idx] = (float(c[0]), float(c[1]))
                    extrap[(t.track_id, idx)] = mapper.is_extrapolated(t.foot_point)
        idx += args.stride
        n += 1
    cap.release()

    raw_series = {tid: [(f, x, y) for f, (x, y) in sorted(d.items())] for tid, d in raw.items()}
    cleaned = clean_trajectories(
        raw, frame_order, jump_sigma=args.jump_sigma, min_jump_dist=args.min_jump_dist,
        max_jump_run=args.max_jump_run, smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)

    n_edited = sum(1 for pts in cleaned.values() for p in pts if p[3])
    n_pts = sum(len(p) for p in cleaned.values())
    log.info("%d frames (%d with homography) · %d tracks · %d trajectory points · %d corrected (%.1f%%)",
             n, n_H, len(cleaned), n_pts, n_edited, 100 * n_edited / max(n_pts, 1))

    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.TRACKING_DIR / f"{args.source.stem}_trajectories.json"
    out_json.write_text(json.dumps({
        str(tid): {"raw": raw_series.get(tid, []),
                   "cleaned": [[f, x, y, ed] for f, x, y, ed in pts]}
        for tid, pts in cleaned.items()}, indent=2))

    # ── synced review video: broadcast (boxes+IDs) | top-down minimap (cleaned dots) ──
    lookup = {tid: {f: (x, y) for f, x, y, _ in pts} for tid, pts in cleaned.items()}
    pos = {f: i for i, f in enumerate(frame_order)}
    scale, margin, trail = 9.0, 15, 12

    def minimap(frame_idx):
        img, _, _ = draw_court_topdown(scale, margin)
        h_mm = img.shape[0]
        i = pos[frame_idx]
        for tid, fmap in lookup.items():
            if frame_idx not in fmap or not np.isfinite(fmap[frame_idx]).all():
                continue
            col = id_color(tid)
            tp = [fmap[frame_order[k]] for k in range(max(0, i - trail), i + 1)
                  if frame_order[k] in fmap and np.isfinite(fmap[frame_order[k]]).all()]
            if len(tp) >= 2:
                cv2.polylines(img, [court_ft_to_px(np.array(tp, np.float32), scale, margin)], False, col, 1, cv2.LINE_AA)
            cx, cy = (int(v) for v in court_ft_to_px(fmap[frame_idx], scale, margin)[0])
            # solid = position constrained by nearby landmarks; hollow ring = extrapolated (unreliable)
            if extrap.get((tid, frame_idx), True):
                cv2.circle(img, (cx, cy), 6, col, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), 6, col, -1, cv2.LINE_AA)
            cv2.putText(img, str(tid), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(img, f"frame {frame_idx}", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
        cv2.putText(img, "solid=trusted  hollow=extrapolated", (10, h_mm - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        return img

    out_mp4 = config.TRACKING_DIR / f"{args.source.stem}_trajectories.mp4"
    if out_mp4.exists():
        out_mp4.unlink()
    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    writer = None
    log.info("Rendering synced review video (broadcast | court minimap) …")
    for f in frame_order:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        # reproject the court model (court→pixel) so the homography fit is visible:
        # green lines that don't sit on the real painted lines = the warp
        hom = hom_by_frame.get(f)
        if hom is not None and hom.is_valid:
            for P1, P2 in court33_segments():
                a = hom.to_pixel_batch(np.array([P1], np.float32))[0]
                b = hom.to_pixel_batch(np.array([P2], np.float32))[0]
                if np.isfinite([a[0], a[1], b[0], b[1]]).all():
                    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 1, cv2.LINE_AA)
        hull = hull_by_frame.get(f)
        if hull is not None:
            cv2.polylines(frame, [hull.astype(np.int32)], True, (0, 220, 255), 1, cv2.LINE_AA)
        for tid, bbox in boxes_by_frame.get(f, []):
            x1, y1, x2, y2 = bbox
            col = id_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, str(tid), (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        mm = minimap(f)
        mm = cv2.resize(mm, (int(mm.shape[1] * frame.shape[0] / mm.shape[0]), frame.shape[0]))
        combo = np.hstack([frame, mm])
        if writer is None:
            h, w = combo.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps / args.stride, (w, h))
        writer.write(combo)
    cap.release()
    if writer:
        writer.release()

    log.info("trajectories → %s", out_json)
    log.info("review video (broadcast | minimap) → %s", out_mp4)


if __name__ == "__main__":
    main()
