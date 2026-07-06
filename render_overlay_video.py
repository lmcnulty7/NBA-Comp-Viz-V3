#!/usr/bin/env python
"""
render_overlay_video.py — real-time court-projection overlay clips from the unseen
game sections, using the NEW-grid detector + per-frame line-snap refinement.

Per game: find the longest continuous HSV-gate-passing stretch across its three
sections (the best proxy for an uninterrupted live possession run), cap at CLIP_S
seconds, then for EVERY frame:

  grid model → H fit → geometric sanity check
    sane   → line-snap refine (guarded) → draw MAGENTA overlay   [status TRACK]
    insane → reuse the last good H for up to HOLD_S seconds, drawn YELLOW [HELD]
    held too long → no overlay                                    [LOST]

The HUD shows status + line residual so you can see exactly when the detector is
solving the frame itself vs coasting. Output: data/court_review/unseen_clips/*.mp4

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/anaconda3/bin/python render_overlay_video.py
"""
from __future__ import annotations
import glob, os
import cv2, numpy as np
from ultralytics import YOLO
from generate_labels import GRID_FT
from snap_projections import ridge_field, match_lines, project
from unseen_benchmark import GAMES, predict_kps, fit_H, h_sane, draw_court
from gate.hsv_baseline import HsvGate

cv2.setRNGSeed(42)
VID_DIR = "data/unseen_videos"
OUT_DIR = "data/court_review/unseen_clips"
MODEL = "models/court_grid_snapped.pt"
CLIP_S = 45.0        # max clip length
HOLD_S = 2.0         # how long a stale H may be reused
GATE_STEP = 0.5      # gate-scan resolution when hunting the live stretch
SNAP_RADII = (10, 6)
MIN_MATCH = 40
MAX_CORNER_PX = 40.0


def longest_live_stretch(path, gate):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    flags = []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        flags.append(bool(ok) and gate.is_court_visible(img))
        t += GATE_STEP
    cap.release()
    best_len = best_start = cur = 0
    for i, f in enumerate(flags):
        cur = cur + 1 if f else 0
        if cur > best_len:
            best_len, best_start = cur, i - cur + 1
    return best_start * GATE_STEP, best_len * GATE_STEP


def snap_refine(P, ridge, w, h):
    """Light guarded line-snap of a sane H (ft->px). Returns (P, res, n_match)."""
    P0 = P.copy()
    res = nm = None
    for radius in SNAP_RADII:
        ft, px = match_lines(P, ridge, w, h, radius)
        if len(ft) < MIN_MATCH:
            break
        Hn, _ = cv2.findHomography(ft.reshape(-1, 1, 2), px.reshape(-1, 1, 2),
                                   cv2.RANSAC, 3.0)
        if Hn is None:
            break
        P = Hn.astype(np.float64)
    ft, px = match_lines(P, ridge, w, h, 6)
    if len(ft):
        res, nm = float(np.median(np.linalg.norm(px - project(P, ft), axis=1))), len(ft)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    try:
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                         P @ np.linalg.inv(P0)).reshape(-1, 2)
        if not np.isfinite(moved).all() or np.linalg.norm(moved - corners, axis=1).max() > MAX_CORNER_PX:
            return P0, res, nm
    except np.linalg.LinAlgError:
        return P0, res, nm
    return P, res, nm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gate = HsvGate()
    model = YOLO(MODEL)

    by_game = {}
    for vp in sorted(glob.glob(os.path.join(VID_DIR, "*.mp4"))):
        vid = os.path.basename(vp).split("_s")[0]
        by_game.setdefault(vid, []).append(vp)

    for vid, paths in by_game.items():
        game = GAMES.get(vid, vid)
        best = (0.0, None, 0.0)                       # (length, path, start)
        for p in paths:
            start, length = longest_live_stretch(p, gate)
            if length > best[0]:
                best = (length, p, start)
        length, path, start = best
        length = min(length, CLIP_S)
        if path is None or length < 5:
            print(f"{game}: no usable live stretch, skipped")
            continue
        print(f"{game}: {os.path.basename(path)} @ {start:.0f}s for {length:.0f}s")

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        out_path = os.path.join(OUT_DIR, f"{game.replace(' ', '_').replace('@', 'at')}.mp4")
        writer = None
        lastP, last_age = None, 1e9
        n_frames = int(length * fps)
        counts = {"TRACK": 0, "HELD": 0, "LOST": 0}
        for k in range(n_frames):
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
            xy, conf = predict_kps(model, img)
            H, npts = fit_H(xy, conf, w, h, GRID_FT) if xy is not None else (None, 0)
            status, res, nm = "LOST", None, None
            if H is not None and h_sane(H, w, h):
                try:
                    P = np.linalg.inv(H.astype(np.float64))
                    ridge = ridge_field(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    if ridge is not None:
                        P, res, nm = snap_refine(P, ridge, w, h)
                    lastP, last_age, status = P, 0.0, "TRACK"
                except np.linalg.LinAlgError:
                    pass
            if status != "TRACK" and lastP is not None and last_age <= HOLD_S:
                status = "HELD"
            if status in ("TRACK", "HELD"):
                draw_court(img, lastP, (255, 0, 255) if status == "TRACK" else (0, 255, 255), 2)
            last_age += 1.0 / fps
            counts[status] += 1
            cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
            hud = f"{game}   NEW-grid + line-snap   {status}"
            if res is not None:
                hud += f"   line-res {res:.1f}px ({nm} matches)"
            col = {"TRACK": (140, 255, 140), "HELD": (0, 255, 255), "LOST": (0, 0, 255)}[status]
            cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            writer.write(img)
        cap.release()
        if writer is not None:
            writer.release()
        tot = max(sum(counts.values()), 1)
        print(f"  -> {out_path}   TRACK {100*counts['TRACK']//tot}%  HELD {100*counts['HELD']//tot}%  LOST {100*counts['LOST']//tot}%")


if __name__ == "__main__":
    main()
