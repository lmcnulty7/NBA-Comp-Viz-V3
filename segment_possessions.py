#!/usr/bin/env python
"""
segment_possessions.py — Component C2: possession segmentation (ball-free v1).

Consumes a clip's trajectories JSON (build_trajectories.py: court positions +
team per canonical track) and segments the timeline into HALFCOURT POSSESSION
spans: which basket is being attacked, and which team is on offense/defense.

Why this is possible without ball tracking:
  • which half — in a halfcourt set all ten players occupy one half; the
    (smoothed) median x of everyone on the floor says which. The band around
    midcourt is transition.
  • who attacks it — the basket in the occupied half is the attacked basket.
  • offense vs defense — defenders position themselves between their man and
    the defended basket, so over a span the DEFENSE's mean distance to the
    attacked basket is systematically smaller. Reported with a confidence =
    the fraction of span frames where that ordering holds; spans where the
    teams are unlabeled (abstained) get offense=None rather than a guess.

This is deliberately a separate stage reading build_trajectories' output file
(same stable-interface idiom as the rest of the pipeline) — perception runs
once, segmentation can be re-run instantly with different thresholds.

Outputs
  data/tracking/<clip>_possessions.json   spans + per-span evidence
  data/tracking/<clip>_possessions.png    timeline strip: occupancy signal,
                                          span shading, per-frame player counts
  console diagnostics

Example
  python segment_possessions.py --trajectories data/tracking/curry_q1_clip_trajectories.json
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
from court.court33 import COURT_LENGTH_FT

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("segment_possessions")

MID_X = COURT_LENGTH_FT / 2.0


def load_frames(traj_path: Path):
    """trajectories.json → {frame: [(team|None, x, y)]}, sorted frame list."""
    data = json.loads(traj_path.read_text())
    frames = defaultdict(list)
    for rec in data.values():
        team = rec.get("team")
        for f, x, y, _ in rec["cleaned"]:
            if np.isfinite([x, y]).all():
                frames[f].append((team, float(x), float(y)))
    return dict(frames), sorted(frames)


def occupancy_signal(frames: dict, order: list) -> np.ndarray:
    """Per-frame median x of everyone positioned (NaN when too few players)."""
    occ = np.full(len(order), np.nan)
    for i, f in enumerate(order):
        xs = [x for _, x, _ in frames[f]]
        if len(xs) >= config.POSS_MIN_PLAYERS:
            occ[i] = np.median(xs)
    return occ


def rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    """NaN-tolerant centered rolling median (k forced odd)."""
    k = max(1, k | 1)
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(len(x)):
        w = x[max(0, i - h): i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def segment(order: list, occ: np.ndarray, fps_eff: float):
    """Occupancy → labeled runs: 'L'/'R' halfcourt candidates, 'T' otherwise.
    Runs also break on frame gaps > POSS_MAX_GAP_SEC (gate skips, lost court)."""
    lo, hi = MID_X - config.POSS_HALF_MARGIN_FT, MID_X + config.POSS_HALF_MARGIN_FT
    state = np.where(np.isnan(occ), "T", np.where(occ < lo, "L", np.where(occ > hi, "R", "T")))
    max_gap = config.POSS_MAX_GAP_SEC * fps_eff  # in source-frame units
    runs, s = [], 0
    for i in range(1, len(order) + 1):
        if (i == len(order) or state[i] != state[s]
                or order[i] - order[i - 1] > max_gap):
            runs.append((state[s], s, i - 1))
            s = i
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment trajectories into possessions.")
    ap.add_argument("--trajectories", type=Path,
                    default=config.TRACKING_DIR / "curry_q1_clip_trajectories.json")
    ap.add_argument("--fps", type=float, default=30.0, help="Source video fps.")
    args = ap.parse_args()

    frames, order = load_frames(args.trajectories)
    if len(order) < 10:
        raise SystemExit("Too few frames in trajectories file.")
    stride = int(np.median(np.diff(order)))
    dt = stride / args.fps                     # seconds between processed frames
    occ = rolling_median(occupancy_signal(frames, order),
                         int(config.POSS_SMOOTH_SEC / dt))
    runs = segment(order, occ, args.fps)

    min_frames = config.POSS_MIN_SPAN_SEC / dt
    spans = []
    for lab, s, e in runs:
        f0, f1 = order[s], order[e]
        dur = (e - s + 1) * dt
        if lab == "T" or (e - s + 1) < min_frames:
            spans.append({"kind": "transition", "start_frame": f0, "end_frame": f1,
                          "duration_s": round(dur, 1)})
            continue
        basket = config.BASKET_LEFT if lab == "L" else config.BASKET_RIGHT
        # offense/defense: mean distance to the attacked basket per team, plus
        # per-frame agreement as the confidence
        dists = {0: [], 1: []}
        agree = []
        for i in range(s, e + 1):
            per = {0: [], 1: []}
            for team, x, y in frames[order[i]]:
                if team in (0, 1):
                    per[team].append(np.hypot(x - basket[0], y - basket[1]))
            if per[0] and per[1]:
                m0, m1 = np.mean(per[0]), np.mean(per[1])
                agree.append(0 if m0 < m1 else 1)   # closer team this frame
                dists[0].append(m0)
                dists[1].append(m1)
        span = {"kind": "halfcourt", "start_frame": f0, "end_frame": f1,
                "duration_s": round(dur, 1), "attacked_basket": "left" if lab == "L" else "right",
                "occupancy_x_med": round(float(np.nanmedian(occ[s:e + 1])), 1)}
        if agree:
            closer = int(np.bincount(agree, minlength=2).argmax())
            span.update({
                "defense_team": closer, "offense_team": 1 - closer,
                "confidence": round(float(np.mean(np.array(agree) == closer)), 2),
                "mean_dist_to_basket_ft": {t: round(float(np.mean(d)), 1)
                                           for t, d in dists.items() if d},
            })
        else:
            span.update({"defense_team": None, "offense_team": None, "confidence": 0.0})
        spans.append(span)

    # ── outputs ────────────────────────────────────────────────────────────────
    stem = args.trajectories.stem.replace("_trajectories", "")
    out_json = config.TRACKING_DIR / f"{stem}_possessions.json"
    out_json.write_text(json.dumps({
        "source": str(args.trajectories), "stride": stride, "fps": args.fps,
        "spans": spans}, indent=2))

    n_half = sum(1 for s in spans if s["kind"] == "halfcourt")
    t_half = sum(s["duration_s"] for s in spans if s["kind"] == "halfcourt")
    t_all = sum(s["duration_s"] for s in spans)
    log.info("── POSSESSION SPANS ──")
    for s in spans:
        if s["kind"] == "halfcourt":
            log.info("  %6d–%-6d %5.1fs  HALFCOURT @%s basket  offense=team %s (conf %.2f)",
                     s["start_frame"], s["end_frame"], s["duration_s"], s["attacked_basket"],
                     s.get("offense_team"), s.get("confidence", 0))
        else:
            log.info("  %6d–%-6d %5.1fs  transition/unknown",
                     s["start_frame"], s["end_frame"], s["duration_s"])
    log.info("%d halfcourt spans, %.1fs of %.1fs (%.0f%%) — rest transition/unknown",
             n_half, t_half, t_all, 100 * t_half / max(t_all, 1e-6))

    # timeline strip: occupancy trace over the court's long axis, spans shaded
    W, H = 1000, 220
    img = np.full((H, W, 3), 24, np.uint8)
    fx = lambda i: int(i / max(len(order) - 1, 1) * (W - 1))
    fy = lambda x: int(H - 20 - (x / COURT_LENGTH_FT) * (H - 50))
    for s in spans:
        i0 = order.index(s["start_frame"]); i1 = order.index(s["end_frame"])
        col = (45, 45, 45) if s["kind"] != "halfcourt" else \
              ((90, 60, 30) if s["attacked_basket"] == "left" else (30, 60, 90))
        img[30:H - 20, fx(i0):fx(i1) + 1] = col
    cv2.line(img, (0, fy(MID_X)), (W, fy(MID_X)), (90, 90, 90), 1)
    for lab, xv in (("mid", MID_X), ("L rim", config.BASKET_LEFT[0]), ("R rim", config.BASKET_RIGHT[0])):
        cv2.putText(img, lab, (4, fy(xv) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    pts = [(fx(i), fy(v)) for i, v in enumerate(occ) if np.isfinite(v)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (60, 255, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stem}: occupancy median-x (green) | orange=left-basket span, "
                     f"blue=right, gray=transition", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    out_png = config.TRACKING_DIR / f"{stem}_possessions.png"
    cv2.imwrite(str(out_png), img)

    log.info("spans → %s", out_json)
    log.info("timeline → %s", out_png)


if __name__ == "__main__":
    main()
