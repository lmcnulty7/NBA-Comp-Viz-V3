#!/usr/bin/env python
"""
matchup_metrics.py — Component C3 Tier 1: matchup assignment + defensive metrics.

Consumes trajectories + possessions JSON; computes, per metrics-eligible
possession (set CORE only — the region the C2 eval certified at 100%):

  • matchup assignment — Hungarian matching offense↔defense on court distance,
    per core frame. Switches show up as assignment changes; per-defender
    aggregation is by (defender, man) pair time.
  • matchup distance — per defender: time on each man, mean/median distance to
    the assigned man; the "primary man" is the most-assigned offender.
  • spacing conceded — offense convex-hull area per frame (needs ≥3 positioned
    offense players), reported as per-possession median + IQR.
  • closeout tendency — DIRECTIONAL only (closing / holding / retreating shares
    plus a median closing rate), computed on rolling-median-smoothed
    matchup-distance series over ≥1.5 s same-pair runs, rate via a 1 s central
    difference. ⚠ This is the noisiest metric in the set: it rides on
    frame-to-frame tracking jitter that the static distance measures average
    out. Treat it as a tendency signal, never as precise ft/s.

Gates (all reported, none silent): span must be metrics_eligible (core ≥ 4 s),
offense/defense confidence ≥ C3_MIN_SPAN_CONF, and only team-labeled tracks
participate (abstained tracks reduce matchup coverage — avg pairs/frame is the
coverage diagnostic; typical labeled lineups here are 4v3–4v4, not 5v5).

Review harness (built FIRST, per project practice): a top-down review video
per clip — team-colored dots, offense hull, matchup lines with distances —
so assignments can be eyeballed before any aggregate number is trusted.

Deferred by scope decision (DEVLOG 2026-07-07): help-position distance (encodes
an unverifiable positioning assumption) and off-ball attentiveness (needs real
ball position; a camera-pan proxy is not trustworthy). Tier 2 (possession-
outcome-adjusted defensive credit) depends on outcome tagging / a shot-quality
model — not built yet.

Example
  python matchup_metrics.py --clip curry_q1_clip
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

import config
from court.court33 import court_ft_to_px, draw_court_topdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("matchup_metrics")


def rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)
    h = k // 2
    return np.array([np.median(x[max(0, i - h): i + h + 1]) for i in range(len(x))])


def load(stem: str):
    traj = json.loads((config.TRACKING_DIR / f"{stem}_trajectories.json").read_text())
    poss = json.loads((config.TRACKING_DIR / f"{stem}_possessions.json").read_text())
    ident = json.loads((config.TRACKING_DIR / f"{stem}_identity.json").read_text())
    frames = defaultdict(list)      # frame -> [(tid, team, x, y)]
    for tid, rec in traj.items():
        team = rec.get("team")
        # raw-support filter: cleaned series interpolate a track's internal gaps
        # unmarked — a cleaned point counts only where the track was OBSERVED
        # (ghost audit: 20–30% of points were invented presence, DEVLOG 07-07c)
        raw_f = {p[0] for p in rec.get("raw", [])}
        for f, x, y, _ in rec["cleaned"]:
            if f in raw_f and np.isfinite([x, y]).all():
                frames[f].append((int(tid), team, float(x), float(y)))
    cols = {int(k): tuple(v) for k, v in ident.get("team_colors_bgr", {}).items()}
    return frames, poss, cols


def possession_metrics(frames: dict, span: dict, dt: float):
    """Per-possession Tier-1 metrics over the core frames. Returns (record, viz_frames)."""
    off_t, def_t = span["offense_team"], span["defense_team"]
    core = sorted(f for f in frames if span["core_start_frame"] <= f <= span["core_end_frame"])

    pair_time = defaultdict(float)              # (d_tid, o_tid) -> seconds assigned
    pair_series = defaultdict(list)             # (d_tid, o_tid) -> [(frame, dist)]
    def_dists = defaultdict(list)               # d_tid -> all assigned distances
    hull_areas, pairs_per_frame = [], []
    assignments_by_frame = {}                   # frame -> [(d_tid, o_tid, dist)]

    invalid_frames = []                          # team count > 5 = basketball-impossible
    count_dist = defaultdict(int)                # (|O|,|D|) -> n frames (coverage diagnostic)
    for f in core:
        O = [(tid, x, y) for tid, team, x, y in frames[f] if team == off_t]
        D = [(tid, x, y) for tid, team, x, y in frames[f] if team == def_t]
        # hard validity gate: >5 tracks on one team means corrupted input for this
        # frame (team misassignment or a residual duplicate) — exclude it from ALL
        # metrics rather than let Hungarian match against phantoms
        if len(O) > 5 or len(D) > 5:
            invalid_frames.append(f)
            continue
        count_dist[(len(O), len(D))] += 1
        if len(O) >= 3:
            try:
                hull_areas.append(ConvexHull(np.array([(x, y) for _, x, y in O])).volume)
            except Exception:
                pass
        if not O or not D:
            continue
        cost = np.array([[np.hypot(dx - ox, dy - oy) for ox, oy in [(o[1], o[2]) for o in O]]
                         for dx, dy in [(d[1], d[2]) for d in D]])
        ri, ci = linear_sum_assignment(cost)
        asg = []
        for r, c in zip(ri, ci):
            d_tid, o_tid, dist = D[r][0], O[c][0], float(cost[r, c])
            pair_time[(d_tid, o_tid)] += dt
            pair_series[(d_tid, o_tid)].append((f, dist))
            def_dists[d_tid].append(dist)
            asg.append((d_tid, o_tid, dist))
        assignments_by_frame[f] = asg
        pairs_per_frame.append(len(asg))

    # closeout tendency: smoothed same-pair runs only, central-difference rate
    k_smooth = max(1, int(config.C3_CLOSEOUT_SMOOTH_S / dt))
    w = max(1, int(config.C3_CLOSEOUT_WIN_S / dt / 2))
    min_run = int(config.C3_MIN_PAIR_RUN_S / dt)
    closing_rates = defaultdict(list)           # d_tid -> closing rates (ft/s, + = closing)
    for (d_tid, o_tid), pts in pair_series.items():
        pts.sort()
        run = [pts[0]]
        runs = []
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] <= 2 * (core[1] - core[0] if len(core) > 1 else 3):
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        runs.append(run)
        for r in runs:
            if len(r) < min_run:
                continue
            d = rolling_median(np.array([p[1] for p in r]), k_smooth)
            for i in range(w, len(d) - w):
                closing_rates[d_tid].append(-(d[i + w] - d[i - w]) / (2 * w * dt))

    defenders = []
    for d_tid, dists in sorted(def_dists.items()):
        men = {o: round(t, 1) for (dd, o), t in pair_time.items() if dd == d_tid}
        primary = max(men, key=men.get)
        rates = closing_rates.get(d_tid, [])
        rec = {"defender": d_tid, "primary_man": primary,
               "time_assigned_s": round(sum(men.values()), 1), "men": men,
               "matchup_dist_mean_ft": round(float(np.mean(dists)), 1),
               "matchup_dist_median_ft": round(float(np.median(dists)), 1)}
        if rates:
            r = np.array(rates)
            rec["closeout"] = {   # directional tendency, NOT precise ft/s (see module doc)
                "pct_closing": round(float((r > 0.5).mean()), 2),
                "pct_holding": round(float((np.abs(r) <= 0.5).mean()), 2),
                "pct_retreating": round(float((r < -0.5).mean()), 2),
                "median_rate_fts": round(float(np.median(r)), 2)}
        defenders.append(rec)

    record = {
        **{k: span[k] for k in ("start_frame", "set_start_frame", "end_frame",
                                "core_start_frame", "core_end_frame", "core_s",
                                "attacked_basket", "offense_team", "defense_team", "confidence")},
        "coverage_pairs_per_frame": round(float(np.mean(pairs_per_frame)), 1) if pairs_per_frame else 0.0,
        "frames_used": len(core) - len(invalid_frames),
        "frames_excluded_team_gt5": len(invalid_frames),
        "pct_excluded": round(100 * len(invalid_frames) / max(len(core), 1), 1),
        # low-trust flag: most of the core failed validity, or too little survived —
        # metrics exist but should be excluded from aggregates
        "degraded": bool(len(invalid_frames) > 0.4 * max(len(core), 1)
                         or (len(core) - len(invalid_frames)) < 30),
        "team_count_distribution": {f"{o}v{d}": n for (o, d), n in
                                    sorted(count_dist.items(), key=lambda kv: -kv[1])},
        "spacing_conceded_ft2": ({"median": round(float(np.median(hull_areas)), 0),
                                  "p25": round(float(np.percentile(hull_areas, 25)), 0),
                                  "p75": round(float(np.percentile(hull_areas, 75)), 0)}
                                 if hull_areas else None),
        "defenders": defenders,
    }
    return record, assignments_by_frame, set(invalid_frames)


def render_review(frames, poss_records, all_assignments, invalid_by_poss, cols, out_path, fps_eff):
    """Top-down review video: team dots, offense hull, matchup lines + distances.
    Frames excluded by the validity gate are shown WITH a red banner (not hidden),
    so what was dropped — and why — stays reviewable."""
    scale, margin = 9.0, 15
    writer = None
    for rec in poss_records:
        asg_map = all_assignments[rec["core_start_frame"]]
        invalid = invalid_by_poss[rec["core_start_frame"]]
        core = sorted(f for f in frames if rec["core_start_frame"] <= f <= rec["core_end_frame"])
        off_t, def_t = rec["offense_team"], rec["defense_team"]
        for f in core:
            img, _, _ = draw_court_topdown(scale, margin)
            if f in invalid:
                cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 120), -1)
                cv2.putText(img, f"frame {f} EXCLUDED — a team has >5 tracks (corrupt input)",
                            (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            pos = {tid: (x, y) for tid, team, x, y in frames[f]}
            O = [(tid, x, y) for tid, team, x, y in frames[f] if team == off_t]
            if len(O) >= 3:
                px = court_ft_to_px(np.array([(x, y) for _, x, y in O], np.float32), scale, margin)
                overlay = img.copy()
                cv2.fillPoly(overlay, [cv2.convexHull(px)], cols.get(off_t, (120, 120, 120)))
                img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
            for d_tid, o_tid, dist in asg_map.get(f, []):
                if d_tid in pos and o_tid in pos:
                    a = court_ft_to_px(np.array(pos[d_tid], np.float32), scale, margin)[0].astype(int)
                    b = court_ft_to_px(np.array(pos[o_tid], np.float32), scale, margin)[0].astype(int)
                    cv2.line(img, tuple(a), tuple(b), (160, 160, 160), 1, cv2.LINE_AA)
                    mid = ((a + b) // 2)
                    cv2.putText(img, f"{dist:.0f}", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (200, 200, 200), 1)
            for tid, team, x, y in frames[f]:
                if team not in (off_t, def_t):
                    continue
                p = court_ft_to_px(np.array((x, y), np.float32), scale, margin)[0].astype(int)
                col = cols.get(team, (150, 150, 150))
                if team == off_t:
                    cv2.circle(img, tuple(p), 7, col, -1, cv2.LINE_AA)
                    cv2.circle(img, tuple(p), 7, (255, 255, 255), 1, cv2.LINE_AA)  # offense ring
                else:
                    cv2.circle(img, tuple(p), 7, col, -1, cv2.LINE_AA)
                cv2.putText(img, str(tid), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (230, 230, 230), 1)
            cv2.putText(img, f"frame {f}  @{rec['attacked_basket']}  offense=team {off_t} "
                             f"(conf {rec['confidence']:.2f})  ring=offense",
                        (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 255, 60), 1)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps_eff, (w, h))
            writer.write(img)
    if writer:
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Component C3 Tier 1: matchup metrics.")
    ap.add_argument("--clip", type=str, default="curry_q1_clip")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    frames, poss, cols = load(args.clip)
    stride = poss.get("stride", 3)
    dt = stride / args.fps

    records, all_asg, invalid_by_poss = [], {}, {}
    skipped = []
    for span in poss["spans"]:
        if span["kind"] != "halfcourt":
            continue
        if not span.get("metrics_eligible"):
            skipped.append((span["set_start_frame"], "core_too_short"))
            continue
        if span.get("offense_team") is None:
            skipped.append((span["set_start_frame"], "offense_unknown"))
            continue
        if span.get("confidence", 0) < config.C3_MIN_SPAN_CONF:
            skipped.append((span["set_start_frame"], f"low_conf_{span['confidence']}"))
            continue
        rec, asg, invalid = possession_metrics(frames, span, dt)
        records.append(rec)
        all_asg[rec["core_start_frame"]] = asg
        invalid_by_poss[rec["core_start_frame"]] = invalid

    out = config.TRACKING_DIR / f"{args.clip}_matchups.json"
    out.write_text(json.dumps({"clip": args.clip, "possessions": records,
                               "skipped_spans": [{"set_start_frame": f, "reason": r}
                                                 for f, r in skipped]}, indent=2))

    log.info("── C3 TIER-1 MATCHUP METRICS — %s ──", args.clip)
    for rec in records:
        sp = rec["spacing_conceded_ft2"]
        log.info(" possession @%s (core %.1fs, offense team %s, conf %.2f) — "
                 "coverage %.1f pairs/frame, spacing median %s ft² | "
                 "frames: %d used, %d excluded (%.0f%%) [counts: %s]",
                 rec["attacked_basket"], rec["core_s"], rec["offense_team"],
                 rec["confidence"], rec["coverage_pairs_per_frame"],
                 sp["median"] if sp else "n/a",
                 rec["frames_used"], rec["frames_excluded_team_gt5"], rec["pct_excluded"],
                 dict(list(rec["team_count_distribution"].items())[:3]))
        for d in rec["defenders"]:
            co = d.get("closeout")
            co_s = (f"closing {co['pct_closing']:.0%}/holding {co['pct_holding']:.0%}/"
                    f"retreating {co['pct_retreating']:.0%}" if co else "n/a (runs too short)")
            log.info("   def %-4d on %-4d (%.1fs): dist median %4.1f ft | %s",
                     d["defender"], d["primary_man"], d["time_assigned_s"],
                     d["matchup_dist_median_ft"], co_s)
    for f, r in skipped:
        log.info(" skipped span @%d: %s", f, r)
    log.info("matchups → %s", out)

    if not args.no_video and records:
        out_mp4 = config.TRACKING_DIR / f"{args.clip}_matchups.mp4"
        render_review(frames, records, all_asg, invalid_by_poss, cols, out_mp4, args.fps / stride)
        log.info("review video → %s", out_mp4)


if __name__ == "__main__":
    main()
