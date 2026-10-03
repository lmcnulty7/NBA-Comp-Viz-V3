"""
sportvu/sync.py: production frame -> (period, game clock) per harvest section (FIX_PLAN B2),
then direction/offset against SportVU (B3, in a later pass).

B2 time map. Anchors come from data/pbp/<section>_outcomes.json (the clock OCR reads that
align_outcomes.py kept: frame, period, clock_s, in PRODUCTION frame numbers). Between two
consecutive anchors in the same period, the clock is RUNNING when the clock delta matches the frame
delta within RUN_TOL_S; those spans are interpolated linearly. Any other span (clock stopped, period
change, or disagreement) is excluded and counted. Frames are the section's processed frames
(the frames carrying raw positions in data/tracking/<section>_trajectories.json).

  python -m sportvu.sync gsw_phx_2016        -> data/sportvu/sync/<section>_timemap.json + reports/sportvu_sync_<game>.json

B3 direction and offset. The pipeline's court frame may be the mirror of SportVU's (which end is
x=0 depends on the grid model's labelling), and the OCR clock is read at 1 s resolution. For each
section, for each of the four mirror candidates (identity, flip x, flip y, both) and each offset in
OFFSET_STEP_S steps over +-OFFSET_MAX_S, project nothing (production raw positions are already in
court ft) and score the median over mapped frames of the median nearest-SportVU-player distance of
the pipeline's positions. Keep the best; if the best OTHER mirror is within AMBIG_FT the section is
ambiguous. Written to <section>_timemap.json under "direction" and to the game report.

  python -m sportvu.sync gsw_phx_2016 --direction 12.16.2015.PHX.at.GSW
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import config
from fetch_pbp import video_path

RUN_TOL_S = 1.0
OFFSET_MAX_S, OFFSET_STEP_S = 1.0, 0.04
AMBIG_FT = 2.0
FAR_FROM_MID_FT = 12.0        # positions farther than this from y=25 discriminate the y-flip
MIRRORS = {"identity": (False, False), "flip_x": (True, False), "flip_y": (False, True), "flip_xy": (True, True)}
SYNC_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "sync"


def section_fps(section: str) -> float:
    from harvest_driver import video_fps
    return video_fps(video_path(section))


def load_anchors(section: str) -> list[dict]:
    p = config.PROJECT_ROOT / "data" / "pbp" / (section + "_outcomes.json")
    if not p.exists():
        return []
    seen, out = set(), []
    for r in json.loads(p.read_text()):
        for a in r.get("anchors", []) or []:
            if a.get("clock_s") is None or a.get("period") is None:
                continue
            k = (a["frame"], a["period"], a["clock_s"])
            if k in seen:
                continue
            seen.add(k); out.append({"frame": int(a["frame"]), "period": int(a["period"]), "clock_s": float(a["clock_s"])})
    return sorted(out, key=lambda a: a["frame"])


def processed_frames(section: str) -> list[int]:
    p = config.TRACKING_DIR / (section + "_trajectories.json")
    if not p.exists():
        return []
    t = json.loads(p.read_text())
    return sorted({int(f) for tr in t.values() for f, _, _ in tr.get("raw", [])})


def build_timemap(section: str) -> dict:
    fps = section_fps(section)
    anchors = load_anchors(section)
    frames = processed_frames(section)
    spans, excluded = [], []
    for a, b in zip(anchors, anchors[1:]):
        dt_video = (b["frame"] - a["frame"]) / fps
        dt_clock = a["clock_s"] - b["clock_s"]
        if a["period"] != b["period"]:
            excluded.append({"from": a["frame"], "to": b["frame"], "why": "period_change", "video_s": round(dt_video, 1)})
        elif dt_video <= 0:
            continue
        elif abs(dt_video - dt_clock) <= RUN_TOL_S:
            spans.append({"f0": a["frame"], "f1": b["frame"], "period": a["period"], "c0": a["clock_s"], "c1": b["clock_s"],
                          "video_s": round(dt_video, 2), "clock_s": round(dt_clock, 2)})
        else:
            excluded.append({"from": a["frame"], "to": b["frame"], "why": "stopped_or_disagree",
                             "video_s": round(dt_video, 1), "clock_s": round(dt_clock, 1)})
    rows = []
    for f in frames:
        for s in spans:
            if s["f0"] <= f <= s["f1"]:
                u = (f - s["f0"]) / (s["f1"] - s["f0"])
                rows.append([f, s["period"], round(s["c0"] + u * (s["c1"] - s["c0"]), 3)])
                break
    mapped_s = sum(s["video_s"] for s in spans)
    excl_s = sum(e["video_s"] for e in excluded)
    return {"section": section, "fps": fps, "anchors": len(anchors), "processed_frames": len(frames),
            "mapped_frames": len(rows), "running_spans": len(spans), "mapped_video_s": round(mapped_s, 1),
            "excluded_spans": len(excluded), "excluded_video_s": round(excl_s, 1),
            "excluded_why": {w: sum(1 for e in excluded if e["why"] == w) for w in {e["why"] for e in excluded}},
            "spans": spans, "excluded": excluded, "frames": rows}


# ── B3 ────────────────────────────────────────────────────────────────────────
class SportVUIndex:
    """Per quarter: clocks (descending play order) and 10x2 player positions; nearest-clock lookup."""

    def __init__(self, moments: list[dict]):
        self.q = {}
        for q in sorted({m["q"] for m in moments}):
            ms = [m for m in moments if m["q"] == q and len(m["players"]) == 10]
            ms.sort(key=lambda m: -m["clock"])
            self.q[q] = {"clock": np.array([m["clock"] for m in ms]),
                         "xy": np.array([[p[2:4] for p in m["players"]] for m in ms], np.float32),
                         "team": np.array([[p[0] for p in m["players"]] for m in ms]),
                         "pid": np.array([[p[1] for p in m["players"]] for m in ms])}

    def at(self, q: int, clock: float, tol: float = 0.1):
        """Index of the moment nearest to `clock` in quarter q, or None if farther than tol."""
        d = self.q.get(q)
        if d is None:
            return None
        i = int(np.argmin(np.abs(d["clock"] - clock)))
        return i if abs(d["clock"][i] - clock) <= tol else None


def pipeline_positions(section: str) -> dict:
    """{frame: (n,2) raw court positions} from the production trajectories."""
    t = json.loads((config.TRACKING_DIR / (section + "_trajectories.json")).read_text())
    by = {}
    for tr in t.values():
        for f, x, y in tr.get("raw", []):
            by.setdefault(int(f), []).append((x, y))
    return {f: np.array(v, np.float32) for f, v in by.items()}


def apply_mirror(xy: np.ndarray, fx: bool, fy: bool) -> np.ndarray:
    from court.court33 import COURT_LENGTH_FT as L, COURT_WIDTH_FT as W
    out = xy.copy()
    if fx: out[:, 0] = L - out[:, 0]
    if fy: out[:, 1] = W - out[:, 1]
    return out


def solve_direction(section: str, index: SportVUIndex, timemap: dict) -> dict:
    pos = pipeline_positions(section)
    rows = [r for r in timemap["frames"] if r[0] in pos and len(pos[r[0]]) >= 3]
    if len(rows) < 30:
        return {"status": "too_few_frames", "frames": len(rows)}
    offsets = np.arange(-OFFSET_MAX_S, OFFSET_MAX_S + 1e-9, OFFSET_STEP_S)
    scores = {}
    for name, (fx, fy) in MIRRORS.items():
        per_off = []
        for off in offsets:
            med = []
            for f, q, clock in rows:
                i = index.at(q, clock + off)
                if i is None:
                    continue
                sv = index.q[q]["xy"][i]
                p = apply_mirror(pos[f], fx, fy)
                d = np.sqrt(((p[:, None, :] - sv[None, :, :]) ** 2).sum(-1)).min(1)
                med.append(float(np.median(d)))
            per_off.append(float(np.median(med)) if med else np.inf)
        k = int(np.argmin(per_off))
        scores[name] = {"offset_s": round(float(offsets[k]), 2), "median_ft": round(per_off[k], 2), "frames": len(rows)}
    best = min(scores, key=lambda n: scores[n]["median_ft"])
    runner = min((n for n in scores if n != best), key=lambda n: scores[n]["median_ft"])
    margin = scores[runner]["median_ft"] - scores[best]["median_ft"]
    # the y-flip is structurally hard to see with nearest-neighbour matching (a player at y=20 only
    # moves to y=30), so also score it on positions far from the mid-line, where it must show
    far = {}
    off = scores[best]["offset_s"]
    for name, (fx, fy) in MIRRORS.items():
        d_all = []
        for f, q, clock in rows:
            i = index.at(q, clock + off)
            if i is None:
                continue
            p = apply_mirror(pos[f], fx, fy); p = p[np.abs(p[:, 1] - 25.0) > FAR_FROM_MID_FT]
            if len(p):
                d_all.extend(np.sqrt(((p[:, None, :] - index.q[q]["xy"][i][None]) ** 2).sum(-1)).min(1).tolist())
        far[name] = round(float(np.median(d_all)), 2) if d_all else None
    far_best = min((n for n in far if far[n] is not None), key=lambda n: far[n]) if any(v is not None for v in far.values()) else None
    return {"status": "ambiguous" if margin < AMBIG_FT else "ok", "mirror": best, "offset_s": off,
            "median_ft": scores[best]["median_ft"], "runner_up": runner, "margin_ft": round(margin, 2),
            "far_from_mid_ft": far, "far_best": far_best, "candidates": scores}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="gsw_phx_2016")
    ap.add_argument("--direction", metavar="SPORTVU_GAME", help="B3: solve mirror + offset per section against this SportVU game")
    a = ap.parse_args()
    game = a.game
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    secs = sorted(p.name.replace("_outcomes.json", "") for p in (config.PROJECT_ROOT / "data" / "pbp").glob(game + "_s*_outcomes.json"))
    rep_path = config.REPORTS_DIR / ("sportvu_sync_%s.json" % game)
    if a.direction:
        moments = json.loads((config.PROJECT_ROOT / "data" / "sportvu" / (a.direction + "_moments.json")).read_text())["moments"]
        index = SportVUIndex(moments)
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {"game": game, "sections": {}}
        rep["sportvu_game"] = a.direction; rep["direction"] = {}
        for s in secs:
            tp = SYNC_DIR / (s + "_timemap.json")
            tm = json.loads(tp.read_text())
            d = solve_direction(s, index, tm)
            tm["direction"] = d; tp.write_text(json.dumps(tm)); rep["direction"][s] = d
            print("  %-20s %-10s mirror %-8s offset %+.2f s  median %.2f ft  margin %.2f ft (runner %s)" % (
                s, d["status"], d.get("mirror", "-"), d.get("offset_s", 0), d.get("median_ft", -1), d.get("margin_ft", -1), d.get("runner_up", "-")) if d["status"] != "too_few_frames" else "  %-20s too few frames (%d)" % (s, d["frames"]), flush=True)
        # game-level resolution: a per-section margin under AMBIG_FT is tolerated when every section
        # picks the same mirror on BOTH statistics (the y-flip is structurally weak, see solve_direction)
        picks = {d["mirror"] for d in rep["direction"].values() if d["status"] != "too_few_frames"}
        far_picks = {d.get("far_best") for d in rep["direction"].values() if d["status"] != "too_few_frames"}
        agree = len(picks) == 1 and picks == far_picks
        rep["direction_resolution"] = {"mirror": next(iter(picks)) if agree else None,
                                       "rule": "all sections agree on the mirror under both the all-positions and the far-from-mid-line statistics",
                                       "sections_agree": agree, "sections": len(rep["direction"])}
        print("game resolution:", rep["direction_resolution"])
        rep_path.write_text(json.dumps(rep, indent=1))
        return
    rep = {"game": game, "sections": {}}
    for s in secs:
        tm = build_timemap(s)
        (SYNC_DIR / (s + "_timemap.json")).write_text(json.dumps(tm))
        rep["sections"][s] = {k: tm[k] for k in ("anchors", "processed_frames", "mapped_frames", "running_spans", "mapped_video_s", "excluded_spans", "excluded_video_s", "excluded_why")}
        print("  %-20s anchors %3d  processed %5d  mapped %5d (%.1f s in %d spans)  excluded %d spans %.1f s %s" % (
            s, tm["anchors"], tm["processed_frames"], tm["mapped_frames"], tm["mapped_video_s"], tm["running_spans"], tm["excluded_spans"], tm["excluded_video_s"], tm["excluded_why"]), flush=True)
    tot = {k: sum(v[k] for v in rep["sections"].values()) for k in ("processed_frames", "mapped_frames", "mapped_video_s", "excluded_video_s")}
    rep["totals"] = tot
    config.REPORTS_DIR.mkdir(exist_ok=True)
    rep_path.write_text(json.dumps(rep, indent=1))
    print("totals:", tot)


if __name__ == "__main__":
    main()
