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
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import config
from fetch_pbp import video_path

RUN_TOL_S = 1.0
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


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    secs = sorted(p.name.replace("_outcomes.json", "") for p in (config.PROJECT_ROOT / "data" / "pbp").glob(game + "_s*_outcomes.json"))
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
    (config.REPORTS_DIR / ("sportvu_sync_%s.json" % game)).write_text(json.dumps(rep, indent=1))
    print("totals:", tot)


if __name__ == "__main__":
    main()
