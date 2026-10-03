#!/usr/bin/env python
"""
track_qc.py: run the label-free quality checks (qc/track_qc.py) over builds and write reports.

  --triage          every data/triage/side/<clip>_frames.json against data/triage/src/<clip>.mp4
  --sidecar S --video V   one build
Writes reports/qc/<clip>.json (per-frame rows + summary) and prints a worst-first ranking.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import config
from qc.track_qc import score_build, summarize, physics, BUG_PAD, SECOND_MAX
from qc.second_tracker import run_second, disagreement
from clock_reader import LAYOUTS, layout_for_clip
import re


def bug_rect_for(clip: str):
    """Scorebug rectangle for a clip: union of the layout's clock and period boxes, padded."""
    section = re.sub(r"^triage_", "", clip); section = re.sub(r"_f\d+$", "", section)
    try:
        lay = LAYOUTS[layout_for_clip(section)]
    except KeyError:
        return None
    xs = [lay["period_box"][0], lay["period_box"][2], lay["clock_box"][0], lay["clock_box"][2]]
    ys = [lay["period_box"][1], lay["period_box"][3], lay["clock_box"][1], lay["clock_box"][3]]
    return (min(xs) - BUG_PAD, min(ys) - BUG_PAD, max(xs) + BUG_PAD, max(ys) + BUG_PAD)

QC_DIR = config.REPORTS_DIR / "qc"


def run_one(sidecar_path: Path, video_path: Path, checks: tuple) -> dict:
    """Run the requested check groups on one build; merge into any existing report by frame."""
    sc = json.loads(sidecar_path.read_text())
    clip = sidecar_path.name.replace("_frames.json", "")
    QC_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = QC_DIR / (clip + ".json")
    rows = {r["frame"]: r for r in json.loads(rep_path.read_text())["frames"]} if rep_path.exists() else {}
    image_checks = tuple(c for c in checks if c in ("overlay", "geometry"))
    if image_checks:
        for r in score_build(video_path, sc, checks=image_checks, bug_rect=bug_rect_for(clip)):
            rows.setdefault(r["frame"], {"frame": r["frame"], "state": r["state"]}).update(r)
    if "physics" in checks:
        traj = json.loads(sidecar_path.with_name(clip + "_trajectories.json").read_text())
        ident_p = sidecar_path.with_name(clip + "_identity.json")
        ident = json.loads(ident_p.read_text()) if ident_p.exists() else {}
        ph = physics(traj, sc, ident, sc["fps"], sc["stride"])
        for f, v in ph["frames"].items():
            rows.setdefault(f, {"frame": f, "state": next(x["state"] for x in sc["frames"] if x["frame"] == f)})["physics"] = v
    if "second" in checks:
        sec = run_second(video_path, sc, sidecar_path.with_name(clip + "_second.json"))
        second_by_frame = {r["frame"]: r["boxes"] for r in sec["frames"]}
        for r in sc["frames"]:
            f = r["frame"]
            if f not in second_by_frame:
                continue
            d = disagreement([b["bbox"] for b in r["boxes"]], [b["bbox"] for b in second_by_frame[f]])
            d["fails"] = ["second_tracker"] if d["disagreement"] > SECOND_MAX else []
            rows.setdefault(f, {"frame": f, "state": r["state"]})["second"] = d
    rows = [rows[f] for f in sorted(rows)]
    summ = summarize(rows)
    rep_path.write_text(json.dumps({"clip": clip, "video": str(video_path), "summary": summ, "frames": rows}, indent=1))
    return {"clip": clip, **summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", action="store_true")
    ap.add_argument("--sidecar", type=Path); ap.add_argument("--video", type=Path)
    ap.add_argument("--checks", default="overlay,geometry,physics,second", help="comma list; image checks read the video, physics only the json")
    a = ap.parse_args()
    jobs = []
    if a.triage:
        side = config.PROJECT_ROOT / "data" / "triage" / "side"
        for s in sorted(side.glob("*_frames.json")):
            jobs.append((s, config.PROJECT_ROOT / "data" / "triage" / "src" / (s.name.replace("_frames.json", "") + ".mp4")))
    elif a.sidecar and a.video:
        jobs.append((a.sidecar, a.video))
    else:
        ap.error("--triage or --sidecar + --video")
    res = []
    for k, (s, v) in enumerate(jobs):
        r = run_one(s, v, tuple(a.checks.split(","))); res.append(r)
        print("  [%d/%d] %-40s dist %5s px  score %s  far %s" % (k + 1, len(jobs), r["clip"], r["dist_px_median"], r["score_median"], r["share_far"]), flush=True)
    res.sort(key=lambda r: (r["dist_px_median"] is None, -(r["dist_px_median"] or 0)))
    (QC_DIR / "_ranking.json").write_text(json.dumps(res, indent=1))
    from fetch_pbp import game_for_clip
    by_game = {}
    for r in res:
        section = re.sub(r"_f\d+$", "", re.sub(r"^triage_", "", r["clip"]))
        g = by_game.setdefault(game_for_clip(section), {"clips": 0, "frames": 0, "frames_failing_any": 0, "rule_counts": {}})
        g["clips"] += 1; g["frames"] += r["frames"]; g["frames_failing_any"] += r["frames_failing_any"]
        for k, v in r["rule_counts"].items():
            g["rule_counts"][k] = g["rule_counts"].get(k, 0) + v
    for g in by_game.values():
        g["fail_rate"] = round(g["frames_failing_any"] / max(g["frames"], 1), 3)
    (QC_DIR / "_by_game.json").write_text(json.dumps(by_game, indent=1))
    print("per game (frames failing any live rule):")
    for g, v in sorted(by_game.items(), key=lambda kv: -kv[1]["fail_rate"]):
        print("  %-14s clips %2d  frames %4d  fail %.2f  %s" % (g, v["clips"], v["frames"], v["fail_rate"], v["rule_counts"]))
    print("worst first (median px from the lines):"); [print("  %-40s %6s px  score %s  far %s" % (r["clip"], r["dist_px_median"], r["score_median"], r["share_far"])) for r in res[:14]]


if __name__ == "__main__":
    main()
