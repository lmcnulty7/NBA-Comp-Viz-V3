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
from qc.track_qc import score_build, summarize, physics, BUG_PAD, SECOND_MAX, frame_badness, tile, sheet
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


def worst_tiles(clip: str, n: int):
    """The n worst frames of one clip as tiles, read from the review render (frame k of the render =
    k-th processed frame). Returns [(badness, tile_image, clip, frame)]."""
    import cv2
    rep = json.loads((QC_DIR / (clip + ".json")).read_text())
    side = config.PROJECT_ROOT / "data" / "triage" / "side"
    sc = json.loads((side / (clip + "_frames.json")).read_text())
    order = [r["frame"] for r in sc["frames"]]
    pos = {f: k for k, f in enumerate(order)}
    sec_p = side / (clip + "_second.json")
    sec = {r["frame"]: [b["bbox"] for b in r["boxes"]] for r in json.loads(sec_p.read_text())["frames"]} if sec_p.exists() else {}
    srow = {r["frame"]: r for r in sc["frames"]}
    rows = sorted(rep["frames"], key=frame_badness, reverse=True)[:n]
    render = config.PROJECT_ROOT / "data" / "triage" / "renders" / (clip.replace("triage_", "") + ".mp4")
    cap = cv2.VideoCapture(str(render)); frames = {}
    want = {pos[r["frame"]] for r in rows if r["frame"] in pos}
    k = 0
    while want:
        ret, fr = cap.read()
        if not ret:
            break
        if k in want:
            frames[k] = fr; want.discard(k)
        k += 1
    cap.release()
    out = []
    for r in rows:
        k = pos.get(r["frame"])
        if k is None or k not in frames:
            continue
        out.append((frame_badness(r), tile(frames[k], r, sec.get(r["frame"]), srow.get(r["frame"])), clip, r["frame"]))
    return out


def make_sheets(n_per_clip: int = 6):
    """One sheet per clip (its 6 worst frames) and one per game (the 6 worst across its clips)."""
    import cv2
    from fetch_pbp import game_for_clip
    out_dir = config.REPORTS_DIR / "qc_sheets"; out_dir.mkdir(parents=True, exist_ok=True)
    by_game = {}
    for p in sorted(QC_DIR.glob("triage_*.json")):
        clip = p.stem
        tiles = worst_tiles(clip, n_per_clip)
        if not tiles:
            continue
        cv2.imwrite(str(out_dir / (clip + ".jpg")), sheet([t[1] for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        section = re.sub(r"_f\d+$", "", re.sub(r"^triage_", "", clip))
        by_game.setdefault(game_for_clip(section), []).extend(tiles)
        print("  sheet", clip, flush=True)
    for g, tiles in by_game.items():
        tiles.sort(key=lambda t: t[0], reverse=True)
        cv2.imwrite(str(out_dir / ("game_%s.jpg" % g)), sheet([t[1] for t in tiles[:n_per_clip]]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("sheets: %d clips, %d games -> %s" % (len(list(out_dir.glob("triage_*.jpg"))), len(by_game), out_dir))


def write_summary():
    """reports/qc_summary.{json,txt}: per clip and per game, every A1..A4 number, worst first."""
    from fetch_pbp import game_for_clip
    clips = []
    for p in sorted(QC_DIR.glob("triage_*.json")):
        rep = json.loads(p.read_text()); s = rep["summary"]
        section = re.sub(r"_f\d+$", "", re.sub(r"^triage_", "", p.stem))
        clips.append({"clip": p.stem, "game": game_for_clip(section), **s})
    games = {}
    for c in clips:
        g = games.setdefault(c["game"], {"clips": 0, "frames": 0, "frames_failing_any": 0, "rule_counts": {}, "dist_px": [], "second": []})
        g["clips"] += 1; g["frames"] += c["frames"]; g["frames_failing_any"] += c["frames_failing_any"]
        for k, v in c["rule_counts"].items():
            g["rule_counts"][k] = g["rule_counts"].get(k, 0) + v
        if c.get("dist_px_median") is not None: g["dist_px"].append(c["dist_px_median"])
        if c.get("second_disagreement_median") is not None: g["second"].append(c["second_disagreement_median"])
    import numpy as np
    for g in games.values():
        g["fail_rate"] = round(g["frames_failing_any"] / max(g["frames"], 1), 3)
        g["dist_px_median"] = round(float(np.median(g["dist_px"])), 1) if g["dist_px"] else None
        g["second_disagreement_median"] = round(float(np.median(g["second"])), 3) if g["second"] else None
        del g["dist_px"], g["second"]
    clips.sort(key=lambda c: -c["frames_failing_any"] / max(c["frames"], 1))
    out = {"method": "label-free indicators (FIX_PLAN Phase A) on the 55 triage renders; thresholds uncalibrated until B6",
           "clips": clips, "games": games}
    (config.REPORTS_DIR / "qc_summary.json").write_text(json.dumps(out, indent=1))
    L = ["QC SUMMARY (label-free, uncalibrated until B6): %d clips, %d frames" % (len(clips), sum(c["frames"] for c in clips)),
         "per game, worst first (fail rate = frames failing any live rule):"]
    for g, v in sorted(games.items(), key=lambda kv: -kv[1]["fail_rate"]):
        L.append("  %-14s clips %2d frames %4d fail %.2f lines %5s px 2nd %5s  %s" % (g, v["clips"], v["frames"], v["fail_rate"], v["dist_px_median"], v["second_disagreement_median"], v["rule_counts"]))
    L.append("per clip, worst first:")
    for c in clips:
        L.append("  %-40s fail %.2f lines %5s px 2nd %5s  %s" % (c["clip"], c["frames_failing_any"] / max(c["frames"], 1), c.get("dist_px_median"), c.get("second_disagreement_median"), c["rule_counts"]))
    txt = "\n".join(L); (config.REPORTS_DIR / "qc_summary.txt").write_text(txt + "\n"); print(txt[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", action="store_true")
    ap.add_argument("--sheets", action="store_true", help="worst-frame contact sheets per clip and per game (reports/qc_sheets/)")
    ap.add_argument("--summary", action="store_true", help="reports/qc_summary.{json,txt}")
    ap.add_argument("--sidecar", type=Path); ap.add_argument("--video", type=Path)
    ap.add_argument("--checks", default="overlay,geometry,physics,second", help="comma list; image checks read the video, physics only the json")
    a = ap.parse_args()
    if a.sheets:
        make_sheets(); return
    if a.summary:
        write_summary(); return
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
