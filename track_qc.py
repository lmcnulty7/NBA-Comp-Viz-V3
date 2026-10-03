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
from qc.track_qc import score_build, summarize

QC_DIR = config.REPORTS_DIR / "qc"


def run_one(sidecar_path: Path, video_path: Path) -> dict:
    sc = json.loads(sidecar_path.read_text())
    rows = score_build(video_path, sc)
    summ = summarize(rows)
    clip = sidecar_path.name.replace("_frames.json", "")
    QC_DIR.mkdir(parents=True, exist_ok=True)
    (QC_DIR / (clip + ".json")).write_text(json.dumps({"clip": clip, "video": str(video_path), "summary": summ, "frames": rows}, indent=1))
    return {"clip": clip, **summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", action="store_true")
    ap.add_argument("--sidecar", type=Path); ap.add_argument("--video", type=Path)
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
        r = run_one(s, v); res.append(r)
        print("  [%d/%d] %-40s dist %5s px  score %s  far %s" % (k + 1, len(jobs), r["clip"], r["dist_px_median"], r["score_median"], r["share_far"]), flush=True)
    res.sort(key=lambda r: (r["dist_px_median"] is None, -(r["dist_px_median"] or 0)))
    (QC_DIR / "_ranking.json").write_text(json.dumps(res, indent=1))
    print("worst first (median px from the lines):"); [print("  %-40s %6s px  score %s  far %s" % (r["clip"], r["dist_px_median"], r["score_median"], r["share_far"])) for r in res[:14]]


if __name__ == "__main__":
    main()
