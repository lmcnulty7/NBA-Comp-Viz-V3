#!/usr/bin/env python
"""
harvest_driver.py — resumable, watchdogged batch driver for harvest-scale runs.

The resilience story for unattended multi-hour Phase B runs (built after native
re-ID crash-looped mid-Phase-A and would have silently burned a night):

  • UNIT = one clip/section, STAGES = build → segment → matchups. Each stage runs
    as a subprocess with a WALL-CLOCK TIMEOUT — a hang (codec stall, GUI wait,
    crash-loop) kills that stage, marks it failed, and the batch moves on. No
    single unit can stall the night.
  • RESUME: per-unit status in data/harvest/status.json, written after every
    stage. Relaunching the driver skips completed stages (verified against the
    output file existing, not just the status flag — artifact over log line).
    Worst case lost to a driver crash = one stage of one unit.
  • RECORD: every stage's outcome (ok / failed / timeout, wall time, output
    path) is in the status file — a dead run tells you exactly where it died.
  • Alignment + join run at the end over whatever units succeeded.

For Phase B, downloaded games get split into ~10-min section files (ffmpeg
stream-copy) so resume granularity stays ~10 min; sections behave exactly like
the existing clip_XXmXX files. Boundary possessions at section edges are lost
(~5-8%, same window-edge behavior the eval characterized) — the price of
resumability.

Usage
  python harvest_driver.py --clips clip_40m00_48m00 curry_q1_clip   # rerun/resume
  python harvest_driver.py --clips ... --force build                # redo a stage
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("harvest_driver")

PY = sys.executable
HARVEST_DIR = config.PROJECT_ROOT / "data" / "harvest"
HARVEST_VIDEO = HARVEST_DIR / "video"
STATUS_PATH = HARVEST_DIR / "status.json"
STAGE_TIMEOUT_S = {"build": 45 * 60, "segment": 5 * 60, "matchups": 10 * 60}


def resolve_src(clip: str) -> Path:
    """Harvested games live in data/harvest/video/; legacy clips in the old raw dir."""
    p = HARVEST_VIDEO / f"{clip}.mp4"
    return p if p.exists() else config.VIDEO_DIR / f"{clip}.mp4"


def split_sections(tag: str, section_s: int = 600) -> list[str]:
    """Split a downloaded game into ~10-min section files (stream copy, no
    re-encode) — the resume/timeout unit. Returns section clip names.
    Idempotent: existing sections are kept."""
    src = HARVEST_VIDEO / f"{tag}.mp4"
    existing = sorted(HARVEST_VIDEO.glob(f"{tag}_s*.mp4"))
    if existing:
        return [p.stem for p in existing]
    out_pattern = str(HARVEST_VIDEO / f"{tag}_s%02d.mp4")
    r = subprocess.run(["ffmpeg", "-i", str(src), "-c", "copy", "-map", "0",
                        "-segment_time", str(section_s), "-f", "segment",
                        "-reset_timestamps", "1", out_pattern],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed for {tag}: {r.stderr[-300:]}")
    return [p.stem for p in sorted(HARVEST_VIDEO.glob(f"{tag}_s*.mp4"))]


def stage_cmd(stage: str, clip: str) -> list[str]:
    src = resolve_src(clip)
    return {
        "build": [PY, "build_trajectories.py", "--source", str(src), "--start", "0",
                  "--max-frames", "99999", "--stride", "3", "--pregate", "--no-video"],
        "segment": [PY, "segment_possessions.py", "--trajectories",
                    str(config.TRACKING_DIR / f"{clip}_trajectories.json")],
        "matchups": [PY, "matchup_metrics.py", "--clip", clip, "--no-video"],
    }[stage]


def stage_output(stage: str, clip: str) -> Path:
    return {
        "build": config.TRACKING_DIR / f"{clip}_trajectories.json",
        "segment": config.TRACKING_DIR / f"{clip}_possessions.json",
        "matchups": config.TRACKING_DIR / f"{clip}_matchups.json",
    }[stage]


def load_status() -> dict:
    return json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}


def save_status(status: dict) -> None:
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=1))


def run_unit(clip: str, status: dict, force: set[str]) -> bool:
    unit = status.setdefault(clip, {})
    for stage in ("build", "segment", "matchups"):
        rec = unit.get(stage, {})
        out = stage_output(stage, clip)
        if stage not in force and rec.get("state") == "ok" and out.exists():
            log.info("[%s] %s: done (skip)", clip, stage)
            continue
        log.info("[%s] %s: running (timeout %ds) …", clip, stage, STAGE_TIMEOUT_S[stage])
        t0 = time.time()
        try:
            r = subprocess.run(stage_cmd(stage, clip), capture_output=True, text=True,
                               timeout=STAGE_TIMEOUT_S[stage])
            state = "ok" if r.returncode == 0 and out.exists() else "failed"
            tail = (r.stderr or r.stdout or "")[-400:]
        except subprocess.TimeoutExpired:
            state, tail = "timeout", ""
        unit[stage] = {"state": state, "wall_s": round(time.time() - t0, 1),
                       "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       **({"tail": tail} if state != "ok" else {})}
        save_status(status)   # heartbeat: written after EVERY stage
        if state != "ok":
            log.warning("[%s] %s: %s — unit abandoned, batch continues", clip, stage, state)
            return False
        log.info("[%s] %s: ok (%.0fs)", clip, stage, unit[stage]["wall_s"])
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable harvest batch driver.")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--games", nargs="*", default=[],
                    help="Downloaded game tags — each is split into 10-min sections first.")
    ap.add_argument("--force", nargs="*", default=[], choices=["build", "segment", "matchups"],
                    help="Redo these stages even if marked done.")
    ap.add_argument("--no-align", action="store_true", help="Skip the alignment+join tail.")
    args = ap.parse_args()

    clips = list(args.clips)
    for tag in args.games:
        secs = split_sections(tag)
        log.info("%s: %d sections", tag, len(secs))
        clips += secs
    if not clips:
        raise SystemExit("Nothing to do — pass --clips and/or --games.")

    status = load_status()
    ok_clips = [c for c in clips if run_unit(c, status, set(args.force))]
    log.info("units complete: %d/%d", len(ok_clips), len(clips))

    if not args.no_align and ok_clips:
        for cmd, name in (([PY, "align_outcomes.py", "--clips"] + ok_clips, "align"),
                          ([PY, "tier2_join.py"], "join")):
            log.info("%s …", name)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print((r.stdout or "")[-1500:])
            if r.returncode != 0:
                log.warning("%s failed: %s", name, (r.stderr or "")[-400:])


if __name__ == "__main__":
    main()
