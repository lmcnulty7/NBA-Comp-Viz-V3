"""
sportvu/rebuild.py: rebuild the mapped spans of a game LOCALLY with the A0 sidecar (FIX_PLAN B4).

Production never saved pixel feet, and local section files are not frame-aligned with production,
so the SportVU comparison runs on a local rebuild: for each section, the longest running spans of
the local time map (up to BUDGET_S seconds per section) are cut with ffmpeg into h264 windows
(data/sportvu/build/<section>_w<k>.mp4, frame 0 = local frame f0 - PAD_S*fps), built with
build_trajectories --pregate (production settings, harvest stride), and the outputs moved next to
the window. Same code, same models, same h264 source as production; the only differences are the
cold start per window (PAD_S of warm-up before the span) and the frames chosen.

  python -m sportvu.rebuild gsw_phx_2016
"""
from __future__ import annotations
import json, shutil, subprocess, sys, time
from pathlib import Path
import config
from fetch_pbp import video_path
from harvest_driver import build_stride
from sportvu.sync import SYNC_DIR

BUILD_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "build"
FF = "/opt/homebrew/bin/ffmpeg"
BUDGET_S = 100.0           # seconds of running clock rebuilt per section
PAD_S = 1.5                # warm-up before each span (positions there are not evaluated)
MIN_SPAN_S = 8.0


def plan_windows(section: str) -> list[dict]:
    tm = json.loads((SYNC_DIR / (section + "_local_timemap.json")).read_text())
    fps = tm["fps"]
    spans = sorted((s for s in tm["spans"] if s["video_s"] >= MIN_SPAN_S), key=lambda s: -s["video_s"])
    out, used = [], 0.0
    for k, s in enumerate(spans):
        if used >= BUDGET_S:
            break
        take = min(s["video_s"], BUDGET_S - used)
        f_start = max(0, int(s["f0"] - PAD_S * fps))
        f_end = int(min(s["f1"], s["f0"] + take * fps))
        out.append({"window": "%s_w%02d" % (section, len(out)), "section": section, "fps": fps,
                    "f_start": f_start, "f0": s["f0"], "f_end": f_end, "period": s["period"]})
        used += take
    return out


def cut(window: dict) -> Path:
    vid = video_path(window["section"]); out = BUILD_DIR / (window["window"] + ".mp4")
    if out.exists():
        return out
    fps = window["fps"]
    start = window["f_start"] / fps; dur = (window["f_end"] - window["f_start"] + 1) / fps
    subprocess.run([FF, "-y", "-v", "error", "-ss", "%.4f" % start, "-t", "%.4f" % dur, "-i", str(vid), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def build(window: dict, snip: Path) -> dict:
    stem = snip.stem
    done = BUILD_DIR / (stem + "_frames.json")
    if done.exists():
        return {"window": stem, "cached": True}
    stride = build_stride(window["fps"])
    n = (window["f_end"] - window["f_start"]) // stride + 2
    import os
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    r = subprocess.run(["/opt/anaconda3/bin/python", "build_trajectories.py", "--source", str(snip), "--start", "0",
                        "--max-frames", str(n), "--stride", str(stride), "--pregate", "--no-video"],
                       cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True)
    moved = []
    for p in list(config.TRACKING_DIR.glob(stem + "_*")) + list((config.PROJECT_ROOT / "reports" / "viz").glob("team_clusters_" + stem + "*")):
        shutil.move(str(p), str(BUILD_DIR / p.name)); moved.append(p.name)
    return {"window": stem, "rc": r.returncode, "moved": moved, "tail": r.stderr[-300:] if r.returncode else ""}


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "gsw_phx_2016"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    secs = sorted(p.name.replace("_local_timemap.json", "") for p in SYNC_DIR.glob(game + "_s*_local_timemap.json"))
    windows = [w for s in secs for w in plan_windows(s)]
    (BUILD_DIR / (game + "_windows.json")).write_text(json.dumps(windows, indent=1))
    print("windows: %d, %.0f s of video" % (len(windows), sum((w["f_end"] - w["f_start"]) / w["fps"] for w in windows)), flush=True)
    t0 = time.time()
    for k, w in enumerate(windows):
        snip = cut(w)
        r = build(w, snip)
        print("  [%d/%d] %-24s %s  (%.0f s elapsed)" % (k + 1, len(windows), w["window"], "cached" if r.get("cached") else ("ok" if r["rc"] == 0 else "FAILED " + r["tail"]), time.time() - t0), flush=True)
    print("rebuild done: %d/%d windows have a sidecar" % (sum(1 for w in windows if (BUILD_DIR / (w["window"] + "_frames.json")).exists()), len(windows)), flush=True)


if __name__ == "__main__":
    main()
