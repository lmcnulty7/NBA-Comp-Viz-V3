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
  COURT_TRACKER=0 python -m sportvu.rebuild gsw_phx_2016 --out data/sportvu/build_public
      another build of the SAME windows (V3's plan, clips and time maps in data/sportvu/build), with the
      environment passed to build_trajectories; only its sidecars go to --out, so sportvu.bench scores it
      on the same frames and truth as V3.
"""
from __future__ import annotations
import json, shutil, subprocess, sys, time
from pathlib import Path
import config
from fetch_pbp import video_path
from harvest_driver import build_stride
from sportvu.sync import SYNC_DIR

BUILD_DIR = config.PROJECT_ROOT / "data" / "sportvu" / "build"
FF = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"     # Colab: system ffmpeg
BUDGET_S = 100.0           # seconds of running clock rebuilt per section
PAD_S = 1.5                # warm-up before each span (positions there are not evaluated)
MIN_SPAN_S = 8.0


def plan_windows(section: str, budget_s: float = BUDGET_S) -> list[dict]:
    tm = json.loads((SYNC_DIR / (section + "_local_timemap.json")).read_text())
    fps = tm["fps"]
    spans = sorted((s for s in tm["spans"] if s["video_s"] >= MIN_SPAN_S), key=lambda s: -s["video_s"])
    out, used = [], 0.0
    for k, s in enumerate(spans):
        if used >= budget_s:
            break
        take = min(s["video_s"], budget_s - used)
        f_start = max(0, int(s["f0"] - PAD_S * fps))
        f_end = int(min(s["f1"], s["f0"] + take * fps))
        out.append({"window": "%s_w%02d" % (section, len(out)), "section": section, "fps": fps,
                    "f_start": f_start, "f0": s["f0"], "f_end": f_end, "period": s["period"]})
        used += take
    return out


def cut(window: dict, out_dir: Path = BUILD_DIR) -> Path:
    vid = video_path(window["section"]); out = out_dir / (window["window"] + ".mp4")
    if out.exists():
        return out
    fps = window["fps"]
    start = window["f_start"] / fps; dur = (window["f_end"] - window["f_start"] + 1) / fps
    subprocess.run([FF, "-y", "-v", "error", "-ss", "%.4f" % start, "-t", "%.4f" % dur, "-i", str(vid), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def build(window: dict, snip: Path, out_dir: Path = BUILD_DIR, extra_args: tuple = ()) -> dict:
    stem = snip.stem
    done = out_dir / (stem + "_frames.json")
    if done.exists():
        return {"window": stem, "cached": True}
    stride = build_stride(window["fps"])
    n = (window["f_end"] - window["f_start"]) // stride + 2
    import os
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    r = subprocess.run([sys.executable, "build_trajectories.py", "--source", str(snip), "--start", "0",
                        "--max-frames", str(n), "--stride", str(stride), "--pregate", "--no-video", *extra_args],
                       cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True)
    moved = []
    for p in list(config.TRACKING_DIR.glob(stem + "_*")) + list((config.PROJECT_ROOT / "reports" / "viz").glob("team_clusters_" + stem + "*")):
        shutil.move(str(p), str(out_dir / p.name)); moved.append(p.name)
    return {"window": stem, "rc": r.returncode, "moved": moved, "tail": r.stderr[-300:] if r.returncode else ""}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="gsw_phx_2016")
    ap.add_argument("--out", type=Path, default=BUILD_DIR, help="sidecar dir for another build of V3's windows")
    a = ap.parse_args()
    game, out = a.game, a.out.resolve()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if out != BUILD_DIR:
        out.mkdir(parents=True, exist_ok=True)
        windows = json.loads((BUILD_DIR / (game + "_windows.json")).read_text())    # V3's plan: same frames
    else:
        secs = sorted(p.name.replace("_local_timemap.json", "") for p in SYNC_DIR.glob(game + "_s*_local_timemap.json"))
        windows = [w for s in secs for w in plan_windows(s)]
    (out / (game + "_windows.json")).write_text(json.dumps(windows, indent=1))
    print("windows: %d, %.0f s of video" % (len(windows), sum((w["f_end"] - w["f_start"]) / w["fps"] for w in windows)), flush=True)
    from clock_reader import ClockReader
    from sportvu.local_sync import build_local_timemap
    reader = ClockReader(json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())[game]["layout"])
    t0 = time.time()
    for k, w in enumerate(windows):
        snip = cut(w)                                           # clips stay in BUILD_DIR, shared by every build
        r = build(w, snip, out_dir=out)
        tm = BUILD_DIR / (w["window"] + "_timemap.json")      # the window's clock map, read by sportvu.truth
        if not tm.exists():
            tm.write_text(json.dumps(build_local_timemap(w["section"], reader, vid=snip)))
        print("  [%d/%d] %-24s %s  (%.0f s elapsed)" % (k + 1, len(windows), w["window"], "cached" if r.get("cached") else ("ok" if r["rc"] == 0 else "FAILED " + r["tail"]), time.time() - t0), flush=True)
    print("rebuild done: %d/%d windows have a sidecar" % (sum(1 for w in windows if (out / (w["window"] + "_frames.json")).exists()), len(windows)), flush=True)


if __name__ == "__main__":
    main()
