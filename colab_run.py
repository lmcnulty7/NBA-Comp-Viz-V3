#!/usr/bin/env python
"""colab_run.py — the ENTIRE Colab harvest run as one repo-versioned script.

Why this exists: three runs (3, 5, 7) were burned wholly or partly by stale
notebook copies — Colab silently reopens Drive-cached copies of a notebook,
so notebook-embedded logic is exactly the code git cannot fix. The notebook
is now a one-cell bootstrap (mount Drive → clone → hard-reset → this script);
everything that can rot lives here, behind the reset. The kernel never
imports project modules, so the stale-module-cache failure mode (run 3) is
gone by construction.

Steps — each timed, each recorded in results/run_report.json:
  env       Drive persistence symlinks (VM-local fallback), GPU check
  deps      pip installs (idempotent)
  preflight required files present + the gate actually scores a frame
  fetch     yt-dlp pinned to avc1≤720p (run-7 postmortem: Colab cv2 has no
            AV1 decoder). ATOMIC cache replacement: download to a temp name,
            ffprobe-verify, then os.replace — a failed download can never
            destroy a previously good cached video.
  localize  copy videos off Drive FUSE to VM disk, then DECODE-PROBE each
            one with cv2 (reads a first + mid-file frame — artifact truth
            that catches AV1, FUSE and corruption identically)
  build     harvest_driver over READY games only (per-stage timeout/resume)
  align     align_outcomes + matchup_metrics + tier2_join + tier2_credit
  package   zips + honesty report: how many section artifacts were NEWLY
            built THIS run — loud warning if zero (never again download a
            re-zip of last run's output thinking it's new)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
GAMES = ["gsw_cle_2017f_g5", "gsw_sac_klay37", "gsw_nyk_curry54",
         "gsw_mia_2017", "gsw_splash62", "gsw_hou_duel"]
LOCAL_V = "/content/video_local"
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit": "", "steps": {}, "games": {}}


def banner(name: str) -> None:
    print(f"\n{'='*62}\n== {name}  (t+{time.time()-T0:5.0f}s)\n{'='*62}", flush=True)


def record(step: str, state: str, **kw) -> None:
    REPORT["steps"][step] = {"state": state, "t_s": round(time.time() - T0), **kw}
    if state == "FAILED":
        print(f"!! step {step} FAILED — see run_report.json", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run streaming to the cell output (no capture) unless told otherwise."""
    return subprocess.run(cmd, **kw)


def vinfo(path: str) -> str:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return (r.stdout or "").strip()          # e.g. "h264,720"


def decodable(path: str) -> bool:
    """Artifact truth: can THIS cv2 actually read frames from THIS file?
    Catches unsupported codecs (AV1 on Colab), FUSE seek failures and
    corruption with one test instead of three heuristics."""
    import cv2
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    if ok:
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if n > 100:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n // 2))
            ok = cap.read()[0]
    cap.release()
    return bool(ok)


# ── env ──────────────────────────────────────────────────────────────────────
def step_env() -> str:
    banner("env: persistence + GPU")
    if not os.path.isdir("/content"):
        sys.exit("colab_run.py is Colab-only (it re-links data/ dirs). "
                 "Run the pipeline directly on other machines.")
    REPORT["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    print("repo commit:", REPORT["commit"])
    if os.path.isdir("/content/drive/MyDrive"):
        persist = "/content/drive/MyDrive/nba_harvest"
    else:
        persist = "/content/nba_harvest"
        print("NO DRIVE MOUNT → VM-local only: a disconnect loses progress.")
    for d in ("video", "tracking", "results"):
        os.makedirs(f"{persist}/{d}", exist_ok=True)
    os.makedirs("data/harvest", exist_ok=True)
    # live persistence: videos, per-stage status and outputs write straight to
    # Drive — a disconnect mid-run costs at most one in-flight stage
    os.system(f"ln -sfn {persist}/video data/harvest/video")
    os.system(f"cp -rn data/tracking/. {persist}/tracking/ 2>/dev/null")
    os.system(f"rm -rf data/tracking && ln -sfn {persist}/tracking data/tracking")
    if not os.path.exists(f"{persist}/status.json"):
        open(f"{persist}/status.json", "w").write("{}")
    os.system(f"ln -sf {persist}/status.json data/harvest/status.json")
    gpu = subprocess.run([sys.executable, "-c",
                          "import torch;print(torch.cuda.get_device_name(0) "
                          "if torch.cuda.is_available() else 'NONE')"],
                         capture_output=True, text=True).stdout.strip()
    print("persist:", persist, "| GPU:", gpu)
    if gpu in ("", "NONE"):
        print("!! CPU runtime — builds will be ~10× slower. "
              "Runtime → Change runtime type → GPU.")
    record("env", "ok", persist=persist, gpu=gpu)
    return persist


# ── deps + preflight ─────────────────────────────────────────────────────────
def step_deps() -> None:
    banner("deps: pip installs")
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                        "ultralytics", "easyocr", "yt-dlp"],
                       capture_output=True, text=True)
    record("deps", "ok" if r.returncode == 0 else "FAILED",
           tail=(r.stderr or "")[-300:])


def step_preflight() -> None:
    banner("preflight: files + gate scores a frame")
    required = ["models/trained_head_coefs.npz", "models/thresholds.json",
                "models/player_detector.pt", "models/court_grid_snapped.pt",
                "models/court_line_seg.pt", "data/harvest/games.json"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        record("preflight", "FAILED", missing=missing)
        sys.exit(f"MISSING RUNTIME FILES: {missing}")
    # score the gate in a SUBPROCESS — the driver's child processes are what
    # actually run this code path, so test it the same way
    probe = ("import json,numpy as np,config;"
             "from gate.backbones import get_backbone;"
             "from gate.trained_head import TrainedHeadGate;"
             "g=TrainedHeadGate.load(config.HEAD_PATH,"
             "backbone=get_backbone('clip',config.get_device()),"
             "threshold=json.loads(config.THRESHOLDS_PATH.read_text())['trained']);"
             "assert g.meta.get('src')=='npz', 'gate loaded pickle not npz';"
             "print('gate OK, score on noise:',"
             "g.score(np.random.randint(0,255,(720,1280,3),np.uint8)))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-500:])
    record("preflight", "ok" if r.returncode == 0 else "FAILED",
           tail=(r.stderr or "")[-400:])
    if r.returncode != 0:
        sys.exit("preflight gate-scoring failed — aborting before any download.")


# ── fetch ────────────────────────────────────────────────────────────────────
YTDLP_FMT = ("bv*[vcodec^=avc1][height<=720][height>=480]+ba/"
             "b[vcodec^=avc1][height<=720]")


def step_fetch() -> None:
    banner("fetch: avc1-only downloads, atomic cache replacement")
    reg = json.load(open("data/harvest/games.json"))
    for t in GAMES:
        dst = f"data/harvest/video/{t}.mp4"
        g = REPORT["games"].setdefault(t, {})
        if os.path.exists(dst) and os.path.getsize(dst) > 1e8:
            info = vinfo(dst)
            if info.startswith("h264"):
                g["fetch"] = f"cached {info}"
                print(f"{t}: cached OK ({info})")
                continue
            print(f"{t}: cached but [{info}] — will replace (old file kept "
                  "until the new one verifies)")
        tmp = f"data/harvest/video/{t}.part.mp4"
        if os.path.exists(tmp):
            os.remove(tmp)
        vid = reg[t]["video_id"]
        print(f"{t}: downloading avc1 …", flush=True)
        try:
            r = subprocess.run(["yt-dlp", "-f", YTDLP_FMT, "--merge-output-format",
                                "mp4", "-o", tmp, "-q", "--no-warnings",
                                f"https://www.youtube.com/watch?v={vid}"],
                               capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            g["fetch"] = "FAILED [timeout 30 min]"
            print(f"{t}: FETCH FAILED — download hung 30 min, moving on")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        info = vinfo(tmp) if os.path.exists(tmp) else "missing"
        if r.returncode == 0 and info.startswith("h264"):
            os.replace(tmp, dst)                      # atomic: verify THEN swap
            for s in glob.glob(f"data/harvest/video/{t}_s*.mp4"):
                os.remove(s)                          # sections from the old file
            g["fetch"] = f"downloaded {info}"
            print(f"{t}: OK ({info})")
        else:
            err = (r.stderr or "")[-300:]
            hint = (" — YouTube is bot-blocking this Colab IP; download the "
                    "game on your laptop (yt-dlp, same avc1 format) and put "
                    "it in Drive: nba_harvest/video/" if "not a bot" in err
                    or "Sign in" in err else "")
            g["fetch"] = f"FAILED [{info}]{hint}"
            print(f"{t}: FETCH FAILED [{info}] {err}{hint}")
            if os.path.exists(tmp):
                os.remove(tmp)
    record("fetch", "ok")


# ── localize + decode probe ──────────────────────────────────────────────────
def step_localize(persist: str) -> list[str]:
    banner("localize: Drive → VM disk + decode probe")
    os.makedirs(LOCAL_V, exist_ok=True)
    for f in sorted(glob.glob(f"{persist}/video/*.mp4")):
        dst = f"{LOCAL_V}/{os.path.basename(f)}"
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f)):
            print("localizing", os.path.basename(f), flush=True)
            shutil.copy(f, dst)
            # game file changed (e.g. AV1→h264): stale local sections split
            # from the old file must go too, or split_sections keeps them
            tag = os.path.basename(f)[:-4]
            for s in glob.glob(f"{LOCAL_V}/{tag}_s*.mp4"):
                os.remove(s)
    ready = []
    for t in GAMES:
        p = f"{LOCAL_V}/{t}.mp4"
        g = REPORT["games"].setdefault(t, {})
        if not os.path.exists(p):
            g["probe"] = "missing"
        elif decodable(p):
            g["probe"] = "decodable"
            ready.append(t)
        else:
            g["probe"] = f"NOT decodable [{vinfo(p)}]"
        print(f"{t}: {g['probe']}")
    os.system(f"ln -sfn {LOCAL_V} data/harvest/video")
    record("localize", "ok" if ready else "FAILED", ready=ready)
    if not ready:
        finish(persist)
        sys.exit("No game is decodable — nothing to build. See report above.")
    if len(ready) < len(GAMES):
        print(f"!! proceeding with {len(ready)}/{len(GAMES)} games — the rest "
              "are listed above with reasons")
    return ready


# ── build / align / package ──────────────────────────────────────────────────
def step_build(ready: list[str]) -> None:
    banner(f"build: harvest_driver over {len(ready)} games")
    r = sh([sys.executable, "harvest_driver.py", "--games"] + ready + ["--no-align"])
    st = json.load(open("data/harvest/status.json"))
    fails = [(c, s, (rec.get("tail") or "")[:140]) for c, u in sorted(st.items())
             for s, rec in u.items() if rec.get("state") != "ok"]
    for c, s, tail in fails:
        print("FAILED", c, s, "→", tail)
    record("build", "ok" if r.returncode == 0 else "FAILED", failed_units=len(fails))


def step_align() -> None:
    banner("align + join + credit")
    st = json.load(open("data/harvest/status.json"))
    secs = sorted(c for c, u in st.items()
                  if "_s" in c and u.get("segment", {}).get("state") == "ok"
                  and any(c.startswith(t) for t in GAMES))
    print(len(secs), "sections to align")
    sh([sys.executable, "align_outcomes.py", "--clips"] + secs)
    for c in secs:
        subprocess.run([sys.executable, "matchup_metrics.py", "--clip", c,
                        "--no-video"], capture_output=True)
    sh([sys.executable, "tier2_join.py"])
    sh([sys.executable, "tier2_credit.py"])
    record("align", "ok", sections=len(secs))


def step_package(persist: str) -> None:
    banner("package + honesty report")
    os.system(f'cd {persist} && zip -qr results/night3_tracking.zip tracking '
              f'-i "tracking/gsw_*"')
    os.system(f"zip -qr {persist}/results/night3_pbp.zip data/pbp")
    new_total = 0
    print(f"{'game':24s} {'built/total':>12s} {'NEW this run':>13s}")
    for t in GAMES:
        trajs = glob.glob(f"{persist}/tracking/{t}_s*_trajectories.json")
        real = [p for p in trajs if os.path.getsize(p) > 1000]
        new = [p for p in real if os.path.getmtime(p) > T0]
        new_total += len(new)
        REPORT["games"].setdefault(t, {})["sections"] = f"{len(real)}/{len(trajs)}"
        REPORT["games"][t]["new_this_run"] = len(new)
        print(f"{t:24s} {len(real):>5d}/{len(trajs):<6d} {len(new):>13d}")
    record("package", "ok", new_sections=new_total)
    if new_total == 0:
        print("\n" + "!" * 62)
        print("!! THIS RUN BUILT NOTHING NEW — the zips are re-packs of old")
        print("!! output. Do NOT download them expecting new data. Check the")
        print("!! fetch/probe lines above for the per-game reason.")
        print("!" * 62)


def finish(persist: str) -> None:
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT["wall_min"] = round((time.time() - T0) / 60, 1)
    path = f"{persist}/results/run_report.json"
    with open(path, "w") as f:
        json.dump(REPORT, f, indent=1)
    print(f"\nrun report → {path}  (send this file if anything looks wrong)")


def main() -> None:
    persist = step_env()
    step_deps()
    step_preflight()
    step_fetch()
    ready = step_localize(persist)
    step_build(ready)
    step_align()
    step_package(persist)
    finish(persist)
    print(f"\nDONE in {REPORT['wall_min']} min — download from Drive: "
          "nba_harvest/results/ (night3_tracking.zip, night3_pbp.zip, "
          "run_report.json + the live status.json in nba_harvest/)")


if __name__ == "__main__":
    main()
