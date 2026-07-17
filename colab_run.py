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
  align     align_outcomes + matchup_metrics + tier2_join + tier2_credit;
            any sub-step's nonzero exit marks the step FAILED in the report
            (run 10: a crashed join went unnoticed and a stale file shipped)
  clips     stream-copy every aligned possession span to Drive clips/
            (±2 s pad; feeds matchup labeling, jersey-OCR batches, demos)
  package   FRESH results/run_<stamp>/ dir per run (fixed-name zips accreted
            stale entries) + honesty report: how many section artifacts were
            NEWLY built THIS run — loud warning if zero. tier2_join/credit
            are EXCLUDED from the pbp zip: the local repo re-runs both after
            ingest, and tier2_credit's fingerprint guard refuses stale joins

Storage policy (2 TB Drive): video/ and crops archives are permanent —
fetch/build once, never prune; sources vanish from YouTube.
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
RUN_STAMP = time.strftime("%Y%m%d_%H%M")   # results/run_<stamp>/ — one dir per
                                           # run, never overwritten (2 TB Drive;
                                           # fixed-name zips caused accretion bugs)
GAMES = ["gsw_cle_2017f_g5", "gsw_sac_klay37", "gsw_nyk_curry54",
         "gsw_mia_2017", "gsw_splash62", "gsw_hou_duel",
         # night 4: one more GSW game tips the GSW defense bucket past
         # MIN_BUCKET=300 (at 295 after run 10). Corrupt-download history
         # in games.json — fetch now ffprobe-verifies, so a bad fetch just
         # marks the game not-ready instead of poisoning the run
         "gsw_phx_2016"]
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
    # video/ is a PERMANENT archive (2 TB Drive): fetch once, never prune —
    # YouTube uploads get taken down, and a lost source video makes its game
    # unreproducible forever. clips/ = per-possession cuts; crops/ lives under
    # tracking/ (Drive-persisted via the symlink below).
    for d in ("video", "tracking", "clips", "easyocr/model", f"results/run_{RUN_STAMP}"):
        os.makedirs(f"{persist}/{d}", exist_ok=True)
    os.environ["HARVEST_SAVE_CROPS"] = "1"   # build persists torso-crop tars
    # easyocr weights live on Drive: a fresh VM must NEVER depend on easyocr's
    # ~80 MB detector download completing (run 12: ContentTooShortError killed
    # align at t+680s). Cache seeded from the local Mac's ~/.EasyOCR (07-16);
    # subprocesses inherit the env var.
    os.environ["EASYOCR_MODULE_PATH"] = f"{persist}/easyocr"
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
    # easyocr must initialize HERE, at t+~2 min, not at t+680 s inside align:
    # with the Drive cache seeded this is instant; if the cache is somehow
    # empty the download happens now, visibly, and a failure aborts the run
    # before any long work (run-12 lesson)
    r2 = subprocess.run([sys.executable, "-c",
                         "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False); "
                         "print('easyocr OK (models from', __import__('os').environ"
                         "['EASYOCR_MODULE_PATH'] + ')')"],
                        capture_output=True, text=True)
    print((r2.stdout or r2.stderr).strip()[-300:])
    ok = r.returncode == 0 and r2.returncode == 0
    record("preflight", "ok" if ok else "FAILED",
           tail=((r.stderr or "") + (r2.stderr or ""))[-400:])
    if not ok:
        sys.exit("preflight failed (gate or easyocr) — aborting before any long work.")


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
    failed = []
    if sh([sys.executable, "align_outcomes.py", "--clips"] + secs).returncode:
        failed.append("align_outcomes")
    m_fail = 0
    for c in secs:
        r = subprocess.run([sys.executable, "matchup_metrics.py", "--clip", c,
                            "--no-video"], capture_output=True, text=True)
        if r.returncode:
            m_fail += 1
            print(f"!! matchup_metrics FAILED {c}: {(r.stderr or '')[-140:]}")
    # run-10 lesson: a crashed join left a stale tier2_join.json that credit
    # aggregated SILENTLY and the package shipped — every sub-step's exit code
    # must surface, and 'align: ok' must mean all of them succeeded.
    # run-12 lesson: when align_outcomes ITSELF failed, do not join/credit at
    # all — a table computed from partial state looks plausible (GSW n=163)
    # and invites reading a broken run as data
    if failed:
        print("!! align_outcomes failed — SKIPPING join+credit (no tables from "
              "partial state); nothing here supersedes the repo's committed join")
    else:
        for tool in ("tier2_join.py", "tier2_credit.py"):
            if sh([sys.executable, tool]).returncode:
                failed.append(tool)
    record("align", "FAILED" if failed else "ok", sections=len(secs),
           matchup_failures=m_fail, **({"failed": failed} if failed else {}))


def step_clips(persist: str) -> None:
    """Cut every ALIGNED possession span to a small mp4 on Drive (stream copy,
    ±2 s pad, keyframe-snapped — context video for labeling/OCR/demos, NOT a
    frame-indexed source). Idempotent: existing non-empty clips are kept."""
    banner("possession clips → Drive")
    fps_cache: dict[str, float] = {}

    def fps_of(path: str) -> float:
        if path not in fps_cache:
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=r_frame_rate",
                                "-of", "csv=p=0", path], capture_output=True, text=True)
            num, den = ((r.stdout.strip() or "30/1").split("/") + ["1"])[:2]
            fps_cache[path] = (float(num) / float(den)) if float(den) else 30.0
        return fps_cache[path]

    new = kept = failed = 0
    for opath in sorted(glob.glob("data/pbp/*_outcomes.json")):
        clip = os.path.basename(opath).replace("_outcomes.json", "")
        src = f"{LOCAL_V}/{clip}.mp4"
        if not os.path.exists(src):        # only games localized this run
            continue
        for o in json.load(open(opath)):
            if o.get("status") != "aligned":
                continue
            f0 = o["set_start_frame"]
            dst = f"{persist}/clips/{clip}_f{f0}.mp4"
            if os.path.exists(dst) and os.path.getsize(dst) > 50_000:
                kept += 1
                continue
            fps = fps_of(src)
            t0 = max(0.0, f0 / fps - 2.0)
            t1 = o["core_end_frame"] / fps + 2.0
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.2f}",
                                "-to", f"{t1:.2f}", "-i", src, "-c", "copy",
                                "-avoid_negative_ts", "make_zero", dst],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.getsize(dst) > 50_000:
                new += 1
            else:
                failed += 1
                if os.path.exists(dst):
                    os.remove(dst)         # never leave a truncated clip behind
    print(f"clips: {new} new, {kept} existing, {failed} failed")
    record("clips", "ok" if not failed else "PARTIAL",
           new=new, existing=kept, failed=failed)


def step_package(persist: str) -> None:
    banner("package + honesty report")
    # one FRESH results dir per run (results/run_<stamp>/) — fixed-name zips
    # on Drive accreted stale entries across runs (zip -r never removes);
    # per-run dirs kill that bug class and give run-over-run history free
    rdir = f"{persist}/results/run_{RUN_STAMP}"
    os.system(f'cd {persist} && zip -qr {rdir}/tracking.zip tracking '
              f'-i "tracking/gsw_*"')
    # join/credit are NEVER packaged: the local repo re-runs both after every
    # ingest (tier2_credit's fingerprint guard enforces it), so a stale pair
    # can't ride along as if it covered this run's outcomes (run-10 incident)
    os.system(f"zip -qr {rdir}/pbp.zip data/pbp "
              f'-x "data/pbp/tier2_join.json" "data/pbp/tier2_credit.json"')
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
    path = f"{persist}/results/run_{RUN_STAMP}/run_report.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    step_clips(persist)
    step_package(persist)
    finish(persist)
    print(f"\nDONE in {REPORT['wall_min']} min — download from Drive: "
          f"nba_harvest/results/run_{RUN_STAMP}/ (tracking.zip, pbp.zip, "
          "run_report.json + the live status.json in nba_harvest/)")


if __name__ == "__main__":
    main()
