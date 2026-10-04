#!/usr/bin/env python
"""colab_label.py: the ENTIRE Colab run for ROADMAP R1.3 (SportVU auto-labels) as one repo script.

Same pattern as colab_run.py: the notebook (colab_label.ipynb) is a one-cell bootstrap that mounts
Drive, clones the repo, hard-resets to origin/track-fix and runs this file, so a stale notebook copy
cannot run stale code.

Steps, each timed and recorded in the run report:
  env       Colab only; Drive must be mounted AND be the project account: My Drive/nba_harvest/
            sportvu/ACCOUNT.txt must read lucienmmcnulty@gmail.com, otherwise the run stops.
            GPU check (A100 expected; any CUDA GPU works).
  deps      ultralytics, easyocr, py7zr (idempotent).
  inputs    the training-game inputs uploaded from the Mac (sportvu/MANIFEST_r13.json): video sections
            copied off Drive to VM disk and sha256-verified against the manifest, SportVU moments and
            local time maps copied into the repo's data dirs. Held-out games are not on Drive.
  preflight gate v2 scores a frame, easyocr initialises from the Drive cache, cv2 decodes a section.
  label     sportvu.autolabel --games <game> --save-images, one process per game in parallel.
  manifest  sportvu.autolabel --manifest-only -> reports/sportvu_labels_manifest.{json,txt}.
  package   results/r13_<stamp>/ on Drive: labels (jsonl, images, contact sheets, per-game results),
            the manifest, logs and this report. Honesty line: accepted frames per game; zero is loud.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, time

T0 = time.time()
STAMP = time.strftime("%Y%m%d_%H%M")
ACCOUNT = "lucienmmcnulty@gmail.com"
PERSIST = "/content/drive/MyDrive/nba_harvest"
LOCAL_V = "/content/r13_video"
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def banner(name: str) -> None:
    print("\n" + "=" * 70 + "\n" + name + "  (t+%.0f s)\n" % (time.time() - T0) + "=" * 70, flush=True)


def record(step: str, state: str, **kw) -> None:
    REPORT["steps"][step] = {"state": state, "t_s": round(time.time() - T0), **kw}
    print("[%s] %s %s" % (step, state, kw if kw else ""), flush=True)


def step_env() -> None:
    banner("env: Drive account + GPU")
    if not os.path.isdir("/content"):
        sys.exit("colab_label.py is Colab-only; run python -m sportvu.autolabel directly elsewhere.")
    acct = os.path.join(PERSIST, "sportvu", "ACCOUNT.txt")
    if not os.path.exists(acct) or open(acct).read().strip() != ACCOUNT:
        sys.exit("WRONG OR UNMOUNTED DRIVE: %s must read %s. Remount Drive with the %s account "
                 "(Runtime > Disconnect, then run again and pick that account)." % (acct, ACCOUNT, ACCOUNT))
    REPORT["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
                         capture_output=True, text=True).stdout.strip()
    os.environ["EASYOCR_MODULE_PATH"] = os.path.join(PERSIST, "easyocr")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    record("env", "ok", account=ACCOUNT, commit=REPORT["commit"], gpu=gpu)
    if gpu in ("", "NONE"):
        print("!! no GPU: Runtime > Change runtime type > A100 GPU. Continuing would be very slow.")


def step_deps() -> None:
    banner("deps")
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics", "easyocr", "py7zr"], capture_output=True, text=True)
    record("deps", "ok" if r.returncode == 0 else "FAILED", tail=(r.stderr or "")[-300:])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def step_inputs() -> list:
    banner("inputs: copy off Drive + verify")
    man = json.load(open(os.path.join(PERSIST, "sportvu", "MANIFEST_r13.json")))
    os.makedirs(LOCAL_V, exist_ok=True)
    os.makedirs("data/sportvu/sync", exist_ok=True)
    bad = []
    for rel, meta in man["files"].items():
        src = os.path.join(PERSIST, rel)
        if rel.startswith("video/"):
            dst = os.path.join(LOCAL_V, os.path.basename(rel))
        elif rel.startswith("sportvu/moments/"):
            dst = os.path.join("data/sportvu", os.path.basename(rel))
        else:
            dst = os.path.join("data/sportvu/sync", os.path.basename(rel))
        if not (os.path.exists(dst) and os.path.getsize(dst) == meta["bytes"]):
            shutil.copyfile(src, dst)
        if sha256(dst) != meta["sha256"]:
            bad.append(rel)
    os.makedirs("data/harvest", exist_ok=True)
    if os.path.islink("data/harvest/video") or os.path.isdir("data/harvest/video"):
        subprocess.run(["rm", "-rf", "data/harvest/video"])
    os.symlink(LOCAL_V, "data/harvest/video")
    if bad:
        record("inputs", "FAILED", checksum_mismatch=bad)
        sys.exit("checksum mismatch (upload incomplete?): %s" % bad[:5])
    record("inputs", "ok", files=len(man["files"]), games=man["games"])
    return man["games"]


def step_preflight(games: list) -> None:
    banner("preflight")
    probe = ("import json,glob,cv2,config;from gate.backbones import get_backbone;from gate.trained_head import TrainedHeadGate;"
             "g=TrainedHeadGate.load(config.HEAD_V2_PATH,backbone=get_backbone('clip',config.get_device()),"
             "threshold=json.loads(config.THRESHOLDS_PATH.read_text())['v2']);"
             "v=sorted(glob.glob('data/harvest/video/%s_s*.mp4'))[1];c=cv2.VideoCapture(v);c.set(1,6000);ok,f=c.read();"
             "assert ok, 'cv2 cannot decode '+v;print('gate v2 scores', round(g.score(f),3), 'on', v);"
             "import easyocr;easyocr.Reader(['en'],gpu=False,verbose=False);print('easyocr OK')" % games[0])
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode != 0:
        record("preflight", "FAILED", tail=r.stderr[-500:])
        sys.exit("preflight failed: %s" % r.stderr[-300:])
    record("preflight", "ok")


def step_label(games: list) -> None:
    banner("label: one process per game")
    os.makedirs("logs", exist_ok=True)
    procs = {}
    for g in games:
        procs[g] = subprocess.Popen([sys.executable, "-u", "-m", "sportvu.autolabel", "--games", g, "--save-images"],
                                    stdout=open("logs/autolabel_%s.log" % g, "w"), stderr=subprocess.STDOUT)
    while procs:
        time.sleep(60)
        for g, p in list(procs.items()):
            if p.poll() is not None:
                print("  %s finished rc=%d (t+%.0f s)" % (g, p.returncode, time.time() - T0), flush=True)
                REPORT.setdefault("label_rc", {})[g] = p.returncode
                del procs[g]
        if procs:
            tails = {g: (open("logs/autolabel_%s.log" % g).read().strip().splitlines() or ["..."])[-1][:100] for g in procs}
            print("  running:", tails, flush=True)
    record("label", "ok" if all(rc == 0 for rc in REPORT["label_rc"].values()) else "FAILED", rc=REPORT["label_rc"])


def step_manifest() -> dict:
    banner("manifest")
    r = subprocess.run([sys.executable, "-m", "sportvu.autolabel", "--manifest-only"], capture_output=True, text=True)
    print(r.stdout[-3000:])
    m = json.load(open("reports/sportvu_labels_manifest.json")) if os.path.exists("reports/sportvu_labels_manifest.json") else {}
    accepted = {g["game"]: g.get("accepted", 0) for g in m.get("games", [])}
    record("manifest", "ok" if r.returncode == 0 else "FAILED", accepted=accepted)
    return accepted


def step_package(accepted: dict) -> None:
    banner("package -> Drive")
    out = os.path.join(PERSIST, "results", "r13_" + STAMP)
    os.makedirs(out, exist_ok=True)
    shutil.copytree("data/sportvu/labels", os.path.join(out, "labels"), dirs_exist_ok=True)
    for f in ("reports/sportvu_labels_manifest.json", "reports/sportvu_labels_manifest.txt"):
        if os.path.exists(f):
            shutil.copy(f, out)
    shutil.copytree("logs", os.path.join(out, "logs"), dirs_exist_ok=True)
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT["honesty"] = {"accepted_frames": accepted, "total": sum(accepted.values())}
    json.dump(REPORT, open(os.path.join(out, "run_report.json"), "w"), indent=1)
    record("package", "ok", out=out)
    if not accepted or sum(accepted.values()) == 0:
        print("\n!!!! ZERO ACCEPTED FRAMES: something upstream failed; read logs/ before using anything.")
    else:
        print("\nDONE: %d accepted frames %s -> %s" % (sum(accepted.values()), accepted, out))


def main() -> None:
    step_env()
    step_deps()
    games = step_inputs()
    step_preflight(games)
    step_label(games)
    step_package(step_manifest())


if __name__ == "__main__":
    main()
