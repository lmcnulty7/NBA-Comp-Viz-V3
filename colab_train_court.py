#!/usr/bin/env python
"""colab_train_court.py: the ENTIRE Colab run for ROADMAP R2.3 (grid court keypoints on the R1.3 labels).

Same pattern as colab_label.py: the notebook (colab_train_court.ipynb) is a one-cell bootstrap that mounts
Drive, clones the repo, hard-resets to origin/track-fix, runs this file, then flushes Drive.

Steps, each timed and recorded in the run report:
  env       Colab only; My Drive/nba_harvest/sportvu/ACCOUNT.txt must read lucienmmcnulty@gmail.com; GPU.
  deps      ultralytics.
  inputs    My Drive/nba_harvest/r23/{r23_dataset.tar, court_grid_snapped.pt} copied to VM disk and
            sha256-checked against r23/MANIFEST_r23.json; the tar is the dataset built locally by
            sportvu.grid_dataset (train games only; every image name passes sportvu.splits.refuse_heldout
            again here). Held-out games never reach Colab.
  baseline  V3's grid model (models/court_grid_snapped.pt) scored on the dataset's val split.
  train     fine-tune V3's grid model on the mix, V3's recipe (train_court_pose.py) except init and epochs.
  package   results/r23_<stamp>/ on Drive: best.pt, r23_run.tar (weights, results.csv, plots, args) and
            this report. Honesty line: best epoch and val pose mAP50-95, V3's model vs fine-tuned.
The held-out scorecards run locally afterwards (COURT_GRID_WEIGHTS=<best.pt> python -m sportvu.rebuild ...).
"""
from __future__ import annotations
import glob, hashlib, json, os, shutil, subprocess, sys, tarfile, time

T0 = time.time()
STAMP = time.strftime("%Y%m%d_%H%M")
ACCOUNT = "lucienmmcnulty@gmail.com"
PERSIST = "/content/drive/MyDrive/nba_harvest"
INPUTS = os.path.join(PERSIST, "r23")
WORK = "/content/r23"
EPOCHS, PATIENCE, IMGSZ, BATCH = 100, 20, 640, 32
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
STATUS_FILE = "/content/run_status.txt"


def banner(name: str) -> None:
    print("\n" + "=" * 70 + "\n" + name + "  (t+%.0f s)\n" % (time.time() - T0) + "=" * 70, flush=True)


def record(step: str, state: str, **kw) -> None:
    REPORT["steps"][step] = {"state": state, "t_s": round(time.time() - T0), **kw}
    print("[%s] %s %s" % (step, state, kw if kw else ""), flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def step_env() -> None:
    banner("env: Drive account + GPU")
    if not os.path.isdir("/content"):
        sys.exit("colab_train_court.py is Colab-only.")
    acct = os.path.join(PERSIST, "sportvu", "ACCOUNT.txt")
    if not os.path.exists(acct) or open(acct).read().strip() != ACCOUNT:
        sys.exit("WRONG OR UNMOUNTED DRIVE: %s must read %s. Remount Drive with the %s account." % (acct, ACCOUNT, ACCOUNT))
    REPORT["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
                         capture_output=True, text=True).stdout.strip()
    record("env", "ok", account=ACCOUNT, commit=REPORT["commit"], gpu=gpu)
    if gpu in ("", "NONE"):
        sys.exit("no GPU: Runtime > Change runtime type > A100 GPU, then run again.")


def step_deps() -> None:
    banner("deps")
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics"], capture_output=True, text=True)
    record("deps", "ok" if r.returncode == 0 else "FAILED", tail=(r.stderr or "")[-300:])


def step_inputs() -> tuple:
    banner("inputs: copy off Drive + verify")
    man = json.load(open(os.path.join(INPUTS, "MANIFEST_r23.json")))
    os.makedirs(WORK, exist_ok=True)
    bad = []
    for name, meta in man["files"].items():
        dst = os.path.join(WORK, name)
        if not (os.path.exists(dst) and os.path.getsize(dst) == meta["bytes"]):
            shutil.copyfile(os.path.join(INPUTS, name), dst)
        if sha256(dst) != meta["sha256"]:
            bad.append(name)
    if bad:
        record("inputs", "FAILED", checksum_mismatch=bad)
        sys.exit("checksum mismatch (upload incomplete?): %s" % bad)
    ds = os.path.join(WORK, "r23_dataset")
    if not os.path.isdir(ds):
        with tarfile.open(os.path.join(WORK, "r23_dataset.tar")) as t:
            t.extractall(WORK)
    yaml_p = os.path.join(ds, "data.yaml")
    lines = open(yaml_p).read().splitlines()
    open(yaml_p, "w").write("\n".join(["path: " + ds] + [l for l in lines if not l.startswith("path:")]) + "\n")
    sys.path.insert(0, os.getcwd())
    from sportvu import splits
    imgs = glob.glob(os.path.join(ds, "images", "*", "*"))
    n = splits.refuse_heldout([os.path.basename(p) for p in imgs], "colab_train_court")
    counts = {s: len(glob.glob(os.path.join(ds, "images", s, "*"))) for s in ("train", "val")}
    record("inputs", "ok", files=list(man["files"]), images=counts, heldout_checked=n)
    return yaml_p, os.path.join(WORK, "court_grid_snapped.pt")


def val_metrics(model, yaml_p: str, name: str) -> dict:
    r = model.val(data=yaml_p, split="val", imgsz=IMGSZ, batch=BATCH, device=0, project="/content/runs", name=name, exist_ok=True, plots=False)
    return {"pose_map50_95": round(float(r.pose.map), 4), "pose_map50": round(float(r.pose.map50), 4),
            "box_map50_95": round(float(r.box.map), 4)}


def step_baseline(yaml_p: str, init: str) -> dict:
    banner("baseline: V3's grid model on the val split")
    from ultralytics import YOLO
    m = val_metrics(YOLO(init), yaml_p, "r23_baseline_val")
    record("baseline", "ok", **m)
    return m


def step_train(yaml_p: str, init: str) -> str:
    banner("train: fine-tune V3's grid model")
    from ultralytics import YOLO
    model = YOLO(init)
    model.train(data=yaml_p, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, patience=PATIENCE, seed=42,
                deterministic=True, mosaic=0.0, degrees=0.0, translate=0.05, scale=0.2, fliplr=0.5,
                project="/content/runs", name="r23", exist_ok=True)
    best = "/content/runs/r23/weights/best.pt"
    ok = os.path.exists(best)
    res = {}
    if ok:
        import csv
        rows = list(csv.DictReader(open("/content/runs/r23/results.csv")))
        key = next(k for k in rows[0] if "pose" in k.lower() and "map50-95" in k.lower().replace(" ", ""))
        b = max(rows, key=lambda r: float(r[key]))
        res = {"epochs_run": len(rows), "best_epoch": int(float(b["epoch"])), "best_val_pose_map50_95": round(float(b[key]), 4)}
        res["finetuned_val"] = val_metrics(YOLO(best), yaml_p, "r23_finetuned_val")
    record("train", "ok" if ok else "FAILED", **res)
    return best if ok else ""


def step_package(best: str) -> None:
    banner("package -> Drive")
    out = os.path.join(PERSIST, "results", "r23_" + STAMP)
    os.makedirs(out, exist_ok=True)
    if best:
        shutil.copy(best, os.path.join(out, "best.pt"))
        with tarfile.open("/content/r23_run.tar", "w") as t:
            t.add("/content/runs", arcname="runs")
        shutil.copy("/content/r23_run.tar", os.path.join(out, "r23_run.tar"))
        REPORT["best_sha256"] = sha256(best)
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(REPORT, open(os.path.join(out, "run_report.json"), "w"), indent=1)
    record("package", "ok", out=out)
    base, tr = REPORT["steps"].get("baseline", {}), REPORT["steps"].get("train", {})
    if not best:
        REPORT["final"] = "RUN FAILED: training produced no weights (read the output above)"
        print("\n!!!! TRAINING PRODUCED NO WEIGHTS: read the cell output above before using anything.")
    else:
        REPORT["final"] = "RUN OK: results in %s" % out
        print("\nDONE: val pose mAP50-95 V3 %s -> fine-tuned %s (best epoch %s of %s) -> %s" % (
            base.get("pose_map50_95"), tr.get("finetuned_val", {}).get("pose_map50_95"), tr.get("best_epoch"), tr.get("epochs_run"), out))


def main() -> None:
    step_env()
    step_deps()
    yaml_p, init = step_inputs()
    step_baseline(yaml_p, init)
    step_package(step_train(yaml_p, init))


if __name__ == "__main__":
    # The notebook prints this file after flushing Drive, so its last line says whether the run worked.
    try:
        main()
        open(STATUS_FILE, "w").write(REPORT.get("final", "RUN OK"))
    except SystemExit as e:
        open(STATUS_FILE, "w").write("RUN FAILED: %s" % e)
        raise
    except Exception as e:
        open(STATUS_FILE, "w").write("RUN FAILED: %r" % e)
        raise
