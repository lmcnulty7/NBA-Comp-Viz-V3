#!/usr/bin/env python
"""colab_scaling.py: the Colab run for the data-scaling study (sportvu/scaling.py), first pass by default.

Same pattern as colab_train_court.py: the notebook (colab_scaling.ipynb) mounts Drive, clones the repo,
hard-resets to origin/track-fix, runs this file, flushes Drive and prints the status this file writes.

  env      Colab only; My Drive/nba_harvest/sportvu/ACCOUNT.txt must read lucienmmcnulty@gmail.com; GPU.
  inputs   the R2.3 inputs already on Drive (r23/r23_dataset.tar, r23/court_grid_snapped.pt), sha256-checked
           against r23/MANIFEST_r23.json; held-out guard on every image name.
  arms     sportvu.scaling builds each arm from the extracted dataset (hard links, no copies).
  train    up to PARALLEL arms at once on the GPU (each a subprocess); an arm that dies (e.g. out of GPU
           memory) is retried alone once. Every arm: same recipe and step budget (sportvu.scaling.train).
  package  each arm as soon as it finishes: results/scaling_<stamp>/<arm>.tar (last.pt, results.csv, args.yaml,
           arm.json) on Drive, so a crash keeps what finished. Held-out scoring runs locally afterwards
           (sportvu.court_replay on each last.pt).
Arms: env SCALING_ARMS="A_full,B_f1,..." (default sportvu.scaling.THIRD_PASS; passes 1 and 2 ran 2026-10-05).
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, tarfile, time

T0 = time.time()
STAMP = time.strftime("%Y%m%d_%H%M")
ACCOUNT = "lucienmmcnulty@gmail.com"
PERSIST = "/content/drive/MyDrive/nba_harvest"
INPUTS = os.path.join(PERSIST, "r23")
WORK = "/content/scaling"
PARALLEL = 2
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}, "arms": {}}
STATUS_FILE = "/content/run_status.txt"


def log(msg: str) -> None:
    print("[t+%5.0f s] %s" % (time.time() - T0, msg), flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def step_env() -> None:
    if not os.path.isdir("/content"):
        sys.exit("colab_scaling.py is Colab-only.")
    acct = os.path.join(PERSIST, "sportvu", "ACCOUNT.txt")
    if not os.path.exists(acct) or open(acct).read().strip() != ACCOUNT:
        sys.exit("WRONG OR UNMOUNTED DRIVE: %s must read %s." % (acct, ACCOUNT))
    REPORT["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
                         capture_output=True, text=True).stdout.strip()
    if gpu in ("", "NONE"):
        sys.exit("no GPU: Runtime > Change runtime type > A100 GPU, then run again.")
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics"], capture_output=True, text=True)
    REPORT["steps"]["env"] = {"gpu": gpu, "commit": REPORT["commit"], "pip_rc": r.returncode}
    log("env ok: %s, commit %s" % (gpu, REPORT["commit"]))


def step_inputs() -> tuple:
    man = json.load(open(os.path.join(INPUTS, "MANIFEST_r23.json")))
    os.makedirs(WORK, exist_ok=True)
    for name, meta in man["files"].items():
        dst = os.path.join(WORK, name)
        if not (os.path.exists(dst) and os.path.getsize(dst) == meta["bytes"]):
            shutil.copyfile(os.path.join(INPUTS, name), dst)
        if sha256(dst) != meta["sha256"]:
            sys.exit("checksum mismatch: %s" % name)
    ds = os.path.join(WORK, "r23_dataset")
    if not os.path.isdir(ds):
        with tarfile.open(os.path.join(WORK, "r23_dataset.tar")) as t:
            t.extractall(WORK)
    sys.path.insert(0, os.getcwd())
    from sportvu import splits
    names = [n for s in ("train", "val") for n in os.listdir(os.path.join(ds, "images", s))]
    splits.refuse_heldout(names, "colab_scaling")
    REPORT["steps"]["inputs"] = {"images": len(names)}
    log("inputs ok: %d images" % len(names))
    return ds, os.path.join(WORK, "court_grid_snapped.pt")


def package(arm: str, out: str) -> None:
    run = os.path.join(WORK, "runs", arm)
    tar_p = os.path.join(WORK, arm + ".tar")
    with tarfile.open(tar_p, "w") as t:
        for f in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
            if os.path.exists(os.path.join(run, f)):
                t.add(os.path.join(run, f), arcname=arm + "/" + f)
        t.add(os.path.join(WORK, "arms", arm, "arm.json"), arcname=arm + "/arm.json")
    shutil.copy(tar_p, os.path.join(out, arm + ".tar"))


def step_train(ds: str, init: str, arms: list) -> None:
    from sportvu import scaling
    out = os.path.join(PERSIST, "results", "scaling_" + STAMP)
    os.makedirs(out, exist_ok=True)
    REPORT["out"] = out
    for arm in arms:
        scaling.build(arm, scaling.Path(os.path.join(WORK, "arms")), scaling.Path(ds))
    queue, running, retry = list(arms), {}, []
    while queue or running or retry:
        cap = PARALLEL if not retry or queue else 1
        while (queue or (retry and not running)) and len(running) < cap:
            arm = queue.pop(0) if queue else retry.pop(0)
            logf = open(os.path.join(WORK, arm + ".log"), "w")
            running[arm] = (subprocess.Popen([sys.executable, "-u", "-m", "sportvu.scaling", "train", arm, os.path.join(WORK, "arms"),
                                              "--init", init, "--workers", "4"], stdout=logf, stderr=subprocess.STDOUT), time.time())
            log("start %s (%d running)" % (arm, len(running)))
        time.sleep(30)
        for arm, (p, t1) in list(running.items()):
            if p.poll() is None:
                continue
            del running[arm]
            ok = p.returncode == 0 and os.path.exists(os.path.join(WORK, "runs", arm, "weights", "last.pt"))
            tail = open(os.path.join(WORK, arm + ".log")).read()[-400:]
            if not ok and arm not in REPORT["arms"]:
                REPORT["arms"][arm] = {"first_try": "failed", "tail": tail}
                retry.append(arm); log("FAILED %s, retry alone: %s" % (arm, tail[-160:].replace("\n", " ")))
                continue
            REPORT["arms"][arm] = {**REPORT["arms"].get(arm, {}), "ok": ok, "minutes": round((time.time() - t1) / 60, 1),
                                   **({} if ok else {"tail": tail})}
            if ok:
                package(arm, out)
            log("%s %s in %.0f min" % ("done" if ok else "FAILED AGAIN", arm, (time.time() - t1) / 60))
            json.dump(REPORT, open(os.path.join(out, "run_report.json"), "w"), indent=1)


def main() -> None:
    step_env()
    ds, init = step_inputs()
    from sportvu import scaling
    arms = [a for a in os.environ.get("SCALING_ARMS", ",".join(scaling.THIRD_PASS)).split(",") if a]
    step_train(ds, init, arms)
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(REPORT, open(os.path.join(REPORT["out"], "run_report.json"), "w"), indent=1)
    ok = [a for a, r in REPORT["arms"].items() if r.get("ok")]
    bad = [a for a in arms if a not in ok]
    REPORT["final"] = ("RUN OK: %d arms in %s" % (len(ok), REPORT["out"])) if not bad else \
                      ("RUN PARTIAL: %d ok, failed %s; results in %s" % (len(ok), bad, REPORT["out"]))
    print("\n" + REPORT["final"])


if __name__ == "__main__":
    try:
        main()
        open(STATUS_FILE, "w").write(REPORT.get("final", "RUN OK"))
    except SystemExit as e:
        open(STATUS_FILE, "w").write("RUN FAILED: %s" % e)
        raise
    except Exception as e:
        open(STATUS_FILE, "w").write("RUN FAILED: %r" % e)
        raise
