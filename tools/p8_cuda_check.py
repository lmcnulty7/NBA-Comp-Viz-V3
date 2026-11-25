"""tools/p8_cuda_check.py: PLAN.md D161 (1), the two D99 windows of gsw_ind_2016 built on CUDA with the frozen path
and compared with the Mac CPU and MPS builds. Design and reading rule, fixed before any run:
docs/research/P8_cuda_check.md.

    <frozen CV python> -m tools.p8_cuda_check pack --dest <pack folder>                     Mac
    python3 -m tools.p8_cuda_check put --pack <pack folder>                                 Mac, COLAB_URL and COLAB_TOKEN
    python -m tools.p8_cuda_check run --pack-dir DIR --out-dir DIR --work DIR [--install-head]   the runtime
    python3 -m tools.p8_cuda_check get --dest <result folder>                               Mac, COLAB_URL and COLAB_TOKEN
    <frozen CV python> -m tools.p8_cuda_check compare --result <result folder> [--pack <pack folder>]   Mac

pack: the two window clips the Mac builds read, the section's local time map and the gitignored gate head, in one tar
with a manifest (each member's sha256 and size, the tar's sha256, the frozen window plan, the sha256 of the tracked
weights, the CLIP parameter fingerprint, the Mac's library versions). put and get move files over the runtime's HTTP
endpoints with tools.p8_colab_xfer. run checks every hash, the plan and the fingerprint, builds each arm (cuda_a,
cuda_b, colab_cpu) through tools.p8_frozen.safe_build with P8_DEVICE, and rewrites the result tar and its
complete.json after each arm. compare reads the result and the Mac builds (tools.p8_truth_v2.DEVICE_BUILDS) and
applies the reading rule; it writes cuda_check/cuda_check_result.json in the P8 workspace.

Only gsw_ind_2016 (an R1.3 training game, D98) is touched; every command guards it first. The module's top level
imports no model library, so tests import the pure parts with the system Python.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_cuda_check"
GAME = "gsw_ind_2016"
SECTION = "gsw_ind_2016_s02"
WINDOWS = ("gsw_ind_2016_s02_w03", "gsw_ind_2016_s02_w04")
TRACKED_WEIGHTS = ("models/player_detector.pt", "models/court_grid_snapped.pt", "models/court_line_seg.pt",
                   "models/thresholds.json")
HEAD = "models/trained_head_v2.joblib"
PACK = "p8_cuda_pack"
RESULT = "p8_cuda_gsw_ind_2016"
ARMS = {"cuda_a": "cuda", "cuda_b": "cuda", "colab_cpu": "cpu"}
REMOTE_IN = "content/p8_cuda_in"
REMOTE_OUT = "content/p8_cuda_out"
KEEP = ("_frames.json", "_trajectories.json", "_identity.json", "_device.json")

# the reading rule (docs/research/P8_cuda_check.md section 4); never changed after a CUDA build is seen
FIXED_OFFSETS = (0.30, 0.62, 0.90, 1.20)
TOL = {"frames_jaccard_min": 0.95, "boxes_mean_abs_max": 0.25, "same_count_share_min": 0.80,
       "h_gap_median_px_max": 1.0, "cand_rel": 0.10, "cand_floor": 3}
# recorded frozen offsets of the Mac builds (P8_PLATFORM.md, truth_v2 device file); compare recomputes and flags them
RECORDED_MAC_OFFSETS = {"gsw_ind_2016_s02_w03": {"mps": 0.90, "cpu": 1.42, "detector_cpu": 0.94},
                        "gsw_ind_2016_s02_w04": {"mps": -1.34, "cpu": -0.78}}
PLATFORM_JSON = REPO / "reports" / "research" / "p8_platform.json"


def guard_all(who: str = WHO) -> None:
    items = [GAME, SECTION, *WINDOWS]
    refuse_cv_test(items, who)
    guard(items, who)


# ── hashes, manifest, tars (pure) ───────────────────────────────────────────────────────────────────────────────────
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def member_records(files: dict) -> dict:
    """{member name: local path} -> {member name: {sha256, bytes}}."""
    return {n: {"sha256": sha256_file(Path(p)), "bytes": Path(p).stat().st_size} for n, p in sorted(files.items())}


def write_tar(path: Path, files: dict) -> str:
    """A deterministic tar (sorted names, mtime 0, owner 0, mode 644) of {member name: local path}, written through a
    .part file -> its sha256."""
    part = path.with_name(path.name + ".part")
    with tarfile.open(part, "w") as tf:
        for name in sorted(files):
            data = Path(files[name]).read_bytes() if not isinstance(files[name], bytes) else files[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
            tf.addfile(ti, io.BytesIO(data))
    os.replace(part, path)
    return sha256_file(path)


def safe_member(name: str) -> bool:
    p = Path(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and not name.startswith(("/", "\\"))


def check_tar(tar_path: Path, records: dict) -> dict:
    """Every member of the tar is a regular file with a safe name named in records, every record is present, and
    each sha256 and size equals its record -> {name: sha256}. SystemExit otherwise."""
    seen = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile() or not safe_member(m.name) or m.name not in records:
                raise SystemExit(f"{WHO}: {Path(tar_path).name} holds an unexpected member {m.name!r}")
            h = hashlib.sha256()
            src = tf.extractfile(m)
            for b in iter(lambda: src.read(1 << 20), b""):
                h.update(b)
            seen[m.name] = h.hexdigest()
            if seen[m.name] != records[m.name]["sha256"] or m.size != records[m.name]["bytes"]:
                raise SystemExit(f"{WHO}: {m.name} in {Path(tar_path).name} differs from its record")
    missing = sorted(set(records) - set(seen))
    if missing:
        raise SystemExit(f"{WHO}: {Path(tar_path).name} lacks {missing}")
    return seen


def extract_checked(tar_path: Path, records: dict, dest: Path) -> Path:
    """check_tar, then extract every member under dest (names checked again on the way)."""
    check_tar(tar_path, records)
    dest = Path(dest)
    with tarfile.open(tar_path) as tf:
        for m in tf:
            out = dest / m.name
            if dest.resolve() not in out.resolve().parents:
                raise SystemExit(f"{WHO}: {m.name!r} would leave {dest}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
    return dest


def tensor_fingerprint(named) -> dict:
    """sha256 over (name, shape, dtype, bytes) of named numpy arrays, sorted by name: equal tensors give equal
    fingerprints whatever file format held them."""
    h, n = hashlib.sha256(), 0
    for name, arr in sorted(named, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
        n += 1
    return {"sha256": h.hexdigest(), "tensors": n}


def clip_fingerprint() -> dict:
    """The gate's CLIP (config.CLIP_MODEL_NAME, loaded as gate.backbones does) on CPU -> its parameter fingerprint and
    the hub revision it came from."""
    import config
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    fp = tensor_fingerprint((k, v.detach().cpu().contiguous().numpy()) for k, v in m.named_parameters())
    return {**fp, "model": config.CLIP_MODEL_NAME, "revision": getattr(m.config, "_commit_hash", None)}


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform(terse=True), "cpu_count": os.cpu_count()}
    for mod in ("torch", "ultralytics", "transformers", "cv2", "numpy", "scipy", "lap"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:                               # noqa: BLE001 (record what is missing, never fail on it)
            out[mod] = None
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:                                   # noqa: BLE001
        pass
    return out


def rename_prev(p: Path) -> None:
    """Never overwrite a result already there: rename it to <name>.prev-<UTC time>."""
    if p.exists():
        p.rename(p.with_name(f"{p.name}.prev-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))


# ── comparison and reading rule (pure) ──────────────────────────────────────────────────────────────────────────────
def frames_of(side: dict) -> dict:
    return {r["frame"]: r for r in side["frames"]}


def pair_metrics(x: dict, r: dict) -> dict:
    """Two builds' frames ({frame: row}) -> the measures of the design's section 3 (X against reference R)."""
    from tools.p8_platform_summary import boxes_mean, h_gap_px
    common, union = set(x) & set(r), set(x) | set(r)
    bx, br = boxes_mean(x), boxes_mean(r)
    return {"frames_x": len(x), "frames_r": len(r), "common": len(common),
            "frames_jaccard": round(len(common) / len(union), 4) if union else None,
            "boxes_mean_x": bx, "boxes_mean_r": br,
            "boxes_mean_diff": round(bx - br, 2) if bx is not None and br is not None else None,
            "same_count_share": round(sum(len(x[f]["boxes"]) == len(r[f]["boxes"]) for f in common) / len(common), 4)
            if common else None,
            "identical_boxes_share": round(sum([b["bbox"] for b in x[f]["boxes"]] == [b["bbox"] for b in r[f]["boxes"]]
                                               for f in common) / len(common), 4) if common else None,
            "h_gap_px": h_gap_px(x, r)}


def offset_set(colab_offset, mac_offsets) -> list:
    """The window's fixed offsets: the P8_PLATFORM sweep, Colab's R1.3 offset and each Mac build's frozen offset."""
    vals = set(FIXED_OFFSETS) | {colab_offset} | set(mac_offsets)
    return sorted(round(float(v), 2) for v in vals if v is not None)


def cand_ok(x, r, tol: dict = TOL) -> bool:
    if x is None or r is None:
        return False
    return abs(x - r) <= max(tol["cand_floor"], tol["cand_rel"] * r)


def window_tests(pm: dict, cands_x: dict, cands_r: dict, tol: dict = TOL) -> dict:
    """The five tests of one window, X against R. cands_*: {offset key: candidates}; every key of cands_r is tested."""
    h = pm.get("h_gap_px") or {}
    return {"frames_jaccard": pm["frames_jaccard"] is not None and pm["frames_jaccard"] >= tol["frames_jaccard_min"],
            "boxes_mean": pm["boxes_mean_diff"] is not None and abs(pm["boxes_mean_diff"]) <= tol["boxes_mean_abs_max"],
            "same_count": pm["same_count_share"] is not None and pm["same_count_share"] >= tol["same_count_share_min"],
            "court_h_gap": h.get("median") is not None and h["median"] <= tol["h_gap_median_px_max"],
            "candidates": bool(cands_r) and all(cand_ok(cands_x.get(k), v, tol) for k, v in cands_r.items())}


def matches(per_window: dict) -> bool:
    """per_window: {window: tests dict}; a match needs every test on every one of the design's windows."""
    return all(w in per_window for w in WINDOWS) and all(all(t.values()) for t in per_window.values())


def failed(per_window: dict) -> dict:
    return {w: sorted(k for k, v in t.items() if not v) for w, t in per_window.items() if not all(t.values())}


def verdict(tests: dict) -> dict:
    """tests: {"X~R": {window: tests}} for the pairs present -> the verdict of the design's section 4."""
    sep = tests.get("mac_cpu~mac_mps")
    out = {"rule_separates_cpu_from_mps": None if sep is None else not matches(sep),
           "cuda_repeatable": None if "cuda_b~cuda_a" not in tests else matches(tests["cuda_b~cuda_a"]),
           "cuda_matches_cpu": None if "cuda_a~mac_cpu" not in tests else matches(tests["cuda_a~mac_cpu"]),
           "cuda_matches_mps": None if "cuda_a~mac_mps" not in tests else matches(tests["cuda_a~mac_mps"]),
           "colab_cpu_matches_mac_cpu": None if "colab_cpu~mac_cpu" not in tests else matches(tests["colab_cpu~mac_cpu"]),
           "failed_tests": {k: failed(v) for k, v in tests.items()}}
    if out["cuda_matches_cpu"] is None:
        v = "no cuda_a build"
    elif out["rule_separates_cpu_from_mps"] is False:
        v = "rule cannot separate"
    elif out["cuda_repeatable"] is False:
        v = "CUDA not repeatable"
    elif out["cuda_matches_cpu"]:
        v = "CUDA matches CPU"
    elif out["cuda_matches_mps"]:
        v = "CUDA matches MPS"
    else:
        v = "matches neither"
    out["verdict"] = v
    out["repeatability"] = "not measured" if out["cuda_repeatable"] is None else "measured"
    return out


# ── pack (Mac, frozen CV python) ────────────────────────────────────────────────────────────────────────────────────
def section_record(P: dict) -> dict:
    vid = P["video"] / f"{SECTION}.mp4"
    rec = {"section": SECTION, "sha256": sha256_file(vid) if vid.exists() else None, "manifest_r13_sha256": None}
    if os.environ.get("DATA_ROOT"):
        man = Path(os.environ["DATA_ROOT"]) / "sportvu" / "MANIFEST_r13.json"
        if man.exists():
            rec["manifest_r13_sha256"] = (json.loads(man.read_text())["files"].get(f"video/{SECTION}.mp4") or {}).get("sha256")
    rec["matches_r13"] = None if not (rec["sha256"] and rec["manifest_r13_sha256"]) else rec["sha256"] == rec["manifest_r13_sha256"]
    return rec


def frozen_plan(P: dict) -> dict:
    from sportvu.autolabel import EVERY_SPAN
    from sportvu.rebuild import plan_windows
    plan = {w["window"]: w for w in plan_windows(SECTION, budget_s=EVERY_SPAN)}
    missing = [w for w in WINDOWS if w not in plan]
    if missing:
        raise SystemExit(f"{WHO}: the frozen plan of {SECTION} has no {missing}")
    return json.loads(json.dumps({w: plan[w] for w in WINDOWS}))


def tracked_weights() -> dict:
    return {rel: sha256_file(REPO / rel) for rel in TRACKED_WEIGHTS}


def cmd_pack(a) -> dict:
    guard_all()
    dest = check_outside(Path(a.dest).expanduser())
    dest.mkdir(parents=True, exist_ok=True)
    from tools import p8_frozen as F
    P = F.install_paths()
    import config
    files = {f"clips/{w}.mp4": P["label_build"] / f"{w}.mp4" for w in WINDOWS}
    files[f"sync/{SECTION}_local_timemap.json"] = P["sync"] / f"{SECTION}_local_timemap.json"
    files[HEAD] = config.HEAD_V2_PATH
    members = member_records(files)
    tar = dest / f"{PACK}.tar"
    rename_prev(tar)
    tar_sha = write_tar(tar, {n: Path(p) for n, p in files.items()})
    weights = {**tracked_weights(), HEAD: members[HEAD]["sha256"]}
    man = {"what": "PLAN.md D161 (1) CUDA check pack", "design": "docs/research/P8_cuda_check.md", "game": GAME,
           "section": section_record(P), "windows": frozen_plan(P), "tar": tar.name, "tar_sha256": tar_sha,
           "members": members, "weights": weights, "clip_fingerprint": clip_fingerprint(), "mac_versions": versions(),
           "commit": _commit(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = dest / "pack_manifest.json"
    rename_prev(p)
    atomic_write(p, json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("tar", "tar_sha256", "section", "clip_fingerprint")}, indent=1))
    return man


def _commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_pack(pack: Path) -> dict:
    man = json.loads((Path(pack) / "pack_manifest.json").read_text())
    if man.get("game") != GAME or sorted(man.get("windows", {})) != sorted(WINDOWS):
        raise SystemExit(f"{WHO}: pack_manifest.json is not this check's ({man.get('game')}, {sorted(man.get('windows', {}))})")
    if sha256_file(Path(pack) / man["tar"]) != man["tar_sha256"]:
        raise SystemExit(f"{WHO}: {man['tar']} does not match pack_manifest.json")
    return man


# ── put and get (Mac, over the runtime's HTTP endpoints) ────────────────────────────────────────────────────────────
def cmd_put(a, rt=None) -> list:
    from tools import p8_colab_xfer as X
    guard_all()
    pack = Path(a.pack)
    man = check_pack(pack)
    rt = rt or X.Runtime()
    X.mkdirs(rt, REMOTE_IN)
    done = [X.put(rt, pack / "pack_manifest.json", f"{REMOTE_IN}/pack_manifest.json"),
            X.put(rt, pack / man["tar"], f"{REMOTE_IN}/{man['tar']}")]
    for d in done:
        print(json.dumps(d), flush=True)
    return done


def cmd_get(a, rt=None) -> dict:
    from tools import p8_colab_xfer as X
    guard_all()
    dest = check_outside(Path(a.dest).expanduser())
    dest.mkdir(parents=True, exist_ok=True)
    rt = rt or X.Runtime()
    tar, comp = dest / f"{RESULT}.tar", dest / f"{RESULT}.complete.json"
    for p in (tar, comp, dest / RESULT):
        rename_prev(p)
    X.get(rt, f"{REMOTE_OUT}/{RESULT}.complete.json", comp)
    info = X.get(rt, f"{REMOTE_OUT}/{RESULT}.tar", tar)
    complete = json.loads(comp.read_text())
    if sha256_file(tar) != complete["tar_sha256"]:
        raise SystemExit(f"{WHO}: {tar.name} does not match its complete.json (a run may have rewritten it; get again)")
    extract_checked(tar, complete["members"], dest)
    info.update({"arms_done": complete["arms_done"], "members": len(complete["members"])})
    print(json.dumps(info, indent=1))
    return info


# ── run (the runtime; on the Mac only as a plumbing dry run) ────────────────────────────────────────────────────────
def check_weights(want: dict, paths: dict) -> dict:
    """{name: sha256 wanted}, {name: path} -> {name: sha256}; SystemExit on a missing or different file."""
    got = {n: (sha256_file(Path(p)) if Path(p).exists() else None) for n, p in paths.items()}
    bad = sorted(n for n in want if got.get(n) != want[n])
    if bad:
        raise SystemExit(f"{WHO}: weights differ from the pack manifest: {bad}")
    return got


def result_members(out_root: Path, arms: list) -> dict:
    """{member name: local path} of the result tar: each arm's kept outputs and run_record.json."""
    files = {f"{RESULT}/run_record.json": out_root / "run_record.json"}
    for arm in arms:
        for w in WINDOWS:
            for suf in KEEP:
                p = out_root / arm / f"{w}{suf}"
                if p.exists():
                    files[f"{RESULT}/{arm}/{p.name}"] = p
    return files


def write_result(out_dir: Path, work_out: Path, record: dict) -> dict:
    """The run record, the result tar and its complete.json (written last), each through a .part file."""
    atomic_write(work_out / "run_record.json", json.dumps(record, indent=1))
    files = result_members(work_out, record["arms_done"])
    members = member_records(files)
    tar_sha = write_tar(out_dir / f"{RESULT}.tar", files)
    complete = {"result": RESULT, "arms_done": record["arms_done"], "tar_sha256": tar_sha, "members": members,
                "purpose": record["purpose"]}
    atomic_write(out_dir / f"{RESULT}.complete.json", json.dumps(complete, indent=1))
    return complete


def cmd_run(a) -> dict:
    guard_all()
    t_start = time.time()
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"{WHO}: arms must be among {sorted(ARMS)}")
    wins = [w for w in (a.windows.split(",") if a.windows else WINDOWS)]
    if any(w not in WINDOWS for w in wins):
        raise SystemExit(f"{WHO}: windows must be among {WINDOWS}")
    if not a.plumbing and (wins != list(WINDOWS) or "cuda_a" not in arms):
        raise SystemExit(f"{WHO}: a real run builds both windows and cuda_a; use --plumbing for a partial dry run")
    pack = Path(a.pack_dir)
    man = check_pack(pack)
    work, out_dir = Path(a.work), Path(a.out_dir)
    os.environ["V3_P8_ROOT"] = str(work)
    for flag in ("P8_DEVICE", "P8_DEVICE_DETECTOR", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "COURT_GRID_WEIGHTS"):
        os.environ.pop(flag, None)
    from tools import p8_frozen as F
    P = F.install_paths(check_outside(work))
    staging = extract_checked(pack / man["tar"], man["members"], work / "pack")
    tm = P["sync"] / f"{SECTION}_local_timemap.json"
    if tm.exists() and sha256_file(tm) != man["members"][f"sync/{SECTION}_local_timemap.json"]["sha256"]:
        raise SystemExit(f"{WHO}: {tm.name} in the work folder differs from the pack; use a fresh --work")
    shutil.copyfile(staging / "sync" / tm.name, tm)
    import config
    if a.install_head and not config.HEAD_V2_PATH.exists():
        config.HEAD_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging / HEAD, config.HEAD_V2_PATH)
    if Path(config.COURT_GRID_WEIGHTS).resolve() != (REPO / "models" / "court_grid_snapped.pt").resolve():
        raise SystemExit(f"{WHO}: COURT_GRID_WEIGHTS is not the frozen V3 court model")
    weights = check_weights(man["weights"], {**{rel: REPO / rel for rel in TRACKED_WEIGHTS}, HEAD: config.HEAD_V2_PATH})
    plan = frozen_plan(P)
    if plan != man["windows"]:
        raise SystemExit(f"{WHO}: the frozen window plan here differs from the pack's")
    fp = clip_fingerprint()
    if fp["sha256"] != man["clip_fingerprint"]["sha256"]:
        raise SystemExit(f"{WHO}: CLIP parameters differ from the Mac's (revision {fp['revision']})")
    from sportvu.autolabel import gate_args
    from tools.p8_device import record_of
    work_out = work / "cuda_check"
    work_out.mkdir(parents=True, exist_ok=True)
    record = {"what": "PLAN.md D161 (1) CUDA check", "design": "docs/research/P8_cuda_check.md",
              "purpose": "plumbing only, not a result" if a.plumbing else "result",
              "commit": _commit(), "pack_tar_sha256": man["tar_sha256"],
              "pack_manifest_sha256": sha256_file(pack / "pack_manifest.json"), "weights": weights,
              "clip_fingerprint": fp, "versions": versions(), "windows": wins, "arms_planned": arms,
              "arms_done": [], "builds": {}, "setup_seconds": round(time.time() - t_start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (out_dir / f"{RESULT}.tar", out_dir / f"{RESULT}.complete.json"):
        rename_prev(p)
    print(json.dumps({"setup": "ok", "seconds": record["setup_seconds"], "gpu": record["versions"].get("gpu")}), flush=True)
    for arm in arms:
        dev = ARMS[arm]
        if dev == "cuda" and not record["versions"].get("cuda_available"):
            raise SystemExit(f"{WHO}: arm {arm} needs CUDA and this runtime has none")
        os.environ["P8_DEVICE"] = dev
        arm_dir = work_out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        rec = {}
        for w in wins:
            t0 = time.time()
            r = F.safe_build(P, plan[w], staging / "clips" / f"{w}.mp4", arm_dir, extra_args=gate_args())
            rec[w] = {"ok": bool(r.get("ok")) and not r.get("empty"), "empty": bool(r.get("empty")),
                      "cached": bool(r.get("cached")), "rc": r.get("rc"), "seconds": round(time.time() - t0, 1),
                      "device": record_of(arm_dir, w), "tail": (r.get("tail") or "")[-300:]}
            print(json.dumps({"arm": arm, "window": w, **{k: rec[w][k] for k in ("ok", "seconds", "device")}}), flush=True)
        record["builds"][arm] = rec
        record["arms_done"].append(arm)
        record["total_seconds"] = round(time.time() - t_start, 1)
        complete = write_result(out_dir, work_out, record)
        print(json.dumps({"written": f"{RESULT}.tar", "arms_done": complete["arms_done"]}), flush=True)
    os.environ.pop("P8_DEVICE", None)
    print(json.dumps({"done": True, "total_seconds": record["total_seconds"]}), flush=True)
    return record


# ── compare (Mac, frozen CV python) ─────────────────────────────────────────────────────────────────────────────────
def check_result(result: Path, pack: Path | None, plumbing: bool) -> tuple:
    """The run record, checked -> (record, usable arms, {unusable arm: why}). An arm is usable when every window it
    built is ok and its device record resolves every model to the arm's device."""
    rec = json.loads((result / RESULT / "run_record.json").read_text())
    if rec["purpose"] != "result" and not plumbing:
        raise SystemExit(f"{WHO}: this result is marked {rec['purpose']!r}; compare it only with --plumbing")
    if pack is not None and sha256_file(Path(pack) / "pack_manifest.json") != rec["pack_manifest_sha256"]:
        raise SystemExit(f"{WHO}: the result was built from another pack manifest")
    usable, unusable = [], {}
    for arm in rec["arms_done"]:
        bad = {w: {"ok": b["ok"], "devices": (b.get("device") or {}).get("resolved")}
               for w, b in rec["builds"][arm].items()
               if not b["ok"] or set(((b.get("device") or {}).get("resolved") or {}).values()) != {ARMS[arm]}}
        if bad:
            unusable[arm] = bad
        else:
            usable.append(arm)
    return rec, usable, unusable


def recorded_offset_mismatch(window: str, now: dict) -> dict:
    """{Mac build: frozen offset recomputed now} -> the builds whose offset differs from the recorded one."""
    out = {}
    for k, v in now.items():
        want = RECORDED_MAC_OFFSETS[window].get(k.replace("mac_", ""))
        if want is not None and (v is None or abs(want - v) > 1e-6):
            out[k] = {"recorded": want, "now": v}
    return out


def cmd_compare(a) -> dict:
    import cv2
    from tools import p8_truth_v2 as V
    guard_all()
    result = Path(a.result).expanduser()
    rec, usable, unusable = check_result(result, Path(a.pack).expanduser() if a.pack else None, a.plumbing)
    P, index, mirror = V._game(GAME)
    plat = json.loads(PLATFORM_JSON.read_text())["windows"]
    out = {"what": "PLAN.md D161 (1) CUDA check, comparison", "design": "docs/research/P8_cuda_check.md",
           "purpose": rec["purpose"], "tolerances": TOL, "runtime": rec["versions"], "commit": rec["commit"],
           "arms_done": rec["arms_done"], "arms_unusable": unusable, "build_seconds": {arm: {w: b["seconds"] for w, b in v.items()}
                                                           for arm, v in rec["builds"].items()},
           "windows": {}}
    tests = {}
    for w in WINDOWS:
        tmj = json.loads((P["label_build"] / f"{w}_timemap.json").read_text())
        cap = cv2.VideoCapture(str(P["label_build"] / f"{w}.mp4"))
        W, H = int(cap.get(3)), int(cap.get(4))
        cap.release()
        mac = {f"mac_{k}" if k in ("cpu", "mps") else k: (P["label_build"] if d is None else V.DIAG / d) / f"{w}_frames.json"
               for k, d in V.DEVICE_BUILDS[w].items()}
        sides = {k: json.loads(p.read_text()) for k, p in mac.items()}
        for arm in usable:
            p = result / RESULT / arm / f"{w}_frames.json"
            if p.exists():
                sides[arm] = json.loads(p.read_text())
        builds = {}
        for k, side in sides.items():
            pc = V.rows_pc_of(side, tmj)
            fz = V.solve_frozen(pc, index, mirror)
            builds[k] = {"frames": len(side["frames"]), "frozen_offset_s": None if fz is None else round(float(fz), 2),
                         "v2": V.solve_v2(pc, index, mirror)}
        mac_offs = [builds[k]["frozen_offset_s"] for k in mac]
        offs = offset_set(plat[w]["colab_offset_s"], mac_offs)
        for k, side in sides.items():
            builds[k]["candidates"] = {"%.2f" % o: V.candidates(w, side, tmj, index, mirror, o, W, H) for o in offs}
        offset_flags = recorded_offset_mismatch(w, {k: builds[k]["frozen_offset_s"] for k in mac})
        fr = {k: frames_of(s) for k, s in sides.items()}
        pairs = {}
        for x, r in (("mac_cpu", "mac_mps"), ("cuda_a", "mac_cpu"), ("cuda_a", "mac_mps"), ("cuda_b", "cuda_a"),
                     ("colab_cpu", "mac_cpu"), ("cuda_a", "colab_cpu")):
            if x in fr and r in fr:
                pm = pair_metrics(fr[x], fr[r])
                t = window_tests(pm, builds[x]["candidates"], builds[r]["candidates"])
                pairs[f"{x}~{r}"] = {"metrics": pm, "tests": t}
                tests.setdefault(f"{x}~{r}", {})[w] = t
        col = "%.2f" % round(plat[w]["colab_offset_s"], 2)
        out["windows"][w] = {"offsets_s": offs, "builds": builds, "pairs": pairs, "recorded_offset_mismatch": offset_flags,
                             "r13": {"colab_offset_s": plat[w]["colab_offset_s"],
                                     "colab_candidates": plat[w]["colab_candidates"],
                                     "cuda_a_candidates_at_colab_offset": (builds.get("cuda_a") or {}).get("candidates", {}).get(col)}}
    out["reading"] = verdict(tests)
    if rec["purpose"] != "result":
        out["reading"]["verdict"] = "plumbing only, not a result"
    d = p8_root() / "cuda_check"
    d.mkdir(parents=True, exist_ok=True)
    name = "cuda_check_result.json" if rec["purpose"] == "result" else "cuda_check_plumbing.json"
    atomic_write(d / name, json.dumps(out, indent=1))
    print(json.dumps({"written": f"P8 workspace cuda_check/{name}", "reading": out["reading"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--dest", required=True)
    u = sub.add_parser("put")
    u.add_argument("--pack", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pack-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--arms", default="cuda_a,cuda_b,colab_cpu")
    r.add_argument("--windows", default="")
    r.add_argument("--install-head", action="store_true", help="copy the pack's gate head into models/ when absent")
    r.add_argument("--plumbing", action="store_true", help="a dry run: marked plumbing only, never read as a result")
    g = sub.add_parser("get")
    g.add_argument("--dest", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--result", required=True)
    c.add_argument("--pack", default=None)
    c.add_argument("--plumbing", action="store_true")
    a = ap.parse_args(argv)
    {"pack": cmd_pack, "put": cmd_put, "run": cmd_run, "get": cmd_get, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
