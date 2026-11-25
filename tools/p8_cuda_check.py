"""tools/p8_cuda_check.py: PLAN.md D161 (1), the two D99 windows of gsw_ind_2016 built on CUDA with the frozen path
and compared with the Mac CPU and MPS builds; plus one report-only window of gsw_bkn_2015 built with the r23 court
model (amendment of 2026-11-25). Design and reading rule, fixed before any run: docs/research/P8_cuda_check.md.

    <frozen CV python> -m tools.p8_cuda_check pack --dest <pack folder>                     Mac
    python3 -m tools.p8_cuda_check put --pack <pack folder>                                 Mac, COLAB_URL and COLAB_TOKEN
    python -m tools.p8_cuda_check run --pack-dir DIR --out-dir DIR --work DIR [--install-models] [--deadline-s S]
                                                                                            the runtime
    python3 -m tools.p8_cuda_check get --dest <result folder>                               Mac, COLAB_URL and COLAB_TOKEN
    <frozen CV python> -m tools.p8_cuda_check compare --result DIR --pack DIR --pin COMMIT  Mac

pack: the window clips the Mac builds read, the sections' local time maps and the two gitignored models (gate head v2,
court model r23), in one tar with a manifest (each member's sha256 and size, the tar's sha256, the frozen window plan,
the sha256 of every weights file, the CLIP parameter fingerprint, the Mac's library versions). put and get move files
over the runtime's HTTP endpoints with tools.p8_colab_xfer. run checks every hash, the plan and the fingerprint,
builds each arm (cuda_a, cuda_b, colab_cpu) through tools.p8_frozen.safe_build with P8_DEVICE, skips any arm not
started by --deadline-s, and rewrites the result tar and its complete.json after each arm. compare reads the result
and the Mac builds and applies the reading rule to the two D99 windows; the gsw_bkn_2015 window is reported only. It
writes cuda_check/cuda_check_result.json in the P8 workspace.

Only gsw_ind_2016 and gsw_bkn_2015 (R1.3 training games) are touched; every command guards them first. The module's
top level imports no model library, so tests import the pure parts with the system Python.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import tarfile
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_cuda_check"
GAME = "gsw_ind_2016"
SECTION = "gsw_ind_2016_s02"
WINDOWS = ("gsw_ind_2016_s02_w03", "gsw_ind_2016_s02_w04")          # the verdict's windows (D99)
# report-only window (amendment 2026-11-25): the most device-sensitive window known (E1 post-mortem M6), built with
# the r23 court model as the registered P8 builds are; never in the verdict
EXTRA = {"gsw_bkn_2015_s04_w00": {"game": "gsw_bkn_2015", "section": "gsw_bkn_2015_s04"}}
EXTRA_ARMS = ("cuda_a", "cuda_b")
# its Mac builds in the P8 workspace builds/: detector on CPU with court and gate on MPS (M6), and the registered MPS
MAC_EXTRA_BUILDS = {"mac_detcpu": "e1m6_r23_detcpu", "mac_mps": "r23"}
ALL_WINDOWS = WINDOWS + tuple(EXTRA)
SECTIONS = (SECTION,) + tuple(sorted({v["section"] for v in EXTRA.values()}))
TRACKED_WEIGHTS = ("models/player_detector.pt", "models/court_grid_snapped.pt", "models/court_line_seg.pt",
                   "models/thresholds.json")
HEAD = "models/trained_head_v2.joblib"
R23_COURT = "models/court_grid_r23.pt"
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
    for w, e in EXTRA.items():
        items += [e["game"], e["section"], w]
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
            src = Path(files[name])
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = src.stat().st_size, 0, 0o644, 0, 0
            with open(src, "rb") as f:
                tf.addfile(ti, f)
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
        # TF32 is used on Ampere and later (A100, L4), not on a T4; the verdict holds for the GPU recorded here
        out["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        out["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:                                   # noqa: BLE001
        pass
    return out


def rename_prev(p: Path) -> None:
    """Never overwrite a result already there: rename it to <name>.prev-<UTC time>."""
    if p.exists():
        p.rename(p.with_name(f"{p.name}.prev-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))


def round_floats(obj, nd: int = 4):
    """A copy with every float rounded, for the written json only; the rule reads the unrounded values."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    return obj


# ── comparison and reading rule (pure) ──────────────────────────────────────────────────────────────────────────────
def frames_of(side: dict) -> dict:
    return {r["frame"]: r for r in side["frames"]}


def boxes_mean(fr: dict):
    return sum(len(r["boxes"]) for r in fr.values()) / len(fr) if fr else None


def with_h(fr: dict) -> int:
    """Frames with a court homography (reported only)."""
    return sum(1 for r in fr.values() if r.get("H"))


def h_gap(a: dict, b: dict):
    """As tools.p8_platform_summary.h_gap_px, unrounded: over common frames where both have an H, the median px gap
    between the two Hs' images of three court points -> median and p90 over frames."""
    import numpy as np
    pts = np.array([[47, 25, 1], [20, 10, 1], [70, 40, 1]], float).T
    gaps = []
    for f in sorted(set(a) & set(b)):
        if a[f].get("H") and b[f].get("H"):
            ua = np.linalg.inv(np.array(a[f]["H"], float).reshape(3, 3)) @ pts
            ub = np.linalg.inv(np.array(b[f]["H"], float).reshape(3, 3)) @ pts
            gaps.append(float(np.median(np.linalg.norm(ua[:2] / ua[2] - ub[:2] / ub[2], axis=0))))
    return {"median": float(np.median(gaps)), "p90": float(np.percentile(gaps, 90)), "frames": len(gaps)} if gaps else None


def pair_metrics(x: dict, r: dict) -> dict:
    """Two builds' frames ({frame: row}) -> the measures of the design's section 3 (X against reference R), unrounded."""
    common, union = set(x) & set(r), set(x) | set(r)
    bx, br = boxes_mean(x), boxes_mean(r)
    return {"frames_x": len(x), "frames_r": len(r), "common": len(common),
            "frames_with_h_x": with_h(x), "frames_with_h_r": with_h(r),
            "frames_jaccard": len(common) / len(union) if union else None,
            "boxes_mean_x": bx, "boxes_mean_r": br,
            "boxes_mean_diff": bx - br if bx is not None and br is not None else None,
            "same_count_share": sum(len(x[f]["boxes"]) == len(r[f]["boxes"]) for f in common) / len(common)
            if common else None,
            "identical_boxes_share": sum([b["bbox"] for b in x[f]["boxes"]] == [b["bbox"] for b in r[f]["boxes"]]
                                         for f in common) / len(common) if common else None,
            "h_gap_px": h_gap(x, r)}


def offset_set(colab_offset, mac_offsets) -> list:
    """The window's fixed offsets: the P8_PLATFORM sweep, Colab's R1.3 offset and each Mac build's frozen offset."""
    vals = set(FIXED_OFFSETS) | {colab_offset} | set(mac_offsets)
    return sorted(round(float(v), 2) for v in vals if v is not None)


def cand_ok(x, r, tol: dict = TOL) -> bool:
    if x is None or r is None:
        return False
    return abs(x - r) <= max(tol["cand_floor"], tol["cand_rel"] * r)


def window_tests(pm: dict, cands_x: dict, cands_r: dict, tol: dict = TOL) -> dict:
    """The five tests of one window, X against R, on unrounded values. cands_*: {offset key: candidates}; every key
    of cands_r is tested."""
    h = pm.get("h_gap_px") or {}
    return {"frames_jaccard": pm["frames_jaccard"] is not None and pm["frames_jaccard"] >= tol["frames_jaccard_min"],
            "boxes_mean": pm["boxes_mean_diff"] is not None and abs(pm["boxes_mean_diff"]) <= tol["boxes_mean_abs_max"],
            "same_count": pm["same_count_share"] is not None and pm["same_count_share"] >= tol["same_count_share_min"],
            "court_h_gap": h.get("median") is not None and h["median"] <= tol["h_gap_median_px_max"],
            "candidates": bool(cands_r) and all(cand_ok(cands_x.get(k), v, tol) for k, v in cands_r.items())}


def matches(per_window: dict) -> bool:
    """per_window: {window: tests dict}; a match needs every test on every one of the verdict's windows."""
    return all(w in per_window for w in WINDOWS) and all(all(per_window[w].values()) for w in WINDOWS)


def failed(per_window: dict) -> dict:
    return {w: sorted(k for k, v in t.items() if not v) for w, t in per_window.items() if not all(t.values())}


def verdict(tests: dict) -> dict:
    """tests: {"X~R": {window: tests}} for the pairs present (D99 windows only) -> the verdict of section 4.
    cuda_a against colab_cpu is reported in failed_tests and never decides the verdict."""
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
    if v == "matches neither" and out["colab_cpu_matches_mac_cpu"] is None:
        out["attribution"] = "colab_cpu not built: a mismatch cannot be traced to the device or the libraries"
    return out


def run_flags(rec: dict, man: dict | None, pin: str | None) -> dict:
    """The compare flags (amendment 2026-11-25): pin, CLIP revision and transformers version against the pack."""
    commit = rec.get("commit") or ""
    mac_t = ((man or {}).get("mac_versions") or {}).get("transformers")
    run_t = (rec.get("versions") or {}).get("transformers")
    return {"commit": commit, "pin": pin,
            "pin_mismatch": None if not pin else not (commit and (commit.startswith(pin) or pin.startswith(commit))),
            "clip_revision_differs": rec.get("clip_revision_matches_manifest") is False,
            "transformers_mac": mac_t, "transformers_runtime": run_t,
            "transformers_differs": None if man is None else mac_t != run_t}


# ── pack (Mac, frozen CV python) ────────────────────────────────────────────────────────────────────────────────────
def section_record(P: dict, section: str) -> dict:
    vid = P["video"] / f"{section}.mp4"
    rec = {"section": section, "sha256": sha256_file(vid) if vid.exists() else None, "manifest_r13_sha256": None}
    if os.environ.get("DATA_ROOT"):
        man = Path(os.environ["DATA_ROOT"]) / "sportvu" / "MANIFEST_r13.json"
        if man.exists():
            rec["manifest_r13_sha256"] = (json.loads(man.read_text())["files"].get(f"video/{section}.mp4") or {}).get("sha256")
    rec["matches_r13"] = None if not (rec["sha256"] and rec["manifest_r13_sha256"]) else rec["sha256"] == rec["manifest_r13_sha256"]
    return rec


def frozen_plan(P: dict) -> dict:
    from sportvu.autolabel import EVERY_SPAN
    from sportvu.rebuild import plan_windows
    plan = {w["window"]: w for s in SECTIONS for w in plan_windows(s, budget_s=EVERY_SPAN)}
    missing = [w for w in ALL_WINDOWS if w not in plan]
    if missing:
        raise SystemExit(f"{WHO}: the frozen plan of {SECTIONS} has no {missing}")
    return json.loads(json.dumps({w: plan[w] for w in ALL_WINDOWS}))


def tracked_weights() -> dict:
    return {rel: sha256_file(REPO / rel) for rel in TRACKED_WEIGHTS}


def cmd_pack(a) -> dict:
    guard_all()
    dest = check_outside(Path(a.dest).expanduser())
    dest.mkdir(parents=True, exist_ok=True)
    from tools import p8_frozen as F
    P = F.install_paths()
    import config
    files = {f"clips/{w}.mp4": P["label_build"] / f"{w}.mp4" for w in ALL_WINDOWS}
    for s in SECTIONS:
        files[f"sync/{s}_local_timemap.json"] = P["sync"] / f"{s}_local_timemap.json"
    files[HEAD] = config.HEAD_V2_PATH
    files[R23_COURT] = F.court_weights("r23")          # checked against reports/r23_verification.json
    members = member_records(files)
    tar = dest / f"{PACK}.tar"
    rename_prev(tar)
    tar_sha = write_tar(tar, files)
    weights = {**tracked_weights(), HEAD: members[HEAD]["sha256"], R23_COURT: members[R23_COURT]["sha256"]}
    man = {"what": "PLAN.md D161 (1) CUDA check pack", "design": "docs/research/P8_cuda_check.md", "game": GAME,
           "report_only_windows": sorted(EXTRA), "sections": {s: section_record(P, s) for s in SECTIONS},
           "windows": frozen_plan(P), "tar": tar.name, "tar_sha256": tar_sha, "members": members, "weights": weights,
           "clip_fingerprint": clip_fingerprint(), "mac_versions": versions(), "commit": _commit(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = dest / "pack_manifest.json"
    rename_prev(p)
    atomic_write(p, json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("tar", "tar_sha256", "sections", "clip_fingerprint")}, indent=1))
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
    if man.get("game") != GAME or sorted(man.get("windows", {})) != sorted(ALL_WINDOWS):
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


def arm_windows(arm: str, wins: list, plumbing: bool = False) -> list:
    """The windows an arm builds: the D99 windows always; the report-only window in the cuda arms (any arm in a
    plumbing run)."""
    return [w for w in wins if w in WINDOWS or (w in EXTRA and (arm in EXTRA_ARMS or plumbing))]


def past_deadline(t_start: float, deadline_s, now: float | None = None) -> bool:
    return deadline_s is not None and ((now if now is not None else time.time()) - t_start) > deadline_s


def build_window(safe_build, P: dict, w_plan: dict, snip: Path, arm_dir: Path, extra_args: tuple, court=None) -> dict:
    """One window through safe_build; a report-only window gets COURT_GRID_WEIGHTS (the r23 court model) for its
    build subprocess only, and the variable is removed afterwards whatever happens."""
    if court is not None:
        os.environ["COURT_GRID_WEIGHTS"] = str(court)
    try:
        return safe_build(P, w_plan, snip, arm_dir, extra_args=extra_args)
    finally:
        os.environ.pop("COURT_GRID_WEIGHTS", None)


def result_members(out_root: Path, arms: list) -> dict:
    """{member name: local path} of the result tar: each arm's kept outputs and run_record.json."""
    files = {f"{RESULT}/run_record.json": out_root / "run_record.json"}
    for arm in arms:
        for w in ALL_WINDOWS:
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
    wins = [w for w in (a.windows.split(",") if a.windows else ALL_WINDOWS)]
    if any(w not in ALL_WINDOWS for w in wins):
        raise SystemExit(f"{WHO}: windows must be among {ALL_WINDOWS}")
    if not a.plumbing and (not set(WINDOWS) <= set(wins) or "cuda_a" not in arms):
        raise SystemExit(f"{WHO}: a real run builds both D99 windows and cuda_a; use --plumbing for a partial dry run")
    pack = Path(a.pack_dir)
    man = check_pack(pack)
    work, out_dir = Path(a.work), Path(a.out_dir)
    os.environ["V3_P8_ROOT"] = str(work)
    for flag in ("P8_DEVICE", "P8_DEVICE_DETECTOR", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "COURT_GRID_WEIGHTS"):
        os.environ.pop(flag, None)
    from tools import p8_frozen as F
    P = F.install_paths(check_outside(work))
    staging = extract_checked(pack / man["tar"], man["members"], work / "pack")
    for s in SECTIONS:
        tm = P["sync"] / f"{s}_local_timemap.json"
        if tm.exists() and sha256_file(tm) != man["members"][f"sync/{tm.name}"]["sha256"]:
            raise SystemExit(f"{WHO}: {tm.name} in the work folder differs from the pack; use a fresh --work")
        shutil.copyfile(staging / "sync" / tm.name, tm)
    import config
    r23_path = main_checkout() / "models" / Path(R23_COURT).name
    if a.install_models:
        for src, dst in ((staging / HEAD, config.HEAD_V2_PATH), (staging / R23_COURT, r23_path)):
            if not Path(dst).exists():
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    if Path(config.COURT_GRID_WEIGHTS).resolve() != (REPO / "models" / "court_grid_snapped.pt").resolve():
        raise SystemExit(f"{WHO}: COURT_GRID_WEIGHTS is not the frozen V3 court model")
    weights = check_weights(man["weights"], {**{rel: REPO / rel for rel in TRACKED_WEIGHTS}, HEAD: config.HEAD_V2_PATH,
                                             R23_COURT: r23_path})
    r23_path = F.court_weights("r23")               # and against reports/r23_verification.json
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
              "clip_fingerprint": fp,
              "clip_revision_matches_manifest": fp["revision"] == man["clip_fingerprint"].get("revision"),
              "versions": versions(), "windows": wins, "report_only_windows": sorted(EXTRA), "arms_planned": arms,
              "deadline_s": a.deadline_s, "arms_done": [], "arms_skipped": {}, "builds": {},
              "setup_seconds": round(time.time() - t_start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (out_dir / f"{RESULT}.tar", out_dir / f"{RESULT}.complete.json"):
        rename_prev(p)
    print(json.dumps({"setup": "ok", "seconds": record["setup_seconds"], "gpu": record["versions"].get("gpu"),
                      "clip_revision_matches_manifest": record["clip_revision_matches_manifest"]}), flush=True)
    for arm in arms:
        if past_deadline(t_start, a.deadline_s):
            record["arms_skipped"][arm] = "deadline"
            print(json.dumps({"arm": arm, "skipped": "deadline", "elapsed_s": round(time.time() - t_start, 1)}), flush=True)
            continue
        dev = ARMS[arm]
        if dev == "cuda" and not record["versions"].get("cuda_available"):
            raise SystemExit(f"{WHO}: arm {arm} needs CUDA and this runtime has none")
        os.environ["P8_DEVICE"] = dev
        arm_dir = work_out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        rec = {}
        for w in arm_windows(arm, wins, a.plumbing):
            t0 = time.time()
            court = r23_path if w in EXTRA else None
            r = build_window(F.safe_build, P, plan[w], staging / "clips" / f"{w}.mp4", arm_dir, gate_args(), court)
            rec[w] = {"ok": bool(r.get("ok")) and not r.get("empty"), "empty": bool(r.get("empty")),
                      "cached": bool(r.get("cached")), "rc": r.get("rc"), "seconds": round(time.time() - t0, 1),
                      "court_weights": Path(court).name if court else "court_grid_snapped.pt",
                      "device": record_of(arm_dir, w), "tail": (r.get("tail") or "")[-300:]}
            print(json.dumps({"arm": arm, "window": w, **{k: rec[w][k] for k in ("ok", "seconds", "court_weights", "device")}}),
                  flush=True)
        record["builds"][arm] = rec
        record["arms_done"].append(arm)
        record["total_seconds"] = round(time.time() - t_start, 1)
        complete = write_result(out_dir, work_out, record)
        print(json.dumps({"written": f"{RESULT}.tar", "arms_done": complete["arms_done"]}), flush=True)
    os.environ.pop("P8_DEVICE", None)
    record["total_seconds"] = round(time.time() - t_start, 1)
    if record["arms_skipped"]:
        write_result(out_dir, work_out, record)
    print(json.dumps({"done": True, "total_seconds": record["total_seconds"], "arms_done": record["arms_done"],
                      "arms_skipped": record["arms_skipped"]}), flush=True)
    return record


# ── compare (Mac, frozen CV python) ─────────────────────────────────────────────────────────────────────────────────
def check_result(result: Path, pack: Path | None, plumbing: bool) -> tuple:
    """The run record, checked -> (record, usable arms, {unusable arm: why}). An arm is usable when every window it
    built is ok, its device record resolves every model to the arm's device, and each window used its court model."""
    rec = json.loads((result / RESULT / "run_record.json").read_text())
    if rec["purpose"] != "result" and not plumbing:
        raise SystemExit(f"{WHO}: this result is marked {rec['purpose']!r}; compare it only with --plumbing")
    if pack is not None and sha256_file(Path(pack) / "pack_manifest.json") != rec["pack_manifest_sha256"]:
        raise SystemExit(f"{WHO}: the result was built from another pack manifest")
    usable, unusable = [], {}
    for arm in rec["arms_done"]:
        bad = {}
        for w, b in rec["builds"][arm].items():
            res = (b.get("device") or {}).get("resolved") or {}
            court = b.get("court_weights", "court_grid_snapped.pt")
            want_court = Path(R23_COURT).name if w in EXTRA else "court_grid_snapped.pt"
            if not b["ok"] or set(res.values()) != {ARMS[arm]} or court != want_court:
                bad[w] = {"ok": b["ok"], "devices": res or None, "court_weights": court}
        if any(w in WINDOWS for w in bad) or not any(w in WINDOWS for w in rec["builds"][arm]):
            unusable[arm] = bad or {"windows": "no D99 window"}
        else:
            usable.append(arm)
            if bad:
                unusable[f"{arm} (report-only window)"] = bad
    return rec, usable, unusable


def recorded_offset_mismatch(window: str, now: dict) -> dict:
    """{Mac build: frozen offset recomputed now} -> the builds whose offset differs from the recorded one."""
    out = {}
    for k, v in now.items():
        want = RECORDED_MAC_OFFSETS.get(window, {}).get(k.replace("mac_", ""))
        if want is not None and (v is None or abs(want - v) > 1e-6):
            out[k] = {"recorded": want, "now": v}
    return out


_GAMES: dict = {}


def game_index(V, game: str):
    """tools.p8_truth_v2._game (workspace, SportVU index, mirror), loaded once per game."""
    if game not in _GAMES:
        _GAMES[game] = V._game(game)
    return _GAMES[game]


def _window_block(V, w: str, game: str, sides: dict, mac_keys: list, extra_offsets: list, pair_list: tuple,
                  wh: tuple) -> tuple:
    """Builds, pairs and tests of one window (frozen helpers of tools.p8_truth_v2, read only)."""
    P, index, mirror = game_index(V, game)
    tmj = json.loads((P["label_build"] / f"{w}_timemap.json").read_text())
    builds = {}
    for k, side in sides.items():
        pc = V.rows_pc_of(side, tmj)
        fz = V.solve_frozen(pc, index, mirror)
        builds[k] = {"frames": len(side["frames"]), "frames_with_h": with_h(frames_of(side)),
                     "frozen_offset_s": None if fz is None else round(float(fz), 2), "v2": V.solve_v2(pc, index, mirror)}
    offs = offset_set(None, [builds[k]["frozen_offset_s"] for k in mac_keys] + list(extra_offsets))
    for k, side in sides.items():
        builds[k]["candidates"] = {"%.2f" % o: V.candidates(w, side, tmj, index, mirror, o, *wh) for o in offs}
    fr = {k: frames_of(s) for k, s in sides.items()}
    pairs, tests = {}, {}
    for x, r in pair_list:
        if x in fr and r in fr:
            pm = pair_metrics(fr[x], fr[r])
            t = window_tests(pm, builds[x]["candidates"], builds[r]["candidates"])
            pairs[f"{x}~{r}"] = {"metrics": pm, "tests": t}
            tests[f"{x}~{r}"] = t
    return offs, builds, pairs, tests


def cmd_compare(a) -> dict:
    import cv2
    from tools import p8_truth_v2 as V
    guard_all()
    result = Path(a.result).expanduser()
    pack = Path(a.pack).expanduser() if a.pack else None
    rec, usable, unusable = check_result(result, pack, a.plumbing)
    if rec["purpose"] == "result" and (pack is None or not a.pin):
        raise SystemExit(f"{WHO}: a result is compared only with --pack and --pin (the pushed commit the run used)")
    man = json.loads((pack / "pack_manifest.json").read_text()) if pack else None
    flags = run_flags(rec, man, a.pin)
    P = game_index(V, GAME)[0]
    plat = json.loads(PLATFORM_JSON.read_text())["windows"]

    def wh(w):
        cap = cv2.VideoCapture(str(P["label_build"] / f"{w}.mp4"))
        out = (int(cap.get(3)), int(cap.get(4)))
        cap.release()
        return out

    def arm_sides(w):
        return {arm: json.loads((result / RESULT / arm / f"{w}_frames.json").read_text()) for arm in usable
                if (result / RESULT / arm / f"{w}_frames.json").exists()}

    out = {"what": "PLAN.md D161 (1) CUDA check, comparison", "design": "docs/research/P8_cuda_check.md",
           "purpose": rec["purpose"], "tolerances": TOL, "flags": flags, "runtime": rec["versions"],
           "arms_done": rec["arms_done"], "arms_skipped": rec.get("arms_skipped", {}), "arms_unusable": unusable,
           "build_seconds": {arm: {w: b["seconds"] for w, b in v.items()} for arm, v in rec["builds"].items()},
           "windows": {}, "report_only": {}}
    tests = {}
    for w in WINDOWS:
        mac = {f"mac_{k}" if k in ("cpu", "mps") else k: (P["label_build"] if d is None else V.DIAG / d) / f"{w}_frames.json"
               for k, d in V.DEVICE_BUILDS[w].items()}
        sides = {**{k: json.loads(p.read_text()) for k, p in mac.items()}, **arm_sides(w)}
        offs, builds, pairs, t = _window_block(
            V, w, GAME, sides, list(mac), [plat[w]["colab_offset_s"]],
            (("mac_cpu", "mac_mps"), ("cuda_a", "mac_cpu"), ("cuda_a", "mac_mps"), ("cuda_b", "cuda_a"),
             ("colab_cpu", "mac_cpu"), ("cuda_a", "colab_cpu")), wh(w))
        for k, v in t.items():
            tests.setdefault(k, {})[w] = v
        col = "%.2f" % round(plat[w]["colab_offset_s"], 2)
        out["windows"][w] = {"offsets_s": offs, "builds": builds, "pairs": pairs,
                             "recorded_offset_mismatch": recorded_offset_mismatch(w, {k: builds[k]["frozen_offset_s"] for k in mac}),
                             "r13": {"colab_offset_s": plat[w]["colab_offset_s"],
                                     "colab_candidates": plat[w]["colab_candidates"],
                                     "cuda_a_candidates_at_colab_offset": (builds.get("cuda_a") or {}).get("candidates", {}).get(col)}}
    for w, e in EXTRA.items():                          # report only: never enters the verdict
        built = arm_sides(w)
        if not built:
            continue
        Pe = game_index(V, e["game"])[0]
        sides = {k: json.loads((Pe["builds"] / d / f"{w}_frames.json").read_text()) for k, d in MAC_EXTRA_BUILDS.items()}
        stored = json.loads((Pe["labels"] / f"{e['game']}_offsets.json").read_text()).get(w)
        offs, builds, pairs, _ = _window_block(
            V, w, e["game"], {**sides, **built}, list(MAC_EXTRA_BUILDS), [stored],
            (("mac_detcpu", "mac_mps"), ("cuda_a", "mac_detcpu"), ("cuda_a", "mac_mps"), ("cuda_b", "cuda_a")), wh(w))
        out["report_only"][w] = {"note": "report only, not in the verdict; mac_detcpu has the detector on CPU and the "
                                         "court and gate on MPS (E1 post-mortem M6); mac_mps is the registered r23 build",
                                 "offsets_s": offs, "stored_label_offset_s": stored, "builds": builds, "pairs": pairs}
    out["reading"] = verdict(tests)
    if rec["purpose"] != "result":
        out["reading"]["verdict"] = "plumbing only, not a result"
    d = p8_root() / "cuda_check"
    d.mkdir(parents=True, exist_ok=True)
    name = "cuda_check_result.json" if rec["purpose"] == "result" else "cuda_check_plumbing.json"
    atomic_write(d / name, json.dumps(round_floats(out), indent=1))
    print(json.dumps({"written": f"P8 workspace cuda_check/{name}", "flags": flags,
                      "reading": round_floats(out["reading"])}, indent=1))
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
    r.add_argument("--install-models", action="store_true",
                   help="copy the pack's gate head and r23 court model into models/ when absent (the runtime)")
    r.add_argument("--deadline-s", type=float, default=None,
                   help="skip any arm not started this many seconds after the run began")
    r.add_argument("--plumbing", action="store_true", help="a dry run: marked plumbing only, never read as a result")
    g = sub.add_parser("get")
    g.add_argument("--dest", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--result", required=True)
    c.add_argument("--pack", default=None)
    c.add_argument("--pin", default=None, help="the pushed commit the runtime was pinned to")
    c.add_argument("--plumbing", action="store_true")
    a = ap.parse_args(argv)
    {"pack": cmd_pack, "put": cmd_put, "run": cmd_run, "get": cmd_get, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
