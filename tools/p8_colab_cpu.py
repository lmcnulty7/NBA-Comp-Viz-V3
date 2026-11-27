"""tools/p8_colab_cpu.py: does a Colab CPU runtime reproduce this Mac's builds exactly? Design and reading rule, fixed
before any run: docs/research/P8_colab_cpu_parity.md.

    <frozen CV python> -m tools.p8_colab_cpu pack --dest <pack folder>                               Mac, no lock
    <frozen CV python> -m tools.p8_colab_cpu run --platform mac --arms mac --windows W,W ...          Mac, in the lock
    python -m tools.p8_colab_cpu run --platform colab --arms plain,isa,plain_b ...                    the runtime
    python3 -m tools.p8_colab_cpu unpack --dest <result folder> --name <result> [--expect-sha256 S]  Mac
    python3 -m tools.p8_colab_cpu compare --mac <result folder> --colab <result folder> --pack <pack folder>
    (internal) <python> -m tools.p8_colab_cpu probe

Every build is the reset-fix build path with every model on CPU: tools.p8_frozen.safe_build, P8_DEVICE=cpu,
P8_TRACKER_RESET_FIX=1, the r23 detector (models/player_detector.pt) and the r23 court model (COURT_GRID_WEIGHTS),
gate v2. The Mac side builds the same windows from the same pack with the same command, so the two sides differ only
in the platform. Colab arms: plain (the pinned stack as it installs), isa (the same, with every library's CPU
instruction set capped at what the Mac's x86_64 process uses: it runs under Rosetta and reaches SSE4.2 only),
plain_b (plain again, repeatability). A build starts only if its predicted time fits before --end-at; after each
build the result tar is rewritten and, when DRIVE_TOKEN and --drive-dir-id are given, uploaded to Drive.

Only gsw_cha_2016 (a pool development game) is touched; every command guards it, its sections and its windows. The
top level imports no model library, so tests import the pure parts with the system Python.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, guard, main_checkout, p8_root
from tools.p8_cuda_check import (extract_checked, h_gap, member_records, rename_prev, round_floats, sha256_file,
                                 write_tar)
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_colab_cpu"
GAME = "gsw_cha_2016"
RESET_W, NORESET_W = "gsw_cha_2016_s02_w03", "gsw_cha_2016_s00_w01"      # the two core windows (short)
TAIL = ("gsw_cha_2016_s00_w00", "gsw_cha_2016_s05_w00")                 # the lead's named windows, if time allows
CORE = (RESET_W, NORESET_W)
ALL_WINDOWS = CORE + TAIL
SECTIONS = tuple(sorted({w.rsplit("_", 1)[0] for w in ALL_WINDOWS}))
# the runtime's build order, fixed before any run; a step starts only if its predicted time fits
ORDER = (("plain", RESET_W), ("plain", NORESET_W), ("isa", RESET_W), ("isa", NORESET_W), ("plain_b", RESET_W),
         ("plain", TAIL[0]), ("plain", TAIL[1]))
COLAB_ARMS = ("plain", "isa", "plain_b")
MAC_ARM = "mac"
RATE_FAMILY = {"plain": "plain", "plain_b": "plain", "isa": "isa", "mac": "mac"}
ISA_FIRST_FACTOR = 3.0          # predicted isa rate before any isa build: the plain rate times this
TRACKED_WEIGHTS = ("models/player_detector.pt", "models/court_line_seg.pt", "models/thresholds.json")
HEAD = "models/trained_head_v2.joblib"
R23_COURT = "models/court_grid_r23.pt"
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
CODE = ("tools/p8_colab_cpu.py",)
PACK = "p8_colabcpu_pack"
KEEP = ("_frames.json", "_trajectories.json", "_identity.json", "_device.json", "_resetfix.json")
SIDE = ("frames", "trajectories", "identity")
BUILD_FLAGS = ("P8_DEVICE", "P8_DEVICE_DETECTOR", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "P8_MPS_CLIP_FIX",
               "P8_TRACKER_RESET_FIX", "COURT_GRID_WEIGHTS", "TRACKER_REID", "P8_RESETFIX_DETECTOR")
ISA_KEYS = ("ATEN_CPU_CAPABILITY", "DNNL_MAX_CPU_ISA", "ONEDNN_MAX_CPU_ISA", "MKL_ENABLE_INSTRUCTIONS",
            "OPENCV_CPU_DISABLE", "NPY_DISABLE_CPU_FEATURES")

# the runtime environment: Python and every build dependency at the Mac's frozen CV versions (pip names)
PYTHON_PIN = "3.12.2"
TORCH_PINS = ("torch==2.2.2", "torchvision==0.17.2")                      # the CPU wheels
CORE_PINS = ("numpy==1.26.4", "scipy==1.13.1", "transformers==4.49.0",
             "tokenizers==0.21.4", "safetensors==0.7.0", "huggingface_hub==0.36.2", "lap==0.5.13", "pillow==12.2.0",
             "scikit-learn==1.8.0", "joblib==1.4.2", "threadpoolctl==3.5.0", "matplotlib==3.10.9")
# without dependencies: ultralytics asks for opencv-python (a second cv2); the opencv-python-headless 4.13.0.92 wheel
# declares numpy>=2, yet the Mac runs it with numpy 1.26.4 (installed the same way), so it goes in as the Mac has it
NODEPS_PINS = ("opencv-python-headless==4.13.0.92", "ultralytics==8.4.48", "ultralytics-thop==2.0.19")
SOFT_PINS = ("pandas==3.0.3", "pyarrow==22.0.0", "beautifulsoup4==4.12.3", "lxml==5.2.1", "requests==2.34.2",
             "tqdm==4.66.5", "PyYAML==6.0.1", "psutil==5.9.0", "filelock==3.13.1", "packaging==24.1",
             "polars==1.40.1", "python-dateutil==2.9.0.post0", "pytz==2022.7.1", "pyparsing==3.1.2",
             "cycler==0.11.0", "fonttools==4.51.0", "contourpy==1.2.0", "kiwisolver==1.4.4", "regex==2026.5.9",
             "typing_extensions==4.15.0", "sympy==1.13.2", "networkx==3.3", "jinja2==3.1.4", "fsspec==2024.6.1",
             "mpmath==1.3.0", "soupsieve==2.5")
REPO_URL = "https://github.com/lmcnulty7/NBA-Comp-Viz-V3.git"
RT = {"venv": "/content/venv", "repo": "/content/repo", "in": "/content/p8_in", "work": "/content/p8_work",
      "out": "/content/p8_out", "ids": "/content/p8_drive_ids.json"}
END_AFTER_S = 2820              # no build starts unless it fits within 47 min of the job's start


def job_script(base: str, folder_id: str, end_after_s: int = END_AFTER_S) -> str:
    """The one detached runtime job: the pinned venv, the clone at the pushed base commit, the Drive fetch, the
    tool overlaid from the pack (sha256 checked by fetch and by check_pack), then the run, which uploads after
    every build. The token sits in an unexported shell variable and reaches the fetch and the run only."""
    if not re.fullmatch(r"[0-9a-f]{40}", base) or not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise SystemExit(f"{WHO}: base must be a full commit sha and the folder id a Drive id")
    py, v = f"{RT['venv']}/bin/python", RT["venv"]
    soft = " ".join(SOFT_PINS)
    return f"""set -euo pipefail
T="$DRIVE_TOKEN"; unset DRIVE_TOKEN
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
END_AT=$(( $(date +%s) + {end_after_s} ))
stamp() {{ echo "JOB $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }}
trap 'echo "JOB FAILED rc=$? line=$LINENO $(date -u +%Y-%m-%dT%H:%M:%SZ)"' ERR
stamp start
pip -q install uv && uv --version
uv python install -q {PYTHON_PIN}
uv venv -q --python {PYTHON_PIN} {v}
uv pip install -q --python {py} --index-url https://download.pytorch.org/whl/cpu {' '.join(TORCH_PINS)}
printf '%s\\n' {' '.join(TORCH_PINS + CORE_PINS)} > /content/p8_core.txt
uv pip install -q --python {py} -c /content/p8_core.txt {' '.join(CORE_PINS)}
uv pip install -q --python {py} --no-deps {' '.join(NODEPS_PINS)}
{py} -c "import cv2, numpy, torch; print('CHECK1 cv2', cv2.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"
uv pip install -q --python {py} -c /content/p8_core.txt {soft} || for p in {soft}; do
  uv pip install -q --python {py} -c /content/p8_core.txt "$p" || {{ echo "PIN_FAILED $p"; uv pip install -q --python {py} -c /content/p8_core.txt "${{p%%==*}}"; }}
done
{py} -c "import cv2, numpy, torch, ultralytics; print('CHECK2 cv2', cv2.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__, 'ultralytics', ultralytics.__version__)"
stamp env
[ -d {RT['repo']} ] || git clone -q --filter=blob:none {REPO_URL} {RT['repo']}
cd {RT['repo']} && git fetch -q origin && git checkout -q --detach {base} && git log --oneline -1
stamp clone
DRIVE_TOKEN="$T" {py} -m tools.p8_drive_xfer fetch --ids {RT['ids']} --dest {RT['in']}
cp {RT['in']}/p8_colab_cpu.py tools/p8_colab_cpu.py
{py} build_trajectories.py --help > /dev/null && echo "imports ok"
stamp fetch
DRIVE_TOKEN="$T" {py} -m tools.p8_colab_cpu run --platform colab --pack-dir {RT['in']} --work {RT['work']} \\
  --out-dir {RT['out']} --arms {','.join(COLAB_ARMS)} --end-at "$END_AT" --drive-dir-id {folder_id} --install-models
stamp end
"""


# the reading rule (docs/research/P8_colab_cpu_parity.md section 4); never changed after a Colab build is seen
BOX_TOL_PX = 1.0                # a box matches when every bbox coordinate is within 1 px (bboxes are integers)
GATE_PX = 32.0                  # pairs farther than this are unmatched boxes, not offsets
NEAR = {"frames_jaccard_min": 1.0, "frames_within_1px_min": 0.99, "h_gap_median_px_max": 1.0}


def result_name(platform_label: str) -> str:
    return f"p8_colabcpu_{platform_label}"


def guard_all(who: str = WHO) -> None:
    items = [GAME, *SECTIONS, *ALL_WINDOWS]
    refuse_cv_test(items, who)
    guard(items, who)


# ── probe: versions, instruction sets and a numeric fingerprint of each library ────────────────────────────────────
def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def distributions() -> dict:
    from importlib import metadata
    return dict(sorted({(d.metadata["Name"] or "").lower(): d.version for d in metadata.distributions()}.items()))


def numeric_fingerprint() -> dict:
    """sha256 of fixed small computations (seeded numpy inputs) per library: equal hashes on two platforms mean that
    operation computes the same bits there."""
    import hashlib
    import cv2
    import numpy as np
    import torch
    rs = np.random.RandomState(0)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    x = rs.standard_normal((1, 32, 64, 64)).astype(np.float32)
    w = rs.standard_normal((32, 32, 3, 3)).astype(np.float32)
    a = rs.standard_normal((256, 256)).astype(np.float32)
    img = (rs.random_sample((360, 640, 3)) * 255).astype(np.float32)
    with torch.no_grad():
        conv = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w), padding=1).numpy()
        mm = (torch.from_numpy(a) @ torch.from_numpy(a)).numpy()
        sm = torch.softmax(torch.from_numpy(a), dim=1).numpy()
    return {"torch_conv2d": h(conv), "torch_matmul": h(mm), "torch_softmax": h(sm),
            "cv2_resize": h(cv2.resize(img, (320, 180), interpolation=cv2.INTER_LINEAR)),
            "cv2_gaussian": h(cv2.GaussianBlur(img, (5, 5), 1.2)),
            "numpy_exp_sum": h(np.array([np.exp(a).sum(dtype=np.float32)], np.float32)),
            "numpy_matmul": h(a @ a)}


def probe_here() -> dict:
    import cv2
    import numpy as np
    import torch
    try:
        from numpy.core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    except ImportError:                                 # numpy 2
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    out = {"python": platform.python_version(), "python_build": platform.python_implementation() + " " +
           " ".join(platform.python_build()), "platform": platform.platform(terse=True), "machine": platform.machine(),
           "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
           "torch": torch.__version__, "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
           "torch_mkldnn": bool(torch.backends.mkldnn.is_available()),
           "torch_nnpack": bool(torch.backends.nnpack.is_available()) if hasattr(torch.backends, "nnpack") else None,
           "torch_config": torch.__config__.show(), "cv2": cv2.__version__,
           "cv2_cpu_features": cv2.getCPUFeaturesLine(), "cv2_threads": cv2.getNumThreads(),
           "cv2_build_cpu": cv2_build_cpu(cv2.getBuildInformation()), "numpy": np.__version__,
           "numpy_cpu_found": sorted(k for k, v in __cpu_features__.items() if v),
           "numpy_dispatch": sorted(__cpu_dispatch__), "isa_env": {k: os.environ.get(k) for k in ISA_KEYS},
           "fingerprint": numeric_fingerprint(), "distributions": distributions()}
    for mod in ("ultralytics", "transformers", "scipy", "lap", "PIL", "sklearn", "torchvision"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:                               # noqa: BLE001 (record what is missing)
            out[mod] = None
    return out


def cv2_build_cpu(info: str) -> list:
    """The CPU lines of cv2.getBuildInformation(): version control, baseline and dispatched code generation."""
    keep = ("Version control", "Baseline", "Dispatched code generation", "requested")
    return [ln.strip() for ln in info.splitlines() if ln.strip().startswith(keep)][:6]


def parse_verbose(text: str) -> dict:
    """The instruction set lines oneDNN and MKL print with ONEDNN_VERBOSE=1 and MKL_VERBOSE=1."""
    out = {"onednn_isa": None, "mkl": None}
    for ln in text.splitlines():
        if ln.startswith("onednn_verbose,info,cpu,isa:") and out["onednn_isa"] is None:
            out["onednn_isa"] = ln.split("isa:", 1)[1].strip()
        if ln.startswith("MKL_VERBOSE") and "enabled processors" in ln and out["mkl"] is None:
            out["mkl"] = ln.strip()[:300]
    return out


def run_probe(extra_env: dict | None = None, python: str | None = None) -> dict:
    """The probe in a fresh subprocess (environment variables take effect at library load) -> its record."""
    env = {k: v for k, v in os.environ.items() if k not in ISA_KEYS and k != "DRIVE_TOKEN"}
    env.update(extra_env or {})
    env.update({"ONEDNN_VERBOSE": "1", "DNNL_VERBOSE": "1", "MKL_VERBOSE": "1"})
    r = subprocess.run([python or sys.executable, "-m", WHO, "probe"], cwd=REPO, env=env, capture_output=True,
                       text=True)
    rec = None
    for ln in r.stdout.splitlines():
        if ln.startswith("PROBE "):
            rec = json.loads(ln[6:])
    if rec is None:
        raise SystemExit(f"{WHO}: the probe failed (rc {r.returncode}): {r.stderr[-500:]}")
    rec.update(parse_verbose(r.stdout + "\n" + r.stderr))
    return rec


def cmd_probe(_a) -> dict:
    import torch
    rec = probe_here()
    x = torch.ones(1, 8, 32, 32)                        # trigger oneDNN's and MKL's verbose info lines
    with torch.no_grad():
        torch.nn.functional.conv2d(x, torch.ones(8, 8, 3, 3), padding=1)
        torch.ones(64, 64) @ torch.ones(64, 64)
    print("PROBE " + json.dumps(rec), flush=True)
    return rec


# ── the isa arm's environment (pure) ────────────────────────────────────────────────────────────────────────────────
def cv2_available(line: str) -> set:
    """cv2.getCPUFeaturesLine() -> the features this process can use, named as OpenCV prints them (OPENCV_CPU_DISABLE
    takes the same names); an item ending in '?' is not available, a leading '*' marks dispatched code."""
    return {t.lstrip("*") for t in line.split() if not t.endswith("?")}


def cv2_listed(line: str) -> set:
    return {t.lstrip("*").rstrip("?") for t in line.split()}


def isa_env(mac: dict, here: dict) -> dict:
    """The environment that caps every library on this runtime at the Mac's instruction sets (both probes taken
    without caps). SystemExit if the Mac's level is not one this rule knows."""
    cap = {"NO AVX": "default", "DEFAULT": "default", "AVX2": "avx2", "AVX512": "avx512"}.get(mac["torch_cpu_capability"])
    isa = mac.get("onednn_isa") or ""
    dnnl = "SSE41" if "SSE4.1" in isa else ("AVX2" if "AVX2" in isa and "512" not in isa else None)
    mkl = mac.get("mkl") or ""
    mkl_set = "SSE4_2" if "SSE4.2" in mkl else ("AVX2" if "AVX2" in mkl and "512" not in mkl else None)
    if cap is None or dnnl is None or mkl_set is None:
        raise SystemExit(f"{WHO}: unknown Mac instruction level (torch {mac['torch_cpu_capability']!r}, "
                         f"oneDNN {isa!r}, MKL {mkl[:80]!r})")
    cv_off = sorted((cv2_available(here["cv2_cpu_features"]) & cv2_listed(mac["cv2_cpu_features"]))
                    - cv2_available(mac["cv2_cpu_features"]))
    np_off = sorted((set(here["numpy_cpu_found"]) & set(here["numpy_dispatch"])) - set(mac["numpy_cpu_found"]))
    env = {"ATEN_CPU_CAPABILITY": cap, "DNNL_MAX_CPU_ISA": dnnl, "ONEDNN_MAX_CPU_ISA": dnnl,
           "MKL_ENABLE_INSTRUCTIONS": mkl_set}
    if cv_off:
        env["OPENCV_CPU_DISABLE"] = ",".join(cv_off)
    if np_off:
        env["NPY_DISABLE_CPU_FEATURES"] = " ".join(np_off)
    return env


# ── the start rule (pure) ───────────────────────────────────────────────────────────────────────────────────────────
def predicted_s(arm: str, n_frames: int, rates: dict):
    """Predicted build seconds from the measured seconds per planned frame of the arm's family; the first isa build
    uses the plain rate times ISA_FIRST_FACTOR; nothing measured yet -> None (the step starts)."""
    fam = RATE_FAMILY[arm]
    if fam in rates:
        return rates[fam] * n_frames
    if fam == "isa" and "plain" in rates:
        return rates["plain"] * ISA_FIRST_FACTOR * n_frames
    return None


def may_start(now: float, end_at, pred) -> bool:
    if end_at is None:
        return True
    if now >= end_at:
        return False
    return pred is None or now + pred <= end_at


def planned_frames(w_plan: dict) -> int:
    """sportvu.rebuild.build's --max-frames for a window."""
    from sportvu.rebuild import build_stride
    return (w_plan["f_end"] - w_plan["f_start"]) // build_stride(w_plan["fps"]) + 2


# ── pack (Mac, frozen CV python; no model and no DATA_ROOT read) ───────────────────────────────────────────────────
def frozen_plan(P: dict) -> dict:
    from tools import p8_frozen as F
    plan = {w["window"]: w for s in SECTIONS for w in F.label_windows(P, GAME, s)}
    missing = [w for w in ALL_WINDOWS if w not in plan]
    if missing:
        raise SystemExit(f"{WHO}: the frozen plan of {SECTIONS} has no {missing}")
    return json.loads(json.dumps({w: plan[w] for w in ALL_WINDOWS}))


def tracked_weights() -> dict:
    return {rel: sha256_file(REPO / rel) for rel in TRACKED_WEIGHTS}


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cuda_pack_clip_fingerprint() -> dict:
    """The Mac's CLIP parameter fingerprint as the CUDA check's pack recorded it (P8 data folder); every run
    recomputes it and must match."""
    p = p8_root().parent / "cuda_check" / "pack" / "pack_manifest.json"
    fp = json.loads(p.read_text())["clip_fingerprint"]
    return {**fp, "source": "the CUDA check pack manifest (P8 data folder cuda_check/pack)"}


def cmd_pack(a) -> dict:
    guard_all()
    dest = check_outside(Path(a.dest).expanduser())
    dest.mkdir(parents=True, exist_ok=True)
    from tools import p8_frozen as F
    P = F.install_paths()
    import config
    if sha256_file(REPO / "models" / "player_detector.pt") != R23_SHA:
        raise SystemExit(f"{WHO}: models/player_detector.pt is not r23")
    files = {f"clips/{w}.mp4": P["label_build"] / f"{w}.mp4" for w in ALL_WINDOWS}
    for s in SECTIONS:
        files[f"sync/{s}_local_timemap.json"] = P["sync"] / f"{s}_local_timemap.json"
    files[HEAD] = config.HEAD_V2_PATH
    files[R23_COURT] = F.court_weights("r23")
    members = member_records(files)
    tar = dest / f"{PACK}.tar"
    rename_prev(tar)
    tar_sha = write_tar(tar, files)
    for rel in CODE:                                    # the tool itself, overlaid on the runtime's clone
        shutil.copyfile(REPO / rel, dest / Path(rel).name)
    plan = frozen_plan(P)
    ws = P["label_build"].parent.parent / "work" / "label_build"
    man = {"what": "P8 Colab CPU parity pack", "design": "docs/research/P8_colab_cpu_parity.md", "game": GAME,
           "windows": plan, "planned_frames": {w: planned_frames(plan[w]) for w in plan},
           "tar": tar.name, "tar_sha256": tar_sha, "members": members,
           "weights": {**tracked_weights(), HEAD: members[HEAD]["sha256"], R23_COURT: members[R23_COURT]["sha256"]},
           "code": {rel: sha256_file(REPO / rel) for rel in CODE},
           "workspace_clip_sha256": {w: (sha256_file(ws / f"{w}.mp4") if (ws / f"{w}.mp4").exists() else None)
                                     for w in ALL_WINDOWS},
           "clip_fingerprint": cuda_pack_clip_fingerprint(), "mac_probe": run_probe(),
           "commit": _commit(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = dest / "pack_manifest.json"
    rename_prev(p)
    atomic_write(p, json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("tar", "tar_sha256", "planned_frames", "code")}, indent=1))
    return man


def check_pack(pack: Path) -> dict:
    man = json.loads((Path(pack) / "pack_manifest.json").read_text())
    if man.get("game") != GAME or sorted(man.get("windows", {})) != sorted(ALL_WINDOWS):
        raise SystemExit(f"{WHO}: pack_manifest.json is not this test's")
    if sha256_file(Path(pack) / man["tar"]) != man["tar_sha256"]:
        raise SystemExit(f"{WHO}: {man['tar']} does not match pack_manifest.json")
    for rel, want in man["code"].items():
        if sha256_file(REPO / rel) != want:
            raise SystemExit(f"{WHO}: {rel} here is not the registered code of the pack")
    return man


# ── run (the runtime, and the Mac for the like-for-like reference) ──────────────────────────────────────────────────
def steps_for(arms: list, windows: list | None) -> list:
    """The (arm, window) steps: ORDER filtered by the arms on the runtime; each window of --windows for one arm."""
    if windows:
        if len(arms) != 1:
            raise SystemExit(f"{WHO}: --windows takes exactly one arm")
        return [(arms[0], w) for w in windows]
    return [(a, w) for a, w in ORDER if a in arms]


def result_members(out_root: Path, res: str, arms: list) -> dict:
    files = {f"{res}/run_record.json": out_root / "run_record.json"}
    for arm in arms:
        for w in ALL_WINDOWS:
            for suf in KEEP:
                p = out_root / arm / f"{w}{suf}"
                if p.exists():
                    files[f"{res}/{arm}/{p.name}"] = p
    return files


def write_result(out_dir: Path, work_out: Path, res: str, record: dict) -> dict:
    atomic_write(work_out / "run_record.json", json.dumps(record, indent=1))
    files = result_members(work_out, res, sorted(record["builds"]))
    members = member_records(files)
    tar_sha = write_tar(out_dir / f"{res}.tar", files)
    complete = {"result": res, "steps_done": record["steps_done"], "tar_sha256": tar_sha, "members": members}
    atomic_write(out_dir / f"{res}.complete.json", json.dumps(complete, indent=1))
    return complete


def upload(token: str | None, folder_id: str | None, out_dir: Path, res: str, k: int) -> list:
    """Both result files to Drive under names numbered by the steps done (tools.p8_drive_xfer upload); the token
    reaches that subprocess's environment only. -> the lines it printed (no secret in them)."""
    if not token or not folder_id:
        return []
    lines = []
    for suffix in (".complete.json", ".tar"):
        name = f"{res}.{k:02d}{suffix}"
        r = subprocess.run([sys.executable, "-m", "tools.p8_drive_xfer", "upload", "--src", str(out_dir / f"{res}{suffix}"),
                            "--drive-dir-id", folder_id, "--name", name], cwd=REPO,
                           env={**os.environ, "DRIVE_TOKEN": token}, capture_output=True, text=True)
        lines.append(r.stdout.strip() if r.returncode == 0 else f"upload of {name} failed: {r.stderr[-300:]}")
    return lines


def side_problems(arm_dir: Path, w: str) -> list:
    """The checks every built window must pass: all three models on CPU, the reset fix on, one reset-log entry
    (each stripping 2 callbacks) per identity reset."""
    bad = []
    dev = json.loads((arm_dir / f"{w}_device.json").read_text()) if (arm_dir / f"{w}_device.json").exists() else {}
    if dev.get("requested") != {"detector": "cpu", "court": "cpu", "gate": "cpu"} or \
            set((dev.get("resolved") or {}).values()) != {"cpu"}:
        bad.append(f"devices {dev.get('requested')} {dev.get('resolved')}")
    if dev.get("reset_fix") is not True or dev.get("clip_fix"):
        bad.append("reset_fix not on, or clip_fix on")
    log = json.loads((arm_dir / f"{w}_resetfix.json").read_text()) if (arm_dir / f"{w}_resetfix.json").exists() else None
    idt = json.loads((arm_dir / f"{w}_identity.json").read_text()) if (arm_dir / f"{w}_identity.json").exists() else None
    if log is None or idt is None:
        bad.append("no reset log or no identity")
    else:
        resets = idt["identity"]["camera_cut_resets"]
        if len(log["resets"]) != resets or any(e.get("stripped") != 2 for e in log["resets"]):
            bad.append(f"reset log {len(log['resets'])} entries against {resets} resets")
    return bad


def cmd_run(a) -> dict:
    guard_all()
    t_start = time.time()
    token = os.environ.pop("DRIVE_TOKEN", None)             # never inherited by a build
    arms = [x for x in a.arms.split(",") if x]
    allowed = (MAC_ARM,) if a.platform == "mac" else COLAB_ARMS
    if not arms or any(x not in allowed for x in arms):
        raise SystemExit(f"{WHO}: on {a.platform} the arms are among {allowed}")
    wins = [w for w in a.windows.split(",") if w] if a.windows else None
    if wins and any(w not in ALL_WINDOWS for w in wins):
        raise SystemExit(f"{WHO}: windows must be among {ALL_WINDOWS}")
    if a.platform == "mac" and os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: on the Mac, run inside the heavy-job lock")
    if a.platform == "mac" and not wins:
        raise SystemExit(f"{WHO}: on the Mac, name the windows (--windows)")
    steps = steps_for(arms, wins)
    res = result_name(a.platform)
    pack = Path(a.pack_dir)
    man = check_pack(pack)
    work, out_dir = check_outside(Path(a.work)), check_outside(Path(a.out_dir))
    os.environ["V3_P8_ROOT"] = str(work)
    for flag in BUILD_FLAGS + ISA_KEYS:
        os.environ.pop(flag, None)
    from tools import p8_frozen as F
    P = F.install_paths(work)
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
    got = {**{rel: sha256_file(REPO / rel) for rel in TRACKED_WEIGHTS}, HEAD: sha256_file(config.HEAD_V2_PATH),
           R23_COURT: sha256_file(r23_path)}
    bad = sorted(n for n in man["weights"] if got.get(n) != man["weights"][n])
    if bad or Path(config.PLAYER_DETECTOR_WEIGHTS).resolve() != (REPO / "models" / "player_detector.pt").resolve():
        raise SystemExit(f"{WHO}: weights differ from the pack manifest: {bad}")
    r23_path = F.court_weights("r23")                   # and against reports/r23_verification.json
    plan = frozen_plan(P)
    if plan != man["windows"]:
        raise SystemExit(f"{WHO}: the frozen window plan here differs from the pack's")
    from tools.p8_cuda_check import clip_fingerprint
    fp = clip_fingerprint()
    if fp["sha256"] != man["clip_fingerprint"]["sha256"]:
        raise SystemExit(f"{WHO}: CLIP parameters differ from the Mac's (revision {fp['revision']})")
    probes = {"plain": run_probe()}
    envs = {"plain": {}, "plain_b": {}, "mac": {}}
    if "isa" in arms:
        envs["isa"] = isa_env(man["mac_probe"], probes["plain"])
        probes["isa"] = run_probe(envs["isa"])
    from sportvu.autolabel import gate_args
    from tools.p8_device import record_of
    work_out = work / "colab_cpu"
    work_out.mkdir(parents=True, exist_ok=True)
    record = {"what": "P8 Colab CPU parity run", "design": "docs/research/P8_colab_cpu_parity.md",
              "platform": a.platform, "commit": _commit(), "code": {rel: sha256_file(REPO / rel) for rel in CODE},
              "pack_tar_sha256": man["tar_sha256"], "pack_manifest_sha256": sha256_file(pack / "pack_manifest.json"),
              "weights": got, "clip_fingerprint": fp,
              "clip_revision_matches_manifest": fp["revision"] == man["clip_fingerprint"].get("revision"),
              "probes": probes, "isa_env": envs.get("isa"), "steps_planned": steps, "end_at": a.end_at,
              "steps_done": [], "steps_skipped": [], "builds": {}, "rates_s_per_frame": {},
              "setup_seconds": round(time.time() - t_start, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (out_dir / f"{res}.tar", out_dir / f"{res}.complete.json"):
        rename_prev(p)
    print(json.dumps({"setup": "ok", "seconds": record["setup_seconds"], "cpu": probes["plain"]["cpu_model"],
                      "torch_cpu_capability": {k: v["torch_cpu_capability"] for k, v in probes.items()},
                      "onednn_isa": {k: v["onednn_isa"] for k, v in probes.items()}, "isa_env": envs.get("isa")}),
          flush=True)
    rates = record["rates_s_per_frame"]
    for arm, w in steps:
        n = man["planned_frames"][w]
        pred = predicted_s(arm, n, rates)
        if not may_start(time.time(), a.end_at, pred):
            record["steps_skipped"].append([arm, w, "end_at", None if pred is None else round(pred)])
            print(json.dumps({"arm": arm, "window": w, "skipped": "end_at", "predicted_s": pred and round(pred)}),
                  flush=True)
            continue
        arm_dir = work_out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update({"P8_DEVICE": "cpu", "P8_TRACKER_RESET_FIX": "1", "COURT_GRID_WEIGHTS": str(r23_path),
                           **envs[arm]})
        t0 = time.time()
        try:
            for attempt in (1, 2):
                r = F.safe_build(P, plan[w], staging / "clips" / f"{w}.mp4", arm_dir, extra_args=gate_args())
                if r.get("ok"):
                    break
        finally:
            for k in ("P8_DEVICE", "P8_TRACKER_RESET_FIX", "COURT_GRID_WEIGHTS", *ISA_KEYS):
                os.environ.pop(k, None)
        secs = time.time() - t0
        ok = bool(r.get("ok")) and not r.get("empty")
        b = {"ok": ok, "attempts": attempt, "empty": bool(r.get("empty")), "cached": bool(r.get("cached")),
             "seconds": round(secs, 1), "planned_frames": n, "predicted_s": pred and round(pred),
             "device": record_of(arm_dir, w), "problems": side_problems(arm_dir, w) if ok else ["build failed"],
             "tail": (r.get("tail") or "")[-300:]}
        record["builds"].setdefault(arm, {})[w] = b
        if ok and not r.get("cached"):
            rates[RATE_FAMILY[arm]] = secs / n
        record["steps_done"].append([arm, w])
        record["total_seconds"] = round(time.time() - t_start, 1)
        complete = write_result(out_dir, work_out, res, record)
        up = upload(token, a.drive_dir_id, out_dir, res, len(record["steps_done"]))
        print(json.dumps({"arm": arm, "window": w, "ok": ok, "seconds": b["seconds"], "problems": b["problems"],
                          "tar_sha256": complete["tar_sha256"], "uploads": up}), flush=True)
        if not ok:
            print(json.dumps({"stopped": f"{arm} {w} failed twice"}), flush=True)
            break
    record["total_seconds"] = round(time.time() - t_start, 1)
    complete = write_result(out_dir, work_out, res, record)
    up = upload(token, a.drive_dir_id, out_dir, res, 99)
    print(json.dumps({"done": True, "total_seconds": record["total_seconds"], "steps_done": record["steps_done"],
                      "steps_skipped": record["steps_skipped"], "tar_sha256": complete["tar_sha256"], "uploads": up}),
          flush=True)
    return record


def cmd_unpack(a) -> dict:
    """A result tar and its complete.json (pulled from Drive, renamed to <result>.tar and <result>.complete.json)
    in --dest: checked against complete.json and, when given, the sha256 the runtime printed; then extracted."""
    guard_all()
    dest = check_outside(Path(a.dest).expanduser())
    tar, comp = dest / f"{a.name}.tar", dest / f"{a.name}.complete.json"
    complete = json.loads(comp.read_text())
    got = sha256_file(tar)
    if got != complete["tar_sha256"] or (a.expect_sha256 and got != a.expect_sha256):
        raise SystemExit(f"{WHO}: {tar.name} does not match its complete.json or the printed sha256")
    if (dest / a.name).exists():
        rename_prev(dest / a.name)
    extract_checked(tar, complete["members"], dest)
    info = {"steps_done": complete["steps_done"], "members": len(complete["members"]), "tar_sha256": got}
    print(json.dumps(info, indent=1))
    return info


# ── compare (pure parts) ────────────────────────────────────────────────────────────────────────────────────────────
def canonical(kind: str, text: str) -> str:
    """frames.json without "source" (the absolute clip path), as canonical JSON; the other files as they are."""
    if kind != "frames":
        return text
    d = json.loads(text)
    return json.dumps({k: v for k, v in d.items() if k != "source"}, sort_keys=True)


def match_boxes(a: list, b: list, gate: float = GATE_PX) -> tuple:
    """Two frames' boxes -> (pairs [(i, j, L-inf px)] within the gate from a one-to-one assignment that minimises
    the summed L-inf distance of the bboxes, unmatched in a, unmatched in b)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not a or not b:
        return [], len(a), len(b)
    A = np.array([x["bbox"] for x in a], float)
    B = np.array([x["bbox"] for x in b], float)
    cost = np.abs(A[:, None, :] - B[None, :, :]).max(axis=2)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci) if cost[i, j] <= gate]
    return pairs, len(a) - len(pairs), len(b) - len(pairs)


def box_metrics(x: dict, r: dict) -> dict:
    """{frame: row} of a build X and a reference R -> frame sets, the share of common frames whose boxes match one
    to one within BOX_TOL_PX, box offsets of the matched pairs, track-id agreement, unmatched boxes."""
    import numpy as np
    common, union = sorted(set(x) & set(r)), set(x) | set(r)
    within, offs, foot, same_tid, n_tol, unmatched = 0, [], [], 0, 0, 0
    mapping: dict = {}
    for f in common:
        bx, br = x[f]["boxes"], r[f]["boxes"]
        pairs, ux, ur = match_boxes(bx, br)
        unmatched += ux + ur
        if len(bx) == len(br) and len(pairs) == len(bx) and all(d <= BOX_TOL_PX for _, _, d in pairs):
            within += 1
        for i, j, d in pairs:
            offs.append(d)
            fa, fb = bx[i].get("foot_stab"), br[j].get("foot_stab")
            if fa and fb:
                foot.append(float(np.hypot(fa[0] - fb[0], fa[1] - fb[1])))
            if d <= BOX_TOL_PX:
                n_tol += 1
                same_tid += bx[i]["tid"] == br[j]["tid"]
                mapping.setdefault(bx[i]["tid"], {}).setdefault(br[j]["tid"], 0)
                mapping[bx[i]["tid"]][br[j]["tid"]] += 1
    mapped = sum(max(v.values()) for v in mapping.values())
    return {"frames_x": len(x), "frames_r": len(r), "common": len(common),
            "frames_jaccard": len(common) / len(union) if union else None,
            "frames_within_1px": within / len(common) if common else None,
            "boxes_x": sum(len(x[f]["boxes"]) for f in x), "boxes_r": sum(len(r[f]["boxes"]) for f in r),
            "pairs": len(offs), "unmatched_boxes": unmatched,
            "offset_px_median": float(np.median(offs)) if offs else None,
            "offset_px_max": float(np.max(offs)) if offs else None,
            "foot_stab_px_median": float(np.median(foot)) if foot else None,
            "foot_stab_px_max": float(np.max(foot)) if foot else None,
            "pairs_within_1px": n_tol, "tid_equal_share": same_tid / n_tol if n_tol else None,
            "tid_mapped_share": mapped / n_tol if n_tol else None, "h_gap_px": h_gap(x, r)}


def window_reading(eq: dict, m: dict, resets_x: int, resets_r: int) -> str:
    """One window: identical, near or different (section 4)."""
    if all(eq.get(k) for k in SIDE):
        return "identical"
    h = (m.get("h_gap_px") or {}).get("median")
    near = (m["frames_jaccard"] is not None and m["frames_jaccard"] >= NEAR["frames_jaccard_min"]
            and m["frames_within_1px"] is not None and m["frames_within_1px"] >= NEAR["frames_within_1px_min"]
            and resets_x == resets_r and (h is None or h <= NEAR["h_gap_median_px_max"]))
    return "near" if near else "different"


def arm_reading(per_window: dict, reset_windows: set) -> str:
    """An arm: "not read" unless it was compared on at least one reset and one no-reset window; else the weakest
    window reading (identical > near > different)."""
    done = set(per_window)
    if not (done & reset_windows) or not (done - reset_windows):
        return "not read"
    vals = set(per_window.values())
    return "different" if "different" in vals else ("near" if "near" in vals else "identical")


def load_side(d: Path, w: str) -> dict:
    return {k: (d / f"{w}_{k}.json").read_text() for k in SIDE if (d / f"{w}_{k}.json").exists()}


def compare_pair(x: dict, r: dict) -> dict:
    """Two builds of a window ({kind: text}) -> equality per file, box metrics and reset counts."""
    from tools.p8_cuda_check import frames_of
    eq = {k: (k in x and k in r and canonical(k, x[k]) == canonical(k, r[k])) for k in SIDE}
    fx, fr = frames_of(json.loads(x["frames"])), frames_of(json.loads(r["frames"]))
    rx = json.loads(x["identity"])["identity"]["camera_cut_resets"]
    rr = json.loads(r["identity"])["identity"]["camera_cut_resets"]
    m = box_metrics(fx, fr)
    return {"equal": eq, "metrics": m, "resets_x": rx, "resets_r": rr, "reading": window_reading(eq, m, rx, rr)}


def reference_build(ref_root: Path, w: str):
    """Report only: the reference-platform build of a window (detector on CPU, court and gate on MPS) with the same
    reset behaviour: the reset-fix arm's folder, else the unfixed baseline when it had no reset, else None."""
    fixed = ref_root / "builds" / f"resetfix_r23_detcpu__{GAME}"
    if (fixed / f"{w}_frames.json").exists():
        return fixed, "resetfix_r23_detcpu (reset fix on)"
    base = ref_root / "builds" / "r23_nofix"
    if (base / f"{w}_identity.json").exists() and \
            json.loads((base / f"{w}_identity.json").read_text())["identity"]["camera_cut_resets"] == 0:
        return base, "r23_nofix (no reset in the window, so the fix changes nothing)"
    return None, None


def cmd_compare(a) -> dict:
    guard_all()
    mac_dir, col_dir, pack = Path(a.mac).expanduser(), Path(a.colab).expanduser(), Path(a.pack).expanduser()
    man = json.loads((pack / "pack_manifest.json").read_text())
    mres, cres = result_name("mac"), result_name("colab")
    mrec = json.loads((mac_dir / mres / "run_record.json").read_text())
    crec = json.loads((col_dir / cres / "run_record.json").read_text())
    pm_sha = sha256_file(pack / "pack_manifest.json")
    flags = {"mac_pack_ok": mrec["pack_manifest_sha256"] == pm_sha, "colab_pack_ok": crec["pack_manifest_sha256"] == pm_sha,
             "code_equal": mrec["code"] == crec["code"] == man["code"], "weights_equal": mrec["weights"] == crec["weights"],
             "clip_equal": mrec["clip_fingerprint"]["sha256"] == crec["clip_fingerprint"]["sha256"]}
    macb = mac_dir / mres / MAC_ARM
    problems = {f"mac/{w}": b["problems"] for w, b in mrec["builds"].get(MAC_ARM, {}).items() if b["problems"]}
    for arm, v in crec["builds"].items():
        problems.update({f"{arm}/{w}": b["problems"] for w, b in v.items() if b["problems"]})
    mac_ok = {w for w, b in mrec["builds"].get(MAC_ARM, {}).items() if b["ok"] and not b["problems"]}
    reset_ws = {w for w in mac_ok if json.loads((macb / f"{w}_identity.json").read_text())["identity"]["camera_cut_resets"] >= 1}
    out = {"what": "P8 Colab CPU parity, comparison", "design": "docs/research/P8_colab_cpu_parity.md",
           "rule": {"box_tol_px": BOX_TOL_PX, "gate_px": GATE_PX, "near": NEAR}, "flags": flags, "problems": problems,
           "reset_windows_mac": sorted(reset_ws),
           "build_seconds": {"mac": {w: b["seconds"] for w, b in mrec["builds"].get(MAC_ARM, {}).items()},
                             **{arm: {w: b["seconds"] for w, b in v.items()} for arm, v in crec["builds"].items()}},
           "steps_skipped": crec["steps_skipped"], "isa_env": crec.get("isa_env"),
           "versions": {side: {k: v for k, v in rec["probes"]["plain"].items()
                               if k not in ("distributions", "torch_config", "fingerprint")}
                        for side, rec in (("mac", mrec), ("colab", crec))},
           "isa_probe": {k: v for k, v in (crec["probes"].get("isa") or {}).items()
                         if k in ("torch_cpu_capability", "onednn_isa", "mkl", "cv2_cpu_features", "numpy_cpu_found")},
           "fingerprints": {"mac": mrec["probes"]["plain"]["fingerprint"],
                            **{f"colab_{k}": v["fingerprint"] for k, v in crec["probes"].items()}},
           "distribution_differences": {n: [mrec["probes"]["plain"]["distributions"].get(n), v]
                                        for n, v in crec["probes"]["plain"]["distributions"].items()
                                        if n in mrec["probes"]["plain"]["distributions"]
                                        and mrec["probes"]["plain"]["distributions"][n] != v},
           "pairs": {}, "readings": {}, "repeatability": {}, "report_only": {}}
    for arm in COLAB_ARMS:
        if arm == "plain_b":
            continue
        per = {}
        for w, b in (crec["builds"].get(arm) or {}).items():
            if w in mac_ok and b["ok"] and not b["problems"]:
                cp = compare_pair(load_side(col_dir / cres / arm, w), load_side(macb, w))
                out["pairs"][f"{arm}~mac/{w}"] = cp
                per[w] = cp["reading"]
        out["readings"][arm] = {"windows": per, "reading": arm_reading(per, reset_ws)}
    for w, b in (crec["builds"].get("plain_b") or {}).items():
        if b["ok"] and (crec["builds"].get("plain") or {}).get(w, {}).get("ok"):
            cp = compare_pair(load_side(col_dir / cres / "plain_b", w), load_side(col_dir / cres / "plain", w))
            out["repeatability"][w] = {"equal": cp["equal"], "reading": cp["reading"],
                                       "frames_within_1px": cp["metrics"]["frames_within_1px"]}
    ref_root = p8_root().parent / "work_ref"
    for w in sorted(mac_ok):
        d, what = reference_build(ref_root, w)
        if d is None:
            continue
        ref = load_side(d, w)
        # r23_nofix was built in the P8 workspace from its own clip; the reset-fix arm read the reference root's clip,
        # which is the pack's source
        same_clip = (man["workspace_clip_sha256"].get(w) == man["members"][f"clips/{w}.mp4"]["sha256"]
                     if what.startswith("r23_nofix") else True)
        blk = {"reference": what, "note": "report only: the reference platform has the court and gate on MPS",
               "clip_equal_to_pack": same_clip}
        for k, side in (("mac", load_side(macb, w)), ("plain", load_side(col_dir / cres / "plain", w))):
            if side.get("frames"):
                cp = compare_pair(side, ref)
                blk[k] = {"equal": cp["equal"], "metrics": cp["metrics"], "resets": [cp["resets_x"], cp["resets_r"]]}
        out["report_only"][w] = blk
    plain, isa = out["readings"]["plain"]["reading"], out["readings"]["isa"]["reading"]
    out["answer"] = {"plain": plain, "isa": isa,
                     "repeatable": all(v["reading"] == "identical" for v in out["repeatability"].values())
                     if out["repeatability"] else None}
    d = p8_root().parent / "colab_cpu"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "colab_cpu_result.json", json.dumps(round_floats(out), indent=1))
    if a.report:
        atomic_write(REPO / "reports" / "research" / "p8_colab_cpu.json", json.dumps(round_floats(out), indent=1))
    print(json.dumps({"flags": flags, "problems": problems, "answer": out["answer"],
                      "readings": out["readings"], "repeatability": out["repeatability"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--dest", required=True)
    p = sub.add_parser("run")
    p.add_argument("--platform", choices=("mac", "colab"), required=True)
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--windows", default="")
    p.add_argument("--end-at", type=float, default=None, help="epoch seconds; no step starts unless it fits")
    p.add_argument("--drive-dir-id", default=None)
    p.add_argument("--install-models", action="store_true")
    p = sub.add_parser("unpack")
    p.add_argument("--dest", required=True)
    p.add_argument("--name", required=True, choices=(result_name("mac"), result_name("colab")))
    p.add_argument("--expect-sha256", default=None)
    p = sub.add_parser("compare")
    p.add_argument("--mac", required=True)
    p.add_argument("--colab", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--report", action="store_true", help="also write reports/research/p8_colab_cpu.json")
    sub.add_parser("probe")
    p = sub.add_parser("job", help="print the runtime job script")
    p.add_argument("--base", required=True)
    p.add_argument("--folder-id", required=True)
    a = ap.parse_args(argv)
    {"pack": cmd_pack, "run": cmd_run, "unpack": cmd_unpack, "compare": cmd_compare, "probe": cmd_probe,
     "job": lambda x: print(job_script(x.base, x.folder_id))}[a.cmd](a)


if __name__ == "__main__":
    main()
