"""tools/p8_platform_summary.py: the numbers of docs/research/P8_PLATFORM.md, from the diagnosis files (PLAN.md D98, D99).

    python3 -m tools.p8_platform_summary        -> reports/research/p8_platform.json

Reads only, and runs no model:
- the Colab R1.3 run log on the Drive mount (DATA_ROOT/results/r13_20261004_1846/logs/autolabel_gsw_ind_2016.log):
  camera-truth candidates per window, and its input manifest (DATA_ROOT/sportvu/MANIFEST_r13.json)
- the Colab R1.3 labels in the main checkout (data/sportvu/labels/gsw_ind_2016/*.jsonl): each window's clock offset
- the P8 workspace (night 1, MPS): labels/gsw_ind_2016_offsets.json, label_build/<window>_frames.json, video sections
- the diagnosis folder <home>/Developer/nba-v3-worktrees/_p8-diag: CPU and mixed builds of two windows, candidate
  counts (cands_*.json) and offset sweeps (sweep_*.json)
The output holds numbers and the files they come from, named relative to those roots, no prose and no local path.
Development game only (gsw_ind_2016, an R1.3 training game).
"""

import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_platform_summary"
GAME = "gsw_ind_2016"
WINDOWS = ("gsw_ind_2016_s02_w04", "gsw_ind_2016_s02_w03")
OUT = REPO / "reports" / "research" / "p8_platform.json"


def diag_root() -> Path:
    return Path(os.environ.get("V3_P8_DIAG") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-diag")


def _frames(p: Path) -> dict:
    return {r["frame"]: r for r in json.loads(p.read_text())["frames"]} if p.exists() else {}


def boxes_mean(fr: dict):
    return round(float(np.mean([len(r["boxes"]) for r in fr.values()])), 2) if fr else None


def h_gap_px(a: dict, b: dict):
    """Median and p90 over common frames of the median px gap between two court Hs at three court points."""
    pts = np.array([[47, 25, 1], [20, 10, 1], [70, 40, 1]], float).T
    gaps = []
    for f in set(a) & set(b):
        if a[f].get("H") and b[f].get("H"):
            ua = np.linalg.inv(np.array(a[f]["H"]).reshape(3, 3)) @ pts
            ub = np.linalg.inv(np.array(b[f]["H"]).reshape(3, 3)) @ pts
            gaps.append(float(np.median(np.linalg.norm(ua[:2] / ua[2] - ub[:2] / ub[2], axis=0))))
    return {"median": round(float(np.median(gaps)), 2), "p90": round(float(np.percentile(gaps, 90)), 2),
            "frames": len(gaps)} if gaps else None


def identical_boxes(a: dict, b: dict):
    c = set(a) & set(b)
    return {"frames": len(c), "identical": sum([x["bbox"] for x in a[f]["boxes"]] == [x["bbox"] for x in b[f]["boxes"]]
                                               for f in c)} if c else None


def same_box_count(a: dict, b: dict):
    c = set(a) & set(b)
    return {"frames": len(c), "same": sum(len(a[f]["boxes"]) == len(b[f]["boxes"]) for f in c)} if c else None


def colab_candidates() -> dict:
    log = Path(os.environ["DATA_ROOT"]) / "results" / "r13_20261004_1846" / "logs" / f"autolabel_{GAME}.log"
    out = {}
    for line in log.read_text().splitlines():
        m = re.search(r"A \[\d+/\d+\] (\S+)\s+\S+ candidates (\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def colab_offsets() -> dict:
    out = {}
    for p in sorted((main_checkout() / "data" / "sportvu" / "labels" / GAME).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r["window"]] = r["offset_s"]
            break
    return out


def inputs_identical() -> dict:
    man = json.loads((Path(os.environ["DATA_ROOT"]) / "sportvu" / "MANIFEST_r13.json").read_text())["files"]
    same = differ = 0
    for k, v in man.items():
        if not k.startswith(f"video/{GAME}_s"):
            continue
        h = hashlib.sha256()
        with open(p8_root() / "video" / Path(k).name, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        same += h.hexdigest() == v["sha256"]
        differ += h.hexdigest() != v["sha256"]
    return {"sections_identical": same, "sections_differ": differ}


def summarise() -> dict:
    refuse_cv_test([GAME, *WINDOWS], WHO)
    guard([GAME, *WINDOWS], WHO)
    D, W = diag_root(), p8_root()
    col_c, col_o = colab_candidates(), colab_offsets()
    mac_o = json.loads((W / "labels" / f"{GAME}_offsets.json").read_text())
    both = sorted(w for w in set(col_o) & set(mac_o) if mac_o[w] is not None)
    gaps = [abs(col_o[w] - mac_o[w]) for w in both]
    col_r = json.loads((main_checkout() / "data" / "sportvu" / "labels" / f"{GAME}_result.json").read_text())
    mac_r = json.loads((W / "labels" / f"{GAME}_result.json").read_text())
    night = json.loads((D / "cands_mac_night.json").read_text()) if (D / "cands_mac_night.json").exists() else {}
    per_window = {}
    for w in WINDOWS:
        short = w.split(f"{GAME}_")[1]
        mps = _frames(W / "label_build" / f"{w}_frames.json")
        cpu = _frames(D / f"{short}_cpu" / f"{w}_frames.json")
        det = _frames(D / f"{short}_detcpu" / f"{w}_frames.json")
        rerun = _frames(D / f"{short}_mps" / f"{w}_frames.json")
        cc = D / f"cands_cpu_{short}.json"
        row = {"colab_candidates": col_c.get(w), "colab_offset_s": col_o.get(w), "mac_mps_offset_s": mac_o.get(w),
               "mps_candidates": night.get(w, {}).get("candidates"),
               "mps_unmapped_frames": night.get(w, {}).get("status", {}).get("unmapped"),
               "cpu_candidates": json.loads(cc.read_text())[w]["candidates"] if cc.exists() else None,
               "frames": len(mps), "boxes_mean": {"mps": boxes_mean(mps), "cpu": boxes_mean(cpu), "detector_cpu_court_mps": boxes_mean(det)},
               "mps_rerun_identical": (same_box_count(mps, rerun) or {}).get("same") == len(mps) and h_gap_px(mps, rerun)["median"] == 0.0 if rerun else None,
               "same_box_count_cpu_vs_mps": same_box_count(mps, cpu),
               "identical_boxes_detector_cpu_vs_cpu": identical_boxes(det, cpu),
               "court_h_gap_px": {"cpu_vs_mps": h_gap_px(cpu, mps), "detector_cpu_vs_mps": h_gap_px(det, mps),
                                  "detector_cpu_vs_cpu": h_gap_px(det, cpu)},
               "offset_sweep_candidates": {tag: json.loads((D / f"sweep_{short}_{tag}.json").read_text())
                                           for tag in ("mps", "cpu", "detcpu") if (D / f"sweep_{short}_{tag}.json").exists()}}
        per_window[w] = row
    return {"game": GAME, "decision": "PLAN.md D98, D99",
            "colab": {"run": "r13_20261004_1846", "commit": "65aa65d", "gpu": "NVIDIA A100-SXM4-40GB", "device": "cuda",
                      "precision": "fp32", "imgsz": 640, "conf": 0.40, "nms_iou": 0.45, "ultralytics": None},
            "mac": {"device": "mps", "precision": "fp32", "imgsz": 640, "conf": 0.40, "nms_iou": 0.45,
                    "ultralytics": "8.4.48", "torch": "2.2.2"},
            "inputs": inputs_identical(),
            "offsets_vs_colab": {"windows": len(both), "over_0_2_s": sum(g > 0.2 for g in gaps),
                                 "median_abs_s": round(statistics.median(gaps), 2) if gaps else None,
                                 "max_abs_s": round(max(gaps), 2) if gaps else None},
            "game_totals": {k: {"colab": col_r[a][b] if b else col_r[a], "mac_mps": mac_r[a][b] if b else mac_r[a]}
                            for k, a, b in (("accepted", "accepted", None), ("camera_pool_inits", "camera", "inits"),
                                            ("camera_reproj_px", "camera", "reproj_px_median_after"))},
            "windows": per_window,
            "build_seconds": json.loads((D / "timings.json").read_text())["build_seconds"] if (D / "timings.json").exists() else None,
            "sources": {"colab_candidates": "DATA_ROOT/results/r13_20261004_1846/logs/autolabel_gsw_ind_2016.log",
                        "colab_offsets": "main checkout data/sportvu/labels/gsw_ind_2016/*.jsonl (offset_s)",
                        "colab_totals": "main checkout data/sportvu/labels/gsw_ind_2016_result.json",
                        "inputs": "DATA_ROOT/sportvu/MANIFEST_r13.json vs P8 workspace video/",
                        "mac": "P8 workspace labels/gsw_ind_2016_offsets.json, labels/gsw_ind_2016_result.json, label_build/",
                        "diagnosis": "_p8-diag: s02_w0*_cpu, s02_w04_mps, s02_w03_detcpu, cands_*.json, sweep_*.json, timings.json"}}


def main(argv=None) -> int:
    rec = summarise()
    atomic_write(OUT, json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
