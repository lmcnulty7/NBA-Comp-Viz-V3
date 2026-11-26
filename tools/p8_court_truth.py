"""tools/p8_court_truth.py: which build's court homography is closest to camera truth (docs/research/P8_court_truth.md,
design committed before any number). Report only; reads JSON only (no model, no video decode, no lock).

    <frozen CV python> -m tools.p8_court_truth run --cuda-result <folder holding p8_cuda_gsw_ind_2016/>

Per window, build and camera-truth frame: the point gap (median px between the images of three court points under the
build's H and under H_truth) and the frozen court line error (sportvu.bench.line_error_px, beside). Per build: median
and p90. Per pair: median paired difference with a frame bootstrap (deciding) and a 10-frame block bootstrap
(beside). Frames: those the GSW gold-key paint check keeps (primary) and all truth frames (sensitivity). The two
gsw_ind_2016 windows (Colab R1.3 truth, clips cut on Colab) are measured only after a frame-alignment check against the
Mac CPU build's boxes. Output: <P8 root>/court_truth/result.json, no local path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, check_outside, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_court_truth"
PTS = np.array([[47, 25, 1], [20, 10, 1], [70, 40, 1]], float).T      # court ft, as tools.p8_platform_summary.h_gap_px
W_PX, H_PX = 1280, 720
BOOT_N, SEED, BLOCK = 2000, 0, 10
CLOSER_PX = 1.0
ALIGN_IOU, ALIGN_SHARE = 0.9, 0.9
CUDA_RESULT = "p8_cuda_gsw_ind_2016"
BKN, IND = "gsw_bkn_2015", "gsw_ind_2016"
W00, W03, W04 = "gsw_bkn_2015_s04_w00", "gsw_ind_2016_s02_w03", "gsw_ind_2016_s02_w04"


# ── pure ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def point_gap(H_build, H_truth):
    """Median px distance between the images of the three court points under the two homographies (image -> court),
    or None when either is missing or singular."""
    if H_build is None or H_truth is None:
        return None
    try:
        ub = np.linalg.inv(np.asarray(H_build, float).reshape(3, 3)) @ PTS
        ut = np.linalg.inv(np.asarray(H_truth, float).reshape(3, 3)) @ PTS
    except np.linalg.LinAlgError:
        return None
    return float(np.median(np.linalg.norm(ub[:2] / ub[2] - ut[:2] / ut[2], axis=0)))


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def align_share(truth: dict, build: dict, offset: int) -> float | None:
    """Share of truth-row boxes with a box of IoU >= ALIGN_IOU in the build at frame + offset."""
    hit = n = 0
    for f, row in truth.items():
        boxes = [b["bbox"] for b in (build.get(f + offset) or {}).get("boxes", [])]
        for tb in row.get("boxes", []):
            n += 1
            hit += any(iou(tb["bbox"], bb) >= ALIGN_IOU for bb in boxes)
    return hit / n if n else None


def alignment(truth: dict, build: dict, step: int) -> dict:
    """The design's check: share at offset 0 at least ALIGN_SHARE and above both neighbours (one build step away)."""
    s = {str(k): align_share(truth, build, k) for k in (-step, 0, step)}
    z, lo, hi = s["0"], s[str(-step)], s[str(step)]
    ok = z is not None and z >= ALIGN_SHARE and all(z > (v if v is not None else -1) for v in (lo, hi))
    return {"step": step, "shares": s, "aligned": bool(ok)}


def boot(diffs: np.ndarray, n: int = BOOT_N, seed: int = SEED, block: int | None = None) -> list:
    """2.5 and 97.5 percentiles of the median of resampled diffs: frames, or runs of `block` consecutive frames."""
    rng = np.random.default_rng(seed)
    if block:
        blocks = [diffs[i:i + block] for i in range(0, len(diffs), block)]
        vals = [float(np.median(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])))
                for _ in range(n)]
    else:
        vals = [float(np.median(diffs[rng.integers(0, len(diffs), len(diffs))])) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reading(median: float, ci: list, a: str, b: str) -> str:
    if median <= -CLOSER_PX and ci[1] < 0:
        return f"{a} closer"
    if median >= CLOSER_PX and ci[0] > 0:
        return f"{b} closer"
    return "no difference resolved"


def pair_row(gaps_a: dict, gaps_b: dict, a: str, b: str) -> dict:
    """Paired difference (A minus B) over frames where both gaps exist, in frame order."""
    frames = sorted(f for f in gaps_a if gaps_a[f] is not None and gaps_b.get(f) is not None)
    if len(frames) < 2:
        return {"frames": len(frames), "reading": "not enough frames"}
    d = np.array([gaps_a[f] - gaps_b[f] for f in frames])
    med, ci = float(np.median(d)), boot(d)
    return {"frames": len(frames), "median_diff_px": med, "frame_ci": ci, "block_ci": boot(d, block=BLOCK),
            "reading": reading(med, ci, a, b)}


def build_row(gaps: dict, lines: dict) -> dict:
    g = np.array([v for v in gaps.values() if v is not None])
    ln = [v for v in lines.values() if v is not None]
    return {"frames": int(len(g)), "missing_h": int(sum(v is None for v in gaps.values())),
            "gap_median_px": float(np.median(g)) if len(g) else None,
            "gap_p90_px": float(np.percentile(g, 90)) if len(g) else None,
            "line_error_median_px": float(np.median(ln)) if ln else None}


def measure(truth: dict, builds: dict, frames: list, line_error) -> dict:
    """truth {frame: row with H_truth}; builds {name: {frame: row with H}}; frames: the frame set."""
    gaps = {n: {f: point_gap((b.get(f) or {}).get("H"), truth[f]["H_truth"]) for f in frames} for n, b in builds.items()}
    lines = {n: {f: (line_error((b.get(f) or {}).get("H"), truth[f]["H_truth"], W_PX, H_PX)
                     if (b.get(f) or {}).get("H") else None) for f in frames} for n, b in builds.items()}
    names = sorted(builds)
    return {"frames": len(frames), "builds": {n: build_row(gaps[n], lines[n]) for n in names},
            "pairs": {f"{a}~{b}": pair_row(gaps[a], gaps[b], a, b) for i, a in enumerate(names) for b in names[i + 1:]}}


def pair_reading(win: dict, a: str, b: str):
    p = ((win.get("primary") or {}).get("pairs") or {})
    if f"{a}~{b}" in p:
        return p[f"{a}~{b}"]["reading"]
    if f"{b}~{a}" in p:
        return p[f"{b}~{a}"]["reading"]
    return None


def summary(windows: dict) -> dict:
    """The design's section 5 readings from the primary frame sets."""
    ind = [w for w in (W03, W04) if (windows.get(w) or {}).get("measured")]
    cpu_mps = [pair_reading(windows[w], "mac_cpu", "mac_mps") for w in ind]
    if len(ind) < 2:
        device = "not enough aligned gsw_ind_2016 windows"
    elif all(r == "mac_cpu closer" for r in cpu_mps):
        device = "CPU court closer on both: propose the court on CPU for reference builds (a D line)"
    elif all(r == "mac_mps closer" for r in cpu_mps):
        device = "MPS court closer on both: D175 stands"
    else:
        device = "no change: mixed or no difference resolved"
    w00 = windows.get(W00) or {}
    mps_on_bkn = w00.get("measured") and pair_reading(w00, "cuda_a", "mac_mps") == "mac_mps closer"
    cuda_on_ind = bool(ind) and all(pair_reading(windows[w], "cuda_a", "mac_mps") == "cuda_a closer" for w in ind)
    return {"court_device": device, "cpu_vs_mps_readings": dict(zip(ind, cpu_mps)),
            "cuda_readings": {w: {r: pair_reading(windows[w], "cuda_a", r) for r in ("mac_mps", "mac_cpu")}
                              for w in [W00, *ind] if (windows.get(w) or {}).get("measured")},
            "bias_signal": bool(mps_on_bkn and cuda_on_ind)}


# ── files ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def frames_of(p: Path) -> dict:
    return {r["frame"]: r for r in json.loads(Path(p).read_text())["frames"]}


def specs(p8: Path, diag: Path, main: Path, cuda: Path) -> dict:
    """Window -> its truth, paint file and builds (the gsw_ind_2016 Mac builds as tools.p8_truth_v2.DEVICE_BUILDS)."""
    from tools.p8_truth_v2 import DEVICE_BUILDS
    out = {W00: {"game": BKN, "truth": p8 / "labels" / BKN / f"{W00}.jsonl", "paint": p8 / "labels" / f"{BKN}_paint.jsonl",
                 "builds": {"mac_mps": p8 / "builds" / "r23", "mac_detcpu": p8 / "builds" / "e1m6_r23_detcpu",
                            "cuda_a": cuda / CUDA_RESULT / "cuda_a"}, "align": False}}
    for w in (W03, W04):
        b = {("mac_" + k if k in ("cpu", "mps") else k): (p8 / "label_build" if d is None else diag / d)
             for k, d in DEVICE_BUILDS[w].items()}
        out[w] = {"game": IND, "truth": main / "data" / "sportvu" / "labels" / IND / f"{w}.jsonl",
                  "paint": main / "data" / "sportvu" / "labels" / f"{IND}_paint.jsonl",
                  "builds": {**b, "cuda_a": cuda / CUDA_RESULT / "cuda_a"}, "align": True}
    return out


def window_result(w: str, spec: dict, line_error) -> dict:
    truth = {r["frame"]: r for r in read_jsonl(spec["truth"]) if r.get("window", w) == w}
    keep = {r["frame"] for r in read_jsonl(spec["paint"]) if r["window"] == w and r.get("keep")}
    builds = {n: frames_of(d / f"{w}_frames.json") for n, d in spec["builds"].items()}
    out = {"truth_frames": len(truth), "paint_kept": len(keep & set(truth))}
    if spec["align"]:
        ref = builds["mac_cpu"]
        fs = sorted(ref)
        step = int(np.median(np.diff(fs))) if len(fs) > 1 else 1
        out["alignment"] = alignment(truth, ref, step)
        if not out["alignment"]["aligned"]:
            out["measured"] = False
            return out
    out["measured"] = True
    out["primary"] = measure(truth, builds, sorted(keep & set(truth)), line_error)
    out["all_truth_frames"] = measure(truth, builds, sorted(truth), line_error)
    return out


def run(cuda_result: Path, line_error=None) -> dict:
    from tools.p8_truth_v2 import DIAG
    items = [BKN, IND, W00, W03, W04]
    refuse_cv_test(items, WHO)
    guard(items, WHO)
    if line_error is None:
        from sportvu.bench import line_error_px as line_error
    cuda = check_outside(Path(cuda_result).expanduser())
    rec = json.loads((cuda / CUDA_RESULT / "run_record.json").read_text())
    if rec.get("purpose") != "result":
        raise SystemExit(f"{WHO}: the CUDA folder holds a {rec.get('purpose')!r} run, not the result")
    windows = {w: window_result(w, s, line_error) for w, s in specs(p8_root(), DIAG, main_checkout(), cuda).items()}
    out = {"what": "court homography against camera truth", "design": "docs/research/P8_court_truth.md",
           "report_only": True, "bars": {"closer_px": CLOSER_PX, "align_iou": ALIGN_IOU, "align_share": ALIGN_SHARE,
                                         "boot_n": BOOT_N, "seed": SEED, "block": BLOCK},
           "cuda_commit": rec.get("commit"), "windows": windows, "summary": summary(windows)}
    d = p8_root() / "court_truth"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "result.json", json.dumps(out, indent=1))
    print(json.dumps({"written": "P8 workspace court_truth/result.json", "summary": out["summary"]}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cuda-result", required=True)
    a = ap.parse_args(argv)
    run(Path(a.cuda_result))


if __name__ == "__main__":
    sys.exit(main())
