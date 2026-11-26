"""tools/p8_court_truth.py: which build's court homography is closest to camera truth (docs/research/P8_court_truth.md,
design committed before any number). Report only; reads JSON only (no model, no video decode, no lock).

    <frozen CV python> -m tools.p8_court_truth run --cuda-result <folder holding p8_cuda_gsw_ind_2016/>

Per window, build and camera-truth frame: the point gap (median px between the images of three court points under the
build's H and under H_truth) and the frozen court line error (sportvu.bench.line_error_px, beside). Per build: median
and p90. Per pair: median paired difference with a frame bootstrap (deciding) and a 10-frame block bootstrap
(beside). Frames: those the GSW gold-key paint check keeps (primary) and all truth frames (sensitivity). The two
gsw_ind_2016 windows (Colab R1.3 truth, clips cut on Colab) are measured only after a frame-alignment check against the
Mac CPU build's boxes. Output: <P8 root>/court_truth/result.json, no local path.

Version 2 (ml-review of d86ac0e2 and 82190894): every reading means "closer to this truth", not "more accurate";
beside each window the median paint edge_px of its primary frames and the on-frame count of the three court points;
beside each pair the median build-to-build gap, the number of bootstrap blocks and a paired line-error difference;
an outcome that would lead to a D line needs the frame and the block interval to exclude zero, and the point gap and
the line error to agree in sign; the frame size is read from each window's clip header and must be 1280x720.
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


def on_frame_points(H, w: int = W_PX, h: int = H_PX):
    """How many of the three court points land inside the frame under H (image -> court), or None."""
    if H is None:
        return None
    try:
        u = np.linalg.inv(np.asarray(H, float).reshape(3, 3)) @ PTS
    except np.linalg.LinAlgError:
        return None
    xy = u[:2] / u[2]
    return int(((xy[0] >= 0) & (xy[0] <= w - 1) & (xy[1] >= 0) & (xy[1] <= h - 1)).sum())


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


def side_of(ci: list):
    """-1 when an interval lies below zero, +1 above, 0 when it holds zero."""
    return -1 if ci[1] < 0 else (1 if ci[0] > 0 else 0)


def pair_row(gaps_a: dict, gaps_b: dict, a: str, b: str, between: dict | None = None, lines_a: dict | None = None,
             lines_b: dict | None = None) -> dict:
    """Paired difference (A minus B) over frames where both gaps exist, in frame order. The frame interval decides the
    reading; beside it the block interval (and its block count), the median build-to-build gap on the same frames
    (between: frame -> gap of A's H to B's H), and the paired line-error difference over frames where both line errors
    exist (on-frame samples only), with the same bootstraps. d_line_ready: the reading is a 'closer' one, the block
    interval excludes zero on the same side, and the line-error difference's median has the same sign."""
    frames = sorted(f for f in gaps_a if gaps_a[f] is not None and gaps_b.get(f) is not None)
    if len(frames) < 2:
        return {"frames": len(frames), "reading": "not enough frames", "d_line_ready": False}
    d = np.array([gaps_a[f] - gaps_b[f] for f in frames])
    med, ci = float(np.median(d)), boot(d)
    bci = boot(d, block=BLOCK)
    out = {"frames": len(frames), "blocks": -(-len(frames) // BLOCK), "median_diff_px": med, "frame_ci": ci,
           "block_ci": bci, "reading": reading(med, ci, a, b)}
    if between is not None:
        bg = [between[f] for f in frames if between.get(f) is not None]
        out["median_build_to_build_gap_px"] = float(np.median(bg)) if bg else None
    if lines_a is not None and lines_b is not None:
        lf = [f for f in frames if lines_a.get(f) is not None and lines_b.get(f) is not None]
        if len(lf) >= 2:
            ld = np.array([lines_a[f] - lines_b[f] for f in lf])
            out["line_error"] = {"frames": len(lf), "median_diff_px": float(np.median(ld)), "frame_ci": boot(ld),
                                 "block_ci": boot(ld, block=BLOCK)}
        else:
            out["line_error"] = {"frames": len(lf)}
    side = -1 if out["reading"] == f"{a} closer" else (1 if out["reading"] == f"{b} closer" else 0)
    lmed = (out.get("line_error") or {}).get("median_diff_px")
    out["line_error_sign_agrees"] = None if lmed is None or side == 0 else bool(np.sign(lmed) == side)
    out["d_line_ready"] = bool(side != 0 and side_of(bci) == side and out["line_error_sign_agrees"])
    return out


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

    def between(a, b):
        return {f: point_gap((builds[a].get(f) or {}).get("H"), (builds[b].get(f) or {}).get("H")) for f in frames}
    on = [on_frame_points(truth[f]["H_truth"]) for f in frames]
    return {"frames": len(frames),
            "truth_on_frame_points": {str(k): sum(1 for v in on if v == k) for k in (0, 1, 2, 3)},
            "builds": {n: build_row(gaps[n], lines[n]) for n in names},
            "pairs": {f"{a}~{b}": pair_row(gaps[a], gaps[b], a, b, between(a, b), lines[a], lines[b])
                      for i, a in enumerate(names) for b in names[i + 1:]}}


def pair_of(win: dict, a: str, b: str):
    """The pair row of A and B on a window's primary frames, oriented as A minus B, or None."""
    p = ((win.get("primary") or {}).get("pairs") or {})
    if f"{a}~{b}" in p:
        return p[f"{a}~{b}"]
    if f"{b}~{a}" in p:
        return p[f"{b}~{a}"]
    return None


def pair_reading(win: dict, a: str, b: str):
    r = pair_of(win, a, b)
    return r["reading"] if r else None


def closer(win: dict, a: str, b: str, for_d_line: bool = False) -> bool:
    """A closer to this truth than B; for an outcome that would lead to a D line, the block interval and the line
    error must agree too."""
    r = pair_of(win, a, b)
    return bool(r) and r["reading"] == f"{a} closer" and (not for_d_line or r.get("d_line_ready", False))


def summary(windows: dict) -> dict:
    """The design's section 5 readings (version 2) from the primary frame sets. Every reading is "closer to this
    truth", not "more accurate"."""
    ind = [w for w in (W03, W04) if (windows.get(w) or {}).get("measured")]
    w00 = windows.get(W00) or {}
    out = {"meaning": "closer to this camera truth, not more accurate (truth was picked and initialised through one "
                      "build's homography, P8_court_truth.md section 4)",
           "cpu_vs_mps_readings": {w: pair_reading(windows[w], "mac_cpu", "mac_mps") for w in ind}}
    if len(ind) < 2:
        out["court_device"] = "not enough aligned gsw_ind_2016 windows"
    elif all(closer(windows[w], "mac_cpu", "mac_mps", True) for w in ind):
        out["court_device"] = ("CPU court closer to this truth on both: propose a confirmation measurement (a D line "
                               "for a test, not a device change)")
    elif all(closer(windows[w], "mac_mps", "mac_cpu", True) for w in ind):
        out["court_device"] = "MPS court closer to this truth on both: D175 stands"
    else:
        out["court_device"] = "no change: mixed, no difference resolved, or the block interval or line error disagrees"
    cuda_w00 = bool(w00.get("measured")) and closer(w00, "cuda_a", "mac_mps", True)
    cuda_ind = len(ind) == 2 and all(closer(windows[w], "cuda_a", "mac_mps", True)
                                     and closer(windows[w], "cuda_a", "mac_cpu", True) for w in ind)
    if cuda_w00 and cuda_ind:
        out["cuda"] = "CUDA court closer to this truth on s04_w00 and both gsw_ind_2016 windows"
    elif cuda_ind:
        out["cuda"] = "CUDA closer on the gsw_ind_2016 windows only: consistent with a lean, no change"
    else:
        out["cuda"] = "CUDA not closer on every window: no change"
    mps_w00 = bool(w00.get("measured")) and closer(w00, "mac_mps", "cuda_a")
    cuda_ind_any = len(ind) == 2 and all(closer(windows[w], "cuda_a", "mac_mps") for w in ind)
    out["bias_signal"] = ("consistent with a lean toward the selecting build's court" if mps_w00 and cuda_ind_any
                          else "does not show a lean, does not rule it out")
    out["cuda_readings"] = {w: {r: pair_reading(windows[w], "cuda_a", r) for r in ("mac_mps", "mac_cpu")}
                            for w in [W00, *ind] if (windows.get(w) or {}).get("measured")}
    return out


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
        names = {"cpu": "mac_cpu", "mps": "mac_mps", "detector_cpu": "mac_detcpu"}
        b = {names[k]: (p8 / "label_build" if d is None else diag / d) for k, d in DEVICE_BUILDS[w].items()}
        out[w] = {"game": IND, "truth": main / "data" / "sportvu" / "labels" / IND / f"{w}.jsonl",
                  "paint": main / "data" / "sportvu" / "labels" / f"{IND}_paint.jsonl",
                  "builds": {**b, "cuda_a": cuda / CUDA_RESULT / "cuda_a"}, "align": True}
    return out


def clip_size(clip: Path) -> tuple:
    """(width, height) from the clip's header (no frame decoded)."""
    import cv2
    cap = cv2.VideoCapture(str(clip))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def window_result(w: str, spec: dict, line_error, size=None) -> dict:
    if size is not None and tuple(size) != (W_PX, H_PX):
        raise SystemExit(f"{WHO}: {w} is {size[0]}x{size[1]}, not {W_PX}x{H_PX}")
    truth = {r["frame"]: r for r in read_jsonl(spec["truth"]) if r.get("window", w) == w}
    paint = {r["frame"]: r for r in read_jsonl(spec["paint"]) if r["window"] == w}
    keep = {f for f, r in paint.items() if r.get("keep")}
    builds = {n: frames_of(d / f"{w}_frames.json") for n, d in spec["builds"].items()}
    edges = [paint[f]["edge_px"] for f in keep & set(truth) if paint[f].get("edge_px") is not None]
    out = {"truth_frames": len(truth), "paint_kept": len(keep & set(truth)), "frame_size": [W_PX, H_PX],
           "primary_paint_edge_px_median": float(np.median(edges)) if edges else None}
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
    P8 = p8_root()
    windows = {w: window_result(w, s, line_error, clip_size(P8 / "label_build" / f"{w}.mp4"))
               for w, s in specs(P8, DIAG, main_checkout(), cuda).items()}
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
