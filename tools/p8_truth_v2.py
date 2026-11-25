"""tools/p8_truth_v2.py: truth v2's per-window clock offset rule, measured for stability against build differences
(PLAN.md D161 (2); pre-registration docs/research/P8_truth_v2.md). Frozen code is imported, never changed.

    <frozen CV python> -m tools.p8_truth_v2 measure TAG       court-model pairs (V3 label build, r23 rebuild) and the
                                                             accuracy guard on one development game
    <frozen CV python> -m tools.p8_truth_v2 device           the gsw_ind_2016 device pairs (D99 diagnosis builds)
    python3 -m tools.p8_truth_v2 report                      the registered measures and the bar
    <frozen CV python> -m tools.p8_truth_v2 measure-dev TAG   report only: the same measures on a development game
                                                             outside the registration (docs/research/P8_truth_arenas.md,
                                                             check C), written to truth_v2/dev/<tag>.json; never read
                                                             by report

Run measure and device under the heavy-job lock, one game at a time. Outputs: truth_v2/ in the P8 workspace. The
frozen truth, its offsets (labels/<game>_offsets.json) and every result scored on them stay as recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_truth_v2"
POOLED = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = ("gsw_ind_2016",)
DELTA_FT = 0.10
TRIM = 0.20
MIN_FRAMES = 20
BOOT_N, BOOT_SEED = 2000, 0
OVER_S = 0.2
DIAG = Path(os.environ.get("V3_P8_DIAG") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-diag")
# the D99 diagnosis builds of gsw_ind_2016: window -> {build name: folder (None = the workspace's label_build)}
DEVICE_BUILDS = {"gsw_ind_2016_s02_w03": {"mps": None, "cpu": "s02_w03_cpu", "detector_cpu": "s02_w03_detcpu"},
                 "gsw_ind_2016_s02_w04": {"mps": "s02_w04_mps", "cpu": "s02_w04_cpu"}}


# ── the rule (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────
def grid() -> np.ndarray:
    from sportvu.truth import OFFSET_MAX, OFFSET_STEP
    return np.round(np.arange(-OFFSET_MAX, OFFSET_MAX + 1e-9, OFFSET_STEP), 2)


def trimmed_match_mean(feet: np.ndarray, sv: np.ndarray, trim: float = TRIM) -> float:
    """Mean distance of the one-to-one Hungarian match of feet to SportVU players, the worst `trim` of pairs dropped
    (rounded down to whole pairs)."""
    from scipy.optimize import linear_sum_assignment
    D = np.sqrt(((feet[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(D)
    d = np.sort(D[ri, ci])
    keep = len(d) - int(np.floor(trim * len(d)))
    return float(d[:max(keep, 1)].mean())


def v2_objective(rows_pc: list, index, mirror: str, offsets: np.ndarray) -> np.ndarray:
    """J(offset): mean over frames with a SportVU moment of the frame's trimmed matched distance (NaN when none)."""
    from sportvu.sync import MIRRORS, apply_mirror
    from sportvu.truth import CLOCK_TOL_S
    fx, fy = MIRRORS[mirror]
    feet = [(q, c, apply_mirror(p, fx, fy)) for q, c, p in rows_pc]
    J = np.full(len(offsets), np.nan)
    for k, off in enumerate(offsets):
        s = []
        for q, c, p in feet:
            i = index.at(q, c + off, CLOCK_TOL_S)
            if i is not None:
                s.append(trimmed_match_mean(p, index.q[q]["xy"][i]))
        if s:
            J[k] = float(np.mean(s))
    return J


def v2_estimate(offsets: np.ndarray, J: np.ndarray, delta: float = DELTA_FT) -> dict:
    """Weighted centre of the near-minimum set {J <= J_min + delta}, weights J_min + delta - J; at_edge when the set
    holds a grid end."""
    ok = np.isfinite(J)
    if not ok.any():
        return {"offset_s": None, "j_min_ft": None, "at_edge": None, "near_min": 0}
    jmin = float(np.nanmin(J))
    in_s = ok & (J <= jmin + delta)
    w = np.where(in_s, jmin + delta - J, 0.0)
    w = np.where(in_s & (w <= 0), 1e-9, w)
    off = float((offsets * w).sum() / w.sum())
    return {"offset_s": round(off, 3), "j_min_ft": round(jmin, 3), "near_min": int(in_s.sum()),
            "at_edge": bool(in_s[0] or in_s[-1])}


def solve_v2(rows_pc: list, index, mirror: str) -> dict:
    if len(rows_pc) < MIN_FRAMES:
        return {"offset_s": None, "j_min_ft": None, "at_edge": None, "near_min": 0, "frames": len(rows_pc)}
    offs = grid()
    return {**v2_estimate(offs, v2_objective(rows_pc, index, mirror, offs)), "frames": len(rows_pc)}


def solve_frozen(rows_pc: list, index, mirror: str) -> float | None:
    from sportvu.truth import solve_window_offset
    return solve_window_offset(rows_pc, index, mirror)[0] if len(rows_pc) >= MIN_FRAMES else None


def stability(pairs: list) -> dict:
    """pairs: (offset A, offset B) -> median absolute difference, share over OVER_S, max, n (pairs with both)."""
    d = np.array([abs(a - b) for a, b in pairs if a is not None and b is not None])
    if not len(d):
        return {"n": 0, "median_abs_s": None, "share_over_0_2_s": None, "max_abs_s": None}
    return {"n": int(len(d)), "median_abs_s": round(float(np.median(d)), 3),
            "share_over_0_2_s": round(float((d > OVER_S + 1e-9).mean()), 4), "max_abs_s": round(float(d.max()), 3)}


def share_reduction_interval(frozen: list, v2: list, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Paired window bootstrap of (frozen share over 0.2 s) minus (v2 share over 0.2 s); windows where both rules have
    both offsets."""
    keep = [(abs(fa - fb) > OVER_S + 1e-9, abs(va - vb) > OVER_S + 1e-9) for (fa, fb), (va, vb) in zip(frozen, v2)
            if None not in (fa, fb, va, vb)]
    if len(keep) < 2:
        return None
    a = np.array(keep, float)
    rng = np.random.default_rng(seed)
    vals = [float((s[:, 0] - s[:, 1]).mean()) for s in (a[rng.integers(0, len(a), len(a))] for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# ── frozen-code helpers (frozen CV python) ──────────────────────────────────────────────────────────────────────────
def rows_pc_of(side: dict, tm: dict) -> list:
    """The frozen rule's inputs (sportvu.truth.window_truth): frames with a clock, an H and at least 4 boxes."""
    import cv2
    from sportvu.truth import frame_clock
    out = []
    for r in side["frames"]:
        q, c = frame_clock(None, tm, r["frame"])
        if q is None or r["H"] is None or len(r["boxes"]) < 4:
            continue
        feet = np.array([b["foot_stab"] for b in r["boxes"]], np.float32)
        out.append((q, c, cv2.perspectiveTransform(feet.reshape(-1, 1, 2), np.array(r["H"]).reshape(3, 3)).reshape(-1, 2)))
    return out


def candidates(window: str, side: dict, tm: dict, index, mirror: str, offset: float, W: int = 1280, H: int = 720) -> int:
    """Camera-truth candidates at a fixed offset: frozen truth_for_frame rows, frozen frames_from_rows."""
    from sportvu import camera as cam
    from sportvu.truth import CLOCK_TOL_S, frame_clock, truth_for_frame
    rows = []
    for r in side["frames"]:
        q, c = frame_clock(None, tm, r["frame"])
        if q is None:
            continue
        i = index.at(q, c + offset, CLOCK_TOL_S)
        if i is None:
            continue
        t = truth_for_frame(r, index.q[q]["xy"][i], index.q[q]["pid"][i], mirror)
        t.update({"frame": r["frame"], "q": q, "clock": c + offset, "moment": int(i), "state": r["state"]})
        rows.append(t)
    return len(cam.frames_from_rows({"window": window}, side, rows, index, mirror, W, H))


def _game(tag: str):
    from sportvu.sync import SportVUIndex
    from tools import p8_frozen as F
    refuse_cv_test([tag], WHO)
    guard([tag], WHO)
    g = F.game_of(tag)
    P = F.install_paths()
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    return P, index, F.mirror_of(P, tag)


def measure_windows(P: dict, index, mirror: str, tag: str) -> dict:
    """Per label window with a V3 label build, an r23 rebuild and a time map: the frozen and v2 offsets of both builds,
    the stored frozen offset and the V3 candidates at each rule's offset."""
    from tools import p8_frozen as F
    stored = json.loads((P["labels"] / f"{tag}_offsets.json").read_text())
    rows = {}
    for w in F.label_windows(P, tag):
        s = w["window"]
        v3_p, r23_p, tm_p = (P["label_build"] / f"{s}_frames.json", P["builds"] / "r23" / f"{s}_frames.json",
                             P["label_build"] / f"{s}_timemap.json")
        if not (v3_p.exists() and r23_p.exists() and tm_p.exists()):
            continue
        tm = json.loads(tm_p.read_text())
        v3, r23 = json.loads(v3_p.read_text()), json.loads(r23_p.read_text())
        pv3, pr23 = rows_pc_of(v3, tm), rows_pc_of(r23, tm)
        rec = {"frozen": {"v3": solve_frozen(pv3, index, mirror), "r23": solve_frozen(pr23, index, mirror)},
               "v2": {"v3": solve_v2(pv3, index, mirror), "r23": solve_v2(pr23, index, mirror)},
               "stored_frozen_v3": stored.get(s)}
        fz, vz = rec["frozen"]["v3"], rec["v2"]["v3"]["offset_s"]
        rec["candidates_v3"] = {"frozen": candidates(s, v3, tm, index, mirror, fz) if fz is not None else None,
                                "v2": candidates(s, v3, tm, index, mirror, vz) if vz is not None else None}
        rows[s] = rec
    return rows


def cmd_measure(args) -> dict:
    import time
    if args.tag not in POOLED + APART:
        raise SystemExit(f"{WHO}: {args.tag} is not a registered game for truth v2")
    P, index, mirror = _game(args.tag)
    t0 = time.time()
    rows = measure_windows(P, index, mirror, args.tag)
    out = {"tag": args.tag, "rule": "docs/research/P8_truth_v2.md", "delta_ft": DELTA_FT, "trim": TRIM,
           "windows": rows, "seconds": round(time.time() - t0, 1)}
    d = p8_root() / "truth_v2"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{args.tag}.json", json.dumps(out, indent=1))
    print(json.dumps({"tag": args.tag, "windows": len(rows), "seconds": out["seconds"]}, indent=1))
    return out


def dev_record(tag: str, rows: dict, seconds: float) -> dict:
    """The report-only record of measure-dev: the windows and the frozen and v2 stability of the game (game_summary),
    marked as outside the registration."""
    summ = game_summary(rows)
    summ.pop("_pairs")
    return {"tag": tag, "report_only": True, "rule": "docs/research/P8_truth_v2.md measures, read by check C of "
            "docs/research/P8_truth_arenas.md; not part of the truth v2 registration", "delta_ft": DELTA_FT,
            "trim": TRIM, "summary": summ, "windows": rows, "seconds": round(seconds, 1)}


def dev_path(tag: str) -> Path:
    return p8_root() / "truth_v2" / "dev" / f"{tag}.json"


def cmd_measure_dev(args) -> dict:
    """Report only. The registered games are refused here (they use measure and the registered report); any other
    development game passes the project guard through _game first."""
    import time
    if args.tag in POOLED + APART:
        raise SystemExit(f"{WHO}: {args.tag} is a registered truth v2 game; use measure")
    P, index, mirror = _game(args.tag)
    t0 = time.time()
    out = dev_record(args.tag, measure_windows(P, index, mirror, args.tag), time.time() - t0)
    p = dev_path(args.tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(out, indent=1))
    print(json.dumps({"tag": args.tag, "windows": len(out["windows"]), "frozen": out["summary"]["frozen"]}, indent=1))
    return out


def cmd_device(args) -> dict:
    P, index, mirror = _game("gsw_ind_2016")
    out = {}
    for s, builds in DEVICE_BUILDS.items():
        tm = json.loads((P["label_build"] / f"{s}_timemap.json").read_text())
        res = {}
        for name, folder in builds.items():
            side = json.loads(((P["label_build"] if folder is None else DIAG / folder) / f"{s}_frames.json").read_text())
            pc = rows_pc_of(side, tm)
            res[name] = {"frozen": solve_frozen(pc, index, mirror), "v2": solve_v2(pc, index, mirror)}
        out[s] = res
    d = p8_root() / "truth_v2"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "device_gsw_ind_2016.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ── report (no model) ───────────────────────────────────────────────────────────────────────────────────────────────
def game_summary(wins: dict) -> dict:
    keep = {s: r for s, r in wins.items() if r["frozen"]["v3"] == r["stored_frozen_v3"]}
    mism = sorted(set(wins) - set(keep))
    fz = [(r["frozen"]["v3"], r["frozen"]["r23"]) for r in keep.values()]
    v2 = [(r["v2"]["v3"]["offset_s"], r["v2"]["r23"]["offset_s"]) for r in keep.values()]
    cf = sum(r["candidates_v3"]["frozen"] or 0 for r in keep.values())
    cv = sum(r["candidates_v3"]["v2"] or 0 for r in keep.values())
    return {"windows": len(wins), "frozen_recompute_mismatch": mism, "frozen": stability(fz), "v2": stability(v2),
            "v2_at_edge": {b: sum(1 for r in keep.values() if r["v2"][b]["at_edge"]) for b in ("v3", "r23")},
            "candidates_v3": {"frozen": cf, "v2": cv, "ratio": round(cv / cf, 4) if cf else None},
            "_pairs": (fz, v2)}


def report(games: dict) -> dict:
    out = {"per_game": {}, "pooled": None, "bar": None}
    pooled_fz, pooled_v2, cf, cv = [], [], 0, 0
    for t, wins in games.items():
        s = game_summary(wins)
        fz, v2 = s.pop("_pairs")
        out["per_game"][t] = s
        if t in POOLED:
            pooled_fz += fz
            pooled_v2 += v2
            cf += s["candidates_v3"]["frozen"]
            cv += s["candidates_v3"]["v2"]
    if all(t in games for t in POOLED):
        st_f, st_v = stability(pooled_fz), stability(pooled_v2)
        iv = share_reduction_interval(pooled_fz, pooled_v2)
        ratio = cv / cf if cf else None
        checks = {"median_at_most_half": st_v["median_abs_s"] is not None and st_f["median_abs_s"] is not None
                  and st_v["median_abs_s"] <= 0.5 * st_f["median_abs_s"],
                  "share_over_0_2_lower": st_v["share_over_0_2_s"] < st_f["share_over_0_2_s"],
                  "share_reduction_interval_above_zero": bool(iv) and iv[0] > 0,
                  "candidates_at_least_95_percent": ratio is not None and ratio >= 0.95}
        out["pooled"] = {"frozen": st_f, "v2": st_v, "share_reduction_interval": iv,
                         "candidates_v3": {"frozen": cf, "v2": cv, "ratio": round(ratio, 4) if ratio else None}}
        out["bar"] = {"pass": all(checks.values()), "checks": checks}
    return out


def cmd_report(args) -> dict:
    d = p8_root() / "truth_v2"
    games = {t: json.loads((d / f"{t}.json").read_text())["windows"] for t in POOLED + APART if (d / f"{t}.json").exists()}
    res = report(games)
    if (d / "device_gsw_ind_2016.json").exists():
        res["device_gsw_ind_2016"] = json.loads((d / "device_gsw_ind_2016.json").read_text())
    atomic_write(d / "truth_v2_result.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.p8_truth_v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("tag")
    md = sub.add_parser("measure-dev")
    md.add_argument("tag")
    sub.add_parser("device")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"measure": cmd_measure, "measure-dev": cmd_measure_dev, "device": cmd_device, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
