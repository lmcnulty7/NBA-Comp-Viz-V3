"""tools/p8_multistart.py: the multi-pool camera fit (docs/research/P8_multistart_fit.md, version 2). Stage 0 only.

    <frozen CV python> -m tools.p8_multistart stage0     (under the heavy-job lock, name
        "p8 multistart stage0 gsw_cha_2016"; V3_P8_ROOT set to the reference root)
        -> <reference root>/diag/multistart/stage0.json and reports/research/p8_multistart_stage0.json

Stage 0 (the design's section 3, registered before any number): on gsw_cha_2016's reset-fixed inputs, rebuilt as
tools.p8_frozen.step_fit builds them, (a) gates: 4,943 inits and fit_game(frames, seed=0) reproducing the recorded
fixed C within 0.01 ft; (b) the common objective J at five recorded positions (the broken fixed C and four good ones,
read from reports/research/p8_cha_fit_stage2.json) and the registered reading; (c) the per-frame optimality check on
the seed-0 pool at the broken C and on the seed-10 pool at G4, each with the copy's C checked against fit_game and
the recorded C (stop above 0.01 ft); a non-finite J or any fit_frame failure stops; (d) a report-only
chunked-continuation smoke; (e) timings. It reads J only for its
reading: never checks A, B or C, accepted frames, line support, the scorer or any score. The frozen code
(sportvu.camera, sportvu.truth, tools.p8_frozen.step_fit) is called, never edited. It writes only under
<reference root>/diag/multistart/ and the record, and checks a SHA-256 manifest of the reference root's labels,
builds/r23, truth pins and label sidecars before and after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, check_outside, guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_multistart"
GAME = "gsw_cha_2016"
F_SCALE = 6.0                      # sportvu.camera.F_SCALE_PX, checked at run time
CHUNK = 200                        # fit_game's evaluation cap; one chunk = one fit_game call
STOP_MARGIN = 0.01                 # stage 0: the good positions must beat the broken one by at least 1 percent of J_B
OPT_SLACK, OPT_SHARE = 0.01, 0.05  # stage 0: optimality check, 1 percent cost slack, 5 percent of frames
SMOKE_CHUNKS = 10
EXPECT_INITS = 4943
REPRO_FT = 0.01
STAGE2_RECORD = REPO / "reports" / "research" / "p8_cha_fit_stage2.json"
DESIGN_C = {"B": (47.0815, 104.3231, -34.5585), "G1": (46.9099, 127.7549, -31.3734), "G2": (46.4770, 130.3553, -29.9345),
            "G3": (47.3303, 130.5270, -33.9374), "G4": (47.8121, 137.3554, -30.9693)}   # as printed in the design


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def soft_l1_cost(r, f_scale: float = F_SCALE) -> float:
    """scipy's soft-L1 cost of residuals r with f_scale: f_scale^2 * sum(sqrt(1 + (r / f_scale)^2) - 1)."""
    r = np.asarray(r, float).ravel()
    return float(f_scale ** 2 * np.sum(np.sqrt(1.0 + (r / f_scale) ** 2) - 1.0))


def reading(j_b: float, j_goods: list, margin: float = STOP_MARGIN) -> str:
    """The stage 0 reading: STOP when the broken position scores at or below every good one, INCONCLUSIVE when the
    best good one is lower by less than `margin` of J_B, CONTINUE otherwise."""
    j_g = min(j_goods)
    if j_b <= j_g:
        return "STOP"
    if j_g > (1.0 - margin) * j_b:
        return "INCONCLUSIVE"
    return "CONTINUE"


def j_valid(js: dict) -> tuple:
    """(valid, reasons): every position's J finite and no fit_frame failure (re-check S0-3)."""
    bad = [f"{k}: J not finite" for k, v in js.items() if not np.isfinite(v.get("J", np.nan))]
    bad += [f"{k}: fit_frame failed on {v['fit_frame_failed']} inits" for k, v in js.items() if v.get("fit_frame_failed")]
    return not bad, bad


def optimality(fit_costs, pooled_costs, slack: float = OPT_SLACK, share: float = OPT_SHARE) -> dict:
    """Frames whose fit_frame cost is more than `slack` above the pooled pose's cost; fails above `share` of frames."""
    a, b = np.asarray(fit_costs, float), np.asarray(pooled_costs, float)
    worse = a > (1.0 + slack) * b + 1e-9
    s = float(worse.mean()) if len(a) else 0.0
    return {"frames": int(len(a)), "worse": int(worse.sum()), "share_worse": round(s, 4), "passes": s <= share,
            "median_ratio_fit_over_pooled": round(float(np.median(a / np.maximum(b, 1e-12))), 4) if len(a) else None}


def pool_of(inits: list, seed: int) -> list:
    """fit_game's pool: the seeded draw of up to POOL_MAX inits, in init order."""
    from sportvu.camera import POOL_MAX
    rng = np.random.default_rng(seed)
    return [inits[i] for i in sorted(rng.choice(len(inits), size=min(POOL_MAX, len(inits)), replace=False))]


def problem(pool: list):
    """fit_game's pooled residual and Jacobian sparsity for a pool (same parameterisation: C, then per frame rvec and
    log f)."""
    from scipy.sparse import lil_matrix
    from sportvu.camera import project
    sizes = [len(fr["px"]) for fr, _ in pool]

    def resid(x):
        C = x[:3]
        return np.concatenate([(project(C, x[3 + 4 * k: 6 + 4 * k], np.exp(x[6 + 4 * k]), fr["w"], fr["h"], fr["ft"])
                                - fr["px"]).ravel() for k, (fr, _) in enumerate(pool)])
    S = lil_matrix((2 * sum(sizes), 3 + 4 * len(pool)), dtype=int)
    row = 0
    for k, n in enumerate(sizes):
        S[row:row + 2 * n, :3] = 1
        S[row:row + 2 * n, 3 + 4 * k: 7 + 4 * k] = 1
        row += 2 * n
    return resid, S


def start_vector(pool: list, C0) -> np.ndarray:
    x0 = [*np.asarray(C0, float)]
    for _, (rv, f, _) in pool:
        x0 += [*rv, np.log(f)]
    return np.array(x0)


def run_chunks(pool: list, x0: np.ndarray, chunks: int, chunk_nfev: int = CHUNK) -> tuple:
    """fit_game's least-squares call (soft-L1, f_scale, x_scale "jac", sparsity) run for `chunks` chunks of
    `chunk_nfev` evaluations, each continuing from the previous chunk's end -> (per-chunk records, final x). The
    first chunk from C0 is exactly fit_game's call."""
    from scipy.optimize import least_squares
    from sportvu.camera import F_SCALE_PX
    resid, S = problem(pool)
    x, out = x0, []
    for _ in range(chunks):
        t = time.time()
        sol = least_squares(resid, x, jac_sparsity=S, loss="soft_l1", f_scale=F_SCALE_PX, x_scale="jac",
                            max_nfev=chunk_nfev)
        x = sol.x
        out.append({"C": x[:3].tolist(), "cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status),
                    "pooled_median_px": round(float(np.median(np.abs(sol.fun))), 4), "seconds": round(time.time() - t, 2)})
    return out, x


def frame_cost(C, fr: dict) -> tuple:
    """(cost_k, rvec, f) of one init at position C: the frozen fit_frame's solution and its soft-L1 cost."""
    from sportvu.camera import fit_frame, project
    fit = fit_frame(np.asarray(C, float), fr)
    if fit.get("status") != "ok":
        return None, None, None
    rv, f = np.asarray(fit["rvec"], float), float(fit["f"])
    r = project(C, rv, f, fr["w"], fr["h"], fr["ft"]) - fr["px"]
    return soft_l1_cost(r), rv, f


def common_objective(C, frames: list) -> dict:
    """J(C): the sum of frame_cost over the frames (the game's inits); also the all-inits median |residual| (reported,
    never ruled) and the frames fit_frame could not fit."""
    from sportvu.camera import project
    t, total, absr, failed = time.time(), 0.0, [], 0
    for fr in frames:
        c, rv, f = frame_cost(C, fr)
        if c is None:
            failed += 1
            continue
        total += c
        absr.append(np.abs(project(C, rv, f, fr["w"], fr["h"], fr["ft"]) - fr["px"]).ravel())
    med = float(np.median(np.concatenate(absr))) if absr else None
    return {"J": total, "frames": len(frames), "fit_frame_failed": failed,
            "report_only_all_inits_median_px": round(med, 4) if med is not None else None,
            "seconds": round(time.time() - t, 1)}


# ── guards ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def manifest(ref: Path) -> dict:
    """SHA-256 of every file the item must never change in the reference root."""
    files = [*sorted((ref / "labels").rglob("*")), *sorted((ref / "builds" / "r23").glob("*")),
             *sorted((ref / "resetfix").glob("truth_pin__*.json")), *sorted((ref / "label_build").glob("*_frames.json"))]
    out = {}
    for p in files:
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p.relative_to(ref))] = h.hexdigest()
    return out


def recorded_positions(path: Path = STAGE2_RECORD) -> dict:
    """The five positions of stage 0 from the stage 2 record, checked against the design's printed values."""
    d = json.loads(Path(path).read_text())
    by_seed = {r["seed"]: r["C"] for r in d["F1"]["fix"]}
    pos = {"B": d["F0"]["fix"]["frozen"]["C"], "G1": d["F0"]["nofix"]["frozen"]["C"],
           "G2": by_seed[1], "G3": by_seed[7], "G4": by_seed[10]}
    for k, c in pos.items():
        if np.linalg.norm(np.asarray(c) - np.asarray(DESIGN_C[k])) > 1e-3:
            raise SystemExit(f"{WHO}: recorded position {k} {c} is not the design's {DESIGN_C[k]}")
    smoke = d["F2"]["report_only_from_C0_2000_evaluations"]
    return {"positions": pos, "single_2000": {"C": smoke["C"], "cost": smoke["cost"]}}


# ── stage 0 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_inputs(P: dict, wins: list, index, mirror: str, side_dir: Path) -> list:
    """tools.p8_frozen.step_fit's camera-fit inputs: every window with a sidecar and a time map, in order."""
    import cv2
    from sportvu.camera import frames_from_rows
    from sportvu.truth import window_truth
    out = []
    for w in wins:
        side_p, tm_p = side_dir / (w["window"] + "_frames.json"), P["label_build"] / (w["window"] + "_timemap.json")
        if not side_p.exists() or not tm_p.exists():
            continue
        sc, tm = json.loads(side_p.read_text()), json.loads(tm_p.read_text())
        _, _, rows = window_truth(w, sc, tm, index, mirror)
        cap = cv2.VideoCapture(str(P["label_build"] / (w["window"] + ".mp4")))
        W, H = int(cap.get(3)), int(cap.get(4))
        cap.release()
        out += frames_from_rows(w, sc, rows, index, mirror, W, H)
    return out


def stage0() -> dict:
    from sportvu import camera as cam
    from sportvu.sync import SportVUIndex
    from tools import p8_frozen as F
    from tools.p8_cha_fit import inits_of
    refuse_cv_test([GAME], WHO)
    guard([GAME], WHO)
    if cam.F_SCALE_PX != F_SCALE or cam.POOL_MAX != 400:
        raise SystemExit(f"{WHO}: sportvu.camera constants are not the ones this design pins")
    base = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
    ref = check_outside(base / "work_ref")
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ref.resolve():
        raise SystemExit(f"{WHO}: run stage0 with V3_P8_ROOT set to the reference root")
    rec = recorded_positions()
    before = manifest(ref)
    times, t0 = {}, time.time()
    P = F.install_paths()
    g = F.game_of(GAME)
    mirror = F.mirror_of(P, GAME)
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    secs = [s for s in F.sections(P, GAME) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    wins = [w for s in secs for w in F.label_windows(P, GAME, s)]
    guard([w["window"] for w in wins], WHO)
    frames = build_inputs(P, wins, index, mirror, ref / "label_build")
    times["inputs_s"] = round(time.time() - t0, 1)
    inits = inits_of(frames)
    t = time.time()
    fz = cam.fit_game(frames, seed=0)
    times["fit_game_s"] = round(time.time() - t, 1)
    gates = {"inits": len(inits), "inits_expected": EXPECT_INITS,
             "fit_game_C": fz.get("C"), "recorded_B": rec["positions"]["B"],
             "C_diff_ft": float(np.linalg.norm(np.asarray(fz["C"]) - np.asarray(rec["positions"]["B"]))) if fz.get("C") else None}
    gates["passes"] = gates["inits"] == EXPECT_INITS and gates["C_diff_ft"] is not None and gates["C_diff_ft"] <= REPRO_FT
    out = {"what": "docs/research/P8_multistart_fit.md version 2, stage 0; the reading reads J only", "game": GAME,
           "gates": gates}
    if gates["passes"]:
        init_frames = [fr for fr, _ in inits]
        js = {k: {"C": list(c), **common_objective(c, init_frames)} for k, c in rec["positions"].items()}
        times["J_s_each"] = [js[k]["seconds"] for k in js]
        j_b, j_g = js["B"]["J"], [js[k]["J"] for k in ("G1", "G2", "G3", "G4")]
        out["J"] = js
        out["reading"] = {"J_B": j_b, "J_G_min": min(j_g), "best_good": min(("G1", "G2", "G3", "G4"), key=lambda k: js[k]["J"]),
                          "relative_margin": (j_b - min(j_g)) / j_b, "verdict": reading(j_b, j_g)}
        # per-frame optimality at B (seed-0 pool) and at G4 (seed-10 pool), with the copy checks (S0-1, S0-2)
        C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
        from sportvu.camera import project
        out["optimality"], stops = {}, []
        for name, seed in (("B", 0), ("G4", 10)):
            t = time.time()
            fz_s = fz if seed == 0 else cam.fit_game(frames, seed=seed)
            pool = pool_of(inits, seed)
            first, x = run_chunks(pool, start_vector(pool, C0), 1)
            Cs = x[:3]
            pooled_c, fit_c = [], []
            for k, (fr, _) in enumerate(pool):
                r = project(Cs, x[3 + 4 * k: 6 + 4 * k], np.exp(x[6 + 4 * k]), fr["w"], fr["h"], fr["ft"]) - fr["px"]
                pooled_c.append(soft_l1_cost(r))
                fit_c.append(frame_cost(Cs, fr)[0])
            opt = optimality([c if c is not None else np.inf for c in fit_c], pooled_c)
            opt["seed"] = seed
            opt["copy_C"] = Cs.tolist()
            opt["copy_C_diff_from_fit_game_ft"] = float(np.linalg.norm(Cs - np.asarray(fz_s["C"])))
            opt["copy_C_diff_from_recorded_ft"] = float(np.linalg.norm(Cs - np.asarray(rec["positions"][name])))
            opt["seconds"] = round(time.time() - t, 1)
            out["optimality"][name] = opt
            if not opt["passes"]:
                stops.append(f"optimality at {name}: {opt['share_worse']} of frames worse")
            if opt["copy_C_diff_from_fit_game_ft"] > REPRO_FT:
                stops.append(f"copy at {name} differs from fit_game by {opt['copy_C_diff_from_fit_game_ft']:.4f} ft")
            if opt["copy_C_diff_from_recorded_ft"] > REPRO_FT:
                stops.append(f"copy at {name} differs from the recorded C by {opt['copy_C_diff_from_recorded_ft']:.4f} ft")
        times["chunk_s"] = first[0]["seconds"]
        pool = pool_of(inits, 0)
        # report-only chunk smoke: 10 chunks from C0 against the recorded single 2,000-evaluation run
        smoke, xs = run_chunks(pool, start_vector(pool, C0), SMOKE_CHUNKS)
        out["report_only_chunk_smoke"] = {
            "chunks": smoke, "final_C": xs[:3].tolist(), "final_cost": smoke[-1]["cost"],
            "single_2000_C": rec["single_2000"]["C"], "single_2000_cost": rec["single_2000"]["cost"],
            "C_diff_ft": float(np.linalg.norm(xs[:3] - np.asarray(rec["single_2000"]["C"]))),
            "cost_diff_rel": (smoke[-1]["cost"] - rec["single_2000"]["cost"]) / rec["single_2000"]["cost"]}
        ok, why = j_valid(js)
        if not ok:
            stops.insert(0, "J invalid: " + "; ".join(why))
        out["stops"] = stops
        if stops:
            out["reading"]["verdict_by_J_alone"] = out["reading"]["verdict"]
            out["reading"]["verdict"] = "STOP (" + "; ".join(stops) + ")"
    else:
        out["stopped"] = "gates failed: inputs or the frozen fit are not reproduced"
    after = manifest(ref)
    out["guard_manifest"] = {"files": len(before), "unchanged": before == after}
    times["total_s"] = round(time.time() - t0, 1)
    out["timings"] = times
    text = json.dumps(out, indent=1, default=str) + "\n"
    d = ref / "diag" / "multistart"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / "stage0.json", text)
    atomic_write(REPO / "reports" / "research" / "p8_multistart_stage0.json", text)
    print(json.dumps({k: out.get(k) for k in ("gates", "reading", "optimality", "guard_manifest", "timings", "stopped")},
                     indent=1, default=str))
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed during stage 0")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage0")
    ap.parse_args(argv)
    stage0()


if __name__ == "__main__":
    sys.exit(main())
