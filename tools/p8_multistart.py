"""tools/p8_multistart.py: the multi-pool camera fit (docs/research/P8_multistart_fit.md, version 2). Stages 0 and 1.

    <frozen CV python> -m tools.p8_multistart multipool GAME   (V3_P8_ROOT = work_ref; lock "p8 multistart stage1 GAME
        multipool") -> work_ms/multistart/GAME/ and reports/research/p8_multistart_multipool_GAME.json
    <frozen CV python> -m tools.p8_multistart truth GAME       (V3_P8_ROOT = work_ms; lock "p8 multistart stage1 GAME
        truth") -> the multipool v1 truth in work_ms/labels and reports/research/p8_multistart_truth_GAME.json

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


# ── stage 1 (design sections 4 to 6, 9, 10, 13) ──────────────────────────────────────────────────────────────────
SEED_SETS = {"truth": tuple(range(0, 8)), "stab_a": tuple(range(16, 24)), "stab_b": tuple(range(24, 32))}
CHUNKS = 20                        # 20 chunks of 200 = the 4,000-evaluation cap
CONV_FT, CONV_COST = 0.05, 0.001   # chunk-rule convergence: C within 0.05 ft and cost within 0.1 percent
STAB_FT = 0.25                     # stability bar, median foot distance between selections
KILL4_FT = 5.0                     # stage 2 runs only if gsw_cha_2016's selection is more than 5 ft from its frozen C
STAGE1_GAMES = ("gsw_cha_2016", "gsw_bkn_2015")
TIE_REL = 1e-9


def converged_at(chunks: list, ft: float = CONV_FT, rel: float = CONV_COST):
    """Index (1-based) of the first chunk end that counts as converged, or None: scipy status 1 to 4, or C within `ft`
    and cost within `rel` of the previous chunk end."""
    for i, c in enumerate(chunks):
        if 1 <= c["status"] <= 4:
            return i + 1
        if i > 0:
            p = chunks[i - 1]
            if np.linalg.norm(np.asarray(c["C"]) - np.asarray(p["C"])) < ft and abs(p["cost"] - c["cost"]) < rel * p["cost"]:
                return i + 1
    return None


def select(runs: list) -> dict:
    """The run with the lowest J; a tie (relative difference under TIE_REL) goes to the lower seed. Every run competes,
    converged or not."""
    best = None
    for r in sorted(runs, key=lambda r: r["seed"]):
        if best is None or r["J"] < best["J"] * (1.0 - TIE_REL):
            best = r
    return best


def foot_distances(maps_a: dict, maps_b: dict) -> np.ndarray:
    """maps_*: {(window, frame): court ft of every box foot pixel, (n, 2)} -> the distances of the same feet."""
    out = [np.linalg.norm(np.asarray(maps_a[k]) - np.asarray(maps_b[k]), axis=1) for k in sorted(set(maps_a) & set(maps_b))
           if len(maps_a[k])]
    return np.concatenate(out) if out else np.zeros(0)


def stability(maps: dict, bar: float = STAB_FT) -> dict:
    """maps: {set name: per-frame foot maps}; pairwise median foot distance, each at most `bar`."""
    names, pairs = sorted(maps), {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = foot_distances(maps[names[i]], maps[names[j]])
            pairs[f"{names[i]}|{names[j]}"] = {"feet": int(len(d)), "median_ft": round(float(np.median(d)), 4) if len(d) else None}
    ok = bool(pairs) and all(v["median_ft"] is not None and v["median_ft"] <= bar for v in pairs.values())
    return {"pairs": pairs, "bar_ft": bar, "passes": ok}


def kill4(selected_C, frozen_C, ft: float = KILL4_FT) -> dict:
    d = float(np.linalg.norm(np.asarray(selected_C) - np.asarray(frozen_C)))
    return {"distance_ft": round(d, 3), "bar_ft": ft, "stage2_runs": d > ft}


def keys_hash(frames: list) -> str:
    return hashlib.sha256(json.dumps([(f["window"], int(f["frame"])) for f in frames]).encode()).hexdigest()


class replaced_fit_game:
    """Within the block, sportvu.camera.fit_game returns `result` (the selected run) instead of fitting; it asserts the
    original function at entry, the frames' (window, frame) hash at the call and exactly one call, and restores the
    original in a finally block (design section 10)."""
    def __init__(self, result: dict, frames_hash: str):
        self.result, self.frames_hash, self.calls = result, frames_hash, 0

    def __enter__(self):
        from sportvu import camera as cam
        self.cam, self.orig = cam, cam.fit_game
        if getattr(self.orig, "__module__", None) != "sportvu.camera" or self.orig.__name__ != "fit_game":
            raise SystemExit(f"{WHO}: sportvu.camera.fit_game is not the original at entry")

        def repl(frames, seed=0):
            if keys_hash(frames) != self.frames_hash:
                raise SystemExit(f"{WHO}: the fit step's frames are not the frames the selection used")
            self.calls += 1
            return dict(self.result)
        cam.fit_game = repl
        return self

    def __exit__(self, *exc):
        self.cam.fit_game = self.orig
        if self.cam.fit_game is not self.orig:
            raise SystemExit(f"{WHO}: the original fit_game was not restored")
        if exc[0] is None and self.calls != 1:
            raise SystemExit(f"{WHO}: the fit step called fit_game {self.calls} times, not once")
        return False


def roots_ms() -> tuple:
    base = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
    work, ref, ms = base / "work", check_outside(base / "work_ref"), check_outside(base / "work_ms")
    check_roots(work, ref, ms)
    return work, ref, ms


def check_roots(work: Path, ref: Path, ms: Path) -> None:
    rp = [Path(os.path.realpath(x)) for x in (work, ref, ms)]
    if len(set(rp)) != 3:
        raise SystemExit(f"{WHO}: work, work_ref and work_ms must be three different folders")


def setup_ms(ref: Path, ms: Path, game: str) -> dict:
    """work_ms: directory links to work_ref's label builds, clips, time maps, sync maps and moments; file links to the
    game's mirror and layout records; copies of the game's label done markers. Returns the layout."""
    import shutil
    from tools import p8_frozen as F
    ms.mkdir(parents=True, exist_ok=True)
    for name in ("label_build", "video", "sync", "sportvu"):
        dst = ms / name
        if not dst.exists():
            dst.symlink_to(ref / name, target_is_directory=True)
        elif not dst.is_symlink() or Path(os.path.realpath(dst)) != Path(os.path.realpath(ref / name)):
            raise SystemExit(f"{WHO}: {dst} is not the link to {ref / name}")
    P = F.layout(ms)
    for k in ("reports", "meta", "labels"):
        P[k].mkdir(parents=True, exist_ok=True)
    for src, dst in ((ref / "reports" / f"sportvu_direction_{game}.json", P["reports"] / f"sportvu_direction_{game}.json"),
                     (ref / "meta" / f"{game}_layout.json", P["meta"] / f"{game}_layout.json")):
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    (P["meta"] / "done").mkdir(parents=True, exist_ok=True)
    for m in sorted((ref / "meta" / "done").glob(f"label__{game}__*.json")):
        if not (P["meta"] / "done" / m.name).exists():
            shutil.copy2(m, P["meta"] / "done" / m.name)
    return P


def unfitted_boxes(fr: dict, boxes: list) -> list:
    """The frame's sidecar boxes minus those of its inlier pairs (the boxes the per-frame fit was fitted to; review R1)."""
    fitted = {p["box"] for p in (fr.get("row") or {}).get("pairs", []) if p.get("inlier")}
    return [b for i, b in enumerate(boxes) if i not in fitted]


def foot_maps(C, init_frames: list, sol: list, sidecars: dict, counts: dict | None = None) -> dict:
    """Every sidecar box foot_stab pixel of each init frame, except the boxes of its inlier pairs, mapped to court ft
    through the frame's per-frame H at C (fit_frame's solution) -> {(window, frame): (n, 2)} (amendment S1-1, review
    R1). counts, if given, receives the feet kept and the inlier-pair feet dropped."""
    import cv2
    from sportvu.camera import h_px2ft
    out = {}
    for fr, (rv, f) in zip(init_frames, sol):
        if rv is None:
            continue
        all_boxes = sidecars.get(fr["window"], {}).get(int(fr["frame"]), [])
        boxes = unfitted_boxes(fr, all_boxes)
        if counts is not None:
            counts["feet_kept"] = counts.get("feet_kept", 0) + len(boxes)
            counts["inlier_feet_dropped"] = counts.get("inlier_feet_dropped", 0) + len(all_boxes) - len(boxes)
        px = np.array([b["foot_stab"] for b in boxes], np.float64).reshape(-1, 1, 2)
        if not len(px):
            continue
        H = h_px2ft(np.asarray(C, float), rv, f, fr["w"], fr["h"])
        out[(fr["window"], int(fr["frame"]))] = cv2.perspectiveTransform(px, H).reshape(-1, 2)
    return out


def per_frame(C, init_frames: list) -> list:
    sol = []
    for fr in init_frames:
        _, rv, f = frame_cost(C, fr)
        sol.append((rv, f))
    return sol


def stale_run(r: dict, frames_hash: str, C0, inits: int) -> str | None:
    """Why a saved run does not belong to these inputs, or None (review R5)."""
    if r.get("frames_hash") != frames_hash:
        return "frames hash differs"
    if r.get("inits") != inits:
        return "inits differ"
    if r.get("C0") is None or np.linalg.norm(np.asarray(r["C0"]) - np.asarray(C0)) > 1e-9:
        return "C0 differs"
    return None


def truth_result(s: dict, rec: dict) -> dict:
    """fit_game's return value for the selected run (review R2): the pooled medians rounded to 2 places as fit_game
    rounds them (the 4-place value stays in the multipool record), pooled frames and inits from the run."""
    return {"status": "ok", "C": s["C"], "C0": rec["C0"], "pooled_frames": s["pooled_frames"], "inits": s["inits"],
            "reproj_px_median_before": round(float(s["pooled_median_px_at_start"]), 2),
            "reproj_px_median_after": round(float(s["pooled_median_px"]), 2), "cost": s["cost"],
            "multipool": {"seed": s["seed"], "J": s["J"], "converged_at_chunk": s["converged_at_chunk"],
                          "pooled_median_px_4dp": s["pooled_median_px"]}}


def check_labels_inside(P: dict, ms: Path, game: str) -> None:
    """P["labels"] and labels/<game> resolve inside work_ms and are not links (review R5)."""
    for p in (P["labels"], P["labels"] / game):
        if p.exists() and (p.is_symlink() or not Path(os.path.realpath(p)).is_relative_to(Path(os.path.realpath(ms)))):
            raise SystemExit(f"{WHO}: {p} is a link or lies outside work_ms")
    if not Path(os.path.realpath(P["labels"])).is_relative_to(Path(os.path.realpath(ms))):
        raise SystemExit(f"{WHO}: the labels folder is not inside work_ms")


def gate_bkn(ms: Path) -> None:
    """gsw_bkn_2015 runs only if gsw_cha_2016's multipool record shows stage 2 runs (kill rule 4) and stability passes,
    and gsw_cha_2016's multipool truth in work_ms reads "use" (review R3, section 15)."""
    mp = ms / "multistart" / "gsw_cha_2016" / "multipool.json"
    ac = ms / "truth_accept" / "gsw_cha_2016.json"
    if not mp.exists():
        raise SystemExit(f"{WHO}: gsw_bkn_2015 waits for gsw_cha_2016's multipool record")
    rec = json.loads(mp.read_text())
    if not (rec.get("kill4") or {}).get("stage2_runs") or not rec["stability"]["passes"]:
        raise SystemExit(f"{WHO}: gsw_bkn_2015 does not run: gsw_cha_2016 failed kill rule 4 or stability")
    if not ac.exists() or json.loads(ac.read_text()).get("use") != "use":
        raise SystemExit(f"{WHO}: gsw_bkn_2015 does not run: gsw_cha_2016's multipool truth does not read use")


def accept(game: str) -> dict:
    """Stage 3 for one game in work_ms (section 9): copies of work_ref's check C composite for the game, then
    tools.p8_truth_v2 measure-dev and tools.p8_truth_accept, unchanged, with section 5's bars."""
    import shutil
    from tools import p8_truth_accept as TA
    from tools import p8_truth_v2 as TV
    if game not in STAGE1_GAMES:
        raise SystemExit(f"{WHO}: {game} is not a stage 1 game")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    work, ref, ms = roots_ms()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ms.resolve():
        raise SystemExit(f"{WHO}: run accept with V3_P8_ROOT set to work_ms")
    if not (ms / "labels" / f"{game}_result.json").exists():
        raise SystemExit(f"{WHO}: {game} has no multipool truth in work_ms")
    before, t0 = manifest(ref), time.time()
    (ms / "builds" / "r23").mkdir(parents=True, exist_ok=True)
    for src in [*sorted((ref / "builds" / "r23").glob(f"{game}_s[0-9][0-9]_w[0-9][0-9]_*")),
                ref / "builds" / f"r23_checkc_{game}.json"]:
        dst = (ms / "builds" / "r23" / src.name) if src.parent.name == "r23" else (ms / "builds" / src.name)
        if not dst.exists():
            shutil.copy2(src, dst)
    TV.cmd_measure_dev(argparse.Namespace(tag=game))
    out = TA.accept(game, ms)
    after = manifest(ref)
    rec = {"game": game, "truth_version": "multipool v1", "acceptance": out,
           "guard_manifest": {"files": len(before), "unchanged": before == after}, "seconds": round(time.time() - t0, 1)}
    atomic_write(ms / "multistart" / game / "accept.json", json.dumps(rec, indent=1, default=str))
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_accept_{game}.json", json.dumps(rec, indent=1, default=str))
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed")
    return rec


def multipool(game: str) -> dict:
    """Section 4 and 6 for one game: three seed sets of 8 runs (each run's first chunk checked against fit_game), J at
    every run's final C, the selection per set, and stability at unfitted pixels. Resumable: each run is saved."""
    from sportvu import camera as cam
    from sportvu.sync import SportVUIndex
    from tools import p8_frozen as F
    from tools.p8_cha_fit import inits_of
    if game not in STAGE1_GAMES:
        raise SystemExit(f"{WHO}: {game} is not a stage 1 game")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    work, ref, ms = roots_ms()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ref.resolve():
        raise SystemExit(f"{WHO}: run with V3_P8_ROOT set to the reference root")
    if game == "gsw_bkn_2015":
        gate_bkn(ms)
    before, t0 = manifest(ref), time.time()
    P = F.install_paths()
    g = F.game_of(game)
    mirror = F.mirror_of(P, game)
    index = SportVUIndex(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    secs = [s for s in F.sections(P, game) if (P["sync"] / f"{s}_local_timemap.json").exists()]
    wins = [w for s in secs for w in F.label_windows(P, game, s)]
    guard([w["window"] for w in wins], WHO)
    frames = build_inputs(P, wins, index, mirror, ref / "label_build")
    inits = inits_of(frames)
    init_frames = [fr for fr, _ in inits]
    C0 = np.median(np.array([p[2] for _, p in inits]), axis=0)
    d = ms / "multistart" / game
    d.mkdir(parents=True, exist_ok=True)
    fh = keys_hash(frames)
    runs = {}
    for set_name, seeds in SEED_SETS.items():
        for seed in seeds:
            rp = d / f"seed_{seed:02d}.json"
            if rp.exists():
                r = json.loads(rp.read_text())
                stale = stale_run(r, fh, C0, len(inits))
                if stale:
                    raise SystemExit(f"{WHO}: {rp.name} is stale ({stale}); move it aside before resuming")
                runs[seed] = r
                continue
            t = time.time()
            fz = cam.fit_game(frames, seed=seed)
            pool = pool_of(inits, seed)
            resid, _ = problem(pool)
            x0 = start_vector(pool, C0)
            first, x1 = run_chunks(pool, x0, 1)
            diff = float(np.linalg.norm(np.asarray(first[0]["C"]) - np.asarray(fz["C"])))
            if diff > REPRO_FT:                          # kill rule 1, checked before the other 19 chunks (review R4)
                raise SystemExit(f"{WHO}: kill rule 1, {game} seed {seed}: first chunk {diff:.4f} ft from fit_game")
            rest, x = run_chunks(pool, x1, CHUNKS - 1)
            chunks = first + rest
            jr = common_objective(x[:3], init_frames)
            r = {"seed": seed, "set": set_name, "frames_hash": fh, "C": x[:3].tolist(), "C0": C0.tolist(), "J": jr["J"],
                 "fit_frame_failed": jr["fit_frame_failed"], "report_only_all_inits_median_px": jr["report_only_all_inits_median_px"],
                 "pooled_median_px": chunks[-1]["pooled_median_px"],
                 "pooled_median_px_at_start": round(float(np.median(np.abs(resid(x0)))), 2),
                 "cost": chunks[-1]["cost"], "converged_at_chunk": converged_at(chunks), "first_chunk_vs_fit_game_ft": diff,
                 "inits": len(inits), "pooled_frames": len(pool), "chunks": chunks, "seconds": round(time.time() - t, 1)}
            atomic_write(rp, json.dumps(r, indent=1))
            runs[seed] = r
    bad_j = [s for s, r in runs.items() if not np.isfinite(r["J"]) or r["fit_frame_failed"]]
    if bad_j:
        raise SystemExit(f"{WHO}: invalid J (non-finite or fit_frame failures) for seeds {bad_j}")
    sel = {name: select([runs[s] for s in seeds]) for name, seeds in SEED_SETS.items()}
    sidecars = {}
    for w in wins:
        sp = ref / "label_build" / (w["window"] + "_frames.json")
        if sp.exists():
            sidecars[w["window"]] = {int(r["frame"]): r["boxes"] for r in json.loads(sp.read_text())["frames"]}
    counts = {name: {} for name in sel}
    maps = {name: foot_maps(r["C"], init_frames, per_frame(r["C"], init_frames), sidecars, counts[name])
            for name, r in sel.items()}
    stab = stability(maps)
    stab["feet_per_selection"] = counts
    out = {"game": game, "design": "docs/research/P8_multistart_fit.md version 2 with sections 13 and 14",
           "inits": len(inits), "frames_hash": keys_hash(frames), "C0": C0.tolist(),
           "selected": {name: {k: r[k] for k in ("seed", "C", "J", "pooled_median_px", "pooled_median_px_at_start", "cost",
                                                 "converged_at_chunk", "report_only_all_inits_median_px", "pooled_frames",
                                                 "inits")} for name, r in sel.items()},
           "runs": {s: {k: r[k] for k in ("set", "C", "J", "pooled_median_px", "converged_at_chunk", "seconds")} for s, r in sorted(runs.items())},
           "converged_per_set": {name: sum(1 for s in seeds if runs[s]["converged_at_chunk"]) for name, seeds in SEED_SETS.items()},
           "stability": stab,
           "report_only_pairwise_C_ft": {f"{a}|{b}": round(float(np.linalg.norm(np.asarray(sel[a]["C"]) - np.asarray(sel[b]["C"]))), 3)
                                         for i, a in enumerate(sorted(sel)) for b in sorted(sel)[i + 1:]}}
    if game == "gsw_cha_2016":
        frozen = json.loads((ref / "labels" / f"{game}_result.json").read_text())["camera"]["C"]
        out["kill4"] = kill4(sel["truth"]["C"], frozen)
    if game == "gsw_bkn_2015":
        out["kill3_bkn_stable"] = stab["passes"]
    after = manifest(ref)
    out["guard_manifest"] = {"files": len(before), "unchanged": before == after}
    out["seconds"] = round(time.time() - t0, 1)
    atomic_write(d / "multipool.json", json.dumps(out, indent=1, default=str))
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_multipool_{game}.json", json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("game", "selected", "converged_per_set", "stability", "guard_manifest", "seconds")}
                     | ({"kill4": out["kill4"]} if "kill4" in out else {}), indent=1, default=str))
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed")
    return out


def truth(game: str) -> dict:
    """The truth step in work_ms: tools.p8_frozen.step_fit unchanged, with fit_game replaced by the selected run of the
    truth seed set (section 4.4). Refuses unless the game's multipool record exists and is stable."""
    from tools import p8_frozen as F
    if game not in STAGE1_GAMES:
        raise SystemExit(f"{WHO}: {game} is not a stage 1 game")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    work, ref, ms = roots_ms()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != ms.resolve():
        raise SystemExit(f"{WHO}: run the truth step with V3_P8_ROOT set to work_ms")
    rec_p = ms / "multistart" / game / "multipool.json"
    if not rec_p.exists():
        raise SystemExit(f"{WHO}: {game} has no multipool record")
    rec = json.loads(rec_p.read_text())
    if not rec["stability"]["passes"]:
        raise SystemExit(f"{WHO}: {game} is not stable; no multipool truth is written (section 6)")
    if (ms / "labels" / f"{game}_result.json").exists():
        raise SystemExit(f"{WHO}: {game} already has a truth in work_ms")
    if game == "gsw_bkn_2015":
        gate_bkn(ms)
    before, t0 = manifest(ref), time.time()
    setup_ms(ref, ms, game)
    P = F.install_paths(ms)
    s = rec["selected"]["truth"]
    result = truth_result(s, rec)
    check_labels_inside(P, ms, game)
    with replaced_fit_game(result, rec["frames_hash"]) as rp:
        res = F.step_fit(P, game)
    check_labels_inside(P, ms, game)
    hashes = {}
    for p in sorted([*(P["labels"] / game).glob("*.jsonl"), *P["labels"].glob(f"{game}_*")]):
        hashes[str(p.relative_to(ms))] = hashlib.sha256(p.read_bytes()).hexdigest()
    after = manifest(ref)
    out = {"game": game, "truth_version": "multipool v1", "selected": s, "fit_step": res, "fit_game_calls": rp.calls,
           "labels_sha256": hashes, "guard_manifest": {"files": len(before), "unchanged": before == after},
           "seconds": round(time.time() - t0, 1)}
    atomic_write(ms / "multistart" / game / "truth.json", json.dumps(out, indent=1, default=str))
    atomic_write(REPO / "reports" / "research" / f"p8_multistart_truth_{game}.json", json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("game", "fit_step", "fit_game_calls", "guard_manifest", "seconds")}, indent=1, default=str))
    if before != after:
        raise SystemExit(f"{WHO}: the reference root's protected files changed")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage0")
    for name in ("multipool", "truth", "accept"):
        sp = sub.add_parser(name)
        sp.add_argument("game")
    a = ap.parse_args(argv)
    {"stage0": lambda: stage0(), "multipool": lambda: multipool(a.game), "truth": lambda: truth(a.game),
     "accept": lambda: accept(a.game)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
