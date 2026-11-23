"""tools/p8_e1_score.py: score E1's leave-one-game-out models on the Mac (PLAN.md D156, D158; D5; D85 (2) and (3)).

    <frozen CV python> -m tools.p8_e1_score rebuild --model M --seed S --weights W --game TAG [--windows N]
    <frozen CV python> -m tools.p8_e1_score score --model M --seed S --game TAG
    python3 -m tools.p8_e1_score report [--models fold names]
    (internal) <python> -m tools.p8_e1_score child build_trajectories.py <its arguments>

Run rebuild and score under the heavy-job lock, one game at a time.

Model names: a fold is named by its held-out game (tools/p8_e1_colab.py GAMES); "stub" is the production detector
itself, used only to prove the plumbing (a stub build must reproduce r23 exactly).

Provenance (rebuild): a fold's weights must sit in their Colab run folder (<run>/train/weights/<file>), whose
run_record.json names this fold as its held-out game, does not list the scored game among its training games, and
carries this seed; its complete.json (written by the Colab run after training) must hold this file's sha256. The
stub's weights must be the production detector's (models/player_detector.pt of the main checkout), by sha256.

Rebuild: the game's label windows are built again exactly as r23 was (tools.p8_frozen.safe_build, gate v2 arguments,
court model r23 through COURT_GRID_WEIGHTS, the same device settings) with one change: the player detector is the
model's weights. sportvu.rebuild launches `python build_trajectories.py ...`; a proxy around its subprocess module (as
tools.p8_device does, no frozen file changed) runs it as `python -m tools.p8_e1_score child build_trajectories.py ...`,
which sets config.PLAYER_DETECTOR_WEIGHTS to the model and records the weights' sha256, the model and the seed in
<window>_detector.json. Output: builds/e1_<model>_seed<S>/. A cached window built with other weights is refused.

Score: the pinned scorer's score_window (tools/p8_scorecard.py, D88), truth camera, for r23 and the model on the same
windows; per window, the counts of every Stage 1 row and the matched players. The two builds' processed frame sets are
compared per window; where they differ, both are scored on the shared frames only (filtered copies of the sidecars)
and the difference is recorded. The output names the expected windows (label windows with an r23 sidecar), the scored
ones, the weights' sha256 and the seed (from the windows' detector records, which must agree).
Output: e1/<model>__seed<S>__<game>.json.

Report (D156, fixed before any training): per seed, the leave-one-game-out pool is each corpus game scored with the
fold that held it out (gsw_bkn_2015 by fold gsw_bkn_2015, and so on), against r23 on the same windows. Rows: position
p50 and p90 and near-field bias (the larger of the near and far thirds' absolute median toward the camera) on the
matched players both builds score; missed players; ghost boxes (non-referee); team labels (majority mapping per
window); coverage. Paired difference model minus r23, bootstrap by window (2000 resamples, seed 0) and by game (the 3
held-out games). D5 (amended 2026-10-04) with D158: a row is worse when its window or its game interval lies wholly on
the bad side of zero; it improves on the point estimate. A seed's pool adopts when every row except coverage improves
and none is worse, and coverage stays at or above 0.50 (or, if r23 is already below, does not fall). A pool with any
missing window is marked partial and never adopts. E1 adopts only when all 3 seeds are scored and each seed's pool
adopts (D85 (3)); until then the verdict says how many seeds are in. hou_cle_2016 is scored with every fold's model and
reported apart (window bootstrap only); gsw_ind_2016 stays apart (D98) and is not in the pool. Identity is not scored
here.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1_score"
FOLDS = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = ("hou_cle_2016",)
MODELS = (*FOLDS, "stub")
SEEDS_NEEDED = 3
ENV = "P8_E1_DETECTOR"
BUILD_SCRIPT = "build_trajectories.py"
BOOT_N, BOOT_SEED = 2000, 0
COVERAGE_FLOOR = 0.50
# row -> True when higher is better
ROWS = {"position_p50_ft": False, "position_p90_ft": False, "nearfield_bias_ft": False, "missed_rate": False,
        "ghost_rate": False, "team_accuracy": True, "coverage": True}


def build_name(model: str, seed: int) -> str:
    if model not in MODELS:
        raise SystemExit(f"{WHO}: unknown model {model}; one of {MODELS}")
    return f"e1_{model}_seed{int(seed)}"


def score_name(model: str, seed: int, game: str) -> str:
    return f"{model}__seed{int(seed)}__{game}.json"


def scored_games(model: str) -> list:
    """The games a model is scored on: a fold's held-out game and hou_cle_2016; the stub on any it is given."""
    return [model, *APART] if model in FOLDS else list(FOLDS + APART)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_detector() -> Path:
    from tools.p8_common import main_checkout
    return main_checkout() / "models" / "player_detector.pt"


def check_provenance(model: str, game: str, seed: int, weights: Path, prod_sha: str | None = None) -> dict:
    """The weights belong to this model and seed and were not trained on the scored game -> {sha256, seed, run}."""
    weights = Path(weights)
    sha = _sha(weights)
    if model == "stub":
        want = prod_sha or _sha(production_detector())
        if sha != want:
            raise SystemExit(f"{WHO}: the stub must be the production detector (sha256 {want[:12]}...), not {sha[:12]}...")
        return {"sha256": sha, "seed": int(seed), "run": None}
    run = weights.parents[2]
    rec_p = run / "run_record.json"
    if not rec_p.exists():
        raise SystemExit(f"{WHO}: no run_record.json in {run} (weights must sit in <run>/train/weights/)")
    rec = json.loads(rec_p.read_text())
    if rec.get("held_out_game") != model or rec.get("fold") != model:
        raise SystemExit(f"{WHO}: {weights.name} was trained for fold {rec.get('fold')}, not {model}")
    if game in rec.get("training_games", []):
        raise SystemExit(f"{WHO}: {game} is among the training games of these weights")
    if int(rec.get("seed", -1)) != int(seed):
        raise SystemExit(f"{WHO}: these weights are seed {rec.get('seed')}, not {seed}")
    comp_p = run / "complete.json"
    if not comp_p.exists():
        raise SystemExit(f"{WHO}: no complete.json in {run}: the Colab run did not finish")
    if json.loads(comp_p.read_text()).get("weights", {}).get(weights.name) != sha:
        raise SystemExit(f"{WHO}: {weights.name} does not match the sha256 in complete.json")
    return {"sha256": sha, "seed": int(seed), "run": run.name}


# ── the build proxy (driver side) and the child ────────────────────────────────────────────────────────────────────
class _Proxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


def rewrite(cmd, env=None):
    env = os.environ if env is None else env
    if env.get(ENV) and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_e1_score", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m tools.p8_e1_score child {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    part = config.TRACKING_DIR / f"{stem}_detector.json.part"
    part.write_text(json.dumps({"weights": weights.name, "sha256": _sha(weights), "model": os.environ.get("P8_E1_MODEL"),
                                "seed": int(os.environ.get("P8_E1_SEED", "-1"))}))
    os.replace(part, config.TRACKING_DIR / f"{stem}_detector.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_rebuild(args) -> dict:
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_device
    from tools import p8_frozen as F
    if args.game not in scored_games(args.model):
        raise SystemExit(f"{WHO}: model {args.model} is not scored on {args.game} (D156)")
    refuse_cv_test([args.game], WHO)
    guard([args.game], WHO)                            # spent held-out games too
    F.game_of(args.game)
    if p8_device.is_set():
        raise SystemExit(f"{WHO}: a P8 device flag is set; E1 builds use r23's device settings")
    weights = Path(args.weights).resolve()
    prov = check_provenance(args.model, args.game, args.seed, weights)
    sha = prov["sha256"]
    P = F.install_paths()
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV], os.environ["P8_E1_MODEL"], os.environ["P8_E1_SEED"] = str(weights), args.model, str(args.seed)
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / build_name(args.model, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    wins = [w for w in F.label_windows(P, args.game) if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    if args.windows:
        wins = wins[:args.windows]
    res = {"model": args.model, "seed": args.seed, "game": args.game, "weights_sha256": sha, "run": prov["run"],
           "windows": 0, "built": 0, "cached": 0}
    for w in wins:
        rec_p = out / f"{w['window']}_detector.json"
        if (out / f"{w['window']}_frames.json").exists():
            if not rec_p.exists() or json.loads(rec_p.read_text())["sha256"] != sha:
                raise SystemExit(f"{WHO}: {w['window']} in {out} was built with other detector weights")
        r = F.safe_build(P, w, P["label_build"] / f"{w['window']}.mp4", out, extra_args=gate_args())
        if not r["ok"]:
            raise SystemExit(f"{WHO}: build of {w['window']} failed: {r.get('tail')}")
        res["windows"] += 1
        res["cached" if r.get("cached") else "built"] += 1
        if not r.get("cached") and not r.get("empty"):
            got = json.loads(rec_p.read_text())["sha256"] if rec_p.exists() else None
            if got != sha:
                raise SystemExit(f"{WHO}: {w['window']} did not record the model's weights (proxy not used?)")
    print(json.dumps(res, indent=1))
    return res


# ── score (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def window_stats(rec: dict) -> dict:
    """The counts of every Stage 1 row in one scored window (a record of tools.p8_scorecard.score_window)."""
    from tools.p8_e3 import window_team
    return {"missed": rec["missed"], "in_frame": rec["in_frame"],
            "nr_ghosts": rec["ghosts"] - rec["ref_ghosts"], "nr_boxes": rec["boxes"] - rec["ref_boxes"],
            "team": window_team(rec["team_rows"]),
            "cov_acc": rec["coverage"]["accepted_live_wide_s"], "cov_live": rec["coverage"]["live_wide_s"],
            "matched": {f"{m['frame']}:{m['sv']}": [m["err"], m["toward_camera"], m["region"]] for m in rec["matched"]}}


def frame_sets(side_a: dict, side_b: dict) -> dict:
    """Processed frames of two builds' sidecars -> shared and one-sided counts."""
    a = {int(r["frame"]) for r in side_a["frames"]}
    b = {int(r["frame"]) for r in side_b["frames"]}
    return {"shared": sorted(a & b), "only_r23": len(a - b), "only_new": len(b - a), "equal": a == b}


def detector_records(new_dir: Path, stems: list) -> dict:
    """The weights sha256 and seed the windows were built with; SystemExit on a mix or a window without a record."""
    shas, seeds, missing = set(), set(), []
    for s in stems:
        p = new_dir / f"{s}_detector.json"
        if not p.exists():
            missing.append(s)
            continue
        r = json.loads(p.read_text())
        shas.add(r["sha256"])
        seeds.add(r.get("seed"))
    if missing:
        raise SystemExit(f"{WHO}: windows without a detector record in {new_dir}: {missing[:3]}")
    if len(shas) > 1 or len(seeds) > 1:
        raise SystemExit(f"{WHO}: {new_dir} mixes detector weights or seeds: {sorted(shas)} {sorted(map(str, seeds))}")
    return {"weights_sha256": next(iter(shas), None), "seed": next(iter(seeds), None)}


def _filtered(src_dir: Path, stem: str, keep: list, dest: Path) -> Path:
    """A build folder holding the window's sidecar cut to the kept frames, with its other files linked."""
    dest.mkdir(parents=True, exist_ok=True)
    side = json.loads((src_dir / f"{stem}_frames.json").read_text())
    k = set(keep)
    side["frames"] = [r for r in side["frames"] if int(r["frame"]) in k]
    atomic_write(dest / f"{stem}_frames.json", json.dumps(side))
    for p in src_dir.glob(f"{stem}_*"):
        if not p.name.endswith("_frames.json") and not (dest / p.name).exists():
            os.symlink(p, dest / p.name)
    return dest


def cmd_score(args) -> dict:
    import time
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    if args.game not in scored_games(args.model):
        raise SystemExit(f"{WHO}: model {args.model} is not scored on {args.game} (D156)")
    refuse_cv_test([args.game], WHO)
    guard([args.game], WHO)                            # spent held-out games too
    g = F.game_of(args.game)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, args.game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    r23_dir, new_dir = P["builds"] / "r23", P["builds"] / build_name(args.model, args.seed)
    t0 = time.time()
    all_w = F.label_windows(P, args.game)
    expected = [w["window"] for w in all_w if (r23_dir / f"{w['window']}_frames.json").exists()]
    built = [s for s in expected if (new_dir / f"{s}_frames.json").exists()]
    gate_empty = [s for s in expected if F.gate_empty_path(new_dir, s).exists()]
    prov = detector_records(new_dir, built + gate_empty)
    if prov["seed"] is not None and int(prov["seed"]) != int(args.seed):
        raise SystemExit(f"{WHO}: {new_dir} was built for seed {prov['seed']}, not {args.seed}")
    wins, frame_diff, skipped = {}, {}, []
    tmp = p8_root() / "e1" / "shared_frames" / build_name(args.model, args.seed)
    for w in all_w:
        s = w["window"]
        if s not in built:
            continue
        fs = frame_sets(json.loads((r23_dir / f"{s}_frames.json").read_text()),
                        json.loads((new_dir / f"{s}_frames.json").read_text()))
        dirs = {"r23": r23_dir, "new": new_dir}
        if not fs["equal"]:
            frame_diff[s] = {k: (len(v) if k == "shared" else v) for k, v in fs.items() if k != "equal"}
            dirs = {"r23": _filtered(r23_dir, s, fs["shared"], tmp / "r23"),
                    "new": _filtered(new_dir, s, fs["shared"], tmp / "new")}
        pair = {}
        for name, bd in dirs.items():
            rec = S.score_window(P, args.game, w, bd, "camera", mom, mirror, refs, events)[0]
            if "skipped" in rec:
                break
            pair[name] = window_stats(rec)
        if len(pair) == 2:
            wins[s] = pair
        else:
            skipped.append(s)
    out = {"model": args.model, "seed": args.seed, "game": args.game, "truth": "camera", **prov,
           "expected_windows": expected, "scored_windows": sorted(wins), "gate_empty_windows": gate_empty,
           "skipped_windows": skipped, "frame_sets_differ": frame_diff, "windows": wins,
           "seconds": round(time.time() - t0, 1)}
    (p8_root() / "e1").mkdir(parents=True, exist_ok=True)
    atomic_write(p8_root() / "e1" / score_name(args.model, args.seed, args.game), json.dumps(out))
    print(json.dumps({"model": args.model, "seed": args.seed, "game": args.game, "expected": len(expected),
                      "scored": len(wins), "frame_sets_differ": len(frame_diff), "seconds": out["seconds"],
                      **{k: v for k, v in rows([v for v in wins.values()]).items()}}, indent=1))
    return out


# ── rows, bootstrap and the D5 verdict (no model) ──────────────────────────────────────────────────────────────────
def _paired(wins: list):
    """Matched players both builds score: aligned (error, toward, region) arrays for r23 and the model."""
    eb, en, tb, tn, reg = [], [], [], [], []
    for w in wins:
        mb, mn = w["r23"]["matched"], w["new"]["matched"]
        for k in mb.keys() & mn.keys():
            eb.append(mb[k][0]); en.append(mn[k][0]); tb.append(mb[k][1]); tn.append(mn[k][1]); reg.append(mb[k][2])
    return np.array(eb), np.array(en), np.array(tb), np.array(tn), np.array(reg)


def _bias(t, reg):
    vals = [abs(float(np.median(t[reg == r]))) for r in ("near", "far") if (reg == r).any()]
    return max(vals) if vals else None


def rows(wins: list) -> dict:
    """Every row for r23 and the model on the same windows -> {row: (r23, model)}."""
    out = {}
    eb, en, tb, tn, reg = _paired(wins)
    out["position_p50_ft"] = (float(np.percentile(eb, 50)), float(np.percentile(en, 50))) if len(eb) else (None, None)
    out["position_p90_ft"] = (float(np.percentile(eb, 90)), float(np.percentile(en, 90))) if len(eb) else (None, None)
    out["nearfield_bias_ft"] = (_bias(tb, reg), _bias(tn, reg)) if len(eb) else (None, None)
    for row, num, den in (("missed_rate", "missed", "in_frame"), ("ghost_rate", "nr_ghosts", "nr_boxes"),
                          ("coverage", "cov_acc", "cov_live")):
        vals = []
        for side in ("r23", "new"):
            d = sum(w[side][den] for w in wins)
            vals.append(sum(w[side][num] for w in wins) / d if d else None)
        out[row] = tuple(vals)
    vals = []
    for side in ("r23", "new"):
        n = sum(w[side]["team"]["n"] for w in wins)
        vals.append(sum(w[side]["team"]["majority"] for w in wins) / n if n else None)
    out["team_accuracy"] = tuple(vals)
    return out


def diffs(wins: list) -> dict:
    """Model minus r23 for every row."""
    return {k: (None if a is None or b is None else b - a) for k, (a, b) in rows(wins).items()}


def boot(groups: dict, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Cluster bootstrap of every row's paired difference (model minus r23) -> {row: [lo, hi]} or None."""
    keys = [k for k, v in groups.items() if v]
    if len(keys) < 2:
        return {k: None for k in ROWS}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ROWS}
    for _ in range(n):
        pick = [w for i in rng.choice(len(keys), len(keys), replace=True) for w in groups[keys[i]]]
        for k, v in diffs(pick).items():
            if v is not None:
                acc[k].append(v)
    return {k: ([round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)] if v else None)
            for k, v in acc.items()}


def d5_verdict(point: dict, window_iv: dict, coverage: tuple, game_iv: dict | None = None) -> dict:
    """D5 as amended, with D158: worse = the window or the game interval wholly on the bad side of zero; improves =
    point estimate on the good side. Adopt when every row but coverage improves, none is worse, and coverage holds its
    floor."""
    checks = {}
    for row, higher in ROWS.items():
        d, iv = point.get(row), window_iv.get(row)
        if row == "coverage":
            base, new = coverage
            ok = new is not None and (new >= COVERAGE_FLOOR if (base or 0) >= COVERAGE_FLOOR else new >= base)
            checks[row] = {"diff": d, "interval": iv, "floor_holds": ok, "pass": ok}
            continue
        giv = (game_iv or {}).get(row)
        improves = d is not None and (d > 0 if higher else d < 0)
        worse = any(bool(v) and (v[1] < 0 if higher else v[0] > 0) for v in (iv, giv))
        checks[row] = {"diff": d, "interval": iv, "game_interval": giv, "improves": improves, "worse": worse,
                       "pass": improves and not worse}
    return {"adopt": all(c["pass"] for c in checks.values()), "rows": checks}


def complete(f: dict) -> bool:
    """A score file covers every expected window (gate-empty windows aside)."""
    return set(f["scored_windows"]) == set(f["expected_windows"]) - set(f.get("gate_empty_windows", []))


def check_hashes(files: dict) -> dict:
    """files: {(model, seed, game): score file} -> {(model, seed): sha256}; SystemExit when one model and seed's files
    disagree on the weights."""
    by = {}
    for (m, s, _g), f in files.items():
        by.setdefault((m, s), set()).add(f.get("weights_sha256"))
    bad = {k: v for k, v in by.items() if len(v) > 1}
    if bad:
        raise SystemExit(f"{WHO}: score files of one model and seed disagree on the weights: {bad}")
    return {k: next(iter(v)) for k, v in by.items()}


def report(files: dict) -> dict:
    """files: {(model, seed, game): score file} -> the D156 report, per seed and overall."""
    hashes = check_hashes(files)
    out = {"per_game": {}, "apart": {}, "seeds": {}, "pools": {}, "weights_sha256": {f"{m}__seed{s}": h for (m, s), h in hashes.items()}}
    for (model, seed, game), f in files.items():
        wins = list(f["windows"].values())
        r = rows(wins)
        entry = {"model": model, "seed": seed, "windows": len(wins), "complete": complete(f),
                 "frame_sets_differ": len(f.get("frame_sets_differ", {})),
                 "rows": {k: {"r23": a, "model": b} for k, (a, b) in r.items()},
                 "diff": diffs(wins), "window_interval": boot({i: [w] for i, w in enumerate(wins)})}
        (out["apart"] if game in APART else out["per_game"])[f"{model}__seed{seed}__{game}"] = entry
    seeds = sorted({s for (m, s, g) in files if m in FOLDS})
    for s in seeds:
        have = {g: files[(g, s, g)] for g in FOLDS if (g, s, g) in files}
        out["seeds"][str(s)] = {"folds_scored": sorted(have), "complete": all(complete(f) for f in have.values())}
        if len(have) < len(FOLDS):
            continue
        pool = {g: list(f["windows"].values()) for g, f in have.items()}
        allw = [w for v in pool.values() for w in v]
        r = rows(allw)
        point = diffs(allw)
        wiv = boot({f"{g}/{i}": [w] for g, v in pool.items() for i, w in enumerate(v)})
        giv = boot(pool)
        v = d5_verdict(point, wiv, r["coverage"], giv)
        partial = not out["seeds"][str(s)]["complete"]
        if partial:
            v = {**v, "adopt": False, "partial": True}
        out["pools"][str(s)] = {"games": list(pool), "rows": {k: {"r23": a, "model": b} for k, (a, b) in r.items()},
                                "diff": point, "window_interval": wiv, "game_interval": giv, "verdict": v}
    full = [s for s, p in out["pools"].items() if not p["verdict"].get("partial")]
    out["verdict"] = {"seeds_scored": f"{len(full)} of {SEEDS_NEEDED} seeds",
                      "adopt": len(full) >= SEEDS_NEEDED and all(out["pools"][s]["verdict"]["adopt"] for s in full)}
    return out


def cmd_report(args) -> dict:
    d = p8_root() / "e1"
    guard(list(FOLDS + APART), WHO)
    models = args.models.split(",") if args.models else list(FOLDS)
    files = {}
    for m in models:
        for g in scored_games(m):
            for p in sorted(d.glob(f"{m}__seed*__{g}.json")):
                f = json.loads(p.read_text())
                files[(m, int(f["seed"]), g)] = f
    res = report(files)
    atomic_write(d / ("e1_result.json" if models == list(FOLDS) else f"e1_result__{'_'.join(models)}.json"),
                 json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e1_score")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--model", required=True, choices=MODELS)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--game", required=True, choices=FOLDS + APART)
    r.add_argument("--windows", type=int, default=None)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--game", required=True, choices=FOLDS + APART)
    p = sub.add_parser("report")
    p.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    {"rebuild": cmd_rebuild, "score": cmd_score, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
