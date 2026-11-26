"""tools/p8_e1r.py: E1R, E1 seed 0 against r23 on the reference platform (docs/research/P8_E1R.md version 2; PLAN.md
D175). Evaluation only; nothing trains. No build runs before the lead confirms the design.

    <frozen CV python> -m tools.p8_e1r rebuild --arm e1 --game G [--fold F] [--smoke]       inside the heavy-job lock
    <frozen CV python> -m tools.p8_e1r rebuild --arm r23 --game gsw_cha_2016 --smoke        smoke gate 1 only
    python3 -m tools.p8_e1r smoke-check                                                     smoke gate 1 (byte identity)
    <frozen CV python> -m tools.p8_e1r score --game G [--fold F] [--smoke]                  inside the heavy-job lock
    python3 -m tools.p8_e1r regress                                                         smoke gate 4 (M6 record)
    python3 -m tools.p8_e1r report                                                          -> e1r/p8_e1r.json
    (internal) <python> -m tools.p8_e1r child build_trajectories.py <its arguments>

Arms: r23 resolves only to builds/e1m6_r23_detcpu (gsw_bkn_2015) and builds/m9_r23_detcpu__<game>; E1 to the fold
named after the game it left out (builds/e1m6_e1_gsw_bkn_2015_detcpu, builds/e1r_e1_<game>_detcpu), and on
hou_cle_2016 to builds/e1r_e1_<fold>_detcpu__hou_cle_2016. Score files: <P8 workspace>/e1r/<fold>__seed0__<game>.json,
in the registered score-file format, read by tools.p8_e1_score.report() called unchanged; tools.p8_e1_score report
itself reads only e1/, never e1r/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1r"
POOL = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
BLIND = ("gsw_cha_2016", "gsw_sac_2015")
APART = "hou_cle_2016"
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
E1_SHA_PREFIX = {"gsw_bkn_2015": "d4ffed33564b", "gsw_cha_2016": "0579a034", "gsw_sac_2015": "bdad8b76"}
SMOKE_GAME, SMOKE_WINDOW = "gsw_cha_2016", "gsw_cha_2016_s00_w00"
SMOKE_DIR = "e1r_smoke_r23__gsw_cha_2016"
SMOKE_FILES = ("frames", "trajectories", "identity")
M6_RECORD = "p8_e1_m6.json"
BUILD_SCRIPT = "build_trajectories.py"
ENV, ENV_MODEL, ENV_FOLD = "P8_E1R_DETECTOR", "P8_E1R_MODEL", "P8_E1R_FOLD"
MISS_FT = 2.5
REQUESTED = {"detector": "cpu", "court": "auto", "gate": "auto"}       # the reference platform (D175)
STAGE2_PASS = "seed 0 passes on the reference platform, including the new arena"
STAGE2_FAIL = "the pass does not carry to the new arena"


def check_game(game: str) -> str:
    if game not in POOL + (APART,):
        raise SystemExit(f"{WHO}: {game} is not an E1R game")
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    return game


# ── arms and folders (pure) ────────────────────────────────────────────────────────────────────────────────────────
def fold_for(game: str, fold: str | None = None) -> str:
    """The E1 fold scored on a game: on a pool game the fold named after it (never a fold trained on it); on
    hou_cle_2016 the fold must be given."""
    if game in POOL:
        if fold not in (None, game):
            raise SystemExit(f"{WHO}: on {game} only fold {game} is scored (it left {game} out)")
        return game
    if game == APART:
        if fold not in POOL:
            raise SystemExit(f"{WHO}: on {APART} name the fold (one of {POOL})")
        return fold
    raise SystemExit(f"{WHO}: {game} is not an E1R game")


def r23_dir(game: str) -> str:
    """The r23 arm on the reference platform; never builds/r23."""
    if game == "gsw_bkn_2015":
        return "e1m6_r23_detcpu"
    if game in ("gsw_cha_2016", "gsw_sac_2015", APART):
        return f"m9_r23_detcpu__{game}"
    raise SystemExit(f"{WHO}: no r23 CPU build for {game}")


def e1_dir(game: str, fold: str | None = None) -> str:
    f = fold_for(game, fold)
    if game == "gsw_bkn_2015":
        return "e1m6_e1_gsw_bkn_2015_detcpu"
    if game in POOL:
        return f"e1r_e1_{game}_detcpu"
    return f"e1r_e1_{f}_detcpu__{APART}"


BUILDABLE = (SMOKE_DIR, "e1r_e1_gsw_cha_2016_detcpu", "e1r_e1_gsw_sac_2015_detcpu",
             *(f"e1r_e1_{f}_detcpu__{APART}" for f in POOL))


def build_folder(arm: str, game: str, fold: str | None, smoke: bool) -> str:
    """The only folders E1R writes: the r23 smoke folder and E1R's own E1 folders."""
    if arm == "r23":
        if not smoke or game != SMOKE_GAME:
            raise SystemExit(f"{WHO}: the r23 arm is built only for smoke gate 1 (on {SMOKE_GAME})")
        name = SMOKE_DIR
    elif arm == "e1":
        name = e1_dir(game, fold)
    else:
        raise SystemExit(f"{WHO}: unknown arm {arm}")
    if name not in BUILDABLE:
        raise SystemExit(f"{WHO}: {name} is not an E1R build folder (the M6 folders are never rebuilt)")
    return name


def score_name(game: str, fold: str | None = None, smoke: bool = False) -> str:
    return f"{fold_for(game, fold)}__seed0__{game}{'__smoke' if smoke else ''}.json"


# ── per-window checks (pure on parsed files) ──────────────────────────────────────────────────────────────────────
def check_records(arm: str, det: dict | None, dev: dict | None, want_sha: str, want_fold: str | None,
                  need_fold: bool) -> list:
    """Problems with one window's detector and device records -> [] when it passes."""
    bad = []
    if dev is None or (dev.get("resolved") or {}).get("detector") != "cpu":
        bad.append(f"{arm}: detector device is not cpu")
    if dev is not None and dev.get("requested") != REQUESTED:
        bad.append(f"{arm}: requested devices are {dev.get('requested')}, not {REQUESTED}")
    if dev is not None and dev.get("clip_fix"):
        bad.append(f"{arm}: built with the MPS clip fix")
    if det is None or det.get("sha256") != want_sha:
        bad.append(f"{arm}: weights sha256 is not the expected one")
    if need_fold and (det or {}).get("fold") != want_fold:
        bad.append(f"{arm}: recorded fold is not {want_fold}")
    if det is not None and "fold" in det and det["fold"] != want_fold:
        bad.append(f"{arm}: recorded fold {det['fold']} is not {want_fold}")
    return bad


def check_frames(side_a: dict, side_b: dict) -> list:
    """Equal processed frame sets and an identical court H in every frame -> [] when they agree."""
    fa = {int(r["frame"]): r.get("H") for r in side_a["frames"]}
    fb = {int(r["frame"]): r.get("H") for r in side_b["frames"]}
    if fa.keys() != fb.keys():
        return [f"frame sets differ ({len(fa)} against {len(fb)} frames)"]
    diff = [f for f in fa if fa[f] != fb[f]]
    return [f"court H differs on {len(diff)} frames"] if diff else []


def duplicates(feet_ft, sv_xy, inside) -> int:
    """In-frame players with two or more box feet within MISS_FT."""
    sv = np.asarray(sv_xy, np.float64).reshape(-1, 2)
    ct = np.asarray(feet_ft, np.float64).reshape(-1, 2)
    if not len(ct):
        return 0
    D = np.sqrt(((sv[:, None, :] - ct[None, :, :]) ** 2).sum(-1))
    return int((np.asarray(inside, bool) & ((D <= MISS_FT).sum(1) >= 2)).sum())


def reading(primary: bool, blind: bool) -> str:
    """The four readings fixed in the design."""
    if primary and blind:
        return "seed 0 passes on the reference platform"
    if primary:
        return "fail: the pass rests on the game already seen and is not supported on the blind games"
    if blind:
        return "fail: the blind games alone do not override the registered pool"
    return "fail: E1 as built is closed on both platforms"


def stage_allowed(stage: str, gates: dict) -> list:
    """Why a stage may not start yet -> [] when it may. gates holds the parsed e1r/ files (None when absent):
    smoke_gate1, smoke_gate4, smoke_score, report."""
    if stage == "smoke":
        return []
    missing = []
    for k in ("smoke_gate1", "smoke_gate4"):
        if not (gates.get(k) or {}).get("passes"):
            missing.append(f"{k} has not passed")
    if gates.get("smoke_score") is None:
        missing.append("smoke gate 3 (the smoke score) has not passed")
    if stage == "stage1":
        return missing
    if stage == "stage2":
        rep = gates.get("report") or {}
        if not (rep.get("primary_pass") and (rep.get("blind") or {}).get("pass")):
            missing.append("Stage 2 runs only when the primary and the blind verdicts both pass")
        return missing
    raise SystemExit(f"{WHO}: unknown stage {stage}")


def stage_of(game: str, smoke: bool) -> str:
    return "smoke" if smoke else ("stage2" if game == APART else "stage1")


def load_gates(d: Path) -> dict:
    def read(name):
        p = d / name
        return json.loads(p.read_text()) if p.exists() else None
    return {"smoke_gate1": read("smoke_gate1.json"), "smoke_gate4": read("smoke_gate4.json"),
            "smoke_score": read(score_name(SMOKE_GAME, smoke=True)), "report": read("p8_e1r.json")}


def stage2_block(claim_passed: bool, fold_verdicts: dict) -> dict:
    """Stage 2 is reported only when the claim passed, with the design's fixed wording."""
    if not claim_passed:
        return {"runs": False, "reason": "Stage 2 runs only when the primary and the blind verdicts both pass"}
    if len(fold_verdicts) < 3:
        return {"runs": True, "folds_scored": sorted(fold_verdicts), "all_three_pass": None, "reading": None}
    ok = stage2_passes(fold_verdicts)
    return {"runs": True, "folds_scored": sorted(fold_verdicts), "all_three_pass": ok,
            "reading": STAGE2_PASS if ok else STAGE2_FAIL}


def stage2_passes(fold_verdicts: dict) -> bool:
    """All 3 folds must pass on hou_cle_2016 (fixed in the design)."""
    return set(fold_verdicts) == set(POOL) and all(bool(v) for v in fold_verdicts.values())


def window_pool_verdict(wins: list) -> dict:
    """d5_verdict called unchanged with the window interval only (the blind pool and each hou_cle_2016 fold)."""
    from tools.p8_e1_score import boot, d5_verdict, diffs, rows
    point = diffs(wins)
    wiv = boot({i: [w] for i, w in enumerate(wins)})
    return d5_verdict(point, wiv, rows(wins)["coverage"], None)


def rows_equal(a: dict, b: dict, tol: float = 1e-12) -> list:
    """Rows of two reports {row: {r23, e1}} that differ beyond tol -> [] when all agree."""
    bad = []
    for k in a:
        for side in ("r23", "e1"):
            x, y = a[k][side], b[k][side]
            if (x is None) != (y is None) or (x is not None and abs(x - y) > tol):
                bad.append(f"{k} {side}: {x} against {y}")
    return bad


# ── build (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_sha(fold: str) -> str:
    from tools.p8_e1_m4 import weights_of
    w = weights_of(fold)
    return json.loads((w.parents[2] / "complete.json").read_text())["weights"][w.name]


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
        return [cmd[0], "-m", "tools.p8_e1r", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    from tools import p8_device as D
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m tools.p8_e1r child {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    req = D.requested()
    if req["detector"] != "cpu" or req["court"] != "auto" or req["gate"] != "auto" or D.clip_fix_requested():
        raise SystemExit(f"{WHO}: the reference platform is P8_DEVICE_DETECTOR=cpu only")
    resolved = D.apply(req)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    for suffix, rec in (("device", D.device_record(req, resolved)),
                        ("detector", {"weights": weights.name, "sha256": sha(weights), "model": os.environ[ENV_MODEL],
                                      "fold": os.environ.get(ENV_FOLD) or None, "seed": 0})):
        part = config.TRACKING_DIR / f"{stem}_{suffix}.json.part"
        part.write_text(json.dumps(rec))
        os.replace(part, config.TRACKING_DIR / f"{stem}_{suffix}.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_rebuild(args) -> dict:
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_frozen as F
    from tools.p8_e1_m4 import weights_of
    from tools.p8_e1_score import check_provenance
    game = check_game(args.game)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    for k in ("P8_DEVICE", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "P8_MPS_CLIP_FIX"):
        os.environ.pop(k, None)
    os.environ["P8_DEVICE_DETECTOR"] = "cpu"
    name = build_folder(args.arm, game, args.fold, args.smoke)
    why = stage_allowed(stage_of(game, args.smoke), load_gates(p8_root() / "e1r"))
    if why:
        raise SystemExit(f"{WHO}: {stage_of(game, args.smoke)} may not start: {why}")
    if args.smoke and (args.arm, game) not in (("r23", SMOKE_GAME), ("e1", SMOKE_GAME)):
        raise SystemExit(f"{WHO}: the smoke stage builds only {SMOKE_WINDOW}")
    if args.arm == "e1":
        fold = fold_for(game, args.fold)
        weights = weights_of(fold).resolve()
        want = check_provenance(fold, game, 0, weights)["sha256"]       # the fold that left this game out
        os.environ[ENV_FOLD] = fold
    else:
        fold = None
        weights = (main_checkout() / "models" / "player_detector.pt").resolve()
        want = sha(weights)
        if want != R23_SHA:
            raise SystemExit(f"{WHO}: the production detector is not r23")
        os.environ.pop(ENV_FOLD, None)
    P = F.install_paths()
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV], os.environ[ENV_MODEL] = str(weights), args.arm
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / name
    out.mkdir(parents=True, exist_ok=True)
    wins = [w for w in F.label_windows(P, game) if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    if args.smoke:
        wins = [w for w in wins if w["window"] == SMOKE_WINDOW]
    res = {"arm": args.arm, "game": game, "fold": fold, "folder": name, "weights_sha256": want, "windows": {}}
    for w in wins:
        s = w["window"]
        det_p = out / f"{s}_detector.json"
        if (out / f"{s}_frames.json").exists() and (not det_p.exists() or json.loads(det_p.read_text())["sha256"] != want):
            raise SystemExit(f"{WHO}: {s} in {out} was built with other weights")
        t0 = time.time()
        r = F.safe_build(P, w, P["label_build"] / f"{s}.mp4", out, extra_args=gate_args())
        if not r["ok"]:
            r = F.safe_build(P, w, P["label_build"] / f"{s}.mp4", out, extra_args=gate_args())   # one retry
            if not r["ok"]:
                raise SystemExit(f"{WHO}: build of {s} failed twice: {r.get('tail')}")
        if not r.get("empty"):
            det = json.loads(det_p.read_text())
            dev = json.loads((out / f"{s}_device.json").read_text())
            bad = check_records(args.arm, det, dev, want, fold, need_fold=args.arm == "e1")
            if bad:
                raise SystemExit(f"{WHO}: {s}: {bad}")
        res["windows"][s] = {"cached": bool(r.get("cached")), "empty": bool(r.get("empty")),
                             "seconds": None if r.get("cached") else round(time.time() - t0, 1)}
    d = p8_root() / "e1r"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"rebuild__{name}{'__smoke' if args.smoke else ''}.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def cmd_smoke_check(args) -> dict:
    """Smoke gate 1: the r23 window rebuilt through the E1R path is byte-identical to M9's."""
    b = p8_root() / "builds"
    out = {f: sha(b / SMOKE_DIR / f"{SMOKE_WINDOW}_{f}.json") == sha(b / r23_dir(SMOKE_GAME) / f"{SMOKE_WINDOW}_{f}.json")
           for f in SMOKE_FILES}
    out = {"window": SMOKE_WINDOW, "byte_identical": out, "passes": all(out.values())}
    atomic_write(p8_root() / "e1r" / "smoke_gate1.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if not out["passes"]:
        raise SystemExit(f"{WHO}: smoke gate 1 fails: CPU builds do not repeat")
    return out


# ── score (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def cmd_score(args) -> dict:
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    from tools.p8_e1_score import _paired, window_stats
    import cv2
    game = check_game(args.game)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    fold = fold_for(game, args.fold)
    want = {"r23": R23_SHA, "new": complete_sha(fold)}
    if not want["new"].startswith(E1_SHA_PREFIX[fold]):
        raise SystemExit(f"{WHO}: {fold}'s complete.json does not hold the expected weights")
    g = F.game_of(game)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, game)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    dirs = {"r23": P["builds"] / r23_dir(game), "new": P["builds"] / e1_dir(game, args.fold)}
    own_e1 = e1_dir(game, args.fold) in BUILDABLE
    t0 = time.time()
    all_w = F.label_windows(P, game)
    expected = [w["window"] for w in all_w if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    if args.smoke:
        expected = [s for s in expected if s == SMOKE_WINDOW]
    wins, gate_empty, gate_mismatch, missing = {}, [], [], []
    counts = {s: {"in_frame": 0, "nr_boxes": 0, "duplicates": 0, "team_n": 0} for s in dirs}
    matched_pairs = 0
    from sportvu.sync import MIRRORS, apply_mirror
    fx, fy = MIRRORS[mirror]
    for w in all_w:
        s = w["window"]
        if s not in expected:
            continue
        empty = {k: F.gate_empty_path(d, s).exists() for k, d in dirs.items()}
        if any(empty.values()):
            (gate_empty if all(empty.values()) else gate_mismatch).append(s)
            continue
        if not all((d / f"{s}_frames.json").exists() for d in dirs.values()):
            missing.append(s)
            continue
        if not (P["cache"] / "scorecard" / f"{s}.json").exists():
            raise SystemExit(f"{WHO}: {s} has no cached scorecard references; refusing to compute them")
        sides = {k: json.loads((d / f"{s}_frames.json").read_text()) for k, d in dirs.items()}
        bad = []
        for k, d in dirs.items():
            det_p, dev_p = d / f"{s}_detector.json", d / f"{s}_device.json"
            det = json.loads(det_p.read_text()) if det_p.exists() else None
            dev = json.loads(dev_p.read_text()) if dev_p.exists() else None
            bad += check_records(k, det, dev, want[k], fold if k == "new" else None, need_fold=(k == "new" and own_e1))
        bad += check_frames(sides["r23"], sides["new"])
        if bad:
            raise SystemExit(f"{WHO}: {s}: {bad}")
        pair, scored_f = {}, {}
        for k, d in dirs.items():
            full = S.score_window(P, game, w, d, "camera", mom, mirror, refs, events)
            rec = full[0]
            if "skipped" in rec:
                raise SystemExit(f"{WHO}: {s} {k} skipped: {rec['skipped']}")
            scored_f[k] = {int(o["f"]) for o in full[1]}
            pair[k] = window_stats(rec)
            counts[k]["in_frame"] += pair[k]["in_frame"]
            counts[k]["nr_boxes"] += pair[k]["nr_boxes"]
            counts[k]["team_n"] += pair[k]["team"]["n"]
        if pair["r23"]["in_frame"] != pair["new"]["in_frame"]:
            raise SystemExit(f"{WHO}: {s}: in-frame players differ between arms "
                             f"({pair['r23']['in_frame']} against {pair['new']['in_frame']})")
        truth, _ = S.truth_rows(P, game, w, "camera", mom.index, mirror)   # descriptive; referee boxes included
        cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4"))
        wd, hd = int(cap.get(3)), int(cap.get(4))
        cap.release()
        for k in dirs:
            srow = {int(r["frame"]): r for r in sides[k]["frames"]}
            for row in truth:
                r = srow.get(int(row["frame"]))
                if r is None or int(row["frame"]) not in scored_f[k]:     # scored live-wide frames only
                    continue
                sv = apply_mirror(mom.index.q[row["q"]]["xy"][row["moment"]], fx, fy)
                Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
                px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
                m = FRAME_MARGIN_PX
                inside = (px[:, 0] > m) & (px[:, 0] < wd - m) & (px[:, 1] > m) & (px[:, 1] < hd - m)
                feet = np.array([b["foot_stab"] for b in r["boxes"]], np.float64).reshape(-1, 1, 2)
                ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2) if len(feet) else np.zeros((0, 2))
                counts[k]["duplicates"] += duplicates(ft, sv, inside)
        matched_pairs += len(_paired([pair])[0])
        wins[s] = pair
    out = {"model": fold, "seed": 0, "game": game, "truth": "camera", "weights_sha256": want["new"],
           "platform": "detector cpu (D175)", "arms": {k: str(d.name) for k, d in dirs.items()},
           "expected_windows": expected, "scored_windows": sorted(wins), "gate_empty_windows": gate_empty,
           "gate_empty_mismatch": gate_mismatch, "missing_windows": missing, "frame_sets_differ": {},
           "windows": wins, "counts": counts, "paired_matched_players": matched_pairs,
           "smoke": bool(args.smoke), "seconds": round(time.time() - t0, 1)}
    d = p8_root() / "e1r"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / score_name(game, args.fold, args.smoke), json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k not in ("windows", "expected_windows", "scored_windows")},
                     indent=1))
    return out


# ── regression and report (system python) ─────────────────────────────────────────────────────────────────────────
def cmd_regress(args) -> dict:
    """Smoke gate 4: the E1R score of M6's gsw_bkn_2015 folders reproduces reports/research/p8_e1_m6.json exactly."""
    from tools.p8_common import REPO
    from tools.p8_e1_score import rows
    f = json.loads((p8_root() / "e1r" / score_name("gsw_bkn_2015")).read_text())
    rec = json.loads((REPO / "reports" / "research" / M6_RECORD).read_text())
    got = {k: {"r23": a, "e1": b} for k, (a, b) in rows(list(f["windows"].values())).items()}
    bad = rows_equal(rec["rows"], got)
    if len(f["windows"]) != rec["windows"]:
        bad.append(f"windows {len(f['windows'])} against {rec['windows']}")
    out = {"windows": len(f["windows"]), "missed": got["missed_rate"], "passes": not bad, "differences": bad}
    atomic_write(p8_root() / "e1r" / "smoke_gate4.json", json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    if bad:
        raise SystemExit(f"{WHO}: smoke gate 4 fails: {bad[:3]}")
    return out


def cmd_report(args) -> dict:
    from tools.p8_e1_score import complete, report
    for g in POOL + (APART,):
        check_game(g)
    d = p8_root() / "e1r"
    files = {}
    for g in POOL:
        p = d / score_name(g)
        if not p.exists():
            raise SystemExit(f"{WHO}: {g} has not been scored")
        files[(g, 0, g)] = json.loads(p.read_text())      # unscored windows (gate mismatch, missing) keep it partial
    apart = {}
    for fold in POOL:
        p = d / score_name(APART, fold)
        if p.exists():
            files[(fold, 0, APART)] = json.loads(p.read_text())
    reg = report(files)                                                  # the registered report, unchanged
    primary = reg["pools"].get("0", {}).get("verdict", {})
    primary_pass = bool(primary.get("adopt"))
    blind_files = [files[(g, 0, g)] for g in BLIND]
    blind_wins = [w for f in blind_files for w in f["windows"].values()]
    blind_v = window_pool_verdict(blind_wins)
    blind_pass = bool(blind_v["adopt"]) and all(complete(f) for f in blind_files)
    for fold in POOL:
        if (fold, 0, APART) in files:
            f = files[(fold, 0, APART)]
            v = window_pool_verdict(list(f["windows"].values()))
            apart[fold] = {"adopt": bool(v["adopt"]) and complete(f), "verdict": v}
    claim_passed = primary_pass and blind_pass
    failing = lambda v: sorted(k for k, c in (v or {}).get("rows", {}).items() if not c.get("pass"))
    out = {"what": "E1R (docs/research/P8_E1R.md version 2): E1 seed 0 against r23 on the reference platform; "
                   "evaluation only", "registered_report": reg, "primary_pass": primary_pass,
           "primary_failing_rows": failing(primary), "partial": bool(primary.get("partial")),
           "blind": {"games": list(BLIND), "windows": len(blind_wins), "verdict": blind_v, "pass": blind_pass,
                     "failing_rows": failing(blind_v)},
           "claim": reading(primary_pass, blind_pass),
           "stage2": stage2_block(claim_passed, {k: v["adopt"] for k, v in apart.items()}),
           "stage2_folds": apart if claim_passed else {},
           "counts": {f"{g}/{f}": v.get("counts") for (f, _s, g), v in files.items()}}
    atomic_write(d / "p8_e1r.json", json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("primary_pass", "claim")} | {"blind_pass": blind_pass}, indent=1))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "child":
        return child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e1r")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebuild")
    r.add_argument("--arm", required=True, choices=("r23", "e1"))
    r.add_argument("--game", required=True, choices=POOL + (APART,))
    r.add_argument("--fold", default=None, choices=POOL)
    r.add_argument("--smoke", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--game", required=True, choices=POOL + (APART,))
    s.add_argument("--fold", default=None, choices=POOL)
    s.add_argument("--smoke", action="store_true")
    sub.add_parser("smoke-check")
    sub.add_parser("regress")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"rebuild": cmd_rebuild, "score": cmd_score, "smoke-check": cmd_smoke_check, "regress": cmd_regress,
     "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
