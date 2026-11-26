"""tools/p8_resetfix.py: the tracker reset fix (docs/research/P8_resetfix.md). This module holds the pure fix used by
tools/p8_device.py's build child and a stored-files estimate for the design. Nothing here builds or trains.

    python3 -m tools.p8_resetfix estimate   -> <P8 workspace>/placement/resetfix_estimate.json and
                                               reports/research/p8_resetfix_estimate.json   (stored files only, no lock)
    python3 -m tools.p8_resetfix truthreset -> reports/research/p8_resetfix_truthreset.json   (design v2 measurement a)
    python3 -m tools.p8_resetfix simulate   -> reports/research/p8_resetfix_simulate.json     (design v2 measurement b)

Design v2 stage 0 (docs/research/P8_resetfix.md section 14), in the reference root (V3_P8_ROOT=<reference root>):
    <frozen CV python> -m tools.p8_resetfix stage0-check                 the real ultralytics check (no lock, no model)
    python3 -m tools.p8_resetfix baseline-copy --game G                  the unfixed baseline in builds/r23_nofix
    python3 -m tools.p8_resetfix plan --game G                           reset and equivalence windows from the baseline
    <frozen CV python> -m tools.p8_resetfix build --game gsw_cha_2016 --smoke         inside the heavy-job lock
    python3 -m tools.p8_resetfix smoke-check                             the smoke gate
    <frozen CV python> -m tools.p8_resetfix build --game G               inside the lock; the planned windows
    python3 -m tools.p8_resetfix equiv --game G                          equivalence and prefix checks
    python3 -m tools.p8_resetfix pin-truth --game G                      pin a final truth (sha256 of its files)
    python3 -m tools.p8_resetfix cache-copy --game G                     scorecard caches from the workspace (same clip)
    <frozen CV python> -m tools.p8_resetfix score --game G               inside the lock; every pool truth pinned
    python3 -m tools.p8_resetfix report                                  the v2 rule -> reports/research/p8_resetfix.json
    (internal) <python> -m tools.p8_resetfix child build_trajectories.py <its arguments>

The bug (docs/research/P8_a3input.md, cause confirmed): detect/tracker.py PlayerTracker.reset sets
self.model.predictor = None; ultralytics 8.4.48 Model.track then re-registers the tracker callbacks
(ultralytics.trackers.track.on_predict_start and on_predict_postprocess_end, as functools.partial objects) and
Model.add_callback appends, so after a reset the tracker updates twice per frame and new detections never confirm.

The fix: after PlayerTracker.reset, remove every tracker callback from the model's callback lists, so Model.track
registers them exactly once again. Applied only in the build child when P8_TRACKER_RESET_FIX=1 (tools/p8_device.py).

Estimate rules (stored files only; the r23 reference-platform sidecars, identity files, the scorer universe, the E1
corpus "ours" boxes on the decoded clip frames):
- Per window: identity["camera_cut_resets"] (cut and exception resets together) and the first reset-like frame of the
  sidecar (80 percent or more of 4 or more previous ids absent; tools.p8_trackreset.reset_like); processed frames from
  that frame on are "after a reset".
- Per pool game, on corpus frames of the scorer's universe: in-frame players; missed (no foot_stab within 2.5 ft);
  raw-reachable misses (missed, and a corpus "ours" box at 0.40 has its bottom-centre foot within 2.5 ft); the same
  split by before or after the window's first reset; far misses (no foot_stab within 5 ft) and those an "ours" box
  reaches within 5 ft; ghost rate of the sidecar's boxes and of the "ours" boxes (more than 3.0 ft from all 10
  players, all boxes) on the same frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
TRACKER_EVENTS = ("on_predict_start", "on_predict_postprocess_end")
TRACKER_MODULE = "ultralytics.trackers.track"
WHO = "tools.p8_resetfix"


# ── the fix (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────
def is_tracker_callback(c) -> bool:
    """A functools.partial of ultralytics.trackers.track.on_predict_start or on_predict_postprocess_end."""
    f = getattr(c, "func", None)
    return f is not None and getattr(f, "__module__", "") == TRACKER_MODULE and \
        getattr(f, "__name__", "") in TRACKER_EVENTS


def strip_tracker_callbacks(callbacks: dict) -> int:
    """Remove every tracker callback from a model's callback lists, in place -> the number removed."""
    n = 0
    for ev in TRACKER_EVENTS:
        keep = [c for c in callbacks.get(ev, []) if not is_tracker_callback(c)]
        n += len(callbacks.get(ev, [])) - len(keep)
        if ev in callbacks:
            callbacks[ev] = keep
    return n


RESET_LOG = {"path": None, "resets": [], "last_frame": None}


def apply_reset_fix() -> str:
    """In the build child, before the build runs: PlayerTracker.reset also strips the tracker callbacks. Each reset
    appends {"frame": the frame index of the last update call, "stripped": n} to RESET_LOG (n = 2 when the tracker had
    registered once: one on_predict_start and one on_predict_postprocess_end); write_reset_log() saves it at exit when
    start_reset_log() named a file. The frame is the frame whose update triggered the reset (a cut, or an exception)."""
    import detect.tracker as DT
    orig = DT.PlayerTracker.reset

    def reset(self):
        orig(self)
        cbs = getattr(getattr(self, "model", None), "callbacks", None)
        RESET_LOG["resets"].append({"frame": RESET_LOG.get("last_frame"),
                                    "stripped": strip_tracker_callbacks(cbs) if isinstance(cbs, dict) else None})
    DT.PlayerTracker.reset = reset
    if hasattr(DT.PlayerTracker, "update"):
        orig_update = DT.PlayerTracker.update

        def update(self, bgr, frame_idx: int = 0, *a, **k):
            RESET_LOG["last_frame"] = int(frame_idx)
            return orig_update(self, bgr, frame_idx, *a, **k)
        DT.PlayerTracker.update = update
    return "detect.tracker.PlayerTracker.reset"


def library_versions() -> dict:
    """Versions of the libraries the build process loaded (None for one it did not load); nothing is imported."""
    import sys
    return {name: getattr(sys.modules.get(mod), "__version__", None)
            for name, mod in (("opencv", "cv2"), ("torch", "torch"), ("ultralytics", "ultralytics"))}


def write_reset_log() -> None:
    import os
    path = RESET_LOG["path"]
    if path is None:
        return
    try:
        import config
        tracker = Path(str(config.TRACKER_CONFIG)).name
    except Exception:                       # noqa: BLE001
        tracker = None
    part = f"{path}.part"
    with open(part, "w") as fh:
        json.dump({"reset_fix": True, "resets": [{k: v for k, v in r.items()} for r in RESET_LOG["resets"]],
                   "tracker_reid_env": os.environ.get("TRACKER_REID"), "tracker_config": tracker,
                   "versions": library_versions()}, fh)
    os.replace(part, path)


def start_reset_log(path) -> None:
    """<window>_resetfix.json in config.TRACKING_DIR (sportvu.rebuild.build moves every <window>_* file next to the
    window's outputs), written when the build child exits."""
    import atexit
    RESET_LOG["path"] = str(path)
    atexit.register(write_reset_log)


# ── estimate (stored files only) ───────────────────────────────────────────────────────────────────────────────────
def first_reset_index(frames: list) -> int | None:
    from tools.p8_trackreset import reset_like
    ids = [{b["tid"] for b in r["boxes"]} for r in frames]
    for i in range(1, len(ids)):
        if reset_like(ids[i - 1], ids[i]):
            return i
    return None


def frame_counts(feet, sv, inside, is_ref, ours_ft) -> dict:
    """One scored frame -> counts. feet: the sidecar's foot_stab in truth court ft (scorer_court of px32); ours_ft: the
    r23 corpus player boxes at 0.40 (bottom centres through H_truth); sv: the 10 players; inside: in frame; is_ref: the
    scorer's referee flag per sidecar box."""
    from tools.p8_footbias import score_frame
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ours_ft = np.asarray(ours_ft, np.float64).reshape(-1, 2)
    is_ref = np.asarray(is_ref, bool)
    sc = score_frame(feet, sv, inside, is_ref)
    c = {"in_frame": sc["in_frame"], "missed": sc["missed"], "nr_boxes": sc["nr_boxes"], "nr_ghosts": sc["nr_ghosts"],
         "reach": 0, "far": 0, "far_reach": 0, "ours_boxes": len(ours_ft), "ours_ghosts": 0,
         "side_boxes_all": len(feet), "side_ghosts_all": 0}
    db = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1)).min(1) if len(feet) else np.full(len(sv), np.inf)
    do = np.sqrt(((sv[:, None, :] - ours_ft[None, :, :]) ** 2).sum(-1)).min(1) if len(ours_ft) else \
        np.full(len(sv), np.inf)
    ins = np.asarray(inside, bool)
    c["reach"] = int((ins & (db > 2.5) & (do <= 2.5)).sum())
    c["far"] = int((ins & (db > 5.0)).sum())
    c["far_reach"] = int((ins & (db > 5.0) & (do <= 5.0)).sum())
    if len(ours_ft):
        c["ours_ghosts"] = int((np.sqrt(((sv[:, None, :] - ours_ft[None, :, :]) ** 2).sum(-1)).min(0) > 3.0).sum())
    if len(feet):
        c["side_ghosts_all"] = int((np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1)).min(0) > 3.0).sum())
    return c


def _add(a: dict, b: dict) -> None:
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def window_counts(game: str, wr=None) -> tuple:
    """Per scored window of a pool game on the registered scored frames (the MPS-truth universe of the score file):
    {window: {"resets", "first_reset_frame", "all": counts, "after": counts}} and the corpus frame count. The scorer's
    own counts (in_frame, missed, nr_ghosts, nr_boxes) are checked against the score file per window."""
    from tools import p8_teachers as T
    from tools.p8_common import p8_root
    from tools.p8_e1data import corpus_root
    from tools.p8_footbias import load_frames, scorer_court
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS, SCORES, SIDE
    wr = wr or p8_root()
    P = layout(wr)
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    ours = {}
    for k in range(len(plan["chunks"])):
        for key, rec in json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"].items():
            g, stem = key.split("/", 1)
            if g == game:
                ours[stem] = [p for p in rec[T.OURS] if p[5] == "player" and p[4] >= 0.40]
    sf = json.loads((wr / "e1" / SCORES[game]).read_text())
    out = {}
    for w in sf["scored_windows"]:
        fr = sorted(json.loads((P["builds"] / BUILDS[game] / f"{w}_frames.json").read_text())["frames"],
                    key=lambda r: int(r["frame"]))
        ident = json.loads((P["builds"] / BUILDS[game] / f"{w}_identity.json").read_text())["identity"]
        i0 = first_reset_index(fr)
        out[w] = {"resets": int(ident.get("camera_cut_resets") or 0),
                  "first_reset_frame": None if i0 is None else int(fr[i0]["frame"]), "all": {}, "after": {}}
    game_frames = load_frames(game)
    n_corpus = 0
    for f in game_frames["frames"]:
        stem = f"{f['w']}_f{int(f['f']):06d}"
        if stem not in ours:
            raise SystemExit(f"{WHO}: scored frame {stem} has no corpus prediction")
        n_corpus += 1
        c = frame_counts(scorer_court(f["Ht"], f["px32"]), f["sv"], f["inside"], f["is_ref"],
                         T.feet_ft(ours[stem], f["Ht"]))
        rec = out[f["w"]]
        _add(rec["all"], c)
        if rec["first_reset_frame"] is not None and int(f["f"]) >= rec["first_reset_frame"]:
            _add(rec["after"], c)
    side = SIDE[game]
    for w, rec in out.items():
        reg = sf["windows"][w][side]
        got = {k: rec["all"].get(k, 0) for k in ("in_frame", "missed", "nr_ghosts", "nr_boxes")}
        if got != {k: reg[k] for k in got}:
            raise SystemExit(f"{WHO}: {w} counts {got} differ from the score file's")
    return out, n_corpus


def cmd_estimate(args) -> dict:
    from tools.p8_common import REPO, atomic_write, p8_root
    from tools.p8_frozen import layout
    from tools.p8_placement import BUILDS, CHECK, POOL, SCORES, check_games, rnd
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(check_games(POOL + CHECK), WHO)
    wr = p8_root()
    P = layout(wr)
    out = {}
    for g in POOL + CHECK:
        if g in POOL:
            wc, n_corpus = window_counts(g, wr)
        else:
            sf = json.loads((wr / "e1" / SCORES[g]).read_text())
            wc, n_corpus = {}, 0
            for w in sf["scored_windows"]:
                fr = sorted(json.loads((P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())["frames"],
                            key=lambda r: int(r["frame"]))
                ident = json.loads((P["builds"] / BUILDS[g] / f"{w}_identity.json").read_text())["identity"]
                i0 = first_reset_index(fr)
                wc[w] = {"resets": int(ident.get("camera_cut_resets") or 0), "frames": len(fr),
                         "first_reset_frame": None if i0 is None else int(fr[i0]["frame"]),
                         "frames_after_first_reset": 0 if i0 is None else len(fr) - i0, "all": {}, "after": {}}
        for w, rec in wc.items():
            if "frames" not in rec:
                fr = json.loads((P["builds"] / BUILDS[g] / f"{w}_frames.json").read_text())["frames"]
                rec["frames"] = len(fr)
                rec["frames_after_first_reset"] = 0 if rec["first_reset_frame"] is None else \
                    sum(1 for r in fr if int(r["frame"]) >= rec["first_reset_frame"])
        A, B = {}, {}
        for rec in wc.values():
            _add(A, rec["all"])
            _add(B, rec["after"])
        n = A.get("in_frame") or None
        rates = None
        if n:
            rates = {"missed": A["missed"] / n, "raw_reachable_missed": None,
                     "raw_reachable_missed_after_reset": B.get("reach", 0) / n,
                     "missed_after_reset": B.get("missed", 0) / n,
                     "far_missed": A["far"] / n, "far_raw_reachable": A["far_reach"] / n,
                     "far_missed_after_reset": B.get("far", 0) / n,
                     "far_raw_reachable_after_reset": B.get("far_reach", 0) / n,
                     "in_frame_after_reset_share": B.get("in_frame", 0) / n,
                     "sidecar_ghost_rate_all_boxes": A["side_ghosts_all"] / A["side_boxes_all"],
                     "ours_ghost_rate_all_boxes": A["ours_ghosts"] / A["ours_boxes"],
                     "nr_ghost_rate": A["nr_ghosts"] / A["nr_boxes"],
                     "after_reset_nr_ghost_rate": B["nr_ghosts"] / B["nr_boxes"] if B.get("nr_boxes") else None,
                     "after_reset_ours_ghost_rate": B["ours_ghosts"] / B["ours_boxes"] if B.get("ours_boxes") else None,
                     "after_reset_excess_boxes": B.get("ours_boxes", 0) - B.get("nr_boxes", 0),
                     "after_reset_excess_ghost_share": excess_ghost_share(B),
                     "sidecar_boxes_per_in_frame": A["side_boxes_all"] / n, "ours_boxes_per_in_frame": A["ours_boxes"] / n}
            rates["raw_reachable_missed"] = A["reach"] / n
        out[g] = {"windows": len(wc), "windows_with_reset": sum(1 for v in wc.values() if v["resets"] > 0),
                  "resets_total": sum(v["resets"] for v in wc.values()),
                  "processed_frames": sum(v["frames"] for v in wc.values()),
                  "frames_in_reset_windows": sum(v["frames"] for v in wc.values() if v["resets"] > 0),
                  "frames_after_first_reset": sum(v["frames_after_first_reset"] for v in wc.values()),
                  "corpus_frames": n_corpus, "counts": {"all": A, "after_first_reset": B}, "rates": rates,
                  "per_window": wc}
        print(g, json.dumps(rnd({k: out[g][k] for k in ("windows", "windows_with_reset", "resets_total",
                                                        "processed_frames", "frames_after_first_reset", "rates")})))
    res = rnd({"what": "Reset-fix design estimate from stored files (docs/research/P8_resetfix.md); not a result",
               "games": out})
    atomic_write(wr / "placement" / "resetfix_estimate.json", json.dumps(res, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_resetfix_estimate.json", json.dumps(res, indent=1))
    return res


def excess_ghost_share(after: dict):
    """Ghost share of the boxes the fix may return after a reset: (r23 corpus ghosts - sidecar non-referee ghosts) over
    (r23 corpus boxes - sidecar non-referee boxes), after the first reset, clipped to [0, 1]; None without excess."""
    e = after.get("ours_boxes", 0) - after.get("nr_boxes", 0)
    if e <= 0:
        return None
    return float(min(1.0, max(0.0, (after.get("ours_ghosts", 0) - after.get("nr_ghosts", 0)) / e)))


# ── design v2: the rule (pure) ───────────────────────────────────────────────────────────────────────────────────
# rows judged by D158 "worse" and by "improves": missed and the detection row
TARGET_ROWS = ("missed_rate", "detection_rate")
# rows judged only by a non-inferiority margin on the bad-side end of their window interval (E1b's margins)
MARGINS = {"ghost_rate": ("hi", 0.010), "position_p50_ft": ("hi", 0.05), "position_p90_ft": ("hi", 0.10),
           "nearfield_bias_ft": ("hi", 0.10), "team_accuracy": ("lo", -0.03)}
ONE_TO_ONE_ROW = "missed_1to1_rate"
COVERAGE_FLOOR = 0.50


def v2_verdict(point: dict, window_iv: dict, coverage: tuple, game_iv: dict | None = None,
               improve_rows: tuple = TARGET_ROWS, one_to_one: bool = True) -> dict:
    """Design v2's rule for one unit (a game, or the pool). point: {row: arm minus baseline}; window_iv and game_iv:
    {row: [lo, hi]} or None; coverage: (baseline, arm).
    - TARGET_ROWS: worse (D158) when the window or the game interval lies wholly above zero; a row in improve_rows must
      also improve (point below zero).
    - MARGINS rows: judged only by the margin, on the bad-side end of the window interval; no interval fails.
    - missed_1to1_rate (when one_to_one): point below zero.
    - coverage: as D5 (the arm holds the 0.50 floor, or the baseline's value when the baseline is under it)."""
    rows, ok = {}, True
    for r in TARGET_ROWS:
        d = point.get(r)
        ivs = [window_iv.get(r), (game_iv or {}).get(r)]
        worse = any(bool(v) and v[0] > 0 for v in ivs)
        improves = d is not None and d < 0
        need = r in improve_rows
        p = (d is not None) and not worse and (improves or not need)
        rows[r] = {"diff": d, "window_interval": ivs[0], "game_interval": ivs[1], "worse": worse,
                   "improves": improves, "improve_required": need, "pass": p}
        ok &= p
    for r, (side, m) in MARGINS.items():
        iv = window_iv.get(r)
        end = None if not iv else (iv[1] if side == "hi" else iv[0])
        p = end is not None and (end <= m if side == "hi" else end >= m)
        rows[r] = {"diff": point.get(r), "window_interval": iv, "bad_end": end, "margin": m, "pass": p}
        ok &= p
    if one_to_one:
        d = point.get(ONE_TO_ONE_ROW)
        p = d is not None and d < 0
        rows[ONE_TO_ONE_ROW] = {"diff": d, "improves": p, "pass": p}
        ok &= p
    base, new = coverage
    cov = new is not None and (new >= COVERAGE_FLOOR if (base or 0) >= COVERAGE_FLOOR else new >= base)
    rows["coverage"] = {"baseline": base, "arm": new, "floor_holds": cov, "pass": cov}
    ok &= cov
    return {"pass": bool(ok), "rows": rows}


def pool_verdict(pooled: dict, per_game: dict) -> dict:
    """pooled: v2_verdict of the pool (improve_rows = TARGET_ROWS, one_to_one True); per_game: {game: v2_verdict with
    improve_rows=() and one_to_one False}. The pool passes when the pooled verdict and every game's pass."""
    return {"pass": bool(pooled["pass"] and all(v["pass"] for v in per_game.values())),
            "pooled": pooled["pass"], "games": {g: v["pass"] for g, v in per_game.items()},
            "failing": sorted({f"pool:{r}" for r, x in pooled["rows"].items() if not x["pass"]} |
                              {f"{g}:{r}" for g, v in per_game.items() for r, x in v["rows"].items() if not x["pass"]})}


def adoption(pv: dict, pooled: dict, bracket_ok: bool) -> dict:
    """Version 2.2 (the lead's decision): the fix is adopted when the pool passes and the bracket guard holds; or, as a
    correctness fix, when every failing condition is a per-game ghost margin and the pooled verdict and the bracket
    guard pass (the claim "no harm on ghosts" is then not made, and each game's ghost cost is reported). Not adopted
    when pooled missed fails, the bracket guard fails, any pooled margin fails, or anything else fails."""
    fails = set(pv.get("failing", []))
    ghost_only = bool(fails) and all(f.endswith(":ghost_rate") and not f.startswith("pool:") for f in fails)
    if pv.get("partial") or not bracket_ok or not pooled["pass"]:
        return {"adopt": False, "kind": None}
    if pv["pass"]:
        return {"adopt": True, "kind": "pass"}
    if ghost_only:
        return {"adopt": True, "kind": "correctness fix; per-game ghost margin failed, ghost cost reported",
                "ghost_failing_games": sorted(f.split(":")[0] for f in fails)}
    return {"adopt": False, "kind": None}


def scale_after(st: dict, after: dict, u: float) -> tuple:
    """Version 2.2 simulation: undo the truth thinning of part (a). A window's counts after its first reset are taken
    u times (the game's under-representation factor): the baseline gains (u - 1) times them, and the after-reset
    counts the arm acts on become u times them (rounded)."""
    if not after or u is None or u <= 1:
        return st, after
    st2 = dict(st)
    for k in ("in_frame", "missed", "far", "nr_ghosts", "nr_boxes"):
        st2[k] = st[k] + int(round((u - 1) * after.get(k, 0)))
    aft2 = {k: int(round(u * v)) if isinstance(v, (int, float)) else v for k, v in after.items()}
    return st2, aft2


def excess_ghost_counts(after: dict) -> tuple:
    """(ghosts, boxes) of the excess r23 boxes after the first reset, for the Beta draw of q."""
    e = max(0, after.get("ours_boxes", 0) - after.get("nr_boxes", 0))
    g = min(e, max(0, after.get("ours_ghosts", 0) - after.get("nr_ghosts", 0)))
    return g, e


def resample_picks(n_keys: int, n: int, seed: int) -> np.ndarray:
    """The resample indices tools.p8_e1_score.boot draws for n_keys non-empty groups (same generator, same order)."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(n_keys, n_keys, replace=True) for _ in range(n)])


def ratio_interval(picks: np.ndarray, bn, bd, nn, nd) -> list | None:
    """2.5 and 97.5 percentiles of (sum nn / sum nd) - (sum bn / sum bd) over the picked groups, rounded as boot."""
    bn, bd, nn, nd = (np.asarray(x, float) for x in (bn, bd, nn, nd))
    sb, sbd, sn, snd = bn[picks].sum(1), bd[picks].sum(1), nn[picks].sum(1), nd[picks].sum(1)
    keep = (sbd > 0) & (snd > 0)
    if not keep.any():
        return None
    v = sn[keep] / snd[keep] - sb[keep] / sbd[keep]
    return [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]


# ── design v2 measurement (a): truth acceptance before and after the first reset ────────────────────────────────
def truth_window(frames: list, accepted: set, resets: int) -> dict:
    """One label window: processed frames and accepted truth frames before and after the first reset-like frame."""
    fr = sorted(frames, key=lambda r: int(r["frame"]))
    i0 = first_reset_index(fr)
    proc = [int(r["frame"]) for r in fr]
    cut = None if i0 is None else proc[i0]
    pb = sum(1 for f in proc if cut is None or f < cut)
    pa = len(proc) - pb
    ps = set(proc)
    ab = sum(1 for f in accepted if f in ps and (cut is None or f < cut))
    aa = sum(1 for f in accepted if f in ps and cut is not None and f >= cut)
    return {"resets": resets, "first_reset_frame": cut, "proc_before": pb, "proc_after": pa, "acc_before": ab,
            "acc_after": aa, "acc_not_processed": sum(1 for f in accepted if f not in ps)}


def truth_summary(wins: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled acceptance rates; the window-cluster interval of rate before minus rate after over windows with a
    located reset (resampled as boot does)."""
    loc = [w for w in wins.values() if w["first_reset_frame"] is not None]
    nores = [w for w in wins.values() if w["resets"] == 0]
    s = lambda ws, k: sum(w[k] for w in ws)
    rb = s(loc, "acc_before") / s(loc, "proc_before") if s(loc, "proc_before") else None
    ra = s(loc, "acc_after") / s(loc, "proc_after") if s(loc, "proc_after") else None
    proc_all = s(wins.values(), "proc_before") + s(wins.values(), "proc_after")
    acc_all = s(wins.values(), "acc_before") + s(wins.values(), "acc_after")
    share_p = s(wins.values(), "proc_after") / proc_all if proc_all else None
    share_a = s(wins.values(), "acc_after") / acc_all if acc_all else None
    iv = None
    if len(loc) >= 2:
        pk = resample_picks(len(loc), n_boot, seed)
        A = np.array([[w["acc_before"], w["proc_before"], w["acc_after"], w["proc_after"]] for w in loc], float)
        S = A[pk].sum(1)
        keep = (S[:, 1] > 0) & (S[:, 3] > 0)
        v = S[keep, 0] / S[keep, 1] - S[keep, 2] / S[keep, 3]
        iv = [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
    return {"windows": len(wins), "windows_with_reset": sum(1 for w in wins.values() if w["resets"] > 0),
            "windows_reset_located": len(loc), "windows_reset_not_located":
                sum(1 for w in wins.values() if w["resets"] > 0 and w["first_reset_frame"] is None),
            "rate_before_in_reset_windows": rb, "rate_after_in_reset_windows": ra,
            "ratio_before_over_after": (rb / ra) if rb and ra else None,
            "rate_before_minus_after_window_interval": iv,
            "rate_no_reset_windows": (s(nores, "acc_before") / s(nores, "proc_before")) if s(nores, "proc_before") else None,
            "share_after_reset_processed": share_p, "share_after_reset_accepted": share_a,
            "under_representation": (share_p / share_a) if share_p and share_a else None,
            "accepted_not_processed": s(wins.values(), "acc_not_processed")}


def cmd_truthreset(args) -> dict:
    from tools.p8_common import REPO, atomic_write, check_outside, p8_root
    from tools.p8_placement import CHECK, POOL, check_games, rnd
    from tools.research_testset import refuse_cv_test
    games = list(POOL + CHECK)
    refuse_cv_test(check_games(games), WHO)
    mps = p8_root()
    ref = check_outside(Path(args.ref_root).expanduser() if args.ref_root else mps.parent / "work_ref")
    out = {}
    for truth, root, gs in (("mps", mps, games), ("cpu", ref, ["gsw_cha_2016"])):
        for g in gs:
            labels = {p.stem: p for p in (root / "labels" / g).glob(f"{g}_s*.jsonl")}
            wins, missing = {}, []
            for w in sorted(labels):
                side = root / "label_build" / f"{w}_frames.json"
                if not side.exists():
                    missing.append(w)
                    continue
                acc = {int(json.loads(x)["frame"]) for x in labels[w].read_text().splitlines() if x.strip()}
                ident = json.loads((root / "label_build" / f"{w}_identity.json").read_text())["identity"]
                wins[w] = truth_window(json.loads(side.read_text())["frames"], acc,
                                       int(ident.get("camera_cut_resets") or 0))
            out[f"{g}__{truth}"] = {"game": g, "truth": truth, "label_windows": len(labels),
                                    "windows_without_label_build": missing, **truth_summary(wins), "per_window": wins}
            r = out[f"{g}__{truth}"]
            print(g, truth, json.dumps(rnd({k: r[k] for k in ("windows", "windows_with_reset", "windows_reset_located",
                                                             "rate_before_in_reset_windows", "rate_after_in_reset_windows",
                                                             "ratio_before_over_after", "under_representation")})))
    res = rnd({"what": "Truth acceptance before and after the first reset of each label-build window "
                       "(docs/research/P8_resetfix.md version 2, measurement a); report only", "units": out})
    atomic_write(REPO / "reports" / "research" / "p8_resetfix_truthreset.json", json.dumps(res, indent=1))
    return res


# ── design v2 measurement (b): simulation of the expected change under the v2 rule ──────────────────────────────
SIM_F = (0.5, 0.9)          # recovery fraction, uniform per draw
SIM_DRAWS, SIM_SEED = 500, 0
SIM_CENTRAL_F = 0.7


def sim_arm(base: dict, after: dict, resets: int, f: float, q: float | None, rng=None) -> dict:
    """The arm's window counts from the baseline's: in a window with a reset, missed falls by about f of the misses an
    r23 box reaches after the first reset (2.5 ft), far misses by f of the far misses an r23 box reaches (5 ft), and f
    of the excess r23 boxes after the reset are added as non-referee boxes, a share q of them ghosts. rng None gives
    the expected values (rounded)."""
    new = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    if resets <= 0 or not after:
        return new
    e = max(0, after.get("ours_boxes", 0) - after.get("nr_boxes", 0))
    if rng is None:
        k, k5, a = round(f * after.get("reach", 0)), round(f * after.get("far_reach", 0)), round(f * e)
        g = round(a * (q or 0.0))
    else:
        k = int(rng.binomial(after.get("reach", 0), f))
        k5 = int(rng.binomial(after.get("far_reach", 0), f))
        a = int(rng.binomial(e, f))
        g = int(rng.binomial(a, q or 0.0))
    new["missed"] -= k
    new["far"] -= k5
    new["nr_boxes"] += a
    new["nr_ghosts"] += g
    return new


def sim_count_rows(wins: list, picks_w: np.ndarray) -> tuple:
    """Point differences and window intervals of the count rows (missed, detection, ghosts, coverage)."""
    spec = {"missed_rate": ("missed", "in_frame"), "detection_rate": ("far", "in_frame"),
            "ghost_rate": ("nr_ghosts", "nr_boxes"), "coverage": ("cov_acc", "cov_live")}
    point, iv = {}, {}
    for r, (num, den) in spec.items():
        bn = [w["r23"][num] for w in wins]; bd = [w["r23"][den] for w in wins]
        nn = [w["new"][num] for w in wins]; nd = [w["new"][den] for w in wins]
        point[r] = sum(nn) / sum(nd) - sum(bn) / sum(bd)
        iv[r] = ratio_interval(picks_w, bn, bd, nn, nd)
    return point, iv


def cmd_simulate(args) -> dict:
    from tools.p8_common import REPO, atomic_write, p8_root
    from tools.p8_e1_score import boot, diffs, rows
    from tools.p8_placement import POOL, SCORES, SIDE, check_games, rnd
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(check_games(list(POOL)), WHO)
    wr = p8_root()
    v22 = getattr(args, "version", "2") == "2.2"
    est = json.loads((REPO / "reports" / "research" / "p8_resetfix_estimate.json").read_text())["games"]
    under = {}
    if v22:
        tr = json.loads((REPO / "reports" / "research" / "p8_resetfix_truthreset.json").read_text())["units"]
        under = {g: tr[f"{g}__mps"]["under_representation"] for g in POOL}
    base, meta = {}, {}
    for g in POOL:
        sf = json.loads((wr / "e1" / SCORES[g]).read_text())
        pw = est[g]["per_window"]
        q = est[g]["rates"]["after_reset_excess_ghost_share"]
        gc, ec = excess_ghost_counts(est[g]["counts"]["after_first_reset"])
        meta[g] = {"q": q if not v22 else (gc + 1) / (ec + 2), "beta": (gc + 1, ec - gc + 1), "u": under.get(g)}
        base[g] = []
        for w in sf["scored_windows"]:
            st = dict(sf["windows"][w][SIDE[g]])
            if st["in_frame"] != pw[w]["all"].get("in_frame", 0) or st["missed"] != pw[w]["all"].get("missed", 0):
                raise SystemExit(f"{WHO}: {w}: estimate and score file disagree")
            st["far"] = pw[w]["all"].get("far", 0)          # a window with no scored frame has no counts
            aft = pw[w]["after"]
            if v22 and pw[w]["resets"] > 0:
                st, aft = scale_after(st, aft, under[g])
            base[g].append((w, st, aft, pw[w]["resets"]))

    def build(f, rng):
        qs = {g: (meta[g]["q"] if (rng is None or not v22) else float(rng.beta(*meta[g]["beta"]))) for g in POOL}
        return {g: [{"r23": st, "new": sim_arm(st, aft, rs, f, qs[g], rng)} for (_w, st, aft, rs) in base[g]]
                for g in POOL}

    def unit_rows(wins, picks):
        point, iv = sim_count_rows(wins, picks)
        for r in ("position_p50_ft", "position_p90_ft", "nearfield_bias_ft", "team_accuracy"):
            point[r], iv[r] = 0.0, [0.0, 0.0]        # unchanged by construction in the simulation
        point[ONE_TO_ONE_ROW] = point["missed_rate"]  # not simulated apart: taken equal to missed
        return point, iv

    n_w = {g: len(base[g]) for g in POOL}
    picks_g = {g: resample_picks(n_w[g], args.boot, 0) for g in POOL}
    picks_pool = resample_picks(sum(n_w.values()), args.boot, 0)
    picks_games = resample_picks(len(POOL), args.boot, 0)

    def verdicts(sim):
        allw = [w for g in POOL for w in sim[g]]
        pp, piv = unit_rows(allw, picks_pool)
        # game interval: groups are games
        gtot = [{s: {k: sum(w[s][k] for w in sim[g]) for k in ("missed", "far", "in_frame", "nr_ghosts", "nr_boxes",
                                                                "cov_acc", "cov_live")} for s in ("r23", "new")}
                for g in POOL]
        _, giv = sim_count_rows(gtot, picks_games)
        cov = (sum(w["r23"]["cov_acc"] for w in allw) / sum(w["r23"]["cov_live"] for w in allw),
               sum(w["new"]["cov_acc"] for w in allw) / sum(w["new"]["cov_live"] for w in allw))
        pooled = v2_verdict(pp, piv, cov, giv)
        per = {}
        for g in POOL:
            gp, giv_ = unit_rows(sim[g], picks_g[g])
            c = (sum(w["r23"]["cov_acc"] for w in sim[g]) / sum(w["r23"]["cov_live"] for w in sim[g]),
                 sum(w["new"]["cov_acc"] for w in sim[g]) / sum(w["new"]["cov_live"] for w in sim[g]))
            per[g] = v2_verdict(gp, giv_, c, None, improve_rows=(), one_to_one=False)
        return pool_verdict(pooled, per), pooled, per

    # central run, checked against the real rows, diffs and boot
    central = build(SIM_CENTRAL_F, None)
    pv, pooled, per = verdicts(central)
    allw = [w for g in POOL for w in central[g]]
    real_point = diffs(allw)
    real_wiv = boot({f"{g}/{i}": [w] for g in POOL for i, w in enumerate(central[g])}, n=args.boot)
    real_giv = boot(central, n=args.boot)
    fast_p, fast_wiv = sim_count_rows(allw, picks_pool)
    gtot = [{sd: {k: sum(w[sd][k] for w in central[g]) for k in ("missed", "far", "in_frame", "nr_ghosts", "nr_boxes",
                                                                  "cov_acc", "cov_live")} for sd in ("r23", "new")}
            for g in POOL]
    fast_giv = sim_count_rows(gtot, picks_games)[1]
    check = {r: {"point_equal": abs(real_point[r] - fast_p[r]) < 1e-12, "window_interval_equal": real_wiv[r] == fast_wiv[r],
                 "game_interval_equal": real_giv[r] == fast_giv[r]} for r in ("missed_rate", "ghost_rate", "coverage")}
    for r in ("position_p50_ft", "position_p90_ft", "nearfield_bias_ft", "team_accuracy"):
        check[r] = {"point": real_point[r], "window_interval": real_wiv[r], "game_interval": real_giv[r],
                    "zero": real_point[r] == 0 and real_wiv[r] == [0.0, 0.0]}
    if not all(all(v for k, v in c.items() if k in ("point_equal", "window_interval_equal", "game_interval_equal",
                                                    "zero")) for c in check.values()):
        raise SystemExit(f"{WHO}: the fast intervals do not reproduce tools.p8_e1_score.boot: {check}")
    # draws
    rng = np.random.default_rng(SIM_SEED)
    draws = []
    for _ in range(args.draws):
        f = float(rng.uniform(*SIM_F))
        sim = build(f, rng)
        v, po, pg = verdicts(sim)
        ad = adoption(v, po, True)
        draws.append({"f": f, "pass": v["pass"], "adopt": ad["adopt"], "pooled_pass": v["pooled"], "games": v["games"],
                      "failing": v["failing"],
                      "missed": po["rows"]["missed_rate"]["diff"], "missed_iv": po["rows"]["missed_rate"]["window_interval"],
                      "detection": po["rows"]["detection_rate"]["diff"],
                      "ghost": po["rows"]["ghost_rate"]["diff"], "ghost_hi": po["rows"]["ghost_rate"]["bad_end"],
                      "ghost_hi_games": {g: pg[g]["rows"]["ghost_rate"]["bad_end"] for g in POOL},
                      "missed_games": {g: pg[g]["rows"]["missed_rate"]["diff"] for g in POOL}})
    q = lambda xs: {"p05": float(np.percentile(xs, 5)), "p50": float(np.percentile(xs, 50)),
                    "p95": float(np.percentile(xs, 95))}
    from collections import Counter
    fails = Counter(x for d in draws for x in d["failing"])
    res = rnd({"what": ("Simulated reset-fix arm under the design v2 rule (docs/research/P8_resetfix.md version 2, "
                        "measurement b); a forecast on the MPS-truth scored frames, not a result") if not v22 else
               ("Simulated reset-fix arm, version 2.2 (docs/research/P8_resetfix.md section 16): post-reset counts scaled "
                "by each game's under-representation factor, q drawn from a Beta on its counts; the v2 rule and the "
                "v2.2 adoption rule (bracket guard taken as passing); a forecast, not a result"),
               "version": "2.2" if v22 else "2",
               "under_representation": under if v22 else None,
               "adoption_probability": sum(d["adopt"] for d in draws) / len(draws),
               "central": {"f": SIM_CENTRAL_F, "verdict": pv, "pooled": pooled, "per_game": per,
                           "checked_against_real_code": check},
               "draws": len(draws), "boot": args.boot,
               "pass_probability": sum(d["pass"] for d in draws) / len(draws),
               "pooled_pass_probability": sum(d["pooled_pass"] for d in draws) / len(draws),
               "game_pass_probability": {g: sum(d["games"][g] for d in draws) / len(draws) for g in POOL},
               "missed_interval_below_zero_probability": sum(1 for d in draws if d["missed_iv"] and d["missed_iv"][1] < 0)
               / len(draws),
               "failing_conditions": {k: v / len(draws) for k, v in fails.most_common()},
               "forecast": {"missed_pooled": q([d["missed"] for d in draws]),
                            "detection_pooled": q([d["detection"] for d in draws]),
                            "ghost_pooled": q([d["ghost"] for d in draws]),
                            "ghost_window_hi_pooled": q([d["ghost_hi"] for d in draws]),
                            "ghost_window_hi_by_game": {g: q([d["ghost_hi_games"][g] for d in draws]) for g in POOL},
                            "missed_by_game": {g: q([d["missed_games"][g] for d in draws]) for g in POOL}},
               "excess_ghost_share": {g: meta[g]["q"] for g in POOL}})
    name = "p8_resetfix_simulate_v22.json" if v22 else "p8_resetfix_simulate.json"
    atomic_write(REPO / "reports" / "research" / name, json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("pass_probability", "adoption_probability", "pooled_pass_probability",
                                          "game_pass_probability",
                                          "missed_interval_below_zero_probability", "failing_conditions", "forecast")},
                     indent=1))
    return res


# ── design v2 stage 0: build, checks, score and report in the reference root ────────────────────────────────────
POOL = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = "hou_cle_2016"
GAMES = POOL + (APART,)
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
REFERENCE = {"detector": "cpu", "court": "auto", "gate": "auto"}       # the reference platform (D175)
PINNED_VERSIONS = {"opencv": "4.13.0", "torch": "2.2.2", "ultralytics": "8.4.48"}   # a3input's build environment
BASE_DIR = "r23_nofix"                         # the unfixed baseline: verified copies of the workspace's r23 builds
# builds/r23 is check C's folder, composed by tools.p8_truth_cpu_r23 compose (r-p8-truth-cpu); nothing here writes it
SMOKE_GAME = "gsw_cha_2016"
SMOKE_RESET, SMOKE_NORESET = "gsw_cha_2016_s05_w00", "gsw_cha_2016_s00_w00"
SIDE_FILES = ("frames", "trajectories", "identity")
ENV, BUILD_SCRIPT = "P8_RESETFIX_DETECTOR", "build_trajectories.py"
FINAL_TRUTH = ("use", "apart", "unused")       # tools.p8_truth_accept's "use" values
BOX_TOL_PX = 1.0                               # a3input's outputs are rounded to 0.01 px; the sidecar truncates


def arm_dir(game: str) -> str:
    return f"resetfix_r23_detcpu__{game}"


def check_game(game: str) -> str:
    if game not in GAMES:
        raise SystemExit(f"{WHO}: {game} is not a reset-fix game ({GAMES})")
    from tools.research_testset import refuse_cv_test
    from tools.p8_common import guard
    refuse_cv_test([game], WHO)
    guard([game], WHO)
    return game


def ref_root() -> Path:
    """The reference root: V3_P8_ROOT names it, it is not the P8 workspace beside it, and it holds label_build/."""
    import os
    from tools.p8_common import p8_root
    r = p8_root()
    if not os.environ.get("V3_P8_ROOT") or r.resolve() == (r.parent / "work").resolve() or \
            not (r / "label_build").is_dir():
        raise SystemExit(f"{WHO}: set V3_P8_ROOT to the reference root (not the P8 workspace)")
    return r


def label_stems(r: Path, game: str) -> list:
    """The game's label windows in a root: one time map each (time maps and clips stay when label builds move)."""
    return sorted(p.name[:-len("_timemap.json")] for p in (r / "label_build").glob(f"{game}_s*_timemap.json"))


def baseline_problems(stem: str, folder: Path, ws: Path, r: Path) -> str | None:
    """None when a workspace r23 build (or a copy of it) may stand for the reference root's unfixed baseline of this
    window, else the reason. The rules of tools.p8_truth_cpu_r23 (verify and check): built from the workspace clip; that
    clip byte-identical to the reference root's; device record on the reference platform with no clip fix (and here
    no reset fix); detector record r23."""
    if (folder / f"{stem}_gate_empty.json").exists() and not (folder / f"{stem}_frames.json").exists():
        same = (ws / "label_build" / f"{stem}.mp4").exists() and (r / "label_build" / f"{stem}.mp4").exists() and \
            sha256_file(ws / "label_build" / f"{stem}.mp4") == sha256_file(r / "label_build" / f"{stem}.mp4")
        return None if same else "gate-empty window whose clips differ or are missing"
    side = _read(folder / f"{stem}_frames.json")
    if side is None:
        return "no build"
    built_from = side.get("source")
    mclip = ws / "label_build" / f"{stem}.mp4"
    if not built_from or Path(built_from).expanduser().resolve() != mclip.resolve():
        return "not built from the workspace clip"
    det, dev = _read(folder / f"{stem}_detector.json"), _read(folder / f"{stem}_device.json")
    if not det or det.get("sha256") != R23_SHA or det.get("model") != "r23":
        return "detector record is not r23"
    if not dev or dev.get("requested") != REFERENCE:
        return "device record is not the reference platform"
    if dev.get("clip_fix") or dev.get("reset_fix"):
        return "built with a fix"
    rclip = r / "label_build" / f"{stem}.mp4"
    if not mclip.exists() or not rclip.exists():
        return "clip missing"
    if sha256_file(mclip) != sha256_file(rclip):
        return "clips differ"
    return None


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# pure checks
def plan_windows(base_identity: dict, gate_empty: set) -> dict:
    """base_identity: {window: camera_cut_resets of the baseline build}. -> the windows built (a reset in the
    baseline), the equivalence sample (first and last no-reset window by name) and the no-reset windows."""
    ws = sorted(w for w in base_identity if w not in gate_empty)
    reset = [w for w in ws if base_identity[w] > 0]
    none = [w for w in ws if base_identity[w] == 0]
    equiv = sorted({none[0], none[-1]}) if none else []
    return {"reset_windows": reset, "equivalence_windows": equiv, "no_reset_windows": none,
            "gate_empty_windows": sorted(gate_empty)}


def check_arm_records(det: dict | None, dev: dict | None, log: dict | None, resets: int | None) -> list:
    """Problems with one fixed-arm window's records -> [] when it passes."""
    bad = []
    if dev is None or dev.get("requested") != REFERENCE or (dev.get("resolved") or {}).get("detector") != "cpu":
        bad.append("device record is not the reference platform with the detector on cpu")
    if dev is not None and dev.get("clip_fix"):
        bad.append("built with the MPS clip fix")
    if dev is None or dev.get("reset_fix") is not True:
        bad.append("device record does not carry reset_fix")
    if det is None or det.get("sha256") != R23_SHA or det.get("model") != "r23":
        bad.append("detector record is not r23")
    if log is None:
        bad.append("no reset log")
        return bad
    if log.get("tracker_reid_env") not in (None,):
        bad.append(f"TRACKER_REID was set ({log.get('tracker_reid_env')!r})")
    if log.get("versions") != PINNED_VERSIONS:
        bad.append(f"library versions {log.get('versions')} are not {PINNED_VERSIONS}")
    rs = log.get("resets", [])
    if any(r.get("stripped") != 2 for r in rs):
        bad.append(f"a reset did not strip exactly 2 callbacks: {[r.get('stripped') for r in rs]}")
    if any(r.get("frame") is None for r in rs):
        bad.append("a reset has no frame index")
    if resets is not None and len(rs) != resets:
        bad.append(f"the reset log has {len(rs)} entries, the identity file {resets}")
    return bad


def check_base_records(det: dict | None, dev: dict | None) -> list:
    bad = []
    if dev is None or dev.get("requested") != REFERENCE or (dev.get("resolved") or {}).get("detector") != "cpu":
        bad.append("baseline device record is not the reference platform")
    if dev is not None and (dev.get("clip_fix") or dev.get("reset_fix")):
        bad.append("baseline built with a fix")
    if det is None or det.get("sha256") != R23_SHA:
        bad.append("baseline detector record is not r23")
    return bad


def prefix_equal(base_rows: list, arm_rows: list, first_reset_frame: int | None) -> dict:
    """Sidecar rows up to and including the arm's first reset frame must be identical (the two builds run the same
    calls until the first reset; later rows, and the window's other outputs, may differ)."""
    b = sorted(base_rows, key=lambda r: int(r["frame"]))
    a = sorted(arm_rows, key=lambda r: int(r["frame"]))
    lim = first_reset_frame if first_reset_frame is not None else float("inf")
    bp = [r for r in b if int(r["frame"]) <= lim]
    ap = [r for r in a if int(r["frame"]) <= lim]
    first_diff = next((int(x["frame"]) for x, y in zip(bp, ap) if x != y), None)
    return {"rows": len(bp), "equal": bp == ap, "first_differing_frame": first_diff}


def boxes_match(side_boxes: list, ref_boxes: list, tol: float = BOX_TOL_PX) -> bool:
    """Same count and a one-to-one match with every coordinate within tol."""
    if len(side_boxes) != len(ref_boxes):
        return False
    if not side_boxes:
        return True
    from scipy.optimize import linear_sum_assignment
    A, B = np.asarray(side_boxes, float)[:, :4], np.asarray(ref_boxes, float)[:, :4]
    D = np.abs(A[:, None, :] - B[None, :, :]).max(-1)
    ri, ci = linear_sum_assignment(D)
    return bool(D[ri, ci].max() <= tol)


def boxes_consistency(arm_rows: list, label_rows: list) -> dict:
    """Arm sidecar against the fixed label build of the same window: on frames both processed, are the boxes (bbox
    and track id) the same? The two share detector and tracker; only court and gate settings differ."""
    a = {int(r["frame"]): sorted((tuple(b["bbox"]), b.get("tid")) for b in r["boxes"]) for r in arm_rows}
    b = {int(r["frame"]): sorted((tuple(x["bbox"]), x.get("tid")) for x in r["boxes"]) for r in label_rows}
    shared = a.keys() & b.keys()
    return {"frames_shared": len(shared), "frames_equal": sum(a[f] == b[f] for f in shared),
            "frames_only_arm": len(a.keys() - b.keys()), "frames_only_label": len(b.keys() - a.keys())}


def one_to_one_missed(feet, sv, inside, miss_ft: float = 2.5) -> int:
    """In-frame players left unmatched by a one-to-one assignment of box feet to players within miss_ft (a box can
    find at most one player, so duplicate boxes cannot lower it)."""
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ins = np.where(np.asarray(inside, bool))[0]
    if not len(ins):
        return 0
    if not len(feet):
        return len(ins)
    from scipy.optimize import linear_sum_assignment
    D = np.sqrt(((sv[ins][:, None, :] - feet[None, :, :]) ** 2).sum(-1))
    C = np.where(D <= miss_ft, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    return int(len(ins) - (C[ri, ci] <= miss_ft).sum())


def frame_extra(feet, sv, inside) -> dict:
    """Report-only and detection counts of one scored frame: far misses (no foot within 5 ft), one-to-one misses,
    duplicates (in-frame players with two or more feet within 2.5 ft)."""
    sv = np.asarray(sv, np.float64).reshape(-1, 2)
    feet = np.asarray(feet, np.float64).reshape(-1, 2)
    ins = np.asarray(inside, bool)
    if len(feet):
        D = np.sqrt(((sv[:, None, :] - feet[None, :, :]) ** 2).sum(-1))
        far = int((ins & (D.min(1) > 5.0)).sum())
        dup = int((ins & ((D <= 2.5).sum(1) >= 2)).sum())
    else:
        far, dup = int(ins.sum()), 0
    return {"far": far, "missed_1to1": one_to_one_missed(feet, sv, inside), "duplicates": dup}


def truth_pin_problems(pin: dict | None, now: dict) -> list:
    """pin: the pinned {file: sha256} of a game's truth; now: the same files hashed now."""
    if pin is None:
        return ["no truth pin"]
    if pin.get("status") not in FINAL_TRUTH:
        return [f"truth status {pin.get('status')!r} is not final"]
    if pin["files"] != now:
        return ["truth files changed since the pin"]
    return []


def pin_order_problems(build_mtimes: list, truth_mtimes: dict, acc_label_windows, fixed_label_windows: int) -> list:
    """Version 2.2: the acceptance record, the fit result, the offsets and every label file must post-date every fixed
    label build, and the acceptance record's count of label windows must equal the fixed label windows."""
    bad = []
    newest = max(build_mtimes) if build_mtimes else None
    old = sorted(k for k, t in truth_mtimes.items() if newest is None or t is None or t <= newest)
    if newest is None:
        bad.append("no fixed label build")
    if old:
        bad.append(f"{len(old)} truth files do not post-date the fixed label builds (first: {old[:3]})")
    if acc_label_windows != fixed_label_windows:
        bad.append(f"the acceptance record counts {acc_label_windows} label windows, the fixed label builds "
                   f"{fixed_label_windows}")
    return bad


def label_fix_problems(device_records: dict) -> list:
    """{label window: its label build's device record or None} -> [] when every built label window ran on the
    reference platform with the reset fix (the truth is rebuilt with the fix, design v2 section 6)."""
    bad = [w for w, d in device_records.items()
           if d is None or d.get("reset_fix") is not True or (d.get("resolved") or {}).get("detector") != "cpu"]
    return [f"{len(bad)} label windows lack the reset fix or the cpu detector (first: {sorted(bad)[:3]})"] if bad else []


# child (frozen CV python, inside the build)
class _Proxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, cmd, *a, **k):
        return self._real.run(rewrite(cmd), *a, **k)


def rewrite(cmd, env=None):
    import os
    env = os.environ if env is None else env
    if env.get(ENV) and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == BUILD_SCRIPT:
        return [cmd[0], "-m", "tools.p8_resetfix", "child", *cmd[1:]]
    return cmd


def child(argv: list) -> int:
    import os
    import runpy
    import sys
    from tools import p8_device as D
    from tools.p8_common import guard
    if not argv or argv[0] != BUILD_SCRIPT:
        raise SystemExit(f"{WHO}: usage: python -m tools.p8_resetfix child {BUILD_SCRIPT} <its arguments>")
    src = argv[argv.index("--source") + 1] if "--source" in argv else ""
    guard([src], WHO)
    req = D.requested()
    if req != REFERENCE or D.clip_fix_requested() or not D.reset_fix_requested():
        raise SystemExit(f"{WHO}: the arm is P8_DEVICE_DETECTOR=cpu with P8_TRACKER_RESET_FIX=1 only")
    if os.environ.get("TRACKER_REID") is not None:
        raise SystemExit(f"{WHO}: TRACKER_REID must be unset")
    resolved = D.apply(req)
    weights = Path(os.environ[ENV])
    import config
    config.PLAYER_DETECTOR_WEIGHTS = weights
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    for suffix, rec in (("device", D.device_record(req, resolved)),
                        ("detector", {"weights": weights.name, "sha256": sha256_file(weights), "model": "r23",
                                      "seed": 0})):
        part = config.TRACKING_DIR / f"{stem}_{suffix}.json.part"
        part.write_text(json.dumps(rec))
        os.replace(part, config.TRACKING_DIR / f"{stem}_{suffix}.json")
    start_reset_log(config.TRACKING_DIR / f"{stem}_resetfix.json")
    script = Path(config.PROJECT_ROOT) / BUILD_SCRIPT
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def cmd_plan(args) -> dict:
    """System python, no lock: the windows built per game, from the reference root's baseline identity files."""
    from tools.p8_common import atomic_write
    game = check_game(args.game)
    r = ref_root()
    src = _read(r / "builds" / f"{BASE_DIR}_source_{game}.json")
    if not src or not src.get("complete"):
        raise SystemExit(f"{WHO}: {game}: the unfixed baseline is not complete in the reference root (baseline-copy)")
    base = r / "builds" / BASE_DIR
    labels = label_stems(r, game)
    ident, empty = {}, set()
    for w in labels:
        if (base / f"{w}_gate_empty.json").exists():
            empty.add(w)
            continue
        i = _read(base / f"{w}_identity.json")
        if i is None:
            raise SystemExit(f"{WHO}: {w} has no baseline build")
        ident[w] = int(i["identity"].get("camera_cut_resets") or 0)
    plan = plan_windows(ident, empty)
    plan.update({"game": game, "baseline_resets": ident,
                 "clip_sha256": {w: sha256_file(r / "label_build" / f"{w}.mp4") for w in labels},
                 "baseline_sha256": {w: {f: sha256_file(base / f"{w}_{f}.json") for f in SIDE_FILES} for w in ident}})
    d = r / "resetfix"
    d.mkdir(exist_ok=True)
    atomic_write(d / f"plan__{game}.json", json.dumps(plan, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in plan.items()
                      if k not in ("baseline_resets", "baseline_sha256")}))
    return plan


def clip_changed(plan: dict, r: Path, w: str) -> bool:
    """Version 2.2: a window's clip must still hash as the plan recorded it."""
    want = (plan.get("clip_sha256") or {}).get(w)
    return want is None or sha256_file(r / "label_build" / f"{w}.mp4") != want


def stage_allowed(stage: str, d: Path) -> list:
    """The gates in force before a stage: smoke before the pool, the pool verdict before hou_cle_2016."""
    why = []
    st0 = _read(d / "stage0.json")
    if not st0 or not st0.get("passes"):       # version 2.2: the real ultralytics check under the frozen CV python
        why.append("stage0-check has not passed")
    if stage in ("pool", "hou", "hou_score"):
        sm = _read(d / "smoke_check.json")
        if not sm or not sm.get("passes"):
            why.append("the smoke check has not passed")
    if stage == "hou_score":                 # hou builds need only the smoke; hou scoring needs the pool pass
        rep = _read(d / "report.json")
        if not rep or not (rep.get("pool") or {}).get("verdict", {}).get("pass"):
            why.append("the pool rule has not passed")
    return why


def stage0_checks(TR, versions: dict, tracker_cls, detector_sha: str) -> dict:
    """The stage 0 gate's checks on the frozen CV interpreter's real modules (pure on what it is given)."""
    from functools import partial
    return {"recognises_on_predict_start": is_tracker_callback(partial(TR.on_predict_start, persist=True)),
            "recognises_on_predict_postprocess_end": is_tracker_callback(partial(TR.on_predict_postprocess_end,
                                                                                 persist=True)),
            "versions_pinned": versions == PINNED_VERSIONS,
            "player_tracker_has_reset_and_update": all(hasattr(tracker_cls, m) for m in ("reset", "update")),
            "detector_is_r23": detector_sha == R23_SHA}


def cmd_stage0_check(args) -> dict:
    """Frozen CV python, no lock (imports only, no model loaded): the real-ultralytics test of the fix, passing and
    not skipped, plus the pinned versions and the r23 weights. Writes resetfix/stage0.json; the smoke needs it."""
    import os
    import cv2  # noqa: F401  (loaded so its version is read)
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
    import ultralytics.trackers.track as TR
    from tools.p8_common import atomic_write, main_checkout
    r = ref_root()
    import detect.tracker as DT
    checks = stage0_checks(TR, library_versions(), DT.PlayerTracker,
                           sha256_file((main_checkout() / "models" / "player_detector.pt").resolve()))
    checks["tracker_reid_unset"] = os.environ.get("TRACKER_REID") is None
    out = {"checks": checks, "versions": library_versions(), "passes": all(checks.values())}
    (r / "resetfix").mkdir(exist_ok=True)
    atomic_write(r / "resetfix" / "stage0.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if not out["passes"]:
        raise SystemExit(f"{WHO}: stage 0 fails: {[k for k, v in checks.items() if not v]}")
    return out


def cmd_build(args) -> dict:
    """Frozen CV python, inside the heavy-job lock: the fixed arm on the planned windows (or the two smoke windows)."""
    import os
    import time
    from sportvu import rebuild
    from sportvu.autolabel import gate_args
    from tools import p8_frozen as F
    from tools.p8_common import atomic_write, main_checkout
    game = check_game(args.game)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = ref_root()
    d = r / "resetfix"
    stage = "smoke" if args.smoke else ("hou" if game == APART else "pool")
    why = stage_allowed(stage, d)
    if why:
        raise SystemExit(f"{WHO}: stage {stage} may not start: {why}")
    for k in ("P8_DEVICE", "P8_DEVICE_COURT", "P8_DEVICE_GATE", "P8_MPS_CLIP_FIX", "TRACKER_REID"):
        os.environ.pop(k, None)
    os.environ["P8_DEVICE_DETECTOR"] = "cpu"
    os.environ["P8_TRACKER_RESET_FIX"] = "1"
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    if sha256_file(weights) != R23_SHA:
        raise SystemExit(f"{WHO}: the production detector is not r23")
    if args.smoke:
        if game != SMOKE_GAME:
            raise SystemExit(f"{WHO}: the smoke builds only {SMOKE_GAME}")
        todo = [SMOKE_NORESET, SMOKE_RESET]
    else:
        plan = _read(d / f"plan__{game}.json")
        if plan is None:
            raise SystemExit(f"{WHO}: no plan for {game}; run plan first")
        todo = plan["reset_windows"] + plan["equivalence_windows"]
    P = F.install_paths()
    rebuild.subprocess = _Proxy(rebuild.subprocess)
    os.environ[ENV] = str(weights)
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    out = P["builds"] / arm_dir(game)
    out.mkdir(parents=True, exist_ok=True)
    wins = {w["window"]: w for w in F.label_windows(P, game)}
    res = {"game": game, "stage": stage, "folder": out.name, "windows": {}}
    for s in todo:
        t0 = time.time()
        clip = P["label_build"] / f"{s}.mp4"
        if not args.smoke and clip_changed(plan, r, s):
            raise SystemExit(f"{WHO}: {s}: the clip changed since the plan")
        for attempt in (1, 2, 3):
            rr = F.safe_build(P, wins[s], clip, out, extra_args=gate_args())
            if rr["ok"]:
                break
        else:
            raise SystemExit(f"{WHO}: build of {s} failed three times: {rr.get('tail')}")
        bad = []
        if not rr.get("empty"):
            why = arm_source_problem(_read(out / f"{s}_frames.json"), s, r, r.parent / "work")
            if why:
                bad.append(why)
            ident = _read(out / f"{s}_identity.json")
            bad = check_arm_records(_read(out / f"{s}_detector.json"), _read(out / f"{s}_device.json"),
                                    _read(out / f"{s}_resetfix.json"),
                                    None if ident is None else int(ident["identity"].get("camera_cut_resets") or 0))
        if bad:
            raise SystemExit(f"{WHO}: {s}: {bad}")
        res["windows"][s] = {"cached": bool(rr.get("cached")), "empty": bool(rr.get("empty")), "attempts": attempt,
                             "seconds": None if rr.get("cached") else round(time.time() - t0, 1)}
    d.mkdir(exist_ok=True)
    atomic_write(d / f"build__{game}{'__smoke' if args.smoke else ''}.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def canonical_sidecar(side: dict) -> str:
    """A sidecar as canonical JSON without "source" (build_trajectories.py stores the absolute clip path there, and the
    baseline and the arm are built from two copies of the same clip in two roots)."""
    return json.dumps({k: v for k, v in side.items() if k != "source"}, sort_keys=True)


def arm_source_problem(side: dict, w: str, r: Path, ws: Path) -> str | None:
    """The arm's sidecar must name the reference root's clip, byte-identical to the workspace clip."""
    src = side.get("source")
    rclip = r / "label_build" / f"{w}.mp4"
    if not src or Path(src).expanduser().resolve() != rclip.resolve():
        return "the arm was not built from the reference root's clip"
    wclip = ws / "label_build" / f"{w}.mp4"
    if not wclip.exists() or sha256_file(wclip) != sha256_file(rclip):
        return "the reference root's clip differs from the workspace clip"
    return None


def compare_window(base: Path, arm: Path, w: str) -> dict:
    """Identity of the three outputs (frames as canonical JSON without "source", trajectories and identity byte for
    byte), the sidecar prefix check, and both arms' reset counts."""
    same = {f: sha256_file(base / f"{w}_{f}.json") == sha256_file(arm / f"{w}_{f}.json") for f in SIDE_FILES
            if f != "frames"}
    same["frames"] = canonical_sidecar(_read(base / f"{w}_frames.json")) == canonical_sidecar(_read(arm / f"{w}_frames.json"))
    log = _read(arm / f"{w}_resetfix.json") or {"resets": []}
    first = min((r["frame"] for r in log["resets"] if r.get("frame") is not None), default=None)
    pre = prefix_equal(_read(base / f"{w}_frames.json")["frames"], _read(arm / f"{w}_frames.json")["frames"], first)
    rb = int(_read(base / f"{w}_identity.json")["identity"].get("camera_cut_resets") or 0)
    ra = int(_read(arm / f"{w}_identity.json")["identity"].get("camera_cut_resets") or 0)
    return {"byte_identical": same, "first_reset_frame_arm": first, "prefix": pre, "resets_baseline": rb,
            "resets_arm": ra}


def cmd_smoke_check(args) -> dict:
    """System python: the smoke gate (design v2, section 8)."""
    from tools.p8_common import atomic_write, p8_root
    r = ref_root()
    base, arm = r / "builds" / BASE_DIR, r / "builds" / arm_dir(SMOKE_GAME)
    out = {"no_reset": compare_window(base, arm, SMOKE_NORESET), "reset": compare_window(base, arm, SMOKE_RESET)}
    # the reset window against a3input's variant B, frame by frame (the a3input record lives in the P8 workspace)
    ws = Path(args.workspace).expanduser() if args.workspace else p8_root().parent / "work"
    a3 = _read(ws / "placement" / "a3input" / f"{SMOKE_RESET}.json")
    clip_same = sha256_file(ws / "label_build" / f"{SMOKE_RESET}.mp4") == \
        sha256_file(r / "label_build" / f"{SMOKE_RESET}.mp4")
    side = {int(x["frame"]): [b["bbox"] for b in x["boxes"]] for x in _read(arm / f"{SMOKE_RESET}_frames.json")["frames"]}
    Bf = a3["variants"]["B"]["per_frame"]
    match = sum(boxes_match(side.get(int(f["frame"]), []), f["output"]) for f in Bf)
    log = _read(arm / f"{SMOKE_RESET}_resetfix.json")
    ident = _read(arm / f"{SMOKE_RESET}_identity.json")["identity"]
    rec = check_arm_records(_read(arm / f"{SMOKE_RESET}_detector.json"), _read(arm / f"{SMOKE_RESET}_device.json"), log,
                            int(ident.get("camera_cut_resets") or 0))
    out["a3input_B"] = {"clip_byte_identical": clip_same, "frames": len(Bf), "frames_matching": match,
                        "frames_equal_sets": set(side) == {int(f["frame"]) for f in Bf},
                        "boxes_arm": sum(len(v) for v in side.values()),
                        "boxes_B": sum(len(f["output"]) for f in Bf), "resets_arm": len((log or {}).get("resets", [])),
                        "resets_B": a3["variants"]["B"]["n_resets"], "record_problems": rec,
                        "versions": (log or {}).get("versions")}
    src_bad = {w: arm_source_problem(_read(arm / f"{w}_frames.json"), w, r, ws) for w in (SMOKE_NORESET, SMOKE_RESET)}
    out["source_problems"] = src_bad
    st0 = _read(r / "resetfix" / "stage0.json")
    gates = {"stage0_passed": bool(st0 and st0.get("passes")),
             "arm_sources": not any(src_bad.values()),
             "no_reset_byte_identical": all(out["no_reset"]["byte_identical"].values()),
             "clip_byte_identical": clip_same,
             "every_frame_matches_B": match == len(Bf) and out["a3input_B"]["frames_equal_sets"],
             "resets_equal_B": out["a3input_B"]["resets_arm"] == out["a3input_B"]["resets_B"],
             "records_pass": not rec,
             "prefix_equal": out["reset"]["prefix"]["equal"]}
    out["gates"] = gates
    out["passes"] = all(gates.values())
    (r / "resetfix").mkdir(exist_ok=True)
    atomic_write(r / "resetfix" / "smoke_check.json", json.dumps(out, indent=1))
    print(json.dumps({"gates": gates, "passes": out["passes"], "a3input_B": out["a3input_B"]}, indent=1))
    if not out["passes"]:
        raise SystemExit(f"{WHO}: the smoke check fails: {[k for k, v in gates.items() if not v]}")
    return out


def cmd_equiv(args) -> dict:
    """System python: after a game's builds, the equivalence windows must be byte-identical to the baseline, and every
    built reset window must pass the prefix check."""
    from tools.p8_common import atomic_write
    game = check_game(args.game)
    r = ref_root()
    plan = _read(r / "resetfix" / f"plan__{game}.json")
    base, arm = r / "builds" / BASE_DIR, r / "builds" / arm_dir(game)
    ws = r.parent / "work"
    out = {"game": game, "equivalence": {w: compare_window(base, arm, w) for w in plan["equivalence_windows"]},
           "reset_windows": {w: compare_window(base, arm, w) for w in plan["reset_windows"]}}
    bad = [w for w, c in out["equivalence"].items() if not all(c["byte_identical"].values())]
    bad += [w for w, c in out["reset_windows"].items() if not c["prefix"]["equal"]]
    for w in plan["equivalence_windows"] + plan["reset_windows"]:
        why = arm_source_problem(_read(arm / f"{w}_frames.json"), w, r, ws)
        if why or clip_changed(plan, r, w):
            bad.append(w)
            out.setdefault("source_or_clip_problems", {})[w] = why or "the clip changed since the plan"
    out["passes"] = not bad
    out["failing_windows"] = bad
    atomic_write(r / "resetfix" / f"equiv__{game}.json", json.dumps(out, indent=1))
    print(json.dumps({"game": game, "passes": out["passes"], "failing_windows": bad}))
    if bad:
        raise SystemExit(f"{WHO}: {game}: equivalence or prefix check fails on {bad}")
    return out


def cmd_cache_copy(args) -> dict:
    """System python, no lock: copy a window's scorecard cache from the P8 workspace into the reference root when the
    two clips are byte-identical and the reference root has none (the cache depends only on the clip and frozen
    models); the score step computes whatever is still missing."""
    import shutil
    from tools.p8_common import atomic_write
    game = check_game(args.game)
    r = ref_root()
    ws = Path(args.workspace).expanduser() if args.workspace else r.parent / "work"
    plan = _read(r / "resetfix" / f"plan__{game}.json")
    if plan is None:
        raise SystemExit(f"{WHO}: no plan for {game}")
    dst = r / "cache" / "scorecard"
    dst.mkdir(parents=True, exist_ok=True)
    out = {"game": game, "copied": [], "present": [], "refused": {}}
    for s in sorted(plan["baseline_resets"]):
        if (dst / f"{s}.json").exists():
            out["present"].append(s)
            continue
        src = ws / "cache" / "scorecard" / f"{s}.json"
        if not src.exists():
            out["refused"][s] = "no workspace cache"
        elif sha256_file(ws / "label_build" / f"{s}.mp4") != sha256_file(r / "label_build" / f"{s}.mp4"):
            out["refused"][s] = "clips differ"
        else:
            shutil.copyfile(src, dst / f"{s}.json")
            out["copied"].append(s)
    atomic_write(r / "resetfix" / f"cache_copy__{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}))
    return out


def cmd_baseline_copy(args) -> dict:
    """System python, no lock: the unfixed baseline in builds/r23_nofix, copied from the workspace's reference-platform
    r23 builds (tools.p8_placement.BUILDS) for windows that pass baseline_problems. Never builds anything."""
    import shutil
    from tools.p8_common import atomic_write
    from tools.p8_placement import BUILDS
    game = check_game(args.game)
    r = ref_root()
    ws = Path(args.workspace).expanduser() if args.workspace else r.parent / "work"
    src, dst = ws / "builds" / BUILDS[game], r / "builds" / BASE_DIR
    dst.mkdir(parents=True, exist_ok=True)
    out = {"game": game, "source": f"builds/{BUILDS[game]} of the P8 workspace", "copied": [], "present": [],
           "refused": {}}
    for s in label_stems(r, game):
        if (dst / f"{s}_frames.json").exists() or (dst / f"{s}_gate_empty.json").exists():
            why = baseline_problems(s, dst, ws, r)
            if why:
                raise SystemExit(f"{WHO}: {s} is in {dst.name} but fails the checks ({why})")
            out["present"].append(s)
            continue
        why = baseline_problems(s, src, ws, r)
        if why:
            out["refused"][s] = why
            continue
        files = sorted(p for p in src.glob(f"{s}_*") if p.name != f"{s}_frames.json")
        for p in files:
            shutil.copy2(p, dst / p.name)
        if (src / f"{s}_frames.json").exists():
            shutil.copy2(src / f"{s}_frames.json", dst / f"{s}_frames.json")    # last: its presence marks the window
        out["copied"].append(s)
    out["label_windows"] = len(label_stems(r, game))
    out["complete"] = not out["refused"] and len(out["copied"]) + len(out["present"]) == out["label_windows"]
    atomic_write(r / "builds" / f"{BASE_DIR}_source_{game}.json", json.dumps(out, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}))
    return out


def truth_files(r: Path, game: str) -> dict:
    files = sorted((r / "labels" / game).glob(f"{game}_s*.jsonl")) + [r / "labels" / f"{game}_result.json",
                                                                       r / "labels" / f"{game}_offsets.json"]
    return {p.name: sha256_file(p) for p in files if p.exists()}


def cmd_pin_truth(args) -> dict:
    """System python: pin a game's truth (sha256 of every label file and the fit result) with its final status, read
    from the reference root's acceptance record. Run before any score; a score refuses truth that changed."""
    from tools.p8_common import atomic_write
    game = check_game(args.game)
    r = ref_root()
    acc = _read(r / "truth_accept" / f"{game}.json")
    if acc is None or acc.get("use") not in FINAL_TRUTH:
        raise SystemExit(f"{WHO}: {game} has no final acceptance record; its truth status is not final")
    labels = sorted(p.name[:-len("_frames.json")] for p in (r / "label_build").glob(f"{game}_s*_frames.json"))
    why = label_fix_problems({w: _read(r / "label_build" / f"{w}_device.json") for w in labels})
    mt = lambda p: p.stat().st_mtime if p.exists() else None
    builds = [t for w in labels for t in (mt(r / "label_build" / f"{w}_frames.json"),
                                          mt(r / "label_build" / f"{w}_device.json")) if t is not None]
    tfiles = {p.name: mt(p) for p in [*(r / "labels" / game).glob(f"{game}_s*.jsonl"), r / "labels" / f"{game}_result.json",
                                      r / "labels" / f"{game}_offsets.json", r / "truth_accept" / f"{game}.json"]}
    why += pin_order_problems(builds, tfiles, acc.get("label_windows"), len(labels))
    if why:
        raise SystemExit(f"{WHO}: {game}: {why}")
    pin = {"game": game, "status": acc["use"], "label_windows_with_fix": len(labels),
           "acceptance_checks": {k: (v or {}).get("pass") for k, v in (acc.get("checks") or {}).items()},
           "files": truth_files(r, game)}
    atomic_write(r / "resetfix" / f"truth_pin__{game}.json", json.dumps(pin, indent=1))
    print(json.dumps({k: v for k, v in pin.items() if k != "files"} | {"files": len(pin["files"])}))
    return pin


def score_pair(P: dict, game: str, w: dict, dirs: dict, sides: dict, ctx: dict) -> dict:
    """One window, both arms, through the pinned score_window and window_stats, plus the detection, one-to-one and
    duplicate counts on the same scored frames (frozen CV python)."""
    import cv2
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import apply_mirror
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    import copy
    s = w["window"]
    pair, scored_f = {}, {}
    for k, dd in dirs.items():
        if k == "new" and dd == dirs["r23"]:          # a no-reset window: the arm is the baseline, scored once
            pair[k], scored_f[k] = copy.deepcopy(pair["r23"]), scored_f["r23"]
            continue
        full = S.score_window(P, game, w, dd, "camera", ctx["mom"], ctx["mirror"], ctx["refs"], ctx["events"])
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} {k} skipped: {full[0]['skipped']}")
        scored_f[k] = {int(o["f"]) for o in full[1]}
        pair[k] = window_stats(full[0])
    if pair["r23"]["in_frame"] != pair["new"]["in_frame"]:
        raise SystemExit(f"{WHO}: {s}: in-frame players differ between arms")
    truth, _ = S.truth_rows(P, game, w, "camera", ctx["mom"].index, ctx["mirror"])
    cap = cv2.VideoCapture(str(P["label_build"] / f"{s}.mp4"))
    wd, hd = int(cap.get(3)), int(cap.get(4))
    cap.release()
    for k in dirs:
        srow = {int(x["frame"]): x for x in sides[k]["frames"]}
        c = {"far": 0, "missed_1to1": 0, "duplicates": 0}
        for row in truth:
            x = srow.get(int(row["frame"]))
            if x is None or int(row["frame"]) not in scored_f[k]:
                continue
            sv = apply_mirror(ctx["mom"].index.q[row["q"]]["xy"][row["moment"]], ctx["fx"], ctx["fy"])
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < wd - m) & (px[:, 1] > m) & (px[:, 1] < hd - m)
            feet = np.array([b["foot_stab"] for b in x["boxes"]], np.float32).reshape(-1, 1, 2)
            ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2).astype(np.float64) if len(feet) else np.zeros((0, 2))
            _add(c, frame_extra(ft, sv, inside))
        pair[k].update(c)
    return pair


def cmd_score(args) -> dict:
    """Frozen CV python, inside the lock: baseline against the fixed arm on the game's scored windows, on the pinned
    truth. A no-reset window of the arm resolves to the baseline's files (read only)."""
    import os
    import shutil
    import time
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    from tools.p8_common import atomic_write
    from tools.p8_e1r import check_frames
    game = check_game(args.game)
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = ref_root()
    d = r / "resetfix"
    for g in POOL + ((APART,) if game == APART else ()):      # nothing is scored until every pool truth is final
        why = truth_pin_problems(_read(d / f"truth_pin__{g}.json"), truth_files(r, g))
        if why:
            raise SystemExit(f"{WHO}: truth of {g}: {why}")
    if game == APART:
        why = stage_allowed("hou_score", d)
        if why:
            raise SystemExit(f"{WHO}: {game} is scored only after the pool passes: {why}")
    if _read(d / f"truth_pin__{game}.json")["status"] != "use":
        raise SystemExit(f"{WHO}: {game}'s truth is not in use (tools.p8_truth_accept); it is not scored")
    plan = _read(d / f"plan__{game}.json")
    eq = _read(d / f"equiv__{game}.json")
    if not eq or not eq.get("passes"):
        raise SystemExit(f"{WHO}: {game}: the equivalence and prefix checks have not passed")
    built = set(plan["reset_windows"])
    g = F.game_of(game)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, game)
    fx, fy = MIRRORS[mirror]
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    base, arm = P["builds"] / BASE_DIR, P["builds"] / arm_dir(game)
    ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": events, "fx": fx, "fy": fy}
    ws = Path(args.workspace).expanduser() if args.workspace else r.parent / "work"
    common = r / "tmp" / "resetfix_common_labels"
    (common / game).mkdir(parents=True, exist_ok=True)
    for extra_name in (f"{game}_offsets.json", f"{game}_result.json"):
        if (P["labels"] / extra_name).exists():
            shutil.copyfile(P["labels"] / extra_name, common / extra_name)
    Pc = {**P, "labels": common}
    t0 = time.time()
    wins, wins_common, extra, cache_s = {}, {}, {}, {}
    for w in F.label_windows(P, game):
        s = w["window"]
        if s in plan["gate_empty_windows"] or s not in plan["baseline_resets"]:
            continue
        dirs = {"r23": base, "new": arm if s in built else base}
        bad = check_base_records(_read(base / f"{s}_detector.json"), _read(base / f"{s}_device.json"))
        if s in built:
            ident = _read(arm / f"{s}_identity.json")["identity"]
            bad += check_arm_records(_read(arm / f"{s}_detector.json"), _read(arm / f"{s}_device.json"),
                                     _read(arm / f"{s}_resetfix.json"), int(ident.get("camera_cut_resets") or 0))
        sides = {k: _read(dd / f"{s}_frames.json") for k, dd in dirs.items()}
        bad += check_frames(sides["r23"], sides["new"])
        if {f: sha256_file(base / f"{s}_{f}.json") for f in SIDE_FILES} != plan["baseline_sha256"][s]:
            bad.append("baseline files changed since the plan")
        if clip_changed(plan, r, s):
            bad.append("the clip changed since the plan")
        if s in built:
            why = arm_source_problem(sides["new"], s, r, ws)
            if why:
                bad.append(why)
        if bad:
            raise SystemExit(f"{WHO}: {s}: {bad}")
        had_cache = (P["cache"] / "scorecard" / f"{s}.json").exists()
        tc = time.time()
        wins[s] = score_pair(P, game, w, dirs, sides, ctx)
        if not had_cache:
            cache_s[s] = round(time.time() - tc, 1)
        extra[s] = {"built": s in built}
        lab = _read(P["label_build"] / f"{s}_frames.json")       # report only: the arm against the fixed label build
        extra[s]["label_build_consistency"] = None if lab is None else \
            boxes_consistency(sides["new"]["frames"], lab["frames"])
        # the bracket (report only): the same pair on the frames the bug-carrying workspace truth also accepted
        old = ws / "labels" / game / f"{s}.jsonl"
        if old.exists() and sha256_file(ws / "label_build" / f"{s}.mp4") == sha256_file(P["label_build"] / f"{s}.mp4"):
            keep = {int(json.loads(x)["frame"]) for x in old.read_text().splitlines() if x.strip()}
            new_rows = (P["labels"] / game / f"{s}.jsonl").read_text().splitlines() \
                if (P["labels"] / game / f"{s}.jsonl").exists() else []
            (common / game / f"{s}.jsonl").write_text(
                "".join(x + "\n" for x in new_rows if x.strip() and int(json.loads(x)["frame"]) in keep))
            wins_common[s] = score_pair(Pc, game, w, dirs, sides, ctx)
            extra[s]["common_source_sha256"] = sha256_file(old)
        else:
            extra[s]["common"] = "left out (no workspace truth or the clips differ)"
    out = {"game": game, "truth": "camera (pinned, reset-fix label builds)", "truth_pin": _read(d / f"truth_pin__{game}.json"),
           "platform": "detector cpu (D175)", "arms": {"r23": BASE_DIR, "new": arm_dir(game)},
           "built_windows": sorted(built), "resolved_to_baseline": sorted(set(wins) - built),
           "expected_windows": sorted(plan["baseline_resets"]), "scored_windows": sorted(wins),
           "gate_empty_windows": plan["gate_empty_windows"], "windows": wins, "windows_common": wins_common,
           "window_notes": extra, "cache_seconds": cache_s,
           "seconds": round(time.time() - t0, 1)}
    atomic_write(d / f"score__{game}.json", json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k not in ("windows", "windows_common", "expected_windows",
                                                               "scored_windows", "truth_pin", "window_notes")},
                     indent=1))
    return out


# report (system python)
def unit_eval(wins: list, picks_w: np.ndarray, picks_g: np.ndarray | None = None, groups: list | None = None) -> dict:
    """Point differences and intervals of every row for one unit: the registered rows through tools.p8_e1_score's
    rows, diffs and boot (window interval; game interval when groups are given), and the detection and one-to-one
    rows through ratio_interval with the same resample indices."""
    from tools.p8_e1_score import boot, diffs, rows
    r = rows(wins)
    point = diffs(wins)
    wiv = boot({i: [w] for i, w in enumerate(wins)})
    giv = boot({i: g for i, g in enumerate(groups)}) if groups else None
    for row, num in (("detection_rate", "far"), (ONE_TO_ONE_ROW, "missed_1to1")):
        bn = [w["r23"][num] for w in wins]; nn = [w["new"][num] for w in wins]
        bd = [w["r23"]["in_frame"] for w in wins]; nd = [w["new"]["in_frame"] for w in wins]
        point[row] = sum(nn) / sum(nd) - sum(bn) / sum(bd)
        r[row] = (sum(bn) / sum(bd), sum(nn) / sum(nd))
        wiv[row] = ratio_interval(picks_w, bn, bd, nn, nd)
        if groups:
            G = [[sum(w[s][k] for w in grp) for s in ("r23", "new") for k in (num, "in_frame")] for grp in groups]
            giv[row] = ratio_interval(picks_g, [x[0] for x in G], [x[1] for x in G], [x[2] for x in G], [x[3] for x in G])
    return {"rows": r, "point": point, "window_interval": wiv, "game_interval": giv}


def report_only(wins: list) -> dict:
    """Boxes, duplicates and ghosts per in-frame player, and the position error of players matched only in the arm."""
    out = {}
    for side in ("r23", "new"):
        n = sum(w[side]["in_frame"] for w in wins) or None
        out[side] = None if n is None else {"boxes_per_in_frame": sum(w[side]["nr_boxes"] for w in wins) / n,
                                            "duplicates_per_in_frame": sum(w[side]["duplicates"] for w in wins) / n,
                                            "ghosts_per_in_frame": sum(w[side]["nr_ghosts"] for w in wins) / n}
    only = [w["new"]["matched"][k][0] for w in wins for k in w["new"]["matched"].keys() - w["r23"]["matched"].keys()]
    both = [w["r23"]["matched"][k][0] for w in wins for k in w["new"]["matched"].keys() & w["r23"]["matched"].keys()]
    q = lambda a: None if not a else {"n": len(a), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}
    out["arm_only_matches"] = q(only)
    out["matched_in_both_baseline_error"] = q(both)
    out["matched_only_in_baseline"] = sum(len(w["r23"]["matched"].keys() - w["new"]["matched"].keys()) for w in wins)
    return out


def cmd_report(args) -> dict:
    from tools.p8_common import REPO, atomic_write
    from tools.p8_placement import rnd
    r = ref_root()
    d = r / "resetfix"
    files = {g: _read(d / f"score__{g}.json") for g in GAMES}
    pins = {g: _read(d / f"truth_pin__{g}.json") for g in POOL}
    if any(p is None for p in pins.values()):
        raise SystemExit(f"{WHO}: not every pool game's truth is pinned; nothing is reported")
    have = [g for g in POOL if files[g] is not None]
    unscored = [g for g in POOL if files[g] is None]
    if any(pins[g]["status"] == "use" for g in unscored):
        raise SystemExit(f"{WHO}: pool games with truth in use are not scored yet: {unscored}")
    if len(have) < 2:
        raise SystemExit(f"{WHO}: fewer than two pool games have truth in use ({have}); no pool")
    out = {"per_game": {}, "pool": None, "hou": None, "partial": bool(unscored), "not_scored": unscored}
    per = {}
    for g in have:
        wins = list(files[g]["windows"].values())
        ev = unit_eval(wins, resample_picks(len(wins), 2000, 0))
        cov = ev["rows"]["coverage"]
        per[g] = v2_verdict(ev["point"], ev["window_interval"], cov, None, improve_rows=(), one_to_one=False)
        out["per_game"][g] = {**ev, "verdict": per[g], "report_only": report_only(wins)}
    groups = [list(files[g]["windows"].values()) for g in have]
    allw = [w for grp in groups for w in grp]
    ev = unit_eval(allw, resample_picks(len(allw), 2000, 0), resample_picks(len(groups), 2000, 0), groups)
    pooled = v2_verdict(ev["point"], ev["window_interval"], ev["rows"]["coverage"], ev["game_interval"])
    pv = pool_verdict(pooled, per)
    if unscored:                                   # a partial pool never passes (as E1R)
        pv = {**pv, "pass": False, "partial": True}
    # the bracket (version 2.2: a guard): the same rows on the frames the bug-carrying workspace truth also accepted
    br = {}
    for g in have:
        wc = list((files[g].get("windows_common") or {}).values())
        if len(wc) >= 2:
            e = unit_eval(wc, resample_picks(len(wc), 2000, 0))
            br[g] = {**e, "verdict": v2_verdict(e["point"], e["window_interval"], e["rows"]["coverage"], None,
                                                improve_rows=(), one_to_one=False)}
    gc = [list((files[g].get("windows_common") or {}).values()) for g in have]
    if all(len(x) >= 1 for x in gc):
        ac = [w for x in gc for w in x]
        e = unit_eval(ac, resample_picks(len(ac), 2000, 0), resample_picks(len(gc), 2000, 0), gc)
        br["pool"] = {**e, "verdict": v2_verdict(e["point"], e["window_interval"], e["rows"]["coverage"],
                                                 e["game_interval"])}
    out["bracket_common_frames"] = br
    bp = br.get("pool")
    bracket_ok = bool(bp) and bp["point"]["missed_rate"] is not None and bp["point"]["missed_rate"] < 0
    biv = (bp or {}).get("window_interval", {}).get("missed_rate")
    out["bracket_guard"] = {"pooled_missed_point_below_zero": bracket_ok,
                            "pooled_missed_interval_below_zero": bool(biv and biv[1] < 0)}
    if not bracket_ok:                              # the guard: the pool passes only if the bracket's missed improves
        pv = {**pv, "pass": False, "bracket_guard_failed": True}
    out["adoption"] = adoption(pv, pooled, bracket_ok)
    miv = ev["window_interval"]["missed_rate"]
    label = None
    if pv["pass"]:
        label = "the fix reduces missed players on the pool" if (miv and miv[1] < 0 and biv and biv[1] < 0) else \
            "the fix does not harm and points to fewer misses"
    elif out["adoption"]["adopt"]:
        label = "adopted as a correctness fix; no claim of no harm on ghosts (ghost cost reported per game)"
    out["pool"] = {**ev, "verdict": pv, "pooled_verdict": pooled, "label": label, "report_only": report_only(allw),
                   "ghost_cost_by_game": {g: out["per_game"][g]["point"]["ghost_rate"] for g in have}}
    if files[APART] is not None and pv["pass"]:        # hou is reported only after a pool pass
        wins = list(files[APART]["windows"].values())
        evh = unit_eval(wins, resample_picks(len(wins), 2000, 0))
        vh = v2_verdict(evh["point"], evh["window_interval"], evh["rows"]["coverage"], None,
                        improve_rows=("missed_rate",), one_to_one=False)
        out["hou"] = {**evh, "verdict": vh, "report_only": report_only(wins),
                      "label": "the fix holds on the new arena" if vh["pass"] else
                      "the pool result does not carry to the new arena"}
    cons = {}
    for g in GAMES:
        if files[g] is None:
            continue
        c = {}
        for note in (files[g].get("window_notes") or {}).values():
            if note.get("label_build_consistency"):
                _add(c, note["label_build_consistency"])
        cons[g] = c
    out["label_build_consistency"] = cons
    out["resets"] = {g: {w: {"baseline": _read(d / f"plan__{g}.json")["baseline_resets"][w],
                             "arm": (_read(d / f"equiv__{g}.json") or {}).get("reset_windows", {}).get(w, {})
                             .get("resets_arm")} for w in _read(d / f"plan__{g}.json")["reset_windows"]}
                     for g in GAMES if files[g] is not None}
    res = rnd(out)
    atomic_write(d / "report.json", json.dumps(res, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_resetfix.json", json.dumps(res, indent=1))
    print(json.dumps({"pool": {"pass": pv["pass"], "failing": pv["failing"], "label": label},
                      "hou": None if out["hou"] is None else out["hou"]["verdict"]["pass"]}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_resetfix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    t = sub.add_parser("truthreset")
    t.add_argument("--ref-root", default=None, help="the reference root (default: work_ref beside the P8 workspace)")
    m = sub.add_parser("simulate")
    m.add_argument("--draws", type=int, default=SIM_DRAWS)
    m.add_argument("--boot", type=int, default=2000)
    m.add_argument("--version", choices=("2", "2.2"), default="2")
    for name in ("plan", "build", "equiv", "pin-truth", "score", "cache-copy", "baseline-copy"):
        x = sub.add_parser(name)
        x.add_argument("--game", required=True)
        if name == "build":
            x.add_argument("--smoke", action="store_true")
        if name in ("score", "cache-copy", "baseline-copy"):
            x.add_argument("--workspace", default=None, help="the P8 workspace (default: work beside the root)")
    sub.add_parser("stage0-check")
    sc = sub.add_parser("smoke-check")
    sc.add_argument("--workspace", default=None, help="the P8 workspace holding the a3input record (default: work)")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"estimate": cmd_estimate, "truthreset": cmd_truthreset, "simulate": cmd_simulate, "plan": cmd_plan,
     "build": cmd_build, "equiv": cmd_equiv, "pin-truth": cmd_pin_truth, "score": cmd_score,
     "cache-copy": cmd_cache_copy, "baseline-copy": cmd_baseline_copy,
     "smoke-check": cmd_smoke_check, "report": cmd_report, "stage0-check": cmd_stage0_check}[a.cmd](a)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        sys.exit(child(sys.argv[2:]))
    main()
