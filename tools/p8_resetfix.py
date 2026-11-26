"""tools/p8_resetfix.py: the tracker reset fix (docs/research/P8_resetfix.md). This module holds the pure fix used by
tools/p8_device.py's build child and a stored-files estimate for the design. Nothing here builds or trains.

    python3 -m tools.p8_resetfix estimate   -> <P8 workspace>/placement/resetfix_estimate.json and
                                               reports/research/p8_resetfix_estimate.json   (stored files only, no lock)
    python3 -m tools.p8_resetfix truthreset -> reports/research/p8_resetfix_truthreset.json   (design v2 measurement a)
    python3 -m tools.p8_resetfix simulate   -> reports/research/p8_resetfix_simulate.json     (design v2 measurement b)

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


RESET_LOG = {"path": None, "resets": []}


def apply_reset_fix() -> str:
    """In the build child, before the build runs: PlayerTracker.reset also strips the tracker callbacks. Each reset
    appends {"stripped": n} to RESET_LOG (n = 2 when the tracker had registered once: one on_predict_start and one
    on_predict_postprocess_end); write_reset_log() saves it at exit when start_reset_log() named a file."""
    import detect.tracker as DT
    orig = DT.PlayerTracker.reset

    def reset(self):
        orig(self)
        cbs = getattr(getattr(self, "model", None), "callbacks", None)
        RESET_LOG["resets"].append({"stripped": strip_tracker_callbacks(cbs) if isinstance(cbs, dict) else None})
    DT.PlayerTracker.reset = reset
    return "detect.tracker.PlayerTracker.reset"


def write_reset_log() -> None:
    import os
    path = RESET_LOG["path"]
    if path is None:
        return
    part = f"{path}.part"
    with open(part, "w") as fh:
        json.dump({"reset_fix": True, "resets": RESET_LOG["resets"]}, fh)
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
    est = json.loads((REPO / "reports" / "research" / "p8_resetfix_estimate.json").read_text())["games"]
    base, meta = {}, {}
    for g in POOL:
        sf = json.loads((wr / "e1" / SCORES[g]).read_text())
        pw = est[g]["per_window"]
        q = est[g]["rates"]["after_reset_excess_ghost_share"]
        meta[g] = {"q": q}
        base[g] = []
        for w in sf["scored_windows"]:
            st = dict(sf["windows"][w][SIDE[g]])
            if st["in_frame"] != pw[w]["all"]["in_frame"] or st["missed"] != pw[w]["all"]["missed"]:
                raise SystemExit(f"{WHO}: {w}: estimate and score file disagree")
            st["far"] = pw[w]["all"]["far"]
            base[g].append((w, st, pw[w]["after"], pw[w]["resets"]))

    def build(f, rng):
        return {g: [{"r23": st, "new": sim_arm(st, aft, rs, f, meta[g]["q"], rng)} for (_w, st, aft, rs) in base[g]]
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
        draws.append({"f": f, "pass": v["pass"], "pooled_pass": v["pooled"], "games": v["games"],
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
    res = rnd({"what": "Simulated reset-fix arm under the design v2 rule (docs/research/P8_resetfix.md version 2, "
                       "measurement b); a forecast on the MPS-truth scored frames, not a result",
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
    atomic_write(REPO / "reports" / "research" / "p8_resetfix_simulate.json", json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("pass_probability", "pooled_pass_probability", "game_pass_probability",
                                          "missed_interval_below_zero_probability", "failing_conditions", "forecast")},
                     indent=1))
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
    a = ap.parse_args(argv)
    {"estimate": cmd_estimate, "truthreset": cmd_truthreset, "simulate": cmd_simulate}[a.cmd](a)


if __name__ == "__main__":
    main()
