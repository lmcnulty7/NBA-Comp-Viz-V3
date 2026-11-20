"""tools/research_epv.py: the EPV research experiment epv-E1 (PLAN.md D70, D116, D117).

    python3 -m export.heavy --name epv-E1 -- python3 -m tools.research_epv e1-run [--max-minutes 26] [--xfg-artifacts DIR]
    python3 -m export.heavy --name epv-E1 -- python3 -m tools.research_epv e1-dev

Pre-registration: docs/research/epv.md, section 6, "epv-E1", with its D117 amendment: development decides; Set A is
not scored whatever development shows, and no Set A path exists in this module (no scorer, no test read).

e1-run   Resumable pieces, one lock job at a time (each job stops before a piece that would pass --max-minutes):
         xfg_<k>  the xFG labels of EPV fold k: the frozen xFG recipe (tools.research_xfg.fit_parts, gated in
                  xfg-E1 against the stored out-of-fold predictions) refit out of fold over the training games of EPV
                  fold k (5 grouped folds by game, seed 0), so the probability on a training row's shot comes from a
                  model that never saw that row's game nor any game of EPV fold k's test games (the pre-registered
                  leakage guard). Records the log loss of those probabilities (gate B).
         fit_<arm>_<k>  one GRU fit of an arm on EPV fold k (the frozen outer folds: analytics.eval.grouped_folds over
                  the training games, 5 folds, seed 0) and its predictions on the fold's test states.
         Order: the five baseline fits (gate A), the five xFG pieces (gate B), then e1 and three_only.
e1-dev   Assembles reports/research/epv-E1-dev.json from the pieces, refusing unless gates A and B passed.

Definitions, fixed in this file before any development number (the pre-registration leaves them open):
- Arms. baseline: the frozen calibrated GRU (analytics.epv.train.Calibrated, slot mlp, deep torch, seed 0) on the
  frozen label points_to_end, exactly as the frozen out_of_fold fits it. e1: the same fit with the training label
  replaced as pre-registered. three_only (the ablation): only 3-point attempts smoothed.
- The label. A possession is smoothed when it ends on its final field-goal attempt (end_kind shot_made or shot_missed;
  an and-one or a shooting foul ends as free_throws and is not smoothed) and that attempt has an xFG row. For each of
  its states strictly before the attempt (state game clock above the attempt's clock), label = points_to_end - the
  attempt's realized points + p * value (value 3 for a three, else 2); free-throw points stay realized. Every other
  state keeps points_to_end.
- The calibration step. The frozen Calibrated fits its isotonic map from expected points to P(points > 0) on the
  training label. Under a smoothed label "points > 0" is almost always true, so for e1 and three_only the regressor
  (and its inner out-of-fold predictions) is trained on the label while the isotonic map is fitted on the realized
  points_to_end > 0 of the same rows, and p_const is the realized share. Evaluation is always against realized
  points.
- Metric (pre-registered): MSE of expected points against realized points_to_end, out of fold; the paired difference
  MSE(baseline) - MSE(e1) with a 2000-resample bootstrap of games (seed 0); the calibration slope of P(points > 0)
  against realized points > 0 in the 0 to 4 s elapsed bucket (analytics.eval.calibration_slope_fast). Verdict: the
  difference's interval above 0 and e1's 0 to 4 s slope inside 0.9 to 1.1. Secondary, reported: Brier and slope per
  bucket, the ablation's difference.
- Gate A: the baseline fits reproduce the stored out-of-fold epv_points of every training state (fold >= 0) within
  GATE_A_TOL. Gate B: the log loss of the nested xFG probabilities of each EPV fold lies inside the frozen xFG
  out-of-fold interval [0.6277, 0.6316] (reports/xfg.md). Either failing stops the experiment.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, artifacts_dir, assert_safe_output
from tools.research_testset import dev_games, refuse_test

WHO = "tools.research_epv"
EXPERIMENT = "epv-E1"
RESULTS = REPO_ROOT / "reports" / "research"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
ARMS = ("baseline", "e1", "three_only")
FROZEN_RECORD = {"train_states": 1739119, "train_games": 604}
XFG_OOF_INTERVAL = (0.6277, 0.6316)       # reports/xfg.md, out-of-fold log loss 0.6294 [0.6277, 0.6316]
GATE_A_TOL = 1e-5
SLOPE_LO, SLOPE_HI = 0.9, 1.1
FIT_MINUTES = 13.0                        # a piece is not started when fewer minutes than this are left
SMOOTH_KINDS = ("shot_made", "shot_missed")


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / "epv_e1")


def pieces_dir(artifacts=None) -> Path:
    return work_dir(artifacts) / "pieces"


# ----------------------------------------------------------------------------- data

class Data:
    """The frozen training tables (the frozen loader and split), the frozen outer folds, built once per job."""

    def __init__(self):
        from analytics.eval import grouped_folds
        from tools import frozen_epv
        root = Path(os.environ["DATA_ROOT"])
        box = frozen_epv._load_training(root)
        counts = box["counts"]
        wrong = {k: (counts[k], v) for k, v in FROZEN_RECORD.items() if counts[k] != v}
        if wrong:
            raise SystemExit(f"{WHO}: the training inputs are not the frozen record's (found, expected): {wrong}")
        self.states, self.shots, self.rosters = box["states"], box["shots"], box["rosters"]
        self.games_all = self.states["sportvu_game"].astype(str).to_numpy()
        uniq = sorted(set(self.games_all))
        if dev_games(uniq, WHO) != uniq:
            raise SystemExit(f"{WHO}: a training game is not a development game")
        self.folds = grouped_folds(uniq, n_folds=FOLDS, seed=SEED)
        self.root = root
        self._geo = None

    def geo(self):
        from analytics import schema
        from analytics.epv import features as F
        if self._geo is None:
            self._geo = F.geometry(self.states)
            schema.assert_time_forward(self._geo, "states_1hz")
        return self._geo

    def fold_rows(self, k: int):
        import numpy as np
        tr_g, te_g = self.folds[k]
        return np.isin(self.games_all, tr_g), np.isin(self.games_all, te_g)

    def fold_features(self, k: int):
        """The frozen fold features (priors from the fold's training games only) and their lagged frames."""
        from analytics.epv import features as F
        from analytics.epv import train as T
        tr, te = self.fold_rows(k)
        geo = self.geo()
        X_tr, X_te, _ = T.fold_features(geo.loc[tr], geo.loc[te], self.states.loc[tr], self.shots, self.rosters)
        return F.lagged(X_tr, T.N_LAGS), F.lagged(X_te, T.N_LAGS)


# ----------------------------------------------------------------------------- the label

def final_attempts(root: Path, games, shot_rows):
    """Per possession that ends on its final field-goal attempt (SMOOTH_KINDS): that attempt's event, clock, value
    and realized points, from the possessions tables and the xFG shot rows (sportvu_game, event, period, clock_s,
    is_three, made)."""
    import pandas as pd
    poss = pd.concat([pd.read_parquet(root / "analytics" / "possessions" / SEASON / f"{g}.parquet",
                                      columns=["sportvu_game", "possession", "period", "start_event", "end_event",
                                               "end_kind"]) for g in games], ignore_index=True)
    poss["sportvu_game"] = poss["sportvu_game"].astype(str)
    poss = poss[poss["end_kind"].astype(str).isin(SMOOTH_KINDS)]
    s = shot_rows[["sportvu_game", "event", "period", "clock_s", "is_three", "made"]].copy()
    s["sportvu_game"] = s["sportvu_game"].astype(str)
    m = poss.merge(s, on=["sportvu_game", "period"], how="inner")
    m = m[(m["event"] >= m["start_event"]) & (m["event"] <= m["end_event"])]
    last = m.sort_values("event").groupby(["sportvu_game", "possession"], as_index=False).tail(1)
    last = last.assign(value=last["is_three"].astype(bool).map({True: 3, False: 2}).astype(int))
    last["realized"] = last["made"].astype(int) * last["value"]
    return last[["sportvu_game", "possession", "event", "clock_s", "value", "realized", "is_three"]].reset_index(drop=True)


def smoothed_label(states, attempts, pmap: dict, three_only: bool = False):
    """The arm's training label for every state (numpy, states' order) and counts."""
    import numpy as np
    y = states["points_to_end"].to_numpy(dtype=float).copy()
    a = attempts.copy()
    a["p"] = [pmap.get((g, int(e)), np.nan) for g, e in zip(a["sportvu_game"], a["event"])]
    a = a[a["p"].notna()]
    if three_only:
        a = a[a["is_three"].astype(bool)]
    if len(a) == 0:
        return y, {"states_smoothed": 0, "possessions_smoothed": 0}
    key = dict(zip(zip(a["sportvu_game"], a["possession"].astype(int)), range(len(a))))
    idx = np.array([key.get((g, int(p)), -1) for g, p in zip(states["sportvu_game"].astype(str), states["possession"])])
    has = idx >= 0
    clock = a["clock_s"].to_numpy(dtype=float)
    before = has & (states["game_clock_s"].to_numpy(dtype=float) > np.where(has, clock[np.maximum(idx, 0)], np.inf))
    j = idx[before]
    y[before] = y[before] - a["realized"].to_numpy(dtype=float)[j] + a["p"].to_numpy(dtype=float)[j] * \
        a["value"].to_numpy(dtype=float)[j]
    return y, {"states_smoothed": int(before.sum()), "possessions_smoothed": int(len(a))}


# ----------------------------------------------------------------------------- the xFG labels of a fold

def xfg_piece(d: Data, k: int, xfg_artifacts) -> dict:
    """Nested out-of-fold xFG probabilities on the shots of EPV fold k's training games, and their log loss."""
    import numpy as np
    import pandas as pd
    from analytics.eval import grouped_folds
    from tools import research_xfg as X
    feats = pd.read_parquet(X.features_path(xfg_artifacts))
    feats["sportvu_game"] = feats["sportvu_game"].astype(str)
    refuse_test(feats["sportvu_game"].unique(), WHO)
    tr_g, te_g = d.folds[k]
    rows = feats[feats["sportvu_game"].isin(set(tr_g))].reset_index(drop=True)
    if rows["sportvu_game"].isin(set(te_g)).any():
        raise SystemExit(f"{WHO}: an EPV test game reached its fold's xFG fit")
    p = np.full(len(rows), np.nan)
    for _a, inner_te in grouped_folds(sorted(rows["sportvu_game"].unique()), n_folds=FOLDS, seed=SEED):
        m = rows["sportvu_game"].isin(set(inner_te)).to_numpy()
        parts = X.fit_parts(rows[~m].reset_index(drop=True), SEED)
        p[m] = X.predict_arms(parts, rows[m].reset_index(drop=True))["p_iso"]
    y = rows["made"].astype(int).to_numpy()
    ll = float(X.per_shot_loss(y, p).mean())
    out = rows[["sportvu_game", "event", "period", "clock_s", "is_three", "made"]].copy()
    out["p"] = p
    return {"frame": out, "log_loss": ll, "shots": int(len(rows)), "games": int(rows["sportvu_game"].nunique())}


# ----------------------------------------------------------------------------- the GRU fits

def fit_label(name, columns, X, y_label, y_real, games):
    """The frozen Calibrated fit with the regressor trained on y_label and the isotonic map on realized points > 0
    (see the module docstring). For the baseline y_label is y_real and this equals the frozen fit."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from analytics import schema
    from analytics.epv import train as T
    from analytics.eval import grouped_folds
    c = T.Calibrated(name, columns, SEED, 100, "torch")
    y_label, y_real = np.asarray(y_label, dtype=float), np.asarray(y_real, dtype=float)
    games = np.asarray(games).astype(str)
    schema.assert_time_forward(X, "states_1hz")
    c.p_const = float(np.mean(y_real > 0))
    uniq = np.unique(games)
    oof = np.full(len(y_label), np.nan)
    for tr_g, te_g in grouped_folds(uniq, n_folds=min(T.INNER_FOLDS, len(uniq)), seed=SEED):
        tr, te = np.isin(games, tr_g), np.isin(games, te_g)
        oof[te] = c._predict_raw(c._fit_raw(X.loc[tr], y_label[tr]), X.loc[te], c.columns)
    c.model = c._fit_raw(X, y_label)
    if len(uniq) < 2:
        oof = c._predict_raw(c.model, X, c.columns)
    if len(np.unique(y_real > 0)) == 2 and np.nanstd(oof) > 1e-9:       # the frozen condition, on realized points
        c.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, (y_real > 0).astype(float))
    else:
        c.iso = None
    return c


def fit_piece(d: Data, arm: str, k: int, feats, labels: dict) -> dict:
    import numpy as np
    from analytics.epv import features as F
    from analytics.epv import train as T
    X_tr, X_te = feats
    tr, te = d.fold_rows(k)
    y_real = d.states["points_to_end"].to_numpy(dtype=float)
    cols = F.lagged_columns(T.N_LAGS)
    if arm == "baseline":
        c = T.Calibrated(T.SEQUENCE, cols, SEED, 100, "torch").fit(X_tr, y_real[tr], d.games_all[tr])
    else:
        c = fit_label(T.SEQUENCE, cols, X_tr, labels[arm][tr], y_real[tr], d.games_all[tr])
    epv, p = c.predict(X_te)
    return {"rows": np.flatnonzero(te), "epv": np.asarray(epv, dtype=float), "p": np.asarray(p, dtype=float)}


def labels_for(d: Data, k: int, artifacts) -> dict:
    """The e1 and three_only labels of every state from fold k's nested xFG probabilities (only fold k's training
    rows are ever used)."""
    import pandas as pd
    folder = pieces_dir(artifacts)
    shots = pd.read_parquet(folder / f"xfg_{k}.parquet")
    pmap = {(g, int(e)): float(p) for g, e, p in zip(shots["sportvu_game"], shots["event"], shots["p"])}
    tr_g, _te = d.folds[k]
    attempts = final_attempts(d.root, sorted(tr_g), shots)
    e1, c1 = smoothed_label(d.states, attempts, pmap)
    th, c3 = smoothed_label(d.states, attempts, pmap, three_only=True)
    return {"e1": e1, "three_only": th, "counts": {"e1": c1, "three_only": c3, "attempts": int(len(attempts))}}


# ----------------------------------------------------------------------------- the resumable run

def plan() -> list:
    return ([("fit", "baseline", k) for k in range(FOLDS)] + [("xfg", None, k) for k in range(FOLDS)]
            + [("fit", arm, k) for k in range(FOLDS) for arm in ("e1", "three_only")])


def piece_path(folder: Path, kind: str, arm, k: int) -> Path:
    return folder / (f"xfg_{k}.json" if kind == "xfg" else f"fit_{arm}_{k}.npz")


def gate_a(d: Data, folder: Path) -> dict:
    """The baseline out-of-fold predictions against the stored out-of-fold epv_points (fold >= 0)."""
    import numpy as np
    import pandas as pd
    games = sorted(set(d.games_all))
    stored = pd.concat([pd.read_parquet(d.root / "analytics" / "epv" / SEASON / f"{g}.parquet",
                                        columns=["sportvu_game", "possession", "frame", "epv_points", "fold"])
                        for g in games], ignore_index=True)
    stored = stored[stored["fold"] >= 0]
    pred = np.full(len(d.states), np.nan)
    fold = np.full(len(d.states), -1)
    for k in range(FOLDS):
        z = np.load(folder / f"fit_baseline_{k}.npz")
        pred[z["rows"]] = z["epv"]
        fold[z["rows"]] = k
    mine = d.states[["sportvu_game", "possession", "frame"]].copy()
    mine["sportvu_game"] = mine["sportvu_game"].astype(str)
    mine["epv"], mine["fold_mine"] = pred, fold
    stored["sportvu_game"] = stored["sportvu_game"].astype(str)
    m = mine.merge(stored, on=["sportvu_game", "possession", "frame"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = float(np.max(np.abs(both["epv"].to_numpy() - both["epv_points"].to_numpy()))) if len(both) else None
    rec = {"states_mine": int(len(mine)), "states_stored": int(len(stored)), "matched": int(len(both)),
           "only_mine": int((m["_merge"] == "left_only").sum()), "only_stored": int((m["_merge"] == "right_only").sum()),
           "fold_mismatches": int((both["fold_mine"].to_numpy() != both["fold"].to_numpy()).sum()),
           "max_abs_diff": diff, "tolerance": GATE_A_TOL}
    rec["passed"] = bool(rec["only_mine"] == 0 and rec["only_stored"] == 0 and rec["fold_mismatches"] == 0
                         and diff is not None and diff <= GATE_A_TOL)
    return rec


def run(artifacts=None, xfg_artifacts=None, max_minutes: float = 26.0, log=print, head: str = "") -> dict:
    import numpy as np
    t0 = time.time()
    folder = pieces_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plan() if not piece_path(folder, *p).exists()]
    if not todo:
        return {"status": "every piece is done; run e1-dev"}
    d = Data()
    feats_k, labels_k = None, {}
    done = []
    for kind, arm, k in todo:
        left = max_minutes - (time.time() - t0) / 60
        need = FIT_MINUTES if kind == "fit" else 6.0
        if left < need:
            break
        if kind == "fit" and arm != "baseline":
            gA = folder / "gate_a.json"
            gB = folder / "gate_b.json"
            if not (gA.exists() and json.loads(gA.read_text())["passed"]):
                raise SystemExit(f"{WHO}: gate A (the baseline reproduction) has not passed; nothing more is fitted")
            if not (gB.exists() and json.loads(gB.read_text())["passed"]):
                raise SystemExit(f"{WHO}: gate B (the nested xFG log loss) has not passed; nothing more is fitted")
        if kind == "xfg":
            r = xfg_piece(d, k, xfg_artifacts)
            r["frame"].to_parquet(folder / f"xfg_{k}.parquet", index=False)
            piece_path(folder, kind, arm, k).write_text(json.dumps(_plain({**{k2: v for k2, v in r.items() if k2 != "frame"},
                                                                    "code_commit": head})) + "\n")
        else:
            if feats_k is None or feats_k[0] != k:
                feats_k = None
                gc.collect()
                feats_k = (k, d.fold_features(k))
            if arm != "baseline" and k not in labels_k:
                labels_k = {k: labels_for(d, k, artifacts)}
                (folder / f"labels_{k}.json").write_text(json.dumps(_plain(labels_k[k]["counts"])) + "\n")
            r = fit_piece(d, arm, k, feats_k[1], labels_k.get(k, {}))
            part = folder / f"fit_{arm}_{k}.part.npz"
            np.savez(part, **r, code_commit=np.array(head))
            os.replace(part, piece_path(folder, kind, arm, k))
        done.append(f"{kind}_{arm or ''}_{k}")
        log(f"[epv-E1] piece {kind} {arm or ''} {k} done {round(time.time() - t0)} s, peak {peak_rss_gb():.1f} GB", flush=True)
        if kind == "fit" and arm == "baseline" and k == FOLDS - 1:
            rec = gate_a(d, folder)
            (folder / "gate_a.json").write_text(json.dumps(_plain(rec), indent=2) + "\n")
            log(f"[epv-E1] gate A {rec}", flush=True)
        if kind == "xfg" and k == FOLDS - 1:
            lls = {i: json.loads((folder / f"xfg_{i}.json").read_text())["log_loss"] for i in range(FOLDS)}
            rec = {"log_loss_by_epv_fold": lls, "frozen_interval": XFG_OOF_INTERVAL,
                   "passed": all(XFG_OOF_INTERVAL[0] <= v <= XFG_OOF_INTERVAL[1] for v in lls.values())}
            (folder / "gate_b.json").write_text(json.dumps(_plain(rec), indent=2) + "\n")
            log(f"[epv-E1] gate B {rec}", flush=True)
    left = [p for p in plan() if not piece_path(folder, *p).exists()]
    return {"done_this_job": done, "left": len(left), "seconds": round(time.time() - t0, 1),
            "peak_gb": round(peak_rss_gb(), 2)}


# ----------------------------------------------------------------------------- assembly

def game_sums_bootstrap(groups, values: dict, n: int = BOOT, seed: int = SEED) -> dict:
    """Row-weighted means of each value with percentile intervals from a bootstrap of games (one shared draw)."""
    import numpy as np
    import pandas as pd
    codes, uniq = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniq)
    cnt = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))
    out = {}
    for k, v in values.items():
        s = np.bincount(codes, weights=v, minlength=G)
        b = s[idx].sum(axis=1) / cnt[idx].sum(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        out[k] = {"point": float(s.sum() / cnt.sum()), "ci95": [float(lo), float(hi)]}
    return out


def dev(artifacts=None) -> dict:
    import numpy as np
    from analytics.epv import train as T
    from analytics.eval import brier, calibration_slope_fast
    head = clean_head()
    folder = pieces_dir(artifacts)
    left = [p for p in plan() if not piece_path(folder, *p).exists()]
    if left:
        raise SystemExit(f"{WHO}: {len(left)} piece(s) left; run e1-run again")
    gA = json.loads((folder / "gate_a.json").read_text())
    gB = json.loads((folder / "gate_b.json").read_text())
    if not (gA["passed"] and gB["passed"]):
        raise SystemExit(f"{WHO}: a gate did not pass")
    d = Data()
    n = len(d.states)
    y = d.states["points_to_end"].to_numpy(dtype=float)
    games = d.games_all
    bucket = T.elapsed_bucket(d.states["elapsed_in_possession_s"].to_numpy(dtype=float))
    E, P = {}, {}
    for arm in ARMS:
        E[arm], P[arm] = np.full(n, np.nan), np.full(n, np.nan)
        for k in range(FOLDS):
            z = np.load(folder / f"fit_{arm}_{k}.npz")
            E[arm][z["rows"]], P[arm][z["rows"]] = z["epv"], z["p"]
    if any(not np.isfinite(E[a]).all() or not np.isfinite(P[a]).all() for a in ARMS):
        raise SystemExit(f"{WHO}: a state lacks an out-of-fold prediction or holds a non-finite one")
    sq = {a: (y - E[a]) ** 2 for a in ARMS}
    yb = (y > 0).astype(float)
    br = {a: (yb - P[a]) ** 2 for a in ARMS}
    overall = game_sums_bootstrap(games, {**{f"mse_{a}": sq[a] for a in ARMS}, **{f"brier_{a}": br[a] for a in ARMS},
                                          "baseline_minus_e1": sq["baseline"] - sq["e1"],
                                          "baseline_minus_three_only": sq["baseline"] - sq["three_only"]})
    buckets = {}
    for b in ("0-4", "4-10", "10-24"):
        m = bucket == b
        st = game_sums_bootstrap(games[m], {**{f"mse_{a}": sq[a][m] for a in ARMS},
                                            "baseline_minus_e1": sq["baseline"][m] - sq["e1"][m]})
        buckets[b] = {"states": int(m.sum()), **st,
                      **{f"slope_{a}": float(calibration_slope_fast(yb[m], P[a][m])) for a in ARMS},
                      **{f"brier_{a}": float(brier(yb[m], P[a][m])) for a in ARMS}}
    d_e1 = overall["baseline_minus_e1"]
    slope04 = buckets["0-4"]["slope_e1"]
    labels = {k: json.loads((folder / f"labels_{k}.json").read_text()) for k in range(FOLDS)}
    piece_commits = sorted({str(np.load(folder / f"fit_{a}_{k}.npz")["code_commit"]) for a in ARMS for k in range(FOLDS)}
                           | {json.loads((folder / f"xfg_{k}.json").read_text())["code_commit"] for k in range(FOLDS)})
    out = {"experiment": EXPERIMENT, "code_commit": head,
           "stage": "development games only, out of fold under the frozen 5 grouped folds by game (seed 0); no Set A "
                    "or held-out row read; Set A is not scored (D117)",
           "piece_commits": piece_commits,
           "gates": {"a_baseline_reproduces_frozen_oof": gA, "b_nested_xfg_log_loss": gB},
           "states": n, "games": int(len(set(games))), "labels": labels,
           "overall": overall, "slope_all": {a: float(calibration_slope_fast(yb, P[a])) for a in ARMS},
           "buckets": buckets,
           "verdict": {"rule": ("MSE(baseline) - MSE(e1) with its 95 percent interval above 0 and e1's 0 to 4 s "
                                "calibration slope inside 0.9 to 1.1, out of fold on the development games (D117)"),
                       "diff_ci_above_zero": bool(d_e1["ci95"][0] > 0),
                       "slope_0_4_e1": slope04, "slope_inside": bool(SLOPE_LO <= slope04 <= SLOPE_HI),
                       "passes": bool(d_e1["ci95"][0] > 0 and SLOPE_LO <= slope04 <= SLOPE_HI)},
           "peak_gb": round(peak_rss_gb(), 2)}
    bad = [k for k, v in _walk(out) if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise SystemExit(f"{WHO}: non-finite value(s) in the record, for example {bad[0]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def _walk(v, path=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _walk(x, f"{path}.{k}")
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            yield from _walk(x, f"{path}[{i}]")
    else:
        yield path, (v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v)


def clean_head() -> str:
    """HEAD, refused unless the whole working tree is clean."""
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty[:300]}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-run", "e1-dev"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--xfg-artifacts", default=os.environ.get("V3_XFG_ARTIFACTS"))
    ap.add_argument("--max-minutes", type=float, default=26.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research epv {a.command}"):
        if a.command == "e1-run":
            head = clean_head()
            out = {"code_commit": head, **run(a.artifacts, a.xfg_artifacts, a.max_minutes, head=head)}
        else:
            out = dev(a.artifacts)
    print(json.dumps(_plain(out.get("verdict", out)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
