"""analytics/epv/train.py: EPV candidates under the evaluation protocol of analytics/eval.py.

    python -m analytics.epv.train --fixtures | --data DATA_ROOT --report reports/epv.md [--folds 5] [--seed 0] [--bootstrap 200]
                                  [--out OUT_ROOT] [--season 2015-16] [--hgb-iter 100]

Unit: a state at 1 Hz inside a possession (docs/DATA_CONTRACT.md section 20). Target: points_to_end (0 to 4) as a
regression of expected points, with P(points > 0) as the classification view for calibration.

Protocol:
  1. load states_1hz, possessions, rosters and the shots table (analytics.epv.validate), set every row of a held-out
     game aside (the directory flag or an id of sportvu/splits.json), call analytics.eval.refuse_heldout on the
     training game ids and analytics.schema.assert_time_forward on the feature frame (features.build_features);
  2. grouped folds by sportvu_game (fewer games than folds: n_folds = number of games; one game: in-sample, labelled);
     inside each fold the priors (features.Priors) come from the training games only and are out-of-fold for the
     training rows themselves (a training game's own shots and possessions are left out of its prior);
  3. candidates: (1) constant = mean points_to_end of the training rows, (2) ridge regression (median imputation,
     standardisation, alpha 1), (3) HistGradientBoostingRegressor (the plan's model), each with an isotonic
     calibration of P(points > 0) fitted on out-of-fold predictions inside the training fold (3 inner grouped folds
     by game), (4) only under the gate: MLPRegressor over the last 5 states' features (features.lagged; one hidden
     layer of 32 units), the stand-in for a GRU;
  4. the gate: the sequence model is trained only if the gradient-boosting calibration slope by shot-clock bucket
     (0 to 7, 7 to 14, 14 to 24 s) is outside 0.9 to 1.1 in any bucket or its MSE lift over the constant is under 5
     percent; it is kept only if it beats gradient boosting under paired bootstrap by game (the 95 percent interval
     of MSE(hgb) - MSE(mlp) above 0);
  5. selection: lowest out-of-fold MSE of expected points among the three (the base), replaced by the sequence model
     only when it beats the base under paired bootstrap by game with a lower out-of-fold MSE; the report shows out-of-fold Brier of calibrated P(points > 0) and MSE by elapsed bucket (0 to 4, 4 to
     10, 10 to 24 s; a state past 24 s joins the last bucket) with bootstrap intervals by game, the calibration
     slope per bucket (target 0.9 to 1.1, reported, never tuned on), the reliability tables, a shuffled-target check
     (possession points permuted across possessions of the training games: the lift must sit at chance), and the
     held-out games scored once by the selected candidate refit on every training row (the constant kept as a labelled
     floor; no other candidate sees a held-out row). The fold column of the outputs is >= 0 out-of-fold, -1 held-out,
     -2 in-sample (the one-training-game mode).

Outputs (docs/contract/epv.md) with --out: OUT_ROOT/analytics/epv/<season>/<game>.parquet (epv_states, one file per
game) and OUT_ROOT/analytics/epv/<season>/epv_possessions.parquet (one file per season), held-out games under
OUT_ROOT/analytics/epv/heldout/<season>/. A fixture run writes the report with the words "fixture output, not a
result" and the numbers are never presented as results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.epv import ELAPSED_BUCKETS, SHOT_CLOCK_BUCKETS, WHO, season_of
from analytics.epv import features as F
from analytics.epv import validate as V
from analytics.eval import (bootstrap_by_game, brier, calibration_slope, calibration_table, grouped_folds,
                            paired_bootstrap_by_game, refuse_heldout)

CANDIDATES = ("constant", "ridge", "hgb")
SEQUENCE = "mlp"
SLOPE_LO, SLOPE_HI, MIN_LIFT = 0.9, 1.1, 0.05
N_LAGS = 5
INNER_FOLDS = 3
MLP_ITER = 300
IN_SAMPLE_FOLD = -2          # repo convention for the fold column: -1 held-out, -2 in-sample, >= 0 out-of-fold
MAC_CMD = ("python -m analytics.epv.train --data /path/to/DATA_ROOT --report reports/epv.md --folds 5 --seed 0 "
           "--bootstrap 1000 --out /path/to/DATA_ROOT")


# ── buckets ────────────────────────────────────────────────────────────────────────────────────────
def elapsed_bucket(t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    labels = np.array(["%g-%g" % b for b in ELAPSED_BUCKETS], dtype=object)
    edges = np.array([b[1] for b in ELAPSED_BUCKETS[:-1]])
    return labels[np.searchsorted(edges, t, side="right")]


def shot_clock_bucket(sc) -> np.ndarray:
    sc = np.asarray(sc, dtype=float)
    labels = np.array(["%g-%g" % (b[0], round(b[1])) for b in SHOT_CLOCK_BUCKETS], dtype=object)
    edges = np.array([b[1] for b in SHOT_CLOCK_BUCKETS[:-1]])
    out = labels[np.searchsorted(edges, np.nan_to_num(sc, nan=12.0), side="right")]
    return np.where(np.isfinite(sc), out, "none")


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def make_model(name: str, seed: int, hgb_iter: int = 100):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=hgb_iter, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50,
                                             l2_regularization=1.0, random_state=seed)
    if name == SEQUENCE:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=MLP_ITER, random_state=seed))
    raise ValueError(name)


class Calibrated:
    """A regression candidate plus an isotonic map from its expected points to P(points > 0), the map fitted on
    out-of-fold predictions inside the training rows (inner grouped folds by game)."""

    def __init__(self, name: str, columns: list, seed: int, hgb_iter: int = 100):
        self.name, self.columns, self.seed, self.hgb_iter = name, list(columns), seed, hgb_iter
        self.model, self.iso, self.const, self.p_const = None, None, 0.0, 0.0

    def _fit_raw(self, X: pd.DataFrame, y: np.ndarray):
        if self.name == "constant":
            return float(np.mean(y))
        m = make_model(self.name, self.seed, self.hgb_iter)
        return m.fit(X[self.columns].to_numpy(dtype=float), y)

    @staticmethod
    def _predict_raw(m, X: pd.DataFrame, columns) -> np.ndarray:
        if isinstance(m, float):
            return np.full(len(X), m)
        return np.asarray(m.predict(X[columns].to_numpy(dtype=float)), dtype=float)

    def fit(self, X: pd.DataFrame, y, games) -> "Calibrated":
        from sklearn.isotonic import IsotonicRegression
        y = np.asarray(y, dtype=float)
        games = np.asarray(games).astype(str)
        schema.assert_time_forward(X, "states_1hz")
        self.p_const = float(np.mean(y > 0))
        uniq = np.unique(games)
        oof = np.full(len(y), np.nan)
        if self.name != "constant" and len(uniq) >= 2:
            for tr_g, te_g in grouped_folds(uniq, n_folds=min(INNER_FOLDS, len(uniq)), seed=self.seed):
                tr, te = np.isin(games, tr_g), np.isin(games, te_g)
                oof[te] = self._predict_raw(self._fit_raw(X.loc[tr], y[tr]), X.loc[te], self.columns)
        self.model = self._fit_raw(X, y)
        if self.name == "constant" or len(uniq) < 2:
            oof = self._predict_raw(self.model, X, self.columns)
        if self.name != "constant" and len(np.unique(y > 0)) == 2 and np.nanstd(oof) > 1e-9:
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, (y > 0).astype(float))
        else:
            self.iso = None
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(expected points, calibrated P(points > 0))."""
        epv = self._predict_raw(self.model, X, self.columns)
        p = np.full(len(X), self.p_const) if self.iso is None else np.clip(self.iso.predict(epv), 0.0, 1.0)
        return epv, p


# ── data ───────────────────────────────────────────────────────────────────────────────────────────
def split_tables(tables: dict) -> dict:
    """Cleaned tables -> dict(states_train, states_heldout, shots_train, rosters, possessions, reports). The guard
    runs on the training game ids; held-out rows are scored once, never fitted."""
    clean, reports = V.validate_all(tables)
    st = clean["states_1hz"]
    if st is None or len(st) == 0:
        raise SystemExit("%s: no states_1hz rows" % WHO)
    held = st["heldout"].astype(bool)
    states_train, states_heldout = st[~held].reset_index(drop=True), st[held].reset_index(drop=True)
    shots = clean.get("shots")
    shots_train = shots[~shots["heldout"].astype(bool)].reset_index(drop=True) if shots is not None and len(shots) else None
    train_games = sorted(states_train["sportvu_game"].astype(str).unique())
    refuse_heldout(train_games, WHO)
    if shots_train is not None:
        refuse_heldout(sorted(shots_train["sportvu_game"].astype(str).unique()), WHO)
    return {"states_train": states_train, "states_heldout": states_heldout, "shots_train": shots_train,
            "rosters": clean.get("rosters"), "possessions": clean.get("possessions"), "reports": reports,
            "train_games": train_games}


def fold_features(geo_tr: pd.DataFrame, geo_te: pd.DataFrame, states_tr: pd.DataFrame, shots_train, rosters):
    """Priors from the training rows only -> (X_train, X_test) feature frames (assert_time_forward inside)."""
    priors = F.Priors.from_training(states_tr, shots_train, rosters)
    return F.build_features(states_tr, priors, geo=geo_tr), F.build_features(None, priors, geo=geo_te), priors


def out_of_fold(states: pd.DataFrame, geo: pd.DataFrame, shots_train, rosters, n_folds: int, seed: int,
                hgb_iter: int = 100, models=CANDIDATES, y_override=None, cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold expected points and calibrated P(points > 0) per candidate -> (pred, protocol note).
    pred holds epv_<m>, p_<m>, fold, and the bucket columns; note["insample"] is each model's mean in-sample MSE over
    the folds (fit and scored on the same training rows), next to which the out-of-fold MSE shows overfit.
    ``cache`` (a dict the caller keeps) stores the fold feature frames by fold index, so the gate and shuffle runs
    reuse the priors instead of rebuilding them."""
    y_all = np.asarray(states["points_to_end"] if y_override is None else y_override, dtype=float)
    games = states["sportvu_game"].astype(str).to_numpy()
    uniq = sorted(set(games))
    pred = states[["sportvu_game", "possession", "frame", "elapsed_in_possession_s"]].copy()
    pred["y"] = y_all
    pred["fold"] = -1
    for m in models:
        pred["epv_" + m], pred["p_" + m] = np.nan, np.nan
    seq_cols = F.lagged_columns(N_LAGS)
    cache = {} if cache is None else cache
    insample = {m: [] for m in models}
    if len(uniq) == 1:
        folds = [(uniq, uniq)]
        note = {"mode": "in-sample", "folds": 0, "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE"}
    else:
        k = min(n_folds, len(uniq))
        folds = grouped_folds(uniq, n_folds=k, seed=seed)
        note = {"mode": "out-of-fold", "folds": k, "text": "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)}
        if k < n_folds:
            note["text"] += "; fewer games (%d) than the %d folds asked, so n_folds = number of games" % (len(uniq), n_folds)
    for fi, (tr_g, te_g) in enumerate(folds):
        tr, te = np.isin(games, tr_g), np.isin(games, te_g)
        key = (fi, len(folds), tuple(tr_g))
        if key not in cache:
            X_tr, X_te, _ = fold_features(geo.loc[tr], geo.loc[te], states.loc[tr], shots_train, rosters)
            cache[key] = (X_tr, X_te)
        X_tr, X_te = cache[key]
        if SEQUENCE in models:
            if (key, "lag") not in cache:
                cache[(key, "lag")] = (F.lagged(X_tr, N_LAGS), F.lagged(X_te, N_LAGS))
            X_tr, X_te = cache[(key, "lag")]
        for m in models:
            cols = seq_cols if m == SEQUENCE else F.FEATURES
            c = Calibrated(m, cols, seed, hgb_iter).fit(X_tr, y_all[tr], games[tr])
            epv, p = c.predict(X_te)
            pred.loc[te, "epv_" + m], pred.loc[te, "p_" + m] = epv, p
            insample[m].append(mse(y_all[tr], c.predict(X_tr)[0]))
        pred.loc[te, "fold"] = fi if note["mode"] == "out-of-fold" else IN_SAMPLE_FOLD
    pred["elapsed_bucket"] = elapsed_bucket(pred["elapsed_in_possession_s"])
    pred["shot_clock_bucket"] = shot_clock_bucket(states["shot_clock_s"].to_numpy(dtype=float))
    note["insample"] = {m: float(np.mean(v)) if v else float("nan") for m, v in insample.items()}
    return pred, note


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────
def mse(y, e) -> float:
    y, e = np.asarray(y, dtype=float), np.asarray(e, dtype=float)
    return float(np.mean((y - e) ** 2))


def _ci(d: dict, nd: int = 4) -> str:
    f = "%%.%df" % nd
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d.get("lo", np.nan)):
        return f % d["point"]
    return (f + " [" + f + ", " + f + "]") % (d["point"], d["lo"], d["hi"])


def _boot(df: pd.DataFrame, fn, n_boot: int, seed: int) -> dict:
    if df["game"].nunique() < 2 or n_boot <= 0:
        return {"point": float(fn(df)), "lo": float("nan"), "hi": float("nan")}
    return bootstrap_by_game(df, "game", fn, n=n_boot, seed=seed)


def metrics_by_bucket(pred: pd.DataFrame, models, bucket_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Rows (bucket, model, n, brier, mse, slope) with bootstrap intervals by game; bucket "all" first."""
    rows = []
    buckets = ["all"] + [b for b in dict.fromkeys(pred[bucket_col]) if b != "none"] + (["none"] if (pred[bucket_col] == "none").any() else [])
    for b in buckets:
        part = pred if b == "all" else pred[pred[bucket_col] == b]
        for m in models:
            if ("epv_" + m) not in part.columns or part["epv_" + m].isna().all():
                continue
            d = pd.DataFrame({"game": part["sportvu_game"].astype(str).to_numpy(), "y": part["y"].to_numpy(dtype=float),
                              "e": part["epv_" + m].to_numpy(dtype=float), "p": part["p_" + m].to_numpy(dtype=float)})
            d = d[np.isfinite(d["e"])]
            if len(d) == 0:
                continue
            yb = (d["y"] > 0).astype(float).to_numpy()
            rows.append({"bucket": b, "model": m, "n": int(len(d)),
                         "brier": _boot(d, lambda s: brier((s["y"] > 0).astype(float), s["p"]), n_boot, seed),
                         "mse": _boot(d, lambda s: mse(s["y"], s["e"]), n_boot, seed),
                         "slope": float(calibration_slope(yb, d["p"].to_numpy())) if m != "constant" and np.std(d["p"]) > 1e-9 else float("nan")})
    return pd.DataFrame(rows)


def gate_decision(pred: pd.DataFrame) -> dict:
    """The sequence-model gate from the gradient-boosting out-of-fold predictions."""
    slopes = {}
    for b in [x for x in dict.fromkeys(pred["shot_clock_bucket"]) if x != "none"]:
        part = pred[pred["shot_clock_bucket"] == b]
        yb = (part["y"] > 0).astype(float).to_numpy()
        p = part["p_hgb"].to_numpy(dtype=float)
        slopes[b] = float(calibration_slope(yb, p)) if len(np.unique(yb)) == 2 and np.std(p) > 1e-9 else float("nan")
    m_const, m_hgb = mse(pred["y"], pred["epv_constant"]), mse(pred["y"], pred["epv_hgb"])
    lift = 1.0 - m_hgb / m_const if m_const > 0 else 0.0
    bad = [b for b, s in slopes.items() if not np.isfinite(s) or s < SLOPE_LO or s > SLOPE_HI]
    return {"slopes": slopes, "lift": float(lift), "slope_out_of_range": bad, "train_sequence": bool(bad) or lift < MIN_LIFT,
            "mse_constant": m_const, "mse_hgb": m_hgb}


def select_model(overall: dict, paired: dict | None) -> tuple[str, str, bool]:
    """(selected, reason, keep_sequence): the lowest out-of-fold MSE among CANDIDATES is the base; the sequence model
    replaces it only when its paired bootstrap against the base (MSE(base) - MSE(mlp)) has its whole 95 percent
    interval above 0 and its point out-of-fold MSE is below the base's."""
    base = min(CANDIDATES, key=lambda m: overall[m])
    selected, reason, keep = base, "lowest out-of-fold MSE among the three candidates", False
    if paired is not None and SEQUENCE in overall:
        keep = bool(paired["excludes_zero"] and paired["diff"] > 0 and overall[SEQUENCE] < overall[base])
        if keep:
            selected, reason = SEQUENCE, "beats %s, the lowest-MSE candidate, under paired bootstrap by game" % base
    return selected, reason, keep


def paired_mse(pred: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Paired bootstrap by game of MSE(a) - MSE(b); positive means b is better."""
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": pred["y"].to_numpy(dtype=float),
                      "a": pred["epv_" + a].to_numpy(dtype=float), "b": pred["epv_" + b].to_numpy(dtype=float)})
    if d["game"].nunique() < 2 or n_boot <= 0:
        diff = mse(d["y"], d["a"]) - mse(d["y"], d["b"])
        return {"diff": diff, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    return paired_bootstrap_by_game(d, "game", lambda s: mse(s["y"], s["a"]), lambda s: mse(s["y"], s["b"]), n=n_boot, seed=seed)


def shuffled_target(states: pd.DataFrame, seed: int) -> np.ndarray:
    """points_to_end with the possession-level points permuted across possessions (the within-possession structure
    kept): a model fitted on this must show no lift over the constant."""
    rng = np.random.default_rng(seed + 12345)
    key = states["sportvu_game"].astype(str) + "#" + states["possession"].astype(str)
    first = states.groupby(key, sort=False)["points_to_end"].first()
    perm = pd.Series(rng.permutation(first.to_numpy()), index=first.index)
    return key.map(perm).to_numpy(dtype=float)


HELDOUT_MIN_GAMES_FOR_INTERVALS = 5


def heldout_once(states_train, geo_train, states_heldout, geo_heldout, shots_train, rosters, seed, hgb_iter, models, n_boot) -> dict:
    """Refit priors and candidates on every training row, score the held-out rows once. Point estimates only (no
    bootstrap) with fewer than HELDOUT_MIN_GAMES_FOR_INTERVALS held-out games."""
    X_tr, X_ho, _ = fold_features(geo_train, geo_heldout, states_train, shots_train, rosters)
    if SEQUENCE in models:
        X_tr, X_ho = F.lagged(X_tr, N_LAGS), F.lagged(X_ho, N_LAGS)
    y_tr = states_train["points_to_end"].to_numpy(dtype=float)
    pred = states_heldout[["sportvu_game", "possession", "frame", "elapsed_in_possession_s"]].copy()
    pred["y"] = states_heldout["points_to_end"].to_numpy(dtype=float)
    pred["fold"] = -1
    for m in models:
        cols = F.lagged_columns(N_LAGS) if m == SEQUENCE else F.FEATURES
        c = Calibrated(m, cols, seed, hgb_iter).fit(X_tr, y_tr, states_train["sportvu_game"].to_numpy())
        pred["epv_" + m], pred["p_" + m] = c.predict(X_ho)
    pred["elapsed_bucket"] = elapsed_bucket(pred["elapsed_in_possession_s"])
    pred["shot_clock_bucket"] = shot_clock_bucket(states_heldout["shot_clock_s"].to_numpy(dtype=float))
    n_games = int(pred["sportvu_game"].nunique())
    boot = n_boot if n_games >= HELDOUT_MIN_GAMES_FOR_INTERVALS else 0
    return {"pred": pred, "by_bucket": metrics_by_bucket(pred, models, "elapsed_bucket", boot, seed), "n_games": n_games, "bootstrap": boot}


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────
def epv_tables(pred: pd.DataFrame, selected: str, heldout: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(epv_states, epv_possessions) of docs/contract/epv.md from a prediction frame."""
    st = pd.DataFrame({"sportvu_game": pred["sportvu_game"].astype(str).to_numpy(), "possession": pred["possession"].to_numpy(dtype=np.int32),
                       "frame": pred["frame"].to_numpy(dtype=np.int32), "elapsed_in_possession_s": pred["elapsed_in_possession_s"].to_numpy(dtype=np.float32),
                       "epv_points": pred["epv_" + selected].astype(float).to_numpy(), "p_score": pred["p_" + selected].astype(float).to_numpy(),
                       "points_to_end": pred["y"].to_numpy(dtype=np.int16), "fold": pred["fold"].to_numpy(dtype=np.int32), "heldout": bool(heldout)})
    first = st.sort_values("elapsed_in_possession_s", kind="stable").drop_duplicates(["sportvu_game", "possession"])
    po = pd.DataFrame({"sportvu_game": first["sportvu_game"].to_numpy(), "possession": first["possession"].to_numpy(dtype=np.int32),
                       "epv_at_start": first["epv_points"].to_numpy(dtype=float), "points": first["points_to_end"].to_numpy(dtype=np.int16)})
    return st.sort_values(["sportvu_game", "possession", "elapsed_in_possession_s"]).reset_index(drop=True), po.reset_index(drop=True)


def write_tables(out_root, st: pd.DataFrame, po: pd.DataFrame) -> list:
    """OUT_ROOT/analytics/epv/<season>/<game>.parquet (the states table, one file per game) and
    OUT_ROOT/analytics/epv/<season>/epv_possessions.parquet (one file per season); held-out games under
    OUT_ROOT/analytics/epv/heldout/<season>/. The layout the decisions and playcalling lanes read."""
    written = []
    held_of = st.drop_duplicates("sportvu_game").set_index("sportvu_game")["heldout"].to_dict()
    for game, part in st.groupby("sportvu_game", sort=True):
        base = Path(out_root) / "analytics" / "epv" / ("heldout" if held_of[game] else "") / season_of(game)
        base.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(base / ("%s.parquet" % game), index=False)
        written.append(base / ("%s.parquet" % game))
    po = po.copy()
    po["heldout"] = po["sportvu_game"].map(held_of).astype(bool)
    po["season"] = po["sportvu_game"].map(season_of)
    for (season, held), part in po.groupby(["season", "heldout"], sort=True):
        base = Path(out_root) / "analytics" / "epv" / ("heldout" if held else "") / season
        base.mkdir(parents=True, exist_ok=True)
        path = base / "epv_possessions.parquet"
        part = part.drop(columns=["season", "heldout"]).reset_index(drop=True)
        if path.exists():      # one file per season: merge with what an earlier run wrote for other games
            old = pd.read_parquet(path)
            old = old[~old["sportvu_game"].isin(part["sportvu_game"])]
            part = pd.concat([old, part], ignore_index=True).sort_values(["sportvu_game", "possession"]).reset_index(drop=True)
        part.to_parquet(path, index=False)
        written.append(path)
    return written


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def _bucket_table(t: pd.DataFrame, models) -> list:
    L = ["| bucket (s) | model | n | Brier P(points > 0) | MSE expected points | calibration slope |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in t.iterrows():
        s = "n/a" if not np.isfinite(r["slope"]) else "%.3f" % r["slope"]
        L.append("| %s | %s | %d | %s | %s | %s |" % (r["bucket"], r["model"], r["n"], _ci(r["brier"]), _ci(r["mse"]), s))
    return L


def _reliability(pred: pd.DataFrame, model: str, bins: int = 5) -> list:
    L = ["| bucket (s) | bin | n | mean p | observed rate |", "| --- | --- | --- | --- | --- |"]
    for b in [x for x in dict.fromkeys(pred["elapsed_bucket"])]:
        part = pred[pred["elapsed_bucket"] == b]
        t = calibration_table((part["y"] > 0).astype(float).to_numpy(), part["p_" + model].to_numpy(dtype=float), bins)
        for _, r in t[t["n"] > 0].iterrows():
            L.append("| %s | %.1f to %.1f | %d | %.3f | %.3f |" % (b, r["lo"], r["hi"], r["n"], r["mean_p"], r["mean_y"]))
    return L


def _why_selected(ctx: dict) -> str:
    """One sentence on why the constant won (or did not), from the numbers in ctx."""
    g, o, i = ctx["gate"], ctx["overall_mse"], ctx["insample_mse"]
    if ctx["selected"] != "constant":
        return ("Why: %s has the lowest out-of-fold MSE (%.4f against the constant's %.4f, lift %.1f percent)."
                % (ctx["selected"], o[ctx["selected"]], o["constant"], 100 * (1 - o[ctx["selected"]] / o["constant"])))
    reasons = ["no learner beats the constant out of fold (gradient boosting lift %.1f percent)" % (100 * g["lift"])]
    bad = g["slope_out_of_range"]
    if bad:
        reasons.append("the gradient-boosting calibration slopes sit outside 0.9 to 1.1 in %s" % ", ".join(bad))
    if g["lift"] <= ctx["shuffle"]["lift"]:
        reasons.append("the real lift (%.1f percent) is no better than the shuffled-target lift (%.1f percent)" % (100 * g["lift"], 100 * ctx["shuffle"]["lift"]))
    over = [m for m in ("ridge", "hgb", SEQUENCE) if m in i and np.isfinite(i[m]) and i[m] < 0.9 * o.get(m, np.inf)]
    if over:
        reasons.append("the learners overfit (in-sample MSE well under out-of-fold for %s: the target is one draw per possession "
                       "while min_samples_leaf counts states, so a leaf can hold a handful of possessions)" % ", ".join(over))
    return "Why the constant: " + "; ".join(reasons) + "."


def write_report(path: Path, ctx: dict) -> str:
    fixture = ctx["fixtures"]
    tag = "fixture output, not a result" if fixture else "real data run"
    L = ["# EPV: expected possession value from 1 Hz states (analytics/epv)", "",
         "Written %s by `%s` (%s)." % (ctx["written"], ctx["command"], tag), ""]
    if fixture:
        L += ["**Every number below is fixture output, not a result**: the inputs are the synthetic season of "
              "tests/_fx_epv.py, built to exercise the pipeline, and say nothing about basketball.", ""]
    L += ["## Inputs", "", "| table | rows in | schema problems | duplicates dropped | rows dropped | rows out |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in ctx["reports"].items():
        L.append("| %s | %d | %d | %d | %d | %d |" % (name, r["rows_in"], len(r["schema_problems"]), r["duplicates_dropped"], r["rows_dropped"], r["rows_out"]))
    L += ["", "Training: %d states, %d possessions, %d game(s) (%s). Held-out, set aside before any fit: %d states from %d game(s) (%s)."
          % (ctx["train_rows"], ctx["train_possessions"], len(ctx["train_games"]), ", ".join(ctx["train_games"]) or "none",
             ctx["heldout_rows"], len(ctx["heldout_games"]), ", ".join(ctx["heldout_games"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked the %d training game ids against sportvu/splits.json; "
          "`analytics.schema.assert_time_forward` passed on every feature frame (the only column source is "
          "`schema.feature_allowlist(\"states_1hz\")`; points_to_end and possession_end_kind never enter)." % len(ctx["train_games"]),
          "Training mean points per state: %.4f; P(points > 0): %.4f." % (ctx["mean_points"], ctx["rate_score"]), "",
          "## Protocol", "",
          "Features (%d): %s." % (len(F.FEATURES), ", ".join(F.FEATURES)),
          "Priors come from the training games of each fold only. The per-player zone counts and the per-lineup counts are "
          "out-of-fold for the training rows (a game's own shots and possessions are left out of its prior); the two league-wide "
          "quantities they shrink towards, the location-only xFG (analytics.features_xfg, four parameters) at the handler's "
          "position and the training turnover rate, are fitted once on the whole training fold, the row's own game included.",
          "Candidates: constant (training mean), ridge (median imputation, standardisation, alpha 1), HistGradientBoostingRegressor "
          "(%d iterations, learning rate 0.05, 8 leaves, min 50 per leaf, fixed in advance, never tuned) with isotonic calibration of P(points > 0) fitted on inner out-of-fold "
          "predictions (%d grouped folds), and under the gate an MLPRegressor over the last %d states' features." % (ctx["hgb_iter"], INNER_FOLDS, N_LAGS),
          "Metrics: %s; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s."
          % (ctx["note"]["text"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if len(ctx["train_games"]) < 2 else ""), "",
          "## Candidates by elapsed-time bucket (%s)" % ctx["note"]["mode"], ""]
    L += _bucket_table(ctx["by_elapsed"], ctx["models"])
    L += ["", "| model | in-sample MSE (mean over folds) | out-of-fold MSE |", "| --- | --- | --- |"]
    for m in ctx["models"]:
        L.append("| %s | %.4f | %.4f |" % (m, ctx["insample_mse"].get(m, float("nan")), ctx["overall_mse"][m]))
    L += ["", "Selected: **%s** (%s)." % (ctx["selected"], ctx["selection_reason"])]
    L.append(_why_selected(ctx))
    L += ["", "## Calibration of P(points > 0), %s model (%s)" % (ctx["selected"], ctx["note"]["mode"]), ""]
    L += _reliability(ctx["pred"], ctx["selected"])
    L += ["", "## Gradient boosting by shot-clock bucket and the gate", "",
          "| shot clock (s) | calibration slope |", "| --- | --- |"]
    for b, s in ctx["gate"]["slopes"].items():
        L.append("| %s | %s |" % (b, "n/a" if not np.isfinite(s) else "%.3f" % s))
    g = ctx["gate"]
    L += ["", "MSE lift of gradient boosting over the constant: %.1f percent (%.4f vs %.4f). Slopes outside %g to %g (n/a counts as outside): %s."
          % (100 * g["lift"], g["mse_hgb"], g["mse_constant"], SLOPE_LO, SLOPE_HI, ", ".join(g["slope_out_of_range"]) or "none"),
          "Gate decision: the sequence model (MLPRegressor over the last %d states, standing in for a GRU) was %s."
          % (N_LAGS, "TRAINED" if g["train_sequence"] else "NOT trained (gradient boosting is calibrated in every bucket and its lift is at least 5 percent)")]
    if ctx.get("paired") is not None:
        p = ctx["paired"]
        L.append("Paired bootstrap by game of MSE(%s) - MSE(mlp), %s being the lowest-MSE candidate: %.4f [%s, %s]; the sequence model is %s."
                 % (ctx["base"], ctx["base"], p["diff"], "n/a" if not np.isfinite(p["lo"]) else "%.4f" % p["lo"], "n/a" if not np.isfinite(p["hi"]) else "%.4f" % p["hi"],
                    "KEPT (interval above 0 and a lower out-of-fold MSE)" if ctx["keep_sequence"] else "DROPPED (no significant gain over the base candidate)"))
    L += ["", "## Shuffled-target check", "",
          "Possession points permuted across possessions of the training games, gradient boosting refit out-of-fold: MSE %.4f vs constant %.4f, "
          "lift %.1f percent (chance is at or below 0; the real lift above is %.1f percent)."
          % (ctx["shuffle"]["mse_hgb"], ctx["shuffle"]["mse_constant"], 100 * ctx["shuffle"]["lift"], 100 * g["lift"])]
    if ctx.get("heldout") is not None:
        L += ["", "## Held-out games, scored once (never fitted or tuned on)", "",
              "%d states from %d held-out game(s): %s; the pipeline was refit on all %d training states. Only the selected "
              "candidate (%s) is scored here%s; the other candidates never see a held-out row."
              % (ctx["heldout_rows"], len(ctx["heldout_games"]), ", ".join(ctx["heldout_games"]), ctx["train_rows"], ctx["selected"],
                 ", with the constant as a labelled floor" if ctx["selected"] != "constant" else ""),
              "Calibration slope per bucket against the 0.9 to 1.1 target is reported, not tuned on.%s" % (
                  "" if ctx["heldout"]["bootstrap"] else " Point estimates only: fewer than %d held-out games, so no bootstrap interval." % HELDOUT_MIN_GAMES_FOR_INTERVALS), ""]
        L += _bucket_table(ctx["heldout"]["by_bucket"], ctx["heldout"]["models"])
    L += ["", "## Status", ""]
    if fixture:
        L.append("Fixture output, not a result. The real run of this protocol on the owner's Mac, where the full data lives, is")
    else:
        L.append("The run of this protocol was")
    L += ["", "    " + MAC_CMD, "",
          "which expects DATA_ROOT/analytics/states_1hz/<season>/<game>.parquet (build it first with "
          "`python -m analytics.epv.states --data DATA_ROOT` when absent), DATA_ROOT/analytics/possessions, rosters, and "
          "DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet, and writes the epv_states and epv_possessions tables of "
          "docs/contract/epv.md under OUT_ROOT/analytics/epv/<season>/.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def run(tables: dict, n_folds: int = 5, seed: int = 0, n_boot: int = 200, hgb_iter: int = 100, fixtures: bool = False,
        command: str = "", report: Path | None = None, out_root=None) -> dict:
    """The whole protocol on loaded tables -> ctx (also written as the report when a path is given)."""
    d = split_tables(tables)
    st, ho = d["states_train"], d["states_heldout"]
    geo = F.geometry(st)
    schema.assert_time_forward(geo, "states_1hz")
    cache = {}
    pred, note = out_of_fold(st, geo, d["shots_train"], d["rosters"], n_folds, seed, hgb_iter, cache=cache)
    gate = gate_decision(pred)
    models = list(CANDIDATES)
    insample = dict(note["insample"])
    paired = None
    base = min(CANDIDATES, key=lambda m: mse(pred["y"], pred["epv_" + m]))
    if gate["train_sequence"]:
        seq_pred, seq_note = out_of_fold(st, geo, d["shots_train"], d["rosters"], n_folds, seed, hgb_iter, models=(SEQUENCE,), cache=cache)
        pred["epv_" + SEQUENCE], pred["p_" + SEQUENCE] = seq_pred["epv_" + SEQUENCE], seq_pred["p_" + SEQUENCE]
        models.append(SEQUENCE)
        insample.update(seq_note["insample"])
        paired = paired_mse(pred, base, SEQUENCE, n_boot, seed)
    by_elapsed = metrics_by_bucket(pred, models, "elapsed_bucket", n_boot, seed)
    overall = {r["model"]: r["mse"]["point"] for _, r in by_elapsed[by_elapsed["bucket"] == "all"].iterrows()}
    selected, reason, keep_seq = select_model(overall, paired)
    y_shuf = shuffled_target(st, seed)
    shuf_pred, _ = out_of_fold(st, geo, d["shots_train"], d["rosters"], n_folds, seed, hgb_iter, models=("constant", "hgb"), y_override=y_shuf, cache=cache)
    shuffle = {"mse_constant": mse(shuf_pred["y"], shuf_pred["epv_constant"]), "mse_hgb": mse(shuf_pred["y"], shuf_pred["epv_hgb"])}
    shuffle["lift"] = 1.0 - shuffle["mse_hgb"] / shuffle["mse_constant"] if shuffle["mse_constant"] > 0 else 0.0
    heldout = None
    if len(ho):
        geo_ho = F.geometry(ho)
        scored = [selected] + (["constant"] if selected != "constant" else [])      # the selected candidate, the constant as a labelled floor
        heldout = heldout_once(st, geo, ho, geo_ho, d["shots_train"], d["rosters"], seed, hgb_iter, scored, n_boot)
        heldout["models"] = scored
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command, "fixtures": fixtures,
           "reports": d["reports"], "train_rows": int(len(st)), "train_possessions": int(st.groupby(["sportvu_game", "possession"]).ngroups),
           "train_games": d["train_games"], "heldout_rows": int(len(ho)), "heldout_games": sorted(ho["sportvu_game"].astype(str).unique()),
           "mean_points": float(st["points_to_end"].mean()), "rate_score": float((st["points_to_end"] > 0).mean()),
           "hgb_iter": hgb_iter, "note": note, "bootstrap": n_boot, "seed": seed, "models": models, "by_elapsed": by_elapsed,
           "selected": selected, "selection_reason": reason, "pred": pred, "gate": gate, "paired": paired, "keep_sequence": keep_seq,
           "shuffle": shuffle, "heldout": heldout, "overall_mse": overall, "insample_mse": insample, "base": base}
    if report is not None:
        ctx["text"] = write_report(Path(report), ctx)
    if out_root is not None:
        st_tab, po_tab = epv_tables(pred, selected, heldout=False)
        written = write_tables(out_root, st_tab, po_tab)
        if heldout is not None:
            hs, hp = epv_tables(heldout["pred"], selected, heldout=True)
            written += write_tables(out_root, hs, hp)
        ctx["written_tables"] = written
    return ctx


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="the synthetic season of tests/_fx_epv.py")
    g.add_argument("--data", help="DATA_ROOT (see analytics.epv.validate)")
    ap.add_argument("--report", default="reports/epv.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", help="OUT_ROOT for the epv_states and epv_possessions tables (never the repo)")
    ap.add_argument("--season", default=None)
    ap.add_argument("--games", type=int, default=8, help="fixture games (every fourth held out)")
    ap.add_argument("--hgb-iter", type=int, default=100)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    tables = V.load_fixtures(a.seed, n_games=a.games, heldout_every=4) if a.fixtures else V.load_root(a.data, a.season)
    cmd = "python -m analytics.epv.train " + " ".join(argv if argv is not None else sys.argv[1:])
    out_root = a.out if a.out else (Path(tempfile.gettempdir()) / "epv_fixture_root" if a.fixtures else None)
    ctx = run(tables, a.folds, a.seed, a.bootstrap, a.hgb_iter, fixtures=a.fixtures, command=cmd, report=Path(a.report), out_root=out_root)
    print(ctx["text"])
    return ctx


if __name__ == "__main__":
    main()
