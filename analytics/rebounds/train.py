"""analytics/rebounds/train.py: rebound probability under the full evaluation protocol.

    python -m analytics.rebounds.train --fixtures | --data DATA_ROOT --report reports/rebounds.md
        [--out OUT_ROOT] [--folds 5] [--seed 0] [--bootstrap 200] [--fixture-games 8] [--fixture-possessions 40]
        [--fixture-heldout-every 4] [--skip-heldout]

Unit: one missed field-goal attempt; target: which player on court secures the rebound, or none (team rebound).
Each miss is a block of candidate rows (analytics.rebounds.features) and every model gives a probability vector
over the block that sums to one.

Candidates
    prior   player-prior baseline: each candidate's smoothed rebound rate by player and is_offense from the
            training folds (rate = (rebounds + 10 base) / (opportunities + 10), base the per-row rate of that side;
            the none row takes the Laplace team-rebound rate), normalised within the miss (a softmax over the log
            rates). No positions.
    logit   binary-logistic approximation of the conditional logit: sklearn LogisticRegression (imputer, scaler,
            C = 1) on the candidate rows with the binary target, then a softmax of the linear scores within each
            miss (the within-miss softmax likelihood itself is not maximised).
    hgb     sklearn HistGradientBoostingClassifier on the candidate rows, probabilities normalised within the miss.
The prior_rate feature of logit and hgb is itself out of fold: training rows carry a prior computed on inner
grouped folds of the training games (leave-one-game-out when fewer than five), test rows the prior of all the
training games of their fold. The prior never sees the row it scores.

Protocol: analytics.eval.grouped_folds by sportvu_game; selection by per-miss multinomial log loss (lower is
better); top-1 accuracy also reported (plan target over 40 percent); 95 percent bootstrap intervals by game;
paired bootstrap of hgb against prior (out-of-fold rows only); calibration of the top candidate's probability via
analytics.eval.report_block; held-out games (heldout flag of the bundle, or analytics.eval.heldout_mask by id) set aside
before any fit and scored once, by the selected candidate alone refit on every training row (next to a uniform floor
that needs no fit); the non-selected models are never scored on held-out rows. analytics.eval.refuse_heldout runs on the training game ids and
analytics.schema.assert_time_forward on the feature frame before any fit. One training game: in-sample, labelled.

Fold column of the predictions: the outer fold index, HELDOUT_FOLD (-1) on held-out rows, IN_SAMPLE_FOLD (-2) when
one training game forced an in-sample fit.
Outputs: the report, and with --out the rebound_predictions table (docs/contract/rebounds.md) under
OUT_ROOT/analytics/rebounds/<season>/<game>.parquet (held-out under heldout/<season>/). Under --fixtures every
number is fixture output, not a result.
--skip-heldout: the held-out games are set aside exactly as without it but never refit for, scored or written; the
report says "held-out games not scored (--skip-heldout)".
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.eval import (EPS, bootstrap_by_game, fit_threads, fmt_ci, fmt_num, grouped_folds, heldout_mask,
                            paired_bootstrap_by_game, refuse_heldout, report_block, add_skip_heldout_arg,
                            skip_heldout_lines)
from analytics.rebounds import features as F
from analytics.rebounds import validate as V
from analytics.schema import assert_time_forward, season_of_game

WHO = "rebounds.train"
MODELS = ("prior", "logit", "hgb")
SMOOTH = 10.0
INNER_FOLDS = 5
TEMPERATURE = True                 # improvement search variant 1 (reports/archive/rebounds.md): sharpen hgb within the miss
TAU_BOUNDS = (0.25, 8.0)
FIT_TAU: list = []                 # the hgb temperature of every fit, for the report and the tests
TOP1_TARGET = 0.40
HELDOUT_FOLD, IN_SAMPLE_FOLD = -1, -2        # fold sentinels of the predictions table (docs/contract/rebounds.md)
PRED_COLUMNS = ["sportvu_game", "event", "candidate_player_id", "p_prior", "p_logit", "p_selected", "is_rebounder",
                "fold", "heldout"]
MAC_CMD = ('python -m analytics.rebounds.train --data "<Drive>/nba_harvest" --out "<Drive>/nba_harvest" '
           "--report reports/rebounds.md --folds 5 --seed 0 --bootstrap 1000")


# ── blocks ─────────────────────────────────────────────────────────────────────────────────────────
def miss_key(df: pd.DataFrame) -> np.ndarray:
    """Integer code of the miss (sportvu_game, event) of every row, in order of first appearance."""
    return pd.factorize(df["sportvu_game"].astype(str) + "#" + df["event"].astype(str))[0]


def normalise_within(values, key: np.ndarray) -> np.ndarray:
    """Non-negative values divided by their sum within each key; a block summing to zero becomes uniform."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, None)
    sums = np.bincount(key, weights=v, minlength=key.max() + 1)
    n = np.bincount(key, minlength=key.max() + 1)
    out = np.where(sums[key] > 0, v / np.where(sums[key] > 0, sums[key], 1.0), 1.0 / n[key])
    return out


def softmax_within(scores, key: np.ndarray) -> np.ndarray:
    """Softmax of ``scores`` within each key (max-shifted for stability)."""
    s = np.asarray(scores, dtype=float)
    mx = pd.Series(s).groupby(key).transform("max").to_numpy()
    return normalise_within(np.exp(s - mx), key)


# ── priors ─────────────────────────────────────────────────────────────────────────────────────────
def player_rates(train: pd.DataFrame) -> dict:
    """Smoothed rebound rates by (player, is_offense) plus the per-side base rates and the none rate, from the
    rows given (training rows only, by construction of the callers)."""
    rows = train[train["is_none"] == 0]
    y = rows[F.TARGET].astype(float)
    base = y.groupby(rows["is_offense"].astype(int)).mean().to_dict()
    for s in (0, 1):
        base.setdefault(s, float(y.mean()) if len(y) else 0.1)
    g = pd.DataFrame({"y": y, "pid": rows["candidate_player_id"].astype(int), "off": rows["is_offense"].astype(int)})
    agg = g.groupby(["pid", "off"])["y"].agg(["sum", "count"])
    player = {(int(p), int(o)): float((r["sum"] + SMOOTH * base[int(o)]) / (r["count"] + SMOOTH))
              for (p, o), r in agg.iterrows()}
    none_rows = train[train["is_none"] == 1]
    none = float((none_rows[F.TARGET].astype(float).sum() + 1.0) / (len(none_rows) + 2.0))
    return {"player": player, "base": base, "none": none}


def apply_prior(frame: pd.DataFrame, rates: dict) -> np.ndarray:
    """The prior rate of each candidate row from player_rates (the side base rate for an unseen player)."""
    pid = frame["candidate_player_id"].astype(int).to_numpy()
    off = frame["is_offense"].astype(int).to_numpy()
    none = frame["is_none"].astype(int).to_numpy() == 1
    out = np.array([rates["player"].get((int(p), int(o)), rates["base"][int(o)]) for p, o in zip(pid, off)], dtype=float)
    out[none] = rates["none"]
    return out


def oof_prior(train: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, str]:
    """The prior_rate of every training row computed without its own game (inner grouped folds). One game: the
    rates of that game, labelled in-sample."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    if len(games) < 2:
        return apply_prior(train, player_rates(train)), "in-sample (one training game)"
    k = min(INNER_FOLDS, len(games))
    out = np.full(len(train), np.nan)
    g = train["sportvu_game"].astype(str).to_numpy()
    for tr_games, te_games in grouped_folds(games, n_folds=k, seed=seed):
        te = np.isin(g, te_games)
        out[te] = apply_prior(train[te], player_rates(train[~te]))
    assert np.isfinite(out).all()
    return out, "out of fold over %d inner grouped folds" % k


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def design(frame: pd.DataFrame, prior: np.ndarray) -> pd.DataFrame:
    """The design matrix (static features plus the prior) after the leak and time-forward checks."""
    X = frame[F.STATIC_FEATURES].astype(float).copy()
    X[F.PRIOR] = np.asarray(prior, dtype=float)
    X = X[F.feature_columns()]
    F.assert_no_leak(X)
    assert_time_forward(X, "rebound_frames")
    return X


def fit_temperature(q, key: np.ndarray, y) -> float:
    """The exponent tau minimising the per-miss log loss of normalise_within(q ** tau) (bounded TAU_BOUNDS); 1.0
    when there is nothing to fit (no rows, or every miss has one candidate)."""
    from scipy.optimize import minimize_scalar
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0)
    y = np.asarray(y).astype(bool)
    if len(q) == 0 or not y.any():
        return 1.0
    logq = np.log(q)

    def loss(tau):
        p = normalise_within(np.exp(tau * (logq - logq.max())), key)
        return -float(np.log(np.clip(p[y], EPS, 1.0)).mean())
    r = minimize_scalar(loss, bounds=TAU_BOUNDS, method="bounded", options={"xatol": 1e-3})
    return float(r.x)


def inner_temperature(train: pd.DataFrame, seed: int = 0) -> tuple[float, int]:
    """(tau, inner folds): the hgb temperature fitted on inner out-of-fold hgb predictions over INNER_FOLDS grouped
    folds of ``train``'s games (never a row outside ``train``); (1.0, 0) with fewer than two games."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    if len(games) < 2:
        return 1.0, 0
    k = min(INNER_FOLDS, len(games))
    g = train["sportvu_game"].astype(str)
    q = np.full(len(train), np.nan)
    for tr_games, te_games in grouped_folds(games, n_folds=k, seed=seed):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        q[te] = _fit(train[tr], seed, only="hgb", temperature=False)["hgb"](train[te])
    return fit_temperature(q, miss_key(train), train[F.TARGET].astype(int).to_numpy()), k


def fit_candidates(train: pd.DataFrame, seed: int = 0, only: str | None = None) -> dict:
    """Fit the three candidates on one training frame -> {name: predict(frame) -> p normalised within miss}.
    With ``only`` set to a model name, only that candidate is fitted (the held-out refit). Every fit goes through
    here (the inner temperature fits go through _fit, under the same guards)."""
    return _fit(train, seed, only)


def _fit(train: pd.DataFrame, seed: int = 0, only: str | None = None, temperature: bool | None = None) -> dict:
    """Fit the three candidates on one training frame -> {name: predict(frame) -> p normalised within miss}.
    With ``only`` set to a model name, only that candidate is fitted (the held-out refit). Every fit goes through
    here: it refuses a held-out row or a held-out game id (refuse_heldout covers the held-out arena too)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_candidates" % WHO)
    games = sorted(train["sportvu_game"].astype(str).unique())
    refuse_heldout(games, WHO)
    rates = player_rates(train)
    prior_tr, _ = oof_prior(train, seed)
    X, y = design(train, prior_tr), train[F.TARGET].astype(int).to_numpy()
    out = {"prior": lambda frame, r=rates: normalise_within(apply_prior(frame, r), miss_key(frame))}
    if only == "prior":
        return out
    if len(np.unique(y)) < 2:
        out["logit"] = out["prior"]; out["hgb"] = out["prior"]
        return out
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    hgb = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=3, min_samples_leaf=10,
                                         early_stopping=False, random_state=seed)
    with fit_threads(1):                        # four cores shared by many jobs: no thread pool per fit
        if only in (None, "logit"):
            lr.fit(X.to_numpy(), y)
        if only in (None, "hgb"):
            hgb.fit(X.to_numpy(), y)

    def logit_predict(frame, m=lr, r=rates):
        Xt = design(frame, apply_prior(frame, r))
        with fit_threads(1):
            scores = m.decision_function(Xt.to_numpy())
        return softmax_within(scores, miss_key(frame))

    tau = 1.0
    if only in (None, "hgb") and (TEMPERATURE if temperature is None else temperature):
        tau, _ = inner_temperature(train, seed)
    FIT_TAU.append(tau)

    def hgb_predict(frame, m=hgb, r=rates, tau=tau):
        Xt = design(frame, apply_prior(frame, r))
        with fit_threads(1):
            p = m.predict_proba(Xt.to_numpy())[:, 1]
        if tau != 1.0:
            p = np.exp(tau * np.log(np.clip(p, EPS, 1.0)))
        return normalise_within(p, miss_key(frame))

    if only in (None, "logit"):
        out["logit"] = logit_predict
    if only in (None, "hgb"):
        out["hgb"] = hgb_predict
    if only is not None:
        out = {k: v for k, v in out.items() if k == only}
    return out


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold probabilities of every model -> (frame with p_prior, p_logit, p_hgb, fold; protocol note)."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    pred = train.copy()
    pred["fold"] = -1
    if len(games) == 1:
        models = fit_candidates(train, seed)
        for m in MODELS:
            pred["p_" + m] = models[m](train)
        pred["fold"] = IN_SAMPLE_FOLD
        return pred, {"mode": "in-sample", "folds": 0, "games": 1, "inner_folds": [0],
                      "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows)"}
    k = min(n_folds, len(games))
    for m in MODELS:
        pred["p_" + m] = np.nan
    g = train["sportvu_game"].astype(str)
    inner = []
    for fold, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        inner.append(min(INNER_FOLDS, len(tr_games)) if len(tr_games) > 1 else 0)
        models = fit_candidates(train[tr], seed)
        for m in MODELS:
            pred.loc[te, "p_" + m] = models[m](train[te])
        pred.loc[te, "fold"] = fold
    assert pred[["p_" + m for m in MODELS]].notna().all().all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(games), "inner_folds": inner, "text": text}


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────
def per_miss(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per miss: game, event, logp (log probability of the true candidate), hit (the top candidate is the
    rebounder; a tie at the maximum counts 1 / number tied, the expected top-1 under random tie-breaking),
    top_is_rebounder (binary: the rebounder is at the maximum, for the calibration block), p_top, n_candidates."""
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "event": pred["event"].to_numpy(),
                      "p": np.clip(pred[col].astype(float).to_numpy(), EPS, 1.0), "y": pred[F.TARGET].astype(bool).to_numpy()})
    d["row"] = np.arange(len(d))
    g = d.groupby(["game", "event"], sort=False)
    n_true = g["y"].sum()
    if not (n_true == 1).all():
        raise ValueError("%s: every miss needs exactly one rebounder row; found counts %s" % (WHO, sorted(set(n_true))))
    d["p_max"] = g["p"].transform("max")
    d["tied"] = (d["p"] >= d["p_max"] - 1e-12)
    n_tied = d[d["tied"]].groupby(["game", "event"], sort=False).size()
    true = d[d["y"]].set_index(["game", "event"])
    # expected top-1 under random tie-breaking: 1 / (number tied at the maximum) when the rebounder is among them
    hit = np.where(true["tied"].to_numpy(), 1.0 / n_tied.reindex(true.index).to_numpy(float), 0.0)
    out = pd.DataFrame({"logp": np.log(true["p"]), "hit": hit, "top_is_rebounder": true["tied"].astype(bool),
                        "p_top": true["p_max"], "n_candidates": g.size().reindex(true.index)}, index=true.index)
    return out.reset_index()


def mlogloss(pm: pd.DataFrame) -> float:
    """Per-miss multinomial log loss (mean of -log p of the true candidate)."""
    return float(-pm["logp"].mean())


def top1(pm: pd.DataFrame) -> float:
    """Top-1 accuracy (ties counted as 1 over the number tied)."""
    return float(pm["hit"].mean())


def uniform_reference(pm: pd.DataFrame) -> dict:
    """Log loss and top-1 of the uniform guess over each miss's candidates."""
    return {"mlogloss": float(np.log(pm["n_candidates"].astype(float)).mean()), "top1": float((1.0 / pm["n_candidates"]).mean())}


def score(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{model: {mlogloss: {point, lo, hi}, top1: {...}}} with bootstrap intervals by game (none over one game)."""
    out = {}
    for m in MODELS:
        pm = per_miss(pred, "p_" + m)
        out[m] = {}
        for name, fn in (("mlogloss", mlogloss), ("top1", top1)):
            r = bootstrap_by_game(pm, "game", fn, n=n_boot, seed=seed)
            if pm["game"].nunique() < 2:
                r = {**r, "lo": float("nan"), "hi": float("nan")}
            out[m][name] = r
    return out


def paired(pred: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Paired bootstrap by game of mlogloss(a) - mlogloss(b): negative favours a."""
    pa, pb = per_miss(pred, "p_" + a), per_miss(pred, "p_" + b)
    d = pa.merge(pb, on=["game", "event"], suffixes=("_a", "_b"))
    r = paired_bootstrap_by_game(d, "game", lambda s: -s["logp_a"].mean(), lambda s: -s["logp_b"].mean(), n=n_boot, seed=seed)
    if d["game"].nunique() < 2:
        r = {**r, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    return r


def select(scores: dict) -> str:
    """The model with the lowest out-of-fold per-miss log loss."""
    return min(MODELS, key=lambda m: scores[m]["mlogloss"]["point"])


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────
def _p_or_nan(pred: pd.DataFrame, col: str) -> np.ndarray:
    """The probability column as float32, or NaN where it was never computed (non-selected models on held-out rows)."""
    if col in pred.columns:
        return pred[col].astype("float32").to_numpy()
    return np.full(len(pred), np.nan, dtype="float32")


def predictions_table(pred: pd.DataFrame, selected: str) -> pd.DataFrame:
    """The rebound_predictions table (docs/contract/rebounds.md). On held-out rows only the selected model was
    refit and scored, so p_prior and p_logit are NaN there unless that model is the selected one."""
    out = pd.DataFrame({"sportvu_game": pred["sportvu_game"].astype("string").to_numpy(),
                        "event": pred["event"].astype("int32").to_numpy(),
                        "candidate_player_id": pred["candidate_player_id"].astype("int32").to_numpy(),
                        "p_prior": _p_or_nan(pred, "p_prior"),
                        "p_logit": _p_or_nan(pred, "p_logit"),
                        "p_selected": pred["p_" + selected].astype("float32").to_numpy(),
                        "is_rebounder": pred[F.TARGET].astype(bool).to_numpy(),
                        "fold": pred["fold"].astype("int16").to_numpy(),
                        "heldout": pred["heldout"].astype(bool).to_numpy()})
    return out[PRED_COLUMNS]


def _season(game: str) -> str:
    """analytics.schema.season_of_game, "unknown" for an id that is not MM.DD.YYYY.AWAY.at.HOME."""
    try:
        return season_of_game(game)
    except ValueError:
        return "unknown"


def write_predictions(table: pd.DataFrame, out_root) -> list:
    """One rebound_predictions parquet per game under OUT_ROOT/analytics/rebounds/<season>/ (held-out under heldout/)."""
    paths = []
    for game, part in table.groupby("sportvu_game", sort=True):
        held = bool(part["heldout"].any())
        p = Path(out_root) / "analytics" / "rebounds" / ("heldout" if held else "")
        p = p / _season(str(game)) / (str(game) + ".parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


ND = 4                                       # decimals of every number in the report


def write_report(path: Path, ctx: dict) -> str:
    """Write the markdown report to ``path`` and return its text; under --fixtures every heading of a section that
    carries a number ends with "(fixture output, not a result)"."""
    fx = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Rebound probability (analytics/rebounds/train.py)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture output, not a result.** Every number below comes from synthetic games (%s); no real tracking "
              "data was read. The real run is the Mac command at the end." % ctx["fixture_source"], ""]
    L += ["## Inputs and rows%s" % tag, "",
          "| game | held out | frames in | frames out | misses in | kept | team rebounds (none) | dropped | candidate rows |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    # --skip-heldout: no outcome count of a held-out game (team rebounds are a target class) reaches the report
    hide = set(ctx["heldout_game_ids"]) if ctx.get("skip_heldout") else set()
    for g in ctx["inputs"]:
        team = "-" if g["game"] in hide else "%d" % g["team"]
        L.append("| %s | %s | %d | %d | %d | %d | %s | %d | %d |" % (g["game"], g["heldout"], g["frames_in"], g["frames_out"],
                                                                     g["misses_in"], g["kept"], team, g["dropped"], g["rows"]))
    t = ctx["drop_total"]
    if hide:
        kept = {k: sum(g["info"].get(k, 0) for g in ctx["inputs"] if g["game"] not in hide) for k in F.INFO_COUNTS}
        kept_line = "Kept misses by kind (training games only; held-out outcomes not shown): "
    else:
        kept, kept_line = t, "Kept misses by kind: "
    L += ["", "Dropped misses by reason (all games): " + ", ".join("%s %d" % (k, t[k]) for k in F.DROP_REASONS) + ".",
          kept_line + ", ".join("%s %d" % (k, kept[k]) for k in F.INFO_COUNTS) + ".",
          "Validation (analytics.rebounds.validate): " + ctx["validate_line"] + ".", "",
          "Training: %d candidate rows, %d misses, %d game(s) (%s). Held out, set aside before any fit: %d rows, %d misses, %d game(s) (%s)."
          % (ctx["train_rows"], ctx["train_misses"], ctx["train_games"], ", ".join(ctx["train_game_ids"]) or "none",
             ctx["heldout_rows"], ctx["heldout_misses"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (%d columns)."
          % (ctx["train_games"], len(F.feature_columns())),
          "", "## Protocol%s" % tag, "",
          "Unit: a missed field-goal attempt; one candidate row per player on court at the rim frame plus a none row (team rebound).",
          "Features: %s." % ", ".join(F.feature_columns()),
          "Models: prior (smoothed player rate by is_offense, normalised within the miss), logit (binary-logistic approximation of the conditional logit: logistic regression on the candidate rows with the binary target, then a softmax of the linear scores within the miss; the within-miss likelihood itself is not maximised), hgb (HistGradientBoostingClassifier, normalised within the miss).",
          "Priors for the logit and hgb training rows: %s; test rows take the prior of their fold's training games." % ctx["prior_note"],
          "Selection: lowest out-of-fold per-miss multinomial log loss. Top-1 accuracy reported against the plan target of %.0f percent." % (100 * TOP1_TARGET),
          "Metrics: %s; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s."
          % (ctx["protocol"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if ctx["train_games"] < 2 else ""),
          "", "## Candidates (%s, %d misses, %d game(s))%s" % (ctx["mode"], ctx["train_misses"], ctx["train_games"], tag), "",
          "| model | per-miss log loss | top-1 accuracy |", "| --- | --- | --- |"]
    for m in MODELS:
        s = ctx["scores"][m]
        L.append("| %s | %s | %s |" % (m + (" (selected)" if m == ctx["selected"] else ""), fmt_ci(s["mlogloss"], ND), fmt_ci(s["top1"], ND)))
    u = ctx["uniform"]
    L += ["| uniform reference | %s | %s |" % (fmt_num(u["mlogloss"], ND), fmt_num(u["top1"], ND)), "",
          "Paired bootstrap by game, hgb minus prior per-miss log loss: %s [%s, %s]; interval excludes zero: %s (negative favours hgb)."
          % (fmt_num(ctx["paired"]["diff"], ND), fmt_num(ctx["paired"]["lo"], ND), fmt_num(ctx["paired"]["hi"], ND),
             ctx["paired"]["excludes_zero"]),
          "Selected: %s." % ctx["selected"], "",
          "## Calibration of the top candidate's probability (%s, %s)%s" % (ctx["selected"], ctx["mode"], tag), "",
          "Per miss: y is whether the top candidate secured the rebound (a tie at the maximum counts as yes), p is the top candidate's probability.", "",
          ctx["calibration_block"]]
    if ctx.get("skip_heldout"):
        L += ["## Held-out games%s" % tag, ""] + skip_heldout_lines(ctx["heldout_games"]) + [""]
    elif ctx["heldout_rows"]:
        L += ["## Held-out games (scored once by the selected candidate refit on all %d training rows; never fitted or tuned on)%s"
              % (ctx["train_rows"], tag), "",
              "%d misses from %d held-out game(s): %s." % (ctx["heldout_misses"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"])), "",
              "| model | per-miss log loss | top-1 accuracy |", "| --- | --- | --- |"]
        for m, sc in ctx["heldout_scores"].items():
            L.append("| %s | %s | %s |" % (m + (" (selected)" if m == ctx["selected"] else " (no fit: 1 over the candidates of the miss)"), fmt_num(sc["mlogloss"], ND), fmt_num(sc["top1"], ND)))
        L += ["", "Held-out numbers are point estimates over %d game(s), scored exactly once. The selection (p_selected = %s) was fixed on the out-of-fold training folds above before any held-out row was scored; only the selected candidate was refit and scored on held-out rows, so the other models' held-out scores do not exist and their columns are NaN on held-out rows of the predictions table. No bootstrap or paired comparison is computed on held-out rows."
              % (ctx["heldout_games"], ctx["selected"]), ""]
    L += ["## Assumptions taken without asking%s" % tag, "",
          "- Team rebounds are kept as the none class when a rebound event follows the miss and names no player; every other null identity_match is dropped and counted (the lane spec's literal rule would drop every team rebound, since its identity_match is null by construction).",
          "- The offence of a miss is the section 5 team of the miss event, else the possession's offense_team_id; a miss with neither is dropped.",
          "- The 0.4 s look-back is the frame of the same period nearest 0.4 s of game clock before the rim frame, within 0.2 s; otherwise the speed is NaN (imputed by the median in the logit, native in hgb).",
          "- shot_distance_ft uses the section 17 release_frame when present, else the ball's horizontal distance from the basket 1 s before the rim frame.",
          "- The prior is smoothed with 10 pseudo-opportunities toward the per-side base rate; the none rate is the Laplace team-rebound rate. Nothing was tuned.",
          "- Top-1 accuracy counts a tie at the maximum as 1 over the number tied (expected accuracy under random tie-breaking); the calibration block counts it as a hit.",
          "- Model settings fixed in advance: logistic regression C = 1 after median imputation and standardisation; HistGradientBoostingClassifier with 80 iterations, depth 3, learning rate 0.1, 10 rows per leaf.",
          "", "## Predictions%s" % tag, "",
          ("Written %d file(s) under OUT_ROOT/analytics/rebounds/ (docs/contract/rebounds.md)." % len(ctx["written_paths"]))
          if ctx["written_paths"] else "No --out given: the rebound_predictions table was not written.", "",
          "## Status%s" % tag, ""]
    if fx:
        L.append("Fixture output, not a result: a synthetic run that exercises the pipeline end to end. The real run on the Mac, where the full data lives, is")
    else:
        L.append("Run on %s. The command was" % ctx["data_root"])
    L += ["", "    " + (MAC_CMD if fx else ctx["command"]), ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def load_candidates(bundles) -> tuple[pd.DataFrame, list, dict, dict]:
    """Clean every bundle, build the candidate rows -> (frame, per-game input summary, drop totals, validate totals)."""
    inputs, cleaned, val = [], [], []
    for b in bundles:
        cb, counts = V.clean_bundle(b)
        cleaned.append(cb); val.append(counts)
    cands, fc = F.build(cleaned)
    for cb, counts in zip(cleaned, val):
        c = fc["per_game"].get(cb["sportvu_game"], {})
        inputs.append({"game": cb["sportvu_game"], "heldout": bool(cb["heldout"]), "frames_in": counts["frames"]["rows_in"],
                       "frames_out": counts["frames"]["rows_out"], "misses_in": c.get("n_in", 0), "kept": c.get("misses_kept", 0),
                       "team": c.get("team_rebounds_kept", 0), "dropped": int(sum(c.get(k, 0) for k in F.DROP_REASONS)), "rows": c.get("rows", 0),
                       "info": {k: int(c.get(k, 0)) for k in F.INFO_COUNTS}})
    return cands, inputs, fc["total"], V.summarise(val)["totals"]


def split_train_heldout(cands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held out: the bundle flag, or a sportvu_game that sportvu/splits.json marks held out, or the CLE arena."""
    held = cands["heldout"].astype(bool).to_numpy() | heldout_mask(cands["sportvu_game"].astype(str))
    train = cands[~held].reset_index(drop=True)
    heldout = cands[held].reset_index(drop=True)
    heldout["heldout"] = True
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    assert_time_forward(train, "rebound_frames")
    return train, heldout


def parse_args(argv=None):
    """The trainer's command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    V.add_source_args(ap)
    ap.add_argument("--report", default="reports/rebounds.md", help="markdown report path (default %(default)s)")
    ap.add_argument("--out", help="OUT_ROOT for the rebound_predictions table (default: not written)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by sportvu_game (default %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=200, help="game bootstrap resamples per interval (default %(default)s)")
    add_skip_heldout_arg(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """CLI: run the protocol, score the held-out games once, write the report and (with --out) the predictions."""
    a = parse_args(argv)
    bundles = V.source_bundles(a)
    fixture_source = V.FIXTURE_SOURCE if a.fixtures else ""
    cands, inputs, drop_total, val_tot = load_candidates(bundles)
    train, heldout = split_train_heldout(cands)
    if len(train) == 0:
        raise SystemExit("%s: no training rows after setting the held-out rows aside" % WHO)
    pred, note = out_of_fold(train, a.folds, a.seed)
    scores = score(pred, a.bootstrap, a.seed)
    selected = select(scores)
    pm = per_miss(pred, "p_" + selected)
    cal = report_block("top candidate, %s%s" % (selected, " (fixture output, not a result)" if a.fixtures else ""), pm["top_is_rebounder"].astype(float).to_numpy(), pm["p_top"].to_numpy(),
                       pm, "game", n=a.bootstrap, seed=a.seed)
    inner = sorted(set(note["inner_folds"]))
    prior_note = ("in-sample (one training game)" if inner == [0] else
                  "out of fold over %s inner grouped folds of each outer fold's training games (0 means in-sample: one game)"
                  % (inner[0] if len(inner) == 1 else "%d to %d" % (inner[0], inner[-1])))
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.rebounds.train " + " ".join(argv if argv is not None else sys.argv[1:]),
           "fixtures": bool(a.fixtures), "fixture_source": fixture_source, "data_root": a.data, "inputs": inputs,
           "drop_total": drop_total,
           "validate_line": "; ".join("%s %d rows in, %d out" % (t, c["rows_in"], c["rows_out"]) for t, c in val_tot.items()),
           "train_rows": int(len(train)), "train_misses": int(train[["sportvu_game", "event"]].drop_duplicates().shape[0]),
           "train_games": int(train["sportvu_game"].nunique()), "train_game_ids": sorted(train["sportvu_game"].astype(str).unique()),
           "heldout_rows": int(len(heldout)), "heldout_misses": int(heldout[["sportvu_game", "event"]].drop_duplicates().shape[0]),
           "heldout_games": int(heldout["sportvu_game"].nunique()), "heldout_game_ids": sorted(heldout["sportvu_game"].astype(str).unique()),
           "prior_note": prior_note, "protocol": note["text"], "mode": note["mode"], "folds": note["folds"],
           "bootstrap": a.bootstrap, "seed": a.seed, "scores": scores, "uniform": uniform_reference(pm),
           "paired": paired(pred, "hgb", "prior", a.bootstrap, a.seed), "selected": selected, "calibration_block": cal,
           "heldout_scores": {}, "written_paths": [], "out_root": a.out, "train": train, "heldout": heldout, "oof": pred,
           "skip_heldout": a.skip_heldout}
    tables = [predictions_table(pred, selected)]
    if len(heldout) and not a.skip_heldout:
        # only the selected candidate is refit and scored on held-out rows; the uniform floor needs no fit
        models = fit_candidates(train, a.seed, only=selected)
        hp = heldout.copy(); hp["fold"] = HELDOUT_FOLD
        hp["p_" + selected] = models[selected](heldout)
        pmh = per_miss(hp, "p_" + selected)
        ctx["heldout_scores"][selected] = {"mlogloss": mlogloss(pmh), "top1": top1(pmh)}
        ctx["heldout_scores"]["uniform floor"] = uniform_reference(pmh)
        tables.append(predictions_table(hp, selected))
    ctx["predictions"] = pd.concat(tables, ignore_index=True)
    if a.out:
        ctx["written_paths"] = write_predictions(ctx["predictions"], a.out)
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
