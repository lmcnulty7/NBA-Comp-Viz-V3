"""analytics/rebounds/train.py: rebound probability under the full evaluation protocol.

    python -m analytics.rebounds.train --fixtures | --data DATA_ROOT --report reports/rebounds.md
        [--out OUT_ROOT] [--folds 5] [--seed 0] [--bootstrap 200] [--games 8]

Unit: one missed field-goal attempt; target: which player on court secures the rebound, or none (team rebound).
Each miss is a block of candidate rows (analytics.rebounds.features) and every model gives a probability vector
over the block that sums to one.

Candidates
    prior   player-prior baseline: each candidate's smoothed rebound rate by player and is_offense from the
            training folds (rate = (rebounds + 10 base) / (opportunities + 10), base the per-row rate of that side;
            the none row takes the Laplace team-rebound rate), normalised within the miss (a softmax over the log
            rates). No positions.
    logit   conditional logit: sklearn LogisticRegression (imputer, scaler, C = 1) on the candidate rows with the
            binary target, then a softmax of the linear scores within each miss.
    hgb     sklearn HistGradientBoostingClassifier on the candidate rows, probabilities normalised within the miss.
The prior_rate feature of logit and hgb is itself out of fold: training rows carry a prior computed on inner
grouped folds of the training games (leave-one-game-out when fewer than five), test rows the prior of all the
training games of their fold. The prior never sees the row it scores.

Protocol: analytics.eval.grouped_folds by sportvu_game; selection by per-miss multinomial log loss (lower is
better); top-1 accuracy also reported (plan target over 40 percent); 95 percent bootstrap intervals by game;
paired bootstrap of hgb against prior; calibration of the top candidate's probability via analytics.eval.report_block;
held-out games (heldout flag of the bundle, or an id of sportvu/splits.json) set aside before any fit, scored once by
the models refit on every training row. analytics.eval.refuse_heldout runs on the training game ids and
analytics.schema.assert_time_forward on the feature frame before any fit. One training game: in-sample, labelled.

Outputs: the report, and with --out the rebound_predictions table (docs/contract/rebounds.md) under
OUT_ROOT/analytics/rebounds/<season>/<game>.parquet (held-out under heldout/<season>/). Under --fixtures every
number is fixture output, not a result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.eval import (EPS, bootstrap_by_game, grouped_folds, paired_bootstrap_by_game, refuse_heldout,
                            report_block, split_games)
from analytics.rebound_frames import season_of_game
from analytics.rebounds import features as F
from analytics.rebounds import validate as V
from analytics.schema import assert_time_forward

WHO = "rebounds.train"
MODELS = ("prior", "logit", "hgb")
SMOOTH = 10.0
INNER_FOLDS = 5
TOP1_TARGET = 0.40
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
    X = frame[F.STATIC_FEATURES].astype(float).copy()
    X[F.PRIOR] = np.asarray(prior, dtype=float)
    X = X[F.feature_columns()]
    F.assert_no_leak(X)
    assert_time_forward(X, "rebound_frames")
    return X


def fit_candidates(train: pd.DataFrame, seed: int = 0) -> dict:
    """Fit the three candidates on one training frame -> {name: predict(frame) -> p normalised within miss}.
    Every fit goes through here: it refuses a held-out row or a held-out game id."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_candidates" % WHO)
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    rates = player_rates(train)
    prior_tr, _ = oof_prior(train, seed)
    X, y = design(train, prior_tr), train[F.TARGET].astype(int).to_numpy()
    out = {"prior": lambda frame, r=rates: normalise_within(apply_prior(frame, r), miss_key(frame))}
    if len(np.unique(y)) < 2:
        out["logit"] = out["prior"]; out["hgb"] = out["prior"]
        return out
    from threadpoolctl import threadpool_limits
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    hgb = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=3, min_samples_leaf=10,
                                         early_stopping=False, random_state=seed)
    with threadpool_limits(limits=1):           # four cores shared by many jobs: no thread pool per fit
        lr.fit(X.to_numpy(), y)
        hgb.fit(X.to_numpy(), y)

    def logit_predict(frame, m=lr, r=rates):
        Xt = design(frame, apply_prior(frame, r))
        return softmax_within(m.decision_function(Xt.to_numpy()), miss_key(frame))

    def hgb_predict(frame, m=hgb, r=rates):
        Xt = design(frame, apply_prior(frame, r))
        return normalise_within(m.predict_proba(Xt.to_numpy())[:, 1], miss_key(frame))

    out["logit"], out["hgb"] = logit_predict, hgb_predict
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
        pred["fold"] = 0
        return pred, {"mode": "in-sample", "folds": 0, "games": 1,
                      "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows)"}
    k = min(n_folds, len(games))
    for m in MODELS:
        pred["p_" + m] = np.nan
    g = train["sportvu_game"].astype(str)
    for fold, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        models = fit_candidates(train[tr], seed)
        for m in MODELS:
            pred.loc[te, "p_" + m] = models[m](train[te])
        pred.loc[te, "fold"] = fold
    assert pred[["p_" + m for m in MODELS]].notna().all().all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


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
    return float(-pm["logp"].mean())


def top1(pm: pd.DataFrame) -> float:
    return float(pm["hit"].mean())


def uniform_reference(pm: pd.DataFrame) -> dict:
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
    return min(MODELS, key=lambda m: scores[m]["mlogloss"]["point"])


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────
def predictions_table(pred: pd.DataFrame, selected: str) -> pd.DataFrame:
    out = pd.DataFrame({"sportvu_game": pred["sportvu_game"].astype("string").to_numpy(),
                        "event": pred["event"].astype("int32").to_numpy(),
                        "candidate_player_id": pred["candidate_player_id"].astype("int32").to_numpy(),
                        "p_prior": pred["p_prior"].astype("float32").to_numpy(),
                        "p_logit": pred["p_logit"].astype("float32").to_numpy(),
                        "p_selected": pred["p_" + selected].astype("float32").to_numpy(),
                        "is_rebounder": pred[F.TARGET].astype(bool).to_numpy(),
                        "fold": pred["fold"].astype("int16").to_numpy(),
                        "heldout": pred["heldout"].astype(bool).to_numpy()})
    return out[PRED_COLUMNS]


def _season(game: str) -> str:
    try:
        return season_of_game(game)
    except ValueError:
        return "unknown"


def write_predictions(table: pd.DataFrame, out_root) -> list:
    paths = []
    for game, part in table.groupby("sportvu_game", sort=True):
        held = bool(part["heldout"].any())
        p = Path(out_root) / "analytics" / "rebounds" / ("heldout" if held else "")
        p = p / _season(str(game)) / (str(game) + ".parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def _f(v, nd=4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d: dict, nd=4) -> str:
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return _f(d["point"], nd)
    return "%s [%s, %s]" % (_f(d["point"], nd), _f(d["lo"], nd), _f(d["hi"], nd))


def write_report(path: Path, ctx: dict) -> str:
    fx = ctx["fixtures"]
    L = ["# Rebound probability (analytics/rebounds/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture output, not a result.** Every number below comes from synthetic games (%s); no real tracking "
              "data was read. The real run is the Mac command at the end." % ctx["fixture_source"], ""]
    L += ["## Inputs and rows", "",
          "| game | held out | frames in | frames out | misses in | kept | team rebounds (none) | dropped | candidate rows |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in ctx["inputs"]:
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d |" % (g["game"], g["heldout"], g["frames_in"], g["frames_out"],
                                                                     g["misses_in"], g["kept"], g["team"], g["dropped"], g["rows"]))
    t = ctx["drop_total"]
    L += ["", "Dropped misses by reason (all games): " + ", ".join("%s %d" % (k, t[k]) for k in F.DROP_REASONS) + ".",
          "Validation (analytics.rebounds.validate): " + ctx["validate_line"] + ".", "",
          "Training: %d candidate rows, %d misses, %d game(s) (%s). Held out, set aside before any fit: %d rows, %d misses, %d game(s) (%s)."
          % (ctx["train_rows"], ctx["train_misses"], ctx["train_games"], ", ".join(ctx["train_game_ids"]) or "none",
             ctx["heldout_rows"], ctx["heldout_misses"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (%d columns)."
          % (ctx["train_games"], len(F.feature_columns())),
          "", "## Protocol", "",
          "Unit: a missed field-goal attempt; one candidate row per player on court at the rim frame plus a none row (team rebound).",
          "Features: %s." % ", ".join(F.feature_columns()),
          "Models: prior (smoothed player rate by is_offense, normalised within the miss), logit (conditional logit: logistic regression on the rows, softmax of the linear scores within the miss), hgb (HistGradientBoostingClassifier, normalised within the miss).",
          "Priors for the logit and hgb training rows: %s; test rows take the prior of their fold's training games." % ctx["prior_note"],
          "Selection: lowest out-of-fold per-miss multinomial log loss. Top-1 accuracy reported against the plan target of %.0f percent." % (100 * TOP1_TARGET),
          "Metrics: %s; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s."
          % (ctx["protocol"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if ctx["train_games"] < 2 else ""),
          "", "## Candidates (%s, %d misses, %d game(s))" % (ctx["mode"], ctx["train_misses"], ctx["train_games"]), "",
          "| model | per-miss log loss | top-1 accuracy |", "| --- | --- | --- |"]
    for m in MODELS:
        s = ctx["scores"][m]
        L.append("| %s | %s | %s |" % (m + (" (selected)" if m == ctx["selected"] else ""), _ci(s["mlogloss"]), _ci(s["top1"])))
    u = ctx["uniform"]
    L += ["| uniform reference | %s | %s |" % (_f(u["mlogloss"]), _f(u["top1"])), "",
          "Paired bootstrap by game, hgb minus prior per-miss log loss: %s [%s, %s]; interval excludes zero: %s (negative favours hgb)."
          % (_f(ctx["paired"]["diff"]), _f(ctx["paired"]["lo"]), _f(ctx["paired"]["hi"]), ctx["paired"]["excludes_zero"]),
          "Selected: %s." % ctx["selected"], "",
          "## Calibration of the top candidate's probability (%s, %s)" % (ctx["selected"], ctx["mode"]), "",
          "Per miss: y is whether the top candidate secured the rebound (a tie at the maximum counts as yes), p is the top candidate's probability.", "",
          ctx["calibration_block"]]
    if ctx["heldout_rows"]:
        L += ["## Held-out games (scored once by the models refit on all %d training rows; never fitted or tuned on)" % ctx["train_rows"], "",
              "%d misses from %d held-out game(s): %s." % (ctx["heldout_misses"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"])), "",
              "| model | per-miss log loss | top-1 accuracy |", "| --- | --- | --- |"]
        for m in MODELS:
            s = ctx["heldout_scores"][m]
            L.append("| %s | %s | %s |" % (m, _f(s["mlogloss"]), _f(s["top1"])))
        L += ["", "Held-out numbers are point estimates over %d game(s), scored exactly once." % ctx["heldout_games"], ""]
    L += ["## Predictions", "",
          ("Written %d file(s) under %s/analytics/rebounds/ (docs/contract/rebounds.md)." % (len(ctx["written_paths"]), ctx["out_root"]))
          if ctx["written_paths"] else "No --out given: the rebound_predictions table was not written.", "",
          "## Status", ""]
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
                       "team": c.get("team_rebounds_kept", 0), "dropped": int(sum(c.get(k, 0) for k in F.DROP_REASONS)), "rows": c.get("rows", 0)})
    return cands, inputs, fc["total"], V.summarise(val)["totals"]


def split_train_heldout(cands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held out: the bundle flag, or a sportvu_game that sportvu/splits.json marks held out, or the CLE arena."""
    ids = cands["sportvu_game"].astype(str)
    by_id = ids.isin(set(split_games(sorted(ids.unique()))["heldout"])) | ids.str.endswith(".at.CLE")
    held = cands["heldout"].astype(bool) | by_id
    train = cands[~held].reset_index(drop=True)
    heldout = cands[held].reset_index(drop=True)
    heldout["heldout"] = True
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    assert_time_forward(train, "rebound_frames")
    return train, heldout


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="synthetic games (fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT (analytics.rebounds.validate docstring)")
    ap.add_argument("--report", default="reports/rebounds.md")
    ap.add_argument("--out", help="OUT_ROOT for the rebound_predictions table")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--games", type=int, default=8, help="fixture games (every fourth held out)")
    ap.add_argument("--possessions", type=int, default=40, help="fixture possessions per game")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.fixtures:
        bundles = V.bundles_from_fixtures(a.seed, a.games, 4, a.possessions)
        fixture_source = V._fixture_module().__name__
    else:
        bundles, fixture_source = V.bundles_from_root(a.data), ""
    cands, inputs, drop_total, val_tot = load_candidates(bundles)
    train, heldout = split_train_heldout(cands)
    if len(train) == 0:
        raise SystemExit("%s: no training rows after setting the held-out rows aside" % WHO)
    pred, note = out_of_fold(train, a.folds, a.seed)
    scores = score(pred, a.bootstrap, a.seed)
    selected = select(scores)
    pm = per_miss(pred, "p_" + selected)
    cal = report_block("top candidate, %s" % selected, pm["top_is_rebounder"].astype(float).to_numpy(), pm["p_top"].to_numpy(),
                       pm, "game", n=a.bootstrap, seed=a.seed)
    _, prior_note = oof_prior(train, a.seed)
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
           "heldout_scores": {}, "written_paths": [], "out_root": a.out, "train": train, "heldout": heldout, "oof": pred}
    tables = [predictions_table(pred, selected)]
    if len(heldout):
        models = fit_candidates(train, a.seed)
        hp = heldout.copy(); hp["fold"] = -1
        for m in MODELS:
            hp["p_" + m] = models[m](heldout)
            pmh = per_miss(hp, "p_" + m)
            ctx["heldout_scores"][m] = {"mlogloss": mlogloss(pmh), "top1": top1(pmh)}
        tables.append(predictions_table(hp, selected))
    ctx["predictions"] = pd.concat(tables, ignore_index=True)
    if a.out:
        ctx["written_paths"] = write_predictions(ctx["predictions"], a.out)
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
