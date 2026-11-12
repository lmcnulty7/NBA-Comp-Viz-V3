"""analytics/winprob/train.py: the live win probability candidates under the multi-season protocol.

    python -m analytics.winprob.train --fixtures | --data DATA_ROOT --report reports/winprob.md
        [--out OUT_ROOT] [--seed 0] [--bootstrap 200] [--folds 5] [--heldout-season 2024-25] [--skip-heldout]
        [--fixture-games 20] [--fixture-seasons 2020-21,2021-22,2024-25]

Protocol
  1. Load the play-by-play, games and odds tables (analytics.winprob.features.load_data or the fixtures), build
     the winprob_states frame (one row per event, time-forward features only).
  2. Split by season: every row of --heldout-season is set aside before any fit and scored once at the end by the
     selected candidate refit on all training rows; seasons later than the held-out season are ignored (listed
     in the report), so nothing from the held-out season's future fits. Rows flagged heldout (sportvu/splits.json games, 2015-16)
     are dropped from training as well, and analytics.eval.refuse_heldout runs on the training game ids.
     analytics.schema.assert_time_forward runs on the design matrix. With --skip-heldout the season is set aside
     exactly as without it but never scored or predicted (no held-out rows in winprob_states), and the report says
     "held-out games not scored (--skip-heldout)".
  3. Candidates (baselines first, no deep model):
       constant    the training-fold home win rate
       logistic    logistic regression on margin, seconds_remaining, possession_home (plus its missing flag) and
                   margin_x_sqrt_time, standardised, C=1
       hgb_iso     HistGradientBoostingClassifier (depth 3, 50 rounds, 200 rows per leaf, L2 1.0) on the same plus
                   period, overtime, spread_home and spread_missing, calibrated with isotonic regression fitted on out-of-fold predictions of the
                   candidate's own training rows (inner grouped folds by game), then refit on all of them
  4. Selection: out-of-fold predictions under --folds grouped folds by game within the training seasons; the
     selection metric is the unweighted mean over time buckets (48-36, 36-24, 24-12, 12-4, 4-0 minutes
     remaining; OT reported, not selected on) of the Brier score, so the late game weighs as much as the
     early game. ECE and calibration slope per bucket carry bootstrap intervals by game; log loss overall.
  5. Outputs: reports/winprob.md and, with --out, OUT_ROOT/analytics/winprob/<season>/winprob_states.parquet
     and winprob_calibration.parquet (docs/contract/winprob.md).

Benchmark: the NBA.com WinProbabilityPBP endpoint is unreachable from here, so the benchmark column of the
report is empty until it exists. Fixture numbers are fixture output, not a result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import (bootstrap_by_game, brier, calibration_slope_fast, calibration_table, ece, fit_threads, fmt_ci,
                            fmt_num, grouped_folds, log_loss, refuse_heldout, add_skip_heldout_arg, skip_heldout_lines)
from analytics.winprob import features as F

WHO = "winprob.train"
MODELS = ("constant", "logistic", "hgb_iso")
LOGISTIC_COLS = ("margin", "seconds_remaining", "possession_home", "margin_x_sqrt_time")
HGB_COLS = tuple(F.FEATURES)
SELECTION_BUCKETS = tuple(b[0] for b in F.BUCKETS)
INNER_FOLDS = 3
MAC_CMD = ("python -m analytics.winprob.train --data \"$DATA_ROOT\" --out \"$DATA_ROOT\" --report reports/winprob.md "
           "--seed 0 --bootstrap 1000 --folds 5 --heldout-season 2024-25")
FIT_LOG: list = []      # every fit appends (models, sorted training game ids); inner isotonic fits log as ("hgb_inner",)
FIXTURE_SEASONS = "2020-21,2021-22,2024-25"
FIXTURE_TAG = "(fixture output, not a result)"


# ── split ──────────────────────────────────────────────────────────────────────────────────────────
def later_seasons(states: pd.DataFrame, heldout_season: str) -> list:
    """Seasons whose key sorts after the held-out season (YYYY-YY keys sort chronologically); they never fit."""
    return sorted(s for s in states["season"].astype(str).unique() if s > str(heldout_season))


def split_seasons(states: pd.DataFrame, heldout_season: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """-> (train, heldout, ignored_seasons). The held-out season and every heldout-flagged row are set aside, and
    every season later than the held-out one is ignored (training on the future of the held-out season would leak
    the era into it); the guard runs on the training game ids; the leakage guard runs on the training design
    matrix."""
    if "2015-16" in str(heldout_season) or heldout_season == "":
        raise SystemExit("%s: --heldout-season must name a modelling season (2015-16 never fits or scores here)" % WHO)
    s = states.copy()
    s["season"] = s["season"].astype(str)
    ignored = later_seasons(s, heldout_season)
    held_mask = (s["season"] == str(heldout_season)) & ~s["heldout"].astype(bool)
    drop = s["heldout"].astype(bool) | (s["season"] == "2015-16") | s["season"].isin(ignored)
    train = s[~held_mask & ~drop].reset_index(drop=True)
    heldout = s[held_mask].reset_index(drop=True)
    refuse_heldout(sorted(train["game_id"].astype(str).unique()), WHO)
    if "2015-16" in set(train["season"]):
        raise SystemExit("%s: 2015-16 rows reached the training frame" % WHO)
    if train["season"].gt(str(heldout_season)).any():
        raise SystemExit("%s: a season later than the held-out season reached the training frame" % WHO)
    schema.assert_time_forward(F.design(train), "states_1hz")
    return train, heldout, ignored


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def _xy(frame: pd.DataFrame, cols):
    X = F.design(frame, cols)
    return X.to_numpy(dtype=float), frame[F.TARGET].astype(int).to_numpy()


def fit_constant(train: pd.DataFrame):
    """The constant candidate: the training home win rate for every row."""
    rate = float(train[F.TARGET].astype(float).mean())
    return lambda frame, r=rate: np.full(len(frame), r)


def fit_logistic(train: pd.DataFrame):
    """The logistic candidate on LOGISTIC_COLS (standardised, C=1); the constant when y has one class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(train, LOGISTIC_COLS)
    if len(np.unique(y)) < 2:
        return fit_constant(train)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    return lambda frame, m=m: m.predict_proba(F.design(frame, LOGISTIC_COLS).to_numpy(dtype=float))[:, 1]


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    # shallow and heavily regularised: the spread is constant inside a game, so deep trees on few games memorise
    # games through it (seen on the fixtures); on real seasons many games share a spread value
    return HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, max_depth=3, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_hgb_iso(train: pd.DataFrame, seed: int = 0):
    """HGB with isotonic calibration on inner out-of-fold predictions (grouped by game), then a refit on all rows."""
    from sklearn.isotonic import IsotonicRegression
    X, y = _xy(train, HGB_COLS)
    if len(np.unique(y)) < 2:
        return fit_constant(train)
    games = train["game_id"].astype(str).to_numpy()
    uniq = np.unique(games)
    k = min(INNER_FOLDS, len(uniq))
    oof = np.full(len(train), np.nan)
    if k >= 2:
        for tr_g, te_g in grouped_folds(uniq, n_folds=k, seed=seed):
            te = np.isin(games, te_g)
            FIT_LOG.append((("hgb_inner",), sorted(set(games[~te].tolist()))))
            if len(np.unique(y[~te])) < 2:
                oof[te] = float(y[~te].mean())
                continue
            oof[te] = _hgb().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    base = _hgb().fit(X, y)
    if k >= 2 and np.isfinite(oof).all():
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
        return lambda frame, m=base, c=iso: c.predict(m.predict_proba(F.design(frame, HGB_COLS).to_numpy(dtype=float))[:, 1])
    return lambda frame, m=base: m.predict_proba(F.design(frame, HGB_COLS).to_numpy(dtype=float))[:, 1]


def fit_models(train: pd.DataFrame, seed: int = 0, models=MODELS) -> dict:
    """Every fit of the trainer goes through here -> {name: predict(frame) -> p}. Refuses a held-out row."""
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    if "2015-16" in set(train["season"].astype(str)):
        raise ValueError("%s: a 2015-16 row reached fit_models" % WHO)
    FIT_LOG.append((tuple(models), sorted(train["game_id"].astype(str).unique())))
    out = {}
    with fit_threads():
        if "constant" in models:
            out["constant"] = fit_constant(train)
        if "logistic" in models:
            out["logistic"] = fit_logistic(train)
        if "hgb_iso" in models:
            out["hgb_iso"] = fit_hgb_iso(train, seed)
    return out


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions p_<model> of every candidate under grouped folds by game, and a protocol note."""
    games = sorted(train["game_id"].astype(str).unique())
    pred = train.copy()
    pred["fold"] = -1
    for m in MODELS:
        pred["p_" + m] = np.nan
    if len(games) < 2:
        raise SystemExit("%s: %d training game(s); grouped folds need at least 2" % (WHO, len(games)))
    k = min(n_folds, len(games))
    g = train["game_id"].astype(str)
    for i, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = g.isin(tr_games).to_numpy(), g.isin(te_games).to_numpy()
        models = fit_models(train[tr], seed)
        for m in MODELS:
            pred.loc[te, "p_" + m] = models[m](train[te])
        pred.loc[te, "fold"] = i
    assert pred[["p_" + m for m in MODELS]].notna().all().all()
    text = "out-of-fold under %d grouped folds by game within the training seasons (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked" % (len(games), n_folds)
    return pred, {"folds": k, "games": len(games), "text": text}


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────
def _nan3():
    return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}


def bucket_scores(pred: pd.DataFrame, p_col: str, n_boot: int, seed: int, intervals: bool = True) -> dict:
    """{bucket: {n, brier, ece, slope}} each a dict(point, lo, hi) by game bootstrap inside the bucket;
    'all' holds the overall log loss and Brier."""
    out = {}
    d = pd.DataFrame({"game": pred["game_id"].astype(str).to_numpy(), "y": pred[F.TARGET].astype(float).to_numpy(),
                      "p": pred[p_col].astype(float).to_numpy(), "b": pred["time_bucket"].astype(str).to_numpy()})
    fns = {"brier": lambda s: brier(s["y"], s["p"]), "ece": lambda s: ece(s["y"], s["p"], 10),
           "slope": lambda s: calibration_slope_fast(s["y"], s["p"])}
    for b in F.BUCKET_NAMES:
        s = d[d["b"] == b]
        r = {"n": int(len(s))}
        for name, fn in fns.items():
            if len(s) == 0 or (name == "slope" and p_col == "p_constant"):
                r[name] = _nan3()
            elif intervals and n_boot > 0 and s["game"].nunique() >= 2:
                r[name] = bootstrap_by_game(s, "game", fn, n=n_boot, seed=seed)
            else:
                r[name] = {"point": float(fn(s)), "lo": float("nan"), "hi": float("nan")}
        out[b] = r
    out["all"] = {"n": int(len(d)), "log_loss": {"point": log_loss(d["y"], d["p"]), "lo": float("nan"), "hi": float("nan")},
                  "brier": {"point": brier(d["y"], d["p"]), "lo": float("nan"), "hi": float("nan")}}
    if intervals and n_boot > 0 and d["game"].nunique() >= 2:
        out["all"]["log_loss"] = bootstrap_by_game(d, "game", lambda s: log_loss(s["y"], s["p"]), n=n_boot, seed=seed)
        out["all"]["brier"] = bootstrap_by_game(d, "game", lambda s: brier(s["y"], s["p"]), n=n_boot, seed=seed)
    return out


def selection_score(scores: dict) -> float:
    """Unweighted mean of the bucket Brier points over the five regulation buckets (NaN buckets skipped)."""
    vals = [scores[b]["brier"]["point"] for b in SELECTION_BUCKETS if np.isfinite(scores[b]["brier"]["point"])]
    return float(np.mean(vals)) if vals else float("nan")


def calibration_rows(pred: pd.DataFrame, p_col: str, season_label: str) -> pd.DataFrame:
    """winprob_calibration rows (season, time_bucket, n, mean_p, mean_y, brier, slope)."""
    rows = []
    for b in F.BUCKET_NAMES:
        s = pred[pred["time_bucket"].astype(str) == b]
        if len(s) == 0:
            continue
        y, p = s[F.TARGET].astype(float).to_numpy(), s[p_col].astype(float).to_numpy()
        rows.append({"season": season_label, "time_bucket": b, "n": int(len(s)), "mean_p": float(p.mean()),
                     "mean_y": float(y.mean()), "brier": brier(y, p), "slope": calibration_slope_fast(y, p)})
    return pd.DataFrame(rows, columns=["season", "time_bucket", "n", "mean_p", "mean_y", "brier", "slope"])


def states_table(pred: pd.DataFrame, p_col: str, heldout: bool) -> pd.DataFrame:
    """winprob_states rows (docs/contract/winprob.md)."""
    return pd.DataFrame({
        "game_id": pred["game_id"].astype(str).to_numpy(), "season": pred["season"].astype(str).to_numpy(),
        "event": pred["event"].astype("int32").to_numpy(), "period": pred["period"].astype("int8").to_numpy(),
        "seconds_remaining": pred["seconds_remaining"].astype("float32").to_numpy(),
        "margin": pred["margin"].astype("int16").to_numpy(), "possession": pred["possession_home"].astype("Int8").to_numpy(),
        "spread": pred["spread_home"].astype("float32").to_numpy(), "p_home_win": pred[p_col].astype("float32").to_numpy(),
        "home_win": pred[F.TARGET].astype(bool).to_numpy(),
        "fold": pred["fold"].astype("int8").to_numpy() if "fold" in pred.columns else np.full(len(pred), -1, dtype="int8"),
        "heldout": np.full(len(pred), bool(heldout))})


def write_tables(out_root, train_pred: pd.DataFrame, heldout_pred: pd.DataFrame | None, p_col: str) -> list:
    """winprob_states and winprob_calibration per season under OUT_ROOT/analytics/winprob/<season>/ -> paths."""
    paths = []
    frames = [(train_pred, False)] + ([(heldout_pred, True)] if heldout_pred is not None and len(heldout_pred) else [])
    for pred, held in frames:
        for season, s in pred.groupby(pred["season"].astype(str), sort=True):
            d = Path(out_root) / "analytics" / "winprob" / str(season)
            d.mkdir(parents=True, exist_ok=True)
            states_table(s, p_col, held).to_parquet(d / "winprob_states.parquet", index=False)
            calibration_rows(s, p_col, str(season)).to_parquet(d / "winprob_calibration.parquet", index=False)
            paths += [str(d / "winprob_states.parquet"), str(d / "winprob_calibration.parquet")]
    return paths


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def clock_rule_text(c: dict) -> str:
    """One table cell for the clock rule's counts (features.clock_rule)."""
    b = c["by_cause"]
    return ("%d of %d events, %d of %d games: %d games (%d events) with a gap in the period numbers, %d games (%d events) with an "
            "overtime clock above 300 s, %d single events with a null or out-of-range clock" % (
                c["rows_dropped"], c["rows_in"], c["games_dropped"], c["games_in"], b["period_gap"]["games"], b["period_gap"]["rows"],
                b["overtime_clock"]["games"], b["overtime_clock"]["rows"], b["bad_clock"]["rows"]))


def write_report(path: Path, ctx: dict) -> str:
    """The markdown report of a run (ctx from run); every heading over fixture numbers carries the fixture tag."""
    fx = ctx["fixtures"]
    tag = " " + FIXTURE_TAG if fx else ""
    L = ["# Live win probability (analytics/winprob/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture run: every number below is fixture output, not a result.** The synthetic play-by-play of "
              "analytics.fixtures (synth_stints, synth_odds) stands in for the five seasons on the owner's Mac.", ""]
    L += ["## Inputs%s" % tag, "", "| input | value |", "| --- | --- |",
          "| files | %s |" % ", ".join(ctx["files"]),
          "| events (states) | %d |" % ctx["rows_all"],
          "| games | %d |" % ctx["games_all"],
          "| seasons | %s |" % ", ".join(ctx["seasons"]),
          "| training seasons | %s (%d events, %d games) |" % (", ".join(ctx["train_seasons"]), ctx["train_rows"], ctx["train_games"]),
          "| held-out season | %s (%d events, %d games), set aside before any fit |" % (ctx["heldout_season"], ctx["heldout_rows"], ctx["heldout_games"]),
          "| rows flagged heldout (sportvu/splits.json games) dropped | %d |" % ctx["dropped_heldout_rows"],
          "| clock rule: dropped before any feature (all seasons) | %s |" % clock_rule_text(ctx["clock_rule"]),
          "| seasons later than the held-out season, ignored (never fitted or scored) | %s |"
          % (("%s (%d events)" % (", ".join(ctx["ignored_seasons"]), ctx["ignored_rows"])) if ctx["ignored_seasons"] else "none"),
          "| guard | `analytics.eval.refuse_heldout` checked %d training game ids; `analytics.schema.assert_time_forward` passed on the design matrix |" % ctx["train_games"],
          "| closing spread coverage (training games) | %s |" % fmt_num(ctx["spread_coverage"], 3),
          "| possession known (training events) | %s |" % fmt_num(ctx["possession_known"], 3),
          "| training home win rate (by game) | %s |" % fmt_num(ctx["home_win_rate"], 3),
          "", "## Protocol%s" % tag, "",
          "Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): %s." % ", ".join(F.feature_columns()),
          "The final score never enters a feature; `prefix_invariance_violations` found %d violation(s) on %d probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k)." % (ctx["prefix_violations"], ctx["prefix_games"]),
          "Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag) and margin_x_sqrt_time (standardised, C=1); "
          "HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.",
          "Selection: %s. The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). "
          "Intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d). %s" % (ctx["protocol"], ctx["bootstrap"], ctx["seed"],
              "The held-out season is set aside and never scored (--skip-heldout)." if ctx.get("skip_heldout")
              else "The held-out season is scored once, by the selected candidate refit on all training rows."),
          "", "## Candidates by time bucket%s" % tag, "",
          "Brier score (lower is better), out-of-fold within the training seasons.", "",
          "| model | " + " | ".join(F.BUCKET_NAMES) + " | selection (mean of 5) | log loss overall | Brier overall |",
          "| --- |" + " --- |" * (len(F.BUCKET_NAMES) + 3)]
    for m in MODELS:
        s = ctx["scores"][m]
        cells = [fmt_ci(s[b]["brier"], 4) for b in F.BUCKET_NAMES]
        L.append("| %s%s | %s | %s | %s | %s |" % (m, " (selected)" if m == ctx["selected"] else "", " | ".join(cells),
                                                   fmt_num(ctx["selection"][m], 4), fmt_ci(s["all"]["log_loss"], 4), fmt_ci(s["all"]["brier"], 4)))
    L += ["", "Rows per bucket: " + ", ".join("%s %d" % (b, ctx["scores"][MODELS[0]][b]["n"]) for b in F.BUCKET_NAMES) + ".", "",
          "Selected: **%s** (lowest mean bucket Brier)." % ctx["selected"], "",
          "## Calibration by time bucket%s" % tag, "", "ECE (10 bins) and calibration slope per bucket, out-of-fold.", "",
          "| model | bucket | n | ECE | slope |", "| --- | --- | --- | --- | --- |"]
    for m in MODELS:
        for b in F.BUCKET_NAMES:
            s = ctx["scores"][m][b]
            if s["n"] == 0:
                continue
            L.append("| %s | %s | %d | %s | %s |" % (m, b, s["n"], fmt_ci(s["ece"], 4), fmt_ci(s["slope"], 4)))
    L += ["", "The constant model has no calibration slope by design.", "",
          "### Reliability table of the selected candidate (out-of-fold)%s" % tag, "",
          "| bin | lo | hi | n | mean p | mean home win |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ctx["calibration"].itertuples():
        L.append("| %d | %.1f | %.1f | %d | %s | %s |" % (r.bin, r.lo, r.hi, r.n, fmt_num(r.mean_p, 3), fmt_num(r.mean_y, 3)))
    L += ["", "### winprob_calibration rows of the selected candidate (training seasons, out-of-fold)%s" % tag, "",
          "| season | time_bucket | n | mean_p | mean_y | brier | slope |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in ctx["calibration_rows"].itertuples():
        L.append("| %s | %s | %d | %s | %s | %s | %s |" % (r.season, r.time_bucket, r.n, fmt_num(r.mean_p, 3), fmt_num(r.mean_y, 3), fmt_num(r.brier, 4), fmt_num(r.slope, 3)))
    L += ["", "## Held-out season %s, %s%s" % (ctx["heldout_season"], "not scored" if ctx.get("skip_heldout") else "scored once", tag), ""]
    if ctx.get("skip_heldout"):
        L += skip_heldout_lines(ctx["heldout_games"])
    elif ctx["heldout_rows"]:
        hs = ctx["heldout_scores"]
        L += ["%d events from %d games of %s, never fitted or tuned on; the selected candidate (%s) refit on all %d training rows scored them once."
              % (ctx["heldout_rows"], ctx["heldout_games"], ctx["heldout_season"], ctx["selected"], ctx["train_rows"]), "",
              "| bucket | n | Brier | ECE | slope | benchmark (NBA.com WinProbabilityPBP) |", "| --- | --- | --- | --- | --- | --- |"]
        for b in F.BUCKET_NAMES:
            s = hs[b]
            if s["n"] == 0:
                continue
            L.append("| %s | %d | %s | %s | %s |  |" % (b, s["n"], fmt_ci(s["brier"], 4), fmt_ci(s["ece"], 4), fmt_ci(s["slope"], 4)))
        L += ["", "Overall: log loss %s, Brier %s, mean bucket Brier %s." % (fmt_ci(hs["all"]["log_loss"], 4), fmt_ci(hs["all"]["brier"], 4), fmt_num(ctx["heldout_selection"], 4))]
    else:
        L.append("No rows of season %s were in the input, so nothing was scored." % ctx["heldout_season"])
    L += ["", "## Benchmark", "",
          "The NBA.com WinProbabilityPBP endpoint is unreachable from here, so the benchmark column is empty until it exists. "
          "When it does, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier.", "",
          "## Outputs", ""]
    L += ["- " + p for p in ctx["outputs"]] if ctx["outputs"] else ["- no tables written (no --out)"]
    L += ["", "## Mac command", "", "The real run over the five seasons under DATA_ROOT (collect/stints/playbyplay/<season>.parquet, collect/extras/odds/<season>.parquet when present):", "",
          "    " + MAC_CMD, "",
          "## Assumptions", "",
          "- A whole season is held out (default 2024-25) and %s; folds inside the training seasons are grouped by game; seasons later than the held-out season are ignored entirely (listed under Inputs)."
          % ("set aside unscored in this run (--skip-heldout)" if ctx.get("skip_heldout") else "scored once"),
          "- 2015-16 games of sportvu/splits.json never fit anything: rows flagged heldout are dropped, the 2015-16 season is refused as a held-out season, and refuse_heldout runs on the training ids.",
          "- The closing spread is the median over books of the is_close spreads market for the home side, joined on game_date and the home team name (analytics.winprob.features.TEAM_NAMES); null with a flag when absent.",
          "- The possession indicator is read from event kinds and is null when the events so far do not say.",
          "- The interaction is margin times sqrt(seconds remaining), as specified; the logistic model also sees margin and time on their own.",
          "- On the fixtures the boosted candidate loses to the logistic one because the spread is nearly unique per game among a few dozen games and the trees memorise games through it; the selection rule picks the logistic candidate. This is a fixture artefact to re-check on the real seasons, where many games share a spread value.",
          "- The HGB hyperparameters (depth 3, 50 rounds, 200 rows per leaf, L2 1.0) were set after seeing the fixture memorise games through the spread (a first setting of 15 leaves and 40 rows per leaf scored worse than the constant out of fold). They were chosen on fixture data, not on any real season, and the held-out season was not involved.",
          "- The closing spread join is analytics.betting.validate's (per-book latest pre-tip is_close snapshot, median over books, both team keys, tip date shifted 8 h to US local with one day of tolerance); the games table's spread_home is used only when no odds table exists.",
          "- The calibration slope is analytics.eval.calibration_slope_fast: the logistic regression of home_win on logit(p) of analytics.eval.calibration_slope, solved by a bounded Newton method because the lbfgs solver runs to its iteration cap on the nearly separable late-game buckets; Brier, ECE, log loss, the reliability table and the game bootstrap are analytics.eval's too.",
          ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The train CLI: --fixtures or --data, the report and output paths, the protocol knobs and the fixture knobs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic fixtures of analytics.fixtures "
                     "(fixture output, not a result)")
    src.add_argument("--data", help="DATA_ROOT with collect/stints/playbyplay/<season>.parquet")
    ap.add_argument("--report", default="reports/winprob.md", help="markdown report path (default reports/winprob.md)")
    ap.add_argument("--out", default=None, help="OUT_ROOT for the winprob_states and winprob_calibration tables")
    ap.add_argument("--seed", type=int, default=0, help="seed of the folds, the bootstrap and the fixtures (default 0)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples of whole games (default 200)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by game inside the training seasons (default 5)")
    ap.add_argument("--heldout-season", default="2024-25", help="the season set aside and scored once (default 2024-25)")
    add_skip_heldout_arg(ap)
    F.add_fixture_args(ap, FIXTURE_SEASONS)
    return ap.parse_args(argv)


def run(d: dict, heldout_season: str, seed: int, n_boot: int, n_folds: int, out_root=None, fixtures: bool = False,
        command: str = "", report: str | None = None, skip_heldout: bool = False) -> dict:
    """The whole protocol on the loaded tables -> ctx (what write_report renders), the report written when asked. With
    skip_heldout the held-out season is set aside as usual but never scored or written."""
    pbp, clock = F.clean_playbyplay(d["playbyplay"])   # the clock rule validate counts; build_states applies it too
    states = F.build_states(pbp, d["games"], d["odds"], d.get("team_names"))
    viol = F.prefix_invariance_violations(pbp, d["games"], d["odds"], n_games=2, cuts=2, seed=seed,
                                          team_names=d.get("team_names"))   # joins the spread once on the full games table
    if viol:
        raise SystemExit("%s: a feature changed when the play-by-play was cut: %s" % (WHO, viol[:3]))
    train, heldout, ignored = split_seasons(states, heldout_season)
    if len(train) == 0:
        raise SystemExit("%s: no training rows after setting season %s aside" % (WHO, heldout_season))
    pred, note = out_of_fold(train, n_folds, seed)
    scores = {m: bucket_scores(pred, "p_" + m, n_boot, seed) for m in MODELS}
    selection = {m: selection_score(scores[m]) for m in MODELS}
    selected = min(MODELS, key=lambda m: (selection[m] if np.isfinite(selection[m]) else np.inf, MODELS.index(m)))
    p_sel = "p_" + selected
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command, "fixtures": fixtures,
           "files": d["files"], "rows_all": int(len(states)), "games_all": int(states["game_id"].nunique()),
           "seasons": sorted(states["season"].astype(str).unique()), "train_seasons": sorted(train["season"].unique()),
           "train_rows": int(len(train)), "train_games": int(train["game_id"].nunique()), "heldout_season": heldout_season,
           "heldout_rows": int(len(heldout)), "heldout_games": int(heldout["game_id"].nunique()), "ignored_seasons": ignored,
           "ignored_rows": int(states["season"].astype(str).isin(ignored).sum()),
           "dropped_heldout_rows": int(states["heldout"].astype(bool).sum()),
           "spread_coverage": float((train.drop_duplicates("game_id")["spread_missing"] == 0).mean()),
           "possession_known": float(train["possession_home"].notna().mean()),
           "home_win_rate": float(train.drop_duplicates("game_id")[F.TARGET].mean()),
           "clock_rule": clock, "prefix_violations": len(viol), "prefix_games": 2, "protocol": note["text"], "bootstrap": n_boot, "seed": seed,
           "scores": scores, "selection": selection, "selected": selected,
           "calibration": calibration_table(pred[F.TARGET].astype(float).to_numpy(), pred[p_sel].astype(float).to_numpy()),
           "calibration_rows": pd.concat([calibration_rows(s, p_sel, str(season)) for season, s in pred.groupby("season", sort=True)], ignore_index=True),
           "heldout_scores": {}, "heldout_selection": float("nan"), "outputs": [], "pred": pred, "heldout_pred": None,
           "skip_heldout": bool(skip_heldout)}
    if len(heldout) and not skip_heldout:
        model = fit_models(train, seed, models=(selected,))[selected]
        hp = heldout.copy()
        hp[p_sel] = model(heldout)
        hp["fold"] = -1
        ctx["heldout_scores"] = bucket_scores(hp, p_sel, n_boot, seed)
        ctx["heldout_selection"] = selection_score(ctx["heldout_scores"])
        ctx["heldout_pred"] = hp
    if out_root:
        ctx["outputs"] = write_tables(out_root, pred, ctx["heldout_pred"], p_sel)
    if report:
        ctx["report"] = write_report(Path(report), ctx)
    return ctx


def main(argv=None) -> dict:
    """Run the trainer from the command line, print the report and return ctx."""
    a = parse_args(argv)
    d = F.load_args(a)
    cmd = "python -m analytics.winprob.train " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = run(d, a.heldout_season, a.seed, a.bootstrap, a.folds, out_root=a.out, fixtures=a.fixtures, command=cmd, report=a.report,
              skip_heldout=a.skip_heldout)
    print(ctx["report"])
    return ctx


if __name__ == "__main__":
    main()
