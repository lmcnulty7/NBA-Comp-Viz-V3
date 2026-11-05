"""analytics/winprob/train.py: the live win probability candidates under the multi-season protocol.

    python -m analytics.winprob.train --fixtures | --data DATA_ROOT --report reports/winprob.md
        [--out OUT_ROOT] [--seed 0] [--bootstrap 200] [--folds 5] [--heldout-season 2024-25]

Protocol
  1. Load the play-by-play, games and odds tables (analytics.winprob.features.load_data or the fixtures), build
     the winprob_states frame (one row per event, time-forward features only).
  2. Split by season: every row of --heldout-season is set aside before any fit and scored once at the end by the
     selected candidate refit on all training rows. Rows flagged heldout (sportvu/splits.json games, 2015-16)
     are dropped from training as well, and analytics.eval.refuse_heldout runs on the training game ids.
     analytics.schema.assert_time_forward runs on the design matrix.
  3. Candidates (baselines first, no deep model):
       constant    the training-fold home win rate
       logistic    logistic regression on margin, seconds_remaining, possession_home (plus its missing flag) and
                   margin_x_sqrt_time, standardised, C=1
       hgb_iso     HistGradientBoostingClassifier on the same plus period, overtime, spread_home and
                   spread_missing, calibrated with isotonic regression fitted on out-of-fold predictions of the
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
from analytics.eval import (bootstrap_by_game, brier, calibration_slope, calibration_table, ece, grouped_folds, log_loss,
                            refuse_heldout)
from analytics.winprob import features as F

WHO = "winprob.train"
MODELS = ("constant", "logistic", "hgb_iso")
LOGISTIC_COLS = ("margin", "seconds_remaining", "possession_home", "margin_x_sqrt_time")
HGB_COLS = tuple(F.FEATURES)
SELECTION_BUCKETS = tuple(b[0] for b in F.BUCKETS)
INNER_FOLDS = 3
MAC_CMD = ("python -m analytics.winprob.train --data \"$DATA_ROOT\" --out \"$DATA_ROOT\" --report reports/winprob.md "
           "--seed 0 --bootstrap 1000 --folds 5 --heldout-season 2024-25")
FIT_LOG: list = []      # every fit appends (model, sorted training game ids); tests read it


def _limits():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception:                                   # noqa: BLE001
        import contextlib
        return contextlib.nullcontext()


# ── split ──────────────────────────────────────────────────────────────────────────────────────────
def split_seasons(states: pd.DataFrame, heldout_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (train, heldout). The held-out season and every heldout-flagged row are set aside; the guard runs on
    the training game ids; the leakage guard runs on the training design matrix."""
    if "2015-16" in str(heldout_season) or heldout_season == "":
        raise SystemExit("%s: --heldout-season must name a modelling season (2015-16 never fits or scores here)" % WHO)
    s = states.copy()
    s["season"] = s["season"].astype(str)
    held_mask = (s["season"] == str(heldout_season)) & ~s["heldout"].astype(bool)
    drop = s["heldout"].astype(bool) | (s["season"] == "2015-16")
    train = s[~held_mask & ~drop].reset_index(drop=True)
    heldout = s[held_mask].reset_index(drop=True)
    refuse_heldout(sorted(train["game_id"].astype(str).unique()), WHO)
    if "2015-16" in set(train["season"]):
        raise SystemExit("%s: 2015-16 rows reached the training frame" % WHO)
    schema.assert_time_forward(F.design(train), "states_1hz")
    return train, heldout


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def _xy(frame: pd.DataFrame, cols):
    X = F.design(frame, cols)
    return X.to_numpy(dtype=float), frame[F.TARGET].astype(int).to_numpy()


def fit_constant(train: pd.DataFrame):
    rate = float(train[F.TARGET].astype(float).mean())
    return lambda frame, r=rate: np.full(len(frame), r)


def fit_logistic(train: pd.DataFrame):
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
    return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                          early_stopping=False, random_state=0)


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
    with _limits():
        if "constant" in models:
            out["constant"] = fit_constant(train)
        if "logistic" in models:
            out["logistic"] = fit_logistic(train)
        if "hgb_iso" in models:
            out["hgb_iso"] = fit_hgb_iso(train, seed)
    return out


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
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
           "slope": lambda s: calibration_slope(s["y"], s["p"])}
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
                     "mean_y": float(y.mean()), "brier": brier(y, p), "slope": calibration_slope(y, p)})
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
def _f(v, nd=4):
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d):
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return _f(d["point"])
    return "%s [%s, %s]" % (_f(d["point"]), _f(d["lo"]), _f(d["hi"]))


def write_report(path: Path, ctx: dict) -> str:
    fx = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Live win probability (analytics/winprob/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture run: every number below is fixture output, not a result.** The synthetic play-by-play of "
              "tests/_fx_winprob.py stands in for the five seasons on the owner's Mac.", ""]
    L += ["## Inputs", "", "| input | value |", "| --- | --- |",
          "| files | %s |" % ", ".join(ctx["files"]),
          "| events (states) | %d |" % ctx["rows_all"],
          "| games | %d |" % ctx["games_all"],
          "| seasons | %s |" % ", ".join(ctx["seasons"]),
          "| training seasons | %s (%d events, %d games) |" % (", ".join(ctx["train_seasons"]), ctx["train_rows"], ctx["train_games"]),
          "| held-out season | %s (%d events, %d games), set aside before any fit |" % (ctx["heldout_season"], ctx["heldout_rows"], ctx["heldout_games"]),
          "| rows flagged heldout (sportvu/splits.json games) dropped | %d |" % ctx["dropped_heldout_rows"],
          "| guard | `analytics.eval.refuse_heldout` checked %d training game ids; `analytics.schema.assert_time_forward` passed on the design matrix |" % ctx["train_games"],
          "| closing spread coverage (training games) | %s |" % _f(ctx["spread_coverage"], 3),
          "| possession known (training events) | %s |" % _f(ctx["possession_known"], 3),
          "| training home win rate (by game) | %s |" % _f(ctx["home_win_rate"], 3),
          "", "## Protocol", "",
          "Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): %s." % ", ".join(F.feature_columns()),
          "The final score never enters a feature; `prefix_invariance_violations` found %d violation(s) on %d probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k)." % (ctx["prefix_violations"], ctx["prefix_games"]),
          "Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag) and margin_x_sqrt_time (standardised, C=1); "
          "HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.",
          "Selection: %s. The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). "
          "Intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d). The held-out season is scored once, by the selected candidate refit on all training rows." % (ctx["protocol"], ctx["bootstrap"], ctx["seed"]),
          "", "## Candidates by time bucket%s" % tag, "",
          "Brier score (lower is better), out-of-fold within the training seasons.", "",
          "| model | " + " | ".join(F.BUCKET_NAMES) + " | selection (mean of 5) | log loss overall | Brier overall |",
          "| --- |" + " --- |" * (len(F.BUCKET_NAMES) + 3)]
    for m in MODELS:
        s = ctx["scores"][m]
        cells = [_ci(s[b]["brier"]) for b in F.BUCKET_NAMES]
        L.append("| %s%s | %s | %s | %s | %s |" % (m, " (selected)" if m == ctx["selected"] else "", " | ".join(cells),
                                                   _f(ctx["selection"][m]), _ci(s["all"]["log_loss"]), _ci(s["all"]["brier"])))
    L += ["", "Rows per bucket: " + ", ".join("%s %d" % (b, ctx["scores"][MODELS[0]][b]["n"]) for b in F.BUCKET_NAMES) + ".", "",
          "Selected: **%s** (lowest mean bucket Brier)." % ctx["selected"], "",
          "## Calibration by time bucket%s" % tag, "", "ECE (10 bins) and calibration slope per bucket, out-of-fold.", "",
          "| model | bucket | n | ECE | slope |", "| --- | --- | --- | --- | --- |"]
    for m in MODELS:
        for b in F.BUCKET_NAMES:
            s = ctx["scores"][m][b]
            if s["n"] == 0:
                continue
            L.append("| %s | %s | %d | %s | %s |" % (m, b, s["n"], _ci(s["ece"]), _ci(s["slope"])))
    L += ["", "The constant model has no calibration slope by design.", "",
          "### Reliability table of the selected candidate (out-of-fold)%s" % tag, "",
          "| bin | lo | hi | n | mean p | mean home win |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ctx["calibration"].itertuples():
        L.append("| %d | %.1f | %.1f | %d | %s | %s |" % (r.bin, r.lo, r.hi, r.n, _f(r.mean_p, 3), _f(r.mean_y, 3)))
    L += ["", "### winprob_calibration rows of the selected candidate (training seasons, out-of-fold)%s" % tag, "",
          "| season | time_bucket | n | mean_p | mean_y | brier | slope |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in ctx["calibration_rows"].itertuples():
        L.append("| %s | %s | %d | %s | %s | %s | %s |" % (r.season, r.time_bucket, r.n, _f(r.mean_p, 3), _f(r.mean_y, 3), _f(r.brier), _f(r.slope, 3)))
    L += ["", "## Held-out season %s, scored once%s" % (ctx["heldout_season"], tag), ""]
    if ctx["heldout_rows"]:
        hs = ctx["heldout_scores"]
        L += ["%d events from %d games of %s, never fitted or tuned on; the selected candidate (%s) refit on all %d training rows scored them once."
              % (ctx["heldout_rows"], ctx["heldout_games"], ctx["heldout_season"], ctx["selected"], ctx["train_rows"]), "",
              "| bucket | n | Brier | ECE | slope | benchmark (NBA.com WinProbabilityPBP) |", "| --- | --- | --- | --- | --- | --- |"]
        for b in F.BUCKET_NAMES:
            s = hs[b]
            if s["n"] == 0:
                continue
            L.append("| %s | %d | %s | %s | %s |  |" % (b, s["n"], _ci(s["brier"]), _ci(s["ece"]), _ci(s["slope"])))
        L += ["", "Overall: log loss %s, Brier %s, mean bucket Brier %s." % (_ci(hs["all"]["log_loss"]), _ci(hs["all"]["brier"]), _f(ctx["heldout_selection"]))]
    else:
        L.append("No rows of season %s were in the input, so nothing was scored." % ctx["heldout_season"])
    L += ["", "## Benchmark", "",
          "The NBA.com WinProbabilityPBP endpoint is unreachable from here, so the benchmark column is empty until it exists. "
          "When it does, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier.", "",
          "## Outputs", ""]
    L += ["- " + p for p in ctx["written"]] if ctx["written"] else ["- no tables written (no --out)"]
    L += ["", "## Mac command", "", "The real run over the five seasons under DATA_ROOT (collect/stints/playbyplay/<season>.parquet, collect/extras/odds/<season>.parquet when present):", "",
          "    " + MAC_CMD, "",
          "## Assumptions", "",
          "- A whole season is held out (default 2024-25) and scored once; folds inside the training seasons are grouped by game.",
          "- 2015-16 games of sportvu/splits.json never fit anything: rows flagged heldout are dropped, the 2015-16 season is refused as a held-out season, and refuse_heldout runs on the training ids.",
          "- The closing spread is the median over books of the is_close spreads market for the home side, joined on game_date and the home team name (analytics.winprob.features.TEAM_NAMES); null with a flag when absent.",
          "- The possession indicator is read from event kinds and is null when the events so far do not say.",
          "- The interaction is margin times sqrt(seconds remaining), as specified; the logistic model also sees margin and time on their own.",
          ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic fixtures (output is not a result)")
    src.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--report", default="reports/winprob.md")
    ap.add_argument("--out", default=None, help="OUT_ROOT for the winprob_states and winprob_calibration tables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--heldout-season", default="2024-25")
    ap.add_argument("--fixture-games", type=int, default=20, help="games per fixture season")
    ap.add_argument("--fixture-seasons", default="2020-21,2021-22,2024-25", help="fixture seasons, comma separated")
    return ap.parse_args(argv)


def run(d: dict, heldout_season: str, seed: int, n_boot: int, n_folds: int, out_root=None, fixtures: bool = False,
        command: str = "", report: str | None = None) -> dict:
    states = F.build_states(d["playbyplay"], d["games"], d["odds"])
    viol = F.prefix_invariance_violations(d["playbyplay"], d["games"], d["odds"], n_games=2, cuts=2, seed=seed)
    if viol:
        raise SystemExit("%s: a feature changed when the play-by-play was cut: %s" % (WHO, viol[:3]))
    train, heldout = split_seasons(states, heldout_season)
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
           "heldout_rows": int(len(heldout)), "heldout_games": int(heldout["game_id"].nunique()),
           "dropped_heldout_rows": int(states["heldout"].astype(bool).sum()),
           "spread_coverage": float((train.drop_duplicates("game_id")["spread_missing"] == 0).mean()),
           "possession_known": float(train["possession_home"].notna().mean()),
           "home_win_rate": float(train.drop_duplicates("game_id")[F.TARGET].mean()),
           "prefix_violations": len(viol), "prefix_games": 2, "protocol": note["text"], "bootstrap": n_boot, "seed": seed,
           "scores": scores, "selection": selection, "selected": selected,
           "calibration": calibration_table(pred[F.TARGET].astype(float).to_numpy(), pred[p_sel].astype(float).to_numpy()),
           "calibration_rows": pd.concat([calibration_rows(s, p_sel, str(season)) for season, s in pred.groupby("season", sort=True)], ignore_index=True),
           "heldout_scores": {}, "heldout_selection": float("nan"), "written": [], "pred": pred, "heldout_pred": None}
    if len(heldout):
        model = fit_models(train, seed, models=(selected,))[selected]
        hp = heldout.copy()
        hp[p_sel] = model(heldout)
        hp["fold"] = -1
        ctx["heldout_scores"] = bucket_scores(hp, p_sel, n_boot, seed)
        ctx["heldout_selection"] = selection_score(ctx["heldout_scores"])
        ctx["heldout_pred"] = hp
    if out_root:
        ctx["written"] = write_tables(out_root, pred, ctx["heldout_pred"], p_sel)
    if report:
        ctx["report"] = write_report(Path(report), ctx)
    return ctx


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.fixtures:
        d = F.load_fixtures(a.seed, n_games=a.fixture_games, seasons=tuple(s for s in a.fixture_seasons.split(",") if s))
    else:
        d = F.load_data(a.data)
    cmd = "python -m analytics.winprob.train " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = run(d, a.heldout_season, a.seed, a.bootstrap, a.folds, out_root=a.out, fixtures=a.fixtures, command=cmd, report=a.report)
    print(ctx["report"])
    return ctx


if __name__ == "__main__":
    main()
