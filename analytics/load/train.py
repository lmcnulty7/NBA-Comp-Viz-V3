"""analytics/load/train.py: three load and fatigue baselines under the evaluation harness, and the report.

    python -m analytics.load.train --fixtures | --data DATA_ROOT --report reports/load.md [--out OUT_ROOT]
                                   [--folds 5] [--seed 0] [--bootstrap 200] [--season 2015-16] [--limit N]
                                   [--fixture-games 8] [--fixture-possessions 40] [--fixture-heldout-every 4]
                                   [--skip-heldout]

Steps: load the inputs (analytics.load.validate), build the tables (analytics.load.features), set every
held-out player-game aside, call analytics.eval.refuse_heldout on the training game ids, run
schema.assert_time_forward and assert_no_leak on the design matrix, then:

  1. descriptive: mean decline_ftps by rest bucket (back_to_back, one_day, two_plus) with bootstrap intervals
     by game; the per-player load_summary, ranked only among players with MIN_STINTS_RANK stints (gated);
  2. ridge regression (alpha 1, median imputation and standardisation fitted on the training folds) of
     decline_ftps on FEATURES, grouped folds by sportvu_game, MSE against the training-fold mean with a paired
     bootstrap by game of the difference;
  3. absence: logistic regression (C 1, same preprocessing) of absent_next on the same FEATURES, log loss and
     Brier against the base rate, ECE, calibration slope and the calibration table. Absence is not injury.

No model is tuned: alpha and C are fixed here before any run. Fewer games than folds: n_folds becomes the
number of games. Held-out rows are scored once by the models refit on every training row and labelled as
such; with --skip-heldout they are set aside exactly as without it but never scored, and the report says "held-out
games not scored (--skip-heldout)" (the descriptive player_stints and player_games tables of held-out games still go
under heldout/<season>/ with --out, as the features step writes them: measurements, not predictions). With
--fixtures the report says "fixture output, not a result" on every table. Deep models: none.
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
                            fmt_num, grouped_folds, heldout_mask, log_loss, paired_bootstrap_by_game, refuse_heldout,
                            add_skip_heldout_arg, skip_heldout_lines)
from analytics.load import validate as V
from analytics.load.eda import markdown, rest_bucket
from analytics.load.features import FEATURES, MIN_STINTS_RANK, assert_no_leak, build_tables, design, write_tables

WHO = "analytics.load.train"
RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
# the first cell of each candidate row in the report tables; the "Selected candidate:" lines name the row the same way
REG_ROWS = {"mean": "mean (training-fold mean)", "ridge": "ridge on FEATURES"}
ABS_ROWS = {"base": "base rate (training-fold rate)", "logistic": "logistic on FEATURES"}
MAC_CMD = ("python -m analytics.load.train --data \"$DATA_ROOT\" --report reports/load.md --out \"$DATA_ROOT\" "
           "--folds 5 --seed 0 --bootstrap 1000")


# ── frames ─────────────────────────────────────────────────────────────────────────────────────────
def training_frames(games_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (train, heldout) player-game frames. The guard runs on the training ids; held-out rows are only scored."""
    held = games_df["heldout"].astype(bool).to_numpy() | heldout_mask(games_df["sportvu_game"])
    train = games_df[~held].reset_index(drop=True)
    heldout = games_df[held].reset_index(drop=True)
    refuse_heldout(sorted(train["sportvu_game"].unique()), WHO)
    schema.assert_time_forward(train[FEATURES], "rosters")
    assert_no_leak(train[FEATURES])
    return train, heldout


def regression_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The player-games with a finite decline_ftps (the regression target)."""
    return df[np.isfinite(df["decline_ftps"].astype(float))].reset_index(drop=True)


def absence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The player-games with a known absent_next (the absence target)."""
    return df[df["absent_next"].notna()].reset_index(drop=True)


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def _pre():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]


def fit_regression(train: pd.DataFrame) -> dict:
    """{mean, ridge}: predict(frame) -> decline estimates. Every regression fit goes through here."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_regression" % WHO)
    X, y = design(train), train["decline_ftps"].astype(float).to_numpy()
    mu = float(y.mean())
    out = {"mean": lambda frame, m=mu: np.full(len(frame), m)}
    if len(y) < 3:
        out["ridge"] = out["mean"]
    else:
        with fit_threads():
            m = make_pipeline(*_pre(), Ridge(alpha=RIDGE_ALPHA)).fit(X.to_numpy(), y)
        out["ridge"] = lambda frame, m=m: m.predict(design(frame).to_numpy())
    return out


def fit_absence(train: pd.DataFrame) -> dict:
    """{base, logistic}: predict(frame) -> probability of absent_next."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    if train["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_absence" % WHO)
    X, y = design(train), train["absent_next"].astype(bool).astype(int).to_numpy()
    rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    out = {"base": lambda frame, r=rate: np.full(len(frame), r)}
    if len(np.unique(y)) < 2 or len(y) < 5:
        out["logistic"] = out["base"]
    else:
        with fit_threads():
            m = make_pipeline(*_pre(), LogisticRegression(C=LOGISTIC_C, max_iter=1000)).fit(X.to_numpy(), y)
        out["logistic"] = lambda frame, m=m: m.predict_proba(design(frame).to_numpy())[:, 1]
    return out


def out_of_fold(rows: pd.DataFrame, fit_fn, names, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Predictions p_<name> of fit_fn's models under grouped folds by sportvu_game (in-sample with one game) and a note."""
    games = sorted(rows["sportvu_game"].unique())
    pred = rows.copy()
    for m in names:
        pred["p_" + m] = np.nan
    if len(games) < 2:
        models = fit_fn(rows)
        for m in names:
            pred["p_" + m] = models[m](rows)
        return pred, {"mode": "in-sample", "folds": 0, "games": len(games),
                      "text": "one training game: no grouped fold is possible, metrics are IN-SAMPLE"}
    k = min(n_folds, len(games))
    for tr_games, te_games in grouped_folds(games, n_folds=k, seed=seed):
        tr, te = rows["sportvu_game"].isin(tr_games), rows["sportvu_game"].isin(te_games)
        models = fit_fn(rows[tr])
        for m in names:
            pred.loc[te, "p_" + m] = models[m](rows[te])
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games" % (len(games), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────
def mse(y, p) -> float:
    """Mean squared error."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def score_regression(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """MSE of the mean and ridge predictions with bootstrap intervals by game, and their paired difference."""
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": pred["decline_ftps"].astype(float).to_numpy(),
                      "mean": pred["p_mean"].astype(float).to_numpy(), "ridge": pred["p_ridge"].astype(float).to_numpy()})
    out = {"rows": int(len(d)), "games": int(d["game"].nunique())}
    for m in ("mean", "ridge"):
        r = bootstrap_by_game(d, "game", lambda s, m=m: mse(s["y"], s[m]), n=n_boot, seed=seed)
        if out["games"] < 2:
            r = {**r, "lo": np.nan, "hi": np.nan}
        out[m] = r
    out["diff"] = paired_bootstrap_by_game(d, "game", lambda s: mse(s["y"], s["ridge"]), lambda s: mse(s["y"], s["mean"]),
                                           n=n_boot, seed=seed)
    return out


def score_absence(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Log loss, Brier and ECE of the base and logistic predictions by game bootstrap, the paired difference, the
    calibration slope and the calibration table of the logistic model."""
    y = pred["absent_next"].astype(bool).astype(float).to_numpy()
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": y,
                      "base": pred["p_base"].astype(float).to_numpy(), "logistic": pred["p_logistic"].astype(float).to_numpy()})
    out = {"rows": int(len(d)), "games": int(d["game"].nunique()), "base_rate": float(y.mean())}
    for m in ("base", "logistic"):
        out[m] = {}
        for name, fn in (("log_loss", log_loss), ("brier", brier), ("ece", ece)):
            r = bootstrap_by_game(d, "game", lambda s, f=fn, m=m: f(s["y"], s[m]), n=n_boot, seed=seed)
            if out["games"] < 2:
                r = {**r, "lo": np.nan, "hi": np.nan}
            out[m][name] = r
    out["diff"] = paired_bootstrap_by_game(d, "game", lambda s: log_loss(s["y"], s["logistic"]),
                                           lambda s: log_loss(s["y"], s["base"]), n=n_boot, seed=seed)
    two = len(np.unique(y)) == 2
    # the slope is one fit (a point estimate), solved by the bounded Newton method of the harness
    out["logistic"]["calibration_slope"] = calibration_slope_fast(y, d["logistic"].to_numpy()) if two else np.nan
    out["calibration"] = calibration_table(y, d["logistic"].to_numpy()) if two else None
    return out


def rest_bucket_table(train: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Mean decline per rest bucket with bootstrap intervals by game (candidate 1)."""
    r = regression_rows(train).assign(bucket=rest_bucket(train.loc[np.isfinite(train["decline_ftps"].astype(float)), "days_rest"]).to_numpy())
    rows = []
    for b in ("back_to_back", "one_day", "two_plus"):
        part = r[r["bucket"] == b]
        row = {"bucket": b, "n": int(len(part)), "games": int(part["sportvu_game"].nunique()),
               "mean_decline_ftps": np.nan, "lo": np.nan, "hi": np.nan}
        if len(part):
            d = pd.DataFrame({"game": part["sportvu_game"].astype(str).to_numpy(), "y": part["decline_ftps"].astype(float).to_numpy()})
            bs = bootstrap_by_game(d, "game", lambda s: float(s["y"].mean()), n=n_boot, seed=seed)
            row.update({"mean_decline_ftps": bs["point"], "lo": bs["lo"] if row["games"] > 1 else np.nan,
                        "hi": bs["hi"] if row["games"] > 1 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregates_check(games_df: pd.DataFrame, aggregates: pd.DataFrame | None) -> dict | None:
    """Tracking totals against the leaguedashptstats SpeedDistance aggregates (section 11) when present: per
    player, our mean distance per game in miles and mean speed in mph against dist_miles and avg_speed; Spearman
    rank correlations over the matched players. Non-held-out rows only."""
    if aggregates is None or not len(aggregates):
        return None
    a = aggregates
    if "endpoint" in a.columns:
        a = a[a["endpoint"].astype(str) == "leaguedashptstats"]
    if "pt_measure_type" in a.columns:
        a = a[a["pt_measure_type"].astype(str).str.lower() == "speeddistance"]
    if "heldout" in a.columns:
        a = a[~a["heldout"].fillna(False).astype(bool)]
    if not len(a) or "metric" not in a.columns:
        return None
    wide = a.pivot_table(index="player_id", columns="metric", values="value", aggfunc="mean")
    g = games_df[~games_df["heldout"].astype(bool)]
    ours = g.groupby("player_id").agg(games=("sportvu_game", "nunique"), dist_ft=("distance_ft", "mean"),
                                      speed_ftps=("avg_speed_ftps", "mean"))
    ours["dist_miles_ours"] = ours["dist_ft"] / 5280.0
    ours["avg_speed_mph_ours"] = ours["speed_ftps"] * 3600.0 / 5280.0
    j = ours.join(wide, how="inner")
    out = {"players_matched": int(len(j)), "aggregate_rows": int(len(a))}
    for ours_col, theirs in (("dist_miles_ours", "dist_miles"), ("avg_speed_mph_ours", "avg_speed")):
        if theirs in j.columns and len(j) >= 3:
            out[theirs] = {"spearman": float(j[ours_col].corr(j[theirs], method="spearman")),
                           "ours_mean": float(j[ours_col].mean()), "theirs_mean": float(j[theirs].mean())}
    return out


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def write_report(path: Path, ctx: dict) -> str:
    """The markdown report of a run (ctx from run); every heading over fixture numbers carries the fixture tag."""
    fx = ctx["fixture"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Load and fatigue baselines (analytics/load/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Every number below is fixture output, not a result**: the inputs are a synthetic season (%s). The real "
              "run is the Mac command at the end." % ctx["source"], ""]
    L += ["## Inputs%s" % tag, "",
          "| games loaded | training games | held-out games | stints | player-games | players | aggregates rows |",
          "| --- | --- | --- | --- | --- | --- | --- |",
          "| %d | %d | %d | %d | %d | %d | %d |" % (ctx["n_games"], ctx["train_games"], ctx["heldout_games"], ctx["n_stints"],
                                                    ctx["n_player_games"], ctx["n_players"], ctx["aggregates_rows"]), "",
          "Validation totals over the loaded frames: %s." % ", ".join("%s %s" % (k, v) for k, v in ctx["validation"].items()),
          "", ctx["injuries"] + ".", "",
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids against sportvu/splits.json; "
          "`schema.assert_time_forward` and `assert_no_leak` passed on the design matrix (%s)." % (ctx["train_games"], ", ".join(FEATURES)),
          "", "## Target definitions and their limits", "",
          "- Unit: the player-stint, a continuous on-court span of one player in the frames (a new stint when the period "
          "changes or the game clock jumps by more than 5 s between consecutive rows of the player); positions smoothed "
          "over 5 rows; steps count when their wall-clock dt is in (0, 1] s. Dead-ball movement counts.",
          "- y_decline (`decline_ftps`): average speed of the player's last stint minus the first, over the stints of at "
          "least 1 minute, in ft per s; NaN with fewer than two such stints. `slope_ftps_per_stint` is the least-squares "
          "slope over those stints. Limits: average speed mixes role and game state (a blowout fourth quarter, a closing "
          "lineup) with fatigue; a player whose late stints are short defensive stands reads as declining; one game is noisy.",
          "- `absent_next`: the team's next manifest game has a loaded roster and the player is not on it. Absence is "
          "not injury: rest, a coach's decision, a suspension, a trade or a tracking miss read the same. The section 15 "
          "injuries table starts in 2021-22 and never overlaps the tracking era, so no injury label exists here.",
          "- Features come from the manifest schedule and from previous games only (minutes_last_3, distance_last_3_ft, "
          "prev_decline_ftps, prev_avg_speed_ftps); same-game totals and later stints are never features. A held-out game "
          "is never read for a feature (the window is NaN instead).",
          "", "## Candidate 1: mean decline by rest bucket%s" % tag, "",
          "Buckets by days_rest: back_to_back (1 or fewer), one_day (2), two_plus (3 or more, including a team's first "
          "listed game, which takes the cap of 7). Bootstrap intervals by game (n=%d, seed=%d); an interval needs at "
          "least two games in the bucket." % (ctx["bootstrap"], ctx["seed"]), "",
          markdown(ctx["rest_table"]), "",
          "Per-player summary (load_summary): %d players, %d with at least %d stints (the others are gated out of any "
          "ranking, `insufficient`). The %d largest mean declines among the gated-in players:" % (
              ctx["summary_players"], ctx["summary_enough"], MIN_STINTS_RANK, len(ctx["summary_top"])), "",
          markdown(ctx["summary_top"]) if len(ctx["summary_top"]) else "(no player passes the gate)", "",
          "## Candidate 2: ridge regression of y_decline%s" % tag, "",
          "%s. %d player-games with a decline from %d games. Ridge alpha %.1f, median imputation and standardisation "
          "fitted inside the training folds." % (ctx["reg_note"]["text"], ctx["reg"]["rows"], ctx["reg"]["games"], RIDGE_ALPHA), "",
          "| model | MSE (ft per s)^2 |", "| --- | --- |",
          "| %s | %s |" % (REG_ROWS["mean"], fmt_ci(ctx["reg"]["mean"], 4)),
          "| %s | %s |" % (REG_ROWS["ridge"], fmt_ci(ctx["reg"]["ridge"], 4)), "",
          "Paired bootstrap by game of MSE(ridge) - MSE(mean): %s [%s, %s], excludes zero: %s." % (
              fmt_num(ctx["reg"]["diff"]["diff"], 4), fmt_num(ctx["reg"]["diff"]["lo"], 4), fmt_num(ctx["reg"]["diff"]["hi"], 4), ctx["reg"]["diff"]["excludes_zero"]),
          "", "Selected candidate: %s." % REG_ROWS[ctx["reg_selected"]],
          "", "## Candidate 3: absence from the next game%s" % tag, ""]
    ab = ctx["abs"]
    if ab is None:
        L += ["No player-game has a known absent_next (no loaded next game), so the absence baseline did not run.", ""]
    else:
        L += ["%s. %d player-games with a known absent_next from %d games, base rate %s. Logistic C %.1f, same preprocessing. "
              "Absence is not injury." % (ctx["abs_note"]["text"], ab["rows"], ab["games"], fmt_num(ab["base_rate"], 4), LOGISTIC_C), "",
              "| model | log loss | Brier | ECE (10 bins) |", "| --- | --- | --- | --- |",
              "| %s | %s | %s | %s |" % (ABS_ROWS["base"], fmt_ci(ab["base"]["log_loss"], 4), fmt_ci(ab["base"]["brier"], 4), fmt_ci(ab["base"]["ece"], 4)),
              "| %s | %s | %s | %s |" % (ABS_ROWS["logistic"], fmt_ci(ab["logistic"]["log_loss"], 4), fmt_ci(ab["logistic"]["brier"], 4), fmt_ci(ab["logistic"]["ece"], 4)), "",
              "Paired bootstrap by game of log loss(logistic) - log loss(base): %s [%s, %s], excludes zero: %s. "
              "Calibration slope of the logistic model: %s (one fit, no interval; 1 is calibrated)." % (
                  fmt_num(ab["diff"]["diff"], 4), fmt_num(ab["diff"]["lo"], 4), fmt_num(ab["diff"]["hi"], 4), ab["diff"]["excludes_zero"],
                  fmt_num(ab["logistic"]["calibration_slope"], 4)), "",
              "Selected candidate: %s." % ABS_ROWS[ctx["abs_selected"]], ""]
        if ab["calibration"] is None:
            L += ["One class only in absent_next: no calibration table, and the logistic model reduces to the base rate.%s" % (
                " On the analytics.fixtures season every listed player plays every game, so no player is ever absent."
                if fx and ab["base_rate"] == 0 else ""), ""]
        else:
            L += ["Calibration table of the logistic model (equal-width bins of p):", "",
                  "| bin | lo | hi | n | mean p | mean absent |", "| --- | --- | --- | --- | --- | --- |"]
            for r in ab["calibration"].itertuples():
                L.append("| %d | %.1f | %.1f | %d | %s | %s |" % (r.bin, r.lo, r.hi, r.n, fmt_num(r.mean_p, 3), fmt_num(r.mean_y, 3)))
            L.append("")
    agg = ctx["aggregates"]
    L += ["## Cross-check against the tracking aggregates (section 11)%s" % tag, ""]
    if agg is None:
        L.append("No leaguedashptstats SpeedDistance aggregates were present; the cross-check did not run.")
    else:
        L += ["%d players matched over %d aggregate rows." % (agg["players_matched"], agg["aggregate_rows"]), "",
              "| quantity | Spearman (ours vs aggregate) | ours mean | aggregate mean |", "| --- | --- | --- | --- |"]
        for k, label in (("dist_miles", "distance per game, miles"), ("avg_speed", "average speed, mph")):
            if k in agg:
                L.append("| %s | %s | %s | %s |" % (label, fmt_num(agg[k]["spearman"], 3), fmt_num(agg[k]["ours_mean"], 3), fmt_num(agg[k]["theirs_mean"], 3)))
        L += ["", "Our distance counts dead-ball movement at the wall clock and only frames that passed validation; the "
              "aggregate is NBA.com's own, so a gap in level with a high rank correlation is the expected picture."]
    L += ["", "## Held-out games, %s%s" % ("not scored" if ctx.get("skip_heldout") else "scored once", tag), ""]
    ho = ctx["heldout"]
    if ctx.get("skip_heldout"):
        L += skip_heldout_lines(ctx["heldout_games"])
    elif ho is None:
        L.append("No held-out player-game was loaded.")
    else:
        L += ["%d player-games from %d held-out game(s) (%s). One model per pair was selected on its out-of-fold score "
              "and scored once here, refit on all training rows; never fitted or tuned on. The constant baseline appears "
              "only as a labelled floor under a selected candidate. Point estimates only." % (ho["rows"], ho["games"], ", ".join(ho["ids"])), "",
              "| quantity | value |", "| --- | --- |",
              "| decline rows | %d |" % ho["reg_rows"],
              "| MSE, selected model (%s) | %s |" % (ho["reg_selected"], fmt_num(ho.get("mse_selected"), 4))]
        if "mse_floor" in ho:
            L.append("| MSE, floor (training mean, not selected) | %s |" % fmt_num(ho["mse_floor"], 4))
        L += ["| absence rows | %d |" % ho["abs_rows"],
              "| log loss, selected model (%s) | %s |" % (ho["abs_selected"], fmt_num(ho.get("ll_selected"), 4))]
        if "ll_floor" in ho:
            L.append("| log loss, floor (base rate, not selected) | %s |" % fmt_num(ho["ll_floor"], 4))
    L += ["", "## Status and the Mac command", ""]
    if fx:
        L.append("Fixture output, not a result. The pipeline is tested (tests/test_load.py); the real run on the owner's Mac, "
                 "where DATA_ROOT holds sportvu_contract/, analytics/rosters/ and, when present, collect/tracking/aggregates/, is")
    else:
        L.append("The run of this protocol was")
    L += ["", "    " + MAC_CMD, "",
          "Tables written under OUT_ROOT/analytics/load/<season>/ (held-out rows under heldout/<season>/): player_stints, "
          "player_games, load_summary (docs/contract/load.md).", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The train CLI: the shared source and fixture arguments of analytics.load.validate plus the protocol knobs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    V.add_source_args(ap)
    ap.add_argument("--report", default="reports/load.md", help="markdown report path (default reports/load.md)")
    ap.add_argument("--out", help="OUT_ROOT: also write the three tables")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by sportvu_game (default 5)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples of whole games (default 200)")
    add_skip_heldout_arg(ap)
    return ap.parse_args(argv)


def run(inputs: dict, n_folds: int = 5, seed: int = 0, n_boot: int = 200, fixture: bool = False, tables: dict | None = None,
        skip_heldout: bool = False) -> dict:
    """The whole protocol on an inputs dict -> ctx (what write_report renders), with the tables under 'tables'. With
    skip_heldout the held-out player-games are set aside as usual but never scored."""
    tables = tables or build_tables(inputs, n_boot=n_boot, seed=seed)
    pg, st, sm = tables["player_games"], tables["player_stints"], tables["load_summary"]
    train, heldout = training_frames(pg)
    if not len(train):
        raise SystemExit("%s: no training player-games after setting the held-out rows aside" % WHO)
    reg_rows = regression_rows(train)
    reg_pred, reg_note = out_of_fold(reg_rows, fit_regression, ("mean", "ridge"), n_folds, seed)
    reg = score_regression(reg_pred, n_boot, seed)
    abs_rows = absence_rows(train)
    if len(abs_rows) and abs_rows["sportvu_game"].nunique() >= 1:
        abs_pred, abs_note = out_of_fold(abs_rows, fit_absence, ("base", "logistic"), n_folds, seed)
        ab = score_absence(abs_pred, n_boot, seed)
    else:
        abs_pred, abs_note, ab = None, {"text": "no rows"}, None
    enough = sm[~sm["insufficient"].astype(bool)]
    top = enough.sort_values("mean_decline").head(10).reset_index(drop=True)
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "fixture": fixture,
           "source": inputs["source"], "n_games": len(inputs["games"]), "train_games": int(train["sportvu_game"].nunique()),
           "heldout_games": int(heldout["sportvu_game"].nunique()), "n_stints": int(len(st)), "n_player_games": int(len(pg)),
           "n_players": int(pg["player_id"].nunique()),
           "aggregates_rows": int(len(inputs["aggregates"])) if inputs.get("aggregates") is not None else 0,
           "validation": _totals(tables["logs"]), "injuries": V.injuries_note(inputs.get("injuries")),
           "bootstrap": n_boot, "seed": seed, "rest_table": rest_bucket_table(train, n_boot, seed),
           "summary_players": int(len(sm)), "summary_enough": int(len(enough)), "summary_top": top,
           "reg": reg, "reg_note": reg_note, "abs": ab, "abs_note": abs_note,
           "aggregates": aggregates_check(pg, inputs.get("aggregates")), "heldout": None, "tables": tables,
           "reg_pred": reg_pred, "abs_pred": abs_pred, "skip_heldout": bool(skip_heldout),
           # one model per pair, selected on the out-of-fold score (lower wins, ties to the constant baseline)
           "reg_selected": "ridge" if reg["ridge"]["point"] <= reg["mean"]["point"] else "mean",
           "abs_selected": ("logistic" if ab["logistic"]["log_loss"]["point"] <= ab["base"]["log_loss"]["point"] else "base") if ab else "base"}
    if len(heldout) and not skip_heldout:
        ho = {"rows": int(len(heldout)), "games": int(heldout["sportvu_game"].nunique()), "ids": sorted(heldout["sportvu_game"].unique())}
        # one model per pair is selected on the out-of-fold score and scored on the held-out rows; the constant
        # baseline is scored only as a labelled floor under a selected candidate, never as a second candidate
        hr = regression_rows(heldout)
        ho["reg_rows"] = int(len(hr))
        ho["reg_selected"] = ctx["reg_selected"]
        if len(hr) and len(reg_rows):
            models = fit_regression(reg_rows)
            y = hr["decline_ftps"].astype(float).to_numpy()
            ho["mse_selected"] = mse(y, models[ho["reg_selected"]](hr))
            if ho["reg_selected"] != "mean":
                ho["mse_floor"] = mse(y, models["mean"](hr))
        ha = absence_rows(heldout)
        ho["abs_rows"] = int(len(ha))
        ho["abs_selected"] = ctx["abs_selected"]
        if len(ha) and len(abs_rows):
            models = fit_absence(abs_rows)
            y = ha["absent_next"].astype(bool).astype(float).to_numpy()
            ho["ll_selected"] = log_loss(y, models[ho["abs_selected"]](ha))
            if ho["abs_selected"] != "base":
                ho["ll_floor"] = log_loss(y, models["base"](ha))
        ctx["heldout"] = ho
    return ctx


def _totals(logs: dict) -> dict:
    keys = ("rows_in", "rows_missing_key_or_position", "rows_duplicate_frame_player", "rows_outside_court",
            "rows_speed_over_30_ftps", "players_distance_over_ceiling", "rows_out")
    return {k: int(sum(l.get(k, 0) for l in logs.values())) for k in keys}


def main(argv=None) -> dict:
    """Run the protocol from the command line, write the report (and the tables with --out) and return ctx."""
    a = parse_args(argv)
    inputs = V.load_args(a)
    ctx = run(inputs, n_folds=a.folds, seed=a.seed, n_boot=a.bootstrap, fixture=a.fixtures, skip_heldout=a.skip_heldout)
    ctx["command"] = "python -m analytics.load.train " + " ".join(argv if argv is not None else sys.argv[1:])
    if a.out:
        for p in write_tables(ctx["tables"], a.out, inputs["season"]):
            print("wrote %s" % p, file=sys.stderr)
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
