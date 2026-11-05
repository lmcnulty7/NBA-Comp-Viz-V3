"""analytics/xfg/train.py: three xFG candidates under the Phase 0 protocol, selection by out-of-fold log loss.

    python -m analytics.xfg.train --fixtures | --data DATA_ROOT --report reports/xfg.md [--folds 5] [--seed 0]
                                  [--bootstrap 200] [--out OUT_ROOT] [--season 2015-16]

Candidates (CANDIDATES, in order):
    location_logistic   the Phase 1 baseline: analytics.train_xfg.fit_models on the four location features
    contest_logistic    logistic regression on FEATURES (location, contest, has_contest, shot clock, period, clock)
                        after median imputation and standardisation (C=1)
    hgb_contest_prior   sklearn HistGradientBoostingClassifier on FEATURES plus the shooter prior, calibrated by
                        isotonic regression fitted on out-of-fold predictions inside the training folds

Shooter prior: the shooter's make rate by zone (rim, paint, mid, corner3, above3), smoothed toward the zone rate
with PRIOR_M = 20 pseudo-attempts: (makes + m * zone_rate) / (n + m). For the rows a model is fitted on it is
computed out-of-fold (INNER_FOLDS grouped folds by game inside the fitting rows), for the rows it predicts from
all fitting rows. The row's own outcome, and its game's, never enter its prior. The isotonic map is fitted on
inner out-of-fold predictions whose inner training rows got their own nested out-of-fold prior, so no label of
an inner test row reaches the booster that scores it.

Protocol: analytics.eval.grouped_folds by sportvu_game (the same folds for every candidate, via grouped_cv);
selection by out-of-fold log loss; Brier, ECE (10 bins, target under 0.02) and calibration slope reported with
bootstrap intervals by game; paired bootstrap by game of candidate 3 against 1 (and 2 against 1) on log loss.
Rows whose heldout flag is true, or whose game id carries a held-out token of sportvu/splits.json, are set
aside before any fit; analytics.eval.refuse_heldout runs on the training game ids and again on every fit; the
feature frame passed schema.assert_time_forward in analytics.xfg.features and every design matrix passes
assert_no_leak. Held-out rows are scored once, by the candidates refit on every training row; the held-out
score reported is the selected candidate's.

Fewer training games than folds: n_folds becomes the number of games (leave-one-game-out) and the report says
so. One training game: SystemExit (no grouped fold is possible; this trainer does not report in-sample numbers).

Output: the report (markdown) and, with --out (default DATA_ROOT for --data, none for --fixtures), the table
xfg_predictions (docs/contract/xfg.md) under OUT_ROOT/analytics/xfg/<season>/xfg_predictions.parquet.
Under --fixtures every number is labelled "fixture output, not a result".
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import train_xfg as baseline
from analytics.eval import (bootstrap_by_game, brier, calibration_slope, calibration_table, ece, grouped_cv,
                            grouped_folds, log_loss, paired_bootstrap_by_game, refuse_heldout, report_block)
from analytics.xfg import features as FX
from analytics.xfg import validate as V
from analytics.xfg.features import FEATURES, PRIOR_FEATURE, assert_no_leak, build, feature_columns, targets

log = logging.getLogger("analytics.xfg.train")

WHO = "analytics.xfg.train"
CANDIDATES = ("location_logistic", "contest_logistic", "hgb_contest_prior")
SHORT = {"location_logistic": "location", "contest_logistic": "contest", "hgb_contest_prior": "selected"}
METRICS = (("log_loss", log_loss), ("brier", brier), ("ece", ece), ("calibration_slope", calibration_slope))
PRIOR_M = 20.0
INNER_FOLDS = 3
ECE_TARGET = 0.02
ISO_CLIP = 0.02              # the isotonic map never returns a probability under ISO_CLIP or over 1 - ISO_CLIP
HGB_PARAMS = dict(max_iter=100, learning_rate=0.1, max_depth=3, min_samples_leaf=10, l2_regularization=1.0,
                  early_stopping=False)
FULL_RUN_CMD = ("python -m analytics.xfg.train --data \"$DATA_ROOT\" --report reports/xfg.md --folds 5 --seed 0 "
                "--bootstrap 1000")
PREDICTION_COLUMNS = ["sportvu_game", "event", "player_id", "p_location", "p_contest", "p_selected", "made", "fold", "heldout"]


def _threads(n: int = 1):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


# ----------------------------------------------------------------------------- inputs

def assemble(tables: dict) -> tuple:
    """Cleaned tables -> (train, heldout) frames: features plus the label, held-out rows set aside, guard run."""
    feat = build(tables["shots"], tables["shot_frames"], tables.get("frames"))
    y = targets(tables["shots"].reset_index(drop=True))
    df = feat.copy()
    df["made"] = y["made"].to_numpy()
    held = baseline.heldout_mask(df)
    train, heldout = df[~held].reset_index(drop=True), df[held].reset_index(drop=True)
    refuse_heldout(sorted(train["sportvu_game"].unique()), WHO)
    assert not train["heldout"].any()
    return train, heldout


# ----------------------------------------------------------------------------- shooter prior

def prior_table(rows: pd.DataFrame, m: float = PRIOR_M) -> dict:
    """Smoothed make rate by (player_id, zone) from ``rows`` only."""
    y = rows["made"].astype(float)
    base = float(y.mean()) if len(rows) else 0.45
    zone_rate = rows.groupby("zone")["made"].mean().astype(float).to_dict()
    g = rows.groupby(["player_id", "zone"])["made"].agg(["sum", "count"])
    table = {}
    for (pid, zone), r in g.iterrows():
        zr = zone_rate.get(zone, base)
        table[(int(pid), str(zone))] = (float(r["sum"]) + m * zr) / (float(r["count"]) + m)
    return {"table": table, "zone_rate": zone_rate, "base": base}


def apply_prior(pt: dict, rows: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(rows))
    pids = pd.to_numeric(rows["player_id"], errors="coerce").to_numpy()
    zones = rows["zone"].astype(str).to_numpy()
    for i in range(len(rows)):
        key = (int(pids[i]) if np.isfinite(pids[i]) else -1, zones[i])
        out[i] = pt["table"].get(key, pt["zone_rate"].get(zones[i], pt["base"]))
    return out


def oof_prior(rows: pd.DataFrame, n_folds: int = INNER_FOLDS, seed: int = 0) -> np.ndarray:
    """The prior of every row from the other grouped folds (by game), never from its own game."""
    games = rows["sportvu_game"].astype(str)
    k = min(n_folds, games.nunique())
    out = np.full(len(rows), np.nan)
    if k < 2:
        pt = prior_table(rows.iloc[0:0])
        return np.full(len(rows), pt["base"])
    for tr, te in grouped_folds(games.unique(), k, seed):
        m = games.isin(te).to_numpy()
        out[m] = apply_prior(prior_table(rows[~m]), rows[m])
    return out


# ----------------------------------------------------------------------------- candidates

def _design(frame: pd.DataFrame, cols: list) -> np.ndarray:
    X = frame[cols].astype(float)
    assert_no_leak(X)
    return X.to_numpy()


def fit_candidate(name: str, train: pd.DataFrame, seed: int = 0):
    """Fit one candidate on ``train`` -> predict(frame) -> p. Every fit of this module goes through here, so the
    guards (held-out flag, held-out ids) run on exactly what reaches a fit. Fits run single-threaded (the shared
    box has 4 cores and many lanes; OpenMP spin-waits are slower than one thread there)."""
    with _threads(1):
        return _fit_candidate(name, train, seed)


def _fit_candidate(name: str, train: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_candidate" % WHO)
    refuse_heldout(sorted(train["sportvu_game"].astype(str).unique()), WHO)
    y = train["made"].astype(int).to_numpy()
    rate = float(y.mean()) if len(y) else 0.45
    constant = lambda frame, r=rate: np.full(len(frame), r)  # noqa: E731
    if len(np.unique(y)) < 2:
        return constant
    if name == "location_logistic":
        return baseline.fit_models(train)["logistic"]
    if name == "contest_logistic":
        cols = feature_columns()
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        pipe.fit(_design(train, cols), y)
        return lambda frame, m=pipe, c=cols: m.predict_proba(_design(frame, c))[:, 1]
    if name == "hgb_contest_prior":
        cols = feature_columns(with_prior=True)
        games = train["sportvu_game"].astype(str)
        k = min(INNER_FOLDS, games.nunique())
        # isotonic map from inner out-of-fold predictions (nested priors, so no inner test label reaches its booster)
        oof = np.full(len(train), np.nan)
        if k >= 2:
            for tr_g, te_g in grouped_folds(games.unique(), k, seed):
                m = games.isin(te_g).to_numpy()
                inner = train[~m].copy()
                inner[PRIOR_FEATURE] = oof_prior(inner, INNER_FOLDS, seed)
                test = train[m].copy()
                test[PRIOR_FEATURE] = apply_prior(prior_table(inner), test)
                h = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS).fit(_design(inner, cols), inner["made"].astype(int))
                with _threads(1):
                    oof[m] = h.predict_proba(_design(test, cols))[:, 1]
        full = train.copy()
        full[PRIOR_FEATURE] = oof_prior(full, INNER_FOLDS, seed)
        hgb = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS).fit(_design(full, cols), y)
        iso = None
        if k >= 2 and len(np.unique(y)) == 2 and np.isfinite(oof).all():
            iso = IsotonicRegression(y_min=ISO_CLIP, y_max=1 - ISO_CLIP, out_of_bounds="clip").fit(oof, y)
        pt = prior_table(train)

        def predict(frame, h=hgb, iso=iso, pt=pt, c=cols):
            f = frame.copy()
            f[PRIOR_FEATURE] = apply_prior(pt, f)
            with _threads(1):
                p = h.predict_proba(_design(f, c))[:, 1]
            return iso.predict(p) if iso is not None else p
        return predict
    raise KeyError("unknown candidate %r" % name)


# ----------------------------------------------------------------------------- out-of-fold

def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int, candidates=CANDIDATES) -> tuple:
    """Out-of-fold predictions of every candidate -> (frame with fold and p_<candidate> columns, protocol note)."""
    games = sorted(train["sportvu_game"].astype(str).unique())
    if len(games) < 2:
        raise SystemExit("%s: %d training game(s); at least 2 are needed for a grouped fold" % (WHO, len(games)))
    k = min(n_folds, len(games))
    pred = train.copy()
    for name in candidates:
        cv = grouped_cv(train, "sportvu_game", lambda d, n=name: fit_candidate(n, d, seed),
                        lambda m, d: m(d), n_folds=k, seed=seed)
        pred["fold"] = cv["fold"].to_numpy()
        pred["p_" + name] = cv["p"].to_numpy()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return pred, {"folds": k, "games": len(games), "text": text}


# ----------------------------------------------------------------------------- scoring

def _frame(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": pred["made"].astype(float).to_numpy(),
                         "p": pred[col].astype(float).to_numpy()})


def fast_slope(y, p, iters: int = 50, ridge: float = 1e-6) -> float:
    """Calibration slope by Newton's method on the 2-parameter logistic regression of y on logit(p) with the same
    negligible ridge as analytics.eval.calibration_slope (C = 1e6), at a fraction of its cost, for the bootstrap."""
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan")
    X = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    w = np.zeros(2)
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (q - y) + ridge * w
        H = (X * (q * (1 - q))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[1])


BOOT_METRICS = (("log_loss", log_loss), ("brier", brier), ("ece", ece), ("calibration_slope", fast_slope))


def score(pred: pd.DataFrame, n_boot: int, seed: int, candidates=CANDIDATES) -> dict:
    """{candidate: {metric: {point, lo, hi}}} with bootstrap intervals by game (the slope by fast_slope; the point
    estimate of the slope is analytics.eval.calibration_slope)."""
    out = {}
    for name in candidates:
        d = _frame(pred, "p_" + name)
        out[name] = {m: bootstrap_by_game(d, "game", lambda s, f=fn: f(s["y"].to_numpy(), s["p"].to_numpy()), n=n_boot, seed=seed)
                     for m, fn in BOOT_METRICS}
        out[name]["calibration_slope"]["point"] = calibration_slope(d["y"].to_numpy(), d["p"].to_numpy())
    return out


def paired(pred: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Paired bootstrap by game of log loss(a) - log loss(b); negative favours a."""
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": pred["made"].astype(float).to_numpy(),
                      "pa": pred["p_" + a].astype(float).to_numpy(), "pb": pred["p_" + b].astype(float).to_numpy()})
    return paired_bootstrap_by_game(d, "game", lambda s: log_loss(s["y"], s["pa"]), lambda s: log_loss(s["y"], s["pb"]),
                                    n=n_boot, seed=seed)


def select(scores: dict, candidates=CANDIDATES) -> str:
    """The candidate with the lowest out-of-fold log loss (ties to the earlier, simpler one)."""
    return min(candidates, key=lambda n: (round(scores[n]["log_loss"]["point"], 6), candidates.index(n)))


def calibration_block(name: str, y, p, bins: int = 10) -> str:
    """The calibration table of one candidate as markdown (the metric intervals are in the candidate table)."""
    L = ["### %s" % name, "", "| bin | lo | hi | n | mean_p | mean_y |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in calibration_table(np.asarray(y, float), np.asarray(p, float), bins).iterrows():
        mp = "%.4f" % r["mean_p"] if r["n"] else "-"
        my = "%.4f" % r["mean_y"] if r["n"] else "-"
        L.append("| %d | %.2f | %.2f | %d | %s | %s |" % (r["bin"], r["lo"], r["hi"], r["n"], mp, my))
    return "\n".join(L) + "\n"


def point_metrics(y, p) -> dict:
    return {m: float(fn(y, p)) for m, fn in METRICS}


# ----------------------------------------------------------------------------- predictions table

def predictions_table(pred: pd.DataFrame, heldout: pd.DataFrame, selected: str, heldout_p: dict) -> pd.DataFrame:
    rows = pd.DataFrame({"sportvu_game": pred["sportvu_game"].astype(str), "event": pred["event"].astype("int64"),
                         "player_id": pd.to_numeric(pred["player_id"], errors="coerce").astype("Int64"),
                         "p_location": pred["p_location_logistic"].astype("float32"),
                         "p_contest": pred["p_contest_logistic"].astype("float32"),
                         "p_selected": pred["p_" + selected].astype("float32"), "made": pred["made"].astype(bool),
                         "fold": pred["fold"].astype("int16"), "heldout": False})
    if len(heldout):
        h = pd.DataFrame({"sportvu_game": heldout["sportvu_game"].astype(str), "event": heldout["event"].astype("int64"),
                          "player_id": pd.to_numeric(heldout["player_id"], errors="coerce").astype("Int64"),
                          "p_location": heldout_p["location_logistic"].astype("float32"),
                          "p_contest": heldout_p["contest_logistic"].astype("float32"),
                          "p_selected": heldout_p[selected].astype("float32"), "made": heldout["made"].astype(bool),
                          "fold": np.int16(-1), "heldout": True})
        rows = pd.concat([rows, h], ignore_index=True)
    rows["heldout"] = rows["heldout"].astype(bool)
    return rows[PREDICTION_COLUMNS]


# ----------------------------------------------------------------------------- report

def _f(v, nd=4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d: dict) -> str:
    if not np.isfinite(d["point"]):
        return "n/a"
    return "%s [%s, %s]" % (_f(d["point"]), _f(d["lo"]), _f(d["hi"]))


ASSUMPTIONS = [
    "The unit is a row of the shots table; shot_frames rows without a shots row are not attempts here.",
    "shots rows are joined to shot_frames by (sportvu_game, event); a shots table without an event column (basketball-reference "
    "rows) is matched by (sportvu_game, period) and nearest clock_s within 1 s, each shot_frames row used once.",
    "Frame rate 25 Hz: 0.4 s is 10 frames, the touch cap of 10 s is 250 frames; speeds use frame counts, not the clock.",
    "An attempt with release_frame -1, confidence_flag unmatched, no shot_frames row or a shooter absent from the release frame "
    "keeps its location features and gets NaN contest features with has_contest 0; the logistic candidate imputes the training median, "
    "the boosting candidate takes NaN natively.",
    "The shooter prior is the smoothed (m = %d) make rate by (player_id, zone) from the fitting rows only, out-of-fold by game for the "
    "rows being fitted; zones are rim (under 4 ft), paint (under 14 ft), mid, corner3, above3." % int(PRIOR_M),
    "Selection is by out-of-fold log loss alone; ECE under %.2f is a target reported, not a selection rule." % ECE_TARGET,
    "The isotonic map of the boosting candidate is bounded to [%.2f, %.2f] so a sparse calibration set cannot emit exact 0 or 1." % (ISO_CLIP, 1 - ISO_CLIP),
    "Held-out rows get a prediction from every refit candidate (each scored once) so the output table is complete; only the "
    "selected candidate's held-out score is reported.",
    "Bootstrap intervals are 2.5 and 97.5 percentiles over whole-game resamples; with one held-out game no interval is meaningful.",
]


def write_report(path: Path, ctx: dict) -> str:
    fx = ctx["fixture"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# xFG upgrade: shot quality with contest features (analytics/xfg)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture output, not a result.** Every number below comes from the synthetic season of tests/_fx_xfg.py "
              "(no real tracking, play-by-play or shot data). The real run is the command in the last section.", ""]
    L += ["## Inputs and rows", "", "| table | rows in | rows out | dropped missing | dropped range | dropped duplicate | schema problems |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in ctx["summaries"]:
        L.append("| %s | %d | %d | %d | %d | %d | %d |" % (s["table"], s["rows_in"], s["rows_out"], s["dropped_missing"],
                                                         s["dropped_range"], s["dropped_duplicate"], len(s["schema_problems"])))
    L += ["", "Training rows: %d attempts from %d game(s), %d with contest features (has_contest 1), make rate %s."
          % (ctx["train_rows"], ctx["train_games"], ctx["train_contest"], _f(ctx["make_rate"])),
          "Held-out rows set aside before any fit: %d attempts from %d game(s) (%s)."
          % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids against sportvu/splits.json, and runs again on every fit; "
          "`schema.assert_time_forward` passed on the feature frame; `assert_no_leak` on every design matrix." % ctx["train_games"], "",
          "## Protocol", "",
          "Features: %s; the boosting candidate adds `%s`." % (", ".join(FEATURES), PRIOR_FEATURE),
          "Candidates: location_logistic (analytics.train_xfg, four location features), contest_logistic (median imputation, "
          "standardisation, C = 1), hgb_contest_prior (HistGradientBoostingClassifier %s, isotonic calibration on inner out-of-fold "
          "predictions, %d inner folds, output bounded to [%.2f, %.2f])." % (HGB_PARAMS, INNER_FOLDS, ISO_CLIP, 1 - ISO_CLIP),
          "Metrics: %s; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d; the slope "
          "resamples use the Newton solve `fast_slope`, the point estimate `analytics.eval.calibration_slope`). "
          "Selection: lowest out-of-fold log loss. ECE target: under %.2f." % (ctx["protocol"], ctx["bootstrap"], ctx["seed"], ECE_TARGET),
          "", "## Candidate table (out-of-fold, %d attempts, %d games)%s" % (ctx["train_rows"], ctx["train_games"], tag), "",
          "| candidate | log loss | Brier | ECE (10 bins) | ECE under %.2f | calibration slope | selected |" % ECE_TARGET,
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in CANDIDATES:
        s = ctx["scores"][name]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (name, _ci(s["log_loss"]), _ci(s["brier"]), _ci(s["ece"]),
                                                          "yes" if s["ece"]["point"] < ECE_TARGET else "no",
                                                          _ci(s["calibration_slope"]), "yes" if name == ctx["selected"] else ""))
    L += ["", "Selected: **%s** (lowest out-of-fold log loss)." % ctx["selected"], "",
          "Paired bootstrap by game, log loss difference (negative favours the first):", "",
          "| pair | diff | lo | hi | interval excludes 0 |", "| --- | --- | --- | --- | --- |"]
    for pair, r in ctx["paired"].items():
        L.append("| %s | %s | %s | %s | %s |" % (pair, _f(r["diff"]), _f(r["lo"]), _f(r["hi"]), "yes" if r["excludes_zero"] else "no"))
    L += ["", "## Calibration%s" % tag, ""]
    for name in CANDIDATES:
        L.append(ctx["blocks"][name])
    L += ["## Held-out score (scored once by the selected candidate refit on all %d training rows)%s" % (ctx["train_rows"], tag), ""]
    if ctx["heldout_rows"]:
        L += ["%d attempts from %d held-out game(s): %s. Never fitted or tuned on." % (ctx["heldout_rows"], ctx["heldout_games"],
                                                                                         ", ".join(ctx["heldout_game_ids"])), "",
              "| candidate | log loss | Brier | ECE (10 bins) | calibration slope |", "| --- | --- | --- | --- | --- |"]
        hs = ctx["heldout_scores"]
        L.append("| %s | %s | %s | %s | %s |" % (ctx["selected"], _f(hs["log_loss"]), _f(hs["brier"]), _f(hs["ece"]), _f(hs["calibration_slope"])))
        L += ["", "Point estimates%s." % ("; no interval over a single held-out game" if ctx["heldout_games"] < 2 else ""), "",
              ctx["heldout_block"]]
    else:
        L += ["No held-out rows in the input.", ""]
    L += ["## Output", "", "xfg_predictions (docs/contract/xfg.md): %s" % (ctx["predictions_path"] or "not written (no --out)"), "",
          "## Assumptions", ""]
    L += ["- " + a for a in ASSUMPTIONS]
    L += ["", "## The real run on the Mac", "",
          "With DATA_ROOT holding sportvu_contract/, collect/bref_shots/ and analytics/{shot_frames,rosters,possessions}/ "
          "(docs/contract/xfg.md):", "", "    " + FULL_RUN_CMD, "",
          "Validate and build the features first with `python -m analytics.xfg.validate --data \"$DATA_ROOT\"` and "
          "`python -m analytics.xfg.features --data \"$DATA_ROOT\" --out \"$DATA_ROOT\"`.", ""]
    if fx:
        L += ["fixture output, not a result", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true", help="run on the synthetic season (fixture output, not a result)")
    src.add_argument("--data", metavar="DATA_ROOT")
    ap.add_argument("--report", default="reports/xfg.md")
    ap.add_argument("--out", metavar="OUT_ROOT", default=None, help="where analytics/xfg/<season>/xfg_predictions.parquet goes")
    ap.add_argument("--season", default=V.SEASON)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--fixture-games", type=int, default=8)
    ap.add_argument("--fixture-possessions", type=int, default=40)
    return ap.parse_args(argv)


def run(tables: dict, summaries: list, a, command: str) -> dict:
    train, heldout = assemble(tables)
    if len(train) == 0:
        raise SystemExit("%s: no training rows after setting the held-out rows aside" % WHO)
    with _threads(1):
        pred, note = out_of_fold(train, a.folds, a.seed)
        scores = score(pred, a.bootstrap, a.seed)
        selected = select(scores)
        pairs = {"hgb_contest_prior vs location_logistic": paired(pred, "hgb_contest_prior", "location_logistic", a.bootstrap, a.seed),
                 "contest_logistic vs location_logistic": paired(pred, "contest_logistic", "location_logistic", a.bootstrap, a.seed)}
        blocks = {selected: report_block("%s (out-of-fold, selected)" % selected, pred["made"].astype(float),
                                         pred["p_" + selected].astype(float), pred, "sportvu_game", n=a.bootstrap, seed=a.seed)}
        for n in CANDIDATES:
            if n != selected:
                blocks[n] = calibration_block("%s (out-of-fold)" % n, pred["made"].astype(float), pred["p_" + n].astype(float))
        heldout_p, heldout_scores, heldout_block = {}, {}, ""
        if len(heldout):
            for n in CANDIDATES:
                heldout_p[n] = np.asarray(fit_candidate(n, train, a.seed)(heldout), dtype=float)
            y = heldout["made"].astype(float).to_numpy()
            heldout_scores = point_metrics(y, heldout_p[selected])
            heldout_block = report_block("%s (held-out, scored once)" % selected, y, heldout_p[selected], heldout, "sportvu_game",
                                         n=a.bootstrap if heldout["sportvu_game"].nunique() > 1 else 1, seed=a.seed)
    table = predictions_table(pred, heldout, selected, heldout_p)
    out_root = a.out or (None if a.fixtures else a.data)
    path = None
    if out_root:
        d = Path(out_root) / "analytics" / "xfg" / a.season
        d.mkdir(parents=True, exist_ok=True)
        path = d / "xfg_predictions.parquet"
        table.to_parquet(path, index=False)
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command, "fixture": bool(a.fixtures),
           "summaries": summaries, "train_rows": int(len(train)), "train_games": int(train["sportvu_game"].nunique()),
           "train_contest": int(train["has_contest"].sum()), "make_rate": float(train["made"].mean()),
           "heldout_rows": int(len(heldout)), "heldout_games": int(heldout["sportvu_game"].nunique()),
           "heldout_game_ids": sorted(heldout["sportvu_game"].astype(str).unique()), "protocol": note["text"], "folds": note["folds"],
           "bootstrap": a.bootstrap, "seed": a.seed, "scores": scores, "selected": selected, "paired": pairs, "blocks": blocks,
           "heldout_scores": heldout_scores, "heldout_block": heldout_block, "predictions": table,
           "predictions_path": str(path) if path else None,
           "calibration": calibration_table(pred["made"].astype(float).to_numpy(), pred["p_" + selected].astype(float).to_numpy())}
    return ctx


def main(argv=None) -> dict:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)
    tables = V.fixture_tables(a.seed, n_games=a.fixture_games, n_possessions=a.fixture_possessions) if a.fixtures \
        else V.load_data_root(a.data, a.season)
    tables, summaries = V.clean(tables)
    command = "python -m analytics.xfg.train " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = run(tables, summaries, a, command)
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
