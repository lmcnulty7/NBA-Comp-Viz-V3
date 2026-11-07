"""analytics/train_xfg.py: the location-only xFG baseline under the full evaluation protocol.

    python -m analytics.train_xfg --shots PATH [PATH ...] --report reports/xfg_baseline.md [--folds 5] [--seed 0] [--bootstrap 200]

Steps: load and concatenate the shots tables (docs/DATA_CONTRACT.md sections 10 and 10a, parquet or csv); set every
row whose heldout column is true, or whose sportvu_game is a held-out game of sportvu/splits.json, aside; call
sportvu.splits.refuse_heldout on the remaining game ids; build the location features (analytics.features_xfg) and
refuse a design matrix with a forbidden column; fit two models, the constant (league average make rate of the
training folds) and a logistic regression on the four location features after standardisation, under grouped
folds by sportvu_game; report out-of-fold log loss, Brier, ECE and calibration slope with bootstrap intervals by
game, the calibration table of the logistic model, and, when the input carried held-out rows, those rows scored once
by the logistic (selected) model refit on every training row (labelled held-out, never used for fitting or tuning;
the constant model is not scored on them). Under --smoke the held-out rows are set aside and never scored: the
held-out game is reserved for the one final scoring on the Mac.

Fewer games than folds: n_folds becomes the number of games (leave-one-game-out) and the report says so. One game:
no out-of-fold fit is possible, so the report carries in-sample metrics, labelled in-sample.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.features_xfg import assert_no_leak, feature_columns, features

from analytics.eval import (bootstrap_by_game, brier, calibration_slope, calibration_table, ece, grouped_folds,
                            log_loss, refuse_heldout)

WHO = "train_xfg"
MODELS = ("constant", "logistic")
METRICS = (("log_loss", log_loss), ("brier", brier), ("ece", ece), ("calibration_slope", calibration_slope))
FULL_RUN_CMD = ("python -m analytics.train_xfg --shots collect/shots/shotchartdetail/2015-16.parquet "
                "collect/shots/heldout/shotchartdetail/2015-16.parquet --report reports/xfg_baseline.md "
                "--folds 5 --seed 0 --bootstrap 1000")


# ── inputs ─────────────────────────────────────────────────────────────────────────────────────────
def load_shots(paths) -> tuple[pd.DataFrame, list[dict]]:
    """Concatenate the shots tables -> (frame with a source_file column, per-file summary)."""
    frames, summary = [], []
    for p in paths:
        p = Path(p)
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
        df = df.copy()
        df["source_file"] = p.name
        frames.append(df)
        h = df["heldout"].fillna(False).astype(bool) if "heldout" in df.columns else pd.Series(False, index=df.index)
        summary.append({"file": p.name, "rows": int(len(df)), "games": int(df["sportvu_game"].nunique()),
                        "heldout_rows": int(h.sum()), "heldout_games": int(df.loc[h, "sportvu_game"].nunique())})
    return pd.concat(frames, ignore_index=True), summary


def heldout_mask(feats: pd.DataFrame) -> pd.Series:
    """heldout column true, or a sportvu_game that sportvu/splits.json marks held out (by id or by the CLE arena)."""
    from sportvu import splits
    toks = splits.heldout_tokens()
    g = feats["sportvu_game"].astype(str)
    by_id = g.map(lambda s: any(t in s for t in toks) or s.endswith(".at." + splits.HELDOUT_ARENA))
    return feats["heldout"].astype(bool) | by_id.astype(bool)


def training_frame(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (train, heldout) feature frames. The guard runs on the training game ids; the held-out frame is scored,
    never fitted."""
    feats = features(shots)
    held = heldout_mask(feats)
    train, heldout = feats[~held].reset_index(drop=True), feats[held].reset_index(drop=True)
    refuse_heldout(sorted(train["sportvu_game"].unique()), WHO)
    assert train["heldout"].eq(False).all()
    return train, heldout


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def design(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[feature_columns()].astype(float)
    assert_no_leak(X)
    return X


def fit_models(train: pd.DataFrame) -> dict:
    """Fit the constant and the logistic model on one training frame -> {name: predict(frame) -> p}. Every fit of
    the trainer goes through here, so a test can watch what reaches a fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    X, y = design(train), train["made"].astype(int).to_numpy()
    rate = float(y.mean())
    out = {"constant": lambda frame, r=rate: np.full(len(frame), r)}
    if len(np.unique(y)) < 2:
        out["logistic"] = out["constant"]
    else:
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X.to_numpy(), y)
        out["logistic"] = lambda frame, m=lr: m.predict_proba(design(frame).to_numpy())[:, 1]
    return out


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions of both models -> (frame with p_constant, p_logistic, and the protocol note)."""
    games = sorted(train["sportvu_game"].unique())
    pred = train.copy()
    if len(games) == 1:
        models = fit_models(train)
        for m in MODELS:
            pred["p_" + m] = models[m](train)
        note = {"mode": "in-sample", "folds": 0, "games": 1,
                "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows)"}
        return pred, note
    k = min(n_folds, len(games))
    for m in MODELS:
        pred["p_" + m] = np.nan
    for tr_games, te_games in grouped_folds(games, n_folds=k, seed=seed):
        tr, te = train["sportvu_game"].isin(tr_games), train["sportvu_game"].isin(te_games)
        models = fit_models(train[tr])
        for m in MODELS:
            pred.loc[te, "p_" + m] = models[m](train[te])
    assert pred[["p_" + m for m in MODELS]].notna().all().all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────
def score(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{model: {metric: {point, lo, hi}}} with bootstrap intervals by game."""
    out = {}
    nan = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    for m in MODELS:
        df = pd.DataFrame({"game": pred["sportvu_game"].to_numpy(), "y": pred["made"].astype(float).to_numpy(),
                           "p": pred["p_" + m].astype(float).to_numpy()})
        out[m] = {}
        for name, fn in METRICS:
            if name == "calibration_slope" and m == "constant":
                out[m][name] = dict(nan)      # a constant prediction has no slope (its out-of-fold values differ only by fold rate)
                continue
            out[m][name] = bootstrap_by_game(df, "game", lambda d, f=fn: f(d["y"].to_numpy(), d["p"].to_numpy()), n=n_boot, seed=seed)
            if df["game"].nunique() < 2:          # an interval over one game is meaningless: report none
                out[m][name] = {**out[m][name], "lo": float("nan"), "hi": float("nan")}
    return out


def _f(v, nd=4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d: dict) -> str:
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return _f(d["point"])
    return "%s [%s, %s]" % (_f(d["point"]), _f(d["lo"]), _f(d["hi"]))


def write_report(path: Path, ctx: dict) -> str:
    L = ["# xFG baseline: location only (analytics/train_xfg.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), "",
         "## Inputs", "", "| file | rows | games | held-out rows | held-out games |", "| --- | --- | --- | --- | --- |"]
    for s in ctx["inputs"]:
        L.append("| %s | %d | %d | %d | %d |" % (s["file"], s["rows"], s["games"], s["heldout_rows"], s["heldout_games"]))
    L += ["", "Training rows: %d attempts from %d game(s) (%s). Held-out rows set aside before any fit: %d attempts from %d game(s) (%s)."
          % (ctx["train_rows"], ctx["train_games"], ", ".join(ctx["train_game_ids"]) or "none",
             ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `sportvu.splits.refuse_heldout` checked %d training game ids against sportvu/splits.json." % ctx["train_games"],
          "Training make rate: %s over %d attempts." % (_f(ctx["make_rate"]), ctx["train_rows"]), "",
          "## Protocol", "",
          "Features: %s (period and clock_s kept for buckets, not fed to the model; `assert_no_leak` passed on the design matrix)." % ", ".join(feature_columns()),
          "Models: constant (training-fold make rate) and logistic regression on the four location features with standardisation (C=1).",
          "Metrics: %s; intervals are the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s."
          % (ctx["protocol"], ctx["bootstrap"], ctx["seed"],
             "; no interval with a single game" if ctx["train_games"] < 2 else ""),
          "", "## Metrics (%s, %d attempts, %d game(s))" % (ctx["mode"], ctx["train_rows"], ctx["train_games"]), "",
          "| model | log loss | Brier | ECE (10 bins) | calibration slope |", "| --- | --- | --- | --- | --- |"]
    for m in MODELS:
        s = ctx["scores"][m]
        L.append("| %s | %s | %s | %s | %s |" % (m, _ci(s["log_loss"]), _ci(s["brier"]), _ci(s["ece"]), _ci(s["calibration_slope"])))
    L += ["", "The constant model's calibration slope is n/a by design: its predictions vary only by fold make rate, so a slope is not defined.", "",
          "## Calibration table, logistic model (%s)" % ctx["mode"], "",
          "| bin | lo | hi | n | mean p | mean made |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ctx["calibration"].itertuples():
        L.append("| %d | %.1f | %.1f | %d | %s | %s |" % (r.bin, r.lo, r.hi, r.n, _f(r.mean_p, 3), _f(r.mean_y, 3)))
    if ctx["heldout_rows"] and ctx["smoke"]:
        L += ["", "## Held-out rows", "",
              "%d attempts from %d held-out game(s) (%s) were set aside and NOT scored: a smoke run never touches them. The held-out "
              "game is reserved for the one final scoring on the Mac, by the real run below."
              % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]))]
    elif ctx["heldout_rows"]:
        L += ["", "## Held-out rows (scored once by the logistic model refit on all %d training rows; never fitted or tuned on)" % ctx["train_rows"], "",
              "%d attempts from %d held-out game(s): %s. Only the selected (logistic) model scores them; the constant model is not scored on held-out rows."
              % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"])),
              "", "| model | log loss | Brier | ECE (10 bins) | calibration slope |", "| --- | --- | --- | --- | --- |"]
        s = ctx["heldout_scores"]["logistic"]
        L.append("| %s | %s | %s | %s | %s |" % ("logistic", _f(s["log_loss"]), _f(s["brier"]), _f(s["ece"]), _f(s["calibration_slope"])))
        L.append("")
        L.append("Held-out numbers are point estimates (no bootstrap: %d game(s))." % ctx["heldout_games"])
    L += ["", "## Status", ""]
    if ctx["smoke"]:
        L.append("This is a smoke run on the saved basketball-reference fixture pages, not the baseline: the real run awaits the full "
                 "2015-16 shots parquet (104,967 attempts, 625 games, on the owner's Mac) with the exact command")
    else:
        L.append("The real run of this protocol is")
    L += ["", "    " + FULL_RUN_CMD, ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", nargs="+", required=True, help="shots tables (parquet or csv), sections 10 and 10a")
    ap.add_argument("--report", default="reports/xfg_baseline.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="label the report as a smoke run on fixture pages")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    shots, inputs = load_shots(a.shots)
    train, heldout = training_frame(shots)
    if len(train) == 0:
        raise SystemExit("%s: no training rows after setting the held-out rows aside" % WHO)
    pred, note = out_of_fold(train, a.folds, a.seed)
    scores = score(pred, a.bootstrap, a.seed)
    cal = calibration_table(pred["made"].astype(float).to_numpy(), pred["p_logistic"].astype(float).to_numpy())
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.train_xfg " + " ".join(argv if argv is not None else sys.argv[1:]),
           "inputs": inputs, "train_rows": int(len(train)), "train_games": int(train["sportvu_game"].nunique()),
           "train_game_ids": sorted(train["sportvu_game"].unique()), "heldout_rows": int(len(heldout)),
           "heldout_games": int(heldout["sportvu_game"].nunique()), "heldout_game_ids": sorted(heldout["sportvu_game"].unique()),
           "make_rate": float(train["made"].mean()), "protocol": note["text"], "mode": note["mode"], "folds": note["folds"],
           "bootstrap": a.bootstrap, "seed": a.seed, "scores": scores, "calibration": cal, "smoke": a.smoke, "heldout_scores": {}}
    if len(heldout) and not a.smoke:
        # the held-out rows are scored once, by the selected (logistic) model refit on every training row; the constant
        # model is not scored on them (its entry is kept with n/a so the table shape is stable). A smoke run never
        # scores them: the held-out game is reserved for the one final scoring on the Mac.
        models = fit_models(train)
        y = heldout["made"].astype(float).to_numpy()
        p = models["logistic"](heldout)
        ctx["heldout_scores"]["constant"] = {name: float("nan") for name, _fn in METRICS}
        ctx["heldout_scores"]["logistic"] = {name: float(fn(y, p)) for name, fn in METRICS}
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
