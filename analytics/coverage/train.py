"""analytics/coverage/train.py: the coverage classifier under the evaluation protocol.

    python -m analytics.coverage.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/coverage.md
                                       [--folds 5] [--seed 0] [--bootstrap 200]

Steps: load the inputs (analytics.coverage.validate), run every check, build the feature frame
(analytics.coverage.features; schema.assert_time_forward runs on it), set every held-out row aside (held-out
game of sportvu/splits.json, the `.at.CLE` arena rule, or frames under heldout/), call
analytics.eval.refuse_heldout on the training game ids, then fit three candidates under grouped folds by
sportvu_game:

    majority    the training-fold class frequencies (predicts the most frequent class)
    logistic    multinomial logistic regression on the standardised features (median imputation, C = 1)
    hgb         HistGradientBoostingClassifier (100 trees, depth 3, learning rate 0.1, min 5 rows per leaf)

Selection: out-of-fold macro F1 (ties go to the simpler model); accuracy, per-class F1 and a confusion table
are reported for the selected model, with 95 percent bootstrap intervals by game. Held-out rows are scored
once by the selected model refit on every training row, never fitted or tuned on. Unlabelled screens of
training games (ball_screen actions without a screen label and the detector's screens) are scored the same way.

Outputs (docs/contract/coverage.md): OUT_ROOT/analytics/coverage/<season>/coverage_predictions.parquet (held-out
rows under heldout/<season>/) and OUT_ROOT/analytics/coverage/<season>/coverage_team_rates.parquet (shares of the
predicted class per defending team over training-game screens, insufficient under 300 screens via
analytics.eval.gated), plus the markdown report.

Fewer games than folds: n_folds becomes the number of training games (leave-one-game-out) and the report says
so. One training game: in-sample metrics, labelled as such. No deep model here: baselines first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.coverage import CLASSES, WHO
from analytics.coverage.features import design, feature_columns, features
from analytics.coverage.validate import add_input_args, load_inputs, season_of, validate_bundle
from analytics.eval import bootstrap_by_game, gated, grouped_folds, refuse_heldout

log = logging.getLogger("analytics.coverage")

MODELS = ("majority", "logistic", "hgb")
N_MIN_TEAM = 300
MAC_CMD = ('python -m analytics.coverage.train --data "<Drive>/nba_harvest" --out "<Drive>/nba_harvest" '
           "--report reports/coverage.md --folds 5 --seed 0 --bootstrap 1000")
PROB_COLUMNS = ["p_" + c for c in CLASSES]
PREDICTION_COLUMNS = ["sportvu_game", "possession", "action_id", "coverage_label"] + PROB_COLUMNS + ["predicted", "fold", "heldout"]
TEAM_RATE_COLUMNS = ["team_id", "n_screens"] + ["share_" + c for c in CLASSES] + ["insufficient"]


# ── training frame ────────────────────────────────────────────────────────────────────────────────
def training_frame(feats: pd.DataFrame) -> tuple:
    """-> (train, score_only, heldout): labelled rows of training games, unlabelled rows of training games, every
    held-out row. The guard runs on the training game ids."""
    held = feats["heldout"].astype(bool)
    lab = feats["coverage_label"].notna()
    train = feats[~held & lab].reset_index(drop=True)
    score_only = feats[~held & ~lab].reset_index(drop=True)
    heldout = feats[held].reset_index(drop=True)
    refuse_heldout(sorted(train["sportvu_game"].unique()), WHO)
    schema.assert_time_forward(train, "states_1hz")
    assert not train["heldout"].any()
    return train, score_only, heldout


# ── models ────────────────────────────────────────────────────────────────────────────────────────
def _align(model_classes, P: np.ndarray) -> np.ndarray:
    out = np.zeros((len(P), len(CLASSES)))
    for j, c in enumerate(model_classes):
        out[:, CLASSES.index(str(c))] = P[:, j]
    return out


def fit_models(train: pd.DataFrame, seed: int = 0) -> dict:
    """Fit the three candidates on one training frame -> {name: predict(frame) -> (n, 7) probabilities in CLASSES
    order}. Every fit goes through here, so a test can watch what reaches a fit."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    if train["coverage_label"].isna().any():
        raise ValueError("%s: an unlabelled row reached fit_models" % WHO)
    X, y = design(train), train["coverage_label"].astype(str).to_numpy()
    freq = np.array([(y == c).mean() for c in CLASSES])
    out = {"majority": lambda frame, f=freq: np.tile(f, (len(frame), 1))}
    if len(np.unique(y)) < 2:
        out["logistic"] = out["hgb"] = out["majority"]
        return out
    # one thread per fit: the data is small and the machine is shared, so thread fan-out only costs time
    with threadpool_limits(limits=1):
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=1.0, max_iter=2000)).fit(X.to_numpy(), y)
        hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, min_samples_leaf=5,
                                             l2_regularization=1.0, early_stopping=False, random_state=seed).fit(X.to_numpy(), y)

    def _predict(m):
        def fn(frame):
            with threadpool_limits(limits=1):
                return _align(m.classes_, m.predict_proba(design(frame).to_numpy()))
        return fn
    out["logistic"], out["hgb"] = _predict(lr), _predict(hgb)
    return out


def out_of_fold(train: pd.DataFrame, n_folds: int, seed: int) -> tuple:
    """-> (fold array, {model: (n, 7) out-of-fold probabilities}, protocol note)."""
    games = sorted(train["sportvu_game"].unique())
    probs = {m: np.full((len(train), len(CLASSES)), np.nan) for m in MODELS}
    fold = np.full(len(train), -1, dtype=int)
    if len(games) == 1:
        models = fit_models(train, seed)
        for m in MODELS:
            probs[m][:] = models[m](train)
        fold[:] = 0
        note = {"mode": "in-sample", "folds": 0, "games": 1,
                "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows)"}
        return fold, probs, note
    k = min(n_folds, len(games))
    for i, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = train["sportvu_game"].isin(tr_games).to_numpy(), train["sportvu_game"].isin(te_games).to_numpy()
        models = fit_models(train[tr], seed)
        for m in MODELS:
            probs[m][te] = models[m](train[te])
        fold[te] = i
    assert all(np.isfinite(p).all() for p in probs.values()) and (fold >= 0).all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return fold, probs, {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


# ── metrics ───────────────────────────────────────────────────────────────────────────────────────
def predicted(P: np.ndarray) -> np.ndarray:
    return np.array(CLASSES, dtype=object)[np.argmax(P, axis=1)]


def macro_f1(y, yhat) -> float:
    """Macro F1 over the classes present in y (sklearn, zero_division = 0)."""
    from sklearn.metrics import f1_score
    y, yhat = np.asarray(y, dtype=object), np.asarray(yhat, dtype=object)
    labels = sorted(set(y.tolist()))
    return float(f1_score(y, yhat, labels=labels, average="macro", zero_division=0))


def accuracy(y, yhat) -> float:
    return float(np.mean(np.asarray(y, dtype=object) == np.asarray(yhat, dtype=object)))


def per_class_f1(y, yhat) -> dict:
    from sklearn.metrics import f1_score
    y, yhat = np.asarray(y, dtype=object), np.asarray(yhat, dtype=object)
    present = set(y.tolist()) | set(yhat.tolist())
    f = f1_score(y, yhat, labels=list(CLASSES), average=None, zero_division=0)
    return {c: (float(v) if c in present else float("nan")) for c, v in zip(CLASSES, f)}


def confusion(y, yhat) -> pd.DataFrame:
    """Rows: true class; columns: predicted class; CLASSES order."""
    y, yhat = pd.Categorical(y, categories=CLASSES), pd.Categorical(yhat, categories=CLASSES)
    return pd.crosstab(y, yhat, dropna=False).reindex(index=list(CLASSES), columns=list(CLASSES), fill_value=0)


def score(train: pd.DataFrame, probs: dict, n_boot: int, seed: int) -> dict:
    """{model: {macro_f1: {point, lo, hi}, accuracy: {...}, per_class: {...}, confusion: DataFrame}}."""
    y = train["coverage_label"].astype(str).to_numpy()
    games = train["sportvu_game"].astype(str).to_numpy()
    out = {}
    for m in MODELS:
        yhat = predicted(probs[m])
        d = pd.DataFrame({"game": games, "y": y, "yhat": yhat})
        res = {}
        for name, fn in (("macro_f1", macro_f1), ("accuracy", accuracy)):
            r = bootstrap_by_game(d, "game", lambda s, f=fn: f(s["y"].to_numpy(), s["yhat"].to_numpy()), n=n_boot, seed=seed)
            if d["game"].nunique() < 2:
                r = {**r, "lo": float("nan"), "hi": float("nan")}
            res[name] = r
        res["per_class"] = per_class_f1(y, yhat)
        res["confusion"] = confusion(y, yhat)
        out[m] = res
    return out


def select(scores: dict) -> str:
    """The model with the highest out-of-fold macro F1; ties go to the earlier (simpler) entry of MODELS."""
    best = MODELS[0]
    for m in MODELS[1:]:
        if scores[m]["macro_f1"]["point"] > scores[best]["macro_f1"]["point"] + 1e-12:
            best = m
    return best


# ── output tables ─────────────────────────────────────────────────────────────────────────────────
def prediction_rows(frame: pd.DataFrame, P: np.ndarray, fold) -> pd.DataFrame:
    out = pd.DataFrame({"sportvu_game": frame["sportvu_game"].astype(str).to_numpy(),
                        "possession": frame["possession"].astype("int32").to_numpy(),
                        "action_id": frame["action_id"].astype("int32").to_numpy(),
                        "coverage_label": frame["coverage_label"].astype("string").to_numpy()})
    for j, c in enumerate(CLASSES):
        out["p_" + c] = P[:, j].astype("float32")
    out["predicted"] = pd.array(predicted(P), dtype="string")
    out["fold"] = np.asarray(np.broadcast_to(fold, len(frame)), dtype="int8")
    out["heldout"] = frame["heldout"].astype(bool).to_numpy()
    return out[PREDICTION_COLUMNS]


def defender_team(feats: pd.DataFrame, games: dict) -> pd.Series:
    """team_id of the on-ball defender from the rosters table, else from the frames; NaN when unknown."""
    team = {}
    for gid, g in games.items():
        src = g.get("rosters")
        if src is None or not len(src):
            src = g["frames"]
        for pid, tid in src.drop_duplicates("player_id")[["player_id", "team_id"]].itertuples(index=False):
            team[(gid, int(pid))] = int(tid)
    keys = list(zip(feats["sportvu_game"].astype(str), feats["on_ball_defender_id"].astype(int)))
    return pd.Series([team.get(k, np.nan) for k in keys], index=feats.index, dtype=float)


def team_rates(pred: pd.DataFrame, team: pd.Series, n_min: int = N_MIN_TEAM) -> pd.DataFrame:
    """Shares of the predicted class per defending team over training-game screens; insufficient under n_min
    screens (analytics.eval.gated). Rows without a team are dropped."""
    d = pred[~pred["heldout"]].copy()
    d["team_id"] = team.loc[d.index].to_numpy() if len(d) else []
    d = d[d["team_id"].notna()]
    if not len(d):
        return pd.DataFrame(columns=TEAM_RATE_COLUMNS)
    d["team_id"] = d["team_id"].astype(int)
    g = gated(d, "team_id", n_min)
    rows = []
    for tid, part in g.groupby("team_id"):
        row = {"team_id": int(tid), "n_screens": int(len(part))}
        for c in CLASSES:
            row["share_" + c] = float((part["predicted"] == c).mean())
        row["insufficient"] = bool(part["insufficient"].any())
        rows.append(row)
    return pd.DataFrame(rows, columns=TEAM_RATE_COLUMNS)


def write_tables(pred: pd.DataFrame, rates: pd.DataFrame, out_root) -> list:
    paths = []
    for (season, held), part in pred.groupby([pred["sportvu_game"].map(season_of), pred["heldout"]]):
        d = Path(out_root) / "analytics" / "coverage" / ("heldout" if held else "") / season
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / "coverage_predictions.parquet", index=False)
        paths.append(d / "coverage_predictions.parquet")
    seasons = sorted(set(pred.loc[~pred["heldout"], "sportvu_game"].map(season_of))) or ["unknown"]
    for season in seasons:
        d = Path(out_root) / "analytics" / "coverage" / season
        d.mkdir(parents=True, exist_ok=True)
        rates.to_parquet(d / "coverage_team_rates.parquet", index=False)
        paths.append(d / "coverage_team_rates.parquet")
    return paths


# ── report ────────────────────────────────────────────────────────────────────────────────────────
def _f(v, nd=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d: dict) -> str:
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return _f(d["point"])
    return "%s [%s, %s]" % (_f(d["point"]), _f(d["lo"]), _f(d["hi"]))


def write_report(path: Path, ctx: dict) -> str:
    c, s = ctx["counts"], ctx["scores"]
    best = ctx["best"]
    L = ["# Coverage recognition: pick-and-roll defensive coverage (analytics/coverage)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), "", "## Status", ""]
    if ctx["fixtures"]:
        L.append("No hand labels exist yet (STATUS.md): this run read the synthetic fixture season (%s generator), so every "
                 "number below is fixture output, not a result. The real run awaits the hand-label tables of lane ACTIONS "
                 "under DATA_ROOT/labels/ on the owner's Mac." % ctx["fixture_kind"])
    else:
        L.append("This run read `%s`. Whether its numbers are a result depends on the label tables it found (counts below); "
                 "with no hand labels under DATA_ROOT/labels/screens/ nothing here is a result." % ctx["data"])
    L += ["", "## Inputs", "",
          "| item | count |", "| --- | --- |",
          "| games with frames | %d (held out: %d) |" % (c["games"], c["games_heldout"]),
          "| games with labels but no frames | %d |" % c["games_without_frames"],
          "| action rows read / kept | %d / %d |" % (c["actions_in"], c["actions_kept"]),
          "| ball_screen actions | %d |" % c["ball_screen_actions"],
          "| screen rows read / kept | %d / %d |" % (c["screens_in"], c["screens_kept"]),
          "| screens dropped: missing value, non-integer, unknown coverage, unknown confidence, duplicate id, no ball_screen action, bad defender ids | %d, %d, %d, %d, %d, %d, %d |" % (
              c["screens_missing_value"], c["screens_non_integer"], c["screens_unknown_coverage"], c["screens_unknown_confidence"],
              c["screens_duplicate_id"], c["screens_without_ball_screen_action"], c["screens_bad_defender_ids"]),
          "| ball_screen actions without a screen label (scored, never fitted) | %d |" % c["ball_screen_actions_without_label"],
          "| detected screens (label_source model; scored, never fitted) | %d |" % c.get("detected_screens", 0),
          "| screens dropped at feature time: no frames, roles untracked, defenders not inferable | %d, %d, %d |" % (
              c["features_dropped_no_frames"], c["features_dropped_roles_untracked"], c["features_dropped_defenders_not_inferable"]),
          "| training rows (labelled, training games) | %d from %d game(s) |" % (ctx["train_rows"], ctx["train_games"]),
          "| held-out rows set aside before any fit | %d from %d game(s) |" % (ctx["heldout_rows"], ctx["heldout_games"]),
          "", "Guard: `analytics.eval.refuse_heldout` checked %d training game ids against sportvu/splits.json; "
          "`analytics.schema.assert_time_forward` passed on the feature frame." % ctx["train_games"],
          "", "## Label counts (training rows)", "", "| class | n | share |", "| --- | --- | --- |"]
    for k in CLASSES:
        n = ctx["label_counts"].get(k, 0)
        L.append("| %s | %d | %s |" % (k, n, _f(n / ctx["train_rows"] if ctx["train_rows"] else float("nan"))))
    L += ["", "Confidence: %d sure, %d unsure (both are fitted; the flag never enters a feature)." % (ctx["n_sure"], ctx["n_unsure"]),
          "", "## Protocol", "",
          "Unit: one ball screen. Window: start_frame to end_frame plus 1.5 s (38 frames at 25 Hz), declared as the object "
          "being classified; no later frame enters. Features (%d): %s." % (len(feature_columns()), ", ".join(feature_columns())),
          "Candidates: majority (training-fold class frequencies), multinomial logistic regression (median imputation, "
          "standardisation, C = 1), HistGradientBoostingClassifier (100 trees, depth 3, learning rate 0.1, min 5 rows per leaf).",
          "Metrics: %s; macro F1 over the classes present in the truth; intervals are the 2.5 and 97.5 percentiles of %d "
          "bootstrap resamples of whole games (seed %d)%s. Selection: highest out-of-fold macro F1, ties to the simpler model." % (
              ctx["protocol"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if ctx["train_games"] < 2 else ""),
          "", "## Candidates (%s, %d rows, %d game(s))" % (ctx["mode"], ctx["train_rows"], ctx["train_games"]), "",
          "| model | macro F1 | accuracy | selected |", "| --- | --- | --- | --- |"]
    for m in MODELS:
        L.append("| %s | %s | %s | %s |" % (m, _ci(s[m]["macro_f1"]), _ci(s[m]["accuracy"]), "yes" if m == best else ""))
    L += ["", "Selected: %s. Fixture target: beat majority (%s)." % (
        best, "met" if s[best]["macro_f1"]["point"] > s["majority"]["macro_f1"]["point"] else "NOT met"),
          "", "## Per-class F1 (%s, %s)" % (best, ctx["mode"]), "", "| class | F1 |", "| --- | --- |"]
    for k in CLASSES:
        L.append("| %s | %s |" % (k, _f(s[best]["per_class"][k])))
    cm = s[best]["confusion"]
    L += ["", "## Confusion (%s, %s): rows true, columns predicted" % (best, ctx["mode"]), "",
          "| true \\ predicted | " + " | ".join(CLASSES) + " |", "| --- |" + " --- |" * len(CLASSES)]
    for k in CLASSES:
        L.append("| %s | " % k + " | ".join(str(int(cm.loc[k, j])) for j in CLASSES) + " |")
    L += ["", "## Team rates (predicted class shares per defending team, training games; insufficient under %d screens)" % N_MIN_TEAM, ""]
    if len(ctx["rates"]):
        L += ["| team_id | n_screens | " + " | ".join("share_" + k for k in CLASSES) + " | insufficient |",
              "| --- | --- |" + " --- |" * len(CLASSES) + " --- |"]
        for r in ctx["rates"].itertuples(index=False):
            L.append("| %d | %d | " % (r.team_id, r.n_screens) + " | ".join(_f(getattr(r, "share_" + k)) for k in CLASSES)
                     + " | %s |" % ("yes" if r.insufficient else "no"))
        L.append("")
        L.append("A team marked insufficient has fewer than %d screens; its shares are shown for completeness and are not to be read." % N_MIN_TEAM)
    else:
        L.append("No team rates: no training-game screen with a known defending team.")
    L += ["", "## Held-out rows (scored once by %s refit on all %d training rows; never fitted or tuned on)" % (best, ctx["train_rows"]), ""]
    if ctx["heldout_rows"]:
        h = ctx["heldout_scores"]
        L += ["%d screens from %d held-out game(s): %s." % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"])),
              "", "| model | macro F1 | accuracy | labelled rows |", "| --- | --- | --- | --- |"]
        for m in MODELS:
            L.append("| %s | %s | %s | %d |" % (m, _f(h[m]["macro_f1"]), _f(h[m]["accuracy"]), h["n_labelled"]))
        L.append("")
        L.append("Held-out numbers are point estimates over %d game(s), no interval." % ctx["heldout_games"])
    else:
        L.append("No held-out rows in this run.")
    L += ["", "## Outputs", ""] + ["- `%s`" % p for p in ctx["paths"]] + [
        "", "## On the Mac (where the labels and frames will be)", "", "    " + MAC_CMD, "",
        "## Assumptions", "",
        "- action_id is the row's position within its game's actions file when the table carries no action_id column.",
        "- The basket a screen attacks is the one nearer the handler at start_frame.",
        "- Detected screens (no defender ids) take the opponent nearest the handler as on-ball defender and the opponent "
        "nearest the screener, other than that one, as the screener's defender.",
        "- A held-out game is one with a sportvu/splits.json token, the `.at.CLE` arena rule, or frames under heldout/.",
        "- Team rates use the predicted class of every training-game screen (out-of-fold for labelled rows) so labelled and "
        "unlabelled screens are counted the same way; the defending team is the on-ball defender's team.",
        "- The feature frame is checked with `schema.assert_time_forward(frame, \"states_1hz\")`: the screen table is not one "
        "of the five contract tables, so the nearest per-frame table name is passed; the check is against the union of "
        "every table's excluded columns whatever the name.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ──────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_input_args(ap)
    ap.add_argument("--report", default="reports/coverage.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    return ap.parse_args(argv)


def run(bundle: dict, n_folds: int, seed: int, n_boot: int, out_root, report: Path, command: str, fixtures: bool,
        data: str = "") -> dict:
    cand, counts = validate_bundle(bundle)
    feats = features(cand, bundle["games"], counts)
    train, score_only, heldout = training_frame(feats)
    if len(train) == 0:
        raise SystemExit("%s: no labelled training rows after setting the held-out rows aside" % WHO)
    fold, probs, note = out_of_fold(train, n_folds, seed)
    scores = score(train, probs, n_boot, seed)
    best = select(scores)
    full = fit_models(train, seed)
    pred = [prediction_rows(train, probs[best], fold)]
    if len(score_only):
        pred.append(prediction_rows(score_only, full[best](score_only), -1))
    heldout_scores = {}
    if len(heldout):
        P = full[best](heldout)
        pred.append(prediction_rows(heldout, P, -1))
        lab = heldout["coverage_label"].notna().to_numpy()
        y = heldout.loc[lab, "coverage_label"].astype(str).to_numpy()
        for m in MODELS:
            yh = predicted(full[m](heldout))[lab]
            heldout_scores[m] = {"macro_f1": macro_f1(y, yh) if lab.any() else float("nan"),
                                 "accuracy": accuracy(y, yh) if lab.any() else float("nan")}
        heldout_scores["n_labelled"] = int(lab.sum())
    pred_all = pd.concat(pred, ignore_index=True)
    feats_all = pd.concat([train, score_only, heldout], ignore_index=True)
    rates = team_rates(pred_all, defender_team(feats_all, bundle["games"]))
    paths = write_tables(pred_all, rates, out_root)
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command,
           "fixtures": fixtures, "fixture_kind": bundle.get("fixture", "local"), "data": data, "counts": counts,
           "train_rows": int(len(train)), "train_games": int(train["sportvu_game"].nunique()),
           "heldout_rows": int(len(heldout)), "heldout_games": int(heldout["sportvu_game"].nunique()),
           "heldout_game_ids": sorted(heldout["sportvu_game"].unique()),
           "label_counts": train["coverage_label"].value_counts().to_dict(),
           "n_sure": int((cand["confidence"] == "sure").sum()), "n_unsure": int((cand["confidence"] == "unsure").sum()),
           "protocol": note["text"], "mode": note["mode"], "folds": note["folds"], "bootstrap": n_boot, "seed": seed,
           "scores": scores, "best": best, "rates": rates, "heldout_scores": heldout_scores,
           "paths": [str(p) for p in paths] + [str(report)], "predictions": pred_all, "features": feats_all}
    ctx["report_text"] = write_report(Path(report), ctx)
    return ctx


def main(argv=None) -> dict:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stderr)
    bundle = load_inputs(a)
    out_root = a.out or tempfile.mkdtemp(prefix="coverage_out_")
    cmd = "python -m analytics.coverage.train " + " ".join(argv if argv is not None else sys.argv[1:])
    ctx = run(bundle, a.folds, a.seed, a.bootstrap, out_root, Path(a.report), cmd, a.fixtures, a.data or "")
    print(ctx["report_text"])
    return ctx


if __name__ == "__main__":
    main()
