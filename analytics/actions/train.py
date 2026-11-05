"""analytics/actions/train.py: the action detector (ball screens first) under the full evaluation protocol.

    python -m analytics.actions.train --fixtures | --data DATA_ROOT --report reports/actions.md
                                      [--out OUT_ROOT] [--folds 5] [--seed 0] [--bootstrap 200]

Steps: load the games (validate.load_inputs) and run the validation checks (a game with a schema problem is
skipped and listed); build the candidate windows and features (features.build_season); set every held-out
game aside (under heldout/, a held-out token of sportvu/splits.json, or the CLE arena); call
analytics.eval.refuse_heldout on the training game ids and analytics.schema.assert_time_forward plus
features.assert_no_leak on the training feature frame; fit two models under grouped folds by sportvu_game:
    rule   the candidate generator's own class (the baseline: p = 1 for the rule class)
    hgb    HistGradientBoostingClassifier, multi-class over {ball_screen, drive, isolation, none}, fitted on one
           thread (threadpoolctl): with 4 cores shared by many processes its OpenMP loops stall for minutes
Hand labels come through analytics.labels.read_labels(DATA_ROOT, game, labeller_pass=1): tombstoned ids are
dropped, the last line per id wins, and the pass-2 relabels never reach a fit.
Selection metric: out-of-fold F1 for ball_screen (plan target over 0.8 on a held-out game); the report also
carries per-class F1, macro F1, accuracy, the confusion table of the selected model, bootstrap intervals by
game, and candidate recall against the labels. Ties go to the rule baseline (simpler first). The held-out
games are scored once by the models refit on every training row and reported apart; they never fit or tune.

Outputs (docs/contract/actions.md), written under OUT_ROOT/analytics/actions/<season>/ when --out is given
(held-out games under heldout/<season>/): action_candidates/<game>.parquet (every candidate with its label
class, the selected model's probabilities, fold and heldout flag) and actions_detected/<game>.parquet plus
<game>.jsonl (the candidates whose selected class is not none with probability above 0.5, in the hand-label
schema with label_source model).

Fewer games than folds: n_folds becomes the number of games (leave-one-game-out) and the report says so. One
training game: no out-of-fold fit is possible, so the report carries in-sample metrics, labelled in-sample.
A fixture run is labelled "fixture output, not a result" everywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.actions import CLASSES, SEASON, WHO
from analytics.actions import features as F
from analytics.actions import validate as V
from analytics.eval import bootstrap_by_game, grouped_folds, refuse_heldout

log = logging.getLogger("analytics.actions.train")

MODELS = ("rule", "hgb")
DETECT_P = 0.5
FULL_RUN_CMD = ("python -m analytics.actions.train --data \"$DATA_ROOT\" --report reports/actions.md "
                "--out \"$DATA_ROOT\" --folds 5 --seed 0 --bootstrap 1000")
HGB_PARAMS = dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=0.1)


# ── inputs ─────────────────────────────────────────────────────────────────────────────────────────
def is_heldout(game: dict) -> bool:
    return bool(game.get("heldout", False)) or V.is_heldout_id(game["sportvu_game"])


def training_split(cands: pd.DataFrame, feats: pd.DataFrame) -> tuple:
    """-> (train candidates, train features, held-out candidates, held-out features). The guard runs on the
    training game ids; the training feature frame passes assert_time_forward and assert_no_leak."""
    held = cands["heldout"].astype(bool) | cands["sportvu_game"].map(V.is_heldout_id).astype(bool)
    train_c, held_c = cands[~held].reset_index(drop=True), cands[held].reset_index(drop=True)
    train_f, held_f = feats[~held.to_numpy()].reset_index(drop=True), feats[held.to_numpy()].reset_index(drop=True)
    refuse_heldout(sorted(train_c["sportvu_game"].unique()), WHO)
    schema.assert_time_forward(train_f, "possessions")
    F.assert_no_leak(train_f.drop(columns=F.ID_COLUMNS))
    return train_c, train_f, held_c, held_f


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def design(feats: pd.DataFrame) -> np.ndarray:
    X = feats[F.feature_columns()]
    F.assert_no_leak(X)
    return X.to_numpy(dtype=float)


def one_hot(classes: pd.Series) -> np.ndarray:
    idx = classes.map({c: i for i, c in enumerate(CLASSES)}).fillna(CLASSES.index("none")).astype(int).to_numpy()
    out = np.zeros((len(idx), len(CLASSES)))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def fit_models(train_c: pd.DataFrame, train_f: pd.DataFrame, seed: int = 0) -> dict:
    """Fit on one training frame -> {name: predict(cands, feats) -> (n, 4) class probabilities in CLASSES order}.
    Every fit of the trainer goes through here."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    if train_c["heldout"].astype(bool).any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    if (train_c[["sportvu_game", "possession", "candidate"]].to_numpy() != train_f[["sportvu_game", "possession", "candidate"]].to_numpy()).any():
        raise ValueError("%s: candidate and feature rows are not aligned" % WHO)
    y = train_c["label_class"].map({c: i for i, c in enumerate(CLASSES)}).astype(int).to_numpy()
    out = {"rule": lambda cands, feats: one_hot(cands["rule_class"])}
    present = np.unique(y)
    if len(present) < 2 or len(y) < 2:
        freq = np.bincount(y, minlength=len(CLASSES)) / max(1, len(y))
        out["hgb"] = lambda cands, feats, p=freq: np.tile(p, (len(cands), 1))
        return out
    with threadpool_limits(limits=1):      # OpenMP oversubscription on shared cores stalls HGB; one thread is plenty here
        m = HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS).fit(design(train_f), y)

    def predict(cands, feats, m=m):
        with threadpool_limits(limits=1):
            p = m.predict_proba(design(feats))
        full = np.zeros((len(feats), len(CLASSES)))
        full[:, m.classes_] = p
        return full
    out["hgb"] = predict
    return out


def out_of_fold(train_c: pd.DataFrame, train_f: pd.DataFrame, n_folds: int, seed: int) -> tuple:
    """-> ({model: (n, 4) out-of-fold probabilities}, fold per row, protocol note)."""
    games = sorted(train_c["sportvu_game"].unique())
    probs = {m: np.full((len(train_c), len(CLASSES)), np.nan) for m in MODELS}
    fold = np.full(len(train_c), -1, dtype=int)
    if len(games) == 1:
        models = fit_models(train_c, train_f, seed)
        for m in MODELS:
            probs[m] = models[m](train_c, train_f)
        fold[:] = 0
        note = {"mode": "in-sample", "folds": 0, "games": 1,
                "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows)"}
        return probs, fold, note
    k = min(n_folds, len(games))
    for i, (tr_games, te_games) in enumerate(grouped_folds(games, n_folds=k, seed=seed)):
        tr, te = train_c["sportvu_game"].isin(tr_games).to_numpy(), train_c["sportvu_game"].isin(te_games).to_numpy()
        models = fit_models(train_c[tr].reset_index(drop=True), train_f[tr].reset_index(drop=True), seed)
        for m in MODELS:
            probs[m][te] = models[m](train_c[te].reset_index(drop=True), train_f[te].reset_index(drop=True))
        fold[te] = i
    assert all(np.isfinite(probs[m]).all() for m in MODELS)
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games (leave-one-game-out)" % (len(games), n_folds)
    return probs, fold, {"mode": "out-of-fold", "folds": k, "games": len(games), "text": text}


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────
def f1_for(y_true, y_pred, cls: str) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def confusion(y_true, y_pred) -> pd.DataFrame:
    """Rows true class, columns predicted class, in CLASSES order."""
    t = pd.Categorical(y_true, categories=CLASSES)
    p = pd.Categorical(y_pred, categories=CLASSES)
    return pd.crosstab(pd.Series(t, name="true"), pd.Series(p, name="pred"), dropna=False).reindex(index=CLASSES, columns=CLASSES, fill_value=0)


def predicted(probs: np.ndarray) -> np.ndarray:
    return np.asarray(CLASSES, dtype=object)[np.argmax(probs, axis=1)]


def score(cands: pd.DataFrame, probs: dict, n_boot: int, seed: int) -> dict:
    """{model: {f1_ball_screen: {point, lo, hi}, per_class_f1, macro_f1, accuracy, confusion}} with the ball_screen
    F1 interval by game bootstrap (no interval with a single game)."""
    out = {}
    y = cands["label_class"].astype(str).to_numpy(dtype=object)
    for m in MODELS:
        yhat = predicted(probs[m])
        d = pd.DataFrame({"game": cands["sportvu_game"].astype(str).to_numpy(), "y": y, "yhat": yhat})
        r = bootstrap_by_game(d, "game", lambda s: f1_for(s["y"], s["yhat"], "ball_screen"), n=n_boot, seed=seed)
        if d["game"].nunique() < 2:
            r = {**r, "lo": float("nan"), "hi": float("nan")}
        per = {c: f1_for(y, yhat, c) for c in CLASSES}
        out[m] = {"f1_ball_screen": r, "per_class_f1": per,
                  "macro_f1": float(np.nanmean([v for v in per.values()])) if any(np.isfinite(v) for v in per.values()) else float("nan"),
                  "accuracy": float((y == yhat).mean()) if len(y) else float("nan"), "confusion": confusion(y, yhat)}
    return out


def select(scores: dict) -> str:
    """The model with the higher out-of-fold ball_screen F1; ties (or NaN) go to the rule baseline."""
    best, best_v = "rule", scores["rule"]["f1_ball_screen"]["point"]
    for m in MODELS[1:]:
        v = scores[m]["f1_ball_screen"]["point"]
        if np.isfinite(v) and (not np.isfinite(best_v) or v > best_v + 1e-12):
            best, best_v = m, v
    return best


# ── output tables ──────────────────────────────────────────────────────────────────────────────────
def candidate_table(cands: pd.DataFrame, probs: np.ndarray, fold) -> pd.DataFrame:
    out = cands.copy()
    out["pred_class"] = predicted(probs)
    out["p_ball_screen"] = probs[:, CLASSES.index("ball_screen")]
    out["p_selected_class"] = probs.max(axis=1)
    out["fold"] = np.asarray(fold, dtype=int)
    out["heldout"] = out["heldout"].astype(bool)
    cols = ["sportvu_game", "possession", "candidate", "start_frame", "end_frame", "rule_class", "ball_handler_id", "screener_id",
            "label_class", "label_iou", "label_action_type", "pred_class", "p_ball_screen", "p_selected_class", "fold", "heldout"]
    return out[cols]


def detected_table(table: pd.DataFrame, labeller_pass: str) -> pd.DataFrame:
    """actions_detected: selected class other than none with probability above DETECT_P, in the hand-label schema
    with label_source model and action_id the row position within the game (possession, start_frame order)."""
    m = (table["pred_class"] != "none") & (table["p_selected_class"] > DETECT_P)
    d = table[m].sort_values(["sportvu_game", "possession", "start_frame", "candidate"], kind="stable")
    action_id = d.groupby("sportvu_game").cumcount().to_numpy()
    return pd.DataFrame({"sportvu_game": d["sportvu_game"].to_numpy(), "possession": d["possession"].to_numpy(),
                         "action_id": action_id.astype("int32"),
                         "action_type": d["pred_class"].to_numpy(), "start_frame": d["start_frame"].to_numpy(),
                         "end_frame": d["end_frame"].to_numpy(), "ball_handler_id": d["ball_handler_id"].to_numpy(),
                         "screener_id": d["screener_id"].to_numpy(), "label_source": "model", "labeller_pass": labeller_pass,
                         "confidence": d["p_selected_class"].to_numpy()})


def write_tables(out_root, table: pd.DataFrame, detected: pd.DataFrame) -> list:
    from analytics.possessions import game_meta
    written = []
    for game, grp in table.groupby("sportvu_game"):
        held = bool(grp["heldout"].iloc[0])
        season = game_meta(str(game))["season"] if str(game).count(".") == 5 else SEASON
        base = Path(out_root) / "analytics" / "actions" / ("heldout/" if held else "") / season
        (base / "action_candidates").mkdir(parents=True, exist_ok=True)
        (base / "actions_detected").mkdir(parents=True, exist_ok=True)
        grp.to_parquet(base / "action_candidates" / ("%s.parquet" % game), index=False)
        det = detected[detected["sportvu_game"] == game]
        det.to_parquet(base / "actions_detected" / ("%s.parquet" % game), index=False)
        V.write_labels_jsonl(base / "actions_detected" / ("%s.jsonl" % game), det, pd.DataFrame(columns=V.SCREEN_COLUMNS))
        written.append({"sportvu_game": str(game), "heldout": held, "candidates": int(len(grp)), "detected": int(len(det)),
                        "dir": str(base)})
    return written


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def _f(v, nd=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else ("%%.%df" % nd) % v


def _ci(d: dict) -> str:
    if not np.isfinite(d["point"]):
        return "n/a"
    if not np.isfinite(d["lo"]):
        return _f(d["point"])
    return "%s [%s, %s]" % (_f(d["point"]), _f(d["lo"]), _f(d["hi"]))


def md_table(df: pd.DataFrame, nd: int = 3) -> list:
    cols = [str(c) for c in df.columns]
    L = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(_f(float(v), nd))
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def write_report(path: Path, ctx: dict) -> str:
    fx = ctx["fixtures"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Action recognition: ball-screen detection (analytics/actions/train.py)", "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Every number below is fixture output, not a result**: the inputs are synthetic games from %s "
              "with scripted screens, drives and isolations. The real run awaits the SportVU frames, possessions and "
              "hand labels on the owner's Mac (command under Status)." % ctx.get("fixture_source", "the fixture generator"), ""]
    L += ["## Inputs" + tag, "", "| game | held-out | frames | possessions | action labels | screen labels | validation |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in ctx["inputs"]:
        L.append("| %s | %s | %d | %d | %d | %d | %s |" % (g["sportvu_game"], "yes" if g["heldout"] else "no", g["frames"],
                                                        g["possessions"], g["actions"], g["screens"], g["validation"]))
    L += ["", "Label counts by action_type over every game (hand-label schema; only ball_screen, drive and isolation are detected): %s."
          % (", ".join("%s %d" % (k, v) for k, v in sorted(ctx["label_counts"].items())) or "none"),
          "Label counts by label_source: %s." % (", ".join("%s %d" % (k, v) for k, v in sorted(ctx["label_sources"].items())) or "none"),
          "Training games: %d (%s). Held-out games set aside before any fit: %d (%s). Skipped possessions (null offense or no rows): %d. "
          "Games skipped by validation: %s."
          % (ctx["train_games"], ", ".join(ctx["train_game_ids"]) or "none", ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]) or "none",
             ctx["skipped_possessions"], ", ".join(ctx["invalid_games"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids against sportvu/splits.json; "
          "`analytics.schema.assert_time_forward` and `features.assert_no_leak` passed on the training feature frame (%d rows, %d features)."
          % (ctx["train_games"], ctx["train_rows"], len(F.FEATURES)), "",
          "## Candidate generation" + tag, "",
          "Rules (5 Hz): ball_screen = 1 s window, teammate within %g ft of the handler's defender, under %g ft per s, handler within %g ft; "
          "drive = %g s window, handler closing on the basket over %g ft per s; isolation = %g s window, same handler with the ball, no teammate within %g ft. "
          "Target: IoU over frames with a label of at least %.2f."
          % (F.SCREEN_DIST_FT, F.SCREEN_SPEED_FTPS, F.SCREEN_HANDLER_DIST_FT, F.DRIVE_WINDOW_S, F.DRIVE_SPEED_FTPS, F.ISO_WINDOW_S, F.ISO_SPREAD_FT, F.IOU_MIN), "",
          "Candidates: %d over %d possessions (%.2f per possession); %d overlap a label of a type this detector does not model (counted as none)."
          % (ctx["n_candidates"], ctx["n_possessions"], ctx["n_candidates"] / max(1, ctx["n_possessions"]), ctx["other_type_overlaps"]), "",
          "Rule class (rows) against label class (columns), training games:", ""]
    L += md_table(ctx["rule_vs_label"].reset_index().rename(columns={"index": "rule_class"}))
    L += ["", "Candidate recall against the labels (training games), IoU >= %.2f:" % F.IOU_MIN, ""]
    L += md_table(ctx["recall"])
    L += ["", "## Protocol", "",
          "Features (%d, positions and velocities inside the window): %s. `def_recovery_ft` is the one declared post-window feature "
          "(defender to handler distance over the 1 s after the window minus inside it); no feature reads the possession outcome and no label column is a feature."
          % (len(F.FEATURES), ", ".join(F.FEATURES)),
          "Models: rule (the generator's own class) and hgb (HistGradientBoostingClassifier, %s) over %s." % (HGB_PARAMS, list(CLASSES)),
          "Metrics: %s; the ball_screen F1 interval is the 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)%s. "
          "Selection: out-of-fold ball_screen F1, ties to the rule baseline."
          % (ctx["protocol"], ctx["bootstrap"], ctx["seed"], "; no interval with a single game" if ctx["train_games"] < 2 else ""), "",
          "## Metrics (%s, %d candidates, %d training game(s))%s" % (ctx["mode"], ctx["train_rows"], ctx["train_games"], tag), "",
          "| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in MODELS:
        s = ctx["scores"][m]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (m, _ci(s["f1_ball_screen"]), _f(s["per_class_f1"]["drive"]),
                                                          _f(s["per_class_f1"]["isolation"]), _f(s["per_class_f1"]["none"]),
                                                          _f(s["macro_f1"]), _f(s["accuracy"])))
    L += ["", "Selected model: **%s** (plan target: ball_screen F1 over 0.8 on a held-out game%s)." % (ctx["selected"], "; a fixture cannot meet or miss it" if fx else ""),
          "", "Confusion of the selected model (rows true, columns predicted, %s):" % ctx["mode"], ""]
    L += md_table(ctx["scores"][ctx["selected"]]["confusion"].reset_index().rename(columns={"true": "true \\ pred"}))
    L += ["", "Detected actions (selected class other than none with probability above %.1f): %d of %d training candidates." % (DETECT_P, ctx["n_detected_train"], ctx["train_rows"])]
    if ctx["heldout_rows"]:
        L += ["", "## Held-out games (scored once by the models refit on all %d training candidates; never fitted or tuned on)%s" % (ctx["train_rows"], tag), "",
              "%d candidates from %d held-out game(s): %s." % (ctx["heldout_rows"], ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"])), "",
              "| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for m in MODELS:
            s = ctx["heldout_scores"][m]
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (m, _f(s["f1_ball_screen"]["point"]), _f(s["per_class_f1"]["drive"]),
                                                              _f(s["per_class_f1"]["isolation"]), _f(s["per_class_f1"]["none"]),
                                                              _f(s["macro_f1"]), _f(s["accuracy"])))
        L += ["", "Held-out numbers are point estimates (no bootstrap over %d game(s)). Confusion of the selected model on the held-out candidates:" % ctx["heldout_games"], ""]
        L += md_table(ctx["heldout_scores"][ctx["selected"]]["confusion"].reset_index().rename(columns={"true": "true \\ pred"}))
    elif ctx["heldout_games"]:
        L += ["", "## Held-out games", "", "%d held-out game(s) set aside (%s) but no candidate was generated in them." % (ctx["heldout_games"], ", ".join(ctx["heldout_game_ids"]))]
    L += ["", "## Outputs", ""]
    if ctx["written_tables"]:
        for w in ctx["written_tables"]:
            L.append("- %s: %d candidates, %d detected, under `%s`" % (w["sportvu_game"], w["candidates"], w["detected"], w["dir"]))
    else:
        L.append("No table written (no --out). With --out OUT_ROOT the action_candidates and actions_detected tables of docs/contract/actions.md go under OUT_ROOT/analytics/actions/<season>/.")
    L += ["", "## Status", ""]
    if fx:
        L.append("Fixture output, not a result: this run shows the pipeline works end to end on scripted motion. The real run, on the owner's Mac where the SportVU frames, "
                 "the possessions tables and the hand labels (DATA_ROOT/labels/actions/<game>.jsonl, lane I4's tool) live, is")
    else:
        L.append("The real run of this protocol is")
    L += ["", "    " + FULL_RUN_CMD, "",
          "Needs real data before any number means anything: hand-labelled ball screens, drives and isolations on training games "
          "(the plan asks for over 0.8 ball_screen F1 on a held-out game, judged once on the held-out block above), and the possessions table of every labelled game.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analytics.actions.train", description=__doc__.split("\n")[0])
    V.add_input_args(ap)
    ap.add_argument("--report", default="reports/actions.md")
    ap.add_argument("--out", help="OUT_ROOT for the action_candidates and actions_detected tables")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=200)
    return ap.parse_args(argv)


def run(games: list, n_folds: int = 5, seed: int = 0, n_boot: int = 200, fixtures: bool = False, out_root=None,
        report: Path | None = None, command: str = "") -> dict:
    """The whole protocol on a list of game dicts -> the report context (scores, tables, selected model)."""
    valid, inputs, invalid = [], [], []
    label_counts, label_sources = {}, {}
    for g in games:
        g["heldout"] = is_heldout(g)
        rep = V.validate_game(g)
        lab = g.get("labels") or {"actions": pd.DataFrame(), "screens": pd.DataFrame()}
        inputs.append({"sportvu_game": g["sportvu_game"], "heldout": g["heldout"], "frames": int(g["frames"]["frame"].nunique()),
                       "possessions": int(len(g["possessions"])), "actions": int(len(lab["actions"])), "screens": int(len(lab["screens"])),
                       "validation": "ok" if rep["ok"] else "%d problem(s)" % sum(len(rep[p]["problems"]) for p in ("frames", "possessions", "rosters", "labels"))})
        if not rep["ok"]:
            invalid.append(g["sportvu_game"])
            continue
        for k, v in rep["labels"]["counts"].get("by_action_type", {}).items():
            label_counts[k] = label_counts.get(k, 0) + v
        for k, v in rep["labels"]["counts"].get("by_label_source", {}).items():
            label_sources[k] = label_sources.get(k, 0) + v
        valid.append(g)
    if not valid:
        raise SystemExit("%s: no valid game" % WHO)
    cands, feats, recall_all, skipped = F.build_season(valid)
    train_c, train_f, held_c, held_f = training_split(cands, feats)
    if len(train_c) == 0:
        raise SystemExit("%s: no training candidate after setting the held-out games aside" % WHO)
    train_games = [g for g in valid if not g["heldout"]]
    train_actions = pd.concat([g["labels"]["actions"] for g in train_games if len(g["labels"]["actions"])], ignore_index=True) \
        if any(len(g["labels"]["actions"]) for g in train_games) else pd.DataFrame(columns=["sportvu_game", "possession", "action_type", "start_frame", "end_frame"])
    recall = F.candidate_recall(train_c, train_actions)
    probs, fold, note = out_of_fold(train_c, train_f, n_folds, seed)
    scores = score(train_c, probs, n_boot, seed)
    selected = select(scores)
    table = candidate_table(train_c, probs[selected], fold)
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "command": command, "fixtures": fixtures,
           "fixture_source": V.fixture_source() if fixtures else "",
           "inputs": inputs, "invalid_games": invalid, "label_counts": label_counts, "label_sources": label_sources,
           "train_games": int(train_c["sportvu_game"].nunique()), "train_game_ids": sorted(train_c["sportvu_game"].unique()),
           "heldout_games": sum(1 for g in valid if g["heldout"]), "heldout_game_ids": sorted(g["sportvu_game"] for g in valid if g["heldout"]),
           "train_rows": int(len(train_c)), "heldout_rows": int(len(held_c)), "skipped_possessions": int(skipped),
           "n_candidates": int(len(train_c)), "n_possessions": int(sum(len(g["possessions"]) for g in train_games)),
           "other_type_overlaps": int((train_c["label_action_type"].notna() & (train_c["label_class"] == "none")).sum()),
           "rule_vs_label": pd.crosstab(train_c["rule_class"], train_c["label_class"]).reindex(index=F.DETECT_CLASSES, columns=CLASSES, fill_value=0),
           "recall": recall, "protocol": note["text"], "mode": note["mode"], "folds": note["folds"], "bootstrap": n_boot, "seed": seed,
           "scores": scores, "selected": selected, "n_detected_train": int(len(detected_table(table, "oof"))),
           "heldout_scores": {}, "written_tables": [], "table": table}
    if len(held_c):
        models = fit_models(train_c, train_f, seed)
        held_probs = {m: models[m](held_c, held_f) for m in MODELS}
        ctx["heldout_scores"] = score(held_c, held_probs, 1, seed)
        held_table = candidate_table(held_c, held_probs[selected], np.full(len(held_c), -1))
        ctx["table"] = pd.concat([table, held_table], ignore_index=True)
    labeller_pass = "%s %s seed%d %s" % (WHO, selected, seed, "fixtures" if fixtures else "data")
    ctx["detected"] = detected_table(ctx["table"], labeller_pass)
    if out_root:
        ctx["written_tables"] = write_tables(out_root, ctx["table"], ctx["detected"])
    if report is not None:
        ctx["report_text"] = write_report(Path(report), ctx)
    return ctx


def main(argv=None) -> dict:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    games = V.load_inputs(a)
    ctx = run(games, n_folds=a.folds, seed=a.seed, n_boot=a.bootstrap, fixtures=bool(a.fixtures), out_root=a.out, report=Path(a.report),
              command="python -m analytics.actions.train " + " ".join(argv if argv is not None else sys.argv[1:]))
    print(ctx["report_text"])
    return ctx


if __name__ == "__main__":
    main()
