"""analytics/gravity/train.py: the gravity trainer, selection, player table, validations and report.

    python -m analytics.gravity.train --fixtures | --data DATA_ROOT --report reports/gravity.md
        [--folds 5] [--seed 0] [--bootstrap 200] [--min-possessions 300] [--out OUT_ROOT] [--season 2015-16]

Protocol
  1. load the game dicts (validate.load_games), clean them (validate.validate_games), set the held-out games
     aside (features.is_heldout: the dict flag, a held-out token of sportvu/splits.json, or the CLE arena) and
     call analytics.eval.refuse_heldout on the training game ids
  2. build the 1 Hz player frame and the spacing frame (features.build); schema.assert_time_forward runs on
     the feature frame there and again in features.design before every fit
  3. grouped folds by sportvu_game (analytics.eval.grouped_folds); in each fold the shooting prior comes from
     the shots of the training-fold games only (features.skill_table), attached to both sides
  4. candidates: constant (training mean, the floor), ridge regression of y on the location features
     (StandardScaler + Ridge alpha 1, the plan's model) and HistGradientBoostingRegressor on the same features;
     selection by out-of-fold MSE; HGB is kept only when the paired bootstrap by game of MSE(HGB) - MSE(ridge)
     excludes zero on the side of HGB, else ridge
  5. gravity_ft = expected_ft - nearest_def_dist_ft (positive: the defender stays closer than location predicts;
     see analytics/gravity/__init__.py for why this sign); a player's gravity is the mean over frames, ranked only
     with at least --min-possessions possessions (analytics.eval.gated), with a bootstrap interval by game
  6. validations: (a) the known high-gravity players (the fixture dict's high_gravity_ids) rank in the top decile
     of the gated players; on real data the top ten are listed without a claim; (b) the sign test: the
     location-only xFG probability of teammates' shots (analytics.features_xfg features, analytics.train_xfg
     logistic, out of fold by game) is higher with a high-gravity player on the floor; one difference per game
     (mean over the candidate players present), a one-sided binomial sign test over games and the mean
     difference with a bootstrap interval by game
  7. held-out games, when present, are scored exactly once by the selected model refit on every training row,
     with the shooting prior from every training shot; they never fit, tune or rank anything
  8. the report (--report) and, with --out, the three tables of docs/contract/gravity.md

Fewer games than folds: the fold count becomes the game count (leave-one-game-out) and the report says so;
one training game: in-sample, labelled. With --fixtures the report is labelled "fixture output, not a result".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")   # 4 cores shared by 21 lanes: HGB must not take them all

import numpy as np
import pandas as pd

from analytics import features_xfg, schema, train_xfg
from analytics.eval import bootstrap_by_game, gated, grouped_folds, paired_bootstrap_by_game, refuse_heldout
from analytics.gravity import MIN_POSSESSIONS, WHO
from analytics.gravity import features as F
from analytics.gravity import validate as V

MODELS = ("constant", "ridge", "hgb")
RIDGE_ALPHA = 1.0
HGB_PARAMS = dict(max_iter=150, learning_rate=0.1, max_depth=4, min_samples_leaf=40)
TOP_DECILE = 0.10
TOP_N_REAL = 10
MAC_CMD = ("python -m analytics.gravity.train --data DATA_ROOT --report reports/gravity.md --folds 5 --seed 0 "
           "--bootstrap 1000 --min-possessions 300 --out DATA_ROOT")


# ── models ─────────────────────────────────────────────────────────────────────────────────────────
def fit_models(train: pd.DataFrame, seed: int = 0) -> dict:
    """{name: predict(frame) -> expected_ft} fitted on one training frame. Refuses a held-out row."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if train["heldout"].any():
        raise ValueError("%s: a held-out row reached fit_models" % WHO)
    X, y = F.design(train), train[F.TARGET].to_numpy(dtype=float)
    mean = float(y.mean())
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(X, y)
    hgb = HistGradientBoostingRegressor(random_state=seed, **HGB_PARAMS).fit(X, y)
    return {"constant": lambda fr, m=mean: np.full(len(fr), m),
            "ridge": lambda fr, m=ridge: m.predict(F.design(fr)),
            "hgb": lambda fr, m=hgb: m.predict(F.design(fr))}


def shots_of(games, ids) -> pd.DataFrame:
    """Concatenated shots of the game dicts whose id is in ids (empty frame when none has shots)."""
    parts = [d["shots"] for d in games if F.game_id(d) in set(ids) and d.get("shots") is not None and len(d["shots"])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=V.SHOT_COLUMNS)


def rosters_of(games) -> pd.DataFrame:
    parts = [d["rosters"] for d in games if d.get("rosters") is not None and len(d["rosters"])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=schema.columns("rosters"))


def out_of_fold(pf: pd.DataFrame, games: list, n_folds: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold expectations of the three candidates -> (frame with fold, skill columns and p_<model>, note)."""
    ids = sorted(pf["sportvu_game"].unique())
    rosters = rosters_of(games)
    pred = pf.copy()
    pred["fold"] = -1
    for m in MODELS:
        pred["p_" + m] = np.nan
    if len(ids) == 1:
        table = F.skill_table(shots_of(games, ids), rosters)
        pred = F.attach_skill(pred, table)
        models = fit_models(pred, seed)
        for m in MODELS:
            pred["p_" + m] = models[m](pred)
        pred["fold"] = 0
        return pred, {"mode": "in-sample", "folds": 0, "games": 1,
                      "text": "one training game: no grouped fold is possible, so the metrics are IN-SAMPLE"}
    k = min(n_folds, len(ids))
    for fold, (tr_ids, te_ids) in enumerate(grouped_folds(ids, n_folds=k, seed=seed)):
        tr, te = pred["sportvu_game"].isin(tr_ids), pred["sportvu_game"].isin(te_ids)
        table = F.skill_table(shots_of(games, tr_ids), rosters)      # training-fold shots only
        train_fr = F.attach_skill(pred[tr], table)
        test_fr = F.attach_skill(pred[te], table)
        models = fit_models(train_fr, seed)
        for m in MODELS:
            pred.loc[te, "p_" + m] = models[m](test_fr)
        for c in F.SKILL_COLUMNS:
            pred.loc[te, c] = test_fr[c].to_numpy()
        pred.loc[te, "fold"] = fold
    assert pred[["p_" + m for m in MODELS]].notna().all().all()
    text = "out-of-fold under %d grouped folds by sportvu_game (seed %d)" % (k, seed)
    if k < n_folds:
        text += "; fewer games (%d) than the %d folds asked, so n_folds = number of games" % (len(ids), n_folds)
    return pred, {"mode": "out-of-fold", "folds": k, "games": len(ids), "text": text}


# ── scoring and selection ──────────────────────────────────────────────────────────────────────────
def mse(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def score(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """{model: {point, lo, hi}} of out-of-fold MSE with bootstrap by game, plus the paired HGB vs ridge test
    and the selection."""
    d = pd.DataFrame({"game": pred["sportvu_game"].astype(str).to_numpy(), "y": pred[F.TARGET].to_numpy(dtype=float)})
    for m in MODELS:
        d["p_" + m] = pred["p_" + m].to_numpy(dtype=float)
    one_game = d["game"].nunique() < 2
    out = {"mse": {}, "var_y": float(d["y"].var())}
    for m in MODELS:
        r = bootstrap_by_game(d, "game", lambda s, m=m: mse(s["y"], s["p_" + m]), n=n_boot, seed=seed)
        if one_game:
            r = {**r, "lo": float("nan"), "hi": float("nan")}
        out["mse"][m] = r
    pair = paired_bootstrap_by_game(d, "game", lambda s: mse(s["y"], s["p_hgb"]), lambda s: mse(s["y"], s["p_ridge"]),
                                    n=n_boot, seed=seed)
    if one_game:
        pair = {**pair, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    out["paired_hgb_minus_ridge"] = pair
    out["selected"] = "hgb" if (pair["excludes_zero"] and pair["diff"] < 0) else "ridge"
    return out


# ── gravity tables ─────────────────────────────────────────────────────────────────────────────────
def gravity_frames(pred: pd.DataFrame, selected: str) -> pd.DataFrame:
    """The gravity_frames table of docs/contract/gravity.md from an out-of-fold (or held-out) prediction frame."""
    expected = pred["p_" + selected].to_numpy(dtype=float)
    y = pred[F.TARGET].to_numpy(dtype=float)
    return pd.DataFrame({"sportvu_game": pred["sportvu_game"].astype(str).to_numpy(),
                         "possession": pred["possession"].to_numpy(dtype=np.int32), "frame": pred["frame"].to_numpy(dtype=np.int32),
                         "player_id": pred["player_id"].to_numpy(dtype=np.int32), "nearest_def_dist_ft": y.astype(np.float32),
                         "expected_ft": expected.astype(np.float32), "gravity_ft": (expected - y).astype(np.float32),
                         "has_ball": pred["has_ball"].to_numpy(dtype=float) > 0.5, "fold": pred["fold"].to_numpy(dtype=np.int16),
                         "heldout": pred["heldout"].to_numpy(dtype=bool)})


def gravity_players(gf: pd.DataFrame, n_min: int = MIN_POSSESSIONS, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """The gravity_players table: mean gravity per player over frames, possessions counted as distinct
    (sportvu_game, possession), insufficient under n_min possessions (analytics.eval.gated), and a percentile
    interval from n_boot resamples of the player's games (NaN with one game)."""
    gf = gf[~gf["heldout"].astype(bool)]
    pp = gf[["player_id", "sportvu_game", "possession"]].drop_duplicates()
    gate = gated(pp, "player_id", n_min).groupby("player_id")["insufficient"].first()
    by_game = gf.groupby(["player_id", "sportvu_game"])["gravity_ft"].agg(["sum", "size"]).reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for pid, grp in by_game.groupby("player_id"):
        s, n = grp["sum"].to_numpy(dtype=float), grp["size"].to_numpy(dtype=float)
        point = float(s.sum() / n.sum())
        if len(grp) >= 2 and n_boot > 0:
            draws = rng.integers(0, len(grp), size=(n_boot, len(grp)))
            vals = s[draws].sum(axis=1) / n[draws].sum(axis=1)
            lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
        else:
            lo = hi = float("nan")
        rows.append({"player_id": int(pid), "n_possessions": int((pp["player_id"] == pid).sum()), "n_frames": int(n.sum()),
                     "n_games": int(len(grp)), "gravity_ft": point, "lo": lo, "hi": hi, "insufficient": bool(gate.get(pid, True))})
    cols = ["player_id", "n_possessions", "n_frames", "n_games", "gravity_ft", "lo", "hi", "insufficient"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["insufficient", "gravity_ft"], ascending=[True, False], kind="stable").reset_index(drop=True)


def top_decile_check(players: pd.DataFrame, known_ids) -> dict:
    """Rank of each known high-gravity id among the gated players, and whether all sit in the top decile."""
    ranked = players[~players["insufficient"]].reset_index(drop=True)
    n = len(ranked)
    cut = max(1, math.ceil(TOP_DECILE * n)) if n else 0
    ranks = {}
    for pid in known_ids:
        hit = np.flatnonzero(ranked["player_id"].to_numpy() == int(pid))
        ranks[int(pid)] = int(hit[0]) + 1 if len(hit) else None
    ok = bool(known_ids) and n > 0 and all(r is not None and r <= cut for r in ranks.values())
    return {"n_gated": int(n), "decile_cut": int(cut), "ranks": ranks, "passed": ok,
            "top": ranked.head(TOP_N_REAL)[["player_id", "n_possessions", "gravity_ft", "lo", "hi"]].to_dict("records")}


# ── sign test with the location-only xFG probability ──────────────────────────────────────────────
def xfg_oof(games: list, n_folds: int, seed: int) -> pd.DataFrame:
    """Shots of the training games with shooter_id, their location-only xFG probability out of fold by game
    (analytics.train_xfg logistic on analytics.features_xfg features) and the shot's period and clock."""
    rosters = rosters_of(games)
    ids = sorted(F.game_id(d) for d in games)
    shots = shots_of(games, ids)
    if len(shots) == 0:
        return pd.DataFrame(columns=["sportvu_game", "period", "clock_s", "team", "shooter_id", "p"])
    feats = features_xfg.features(shots)
    feats["heldout"] = False
    out = pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype(str).to_numpy(), "period": shots["period"].to_numpy(),
                        "clock_s": pd.to_numeric(shots["clock_s"]).to_numpy(dtype=float), "team": shots["team"].astype(str).to_numpy(),
                        "shooter_id": F.shot_player_ids(shots, rosters).to_numpy(), "p": np.nan})
    g = feats["sportvu_game"].astype(str)
    uniq = sorted(g.unique())
    if len(uniq) < 2:
        models = train_xfg.fit_models(feats)
        out["p"] = models["logistic"](feats)
        return out
    for tr_ids, te_ids in grouped_folds(uniq, n_folds=min(n_folds, len(uniq)), seed=seed):
        tr, te = g.isin(tr_ids).to_numpy(), g.isin(te_ids).to_numpy()
        models = train_xfg.fit_models(feats[tr])
        out.loc[te, "p"] = models["logistic"](feats[te])
    return out


class _OnCourt:
    """Who is on the floor at the frame nearest a (period, clock) of one game."""

    def __init__(self, frames: pd.DataFrame):
        fr = frames.sort_values("frame", kind="stable")
        self.frame = fr["frame"].to_numpy(dtype=np.int64)
        self.pid = fr["player_id"].to_numpy(dtype=np.int64)
        self.uniq, self.start = np.unique(self.frame, return_index=True)
        self.end = np.append(self.start[1:], len(self.frame))
        fl = fr.drop_duplicates("frame")
        self.fl_frame = fl["frame"].to_numpy(dtype=np.int64)
        self.fl_period = fl["period"].to_numpy(dtype=np.int64)
        self.fl_clock = fl["game_clock_s"].to_numpy(dtype=float)

    def players(self, period: int, clock_s: float) -> set:
        m = self.fl_period == int(period)
        if not m.any():
            return set()
        idx = np.flatnonzero(m)
        f = self.fl_frame[idx[int(np.argmin(np.abs(self.fl_clock[idx] - float(clock_s))))]]
        k = int(np.searchsorted(self.uniq, f))
        return set(self.pid[self.start[k]:self.end[k]].tolist())


def sign_test(games: list, candidates, xfg: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Per game: mean over the candidate players of (mean xFG p of teammates' shots with the player on the
    floor) - (the same with the player off). Then a one-sided binomial sign test over games (p > 0.5 of a
    positive difference) and the mean difference with a bootstrap interval by game."""
    from scipy.stats import binomtest
    rosters = rosters_of(games)
    team_of = {}
    for pid, team, g in zip(rosters["player_id"], rosters["team"], rosters["sportvu_game"]):
        team_of[(str(g), int(pid))] = str(team)
    per_game = []
    for d in games:
        g = F.game_id(d)
        sh = xfg[xfg["sportvu_game"] == g]
        if len(sh) == 0:
            continue
        oc = _OnCourt(d["frames"])
        on_sets = [oc.players(p, c) for p, c in zip(sh["period"], sh["clock_s"])]
        diffs = []
        for pid in candidates:
            team = team_of.get((g, int(pid)))
            if team is None:
                continue
            mates = (sh["team"].to_numpy() == team) & (sh["shooter_id"].to_numpy() != int(pid))
            on = np.array([int(pid) in s for s in on_sets])
            p = sh["p"].to_numpy(dtype=float)
            a, b = p[mates & on], p[mates & ~on]
            if len(a) and len(b):
                diffs.append(float(a.mean() - b.mean()))
        if diffs:
            per_game.append({"game": g, "diff": float(np.mean(diffs)), "n_players": len(diffs)})
    out = {"n_games": len(per_game), "n_positive": 0, "p_value": float("nan"), "mean_diff": float("nan"),
           "lo": float("nan"), "hi": float("nan"), "candidates": [int(c) for c in candidates], "per_game": per_game}
    if not per_game:
        return out
    df = pd.DataFrame(per_game)
    out["n_positive"] = int((df["diff"] > 0).sum())
    out["p_value"] = float(binomtest(out["n_positive"], len(df), 0.5, alternative="greater").pvalue)
    r = bootstrap_by_game(df, "game", lambda s: float(s["diff"].mean()), n=n_boot, seed=seed)
    out.update({"mean_diff": r["point"], "lo": r["lo"] if len(df) > 1 else float("nan"), "hi": r["hi"] if len(df) > 1 else float("nan")})
    return out


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def _f(v, nd=3) -> str:
    try:
        return "n/a" if v is None or not np.isfinite(float(v)) else ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return str(v)


def _ci(d: dict, nd=3) -> str:
    if not np.isfinite(d.get("point", float("nan"))):
        return "n/a"
    if not np.isfinite(d.get("lo", float("nan"))):
        return _f(d["point"], nd)
    return "%s [%s, %s]" % (_f(d["point"], nd), _f(d["lo"], nd), _f(d["hi"], nd))


def write_report(path: Path, ctx: dict) -> str:
    fx = ctx["fixture"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Gravity and spacing (analytics/gravity)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written"], ctx["command"]), ""]
    if fx:
        L += ["**Every number below is fixture output, not a result**: the inputs are synthetic games from `%s` "
              "with two planted high-gravity players, used to exercise the pipeline. The real run is the Mac command at the end." % ctx["fixture_source"], ""]
    L += ["## Inputs", "",
          "| games | training | held-out | player-frames (train) | spacing frames (train) | shots (train) |", "| --- | --- | --- | --- | --- | --- |",
          "| %d | %d | %d | %d | %d | %d |" % (ctx["n_games"], ctx["train_games"], ctx["heldout_games"], ctx["train_rows"],
                                             ctx["spacing_rows"], ctx["train_shots"]), "",
          "Training games: %s. Held-out games set aside before any fit: %s." % (", ".join(ctx["train_game_ids"]) or "none",
                                                                               ", ".join(ctx["heldout_game_ids"]) or "none"),
          "Guard: `analytics.eval.refuse_heldout` checked %d training game ids; `schema.assert_time_forward` passed on the feature frame." % ctx["train_games"],
          "Sampling: %d frames sampled at 1 Hz inside possessions, %d kept (not ten players: %d, no ball: %d, degenerate hull: %d)."
          % (ctx["sampled"], ctx["kept"], ctx["not_ten"], ctx["no_ball"], ctx["degenerate"]),
          "Validation counts: %d schema problem(s), dropped rows %s." % (ctx["schema_problems"], json.dumps(ctx["dropped"])), "",
          "## Protocol", "",
          "Unit: player-frame at 1 Hz inside possessions. Target: nearest-defender distance (ft). Features: %s." % ", ".join(F.FEATURES),
          "Shooting prior: smoothed make rate by zone (K = %d) from training-fold shots only, matched to SportVU ids by name. "
          "Gravity sign: `gravity_ft = expected_ft - nearest_def_dist_ft` (positive: defenders closer than location predicts)." % int(F.K_SMOOTH),
          "Folds: %s. Intervals: 2.5 and 97.5 percentiles of %d bootstrap resamples of whole games (seed %d)." % (ctx["protocol"], ctx["bootstrap"], ctx["seed"]),
          "", "## Candidates (%s MSE, %d player-frames, %d games; variance of y %s)" % (ctx["mode"], ctx["train_rows"], ctx["train_games"], _f(ctx["scores"]["var_y"])), "",
          "| model | MSE | R2 (1 - MSE / var y) |", "| --- | --- | --- |"]
    for m in MODELS:
        s = ctx["scores"]["mse"][m]
        L.append("| %s | %s | %s |" % (m, _ci(s), _f(1 - s["point"] / ctx["scores"]["var_y"])))
    pair = ctx["scores"]["paired_hgb_minus_ridge"]
    L += ["", "Paired bootstrap by game, MSE(hgb) - MSE(ridge): %s [%s, %s], excludes zero: %s. Selected: **%s** (HGB only when the "
          "interval excludes zero on HGB's side)." % (_f(pair["diff"], 4), _f(pair["lo"], 4), _f(pair["hi"], 4), pair["excludes_zero"], ctx["selected"]),
          "", "## Players (gate: %d possessions; %d gated of %d)%s" % (ctx["min_possessions"], ctx["n_gated"], ctx["n_players"], tag), "",
          "| rank | player_id | possessions | gravity_ft | lo | hi |", "| --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(ctx["top_decile"]["top"]):
        L.append("| %d | %d | %d | %s | %s | %s |" % (i + 1, r["player_id"], r["n_possessions"], _f(r["gravity_ft"]), _f(r["lo"]), _f(r["hi"])))
    td = ctx["top_decile"]
    L += ["", "### Top-decile check%s" % tag, ""]
    if ctx["known_ids"]:
        L.append("Known high-gravity players %s; ranks among %d gated players %s; decile cut %d; passed: %s."
                 % (ctx["known_ids"], td["n_gated"], td["ranks"], td["decile_cut"], td["passed"]))
        L += ["", "Mean out-of-fold gravity (ft) of the known players under each candidate (the skill prior nearly identifies a player, "
              "so a flexible model can absorb part of a player's pull into its expectation; the fixture plants a pull of 2 ft):", "",
              "| model | " + " | ".join(str(p) for p in ctx["known_ids"]) + " |", "| --- |" + " --- |" * len(ctx["known_ids"])]
        for m in MODELS:
            L.append("| %s | %s |" % (m, " | ".join(_f(ctx["known_by_model"][m][p]) for p in ctx["known_ids"])))
    else:
        L.append("No known high-gravity player on this input: the top ten above are listed without a claim.")
    st = ctx["sign_test"]
    L += ["", "### Sign test: teammates' location-only xFG with a high-gravity player on the floor%s" % tag, "",
          "Candidates %s (%s). Games with a difference: %d; positive: %d; one-sided binomial p = %s; mean difference %s [%s, %s] (bootstrap by game)."
          % (st["candidates"], ctx["candidate_source"], st["n_games"], st["n_positive"], _f(st["p_value"], 4), _f(st["mean_diff"], 4), _f(st["lo"], 4), _f(st["hi"], 4)),
          "", "## Spacing%s" % tag, "",
          "Mean hull area %s ft2 (sd %s), mean nearest-defender distance %s ft over %d training possession-frames."
          % (_f(ctx["hull_mean"], 1), _f(ctx["hull_sd"], 1), _f(ctx["mnd_mean"], 2), ctx["spacing_rows"]),
          "", "## Held-out, scored once%s" % tag, ""]
    if ctx["heldout_games"]:
        h = ctx["heldout_scores"]
        L.append("%d held-out game(s), %d player-frames, scored once by the %s model refit on all %d training rows: MSE %s (constant %s, ridge %s, hgb %s). "
                 "Never fitted, tuned or ranked on." % (ctx["heldout_games"], ctx["heldout_rows"], ctx["selected"], ctx["train_rows"],
                                                        _f(h[ctx["selected"]]), _f(h["constant"]), _f(h["ridge"]), _f(h["hgb"])))
    else:
        L.append("No held-out game in this input.")
    L += ["", "## Outputs", "", ctx["outputs"], "", "## Assumptions", ""]
    L += ["- " + s for s in ASSUMPTIONS]
    L += ["", "## Mac command (the real run)", "", "    " + MAC_CMD, "",
          "DATA_ROOT holds sportvu_contract/, analytics/possessions/2015-16/ (and heldout/2015-16/), analytics/rosters/, analytics/shot_frames/ "
          "and collect/bref_shots/shotchart/2015-16.parquet; the tables land under DATA_ROOT/analytics/gravity/.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


ASSUMPTIONS = [
    "The spec gives gravity as y minus the expectation and also says positive means defenders stay closer; these conflict, "
    "and the stated meaning wins: gravity_ft = expected_ft - nearest_def_dist_ft.",
    "A possession's frames run from start_frame to the next possession's start_frame (exclusive) in the same period, not to "
    "end_frame, which the contract marks post-outcome; analytics.possessions makes possessions contiguous so the span is the same.",
    "The attacked basket per (offense team, period) is the one nearer the mean x of the sampled offensive positions.",
    "has_ball marks the offensive player nearest the ball (the section 20 handler), with no distance threshold.",
    "The shooting prior uses zones by distance only (rim, paint, short mid, long mid, three) and K = 25 shots of smoothing; shooters of "
    "basketball-reference rows (null player_id) are matched to SportVU ids by name against the game's roster.",
    "shot_clock_s null is filled with 24 and flagged; frames without a ball, or without exactly five offensive and five defensive players, are dropped and counted.",
    "A player's interval resamples that player's games; a player seen in one game gets no interval. The player table is built on training frames only.",
    "The sign test's candidates are the fixture's planted players on fixtures and the top ten gated players on real data; the xFG probability "
    "is the location-only logistic of analytics.train_xfg fitted out of fold by game on the training games' shots.",
    "On-court at a shot means the player has a row in the frame of the same period nearest the shot's clock.",
]


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", default=None, help="DATA_ROOT (owner's Mac layout)")
    ap.add_argument("--report", default="reports/gravity.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--min-possessions", type=int, default=MIN_POSSESSIONS)
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes OUT_ROOT/analytics/gravity/{gravity_frames,gravity_players,spacing_frames}.parquet")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--n-games", type=int, default=8, help="fixture games (--fixtures)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    raw = V.load_games(a, who=WHO)
    games, counts = V.validate_games(raw)
    train_games = [d for d in games if not F.is_heldout(d)]
    held_games = [d for d in games if F.is_heldout(d)]
    train_ids = sorted(F.game_id(d) for d in train_games)
    refuse_heldout(train_ids, WHO)
    if not train_games:
        raise SystemExit("%s: no training game after setting the held-out games aside" % WHO)
    pf, sp, fcounts = F.build(train_games)
    schema.assert_time_forward(pf, "states_1hz")
    if len(pf) == 0:
        raise SystemExit("%s: no player-frame survived sampling" % WHO)
    assert not pf["heldout"].any()
    pred, note = out_of_fold(pf, train_games, a.folds, a.seed)
    scores = score(pred, a.bootstrap, a.seed)
    selected = scores["selected"]
    gf = gravity_frames(pred, selected)
    players = gravity_players(gf, a.min_possessions, a.bootstrap, a.seed)
    known = sorted({int(p) for d in train_games for p in (d.get("high_gravity_ids") or [])})
    td = top_decile_check(players, known)
    known_by_model = {m: {int(p): float((pred.loc[pred["player_id"] == p, "p_" + m] - pred.loc[pred["player_id"] == p, F.TARGET]).mean())
                          for p in known} for m in MODELS}
    if known:
        candidates, candidate_source = known, "the fixture's planted high-gravity players"
    else:
        candidates = [int(r["player_id"]) for r in td["top"]]
        candidate_source = "the top ten gated players by gravity, no claim attached"
    xfg = xfg_oof(train_games, a.folds, a.seed)
    st = sign_test(train_games, candidates, xfg, a.bootstrap, a.seed)
    heldout_scores, hf, heldout_rows = {}, pd.DataFrame(), 0
    if held_games:
        hpf, hsp, hcounts = F.build(held_games)
        if len(hpf):
            table = F.skill_table(shots_of(train_games, train_ids), rosters_of(train_games))
            full = F.attach_skill(pf, table)
            models = fit_models(full, a.seed)
            hpf = F.attach_skill(hpf, table)
            hpf["fold"] = -1
            for m in MODELS:
                hpf["p_" + m] = models[m](hpf)
                heldout_scores[m] = mse(hpf[F.TARGET], hpf["p_" + m])
            hf = gravity_frames(hpf, selected)
            heldout_rows = int(len(hf))
            sp = pd.concat([sp, hsp.assign(heldout=True)], ignore_index=True) if len(hsp) else sp
    sp = sp.assign(heldout=sp["heldout"].fillna(False).astype(bool)) if "heldout" in sp.columns else sp.assign(heldout=False)
    all_gf = pd.concat([gf, hf], ignore_index=True) if len(hf) else gf
    outputs = "No tables written (pass --out OUT_ROOT)."
    if a.out:
        root = Path(a.out) / "analytics" / "gravity"
        root.mkdir(parents=True, exist_ok=True)
        all_gf.to_parquet(root / "gravity_frames.parquet", index=False)
        players.to_parquet(root / "gravity_players.parquet", index=False)
        sp.to_parquet(root / "spacing_frames.parquet", index=False)
        outputs = "Tables written under `%s`: gravity_frames (%d rows), gravity_players (%d rows), spacing_frames (%d rows)." % (
            root, len(all_gf), len(players), len(sp))
    sp_train = sp[~sp["heldout"]]
    ctx = {"written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "command": "python -m analytics.gravity.train " + " ".join(argv if argv is not None else sys.argv[1:]),
           "fixture": bool(a.fixtures), "fixture_source": _fixture_source() if a.fixtures else "",
           "n_games": len(games), "train_games": len(train_games), "heldout_games": len(held_games), "train_rows": int(len(pf)),
           "spacing_rows": int(len(sp_train)), "train_shots": int(len(shots_of(train_games, train_ids))),
           "train_game_ids": train_ids, "heldout_game_ids": sorted(F.game_id(d) for d in held_games),
           "sampled": sum(c["sampled"] for c in fcounts), "kept": sum(c["kept"] for c in fcounts),
           "not_ten": sum(c["not_ten"] for c in fcounts), "no_ball": sum(c["no_ball"] for c in fcounts),
           "degenerate": sum(c["degenerate_hull"] for c in fcounts),
           "schema_problems": int(sum(len(v) for c in counts for v in c["schema"].values())),
           "dropped": {k: int(sum(c["dropped"].get(k, 0) for c in counts)) for k in sorted({k for c in counts for k in c["dropped"]})},
           "protocol": note["text"], "mode": note["mode"], "bootstrap": a.bootstrap, "seed": a.seed, "scores": scores, "selected": selected,
           "min_possessions": a.min_possessions, "n_gated": int((~players["insufficient"]).sum()), "n_players": int(len(players)),
           "top_decile": td, "known_ids": known, "known_by_model": known_by_model, "sign_test": st, "candidate_source": candidate_source,
           "hull_mean": float(sp_train["hull_area_ft2"].mean()) if len(sp_train) else float("nan"),
           "hull_sd": float(sp_train["hull_area_ft2"].std()) if len(sp_train) else float("nan"),
           "mnd_mean": float(sp_train["mean_nearest_def_ft"].mean()) if len(sp_train) else float("nan"),
           "heldout_scores": heldout_scores, "heldout_rows": heldout_rows, "outputs": outputs,
           "gravity_frames": all_gf, "gravity_players": players, "spacing_frames": sp, "pred": pred, "xfg": xfg}
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


def _fixture_source() -> str:
    try:
        from tests import _fx_gravity
        return _fx_gravity.SOURCE
    except ImportError:
        return "tests._fx_gravity"


if __name__ == "__main__":
    main()
