"""tools/research_winprob.py: winprob-E1, a pregame strength drift term from past results (docs/research/winprob.md
section 6; PLAN.md D71, D79, D89).

    python3 -m export.heavy --name winprob-E1 -- python3 -m tools.research_winprob e1-features [--artifacts DIR]
    python3 -m export.heavy --name winprob-E1 -- python3 -m tools.research_winprob e1-gate     [--artifacts DIR]
    python3 -m export.heavy --name winprob-E1 -- python3 -m tools.research_winprob e1-dev      [--artifacts DIR]
    python3 -m export.heavy --name winprob-E1 -- python3 -m tools.research_winprob e1-final    [--artifacts DIR]

e1-features  The frozen state frame (analytics.winprob.features, unchanged) of the development seasons 2020-21 to
             2023-24, through the frozen clock rule and the frozen season split, cached under the artifacts folder.
e1-gate      The baseline arm is the frozen logistic (analytics.winprob.train.fit_models, unchanged). Refit here on the
             development seasons under the frozen 5 grouped folds (seed 0), it must reproduce the stored out-of-fold
             p_home_win of DATA_ROOT/analytics/winprob/<season>/winprob_states.parquet (fold >= 0, training rows only)
             to GATE_TOLERANCE, with no fold mismatch. No held-out row and no metric.
e1-dev       Elo parameters chosen by forward chaining (tools.research_winprob_elo), the Elo prefix check and the
             frozen prefix_invariance_violations check over the new columns, then the development evaluation: forward
             chaining by season (fit the three arms on the seasons before 2021-22, 2022-23 and 2023-24, score that
             season), paired bootstrap by game (2000 resamples, seed 0). Secondary: the frozen protocol (5 grouped
             folds inside the development seasons). Writes reports/research/winprob-E1-dev.json.
e1-final     The one final scoring call: the three arms fitted on every development row, Elo carried through 2024-25 as
             state (D89), the 2025-26 season scored once through tools.research_testset.score_once. The scorer alone
             reads the test files, through scoring_data_root, and updates Elo game by game from the season's results.
             Writes reports/research/winprob-E1-final.json.

Arms (the logistic recipe of the frozen trainer: standardised, C=1, lbfgs, max_iter 2000, one thread):
    baseline   the frozen logistic on margin, seconds_remaining, possession_home (with its missing flag),
               margin_x_sqrt_time, margin_over_sqrt_time
    e1         the same plus mu_hat and mu_drift = mu_hat * sqrt(seconds_remaining / 2880) (Stern's drift term)
    ablation   the same plus mu_hat only (no root-time drift term)

Metric (pre-registered): mean bucket Brier over the five regulation buckets, per-bucket Brier and log loss beside it.
Adoption: e1 improves mean bucket Brier by at least THRESHOLD with the paired 95 percent interval excluding zero, and
no regulation bucket is worse beyond its interval (the lower end of that bucket's paired interval above zero).

Artifacts go only under the artifacts folder (never under DATA_ROOT, never under reports/); results go to
reports/research/. Every command runs under the heavy-job lock.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from export.heavy import heavy_job, peak_rss_gb
from tools import research_testset as R
from tools import research_winprob_elo as E
from tools.frozen_common import (REPO_ROOT, MemoryGuard, _plain, _sha256, artifacts_dir, assert_safe_output)
from tools.research_testset import dev_games, refuse_test, score_once, scoring_data_root

WHO = "tools.research_winprob"
EXPERIMENT = "winprob-E1"
RESULTS = REPO_ROOT / "reports" / "research"
SEED, BOOT, FOLDS = 0, 2000, 5
THRESHOLD = 0.002               # pre-registered: mean bucket Brier improves by at least this
GATE_TOLERANCE = 1e-6           # stored p_home_win is float32; a refit of the same fit differs by float32 rounding
FULL_GAME_S = 2880.0
NEW = ("mu_hat", "mu_drift")
ARM_NAMES = ("baseline", "e1", "ablation")
PAIRS = (("e1", "baseline"), ("ablation", "baseline"), ("e1", "ablation"))
TEST_GAMES_EXPECTED = 1316      # the game count of the fixed 2025-26 source (PLAN.md D89), for the power note only
STATES_FILE = "winprob_states_dev.parquet"
SPLIT_SEASON = "2024-25"        # the frozen split's held-out season; absent from the development frame


def _F():
    from analytics.winprob import features as F
    return F


def _T():
    from analytics.winprob import train as T
    return T


def arm_columns() -> dict:
    T = _T()
    base = tuple(T.LOGISTIC_COLS)
    return {"baseline": base, "e1": base + NEW, "ablation": base + ("mu_hat",)}


def research_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research")


def data_root() -> Path:
    root = os.environ.get("DATA_ROOT")
    if not root:
        raise SystemExit(f"{WHO}: DATA_ROOT is not set")
    return Path(root)


# ----------------------------------------------------------------------------- features

def build_dev_states(artifacts=None) -> dict:
    """The frozen state frame of the development seasons, through the frozen split and the door -> a record."""
    F, T = _F(), _T()
    started = time.time()
    guard = MemoryGuard()
    d = F.load_data(data_root(), seasons=R.PBP_DEV_SEASONS)
    pbp, clock = F.clean_playbyplay(d["playbyplay"])
    states = F.build_states(pbp, d["games"], None)
    del d, pbp
    guard.check("states built")
    train, held, ignored = T.split_seasons(states, SPLIT_SEASON)
    if len(held) or ignored:
        raise SystemExit(f"{WHO}: rows outside the development seasons reached the state frame")
    ids = sorted(train["game_id"].astype(str).unique())
    if dev_games(ids, WHO, R.PBP) != ids:
        raise SystemExit(f"{WHO}: a game of the state frame is not a development game")
    out = research_dir(artifacts) / STATES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out, index=False)
    record = {"file": out.name, "rows": int(len(train)), "games": len(ids),
              "seasons": sorted(train["season"].astype(str).unique()),
              "clock_rule": {k: clock[k] for k in ("rows_in", "rows_dropped", "games_in", "games_dropped")},
              "possession_known": float(train["possession_home"].notna().mean()),
              "sha256": _sha256(out), "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}
    (out.parent / "winprob_states_dev.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_dev_states(artifacts=None) -> pd.DataFrame:
    folder = research_dir(artifacts)
    record = json.loads((folder / "winprob_states_dev.json").read_text())
    path = folder / STATES_FILE
    if _sha256(path) != record["sha256"]:
        raise SystemExit(f"{WHO}: the cached state frame changed after it was built; run e1-features again")
    states = pd.read_parquet(path)
    ids = sorted(states["game_id"].astype(str).unique())
    if dev_games(ids, WHO, R.PBP) != ids or states["heldout"].astype(bool).any():
        raise SystemExit(f"{WHO}: the cached state frame holds a game outside development")
    return states


def attach_mu(states: pd.DataFrame, pregame: pd.DataFrame) -> pd.DataFrame:
    """mu_hat of each row's game (its pregame Elo value) and the drift term mu_hat * sqrt(seconds left / 2880)."""
    mu = pd.Series(pregame["mu_hat"].to_numpy(dtype=float), index=pregame["game_id"].astype(str))
    if mu.index.duplicated().any():
        raise SystemExit(f"{WHO}: a game has two pregame Elo values")
    out = states.copy()
    m = out["game_id"].astype(str).map(mu)
    if m.isna().any():
        missing = sorted(set(out.loc[m.isna(), "game_id"].astype(str)))
        raise SystemExit(f"{WHO}: {len(missing)} game(s) of the play-by-play have no schedule row for Elo")
    out["mu_hat"] = m.to_numpy(dtype=float)
    sec = np.maximum(out["seconds_remaining"].astype(float).to_numpy(), 0.0)
    out["mu_drift"] = out["mu_hat"].to_numpy() * np.sqrt(sec / FULL_GAME_S)
    return out


def build_with_mu(schedule: pd.DataFrame, params: dict):
    """A build function for the frozen prefix_invariance_violations: the frozen build_states plus mu_hat and
    mu_drift, with Elo run over the development schedule after the finals of the given games table replace the
    schedule's (so the check's scrambled finals reach Elo, and a feature that read a game's own result would move)."""
    F = _F()

    def build(pbp, games, odds=None, **kw):
        states = F.build_states(pbp, games, odds, **kw)
        s = schedule.copy()
        fin = games.assign(game_id=games["game_id"].astype(str)).drop_duplicates("game_id").set_index("game_id")
        hit = s["game_id"].isin(fin.index)
        s.loc[hit, "home_score"] = s.loc[hit, "game_id"].map(fin["home_score"]).astype(float).to_numpy()
        s.loc[hit, "away_score"] = s.loc[hit, "game_id"].map(fin["away_score"]).astype(float).to_numpy()
        return attach_mu(states, E.advance(E.Elo(**params), s))
    return build


def frozen_prefix_check(pbp, games, build, n_games: int = 5, cuts: int = 3, seed: int = SEED) -> list:
    """The frozen prefix_invariance_violations, unchanged, with its column list extended by the new columns (the
    frozen function compares feature_columns(); the new ones are added for the duration of the call)."""
    F = _F()
    cols = list(F.FEATURES) + list(NEW)
    with mock.patch.object(F, "feature_columns", lambda: list(cols)):
        return F.prefix_invariance_violations(pbp, games, None, build=build, n_games=n_games, cuts=cuts, seed=seed)


def dev_prefix_probe(schedule, params, season: str = "2022-23", n: int = 60, seed: int = SEED) -> dict:
    F = _F()
    d = F.load_data(data_root(), seasons=(season,))
    gids = sorted(d["playbyplay"]["game_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    pick = sorted(gids[i] for i in rng.choice(len(gids), size=min(n, len(gids)), replace=False))
    refuse_test(pick, WHO, R.PBP)
    pbp = d["playbyplay"][d["playbyplay"]["game_id"].astype(str).isin(set(pick))].reset_index(drop=True)
    del d
    games = F.games_from_playbyplay(pbp)
    viol = frozen_prefix_check(pbp, games, build_with_mu(schedule, params))
    return {"season": season, "games_in_frame": len(pick), "probed_games": 5, "cuts_per_game": 3,
            "columns": list(F.FEATURES) + list(NEW), "violations": [list(map(str, v)) for v in viol]}


# ----------------------------------------------------------------------------- arms

def _fit_logistic(train: pd.DataFrame, cols):
    """The frozen trainer's logistic recipe (analytics.winprob.train.fit_logistic) on the given columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from analytics.eval import fit_threads
    F = _F()
    X = F.design(train, cols)
    y = train[F.TARGET].astype(int).to_numpy()
    with fit_threads():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X.to_numpy(dtype=float), y)
    info = {"columns": list(X.columns), "coef_standardised": [float(c) for c in m[-1].coef_[0]],
            "intercept": float(m[-1].intercept_[0]), "iterations": int(m[-1].n_iter_[0])}
    return (lambda frame, m=m, cols=tuple(cols): m.predict_proba(F.design(frame, cols).to_numpy(dtype=float))[:, 1]), info


def fit_arms(train: pd.DataFrame) -> tuple:
    """The three arms on development rows only -> ({arm: predict(frame) -> p}, {arm: fit record})."""
    T = _T()
    if train["heldout"].astype(bool).any():
        raise SystemExit(f"{WHO}: a held-out row reached the fit")
    ids = sorted(train["game_id"].astype(str).unique())
    if dev_games(ids, WHO, R.PBP) != ids:
        raise SystemExit(f"{WHO}: a game outside development reached the fit")
    cols = arm_columns()
    models = {"baseline": T.fit_models(train, SEED, models=("logistic",))["logistic"]}
    info = {"baseline": {"columns": list(cols["baseline"]), "fit": "analytics.winprob.train.fit_models (frozen)"}}
    for arm in ("e1", "ablation"):
        models[arm], info[arm] = _fit_logistic(train, cols[arm])
    for arm in info:
        info[arm]["rows"], info[arm]["games"] = int(len(train)), len(ids)
    return models, info


# ----------------------------------------------------------------------------- one season, scored

# The only files score_season reads under a data root (relative paths; at test time both are pinned files of the
# test source, hashed by score_once before the scorer runs; tests/test_research_winprob.py checks that).
READ_FILES = ("collect/stints/playbyplay/{season}.parquet", "collect/bref_schedule/games/{season}.parquet")


def read_schedule(root, season: str) -> pd.DataFrame:
    path = Path(root) / READ_FILES[1].format(season=season)
    s = E.schedule_frame(pd.read_parquet(path, columns=list(E.COLUMNS)))
    if set(s["season"]) != {season}:
        raise SystemExit(f"{WHO}: the {season} schedule holds games of {sorted(set(s['season']))}")
    return s


def schedule_agreement(pbp_games: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Report only: for each listed game with play-by-play, does the whole-game final margin of the play-by-play
    (its last score) equal the schedule's final score margin? -> counts and the share that agree."""
    s = schedule.set_index("game_id")
    g = pbp_games.set_index("game_id")
    both = g.index.intersection(s.index)
    a = (g.loc[both, "home_score"].astype(float) - g.loc[both, "away_score"].astype(float)).to_numpy()
    b = (s.loc[both, "home_score"].astype(float) - s.loc[both, "away_score"].astype(float)).to_numpy()
    agree = int((a == b).sum())
    return {"games_compared": int(len(both)), "games_agree": agree, "games_disagree": int(len(both) - agree),
            "games_without_a_schedule_row": int(len(g.index.difference(s.index))),
            "share_agree": float(agree / len(both)) if len(both) else float("nan")}


def score_season(root, season: str, allowed, elo_start, models: dict) -> tuple:
    """Build the frozen state frame of one season from a data root, run Elo game by game through the season's
    schedule from elo_start (a copy), attach mu_hat, and predict every arm -> (rows, pregame, info, states). The
    same code scores a development season (from DATA_ROOT) and the test season (from the scorer's root)."""
    F = _F()
    d = F.load_data(root, seasons=(season,))
    read = sorted(str(Path(f).resolve().relative_to(Path(root).resolve())) for f in d["files"])
    allowed_files = [f.format(season=season) for f in READ_FILES]
    if read != allowed_files[:1]:
        raise SystemExit(f"{WHO}: load_data read {read}, not only the play-by-play file of {season}")
    in_pbp = set(d["playbyplay"]["game_id"].astype(str))
    pbp, clock = F.clean_playbyplay(d["playbyplay"])
    states = F.build_states(pbp, d["games"], None)
    pbp_finals = d["games"].assign(game_id=d["games"]["game_id"].astype(str)).drop_duplicates("game_id")
    del d, pbp
    allowed = {str(g) for g in allowed}
    built = set(states["game_id"].astype(str))
    if built - allowed:
        raise SystemExit(f"{WHO}: {len(built - allowed)} game(s) of the {season} frame are not allowed here")
    schedule = read_schedule(root, season)
    agreement = schedule_agreement(pbp_finals[pbp_finals["game_id"].isin(allowed)], schedule)
    elo_violations = E.prefix_violations(elo_start, schedule)
    pregame = E.advance(elo_start.copy(), schedule)
    states = attach_mu(states, pregame)
    rows = pd.DataFrame({"game_id": states["game_id"].astype(str).to_numpy(),
                         "time_bucket": states["time_bucket"].astype(str).to_numpy(),
                         "home_win": states[F.TARGET].astype(int).to_numpy()})
    for arm in ARM_NAMES:
        rows["p_" + arm] = models[arm](states)
    pre = pregame[pregame["game_id"].isin(built)].reset_index(drop=True)
    p_cols = rows[["p_" + a for a in ARM_NAMES]].to_numpy(dtype=float)
    info = {"season": season, "rows": int(len(rows)), "games_scored": len(built),
            "files_read": read + [allowed_files[1]],
            "games_dropped_by_builders_ids": sorted((allowed & in_pbp) - built),
            "games_without_playbyplay_ids": sorted(allowed - in_pbp),
            "clock_rule_by_cause": clock["by_cause"],
            "counts": {"games_listed": len(allowed), "games_without_playbyplay": len(allowed - in_pbp),
                       "games_dropped_by_builders": len((allowed & in_pbp) - built),
                       "games_scored": len(built & allowed),
                       "rows_with_a_non_finite_prediction": int((~np.isfinite(p_cols)).any(axis=1).sum()),
                       "rows_with_a_non_finite_mu": int((~np.isfinite(states[list(NEW)].to_numpy(dtype=float))).any(axis=1).sum()),
                       "games_with_a_non_finite_pregame_value": int((~np.isfinite(pre["mu_hat"].to_numpy(dtype=float))).sum())},
            "schedule_agreement": agreement,
            "clock_rule": {k: clock[k] for k in ("rows_in", "rows_dropped", "games_in", "games_dropped")},
            "schedule_games": int(len(schedule)), "schedule_games_without_playbyplay": int(len(set(schedule["game_id"]) - built)),
            "possession_known": float(states["possession_home"].notna().mean()),
            "elo_prefix_violations": elo_violations, "elo_ratings_before_season": len(elo_start.ratings)}
    return rows, pre, info, states


# ----------------------------------------------------------------------------- metrics

def _boot_weights(n_games: int, n: int = BOOT, seed: int = SEED) -> np.ndarray:
    """How often each game is drawn in each of n resamples of whole games (n x games)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, size=(n, n_games))
    return np.stack([np.bincount(r, minlength=n_games) for r in idx]).astype(float)


def _ci(point, boots) -> dict:
    boots = np.asarray(boots, dtype=float)
    if not np.isfinite(boots).any():
        return {"point": float(point), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def summarize(rows: pd.DataFrame, pregame: pd.DataFrame | None = None, home_rate: float | None = None,
              n_boot: int = BOOT, seed: int = SEED) -> dict:
    """Every arm's bucket Brier, mean bucket Brier, overall Brier and log loss with 95 percent percentile intervals
    from one bootstrap of whole games (n_boot resamples, seed), the paired differences on the same resamples, per
    bucket ECE and calibration slope as point estimates, and the pregame Elo Brier of the scored games."""
    from analytics.eval import EPS, calibration_slope_fast, ece
    F = _F()
    names = list(F.BUCKET_NAMES)
    reg = [b[0] for b in F.BUCKETS]
    codes, uniq = pd.factorize(rows["game_id"].astype(str), sort=True)
    G, B = len(uniq), len(names)
    bidx = pd.Categorical(rows["time_bucket"].astype(str), categories=names).codes.astype(int)
    if (bidx < 0).any():
        raise SystemExit(f"{WHO}: a row has an unknown time bucket")
    cell = codes * B + bidx
    y = rows["home_win"].astype(float).to_numpy()
    n_gb = np.bincount(cell, minlength=G * B).reshape(G, B).astype(float)
    cnt = n_gb.sum(axis=1)
    W = _boot_weights(G, n_boot, seed)
    Wn = W @ n_gb
    Wc = W @ cnt
    point, boot, out = {}, {}, {"rows": int(len(rows)), "games": int(G), "resamples": n_boot, "seed": seed,
                                "rows_by_bucket": {b: int(n_gb[:, i].sum()) for i, b in enumerate(names)},
                                "arms": {}, "paired": {}}
    with np.errstate(divide="ignore", invalid="ignore"):
        for arm in ARM_NAMES:
            p = np.clip(rows["p_" + arm].astype(float).to_numpy(), EPS, 1 - EPS)
            se = np.bincount(cell, weights=(p - y) ** 2, minlength=G * B).reshape(G, B)
            row_ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
            ll = np.bincount(codes, weights=row_ll, minlength=G)
            llb = np.bincount(cell, weights=row_ll, minlength=G * B).reshape(G, B)
            pb = se.sum(axis=0) / n_gb.sum(axis=0)
            bb = (W @ se) / Wn
            point[arm] = {"buckets": pb, "mean": float(np.mean(pb[: len(reg)])), "brier": float(se.sum() / cnt.sum()),
                          "log_loss": float(ll.sum() / cnt.sum()), "bucket_ll": llb.sum(axis=0) / n_gb.sum(axis=0)}
            boot[arm] = {"buckets": bb, "mean": bb[:, : len(reg)].mean(axis=1), "brier": (W @ se.sum(axis=1)) / Wc,
                         "log_loss": (W @ ll) / Wc, "bucket_ll": (W @ llb) / Wn}
            a = {"mean_bucket_brier": _ci(point[arm]["mean"], boot[arm]["mean"]),
                 "bucket_brier": {b: _ci(pb[i], bb[:, i]) for i, b in enumerate(names)},
                 "brier": _ci(point[arm]["brier"], boot[arm]["brier"]),
                 "log_loss": _ci(point[arm]["log_loss"], boot[arm]["log_loss"]),
                 "bucket_log_loss": {b: _ci(point[arm]["bucket_ll"][i], boot[arm]["bucket_ll"][:, i])
                                     for i, b in enumerate(names)},
                 "ece_10_bins_point_estimate": {}, "calibration_slope_point_estimate": {}}
            for i, b in enumerate(names):
                m = bidx == i
                if m.sum():
                    a["ece_10_bins_point_estimate"][b] = float(ece(y[m], p[m], 10))
                    a["calibration_slope_point_estimate"][b] = float(calibration_slope_fast(y[m], p[m]))
            out["arms"][arm] = a
        for x, z in PAIRS:
            d = {"mean_bucket_brier": _ci(point[x]["mean"] - point[z]["mean"], boot[x]["mean"] - boot[z]["mean"]),
                 "bucket_brier": {b: _ci(point[x]["buckets"][i] - point[z]["buckets"][i],
                                         boot[x]["buckets"][:, i] - boot[z]["buckets"][:, i]) for i, b in enumerate(names)},
                 "brier": _ci(point[x]["brier"] - point[z]["brier"], boot[x]["brier"] - boot[z]["brier"]),
                 "log_loss": _ci(point[x]["log_loss"] - point[z]["log_loss"], boot[x]["log_loss"] - boot[z]["log_loss"]),
                 "bucket_log_loss_reported_no_verdict": {
                     b: _ci(point[x]["bucket_ll"][i] - point[z]["bucket_ll"][i],
                            boot[x]["bucket_ll"][:, i] - boot[z]["bucket_ll"][:, i]) for i, b in enumerate(names)}}
            out["paired"][f"{x}_minus_{z}"] = d
    if pregame is not None:
        pre = pregame.drop_duplicates("game_id")
        pre = pre.set_axis(pre["game_id"].astype(str).to_numpy(), axis=0).reindex(uniq)
        if pre["p_home"].isna().any():
            raise SystemExit(f"{WHO}: a scored game has no pregame Elo value")
        game_y = pd.Series(y).groupby(codes).first().to_numpy()
        e = (pre["p_home"].to_numpy(dtype=float) - game_y) ** 2
        res = {"games": int(G), "elo_brier": _ci(e.mean(), (W @ e) / G),
               "outcome_mismatches_schedule_vs_playbyplay": int((pre["home_win"].to_numpy() != game_y).sum()),
               "mu_hat_sd_points": float(pre["mu_hat"].std())}
        if home_rate is not None:
            c = (home_rate - game_y) ** 2
            res["constant_brier"] = _ci(c.mean(), (W @ c) / G)
            res["constant_home_rate"] = float(home_rate)
        out["pregame"] = res
    bad = int((~np.isfinite(rows[["p_" + a for a in ARM_NAMES]].to_numpy(dtype=float))).sum())
    if pregame is not None:
        bad += int((~np.isfinite(pregame["mu_hat"].to_numpy(dtype=float))).sum())
    out["non_finite_values"] = bad
    if bad:
        out["verdict"] = {"refused": True, "reading": f"refused: {bad} non-finite value(s) reached the verdict"}
    else:
        out["verdict"] = verdict(out["paired"], reg)
    return out


def _reading(size: bool, below: bool, worse: list, lo: float) -> str:
    if size and below and not worse:
        return "pass"
    if worse:
        return f"fail: bucket(s) worse beyond the interval: {', '.join(worse)}"
    if lo > 0:
        return "fail: e1 is worse than the baseline"
    if not below:
        return "fail: the interval of the difference includes zero"
    return "fail: the gain is real (interval below zero) but smaller than the threshold"


def verdict(paired: dict, regulation=None) -> dict:
    """The pre-registered rule, read on e1 minus baseline: adopt when the mean bucket Brier difference is at most
    -THRESHOLD, its 95 percent interval lies below zero, and no bucket's interval lies above zero. "No bucket" is read
    as written (PLAN.md D91): every bucket, overtime included. The regulation-only reading is reported beside it with
    whether it would differ."""
    reg = list(regulation or [b[0] for b in _F().BUCKETS])
    d = paired["e1_minus_baseline"]
    mb = d["mean_bucket_brier"]
    worse = [b for b in d["bucket_brier"] if d["bucket_brier"][b]["lo"] > 0]
    worse_reg = [b for b in worse if b in reg]
    size = mb["point"] <= -THRESHOLD
    below = mb["hi"] < 0
    adopt = bool(size and below and not worse)
    reading = _reading(size, below, worse, mb["lo"])
    reg_reading = _reading(size, below, worse_reg, mb["lo"])
    ab = paired["e1_minus_ablation"]["mean_bucket_brier"]
    return {"threshold": THRESHOLD, "difference": mb["point"], "lo": mb["lo"], "hi": mb["hi"],
            "size_at_least_threshold": bool(size), "interval_excludes_zero": bool(below),
            "interval_straddles_threshold": bool(mb["lo"] < -THRESHOLD < mb["hi"]),
            "bucket_guard": "every bucket, overtime included (PLAN.md D91)", "buckets_worse": worse,
            "adopt": adopt, "reading": reading,
            "regulation_only_reading": {"buckets_worse": worse_reg, "adopt": bool(size and below and not worse_reg),
                                        "reading": reg_reading, "differs": bool(reg_reading != reading)},
            "ablation": {"e1_minus_ablation": ab["point"], "lo": ab["lo"], "hi": ab["hi"],
                         "drift_form_adds_beyond_mu_alone": bool(ab["hi"] < 0)}}


def power_note(summary: dict, n_test: int = TEST_GAMES_EXPECTED) -> dict:
    """The development half-width of the e1 minus baseline interval scaled to the test season's game count."""
    mb = summary["paired"]["e1_minus_baseline"]["mean_bucket_brier"]
    half = (mb["hi"] - mb["lo"]) / 2
    return {"development_games": summary["games"], "development_half_width": half,
            "test_games": n_test, "scaled_half_width_at_test_game_count": half * float(np.sqrt(summary["games"] / n_test))}


# ----------------------------------------------------------------------------- gate

def gate(artifacts=None) -> dict:
    from analytics.eval import grouped_folds
    T = _T()
    folder = research_dir(artifacts) / "winprob_e1_gate"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "check.json").write_text(json.dumps({"passed": False, "status": "gate started"}) + "\n")
    train = load_dev_states(artifacts)
    stored = []
    for season in R.PBP_DEV_SEASONS:
        path = data_root() / "analytics" / "winprob" / season / "winprob_states.parquet"
        s = pd.read_parquet(path, columns=["game_id", "event", "p_home_win", "fold"])
        stored.append(s[s["fold"] >= 0])
    stored = pd.concat(stored, ignore_index=True)
    stored["game_id"] = stored["game_id"].astype(str)
    g = train["game_id"].astype(str)
    p = np.full(len(train), np.nan)
    fold = np.full(len(train), -1)
    started = time.time()
    for i, (tr, te) in enumerate(grouped_folds(sorted(g.unique()), n_folds=FOLDS, seed=SEED)):
        mtr, mte = g.isin(tr).to_numpy(), g.isin(te).to_numpy()
        model = T.fit_models(train[mtr], SEED, models=("logistic",))["logistic"]
        p[mte] = model(train[mte])
        fold[mte] = i
        print(f"[gate] fold {i}: {int(mte.sum())} rows, {time.time() - started:.0f} s", flush=True)
    mine = pd.DataFrame({"game_id": g.to_numpy(), "event": train["event"].astype(int).to_numpy(), "p": p, "fold": fold})
    stored = stored.rename(columns={"fold": "stored_fold"}).assign(event=stored["event"].astype(int))
    joined = mine.merge(stored, on=["game_id", "event"], how="inner", validate="one_to_one")
    diff = float(np.max(np.abs(joined["p"].to_numpy() - joined["p_home_win"].astype(float).to_numpy())))
    folds = int((joined["fold"].to_numpy() != joined["stored_fold"].to_numpy()).sum())
    mine[["p", "fold"]].to_parquet(folder / "gate_oof.parquet", index=False)
    n, expected = int(len(joined)), int(len(stored))
    record = {"name": "winprob", "compared": ("the frozen logistic refit by tools.research_winprob on the development "
              "seasons under the frozen 5 grouped folds (seed 0) against the stored out-of-fold p_home_win of the "
              "training seasons (fold >= 0); no held-out row, no metric"),
              "n": n, "n_expected": expected, "rows_in_frame": int(len(train)), "max_abs_diff": diff,
              "tolerance": GATE_TOLERANCE, "fold_mismatches": folds,
              "passed": bool(n == expected == len(train) and folds == 0 and diff <= GATE_TOLERANCE),
              "files": {"gate_oof.parquet": _sha256(folder / "gate_oof.parquet")},
              "seconds": round(time.time() - started, 1)}
    (folder / "check.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["passed"]:
        raise SystemExit(f"{WHO}: the gate did not pass: {record}")
    return record


def require_gate(artifacts=None) -> dict:
    folder = research_dir(artifacts) / "winprob_e1_gate"
    record = json.loads((folder / "check.json").read_text()) if (folder / "check.json").exists() else {}
    if not record.get("passed"):
        raise SystemExit(f"{WHO}: run e1-gate first; the baseline must reproduce the frozen record")
    for name, digest in record["files"].items():
        if _sha256(folder / name) != digest:
            raise SystemExit(f"{WHO}: {name} changed after the gate was written")
    return {k: v for k, v in record.items() if k != "files"}


# ----------------------------------------------------------------------------- development

def _home_rate(train: pd.DataFrame) -> float:
    F = _F()
    return float(train.drop_duplicates("game_id")[F.TARGET].astype(float).mean())


def dev(artifacts=None) -> dict:
    from analytics.eval import grouped_folds
    gate_record = require_gate(artifacts)
    guard = MemoryGuard()
    root = data_root()
    schedule = E.dev_schedule(root)
    choice = E.choose_params(schedule)
    params = choice["chosen"]
    elo_viol = E.prefix_violations(E.Elo(**params), schedule)
    probe = dev_prefix_probe(schedule, params)
    if elo_viol or probe["violations"]:
        raise SystemExit(f"{WHO}: a new feature changed when later results were cut: {elo_viol[:3]} {probe['violations'][:3]}")
    states = attach_mu(load_dev_states(artifacts), E.advance(E.Elo(**params), schedule))
    guard.check("dev states")
    rows_all, pre_all, seasons, fits = [], [], {}, {}
    for season in E.CHECK_SEASONS:
        train = states[states["season"].astype(str) < season]
        models, info = fit_arms(train)
        allowed = sorted(states.loc[states["season"].astype(str) == season, "game_id"].astype(str).unique())
        rows, pre, sinfo, rebuilt = score_season(root, season, allowed, E.state_after(params, schedule, season), models)
        cached = states[states["season"].astype(str) == season].reset_index(drop=True)
        same = {c: float(np.nanmax(np.abs(rebuilt[c].astype(float).to_numpy() - cached[c].astype(float).to_numpy())))
                for c in ("margin", "seconds_remaining", "mu_hat", "mu_drift")} if len(rebuilt) == len(cached) else None
        if same is None or any(v != 0 for v in same.values()) or sinfo["elo_prefix_violations"]:
            raise SystemExit(f"{WHO}: the season path disagrees with the cached frame for {season}: {same}")
        sinfo["rebuilt_equals_cached"] = True
        hr = _home_rate(train)
        seasons[season] = {"info": sinfo, "home_rate_of_training_games": hr,
                           "summary": summarize(rows, pre, hr)}
        fits[season] = info
        rows_all.append(rows.assign(fold_season=season))
        pre_all.append(pre)
        print(f"[dev] {season}: {sinfo['games_scored']} games scored", flush=True)
        guard.check(season)
    rows = pd.concat(rows_all, ignore_index=True)
    pooled = summarize(rows, pd.concat(pre_all, ignore_index=True), None)
    # secondary: the frozen protocol, 5 grouped folds inside the development seasons (not the forward test)
    gate_oof = pd.read_parquet(research_dir(artifacts) / "winprob_e1_gate" / "gate_oof.parquet")
    g = states["game_id"].astype(str)
    sec_rows = pd.DataFrame({"game_id": g.to_numpy(), "time_bucket": states["time_bucket"].astype(str).to_numpy(),
                             "home_win": states["home_win"].astype(int).to_numpy(),
                             "p_baseline": gate_oof["p"].to_numpy()})
    cols = arm_columns()
    for arm in ("e1", "ablation"):
        sec_rows["p_" + arm] = np.nan
    for i, (tr, te) in enumerate(grouped_folds(sorted(g.unique()), n_folds=FOLDS, seed=SEED)):
        mtr, mte = g.isin(tr).to_numpy(), g.isin(te).to_numpy()
        for arm in ("e1", "ablation"):
            model, _ = _fit_logistic(states[mtr], cols[arm])
            sec_rows.loc[mte, "p_" + arm] = model(states[mte])
        print(f"[dev] grouped fold {i}", flush=True)
    secondary = summarize(sec_rows)
    out = {"experiment": EXPERIMENT,
           "stage": ("development seasons 2020-21 to 2023-24; primary: forward chaining by season (fit on the seasons "
                     "before 2021-22, 2022-23 and 2023-24, score that season); paired bootstrap by game, 2000 resamples, "
                     "seed 0"),
           "pre_registration": "docs/research/winprob.md section 6, winprob-E1",
           "gate": gate_record,
           "elo": {k: v for k, v in choice.items()} | {"prefix_violations": elo_viol,
                                                        "points_per_elo": E.POINTS_PER_ELO, "mean": E.MEAN},
           "frozen_prefix_check": probe,
           "fits_by_season": fits, "by_season": seasons, "pooled_forward": pooled,
           "power": power_note(pooled),
           "secondary_grouped_folds": {"what": ("the frozen protocol: 5 grouped folds by game inside the development "
                                                "seasons (seed 0), mu_hat from the forward Elo chain; within-season, "
                                                "not the forward test; reported beside the primary, no verdict"),
                                       "summary": secondary},
           "peak_gb": round(peak_rss_gb(), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "winprob-E1-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


# ----------------------------------------------------------------------------- the one final score

def clean_head() -> str:
    """HEAD, after checking that the whole working tree is committed (no change, no untracked file), so the score
    belongs to one commit of the code, the frozen code and the development record."""
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def final(artifacts=None) -> dict:
    head = clean_head()
    gate_record = require_gate(artifacts)
    dev_path = RESULTS / "winprob-E1-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test season is scored after the development result exists")
    dev_out = json.loads(dev_path.read_text())
    root = data_root()
    schedule = E.dev_schedule(root)
    choice = E.choose_params(schedule)
    params = choice["chosen"]
    if params != dev_out["elo"]["chosen"]:
        raise SystemExit(f"{WHO}: the Elo choice differs from the development record")
    states = attach_mu(load_dev_states(artifacts), E.advance(E.Elo(**params), schedule))
    states_sha = json.loads((research_dir(artifacts) / "winprob_states_dev.json").read_text())["sha256"]
    gate_files = json.loads((research_dir(artifacts) / "winprob_e1_gate" / "check.json").read_text())["files"]
    models, fit_info = fit_arms(states)
    hr = _home_rate(states)
    elo, state_record = E.state_through_2024_25(params, schedule, root)
    del states

    def scorer(test_games):
        troot = scoring_data_root(R.PBP)
        rows, pre, info, _ = score_season(troot, R.PBP_TEST_SEASON, test_games, elo, models)
        results = summarize(rows, pre, hr)
        if info["elo_prefix_violations"]:
            results["verdict"] = {"refused": True, "computed_but_not_read": results["verdict"],
                                  "reading": (f"refused: the test-season Elo prefix check found "
                                              f"{len(info['elo_prefix_violations'])} violation(s)")}
        return {"experiment": EXPERIMENT,
                "stage": ("the three arms fitted on every development row (2020-21 to 2023-24), Elo carried through "
                          "2024-25 as state (D89) and updated game by game through the test season; scored once"),
                "code_commit": head, "dev_states_sha256": states_sha, "gate_files_sha256": gate_files,
                "test_season": info, "results": results}

    out = score_once(EXPERIMENT, R.PBP, scorer)
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate": gate_record, "elo_params": params, "elo_state": state_record,
                        "home_rate_of_training_games": hr, "fits": fit_info}}
    (RESULTS / "winprob-E1-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-features", "e1-gate", "e1-dev", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"e1-features": build_dev_states, "e1-gate": gate, "e1-dev": dev, "e1-final": final}[a.command]
    with heavy_job(f"research winprob {a.command}"):
        out = run(a.artifacts)
    show = out.get("pooled_forward", out).get("verdict", out) if isinstance(out, dict) else out
    print(json.dumps(_plain(show) if isinstance(show, dict) else show, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
