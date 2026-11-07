"""analytics/lineups/train.py: lineup synergy under the season-split protocol (baselines first, no deep model).

    python -m analytics.lineups.train --fixtures | --data DATA_ROOT --report reports/lineups.md [--out OUT_ROOT]
        [--seed 0] [--bootstrap 200] [--heldout-season 2024-25] [--gate 300] [--min-poss 100] [--folds 5]

Unit: the stint (analytics.lineups.features). Target: points per 100 possessions differential, home minus
away, possessions from duration at 28.8 s each when the table has no count.

Candidates (every fit goes through fit_candidates, the one place a test can watch)
    plus_minus     a lineup's raw net rating pooled over the fitting seasons: sum of its signed point
                   differentials over its possessions, times 100 (from both sides of the floor)
    rapm           weighted ridge regression of the stint differential on player indicators (+1 home, -1 away)
                   with an unpenalised home-court intercept, weights = possessions, alpha chosen by grouped
                   cross-validation by game (weighted out-of-fold MSE over ALPHAS) within the fitting seasons;
                   a lineup's rating is the sum of its five coefficients (an unseen player counts 0)
    rapm_synergy   rapm plus one indicator per five-man lineup with at least --gate possessions in the fitting
                   seasons (analytics.eval.min_sample_gate on possessions, the same n_min as eval.gated),
                   +1 when it is the home lineup, -1 away, ridge-penalised with the same alpha (re-chosen by
                   the same grouped CV); the lineup coefficient is the synergy and the rating is the player
                   sum plus the synergy
Selection: for every training season after the first, the candidates are fitted on the seasons before it and
predict the raw net rating of its lineups present in both (at least --min-poss possessions on each side);
Spearman and MSE against the raw next-season rating; every interval of the selection table and of the held-out
table is over lineups (analytics.eval.bootstrap_by_game with lineup_id as the group, lineups resampled with
replacement), as is the paired bootstrap against plus_minus. The lo and hi of the output tables are different
intervals: over games (lineup_ratings) and over game-resampled refits (player_rapm).
The candidate with the highest mean Spearman over the validation pairs is selected (ties go to plus_minus).
The held-out season is scored ONCE, after selection, by the SELECTED candidate refitted on every training season;
plus_minus is reported beside it as a labelled floor and the other candidates are never scored on it (no paired
test on the held-out season either).

Leakage: no row of the target season or of any later season reaches a fit (assert_season_split, in code,
on every fit); the held-out season never reaches a fit; analytics.eval.refuse_heldout checks the training game
ids; analytics.schema.assert_time_forward runs on the feature frame; rows flagged heldout are dropped by
validate. Seasons later than --heldout-season are ignored (a season is never predicted from a later one).

Outputs (docs/contract/lineups.md), OUT_ROOT/analytics/lineups/<season>/
    lineup_ratings.parquet   season, lineup_id, team_id, n_possessions, plus_minus_per100 (that season),
                             rapm_sum_per100, synergy_per100 (the fit on the seasons up to and including it,
                             in-sample descriptive ratings; for the held-out season the training fit, out of
                             sample), lo, hi (bootstrap by game of plus_minus_per100, NaN under two games),
                             insufficient (n_possessions under --gate), fit_seasons
    player_rapm.parquet      season, player_id, rapm, lo, hi (bootstrap by game refits), team_id (the player's main
                             team in the output season's lineups), as_of (the last game date of the fit seasons:
                             the rating is known from that date on), in_sample, fit_seasons
    player_rapm_prior.parquet  for every training season after the first: the same columns from the fit on the
                             seasons BEFORE it (the validation-pair fit), in_sample false, as_of the last date of
                             the previous season: the dated table a consumer (betting) may use for that season
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from analytics import schema
from analytics.eval import (bootstrap_by_game, fit_threads, fmt_ci, fmt_num, grouped_folds, min_sample_gate, paired_bootstrap_by_game,
                           refuse_heldout)
from analytics.lineups import FIXTURE_SEASONS, SEASONS, WHO, add_fixture_args, load_inputs, out_dir
from analytics.lineups.features import TARGET, assert_no_leak, feature_columns, features
from analytics.lineups.validate import validate

CANDIDATES = ("plus_minus", "rapm", "rapm_synergy")
ALPHAS = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)
GATE_POSS = 300
MIN_POSS_EVAL = 100
MAC_CMD = ("python -m analytics.lineups.train --data \"$DATA_ROOT\" --report reports/lineups.md --out \"$DATA_ROOT\" "
           "--seed 0 --bootstrap 1000 --heldout-season 2024-25")


# ── season split ───────────────────────────────────────────────────────────────────────────────────
def season_key(s) -> int:
    """The starting year of a season string, for ordering."""
    return int(str(s)[:4])


def assert_season_split(frame: pd.DataFrame, target_season: str | None, heldout_season: str | None) -> None:
    """ValueError when a fitting frame holds the target season, any later season, or the held-out season."""
    seasons = sorted(map(str, frame["season"].unique()))
    if heldout_season is not None and str(heldout_season) in seasons:
        raise ValueError("%s: held-out season %s reached a fit" % (WHO, heldout_season))
    if target_season is not None:
        late = [s for s in seasons if season_key(s) >= season_key(target_season)]
        if late:
            raise ValueError("%s: season(s) %s are not before the target season %s but reached a fit" % (WHO, late, target_season))


# ── lineup tables ──────────────────────────────────────────────────────────────────────────────────
def lineup_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Both sides of every stint from the lineup's point of view: lineup_id, team_id, game_id, possessions, diff."""
    h = pd.DataFrame({"lineup_id": feats["lineup_home"], "team_id": feats["home_team_id"], "game_id": feats["game_id"],
                      "season": feats["season"], "possessions": feats["possessions"], "diff": feats["points_diff"]})
    a = pd.DataFrame({"lineup_id": feats["lineup_away"], "team_id": feats["away_team_id"], "game_id": feats["game_id"],
                      "season": feats["season"], "possessions": feats["possessions"], "diff": -feats["points_diff"]})
    return pd.concat([h, a], ignore_index=True)


def lineup_table(feats: pd.DataFrame, n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """One row per lineup: team_id (the most frequent), n_games, n_possessions, plus_minus_per100 and, with
    n_boot > 0, lo and hi from a bootstrap by game of the lineup's own stints (games resampled with
    replacement, the 2.5 and 97.5 percentiles; NaN under two games)."""
    rows = lineup_rows(feats)
    per_game = rows.groupby(["lineup_id", "game_id"], sort=True).agg(possessions=("possessions", "sum"), diff=("diff", "sum")).reset_index()
    tot = per_game.groupby("lineup_id", sort=True).agg(n_games=("game_id", "size"), n_possessions=("possessions", "sum"), diff=("diff", "sum"))
    tot["plus_minus_per100"] = np.where(tot["n_possessions"] > 0, tot["diff"] / tot["n_possessions"].replace(0, np.nan) * 100.0, np.nan)
    team = rows.dropna(subset=["team_id"]).groupby("lineup_id")["team_id"].agg(lambda s: int(s.mode().iloc[0]))
    df = tot.reset_index()
    df["team_id"] = df["lineup_id"].map(team).astype("Int64")
    df["lo"], df["hi"] = np.nan, np.nan
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        codes, uniq = pd.factorize(per_game["lineup_id"], sort=True)
        poss_all, diff_all = per_game["possessions"].to_numpy(dtype=float), per_game["diff"].to_numpy(dtype=float)
        lo, hi = {}, {}
        for k, lid in enumerate(uniq):
            idx = np.flatnonzero(codes == k)
            if len(idx) < 2:
                continue
            pick = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
            vals = diff_all[pick].sum(axis=1) / np.maximum(poss_all[pick].sum(axis=1), 1e-9) * 100.0
            lo[lid], hi[lid] = np.nanpercentile(vals, [2.5, 97.5])
        df["lo"] = df["lineup_id"].map(lo).astype(float)
        df["hi"] = df["lineup_id"].map(hi).astype(float)
    return df[["lineup_id", "team_id", "n_games", "n_possessions", "plus_minus_per100", "lo", "hi"]]


# ── ridge on sparse indicators ─────────────────────────────────────────────────────────────────────
def design(feats: pd.DataFrame, players: list, lineups: list) -> sp.csr_matrix:
    """Sparse [1 | player indicators (+1 home, -1 away) | lineup indicators (+1 home, -1 away)]."""
    assert_no_leak(feats[feature_columns()])
    n = len(feats)
    pix = {p: i + 1 for i, p in enumerate(players)}
    lix = {l: i + 1 + len(players) for i, l in enumerate(lineups)}
    rows, cols, vals = [np.arange(n)], [np.zeros(n, dtype=int)], [np.ones(n)]
    for k in range(1, 6):
        for side, sign in (("home", 1.0), ("away", -1.0)):
            ids = feats[f"{side}_{k}"].to_numpy()
            c = np.array([pix.get(int(p), -1) for p in ids])
            m = c >= 0
            rows.append(np.flatnonzero(m)); cols.append(c[m]); vals.append(np.full(int(m.sum()), sign))
    if lineups:
        for side, sign in (("lineup_home", 1.0), ("lineup_away", -1.0)):
            c = np.array([lix.get(l, -1) for l in feats[side].to_numpy()])
            m = c >= 0
            rows.append(np.flatnonzero(m)); cols.append(c[m]); vals.append(np.full(int(m.sum()), sign))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, 1 + len(players) + len(lineups)))
    return X


def normal_equations(X: sp.csr_matrix, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The weighted normal equations (X'WX, X'Wy) of a sparse design."""
    Xw = X.multiply(w[:, None]).tocsr()
    return (X.T @ Xw).toarray(), np.asarray(X.T @ (w * y)).reshape(-1)


def solve_ridge(A: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """(A + alpha I with the intercept unpenalised)^-1 b."""
    D = np.eye(A.shape[0]) * alpha
    D[0, 0] = 0.0
    return np.linalg.solve(A + D, b)


def choose_alpha(X: sp.csr_matrix, y: np.ndarray, w: np.ndarray, games: np.ndarray, alphas=ALPHAS, n_folds: int = 5, seed: int = 0) -> tuple[float, dict]:
    """Grouped CV by game: alpha with the smallest possession-weighted out-of-fold MSE, ties to the larger alpha.
    Returns (alpha, {alpha: mse})."""
    uniq = np.unique(games)
    k = min(n_folds, len(uniq))
    if k < 2:
        return float(alphas[len(alphas) // 2]), {}
    A, b = normal_equations(X, y, w)
    se = {a: 0.0 for a in alphas}
    for _tr, te in grouped_folds(uniq, n_folds=k, seed=seed):
        m = np.isin(games, te)
        A_te, b_te = normal_equations(X[m], y[m], w[m])
        for a in alphas:
            beta = solve_ridge(A - A_te, b - b_te, a)
            pred = X[m] @ beta
            se[a] += float(np.sum(w[m] * (y[m] - pred) ** 2))
    tot = float(w.sum())
    mse = {float(a): v / tot for a, v in se.items()}
    best = min(mse, key=lambda a: (mse[a], -a))          # ties go to the larger alpha (more shrinkage)
    return best, mse


def fit_ridge(feats: pd.DataFrame, players: list, lineups: list, alpha: float | None, n_folds: int, seed: int, alphas=ALPHAS) -> dict:
    """A possession-weighted ridge fit of the target on the design (alpha by grouped CV when None) -> coefficients, alpha, CV MSE."""
    X = design(feats, players, lineups)
    y = feats[TARGET].to_numpy(dtype=float)
    w = feats["possessions"].to_numpy(dtype=float)
    games = feats["game_id"].astype(str).to_numpy()
    cv = {}
    if alpha is None:
        alpha, cv = choose_alpha(X, y, w, games, alphas, n_folds, seed)
    beta = solve_ridge(*normal_equations(X, y, w), alpha)
    return {"alpha": float(alpha), "cv_mse": cv, "intercept": float(beta[0]),
            "player_coef": dict(zip(players, beta[1:1 + len(players)].tolist())),
            "lineup_coef": dict(zip(lineups, beta[1 + len(players):].tolist())), "players": list(players), "lineups": list(lineups)}


def bootstrap_player_coef(feats: pd.DataFrame, players: list, lineups: list, alpha: float, n: int, seed: int) -> pd.DataFrame:
    """player_id, lo, hi of the player coefficients over n refits on games resampled with replacement."""
    if n <= 0 or feats["game_id"].nunique() < 2:
        return pd.DataFrame({"player_id": players, "lo": np.nan, "hi": np.nan})
    X = design(feats, players, lineups)
    y = feats[TARGET].to_numpy(dtype=float)
    w = feats["possessions"].to_numpy(dtype=float)
    codes, uniq = pd.factorize(feats["game_id"].astype(str), sort=True)
    groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        beta = solve_ridge(*normal_equations(X[idx], y[idx], w[idx]), alpha)
        draws.append(beta[1:1 + len(players)])
    q = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
    return pd.DataFrame({"player_id": players, "lo": q[0], "hi": q[1]})


# ── candidates ─────────────────────────────────────────────────────────────────────────────────────
def gated_lineups(feats: pd.DataFrame, gate: int) -> list:
    """Lineups with at least `gate` possessions in feats (analytics.eval.min_sample_gate on possessions)."""
    poss = lineup_rows(feats).groupby("lineup_id")["possessions"].sum()
    ok = min_sample_gate(poss, gate)
    return sorted(poss.index[ok.to_numpy()].tolist())


def fit_candidates(train: pd.DataFrame, target_season: str | None, heldout_season: str | None, gate: int = GATE_POSS,
                   n_folds: int = 5, seed: int = 0, alphas=ALPHAS) -> dict:
    """Fit the three candidates on `train` -> {"plus_minus": table, "rapm": fit, "rapm_synergy": fit, "players", "lineups"}.
    The split guards run here, on every fit."""
    if len(train) == 0:
        raise ValueError("%s: empty training frame" % WHO)
    assert_season_split(train, target_season, heldout_season)
    refuse_heldout(sorted(train["game_id"].astype(str).unique()), WHO)
    schema.assert_time_forward(train[feature_columns()], "possessions")
    players = sorted({int(p) for c in feature_columns()[:10] for p in train[c].unique()})
    lineups = gated_lineups(train, gate)
    pm = lineup_table(train).set_index("lineup_id")["plus_minus_per100"]
    rapm = fit_ridge(train, players, [], None, n_folds, seed, alphas)
    syn = fit_ridge(train, players, lineups, None, n_folds, seed, alphas)
    return {"plus_minus": pm, "rapm": rapm, "rapm_synergy": syn, "players": players, "lineups": lineups,
            "fit_seasons": sorted(map(str, train["season"].unique())), "n_rows": int(len(train))}


def predict_lineups(fits: dict, lineup_ids) -> pd.DataFrame:
    """lineup_id, pred_plus_minus, pred_rapm, pred_rapm_synergy for the given lineup ids."""
    out = []
    for lid in lineup_ids:
        ids = [int(x) for x in str(lid).split("|")]
        r = sum(fits["rapm"]["player_coef"].get(p, 0.0) for p in ids)
        s = sum(fits["rapm_synergy"]["player_coef"].get(p, 0.0) for p in ids) + fits["rapm_synergy"]["lineup_coef"].get(lid, 0.0)
        out.append({"lineup_id": lid, "pred_plus_minus": float(fits["plus_minus"].get(lid, np.nan)), "pred_rapm": float(r), "pred_rapm_synergy": float(s),
                    "synergy_per100": float(fits["rapm_synergy"]["lineup_coef"].get(lid, 0.0)),
                    "n_unseen_players": int(sum(p not in fits["rapm"]["player_coef"] for p in ids))})
    return pd.DataFrame(out, columns=["lineup_id", "pred_plus_minus", "pred_rapm", "pred_rapm_synergy", "synergy_per100", "n_unseen_players"])


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────
def _spearman(d: pd.DataFrame, col: str) -> float:
    a, b = d[col].to_numpy(dtype=float), d["actual"].to_numpy(dtype=float)
    if len(d) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _mse(d: pd.DataFrame, col: str) -> float:
    return float(np.mean((d[col] - d["actual"]) ** 2))


def evaluate_pair(fits: dict, train: pd.DataFrame, test: pd.DataFrame, min_poss: int, n_boot: int, seed: int,
                  candidates=CANDIDATES, paired: bool = True) -> dict:
    """Score `candidates` on the lineups present in both frames with at least min_poss possessions each side;
    `paired` adds the paired bootstrap of each candidate's Spearman against plus_minus (validation pairs only)."""
    tr = lineup_table(train).set_index("lineup_id")
    te = lineup_table(test)
    both = te[te["lineup_id"].isin(tr.index[tr["n_possessions"] >= min_poss]) & (te["n_possessions"] >= min_poss)].copy()
    both = both.rename(columns={"plus_minus_per100": "actual", "n_possessions": "n_poss_test"})
    pred = predict_lineups(fits, both["lineup_id"])
    d = both.merge(pred, on="lineup_id")
    res = {"n_lineups": int(len(d)), "n_lineups_test": int(len(te)), "n_lineups_train": int(len(tr)), "metrics": {}, "frame": d}
    nan = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    for c in [c for c in CANDIDATES if c in candidates]:
        col = "pred_" + c
        if len(d) < 3:
            res["metrics"][c] = {"spearman": dict(nan), "mse": dict(nan), "spearman_vs_plus_minus": None}
            continue
        m = {"spearman": bootstrap_by_game(d, "lineup_id", lambda s, col=col: _spearman(s, col), n=n_boot, seed=seed),
             "mse": bootstrap_by_game(d, "lineup_id", lambda s, col=col: _mse(s, col), n=n_boot, seed=seed)}
        if c != "plus_minus" and paired:
            m["spearman_vs_plus_minus"] = paired_bootstrap_by_game(d, "lineup_id", lambda s, col=col: _spearman(s, col),
                                                                   lambda s: _spearman(s, "pred_plus_minus"), n=n_boot, seed=seed)
        else:
            m["spearman_vs_plus_minus"] = None
        res["metrics"][c] = m
    return res


def select(pairs: list[dict]) -> tuple[str, dict]:
    """The candidate with the highest mean validation Spearman (ties and no pairs: plus_minus)."""
    means = {}
    for c in CANDIDATES:
        vals = [p["metrics"][c]["spearman"]["point"] for p in pairs if c in p["metrics"]]
        vals = [v for v in vals if np.isfinite(v)]
        means[c] = float(np.mean(vals)) if vals else float("nan")
    best = "plus_minus"
    for c in CANDIDATES:
        if np.isfinite(means[c]) and (not np.isfinite(means[best]) or means[c] > means[best] + 1e-12):
            best = c
    return best, means


# ── the run ────────────────────────────────────────────────────────────────────────────────────────
def run(feats: pd.DataFrame, heldout_season: str, seed: int = 0, n_boot: int = 200, gate: int = GATE_POSS, min_poss: int = MIN_POSS_EVAL,
        n_folds: int = 5, alphas=ALPHAS, out_root=None) -> dict:
    """The season-split protocol on a feature frame -> context dict (pairs, selection, held-out scores, written paths).
    Every ridge solve and refit runs inside analytics.eval.fit_threads() (one context for the whole run)."""
    with fit_threads():
        return _run(feats, heldout_season, seed, n_boot, gate, min_poss, n_folds, alphas, out_root)


def _run(feats: pd.DataFrame, heldout_season: str, seed: int, n_boot: int, gate: int, min_poss: int, n_folds: int, alphas,
         out_root) -> dict:
    seasons = sorted(map(str, feats["season"].unique()), key=season_key)
    train_seasons = [s for s in seasons if season_key(s) < season_key(heldout_season)]
    ignored = [s for s in seasons if season_key(s) > season_key(heldout_season)]
    has_heldout = heldout_season in seasons
    if not train_seasons:
        raise SystemExit("%s: no training season before the held-out season %s (seasons %s)" % (WHO, heldout_season, seasons))
    ctx = {"seasons": seasons, "train_seasons": train_seasons, "heldout_season": heldout_season if has_heldout else None,
           "ignored_seasons": ignored, "pairs": [], "written": [], "gate": gate, "min_poss": min_poss, "n_boot": n_boot, "seed": seed}
    by_season = {s: feats[feats["season"].astype(str) == s] for s in seasons}
    # validation pairs within the training seasons
    for i in range(1, len(train_seasons)):
        target = train_seasons[i]
        tr = pd.concat([by_season[s] for s in train_seasons[:i]], ignore_index=True)
        fits = fit_candidates(tr, target, heldout_season, gate, n_folds, seed, alphas)
        r = evaluate_pair(fits, tr, by_season[target], min_poss, n_boot, seed)
        r.update({"fit_seasons": train_seasons[:i], "target": target, "alpha_rapm": fits["rapm"]["alpha"], "alpha_synergy": fits["rapm_synergy"]["alpha"],
                  "n_gated": len(fits["lineups"]), "n_players": len(fits["players"])})
        ctx["pairs"].append(r)
        if out_root is not None:                   # the dated, prior-seasons-only player table of the target season
            ctx["written"].append(write_player_rapm(out_dir(out_root, target) / "player_rapm_prior.parquet", target, by_season[target], tr, fits,
                                                    n_boot, seed, in_sample=False))
    ctx["selected"], ctx["mean_spearman"] = select(ctx["pairs"])
    # the held-out season, once, by the candidates refitted on every training season
    tr_all = pd.concat([by_season[s] for s in train_seasons], ignore_index=True)
    fits_all = fit_candidates(tr_all, heldout_season, heldout_season, gate, n_folds, seed, alphas)
    ctx["final"] = {"fit_seasons": train_seasons, "alpha_rapm": fits_all["rapm"]["alpha"], "alpha_synergy": fits_all["rapm_synergy"]["alpha"],
                    "n_gated": len(fits_all["lineups"]), "n_players": len(fits_all["players"]), "cv_mse_rapm": fits_all["rapm"]["cv_mse"],
                    "cv_mse_synergy": fits_all["rapm_synergy"]["cv_mse"], "home_court_rapm": fits_all["rapm"]["intercept"]}
    if has_heldout:
        # the selected candidate only (plus_minus stays as the labelled floor); no paired test on the held-out season
        ctx["heldout"] = evaluate_pair(fits_all, tr_all, by_season[heldout_season], min_poss, n_boot, seed,
                                       candidates=("plus_minus", ctx["selected"]), paired=False)
    # output tables: expanding-window fits per training season, the training fit for the held-out season
    if out_root is not None:
        for i, s in enumerate(train_seasons):
            tr = pd.concat([by_season[x] for x in train_seasons[:i + 1]], ignore_index=True)
            fits = fits_all if i == len(train_seasons) - 1 else fit_candidates(tr, None, heldout_season, gate, n_folds, seed, alphas)
            ctx["written"] += write_tables(out_root, s, by_season[s], tr, fits, gate, n_boot, seed, in_sample=True)
        if has_heldout:
            ctx["written"] += write_tables(out_root, heldout_season, by_season[heldout_season], tr_all, fits_all, gate, n_boot, seed, in_sample=False)
    return ctx


def player_teams(feats: pd.DataFrame) -> dict:
    """player_id -> the team id the player had most possessions with in feats (from the games table; empty without it)."""
    parts = []
    for side in ("home", "away"):
        for k in range(1, 6):
            parts.append(pd.DataFrame({"player_id": feats[f"{side}_{k}"].astype("int64"), "team_id": feats[f"{side}_team_id"], "w": feats["possessions"]}))
    m = pd.concat(parts).dropna(subset=["team_id"])
    if len(m) == 0:
        return {}
    w = m.groupby(["player_id", "team_id"])["w"].sum().reset_index().sort_values("w", ascending=False)
    return {int(r.player_id): int(r.team_id) for r in w.drop_duplicates("player_id").itertuples()}


def write_tables(out_root, season: str, season_feats: pd.DataFrame, fit_feats: pd.DataFrame, fits: dict, gate: int, n_boot: int, seed: int,
                 in_sample: bool) -> list[str]:
    """Write lineup_ratings and player_rapm of one output season; returns the paths."""
    d = out_dir(out_root, season)
    lt = lineup_table(season_feats, n_boot=n_boot, seed=seed)
    pred = predict_lineups(fits, lt["lineup_id"])
    lr = lt.merge(pred, on="lineup_id")
    lr.insert(0, "season", season)
    lr = lr.rename(columns={"pred_rapm": "rapm_sum_per100"})
    lr["insufficient"] = ~min_sample_gate(lr["n_possessions"], gate).to_numpy()
    lr["fit_seasons"] = ",".join(fits["fit_seasons"])
    lr["in_sample"] = bool(in_sample)
    lr = lr[["season", "lineup_id", "team_id", "n_possessions", "plus_minus_per100", "rapm_sum_per100", "synergy_per100", "lo", "hi", "insufficient",
             "n_games", "fit_seasons", "in_sample"]]
    lr.to_parquet(d / "lineup_ratings.parquet", index=False)
    return [str(d / "lineup_ratings.parquet"), write_player_rapm(d / "player_rapm.parquet", season, season_feats, fit_feats, fits, n_boot, seed, in_sample)]


def write_player_rapm(path: Path, season: str, season_feats: pd.DataFrame, fit_feats: pd.DataFrame, fits: dict, n_boot: int, seed: int,
                      in_sample: bool) -> str:
    """The player table: team_id is the player's main team in the OUTPUT season's lineups (fit seasons only for a
    player absent from it), as_of the last game date of the fit seasons."""
    coef = fits["rapm"]["player_coef"]
    ci = bootstrap_player_coef(fit_feats, fits["players"], [], fits["rapm"]["alpha"], n_boot, seed)
    pr = pd.DataFrame({"season": season, "player_id": fits["players"], "rapm": [coef[p] for p in fits["players"]]}).merge(ci, on="player_id")
    pr["player_id"] = pr["player_id"].astype("int64")
    teams = {**player_teams(fit_feats), **player_teams(season_feats)}
    pr["team_id"] = pr["player_id"].map(teams).astype("Int64")
    pr["as_of"] = str(fit_feats["game_date"].astype(str).max()) if "game_date" in fit_feats.columns and len(fit_feats) else None
    pr["in_sample"] = bool(in_sample)
    pr["fit_seasons"] = ",".join(fits["fit_seasons"])
    pr = pr[["season", "player_id", "rapm", "lo", "hi", "team_id", "as_of", "in_sample", "fit_seasons"]]
    pr.to_parquet(path, index=False)
    return str(path)


# ── report ─────────────────────────────────────────────────────────────────────────────────────────
def _rel(path: str, root) -> str:
    """path relative to root (OUT_ROOT) when it lies under it, as written otherwise."""
    if root is None:
        return str(path)
    try:
        return Path(path).relative_to(Path(root)).as_posix()
    except ValueError:
        return str(path)


def write_report(path: Path, ctx: dict) -> str:
    """Write the markdown report to path and return its text; on fixtures every section heading says fixture output."""
    fx = ctx["fixture"]
    tag = " (fixture output, not a result)" if fx else ""
    L = ["# Lineup synergy and stints (analytics/lineups)%s" % tag, "",
         "Written %s by `%s`." % (ctx["written_at"], ctx["command"]), ""]
    if fx:
        L += ["**Fixture output, not a result.** This run used the synthetic fixture generator %s; every number below is fixture output. "
              "The real run is the Mac command at the end." % ctx["generator"], ""]
    L += ["## Inputs%s" % tag, "", "| item | value |", "| --- | --- |"]
    for k, v in ctx["inputs"].items():
        L.append("| %s | %s |" % (k, v))
    L += ["", "## Protocol%s" % tag, "",
          "Unit: stint; target: (points_home - points_away) / possessions x 100 with possessions = duration_s / 28.8 when the table has no count "
          "(100 per team per 48 minutes). Candidates: plus_minus (raw net rating pooled over the fitting seasons), rapm (possession-weighted ridge "
          "on player indicators, alpha by grouped CV by game over %s), rapm_synergy (rapm plus ridge-penalised indicators for five-man lineups with at "
          "least %d possessions in the fitting seasons; the lineup coefficient is the synergy). Selection metric: out-of-season Spearman and MSE of the "
          "predicted against the raw next-season lineup net rating for lineups with at least %d possessions on both sides, intervals from %d bootstrap "
          "resamples over lineups (seed %d), paired against plus_minus. The candidate with the highest mean validation Spearman is selected and the "
          "held-out season %s is scored once by the candidates refitted on %s. Guards: assert_season_split on every fit, refuse_heldout on the training "
          "game ids, assert_time_forward on the feature frame."
          % (list(ctx["alphas"]), ctx["gate"], ctx["min_poss"], ctx["n_boot"], ctx["seed"], ctx["heldout_season"] or "(absent)", ", ".join(ctx["train_seasons"])), "",
          "## Candidate table: out-of-season validation within the training seasons%s" % tag, ""]
    if not ctx["pairs"]:
        L.append("One training season only: no validation pair, plus_minus selected by default.")
    for p in ctx["pairs"]:
        L += ["Fit on %s, predict %s: %d lineups present in both (%d in the fit seasons, %d in the target), %d gated lineups, %d players, alpha rapm %g, alpha synergy %g."
              % (", ".join(p["fit_seasons"]), p["target"], p["n_lineups"], p["n_lineups_train"], p["n_lineups_test"], p["n_gated"], p["n_players"], p["alpha_rapm"], p["alpha_synergy"]), "",
              "| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |", "| --- | --- | --- | --- |"]
        for c in CANDIDATES:
            m = p["metrics"][c]
            pv = m.get("spearman_vs_plus_minus")
            pvs = "baseline" if pv is None else "%s [%s, %s]%s" % (fmt_num(pv["diff"]), fmt_num(pv["lo"]), fmt_num(pv["hi"]), " (excludes 0)" if pv["excludes_zero"] else "")
            L.append("| %s | %s | %s | %s |" % (c, fmt_ci(m["spearman"]), fmt_ci(m["mse"]), pvs))
        L.append("")
    L += ["Mean validation Spearman: %s. Selected: **%s**." % (", ".join("%s %s" % (c, fmt_num(v)) for c, v in ctx["mean_spearman"].items()), ctx["selected"]), "",
          "## Out-of-season correlation of raw lineup net rating%s" % tag, "",
          "| from | to | lineups | Pearson | Spearman |", "| --- | --- | --- | --- | --- |"]
    for r in ctx["cross_season"].itertuples():
        held = " (held-out season: descriptive, scored once below, not used for selection)" if r.season_to == ctx["heldout_season"] else ""
        L.append("| %s | %s%s | %d | %s | %s |" % (r.season_from, r.season_to, held, r.n_lineups, fmt_num(r.pearson), fmt_num(r.spearman)))
    L += ["", "Seasons later than the held-out season are not in this table; the held-out row is labelled."]
    L += ["", "## Held-out season, scored once%s" % tag, ""]
    h = ctx.get("heldout")
    if h is None:
        L.append("The held-out season %s is not in the inputs: nothing scored." % ctx["heldout_season"])
    else:
        L += ["Season %s scored once by the selected candidate refitted on %s: %d lineups present in both, %d gated lineups, %d players, alpha rapm %g, alpha synergy %g, home court (rapm intercept) %s per 100."
              % (ctx["heldout_season"], ", ".join(ctx["final"]["fit_seasons"]), h["n_lineups"], ctx["final"]["n_gated"], ctx["final"]["n_players"],
                 ctx["final"]["alpha_rapm"], ctx["final"]["alpha_synergy"], fmt_num(ctx["final"]["home_court_rapm"], 2)), "",
              "Only the candidate selected on the training seasons is scored here; plus_minus is shown as the labelled floor "
              "(not a comparison, no paired test): the other candidates are never scored on the held-out season.", "",
              "| candidate | Spearman | MSE |", "| --- | --- | --- |"]
        for c in CANDIDATES:
            if c not in h["metrics"]:
                continue
            m = h["metrics"][c]
            mark = " (selected)" if c == ctx["selected"] else " (floor)"
            L.append("| %s%s | %s | %s |" % (c, mark, fmt_ci(m["spearman"]), fmt_ci(m["mse"])))
    L += ["", "## Outputs (relative to OUT_ROOT)%s" % tag, ""]
    L += ["- " + _rel(w, ctx.get("out_root")) for w in ctx["written"]] or ["No tables written (pass --out OUT_ROOT)."]
    L += ["", "## Status and the Mac command%s" % tag, ""]
    if fx:
        L.append("Fixture run: the pipeline is exercised end to end on synthetic stints with hidden ratings and a lineup interaction; nothing here is a result.")
    else:
        L.append("Real data run over %s." % ", ".join(ctx["seasons"]))
    L += ["", "    " + MAC_CMD, "",
          "Needs on the Mac: DATA_ROOT/collect/stints/{stints,playbyplay}/<season>.parquet for 2020-21 to 2024-25 (collect/stints.py, or "
          "`python -m analytics.lineups.stints --data DATA_ROOT` from the play-by-play), games/<season>.parquet for team ids, and the optional "
          "starters/<season>.parquet. Nothing under collect/stints/heldout/ is read.", ""]
    text = "\n".join(L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ── main ───────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    """The CLI's arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic stints (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT")
    ap.add_argument("--report", default="reports/lineups.md", help="markdown report path (default %(default)s)")
    ap.add_argument("--out", help="OUT_ROOT for the tables (default DATA_ROOT; none for --fixtures without it)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the fixtures and the bootstrap (default %(default)s)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples (default %(default)s)")
    ap.add_argument("--heldout-season", default=None, help="default 2024-25 (--data) or the last fixture season (--fixtures)")
    ap.add_argument("--gate", type=int, default=GATE_POSS, help="possession gate of the lineup terms and the insufficient flag (default %(default)s)")
    ap.add_argument("--min-poss", type=int, default=MIN_POSS_EVAL, help="possessions a lineup needs on both sides of a validation pair (default %(default)s)")
    ap.add_argument("--folds", type=int, default=5, help="grouped folds by game for the ridge alpha (default %(default)s)")
    ap.add_argument("--seasons", nargs="*", default=list(SEASONS), help="--data only: seasons to read (default %(default)s)")
    add_fixture_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    """Validate, build the features, run the protocol, write the tables under OUT_ROOT and the report."""
    from analytics.lineups.eda import cross_season_correlation
    a = parse_args(argv)
    inputs = load_inputs(a.fixtures, a.data, a.seasons, a.seed, a.fixture_generator, a.n_games)
    if len(inputs["stints"]) == 0:
        raise SystemExit("%s: no stints rows found" % WHO)
    _clean, counts = validate(inputs["stints"])
    feats = features(inputs["stints"], inputs["games"])
    heldout = a.heldout_season or (FIXTURE_SEASONS[-1] if a.fixtures else "2024-25")
    out_root = a.out or (a.data if not a.fixtures else None)
    ctx = run(feats, heldout, a.seed, a.bootstrap, a.gate, a.min_poss, a.folds, ALPHAS, out_root)
    ctx.update({"fixture": inputs["fixture"], "generator": inputs["generator"], "out_root": out_root, "alphas": ALPHAS, "written_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "command": "python -m analytics.lineups.train " + " ".join(argv if argv is not None else sys.argv[1:]),
                "cross_season": cross_season_correlation(feats[feats["season"].astype(str).isin(ctx["train_seasons"] + ([heldout] if ctx["heldout_season"] else []))],
                                                        min_poss=a.min_poss),
                "inputs": {"source": "fixture generator %s" % inputs["generator"] if inputs["fixture"] else "DATA_ROOT " + str(a.data),
                           "seasons": ", ".join(ctx["seasons"]), "stints in": counts["rows_in"], "stints after validate": counts["rows_out"],
                           "games": counts["games_out"], "lineup_ok share": "%.3f" % counts["lineup_ok_share"],
                           "dropped not 5v5": counts["dropped_not_5v5"], "dropped duration": counts["dropped_duration"],
                           "dropped points": counts["dropped_points"], "dropped duplicate": counts["dropped_duplicate"],
                           "dropped heldout": counts["dropped_heldout"], "possessions": "%.0f" % feats["possessions"].sum(),
                           "feature frame rows": len(feats)}})
    text = write_report(Path(a.report), ctx)
    print(text)
    return ctx


if __name__ == "__main__":
    main()
