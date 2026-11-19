"""tools/research_lineups_e2.py: the lineups research experiment lineups-E2 (PLAN.md D70, D71, D79, D102).

    python3 -m export.heavy --name lineups-E2 -- python3 -m tools.research_lineups_e2 e2-dev   [--artifacts DIR]
    python3 -m export.heavy --name lineups-E2 -- python3 -m tools.research_lineups_e2 e2-final [--artifacts DIR]

Pre-registration: docs/research/lineups.md, section 6, "lineups-E2". The question: inside a season, do lineup offensive
and defensive ratings shrunk toward the sum of their players' ratings from earlier seasons (L-RAPM, Petridis and
Pelechrinis 2026) predict next week's points per possession better than raw lineup ratings, most of all for lineups
with few possessions?

Inputs are the gated tables of lineups-E1 (tools.research_lineups e1-gate: the frozen stints of 2020-21 to 2023-24,
equal to the stored ones, and the frozen RAPM reproduced exactly), so no new DATA_ROOT read happens here beyond the
stored lineup_ratings table of the synergy gate. 2024-25 is never read. The 2025-26 season is read only inside the
scorer that score_once runs, as its own ledger entry; lineups-E1 and winprob-E1 have each scored that season once.

Definitions, fixed in this file before any test score (the pre-registration leaves them open):
- Observations. Each kept stint (the frozen validation's five-against-five stints) gives two: the home lineup on
  offense against the away lineup on defense (y = points_home / possessions), and the reverse (y = points_away /
  possessions). The weight is the stint's possessions (duration / 28.8, frozen). The possession-weighted squared error
  of y equals the possession-level squared error up to a within-stint term no prediction can change, so RMSE here is
  in points per possession but smaller than a possession-level RMSE; relative changes are not comparable to the
  paper's figures.
- Weeks. Week k holds the games dated 7k to 7k + 6 days after the season's first game. Every week from the fifth on
  (k >= WARMUP_WEEKS) is predicted from the season's earlier weeks only (k' < k): the fit, the league average, the raw
  ratings and the lineup possession counts all come from earlier weeks, and a check refuses any training row dated
  on or after the first predicted date.
- Arms (each predicts y for an observation with lineup A on offense and lineup B on defense):
  raw              the baseline: A's points scored per possession in earlier weeks plus B's points allowed per
                   possession in earlier weeks minus the league average; a lineup with no earlier possession counts
                   at league average (the paper's baseline).
  lrapm            L-RAPM: y = b0 + O_A - D_B by possession-weighted ridge on lineup indicators over earlier weeks,
                   each O and D shrunk toward its prior (penalty lambda), the prior the sum of the five players'
                   offensive (defensive) ratings. D is points per possession saved (positive is good defense), so a
                   better defense lowers the prediction (unit-tested). A lineup not seen in earlier weeks is
                   predicted by its prior. The player ratings are an offensive and defensive RAPM fitted on the
                   development seasons (Sill's construction as the paper uses it, the frozen solver and weights, one
                   penalty for each side chosen by grouped CV by game over the frozen ALPHAS); a player never seen
                   counts 0 (frozen behaviour; the paper uses -1). The paper's priors include the league average in
                   both O and D; it cancels in O_A - D_B and is left out.
  shrink_zero      the ablation: the same ridge with every prior 0 (pure shrinkage to league average).
  rapm_synergy     the second pre-registered baseline: the frozen candidate refit with frozen code on the development
                   seasons (fit_candidates), fixed for the whole season: its stint margin per 100 d (home minus away)
                   is split evenly, y_home = league average + d / 200 and y_away = league average - d / 200, the
                   league average from earlier weeks. Handicapped by construction (no in-season update).
  lrapm_net_prior  context, not pre-registered, reads no verdict: L-RAPM with the prior from the frozen net RAPM
                   split in halves (each player gives rapm / 200 to O and rapm / 200 to D).
  Every penalty (lrapm, shrink_zero, lrapm_net_prior, each its own) is chosen on the development stage's expanding
  windows inside 2023-24, priors fitted on 2020-21 to 2022-23, by the smallest mean weekly RMSE (ties to the larger
  penalty; the grid doubles past its top while the top wins).
- Metric (pre-registered): weekly RMSE of points per possession, averaged over the predicted weeks, and the relative
  improvement d = (mean RMSE(lrapm) - mean RMSE(raw)) / mean RMSE(raw) (the paper's equation 1). Bins by the
  possessions of the less-seen of the two lineups in earlier weeks: under 25, 25 to 100, over 100, each a pooled RMSE
  over all predicted weeks. Every interval: 2000 resamples of games (seed 0), all arms paired on the same games.
- Verdict (pre-registered, read against both named baselines, PLAN.md D104): lrapm is adopted when, against raw and
  against rapm_synergy each, d is at most -0.01, the paired interval of mean RMSE(lrapm) minus the baseline's lies
  wholly below zero, and no bin's paired interval of RMSE(lrapm) minus the baseline's lies wholly above zero.
  "Toward their players" is claimed only when the interval of lrapm minus shrink_zero lies wholly below zero. A
  non-finite value refuses the verdict.
"""

import argparse
import json
import math
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from export.heavy import heavy_job, peak_rss_gb
from tools import research_lineups as E1
from tools.frozen_common import REPO_ROOT, _plain
from tools.research_testset import PBP, PBP_TEST_SEASON, dev_games, refuse_test, score_once, scoring_data_root

WHO = "tools.research_lineups_e2"
EXPERIMENT = "lineups-E2"
RESULTS = E1.RESULTS
SEED, BOOT, FOLDS = 0, 2000, 5
WARMUP_WEEKS = 4
LAMBDA_GRID = (5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
LAMBDA_MAX = 102400.0
THRESHOLD = -0.01
BINS = (("under_25", 0.0, 25.0), ("25_to_100", 25.0, 100.0), ("over_100", 100.0, math.inf))
PRIOR_ARMS = ("lrapm", "shrink_zero", "lrapm_net_prior")
ARMS = ("raw", "lrapm", "shrink_zero", "rapm_synergy", "lrapm_net_prior")
CONTEXT = {"rapm_synergy": "pre-registered second baseline; the verdict needs the rule against it too (PLAN.md D104)",
           "lrapm_net_prior": "not pre-registered, reads no verdict"}
DRY_SEASON, DRY_PRIOR = "2023-24", ("2020-21", "2021-22", "2022-23")
CHECK_SEASON, CHECK_PRIOR = "2022-23", ("2020-21", "2021-22")
DEV_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24")
SYNERGY_TOLERANCE = 1e-6


# ----------------------------------------------------------------------------- observations and weeks

def observations(feats) -> pd.DataFrame:
    """Two rows per kept stint: offense lineup, defense lineup, their players, y (points per possession), w."""
    w = feats["possessions"].to_numpy(dtype=float)
    if "points_home" not in feats.columns:      # the frozen feature frame keeps only the difference (with_points)
        raise ValueError(f"{WHO}: the frame lacks points_home and points_away (use with_points)")
    ph, pa = feats["points_home"].to_numpy(dtype=float), feats["points_away"].to_numpy(dtype=float)
    base = {"game_id": feats["game_id"].astype(str).to_numpy(), "game_date": feats["game_date"].astype(str).str[:10].to_numpy(),
            "w": w}
    home = pd.DataFrame({**base, "off": feats["lineup_home"].to_numpy(), "dfn": feats["lineup_away"].to_numpy(),
                         "y": ph / w, "off_home": True,
                         **{f"o{k}": feats[f"home_{k}"].astype("int64").to_numpy() for k in range(1, 6)},
                         **{f"d{k}": feats[f"away_{k}"].astype("int64").to_numpy() for k in range(1, 6)},
                         "stint_row": np.arange(len(feats))})
    away = pd.DataFrame({**base, "off": feats["lineup_away"].to_numpy(), "dfn": feats["lineup_home"].to_numpy(),
                         "y": pa / w, "off_home": False,
                         **{f"o{k}": feats[f"away_{k}"].astype("int64").to_numpy() for k in range(1, 6)},
                         **{f"d{k}": feats[f"home_{k}"].astype("int64").to_numpy() for k in range(1, 6)},
                         "stint_row": np.arange(len(feats))})
    return pd.concat([home, away], ignore_index=True)


def with_points(feats, stints_points) -> pd.DataFrame:
    """The feature frame with points_home and points_away joined back by (game_id, stint)."""
    key = ["game_id", "stint"]
    p = stints_points[key + ["points_home", "points_away"]].copy()
    p["game_id"] = p["game_id"].astype(str)
    out = feats.copy()
    out["game_id"] = out["game_id"].astype(str)
    out = out.merge(p, on=key, how="left", validate="one_to_one")
    if out[["points_home", "points_away"]].isna().any().any():
        raise ValueError(f"{WHO}: a kept stint has no points")
    if not np.allclose(out["points_home"] - out["points_away"], out["points_diff"]):
        raise ValueError(f"{WHO}: the joined points do not match points_diff")
    return out


def week_of(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates))
    return ((d - d.min()).dt.days // 7).to_numpy()


# ----------------------------------------------------------------------------- player offensive and defensive RAPM

def player_od_design(obs, players: list):
    """Sparse [1 | offense players +1 | defense players -1]."""
    import scipy.sparse as sp
    n, P = len(obs), len(players)
    pix = {p: i for i, p in enumerate(players)}
    rows, cols, vals = [np.arange(n)], [np.zeros(n, dtype=int)], [np.ones(n)]
    for pre, off, sign in (("o", 1, 1.0), ("d", 1 + P, -1.0)):
        for k in range(1, 6):
            c = np.array([pix.get(int(p), -1) for p in obs[f"{pre}{k}"].to_numpy()])
            m = c >= 0
            rows.append(np.flatnonzero(m)); cols.append(c[m] + off); vals.append(np.full(int(m.sum()), sign))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, 1 + 2 * P))


def _od_penalty(P: int, a_off: float, a_def: float):
    return np.concatenate([[0.0], np.full(P, a_off), np.full(P, a_def)])


def fit_player_od(obs, n_folds: int = FOLDS, seed: int = SEED) -> dict:
    """Offensive and defensive player ratings in points per possession, one penalty a side by grouped CV by game
    (possession-weighted out-of-fold MSE over the frozen ALPHAS grid; ties to the larger penalties)."""
    from analytics.eval import grouped_folds
    from analytics.lineups.train import ALPHAS, normal_equations
    refuse_test(obs["game_id"].unique(), WHO, PBP)
    players = sorted({int(p) for c in [f"{s}{k}" for s in "od" for k in range(1, 6)] for p in obs[c].unique()})
    P = len(players)
    X = player_od_design(obs, players)
    y, w, g = obs["y"].to_numpy(), obs["w"].to_numpy(), obs["game_id"].to_numpy()
    A, b = normal_equations(X, y, w)
    se = {(a, d): 0.0 for a in ALPHAS for d in ALPHAS}
    for _tr, te in grouped_folds(np.unique(g), n_folds=n_folds, seed=seed):
        m = np.isin(g, te)
        A_te, b_te = normal_equations(X[m], y[m], w[m])
        for (a, d) in se:
            beta = np.linalg.solve(A - A_te + np.diag(_od_penalty(P, a, d)), b - b_te)
            se[(a, d)] += float(np.sum(w[m] * (y[m] - X[m] @ beta) ** 2))
    tot = float(w.sum())
    mse = {k: v / tot for k, v in se.items()}
    a_off, a_def = min(mse, key=lambda k: (mse[k], -k[0], -k[1]))
    beta = np.linalg.solve(A + np.diag(_od_penalty(P, a_off, a_def)), b)
    return {"players": players, "intercept": float(beta[0]), "off": dict(zip(players, beta[1:1 + P].tolist())),
            "def": dict(zip(players, beta[1 + P:].tolist())), "alpha_off": float(a_off), "alpha_def": float(a_def),
            "cv_mse_best": float(mse[(a_off, a_def)]), "fit_games": int(len(np.unique(g)))}


def lineup_prior(lineups, off: dict, dfn: dict):
    """lineup id -> (sum of the players' offensive ratings, sum of their defensive ratings); unseen players 0."""
    out = {}
    if not off and not dfn:
        return {lid: (0.0, 0.0) for lid in lineups}
    for lid in lineups:
        ids = [int(x) for x in str(lid).split("|")]
        out[lid] = (sum(off.get(p, 0.0) for p in ids), sum(dfn.get(p, 0.0) for p in ids))
    return out


def prior_source(name: str, od: dict, frozen_rapm: dict):
    """The (off, def) player dicts behind an arm's prior."""
    if name == "lrapm":
        return od["off"], od["def"]
    if name == "lrapm_net_prior":
        half = {p: v / 200.0 for p, v in frozen_rapm.items()}
        return half, half
    if name == "shrink_zero":
        return {}, {}
    raise ValueError(name)


# ----------------------------------------------------------------------------- the weekly expanding window

CG_RTOL = 1e-12


def solve_spd(M, rhs):
    """The solution of a sparse symmetric positive definite system by conjugate gradients with a Jacobi
    preconditioner (a direct sparse solve fills in badly on the lineup graph); SystemExit unless it converges to a
    relative residual of CG_RTOL."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg
    d = M.diagonal()
    pre = sp.diags(1.0 / d)
    x, info = cg(M, rhs, rtol=CG_RTOL, atol=0.0, maxiter=20 * M.shape[0], M=pre)
    resid = float(np.linalg.norm(M @ x - rhs) / max(np.linalg.norm(rhs), 1e-300))
    if info != 0 or resid > 1e-9:
        raise SystemExit(f"{WHO}: the conjugate gradient solve did not converge (info {info}, residual {resid:.2e})")
    return x


def fit_lineups(train, prior_off: dict, prior_def: dict, lam: float) -> dict:
    """y = b0 + O_off - D_def, possession-weighted, sum of lam * (O - prior)^2 + lam * (D - prior)^2 (b0 free)."""
    import scipy.sparse as sp
    lineups = sorted(set(train["off"]) | set(train["dfn"]))
    L = len(lineups)
    ix = {l: i for i, l in enumerate(lineups)}
    n = len(train)
    oc = 1 + train["off"].map(ix).to_numpy()
    dc = 1 + L + train["dfn"].map(ix).to_numpy()
    X = sp.csr_matrix((np.concatenate([np.ones(n), np.ones(n), -np.ones(n)]),
                       (np.tile(np.arange(n), 3), np.concatenate([np.zeros(n, dtype=int), oc, dc]))), shape=(n, 1 + 2 * L))
    w, y = train["w"].to_numpy(), train["y"].to_numpy()
    A = (X.T @ X.multiply(w[:, None]).tocsr()).tocsc()
    pri = lineup_prior(lineups, prior_off, prior_def)
    pi = np.concatenate([[0.0], [pri[l][0] for l in lineups], [pri[l][1] for l in lineups]])
    pen = np.concatenate([[0.0], np.full(2 * L, lam)])
    rhs = np.asarray(X.T @ (w * y)).ravel() + pen * pi
    beta = solve_spd((A + sp.diags(pen)).tocsr(), rhs)
    return {"b0": float(beta[0]), "O": dict(zip(lineups, beta[1:1 + L])), "D": dict(zip(lineups, beta[1 + L:]))}


def predict_lineups(fit: dict, test, prior_off: dict, prior_def: dict) -> np.ndarray:
    """b0 + O_A - D_B; a lineup not in the fit is predicted by its prior."""
    new = (set(test["off"]) | set(test["dfn"])) - set(fit["O"])
    pri = lineup_prior(new, prior_off, prior_def)
    O = test["off"].map(lambda l: fit["O"][l] if l in fit["O"] else pri[l][0]).to_numpy(dtype=float)
    D = test["dfn"].map(lambda l: fit["D"][l] if l in fit["D"] else pri[l][1]).to_numpy(dtype=float)
    return fit["b0"] + O - D


def raw_predict(train, test) -> np.ndarray:
    """A's points scored per possession plus B's points allowed per possession minus the league average, all from
    earlier weeks; an unseen lineup counts at league average."""
    lg = float((train["y"] * train["w"]).sum() / train["w"].sum())
    pts = train["y"] * train["w"]
    off = (pts.groupby(train["off"]).sum() / train["w"].groupby(train["off"]).sum())
    allowed = (pts.groupby(train["dfn"]).sum() / train["w"].groupby(train["dfn"]).sum())
    o = test["off"].map(off).fillna(lg).to_numpy(dtype=float)
    a = test["dfn"].map(allowed).fillna(lg).to_numpy(dtype=float)
    return o + a - lg


def synergy_margin(feats, syn: dict) -> np.ndarray:
    """The frozen rapm_synergy stint margin per 100 (home minus away) of every stint of feats."""
    from analytics.lineups.train import design
    players, lineups = syn["players"], syn["lineups"]
    beta = np.array([syn["intercept"]] + [syn["player_coef"][p] for p in players] + [syn["lineup_coef"][l] for l in lineups])
    return np.asarray(design(feats, players, lineups) @ beta).ravel()


def window_possessions(train) -> pd.Series:
    """lineup -> possessions played in earlier weeks (each stint counted once, from its offense row)."""
    return train["w"].groupby(train["off"]).sum()


def weekly(obs, feats, arms: dict, synergy=None) -> pd.DataFrame:
    """The expanding window over the season of obs. arms: name -> (prior_off, prior_def, lam) for the ridge arms.
    -> one row per predicted observation: game_id, week, w, y, bin_poss and a pred_<arm> column per arm."""
    weeks = week_of(obs["game_date"])
    obs = obs.assign(week=weeks)
    margin = synergy_margin(feats, synergy) if synergy is not None else None
    out = []
    for k in range(WARMUP_WEEKS, int(weeks.max()) + 1):
        train, test = obs[obs["week"] < k], obs[obs["week"] == k]
        if len(test) == 0:
            continue
        if train["game_date"].max() >= test["game_date"].min():
            raise SystemExit(f"{WHO}: week {k} would see its own outcomes")
        res = test[["game_id", "week", "w", "y"]].copy()
        poss = window_possessions(train)
        res["bin_poss"] = np.minimum(test["off"].map(poss).fillna(0.0).to_numpy(),
                                     test["dfn"].map(poss).fillna(0.0).to_numpy())
        res["pred_raw"] = raw_predict(train, test)
        for name, (po, pd_, lam) in arms.items():
            res[f"pred_{name}"] = predict_lineups(fit_lineups(train, po, pd_, lam), test, po, pd_)
        if margin is not None:
            lg = float((train["y"] * train["w"]).sum() / train["w"].sum())
            d = margin[test["stint_row"].to_numpy()]
            res["pred_rapm_synergy"] = lg + np.where(test["off_home"].to_numpy(), d, -d) / 200.0
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- metrics

def mean_weekly_rmse(rows, arm: str) -> float:
    e = rows["w"] * (rows["y"] - rows[f"pred_{arm}"]) ** 2
    s = e.groupby(rows["week"]).sum() / rows["w"].groupby(rows["week"]).sum()
    return float(np.sqrt(s).mean())


def choose_lambda(obs, feats, name: str, po: dict, pd_: dict, grid=LAMBDA_GRID) -> tuple:
    """The penalty with the smallest mean weekly RMSE (ties to the larger; the grid doubles past its top while the top
    wins, up to LAMBDA_MAX) -> (lam, {lam: mean weekly rmse})."""
    curve = {}
    grid = list(grid)
    while True:
        for lam in grid:
            if lam not in curve:
                rows = weekly(obs, feats, {name: (po, pd_, lam)})
                curve[lam] = mean_weekly_rmse(rows, name)
        best = min(curve, key=lambda l: (curve[l], -l))
        if best == max(curve) and best * 2 <= LAMBDA_MAX:
            grid = [best * 2]
            continue
        return float(best), {str(k): v for k, v in sorted(curve.items())}


def score_rows(rows, arms=ARMS, n: int = BOOT, seed: int = SEED) -> dict:
    """Mean weekly RMSE per arm, d, the paired differences against raw, and the bins, with one shared bootstrap of
    games; the verdict as pre-registered."""
    cols = ["w", "y"] + [f"pred_{a}" for a in arms]
    finite = np.isfinite(rows[cols].to_numpy(dtype=float)).all(axis=1)
    games = pd.Index(sorted(rows["game_id"].unique()))
    gi = games.get_indexer(rows["game_id"])
    weeks = np.sort(rows["week"].unique())
    wi = np.searchsorted(weeks, rows["week"].to_numpy())
    G, K = len(games), len(weeks)
    week_of_game = np.zeros(G, dtype=int)
    week_of_game[gi] = wi
    # the three bins as written: under 25, 25 to 100 (inclusive), over 100
    bp = rows["bin_poss"].to_numpy()
    binlab = np.where(bp < 25.0, 0, np.where(bp <= 100.0, 1, 2))
    w = rows["w"].to_numpy(dtype=float)
    per_game_w = np.bincount(gi, weights=w, minlength=G)
    per_game_bin_w = np.stack([np.bincount(gi, weights=w * (binlab == j), minlength=G) for j in range(3)])
    sse, sse_bin = {}, {}
    for a in arms:
        e = w * (rows["y"].to_numpy(dtype=float) - rows[f"pred_{a}"].to_numpy(dtype=float)) ** 2
        sse[a] = np.bincount(gi, weights=e, minlength=G)
        sse_bin[a] = np.stack([np.bincount(gi, weights=e * (binlab == j), minlength=G) for j in range(3)])
    rng = np.random.default_rng(seed)
    idx = np.vstack([np.arange(G)[None, :], rng.integers(0, G, size=(n, G))])      # row 0 is the point estimate

    def weekly_mean(per_game):
        wk = week_of_game[idx]                                                    # (n+1, G)
        num = np.zeros((idx.shape[0], K))
        den = np.zeros((idx.shape[0], K))
        for r in range(idx.shape[0]):
            num[r] = np.bincount(wk[r], weights=per_game[idx[r]], minlength=K)
            den[r] = np.bincount(wk[r], weights=per_game_w[idx[r]], minlength=K)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.sqrt(num / den), axis=1)

    def pooled(per_game, per_game_den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(per_game[idx].sum(axis=1) / per_game_den[idx].sum(axis=1))

    def ci(v):
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        return [float(lo), float(hi)]

    mw = {a: weekly_mean(sse[a]) for a in arms}
    out = {"games": int(G), "weeks_predicted": int(K), "observations": int(len(rows)), "possessions": float(w.sum()),
           "resamples": n, "seed": seed, "context": CONTEXT, "non_finite_rows": int((~finite).sum()),
           "mean_weekly_rmse": {a: {"point": float(mw[a][0]), "ci95": ci(mw[a])} for a in arms},
           "relative_improvement_vs_raw": {}, "paired_minus_raw": {}, "bins": {}}
    for a in arms:
        if a == "raw":
            continue
        rel = (mw[a] - mw["raw"]) / mw["raw"]
        diff = mw[a] - mw["raw"]
        out["relative_improvement_vs_raw"][a] = {"point": float(rel[0]), "ci95": ci(rel)}
        out["paired_minus_raw"][a] = {"point": float(diff[0]), "ci95": ci(diff)}
    d = mw["lrapm"] - mw["shrink_zero"]
    out["paired_lrapm_minus_shrink_zero"] = {"point": float(d[0]), "ci95": ci(d)}
    d = mw["lrapm"] - mw["rapm_synergy"]
    rel = d / mw["rapm_synergy"]
    out["paired_lrapm_minus_rapm_synergy"] = {"point": float(d[0]), "ci95": ci(d)}
    out["relative_improvement_vs_rapm_synergy"] = {"point": float(rel[0]), "ci95": ci(rel)}
    for j, (name, _lo, _hi) in enumerate(BINS):
        b = {"observations": int((binlab == j).sum()), "possessions": float(w[binlab == j].sum()), "rmse": {}, "minus_raw": {},
             "relative_vs_raw": {}}
        r = {a: pooled(sse_bin[a][j], per_game_bin_w[j]) for a in arms}
        for a in arms:
            b["rmse"][a] = {"point": float(r[a][0]), "ci95": ci(r[a])}
            if a != "raw":
                b["minus_raw"][a] = {"point": float(r[a][0] - r["raw"][0]), "ci95": ci(r[a] - r["raw"])}
                b["relative_vs_raw"][a] = {"point": float((r[a][0] - r["raw"][0]) / r["raw"][0]),
                                           "ci95": ci((r[a] - r["raw"]) / r["raw"])}
        ds = r["lrapm"] - r["rapm_synergy"]
        b["lrapm_minus_rapm_synergy"] = {"point": float(ds[0]), "ci95": ci(ds)}
        b["lrapm_relative_vs_rapm_synergy"] = {"point": float(ds[0] / r["rapm_synergy"][0]), "ci95": ci(ds / r["rapm_synergy"])}
        out["bins"][name] = b
    def rule(rel, diff, bin_key):
        worse = [n for n in out["bins"] if out["bins"][n][bin_key]["ci95"][0] is not None
                 and out["bins"][n][bin_key]["ci95"][0] > 0]
        r = {"d": rel["point"], "d_at_most_minus_0_01": bool(rel["point"] <= THRESHOLD),
             "diff_ci_below_zero": bool(diff["ci95"][1] < 0), "bins_worse_beyond_interval": worse}
        r["holds"] = bool(r["d_at_most_minus_0_01"] and r["diff_ci_below_zero"] and not worse)
        return r

    for n in out["bins"]:
        out["bins"][n]["lrapm_minus_raw"] = out["bins"][n]["minus_raw"]["lrapm"]
    v = {"rule": ("PLAN.md D104: lrapm adopted when, against each pre-registered baseline (raw and rapm_synergy), d <= "
                  "-0.01, the paired interval of mean weekly RMSE(lrapm) minus the baseline's lies wholly below 0, and "
                  "no bin's interval of RMSE(lrapm) minus the baseline's lies wholly above 0"),
         "against_raw": rule(out["relative_improvement_vs_raw"]["lrapm"], out["paired_minus_raw"]["lrapm"], "lrapm_minus_raw"),
         "against_rapm_synergy": rule(out["relative_improvement_vs_rapm_synergy"], out["paired_lrapm_minus_rapm_synergy"],
                                      "lrapm_minus_rapm_synergy"),
         "rapm_synergy_note": ("rapm_synergy is handicapped by construction: fixed from the development seasons, no "
                               "in-season update, its stint margin split evenly between the two sides"),
         "toward_their_players_supported": bool(out["paired_lrapm_minus_shrink_zero"]["ci95"][1] < 0)}
    v["passes"] = bool(v["against_raw"]["holds"] and v["against_rapm_synergy"]["holds"])
    if out["non_finite_rows"]:
        v = {"rule": v["rule"], "passes": None, "refused": f"{out['non_finite_rows']} row(s) carry a non-finite value"}
    out["verdict"] = v
    return out


def coverage(rows, obs_test, prior_players: set) -> dict:
    """Counts only (no outcome): how much of the predicted possessions the window and the priors know."""
    w = rows["w"].to_numpy()
    pl = obs_test[[f"{s}{k}" for s in "od" for k in range(1, 6)]].to_numpy()
    unseen = np.isin(pl, list(prior_players), invert=True).mean(axis=1)
    return {"share_of_possessions_with_a_lineup_unseen_in_earlier_weeks": float(w[rows["bin_poss"].to_numpy() == 0].sum() / w.sum()),
            "share_of_player_slots_unseen_by_the_prior_fit": float(np.average(unseen, weights=obs_test["w"].to_numpy()))}


# ----------------------------------------------------------------------------- stages

def season_run(obs_prior, feats_prior, obs_season, feats_season, target_season, lambdas=None):
    """Priors and synergy fitted on the prior seasons; the weekly window over the season. lambdas None: choose each
    ridge arm's penalty on this season (the development choice)."""
    from analytics.eval import fit_threads
    from analytics.lineups.train import fit_candidates
    with fit_threads():
        od = fit_player_od(obs_prior)
        fits = fit_candidates(feats_prior, target_season, E1.SPENT)
    syn, rapm = fits["rapm_synergy"], fits["rapm"]["player_coef"]
    chosen, curves = {}, {}
    for name in PRIOR_ARMS:
        po, pd_ = prior_source(name, od, rapm)
        if lambdas is None:
            chosen[name], curves[name] = choose_lambda(obs_season, feats_season, name, po, pd_)
        else:
            chosen[name] = float(lambdas[name])
    arms = {name: (*prior_source(name, od, rapm), chosen[name]) for name in PRIOR_ARMS}
    rows = weekly(obs_season, feats_season, arms, synergy=syn)
    weeks = week_of(obs_season["game_date"])
    info = {"prior_fit_games": od["fit_games"], "prior_players": len(od["players"]), "alpha_off": od["alpha_off"],
            "alpha_def": od["alpha_def"], "prior_intercept": od["intercept"],
            "od_net_vs_frozen_rapm_correlation": float(np.corrcoef(
                [od["off"][p] + od["def"][p] for p in od["players"] if p in rapm],
                [rapm[p] for p in od["players"] if p in rapm])[0, 1]),
            "synergy_alpha": syn["alpha"], "synergy_lineups": len(syn["lineups"]), "lambdas": chosen,
            "lambda_curves": curves, "season_weeks": int(weeks.max()) + 1, "warmup_weeks": WARMUP_WEEKS}
    return rows, info, od


def e2_dir(artifacts=None):
    from tools.frozen_common import assert_safe_output
    return assert_safe_output(E1.work_dir(artifacts).parent / "lineups_e2")


def dev_tables(artifacts, seasons):
    """The gated feature frames of lineups-E1 with each side's points joined back (checked against points_diff of
    every kept stint), and their observations."""
    feats, _games = E1.load_dev(artifacts, seasons)
    folder = e2_dir(artifacts)
    pts = pd.concat([pd.read_parquet(folder / f"stint_points_{s}.parquet") for s in seasons], ignore_index=True)
    feats = with_points(feats, pts)
    return feats, observations(feats)


def stint_points(artifacts=None) -> dict:
    """The points of each side of every development stint, rebuilt with the frozen builder from DATA_ROOT (the frozen
    feature frame keeps only their difference). Checked against the gated feature frames; written beside them."""
    import os
    root = os.environ["DATA_ROOT"]
    gated_folder, folder = E1.work_dir(artifacts), e2_dir(artifacts)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in DEV_SEASONS:
        if (folder / f"stint_points_{s}.parquet").exists():
            pts = pd.read_parquet(folder / f"stint_points_{s}.parquet")
            gated = pd.read_parquet(gated_folder / f"features_{s}.parquet")
            with_points(gated, pts)
            out[s] = {"stints": int(len(pts)), "kept_stints_matched": int(len(gated)), "cached": True}
            continue
        pbp, starters = E1.read_dev_season(root, s)
        feats, _g, _d, stints = E1.season_tables(pbp, starters)
        gated = pd.read_parquet(gated_folder / f"features_{s}.parquet")
        pts = stints[["game_id", "stint", "points_home", "points_away"]].copy()
        with_points(gated, pts)                                     # raises unless every kept stint matches
        pts.to_parquet(folder / f"stint_points_{s}.parquet", index=False)
        out[s] = {"stints": int(len(pts)), "kept_stints_matched": int(len(gated)), "cached": False}
        del pbp, starters, feats, stints
    return out


def synergy_gate(feats_dev) -> dict:
    """The frozen rapm_synergy refit on every development season against the stored synergy_per100 of the 2023-24
    lineup table (the fit on 2020-21 to 2023-24)."""
    import os
    from analytics.eval import fit_threads
    from analytics.lineups.train import fit_candidates
    with fit_threads():
        fits = fit_candidates(feats_dev, E1.SPENT, E1.SPENT)
    stored = pd.read_parquet(f"{os.environ['DATA_ROOT']}/analytics/lineups/{DRY_SEASON}/lineup_ratings.parquet",
                             columns=["lineup_id", "synergy_per100", "fit_seasons"])
    if set(stored["fit_seasons"]) != {",".join(DEV_SEASONS)}:
        raise SystemExit(f"{WHO}: the stored lineup table is not the fit on the development seasons")
    coef = fits["rapm_synergy"]["lineup_coef"]
    diff = np.abs(stored["lineup_id"].map(lambda l: coef.get(l, 0.0)).to_numpy() - stored["synergy_per100"].to_numpy())
    rec = {"compared": "rapm_synergy lineup coefficients refit on 2020-21 to 2023-24 against stored synergy_per100 "
                       "(DATA_ROOT analytics/lineups/2023-24/lineup_ratings)",
           "lineups_compared": int(len(stored)), "gated_lineups": len(coef), "alpha": fits["rapm_synergy"]["alpha"],
           "non_zero_stored_compared": int((stored["synergy_per100"] != 0).sum()),
           "max_abs_diff": float(diff.max()), "tolerance": SYNERGY_TOLERANCE}
    rec["passed"] = bool(rec["max_abs_diff"] <= SYNERGY_TOLERANCE and len(stored) > 0)
    if not rec["passed"]:
        raise SystemExit(f"{WHO}: the synergy gate failed: {rec}")
    return rec


def dev(artifacts=None) -> dict:
    t0 = time.time()
    head = clean_head()
    gate = E1.require_gate(artifacts)
    points = stint_points(artifacts)
    feats_all, obs_all = dev_tables(artifacts, DEV_SEASONS)
    syn_gate = synergy_gate(feats_all)
    by = {s: (feats_all["season"].astype(str) == s).to_numpy() for s in DEV_SEASONS}
    ob = {s: obs_all["game_id"].isin(set(feats_all.loc[by[s], "game_id"])).to_numpy() for s in DEV_SEASONS}

    def part(seasons):
        fm = np.any([by[s] for s in seasons], axis=0)
        om = np.any([ob[s] for s in seasons], axis=0)
        return feats_all[fm].reset_index(drop=True), obs_all[om].reset_index(drop=True)

    def season(s):
        f, _o = part([s])
        return f, observations(f)

    # 1. pre-registered: penalties chosen on the expanding windows inside 2023-24, priors from 2020-21 to 2022-23
    fp, op = part(DRY_PRIOR)
    fs, os_ = season(DRY_SEASON)
    rows, info, _od = season_run(op, fp, os_, fs, DRY_SEASON)
    dry = {"what": ("pre-registered: priors and synergy fitted on 2020-21 to 2022-23, every penalty chosen on the "
                    "expanding windows inside 2023-24 and scored on the same windows (the penalty is in sample)"),
           **info, "coverage": coverage(rows, os_[week_of(os_["game_date"]) >= WARMUP_WEEKS], set(_od["players"])),
           **score_rows(rows)}
    print(f"[lineups-E2] dry run {dry['verdict']} {round(time.time() - t0)} s", flush=True)
    # 2. not pre-registered, context: 2022-23 with priors from 2020-21 and 2021-22 and the 2023-24 penalties
    fp, op = part(CHECK_PRIOR)
    fs, os2 = season(CHECK_SEASON)
    rows2, info2, od2 = season_run(op, fp, os2, fs, CHECK_SEASON, lambdas=info["lambdas"])
    check = {"what": ("not pre-registered, context and power only: priors and synergy fitted on 2020-21 and 2021-22, "
                      "the 2023-24 penalties, scored on the expanding windows inside 2022-23 (penalties out of sample)"),
             **info2, "coverage": coverage(rows2, os2[week_of(os2["game_date"]) >= WARMUP_WEEKS], set(od2["players"])),
             **score_rows(rows2)}
    print(f"[lineups-E2] check season {check['verdict']} {round(time.time() - t0)} s", flush=True)
    # 3. the test fit's priors: every development season
    from analytics.eval import fit_threads
    with fit_threads():
        od_all = fit_player_od(obs_all)
    out = {"experiment": EXPERIMENT, "code_commit": head,
           "stage": "development seasons only (2020-21 to 2023-24); no 2024-25 or 2025-26 row read",
           "inputs": {"lineups_e1_gate_passed": gate["passed"], "stint_points": points, "synergy_gate": syn_gate},
           "dry_run": dry, "check_season": check,
           "test_fit": {"what": "priors and synergy fitted on every development season; the penalties of the dry run",
                        "prior_fit_games": od_all["fit_games"], "prior_players": len(od_all["players"]),
                        "alpha_off": od_all["alpha_off"], "alpha_def": od_all["alpha_def"], "lambdas": info["lambdas"]},
           "power": power(dry, check), "seconds": round(time.time() - t0, 1), "peak_memory_gb": round(peak_rss_gb(), 2)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lineups-E2-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def power(dry: dict, check: dict, test_games: int = 1300) -> dict:
    out = {"what": ("half-widths of the interval of d (lrapm against raw) in each development check, scaled to the "
                    "test game count by the square root of the game counts (an estimate, not a result)"),
           "test_games_assumed": test_games}
    for name, r in (("dry_run", dry), ("check_season", check)):
        lo, hi = r["relative_improvement_vs_raw"]["lrapm"]["ci95"]
        out[name] = {"games": r["games"], "half_width_d": (hi - lo) / 2,
                     "projected_test_half_width_d": (hi - lo) / 2 * math.sqrt(r["games"] / test_games)}
    return out


def clean_head() -> str:
    """HEAD, refused unless the whole working tree is clean."""
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first, nothing scored ({dirty[:200]})")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def make_scorer(od: dict, syn: dict, rapm: dict, lambdas: dict, commit):
    """The scorer score_once runs: the 2025-26 stints built from the pinned files with the frozen code, the weekly
    expanding window inside the season, every arm scored once. The test data is reached only here."""
    from analytics.lineups import read_season_tables

    def scorer(test_games):
        root = scoring_data_root(PBP)
        pbp = read_season_tables(root, "playbyplay", [PBP_TEST_SEASON])
        starters = read_season_tables(root, "starters", [PBP_TEST_SEASON])
        listed = set(map(str, test_games))
        pbp = pbp[pbp["game_id"].astype(str).isin(listed)].reset_index(drop=True)
        starters = starters[starters["game_id"].astype(str).isin(listed)].reset_index(drop=True)
        feats, games, dropped, stints = E1.season_tables(pbp, starters)
        feats = with_points(feats, stints)
        obs = observations(feats)
        arms = {name: (*prior_source(name, od, rapm), lambdas[name]) for name in PRIOR_ARMS}
        rows = weekly(obs, feats, arms, synergy=syn)
        with_pbp, built = set(pbp["game_id"].astype(str)), set(games["game_id"].astype(str))
        return _plain({"experiment": EXPERIMENT, "code_commit": commit,
                "test_games_listed": len(listed), "games_with_playbyplay": len(with_pbp),
                "games_listed_without_playbyplay": len(listed - with_pbp),
                "games_dropped_shifted_periods": len(dropped),
                "games_built_without_stints": len(with_pbp - built - set(map(str, dropped))),
                "games_without_kept_stints": int(len(built - set(feats["game_id"].astype(str)))),
                "stints_built": int(len(stints)), "stints_kept": int(len(feats)),
                "games_in_warmup_weeks": int(len(set(feats["game_id"].astype(str)) - set(rows["game_id"]))),
                "season_weeks": int(week_of(obs["game_date"]).max()) + 1,
                "schedule_agreement": E1.schedule_agreement(games, root, PBP_TEST_SEASON),
                "coverage": coverage(rows, obs[week_of(obs["game_date"]) >= WARMUP_WEEKS], set(od["players"])),
                "results": score_rows(rows)})                 # NaN (an empty bin) becomes null in the ledger

    return scorer


def final(artifacts=None, testset_path=None, ledger_path=None, data_root=None) -> dict:
    head = clean_head()
    E1.require_gate(artifacts)
    dev_path = RESULTS / "lineups-E2-dev.json"
    if not dev_path.exists():
        raise SystemExit(f"{WHO}: run e2-dev first; the test season is scored after the development result exists")
    planned = json.loads(dev_path.read_text())["test_fit"]
    feats, obs = dev_tables(artifacts, DEV_SEASONS)
    from analytics.eval import fit_threads
    from analytics.lineups.train import fit_candidates
    with fit_threads():
        od = fit_player_od(obs)
        fits = fit_candidates(feats, PBP_TEST_SEASON, E1.SPENT)
    if (od["alpha_off"], od["alpha_def"], len(od["players"])) != (planned["alpha_off"], planned["alpha_def"],
                                                                 planned["prior_players"]):
        raise SystemExit(f"{WHO}: the prior fit differs from the one e2-dev recorded; nothing scored")
    scorer = make_scorer(od, fits["rapm_synergy"], fits["rapm"]["player_coef"], planned["lambdas"], head)
    out = score_once(EXPERIMENT, PBP, scorer, testset_path=testset_path, ledger_path=ledger_path, data_root=data_root)
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"prior_fit_games": od["fit_games"], "prior_players": len(od["players"]),
                        "alpha_off": od["alpha_off"], "alpha_def": od["alpha_def"], "lambdas": planned["lambdas"],
                        "synergy_alpha": fits["rapm_synergy"]["alpha"], "synergy_lineups": len(fits["rapm_synergy"]["lineups"])}}
    (RESULTS / "lineups-E2-final.json").write_text(json.dumps(_plain(record), indent=2) + "\n")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e2-dev", "e2-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"e2-dev": dev, "e2-final": final}[a.command]
    with heavy_job(f"research lineups {a.command}"):
        out = run(a.artifacts)
    v = (out.get("dry_run") or out.get("scored", {}).get("results") or {}).get("verdict", {})
    print(json.dumps(_plain(v), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
