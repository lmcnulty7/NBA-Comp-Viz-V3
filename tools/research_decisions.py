"""tools/research_decisions.py: the decisions research experiment decisions-E1 (PLAN.md D70, D71, D83, D84).

    python3 -m tools.research_decisions e1-gate   [--artifacts DIR]
    python3 -m tools.research_decisions e1-dryrun [--artifacts DIR]
    python3 -m tools.research_decisions e1-dev    [--artifacts DIR]
    python3 -m tools.research_decisions e1-final  [--artifacts DIR]

Pre-registration: docs/research/decisions.md, section 6, "decisions-E1". The question: is the frozen validation target
(per-player on-court offensive rating) reliable enough to validate against, and is the frozen per-player mean decision
quality more than the player's mix of shots, passes and drives? Nothing is fitted here beyond a descriptive linear
regression of player means on kind shares.

e1-gate     The baseline: the frozen decisions lane (analytics.decisions.train.run, unchanged) rerun in this process on
            the development games only (tools.research_testset.dev_games; the 17 spent held-out games are never
            loaded), reading the stored EPV and xFG lane outputs as the frozen run did. Nothing is written under
            DATA_ROOT or reports/; the lane's command line is never run. The gate: the refit must reproduce the stored
            decision_events of every game (keys equal, values within GATE_TOLERANCE points), the stored
            decision_players table, and the validation numbers printed in reports/decisions.md (to half a unit of the
            last printed digit; this also checks the recomputed on-court ratings, which were never stored). Writes the
            refit tables under <artifacts>/research/decisions_e1_gate/ with check.json.
e1-dryrun   The test-set valuation path, run once on a few development games with the stored upstream tables in place
            of the frozen predictors, so a mechanical fault cannot spend the single test score. Prints row counts only;
            writes nothing tracked.
e1-dev      Metrics (1) to (4) on the development games, with every interval from a bootstrap of games (2000 resamples,
            seed 0). Writes reports/research/decisions-E1-dev.json.
e1-final    The same statistics once on the D71 test set (Set A) through tools.research_testset.score_once: decisions
            valued as the frozen lane valued its held-out games (value model refit on every development game, the
            frozen EPV and xFG predictors rebuilt under tools/, D45), on-court rating from the test possessions.
            Writes reports/research/decisions-E1-final.json.

Definitions, fixed in this file before any statistic was computed (the pre-registration leaves them open):
- Halves. The development games sorted by date (then id). Odd/even: alternate games (metric 1). Chronological: the
  first ceil(G/2) games against the rest (the "seasons" of Franks stability; interleaved halves would make
  stability repeat split-half reliability). The outer bootstrap resamples games within the 4 cells of the two splits,
  so every half keeps its size.
- The target (frozen): a player's value in a set of games is the mean, over their player-games with at least 3
  decisions (the frozen validation's unit, MIN_DECISIONS_GAME), of the per-game on-court points per 100 possessions.
- The metric (frozen): a player's mean decision quality over their decisions in the set of games.
- Kind-adjusted quality (metric 4): regret minus the development-game mean regret of the decisions of the same kind
  and court zone; the zone is the frozen EPV lane's zone of the decision-frame distance to the basket
  (analytics.epv.features.zone_of: rim under 4 ft, paint under 14, mid under 22, three), with a fifth zone for a
  missing distance. Ablation: the kind mean only. On the test set the development-game cell means are used.
- Player sets. Split-half and stability: players with at least N_MIN = 300 decisions in each half of that split.
  Discrimination and R squared: players with at least 300 decisions in the whole set (the frozen player table gate).
  The sets are fixed on the data and kept in every bootstrap resample.
- (1) Split-half reliability: Pearson r between the two odd/even half values across players (the H1 statistic);
  Spearman's rho and the Spearman-Brown full-length value 2r / (1 + r) are reported beside it.
- (2) Franks et al. 2016: discrimination D = 1 - mean_p BV_p / mean_p (X_p - mean X)^2 over all the games, with BV_p
  the variance of the player's value over a bootstrap of games (2000 for the point value, 200 inside each outer
  resample). Stability (Eq. 4) over the two chronological halves: S = 1 - mean_p (V_p - BVbar_p) / (V_m - mean_p
  BVbar_p), V_p the player's between-half variance and V_m the variance of all player-half values, both with ddof 1
  (with two periods the paper's 1/S_p sample moment halves V_p and biases S upward; it is reported beside as
  "printed estimator"). BVbar_p averages the player's bootstrap variance in the two halves, games resampled within
  each half. The cell means of the adjusted quality are held at the resample's values inside the inner bootstrap.
- (3) R squared of the ordinary least squares fit of the player values on [1, shot share, pass share] (drive share is
  the omitted third), players unweighted.
- Hypotheses read on the point value, with the interval reported: H1 target r < 0.2; H2 frozen S >= 0.5; H3 frozen
  R squared >= 0.5. Adoption of the kind-adjusted quality (development games): S >= 0.5 with its 95 percent interval
  above 0.3, and R squared on kind shares below 0.2. Paired test: the stability difference adjusted minus frozen.
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, _sha256, artifacts_dir, assert_safe_output, training_games_only
from tools.research_testset import TRACKING, dev_games, refuse_test, score_once

WHO = "tools.research_decisions"
EXPERIMENT = "decisions-E1"
RESULTS = REPO_ROOT / "reports" / "research"
LANE_REPORT = REPO_ROOT / "reports" / "decisions.md"
SEASON = "2015-16"
SEED, BOOT, INNER_BOOT, INNER_POINT = 0, 2000, 200, 2000
N_MIN = 300                      # the frozen player gate (analytics.decisions.train.N_MIN)
MIN_DECISIONS_GAME = 3           # the frozen validation's player-game unit
GATE_TOLERANCE = 1e-6            # points; decision values are float64 from deterministic frozen code
H1_MAX, H2_MIN, H3_MIN = 0.2, 0.5, 0.5
ADOPT_S, ADOPT_S_LO, ADOPT_R2 = 0.5, 0.3, 0.2
LOW_POWER_GAMES = 60
KINDS = ("shot", "pass", "drive")
ZONES = ("rim", "paint", "mid", "three", "none")
MEASURES = ("target", "frozen", "adjusted", "adjusted_kind")
METRICS = ("frozen", "adjusted", "adjusted_kind")
TEST_FOLD = -3                   # research label of a test-set decision (never written to a lane table)
GATE_DIR = "decisions_e1_gate"
CHECK = "check.json"
EVENT_KEYS = ["sportvu_game", "possession", "frame", "player_id", "decision_kind", "event", "receiver_id"]
EVENT_EXACT = EVENT_KEYS + ["best_option", "prior", "fold", "heldout"]
EVENT_VALUES = ["chosen_value", "best_value", "decision_quality", "shot_value", "hold_value", "drive_value"]
PLAYER_INTS = ["player_id", "n_decisions", "n_games", "n_shot", "n_pass", "n_drive", "insufficient"]
PLAYER_VALUES = ["mean_quality", "lo", "hi", "sum_quality", "mean_quality_shot", "mean_quality_pass",
                 "mean_quality_drive", "rank"]


def research_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research")


def gate_dir(artifacts=None) -> Path:
    return research_dir(artifacts) / GATE_DIR


# ----------------------------------------------------------------------------- halves and the player-game table

def game_date(game: str) -> tuple:
    """(year, month, day, id) of a SportVU-style id MM.DD.YYYY.AAA.at.BBB, for a chronological sort."""
    month, day, year = (int(x) for x in str(game).split(".")[:3])
    return (year, month, day, str(game))


def halves(games) -> dict:
    """game -> (odd/even half, chronological half), both 0 or 1, from the games sorted by date then id."""
    order = sorted((str(g) for g in games), key=game_date)
    cut = math.ceil(len(order) / 2)
    return {g: (i % 2, 0 if i < cut else 1) for i, g in enumerate(order)}


def zone_index(dist) -> "np.ndarray":
    """Index into ZONES of each decision-frame distance (the frozen EPV zones; a missing distance is 'none')."""
    import numpy as np
    from analytics.epv.features import ZONES as EPV_ZONES, zone_of
    if tuple(EPV_ZONES) != ZONES[:4]:
        raise RuntimeError(f"{WHO}: the frozen EPV zones changed: {EPV_ZONES}")
    d = np.asarray(dist, dtype=float)
    out = np.full(len(d), len(ZONES) - 1)
    ok = np.isfinite(d)
    names = zone_of(d[ok])
    out[ok] = [ZONES.index(z) for z in names]
    return out


class Table:
    """The player-game table every statistic reads: one row per (game, player) with decisions, with counts and sums
    by cell (kind x zone), and the player-game's on-court rating where it is in the frozen validation's unit."""

    def __init__(self, events, ortg, games):
        import numpy as np
        import pandas as pd
        from scipy import sparse
        games = [str(g) for g in games]
        self.games = sorted(games, key=game_date)
        gi = {g: i for i, g in enumerate(self.games)}
        ev = events[events["sportvu_game"].astype(str).isin(set(self.games))]
        kind = ev["decision_kind"].astype(str).map({k: i for i, k in enumerate(KINDS)}).to_numpy()
        if np.isnan(kind.astype(float)).any():
            raise ValueError(f"{WHO}: a decision kind outside {KINDS}")
        cell = kind.astype(int) * len(ZONES) + ev["zone"].to_numpy(dtype=int)
        self.C = len(KINDS) * len(ZONES)
        key = pd.DataFrame({"g": ev["sportvu_game"].astype(str).map(gi).to_numpy(),
                            "p": ev["player_id"].to_numpy(dtype=np.int64), "c": cell,
                            "q": ev["decision_quality"].to_numpy(dtype=float)})
        agg = key.groupby(["g", "p", "c"]).agg(n=("q", "size"), s=("q", "sum")).reset_index()
        pg = agg[["g", "p"]].drop_duplicates().sort_values(["g", "p"]).reset_index(drop=True)
        pg["r"] = np.arange(len(pg))
        agg = agg.merge(pg, on=["g", "p"])
        R = len(pg)
        self.nc = np.zeros((R, self.C))
        self.sc = np.zeros((R, self.C))
        self.nc[agg["r"].to_numpy(), agg["c"].to_numpy()] = agg["n"].to_numpy()
        self.sc[agg["r"].to_numpy(), agg["c"].to_numpy()] = agg["s"].to_numpy()
        self.kind_of_cell = np.repeat(np.arange(len(KINDS)), len(ZONES))
        self.nk = np.stack([self.nc[:, self.kind_of_cell == k].sum(axis=1) for k in range(len(KINDS))], axis=1)
        self.n = self.nc.sum(axis=1)
        self.q = self.sc.sum(axis=1)
        self.g = pg["g"].to_numpy()
        self.players = np.unique(pg["p"].to_numpy())
        self.p = np.searchsorted(self.players, pg["p"].to_numpy())
        o = ortg[["sportvu_game", "player_id", "ortg"]].copy()
        o["g"] = o["sportvu_game"].astype(str).map(gi)
        o = o.dropna(subset=["g"])
        o["g"] = o["g"].astype(int)
        o["p"] = o["player_id"].astype(np.int64)
        m = pg.merge(o[["g", "p", "ortg"]], on=["g", "p"], how="left").sort_values("r")
        r = m["ortg"].to_numpy(dtype=float)
        self.ok = (np.isfinite(r) & (self.n >= MIN_DECISIONS_GAME)).astype(float)
        self.ortg = np.where(self.ok > 0, r, 0.0)
        self.R, self.P, self.G = R, len(self.players), len(self.games)
        self.A = sparse.csr_matrix((np.ones(R), (np.arange(R), self.p)), shape=(R, self.P))
        h = halves(self.games)
        self.oddeven = np.array([h[g][0] for g in self.games])
        self.chrono = np.array([h[g][1] for g in self.games])
        self.cells4 = self.oddeven * 2 + self.chrono

    def agg(self, Wr, v):
        """Per-player sums of v (length R, or B x R) under row weights Wr (B x R) -> B x P."""
        import numpy as np
        X = Wr * (v if np.ndim(v) == 2 else v[None, :])
        return np.asarray((self.A.T @ X.T).T)


def cell_means(T: Table, W):
    """Mean regret per (kind, zone) cell and per kind under game weights W (B x G) -> (B x C, B x 3)."""
    import numpy as np
    Wr = W[:, T.g]
    num, den = Wr @ T.sc, Wr @ T.nc
    mu = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    nk = np.stack([num[:, T.kind_of_cell == k].sum(axis=1) for k in range(len(KINDS))], axis=1)
    dk = np.stack([den[:, T.kind_of_cell == k].sum(axis=1) for k in range(len(KINDS))], axis=1)
    return mu, np.divide(nk, dk, out=np.zeros_like(nk), where=dk > 0)


def player_values(T: Table, W, mu, mu_k) -> dict:
    """Every measure per player under game weights W (B x G); mu, mu_k are the cell and kind means (B x C or C)."""
    import numpy as np
    W = np.atleast_2d(W)
    mu, mu_k = np.atleast_2d(mu), np.atleast_2d(mu_k)
    Wr = W[:, T.g]
    N = T.agg(Wr, T.n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"N": N, "frozen": T.agg(Wr, T.q) / N,
               "adjusted": T.agg(Wr, T.q[None, :] - mu @ T.nc.T) / N,
               "adjusted_kind": T.agg(Wr, T.q[None, :] - mu_k @ T.nk.T) / N,
               "target": T.agg(Wr, T.ok * T.ortg) / T.agg(Wr, T.ok)}
        out["shares"] = [T.agg(Wr, T.nk[:, k]) / N for k in range(len(KINDS))]
    return out


def resample(rng, w, strata, B: int):
    """B multinomial resamples of the games (weights w, counts of each game) within each stratum -> B x G counts."""
    import numpy as np
    W = np.zeros((B, len(w)))
    for s in np.unique(strata):
        idx = np.flatnonzero((strata == s) & (w > 0))
        if not len(idx):
            continue
        total = int(round(w[idx].sum()))
        W[:, idx] = rng.multinomial(total, w[idx] / w[idx].sum(), size=B)
    return W


# ----------------------------------------------------------------------------- the statistics

def pearson(a, b) -> float:
    import numpy as np
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a, b) -> float:
    import numpy as np
    from scipy.stats import rankdata
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return pearson(rankdata(a[m]), rankdata(b[m]))


def r_squared(y, X) -> float:
    """R squared of the OLS fit of y on [1, X] (rows with a missing value dropped)."""
    import numpy as np
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < X.shape[1] + 2:
        return float("nan")
    D = np.column_stack([np.ones(int(m.sum())), X[m]])
    beta, *_ = np.linalg.lstsq(D, y[m], rcond=None)
    resid = y[m] - D @ beta
    tss = float(((y[m] - y[m].mean()) ** 2).sum())
    return float(1 - (resid ** 2).sum() / tss) if tss > 0 else float("nan")


def discrimination(X, BV) -> float:
    """Franks et al. 2016, Eq. 2 with the Section 3 sample moments: 1 - mean BV / mean (X - mean X)^2."""
    import numpy as np
    m = np.isfinite(X) & np.isfinite(BV)
    if m.sum() < 3:
        return float("nan")
    total = float(np.mean((X[m] - X[m].mean()) ** 2))
    return float(1 - np.mean(BV[m]) / total) if total > 0 else float("nan")


def stability(X0, X1, BV0, BV1, printed: bool = False) -> float:
    """Franks et al. 2016, Eq. 4 over two periods. ddof 1 for the variances unless `printed` (the paper's 1/S_p and
    1/N sample moments, biased upward with two periods)."""
    import numpy as np
    m = np.isfinite(X0) & np.isfinite(X1) & np.isfinite(BV0) & np.isfinite(BV1)
    if m.sum() < 3:
        return float("nan")
    x0, x1 = X0[m], X1[m]
    bv = (BV0[m] + BV1[m]) / 2
    vp = (x0 - x1) ** 2 / (4 if printed else 2)
    vm = float(np.var(np.concatenate([x0, x1]), ddof=0 if printed else 1))
    denom = vm - float(bv.mean())
    return float(1 - (vp - bv).mean() / denom) if denom > 0 else float("nan")


def player_sets(T: Table) -> dict:
    """The fixed player sets (boolean over T.players), from the unweighted data."""
    import numpy as np
    one = np.ones((1, T.G))
    n = lambda mask: T.agg(one[:, T.g] * mask[T.g][None, :], T.n)[0]
    every = np.ones(T.G)
    return {"all": n(every) >= N_MIN,
            "oddeven": (n((T.oddeven == 0).astype(float)) >= N_MIN) & (n((T.oddeven == 1).astype(float)) >= N_MIN),
            "chrono": (n((T.chrono == 0).astype(float)) >= N_MIN) & (n((T.chrono == 1).astype(float)) >= N_MIN)}


def statistics(T: Table, w, sets: dict, rng, inner: int, mu_fixed=None) -> dict:
    """Every statistic of metrics (1) to (4) on one data set: the games with weights w (counts; all ones for the data
    itself, a resample otherwise). mu_fixed: the cell and kind means to use (the test set uses the development
    games' means); otherwise they are computed from w."""
    import numpy as np
    w = np.asarray(w, dtype=float)
    mu, mu_k = mu_fixed if mu_fixed is not None else (x[0] for x in cell_means(T, w[None, :]))
    vals = lambda W: player_values(T, W, mu, mu_k)
    mask = lambda arr, h: w * (arr == h)
    full = vals(w[None, :])
    oe = [vals(mask(T.oddeven, h)[None, :]) for h in (0, 1)]
    ch = [vals(mask(T.chrono, h)[None, :]) for h in (0, 1)]
    inner_all = vals(resample(rng, w, np.zeros(T.G, dtype=int), inner))
    inner_ch = [vals(resample(rng, mask(T.chrono, h), np.zeros(T.G, dtype=int), inner)) for h in (0, 1)]
    a, o, c = sets["all"], sets["oddeven"], sets["chrono"]
    out = {}
    with np.errstate(invalid="ignore"):
        for m in MEASURES:
            x0, x1 = oe[0][m][0][o], oe[1][m][0][o]
            r = pearson(x0, x1)
            bv_all = np.nanvar(inner_all[m][:, a], axis=0, ddof=1)
            bv0 = np.nanvar(inner_ch[0][m][:, c], axis=0, ddof=1)
            bv1 = np.nanvar(inner_ch[1][m][:, c], axis=0, ddof=1)
            X0, X1 = ch[0][m][0][c], ch[1][m][0][c]
            out[m] = {"split_half_r": r, "split_half_spearman": spearman(x0, x1),
                      "split_half_spearman_brown": float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else float("nan"),
                      "discrimination": discrimination(full[m][0][a], bv_all),
                      "stability": stability(X0, X1, bv0, bv1),
                      "stability_printed_estimator": stability(X0, X1, bv0, bv1, printed=True),
                      "sd_between_players": float(np.nanstd(full[m][0][a], ddof=1)) if a.sum() > 1 else float("nan")}
            if m != "target":
                S = np.column_stack([full["shares"][0][0][a], full["shares"][1][0][a]])
                out[m]["r2_on_kind_shares"] = r_squared(full[m][0][a], S)
    for m in ("adjusted", "adjusted_kind"):
        out[f"stability_{m}_minus_frozen"] = out[m]["stability"] - out["frozen"]["stability"]
    return out


def _flat(stats: dict) -> dict:
    return {f"{m}.{k}" if isinstance(v, dict) else m: x for m, v in stats.items()
            for k, x in (v.items() if isinstance(v, dict) else [(None, v)])}


def summarize(T: Table, mu_fixed=None, n: int = BOOT, inner: int = INNER_BOOT, seed: int = SEED, log=print) -> dict:
    """The statistics on the data with a 95 percent percentile interval for each from n outer resamples of games
    (stratified by the 4 half cells), each with its own inner bootstrap of `inner` resamples."""
    import numpy as np
    sets = player_sets(T)
    w = np.ones(T.G)
    point = statistics(T, w, sets, np.random.default_rng([seed, 1]), INNER_POINT, mu_fixed)
    outer = resample(np.random.default_rng(seed), w, T.cells4, n)
    draws = {k: [] for k in _flat(point)}
    started = time.time()
    for b in range(n):
        s = _flat(statistics(T, outer[b], sets, np.random.default_rng([seed, 2, b]), inner, mu_fixed))
        for k in draws:
            draws[k].append(s[k])
        if log and (b + 1) % 250 == 0:
            log(f"[research] {b + 1} of {n} resamples, {time.time() - started:.0f} s")
    flat = _flat(point)
    intervals = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        ok = v[np.isfinite(v)]
        intervals[k] = {"value": flat[k],
                        "lo": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                        "hi": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                        "finite_resamples": int(len(ok))}
    mu, mu_k = mu_fixed if mu_fixed is not None else (x[0] for x in cell_means(T, w[None, :]))
    return {"games": T.G, "decisions": int(T.n.sum()), "player_games": T.R, "players_with_decisions": T.P,
            "players": {k: int(v.sum()) for k, v in sets.items()},
            "games_per_half": {"oddeven": [int((T.oddeven == h).sum()) for h in (0, 1)],
                               "chrono": [int((T.chrono == h).sum()) for h in (0, 1)]},
            "chrono_split": {"first_half_last_game": T.games[int((T.chrono == 0).sum()) - 1],
                             "second_half_first_game": T.games[int((T.chrono == 0).sum())]},
            "bootstrap": {"outer_resamples": n, "inner_resamples": inner, "inner_resamples_point": INNER_POINT,
                          "seed": seed, "unit": "game", "outer_strata": "odd/even half x chronological half"},
            "statistics": intervals, "cell_means": cell_table(T, mu, mu_k)}


def cell_table(T: Table, mu, mu_k) -> list:
    import numpy as np
    n = T.nc.sum(axis=0)
    return [{"kind": KINDS[c // len(ZONES)], "zone": ZONES[c % len(ZONES)], "decisions": int(n[c]),
             "mean_regret": float(np.asarray(mu)[c]), "kind_mean_regret": float(np.asarray(mu_k)[c // len(ZONES)])}
            for c in range(T.C)]


def verdict(stats: dict) -> dict:
    """The pre-registered reading of the development statistics (point values; intervals reported beside)."""
    v = lambda k: stats[k]["value"]
    iv = lambda k: [stats[k]["lo"], stats[k]["hi"]]
    h1 = v("target.split_half_r") < H1_MAX
    out = {"H1": {"statement": f"target split-half r below {H1_MAX}", "value": v("target.split_half_r"),
                  "interval_95": iv("target.split_half_r"), "holds": bool(h1)},
           "H2": {"statement": f"frozen metric stability at least {H2_MIN}", "value": v("frozen.stability"),
                  "interval_95": iv("frozen.stability"), "holds": bool(v("frozen.stability") >= H2_MIN)},
           "H3": {"statement": f"frozen metric R squared on kind shares at least {H3_MIN}",
                  "value": v("frozen.r2_on_kind_shares"), "interval_95": iv("frozen.r2_on_kind_shares"),
                  "holds": bool(v("frozen.r2_on_kind_shares") >= H3_MIN)}}
    for name, h in out.items():
        lo, hi = h["interval_95"]
        thr = H1_MAX if name == "H1" else (H2_MIN if name == "H2" else H3_MIN)
        h["interval_crosses_threshold"] = bool(lo < thr < hi)
    s, (s_lo, _), r2 = v("adjusted.stability"), iv("adjusted.stability"), v("adjusted.r2_on_kind_shares")
    out["adoption"] = {"rule": f"kind-adjusted stability >= {ADOPT_S} with 95 percent interval above {ADOPT_S_LO}, "
                               f"and its R squared on kind shares below {ADOPT_R2}",
                       "stability": s, "stability_lo": s_lo, "r2_on_kind_shares": r2,
                       "adopted": bool(s >= ADOPT_S and s_lo > ADOPT_S_LO and r2 < ADOPT_R2)}
    out["protocol"] = ("H1 holds: the lane report adds, next to the unchanged failed verdict, that the frozen "
                       "validation could not have detected a signal" if h1 else
                       "H1 fails: the target is reliable, so the failed verdict is read as evidence against the metric")
    return out


# ----------------------------------------------------------------------------- the gate (baseline refit)

def _stored_events(root: Path, games) -> "pd.DataFrame":
    import pandas as pd
    d = root / "analytics" / "decisions" / SEASON / "decision_events"
    return pd.concat([pd.read_parquet(d / f"{g}.parquet") for g in games], ignore_index=True)


def printed_validation(text: str) -> dict:
    """The validation rows printed in the lane report's training section: prior -> numbers as printed (strings)."""
    section = text.split("## Validation: mean decision quality per player-game", 1)[1].split("\n## ", 1)[0]
    out = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] in ("location", "xfg_selected"):
            pos, signed = (x.strip() for x in cells[5].split("/"))
            out[cells[0]] = {"games": cells[1], "mean_rho": cells[2], "lo": cells[3], "hi": cells[4],
                             "n_positive": pos, "signed": signed, "sign_p": cells[6], "shuffled_mean_rho": cells[7]}
    return out


def half_unit(printed: str) -> float:
    """Half a unit of the last printed digit of a number."""
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return 0.5 * 10 ** (-decimals)


def compare_printed(printed: dict, summary: dict, shuffled: dict) -> dict:
    """The recomputed validation numbers against the printed ones; integers must be equal."""
    recomputed = {"games": summary["n_games"] + 0, "mean_rho": summary["mean_rho"], "lo": summary["lo"],
                  "hi": summary["hi"], "n_positive": summary["n_positive"],
                  "signed": summary["n_positive"] + summary["n_negative"], "sign_p": summary["sign_p"],
                  "shuffled_mean_rho": shuffled["mean_rho"]}
    rows, ok = {}, True
    for k, p in printed.items():
        r = recomputed[k]
        if k in ("games", "n_positive", "signed"):
            good = int(p) == int(r)
        else:
            good = abs(float(p) - float(r)) <= half_unit(p) + 1e-12
        rows[k] = {"printed": p, "recomputed": r, "ok": bool(good)}
        ok &= good
    return {"rows": rows, "passed": bool(ok)}


def compare_events(stored, refit) -> dict:
    """Row-aligned comparison per game (both in the order the frozen code writes them)."""
    import numpy as np
    out = {"n": int(len(refit)), "n_expected": int(len(stored)), "exact_mismatches": {}, "max_abs_diff": {}}
    if len(stored) != len(refit):
        out["passed"] = False
        return out
    a, b = stored.reset_index(drop=True), refit.reset_index(drop=True)
    for c in EVENT_EXACT:
        out["exact_mismatches"][c] = int((a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy()).sum())
    worst = 0.0
    for c in EVENT_VALUES:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
        both = np.isfinite(x) & np.isfinite(y)
        d = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
        out["max_abs_diff"][c] = d
        out["exact_mismatches"][c + "_nan"] = nan_mismatch
        worst = max(worst, d)
    out["worst_abs_diff"] = worst
    out["passed"] = bool(not any(out["exact_mismatches"].values()) and worst <= GATE_TOLERANCE)
    return out


def compare_players(stored, refit) -> dict:
    import numpy as np
    a = stored.sort_values("player_id").reset_index(drop=True)
    b = refit.sort_values("player_id").reset_index(drop=True)
    out = {"n": int(len(b)), "n_expected": int(len(a)), "exact_mismatches": {}, "max_abs_diff": {}}
    if len(a) != len(b):
        out["passed"] = False
        return out
    for c in PLAYER_INTS:
        out["exact_mismatches"][c] = int((a[c].astype(int).to_numpy() != b[c].astype(int).to_numpy()).sum())
    worst = 0.0
    for c in PLAYER_VALUES:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        out["exact_mismatches"][c + "_nan"] = int((np.isnan(x) != np.isnan(y)).sum())
        both = np.isfinite(x) & np.isfinite(y)
        d = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
        out["max_abs_diff"][c] = d
        worst = max(worst, d)
    out["worst_abs_diff"] = worst
    out["passed"] = bool(not any(out["exact_mismatches"].values()) and worst <= GATE_TOLERANCE)
    return out


def with_zones(events, contexts: dict):
    """The events with the decision-frame distance and zone of their context (joined on the context keys, which are
    unique per game; option_values keeps the contexts whose chosen value is finite, in order)."""
    import pandas as pd
    keys = EVENT_KEYS[1:]
    ints = {k: "int64" for k in keys if k != "decision_kind"}

    def norm(df):
        out = df[keys].astype(ints)
        out["decision_kind"] = out["decision_kind"].astype(str)
        return out

    parts = []
    for g, ev in events.groupby("sportvu_game", sort=False):
        ctx = contexts[str(g)]
        c = norm(ctx)
        c["dist_to_basket_ft"] = ctx["dist_to_basket_ft"].to_numpy(dtype=float)
        c["_hit"] = True
        if c.duplicated(keys).any():
            raise ValueError(f"{WHO}: duplicate context keys in {g}")
        e = ev.reset_index(drop=True).copy()
        m = norm(e).merge(c, on=keys, how="left", validate="one_to_one")
        if len(m) != len(e) or not m["_hit"].fillna(False).astype(bool).all():
            raise ValueError(f"{WHO}: events of {g} do not join their contexts")
        e["dist_to_basket_ft"] = m["dist_to_basket_ft"].to_numpy()
        parts.append(e)
    out = pd.concat(parts, ignore_index=True)
    out["zone"] = zone_index(out["dist_to_basket_ft"])
    return out


def _check_record(folder: Path, record: dict, files) -> Path:
    record = _plain(dict(record))
    record["files"] = {f: _sha256(folder / f) for f in files}
    record["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["peak_memory_gb"] = round(peak_rss_gb(), 2)
    path = folder / CHECK
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return path


def require_gate(artifacts=None) -> dict:
    """The gate record, or RuntimeError when the refit is missing, did not reproduce the frozen record, or a file it
    vouches for changed after it was written (the same rules as tools.frozen_common.require_passed)."""
    folder = gate_dir(artifacts)
    path = folder / CHECK
    if not path.exists():
        raise RuntimeError(f"{EXPERIMENT}: no gate record (run e1-gate first)")
    record = json.loads(path.read_text())
    if record.get("passed") is not True or not record.get("files"):
        raise RuntimeError(f"{EXPERIMENT}: the refit did not reproduce the frozen decisions record, so it is not used")
    for name, digest in record["files"].items():
        if not (folder / name).exists() or _sha256(folder / name) != digest:
            raise RuntimeError(f"{EXPERIMENT}: {name} changed after the gate record was written")
    return record


def gate(artifacts=None) -> dict:
    import os
    import pandas as pd
    from analytics.decisions import train as DT
    from analytics.decisions import validate as V
    root = Path(os.environ["DATA_ROOT"])
    folder = assert_safe_output(gate_dir(artifacts))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CHECK).write_text(json.dumps({"passed": False, "status": "refit started"}) + "\n")
    candidates = sorted(p.stem for p in (root / "analytics" / "possessions" / SEASON).glob("*.parquet"))
    games = dev_games(candidates, WHO)
    training_games_only(games, WHO)
    started = time.time()
    bundles = V.load_root_bundles(root, SEASON, root, games=games)
    if any(b["heldout"] for b in bundles) or sorted(b["sportvu_game"] for b in bundles) != games:
        raise SystemExit(f"{WHO}: the loaded bundles are not exactly the development games")
    # the frozen run's settings (reports/decisions.md header: --seed 0, default --folds 5 and --bootstrap 200)
    res = DT.run(bundles, n_folds=5, seed=0, n_boot=200, skip_heldout=True)
    del bundles
    stored_games = sorted(p.stem for p in (root / "analytics" / "decisions" / SEASON / "decision_events").glob("*.parquet"))
    refit = res["events"]
    order = {g: i for i, g in enumerate(stored_games)}
    refit_sorted = refit.assign(_o=refit["sportvu_game"].astype(str).map(order)).sort_values("_o", kind="stable").drop(columns="_o")
    ev_check = compare_events(_stored_events(root, stored_games), refit_sorted) if games == stored_games else \
        {"passed": False, "games": [len(games), len(stored_games)]}
    pl_check = compare_players(pd.read_parquet(root / "analytics" / "decisions" / SEASON / "decision_players.parquet"),
                               DT._cast_players(res["players"]))
    printed = printed_validation(LANE_REPORT.read_text())
    val_check = {p: compare_printed(printed[p], res["validation"][p]["summary"], res["validation"][p]["shuffled"])
                 for p in ("location", "xfg_selected")}
    events = with_zones(refit_sorted, res["contexts"])
    events.to_parquet(folder / "events.parquet", index=False)
    res["ortg"].to_parquet(folder / "ortg.parquet", index=False)
    with open(folder / "value_model.pkl", "wb") as fh:
        pickle.dump(res["full_model"], fh)
    passed = bool(ev_check["passed"] and pl_check["passed"] and all(v["passed"] for v in val_check.values())
                  and res["prior"] == "xfg_selected")
    record = {"experiment": EXPERIMENT, "name": "decisions",
              "compared": "the frozen decisions lane rerun in-process on the development games (seed 0, 5 grouped "
                          "folds, bootstrap 200, stored EPV and xFG outputs) against the stored decision_events, the "
                          "stored decision_players and the validation numbers printed in reports/decisions.md; no "
                          "held-out game loaded, nothing scored",
              "games": len(games), "prior": res["prior"], "xfg_coverage": res["xfg_coverage"],
              "tolerance_points": GATE_TOLERANCE, "events": ev_check, "players": pl_check, "validation": val_check,
              "seconds": round(time.time() - started, 1), "passed": passed}
    _check_record(folder, record, ["events.parquet", "ortg.parquet", "value_model.pkl"])
    if not passed:
        print(json.dumps(_plain(record), indent=2, default=str))
        raise SystemExit(f"{WHO}: the refit did not reproduce the frozen decisions record; nothing downstream runs")
    return require_gate(artifacts)


# ----------------------------------------------------------------------------- development and test statistics

def _gate_tables(artifacts=None):
    import pandas as pd
    require_gate(artifacts)
    folder = gate_dir(artifacts)
    events = pd.read_parquet(folder / "events.parquet")
    ortg = pd.read_parquet(folder / "ortg.parquet")
    games = sorted(events["sportvu_game"].astype(str).unique())
    training_games_only(games, WHO)
    refuse_test(games, WHO)
    if sorted(dev_games(games, WHO)) != games:
        raise SystemExit(f"{WHO}: the refit holds a game that is not a development game")
    return events, ortg, games


def dev(artifacts=None) -> dict:
    events, ortg, games = _gate_tables(artifacts)
    T = Table(events, ortg, games)
    s = summarize(T)
    out = {"experiment": EXPERIMENT, "stage": "development games (the frozen decisions output refit with frozen code; "
                                              "gate passed)", **s, "verdict": verdict(s["statistics"])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def value_games(bundles, model, epv_of, xfg_table, fold: int, log=None):
    """The frozen valuation of each game (analytics.decisions.features, unchanged) as the frozen lane values a game
    it never fitted on: contexts, the hold value from epv_of(cleaned bundle), the shot prior from xfg_table (the
    xfg_selected prior), the value model `model`; plus the on-court rating -> (events with zones, ortg)."""
    import pandas as pd
    from analytics.decisions import features as F
    from analytics.decisions import validate as V
    events, ortg, contexts = [], [], {}

    def per_game(c):
        g = c["sportvu_game"]
        ctx = F.decision_contexts(c, {})
        F.assert_decisions_before_outcome(ctx, c["shot_frames"], c["possessions"])
        F.feature_frame(ctx)
        hold = F.hold_values(ctx, epv_of(c))
        x = xfg_table[xfg_table["sportvu_game"].astype(str) == g] if xfg_table is not None else None
        ev = F.option_values(ctx, model, hold=hold, p_shot_override=F.xfg_override(ctx, x), prior="xfg_selected")
        ev["fold"] = fold
        events.append(ev)
        contexts[g] = ctx
        ortg.append(F.on_court_offensive_rating(V.frames_of(c), c["possessions"], g))
        if log:
            log(f"[research] valued a game: {len(ev)} decisions, hold present on {int(hold.notna().sum())}")

    V.stream_bundles(bundles, per_game)
    ev = pd.concat(events, ignore_index=True)
    return with_zones(ev, contexts), pd.concat(ortg, ignore_index=True)


def _load_value_model(artifacts=None):
    require_gate(artifacts)
    with open(gate_dir(artifacts) / "value_model.pkl", "rb") as fh:
        return pickle.load(fh)


def dryrun(artifacts=None, n_games: int = 3) -> dict:
    """The test valuation path on the first n development games, with the stored EPV and xFG tables (out-of-fold
    rows) in place of the frozen predictors. Mechanics only: counts are printed, nothing tracked is written."""
    import os
    import numpy as np
    import pandas as pd
    from analytics.decisions import features as F
    from analytics.decisions import validate as V
    events, ortg, games = _gate_tables(artifacts)
    root = Path(os.environ["DATA_ROOT"])
    pick = games[:n_games]
    bundles = V.load_root_bundles(root, SEASON, root, games=pick)
    xfg = pd.concat([F.upstream_rows(b["xfg"], False) for b in bundles if b.get("xfg") is not None], ignore_index=True)
    epv = {b["sportvu_game"]: F.upstream_rows(b["epv"], False) for b in bundles}
    ev, o = value_games(bundles, _load_value_model(artifacts), lambda c: epv[c["sportvu_game"]], xfg, TEST_FOLD, print)
    ref = events[events["sportvu_game"].isin(pick)]
    T = Table(ev, o, pick)
    dev_mu = cell_means(Table(events, ortg, games), np.ones((1, len(games))))
    s = statistics(T, np.ones(T.G), player_sets(T), np.random.default_rng(0), 20, (dev_mu[0][0], dev_mu[1][0]))
    return {"games": len(pick), "decisions": int(len(ev)), "decisions_in_refit": int(len(ref)),
            "same_keys": bool((ev[EVENT_KEYS].astype(str).to_numpy() == ref[EVENT_KEYS].astype(str).to_numpy()).all())
            if len(ev) == len(ref) else False, "ortg_rows": int(len(o)), "statistics_keys": sorted(_flat(s))[:5]}


def final(artifacts=None) -> dict:
    import numpy as np
    import pandas as pd
    from analytics.decisions import validate as V
    from tools import frozen_epv
    from tools.frozen_models import load_xfg
    from tools.research_xfg import research_data_root, test_root_ready, test_features
    if not (RESULTS / f"{EXPERIMENT}-dev.json").exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    events, ortg, games = _gate_tables(artifacts)
    dev_T = Table(events, ortg, games)
    mu, mu_k = (x[0] for x in cell_means(dev_T, np.ones((1, dev_T.G))))
    model = _load_value_model(artifacts)
    epv_model = frozen_epv.load(artifacts)
    xfg_model = load_xfg(artifacts)
    if not test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")

    def scorer(test_games):
        root = research_data_root(artifacts)
        feats = test_features(test_games, artifacts)
        xfg = pd.DataFrame({"sportvu_game": feats["sportvu_game"].astype(str).to_numpy(),
                            "event": feats["event"].to_numpy(), "p_selected": xfg_model.predict(feats)})
        bundles = V.load_root_bundles(root, SEASON, root, games=list(test_games))
        if sorted(b["sportvu_game"] for b in bundles) != sorted(test_games) or any(b["heldout"] for b in bundles):
            raise SystemExit(f"{WHO}: the test bundles are not exactly the test games; score them all or none")
        ev, o = value_games(bundles, model, lambda c: epv_model.predict_states(V.frames_of(c), c["possessions"],
                                                                             c["sportvu_game"]), xfg, TEST_FOLD, print)
        T = Table(ev, o, test_games)
        s = summarize(T, mu_fixed=(mu, mu_k))
        return {"experiment": EXPERIMENT, "stage": "valued as the frozen lane values unseen games (value model refit on "
                                                  "every development game, frozen EPV and xFG predictors), scored once on "
                                                  "the D71 test set",
                "low_powered": bool(T.G < LOW_POWER_GAMES),
                "hold_present_share": float(np.isfinite(ev["hold_value"].to_numpy(dtype=float)).mean()),
                "kind_counts": ev["decision_kind"].value_counts().to_dict(), **s}

    out = score_once(EXPERIMENT, TRACKING, scorer)
    (RESULTS / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def posthoc(artifacts=None) -> dict:
    """Descriptive numbers written after the development result, for reading it; none enters a verdict.
    (a) The most decisions of any player in one development game, against the 300-decision gate: whether the
    pre-registered statistics can exist on a test set in which every team plays one game. (b) The per-game
    reliability that the split-half r of the target implies by the Spearman-Brown formula stepped down from the mean
    number of qualifying games per player per half: r / (k - (k - 1) r)."""
    import numpy as np
    events, ortg, games = _gate_tables(artifacts)
    dev_out = json.loads((RESULTS / f"{EXPERIMENT}-dev.json").read_text())
    T = Table(events, ortg, games)
    sets = player_sets(T)
    per_pg = events.groupby(["sportvu_game", "player_id"]).size()
    k = np.concatenate([np.bincount(T.p[(T.oddeven[T.g] == h) & (T.ok > 0)], minlength=T.P)[sets["oddeven"]]
                        for h in (0, 1)])
    kbar = float(k.mean())
    out = {"experiment": EXPERIMENT, "stage": "post hoc, descriptive, written after the development result; no verdict "
                                              "reads these numbers"}
    out["player_game_decisions"] = {"max": int(per_pg.max()), "p99": float(per_pg.quantile(0.99)),
                                    "player_games": int(len(per_pg)), "gate": N_MIN}
    r = dev_out["statistics"]["target.split_half_r"]["value"]
    out["target_per_game_reliability"] = {
        "split_half_r": r, "mean_qualifying_games_per_player_half": kbar,
        "median_qualifying_games_per_player_half": float(np.median(k)),
        "spearman_brown_one_game": float(r / (kbar - (kbar - 1) * r))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-posthoc.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-gate", "e1-dryrun", "e1-dev", "e1-posthoc", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(argv)
    run = {"e1-gate": gate, "e1-dryrun": dryrun, "e1-dev": dev, "e1-posthoc": posthoc, "e1-final": final}[a.command]
    with heavy_job(f"research decisions {a.command}"):
        out = run(a.artifacts)
    shown = out.get("verdict", out) if isinstance(out, dict) else out
    if a.command == "e1-final":
        shown = {"low_powered": out.get("low_powered"), "players": out.get("players")}
    print(json.dumps(_plain(shown), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
