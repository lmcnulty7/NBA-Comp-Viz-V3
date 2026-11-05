"""analytics/matchups/eda.py: tables for notebooks/eda_matchups.ipynb (no plotting; output is tables and markdown).

    partial_possession_distribution(mp)      histogram of partial_possession in ten equal-width bins
    switches_per_possession(mp)              distribution of switches per defender-possession
    defended_fg_by_distance(shots)           FG% by distance bin between the shooter and its assigned defender
    shuffled_assignment_check(shots, mf, seed)  the within-4-ft minus beyond-4-ft FG% gap under the real
                                             assignment vs under assignments shuffled among the defenders of
                                             the same sampled frame (the gap should shrink toward zero when
                                             the assignment carries information)
    fixture_tables(seed, n_games)            every table above on the synthetic season (fixture output, not a result)

Everything here reads the tables analytics.matchups.features builds; nothing is fitted. Numbers from the
fixtures are fixture output, not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.matchups import features as F

DIST_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, np.inf)


def partial_possession_distribution(mp: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Rows: bin, lo, hi, n, share of rows, mean switches in the bin."""
    s = mp["partial_possession"].to_numpy(dtype=float)
    b = np.clip(np.floor(s * bins), 0, bins - 1).astype(int)
    rows = []
    for k in range(bins):
        m = b == k
        rows.append({"bin": k, "lo": k / bins, "hi": (k + 1) / bins, "n": int(m.sum()),
                     "share": float(m.mean()) if len(s) else np.nan,
                     "mean_switches": float(mp.loc[m, "switches"].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def switches_per_possession(mp: pd.DataFrame) -> pd.DataFrame:
    """Rows: switches, n defender-possessions, share."""
    d = mp.groupby(["sportvu_game", "possession", "def_player_id"])["switches"].first()
    c = d.value_counts().sort_index()
    return pd.DataFrame({"switches": c.index.astype(int), "n": c.to_numpy(), "share": (c / c.sum()).to_numpy()})


def defended_fg_by_distance(shots: pd.DataFrame, edges=DIST_BINS) -> pd.DataFrame:
    """Rows: lo, hi, attempts, fg_pct, mean_xfg (the p column), per distance bin to the assigned defender."""
    d = shots.dropna(subset=["def_dist_ft"])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d["def_dist_ft"] >= lo) & (d["def_dist_ft"] < hi)
        n = int(m.sum())
        rows.append({"lo": lo, "hi": hi, "attempts": n, "fg_pct": float(d.loc[m, "made"].astype(float).mean()) if n else np.nan,
                     "mean_xfg": float(d.loc[m, "p"].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def _gap(shots: pd.DataFrame, dist_col: str) -> float:
    near = shots[dist_col] <= F.DEFENDED_FT
    if near.sum() == 0 or (~near).sum() == 0:
        return float("nan")
    return float(shots.loc[near, "made"].astype(float).mean() - shots.loc[~near, "made"].astype(float).mean())


def shuffled_assignment_check(shots: pd.DataFrame, mf: pd.DataFrame, seed: int = 0, n_shuffles: int = 20) -> pd.DataFrame:
    """The FG% gap (within 4 ft minus beyond 4 ft of the assigned defender) under the real assignment and under
    n_shuffles shuffles in which each attempt takes the defender distance of a defender drawn at random among
    those present at the attempt's sampled frame (the distance distribution is kept, who is counted as guarding
    the shooter is broken). Rows: kind, gap, lo, hi, n. When the assignment carries information the real gap
    is more negative than the shuffled band."""
    d = shots.dropna(subset=["def_dist_ft", "shooter_id"]).copy()
    real = _gap(d, "def_dist_ft")
    rng = np.random.default_rng(seed)
    by_frame = {k: g["dist_ft"].to_numpy(dtype=float) for k, g in mf.groupby(["sportvu_game", "possession", "frame"])}
    frames_of = {k: np.sort(g["frame"].unique()) for k, g in mf.groupby(["sportvu_game", "possession"])}
    cands = []
    for s in d.itertuples(index=False):
        frames = frames_of.get((s.sportvu_game, s.possession))
        if frames is None:
            cands.append(np.array([]))
            continue
        before = frames[frames <= int(s.release_frame)] if int(s.release_frame) >= 0 else frames
        fr = int(before[-1]) if len(before) else int(frames[0])
        cands.append(by_frame.get((s.sportvu_game, s.possession, fr), np.array([])))
    gaps = []
    for _ in range(n_shuffles):
        d["shuffled_dist_ft"] = [float(c[rng.integers(0, len(c))]) if len(c) else np.nan for c in cands]
        gaps.append(_gap(d.dropna(subset=["shuffled_dist_ft"]), "shuffled_dist_ft"))
    gaps = np.array(gaps, dtype=float)
    return pd.DataFrame([{"kind": "real assignment", "gap": real, "lo": np.nan, "hi": np.nan, "n": int(len(d))},
                         {"kind": "shuffled assignment", "gap": float(np.nanmean(gaps)), "lo": float(np.nanpercentile(gaps, 2.5)),
                          "hi": float(np.nanpercentile(gaps, 97.5)), "n": int(len(d))}])


def fixture_tables(seed: int = 0, n_games: int = 2) -> dict:
    """All EDA tables on the synthetic season (fixture output, not a result)."""
    from analytics.matchups import validate as V
    games = [g for g in V.fixture_games(seed, n_games, 0) if not g["heldout"]]
    res = F.run_games(games)
    mp, mf, shots = res["matchup_possessions"], res["matchup_frames"], res["shots"]
    return {"partial_possession": partial_possession_distribution(mp), "switches": switches_per_possession(mp),
            "defended_fg": defended_fg_by_distance(shots), "shuffled": shuffled_assignment_check(shots, mf, seed),
            "n_rows": int(len(mp)), "n_games": len(games)}


__all__ = ["partial_possession_distribution", "switches_per_possession", "defended_fg_by_distance",
           "shuffled_assignment_check", "fixture_tables"]
