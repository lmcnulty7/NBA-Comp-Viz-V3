"""analytics/lineups/eda.py: tables for notebooks/eda_lineups.ipynb (no plots: the environment has no matplotlib).

Every function takes the stints table or the feature frame (analytics.lineups.features.features) and returns
a DataFrame the notebook prints. Nothing here fits anything.

    duration_summary(stints)             per season: stints, duration quantiles, share of lineup_ok rows
    differential_summary(feats)          per season: quantiles of points_diff and diff_per100 (weighted mean)
    lineup_possessions(feats, gate)      per season: lineups seen, how many pass the possession gate, quantiles
    player_minutes(stints, top)          per season: the top players by minutes on court (from stint durations)
    cross_season_correlation(feats, min_poss)   raw lineup net rating of consecutive seasons, lineups with at
                                         least min_poss possessions in both: Pearson and Spearman
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analytics.lineups import PLAYER_COLUMNS


def _season_sorted(values) -> list:
    return sorted(map(str, values), key=lambda s: int(s[:4]))


def duration_summary(stints: pd.DataFrame) -> pd.DataFrame:
    """Per season: stints, games, duration quantiles and the lineup_ok share."""
    rows = []
    for season in _season_sorted(stints["season"].unique()):
        s = stints[stints["season"].astype(str) == season]
        d = s["duration_s"].astype(float)
        rows.append({"season": season, "stints": int(len(s)), "games": int(s["game_id"].nunique()),
                     "duration_min_s": float(d.min()), "duration_p25_s": float(d.quantile(0.25)), "duration_median_s": float(d.median()),
                     "duration_p75_s": float(d.quantile(0.75)), "duration_max_s": float(d.max()), "lineup_ok_share": float(s["lineup_ok"].astype(bool).mean())})
    return pd.DataFrame(rows)


def differential_summary(feats: pd.DataFrame) -> pd.DataFrame:
    """Per season: points_diff moments and diff_per100 quantiles with the possession-weighted mean."""
    rows = []
    for season in _season_sorted(feats["season"].unique()):
        f = feats[feats["season"].astype(str) == season]
        d, y, w = f["points_diff"].astype(float), f["diff_per100"].astype(float), f["possessions"].astype(float)
        rows.append({"season": season, "stints": int(len(f)), "points_diff_mean": float(d.mean()), "points_diff_sd": float(d.std()),
                     "points_diff_min": float(d.min()), "points_diff_max": float(d.max()),
                     "diff_per100_weighted_mean": float(np.average(y, weights=w)), "diff_per100_p10": float(y.quantile(0.1)),
                     "diff_per100_median": float(y.median()), "diff_per100_p90": float(y.quantile(0.9))})
    return pd.DataFrame(rows)


def lineup_possessions(feats: pd.DataFrame, gate: int = 300) -> pd.DataFrame:
    """Per season: lineups seen, how many pass the possession gate, and possession quantiles."""
    from analytics.lineups.train import lineup_rows
    rows = []
    lr = lineup_rows(feats)
    for season in _season_sorted(lr["season"].unique()):
        p = lr[lr["season"].astype(str) == season].groupby("lineup_id")["possessions"].sum()
        rows.append({"season": season, "lineups": int(len(p)), "pass_gate_%d" % gate: int((p >= gate).sum()),
                     "possessions_median": float(p.median()), "possessions_p90": float(p.quantile(0.9)), "possessions_max": float(p.max()),
                     "share_of_possessions_in_gated": float(p[p >= gate].sum() / p.sum()) if p.sum() else float("nan")})
    return pd.DataFrame(rows)


def player_minutes(stints: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per season: the top players by minutes on court, from stint durations."""
    parts = []
    for c in PLAYER_COLUMNS:
        parts.append(pd.DataFrame({"season": stints["season"].astype(str), "player_id": stints[c], "minutes": stints["duration_s"].astype(float) / 60.0}))
    m = pd.concat(parts).dropna(subset=["player_id"])
    m["player_id"] = m["player_id"].astype("int64")
    out = m.groupby(["season", "player_id"], as_index=False)["minutes"].sum()
    out = out.sort_values(["season", "minutes"], ascending=[True, False])
    return out.groupby("season", group_keys=False).head(top).reset_index(drop=True)


def cross_season_correlation(feats: pd.DataFrame, min_poss: int = 100) -> pd.DataFrame:
    """Pearson and Spearman of raw lineup net rating between consecutive seasons, lineups with min_poss in both."""
    from analytics.lineups.train import lineup_table
    seasons = _season_sorted(feats["season"].unique())
    rows = []
    tables = {s: lineup_table(feats[feats["season"].astype(str) == s]) for s in seasons}
    for a, b in zip(seasons[:-1], seasons[1:]):
        ta, tb = tables[a], tables[b]
        m = ta[ta["n_possessions"] >= min_poss].merge(tb[tb["n_possessions"] >= min_poss], on="lineup_id", suffixes=("_a", "_b"))
        rec = {"season_from": a, "season_to": b, "n_lineups": int(len(m)), "pearson": float("nan"), "spearman": float("nan")}
        if len(m) >= 3 and m["plus_minus_per100_a"].std() > 0 and m["plus_minus_per100_b"].std() > 0:
            rec["pearson"] = float(pearsonr(m["plus_minus_per100_a"], m["plus_minus_per100_b"])[0])
            rec["spearman"] = float(spearmanr(m["plus_minus_per100_a"], m["plus_minus_per100_b"]).correlation)
        rows.append(rec)
    return pd.DataFrame(rows, columns=["season_from", "season_to", "n_lineups", "pearson", "spearman"])


__all__ = ["duration_summary", "differential_summary", "lineup_possessions", "player_minutes", "cross_season_correlation"]
