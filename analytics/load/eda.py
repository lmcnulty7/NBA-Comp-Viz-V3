"""analytics/load/eda.py: descriptive tables for notebooks/eda_load.ipynb (tables and markdown, no plots).

Every function takes the tables of analytics.load.features.build_tables and returns a DataFrame; markdown()
renders one without tabulate. Nothing here fits a model: the shuffled check is a rank correlation of each
feature with the decline target against the same correlation after the target is permuted within game, so a
feature whose true correlation sits inside the shuffled spread carries no more than chance. Held-out rows are
excluded from every table (the heldout column), so an EDA never looks at a held-out game. On fixtures the
numbers are fixture output, not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.eval import md_table
from analytics.load.features import FEATURES, MIN_STINTS_RANK, SAME_GAME, TARGETS

REST_BUCKETS = (("back_to_back", 0.0, 1.0), ("one_day", 1.0, 2.0), ("two_plus", 2.0, np.inf))


def rest_bucket(days_rest) -> pd.Series:
    """days_rest -> back_to_back (1 day or less), one_day (2), two_plus (3 or more); NaN stays NaN."""
    d = pd.to_numeric(pd.Series(days_rest), errors="coerce")
    out = pd.Series(np.where(d.isna(), None, "two_plus"), index=d.index, dtype=object)
    for name, lo, hi in REST_BUCKETS:
        out[(d > lo) & (d <= hi)] = name
    out[d <= 1.0] = "back_to_back"
    return out


def _train(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["heldout"].astype(bool)] if "heldout" in df.columns else df


def speed_by_stint_index(stints: pd.DataFrame, max_index: int = 8) -> pd.DataFrame:
    """Per stint index within the game: stints, mean and median avg speed, mean minutes, mean high-intensity share."""
    s = _train(stints)
    s = s[np.isfinite(s["avg_speed_ftps"])]
    s = s.assign(stint=np.minimum(s["stint"], max_index))
    g = s.groupby("stint")
    out = g.agg(n=("avg_speed_ftps", "size"), mean_speed_ftps=("avg_speed_ftps", "mean"),
                median_speed_ftps=("avg_speed_ftps", "median"), mean_minutes=("minutes", "mean"),
                mean_high_intensity_share=("high_intensity_share", "mean")).reset_index()
    out.loc[out["stint"] == max_index, "stint"] = max_index
    return out


def decline_by_rest_bucket(games: pd.DataFrame) -> pd.DataFrame:
    """Per rest bucket: player-games with a decline, games, mean and sd of decline_ftps and of the slope."""
    g = _train(games)
    g = g[np.isfinite(g["decline_ftps"])].assign(bucket=rest_bucket(g["days_rest"]))
    order = [b for b, _lo, _hi in REST_BUCKETS]
    out = g.groupby("bucket").agg(n=("decline_ftps", "size"), games=("sportvu_game", "nunique"),
                                  mean_decline_ftps=("decline_ftps", "mean"), sd_decline_ftps=("decline_ftps", "std"),
                                  mean_slope=("slope_ftps_per_stint", "mean")).reindex(order).reset_index()
    out["n"] = out["n"].fillna(0).astype(int)
    out["games"] = out["games"].fillna(0).astype(int)
    return out


def distance_distribution(games: pd.DataFrame, stints: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quantiles of distance (ft), minutes, average speed and distance per minute per player-game, and of
    distance per stint when the stints are given."""
    g = _train(games)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    cols = {"distance_ft": g["distance_ft"], "minutes": g["minutes"], "avg_speed_ftps": g["avg_speed_ftps"],
            "ft_per_minute": g["distance_ft"] / g["minutes"].replace(0, np.nan)}
    if stints is not None:
        cols["stint_distance_ft"] = _train(stints)["distance_ft"]
        cols["stint_minutes"] = _train(stints)["minutes"]
    rows = []
    for name, s in cols.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        row = {"quantity": name, "n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan}
        for p in q:
            row["q%02d" % int(p * 100)] = float(s.quantile(p)) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(games: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Rank correlations among the features, the same-game totals and the targets (absent_next as 0/1)."""
    g = _train(games).copy()
    g["absent_next"] = g["absent_next"].astype("float")
    g["back_to_back"] = g["back_to_back"].astype(float)
    cols = [c for c in FEATURES + SAME_GAME + TARGETS if c in g.columns]
    return g[cols].astype(float).corr(method=method).round(3)


def shuffled_check(games: pd.DataFrame, target: str = "decline_ftps", n_shuffles: int = 50, seed: int = 0) -> pd.DataFrame:
    """Per feature: Spearman rho with the target on the real rows, and the mean and 2.5 / 97.5 percentiles of
    rho when the target is permuted within game n_shuffles times. A real rho inside the shuffled band is chance."""
    g = _train(games)
    g = g[np.isfinite(g[target].astype(float))].copy()
    rng = np.random.default_rng(seed)
    y = g[target].astype(float).to_numpy()
    codes = pd.factorize(g["sportvu_game"])[0]
    rows = []
    feats = {c: pd.to_numeric(g[c].astype(float), errors="coerce").to_numpy() for c in FEATURES}

    def rho(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            return np.nan
        return float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))

    shuffled = []
    for _ in range(n_shuffles):
        ys = y.copy()
        for k in np.unique(codes):
            ix = np.flatnonzero(codes == k)
            ys[ix] = y[rng.permutation(ix)]
        shuffled.append(ys)
    for c, x in feats.items():
        rs = np.array([rho(x, ys) for ys in shuffled], dtype=float)
        rows.append({"feature": c, "n": int(np.isfinite(x).sum()), "rho_real": rho(x, y),
                     "rho_shuffled_mean": float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan,
                     "rho_shuffled_lo": float(np.nanpercentile(rs, 2.5)) if np.isfinite(rs).any() else np.nan,
                     "rho_shuffled_hi": float(np.nanpercentile(rs, 97.5)) if np.isfinite(rs).any() else np.nan})
    out = pd.DataFrame(rows)
    out["outside_band"] = (out["rho_real"] < out["rho_shuffled_lo"]) | (out["rho_real"] > out["rho_shuffled_hi"])
    return out


def stint_gate(summary: pd.DataFrame) -> pd.DataFrame:
    """How many players pass the MIN_STINTS_RANK gate and the stint count quantiles."""
    s = summary
    return pd.DataFrame([{"players": int(len(s)), "min_stints": MIN_STINTS_RANK,
                          "enough": int((~s["insufficient"].astype(bool)).sum()),
                          "insufficient": int(s["insufficient"].astype(bool).sum()),
                          "stints_median": float(s["n_stints"].median()) if len(s) else np.nan,
                          "stints_max": int(s["n_stints"].max()) if len(s) else 0}])


def markdown(df: pd.DataFrame, nd: int = 3) -> str:
    """A markdown table of a frame (analytics.eval.md_table): floats to nd decimals, NaN blank."""
    return "\n".join(md_table(df, nd))


__all__ = ["rest_bucket", "speed_by_stint_index", "decline_by_rest_bucket", "distance_distribution",
           "correlation_table", "shuffled_check", "stint_gate", "markdown"]
