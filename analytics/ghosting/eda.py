"""analytics/ghosting/eda.py: tables for notebooks/eda_ghosting.ipynb.

No plotting library exists in the build environment, so the EDA is tables (pandas frames the notebook
prints) and markdown. Every function is pure over the frames the trainer returns (analytics.ghosting.train.run):
    displacement_distributions(frame)      quantiles of the 0.2 s displacement target and the defender speed
    feature_summary(frame)                 mean, sd, min, max of every lag-1 feature
    error_by_horizon(errors)               mean and median error per candidate and horizon, with row counts
    deviation_by_end_kind(gp, possessions) mean deviation of a possession by its section 19 end_kind
    correlation_table(gp)                  per-game Spearman of mean deviation against points plus the sign test
Nothing here fits anything; fixture numbers are fixture output, not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.eval import md_table
from analytics.ghosting import features as F

QUANTILES = (0.5, 0.9, 0.99)


def displacement_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows: displacement magnitude per 0.2 s step, its x and y parts, the defender speed at t minus 1 (ft/s)
    and the distance to the assignment at t minus 1; columns: n, mean, p50, p90, p99, max."""
    mag = np.hypot(frame["target_dx"], frame["target_dy"])
    speed = np.hypot(frame["def_vx"], frame["def_vy"])
    asg = np.hypot(frame["rel_asg_dx"], frame["rel_asg_dy"])
    series = {"step_displacement_ft": mag, "target_dx_ft": frame["target_dx"], "target_dy_ft": frame["target_dy"],
              "def_speed_ft_s": speed, "assignment_distance_ft": asg}
    rows = []
    for name, s in series.items():
        s = pd.Series(np.asarray(s, dtype=float))
        q = s.quantile(QUANTILES)
        rows.append({"quantity": name, "n": int(s.notna().sum()), "mean": float(s.mean()), "p50": float(q.iloc[0]),
                     "p90": float(q.iloc[1]), "p99": float(q.iloc[2]), "max": float(s.max())})
    return pd.DataFrame(rows)


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """mean, sd, min and max of every lag-1 feature."""
    cols = F.feature_columns()
    d = frame[cols].astype(float)
    return pd.DataFrame({"feature": cols, "mean": d.mean().to_numpy(), "sd": d.std().to_numpy(),
                         "min": d.min().to_numpy(), "max": d.max().to_numpy()})


def error_by_horizon(errors: pd.DataFrame) -> pd.DataFrame:
    """candidate x horizon_s -> rows, mean_error_ft, median_error_ft, p90_error_ft, games."""
    g = errors.groupby(["candidate", "horizon_s"])
    out = g.agg(rows=("error_ft", "size"), mean_error_ft=("error_ft", "mean"), median_error_ft=("error_ft", "median"),
                p90_error_ft=("error_ft", lambda s: float(np.quantile(s, 0.9))), games=("sportvu_game", "nunique"))
    return out.reset_index().sort_values(["horizon_s", "mean_error_ft"]).reset_index(drop=True)


def deviation_by_end_kind(gp: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    """ghost_possessions joined to the section 19 end_kind (a TARGET, joined here after every fit) ->
    end_kind, possessions, mean_deviation_ft, sd, mean_points."""
    ek = possessions[["sportvu_game", "possession", "end_kind"]].drop_duplicates(["sportvu_game", "possession"])
    d = gp.merge(ek, on=["sportvu_game", "possession"], how="left")
    d["end_kind"] = d["end_kind"].fillna("other")
    out = d.groupby("end_kind").agg(possessions=("possession", "size"), mean_deviation_ft=("mean_deviation_ft", "mean"),
                                    sd_deviation_ft=("mean_deviation_ft", "std"), mean_points=("points", "mean"))
    return out.reset_index().sort_values("mean_deviation_ft", ascending=False).reset_index(drop=True)


def correlation_table(gp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(per-game Spearman table, sign test dict) from analytics.ghosting.train.sign_test."""
    from analytics.ghosting.train import sign_test
    st = sign_test(gp)
    return st["per_game"], {k: v for k, v in st.items() if k != "per_game"}


def markdown(df: pd.DataFrame, nd: int = 3) -> str:
    """A DataFrame as one markdown string (analytics.eval.md_table, floats to nd decimals)."""
    return "\n".join(md_table(df, nd))


__all__ = ["displacement_distributions", "feature_summary", "error_by_horizon", "deviation_by_end_kind",
           "correlation_table", "markdown"]
