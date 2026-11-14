"""analytics/epv/eda.py: the tables behind notebooks/eda_epv.ipynb (EPV exploratory data analysis).

Every function returns a pandas DataFrame the notebook prints as a markdown table (no plotting library here).
On the synthetic fixtures the output is fixture output, not a result; the notebook says so in its first cell.

    load_tables(fixtures=True, data_root=None, seed=0, n_games=4)   the EPV inputs (analytics.epv.validate; the
                                                                    fixtures are analytics.fixtures' synthetic season)
    points_by_elapsed_bucket(states)                                points_to_end by elapsed bucket and end kind
    feature_distributions(feats)                                    summary of every feature, with the missing share
    target_correlation(feats, y)                                    Pearson and Spearman correlation with the target
    shuffled_check(tables, folds, seed, hgb_iter)                   the real lift against the shuffled-target lift
    markdown(df)                                                    a markdown table (analytics.eval.md_table)

The trainer's leakage guards apply here as well: ``feature_distributions`` and ``target_correlation`` take the
frame of ``analytics.epv.features.build_features``, which has passed ``analytics.schema.assert_time_forward``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import schema
from analytics.epv import ELAPSED_BUCKETS
from analytics.eval import md_table
from analytics.epv import features as F
from analytics.epv import validate as V


def load_tables(fixtures: bool = True, data_root=None, seed: int = 0, n_games: int = 4, heldout_every: int = 4, out_root=None,
                n_possessions: int = 40) -> dict:
    """The cleaned EPV inputs: the fixture season (default) or DATA_ROOT (with the tables under OUT_ROOT first, as
    analytics.epv.validate.load_root)."""
    tables = (V.load_fixtures(seed, n_games=n_games, heldout_every=heldout_every, n_possessions=n_possessions) if fixtures
              else V.load_root(data_root, out_root=out_root))
    clean, reports = V.validate_all(tables)
    clean["reports"] = reports
    return clean


def _bucket(t) -> np.ndarray:
    from analytics.epv.train import elapsed_bucket
    return elapsed_bucket(t)


def points_by_elapsed_bucket(states: pd.DataFrame) -> pd.DataFrame:
    """Per elapsed bucket: states, possessions, mean points_to_end, P(points > 0), and the share of each end kind."""
    s = states.copy()
    s["bucket"] = _bucket(s["elapsed_in_possession_s"].to_numpy(dtype=float))
    rows = []
    for b in ["%g-%g" % x for x in ELAPSED_BUCKETS]:
        part = s[s["bucket"] == b]
        if len(part) == 0:
            continue
        row = {"bucket_s": b, "states": int(len(part)), "possessions": int(part.groupby(["sportvu_game", "possession"]).ngroups),
               "mean_points_to_end": float(part["points_to_end"].mean()), "p_score": float((part["points_to_end"] > 0).mean())}
        for k in schema.END_KINDS:
            row["share_" + k] = float((part["possession_end_kind"].astype(str) == k).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def feature_distributions(feats: pd.DataFrame) -> pd.DataFrame:
    """mean, std, min, quartiles, max and the missing share of every feature column present."""
    schema.assert_time_forward(feats, "states_1hz")
    cols = [c for c in F.FEATURES if c in feats.columns]
    d = feats[cols].astype(float).describe(percentiles=[0.25, 0.5, 0.75]).T
    d["missing_share"] = feats[cols].isna().mean()
    d = d.drop(columns=["count"]).reset_index().rename(columns={"index": "feature", "25%": "p25", "50%": "p50", "75%": "p75"})
    return d


def target_correlation(feats: pd.DataFrame, y) -> pd.DataFrame:
    """Pearson and Spearman correlation of every feature with the target, missing values dropped pairwise,
    sorted by absolute Pearson correlation."""
    schema.assert_time_forward(feats, "states_1hz")
    y = pd.Series(np.asarray(y, dtype=float), index=feats.index)
    rows = []
    for c in [c for c in F.FEATURES if c in feats.columns]:
        x = feats[c].astype(float)
        m = x.notna() & y.notna()
        if m.sum() < 3 or x[m].std() == 0:
            rows.append({"feature": c, "n": int(m.sum()), "pearson": np.nan, "spearman": np.nan})
            continue
        rows.append({"feature": c, "n": int(m.sum()), "pearson": float(x[m].corr(y[m])), "spearman": float(x[m].corr(y[m], method="spearman"))})
    out = pd.DataFrame(rows)
    return out.reindex(out["pearson"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def shuffled_check(tables: dict, folds: int = 3, seed: int = 0, hgb_iter: int = 60) -> pd.DataFrame:
    """Out-of-fold MSE of the constant and gradient boosting on the real target and on the shuffled target
    (possession points permuted across possessions). The shuffled lift must sit at or below chance."""
    from analytics.epv import train as T
    d = T.split_tables(tables)
    st = d["states_train"]
    geo = F.geometry(st)
    rows = []
    for label, y in (("real", None), ("shuffled", T.shuffled_target(st, seed))):
        pred, _ = T.out_of_fold(st, geo, d["shots_train"], d["rosters"], folds, seed, hgb_iter, models=("constant", "hgb"), y_override=y)
        mc, mh = T.mse(pred["y"], pred["epv_constant"]), T.mse(pred["y"], pred["epv_hgb"])
        rows.append({"target": label, "mse_constant": mc, "mse_hgb": mh, "lift": 1.0 - mh / mc if mc > 0 else 0.0})
    return pd.DataFrame(rows)


def markdown(df: pd.DataFrame, nd: int = 4) -> str:
    """A markdown table of a frame (floats to nd decimals, NaN blank): analytics.eval.md_table joined into one string."""
    return "\n".join(md_table(df, nd))


__all__ = ["load_tables", "points_by_elapsed_bucket", "feature_distributions", "target_correlation", "shuffled_check", "markdown"]
