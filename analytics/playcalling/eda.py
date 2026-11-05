"""analytics/playcalling/eda.py: tables for notebooks/eda_playcalling.ipynb (no plots; pandas and numpy only).

Every function takes the action table of analytics.playcalling.features.build_action_table and returns a
DataFrame; ``to_markdown`` renders one without the tabulate dependency. Nothing here fits a model. On the
fixtures every table is fixture output, not a result.

    action_counts(table)          actions, possessions and games per action type, by label_source
    coverage_counts(table)        actions per action type x coverage, by coverage_source
    ppp_distribution(table)       points distribution per action type (share of 0, 1, 2, 3+ and the mean)
    lineup_counts(table)          distinct lineups and how concentrated the rows are (share in the top 5)
    shuffled_action_check(table)  the spread of mean PPP across action types, observed against the spread
                                  under action labels shuffled within each game (a null reference for the
                                  "action type carries information about PPP" reading; a test of association,
                                  not of a causal effect)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_markdown(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else ("%%.%df" % nd) % v)
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def action_counts(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby(["action_type", "label_source"])
    out = g.agg(actions=("possession", "size"),
                possessions=("possession", lambda s: int(len(set(zip(table.loc[s.index, "sportvu_game"], s))))),
                games=("sportvu_game", "nunique"),
                heldout_actions=("heldout", "sum")).reset_index()
    return out.sort_values(["label_source", "actions"], ascending=[True, False]).reset_index(drop=True)


def coverage_counts(table: pd.DataFrame) -> pd.DataFrame:
    out = (table.groupby(["coverage", "action_type", "coverage_source"]).size().rename("actions").reset_index())
    return out.sort_values(["coverage", "actions"], ascending=[True, False]).reset_index(drop=True)


def ppp_distribution(table: pd.DataFrame) -> pd.DataFrame:
    t = table[table["ppp"].notna()]
    rows = []
    for at, grp in t.groupby("action_type"):
        p = grp["ppp"].to_numpy(dtype=float)
        rows.append({"action_type": at, "n": int(len(p)), "mean_ppp": float(p.mean()), "share_0": float((p == 0).mean()),
                     "share_1": float((p == 1).mean()), "share_2": float((p == 2).mean()), "share_3plus": float((p >= 3).mean()),
                     "epv_gain_mean": float(grp["epv_gain"].mean()) if grp["epv_gain"].notna().any() else float("nan")})
    return pd.DataFrame(rows).sort_values("mean_ppp", ascending=False).reset_index(drop=True)


def lineup_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("off_lineup", "def_lineup"):
        c = table[side].dropna().value_counts()
        rows.append({"side": side, "distinct_lineups": int(len(c)), "rows": int(c.sum()), "null_rows": int(table[side].isna().sum()),
                     "share_top5": float(c.head(5).sum() / c.sum()) if c.sum() else float("nan"),
                     "lineups_with_300_plus": int((c >= 300).sum())})
    return pd.DataFrame(rows)


def _spread(t: pd.DataFrame, col: str) -> float:
    m = t.groupby(col)["ppp"].mean()
    return float(m.max() - m.min()) if len(m) > 1 else 0.0


def shuffled_action_check(table: pd.DataFrame, n_shuffles: int = 200, seed: int = 0) -> pd.DataFrame:
    """Observed max minus min of mean PPP across action types against the same spread with action_type shuffled
    within each game (n_shuffles times). Columns: statistic, observed, shuffled_mean, shuffled_p95, share_shuffled_ge_observed.
    A share near zero says the action labels carry information about PPP beyond game-to-game differences; it says
    nothing about why."""
    t = table[table["ppp"].notna()][["sportvu_game", "action_type", "ppp"]].reset_index(drop=True)
    obs = _spread(t, "action_type")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_shuffles):
        s = t.copy()
        s["action_type"] = s.groupby("sportvu_game")["action_type"].transform(lambda x: x.sample(frac=1.0, random_state=int(rng.integers(2**31))).to_numpy())
        vals.append(_spread(s, "action_type"))
    vals = np.array(vals)
    return pd.DataFrame([{"statistic": "max minus min of mean PPP across action types", "observed": obs,
                          "shuffled_mean": float(vals.mean()), "shuffled_p95": float(np.percentile(vals, 95)),
                          "share_shuffled_ge_observed": float((vals >= obs).mean()), "n_shuffles": int(n_shuffles)}])


__all__ = ["to_markdown", "action_counts", "coverage_counts", "ppp_distribution", "lineup_counts", "shuffled_action_check"]
