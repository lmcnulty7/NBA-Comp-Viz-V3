"""analytics/xfg/eda.py: exploratory tables for the xFG upgrade (notebooks/eda_xfg.ipynb calls these).

    python -m analytics.xfg.eda --fixtures | --data DATA_ROOT [--seed 0] [--folds 3]

Every function takes the trainer frame (analytics.xfg.train.assemble: features plus ``made``) and returns a
DataFrame; nothing is plotted (no matplotlib here) and nothing is fitted except in ``shuffle_check``, which is the
leakage check: ``made`` is permuted and the three candidates are run out-of-fold; a pipeline without leakage
scores at chance (log loss at the entropy of the base rate or above, Brier at p(1 - p) or above). A model that
beats chance on shuffled labels is reading the outcome from somewhere, and the check says which.

Under --fixtures every table is fixture output, not a result.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from analytics.eval import brier, log_loss
from analytics.xfg import validate as V
from analytics.xfg.features import CONTEST_FEATURES, FEATURES, LOCATION_FEATURES

log = logging.getLogger("analytics.xfg.eda")

DISTANCE_EDGES = (0, 4, 8, 14, 18, 22, 26, 31, 50)
DEFENDER_EDGES = (0, 2, 4, 6, 8, 12, 100)


def _rate_table(df: pd.DataFrame, key: pd.Series, label: str) -> pd.DataFrame:
    """n, make rate and its binomial standard error per value of ``key``."""
    g = df.groupby(key, observed=True)["made"]
    out = pd.DataFrame({"n": g.size(), "make_rate": g.mean().astype(float)})
    out.index.name = label
    out["se"] = np.sqrt(out["make_rate"] * (1 - out["make_rate"]) / out["n"].clip(lower=1))
    return out.reset_index()


def make_rate_by_distance(df: pd.DataFrame, edges=DISTANCE_EDGES) -> pd.DataFrame:
    """n, make rate and its standard error per distance bin (feet)."""
    b = pd.cut(df["distance_ft"].astype(float), bins=list(edges), right=False)
    return _rate_table(df, b, "distance_ft_bin")


def make_rate_by_defender(df: pd.DataFrame, edges=DEFENDER_EDGES) -> pd.DataFrame:
    """n, make rate and its standard error per closest-defender bin (feet), plus a row for attempts without a
    measured contest (has_contest 0)."""
    d = df["closest_defender_dist_ft"].astype(float)
    b = pd.cut(d, bins=list(edges), right=False).astype(object)
    b = b.where(d.notna(), "no contest measured")
    out = _rate_table(df, b.astype(str), "closest_defender_bin")
    return out


def feature_distributions(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """count, null share, mean, std, min, quartiles, max of every feature."""
    cols = list(columns or FEATURES)
    x = df[cols].astype(float)
    desc = x.describe().T
    desc.insert(0, "null_share", x.isna().mean())
    return desc.reset_index().rename(columns={"index": "feature"})


def correlation_table(df: pd.DataFrame, columns=None, method: str = "spearman") -> pd.DataFrame:
    """Rank correlation among the features and with made (descriptive; the trainer never sees this)."""
    cols = list(columns or (LOCATION_FEATURES + CONTEST_FEATURES))
    x = df[cols].astype(float).copy()
    x["made"] = df["made"].astype(float)
    return x.corr(method=method).round(3)


def make_rate_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    """n, make rate and its standard error per shooter-prior zone."""
    return _rate_table(df, df["zone"].astype(str), "zone")


def chance_levels(y) -> dict:
    """Log loss and Brier of the best constant predictor: the floor a leak-free model reaches on shuffled labels."""
    y = np.asarray(y, dtype=float)
    r = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    return {"base_rate": r, "log_loss_chance": float(-(r * np.log(r) + (1 - r) * np.log(1 - r))), "brier_chance": float(r * (1 - r))}


def shuffle_check(train: pd.DataFrame, n_folds: int = 3, seed: int = 0, tolerance: float = 0.02) -> pd.DataFrame:
    """The leakage check: permute ``made`` (within game, so the base rate per game is kept), run the three candidates
    out-of-fold, and compare each to chance. ``at_chance`` is true when the out-of-fold log loss is no better than
    chance minus ``tolerance`` (small gains come from the fold-rate constant and from noise)."""
    from analytics.xfg import train as T
    rng = np.random.default_rng(seed)
    df = train.copy()
    y = df["made"].to_numpy().copy()
    for _g, idx in df.groupby("sportvu_game").indices.items():
        y[idx] = rng.permutation(y[idx])
    df["made"] = y
    pred, _note = T.out_of_fold(df, n_folds, seed)
    ch = chance_levels(df["made"])
    rows = []
    for name in T.CANDIDATES:
        ll = log_loss(df["made"], pred["p_" + name])
        rows.append({"candidate": name, "log_loss_shuffled": ll, "log_loss_chance": ch["log_loss_chance"],
                     "brier_shuffled": brier(df["made"], pred["p_" + name]), "brier_chance": ch["brier_chance"],
                     "at_chance": bool(ll >= ch["log_loss_chance"] - tolerance)})
    return pd.DataFrame(rows)


def run(train: pd.DataFrame, n_folds: int = 3, seed: int = 0) -> dict:
    """Every EDA table at once: {name: DataFrame}."""
    return {"make_rate_by_distance": make_rate_by_distance(train), "make_rate_by_defender": make_rate_by_defender(train),
            "make_rate_by_zone": make_rate_by_zone(train), "feature_distributions": feature_distributions(train),
            "correlation_table": correlation_table(train), "shuffle_check": shuffle_check(train, n_folds, seed)}


def main(argv=None) -> dict:
    """CLI: print every EDA table (fixture output, not a result, under --fixtures)."""
    from analytics.xfg import train as T
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    V.add_source_args(ap, games=6, possessions=24)
    ap.add_argument("--folds", type=int, default=3, help="grouped folds of the shuffled-label check (default %(default)s)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)
    tables = V.source_tables(a)
    tables, _s = V.clean(tables)
    train, _held = T.assemble(tables)
    out = run(train, a.folds, a.seed)
    for name, t in out.items():
        print("## %s%s\n" % (name, " (fixture output, not a result)" if a.fixtures else ""))
        print(t.to_string(index=False))
        print()
    return out


if __name__ == "__main__":
    main()
