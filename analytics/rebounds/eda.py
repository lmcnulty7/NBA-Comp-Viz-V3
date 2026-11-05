"""analytics/rebounds/eda.py: tables for notebooks/eda_rebounds.ipynb (no plots: tables and markdown only).

    python -m analytics.rebounds.eda --fixtures | --data DATA_ROOT [--seed 0] [--games 8]

Functions take the candidate frame of analytics.rebounds.features.build and return DataFrames:
    rebound_share_by_side          share of misses secured by the offence, the defence and nobody (none)
    rebound_share_by_distance_bin  rebound rate per candidate row by distance-to-rim bin and side
    feature_distributions          count, mean, std and quartiles of every static feature, plus the mean on
                                   rebounder rows and on the other rows
    correlation_table              Pearson correlations among the static features and the target
    shuffled_label_check           out-of-fold per-miss log loss of the three candidates on the real labels and
                                   on labels permuted within each miss; a model that beats the uniform reference
                                   on shuffled labels is leaking
``markdown(df)`` renders a frame as a pipe table. Nothing here fits on held-out rows: the shuffled-label check
calls train.out_of_fold on the training rows only, which itself calls refuse_heldout and assert_time_forward.
Under --fixtures every number is fixture output, not a result.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analytics.rebounds import features as F

DIST_BINS = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, np.inf)


def markdown(df: pd.DataFrame, nd: int = 4) -> str:
    """A pipe table of ``df`` (index included when it is named)."""
    d = df.reset_index() if df.index.name is not None or isinstance(df.index, pd.MultiIndex) else df
    cols = [str(c) for c in d.columns]
    L = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else ("%%.%df" % nd) % v)
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def rebound_share_by_side(cands: pd.DataFrame) -> pd.DataFrame:
    """One row per outcome side: misses, share. The side of a miss is that of its rebounder row."""
    reb = cands[cands[F.TARGET].astype(bool)]
    side = np.where(reb["is_none"].astype(int) == 1, "none", np.where(reb["is_offense"].astype(int) == 1, "offense", "defense"))
    counts = pd.Series(side).value_counts().reindex(["offense", "defense", "none"]).fillna(0).astype(int)
    out = pd.DataFrame({"side": counts.index, "misses": counts.to_numpy(), "share": counts.to_numpy() / max(int(counts.sum()), 1)})
    return out


def rebound_share_by_distance_bin(cands: pd.DataFrame) -> pd.DataFrame:
    """Rebound rate per candidate row (players only) by distance_to_rim_ft bin and side."""
    pl = cands[cands["is_none"].astype(int) == 0]
    b = pd.cut(pl["distance_to_rim_ft"].astype(float), bins=list(DIST_BINS), right=False)
    g = pl.groupby([b, pl["is_offense"].astype(int)], observed=True)[F.TARGET]
    out = g.agg(rows="size", rebounds="sum").reset_index()
    out.columns = ["distance_bin", "is_offense", "rows", "rebounds"]
    out["distance_bin"] = out["distance_bin"].astype(str)
    out["rebound_rate"] = out["rebounds"] / out["rows"].clip(lower=1)
    return out


def feature_distributions(cands: pd.DataFrame) -> pd.DataFrame:
    X = cands[F.STATIC_FEATURES].astype(float)
    d = X.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    y = cands[F.TARGET].astype(bool).to_numpy()
    d["mean_rebounder"] = X[y].mean()
    d["mean_other"] = X[~y].mean()
    d.index.name = "feature"
    return d


def correlation_table(cands: pd.DataFrame) -> pd.DataFrame:
    X = cands[F.STATIC_FEATURES].astype(float).copy()
    X[F.TARGET] = cands[F.TARGET].astype(float)
    c = X.corr()
    c.index.name = "feature"
    return c


def _shuffle_within_miss(cands: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The target permuted within each miss (every miss keeps exactly one rebounder row)."""
    from analytics.rebounds.train import miss_key
    rng = np.random.default_rng(seed)
    key = miss_key(cands)
    y = cands[F.TARGET].astype(bool).to_numpy().copy()
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        y[idx] = y[rng.permutation(idx)]
    out = cands.copy()
    out[F.TARGET] = y
    return out


def shuffled_label_check(train: pd.DataFrame, seed: int = 0, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold per-miss log loss and top-1 of every candidate on the real labels and on labels permuted
    within each miss, next to the uniform reference. Shuffled rows should sit at the uniform reference."""
    from analytics.rebounds import train as T
    rows = []
    for label, frame in (("real", train), ("shuffled", _shuffle_within_miss(train, seed))):
        pred, note = T.out_of_fold(frame, n_folds, seed)
        for m in T.MODELS:
            pm = T.per_miss(pred, "p_" + m)
            rows.append({"labels": label, "model": m, "per_miss_log_loss": T.mlogloss(pm), "top1": T.top1(pm),
                         "uniform_log_loss": T.uniform_reference(pm)["mlogloss"], "misses": int(len(pm)), "mode": note["mode"]})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from analytics.rebounds import train as T
    from analytics.rebounds import validate as V
    a = parse_args(argv)
    bundles = V.bundles_from_fixtures(a.seed, a.games) if a.fixtures else V.bundles_from_root(a.data)
    cands, _, _, _ = T.load_candidates(bundles)
    train, _ = T.split_train_heldout(cands)
    out = {"by_side": rebound_share_by_side(train), "by_distance": rebound_share_by_distance_bin(train),
           "distributions": feature_distributions(train), "correlations": correlation_table(train),
           "shuffled": shuffled_label_check(train, a.seed, a.folds)}
    head = "# Rebound EDA" + (" (fixture output, not a result)" if a.fixtures else "")
    print(head + "\n")
    for name, df in out.items():
        print("## %s\n\n%s" % (name, markdown(df)))
    return out


if __name__ == "__main__":
    main()
