"""analytics/_eval_local.py: a temporary copy of the evaluation helpers, to be replaced by analytics.eval at merge.

Lane H wrote these so analytics.train_xfg can run before analytics.eval lands. The names and signatures match the
shared module exactly, and the callers import them as

    try:
        from analytics.eval import grouped_folds, log_loss, ...
    except ImportError:
        from analytics._eval_local import grouped_folds, log_loss, ...

so the merge only deletes this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportvu.splits import refuse_heldout  # noqa: F401  (re-exported: refuse_heldout(items, who) -> sportvu.splits.refuse_heldout)

EPS = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, float), EPS, 1.0 - EPS)


def grouped_folds(games, n_folds: int = 5, seed: int = 0) -> list:
    """Split the unique game ids into n_folds groups at random -> [(train_games, test_games), ...]. Every game sits in
    exactly one test fold, so a shot is never scored by a model that saw its game."""
    uniq = sorted(set(str(g) for g in games))
    if n_folds < 1 or n_folds > len(uniq):
        raise ValueError("grouped_folds: n_folds=%d must be between 1 and the number of games (%d)" % (n_folds, len(uniq)))
    order = np.array(uniq, dtype=object)
    np.random.default_rng(seed).shuffle(order)
    folds = [list(f) for f in np.array_split(order, n_folds)]
    out = []
    for i, test in enumerate(folds):
        train = [g for j, f in enumerate(folds) if j != i for g in f]
        out.append((sorted(train), sorted(test)))
    return out


def log_loss(y, p) -> float:
    y, p = np.asarray(y, float), _clip(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def _bin_index(p, bins: int) -> np.ndarray:
    return np.minimum((np.asarray(p, float) * bins).astype(int), bins - 1)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: the row-weighted mean over equal-width probability bins of |mean(y) - mean(p)|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = _bin_index(p, bins)
    total = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """One row per equal-width probability bin: bin, lo, hi, n, mean_p, mean_y (NaN means for an empty bin)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = _bin_index(p, bins)
    rows = []
    for k in range(bins):
        m = b == k
        rows.append({"bin": k, "lo": k / bins, "hi": (k + 1) / bins, "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else np.nan,
                     "mean_y": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "mean_p", "mean_y"])


def calibration_slope(y, p) -> float:
    """The slope of a logistic regression of y on logit(p) (C=1e6, so effectively unpenalised). 1 is calibrated,
    below 1 over-confident, above 1 under-confident. NaN when p is constant or y has one class."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, float)
    z = np.log(_clip(p) / (1.0 - _clip(p)))
    if np.std(z) < 1e-9 or len(np.unique(y)) < 2:
        return float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y.astype(int))
    return float(lr.coef_[0, 0])


def bootstrap_by_game(df: pd.DataFrame, game_col: str, metric_fn, n: int = 1000, seed: int = 0) -> dict:
    """metric_fn(df) on the full frame (point) and on n resamples of whole games with replacement (lo, hi are the
    2.5 and 97.5 percentiles). Resampling games, not rows, respects that shots of one game are not independent."""
    point = float(metric_fn(df))
    groups = [np.flatnonzero((df[game_col] == g).to_numpy()) for g in pd.unique(df[game_col])]
    if n <= 0 or len(groups) < 2 or not np.isfinite(point):
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(float(metric_fn(df.iloc[idx])))
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    return {"point": point, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}
