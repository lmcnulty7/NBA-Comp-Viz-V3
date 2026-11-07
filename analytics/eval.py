"""analytics/eval.py: the evaluation harness (docs/DATA_CONTRACT.md sections 16 to 20).

Pure functions over numpy arrays and pandas frames, deterministic given a seed. Trainers use them for
game-grouped cross-validation, calibration metrics with game-level bootstrap intervals, the minimum-sample
gate and the held-out guard. Nothing here reads data files; the only repo dependency is sportvu.splits
for the held-out ids.

Rules the harness enforces or supports:
  - folds are grouped by game, never by row, so frames of one game never sit on both sides of a fit;
  - uncertainty is estimated by resampling whole games, not rows (rows of a game are not independent);
  - held-out games (sportvu/splits.json) never fit or tune anything: refuse_heldout stops a trainer that
    touches one, split_games separates them before any fit;
  - entities with fewer than n_min samples are flagged insufficient rather than reported.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from sportvu import splits as _splits

EPS = 1e-6


# ----------------------------------------------------------------------------- folds and cross-validation

def grouped_folds(games: Sequence[str], n_folds: int = 5, seed: int = 0) -> list:
    """Split game ids into n_folds (train_games, test_games) pairs by a seeded shuffle.

    Every distinct game lands in exactly one test fold; the input order does not matter (ids are sorted
    before the shuffle). Raises ValueError when there are fewer distinct games than folds."""
    uniq = sorted({str(g) for g in games})
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1, got %r" % n_folds)
    if len(uniq) < n_folds:
        raise ValueError("%d distinct games is fewer than %d folds" % (len(uniq), n_folds))
    rng = np.random.default_rng(seed)
    shuffled = [uniq[i] for i in rng.permutation(len(uniq))]
    out = []
    for k in range(n_folds):
        test = sorted(shuffled[k::n_folds])
        train = sorted(set(uniq) - set(test))
        out.append((train, test))
    return out


def grouped_cv(df: pd.DataFrame, game_col: str, fit_fn: Callable, predict_fn: Callable,
               n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold predictions with folds grouped by game.

    fit_fn(train_df) -> model; predict_fn(model, test_df) -> array of probabilities aligned to test_df.
    Returns a DataFrame on df's index with columns fold (int) and p (float)."""
    games = df[game_col].astype(str)
    out = pd.DataFrame({"fold": np.full(len(df), -1, dtype=int), "p": np.full(len(df), np.nan)}, index=df.index)
    for k, (_train, test) in enumerate(grouped_folds(games.unique(), n_folds, seed)):
        m_test = games.isin(test).to_numpy()
        model = fit_fn(df.loc[~m_test])
        p = np.asarray(predict_fn(model, df.loc[m_test]), dtype=float).reshape(-1)
        if len(p) != int(m_test.sum()):
            raise ValueError("predict_fn returned %d values for %d test rows" % (len(p), int(m_test.sum())))
        out.loc[m_test, "fold"] = k
        out.loc[m_test, "p"] = p
    out["fold"] = out["fold"].astype(int)
    return out


# ----------------------------------------------------------------------------- scoring rules and calibration

def _yp(y, p):
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), EPS, 1 - EPS)
    if len(y) != len(p):
        raise ValueError("y and p differ in length: %d vs %d" % (len(y), len(p)))
    return y, p


def log_loss(y, p) -> float:
    """Mean negative log likelihood of binary outcomes under probabilities p (clipped to [1e-6, 1 - 1e-6])."""
    y, p = _yp(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y, p) -> float:
    """Mean squared error of probabilities against binary outcomes."""
    y, p = _yp(y, p)
    return float(np.mean((p - y) ** 2))


def _bins(p, bins):
    return np.clip(np.floor(p * bins), 0, bins - 1).astype(int)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability table: one row per bin with columns bin, lo, hi, n, mean_p, mean_y.
    Empty bins keep their row with n = 0 and NaN means."""
    y, p = _yp(y, p)
    b = _bins(p, bins)
    rows = []
    for k in range(bins):
        m = b == k
        n = int(m.sum())
        rows.append({"bin": k, "lo": k / bins, "hi": (k + 1) / bins, "n": n,
                     "mean_p": float(p[m].mean()) if n else np.nan,
                     "mean_y": float(y[m].mean()) if n else np.nan})
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "mean_p", "mean_y"])


def brier_decomposition(y, p, bins: int = 10) -> dict:
    """Murphy decomposition of the Brier score over equal-width bins.

    reliability = sum_k n_k (mean_p_k - mean_y_k)^2 / N (lower is better calibrated),
    resolution  = sum_k n_k (mean_y_k - mean_y)^2 / N (higher is sharper),
    uncertainty = mean_y (1 - mean_y).
    Brier is reliability - resolution + uncertainty up to the within-bin variance of p."""
    y, p = _yp(y, p)
    t = calibration_table(y, p, bins)
    t = t[t["n"] > 0]
    n = len(y)
    ybar = float(y.mean())
    rel = float((t["n"] * (t["mean_p"] - t["mean_y"]) ** 2).sum() / n)
    res = float((t["n"] * (t["mean_y"] - ybar) ** 2).sum() / n)
    return {"reliability": rel, "resolution": res, "uncertainty": ybar * (1 - ybar)}


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: the n-weighted mean absolute gap between mean_p and mean_y per bin."""
    y, p = _yp(y, p)
    t = calibration_table(y, p, bins)
    t = t[t["n"] > 0]
    return float((t["n"] * (t["mean_p"] - t["mean_y"]).abs()).sum() / len(y))


def calibration_slope(y, p) -> float:
    """Slope of a logistic regression of y on logit(p) (p clipped to [1e-6, 1 - 1e-6]).

    1 is calibrated, above 1 means p is under-confident (shrunk towards 0.5), below 1 over-confident.
    Fitted with sklearn LogisticRegression at C=1e6 (no regularisation to speak of). NaN when y has one class."""
    from sklearn.linear_model import LogisticRegression

    y, p = _yp(y, p)
    if len(np.unique(y)) < 2:
        return float("nan")
    x = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y.astype(int))
    return float(m.coef_[0][0])


# ----------------------------------------------------------------------------- game-level bootstrap

def _game_index(df: pd.DataFrame, game_col: str) -> list:
    """Row positions of each game, in sorted game order (so the draw is deterministic given the seed)."""
    codes, uniq = pd.factorize(df[game_col].astype(str), sort=True)
    return [np.flatnonzero(codes == k) for k in range(len(uniq))]


def _resample(df: pd.DataFrame, groups: list, rng) -> pd.DataFrame:
    pick = rng.integers(0, len(groups), size=len(groups))
    return df.iloc[np.concatenate([groups[i] for i in pick])]


def bootstrap_by_game(df: pd.DataFrame, game_col: str, metric_fn: Callable, n: int = 1000, seed: int = 0) -> dict:
    """metric_fn(df) -> float on the full frame (point) and on n resamples of whole games with replacement.
    Returns dict(point, lo, hi) with the 2.5 and 97.5 percentiles of the resampled metric."""
    groups = _game_index(df, game_col)
    if not groups:
        raise ValueError("bootstrap_by_game needs at least one game")
    rng = np.random.default_rng(seed)
    vals = np.array([float(metric_fn(_resample(df, groups, rng))) for _ in range(n)])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"point": float(metric_fn(df)), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap_by_game(df: pd.DataFrame, game_col: str, metric_a_fn: Callable, metric_b_fn: Callable,
                             n: int = 1000, seed: int = 0) -> dict:
    """Bootstrap of metric_a_fn(df) - metric_b_fn(df) over the same game resamples (paired).
    Returns dict(diff, lo, hi, excludes_zero) where excludes_zero is true when the 95 percent interval
    does not contain 0."""
    groups = _game_index(df, game_col)
    if not groups:
        raise ValueError("paired_bootstrap_by_game needs at least one game")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = _resample(df, groups, rng)
        vals.append(float(metric_a_fn(s)) - float(metric_b_fn(s)))
    lo, hi = np.nanpercentile(np.array(vals), [2.5, 97.5])
    diff = float(metric_a_fn(df)) - float(metric_b_fn(df))
    return {"diff": diff, "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


# ----------------------------------------------------------------------------- minimum-sample gate

def min_sample_gate(counts: pd.Series, n_min: int = 300) -> pd.Series:
    """True where the entity has at least n_min samples."""
    return (pd.to_numeric(counts, errors="coerce").fillna(0) >= n_min).rename("enough")


def gated(df: pd.DataFrame, entity_col: str, n_min: int = 300) -> pd.DataFrame:
    """df with a bool column insufficient: true for rows of entities with fewer than n_min rows."""
    counts = df.groupby(entity_col, dropna=False)[entity_col].transform("size")
    out = df.copy()
    out["insufficient"] = ~min_sample_gate(counts, n_min).to_numpy()
    return out


# ----------------------------------------------------------------------------- held-out guard

def refuse_heldout(items, who: str) -> int:
    """SystemExit when any item (path, game tag, SportVU id, video id) belongs to a held-out game of
    sportvu/splits.json; otherwise the number of items checked. Every trainer calls this before fitting."""
    return _splits.refuse_heldout(items, who)


def split_games(games: Sequence[str]) -> dict:
    """dict(train, heldout): the ids that carry a held-out token (sportvu.splits.heldout_tokens) and the rest.
    Order is preserved; duplicates are kept so the caller sees exactly what it passed."""
    toks = _splits.heldout_tokens()
    held = [g for g in games if any(t in str(g) for t in toks)]
    held_set = set(map(str, held))
    return {"train": [g for g in games if str(g) not in held_set], "heldout": held}


# ----------------------------------------------------------------------------- swap test (CV rows for tracking rows)

def spearman(ranking_a: Sequence, ranking_b: Sequence):
    """Spearman rank correlation of two orderings of (mostly) the same entities, over the entities both
    contain. None when fewer than two entities are shared."""
    pos_a = {e: i for i, e in enumerate(ranking_a)}
    pos_b = {e: i for i, e in enumerate(ranking_b)}
    common = [e for e in ranking_a if e in pos_b]
    if len(common) < 2:
        return None
    ra = pd.Series([pos_a[e] for e in common], dtype=float).rank().to_numpy()
    rb = pd.Series([pos_b[e] for e in common], dtype=float).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def swap_test(quantities_a: Mapping[str, float], quantities_b: Mapping[str, float],
              interval_widths: Mapping[str, float], ranking_a: Sequence, ranking_b: Sequence,
              min_spearman: float = 0.8) -> dict:
    """The swap test: may rows from source B stand in for rows from source A?

    This is the stub of the test that gates computer-vision rows replacing tracking rows (docs/DATA_CONTRACT.md
    and the plan): the same quantities are computed once from tracking rows (a) and once from the CV rows
    that would replace them (b), and the swap is accepted only when every quantity's absolute difference is
    under its interval width (the bootstrap interval width of that quantity on tracking rows) and the
    Spearman rank correlation of the two entity rankings is at least min_spearman. The tolerance (the
    interval widths and min_spearman) is written down before any CV rows are generated, so the CV pipeline
    cannot be tuned to the test.

    Returns dict(passed, per_quantity={name: {diff, width, ok}}, spearman). A quantity missing from b or
    without a width fails; a missing ranking comparison (fewer than two shared entities) fails unless both
    rankings are empty."""
    per = {}
    for name, a in quantities_a.items():
        b = quantities_b.get(name)
        width = interval_widths.get(name)
        if b is None or width is None or not np.isfinite(a) or not np.isfinite(b):
            per[name] = {"diff": None if b is None else float(abs(float(a) - float(b))), "width": width, "ok": False}
            continue
        diff = float(abs(float(a) - float(b)))
        per[name] = {"diff": diff, "width": float(width), "ok": bool(diff < float(width))}
    for name in quantities_b:
        if name not in per:
            per[name] = {"diff": None, "width": interval_widths.get(name), "ok": False}
    rho = spearman(ranking_a, ranking_b)
    if rho is None:
        rank_ok = len(ranking_a) == 0 and len(ranking_b) == 0
    else:
        rank_ok = rho >= min_spearman
    passed = bool(per) and all(v["ok"] for v in per.values()) and rank_ok
    return {"passed": bool(passed), "per_quantity": per, "spearman": rho}


# ----------------------------------------------------------------------------- report block for trainers

def report_block(name: str, y, p, df: pd.DataFrame, game_col: str, n: int = 1000, seed: int = 0,
                 bins: int = 10) -> str:
    """Markdown block: log loss, Brier, ECE and calibration slope with game-level bootstrap intervals, then
    the calibration table. y and p are aligned to df, whose game_col gives the resampling unit."""
    y, p = _yp(y, p)
    if len(df) != len(y):
        raise ValueError("df has %d rows, y has %d" % (len(df), len(y)))
    d = pd.DataFrame({"game": df[game_col].astype(str).to_numpy(), "y": y, "p": p})
    metrics = [("log loss", lambda s: log_loss(s["y"], s["p"])),
               ("Brier", lambda s: brier(s["y"], s["p"])),
               ("ECE", lambda s: ece(s["y"], s["p"], bins)),
               ("calibration slope", lambda s: calibration_slope(s["y"], s["p"]))]
    games = d["game"].nunique()
    L = ["### %s" % name, "",
         "%d rows, %d games, base rate %.4f, bootstrap by game (n=%d, seed=%d, 95 percent percentile interval)"
         % (len(d), games, float(y.mean()), n, seed), "",
         "| metric | point | lo | hi |", "| --- | --- | --- | --- |"]
    for label, fn in metrics:
        r = bootstrap_by_game(d, "game", fn, n=n, seed=seed)
        L.append("| %s | %.4f | %.4f | %.4f |" % (label, r["point"], r["lo"], r["hi"]))
    L += ["", "| bin | lo | hi | n | mean_p | mean_y |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in calibration_table(y, p, bins).iterrows():
        mp = "%.4f" % r["mean_p"] if r["n"] else "-"
        my = "%.4f" % r["mean_y"] if r["n"] else "-"
        L.append("| %d | %.2f | %.2f | %d | %s | %s |" % (r["bin"], r["lo"], r["hi"], r["n"], mp, my))
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- shared helpers (consolidated after the
# code-quality review: every lane imports these instead of carrying its own copy)

def is_heldout_game(game_id) -> bool:
    """True when a SportVU-style game id carries a held-out token of sportvu/splits.json (game tags, SportVU ids,
    video ids) or the held-out arena suffix (".at.<arena>"). The arena rule is part of heldout_tokens, so one check
    covers both."""
    g = str(game_id)
    return any(t in g for t in _splits.heldout_tokens())


def heldout_mask(games) -> np.ndarray:
    """Boolean array, one entry per element of ``games`` (a Series, list or array of game ids): is_heldout_game each."""
    return np.array([is_heldout_game(g) for g in np.asarray(games, dtype=object)], dtype=bool)


def fit_threads(n: int = 1):
    """Context manager that pins BLAS and OpenMP pools to ``n`` threads for the fits inside it. Imports sklearn's
    HistGradientBoosting module first so its OpenMP runtime is loaded before the limit is taken (a limit taken earlier
    does not reach a library that is not loaded yet). Without threadpoolctl it is a no-op and says so once."""
    import contextlib, warnings
    try:
        import sklearn.ensemble._hist_gradient_boosting  # noqa: F401  (loads the OpenMP runtime)
    except Exception:  # noqa: BLE001  (sklearn absent or built without it; nothing to pin)
        pass
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        warnings.warn("threadpoolctl is not installed: fits run with the default thread pools", RuntimeWarning)
        return contextlib.nullcontext()
    return threadpool_limits(limits=n)


def calibration_slope_fast(y, p, ridge: float = 1e-6, max_iter: int = 50) -> float:
    """calibration_slope solved by Newton's method with a tiny ridge: the same regression (y on logit(p) with an
    intercept) and the same answer to two decimals on ordinary data, but bounded on nearly separable data where the
    lbfgs fit at C=1e6 runs to its iteration cap. Use it inside bootstrap loops. NaN when y has one class."""
    y, p = _yp(y, p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(max_iter):
        z = np.clip(X @ w, -30, 30)
        q = 1 / (1 + np.exp(-z))
        g = X.T @ (q - y) + ridge * w
        H = (X * (q * (1 - q))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[1])


def fmt_num(v, nd: int = 3) -> str:
    """A float to ``nd`` decimals, "n/a" when not finite."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return ("%%.%df" % nd) % v if np.isfinite(v) else "n/a"


def fmt_ci(d: Mapping, nd: int = 3, key: str = "point") -> str:
    """"point [lo, hi]" from a dict with ``key``, lo and hi; "point" alone when lo is not finite; "n/a" when the point
    is not finite. The one formatter every report uses for a bootstrap interval."""
    pt = d.get(key, float("nan"))
    if not np.isfinite(float(pt)) if pt is not None else True:
        return "n/a"
    lo, hi = d.get("lo", float("nan")), d.get("hi", float("nan"))
    if lo is None or not np.isfinite(float(lo)):
        return fmt_num(pt, nd)
    return "%s [%s, %s]" % (fmt_num(pt, nd), fmt_num(lo, nd), fmt_num(hi, nd))


def md_table(df: pd.DataFrame, nd: int = 3) -> list:
    """A DataFrame as markdown table lines (header, rule, rows); floats to ``nd`` decimals, NaN and None blank."""
    cols = [str(c) for c in df.columns]
    L = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else fmt_num(float(v), nd))
            else:
                cells.append("" if v is None else str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L
