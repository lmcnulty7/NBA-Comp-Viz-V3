"""analytics/coverage/eda.py: exploratory tables of the coverage feature frame (notebooks/eda_coverage.ipynb
calls these; output is tables and markdown, no plots).

    class_balance(feats)                  n and share per class, labelled rows only
    feature_distributions_by_class(feats) mean, std, min, max of every feature per class (long table)
    correlation_table(feats)              Pearson correlation matrix of the features (NaN pairs dropped pairwise)
    class_separation(feats)               per feature: the F statistic of a one-way ANOVA across classes (scipy)
    shuffled_label_check(feats)           out-of-fold macro F1 of the logistic candidate on the true labels and on
                                          labels permuted across rows, next to the majority baseline: a model that
                                          scores well on permuted labels is reading something it should not
    fixture_feature_frame(seed)           the feature frame of the in-memory fixture season (fixture output)

Every function takes the feature frame of analytics.coverage.features.features (held-out rows are dropped
first: nothing here tunes anything, but the held-out games stay unseen all the same). The feature frame has
passed analytics.schema.assert_time_forward; the shuffled-label check refits through
analytics.coverage.train.fit_models, which refuses held-out rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.coverage import CLASSES
from analytics.coverage.features import feature_columns


def _labelled(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[(~feats["heldout"].astype(bool)) & feats["coverage_label"].notna()].reset_index(drop=True)


def class_balance(feats: pd.DataFrame) -> pd.DataFrame:
    """n and share per class over the labelled rows of training games."""
    d = _labelled(feats)
    n = d["coverage_label"].value_counts().reindex(list(CLASSES), fill_value=0)
    return pd.DataFrame({"class": list(CLASSES), "n": n.to_numpy(), "share": (n / max(1, n.sum())).round(4).to_numpy()})


def feature_distributions_by_class(feats: pd.DataFrame) -> pd.DataFrame:
    """Long table: feature, class, n, mean, std, min, max over the labelled rows of training games."""
    d = _labelled(feats)
    rows = []
    for f in feature_columns():
        for k, part in d.groupby("coverage_label"):
            v = part[f].astype(float)
            rows.append({"feature": f, "class": k, "n": int(v.notna().sum()), "mean": float(v.mean()), "std": float(v.std()),
                         "min": float(v.min()), "max": float(v.max())})
    return pd.DataFrame(rows, columns=["feature", "class", "n", "mean", "std", "min", "max"])


def correlation_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the features over the labelled rows of training games."""
    d = _labelled(feats)
    return d[feature_columns()].astype(float).corr(method="pearson")


def class_separation(feats: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA F per feature across the classes present (scipy.stats.f_oneway); NaN when a feature is
    constant or a class has one row."""
    from scipy import stats
    d = _labelled(feats)
    rows = []
    for f in feature_columns():
        groups = [part[f].astype(float).dropna().to_numpy() for _, part in d.groupby("coverage_label")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2 or all(np.allclose(g, g[0]) for g in groups):
            F, p = float("nan"), float("nan")
        else:
            res = stats.f_oneway(*groups)
            F, p = float(res.statistic), float(res.pvalue)
        rows.append({"feature": f, "F": F, "p": p})
    return pd.DataFrame(rows).sort_values("F", ascending=False, na_position="last").reset_index(drop=True)


def shuffled_label_check(feats: pd.DataFrame, seed: int = 0, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold macro F1 and accuracy of the logistic candidate on the true labels and on labels permuted
    across rows (same folds), with the majority baseline. Expect the permuted row near or below majority."""
    from analytics.coverage.train import accuracy, macro_f1, out_of_fold, predicted
    d = _labelled(feats)
    rng = np.random.default_rng(seed)
    rows = []
    for name, labels in (("true labels", d["coverage_label"].to_numpy()),
                         ("permuted labels", rng.permutation(d["coverage_label"].to_numpy()))):
        dd = d.copy()
        dd["coverage_label"] = labels
        _fold, probs, note = out_of_fold(dd, n_folds, seed)
        y = dd["coverage_label"].astype(str).to_numpy()
        for m in ("majority", "logistic"):
            yh = predicted(probs[m])
            rows.append({"labels": name, "model": m, "macro_f1": macro_f1(y, yh), "accuracy": accuracy(y, yh), "mode": note["mode"]})
    return pd.DataFrame(rows)


def fixture_feature_frame(seed: int = 0, n_games: int = 8, heldout_every: int = 0) -> pd.DataFrame:
    """The feature frame of the in-memory fixture season: fixture output, not a result."""
    from analytics.coverage.features import features
    from analytics.coverage.validate import fixture_bundle, validate_bundle
    b = fixture_bundle(seed=seed, n_games=n_games, heldout_every=heldout_every)
    cand, counts = validate_bundle(b)
    return features(cand, b["games"], counts)


__all__ = ["class_balance", "feature_distributions_by_class", "correlation_table", "class_separation",
           "shuffled_label_check", "fixture_feature_frame"]
