"""analytics/embeddings/eda.py: tables for notebooks/eda_embeddings.ipynb (no plots: pandas tables and markdown).

Every function takes the frames the lane already builds (player_season_features from analytics.embeddings.train
.build_features, a positions frame, an embedding matrix) and returns a DataFrame:

    feature_distribution_by_position(feats, positions)  long table feature, position, n, mean, sd, median, q10, q90
    correlation_table(feats, top=20)                    the top absolute pairwise correlations between features
    explained_variance(feats, seed=0)                   PCA component, explained_variance_ratio, cumulative
    archetype_separation(E, labels, k=5)                per label: n, within-label and between-label mean distance,
                                                        their ratio, and the leave-one-out k-NN accuracy

The notebook runs them on the fixture season: every table it prints is fixture output, not a result. The fits here
go through analytics.embeddings.train (standardise, fit_pca), whose feature frame passes
analytics.schema.assert_time_forward before anything is fitted; this module fits nothing on held-out games because
the loader never hands it one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.embeddings import features as F
from analytics.embeddings import train as T


def _fit_rows(feats: pd.DataFrame) -> pd.DataFrame:
    if "insufficient" in feats.columns:
        return feats[~feats["insufficient"].astype(bool)].reset_index(drop=True)
    return feats.reset_index(drop=True)


def feature_distribution_by_position(feats: pd.DataFrame, positions: pd.DataFrame | None) -> pd.DataFrame:
    """feature, position, n, mean, sd, median, q10, q90 over the sufficient rows; position 'all' when no positions."""
    rows = _fit_rows(feats)
    cols = F.available_features(rows)
    if positions is not None and len(positions):
        pos = positions.drop_duplicates(["player_id", "season"])[["player_id", "season", "position"]]
        rows = rows.merge(pos, how="left", on=["player_id", "season"])
        rows["position"] = rows["position"].fillna("unknown")
    else:
        rows["position"] = "all"
    out = []
    for c in cols:
        for p, grp in rows.groupby("position"):
            v = grp[c].astype(float).dropna()
            out.append({"feature": c, "position": p, "n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
                        "sd": float(v.std(ddof=0)) if len(v) else np.nan, "median": float(v.median()) if len(v) else np.nan,
                        "q10": float(v.quantile(0.1)) if len(v) else np.nan, "q90": float(v.quantile(0.9)) if len(v) else np.nan})
    return pd.DataFrame(out, columns=["feature", "position", "n", "mean", "sd", "median", "q10", "q90"])


def correlation_table(feats: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """feature_a, feature_b, r: the top absolute Pearson correlations between distinct features (sufficient rows)."""
    rows = _fit_rows(feats)
    cols = F.available_features(rows)
    if len(rows) < 3 or len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "r"])
    C = rows[cols].astype(float).corr()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = C.loc[a, b]
            if np.isfinite(r):
                out.append({"feature_a": a, "feature_b": b, "r": float(r)})
    t = pd.DataFrame(out, columns=["feature_a", "feature_b", "r"])
    t["abs_r"] = t["r"].abs()
    return t.sort_values("abs_r", ascending=False).drop(columns="abs_r").head(top).reset_index(drop=True)


def explained_variance(feats: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """component (1..8), explained_variance_ratio, cumulative of PCA on the standardised sufficient rows."""
    rows = _fit_rows(feats)
    cols = F.available_features(rows)
    if len(rows) < 3 or not cols:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative"])
    F.assert_no_leak(rows[cols])
    Z, _, _ = T.standardise(rows[cols])
    pca = T.fit_pca(Z, seed)
    ratio = np.asarray(pca["explained_variance_ratio"], dtype=float)
    return pd.DataFrame({"component": np.arange(1, len(ratio) + 1), "explained_variance_ratio": ratio,
                         "cumulative": np.cumsum(ratio)})


def archetype_separation(E: np.ndarray, labels, k: int = T.K_NN) -> pd.DataFrame:
    """Per label: n, within (mean distance to same-label rows), between (mean distance to other-label rows), ratio
    within / between (under 1 means the label clusters), knn_accuracy (leave-one-out k-NN over that label's rows)."""
    from scipy.spatial.distance import cdist
    labels = np.asarray(labels, dtype=object)
    if len(labels) < 2:
        return pd.DataFrame(columns=["label", "n", "within", "between", "ratio", "knn_accuracy"])
    D = cdist(E, E)
    pred = T.knn_position_predictions(E, labels, k)
    out = []
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        same = D[np.ix_(m, m)]
        n = int(m.sum())
        within = float(same[~np.eye(n, dtype=bool)].mean()) if n > 1 else np.nan
        between = float(D[np.ix_(m, ~m)].mean()) if (~m).any() else np.nan
        ratio = within / between if (np.isfinite(within) and np.isfinite(between) and between > 0) else np.nan
        out.append({"label": lab, "n": n, "within": within, "between": between, "ratio": ratio,
                    "knn_accuracy": float(np.mean(pred[m] == labels[m]))})
    return pd.DataFrame(out, columns=["label", "n", "within", "between", "ratio", "knn_accuracy"])


def embedding_with_labels(ctx: dict, positions: pd.DataFrame | None) -> tuple:
    """(E, labels) of the chosen method from a train.run context, rows without a position labelled 'unknown'."""
    emb = ctx["tables"]["player_embeddings"]
    emb = emb[emb["method"] == ctx["decision"]["chosen"]].reset_index(drop=True)
    E = emb[T.EMB_COLS].to_numpy(dtype=float)
    if positions is None or len(positions) == 0:
        return E, np.array(["unknown"] * len(emb), dtype=object)
    pos = positions.drop_duplicates(["player_id", "season"])[["player_id", "season", "position"]]
    j = emb[["player_id", "season"]].merge(pos, how="left", on=["player_id", "season"])
    return E, j["position"].fillna("unknown").astype(str).to_numpy(dtype=object)


__all__ = ["feature_distribution_by_position", "correlation_table", "explained_variance", "archetype_separation",
           "embedding_with_labels"]
