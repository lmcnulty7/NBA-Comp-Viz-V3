"""analytics/gravity/eda.py: the tables of notebooks/eda_gravity.ipynb (tables and markdown only, no plots).

Every function takes frames the lane builds (features.build: the player frame pf and the spacing frame sp) or
the contract tables, and returns a DataFrame; markdown renders one (analytics.eval.md_table). run(games) builds everything
from a list of game dicts. On fixtures every table is fixture output, not a result.

Tables:
    nearest_def_by_distance_bin   mean and sd of the nearest-defender distance by distance-to-basket bin
    nearest_def_by_has_ball       the same by has_ball
    hull_area_by_end_kind         hull area and mean nearest-defender distance by the possession's end_kind
                                  (end_kind is a target column: it is joined here for exploration only, never
                                  into a feature frame)
    correlation_table             Pearson correlation of each feature with the target and with dist_basket_ft
    shuffled_y_check              ridge out of fold by game on the real target and on a permuted target: the
                                  permuted run should sit at R2 about 0 (a leak would score well on both)
The correlation table uses the zone prior from every game's shots (exploration, not a fit); the shuffled-y
check attaches the prior per fold from the training-fold games' shots, as the trainer does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import schema
from analytics.eval import fit_threads, grouped_folds, md_table, refuse_heldout
from analytics.gravity import features as F

DIST_BINS = (0.0, 4.0, 10.0, 16.0, 22.0, 30.0, 100.0)


def markdown(df: pd.DataFrame, nd: int = 3) -> str:
    """A DataFrame as one markdown string (analytics.eval.md_table, floats to nd decimals)."""
    return "\n".join(md_table(df, nd))


def nearest_def_by_distance_bin(pf: pd.DataFrame, bins=DIST_BINS) -> pd.DataFrame:
    """n, mean and sd of the nearest-defender distance by distance-to-basket bin."""
    cut = pd.cut(pf["dist_basket_ft"].astype(float), bins=list(bins), right=False)
    g = pf.groupby(cut, observed=True)[F.TARGET].agg(["size", "mean", "std"]).reset_index()
    g.columns = ["dist_basket_bin", "n", "mean_nearest_def_ft", "sd_nearest_def_ft"]
    g["dist_basket_bin"] = g["dist_basket_bin"].astype(str)
    return g


def nearest_def_by_has_ball(pf: pd.DataFrame) -> pd.DataFrame:
    """n, mean and sd of the nearest-defender distance for the handler and the other four."""
    g = pf.groupby(pf["has_ball"].astype(float) > 0.5)[F.TARGET].agg(["size", "mean", "std"]).reset_index()
    g.columns = ["has_ball", "n", "mean_nearest_def_ft", "sd_nearest_def_ft"]
    return g


def hull_area_by_end_kind(sp: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    """Mean hull area and mean nearest-defender distance by the possession's end_kind (a target: exploration only)."""
    keys = possessions[["sportvu_game", "possession", "end_kind"]].drop_duplicates()
    keys = keys.assign(sportvu_game=keys["sportvu_game"].astype(str), possession=keys["possession"].astype(np.int64))
    m = sp.assign(sportvu_game=sp["sportvu_game"].astype(str), possession=sp["possession"].astype(np.int64)).merge(
        keys, on=["sportvu_game", "possession"], how="left")
    m["end_kind"] = m["end_kind"].astype(object).fillna("unknown")
    g = m.groupby("end_kind")[["hull_area_ft2", "mean_nearest_def_ft"]].agg(["size", "mean"])
    out = pd.DataFrame({"end_kind": g.index, "n": g[("hull_area_ft2", "size")].to_numpy(),
                        "mean_hull_area_ft2": g[("hull_area_ft2", "mean")].to_numpy(),
                        "mean_nearest_def_ft": g[("mean_nearest_def_ft", "mean")].to_numpy()}).reset_index(drop=True)
    return out


def correlation_table(pf: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each feature with the target and with dist_basket_ft (NaN for a constant feature)."""
    rows = []
    y = pf[F.TARGET].astype(float)
    d = pf["dist_basket_ft"].astype(float)
    for c in F.FEATURES:
        v = pf[c].astype(float)
        if v.isna().all() or v.std() == 0:
            rows.append({"feature": c, "corr_with_target": float("nan"), "corr_with_dist_basket": float("nan")})
            continue
        rows.append({"feature": c, "corr_with_target": float(v.corr(y)), "corr_with_dist_basket": float(v.corr(d))})
    return pd.DataFrame(rows)


def _shots_of(games, ids) -> pd.DataFrame:
    """Concatenated shots of the game dicts whose id is in ids."""
    parts = [d["shots"] for d in games if F.game_id(d) in set(ids) and d.get("shots") is not None and len(d["shots"])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["made", "loc_x", "loc_y"])


def _ridge_oof(pf: pd.DataFrame, games: list, y: np.ndarray, seed: int, n_folds: int) -> np.ndarray:
    """Ridge out of fold by game for the shuffled-y check, with the zone prior attached per fold from the
    training-fold games' shots (as the trainer does). The same guards as the trainer: the frame passes
    schema.assert_time_forward and the game ids pass analytics.eval.refuse_heldout before any fit."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    schema.assert_time_forward(pf, "states_1hz")
    gid = pf["sportvu_game"].astype(str).to_numpy()
    uniq = sorted(set(gid))
    refuse_heldout(uniq, "gravity.eda")
    p = np.full(len(pf), np.nan)
    if len(uniq) < 2:
        X = F.design(F.attach_skill(pf, F.skill_table(_shots_of(games, uniq))))
        with fit_threads():
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y).predict(X)
    for tr_ids, te_ids in grouped_folds(uniq, n_folds=min(n_folds, len(uniq)), seed=seed):
        tr, te = np.isin(gid, tr_ids), np.isin(gid, te_ids)
        table = F.skill_table(_shots_of(games, tr_ids))
        Xtr, Xte = F.design(F.attach_skill(pf[tr], table)), F.design(F.attach_skill(pf[te], table))
        with fit_threads():
            p[te] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, y[tr]).predict(Xte)
    return p


def shuffled_y_check(pf: pd.DataFrame, games: list, seed: int = 0, n_folds: int = 5) -> pd.DataFrame:
    """Ridge out of fold by game on the real target and on a permuted one: rows (target, mse, r2)."""
    y = pf[F.TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for name, yy in (("real", y), ("shuffled", rng.permutation(y))):
        p = _ridge_oof(pf, games, yy, seed, n_folds)
        m = float(np.mean((yy - p) ** 2))
        rows.append({"target": name, "mse": m, "r2": float(1 - m / np.var(yy))})
    return pd.DataFrame(rows)


def run(games, seed: int = 0) -> dict:
    """All EDA tables from game dicts -> {name: DataFrame}."""
    pf, sp, counts = F.build(games)
    shots = pd.concat([d["shots"] for d in games if d.get("shots") is not None and len(d["shots"])], ignore_index=True) \
        if any(d.get("shots") is not None and len(d["shots"]) for d in games) else pd.DataFrame(columns=["sportvu_game", "player", "made", "loc_x", "loc_y"])
    pf_all = F.attach_skill(pf, F.skill_table(shots))      # every game's shots: the correlation table only (exploration)
    poss = pd.concat([d["possessions"] for d in games], ignore_index=True)
    return {"counts": pd.DataFrame(counts),
            "nearest_def_by_distance_bin": nearest_def_by_distance_bin(pf),
            "nearest_def_by_has_ball": nearest_def_by_has_ball(pf),
            "hull_area_by_end_kind": hull_area_by_end_kind(sp, poss),
            "correlation_table": correlation_table(pf_all),
            "shuffled_y_check": shuffled_y_check(pf, games, seed)}     # the prior per fold, as the trainer
