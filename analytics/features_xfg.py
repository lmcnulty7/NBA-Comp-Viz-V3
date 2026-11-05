"""analytics/features_xfg.py: the location-only features of the xFG baseline (docs/DATA_CONTRACT.md sections 10 and 10a).

features(shots) keeps the keys (sportvu_game, game_id when present, event), the target (made), the held-out flag and
four location features computed from the shot table alone:

    distance_ft  shot_distance_ft, else sqrt(loc_x^2 + loc_y^2) / 10 (loc_* are tenths of a foot from the basket)
    angle_rad    atan2(loc_x, loc_y): 0 straight on from the basket, positive toward page right
    is_three     shot_type == 3PT when the page states it, else distance_ft >= 22
    is_corner    abs(loc_x) > 220 and loc_y < 90 (a corner three, in tenths of a foot)

plus period and clock_s for bucketing the metrics. Nothing else leaves this module: the SportVU-derived defender
columns and anything that encodes the outcome are listed in FORBIDDEN and assert_no_leak refuses a design matrix
that carries one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORBIDDEN = {"made", "points", "score", "release_frame", "closest_defender_id", "closest_defender_dist_ft",
             "defender_closing_speed_ftps"}
THREE_FT = 22.0
CORNER_X_TENTHS, CORNER_Y_TENTHS = 220.0, 90.0


def feature_columns() -> list:
    """The columns the baseline model is fed (period and clock_s are kept for buckets only)."""
    return ["distance_ft", "angle_rad", "is_three", "is_corner"]


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a column that encodes the outcome or a SportVU-derived field."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("features_xfg: forbidden columns in the design matrix: %s" % bad)


def _heldout(shots: pd.DataFrame) -> pd.Series:
    if "heldout" in shots.columns:
        return shots["heldout"].fillna(False).astype(bool)
    if "split" in shots.columns:
        return shots["split"].astype("string").fillna("").str.startswith("heldout").astype(bool)
    return pd.Series(False, index=shots.index)


def features(shots: pd.DataFrame) -> pd.DataFrame:
    """The section 10 or 10a shots table -> the xFG feature frame (one row per input row, same order)."""
    for c in ("sportvu_game", "made", "loc_x", "loc_y", "period", "clock_s"):
        if c not in shots.columns:
            raise ValueError("features_xfg: shots table lacks column %r" % c)
    x = shots["loc_x"].astype(float).to_numpy()
    y = shots["loc_y"].astype(float).to_numpy()
    pixel_ft = np.hypot(x, y) / 10.0
    if "shot_distance_ft" in shots.columns:
        stated = shots["shot_distance_ft"].astype(float).to_numpy()
        distance = np.where(np.isfinite(stated), stated, pixel_ft)
    else:
        distance = pixel_ft
    by_dist = distance >= THREE_FT
    if "shot_type" in shots.columns:
        st = shots["shot_type"].astype("string")
        is_three = np.where(st.isna().to_numpy(), by_dist, (st == "3PT").fillna(False).to_numpy())
    else:
        is_three = by_dist
    out = pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype("string").to_numpy()}, index=shots.index)
    if "game_id" in shots.columns:
        out["game_id"] = shots["game_id"].astype("string").to_numpy()
    if "event_id" in shots.columns and shots["event_id"].notna().any():
        out["event"] = shots["event_id"].to_numpy()
    else:
        out["event"] = np.arange(len(shots))
    out["made"] = shots["made"].astype(bool).to_numpy()
    out["heldout"] = _heldout(shots).to_numpy()
    out["distance_ft"] = distance.astype(float)
    out["angle_rad"] = np.arctan2(x, y)
    out["is_three"] = is_three.astype(bool)
    out["is_corner"] = ((np.abs(x) > CORNER_X_TENTHS) & (y < CORNER_Y_TENTHS)).astype(bool)
    out["period"] = shots["period"].to_numpy()
    out["clock_s"] = shots["clock_s"].astype(float).to_numpy()
    return out
