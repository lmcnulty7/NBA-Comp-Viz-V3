"""analytics/gravity/features.py: the 1 Hz player-frame feature frame and the spacing frame.

    python -m analytics.gravity.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]
        [--fixture-generator auto|fallback]

Inputs are game dicts (frames: contract section 4; possessions: section 19; rosters: section 16; shots:
section 10a) as tests/_fx_gravity.py, analytics.fixtures or validate.load_data_root give them.

Sampling (the unit). For every possession with gap false and a known offense_team_id, the span is the
frames of the same period from start_frame up to the next possession's start_frame (exclusive), or to the
period's last frame for the period's last possession. end_frame is a POST-OUTCOME column of the contract,
so the span is cut by the next start instead; on analytics.possessions output the next possession starts at
the previous one's end frame, so the span equals [start_frame, end_frame - 1]. Inside the span, elapsed =
start_clock_s minus game_clock_s, and the frame nearest each whole second of elapsed time is the sample (one
frame per second; frames over a stopped clock share one elapsed value and collapse to one sample). A sampled
frame is kept only with exactly five offensive and five defensive players and a ball position.

Target: nearest_def_dist_ft, the distance from the offensive player to the nearest defender.

FEATURES (time-forward, about location only; none is a target or post-outcome column of the contract):
    x_ft, y_ft                 the player's position (contract court frame, feet)
    dist_basket_ft             distance to the attacked basket
    angle_rad                  atan2(y - 25, distance along x from the basket toward midcourt); 0 straight on
    dist_ball_ft               distance to the ball
    has_ball                   1 when the player is the offensive player nearest the ball (the section 20 handler)
    ball_dist_basket_ft        the ball's distance to the attacked basket
    zone_make_rate             the league make rate in the zone the player stands in, from training-fold shots only
    game_clock_s               seconds remaining in the period
    shot_clock_s               seconds on the shot clock (SHOT_CLOCK_FILL when the source has none)
    shot_clock_missing         1 when shot_clock_s was null

The shooting prior is deliberately NOT player-specific. Gravity is the residual of the expectation, so any
feature that identifies a player lets a flexible model absorb that player's pull into the expectation (a
per-player skill prior did exactly that on the fixtures: a planted 2 ft pull came out at 0.6 ft under HGB).
zone_make_rate carries the location's shooting value (a corner is worth more than a long two) and nothing
about who stands there. ZONES are distance bins; the rate per zone comes from the shots of the training-fold
games (skill_table) and is attached per fold by the trainer (attach_skill); build() leaves it NaN.

Attacked basket (assumption): per (offense_team_id, period), the basket nearer the mean x of the sampled
offensive positions; the offence spends most of a possession in its frontcourt. The fixture does not swap
ends at half time and the features do not assume a swap either.

Spacing frame (one row per sampled possession-frame): hull_area_ft2, the convex hull area of the five
offensive players (scipy.spatial.ConvexHull; its volume attribute is the area in 2-D; NaN when the five
points are collinear), mean_nearest_def_ft, the mean of the five nearest-defender distances, and heldout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from analytics import schema

FEATURES = ["x_ft", "y_ft", "dist_basket_ft", "angle_rad", "dist_ball_ft", "has_ball", "ball_dist_basket_ft",
            "zone_make_rate", "game_clock_s", "shot_clock_s", "shot_clock_missing"]
KEYS = ["sportvu_game", "possession", "frame", "player_id"]
TARGET = "nearest_def_dist_ft"
LEFT_BASKET, RIGHT_BASKET = (5.25, 25.0), (88.75, 25.0)
MID_X = 47.0
SHOT_CLOCK_FILL = 24.0
ZONES = ((0.0, 4.0, "rim"), (4.0, 10.0, "paint"), (10.0, 16.0, "short_mid"), (16.0, 22.0, "long_mid"), (22.0, np.inf, "three"))
THREE_FT = 22.0
SKILL_COLUMNS = ["zone_make_rate"]
PRIOR_DEFAULT = 0.45
SPACING_COLUMNS = ["sportvu_game", "possession", "frame", "hull_area_ft2", "mean_nearest_def_ft", "heldout"]


def feature_columns() -> list:
    """The columns the models are fed, in FEATURES order."""
    return list(FEATURES)


def game_id(d: dict) -> str:
    """The sportvu_game of a game dict: its sportvu_game key, else meta.sportvu_game (analytics.fixtures), else
    the possessions table, else the frames' game column."""
    if d.get("sportvu_game"):
        return str(d["sportvu_game"])
    meta = d.get("meta") or {}
    if meta.get("sportvu_game"):
        return str(meta["sportvu_game"])
    poss = d.get("possessions")
    if poss is not None and len(poss):
        return str(poss["sportvu_game"].iloc[0])
    return str(d["frames"]["game"].iloc[0])


def is_heldout(d: dict) -> bool:
    """The dict's heldout flag (its key or meta.heldout), or a held-out token of sportvu/splits.json in its id,
    or a game at the CLE arena."""
    from sportvu import splits
    if d.get("heldout") or (d.get("meta") or {}).get("heldout"):
        return True
    g = game_id(d)
    toks = splits.heldout_tokens()
    return any(t in g for t in toks) or g.endswith(".at." + splits.HELDOUT_ARENA)


def known_gravity_ids(d: dict) -> list:
    """The planted high-gravity players of a fixture game: high_gravity_ids (tests/_fx_gravity.py fallback) or
    truth.star_player_ids (analytics.fixtures); empty on real data."""
    ids = d.get("high_gravity_ids") or (d.get("truth") or {}).get("star_player_ids") or []
    return [int(p) for p in ids]


# ── sampling ───────────────────────────────────────────────────────────────────────────────────────
def frame_level(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per frame: frame, period, game_clock_s, shot_clock_s, ball_x, ball_y (first row of the frame)."""
    cols = ["frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y"]
    fl = frames.drop_duplicates("frame", keep="first")[[c for c in cols if c in frames.columns]]
    if "shot_clock_s" not in fl.columns:
        fl = fl.assign(shot_clock_s=np.nan)
    return fl.sort_values("frame", kind="stable").reset_index(drop=True)


def sample_frames(frames: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    """The 1 Hz samples: one row per (possession, frame) with elapsed_s, period, offense_team_id, start_clock_s."""
    fl = frame_level(frames)
    fr = fl["frame"].to_numpy(dtype=np.int64)
    per = fl["period"].to_numpy(dtype=np.int64)
    clk = fl["game_clock_s"].to_numpy(dtype=np.float64)
    poss = possessions.sort_values(["period", "start_frame"], kind="stable").reset_index(drop=True)
    last_frame_of_period = {int(p): int(fr[per == p].max()) for p in np.unique(per)}
    rows = []
    for i, r in enumerate(poss.itertuples(index=False)):
        if bool(r.gap) or pd.isna(r.offense_team_id):
            continue
        period = int(r.period)
        if period not in last_frame_of_period:
            continue
        nxt = poss.iloc[i + 1] if i + 1 < len(poss) else None
        if nxt is not None and int(nxt["period"]) == period:
            stop = int(nxt["start_frame"]) - 1
        else:
            stop = last_frame_of_period[period]
        start = int(r.start_frame)
        m = (fr >= start) & (fr <= stop) & (per == period)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        elapsed = float(r.start_clock_s) - clk[idx]
        elapsed = np.where(elapsed < 0, 0.0, elapsed)
        seen = set()
        for s in range(int(np.floor(elapsed.max())) + 1):
            j = idx[int(np.argmin(np.abs(elapsed - s)))]
            f = int(fr[j])
            if f in seen:
                continue
            seen.add(f)
            rows.append({"possession": int(r.possession), "frame": f, "elapsed_s": float(s), "period": period,
                         "offense_team_id": int(r.offense_team_id), "start_clock_s": float(r.start_clock_s)})
    return pd.DataFrame(rows, columns=["possession", "frame", "elapsed_s", "period", "offense_team_id", "start_clock_s"])


# ── per-game build ─────────────────────────────────────────────────────────────────────────────────
def _hull_area(pts: np.ndarray) -> float:
    try:
        return float(ConvexHull(pts).volume)      # 2-D: volume is the area, area would be the perimeter
    except (QhullError, ValueError):
        return float("nan")


def hull_area_ft2(points) -> float:
    """Convex hull area in square feet of an (n, 2) array of positions; NaN when degenerate."""
    return _hull_area(np.asarray(points, dtype=float))


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    pf = pd.DataFrame(columns=KEYS + ["heldout", "period", "elapsed_s", "offense_team_id"] + FEATURES + [TARGET])
    return pf, pd.DataFrame(columns=SPACING_COLUMNS)


def build_game(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(player_frame, spacing_frame, counts) of one game dict. zone_make_rate is left at NaN (train.py attaches it
    per fold with attach_skill); the frame carries sportvu_game, possession, frame, player_id, heldout, period,
    elapsed_s, offense_team_id, the FEATURES and the TARGET."""
    g = game_id(d)
    held = bool(is_heldout(d))
    frames, poss = d["frames"], d["possessions"]
    counts = {"sportvu_game": g, "possessions": int(len(poss)), "sampled": 0, "kept": 0, "not_ten": 0, "no_ball": 0,
              "degenerate_hull": 0}
    samp = sample_frames(frames, poss)
    counts["sampled"] = int(len(samp))
    if samp.empty:
        return (*_empty_frames(), counts)
    sub = frames[frames["frame"].isin(samp["frame"])].copy()
    sub = sub.merge(samp[["frame", "possession", "elapsed_s", "offense_team_id", "start_clock_s"]], on="frame", how="inner")
    sub["is_def"] = (sub["team_id"].to_numpy() != sub["offense_team_id"].to_numpy())
    cnt = sub.groupby("frame")["is_def"].agg(["size", "sum"])
    ok = cnt.index[(cnt["size"] == 10) & (cnt["sum"] == 5)]
    counts["not_ten"] = int(len(cnt) - len(ok))
    sub = sub[sub["frame"].isin(ok)]
    has_ball = sub["ball_x"].notna() & sub["ball_y"].notna()
    no_ball = sub.loc[~has_ball, "frame"].unique()
    counts["no_ball"] = int(len(no_ball))
    sub = sub[~sub["frame"].isin(no_ball)]
    if sub.empty:
        return (*_empty_frames(), counts)
    sub = sub.sort_values(["frame", "is_def", "player_id"], kind="stable").reset_index(drop=True)
    n = len(sub) // 10
    xy = sub[["x", "y"]].to_numpy(dtype=float).reshape(n, 10, 2)
    off, dfn = xy[:, :5, :], xy[:, 5:, :]
    dist = np.sqrt(((off[:, :, None, :] - dfn[:, None, :, :]) ** 2).sum(-1))     # (n, 5, 5)
    nearest = dist.min(axis=2)                                                   # (n, 5)
    head = sub.iloc[::10].reset_index(drop=True)                                 # frame-level rows, offence first
    off_rows = sub[~sub["is_def"]]
    mean_x = off_rows.groupby(["offense_team_id", "period"])["x"].mean()
    basket_left = {k: bool(v < MID_X) for k, v in mean_x.items()}
    left = np.array([basket_left[(int(t), int(p))] for t, p in zip(head["offense_team_id"], head["period"])])
    bx = np.where(left, LEFT_BASKET[0], RIGHT_BASKET[0])[:, None]
    by = np.full((n, 1), LEFT_BASKET[1])
    toward_mid = np.where(left, 1.0, -1.0)[:, None]
    ball = head[["ball_x", "ball_y"]].to_numpy(dtype=float)
    dist_basket = np.hypot(off[:, :, 0] - bx, off[:, :, 1] - by)
    angle = np.arctan2(off[:, :, 1] - by, (off[:, :, 0] - bx) * toward_mid)
    dist_ball = np.hypot(off[:, :, 0] - ball[:, None, 0], off[:, :, 1] - ball[:, None, 1])
    handler = dist_ball.argmin(axis=1)
    has = np.zeros((n, 5), dtype=float)
    has[np.arange(n), handler] = 1.0
    ball_basket = np.hypot(ball[:, 0] - bx[:, 0], ball[:, 1] - by[:, 0])
    sc = head["shot_clock_s"].to_numpy(dtype=float) if "shot_clock_s" in head.columns else np.full(n, np.nan)
    sc_missing = ~np.isfinite(sc)
    sc = np.where(sc_missing, SHOT_CLOCK_FILL, sc)
    rep = lambda a: np.repeat(np.asarray(a), 5)  # noqa: E731
    off_ids = sub.loc[~sub["is_def"], "player_id"].to_numpy(dtype=np.int64)
    pf = pd.DataFrame({
        "sportvu_game": g, "possession": rep(head["possession"].to_numpy(dtype=np.int64)),
        "frame": rep(head["frame"].to_numpy(dtype=np.int64)), "player_id": off_ids, "heldout": held,
        "period": rep(head["period"].to_numpy(dtype=np.int64)), "elapsed_s": rep(head["elapsed_s"].to_numpy(dtype=float)),
        "offense_team_id": rep(head["offense_team_id"].to_numpy(dtype=np.int64)),
        "x_ft": off[:, :, 0].ravel(), "y_ft": off[:, :, 1].ravel(), "dist_basket_ft": dist_basket.ravel(),
        "angle_rad": angle.ravel(), "dist_ball_ft": dist_ball.ravel(), "has_ball": has.ravel(),
        "ball_dist_basket_ft": rep(ball_basket), "zone_make_rate": np.nan,
        "game_clock_s": rep(head["game_clock_s"].to_numpy(dtype=float)), "shot_clock_s": rep(sc),
        "shot_clock_missing": rep(sc_missing.astype(float)), TARGET: nearest.ravel()})
    areas = np.array([_hull_area(off[i]) for i in range(n)])
    counts["degenerate_hull"] = int(np.isnan(areas).sum())
    sp = pd.DataFrame({"sportvu_game": g, "possession": head["possession"].to_numpy(dtype=np.int32),
                       "frame": head["frame"].to_numpy(dtype=np.int32), "hull_area_ft2": areas,
                       "mean_nearest_def_ft": nearest.mean(axis=1), "heldout": held})
    counts["kept"] = int(n)
    return pf, sp, counts


def build(games) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Concatenate build_game over game dicts -> (player_frame, spacing_frame, per-game counts).
    schema.assert_time_forward runs on the returned player frame (states_1hz is the nearest contract table)."""
    pfs, sps, counts = [], [], []
    for d in games:
        pf, sp, c = build_game(d)
        pfs.append(pf)
        sps.append(sp)
        counts.append(c)
    if pfs:
        pf, sp = pd.concat(pfs, ignore_index=True), pd.concat(sps, ignore_index=True)
    else:
        pf, sp = _empty_frames()
    if len(sp):
        sp = sp.astype({"possession": np.int32, "frame": np.int32, "heldout": bool})
    schema.assert_time_forward(pf, "states_1hz")
    return pf, sp, counts


# ── shooting prior (league rate by zone, non-identifying) ─────────────────────────────────────────
def zone_of(dist_ft) -> np.ndarray:
    """Zone name of each distance to the basket (ZONES)."""
    d = np.asarray(dist_ft, dtype=float)
    out = np.full(d.shape, ZONES[-1][2], dtype=object)
    for lo, hi, name in ZONES:
        out[(d >= lo) & (d < hi)] = name
    return out


def shot_player_ids(shots: pd.DataFrame, rosters: pd.DataFrame) -> pd.Series:
    """SportVU player id of each shot: the shots' player_id when present, else the name matched against the
    roster of the same game (bref_name, then player, then pbp_name). NaN when unmatched. Used by the sign test
    and validate, never by the prior (which is not player-specific)."""
    out = pd.Series(np.nan, index=shots.index, dtype=float)
    if "player_id" in shots.columns:
        out = pd.to_numeric(shots["player_id"], errors="coerce").astype(float)
    if out.notna().all() or rosters is None or len(rosters) == 0 or "player" not in shots.columns:
        return out
    lookup = {}
    for col in ("pbp_name", "player", "bref_name"):   # later columns win: bref_name is the shot chart spelling
        if col in rosters.columns:
            for g, name, pid in zip(rosters["sportvu_game"], rosters[col], rosters["player_id"]):
                if isinstance(name, str) and name:
                    lookup[(str(g), name)] = int(pid)
    miss = out.isna()
    keys = list(zip(shots.loc[miss, "sportvu_game"].astype(str), shots.loc[miss, "player"].astype(str)))
    out.loc[miss] = [lookup.get(k, np.nan) for k in keys]
    return out


def _shot_distance(shots: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(shots["loc_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(shots["loc_y"], errors="coerce").to_numpy(dtype=float)
    pixel = np.hypot(x, y) / 10.0
    if "shot_distance_ft" in shots.columns:
        stated = pd.to_numeric(shots["shot_distance_ft"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(stated), stated, pixel)
    return pixel


def skill_table(shots: pd.DataFrame, rosters: pd.DataFrame = None) -> dict:
    """The zone prior from a shots table (training folds only, the caller filters):
    dict(prior_zone={zone: league make rate}, n_shots, n_by_zone). An empty table gives PRIOR_DEFAULT everywhere.
    Nothing here is per player: rosters is accepted for signature stability and not used."""
    out = {"prior_zone": {z[2]: PRIOR_DEFAULT for z in ZONES}, "n_shots": int(len(shots)) if shots is not None else 0,
           "n_by_zone": {z[2]: 0 for z in ZONES}}
    if shots is None or len(shots) == 0:
        return out
    made = shots["made"].astype(bool).to_numpy().astype(float)
    zone = zone_of(_shot_distance(shots))
    for z in np.unique(zone):
        m = zone == z
        out["prior_zone"][str(z)] = float(made[m].mean())
        out["n_by_zone"][str(z)] = int(m.sum())
    return out


def attach_skill(pf: pd.DataFrame, table: dict) -> pd.DataFrame:
    """pf with zone_make_rate filled from a skill_table by the zone of the player's position."""
    out = pf.copy()
    zone = zone_of(out["dist_basket_ft"].to_numpy(dtype=float))
    pz = table["prior_zone"]
    out["zone_make_rate"] = np.array([pz.get(str(z), PRIOR_DEFAULT) for z in zone], dtype=float)
    return out


def design(pf: pd.DataFrame) -> np.ndarray:
    """The design matrix (FEATURES, float64). assert_time_forward guards the frame first."""
    schema.assert_time_forward(pf, "states_1hz")
    X = pf[FEATURES].astype(float)
    if X[SKILL_COLUMNS].isna().any().any():
        raise ValueError("gravity.features.design: skill columns are NaN; call attach_skill first")
    return X.to_numpy()


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> dict:
    from analytics.gravity import validate as V
    ap = argparse.ArgumentParser(description="gravity feature frame and spacing frame")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", default=None, help="DATA_ROOT (owner's Mac layout)")
    ap.add_argument("--out", default=None, help="OUT_ROOT: writes OUT_ROOT/analytics/gravity/{player_frames,spacing_frames}.parquet")
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-games", type=int, default=8)
    ap.add_argument("--n-possessions", type=int, default=40)
    ap.add_argument("--fixture-generator", default="auto", choices=("auto", "fallback"))
    a = ap.parse_args(argv)
    games, vcounts = V.validate_games(V.load_games(a, who="gravity.features"))
    pf, sp, counts = build(games)
    summary = {"games": len(games), "player_frames": int(len(pf)), "spacing_frames": int(len(sp)), "counts": counts,
               "validation": vcounts, "features": FEATURES, "fixture": bool(a.fixtures)}
    if a.out:
        root = Path(a.out) / "analytics" / "gravity"
        root.mkdir(parents=True, exist_ok=True)
        pf.to_parquet(root / "player_frames.parquet", index=False)
        sp.to_parquet(root / "spacing_frames.parquet", index=False)      # same schema as train.py's spacing_frames
        summary["written"] = str(root)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("counts", "validation")}, indent=1))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
