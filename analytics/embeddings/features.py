"""analytics/embeddings/features.py: player-season style features (docs/contract/embeddings.md, table player_season_features).

    python -m analytics.embeddings.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--n-min 300]

Unit: player-season. Every aggregate is built from TRAINING games only: the loader (analytics.embeddings.train.load_inputs)
drops held-out games (sportvu/splits.json ids, the CLE arena, heldout/ directories, heldout flags) before any table
reaches this module, and ``player_season_features`` refuses a game id that carries a held-out token.

Features (the FEATURES constant, in order):

  tracking (section 4 frames joined to section 19 possessions; sums per game, then combined over games so a season of
  600 games never sits in memory at once):
    avg_speed_ftps         mean speed over consecutive frames of the same game and period, 0 < dt <= 2 s (the gap rule),
                           steps faster than 30 ft/s dropped as tracking glitches
    dist_per_poss_ft       distance covered over those steps / n_possessions
    zone_<near|mid|far>_<left|right>  share of the player's OFFENSIVE frames (own team is the possession's offence) in each
                           cell of a 3 by 2 grid of the attacked half: x bands of 47/3 ft from the attacked baseline, y
                           halves; positions are mirrored so the attacked rim sits at (5.25, 25). The attacked basket of a
                           (game, period, team) is the side where the ball spends its offensive frames. Frames in the
                           backcourt count in the denominator only, so the six shares sum to at most 1.
    ball_share             share of offensive frames where the player is the handler: nearest to the ball among the
                           frame's players, within 3 ft horizontally, ball below 8 ft
    nearest_teammate_ft    mean distance to the nearest teammate on the floor, all frames
    offball_ball_dist_ft   mean distance to the ball over offensive frames where the player is not the handler
  shots (section 10 or 10a table of the tracked training games; a null player_id is resolved through the section 16
  rosters by player_slug = bref_slug):
    shot_share_<zone>      share of the player's attempts by zone_basic in {restricted_area, paint_non_ra, mid_range,
                           corner_3, above_break_3, backcourt}; NBA.com zone names map onto these, and a table without a
                           zone column gets zones derived from (court_x, court_y, shot_type) in the left-basket frame
    shot_share_3pt         share of attempts that are 3PT
    attempts_per_poss      attempts / n_possessions (tracking possessions the player was on the floor for)
  NBA.com aggregates (section 11 long table of the SAME season pivoted wide; absent on a machine without them, in which
  case the trainer uses the tracking and shot features only):
    agg_drives, agg_touches, agg_passes_made (leaguedashptstats), agg_catch_shoot_fga, agg_pull_up_fga
    (leaguedashplayerptshot), agg_deflections, agg_contested_shots, agg_screen_assists (leaguehustlestatsplayer)
    pt_<play_type>_freq    synergyplaytypes poss_pct per play type (isolation, transition, prballhandler, prrollman,
                           postup, spotup, handoff, cut, offscreen, offrebound, misc), type_grouping offensive; a player
                           without a row for a play type has frequency 0 (NBA.com lists qualifying players only)

n_possessions is the number of (game, possession) spans the player has frames in; ``insufficient`` is true under
n_min (300 in the contract; fixture runs lower it and say so). The returned frame passes
analytics.schema.assert_time_forward and carries no target, post-outcome or next-season column.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from sportvu import splits as _splits

log = logging.getLogger("analytics.embeddings.features")

WHO = "embeddings.features"
N_MIN = 300
COURT_X, COURT_Y = 94.0, 50.0
RIM_X, RIM_Y = 5.25, 25.0
HALF_X = COURT_X / 2.0
BAND_X = HALF_X / 3.0
MAX_STEP_S = 2.0           # the section 4 gap rule
MAX_SPEED_FTPS = 30.0
HANDLER_FT, HANDLER_BALL_Z = 3.0, 8.0

ZONE_X = ("near", "mid", "far")
ZONE_Y = ("left", "right")
ZONE_FEATURES = ["zone_%s_%s" % (bx, by) for bx in ZONE_X for by in ZONE_Y]
TRACKING_FEATURES = ["avg_speed_ftps", "dist_per_poss_ft"] + ZONE_FEATURES + \
    ["ball_share", "nearest_teammate_ft", "offball_ball_dist_ft"]
SHOT_ZONES = ("restricted_area", "paint_non_ra", "mid_range", "corner_3", "above_break_3", "backcourt")
SHOT_FEATURES = ["shot_share_%s" % z for z in SHOT_ZONES] + ["shot_share_3pt", "attempts_per_poss"]
PLAY_TYPES = ("isolation", "transition", "prballhandler", "prrollman", "postup", "spotup", "handoff", "cut",
              "offscreen", "offrebound", "misc")
AGG_METRICS = (("leaguedashptstats", "drives", "agg_drives"),
               ("leaguedashptstats", "touches", "agg_touches"),
               ("leaguedashptstats", "passes_made", "agg_passes_made"),
               ("leaguedashplayerptshot", "catch_shoot_fga", "agg_catch_shoot_fga"),
               ("leaguedashplayerptshot", "pull_up_fga", "agg_pull_up_fga"),
               ("leaguehustlestatsplayer", "deflections", "agg_deflections"),
               ("leaguehustlestatsplayer", "contested_shots", "agg_contested_shots"),
               ("leaguehustlestatsplayer", "screen_assists", "agg_screen_assists"))
PLAY_TYPE_FEATURES = ["pt_%s_freq" % p for p in PLAY_TYPES]
AGG_FEATURES = [c for _, _, c in AGG_METRICS] + PLAY_TYPE_FEATURES
FEATURES = TRACKING_FEATURES + SHOT_FEATURES + AGG_FEATURES
KEYS = ["player_id", "season"]
NBA_ZONE_MAP = {"restricted area": "restricted_area", "in the paint (non-ra)": "paint_non_ra", "mid-range": "mid_range",
                "left corner 3": "corner_3", "right corner 3": "corner_3", "above the break 3": "above_break_3",
                "backcourt": "backcourt"}


def feature_columns(with_aggregates: bool = True) -> list:
    """The feature names in order; without the NBA.com aggregate block when the machine has no aggregates."""
    return list(FEATURES) if with_aggregates else TRACKING_FEATURES + SHOT_FEATURES


def assert_no_leak(df: pd.DataFrame) -> None:
    """ValueError when the frame carries a target, post-outcome or id column of the five tables, or a position label."""
    schema.assert_time_forward(df, "possessions")
    bad = sorted(c for c in df.columns if c in {"position", "next_season_target"} or c.endswith("_next"))
    if bad:
        raise ValueError("%s: label or next-season column(s) in the feature frame: %s" % (WHO, bad))


def refuse_heldout_games(games, who: str = WHO) -> int:
    """SystemExit when a game id carries a held-out token of sportvu/splits.json or sits in the held-out arena."""
    n = _splits.refuse_heldout(list(games), who)
    arena = [g for g in games if str(g).endswith(".at." + _splits.HELDOUT_ARENA)]
    if arena:
        raise SystemExit("%s refuses to run: %d game(s) in the held-out arena %s, e.g. %s"
                         % (who, len(arena), _splits.HELDOUT_ARENA, arena[:3]))
    return n


def season_of_date(date) -> str:
    d = pd.to_datetime(date)
    y = d.year if d.month >= 10 else d.year - 1
    return "%d-%02d" % (y, (y + 1) % 100)


# ----------------------------------------------------------------------------- tracking: one game -> partial sums

PARTIAL_SUMS = ["speed_sum", "speed_n", "dist_sum", "off_frames", "handler_frames", "all_frames", "teammate_sum",
                "teammate_n", "offball_sum", "offball_n"] + ["%s_n" % z for z in ZONE_FEATURES]


def _possession_of_frames(frame_ids: np.ndarray, poss: pd.DataFrame):
    """(possession index, offense_team_id) per frame id; -1 and -1 outside every span."""
    if len(poss) == 0:
        return np.full(len(frame_ids), -1), np.full(len(frame_ids), -1)
    p = poss.sort_values("start_frame")
    starts = p["start_frame"].to_numpy(dtype=np.int64)
    ends = p["end_frame"].to_numpy(dtype=np.int64)
    pid = p["possession"].to_numpy(dtype=np.int64)
    off = p["offense_team_id"].astype("float").fillna(-1).to_numpy(dtype=np.int64)
    k = np.searchsorted(starts, frame_ids, side="right") - 1
    ok = (k >= 0) & (frame_ids <= ends[np.clip(k, 0, len(ends) - 1)])
    k = np.where(ok, k, -1)
    return np.where(ok, pid[np.clip(k, 0, len(pid) - 1)], -1), np.where(ok, off[np.clip(k, 0, len(off) - 1)], -1)


def _nearest_teammate(df: pd.DataFrame) -> np.ndarray:
    """Distance to the nearest teammate on the same frame; NaN for a player alone on the floor."""
    key = df["frame"].to_numpy(dtype=np.int64) * 4 + df["team_id"].rank(method="dense").to_numpy(dtype=np.int64) % 4
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    grp_start = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    grp_id = np.cumsum(np.r_[True, k_sorted[1:] != k_sorted[:-1]]) - 1
    within = np.arange(len(key)) - grp_start[grp_id]
    width = int(within.max()) + 1 if len(within) else 1
    n_grp = len(grp_start)
    X = np.full((n_grp, width), np.nan)
    Y = np.full((n_grp, width), np.nan)
    X[grp_id, within] = df["x"].to_numpy(dtype=float)[order]
    Y[grp_id, within] = df["y"].to_numpy(dtype=float)[order]
    D = np.hypot(X[:, :, None] - X[:, None, :], Y[:, :, None] - Y[:, None, :])
    D[:, np.arange(width), np.arange(width)] = np.nan
    with np.errstate(all="ignore"):
        nearest = np.nanmin(np.where(np.isnan(D), np.inf, D), axis=2)
    nearest = np.where(np.isfinite(nearest), nearest, np.nan)
    out = np.empty(len(key))
    out[order] = nearest[grp_id, within]
    return out


def tracking_partials(frames: pd.DataFrame, possessions: pd.DataFrame) -> pd.DataFrame:
    """One game's frames and possessions -> per player partial sums (columns PARTIAL_SUMS plus player_id, season,
    possessions as a set-like string count). Combine games with combine_partials."""
    need = ["game", "period", "frame", "ts_ms", "team_id", "player_id", "x", "y", "ball_x", "ball_y", "ball_z"]
    missing = [c for c in need if c not in frames.columns]
    if missing:
        raise ValueError("%s: frames lack column(s) %s" % (WHO, missing))
    f = frames[need + [c for c in ("date",) if c in frames.columns]].copy()
    f = f.drop_duplicates(["frame", "player_id"]).sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
    if "date" in f.columns and len(f):
        season = season_of_date(str(f["date"].iloc[0]))
    else:
        season = season_of_date(_date_of_game(str(frames["game"].iloc[0])))
    game = str(f["game"].iloc[0]) if len(f) else ""
    poss = possessions[possessions["sportvu_game"].astype(str) == game] if "sportvu_game" in possessions.columns else possessions
    frame_ids = f["frame"].to_numpy(dtype=np.int64)
    poss_idx, off_team = _possession_of_frames(frame_ids, poss)
    team = f["team_id"].to_numpy(dtype=np.int64)
    in_poss = poss_idx >= 0
    is_off = in_poss & (team == off_team)
    x = f["x"].to_numpy(dtype=float)
    y = f["y"].to_numpy(dtype=float)
    bx = f["ball_x"].to_numpy(dtype=float)
    by = f["ball_y"].to_numpy(dtype=float)
    bz = f["ball_z"].to_numpy(dtype=float)
    period = f["period"].to_numpy(dtype=np.int64)

    # attacked basket per (period, offence team): the side where the ball spends that team's offensive frames
    att = pd.DataFrame({"period": period[is_off], "team": team[is_off], "bx": bx[is_off]})
    att_left = att.groupby(["period", "team"])["bx"].mean() < HALF_X
    key = pd.MultiIndex.from_arrays([period, team])
    attack_left = att_left.reindex(key).to_numpy()
    attack_left = np.where(pd.isna(attack_left), True, attack_left).astype(bool)
    xa = np.where(attack_left, x, COURT_X - x)
    ya = np.where(attack_left, y, COURT_Y - y)
    band_x = np.floor(xa / BAND_X).astype(int)
    band_y = (ya >= RIM_Y).astype(int)
    in_half = is_off & (xa >= 0) & (xa < HALF_X)

    # handler: nearest to the ball in the frame, within HANDLER_FT, ball under HANDLER_BALL_Z
    dball = np.hypot(x - bx, y - by)
    dball_ok = np.where(np.isfinite(dball), dball, np.inf)
    frame_min = pd.Series(dball_ok).groupby(frame_ids).transform("min").to_numpy()
    is_handler = (dball_ok == frame_min) & (dball_ok < HANDLER_FT) & (bz < HANDLER_BALL_Z) & np.isfinite(bz)
    # one handler per frame at most: keep the first row at the minimum
    first = pd.Series(is_handler).groupby(frame_ids).cumsum().to_numpy()
    is_handler = is_handler & (first == 1)

    # speed: consecutive frames of the same player, period, 0 < dt <= 2 s
    g = f.sort_values(["player_id", "frame"], kind="stable")
    pid_s = g["player_id"].to_numpy(dtype=np.int64)
    per_s = g["period"].to_numpy(dtype=np.int64)
    ts = g["ts_ms"].to_numpy(dtype=float) / 1000.0
    xs, ys = g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)
    same = (pid_s[1:] == pid_s[:-1]) & (per_s[1:] == per_s[:-1])
    dtm = ts[1:] - ts[:-1]
    ds = np.hypot(xs[1:] - xs[:-1], ys[1:] - ys[:-1])
    with np.errstate(all="ignore"):
        sp = ds / dtm
    valid = same & (dtm > 0) & (dtm <= MAX_STEP_S) & np.isfinite(sp) & (sp <= MAX_SPEED_FTPS)
    steps = pd.DataFrame({"player_id": pid_s[1:][valid], "sp": sp[valid], "ds": ds[valid]})
    step_sums = steps.groupby("player_id").agg(speed_sum=("sp", "sum"), speed_n=("sp", "size"), dist_sum=("ds", "sum"))

    teammate = _nearest_teammate(f)
    rows = pd.DataFrame({
        "player_id": f["player_id"].to_numpy(dtype=np.int64),
        "all_frames": 1,
        "off_frames": is_off.astype(int),
        "handler_frames": (is_off & is_handler).astype(int),
        "teammate_sum": np.where(np.isfinite(teammate), teammate, 0.0),
        "teammate_n": np.isfinite(teammate).astype(int),
        "offball_sum": np.where(is_off & ~is_handler & np.isfinite(dball), dball, 0.0),
        "offball_n": (is_off & ~is_handler & np.isfinite(dball)).astype(int),
    })
    for i, bxn in enumerate(ZONE_X):
        for j, byn in enumerate(ZONE_Y):
            rows["zone_%s_%s_n" % (bxn, byn)] = (in_half & (band_x == i) & (band_y == j)).astype(int)
    sums = rows.groupby("player_id").sum()
    out = sums.join(step_sums, how="left").fillna(0.0)
    n_poss = pd.Series(poss_idx[in_poss]).groupby(f["player_id"].to_numpy(dtype=np.int64)[in_poss]).nunique()
    out["n_possessions"] = n_poss.reindex(out.index).fillna(0).astype(int)
    out = out.reset_index()
    out["season"] = season
    out["sportvu_game"] = game
    return out[["player_id", "season", "sportvu_game", "n_possessions"] + PARTIAL_SUMS]


def _date_of_game(game: str) -> str:
    m, d, y = game.split(".")[:3]
    return "%s-%s-%s" % (y, m, d)


def combine_partials(parts) -> pd.DataFrame:
    """Sum the per-game partial sums over games -> tracking features per (player_id, season) plus n_possessions."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=KEYS + ["n_possessions"] + TRACKING_FEATURES)
    allp = pd.concat(parts, ignore_index=True)
    refuse_heldout_games(sorted(allp["sportvu_game"].astype(str).unique()))
    s = allp.groupby(KEYS)[["n_possessions"] + PARTIAL_SUMS].sum()
    with np.errstate(all="ignore"):
        out = pd.DataFrame(index=s.index)
        out["n_possessions"] = s["n_possessions"].astype(int)
        out["avg_speed_ftps"] = s["speed_sum"] / s["speed_n"].replace(0, np.nan)
        out["dist_per_poss_ft"] = s["dist_sum"] / s["n_possessions"].replace(0, np.nan)
        for z in ZONE_FEATURES:
            out[z] = s[z + "_n"] / s["off_frames"].replace(0, np.nan)
        out["ball_share"] = s["handler_frames"] / s["off_frames"].replace(0, np.nan)
        out["nearest_teammate_ft"] = s["teammate_sum"] / s["teammate_n"].replace(0, np.nan)
        out["offball_ball_dist_ft"] = s["offball_sum"] / s["offball_n"].replace(0, np.nan)
    return out.reset_index()


# ----------------------------------------------------------------------------- shots

def derive_zone_basic(shots: pd.DataFrame) -> pd.Series:
    """zone_basic per attempt: the NBA.com column mapped onto SHOT_ZONES when present, else derived from
    (court_x, court_y, shot_type) in the left-basket frame of section 10a (rim at 5.25, 25)."""
    if "zone_basic" in shots.columns and shots["zone_basic"].notna().any():
        z = shots["zone_basic"].astype("string").str.strip().str.lower().map(NBA_ZONE_MAP)
        if z.notna().all():
            return z.astype(object)
        derived = derive_zone_basic(shots.drop(columns=["zone_basic"]))
        return z.astype(object).where(z.notna(), derived)
    for c in ("court_x", "court_y", "shot_type"):
        if c not in shots.columns:
            raise ValueError("%s: shots table lacks zone_basic and column %r to derive it" % (WHO, c))
    dx = shots["court_x"].astype(float).to_numpy() - RIM_X
    dy = shots["court_y"].astype(float).to_numpy() - RIM_Y
    d = np.hypot(dx, dy)
    three = shots["shot_type"].astype(str).str.upper().str.startswith("3").to_numpy()
    z = np.where(d < 4.0, "restricted_area",
         np.where(~three & (np.abs(dy) <= 8.0) & (dx <= 13.75), "paint_non_ra",
          np.where(~three, "mid_range",
           np.where(dx + RIM_X > HALF_X, "backcourt",
            np.where(dx < 8.75, "corner_3", "above_break_3")))))
    return pd.Series(z, index=shots.index, dtype=object)


def resolve_shooter_ids(shots: pd.DataFrame, rosters: pd.DataFrame | None) -> pd.Series:
    """player_id per attempt: the table's own when filled, else the section 16 roster row whose bref_slug equals the
    attempt's player_slug in the same game. NaN when neither resolves."""
    pid = pd.to_numeric(shots["player_id"], errors="coerce") if "player_id" in shots.columns else pd.Series(np.nan, index=shots.index)
    if rosters is None or "player_slug" not in shots.columns or pid.notna().all():
        return pid.astype(float)
    r = rosters[["sportvu_game", "bref_slug", "player_id"]].dropna().drop_duplicates(["sportvu_game", "bref_slug"])
    r = r.rename(columns={"player_id": "_roster_pid", "bref_slug": "player_slug"})
    j = shots[["sportvu_game", "player_slug"]].merge(r, how="left", on=["sportvu_game", "player_slug"])
    return pid.astype(float).fillna(pd.Series(j["_roster_pid"].to_numpy(dtype=float), index=shots.index))


def shot_features(shots: pd.DataFrame, rosters: pd.DataFrame | None, n_possessions: pd.DataFrame) -> pd.DataFrame:
    """Shot mix per (player_id, season) and attempts per possession (n_possessions: player_id, season, n_possessions)."""
    if shots is None or len(shots) == 0:
        return pd.DataFrame(columns=KEYS + SHOT_FEATURES)
    s = shots.copy()
    s["player_id"] = resolve_shooter_ids(s, rosters)
    n_unres = int(s["player_id"].isna().sum())
    if n_unres:
        log.warning("%s: %d of %d attempts have no resolvable shooter id and are dropped", WHO, n_unres, len(s))
    s = s[s["player_id"].notna()].copy()
    s["player_id"] = s["player_id"].astype(int)
    if "season" in s.columns and s["season"].notna().all():
        s["season"] = s["season"].astype(str)
    else:
        s["season"] = s["game_date"].map(season_of_date) if "game_date" in s.columns else \
            s["sportvu_game"].astype(str).map(lambda g: season_of_date(_date_of_game(g)))
    s["zone"] = derive_zone_basic(s)
    s["three"] = s["shot_type"].astype(str).str.upper().str.startswith("3").astype(int)
    g = s.groupby(KEYS)
    out = pd.DataFrame({"attempts": g.size()})
    for z in SHOT_ZONES:
        out["shot_share_%s" % z] = g["zone"].apply(lambda v, z=z: float((v == z).mean()))
    out["shot_share_3pt"] = g["three"].mean()
    out = out.reset_index()
    out = out.merge(n_possessions[KEYS + ["n_possessions"]], how="left", on=KEYS)
    out["attempts_per_poss"] = out["attempts"] / out["n_possessions"].replace(0, np.nan)
    return out[KEYS + SHOT_FEATURES]


# ----------------------------------------------------------------------------- NBA.com aggregates

def _norm_play_type(v) -> str:
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def aggregate_features(aggregates: pd.DataFrame | None, season: str) -> pd.DataFrame:
    """The section 11 long table filtered to ``season`` (and Regular Season when season_type is present), pivoted
    wide onto AGG_FEATURES per player_id. Only rows of the embedding season are read: a next-season value never enters."""
    if aggregates is None or len(aggregates) == 0:
        return pd.DataFrame(columns=KEYS + AGG_FEATURES)
    a = aggregates[aggregates["season"].astype(str) == season]
    if "season_type" in a.columns and (a["season_type"] == "Regular Season").any():
        a = a[a["season_type"].fillna("Regular Season") == "Regular Season"]
    if "player_id" not in a.columns:
        return pd.DataFrame(columns=KEYS + AGG_FEATURES)
    a = a[pd.to_numeric(a["player_id"], errors="coerce").notna()].copy()
    a["player_id"] = pd.to_numeric(a["player_id"]).astype(int)
    a["metric"] = a["metric"].astype(str).str.lower()
    a["endpoint"] = a["endpoint"].astype(str).str.lower()
    out = pd.DataFrame({"player_id": sorted(a["player_id"].unique())}).set_index("player_id")
    missing = []
    for endpoint, metric, col in AGG_METRICS:
        m = a[(a["endpoint"] == endpoint) & (a["metric"] == metric)]
        if len(m) == 0:
            missing.append(col)
        out[col] = m.groupby("player_id")["value"].mean().reindex(out.index)
    pt = a[a["endpoint"] == "synergyplaytypes"]
    if len(pt):
        metric_pref = [m for m in ("poss_pct", "freq", "frequency", "poss") if (pt["metric"] == m).any()]
        if metric_pref:
            pt = pt[pt["metric"] == metric_pref[0]]
        if "type_grouping" in pt.columns and (pt["type_grouping"].astype(str).str.lower() == "offensive").any():
            pt = pt[pt["type_grouping"].astype(str).str.lower() == "offensive"]
        ptn = pt.assign(pt=pt["play_type"].map(_norm_play_type))
        wide = ptn.groupby(["player_id", "pt"])["value"].mean().unstack("pt")
    else:
        wide = pd.DataFrame(index=out.index)
    for p in PLAY_TYPES:
        col = "pt_%s_freq" % p
        out[col] = wide[p].reindex(out.index).fillna(0.0) if p in wide.columns else 0.0
        if p not in wide.columns:
            missing.append(col)
    if missing:
        log.warning("%s: aggregate metric(s) absent for %s, left NaN or 0: %s", WHO, season, missing)
    out = out.reset_index()
    out["season"] = season
    return out[KEYS + AGG_FEATURES]


# ----------------------------------------------------------------------------- assembly

def player_season_features(tracking: pd.DataFrame, shots: pd.DataFrame | None, rosters: pd.DataFrame | None,
                           aggregates: pd.DataFrame | None, n_min: int = N_MIN) -> pd.DataFrame:
    """player_season_features: KEYS + FEATURES + n_possessions + insufficient, one row per player-season with
    tracking frames. ``tracking`` is combine_partials(...) over the training games."""
    base = tracking.copy()
    if len(base) == 0:
        cols = KEYS + FEATURES + ["n_possessions", "insufficient"]
        return pd.DataFrame(columns=cols)
    sf = shot_features(shots, rosters, base[KEYS + ["n_possessions"]])
    base = base.merge(sf, how="left", on=KEYS)
    aggs = [aggregate_features(aggregates, s) for s in sorted(base["season"].unique())]
    aggs = [a for a in aggs if len(a)]
    if aggs:
        base = base.merge(pd.concat(aggs, ignore_index=True), how="left", on=KEYS)
    for c in FEATURES:
        if c not in base.columns:
            base[c] = np.nan
        base[c] = base[c].astype(float)
    base["insufficient"] = base["n_possessions"].astype(int) < int(n_min)
    out = base[KEYS + FEATURES + ["n_possessions", "insufficient"]].sort_values(KEYS).reset_index(drop=True)
    assert_no_leak(out)
    return out


def available_features(feats: pd.DataFrame) -> list:
    """FEATURES minus the columns that are NaN on every row (the aggregate block on a machine without NBA.com tables)."""
    return [c for c in FEATURES if c in feats.columns and feats[c].notna().any()]


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> dict:
    from analytics.embeddings import train as _train
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--data", help="DATA_ROOT (see analytics/embeddings/train.py)")
    ap.add_argument("--out", help="OUT_ROOT: writes analytics/embeddings/<season>/player_season_features.parquet")
    ap.add_argument("--n-min", type=int, default=None, help="possessions under which a player-season is insufficient (300)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.fixtures and not a.data:
        ap.error("one of --fixtures or --data is required")
    inputs = _train.fixture_inputs(a.seed) if a.fixtures else _train.load_inputs(a.data)
    n_min = a.n_min if a.n_min is not None else (_train.FIXTURE_N_MIN if a.fixtures else N_MIN)
    feats = _train.build_features(inputs, n_min)
    summary = {"rows": int(len(feats)), "insufficient": int(feats["insufficient"].sum()), "n_min": n_min,
               "features_available": available_features(feats), "fixture": bool(a.fixtures)}
    if a.out:
        for season, part in feats.groupby("season"):
            d = Path(a.out) / "analytics" / "embeddings" / str(season)
            d.mkdir(parents=True, exist_ok=True)
            part.to_parquet(d / "player_season_features.parquet", index=False)
        summary["out"] = str(Path(a.out) / "analytics" / "embeddings")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
