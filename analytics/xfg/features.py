"""analytics/xfg/features.py: time-forward features of a field-goal attempt at its release frame.

    python -m analytics.xfg.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]

Unit: one field-goal attempt of the shots table (docs/DATA_CONTRACT.md sections 10 and 10a). The attempt is
joined to shot_frames (section 17) by (sportvu_game, event), then to the frames (section 4) at its release
frame and the frames before it. Every feature is measured at or before the release; nothing from the flight
or the outcome enters (schema.assert_time_forward runs on the returned frame, assert_no_leak on every design
matrix).

Features (FEATURES, in order):
    distance_ft, angle_rad, is_three, is_corner      the location-only baseline (analytics.features_xfg)
    closest_defender_dist_ft                         distance from the shooter to the nearest opponent at release
    closest_defender_closing_speed_ftps              (that defender's distance 0.4 s before release minus at release) / 0.4,
                                                     positive when closing; the defender is the one closest at release
    n_defenders_within_6ft                           opponents within 6 ft of the shooter at release
    shooter_speed_ftps                               the shooter's displacement over the 0.4 s before release / 0.4
    touch_time_s                                     seconds the ball stayed within 2.5 ft (horizontal) of the shooter
                                                     before the release, walking back frame by frame, capped at 10 s
    dribbles_proxy                                   local minima of ball z under 2 ft inside that touch window
    shot_clock_s                                     the shot clock at the release frame (NaN when the source has none)
    has_contest                                      1 when the contest features could be measured, else 0
    period, clock_s                                  the attempt's period and logged game clock (the shots table)

Join rule: an attempt whose shot_frames row has release_frame -1 or confidence_flag unmatched, or whose shooter is
not in the release frame, keeps the location features and gets NaN contest features with has_contest 0. An
attempt with no shot_frames row at all is treated the same way (has_contest 0). The shooter is shooter_id of
shot_frames, else player_id of the shots table, else shooter_at_release_id.

Frame rate: 25 Hz (SportVU); 0.4 s is 10 frames, 10 s is 250 frames. Speeds use the frame count, not the clock,
so a clock gap inside the window (rare, section 4) does not blow a speed up; a window that crosses a period or
holds fewer than 2 frames gives NaN speeds.

The shooter prior (make rate by zone, out-of-fold target encoding) is NOT built here: it needs labels and is
added fold by fold in analytics.xfg.train so the row's own outcome never enters. ``zone`` (rim, paint, mid,
corner3, above3) is the key it uses.

Event matching when the shots table lacks an ``event`` column (basketball-reference rows carry a null event_id):
the attempt is matched to the shot_frames row of the same sportvu_game and period whose clock_s is nearest
within MATCH_CLOCK_S = 1.0 s, each shot_frames row used at most once (greedy by clock distance); unmatched
attempts get has_contest 0. The count is logged.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from analytics import schema
from analytics import features_xfg as fx

log = logging.getLogger("analytics.xfg.features")

FPS = 25
SPEED_FRAMES = 10            # 0.4 s
TOUCH_CAP_FRAMES = 250       # 10 s
TOUCH_RADIUS_FT = 2.5
DRIBBLE_Z_FT = 2.0
CONTEST_RADIUS_FT = 6.0
MATCH_CLOCK_S = 1.0
ZONES = ("rim", "paint", "mid", "corner3", "above3")

LOCATION_FEATURES: list[str] = list(fx.feature_columns())
CONTEST_FEATURES: list[str] = ["closest_defender_dist_ft", "closest_defender_closing_speed_ftps", "n_defenders_within_6ft",
                               "shooter_speed_ftps", "touch_time_s", "dribbles_proxy", "shot_clock_s"]
CONTEXT_FEATURES: list[str] = ["has_contest", "period", "clock_s"]
FEATURES: list[str] = LOCATION_FEATURES + CONTEST_FEATURES + CONTEXT_FEATURES
PRIOR_FEATURE = "shooter_prior"
KEYS: list[str] = ["sportvu_game", "event", "player_id", "heldout"]

# Anything that encodes the outcome, or is measured at or after it, never enters a design matrix.
FORBIDDEN: frozenset = frozenset(schema.ALL_EXCLUDED | {"made", "points", "score", "outcome_frame", "apex_frame",
                                                        "identity_match", "confidence", "confidence_flag"})


def feature_columns(with_prior: bool = False) -> list[str]:
    """The model columns: FEATURES, plus the shooter prior when asked."""
    return FEATURES + ([PRIOR_FEATURE] if with_prior else [])


def assert_no_leak(X) -> None:
    """ValueError when a design matrix carries a forbidden column (outcome, post-outcome or label-derived)."""
    cols = set(getattr(X, "columns", X))
    bad = sorted(cols & FORBIDDEN)
    if bad:
        raise ValueError("analytics.xfg: forbidden column(s) in the design matrix: %s" % bad)


# ----------------------------------------------------------------------------- zones

def zone_of(distance_ft, is_three, is_corner) -> np.ndarray:
    d = np.asarray(distance_ft, dtype=float)
    three = np.asarray(is_three, dtype=bool)
    corner = np.asarray(is_corner, dtype=bool)
    out = np.where(d < 4.0, "rim", np.where(d < 14.0, "paint", "mid")).astype(object)
    out[three & corner] = "corner3"
    out[three & ~corner] = "above3"
    return out


# ----------------------------------------------------------------------------- event matching

def match_events(shots: pd.DataFrame, shot_frames: pd.DataFrame) -> pd.Series:
    """The shot_frames event of each shots row: the row's own event when it has one, else the nearest-clock
    shot_frames row of the same game and period within MATCH_CLOCK_S, each used once. -1 when none."""
    out = pd.Series(-1, index=shots.index, dtype="int64")
    if "event" in shots.columns and shots["event"].notna().any():
        out = pd.to_numeric(shots["event"], errors="coerce").fillna(-1).astype("int64")
        if out.ge(0).all():
            return out
    if "event_id" in shots.columns and shots["event_id"].notna().all() and len(shots):
        return pd.to_numeric(shots["event_id"], errors="coerce").fillna(-1).astype("int64")
    sf = shot_frames[["sportvu_game", "event", "period", "clock_s"]].copy()
    sf["sportvu_game"] = sf["sportvu_game"].astype(str)
    sf_groups = {k: g for k, g in sf.groupby(["sportvu_game", "period"])}
    used: set = set()
    cand = []
    for idx, row in shots.iterrows():
        if out.loc[idx] >= 0:
            continue
        g = sf_groups.get((str(row["sportvu_game"]), int(row["period"])))
        if g is None:
            continue
        diff = (g["clock_s"].astype(float) - float(row["clock_s"])).abs()
        for sf_idx in diff.sort_values().index:
            if diff.loc[sf_idx] > MATCH_CLOCK_S:
                break
            cand.append((float(diff.loc[sf_idx]), idx, int(g.loc[sf_idx, "event"]), str(row["sportvu_game"])))
            break
    for _d, idx, ev, game in sorted(cand):
        if (game, ev) in used:
            continue
        used.add((game, ev))
        out.loc[idx] = ev
    log.info("match_events: %d of %d attempts matched to a shot_frames row by clock", int(out.ge(0).sum()), len(shots))
    return out


# ----------------------------------------------------------------------------- the frames index of one game

class GameFrames:
    """Per-frame arrays of one game: ball, clocks and player positions keyed by frame index (section 4)."""

    def __init__(self, frames: pd.DataFrame):
        f = frames.sort_values(["frame"], kind="stable")
        first = f.groupby("frame", sort=True).first()
        self.frames = first.index.to_numpy().astype(np.int64)
        self.pos = {int(fr): i for i, fr in enumerate(self.frames)}
        self.period = first["period"].to_numpy().astype(np.int64)
        self.ball = np.column_stack([first["ball_x"].to_numpy(float), first["ball_y"].to_numpy(float), first["ball_z"].to_numpy(float)])
        self.shot_clock = first["shot_clock_s"].to_numpy(float) if "shot_clock_s" in first.columns else np.full(len(first), np.nan)
        self.game_clock = first["game_clock_s"].to_numpy(float)
        px = f.pivot_table(index="frame", columns="player_id", values="x", aggfunc="first")
        py = f.pivot_table(index="frame", columns="player_id", values="y", aggfunc="first")
        px = px.reindex(self.frames)
        py = py.reindex(self.frames)
        self.players = px.columns.to_numpy().astype(np.int64)
        self.col = {int(p): j for j, p in enumerate(self.players)}
        self.x = px.to_numpy(float)
        self.y = py.to_numpy(float)
        team = f.drop_duplicates("player_id").set_index("player_id")["team_id"]
        self.team = np.array([int(team.loc[p]) for p in self.players])

    def contest(self, release_frame: int, shooter: int) -> dict:
        """The contest features at ``release_frame`` for ``shooter``; NaNs with has_contest 0 when not measurable."""
        nan = {c: np.nan for c in CONTEST_FEATURES}
        nan["has_contest"] = 0
        i = self.pos.get(int(release_frame))
        j = self.col.get(int(shooter))
        if i is None or j is None or not np.isfinite(self.x[i, j]):
            return nan
        sx, sy = self.x[i, j], self.y[i, j]
        opp = (self.team != self.team[j]) & np.isfinite(self.x[i])
        if not opp.any():
            return nan
        d = np.hypot(self.x[i] - sx, self.y[i] - sy)
        d_opp = np.where(opp, d, np.inf)
        k = int(np.argmin(d_opp))
        out = {"closest_defender_dist_ft": float(d_opp[k]), "n_defenders_within_6ft": int((d_opp <= CONTEST_RADIUS_FT).sum()),
               "shot_clock_s": float(self.shot_clock[i]) if np.isfinite(self.shot_clock[i]) else np.nan, "has_contest": 1}
        # 0.4 s window before the release: same period, consecutive frame indices
        i0 = i - SPEED_FRAMES
        if i0 >= 0 and self.period[i0] == self.period[i] and self.frames[i] - self.frames[i0] == SPEED_FRAMES \
                and np.isfinite(self.x[i0, j]) and np.isfinite(self.x[i0, k]):
            dt = SPEED_FRAMES / FPS
            d0 = float(np.hypot(self.x[i0, k] - self.x[i0, j], self.y[i0, k] - self.y[i0, j]))
            out["closest_defender_closing_speed_ftps"] = (d0 - float(d_opp[k])) / dt
            out["shooter_speed_ftps"] = float(np.hypot(sx - self.x[i0, j], sy - self.y[i0, j])) / dt
        else:
            out["closest_defender_closing_speed_ftps"] = np.nan
            out["shooter_speed_ftps"] = np.nan
        # touch: walk back while the ball stays within 2.5 ft (horizontal) of the shooter, same period, consecutive frames
        n = 0
        t = i
        while t >= 0 and n < TOUCH_CAP_FRAMES:
            if self.period[t] != self.period[i] or not np.isfinite(self.x[t, j]) or not np.isfinite(self.ball[t, 0]):
                break
            if t < i and self.frames[t + 1] - self.frames[t] != 1:
                break
            if np.hypot(self.ball[t, 0] - self.x[t, j], self.ball[t, 1] - self.y[t, j]) > TOUCH_RADIUS_FT:
                break
            n += 1
            t -= 1
        out["touch_time_s"] = n / FPS
        z = self.ball[i - n + 1:i + 1, 2] if n else np.array([])
        out["dribbles_proxy"] = int(count_dribbles(z))
        return out


def count_dribbles(z) -> int:
    """Local minima of ball z under DRIBBLE_Z_FT, strictly inside the window (z[t] <= z[t-1], z[t] < z[t+1])."""
    z = np.asarray(z, dtype=float)
    if len(z) < 3:
        return 0
    inner = z[1:-1]
    return int(((inner < DRIBBLE_Z_FT) & (inner <= z[:-2]) & (inner < z[2:])).sum())


# ----------------------------------------------------------------------------- the feature frame

def _frames_of(frames, game: str):
    if frames is None:
        return None
    if callable(frames):
        return frames(game)
    if isinstance(frames, Mapping):
        return frames.get(game)
    if isinstance(frames, pd.DataFrame):
        col = "game" if "game" in frames.columns else "sportvu_game"
        return frames[frames[col].astype(str) == game]
    raise TypeError("frames must be a DataFrame, a mapping game -> DataFrame or a callable game -> DataFrame")


def build(shots: pd.DataFrame, shot_frames: pd.DataFrame, frames=None) -> pd.DataFrame:
    """The feature frame: KEYS, release_frame, zone and FEATURES, one row per shots row, same order. No label.

    ``frames`` is the section 4 table (all games), a mapping sportvu_game -> frames, or a callable that loads one
    game's frames (so a season of 600 games is read one game at a time). None gives has_contest 0 everywhere."""
    shots = shots.reset_index(drop=True)
    loc = fx.features(shots)
    out = pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype(str).to_numpy()})
    out["event"] = match_events(shots, shot_frames).to_numpy()
    out["player_id"] = pd.to_numeric(shots["player_id"], errors="coerce").astype("Int64").to_numpy() if "player_id" in shots.columns \
        else pd.array([pd.NA] * len(shots), dtype="Int64")
    out["heldout"] = loc["heldout"].to_numpy()
    for c in LOCATION_FEATURES:
        out[c] = loc[c].to_numpy()
    out["zone"] = zone_of(out["distance_ft"], out["is_three"], out["is_corner"])
    out["period"] = pd.to_numeric(shots["period"]).astype(int).to_numpy()
    out["clock_s"] = shots["clock_s"].astype(float).to_numpy()
    # shot_frames join by (sportvu_game, event): release frame and shooter, nothing post-outcome
    sf = shot_frames[["sportvu_game", "event", "release_frame", "shooter_id", "shooter_at_release_id", "confidence_flag"]].copy()
    sf["sportvu_game"] = sf["sportvu_game"].astype(str)
    sf["event"] = sf["event"].astype("int64")
    sf = sf.drop_duplicates(["sportvu_game", "event"])
    j = out[["sportvu_game", "event"]].merge(sf, on=["sportvu_game", "event"], how="left")
    rel = pd.to_numeric(j["release_frame"], errors="coerce").fillna(schema.NOT_FOUND).astype(int)
    unmatched = j["confidence_flag"].astype("string").fillna("unmatched").eq("unmatched").to_numpy()
    out["release_frame"] = np.where(unmatched, schema.NOT_FOUND, rel.to_numpy())
    shooter = pd.to_numeric(j["shooter_id"], errors="coerce")
    shooter = shooter.fillna(pd.to_numeric(out["player_id"], errors="coerce"))
    shooter = shooter.fillna(pd.to_numeric(j["shooter_at_release_id"], errors="coerce"))
    for c in CONTEST_FEATURES:
        out[c] = np.nan
    out["has_contest"] = 0
    todo = (out["release_frame"] >= 0) & shooter.notna().to_numpy()
    n_matched = 0
    for game, idx in out.index[todo].to_series().groupby(out.loc[todo, "sportvu_game"].to_numpy()):
        gf = _frames_of(frames, str(game))
        if gf is None or len(gf) == 0:
            continue
        G = GameFrames(gf)
        for i in idx:
            c = G.contest(int(out.at[i, "release_frame"]), int(shooter.loc[i]))
            for k, v in c.items():
                out.at[i, k] = v
            n_matched += int(c["has_contest"])
    out["has_contest"] = out["has_contest"].astype(int)
    out["n_defenders_within_6ft"] = out["n_defenders_within_6ft"].astype(float)
    out["dribbles_proxy"] = out["dribbles_proxy"].astype(float)
    log.info("features: %d attempts, %d with contest features, %d without", len(out), n_matched, len(out) - n_matched)
    schema.assert_time_forward(out, "shot_frames")
    assert_no_leak(out[FEATURES])
    return out


def targets(shots: pd.DataFrame) -> pd.DataFrame:
    """(sportvu_game, event, made): the label, kept apart from the feature frame."""
    ev = pd.to_numeric(shots["event"], errors="coerce") if "event" in shots.columns else pd.Series(np.arange(len(shots)))
    return pd.DataFrame({"sportvu_game": shots["sportvu_game"].astype(str).to_numpy(), "event": ev.fillna(-1).astype("int64").to_numpy(),
                         "made": shots["made"].astype(bool).to_numpy()})


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> dict:
    from analytics.xfg import validate as V
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", action="store_true")
    src.add_argument("--data", metavar="DATA_ROOT")
    ap.add_argument("--out", metavar="OUT_ROOT", default=None)
    ap.add_argument("--season", default="2015-16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixture-games", type=int, default=8)
    ap.add_argument("--fixture-possessions", type=int, default=40)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)
    tables = V.fixture_tables(a.seed, n_games=a.fixture_games, n_possessions=a.fixture_possessions) if a.fixtures \
        else V.load_data_root(a.data, a.season)
    tables, _summary = V.clean(tables)
    feat = build(tables["shots"], tables["shot_frames"], tables["frames"])
    info = {"rows": int(len(feat)), "with_contest": int(feat["has_contest"].sum()), "games": int(feat["sportvu_game"].nunique()),
            "features": FEATURES, "fixture": bool(a.fixtures)}
    if a.out:
        d = Path(a.out) / "analytics" / "xfg" / a.season
        d.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(d / "xfg_features.parquet", index=False)
        info["path"] = str(d / "xfg_features.parquet")
    print(json.dumps(info, indent=1) + ("\nfixture output, not a result" if a.fixtures else ""))
    return info


if __name__ == "__main__":
    main()
