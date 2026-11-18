"""A small synthetic data root in the real layout, for the tests of export/site_data.py.

Every table has the columns and dtypes of the real one (docs/DATA_CONTRACT.md and the lane writers). Every player,
position and number is invented: nothing here is a result. The root carries the marker file the export looks for
(export.site_common.FIXTURE_MARKER), so the site data command refuses it outside a test.

Layout written by write_data_root(root):
  sportvu_contract/heldout/<game>.parquet, <game>_pbp.parquet        two held-out games
  sportvu_contract/<game>.parquet                                     frame and player_id at the rim frame of the
                                                                      dropped misses, for three training games
  analytics/{possessions,rosters,shot_frames}/heldout/2015-16/<game>.parquet
  analytics/epv/heldout/2015-16/<game>.parquet
  analytics/ghosting/heldout/2015-16/ghost_frames_<game>.parquet, ghost_possessions_<game>.parquet
  analytics/xfg/2015-16/xfg_predictions.parquet                      held-out rows (fold -1), out-of-fold rows,
                                                                     in-sample decoys (fold -2)
  collect/bref_shots/heldout/shotchart/2015-16.parquet
  analytics/{rosters,shot_frames,rebounds,rebound_frames}/2015-16/<game>.parquet    twelve training games
  analytics/gravity/gravity_players.parquet, analytics/load/2015-16/player_games.parquet, collect/positions.parquet
write_artifacts(folder, data_root) adds the rebuilt outputs another job would write: absence/, embeddings/ (each
with a passed check), demo/possessions.json, and the copy of the check records that the repository tracks
(tracked_checks(folder); in the repository it is reports/export/frozen_checks.json).

The held-out games are built from a plan (PLAN_A, PLAN_B): one Spec per possession. The specs name the cases the
tests look for: a clock gap, a single frame, a release that was not located, an unresolved shooter, missing rows,
a possession whose length is not a whole number of grid steps.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = "2015-16"
FPS, STRIDE = 25, 5
HELD_A = "12.16.2015.PHX.at.GSW"          # the scorecard game id: split heldout_game
HELD_B = "01.05.2016.FXA.at.CLE"          # every game at CLE is held out: split heldout_arena
TRAIN_GAMES = [f"11.{day:02d}.2015.FXA.at.FXB" for day in range(1, 13)]
FILLER_GAME = "10.30.2015.FXC.at.FXD"     # out-of-fold attempts with no shooter: they only fill calibration bins
MARKER = "fixture_root.json"

TEAM_IDS = {"PHX": 1610612756, "GSW": 1610612744, "FXA": 1610612901, "FXB": 1610612902, "CLE": 1610612739}
ROSTERS = {            # team -> player ids; the first five are on the floor
    "PHX": [9101, 9102, 9103, 9104, 9105, 9106], "GSW": [9201, 9202, 9203, 9204, 9205, 9206],
    "FXA": [9301, 9302, 9303, 9304, 9305, 9306], "CLE": [9401, 9402, 9403, 9404, 9405, 9406],
    "FXB": [9501, 9502, 9503, 9504, 9505, 9506],
}
SPOT_Y = np.array([8.0, 16.5, 25.0, 33.5, 42.0])
CHART_OFFSET = (1.0, -0.5)                # how far a chart marker sits from the true release position, feet
FAR_GHOST_X = 130.0                       # a ghost point far outside the court (dropped by the display rule)


def player_name(player_id: int) -> str:
    return f"Fixture Player {player_id}"


def slug_of(player_id: int) -> str:
    return f"fixp{player_id}"


@dataclass
class Shot:
    shooter: int = 0                      # index into the five attackers
    made: bool = True
    three: bool = False
    at: float = 1.0                       # where in the possession the logged outcome lies (1.0 = its last frame)
    located: bool = True                  # shot_frames holds a release frame
    resolved: bool = True                 # the play-by-play name matched a tracked player
    predicted: bool = True                # the shot has a shot chart row and an xFG prediction
    event_frame: bool = True              # the play-by-play event has a frame
    release_before_start: bool = False    # the release frame lies before the possession's first frame
    ball_out_of_bounds: bool = False      # at the release frame the ball is outside the court (cleaned away by xFG)
    xfg: float = 0.4


@dataclass
class Spec:
    off: str                              # "away" or "home"
    period: int
    seconds: float
    end_kind: str = "shot_made"
    points: int = 2
    shots: list = field(default_factory=list)
    gap: bool = False
    ghosts: str = "full"                  # full, none, cut (stop half way), far (one point far outside the court)
    epv_states: int = None                # None: one a second
    drop_player: bool = False             # one defender has no rows in the second half of the possession
    drop_frame: bool = False              # one sampled frame has no rows at all
    null_ball: bool = False
    null_shot_clock: bool = False
    skip_event: bool = False              # leave one event number unused before this possession (see EXTRA_SHOT)


def _shot_possession(off, period, seconds, made=True, shooter=0, xfg=0.4, **kw):
    shot_keys = {k: kw.pop(k) for k in list(kw) if k in Shot.__dataclass_fields__}
    three = shot_keys.get("three", False)
    return Spec(off=off, period=period, seconds=seconds, end_kind="shot_made" if made else "shot_missed",
                points=(3 if three else 2) if made else 0,
                shots=[Shot(shooter=shooter, made=made, xfg=xfg, **shot_keys)], **kw)


# Game A: the shooters stand away from the middle of the court, so the shot chart's orientation can be verified.
PLAN_A = [
    _shot_possession("away", 1, 10, made=True, shooter=0, xfg=0.30),                       # 0 eligible
    _shot_possession("home", 1, 9, made=False, shooter=1, xfg=0.34),                       # 1 eligible
    _shot_possession("away", 1, 8, made=True, shooter=4, xfg=0.52, three=True),            # 2 eligible, demo
    _shot_possession("home", 1, 12, made=False, shooter=3, xfg=0.47),                      # 3 eligible
    Spec("away", 1, 7, end_kind="turnover", points=0, skip_event=True),                    # 4 no shot
    _shot_possession("home", 1, 10, made=True, shooter=0, xfg=0.58, located=False),        # 5 release not located
    _shot_possession("away", 1, 9, made=False, shooter=1, xfg=0.41, resolved=False),       # 6 unresolved shooter
    _shot_possession("home", 1, 6, made=True, shooter=4, xfg=0.36),                        # 7 ghosts under 80 percent
    _shot_possession("away", 1, 10, made=True, shooter=3, xfg=0.63, gap=True, ghosts="none"),   # 8 clock gap
    Spec("home", 1, 11, end_kind="shot_made", points=2,                                    # 9 miss, rebound, make
         shots=[Shot(shooter=1, made=False, at=0.5, xfg=0.31), Shot(shooter=0, made=True, xfg=0.71)]),
    Spec("away", 1, 0, end_kind="turnover", points=0, ghosts="none"),                      # 10 single frame
    _shot_possession("home", 1, 9, made=True, shooter=3, xfg=0.44, ball_out_of_bounds=True),    # 11 eligible
    _shot_possession("home", 2, 10, made=False, shooter=4, xfg=0.22),                      # 12 eligible
    _shot_possession("away", 2, 9, made=True, shooter=1, xfg=0.35, ghosts="far"),          # 13 eligible
    _shot_possession("home", 2, 8, made=False, shooter=0, predicted=False),                # 14 no prediction
    _shot_possession("away", 2, 10, made=True, shooter=3, xfg=0.55, release_before_start=True),  # 15
    _shot_possession("home", 2, 9, made=False, shooter=1, xfg=0.39, located=False, event_frame=False),  # 16
    _shot_possession("away", 2, 10, made=True, shooter=4, xfg=0.61, drop_player=True),     # 17 nine tracked
    Spec("home", 2, 9, end_kind="period_end", points=0, null_ball=True, drop_frame=True, null_shot_clock=True),  # 18
    _shot_possession("away", 2, 10, made=True, shooter=0, xfg=0.33, ghosts="cut", epv_states=3),  # 19
]
# An attempt with a prediction whose event lies in no possession (the event number skipped before possession 4).
EXTRA_SHOT_A = dict(period=1, shooter=3, made=False, xfg=0.27, clock_s=5.0)

# Game B: every shooter stands in the middle of the court, so the orientation cannot be verified there.
PLAN_B = [
    _shot_possession("away", 1, 10, made=True, shooter=2, xfg=0.28),                       # 0 eligible (player 9303)
    _shot_possession("home", 1, 9, made=False, shooter=2, xfg=0.37),                       # 1 eligible
    _shot_possession("away", 1, 8, made=True, shooter=2, xfg=0.49),                        # 2 eligible
    _shot_possession("home", 1, 11.1, made=True, shooter=2, xfg=0.66, located=False),      # 3 no release; 277 frames
                                                                                           # long, not a multiple of 5
    Spec("away", 1, 9, end_kind="free_throws", points=2),                                  # 4
    _shot_possession("home", 1, 10, made=False, shooter=2, xfg=0.24),                      # 5 eligible
    _shot_possession("away", 2, 9, made=False, shooter=2, xfg=0.45),                       # 6 eligible
    _shot_possession("home", 2, 10, made=True, shooter=2, xfg=0.53),                       # 7 eligible
    Spec("away", 2, 8, end_kind="turnover", points=0, ghosts="none"),                      # 8
    _shot_possession("home", 2, 12, made=False, shooter=2, xfg=0.32),                      # 9 eligible
]
PLANS = {HELD_A: ("PHX", "GSW", PLAN_A, EXTRA_SHOT_A), HELD_B: ("FXA", "CLE", PLAN_B, None)}


def eligible(spec: Spec) -> bool:
    """PLAN.md D48 over the plan: a made or missed shot with a located release and an xFG, at least 5 EPV states,
    ghosts on at least 80 percent of the frames, no gap, 10 players tracked, 6 to 24 s."""
    if spec.end_kind not in ("shot_made", "shot_missed") or not spec.shots:
        return False
    last = spec.shots[-1]
    located = last.located and last.resolved and last.predicted and not last.release_before_start
    frames = int(spec.seconds * FPS) // STRIDE + 1
    ghost_share = (frames - 7) / frames if spec.ghosts in ("full", "far") else 0.0
    states = spec.epv_states if spec.epv_states is not None else int(spec.seconds) + 1
    return bool(located and states >= 5 and ghost_share >= 0.8 and not spec.gap and not spec.drop_player
                and not spec.drop_frame and 6 <= frames_to_seconds(frames) <= 24)


def frames_to_seconds(frames: int) -> float:
    return round((frames - 1) * STRIDE / FPS, 2)


def _paths(n: int, attack_right: bool, phase: float):
    """Positions of five attackers, five defenders and the ball over n frames of one possession."""
    s = np.linspace(0.0, 1.0, n)[:, None]
    rim = 88.75 if attack_right else 5.25
    out = -1.0 if attack_right else 1.0                       # from the rim toward the middle of the court
    far = np.array([8.0, 12.0, 16.0, 20.0, 24.0])[None, :]
    dist = 40.0 + (far - 40.0) * s
    off = np.stack([rim + out * dist, SPOT_Y[None, :] + 2.0 * np.sin(2 * np.pi * s + phase + np.arange(5))], axis=2)
    de = np.stack([rim + out * (dist - 3.0), off[:, :, 1] + 1.0], axis=2)
    ball = np.stack([off[:, 0, 0] + 1.0, off[:, 0, 1], 4.0 + np.sin(6 * np.pi * s[:, 0])], axis=1)
    return off, de, ball


def _heldout_game(game: str, seed: int) -> dict:
    """The tables of one held-out game, built from its plan."""
    away, home, plan, extra = PLANS[game]
    rng = np.random.default_rng(seed)
    frames, possessions, events, attempts, chart, predictions, states, ghosts, ghost_poss = ([] for _ in range(9))
    frame, event, period, period_frame0, clock_jump = 0, 0, 0, 0, 0.0
    previous = None                                            # (offense team, attackers, defenders, ball) at the last frame

    def clock_at(f: int) -> float:
        return 720.0 - (f - period_frame0) / FPS - clock_jump

    for number, spec in enumerate(plan):
        if spec.period != period:                              # a new period: the clock starts again
            period, period_frame0, clock_jump = spec.period, frame, 0.0
        if spec.skip_event:
            event += 1
        offense, defense = (away, home) if spec.off == "away" else (home, away)
        attack_right = (spec.off == "away") == (spec.period % 2 == 1)
        n = int(spec.seconds * FPS) + 1
        start, end = frame, frame + n - 1
        start_clock = clock_at(start)
        off, de, ball = _paths(n, attack_right, phase=0.7 * number)
        if previous is not None:                               # the first frame is the previous possession's last
            team, attackers, defenders, ball[0] = previous
            off[0], de[0] = (attackers, defenders) if team == offense else (defenders, attackers)
        gap_at = start + n // 2 if spec.gap else None
        for i in range(n):
            f = start + i
            if i == 0 and number > 0:
                continue                                       # that frame's rows were written by the previous possession
            if gap_at is not None and f == gap_at:
                clock_jump += 3.0
            shot_clock = np.nan if spec.null_shot_clock else max(24.0 - i / FPS, 0.0)
            b = (np.nan, np.nan, np.nan) if spec.null_ball and i > n // 2 else tuple(ball[i])
            for side, team, xy in (("off", offense, off[i]), ("def", defense, de[i])):
                for j, player in enumerate(ROSTERS[team][:5]):
                    frames.append((f, spec.period, clock_at(f), shot_clock, b[0], b[1], b[2], TEAM_IDS[team], team,
                                   team == home, player, xy[j, 0], xy[j, 1], number, side, j))
        end_clock = clock_at(end)
        first_event = event
        for shot in spec.shots:
            index = int(round(shot.at * (n - 1)))
            event_frame = start + index
            release = max(index - 20, 0)
            shooter_id = ROSTERS[offense][shot.shooter]
            release_frame = (start - 10) if shot.release_before_start else (start + release)
            tx, ty = off[release, shot.shooter]
            cx, cy = (94.0 - tx, ty) if attack_right else (tx, 50.0 - ty)       # the chart's frame: left rim, mirrored
            cx, cy = cx + CHART_OFFSET[0], cy + CHART_OFFSET[1]
            kind = "shot_made" if shot.made else "shot_missed"
            desc = "F. Player %s %s-pt jump shot from 15 ft" % ("makes" if shot.made else "misses", 3 if shot.three else 2)
            events.append((event, spec.period, clock_at(event_frame), "away" if spec.off == "away" else "home", desc,
                           (3 if shot.three else 2) if shot.made else 0, kind, event_frame if shot.event_frame else -1))
            attempts.append(dict(event=event, period=spec.period, clock_s=clock_at(event_frame), made=shot.made,
                                 shooter_id=shooter_id if shot.resolved else None,
                                 release_frame=release_frame if shot.located else -1,
                                 shooter_at_release_id=shooter_id if shot.located else None,
                                 identity_match=True if (shot.located and shot.resolved) else None))
            if shot.ball_out_of_bounds:
                frames = [row[:4] + (95.5,) + row[5:] if row[0] == release_frame else row for row in frames]
            if shot.predicted:
                chart.append(dict(period=spec.period, clock_s=clock_at(event_frame), made=shot.made, court_x=cx, court_y=cy,
                                  shot_type="3PT" if shot.three else "2PT", player=player_name(shooter_id),
                                  player_slug=slug_of(shooter_id), team=offense))
                predictions.append(dict(event=event, player_id=shooter_id if (shot.resolved or shot.located) else None,
                                        p_selected=shot.xfg, made=shot.made))
            event += 1
        if not spec.shots:
            kind = {"turnover": "turnover", "free_throws": "ft_made", "period_end": "other"}.get(spec.end_kind, "other")
            events.append((event, spec.period, end_clock, "away" if spec.off == "away" else "home", "fixture event",
                           spec.points, kind, end))
            event += 1
        possessions.append(dict(possession=number, period=spec.period, start_frame=start, end_frame=end,
                                start_clock_s=start_clock, end_clock_s=end_clock, start_event=first_event,
                                end_event=event - 1, offense_team_id=TEAM_IDS[offense], end_kind=spec.end_kind,
                                points=spec.points, gap=spec.gap, n_frames=n))
        # EPV: about one state a second, on frames that need not sit on the 5-frame grid
        n_states = spec.epv_states if spec.epv_states is not None else int(spec.seconds) + 1
        for k in range(n_states):
            f = min(start + k * FPS + (k % 3 if k else 0), end)
            states.append((number, f, float(k), round(1.0 + 0.2 * np.sin(k + number) - (0.9 if k == 2 else 0.0)
                                                      + 0.01 * rng.random(), 6)))
        # ghosts on the grid: 0.2 s rows from step 3 (never read by the export), 1.0 s rows from step 7
        if spec.ghosts != "none":
            grid = np.arange(start, end + 1, STRIDE)
            last_step = len(grid) // 2 if spec.ghosts == "cut" else len(grid) - 1
            for horizon, first_step, shift in ((0.2, 3, 0.1), (1.0, 7, 1.5)):
                for j, player in enumerate(ROSTERS[defense][:5]):
                    for step in range(first_step, last_step + 1):
                        x, y = de[grid[step] - start, j]
                        far = spec.ghosts == "far" and horizon == 1.0 and j == 2 and step == 9
                        ghosts.append((number, int(grid[step]), player, ROSTERS[offense][j], x, y,
                                       FAR_GHOST_X if far else x + shift, y - shift, 0.5, horizon))
            if last_step >= 7:
                ghost_poss.append((number, round(0.5 + 0.1 * number, 3), spec.points))
        previous = (offense, off[-1].copy(), de[-1].copy(), ball[-1].copy())
        frame = end
    if extra:
        shooter_id = ROSTERS[away][extra["shooter"]]
        skipped = next(p["start_event"] - 1 for p, s in zip(possessions, plan) if s.skip_event)
        events.append((skipped, extra["period"], extra["clock_s"], "away", "F. Player misses 2-pt jump shot from 9 ft", 0,
                       "shot_missed", -1))
        attempts.append(dict(event=skipped, period=extra["period"], clock_s=extra["clock_s"], made=extra["made"],
                             shooter_id=shooter_id, release_frame=-1, shooter_at_release_id=None, identity_match=None))
        chart.append(dict(period=extra["period"], clock_s=extra["clock_s"], made=extra["made"], court_x=20.0,
                          court_y=30.0, shot_type="2PT", player=player_name(shooter_id), player_slug=slug_of(shooter_id),
                          team=away))
        predictions.append(dict(event=skipped, player_id=shooter_id, p_selected=extra["xfg"], made=extra["made"]))

    fr = pd.DataFrame(frames, columns=["frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z",
                                       "team_id", "team", "home", "player_id", "x", "y", "_possession", "_side", "_slot"])
    for number, spec in enumerate(plan):                        # rows a tracker would have lost
        span = possessions[number]
        grid = np.arange(span["start_frame"], span["end_frame"] + 1, STRIDE)
        if spec.drop_player:
            lost = (fr["player_id"] == ROSTERS[home if spec.off == "away" else away][2]) & (fr["frame"] > grid[len(grid) // 2]) \
                & (fr["frame"] <= span["end_frame"])
            fr = fr[~lost]
        if spec.drop_frame:
            fr = fr[fr["frame"] != grid[4]]
    fr = fr.reset_index(drop=True)
    meta = dict(mm=game[0:2], dd=game[3:5], yyyy=game[6:10])
    frames_table = pd.DataFrame({
        "game": game, "gameid": "0021500000", "date": "%(yyyy)s-%(mm)s-%(dd)s" % meta,
        "period": fr["period"].astype("int8"), "frame": fr["frame"].astype("int32"),
        "ts_ms": (1450000000000 + fr["frame"] * 40).astype("int64"),
        "game_clock_s": fr["game_clock_s"].astype("float32"), "shot_clock_s": fr["shot_clock_s"].astype("float32"),
        "n_players": fr.groupby("frame")["player_id"].transform("size").astype("int8"),
        "ball_x": fr["ball_x"].astype("float32"), "ball_y": fr["ball_y"].astype("float32"),
        "ball_z": fr["ball_z"].astype("float32"), "team_id": fr["team_id"].astype("int32"), "team": fr["team"],
        "home": fr["home"].astype(bool), "player_id": fr["player_id"].astype("int32"),
        "player": [player_name(p) for p in fr["player_id"]], "jersey": [str(p % 100) for p in fr["player_id"]],
        "x": fr["x"].round(2).astype("float32"), "y": fr["y"].round(2).astype("float32")})
    ev = pd.DataFrame(events, columns=["event", "period", "clock_s", "team", "desc", "points", "kind", "frame"])
    ev = ev.sort_values("event").reset_index(drop=True)
    pbp = pd.DataFrame({
        "game": game, "event": ev["event"].astype("int32"), "period": ev["period"].astype("int8"),
        "clock_s": ev["clock_s"].astype("float32"), "team": ev["team"], "desc": ev["desc"],
        "points": ev["points"].astype("int8"), "score": "0-0", "kind": ev["kind"], "frame": ev["frame"].astype("int32"),
        "frame_clock_diff_s": np.float32(0.0), "source": "fixture", "endpoint": "fixture", "season": SEASON,
        "fetched_at": "2000-01-01T00:00:00Z"})
    at = pd.DataFrame(attempts).sort_values("event").reset_index(drop=True)
    shot_frames = pd.DataFrame({
        "sportvu_game": game, "event": at["event"].astype("int32"), "period": at["period"].astype("int8"),
        "clock_s": at["clock_s"].astype("float32"), "made": at["made"].astype(bool),
        "pbp_shooter_name": "F. Player", "shooter_id": at["shooter_id"].astype("Int32"),
        "release_frame": at["release_frame"].astype("int32"), "apex_frame": np.int32(-1), "outcome_frame": np.int32(-1),
        "shooter_at_release_id": at["shooter_at_release_id"].astype("Int32"),
        "identity_match": at["identity_match"].astype("boolean"), "confidence": np.float32(0.7),
        "confidence_flag": np.where(at["shooter_id"].isna(), "unmatched", "high")})
    poss = pd.DataFrame(possessions)
    poss = pd.DataFrame({
        "sportvu_game": game, "possession": poss["possession"].astype("int32"), "period": poss["period"].astype("int8"),
        "start_frame": poss["start_frame"].astype("int32"), "end_frame": poss["end_frame"].astype("int32"),
        "start_clock_s": poss["start_clock_s"].astype("float32"), "end_clock_s": poss["end_clock_s"].astype("float32"),
        "start_event": poss["start_event"].astype("int32"), "end_event": poss["end_event"].astype("int32"),
        "offense_team_id": poss["offense_team_id"].astype("int32"), "end_kind": poss["end_kind"],
        "points": poss["points"].astype("int16"), "gap": poss["gap"].astype(bool), "n_frames": poss["n_frames"].astype("int32")})
    roster_ids = ROSTERS[home] + ROSTERS[away]
    rosters = pd.DataFrame({
        "sportvu_game": game, "player_id": np.array(roster_ids, dtype="int32"),
        "player": [player_name(p) for p in roster_ids],
        "team_id": np.array([TEAM_IDS[home]] * 6 + [TEAM_IDS[away]] * 6, dtype="int32"), "team": [home] * 6 + [away] * 6,
        "home": [True] * 6 + [False] * 6, "jersey": [str(p % 100) for p in roster_ids],
        "bref_slug": [slug_of(p) for p in roster_ids], "bref_name": [player_name(p) for p in roster_ids],
        "pbp_name": "F. Player", "match_method": "exact"})
    ch = pd.DataFrame(chart)
    chart_table = pd.DataFrame({
        "game": game, "player": ch["player"], "player_slug": ch["player_slug"], "team": ch["team"],
        "period": ch["period"].astype("int64"), "clock_s": ch["clock_s"].astype("float64"), "made": ch["made"].astype(bool),
        "shot_type": ch["shot_type"], "shot_distance_ft": np.int64(15),
        "loc_x": (ch["court_y"] - 25.0) * 10.0, "loc_y": (ch["court_x"] - 5.25) * 10.0,
        "court_x": ch["court_x"].astype(float), "court_y": ch["court_y"].astype(float), "basket": "left",
        "sportvu_game": game, "heldout": True, "season": SEASON})
    pr = pd.DataFrame(predictions)
    xfg = pd.DataFrame({
        "sportvu_game": game, "event": pr["event"].astype("int64"), "player_id": pr["player_id"].astype("Int64"),
        "p_location": np.float32(np.nan), "p_contest": np.float32(np.nan), "p_selected": pr["p_selected"].astype("float32"),
        "made": pr["made"].astype(bool), "fold": np.int16(-1), "heldout": True})
    st = pd.DataFrame(states, columns=["possession", "frame", "elapsed_in_possession_s", "epv_points"])
    st = st.drop_duplicates(["possession", "frame"]).reset_index(drop=True)
    epv = pd.DataFrame({
        "sportvu_game": game, "possession": st["possession"].astype("int32"), "frame": st["frame"].astype("int32"),
        "elapsed_in_possession_s": st["elapsed_in_possession_s"].astype("float32"), "epv_points": st["epv_points"],
        "p_score": 0.45, "points_to_end": np.int16(0), "fold": np.int32(-1), "heldout": True})
    gh = pd.DataFrame(ghosts, columns=["possession", "frame", "def_player_id", "off_player_id", "actual_x", "actual_y",
                                       "ghost_x", "ghost_y", "deviation_ft", "horizon_s"])
    ghost_frames = pd.DataFrame({
        "sportvu_game": game, "possession": gh["possession"].astype("int32"), "frame": gh["frame"].astype("int32"),
        "def_player_id": gh["def_player_id"].astype("int32"), "off_player_id": gh["off_player_id"].astype("int32"),
        **{c: gh[c].astype("float32") for c in ("actual_x", "actual_y", "ghost_x", "ghost_y", "deviation_ft", "horizon_s")},
        "fold": np.int32(-1), "heldout": True})
    gp = pd.DataFrame(ghost_poss, columns=["possession", "mean_deviation_ft", "points"])
    ghost_possessions = pd.DataFrame({
        "sportvu_game": game, "possession": gp["possession"].astype("int32"),
        "mean_deviation_ft": gp["mean_deviation_ft"].astype("float32"), "points": gp["points"].astype("int16")})
    return dict(frames=frames_table, pbp=pbp, shot_frames=shot_frames, possessions=poss, rosters=rosters,
                chart=chart_table, xfg=xfg, epv=epv, ghost_frames=ghost_frames, ghost_possessions=ghost_possessions)


# ------------------------------------------------------------------------------------------------ training side

# player -> (games he shoots in, attempts a game, share made, shooter resolved)
SHOOTERS = {
    9301: (range(0, 8), 30, 0.50, True),      # 240 attempts: shown
    9302: (range(0, 10), 9, 0.40, True),      # 90 attempts: under the minimum
    9303: (range(0, 2), 60, 0.45, True),      # 120 attempts in two games
    9304: (range(2, 3), 150, 0.60, True),     # 150 attempts in one game: a point estimate
    9306: (range(0, 10), 20, 0.50, False),    # the name match failed: none of these may count for anybody
    9501: (range(0, 5), 22, 0.35, True),      # 110 attempts
}
CONTEST_VALUES = (0.25, 0.35, 0.45, 0.55, 0.65)     # predictions of the contest candidate, one calibration bin each
IN_SAMPLE_DECOYS = 5                                # rows with fold -2 for player 9301: never read
GRAVITY = {            # player -> (possessions, games, gravity_ft, lo, hi)
    9301: (900, 8, 1.234, 1.0, 1.5), 9302: (400, 10, 0.2, -0.1, 0.5), 9303: (120, 2, 0.4, 0.1, 0.8),
    9304: (350, 1, -0.3, np.nan, np.nan), 9306: (50, 4, 0.9, 0.2, 1.4), 9501: (500, 5, -0.6, -0.9, -0.2),
}
REBOUND_CHANCES = {9301: (range(0, 8), 40, 4, 0.12), 9302: (range(0, 10), 20, 3, 0.2),   # games, chances, rebounds, p
                   9501: (range(0, 12), 30, 2, 0.05),
                   9303: (range(0, 2), 150, 0, 0.01)}     # no rebound: the lane dropped every miss he rebounded
# Misses the rebound lane dropped because the play-by-play rebounder did not resolve (rebound_frames rows with a name
# and no pbp_rebounder_id): (game index, name, tracked rebounder, how many). The name "F. 9303" fits player 9303 only; a
# name that fits nobody falls to the player the tracking credits most often with it, or to nobody.
LOST_REBOUNDS = [(0, "F. 9303", None, 40), (1, "F. 9303", 9303, 40), (3, "Z. Nobody", 9301, 1),
                 (4, "Y. Nobody", None, 1)]
# Who the tracking frames list on the floor at the rim frame (100) of those misses, by game index. Game 4 has no frames
# table, so its one miss counts for every player of its roster (the conservative bound). 9501 was on the floor for the
# 80 misses of 9303: 80 chances he did not get, so his shown rate would read too high (withheld, the upward side).
FILLERS = [9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608]
ON_FLOOR = {0: [9303, 9501] + FILLERS, 1: [9303, 9501] + FILLERS, 3: [9301, 9502] + FILLERS}
RIM_FRAME = 100
LOST_SHOOTER_NAME = "F. 9306"             # the name of the attempts of 9306, whose shooter the lane did not resolve
POSITIONS = {9301: "PG", 9302: "SF-SG", 9303: "C-PF", 9306: "PF", 9501: "SG"}     # 9304 and 9305 are not listed
PLAYS_IN = {9301: range(0, 12), 9302: range(0, 10), 9303: range(0, 2), 9304: range(2, 3), 9305: range(0, 12),
            9306: range(0, 10), 9501: range(0, 12), 9502: range(0, 12)}
ABSENT_AFTER = {9301: (2, 5, 9), 9302: (1,), 9303: (), 9501: (0, 4, 7, 10)}        # game indexes he was absent after


TRADED = 9302                             # moves from FXA to FXB after six games


def team_of(player_id: int, game_index: int) -> str:
    if player_id == TRADED and game_index >= 6:
        return "FXB"
    return "FXA" if player_id < 9400 else "FXB"


def _training_tables(report_bin_counts) -> dict:
    """The out-of-fold lane tables of the training games. `report_bin_counts` are the attempts per calibration bin
    that reports/xfg.md prints for the contest candidate: filler attempts with no shooter top every bin up to it,
    because the export checks the stored predictions against that table."""
    rng = np.random.default_rng(7)
    rows, located = [], []
    for player, (games, per_game, share, resolved) in SHOOTERS.items():
        for g in games:
            for k in range(per_game):
                event = len([r for r in located if r[0] == TRAIN_GAMES[g]])
                credited = player if resolved else (9301 if k % 2 else None)     # a fallback id, or none
                rows.append((TRAIN_GAMES[g], event, credited, CONTEST_VALUES[k % 5], 0.3 + 0.4 * rng.random(),
                             bool(rng.random() < share), g % 5, False))
                located.append((TRAIN_GAMES[g], event, player if resolved else None,
                                "F. Player" if resolved else LOST_SHOOTER_NAME))
    for k in range(IN_SAMPLE_DECOYS):
        event = len([r for r in located if r[0] == TRAIN_GAMES[0]])
        rows.append((TRAIN_GAMES[0], event, 9301, 0.45, 0.5, True, -2, False))
        located.append((TRAIN_GAMES[0], event, 9301, "F. Player"))
    designed = pd.DataFrame(rows, columns=["sportvu_game", "event", "player_id", "p_contest", "p_selected", "made", "fold",
                                           "heldout"])
    n_bins = len(report_bin_counts)
    have = np.bincount(np.floor(designed.loc[designed["fold"] >= 0, "p_contest"] * n_bins).astype(int), minlength=n_bins)
    need = np.asarray(report_bin_counts) - have
    assert (need >= 0).all(), "the designed attempts overfill a calibration bin of the report"
    filler_p = np.concatenate([(b + 0.2 + 0.6 * rng.random(int(k))) / n_bins for b, k in enumerate(need)])
    filler = pd.DataFrame({"sportvu_game": FILLER_GAME, "event": -1, "player_id": None, "p_contest": filler_p,
                           "p_selected": 0.45, "made": rng.random(len(filler_p)) < 0.45,
                           "fold": rng.integers(0, 5, len(filler_p)), "heldout": False})
    xfg = pd.concat([designed, filler], ignore_index=True)
    xfg = pd.DataFrame({
        "sportvu_game": xfg["sportvu_game"], "event": xfg["event"].astype("int64"),
        "player_id": xfg["player_id"].astype("Int64"), "p_location": np.float32(0.4),
        "p_contest": xfg["p_contest"].astype("float32"), "p_selected": xfg["p_selected"].astype("float32"),
        "made": xfg["made"].astype(bool), "fold": xfg["fold"].astype("int16"), "heldout": xfg["heldout"].astype(bool)})
    located = pd.DataFrame(located, columns=["sportvu_game", "event", "shooter_id", "pbp_shooter_name"])
    shot_frames = {game: pd.DataFrame({
        "sportvu_game": game, "event": part["event"].astype("int32"), "period": np.int8(1), "clock_s": np.float32(300.0),
        "made": True, "pbp_shooter_name": part["pbp_shooter_name"].astype("string"),
        "shooter_id": part["shooter_id"].astype("Int32"),
        "release_frame": np.int32(-1), "apex_frame": np.int32(-1), "outcome_frame": np.int32(-1),
        "shooter_at_release_id": part["shooter_id"].astype("Int32"), "identity_match": pd.array([None] * len(part), "boolean"),
        "confidence": np.float32(0.5), "confidence_flag": "low"}) for game, part in located.groupby("sportvu_game")}
    for game in TRAIN_GAMES:
        shot_frames.setdefault(game, shot_frames[TRAIN_GAMES[0]].iloc[0:0].assign(sportvu_game=game))

    rebounds = {}
    for g, game in enumerate(TRAIN_GAMES):
        part = []
        for player, (games, chances, got, p) in REBOUND_CHANCES.items():
            if g in games:
                for k in range(chances):
                    part.append((game, k, player, p, k < got))
                    part.append((game, k, -1, 0.1, False))      # the team rebound row of the same miss
        t = pd.DataFrame(part, columns=["sportvu_game", "event", "candidate_player_id", "p_selected", "is_rebounder"])
        rebounds[game] = pd.DataFrame({
            "sportvu_game": t["sportvu_game"], "event": t["event"].astype("int32"),
            "candidate_player_id": t["candidate_player_id"].astype("int32"), "p_prior": np.float32(0.1),
            "p_logit": np.float32(0.1), "p_selected": t["p_selected"].astype("float32"),
            "is_rebounder": t["is_rebounder"].astype(bool), "fold": np.int16(g % 5), "heldout": False})

    rebound_frames = {}
    for g, game in enumerate(TRAIN_GAMES):
        rows = [(k, 1000 + k, 100, 1000 + k + 20, 9501, "F. Player", 9501) for k in range(3)]   # resolved misses
        rows.append((3, None, 100, -1, None, None, None))                    # no rebound event: not a lost miss
        rows.append((4, 1004, -1, -1, None, "F. 9303", None))                # no rim frame: dropped, not lost
        rows.append((5, 1005, 100, -1, None, None, None))                    # a team rebound: kept, not lost
        for index, name, tracked, n in LOST_REBOUNDS:
            if index == g:
                rows += [(6 + k, 1006 + k, 100, -1 if tracked is None else 130, tracked, name, None) for k in range(n)]
        t = pd.DataFrame(rows, columns=["event", "rebound_event", "rim_frame", "rebound_frame", "rebounder_id",
                                        "pbp_rebounder_name", "pbp_rebounder_id"])
        rebound_frames[game] = pd.DataFrame({
            "sportvu_game": game, "event": t["event"].astype("int32"),
            "rebound_event": t["rebound_event"].astype("Int32"),
            "period": np.int8(1), "clock_s": np.float32(300.0), "rim_frame": t["rim_frame"].astype("int32"),
            "rebound_frame": t["rebound_frame"].astype("int32"), "rebounder_id": t["rebounder_id"].astype("Int32"),
            "pbp_rebounder_name": t["pbp_rebounder_name"].astype("string"),
            "pbp_rebounder_id": t["pbp_rebounder_id"].astype("Int32"),
            "identity_match": pd.array([None] * len(t), "boolean"), "confidence": np.float32(0.5),
            "confidence_flag": "low"})

    floor_frames = {TRAIN_GAMES[g]: pd.DataFrame({"frame": np.int32(RIM_FRAME), "player_id": np.array(ids, "int32")})
                    for g, ids in ON_FLOOR.items()}

    rosters, games_rows = {}, []
    for g, game in enumerate(TRAIN_GAMES):
        ids = [p for p, games in PLAYS_IN.items() if g in games]
        table = pd.DataFrame({
            "sportvu_game": game, "player_id": np.array(ids, dtype="int32"), "player": [player_name(p) for p in ids],
            "team_id": np.array([TEAM_IDS[team_of(p, g)] for p in ids], dtype="int32"), "team": [team_of(p, g) for p in ids],
            "home": [team_of(p, g) == "FXB" for p in ids], "jersey": [str(p % 100) for p in ids],
            "bref_slug": [None if p == 9305 else slug_of(p) for p in ids], "bref_name": [player_name(p) for p in ids],
            "pbp_name": "F. Player", "match_method": "exact"})
        if g == 0:                                              # the nameless placeholder some frames carry
            table = pd.concat([table, pd.DataFrame({"sportvu_game": [game], "player_id": np.array([0], dtype="int32"),
                                                    "player": [None], "team_id": np.array([TEAM_IDS["FXA"]], dtype="int32"),
                                                    "team": ["FXA"], "home": [False], "jersey": [None], "bref_slug": [None],
                                                    "bref_name": [None], "pbp_name": [None], "match_method": ["none"]})],
                              ignore_index=True)
        rosters[game] = table
        for p in ids:
            last = g == max(PLAYS_IN[p])
            games_rows.append((game, "2015-11-%02d" % (g + 1), p, team_of(p, g),
                               None if last else g in ABSENT_AFTER.get(p, ())))
    games_rows.append((TRAIN_GAMES[0], "2015-11-01", 0, "FXA", None))
    pg = pd.DataFrame(games_rows, columns=["sportvu_game", "game_date", "player_id", "team", "absent_next"])
    player_games = pd.DataFrame({
        "sportvu_game": pg["sportvu_game"], "game_date": pg["game_date"], "player_id": pg["player_id"].astype("int32"),
        "team": pg["team"], "minutes": 20.0, "distance_ft": 9000.0, "avg_speed_ftps": 6.0, "decline_ftps": -0.1,
        "slope_ftps_per_stint": 0.0, "days_rest": 2.0, "back_to_back": False, "games_last_7": 2.0, "minutes_last_3": 60.0,
        "distance_last_3_ft": 27000.0, "prev_decline_ftps": -0.1, "prev_avg_speed_ftps": 6.0,
        "absent_next": pg["absent_next"].astype("boolean"), "heldout": False})
    gravity = pd.DataFrame({
        "player_id": np.array(list(GRAVITY), dtype="int32"),
        "n_possessions": np.array([v[0] for v in GRAVITY.values()], dtype="int32"),
        "n_frames": np.array([v[0] * 12 for v in GRAVITY.values()], dtype="int32"),
        "n_games": np.array([v[1] for v in GRAVITY.values()], dtype="int32"),
        "gravity_ft": [v[2] for v in GRAVITY.values()], "lo": [v[3] for v in GRAVITY.values()],
        "hi": [v[4] for v in GRAVITY.values()], "insufficient": [v[0] < 300 for v in GRAVITY.values()]})
    listed = [(slug_of(p), player_name(p), SEASON, pos) for p, pos in POSITIONS.items()]
    listed += [(slug_of(9301), player_name(9301), "2016-17", "C")]             # another season: never read
    positions = pd.DataFrame(listed, columns=["player_slug", "player", "season", "position"]).astype("string")
    positions = positions.assign(team="FXA", age=pd.array([25] * len(listed), "Int16"),
                                 games=pd.array([40] * len(listed), "Int16"), mp_per_g=np.float32(20.0),
                                 source="fixture", endpoint="fixture", fetched_at="2000-01-01T00:00:00Z")
    return dict(xfg=xfg, shot_frames=shot_frames, rebounds=rebounds, rebound_frames=rebound_frames, rosters=rosters,
                player_games=player_games, floor_frames=floor_frames,
                gravity=gravity, positions=positions)


# ------------------------------------------------------------------------------------------------ writing


def write_data_root(root, report_bin_counts) -> dict:
    """Write the synthetic data root and return its tables: {"heldout": {game: tables}, "training": tables}."""
    root = Path(root)
    held = {game: _heldout_game(game, seed) for seed, game in enumerate(PLANS)}
    training = _training_tables(report_bin_counts)
    analytics, tracked = root / "analytics", root / "sportvu_contract" / "heldout"

    def write(table: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)

    for game, t in held.items():
        write(t["frames"], tracked / f"{game}.parquet")
        write(t["pbp"], tracked / f"{game}_pbp.parquet")
        for name in ("possessions", "rosters", "shot_frames"):
            write(t[name], analytics / name / "heldout" / SEASON / f"{game}.parquet")
        write(t["epv"], analytics / "epv" / "heldout" / SEASON / f"{game}.parquet")
        write(t["ghost_frames"], analytics / "ghosting" / "heldout" / SEASON / f"ghost_frames_{game}.parquet")
        write(t["ghost_possessions"], analytics / "ghosting" / "heldout" / SEASON / f"ghost_possessions_{game}.parquet")
    write(pd.concat([t["chart"] for t in held.values()], ignore_index=True),
          root / "collect" / "bref_shots" / "heldout" / "shotchart" / f"{SEASON}.parquet")
    write(pd.concat([training["xfg"]] + [t["xfg"] for t in held.values()], ignore_index=True),
          analytics / "xfg" / SEASON / "xfg_predictions.parquet")
    for name in ("rosters", "shot_frames", "rebounds", "rebound_frames"):
        for game, table in training[name].items():
            write(table, analytics / name / SEASON / f"{game}.parquet")
    for game, table in training["floor_frames"].items():
        write(table, root / "sportvu_contract" / f"{game}.parquet")
    write(training["gravity"], analytics / "gravity" / "gravity_players.parquet")
    write(training["player_games"], analytics / "load" / SEASON / "player_games.parquet")
    write(training["positions"], root / "collect" / "positions.parquet")
    (root / MARKER).write_text(json.dumps({"what": "synthetic data root written by tests/_fx_site_data.py"}) + "\n")
    return {"heldout": held, "training": training}


NEIGHBOURS = {9301: [9501, 9302, 9303, 9304, 9306], 9501: [9301, 9306, 9302, 9304, 9303]}
EMBEDDING_POSSESSIONS = {9301: (1200, False), 9302: (250, True), 9501: (800, False)}
DEMO = [(HELD_A, 2)]
TRACKED_CHECKS = "frozen_checks.json"


def tracked_checks(folder) -> Path:
    """Where write_artifacts puts the tracked copy of the check records of an artifacts folder."""
    return Path(folder) / TRACKED_CHECKS


def absence_probability(player_id: int, game_index: int) -> float:
    return round(0.05 + 0.01 * (player_id % 10) + 0.005 * game_index, 4)


def write_artifacts(folder, data_root, tables: dict, absence: bool = True, embeddings: bool = True,
                    demo: bool = True) -> Path:
    """The rebuilt outputs a separate job writes (PLAN.md D45): each folder ends with a check record that passed,
    and the check records are published the way the repository tracks them (PLAN.md D49)."""
    from tools import frozen_common

    folder = Path(folder)
    before = os.environ.get("DATA_ROOT")
    os.environ["DATA_ROOT"] = str(data_root)                    # the output guard of the rebuild helpers needs it
    try:
        if absence:
            known = tables["training"]["player_games"]
            known = known[(known["player_id"] > 0) & known["absent_next"].notna()]
            oof = pd.DataFrame({
                "sportvu_game": known["sportvu_game"].to_numpy(), "player_id": known["player_id"].to_numpy(),
                "absent_next": known["absent_next"].astype(bool).to_numpy(),
                "p_base": 0.1, "p_logistic": [absence_probability(p, TRAIN_GAMES.index(g))
                                              for p, g in zip(known["player_id"], known["sportvu_game"])]})
            (folder / "absence").mkdir(parents=True, exist_ok=True)
            oof.to_parquet(folder / "absence" / "oof.parquet", index=False)
            frozen_common.write_check(folder / "absence", dict(
                name="absence", compared="out-of-fold log loss and Brier with the lane report (fixture)", n=2,
                max_abs_diff=1e-5, tolerance=5e-5), files=["oof.parquet"])
        if embeddings:
            near = pd.DataFrame([(p, SEASON, "pca", rank + 1, other, SEASON, 0.4 + 0.3 * rank)
                                 for p, others in NEIGHBOURS.items() for rank, other in enumerate(others)],
                                columns=["player_id", "season", "method", "neighbour_rank", "neighbour_player_id",
                                         "neighbour_season", "distance"])
            feats = pd.DataFrame([(p, SEASON, n, low) for p, (n, low) in EMBEDDING_POSSESSIONS.items()],
                                 columns=["player_id", "season", "n_possessions", "insufficient"])
            (folder / "embeddings").mkdir(parents=True, exist_ok=True)
            near.to_parquet(folder / "embeddings" / "embedding_neighbours.parquet", index=False)
            feats.to_parquet(folder / "embeddings" / "player_season_features.parquet", index=False)
            frozen_common.write_check(folder / "embeddings", dict(
                name="embeddings", compared="position accuracy with the lane report (fixture)", n=2, max_abs_diff=1e-4,
                tolerance=5e-4), files=["embedding_neighbours.parquet", "player_season_features.parquet"])
        if demo:
            (folder / "demo").mkdir(parents=True, exist_ok=True)
            (folder / "demo" / "possessions.json").write_text(json.dumps(
                {"possessions": [{"game_id": game, "possession": number} for game, number in DEMO]}))
        folder.mkdir(parents=True, exist_ok=True)
        frozen_common.publish_checks(folder, out=tracked_checks(folder))
    finally:
        if before is None:
            os.environ.pop("DATA_ROOT", None)
        else:
            os.environ["DATA_ROOT"] = before
    return folder
