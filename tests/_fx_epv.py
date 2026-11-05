"""tests/_fx_epv.py: fixtures for the EPV lane (analytics/epv).

Prefers the shared generator ``analytics.fixtures`` (lane I1) and falls back to a minimal local generator with
the same signatures and column names, so the EPV tests pass with the fallback alone. ``season_tables`` gives the
season in one shape whichever generator made it (the shared synth_season returns a list of game dicts, the local
one a dict of concatenated tables); ``fallback=True`` or the environment variable EPV_FIXTURE_FALLBACK=1 forces the
local generator. Everything here is synthetic and deterministic given the seed; nothing is read from disk and
nothing is written into the repo.

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None)
        -> dict(frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz, sportvu_game, heldout)
    synth_season(seed, n_games=8, heldout_every=0)
        -> the same keys with the per-game tables concatenated, plus games: [(sportvu_game, heldout), ...]
    write_fixture_root(root, seed, n_games, heldout_every)
        -> writes the DATA_ROOT layout the EPV CLIs read (sportvu_contract/, analytics/<table>/2015-16/,
           collect/bref_shots/shotchart/2015-16.parquet, held-out games under heldout/) and returns the game list

Fallback generator (the assumptions a test may rely on):
  - frames at 5 Hz (one frame every 0.2 s), ten players per frame, the offence in its front court attacking the
    left basket (5.25, 25) when it is the home team and the right basket (88.75, 25) when it is the away team;
  - possession durations 3 to 22 s, the game clock continuous inside a period (no gaps), the last possession of
    a period cut at 0 s with end_kind period_end;
  - the handler is one offensive player, drifting from 20 to 35 ft out towards an end distance; the points of the
    possession come from a logistic in the end distance, the handler's defender gap at the end and the handler's
    latent shooting skill (per game), and the turnover rate is a per-team latent, so a model over the states has
    something to find and the shooter and lineup priors carry signal;
  - the shots table is 10a-shaped with player_id null (as basketball-reference gives it) and player_slug matching
    rosters.bref_slug, so the shooter is found through the roster join;
  - states_1hz is built by analytics.epv.states.build_states from the frames and possessions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import os

try:  # the shared generator of lane I1, when it exists
    from analytics.fixtures import synth_game as _shared_synth_game, synth_season as _shared_synth_season  # type: ignore
    try:
        from analytics.fixtures import write_fixture_root as _shared_write_fixture_root  # type: ignore
    except ImportError:  # pragma: no cover
        _shared_write_fixture_root = None
    SHARED = True
except ImportError:
    _shared_synth_game = _shared_synth_season = _shared_write_fixture_root = None
    SHARED = False

HOME_ID, AWAY_ID = 1610612744, 1610612751
HOME_ABBR, AWAY_ABBR = "GSW", "BKN"
HOME_IDS = [101, 102, 103, 104, 105]
AWAY_IDS = [201, 202, 203, 204, 205]
HELDOUT_IDS = ["12.16.2015.PHX.at.GSW", "12.23.2015.NYK.at.CLE", "01.18.2016.GSW.at.CLE"]
DT = 0.2
PERIOD_S = 720.0
LEFT_BASKET, RIGHT_BASKET = (5.25, 25.0), (88.75, 25.0)
SEASON = "2015-16"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _game_id(seed: int, heldout: bool) -> str:
    if heldout:
        return HELDOUT_IDS[seed % len(HELDOUT_IDS)] if seed < len(HELDOUT_IDS) else HELDOUT_IDS[0] + ".h%d" % seed
    return "01.%02d.2016.%s.at.%s" % (seed % 28 + 1, AWAY_ABBR, HOME_ABBR)


def _fallback_synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game: str | None = None) -> dict:
    rng = np.random.default_rng(1000 + seed)
    game = sportvu_game or _game_id(seed, heldout)
    skill = {pid: float(rng.normal(0.0, 0.6)) for pid in HOME_IDS + AWAY_IDS}
    lineup_tov = {HOME_ID: float(rng.uniform(0.0, 0.6)), AWAY_ID: float(rng.uniform(0.0, 0.6))}
    poss_rows, frame_chunks, event_rows, shot_rows, sf_rows, rb_rows = [], [], [], [], [], []
    frame_counter, event_counter, poss_idx = 0, 0, 0
    period, clock = 1, PERIOD_S
    offense_home = bool(rng.integers(0, 2))
    event_rows.append(dict(event=event_counter, period=period, clock_s=PERIOD_S, team=None, desc="Start of 1st quarter",
                           points=0, score=None, kind="period_start", frame=frame_counter))
    event_counter += 1
    while poss_idx < n_possessions:
        if clock <= 1.0:
            event_rows.append(dict(event=event_counter, period=period, clock_s=0.0, team=None, desc="End of quarter",
                                   points=0, score=None, kind="period_end", frame=max(frame_counter - 1, 0)))
            event_counter += 1
            period += 1
            clock = PERIOD_S
            event_rows.append(dict(event=event_counter, period=period, clock_s=PERIOD_S, team=None, desc="Start of quarter",
                                   points=0, score=None, kind="period_start", frame=frame_counter))
            event_counter += 1
        dur = float(rng.uniform(3.0, 22.0))
        cut = dur >= clock
        if cut:
            dur = float(clock)
        n = int(round(dur / DT)) + 1
        t = np.arange(n) * DT
        clocks = (clock - t).astype(np.float32)
        frac = t / max(dur, DT)
        basket = LEFT_BASKET if offense_home else RIGHT_BASKET
        sx = 1.0 if offense_home else -1.0
        off_ids = HOME_IDS if offense_home else AWAY_IDS
        def_ids = AWAY_IDS if offense_home else HOME_IDS
        off_team, def_team = (HOME_ID, AWAY_ID) if offense_home else (AWAY_ID, HOME_ID)
        handler_k = int(rng.integers(0, 5))
        handler = off_ids[handler_k]
        d_start, d_end = float(rng.uniform(20, 35)), float(rng.uniform(2, 28))
        g_start, g_end = float(rng.uniform(3, 8)), float(rng.uniform(0.5, 8))
        th0 = float(rng.uniform(-1.2, 1.2))
        pos_off = np.zeros((n, 5, 2))
        pos_def = np.zeros((n, 5, 2))
        for k in range(5):
            if k == handler_k:
                d = d_start + (d_end - d_start) * frac
                th = th0 + rng.normal(0, 0.02, n).cumsum()
                gap = g_start + (g_end - g_start) * frac
            else:
                d = rng.uniform(8, 26) + rng.normal(0, 0.3, n).cumsum()
                th = rng.uniform(-1.3, 1.3) + rng.normal(0, 0.02, n).cumsum()
                gap = rng.uniform(2, 6) + rng.normal(0, 0.1, n).cumsum()
            d = np.clip(d, 1.0, 45.0)
            x = basket[0] + sx * d * np.cos(th)
            y = basket[1] + d * np.sin(th)
            pos_off[:, k, 0], pos_off[:, k, 1] = np.clip(x, 1, 93), np.clip(y, 1, 49)
            u = np.stack([basket[0] - pos_off[:, k, 0], basket[1] - pos_off[:, k, 1]], axis=1)
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
            dpos = pos_off[:, k] + np.clip(gap, 0.3, 12)[:, None] * u
            pos_def[:, k, 0], pos_def[:, k, 1] = np.clip(dpos[:, 0], 1, 93), np.clip(dpos[:, 1], 1, 49)
        ball = pos_off[:, handler_k] + rng.normal(0, 0.4, (n, 2))
        ball_z = 4.0 + 1.5 * np.sin(2 * np.pi * t) + rng.normal(0, 0.2, n)
        shot_clock = np.clip(24.0 - t, 0, 24).astype(np.float32)
        if rng.uniform() < 0.1:
            shot_clock = np.full(n, np.nan, dtype=np.float32)
        # outcome
        logit = 2.2 - 0.16 * d_end + 0.35 * g_end + skill[handler] - 0.03 * dur + rng.normal(0, 0.2)
        if cut:
            end_kind, points, shot = "period_end", 0, False
        elif rng.uniform() < 0.05 + 0.25 * lineup_tov[off_team]:
            end_kind, points, shot = "turnover", 0, False
        elif rng.uniform() < 0.08:
            end_kind, points, shot = "free_throws", int(rng.choice([0, 1, 2], p=[0.1, 0.35, 0.55])), False
        else:
            made = bool(rng.uniform() < _sigmoid(logit))
            end_kind = "shot_made" if made else "shot_missed"
            points = (3 if d_end >= 22 else 2) if made else 0
            shot = True
        frames_ids = np.arange(frame_counter, frame_counter + n, dtype=np.int32)
        ids = np.array(off_ids + def_ids, dtype=np.int32)
        pos = np.concatenate([pos_off, pos_def], axis=1).reshape(n * 10, 2)
        chunk = pd.DataFrame({
            "game": game, "gameid": "0021500%03d" % (seed % 1000), "date": "2016-01-%02d" % (seed % 28 + 1),
            "period": np.int8(period), "frame": np.repeat(frames_ids, 10),
            "ts_ms": np.repeat((frames_ids.astype(np.int64) * 200), 10),
            "game_clock_s": np.repeat(clocks, 10), "shot_clock_s": np.repeat(shot_clock, 10),
            "n_players": np.int8(10),
            "ball_x": np.repeat(ball[:, 0].astype(np.float32), 10), "ball_y": np.repeat(ball[:, 1].astype(np.float32), 10),
            "ball_z": np.repeat(ball_z.astype(np.float32), 10),
            "team_id": np.tile(np.array([off_team] * 5 + [def_team] * 5, dtype=np.int32), n),
            "team": np.tile(np.array([HOME_ABBR if off_team == HOME_ID else AWAY_ABBR] * 5
                                     + [HOME_ABBR if def_team == HOME_ID else AWAY_ABBR] * 5), n),
            "home": np.tile(np.array([off_team == HOME_ID] * 5 + [def_team == HOME_ID] * 5), n),
            "player_id": np.tile(ids, n),
            "player": np.tile(np.array(["Player %d" % i for i in ids]), n),
            "jersey": np.tile(np.array([str(i % 100) for i in ids]), n),
            "x": pos[:, 0].astype(np.float32), "y": pos[:, 1].astype(np.float32)})
        frame_chunks.append(chunk)
        start_frame, end_frame = int(frames_ids[0]), int(frames_ids[-1])
        end_clock = float(clocks[-1])
        side = "home" if off_team == HOME_ID else "away"
        start_event = event_counter
        if end_kind == "period_end":
            ev = [("period_end", None, 0, "End of quarter")]
        elif end_kind == "turnover":
            ev = [("turnover", side, 0, "Turnover by Player %d (bad pass)" % handler)]
        elif end_kind == "free_throws":
            ev = [("foul", "away" if side == "home" else "home", 0, "Shooting foul"),
                  ("ft_made" if points >= 1 else "ft_missed", side, min(points, 1), "Player %d free throw 1 of 2" % handler),
                  ("ft_made" if points == 2 else "ft_missed", side, 1 if points == 2 else 0, "Player %d free throw 2 of 2" % handler)]
        elif end_kind == "shot_made":
            ev = [("shot_made", side, points, "Player %d makes %d-pt shot" % (handler, points))]
        else:
            ev = [("shot_missed", side, 0, "Player %d misses shot" % handler),
                  ("rebound", "away" if side == "home" else "home", 0, "Defensive rebound by Player %d" % def_ids[handler_k])]
        for kind, team, pts, desc in ev:
            event_rows.append(dict(event=event_counter, period=period, clock_s=end_clock, team=team, desc=desc,
                                   points=pts, score=None, kind=kind, frame=end_frame))
            event_counter += 1
        end_event = event_counter - 1
        poss_rows.append(dict(sportvu_game=game, possession=poss_idx, period=period, start_frame=start_frame,
                              end_frame=end_frame, start_clock_s=float(clocks[0]), end_clock_s=end_clock,
                              start_event=start_event, end_event=end_event, offense_team_id=off_team,
                              end_kind=end_kind, points=points, gap=False, n_frames=n))
        if shot:
            hx, hy = pos_off[-1, handler_k]
            dx = (hx - basket[0]) * sx
            dy = hy - basket[1]
            dist = float(np.hypot(dx, dy))
            shot_rows.append(dict(game=game, sportvu_game=game, br_code="2016%02d0%s" % (seed % 28 + 1, HOME_ABBR),
                                  player="Player %d" % handler, player_slug="player%d" % handler,
                                  team=HOME_ABBR if off_team == HOME_ID else AWAY_ABBR,
                                  team_br=HOME_ABBR if off_team == HOME_ID else AWAY_ABBR, team_side=side,
                                  period=period, clock_s=end_clock, clock_str="%d:%04.1f" % divmod(end_clock, 60),
                                  made=end_kind == "shot_made", shot_type="3PT" if dist >= 22 else "2PT",
                                  shot_distance_ft=float(round(dist)), court_x=5.25 + dx, court_y=25.0 + dy, basket="left",
                                  loc_x=float(dy * 10), loc_y=float(dx * 10), desc="Player %d shot" % handler,
                                  game_date="2016-01-%02d" % (seed % 28 + 1), home_team=HOME_ABBR, away_team=AWAY_ABBR,
                                  split="heldout_game" if heldout else None, heldout=bool(heldout), season_type="Regular Season",
                                  game_id=None, event_id=None, player_id=None, team_id=None, release_frame=None,
                                  closest_defender_id=None, closest_defender_dist_ft=None, defender_closing_speed_ftps=None,
                                  source="bref", endpoint="boxscores/shot-chart/x.html", season=SEASON, fetched_at="2026-01-01T00:00:00Z"))
            sf_rows.append(dict(sportvu_game=game, event=start_event, period=period, clock_s=end_clock, made=end_kind == "shot_made",
                                pbp_shooter_name="Player %d" % handler, shooter_id=handler, release_frame=max(end_frame - 5, start_frame),
                                apex_frame=max(end_frame - 2, start_frame), outcome_frame=end_frame, shooter_at_release_id=handler,
                                identity_match=True, confidence=0.8, confidence_flag="high"))
            if end_kind == "shot_missed":
                rb_rows.append(dict(sportvu_game=game, event=start_event, rebound_event=start_event + 1, period=period,
                                    clock_s=end_clock, rim_frame=end_frame, rebound_frame=end_frame, rebounder_id=def_ids[handler_k],
                                    pbp_rebounder_name="Player %d" % def_ids[handler_k], pbp_rebounder_id=def_ids[handler_k],
                                    identity_match=True, confidence=0.8, confidence_flag="high"))
        frame_counter += n
        clock = end_clock - DT
        poss_idx += 1
        offense_home = not offense_home
    frames = pd.concat(frame_chunks, ignore_index=True)
    events = pd.DataFrame(event_rows)
    events["game"] = game
    events["frame_clock_diff_s"] = np.float32(0.0)
    possessions = pd.DataFrame(poss_rows)
    rosters = pd.DataFrame([dict(sportvu_game=game, player_id=pid, player="Player %d" % pid,
                                 team_id=HOME_ID if pid in HOME_IDS else AWAY_ID,
                                 team=HOME_ABBR if pid in HOME_IDS else AWAY_ABBR, home=pid in HOME_IDS,
                                 jersey=str(pid % 100), bref_slug="player%d" % pid, bref_name="Player %d" % pid,
                                 pbp_name="Player %d" % pid, match_method="exact") for pid in HOME_IDS + AWAY_IDS])
    shots = pd.DataFrame(shot_rows)
    shot_frames = pd.DataFrame(sf_rows)
    rebound_frames = pd.DataFrame(rb_rows)
    from analytics.epv.states import build_states
    states = build_states(frames, possessions, game, events=events)
    return dict(frames=frames, events=events, shots=shots, rosters=rosters, shot_frames=shot_frames,
                rebound_frames=rebound_frames, possessions=possessions, states_1hz=states, sportvu_game=game, heldout=bool(heldout))


def _fallback_synth_season(seed: int, n_games: int = 8, heldout_every: int = 0) -> dict:
    games = []
    parts = {k: [] for k in ("frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")}
    n_held = 0
    for i in range(n_games):
        held = bool(heldout_every) and ((i + 1) % heldout_every == 0)
        if held:
            g = _fallback_synth_game(seed * 100 + i, heldout=True, sportvu_game=HELDOUT_IDS[n_held % len(HELDOUT_IDS)]
                                     + ("" if n_held < len(HELDOUT_IDS) else ".h%d" % n_held))
            n_held += 1
        else:
            g = _fallback_synth_game(seed * 100 + i, heldout=False,
                                     sportvu_game="01.%02d.2016.%s.at.%s" % (i % 28 + 1, AWAY_ABBR, HOME_ABBR))
        games.append((g["sportvu_game"], g["heldout"]))
        for k in parts:
            parts[k].append(g[k])
    out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    out["games"] = games
    return out


def _game_meta(g: dict) -> tuple:
    """(sportvu_game, heldout) of a synth_game dict from either generator."""
    meta = g.get("meta") or {}
    game = g.get("sportvu_game") or meta.get("sportvu_game") or str(g["possessions"]["sportvu_game"].iloc[0])
    held = g.get("heldout", meta.get("heldout", False))
    return str(game), bool(held)


def use_fallback(flag=None) -> bool:
    """True when the local generator is to be used: no shared generator, an explicit flag, or EPV_FIXTURE_FALLBACK=1."""
    if flag is not None:
        return bool(flag) or not SHARED
    return (not SHARED) or os.environ.get("EPV_FIXTURE_FALLBACK", "") == "1"


def season_tables(seed: int = 0, n_games: int = 8, heldout_every: int = 0, fallback=None) -> dict:
    """The season as concatenated tables whichever generator made it: keys frames, events, shots, rosters,
    shot_frames, rebound_frames, possessions, states_1hz (each one DataFrame) and games [(sportvu_game, heldout)]."""
    if use_fallback(fallback):
        return _fallback_synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    season = _shared_synth_season(seed, n_games=n_games, heldout_every=heldout_every)
    if isinstance(season, dict) and "games" in season:
        return season
    parts = {k: [] for k in ("frames", "events", "shots", "rosters", "shot_frames", "rebound_frames", "possessions", "states_1hz")}
    games = []
    for g in season:
        games.append(_game_meta(g))
        for k in parts:
            parts[k].append(g[k])
    out = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    out["games"] = games
    return out


synth_game = _shared_synth_game if SHARED else _fallback_synth_game
synth_season = _shared_synth_season if SHARED else _fallback_synth_season
fallback_synth_game, fallback_synth_season = _fallback_synth_game, _fallback_synth_season


def _fallback_write_fixture_root(root, seed: int, n_games: int, heldout_every: int = 0, fallback=None) -> list:
    """Write the DATA_ROOT layout the EPV CLIs read. Returns [(sportvu_game, heldout), ...]."""
    root = Path(root)
    season = season_tables(seed, n_games=n_games, heldout_every=heldout_every, fallback=fallback)
    games = season["games"]
    for game, held in games:
        sub = "heldout/" if held else ""
        d = root / "sportvu_contract" / sub
        d.mkdir(parents=True, exist_ok=True)
        f = season["frames"][season["frames"]["game"] == game]
        f.to_parquet(d / ("%s.parquet" % game), index=False)
        season["events"][season["events"]["game"] == game].to_parquet(d / ("%s_pbp.parquet" % game), index=False)
        for table in ("possessions", "states_1hz", "shot_frames", "rebound_frames", "rosters"):
            t = season[table]
            t = t[t["sportvu_game"] == game] if len(t) else t
            td = root / "analytics" / table / sub / SEASON
            td.mkdir(parents=True, exist_ok=True)
            t.to_parquet(td / ("%s.parquet" % game), index=False)
    shots = season["shots"]
    held = shots["heldout"].astype(bool) if len(shots) else pd.Series([], dtype=bool)
    for flag, sub in ((False, ""), (True, "heldout/")):
        part = shots[held == flag] if len(shots) else shots
        if len(part) == 0 and flag:
            continue
        sd = root / "collect" / "bref_shots" / sub / "shotchart"
        sd.mkdir(parents=True, exist_ok=True)
        part.to_parquet(sd / ("%s.parquet" % SEASON), index=False)
    return games


write_fixture_root = _shared_write_fixture_root if (SHARED and _shared_write_fixture_root is not None) else _fallback_write_fixture_root
fallback_write_fixture_root = _fallback_write_fixture_root

__all__ = ["synth_game", "synth_season", "write_fixture_root", "season_tables", "use_fallback", "fallback_synth_game",
           "fallback_synth_season", "fallback_write_fixture_root", "SHARED", "HELDOUT_IDS", "SEASON"]
