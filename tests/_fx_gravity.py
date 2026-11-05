"""tests/_fx_gravity.py: fixtures for the gravity lane (analytics/gravity).

Imports the shared generator ``analytics.fixtures`` (lane I1) when it exists, else falls back to the
minimal local generator below with the same signatures and column names:

    synth_game(seed, n_possessions=40, heldout=False, sportvu_game=None) -> dict(
        frames, events, shots, rosters, shot_frames, rebound_frames, possessions, states_1hz)
    synth_season(seed, n_games=8, heldout_every=0) -> list of such dicts

The fallback is deterministic given the seed and writes nothing. Each game dict also carries
``sportvu_game``, ``heldout`` and ``high_gravity_ids`` (the two players, one per team, whose
defender stays about HG_PULL_FT closer than their location alone would say). The tests of this lane
pass with the fallback alone; whether the shared generator is in use is recorded in ``SOURCE``.

Fallback world (docs/DATA_CONTRACT.md sections 4, 5, 10a, 16 to 20 column names):
  - court 94 x 50 ft, baskets at (5.25, 25) and (88.75, 25); the home team attacks the left basket,
    the away team the right one, in every period (no half-time swap, which the features must not assume)
  - frames at HZ = 5 per second (the contract fixes no rate; the analytics sample at 1 Hz anyway),
    ten rows per frame, ball carried by the handler at z = 4 ft
  - seven players per team, five on the floor per possession; the high-gravity player of each team
    is on the floor for about 60 percent of the possessions
  - every possession alternates offence; it ends with a shot (made or missed) or a turnover;
    a missed shot is followed by a defensive rebound event
  - the nearest-defender distance of an offensive player is
        DEF_BASE + DEF_PER_FT * distance_to_basket - DEF_BALL * has_ball - HG_PULL_FT * is_high_gravity + noise
    so a location-only model leaves the pull as a residual (the thing the lane measures)
  - teammates of the high-gravity player shoot from nearer the basket when that player is on the floor,
    so a location-only xFG probability rises with the player on court (the sign test of the lane)
  - shots are 10a rows with a null ``player_id`` (as basketball-reference gives them); the shooter is
    matched to the roster by name, as on real data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # the shared generator of lane I1, when it exists (GRAVITY_FX_FALLBACK=1 forces the fallback, for its own test)
    import os as _os
    if _os.environ.get("GRAVITY_FX_FALLBACK"):
        raise ImportError("fallback forced by GRAVITY_FX_FALLBACK")
    from analytics.fixtures import synth_game, synth_season  # type: ignore  # noqa: F401
    SOURCE = "analytics.fixtures"
except ImportError:  # the local fallback
    SOURCE = "tests._fx_gravity"

    HZ = 5
    PERIOD_S = 720.0
    COURT_X, COURT_Y = 94.0, 50.0
    LEFT_BASKET, RIGHT_BASKET = (5.25, 25.0), (88.75, 25.0)
    HOME_ID, AWAY_ID = 100, 200
    HOME_TEAM, AWAY_TEAM = "HHH", "AAA"
    N_ROSTER, N_ON_FLOOR = 7, 5
    HG_HOME, HG_AWAY = 1001, 2001
    HG_PULL_FT = 2.0
    HG_ON_P = 0.6
    DEF_BASE, DEF_PER_FT, DEF_BALL, DEF_NOISE = 2.5, 0.22, 1.5, 0.8
    TURNOVER_P = 0.1
    HELDOUT_GAME = "12.16.2015.PHX.at.GSW"

    def _roster_rows(sportvu_game: str) -> list[dict]:
        rows = []
        for team_id, team, home, base in ((HOME_ID, HOME_TEAM, True, 1000), (AWAY_ID, AWAY_TEAM, False, 2000)):
            for k in range(1, N_ROSTER + 1):
                pid = base + k
                name = "%s Player%d" % (team, k)
                rows.append({"sportvu_game": sportvu_game, "player_id": pid, "player": name, "team_id": team_id,
                             "team": team, "home": home, "jersey": str(k), "bref_slug": "%splay%02d" % (team.lower(), k),
                             "bref_name": name, "pbp_name": "%s. Player%d" % (team[0], k), "match_method": "exact"})
        return rows

    def _basket(team_id: int) -> tuple[float, float]:
        return LEFT_BASKET if team_id == HOME_ID else RIGHT_BASKET

    def synth_game(seed: int, n_possessions: int = 40, heldout: bool = False, sportvu_game=None) -> dict:
        """One synthetic game as the dict of tables described in the module docstring."""
        rng = np.random.default_rng(seed)
        if sportvu_game is None:
            sportvu_game = HELDOUT_GAME if heldout else "01.%02d.2016.%s.at.%s" % (seed % 28 + 1, AWAY_TEAM, HOME_TEAM)
        rosters = pd.DataFrame(_roster_rows(sportvu_game))
        skill = {int(p): float(rng.normal(0.0, 0.06)) for p in rosters["player_id"]}
        team_players = {HOME_ID: rosters.loc[rosters["team_id"] == HOME_ID, "player_id"].to_numpy(),
                        AWAY_ID: rosters.loc[rosters["team_id"] == AWAY_ID, "player_id"].to_numpy()}
        hg_of = {HOME_ID: HG_HOME, AWAY_ID: HG_AWAY}
        side_of = {HOME_ID: "home", AWAY_ID: "away"}
        abbr_of = {HOME_ID: HOME_TEAM, AWAY_ID: AWAY_TEAM}
        per_period = max(1, int(np.ceil(n_possessions / 2)))

        frame_rows, event_rows, shot_rows, sf_rows, poss_rows, state_rows = [], [], [], [], [], []
        frame_id, event_id, ts = 0, 0, 1_000_000
        score = [0, 0]  # away, home
        period_open = {}
        for p in range(n_possessions):
            period = 1 + p // per_period
            first_in_period = p % per_period == 0
            if first_in_period:
                clock = PERIOD_S
                event_rows.append({"game": sportvu_game, "event": event_id, "period": period, "clock_s": clock, "team": None,
                                   "desc": "Start of period %d" % period, "points": 0, "score": "%d-%d" % tuple(score),
                                   "kind": "period_start", "frame": frame_id, "frame_clock_diff_s": 0.0})
                event_id += 1
            else:
                clock = period_open[period]
            off_id = HOME_ID if p % 2 == 0 else AWAY_ID
            def_id = AWAY_ID if off_id == HOME_ID else HOME_ID
            dur_s = int(rng.integers(8, 13))
            n_frames = dur_s * HZ

            def pick(team_id):
                pool = team_players[team_id]
                hg = hg_of[team_id]
                others = pool[pool != hg]
                if rng.uniform() < HG_ON_P:
                    chosen = np.concatenate([[hg], rng.choice(others, N_ON_FLOOR - 1, replace=False)])
                else:
                    chosen = rng.choice(others, N_ON_FLOOR, replace=False)
                return np.sort(chosen)

            off_players, def_players = pick(off_id), pick(def_id)
            hg_on = hg_of[off_id] in off_players
            bx, by = _basket(off_id)
            toward_mid = 1.0 if bx < COURT_X / 2 else -1.0
            # starting spots of the five offensive players in the frontcourt, then a slow random walk
            pos = np.column_stack([bx + toward_mid * rng.uniform(3, 28, N_ON_FLOOR), rng.uniform(4, 46, N_ON_FLOOR)])
            handler_slot = int(rng.integers(0, N_ON_FLOOR))
            ends_turnover = rng.uniform() < TURNOVER_P
            shooter_slot = int(rng.integers(0, N_ON_FLOOR))
            shot_dist = float(rng.uniform(2, 18)) if hg_on else float(rng.uniform(8, 28))
            shot_angle = float(rng.uniform(-1.2, 1.2))
            shot_xy = np.array([bx + toward_mid * shot_dist * np.cos(shot_angle), by + shot_dist * np.sin(shot_angle)])
            start_frame, start_clock = frame_id, clock
            poss_states = []
            for k in range(n_frames):
                if k > 0:
                    pos = pos + rng.normal(0, 0.5, pos.shape)
                    pos[:, 0] = np.clip(pos[:, 0], 0.5, COURT_X - 0.5)
                    pos[:, 1] = np.clip(pos[:, 1], 0.5, COURT_Y - 0.5)
                    if k % (3 * HZ) == 0:
                        handler_slot = int(rng.integers(0, N_ON_FLOOR))
                last = k == n_frames - 1
                if last and not ends_turnover:
                    pos[shooter_slot] = shot_xy
                    handler_slot = shooter_slot
                ball = np.array([pos[handler_slot, 0] + 0.8 * toward_mid, pos[handler_slot, 1], 4.0])
                dist_b = np.hypot(pos[:, 0] - bx, pos[:, 1] - by)
                dpos = np.zeros_like(pos)
                for i in range(N_ON_FLOOR):
                    d = (DEF_BASE + DEF_PER_FT * dist_b[i] - DEF_BALL * (i == handler_slot)
                         - HG_PULL_FT * (off_players[i] == hg_of[off_id]) + rng.normal(0, DEF_NOISE))
                    d = max(0.5, d)
                    v = np.array([bx, by]) - pos[i]
                    nv = np.linalg.norm(v)
                    u = v / nv if nv > 1e-6 else np.array([toward_mid * -1.0, 0.0])
                    u = u + rng.normal(0, 0.15, 2)
                    u = u / max(np.linalg.norm(u), 1e-6)
                    dpos[i] = pos[i] + d * u
                dpos[:, 0] = np.clip(dpos[:, 0], 0.5, COURT_X - 0.5)
                dpos[:, 1] = np.clip(dpos[:, 1], 0.5, COURT_Y - 0.5)
                clock_k = start_clock - k / HZ
                shot_clock = max(0.0, 24.0 - k / HZ)
                common = {"game": sportvu_game, "gameid": "002150%04d" % (seed % 10000), "date": "2016-01-01", "period": period,
                          "frame": frame_id, "ts_ms": ts, "game_clock_s": clock_k, "shot_clock_s": shot_clock, "n_players": 10,
                          "ball_x": float(ball[0]), "ball_y": float(ball[1]), "ball_z": float(ball[2])}
                for side_players, side_pos, tid in ((off_players, pos, off_id), (def_players, dpos, def_id)):
                    for i, pid in enumerate(side_players):
                        frame_rows.append({**common, "team_id": tid, "team": abbr_of[tid], "home": tid == HOME_ID,
                                           "player_id": int(pid), "player": "%s Player%d" % (abbr_of[tid], int(pid) % 1000),
                                           "jersey": str(int(pid) % 1000), "x": float(side_pos[i, 0]), "y": float(side_pos[i, 1])})
                if k % HZ == 0:
                    nearest = float(np.min(np.hypot(dpos[:, 0] - pos[handler_slot, 0], dpos[:, 1] - pos[handler_slot, 1])))
                    st = {"sportvu_game": sportvu_game, "possession": p, "frame": frame_id, "period": period, "game_clock_s": clock_k,
                          "shot_clock_s": shot_clock, "elapsed_in_possession_s": float(k / HZ), "offense_team_id": off_id,
                          "handler_id": int(off_players[handler_slot]), "ball_x": float(ball[0]), "ball_y": float(ball[1]),
                          "ball_z": float(ball[2])}
                    for i in range(N_ON_FLOOR):
                        st["off%d_id" % (i + 1)] = int(off_players[i])
                        st["off%d_x" % (i + 1)], st["off%d_y" % (i + 1)] = float(pos[i, 0]), float(pos[i, 1])
                    for i in range(N_ON_FLOOR):
                        st["def%d_id" % (i + 1)] = int(def_players[i])
                        st["def%d_x" % (i + 1)], st["def%d_y" % (i + 1)] = float(dpos[i, 0]), float(dpos[i, 1])
                    st["handler_nearest_def_dist_ft"] = nearest
                    poss_states.append(st)
                frame_id += 1
                ts += 1000 // HZ
            end_frame = frame_id - 1
            end_clock = start_clock - (n_frames - 1) / HZ
            shooter = int(off_players[shooter_slot])
            points = 0
            if ends_turnover:
                end_kind = "turnover"
                event_rows.append({"game": sportvu_game, "event": event_id, "period": period, "clock_s": end_clock, "team": side_of[off_id],
                                   "desc": "Turnover by %s" % shooter, "points": 0, "score": "%d-%d" % tuple(score), "kind": "turnover",
                                   "frame": end_frame, "frame_clock_diff_s": 0.0})
                start_event, end_event = event_id, event_id
                event_id += 1
            else:
                is_three = shot_dist >= 22.0
                p_make = float(np.clip(0.62 - 0.014 * shot_dist + skill[shooter], 0.05, 0.95))
                made = bool(rng.uniform() < p_make)
                points = (3 if is_three else 2) if made else 0
                if made:
                    score[1 if off_id == HOME_ID else 0] += points
                end_kind = "shot_made" if made else "shot_missed"
                name = "%s Player%d" % (abbr_of[off_id], shooter % 1000)
                event_rows.append({"game": sportvu_game, "event": event_id, "period": period, "clock_s": end_clock, "team": side_of[off_id],
                                   "desc": "%s %s %d-pt shot" % (name, "makes" if made else "misses", 3 if is_three else 2),
                                   "points": points, "score": "%d-%d" % tuple(score), "kind": end_kind, "frame": end_frame,
                                   "frame_clock_diff_s": 0.0})
                start_event, end_event = event_id, event_id
                event_id += 1
                dx_mid = abs(shot_xy[0] - bx)
                loc_x = (shot_xy[1] - by) * 10.0
                loc_y = dx_mid * 10.0
                shot_rows.append({"game": sportvu_game, "sportvu_game": sportvu_game, "br_code": "20160101" + "0" + HOME_TEAM,
                                  "player": name, "player_slug": "%splay%02d" % (abbr_of[off_id].lower(), shooter % 1000),
                                  "team": abbr_of[off_id], "team_br": abbr_of[off_id], "team_side": side_of[off_id], "period": period,
                                  "clock_s": end_clock, "clock_str": "%d:%04.1f" % divmod(end_clock, 60), "made": made,
                                  "shot_type": "3PT" if is_three else "2PT", "shot_distance_ft": float(round(shot_dist)),
                                  "px_left": 239.5 + loc_x, "px_top": 35.5 + loc_y, "court_x": 5.25 + dx_mid, "court_y": 25.0 + (shot_xy[1] - by),
                                  "basket": "left", "loc_x": loc_x, "loc_y": loc_y, "dist_resid_ft": 0.0, "desc": "fixture",
                                  "game_date": "2016-01-01", "home_team": HOME_TEAM, "away_team": AWAY_TEAM,
                                  "split": "heldout_game" if heldout else "train", "heldout": bool(heldout), "season_type": "Regular Season",
                                  "game_id": None, "event_id": None, "player_id": None, "team_id": None, "release_frame": None,
                                  "closest_defender_id": None, "closest_defender_dist_ft": None, "defender_closing_speed_ftps": None,
                                  "source": "bref", "endpoint": "fixture", "season": "2015-16", "fetched_at": "2026-01-01T00:00:00Z"})
                sf_rows.append({"sportvu_game": sportvu_game, "event": end_event, "period": period, "clock_s": end_clock, "made": made,
                                "pbp_shooter_name": name, "shooter_id": shooter, "release_frame": end_frame, "apex_frame": end_frame,
                                "outcome_frame": end_frame, "shooter_at_release_id": shooter, "identity_match": True,
                                "confidence": 0.9, "confidence_flag": "high"})
                if not made:
                    event_rows.append({"game": sportvu_game, "event": event_id, "period": period, "clock_s": end_clock, "team": side_of[def_id],
                                       "desc": "Defensive rebound", "points": 0, "score": "%d-%d" % tuple(score), "kind": "rebound",
                                       "frame": end_frame, "frame_clock_diff_s": 0.0})
                    event_id += 1
            for st in poss_states:
                st["points_to_end"] = points
                st["possession_end_kind"] = end_kind
            state_rows.extend(poss_states)
            poss_rows.append({"sportvu_game": sportvu_game, "possession": p, "period": period, "start_frame": start_frame,
                              "end_frame": end_frame, "start_clock_s": start_clock, "end_clock_s": end_clock, "start_event": start_event,
                              "end_event": end_event, "offense_team_id": off_id, "end_kind": end_kind, "points": points, "gap": False,
                              "n_frames": n_frames})
            period_open[period] = end_clock - 1.0 / HZ
            last_in_period = (p + 1) % per_period == 0 or p == n_possessions - 1
            if last_in_period:
                event_rows.append({"game": sportvu_game, "event": event_id, "period": period, "clock_s": period_open[period], "team": None,
                                   "desc": "End of period %d" % period, "points": 0, "score": "%d-%d" % tuple(score), "kind": "period_end",
                                   "frame": end_frame, "frame_clock_diff_s": 0.0})
                event_id += 1

        frames = pd.DataFrame(frame_rows)
        frames["period"] = frames["period"].astype("int8")
        frames["frame"] = frames["frame"].astype("int32")
        for c in ("game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "x", "y"):
            frames[c] = frames[c].astype("float32")
        possessions = pd.DataFrame(poss_rows)
        shot_frames = pd.DataFrame(sf_rows)
        rebound_frames = pd.DataFrame(columns=["sportvu_game", "event", "rebound_event", "period", "clock_s", "rim_frame", "rebound_frame",
                                               "rebounder_id", "pbp_rebounder_name", "pbp_rebounder_id", "identity_match", "confidence",
                                               "confidence_flag"])
        return {"sportvu_game": sportvu_game, "heldout": bool(heldout), "high_gravity_ids": [HG_HOME, HG_AWAY],
                "frames": frames, "events": pd.DataFrame(event_rows), "shots": pd.DataFrame(shot_rows), "rosters": rosters,
                "shot_frames": shot_frames, "rebound_frames": rebound_frames, "possessions": possessions,
                "states_1hz": pd.DataFrame(state_rows)}

    def synth_season(seed: int, n_games: int = 8, heldout_every: int = 0, n_possessions: int = 40) -> list:
        """n_games games with distinct ids; every heldout_every-th game (1-based) is held out and named with a
        held-out token of sportvu/splits.json (the PHX game first, then games at the CLE arena). The same seven
        players per team appear in every game (ids 1001 to 1007 and 2001 to 2007), as on real data where a
        player table spans a season."""
        out, k_held = [], 0
        for i in range(n_games):
            held = bool(heldout_every) and (i + 1) % heldout_every == 0
            if held:
                name = HELDOUT_GAME if k_held == 0 else "02.%02d.2016.%s.at.CLE" % (k_held, AWAY_TEAM)
                k_held += 1
            else:
                name = "01.%02d.2016.%s.at.%s" % (i + 1, AWAY_TEAM, HOME_TEAM)
            out.append(synth_game(seed * 1000 + i, n_possessions=n_possessions, heldout=held, sportvu_game=name))
        return out
