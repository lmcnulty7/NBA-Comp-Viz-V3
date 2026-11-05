"""tests/_fx_winprob.py: fixtures of the winprob lane.

Uses the shared generators of analytics.fixtures (lane I1) when they exist and otherwise a minimal local
generator with the same signatures and column names:

    synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9)
        -> dict(stints, playbyplay, games)
    synth_odds(seed, games) -> the section 14 odds table

The fallback simulates possessions with a per-team strength so that the home win depends on the running
margin and the time left; the score string is "away-home" on scoring events and null otherwise (as NBA.com
prints it). Deterministic for a seed. Never real data.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

try:
    from analytics.fixtures import synth_odds, synth_stints  # noqa: F401
    SHARED = True
except ImportError:
    SHARED = False

from analytics.winprob.features import TEAM_NAMES

TEAM_IDS = sorted(TEAM_NAMES)
PBP_COLUMNS = ["game", "event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame",
               "frame_clock_diff_s", "source", "endpoint", "season", "fetched_at", "event_id", "event_type",
               "action_type", "player1_id", "player2_id", "player3_id", "team_id", "clock_str", "game_id", "split",
               "heldout", "season_type", "game_date"]
GAME_COLUMNS = ["game_id", "season", "game_date", "home_team_id", "away_team_id", "home_score", "away_score",
                "home_win", "spread_home"]
ODDS_COLUMNS = ["game_key", "commence_time", "home_team", "away_team", "book", "market", "side", "point", "price",
                "snapshot_at", "is_close", "source", "endpoint", "season", "fetched_at"]


def _season_start(season: str) -> dt.date:
    return dt.date(int(season[:4]), 10, 20)


def _simulate_game(rng, game_id, season, date, home, away, s_home, s_away, n_players):
    """One game of events -> (rows, home_score, away_score)."""
    rows = []
    hs = as_ = 0
    edge = 0.012 * (s_home - s_away) + 0.01            # home court plus strength
    ev = 0
    players = {home: [home * 100 + i for i in range(n_players)], away: [away * 100 + i for i in range(n_players)]}

    def push(period, clock, team, desc, pts, kind, scored):
        nonlocal ev
        side = None if team is None else ("home" if team == home else "away")
        rows.append({"game": game_id, "event": ev, "period": period, "clock_s": float(round(clock, 1)), "team": side,
                     "desc": desc, "points": pts, "score": ("%d-%d" % (as_, hs)) if scored else None, "kind": kind,
                     "frame": -1, "frame_clock_diff_s": None, "source": "stats.nba.com", "endpoint": "gamerotation+playbyplayv2",
                     "season": season, "fetched_at": "2026-01-01T00:00:00Z", "event_id": ev + 1,
                     "event_type": None, "action_type": None,
                     "player1_id": int(rng.choice(players[team])) if team is not None else None, "player2_id": None,
                     "player3_id": None, "team_id": team, "clock_str": "%d:%02d" % (int(clock) // 60, int(clock) % 60),
                     "game_id": game_id, "split": "train", "heldout": False, "season_type": "Regular Season",
                     "game_date": date.isoformat()})
        ev += 1

    period = 0
    offense = home if rng.random() < 0.5 else away
    while True:
        period += 1
        length = 720.0 if period <= 4 else 300.0
        clock = length
        push(period, clock, None, "Start of period %d" % period, 0, "period_start", False)
        if period == 1:
            push(period, clock, offense, "Jump ball", 0, "jump_ball", False)
        while True:
            clock -= float(rng.uniform(6, 22))
            if clock <= 0:
                break
            defense = away if offense == home else home
            adv = edge if offense == home else -edge
            r = rng.random()
            if r < 0.13:
                push(period, clock, offense, "Turnover", 0, "turnover", False)
                offense = defense
            elif r < 0.25:
                push(period, clock, defense, "Shooting foul", 0, "foul", False)
                n_ft = 2
                for k in range(1, n_ft + 1):
                    made = rng.random() < 0.77
                    if made:
                        if offense == home:
                            hs += 1
                        else:
                            as_ += 1
                    push(period, clock, offense, "Free Throw %d of %d%s" % (k, n_ft, "" if made else " MISS"), 1 if made else 0,
                         "ft_made" if made else "ft_missed", made)
                if made:
                    offense = defense
                else:
                    reb = offense if rng.random() < 0.2 else defense
                    push(period, max(clock - 1, 0.0), reb, "Rebound", 0, "rebound", False)
                    offense = reb
            else:
                three = rng.random() < 0.38
                p_make = (0.35 if three else 0.52) + adv
                if rng.random() < p_make:
                    pts = 3 if three else 2
                    if offense == home:
                        hs += pts
                    else:
                        as_ += pts
                    push(period, clock, offense, "%s shot" % ("3PT" if three else "Jump"), pts, "shot_made", True)
                    offense = defense
                else:
                    push(period, clock, offense, "MISS %s shot" % ("3PT" if three else "Jump"), 0, "shot_missed", False)
                    reb = offense if rng.random() < 0.25 else defense
                    push(period, max(clock - 1, 0.0), reb, "Rebound", 0, "rebound", False)
                    offense = reb
        push(period, 0.0, None, "End of period %d" % period, 0, "period_end", True)
        if period >= 4 and hs != as_:
            break
    return rows, hs, as_


def _fallback_synth_stints(seed=0, n_games=20, seasons=("2020-21",), n_players_per_team=9):
    rng = np.random.default_rng(seed)
    n_teams = 10
    teams = TEAM_IDS[:n_teams]
    strength = rng.normal(0, 3.0, size=n_teams)
    pbp_rows, games, stints = [], [], []
    for si, season in enumerate(seasons):
        start = _season_start(season)
        for g in range(n_games):
            i, j = rng.choice(n_teams, size=2, replace=False)
            home, away = teams[i], teams[j]
            gid = "00%d%03d" % (22000 + si, g + 1)
            date = start + dt.timedelta(days=int(g))
            rows, hs, as_ = _simulate_game(rng, gid, season, date, home, away, strength[i], strength[j], n_players_per_team)
            pbp_rows.extend(rows)
            spread = float(np.round(-(strength[i] - strength[j] + 2.5) * 2) / 2)
            if rng.random() < 0.15:
                spread = np.nan
            games.append({"game_id": gid, "season": season, "game_date": date.isoformat(), "home_team_id": home,
                          "away_team_id": away, "home_score": hs, "away_score": as_, "home_win": hs > as_,
                          "spread_home": spread})
            n_periods = max(r["period"] for r in rows)
            for p in range(n_periods):
                hp = [home * 100 + k for k in range(5)]
                ap = [away * 100 + k for k in range(5)]
                length = 720.0 if p < 4 else 300.0
                start_s = sum(720.0 if q < 4 else 300.0 for q in range(p))
                row = {"game_id": gid, "stint": p, "start_s": start_s, "end_s": start_s + length, "duration_s": length,
                       "period_start": p + 1, "period_end": p + 1}
                for k in range(5):
                    row["home_%d" % (k + 1)] = hp[k]
                    row["away_%d" % (k + 1)] = ap[k]
                row.update({"home_ids": "|".join(map(str, hp)), "away_ids": "|".join(map(str, ap)), "n_home": 5, "n_away": 5,
                            "lineup_ok": True, "points_home": 0, "points_away": 0, "score_home": hs, "score_away": as_,
                            "split": "train", "heldout": False, "season_type": "Regular Season", "game_date": date.isoformat(),
                            "source": "stats.nba.com", "endpoint": "gamerotation+playbyplayv2", "season": season,
                            "fetched_at": "2026-01-01T00:00:00Z"})
                stints.append(row)
    pbp = pd.DataFrame(pbp_rows, columns=PBP_COLUMNS)
    return {"stints": pd.DataFrame(stints), "playbyplay": pbp, "games": pd.DataFrame(games, columns=GAME_COLUMNS)}


def _fallback_synth_odds(seed, games):
    rng = np.random.default_rng(seed + 1)
    rows = []
    for r in games.itertuples():
        if r.spread_home is None or (isinstance(r.spread_home, float) and np.isnan(r.spread_home)):
            continue
        hn, an = TEAM_NAMES.get(int(r.home_team_id), str(r.home_team_id)), TEAM_NAMES.get(int(r.away_team_id), str(r.away_team_id))
        key = "%s_%s_%s" % (r.game_date, an, hn)
        for book, jitter in (("pinnacle", 0.0), ("draftkings", float(rng.choice([-0.5, 0.0, 0.5])))):
            for is_close, extra in ((True, 0.0), (False, 1.0)):
                for side, sign in ((hn, 1.0), (an, -1.0)):
                    rows.append({"game_key": key, "commence_time": "%sT23:00:00Z" % r.game_date, "home_team": hn, "away_team": an,
                                 "book": book, "market": "spreads", "side": side, "point": sign * (float(r.spread_home) + jitter + extra),
                                 "price": -110, "snapshot_at": "%sT%s:00:00Z" % (r.game_date, "22" if is_close else "12"),
                                 "is_close": is_close, "source": "the-odds-api.com", "endpoint": "historical/odds",
                                 "season": r.season, "fetched_at": "2026-01-01T00:00:00Z"})
            rows.append({"game_key": key, "commence_time": "%sT23:00:00Z" % r.game_date, "home_team": hn, "away_team": an,
                         "book": book, "market": "h2h", "side": hn, "point": None, "price": -150, "snapshot_at": "%sT22:00:00Z" % r.game_date,
                         "is_close": True, "source": "the-odds-api.com", "endpoint": "historical/odds", "season": r.season,
                         "fetched_at": "2026-01-01T00:00:00Z"})
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


if not SHARED:
    synth_stints = _fallback_synth_stints
    synth_odds = _fallback_synth_odds

__all__ = ["synth_stints", "synth_odds", "SHARED", "PBP_COLUMNS", "GAME_COLUMNS", "ODDS_COLUMNS"]
