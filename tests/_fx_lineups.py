"""tests/_fx_lineups.py: fixtures for the lineups lane (analytics/lineups).

Prefers the shared generator ``analytics.fixtures.synth_stints`` (lane I1) and falls back to the local
generator below with the same signature and column names, so the lineups tests pass with the fallback alone.

Fallback generator: synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9) -> dict
    stints       docs/DATA_CONTRACT.md section 12 columns plus season, source, endpoint, fetched_at
    playbyplay   section 13 columns: scoring and non-scoring events with player1_id, substitutions
                 (kind substitution, player1_id out, player2_id in, team home or away, a basketball-reference
                 style description "P2 enters the game for P1"), period_start and period_end rows,
                 a running away-home score
    games        game_id, season, game_date, home_team_id, away_team_id, home_score, away_score, home_win, spread_home
    starters     game_id, team_id, player_id: the period-1 starters (the starters table of the lineups lane)
    truth        hidden player ratings (player_id -> per 100) and lineup synergies (lineup_id -> per 100)

Hidden structure: four teams (ids 1 to 4) with a fixed roster across seasons; every player carries a rating
(per 100 possessions, starters biased upward) and every five-man lineup a synergy drawn once from a generator
seeded by the sorted ids, larger for the starting fives. A stint's expected differential per 100 is home
court (2.5) plus the home ratings and synergy minus the away ratings and synergy; possessions per team
are Poisson at one per 28.8 s (100 per 48 minutes) and each scores with probability ppp / 2.18 for 1, 2 or 3
points (6, 70, 24 percent), ppp being 1.10 plus or minus the differential over 200. Rotations:
starters open period 1, each later period opens with the starters (60 percent) or the lineup that ended the
previous period, and substitutions arrive every 200 s on average: a starter returns for a bench player (60 percent when one sits) or a bench player (the shallow bench first) replaces someone.
Deterministic given the seed. Fixture numbers are never results.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

try:
    from analytics.fixtures import synth_stints  # noqa: F401  (lane I1's shared generator)
except ImportError:
    synth_stints = None

N_TEAMS = 4
PERIOD_S = 720.0
HOME_COURT = 2.5
PACE_S_PER_POSS = 28.8
SUB_MEAN_S = 200.0
OTHER_EVENT_S = 20.0


def _elapsed(period: int, clock_s: float) -> float:
    return (period - 1) * PERIOD_S + (PERIOD_S - clock_s)


def _period_clock(elapsed: float) -> tuple[int, float]:
    p = int(min(elapsed // PERIOD_S, 3)) + 1
    return p, round(p * PERIOD_S - elapsed, 1)


def _lineup_id(ids) -> str:
    return "|".join(str(i) for i in sorted(int(x) for x in ids))


def _synergy(seed: int, ids, starters: bool) -> float:
    rng = np.random.default_rng([seed, 7919] + sorted(int(x) for x in ids))
    return float(rng.normal(0, 3.0 if starters else 1.5))


def _season_start(season: str) -> dt.date:
    return dt.date(int(season[:4]), 10, 20)


def _fallback_synth_stints(seed: int = 0, n_games: int = 20, seasons=("2020-21",), n_players_per_team: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    rosters = {t: [t * 100 + k for k in range(1, n_players_per_team + 1)] for t in range(1, N_TEAMS + 1)}
    ratings = {}
    for t, ids in rosters.items():
        for k, pid in enumerate(ids):
            ratings[pid] = float(rng.normal(1.0 if k < 5 else -1.0, 2.0))
    starters_of = {t: ids[:5] for t, ids in rosters.items()}
    synergies: dict[str, float] = {}

    def synergy(ids, team):
        key = _lineup_id(ids)
        if key not in synergies:
            synergies[key] = _synergy(seed, ids, sorted(ids) == sorted(starters_of[team]))
        return synergies[key]

    pairs = [(h, a) for h in rosters for a in rosters if h != a]
    stints, pbp, games, starters = [], [], [], []
    for season in seasons:
        day0 = _season_start(season)
        for g in range(n_games):
            home, away = pairs[g % len(pairs)]
            game_id = "00%s%05d" % (season[2:4], g + 1 + 1000 * seasons.index(season))
            game_date = (day0 + dt.timedelta(days=2 * g)).isoformat()
            fields = {"season": season, "split": "train", "heldout": False, "season_type": "Regular Season", "game_date": game_date}
            for t in (home, away):
                for pid in starters_of[t]:
                    starters.append({"game_id": game_id, "team_id": t, "player_id": pid})
            # rotation: on-court sets per side as a list of (elapsed cut, {side: set})
            on = {"home": list(starters_of[home]), "away": list(starters_of[away])}
            teams = {"home": home, "away": away}
            cuts = [0.0]
            subs = []        # (elapsed, side, out, in)
            lineups_at = {0.0: {s: list(on[s]) for s in on}}
            for period in range(1, 5):
                p0 = (period - 1) * PERIOD_S
                if period > 1:
                    for s in on:
                        if rng.uniform() < 0.6:
                            on[s] = list(starters_of[teams[s]])
                    cuts.append(p0)
                    lineups_at[p0] = {s: list(on[s]) for s in on}
                t = p0
                while True:
                    t = round(t + rng.exponential(SUB_MEAN_S), 1)
                    if t >= p0 + PERIOD_S - 5:
                        break
                    s = "home" if rng.uniform() < 0.5 else "away"
                    st5 = starters_of[teams[s]]
                    bench = [p for p in rosters[teams[s]] if p not in on[s]]
                    pref = [p for p in bench if p in st5]
                    if pref and rng.uniform() < 0.6:          # a starter comes back for a bench player
                        in_p = int(rng.choice(pref))
                        out_p = int(rng.choice([p for p in on[s] if p not in st5] or on[s]))
                    else:                                       # a bench player comes in, the shallow bench first
                        w = np.array([0.5 ** bench.index(p) for p in bench]); w /= w.sum()
                        in_p = int(rng.choice(bench, p=w))
                        out_p = int(rng.choice(on[s]))
                    on[s] = [p for p in on[s] if p != out_p] + [in_p]
                    subs.append((t, s, out_p, in_p))
                    cuts.append(t)
                    lineups_at[t] = {s2: list(on[s2]) for s2 in on}
            cuts = sorted(set(cuts)) + [4 * PERIOD_S]
            # events
            events = []
            for t0, s, out_p, in_p in subs:
                period, clock = _period_clock(t0)
                events.append({"elapsed": t0, "order": 0, "period": period, "clock_s": clock, "team": s, "team_id": teams[s],
                               "desc": "P%d enters the game for P%d" % (in_p, out_p), "points": 0, "kind": "substitution",
                               "player1_id": out_p, "player2_id": in_p, "player3_id": None, "event_type": 8})
            for period in range(1, 5):
                events.append({"elapsed": (period - 1) * PERIOD_S, "order": -2, "period": period, "clock_s": PERIOD_S, "team": None, "team_id": None,
                               "desc": "Start of %d period" % period, "points": 0, "kind": "period_start",
                               "player1_id": None, "player2_id": None, "player3_id": None, "event_type": 12})
                events.append({"elapsed": period * PERIOD_S, "order": 2, "period": period, "clock_s": 0.0, "team": None, "team_id": None,
                               "desc": "End of %d period" % period, "points": 0, "kind": "period_end",
                               "player1_id": None, "player2_id": None, "player3_id": None, "event_type": 13})
            score = {"home": 0, "away": 0}
            n = 0
            for t0, t1 in zip(cuts[:-1], cuts[1:]):
                if t1 <= t0:
                    continue
                lu = lineups_at[t0]
                dur = t1 - t0
                poss = dur / PACE_S_PER_POSS
                d = (HOME_COURT + sum(ratings[p] for p in lu["home"]) + synergy(lu["home"], home)
                     - sum(ratings[p] for p in lu["away"]) - synergy(lu["away"], away))
                pts = {"home": 0, "away": 0}
                for s, sign in (("home", 1), ("away", -1)):
                    ppp = max(0.2, 1.10 + sign * d / 200.0)
                    n_poss = int(rng.poisson(poss))
                    scored = rng.uniform(size=n_poss) < min(0.95, ppp / 2.18)
                    for v in rng.choice([1, 2, 3], p=[0.06, 0.70, 0.24], size=int(scored.sum())):
                        v = int(v)
                        t = min(max(round(t0 + rng.uniform() * dur, 1), t0 + 0.1), t1)
                        period, clock = _period_clock(t)
                        pts[s] += v
                        events.append({"elapsed": t, "order": 1, "period": period, "clock_s": clock, "team": s, "team_id": teams[s],
                                       "desc": "P%d makes %s" % (int(rng.choice(lu[s])), "free throw" if v == 1 else "%d-pt shot" % v),
                                       "points": v, "kind": "ft_made" if v == 1 else "shot_made",
                                       "player1_id": int(rng.choice(lu[s])), "player2_id": None, "player3_id": None, "event_type": 1 if v > 1 else 3})
                    for _ in range(rng.poisson(dur / OTHER_EVENT_S)):
                        kind = str(rng.choice(["rebound", "turnover", "foul", "shot_missed"]))
                        t = min(max(round(t0 + rng.uniform() * dur, 1), t0 + 0.1), t1)
                        period, clock = _period_clock(t)
                        events.append({"elapsed": t, "order": 1, "period": period, "clock_s": clock, "team": s, "team_id": teams[s],
                                       "desc": "P%d %s" % (int(rng.choice(lu[s])), kind), "points": 0, "kind": kind,
                                       "player1_id": int(rng.choice(lu[s])), "player2_id": None, "player3_id": None,
                                       "event_type": {"rebound": 4, "turnover": 5, "foul": 6, "shot_missed": 2}[kind]})
                score["home"] += pts["home"]; score["away"] += pts["away"]
                hs, as_ = sorted(lu["home"]), sorted(lu["away"])
                row = {"game_id": game_id, "stint": n, "start_s": round(t0, 1), "end_s": round(t1, 1), "duration_s": round(dur, 1),
                       "period_start": int(min(t0 // PERIOD_S, 3)) + 1, "period_end": int(min(max(t1 - 0.1, t0) // PERIOD_S, 3)) + 1}
                for k in range(5):
                    row["home_%d" % (k + 1)] = hs[k]
                    row["away_%d" % (k + 1)] = as_[k]
                row.update({"home_ids": _lineup_id(hs), "away_ids": _lineup_id(as_), "n_home": 5, "n_away": 5, "lineup_ok": True,
                            "points_home": pts["home"], "points_away": pts["away"], "score_home": score["home"], "score_away": score["away"],
                            **fields, "source": "fixture", "endpoint": "synth_stints", "fetched_at": "2026-01-01T00:00:00Z"})
                stints.append(row)
                n += 1
            events.sort(key=lambda e: (e["elapsed"], e["order"]))
            run = {"home": 0, "away": 0}
            for i, e in enumerate(events):
                if e["points"]:
                    run[e["team"]] += e["points"]
                pbp.append({"game": game_id, "game_id": game_id, "event": i, "period": e["period"], "clock_s": e["clock_s"], "team": e["team"],
                            "desc": e["desc"], "points": e["points"], "score": "%d-%d" % (run["away"], run["home"]) if e["points"] else None,
                            "kind": e["kind"], "frame": -1, "frame_clock_diff_s": None, "source": "fixture", "endpoint": "synth_stints",
                            "fetched_at": "2026-01-01T00:00:00Z", "event_id": i, "event_type": e["event_type"], "action_type": 0,
                            "player1_id": e["player1_id"], "player2_id": e["player2_id"], "player3_id": e["player3_id"],
                            "team_id": e["team_id"], "clock_str": "%d:%04.1f" % divmod(e["clock_s"], 60), **fields})
            games.append({"game_id": game_id, "season": season, "game_date": game_date, "home_team_id": home, "away_team_id": away,
                          "home_score": score["home"], "away_score": score["away"], "home_win": score["home"] > score["away"],
                          "spread_home": float(score["home"] - score["away"])})
    st = pd.DataFrame(stints)
    for c in [f"home_{k}" for k in range(1, 6)] + [f"away_{k}" for k in range(1, 6)]:
        st[c] = st[c].astype("Int32")
    pb = pd.DataFrame(pbp)
    for c in ("player1_id", "player2_id", "player3_id", "team_id", "event_id"):
        pb[c] = pb[c].astype("Int32")
    return {"stints": st, "playbyplay": pb, "games": pd.DataFrame(games), "starters": pd.DataFrame(starters),
            "truth": {"ratings": ratings, "synergies": dict(synergies)}}


if synth_stints is None:
    synth_stints = _fallback_synth_stints

__all__ = ["synth_stints", "_fallback_synth_stints"]
