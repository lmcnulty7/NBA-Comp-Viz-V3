"""tests/_fx_betting.py: fixtures for the betting lane.

Prefers the shared generators of analytics.fixtures (lane I1) when they import; otherwise a minimal
local generator with the same signatures and column names, deterministic given the seed. SOURCE names which one
is in use and team_names() gives the id -> name map the join needs for the shared fixtures' synthetic ids:

    synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9)
        -> dict(stints, playbyplay, games)
    synth_odds(seed, games) -> section 14 odds rows consistent with games.spread_home plus noise

The fallback games carry a latent team strength so that the closing spread, the final margin and the
rolling form features are related (the fixture must let the pipeline run end to end; its numbers are
fixture output, never a result). Team ids are real stats.nba.com ids so the name join is exercised.
Tip-offs are written as the real feed gives them (UTC, which for a 7pm ET tip is the next calendar day)
so the date tolerance of the join is exercised too. Some games carry two closing snapshots and two books
so the duplicate resolution has something to count.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

try:  # the shared fixtures of lane I1, when present
    from analytics import fixtures as _shared
    from analytics.fixtures import synth_odds, synth_stints  # noqa: F401
    SOURCE = "analytics/fixtures.py"

    def team_names(games=None) -> dict:
        """id -> feed name of the shared fixtures' synthetic teams (their ids are not stats.nba.com ids), for
        validate.validate(..., team_names=...); the join also infers them from dates when this is not passed.
        From the games frame's home_team / away_team codes when given (synth_odds names its teams from those codes),
        plus the league formula LEAGUE_TEAM_ID_BASE + LEAGUE_CODES.index(code) for every code with a name."""
        out = {}
        league_base, league_codes = getattr(_shared, "LEAGUE_TEAM_ID_BASE", None), getattr(_shared, "LEAGUE_CODES", ())
        if league_base is not None:
            out.update({league_base + i: _shared.TEAM_NAMES[c] for i, c in enumerate(league_codes) if c in _shared.TEAM_NAMES})
        if games is not None and {"home_team", "away_team"} <= set(games.columns):
            for side in ("home", "away"):
                for tid, code in zip(games[side + "_team_id"], games[side + "_team"]):
                    if str(code) in _shared.TEAM_NAMES:
                        out[int(tid)] = _shared.TEAM_NAMES[str(code)]
        return out
except ImportError:  # pragma: no cover - exercised when the shared module is absent
    SOURCE = "tests/_fx_betting.py (fallback generator)"

    def team_names(games=None) -> dict:
        return {}      # the fallback uses real stats.nba.com ids, which validate.TEAM_NAMES knows

    _TEAMS = [(1610612744, "Golden State Warriors"), (1610612747, "Los Angeles Lakers"),
              (1610612738, "Boston Celtics"), (1610612748, "Miami Heat"),
              (1610612743, "Denver Nuggets"), (1610612749, "Milwaukee Bucks")]
    _NAME = dict(_TEAMS)

    def _season_start(season: str) -> dt.date:
        return dt.date(int(season[:4]), 12, 1)

    def synth_stints(seed: int = 0, n_games: int = 20, seasons=("2020-21",), n_players_per_team: int = 9) -> dict:
        rng = np.random.default_rng(seed)
        games, stints, pbp = [], [], []
        for si, season in enumerate(seasons):
            strength = {tid: float(rng.normal(0, 4)) for tid, _ in _TEAMS}
            start = _season_start(season)
            for k in range(n_games):
                home, away = rng.choice(len(_TEAMS), size=2, replace=False)
                hid, aid = _TEAMS[home][0], _TEAMS[away][0]
                gid = "002%d%05d" % (int(season[2:4]), k + 1)
                date = start + dt.timedelta(days=int(k * 1.5 + rng.integers(0, 2)))
                expected = strength[hid] - strength[aid] + 3.0
                margin = int(round(expected + rng.normal(0, 12)))
                if margin == 0:
                    margin = 1
                total = int(round(rng.normal(222, 10)))
                hs, as_ = (total + margin) // 2, (total - margin) // 2
                if hs - as_ != margin:
                    hs += 1
                spread = -round((expected + rng.normal(0, 1.5)) * 2) / 2
                if spread == int(spread):
                    spread += 0.5
                games.append({"game_id": gid, "season": season, "game_date": date.isoformat(),
                              "home_team_id": int(hid), "away_team_id": int(aid), "home_score": int(hs),
                              "away_score": int(as_), "home_win": bool(hs > as_),
                              "spread_home": float(spread) if rng.uniform() > 0.05 else None,
                              "heldout": False, "split": None, "season_type": "Regular Season"})
                hp = [int(hid) * 100 + i for i in range(n_players_per_team)]
                ap = [int(aid) * 100 + i for i in range(n_players_per_team)]
                for s in range(4):
                    row = {"game_id": gid, "stint": s, "start_s": 720.0 * s, "end_s": 720.0 * (s + 1), "duration_s": 720.0,
                           "period_start": s + 1, "period_end": s + 1}
                    for i in range(5):
                        row["home_%d" % (i + 1)] = hp[(s + i) % n_players_per_team]
                        row["away_%d" % (i + 1)] = ap[(s + i) % n_players_per_team]
                    row.update({"home_ids": "|".join(map(str, hp[:5])), "away_ids": "|".join(map(str, ap[:5])),
                                "n_home": 5, "n_away": 5, "lineup_ok": True, "points_home": int(hs // 4),
                                "points_away": int(as_ // 4), "score_home": int(hs * (s + 1) // 4),
                                "score_away": int(as_ * (s + 1) // 4), "split": None, "heldout": False,
                                "season_type": "Regular Season", "game_date": date.isoformat(), "season": season})
                    stints.append(row)
                for e, (per, score) in enumerate([(1, "0-0"), (4, "%d-%d" % (as_, hs))]):
                    pbp.append({"game": gid, "game_id": gid, "event": e, "period": per, "clock_s": 720.0 if e == 0 else 0.0,
                                "team": "home", "desc": "tip" if e == 0 else "end", "points": 0, "score": score,
                                "kind": "period_start" if e == 0 else "period_end", "frame": -1, "team_id": int(hid),
                                "season": season, "game_date": date.isoformat(), "split": None, "heldout": False,
                                "season_type": "Regular Season"})
                pbp.append({"game": gid, "game_id": gid, "event": 2, "period": 1, "clock_s": 700.0, "team": "away",
                            "desc": "shot", "points": 2, "score": "2-0", "kind": "shot_made", "frame": -1,
                            "team_id": int(aid), "season": season, "game_date": date.isoformat(), "split": None,
                            "heldout": False, "season_type": "Regular Season"})
        return {"stints": pd.DataFrame(stints), "playbyplay": pd.DataFrame(pbp), "games": pd.DataFrame(games)}

    def _price(p: float) -> int:
        p = min(max(p, 0.05), 0.95)
        return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

    def synth_odds(seed: int = 0, games: pd.DataFrame | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed + 1)
        rows = []
        for g in games.itertuples(index=False):
            date = dt.date.fromisoformat(g.game_date)
            tip = dt.datetime(date.year, date.month, date.day, 19, 0) + dt.timedelta(hours=5)  # 7pm ET as UTC
            spread = g.spread_home if g.spread_home is not None and not pd.isna(g.spread_home) else -2.5
            home, away = _NAME[int(g.home_team_id)], _NAME[int(g.away_team_id)]
            key = "%s_%s_%s" % (tip.strftime("%Y-%m-%d"), away.replace(" ", ""), home.replace(" ", ""))
            p_home = 1 / (1 + np.exp(spread / 6.0))
            snaps = [(tip - dt.timedelta(hours=20), False), (tip - dt.timedelta(minutes=5), True)]
            if rng.uniform() < 0.3:
                snaps.append((tip - dt.timedelta(minutes=35), True))  # a second closing snapshot (duplicate)
            books = ["book_a", "book_b"] if rng.uniform() < 0.5 else ["book_a"]
            for at, close in snaps:
                for b in books:
                    sp = spread + (0 if close else rng.choice([-1, -0.5, 0, 0.5, 1]))
                    tot = 222.0 + (round(rng.normal(0, 4)) if not close else 0) + 0.5
                    hv = 0.045 + 0.005 * rng.uniform()
                    for market, side, point, price in [
                            ("h2h", home, None, _price(p_home + hv)), ("h2h", away, None, _price(1 - p_home + hv)),
                            ("spreads", home, sp, -110), ("spreads", away, -sp, -110),
                            ("totals", "Over", tot, -110), ("totals", "Under", tot, -110)]:
                        rows.append({"game_key": key, "commence_time": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "home_team": home, "away_team": away, "book": b, "market": market, "side": side,
                                     "point": point, "price": price, "snapshot_at": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "is_close": close, "source": "the-odds-api.com", "endpoint": "historical/odds",
                                     "season": g.season, "fetched_at": "2026-01-01T00:00:00Z"})
        df = pd.DataFrame(rows)
        df["point"] = df["point"].astype("float32")
        df["price"] = df["price"].astype("int32")
        return df
