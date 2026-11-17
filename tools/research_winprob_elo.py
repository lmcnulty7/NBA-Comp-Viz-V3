"""tools/research_winprob_elo.py: the Elo ratings of winprob-E1 (docs/research/winprob.md section 6, PLAN.md D89).

winprob-E1 adds a pregame expected margin mu_hat to the frozen live win probability model. mu_hat comes from an Elo
rating computed only from games played before tip-off: every game updates the two ratings after it ends, and the
pregame difference of a game is read before its own result enters. Three parameters are chosen on the development
seasons 2020-21 to 2023-24 by forward chaining (CHECK_SEASONS): K (the update size), the home advantage in Elo points,
and the share of each rating pulled back to the mean between seasons. Then:

- the development chain runs through 2020-21 to 2023-24 (2020-21 starts every team at MEAN);
- 2024-25 is read as Elo state only (D89): its results move the ratings game by game, and nothing is fitted, tuned or
  scored on it. Its games come through tools.research_testset.elo_state_games_2024_25, which refuses any caller but
  this module;
- the 2025-26 test season is updated game by game inside the scorer of tools.research_winprob, from the schedule
  table the scorer hands to advance(). This module never opens a test file.

Fixed before any result (not tuned): the margin of victory multiplier of the FiveThirtyEight NBA Elo,
((|margin| + 3) ** 0.8) / (7.5 + 0.006 * winner's Elo edge), the logistic Elo curve on a 400-point scale for the
pregame win chance, and POINTS_PER_ELO = 1 / 28 to read an Elo difference as points. The logistic model standardises
mu_hat, so the points scale changes no prediction; it only makes mu_hat readable as a margin. Neutral sites (the NBA
Cup final) are treated like any home game.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tools import research_testset as R

WHO = "tools.research_winprob_elo"
MEAN = 1500.0
POINTS_PER_ELO = 1.0 / 28.0
GRID_K = (10.0, 15.0, 20.0, 25.0, 30.0)
GRID_HOME = (0.0, 25.0, 50.0, 75.0, 100.0)
GRID_REGRESS = (0.0, 0.25, 0.5, 0.75)
BURN_IN_SEASON = "2020-21"                              # every team starts at MEAN; never a check season
CHECK_SEASONS = ("2021-22", "2022-23", "2023-24")       # forward chaining: each is checked on ratings built before it
COLUMNS = R.ELO_STATE_COLUMNS                           # game_id, date, home, away, home_score, away_score


@dataclass
class Elo:
    k: float
    home: float
    regress: float
    ratings: dict = field(default_factory=dict)
    season: str | None = None
    games: int = 0

    def copy(self) -> "Elo":
        return Elo(self.k, self.home, self.regress, dict(self.ratings), self.season, self.games)

    def params(self) -> dict:
        return {"k": self.k, "home": self.home, "regress": self.regress}


def mov_multiplier(margin: float, winner_edge: float) -> float:
    """The FiveThirtyEight margin of victory multiplier; the denominator is kept at 1 or above."""
    return (abs(margin) + 3.0) ** 0.8 / max(7.5 + 0.006 * winner_edge, 1.0)


def win_chance(diff):
    """The Elo pregame win chance of the home side for an Elo difference that includes the home advantage."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / 400.0))


def schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Played games only, one row per game, a season column (from the game code unless given), sorted by date and
    game id: the order advance() reads them in."""
    s = df[df["home_score"].notna() & df["away_score"].notna()].drop_duplicates("game_id").copy()
    s["game_id"] = s["game_id"].astype(str)
    if "season" not in s.columns:
        s["season"] = [R.pbp_season(g) for g in s["game_id"]]
    s["season"] = s["season"].astype(str)
    s["date"] = s["date"].astype(str)
    return s.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)[list(COLUMNS) + ["season"]]


def advance(elo: Elo, schedule: pd.DataFrame, record: bool = True):
    """Run the ratings forward through a schedule frame (schedule_frame), in place. With record, return one row per
    game with the pregame difference, win chance and mu_hat, read before the game's own result moves the ratings,
    and the game's outcome (for scoring by the caller). Without record, return None (the 2024-25 state run: no
    pregame value of those games is kept, so none can be scored)."""
    out = [] if record else None
    for g in schedule.itertuples(index=False):
        if elo.season is not None and g.season != elo.season:
            for team, r in elo.ratings.items():
                elo.ratings[team] = (1.0 - elo.regress) * r + elo.regress * MEAN
        elo.season = g.season
        rh = elo.ratings.get(g.home, MEAN)
        ra = elo.ratings.get(g.away, MEAN)
        diff = rh + elo.home - ra
        if record:
            out.append((g.game_id, g.season, g.date, diff))
        margin = float(g.home_score) - float(g.away_score)
        won = 1.0 if margin > 0 else 0.0
        edge = diff if margin > 0 else -diff
        delta = elo.k * mov_multiplier(margin, edge) * (won - float(win_chance(diff)))
        elo.ratings[g.home] = rh + delta
        elo.ratings[g.away] = ra - delta
        elo.games += 1
    if not record:
        return None
    pre = pd.DataFrame(out, columns=["game_id", "season", "date", "elo_diff"])
    pre["p_home"] = win_chance(pre["elo_diff"].to_numpy())
    pre["mu_hat"] = pre["elo_diff"] * POINTS_PER_ELO
    pre["home_win"] = (schedule["home_score"].astype(float).to_numpy()
                       > schedule["away_score"].astype(float).to_numpy()).astype(int)
    return pre


def dev_schedule(data_root) -> pd.DataFrame:
    """The played games of the development seasons from the frozen schedule tables of DATA_ROOT, every id checked
    by the door (refuse_test refuses any 2024-25 or 2025-26 game)."""
    from pathlib import Path
    frames = []
    for season in R.PBP_DEV_SEASONS:
        path = Path(data_root) / "collect" / "bref_schedule" / "games" / f"{season}.parquet"
        frames.append(pd.read_parquet(path, columns=list(COLUMNS)))
    s = schedule_frame(pd.concat(frames, ignore_index=True))
    R.refuse_test(s["game_id"], WHO, R.PBP)
    if set(s["season"]) != set(R.PBP_DEV_SEASONS):
        raise SystemExit(f"{WHO}: the development schedule holds {sorted(set(s['season']))}")
    return s


def choose_params(schedule: pd.DataFrame) -> dict:
    """Forward chaining on the development schedule: for every grid point, one Elo run from 2020-21; the pregame win
    chance of each check season's games (ratings built only from earlier games) is scored by Brier, and the grid point
    with the lowest mean over the check seasons wins (first in grid order on a tie). -> the choice and the table."""
    if set(schedule["season"]) - set(R.PBP_DEV_SEASONS):
        raise SystemExit(f"{WHO}: Elo parameters are chosen on the development seasons only")
    rows = []
    for k, home, regress in itertools.product(GRID_K, GRID_HOME, GRID_REGRESS):
        pre = advance(Elo(k, home, regress), schedule)
        per = {}
        for season in CHECK_SEASONS:
            m = (pre["season"] == season).to_numpy()
            per[season] = float(np.mean((pre["p_home"].to_numpy()[m] - pre["home_win"].to_numpy()[m]) ** 2))
        rows.append({"k": k, "home": home, "regress": regress, "brier_by_season": per,
                     "mean_brier": float(np.mean(list(per.values())))})
    best = min(range(len(rows)), key=lambda i: (rows[i]["mean_brier"], i))
    return {"criterion": "mean over the check seasons of the pregame Brier of the Elo win chance",
            "check_seasons": list(CHECK_SEASONS), "burn_in": BURN_IN_SEASON,
            "grid": {"k": list(GRID_K), "home": list(GRID_HOME), "regress": list(GRID_REGRESS)},
            "chosen": {k: rows[best][k] for k in ("k", "home", "regress")}, "chosen_mean_brier": rows[best]["mean_brier"],
            "chosen_brier_by_season": rows[best]["brier_by_season"], "table": rows}


def state_after(params: dict, schedule: pd.DataFrame, before_season: str) -> Elo:
    """The ratings after every development game of a season earlier than before_season (a dev forward-chaining
    fold starts from these)."""
    elo = Elo(params["k"], params["home"], params["regress"])
    advance(elo, schedule[schedule["season"] < before_season], record=False)
    return elo


def state_through_2024_25(params: dict, schedule: pd.DataFrame, data_root=None) -> tuple:
    """D89: the ratings after the development seasons and then every played game of 2024-25, read as state only.
    -> (Elo, a record of counts). Nothing is fitted, tuned or scored on 2024-25: advance runs without record, so no
    pregame value of a 2024-25 game exists to score."""
    elo = Elo(params["k"], params["home"], params["regress"])
    advance(elo, schedule, record=False)
    after_dev = elo.games
    spent = schedule_frame(R.elo_state_games_2024_25(data_root))
    if set(spent["season"]) != {"2024-25"}:
        raise SystemExit(f"{WHO}: the 2024-25 door returned games of {sorted(set(spent['season']))}")
    advance(elo, spent, record=False)
    return elo, {"development_games": after_dev, "games_2024_25_read_as_state": int(len(spent)),
                 "teams": len(elo.ratings), "season_after": elo.season}


def prefix_violations(start: Elo, schedule: pd.DataFrame, n: int = 25, seed: int = 0) -> list:
    """The Elo form of the frozen prefix check: for n random games g, cut the schedule after g and scramble g's own
    final score, rerun from the same start, and compare mu_hat of every game up to g with the full run. -> the games
    where any value changed (empty when no game's own or later result reaches its pregame value)."""
    full = advance(start.copy(), schedule)
    rng = np.random.default_rng(seed)
    out = []
    for i in sorted(set(int(v) for v in rng.integers(0, len(schedule), size=min(n, len(schedule))))):
        cut = schedule.iloc[: i + 1].copy()
        cut.iloc[i, cut.columns.get_loc("home_score")] = float(cut.iloc[i]["home_score"]) + int(rng.integers(1, 40))
        cut.iloc[i, cut.columns.get_loc("away_score")] = float(cut.iloc[i]["away_score"]) + int(rng.integers(41, 80))
        part = advance(start.copy(), cut)
        if not np.allclose(part["mu_hat"].to_numpy(), full["mu_hat"].to_numpy()[: i + 1]):
            out.append(str(schedule.iloc[i]["game_id"]))
    return out
