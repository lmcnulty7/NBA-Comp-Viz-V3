"""analytics.lineups: lineup synergy and stints (Phase 1 lane LINEUPS).

Modules
    stints     section 12 stints built from section 13 play-by-play substitutions (NBA.com ids or
               basketball-reference descriptions) plus the starters table when present
    validate   schema and range checks of a stints table, the lineup_ok share, dropped and duplicate counts
    features   the stint feature frame: ten player id columns, possessions, the differential per 100 target
    train      the three candidates (raw plus-minus, RAPM, RAPM plus gated lineup terms), the out-of-season
               selection and the held-out season scored once, the report and the output tables
    eda        tables for notebooks/eda_lineups.ipynb
    synthetic  the lane's own synthetic stints generator (--fixture-generator local); fixture output only

Inputs (DATA_ROOT layout on the owner's Mac)
    DATA_ROOT/collect/stints/stints/<season>.parquet        section 12
    DATA_ROOT/collect/stints/playbyplay/<season>.parquet    section 13
    DATA_ROOT/collect/stints/starters/<season>.parquet      game_id, team_id, player_id (optional)
    DATA_ROOT/collect/stints/games/<season>.parquet         game_id, season, game_date, home_team_id,
                                                            away_team_id, ... (optional; also looked for at
                                                            DATA_ROOT/games/<season>.parquet)
    DATA_ROOT/collect/stints/heldout/...                    never read by anything here
Outputs: OUT_ROOT/analytics/lineups/<season>/{stints,lineup_ratings,player_rapm}.parquet
(docs/contract/lineups.md).

Assumptions recorded here and in reports/lineups.md
    - Possessions: the stints table carries no possession count, so a stint's possessions are
      duration_s / 28.8 (100 possessions per team per 48 minutes, PACE_POSS_PER_48) unless the table
      carries a `possessions` column, which is then used as is.
    - Target: (points_home - points_away) / possessions * 100 per stint, home minus away.
    - Seasons are the unit of the split: a season is held out whole, no row of a season later than the
      target season ever reaches a fit, and the 2015-16 rows under heldout/ are never read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PACE_POSS_PER_48 = 100.0
SECONDS_PER_POSS = 48.0 * 60.0 / PACE_POSS_PER_48      # 28.8 s
SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
FIXTURE_SEASONS = ("2020-21", "2021-22", "2022-23")
FIXTURE_GAMES = 30
WHO = "analytics.lineups"
PLAYER_COLUMNS = tuple(f"home_{k}" for k in range(1, 6)) + tuple(f"away_{k}" for k in range(1, 6))


def lineup_id(ids) -> str:
    """Sorted integer ids joined with |."""
    return "|".join(str(i) for i in sorted(int(x) for x in ids))


def stints_dir(data_root) -> Path:
    """DATA_ROOT/collect/stints."""
    return Path(data_root) / "collect" / "stints"


def read_season_tables(data_root, kind: str, seasons=SEASONS) -> pd.DataFrame:
    """Concatenate DATA_ROOT/collect/stints/<kind>/<season>.parquet for the seasons present (never heldout/)."""
    frames = []
    for season in seasons:
        candidates = [stints_dir(data_root) / kind / f"{season}.parquet"]
        if kind == "games":
            candidates.append(Path(data_root) / "games" / f"{season}.parquet")
        for p in candidates:
            if "heldout" in p.parts:
                raise SystemExit(f"{WHO}: refusing to read {p} (heldout/)")
            if p.exists():
                df = pd.read_parquet(p)
                if "season" not in df.columns:
                    df["season"] = season
                frames.append(df)
                break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_FIXTURE_CACHE: dict = {}


GENERATORS = ("auto", "shared", "local")


def fixture_tables(seed: int = 0, n_games: int = FIXTURE_GAMES, seasons=FIXTURE_SEASONS, generator: str = "auto") -> dict:
    """The synthetic tables. generator: "shared" (or "auto") is analytics.fixtures.synth_stints, called without a
    players frame so its team ids match the betting lane's fixture; "local" is the lane's own synthetic generator
    (analytics/lineups/synthetic.py: a starters table and recoverable player ratings). Fixture numbers are never results."""
    if generator not in GENERATORS:
        raise ValueError(f"{WHO}: unknown fixture generator {generator!r} (one of {GENERATORS})")
    if generator == "local":
        from analytics.lineups.synthetic import synth_stints_local as fn
    else:
        from analytics.fixtures import synth_stints as fn
    key = (fn.__module__, fn.__name__, int(seed), int(n_games), tuple(seasons))
    if key not in _FIXTURE_CACHE:                      # memoised per process: callers never modify the tables in place
        t = fn(seed, n_games=n_games, seasons=tuple(seasons))
        t["generator"] = "%s.%s" % (fn.__module__, fn.__name__)
        _FIXTURE_CACHE[key] = t
    return _FIXTURE_CACHE[key]


def add_fixture_args(ap) -> None:
    """The fixture knobs of every lineups CLI: --fixture-games (alias --n-games) and --fixture-generator. The lane splits
    by season (--heldout-season), so it has no possessions or held-out-every knob."""
    ap.add_argument("--fixture-games", "--n-games", dest="n_games", type=int, default=None,
                    help="--fixtures only: games per fixture season (default %d)" % FIXTURE_GAMES)
    ap.add_argument("--fixture-generator", choices=GENERATORS, default="auto",
                    help="--fixtures only: analytics.fixtures.synth_stints (shared, or auto), or the lane's synthetic "
                         "generator analytics/lineups/synthetic.py (local)")


def load_inputs(fixtures: bool, data_root=None, seasons=SEASONS, seed: int = 0, generator: str = "auto", n_games=None) -> dict:
    """dict(stints, playbyplay, games, starters, fixture, generator): the input tables of a run."""
    if fixtures:
        t = fixture_tables(seed, n_games=n_games or FIXTURE_GAMES, generator=generator)
        return {"stints": t["stints"], "playbyplay": t["playbyplay"], "games": t.get("games", pd.DataFrame()),
                "starters": t.get("starters", pd.DataFrame()), "fixture": True, "generator": t["generator"]}
    if data_root is None:
        raise SystemExit(f"{WHO}: pass --fixtures or --data DATA_ROOT")
    return {"stints": read_season_tables(data_root, "stints", seasons),
            "playbyplay": read_season_tables(data_root, "playbyplay", seasons),
            "games": read_season_tables(data_root, "games", seasons),
            "starters": read_season_tables(data_root, "starters", seasons), "fixture": False, "generator": None}


def out_dir(out_root, season: str) -> Path:
    """OUT_ROOT/analytics/lineups/<season>, created."""
    p = Path(out_root) / "analytics" / "lineups" / season
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = ["PACE_POSS_PER_48", "SECONDS_PER_POSS", "SEASONS", "FIXTURE_SEASONS", "PLAYER_COLUMNS", "WHO",
           "lineup_id", "read_season_tables", "fixture_tables", "add_fixture_args", "load_inputs", "out_dir"]
