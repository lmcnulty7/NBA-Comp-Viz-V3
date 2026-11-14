"""tests/_fx_epv.py: test-only adapter between analytics.fixtures and tests/test_epv.py.

Production code never imports this module: the EPV CLIs build their fixture season through
analytics.epv.validate.fixture_season, which reads analytics.fixtures directly. This file only gives the tests one
game in a flat shape and a DATA_ROOT on disk. Everything is synthetic and deterministic given the seed; nothing is
read from raw data and nothing is written into the repo.

    game(seed, n_possessions)                                   one analytics.fixtures.synth_game with sportvu_game and
                                                                heldout at the top level
    write_root(root, seed, n_games, heldout_every, n_possessions) analytics.fixtures.write_fixture_root, returning
                                                                [(sportvu_game, heldout), ...] in generation order

The season of analytics.fixtures.synth_season is one league (synth_league): teams and their 7-man rosters recur
across games, which the out-of-fold priors test relies on.
"""
from __future__ import annotations

from analytics import fixtures

HELD = fixtures.HELDOUT_GAME
SEASON = fixtures.SEASON_2015


def game(seed: int = 1, n_possessions: int = 20) -> dict:
    """One synthetic game (analytics.fixtures.synth_game) with sportvu_game and heldout copied out of meta."""
    g = fixtures.synth_game(seed, n_possessions=n_possessions)
    return {**g, "sportvu_game": str(g["meta"]["sportvu_game"]), "heldout": bool(g["meta"]["heldout"])}


def write_root(root, seed: int, n_games: int, heldout_every: int, n_possessions: int = 16) -> list:
    """The DATA_ROOT layout the EPV CLIs read, written by analytics.fixtures.write_fixture_root (small stint family)
    -> [(sportvu_game, heldout), ...]."""
    man = fixtures.write_fixture_root(root, seed=seed, n_games=n_games, heldout_every=heldout_every,
                                      n_possessions=n_possessions, n_stint_games=2)
    held = set(man["heldout"])
    return [(g, g in held) for g in man["games"]]


__all__ = ["HELD", "SEASON", "game", "write_root"]
