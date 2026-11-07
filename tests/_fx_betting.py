"""tests/_fx_betting.py: the test-only adapter of the betting lane's fixtures.

The generators are analytics.fixtures' (imported unconditionally; there is no local fallback):

    synth_stints(seed, n_games=20, seasons=("2020-21",), n_players_per_team=9) -> dict(stints, playbyplay, games)
    synth_odds(seed, games) -> section 14 odds rows consistent with games.spread_home plus noise
    team_names(games=None) -> id -> feed name for the fixture's synthetic team ids
                              (analytics.betting.validate.fixture_team_names, the map production uses too)

Production code loads the fixtures through analytics.betting.validate.fixture_data and never imports this file.
Fixture numbers are fixture output, never a result.
"""
from __future__ import annotations

from analytics.betting.validate import FIXTURE_SOURCE as SOURCE
from analytics.betting.validate import fixture_team_names as team_names
from analytics.fixtures import synth_odds, synth_stints

__all__ = ["synth_stints", "synth_odds", "team_names", "SOURCE"]
