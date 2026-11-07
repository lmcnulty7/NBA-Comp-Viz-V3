"""tests/_fx_ghosting.py: the test-only fixture adapter of the ghosting lane (analytics/ghosting).

The lane runs on the shared generator ``analytics.fixtures`` (analytics.ghosting.validate.fixture_games wraps its
synth_season into the lane's game dicts). This module re-exports the generator for tests that want the raw dicts;
it carries no generator of its own. Every number computed from these is fixture output, not a result.
"""
from __future__ import annotations

from analytics.fixtures import synth_game, synth_season, write_fixture_root  # noqa: F401

__all__ = ["synth_game", "synth_season", "write_fixture_root"]
