"""tests/_fx_embeddings.py: the test-only fixture adapter of the player style embeddings lane (analytics/embeddings).

The lane runs on the shared generator ``analytics.fixtures``: write_fixture_root builds one league whose rosters
persist across games, and its aggregates and positions are keyed by the same players (archetype from the roster
slot), so the 5-NN position accuracy and the next-season R squared run on it. This module re-exports what the tests
use; it carries no generator of its own. Every number computed from these is fixture output, not a result.
"""
from __future__ import annotations

from analytics.fixtures import HELDOUT_GAME as HELDOUT_GAME_ID  # noqa: F401
from analytics.fixtures import synth_aggregates, synth_game, synth_season, write_fixture_root  # noqa: F401

__all__ = ["synth_game", "synth_season", "synth_aggregates", "write_fixture_root", "HELDOUT_GAME_ID"]
