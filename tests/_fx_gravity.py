"""tests/_fx_gravity.py: the test-only fixture adapter of the gravity lane (analytics/gravity).

``synth_game`` and ``synth_season`` are the shared generator ``analytics.fixtures``. The lane's planted-pull
generator lives in ``analytics/gravity/fixture_pull.py`` (synthetic, documented there) and is re-exported here
under its own names, so the tests that check the recovered pull run on it explicitly:

    synth_pull_game, synth_pull_season, effective_pull, HG_PULL_FT, HG_HOME, HG_AWAY

Nothing here writes to the repo. Every number a test computes from these is fixture output, not a result.
"""
from __future__ import annotations

from analytics.fixtures import synth_game, synth_season  # noqa: F401
from analytics.gravity.fixture_pull import (HG_AWAY, HG_HOME, HG_PULL_FT, effective_pull,  # noqa: F401
                                            synth_pull_game, synth_pull_season)

__all__ = ["synth_game", "synth_season", "synth_pull_game", "synth_pull_season", "effective_pull", "HG_PULL_FT",
           "HG_HOME", "HG_AWAY"]
