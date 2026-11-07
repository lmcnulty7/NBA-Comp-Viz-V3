"""tests/_fx_rebounds.py: the test-only adapter of the rebound lane over analytics.fixtures.

The games come from ``analytics.fixtures`` (no local generator): synth_game, synth_season and write_fixture_root
are its functions, re-exported so the tests read one name. Production code never imports this module
(analytics.rebounds.validate reads analytics.fixtures directly). No real tracking data: every number here is
fixture output, never a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.fixtures import synth_game, synth_season, write_fixture_root  # noqa: E402

__all__ = ["synth_game", "synth_season", "write_fixture_root"]
