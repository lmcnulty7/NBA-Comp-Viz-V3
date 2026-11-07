"""tests/_fx_lineups.py: test-only fixture adapter of the lineups lane (tests/test_lineups.py).

synth_stints is the shared generator (analytics.fixtures.synth_stints); synth_stints_local is the lane's own
synthetic generator (analytics/lineups/synthetic.py: a starters table and recoverable player ratings), which the
tests exercise on purpose. Fixture numbers are never results.
"""
from __future__ import annotations

from analytics.fixtures import synth_stints  # noqa: F401  (the shared generator)
from analytics.lineups.synthetic import synth_stints_local  # noqa: F401  (the lane's synthetic generator)

__all__ = ["synth_stints", "synth_stints_local"]
