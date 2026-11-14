"""analytics.matchups: individual matchup assessment from SportVU frames (lane MATCHUPS).

Unit: defender x offensive player partial possession. The pipeline is

    validate   schema and range checks on the input tables (python -m analytics.matchups.validate)
    features   1 Hz assignment of defenders to offensive players by linear_sum_assignment, the
               matchup_possessions table (python -m analytics.matchups.features)
    train      defender metrics with a 300-possession gate, a ridge adjustment with alpha chosen by
               grouped CV by game, internal validation, one held-out game scored once
               (python -m analytics.matchups.train)
    eda        tables for notebooks/eda_matchups.ipynb

Tables are documented in docs/contract/matchups.md. Assumptions taken without asking are recorded in
each module docstring and in reports/matchups.md.
"""
from __future__ import annotations

SEASON = "2015-16"
XFG_SUBDIR = ("analytics", "xfg")
MATCHUP_SUBDIR = ("analytics", "matchups")

__all__ = ["SEASON", "XFG_SUBDIR", "MATCHUP_SUBDIR"]
