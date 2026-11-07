"""analytics.gravity: gravity and spacing from SportVU frames inside possessions (lane GRAVITY, Phase 2).

Unit: the player-frame, sampled at 1 Hz inside possessions (the frame nearest each whole second of
elapsed time since the possession's start frame). Target: the distance in feet from the offensive
player to the nearest defender. A location-only ridge regression (the plan's model) gives the expected
distance; gravity is what the defence does beyond that expectation. A histogram gradient boosting
regressor on the same features is reported as a diagnostic only: gravity is a residual, and a flexible
model lowers MSE partly by fitting what should stay in the residual, so MSE is not the selection
criterion. The shooting prior is the league make rate by zone of the player's position, never a
per-player rate, for the same reason (a per-player prior identifies the player and lets the
expectation absorb the pull).

Sign convention (an assumption recorded here because the spec text states both "y minus the
expectation" and "positive means defenders stay closer than location predicts", which conflict):
    gravity_ft = expected_ft - nearest_def_dist_ft
so a positive gravity means the nearest defender stays closer than the player's location alone
predicts. The spec's stated meaning of the sign wins over its formula.

Modules:
    validate   schema and range checks of the input tables, missing and duplicate handling
    features   the 1 Hz player-frame feature frame, the spacing frame, the out-of-fold shooting prior
    train      candidates, selection, gravity tables, validations and the report
    eda        tables for notebooks/eda_gravity.ipynb

Every module exposes main(argv=None) where the layout asks for a CLI (validate, features, train).
Tables written are described in docs/contract/gravity.md.
"""
from __future__ import annotations

WHO = "gravity"
SEASON = "2015-16"
MIN_POSSESSIONS = 300

__all__ = ["WHO", "SEASON", "MIN_POSSESSIONS"]
