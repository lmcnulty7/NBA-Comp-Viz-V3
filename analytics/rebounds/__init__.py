"""analytics.rebounds: rebound probability (which of the ten players on court secures a missed field goal, or
nobody, the team rebound), built on the Phase 0 tables of docs/DATA_CONTRACT.md sections 4, 5, 16 to 19.

Modules
    validate   load the DATA_ROOT layout or the synthetic fixtures, schema and range checks, dedupe with counts
    features   one candidate row per player on court at the rim frame plus a "none" row, time-forward features
    train      three candidates (player prior, conditional logit, gradient boosting) under grouped folds by game,
               selection by per-miss multinomial log loss, held-out games scored once, report and predictions
    eda        tables for notebooks/eda_rebounds.ipynb (shares, distributions, correlations, shuffled labels)

Every command takes --fixtures (synthetic games from analytics.fixtures) or --data DATA_ROOT. Fixture numbers are
fixture output, not a result. The leakage rules are in features.py. Fits run under analytics.eval.fit_threads(1)
(four shared cores: one BLAS and OpenMP thread per fit); nothing here touches the environment at import time.
"""
from __future__ import annotations

WHO = "analytics.rebounds"
SEASON_DEFAULT = "2015-16"
