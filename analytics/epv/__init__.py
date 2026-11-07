"""analytics.epv: expected possession value from the 1 Hz states table (docs/DATA_CONTRACT.md section 20).

Modules:
  states    builds states_1hz from frames (section 4) and possessions (section 19) when the table is absent
  validate  schema and range checks, missing and duplicate handling, with logged counts, and the input loaders
  features  the time-forward feature frame (FEATURES, feature_columns) and the out-of-fold priors
  train     candidates (constant, ridge, gradient boosting with isotonic calibration, gated sequence model),
            grouped folds by game, bootstrap by game, the report
  eda       tables for notebooks/eda_epv.ipynb

Unit: one state at 1 Hz inside a possession. Target: points the offence scores from the state to the end of the
possession (0 to 4), as a regression of expected points with a classification view P(points > 0) for calibration.
Held-out games of sportvu/splits.json never fit or tune anything. The contract of the outputs is docs/contract/epv.md.
Every fit runs inside analytics.eval.fit_threads() (BLAS and OpenMP pinned to one thread); the season directory of a
game is analytics.schema.season_of_game.
"""
from __future__ import annotations

WHO = "train_epv"
ELAPSED_BUCKETS = ((0.0, 4.0), (4.0, 10.0), (10.0, 24.0))
SHOT_CLOCK_BUCKETS = ((0.0, 7.0), (7.0, 14.0), (14.0, 24.01))

__all__ = ["WHO", "ELAPSED_BUCKETS", "SHOT_CLOCK_BUCKETS"]
