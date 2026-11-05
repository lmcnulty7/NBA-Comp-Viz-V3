"""analytics.epv: expected possession value from the 1 Hz states table (docs/DATA_CONTRACT.md section 20).

Modules:
  states    builds states_1hz from frames (section 4) and possessions (section 19) when the table is absent
  validate  schema and range checks, missing and duplicate handling, with logged counts
  features  the time-forward feature frame (FEATURES, feature_columns) and the out-of-fold priors
  train     candidates (constant, ridge, gradient boosting with isotonic calibration, gated sequence model),
            grouped folds by game, bootstrap by game, the report
  eda       tables for notebooks/eda_epv.ipynb

Unit: one state at 1 Hz inside a possession. Target: points the offence scores from the state to the end of the
possession (0 to 4), as a regression of expected points with a classification view P(points > 0) for calibration.
Held-out games of sportvu/splits.json never fit or tune anything. The contract of the outputs is docs/contract/epv.md.
"""
from __future__ import annotations

WHO = "train_epv"
SEASON_DEFAULT = "2015-16"
ELAPSED_BUCKETS = ((0.0, 4.0), (4.0, 10.0), (10.0, 24.0))
SHOT_CLOCK_BUCKETS = ((0.0, 7.0), (7.0, 14.0), (14.0, 24.01))


def season_of(sportvu_game: str) -> str:
    """`MM.DD.YYYY.AWAY.at.HOME` -> `YYYY-YY` (a game from September on belongs to the season starting that year)."""
    try:
        mm, _dd, yyyy = sportvu_game.split(".")[:3]
        y, m = int(yyyy), int(mm)
    except (ValueError, AttributeError):
        return SEASON_DEFAULT
    return "%d-%02d" % (y, (y + 1) % 100) if m >= 9 else "%d-%02d" % (y - 1, y % 100)


__all__ = ["WHO", "SEASON_DEFAULT", "ELAPSED_BUCKETS", "SHOT_CLOCK_BUCKETS", "season_of"]
