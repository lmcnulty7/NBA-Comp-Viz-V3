"""analytics.coverage: pick-and-roll defensive coverage recognition (unit: one ball-screen event; target: the
coverage class, 7 classes).

Modules
    validate   label schema, enum, reference and range checks on the hand-label tables and the detector's table
    features   the per-screen feature frame (positions at the screen frame and over the post-screen window)
    train      majority, multinomial logistic and gradient-boosting candidates under grouped folds by game
    eda        class balance, feature distributions by class, correlation table, shuffled-label check

No hand labels exist yet (STATUS.md): every number this package has produced so far is fixture output, not
a result. See docs/contract/coverage.md for the tables it consumes and writes.
"""
from __future__ import annotations

CLASSES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")
CONFIDENCES = ("sure", "unsure")
LABEL_SOURCES = ("hand", "model", "synthetic")   # as analytics/labels.py writes them
TRAIN_PASS = 1                                       # only first-pass labels fit; pass 2 is the consistency relabel
SCREEN_ACTION = "ball_screen"
WHO = "analytics.coverage.train"

__all__ = ["CLASSES", "CONFIDENCES", "LABEL_SOURCES", "SCREEN_ACTION", "TRAIN_PASS", "WHO"]
