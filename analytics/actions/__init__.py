"""analytics.actions: action recognition from SportVU frames, starting with ball-screen detection.

Unit: a candidate window inside a possession (docs/contract/actions.md). The pipeline is

    validate.py   schema, enum, range, duplicate and overlap checks on the inputs and the hand labels
    features.py   rule-based candidate generation at 5 Hz, label matching at IoU over frames >= 0.5,
                  per-candidate features (FEATURES) and the leakage guard
    train.py      rule baseline and a HistGradientBoostingClassifier over {ball_screen, drive, isolation,
                  none} under grouped folds by sportvu_game, bootstrap intervals by game, the held-out
                  games scored once, reports/actions.md and the action_candidates / actions_detected tables
    eda.py        candidate counts, class balance, feature distributions by class, a shuffled-label check

Hand-label schema (lane I4 writes the tool; this package codes against it, DATA_ROOT/labels/actions/<game>.jsonl):
    actions  sportvu_game, possession, action_type, start_frame, end_frame, ball_handler_id, screener_id
             (nullable), label_source, labeller_pass [, action_id]
    screens  sportvu_game, possession, action_id, coverage, on_ball_defender_id, screener_defender_id,
             label_source, confidence

Assumptions recorded here because nobody could be asked:
  - hand labels are read only through analytics.labels.read_labels(DATA_ROOT, game, labeller_pass=1): the
    last line per id wins, tombstoned ids are dropped, pass-2 relabels never reach a fit; the vocabularies
    (action_type, coverage, label_source) are analytics.labels', and a coverage outside them is a warning;
    screens.confidence may be a text grade (sure, unsure) or a number in [0, 1], never null
  - only ball_screen, drive and isolation are detected; a candidate overlapping a label of another type
    (handoff, off_ball_screen, post_up, cut) counts as "none" and the count is reported
  - frames are read at 5 Hz (every round(hz / 5)-th frame of the possession); hand labels and candidate
    frames stay in the 25 Hz frame ids of the contract
"""
from __future__ import annotations

try:   # the label contract of lane I4 (docs/contract/labels.md) owns these vocabularies
    from analytics.labels import ACTION_TYPES, COVERAGES, LABEL_SOURCES
except ImportError:   # pragma: no cover - the schema as the lead stated it
    ACTION_TYPES = ("ball_screen", "handoff", "off_ball_screen", "isolation", "post_up", "drive", "cut")
    LABEL_SOURCES = ("hand", "model", "synthetic")
    COVERAGES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")
CLASSES = ("ball_screen", "drive", "isolation", "none")      # the detector's classes, in output order
SEASON = "2015-16"
WHO = "actions.train"
IOU_MIN = 0.5

__all__ = ["ACTION_TYPES", "LABEL_SOURCES", "COVERAGES", "CLASSES", "SEASON", "WHO", "IOU_MIN"]
