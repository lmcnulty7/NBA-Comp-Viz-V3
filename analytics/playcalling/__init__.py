"""analytics.playcalling: optimal play calling reported as associations, never as causal claims.

Unit of analysis: one action event (a labelled or detected offensive action such as a pick and roll)
inside a possession of docs/DATA_CONTRACT.md section 19. Outcomes: the points per possession (PPP)
of the containing possession and, when an EPV lane output exists, the EPV gained across the action.

Modules
    validate   schema, enum, range, missing and duplicate checks on every input; data root loader
    features   the action-level table (docs/contract/playcalling.md, action_outcomes) and FEATURES
    train      PPP by action type, by action type x coverage, the ridge adjustment and the
               recommendation table, all with bootstrap intervals by game and the 300-possession gate
    eda        tables for notebooks/eda_playcalling.ipynb

Every output of this package is an observational association with an interval and a sample gate.
Nothing here identifies the effect of calling a play: the defence, the lineup and the game state
choose the play as much as the coach does. The report says so in its first line.

Assumptions recorded here because the upstream lanes (ACTIONS, COVERAGE, EPV) were built in parallel:
  - ACTION_TYPES is the ACTIONS lane vocabulary (analytics/actions, also analytics/fixtures) and
    COVERAGES the COVERAGE lane classes (analytics/coverage CLASSES) plus "none". An unknown value in
    an input is counted and kept as its own category, never dropped and never refused, so a
    different upstream vocabulary flows through with a logged count.
  - Actions without a screen label (or whose screen row is missing) get coverage "none".
  - An actions table without an action_id column gets one: the 0-based position of the row within
    its game (jsonl line order, parquet row order), the convention of analytics/coverage and of the
    shared fixtures. The screens and coverage tables join on (sportvu_game, possession, action_id).
  - Hand labels (labels/actions, labels/screens, read with analytics.labels.read_labels at
    labeller_pass 1 so tombstones and pass-2 relabels never reach training) win over detected rows
    for a game that has them; detected rows (label_source "model") fill the other games and get no
    pass filter. label_source records which one was used.
  - The score margin enters the ridge as a binned indicator group (trailing by more than 5, within
    5, leading by more than 5), so it is a control in the adjustment, not an outcome.
"""
from __future__ import annotations

ACTION_TYPES = ("ball_screen", "handoff", "off_ball_screen", "isolation", "post_up", "drive", "cut")
COVERAGES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over", "none")
try:
    from analytics.labels import LABEL_SOURCES          # ("hand", "model", "synthetic")
except ImportError:                                      # the labels module is another lane's file
    LABEL_SOURCES = ("hand", "model", "synthetic")
SEASON = "2015-16"
N_MIN = 300            # the possession gate of analytics.eval.gated
WHO = "playcalling"

__all__ = ["ACTION_TYPES", "COVERAGES", "LABEL_SOURCES", "SEASON", "N_MIN", "WHO"]
