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
  - ACTION_TYPES and COVERAGES below are the vocabularies this lane codes against. An unknown value
    in an input is counted and kept as its own category, never dropped and never refused, so a
    different upstream vocabulary flows through with a logged count.
  - Actions without a screen label (or whose screen row is missing) get coverage "none".
  - An actions table without an action_id column gets one: the 0-based rank of the action by
    start_frame inside its possession. The screens and coverage tables join on
    (sportvu_game, possession, action_id).
  - Hand labels (labels/actions, labels/screens) win over detected or predicted rows when both
    exist for the same key; label_source records which one was used.
"""
from __future__ import annotations

ACTION_TYPES = ("pick_and_roll", "isolation", "post_up", "handoff", "off_ball_screen", "spot_up", "transition", "cut")
COVERAGES = ("drop", "switch", "hedge", "blitz", "ice", "under", "none")
LABEL_SOURCES = ("hand", "detected", "predicted")
SEASON = "2015-16"
N_MIN = 300            # the possession gate of analytics.eval.gated
WHO = "playcalling"

__all__ = ["ACTION_TYPES", "COVERAGES", "LABEL_SOURCES", "SEASON", "N_MIN", "WHO"]
