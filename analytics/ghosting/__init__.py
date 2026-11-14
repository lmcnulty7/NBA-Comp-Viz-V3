"""analytics.ghosting: defensive ghosting, the one deep model of the plan (docs/contract/ghosting.md).

Unit: a defender-frame sequence inside a possession, sampled at 5 Hz from the 25 Hz frames. Target: the
defender's x, y at frame t given the five offensive positions, the ball and the other defenders at frames
t minus k for k in 1 to 5; the model never sees the defender's own position at t or anything after t.

Modules
  validate   schema and range checks on the inputs, plus the loaders (fixtures or DATA_ROOT) every module shares
  features   the 5 Hz defender panel, the FEATURES constant, the lagged window and the leakage probe
  train      the three candidates (persistence, ridge, MLP window model), grouped folds by game, bootstrap
             intervals, the keep-or-drop decision, the deviation tables and reports/ghosting.md
  eda        tables for notebooks/eda_ghosting.ipynb (no plotting library in this environment)

Assumptions recorded here because the spec is silent (details in each module docstring):
  - assignment of a defender = the nearest offensive player at the possession's first sampled frame (many
    defenders may share one assignment; no one-to-one matching)
  - the attacking side flag is the half the ball is in at t minus 1 (ball_x > 47); the basket of that
    side is the one the distances are measured to
  - a null shot clock is filled with max(24 minus elapsed, 0)
  - possessions with gap == true or a null offense_team_id are skipped; a possession's lineup is the ten
    players of its first sampled frame and the sequence is cut at the first sampled frame without them
    or without a ball
  - the 1.0 s horizon rolls the ghost forward five sampled steps, feeding its own position back into the
    features while the offence, the ball and the other defenders stay actual
  - the deep candidate is sklearn.neural_network.MLPRegressor over the window of the last five sampled
    feature vectors; a torch GRU reading the same (n, 5, n_features) window is the drop-in once torch is
    installed on the Mac
"""
from __future__ import annotations

__all__ = ["validate", "features", "train", "eda"]
