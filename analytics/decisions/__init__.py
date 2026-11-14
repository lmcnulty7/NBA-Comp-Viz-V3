"""analytics.decisions: decision quality of the ball handler (docs/contract/decisions.md).

Unit: a decision event, which is a shot, a pass or a drive start, found in the SportVU frames of a possession.
Target: the value of the chosen option minus the best available option, in expected points, where an option is
the handler's own shot, a pass to each teammate or (when the EPV lane's states are present) holding the ball.

Modules
    validate   load the inputs (fixtures or DATA_ROOT), schema.validate per table, range checks, duplicates
    features   event detection from the ball path, the decision contexts, option values, the leakage guards
    train      the value model (location-only xFG fallback, or the xFG lane's p_selected), decision events,
               the player table and the validation against on-court offensive rating; writes the report
    eda        tables for notebooks/eda_decisions.ipynb

No deep model: the value function is a logistic regression on four location features (analytics.features_xfg),
fit under grouped folds by game, with held-out games scored once and never fitted. Every number computed on the
synthetic fixtures is fixture output, not a result.
"""
__all__ = ["validate", "features", "train", "eda"]
