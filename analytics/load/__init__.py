"""analytics/load: player load and fatigue from the SportVU frames (docs/contract/load.md).

Units: the player-stint (a continuous on-court span of one player in the section 4 frames) and the
player-game. Modules:

    validate   input loaders (fixtures or DATA_ROOT), schema and range checks on the frames, logged counts
    features   player_stints, player_games and load_summary tables; FEATURES and the leakage guards
    eda        descriptive tables for notebooks/eda_load.ipynb (tables and markdown, no plots)
    train      three baselines (rest-bucket means, ridge on y_decline, logistic on absence) and the report

Rules: held-out games never fit or tune anything (analytics.eval.refuse_heldout on the training ids);
schema.assert_time_forward on every feature frame; baselines only; fixture numbers are never results.
"""
