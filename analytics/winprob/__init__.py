"""analytics.winprob: live win probability from multi-season play-by-play (docs/contract/winprob.md).

Unit: the game state at each play-by-play event. Target: the home team wins (1/0). Modules:

    validate   schema and sanity checks of the play-by-play, odds and games tables
    features   the time-forward state frame (FEATURES, feature_columns, build_states)
    eda        tables for notebooks/eda_winprob.ipynb
    train      the candidates, grouped-fold selection by time bucket, the held-out season scored once

Every module exposes main(argv=None) and runs as python -m analytics.winprob.<module>.
"""
