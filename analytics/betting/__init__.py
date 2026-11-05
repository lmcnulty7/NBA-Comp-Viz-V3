"""analytics.betting: the betting market model (docs/contract/betting.md; the plan ranks it last).

Unit: one NBA game. Targets: home_cover (the home side covers the closing spread, 1/0, null on a push)
and the closing-line value clv = model margin forecast minus the market margin forecast (the market
margin forecast is minus the closing home spread). Features use previous games only (analytics.betting.features).

Modules
    validate  odds schema checks, one closing line per game and market, vig removal, join to the games table
    features  FEATURES, feature_columns(), build_games() -> betting_games frame, leak guards
    train     candidates, grouped CV inside training seasons, the staking rule, the held-out season scored once
    eda       tables for notebooks/eda_betting.ipynb (spread and margin distributions, cover rate by spread bin,
              join coverage, shuffled-label check)

Each of validate, features and train exposes main(argv=None) with --fixtures | --data DATA_ROOT [--out OUT_ROOT].

Assumptions recorded here because nobody could be asked
  - Odds history is a paid feed (collect/extras.py --odds, The Odds API) and has not been collected; every
    number this package has produced so far is fixture output, not a result.
  - The games table is DATA_ROOT/collect/stints/games/<season>.parquet when present; otherwise it is rebuilt
    from the final score of the multi-season play-by-play (section 13) and the team ids of its events.
  - "Net rating" is point margin per game over the previous 10 and 30 games (the games table carries no
    possession count); a team with no previous game in the season gets 0 (league average) and its first rest
    value is the cap of 7 days.
  - The closing line is the is_close snapshot nearest tip; when a game has no is_close snapshot the latest
    pre-tip snapshot stands in and is counted as a fallback. Books are combined by the median point and the
    mean de-vigged probability.
  - The held-out unit is a whole season (default 2024-25; when absent, the latest season present when there
    are at least two). It is scored once per run with the staking rule fixed in train.py before any scoring.
  - No raw odds rows are ever written into the repository; outputs go under OUT_ROOT/analytics/betting/<season>/.
"""
