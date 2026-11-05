# Live win probability tables (analytics/winprob)

Unit: the game state at each play-by-play event of the multi-season play-by-play
(docs/DATA_CONTRACT.md section 13). Target: the home team wins the game (1/0). Written by
`python -m analytics.winprob.train --data DATA_ROOT --out OUT_ROOT`, one folder per season:

    OUT_ROOT/analytics/winprob/<season>/winprob_states.parquet
    OUT_ROOT/analytics/winprob/<season>/winprob_calibration.parquet

`python -m analytics.winprob.features --data DATA_ROOT --out OUT_ROOT` writes the full feature frame beside them as
`OUT_ROOT/analytics/winprob/<season>/winprob_features.parquet` (the keys, the FEATURES columns, time_bucket, heldout
and home_win; never the final score), which the trainer does not overwrite.

CLI (each module also exposes main(argv=None); --fixtures runs on tests/_fx_winprob.py and its output is never a result):

    python -m analytics.winprob.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
    python -m analytics.winprob.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
    python -m analytics.winprob.eda      --fixtures | --data DATA_ROOT [--seed 0] [--heldout-season 2024-25]
    python -m analytics.winprob.train    --fixtures | --data DATA_ROOT --report reports/winprob.md [--out OUT_ROOT]
                                         [--seed 0] [--bootstrap 200] [--folds 5] [--heldout-season 2024-25]

A held-out season (default 2024-25) is written like the others with `heldout = true` and `fold = -1`; its rows
were scored once by the selected candidate refit on every training row. Seasons later than the held-out season are
ignored entirely (never fitted, scored or written) and listed under Inputs in reports/winprob.md. The 2015-16 games of
sportvu/splits.json never fit anything and never appear here.

## winprob_states

One row per play-by-play event, sorted by (game_id, period, event). Every feature is a function of the
events of the same game up to and including the row and of the pre-game odds; the final score never enters.

| column | type | meaning |
| --- | --- | --- |
| `game_id` | string | NBA game id |
| `season` | string | `YYYY-YY` |
| `event` | int32 | the play-by-play event index inside the game |
| `period` | int8 | period of the event (5 and above are overtime) |
| `seconds_remaining` | float32 | seconds left in regulation after periods 1 to 4 ((4 - period) * 720 + clock_s); in overtime the seconds left in the current overtime period |
| `margin` | int16 | home score minus away score after the event (running score forward-filled, 0 before the first scoring event) |
| `possession` | int8, nullable | 1 when the home team has the ball after the event, 0 when the away team has it, null when the events so far do not say |
| `spread` | float32, nullable | the closing spread of the home side from the odds table (section 14) through analytics.betting.validate (per-book latest pre-tip is_close snapshot, median over books, joined on both team keys and the US local tip date), negative when the home team is favoured; null when the game has none; the games table's spread_home only when there is no odds table |
| `p_home_win` | float32 | the selected candidate's probability that the home team wins, out-of-fold for training seasons, once-scored for the held-out season |
| `home_win` | bool | the target: the home team won |
| `fold` | int8 | the grouped fold (by game) whose test side the row was predicted on; -1 for the held-out season |
| `heldout` | bool | true for the held-out season |

Features the models see beyond these columns (analytics.winprob.features.FEATURES): `overtime` (period above
4), `spread_missing` (spread null), `margin_x_sqrt_time` (margin times the square root of seconds_remaining),
plus a `possession_missing` flag in the design matrix. They are recomputed from the columns above.

## winprob_calibration

One row per (season, time bucket) of the selected candidate's predictions.

| column | type | meaning |
| --- | --- | --- |
| `season` | string | `YYYY-YY` |
| `time_bucket` | string | minutes remaining in regulation: `48-36`, `36-24`, `24-12`, `12-4`, `4-0`, or `OT` |
| `n` | int64 | events in the bucket |
| `mean_p` | float64 | mean predicted home win probability |
| `mean_y` | float64 | observed home win rate |
| `brier` | float64 | Brier score in the bucket |
| `slope` | float64 | calibration slope (logistic regression of home_win on logit p; 1 is calibrated), NaN when the bucket holds one class |

## Protocol (analytics/winprob/train.py)

Candidates, baselines first and no deep model: constant (training-fold home win rate); logistic regression on
margin, seconds_remaining, possession and margin_x_sqrt_time; HistGradientBoostingClassifier on the same
plus period, overtime, spread and spread_missing, calibrated by isotonic regression on inner out-of-fold
predictions grouped by game. Selection: the unweighted mean over the five regulation buckets of the
out-of-fold Brier score under grouped folds by game within the training seasons. ECE and calibration slope per
bucket carry bootstrap intervals by game; log loss overall. The held-out season is scored once by the selected
candidate. Guards: analytics.eval.refuse_heldout on the training game ids, analytics.schema.assert_time_forward on
the design matrix, a name allowlist (analytics.winprob.features.assert_no_leak) and a value check
(prefix_invariance_violations: a feature at event k is unchanged when the play-by-play is cut after k and the games
row's finals and outcome are scrambled). The games table's home_score, away_score and home_win are dropped before any
feature is built; the target is mapped onto the states afterwards.

Benchmark: the NBA.com WinProbabilityPBP endpoint is unreachable, so the benchmark column of reports/winprob.md
stays empty until it exists; it would join winprob_states on (game_id, event).
