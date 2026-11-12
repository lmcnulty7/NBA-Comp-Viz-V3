# Live win probability tables (analytics/winprob)

Unit: the game state at each play-by-play event of the multi-season play-by-play
(docs/DATA_CONTRACT.md section 13). Target: the home team wins the game (1/0). Written by
`python -m analytics.winprob.train --data DATA_ROOT --out OUT_ROOT` (OUT_ROOT defaults to DATA_ROOT on real data,
as run_all passes no --out; fixture runs without --out write no tables), one folder per season:

    OUT_ROOT/analytics/winprob/<season>/winprob_states.parquet
    OUT_ROOT/analytics/winprob/<season>/winprob_calibration.parquet

`python -m analytics.winprob.features --data DATA_ROOT --out OUT_ROOT` writes the full feature frame beside them as
`OUT_ROOT/analytics/winprob/<season>/winprob_features.parquet` (the keys, the FEATURES columns, time_bucket, heldout
and home_win; never the final score), which the trainer does not overwrite.

CLI (each module also exposes main(argv=None); --fixtures runs on the synth_stints and synth_odds fixtures of
analytics.fixtures, sized by --fixture-games and --fixture-seasons, and its output is never a result):

    python -m analytics.winprob.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
    python -m analytics.winprob.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--seed 0]
    python -m analytics.winprob.eda      --fixtures | --data DATA_ROOT [--seed 0] [--heldout-season 2024-25]
    python -m analytics.winprob.train    --fixtures | --data DATA_ROOT --report reports/winprob.md [--out OUT_ROOT]
                                         [--seed 0] [--bootstrap 200] [--folds 5] [--heldout-season 2024-25]
    (every --fixtures run also takes [--fixture-games 20] [--fixture-seasons 2020-21,2021-22]; train defaults to
    the seasons 2020-21,2021-22,2024-25 so the held-out season is present)

A held-out season (default 2024-25) is written like the others with `heldout = true` and `fold = -1`; its rows
were scored once by the selected candidate refit on every training row. Seasons later than the held-out season are
ignored entirely (never fitted, scored or written) and listed under Inputs in reports/winprob.md. The 2015-16 games of
sportvu/splits.json never fit anything and never appear here.

## Clock rule

`analytics.winprob.features.clock_rule` decides which play-by-play events reach a feature. validate, features
(build_states) and train all apply it, so no bad clock reaches a feature:

1. A game is dropped whole when its period numbers skip a value (for example 1, 2, 4, 5, 6). The
   basketball-reference parser (`fetch_pbp.parse_events`) numbers periods by counting quarter header rows, and a
   page repeats a header (3rd Q, 2nd Q, 3rd Q) around an event filed under another quarter, so every later period
   is shifted up by two and regulation quarters land in periods 5 and 6.
2. A game is dropped whole when a period above 4 holds a clock above 300 s (a regulation quarter labelled
   overtime).
3. Any remaining event is dropped when its period or clock is null, its period is below 1, or its clock is outside
   [0, 720] in periods 1 to 4 or [0, 300] in overtime (the page sometimes prints "24:00.0").

validate writes the counts to validate.json under `counts.clock_rule` (rows and games per cause and per season;
`counts.clock_out_of_range` is the number of out-of-range events before the rule). The check fails the run only when
a season loses more than 2% of its events or games. The trainer's report lists the drops under Inputs. On the five
real seasons (2020-21 to 2024-25) rule 1 drops 53 of 6,416 games; every one of the 4,388 out-of-range events is in
one of them.

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
`margin_over_sqrt_time` (margin over the square root of seconds_remaining plus 1; improvement search variant 1,
reports/archive/winprob.md), plus a `possession_missing` flag in the design matrix. They are recomputed from the columns above.

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
margin, seconds_remaining, possession, margin_x_sqrt_time and margin_over_sqrt_time; HistGradientBoostingClassifier on the same
plus period, overtime, spread and spread_missing, calibrated by isotonic regression on inner out-of-fold
predictions grouped by game. Selection: the unweighted mean over the five regulation buckets of the
out-of-fold Brier score under grouped folds by game within the training seasons. ECE and calibration slope per
bucket carry bootstrap intervals by game; log loss overall. The held-out season is scored once by the selected
candidate. Guards: analytics.eval.refuse_heldout on the training game ids, analytics.schema.assert_time_forward on
the design matrix, a name allowlist (analytics.winprob.features.assert_no_leak) and a value check
(prefix_invariance_violations: a feature at event k is unchanged when the play-by-play is cut after k and the games
row's finals and outcome are scrambled). The games table's home_score, away_score and home_win are dropped before any
feature is built; the target is mapped onto the states afterwards.

Benchmark: the NBA.com tables were skipped by the owner (gates/nbacom-skip, 2026-11-11), so the WinProbabilityPBP
benchmark is not collected and the benchmark column of reports/winprob.md stays empty, with that reason stated
(train.BENCHMARK_NOTE); if collected later it would join winprob_states on (game_id, event).

## Frozen for the held-out run (freeze-v1)

Frozen from the code on branch track-fix, reports/winprob.md (rerun with variant 1 merged) and
reports/archive/winprob.md. Every number below is out-of-fold within the training seasons, from a run with
`--skip-heldout`; no held-out number has been read.

**Selected model:** `logistic` with the accepted improvement-search variant 1 (the feature
`margin_over_sqrt_time = margin / sqrt(seconds_remaining + 1)` added to the incumbent's features).

| setting (analytics/winprob/train.py, features.py, validate.py) | frozen value |
| --- | --- |
| logistic | StandardScaler then `LogisticRegression(C=1.0, max_iter=2000)` on margin, seconds_remaining, possession_home, margin_x_sqrt_time, margin_over_sqrt_time |
| hgb_iso | `HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, max_depth=3, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)` on every feature (random_state is fixed at 0, not --seed; at seed 0 they agree), isotonic on 3 inner grouped folds |
| clock rule | a game whose period numbers skip a value, or whose period above 4 holds a clock above 300 s, is dropped whole; then any event with a null period or clock, a period under 1, or a clock outside [0, 720] (300 in overtime) is dropped (53 of 6416 games, 24584 events in the training run) |
| validate failure | `MAX_DROPPED_SHARE = 0.02`: a season losing more than 2 percent of its events or games under the clock rule makes validate return ok False; run_all then marks the step failed and runs neither features nor train for winprob |
| buckets | minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0 (selection), OT reported only |

Rules the held-out run applies: selection by the unweighted mean of the five regulation bucket Brier scores (lowest
wins, ties to the earlier model); 5 grouped folds by game within 2020-21 to 2023-24, seed 0; 200 game bootstrap
resamples, seed 0. Held out: the 2024-25 season (1304 games, 596840 events in the training run), set aside before any
fit; rows flagged held out are dropped, 2015-16 is refused, seasons after 2024-25 are ignored. The selected logistic
is refit on every training row and scores 2024-25 once: Brier, ECE and slope per bucket with intervals, overall log
loss and Brier, and the mean bucket Brier.

Out-of-fold headline (5059 games): selection metric 0.1468 (no interval in the report); log loss overall 0.4845
[0.4767, 0.4945], Brier overall 0.1645 [0.1616, 0.1683]; bucket Brier 48-36 0.2306 [0.2278, 0.2332], 4-0 0.0530
[0.0496, 0.0564]; constant 0.2472. Variant 1 against the incumbent (archive, 1000 game resamples):
0.00149 [0.00132, 0.00169] in favour of the variant.

Stage 8 command:

    python3 run_all.py --data "$DATA_ROOT" --ext xfg rebounds matchups gravity ghosting embeddings epv decisions lineups winprob load --continue-on-error

run_all.step_argv gives `analytics.winprob.train` the argv `--data DATA_ROOT --seed 0 --report reports/winprob.md`
(`--heldout-season` defaults to 2024-25; OUT_ROOT to DATA_ROOT); validate and features get `--data DATA_ROOT --seed 0`.
