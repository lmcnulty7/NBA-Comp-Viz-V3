# Ghosting contract: `analytics/ghosting/`

Defensive ghosting: where a defender would have been, given what everyone else did. The one deep model
of the plan, kept only if it beats the linear baseline under a paired bootstrap by game. Code:
`analytics/ghosting/{validate,features,train,eda}.py`; tests: `tests/test_ghosting.py`; report:
`reports/ghosting.md`; EDA: `notebooks/eda_ghosting.ipynb`.

## Unit, sampling, assignment

| item | definition |
| --- | --- |
| unit | one defender at one sampled frame of one possession (a defender-frame sequence inside a possession) |
| sampling | every 5th frame of the 25 Hz frames (section 4) from the possession's `start_frame`: 5 Hz, `DT` = 0.2 s |
| possessions used | section 19 rows with `gap == false` and a non-null `offense_team_id`; the lineup is the ten players of the first sampled frame, split five and five by `team_id`; the sequence stops at the first sampled frame without those ten players, without a ball, whose id is not exactly 5 on from the previous sampled frame, or whose game clock advanced by more than 0.2 s plus 0.1 s since it (a dropped frame or a source gap is never sampled as 0.2 s; a stopped clock with frames still flowing is a dead ball and allowed); sequences shorter than 4 steps are dropped; every skip and cut is counted per game |
| assignment | for each defender, the nearest offensive player at the possession's first sampled frame (defender-centric nearest; two defenders may share an assignment, no one-to-one matching) |
| attacking side | `attack_right = ball_x(t-1) > 47`; the attacking basket is the one on that side, (88.75, 25) or (5.25, 25). A past-only proxy for the offence's direction; a per-team per-period direction from the play-by-play is a follow-up |
| shot clock | null filled with max(24 minus elapsed, 0) |

## Target and horizons

The target is the defender's displacement from t minus 1 to t (both axes, feet). The model never sees the
defender's own position at t or anything after t.

| horizon | definition |
| --- | --- |
| 0.2 s | one step: ghost = position at t minus 1 plus the predicted displacement |
| 1.0 s | five steps rolled forward: at each step the ghost's own position and velocity re-enter the features while the offence, the ball and the other defenders stay actual; rows whose sequence ends before the fifth step have no 1.0 s row |

## Features (`features.FEATURES`, lag 1; `features.window_columns()` repeats them for lags 1 to 5)

All measured at t minus k (k = 1 for the linear baseline, 1 to 5 for the window model); lags before the
sequence start repeat the first step's vector.

| feature | meaning |
| --- | --- |
| `rel_asg_dx`, `rel_asg_dy` | assignment minus defender |
| `rel_off1..5_dx/dy` | each offensive player minus defender (slots sorted by `player_id`, as in section 20) |
| `rel_ball_dx`, `rel_ball_dy` | ball minus defender |
| `rel_def1..4_dx/dy` | the other four defenders minus defender (slot order, self removed) |
| `ball_basket_dist_ft` | ball to the attacking basket |
| `def_basket_dist_ft` | defender to the attacking basket |
| `def_vx`, `def_vy` | defender velocity over the last 0.4 s (two sampled steps), ft/s |
| `def_x`, `def_y` | defender position, feet (absolute; the persistence baseline is this position) |
| `shot_clock_s`, `elapsed_s` | shot clock; seconds since the possession's `start_frame` |
| `attack_right` | 1 when the ball is in the right half |

Window layout: lag 1 block first, then lag 2, ... lag 5, so `X.reshape(n, 5, 31)` is the (batch, time,
features) tensor a GRU consumes with the most recent step first.

## Candidates and selection

| candidate | model |
| --- | --- |
| 1 persistence | zero displacement (position at t minus 1) |
| 2 ridge (linear baseline) | StandardScaler + Ridge(alpha = 1) on the lag-1 features |
| 3 MLP window (GRU stand-in) | StandardScaler + MLPRegressor(hidden_layer_sizes = (64, 64), max_iter = 200, early_stopping = True) on the 5-lag window; a torch GRU over the same window is the drop-in once torch is installed on the Mac |

Selection metric: out-of-fold mean position error in feet at 0.2 s and 1.0 s under grouped folds by
`sportvu_game` (`analytics.eval.grouped_folds`), bootstrap intervals by game
(`analytics.eval.bootstrap_by_game`), paired bootstrap of 3 vs 2 and 2 vs 1
(`analytics.eval.paired_bootstrap_by_game`). Rule, fixed in advance: candidate 3 is kept only when its
1.0 s error minus candidate 2's is negative with a paired interval excluding zero; otherwise the selected
ghost is candidate 2. One training game: in-sample, labelled, no interval, candidate 3 dropped.

## Leakage guards

- `analytics.eval.refuse_heldout` on the training game ids; a game is held out by its `heldout` flag or
  by an id carrying a held-out token of `sportvu/splits.json` (the flag is not what protects it);
  `fit_candidates` raises on any held-out row
- `analytics.schema.assert_time_forward` on the feature frame and on every design matrix
- strictly past frames: the lag-1 block is measured at t minus 1; `features.shift_features_forward` is
  the test probe that moves every window one step later (lag 1 measured at the target frame), after which
  a near-unpenalised linear probe recovers the target exactly and the error collapses
- points and end_kind (section 19 TARGETS) are joined to the possession table only after every fit

## Tables

Layout: `OUT_ROOT/analytics/ghosting/<season>/<table>_<sportvu_game>.parquet`, held-out games under
`OUT_ROOT/analytics/ghosting/heldout/<season>/`; OUT_ROOT defaults to DATA_ROOT. The features CLI writes
`ghost_features_<sportvu_game>.parquet` the same way. Dtypes follow the contract conventions (ids int32,
positions float32, points int16, `sportvu_game` string).

### `ghost_frames` (`ghost_frames_<sportvu_game>.parquet`)

One row per defender-step, horizon and the selected candidate.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame of the target step (t for 0.2 s, t + 4 for 1.0 s) |
| `def_player_id` | int32 | the defender |
| `off_player_id` | int32 | the defender's assignment |
| `actual_x`, `actual_y` | float32 | the defender's position at `frame`, feet |
| `ghost_x`, `ghost_y` | float32 | the selected candidate's ghost at `frame`, feet |
| `deviation_ft` | float32 | distance(actual, assignment) minus distance(ghost, assignment) at `frame`; positive = farther from his man than the ghost |
| `horizon_s` | float32 | 0.2 or 1.0 |
| `fold` | int32 | the grouped fold whose test games hold the row; -1 for held-out rows scored once |
| `heldout` | bool | the game is held out (never fitted or tuned on) |

### `ghost_possessions` (`ghost_possessions_<sportvu_game>.parquet`)

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `mean_deviation_ft` | float32 | mean `deviation_ft` over the possession's 1.0 s rows of the selected candidate |
| `points` | int16 | section 19 points (TARGET; joined after every fit, for the validation only) |

## Validation

Per game, the Spearman correlation of `mean_deviation_ft` with `points` over the game's possessions (games
with fewer than 3 possessions or no variance give no correlation); then a sign test over games (one-sided
binomial, more positive than negative). The hypothesis: a defence that strays farther from its
assignments than the ghost allows more points.

## Commands

    python -m analytics.ghosting.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.ghosting.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.ghosting.train --fixtures | --data DATA_ROOT --report reports/ghosting.md [--folds 5] [--seed 0] [--bootstrap 200] [--out OUT_ROOT] [--threads 1]

`DATA_ROOT/sportvu_contract/<game>.parquet` (frames; held-out under `heldout/`) and
`DATA_ROOT/analytics/possessions/2015-16/<game>.parquet` (held-out under `heldout/2015-16/`) are the inputs;
rosters are read when present and checked but not needed. A game read from any `heldout/` path is held out
whatever its id. `--out` defaults to DATA_ROOT; the Mac command is `--out "<Drive>/nba_harvest"`. Fixture runs print and write "fixture output,
not a result".
