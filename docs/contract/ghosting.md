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

Window layout: lag 1 block first in columns, then lag 2, ... lag 5, so `X.reshape(n, 5, 31)` is the (batch,
time, features) tensor with lag 1 first; the GRU reverses it to oldest first.

## Candidates and selection

| candidate | model |
| --- | --- |
| 1 persistence | zero displacement (position at t minus 1) |
| 2 ridge (linear baseline) | StandardScaler + Ridge(alpha = 1) on the lag-1 features |
| 3 MLP window (GRU stand-in) | StandardScaler + MLPRegressor(hidden_layer_sizes = (64, 64), max_iter = 200, early_stopping = True) on the 5-lag window (the default, `--deep mlp`); with `--deep torch` the torch GRU below fills this slot instead |

Selection metric: out-of-fold mean position error in feet at 0.2 s and 1.0 s under grouped folds by
`sportvu_game` (`analytics.eval.grouped_folds`), bootstrap intervals by game
(`analytics.eval.bootstrap_by_game`), paired bootstrap of 3 vs 2 and 2 vs 1
(`analytics.eval.paired_bootstrap_by_game`). Rule, fixed in advance: candidate 3 is kept only when its
1.0 s error minus candidate 2's is negative with a paired interval excluding zero; otherwise the selected
ghost is candidate 2. One training game: in-sample, labelled, no interval, candidate 3 dropped. The
held-out games are scored once by the selected candidate only (plus persistence as a labelled floor); the
dropped candidate never sees them, so nothing can be chosen on held-out numbers.

## Deep model (stage 6): the torch GRU behind `--deep torch`

Code: `analytics/ghosting/deep.py` (the model) and `analytics/deep_torch.py` (shared plumbing); tests:
`tests/test_deep_models.py`.

| item | definition |
| --- | --- |
| option | `--deep {mlp,torch}` on `analytics.ghosting.train`, default `mlp` (the sklearn stand-in above, byte for byte the same candidates and decision) |
| slot | with `--deep torch` the GRU replaces the MLP as candidate 3, still keyed `mlp` in the code (`pairs["mlp_vs_ridge"]`, `decision["keep_mlp"]`), so one deep candidate per run and the comparison count is unchanged |
| inputs | the same `features.window_columns()` matrix (5 lags of the 31 base features, lag 1 block first) reshaped to (n, 5, 31) and reversed on the time axis, so the GRU runs from t minus 5 to t minus 1 |
| target | the same displacement from t minus 1 to t (both axes) |
| roll-forward | the same `train.roll_forward`, unchanged: the GRU is a `predict(window_matrix) -> (n, 2)` callable like every candidate |
| architecture | inputs and targets standardised on the training rows; `GRU(31 -> 32, one layer)`, last hidden state, `Linear(32 -> 2)` |
| training | Adam (learning rate 0.002), batch 256, at most 100 epochs, MSE loss; early stopping after 8 epochs without a lower validation MSE. The fit is two passes on the rows the fit receives: early stopping on the inner split picks the best epoch, then the net is reset to its initial weights (the same seed) and refit for that many epochs on every one of those rows, so it learns from the same rows as the sklearn stand-in; a validation loss that is never finite raises ValueError. `torch.use_deterministic_algorithms(True)` is on during the fit and the predictions (restored after), so `--threads` above 1 reproduces too. |
| early-stopping split | from the training games of the current fold only (the rows `fit_candidates` receives): whole games when there are at least 5, else a seeded 20 percent row split; never a held-out row, never the fold being scored |
| determinism | `torch.manual_seed(--seed)` before the net is built, a seeded shuffling generator, CPU only, `torch.set_num_threads(--threads)` for the fit and the predictions (restored after) |
| keep rule | unchanged and fixed in advance: `train.decide`, kept only when its 1.0 s error minus the ridge's is negative with a paired interval excluding zero; otherwise ridge |
| held-out | the same code path as the MLP: `split_games` and `refuse_heldout` first, `fit_candidates` raises on a held-out row, and `deep.fit_gru` raises ValueError on a true `heldout` flag or a held-out game id before torch is touched; the held-out games are scored once by the selected candidate only |
| fallback | `--deep torch` without an importable torch logs "torch not available; MLP stand-in used", runs the MLP stand-in, and the report says so; torch is imported lazily inside the fit, so the package imports without it |
| report | with `--deep torch` the candidate label reads "3 torch GRU (hidden 32) over the 5-step window" and a "Deep candidate (--deep torch)" line gives these settings; the default report is unchanged |

Stage 6 command (only when this lane's gate opened; `--skip-heldout` is the stage 4 flag every real run carries):

    python -m analytics.ghosting.train --data "$DATA_ROOT" --report reports/ghosting.md --folds 5 --seed 0 --bootstrap 1000 --out "$DATA_ROOT" --threads 4 --deep torch --skip-heldout

This command needs PR #8 (the stage 4a `--skip-heldout` flag) merged first; without it the trainer rejects the flag.

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
| `fold` | int32 | >= 0: the grouped fold whose test games hold the row (out-of-fold); -1: held-out rows scored once; -2: in-sample rows (one training game, no fold possible) |
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
        [--deep mlp|torch]

`DATA_ROOT/sportvu_contract/<game>.parquet` (frames; held-out under `heldout/`) and
`DATA_ROOT/analytics/possessions/2015-16/<game>.parquet` (held-out under `heldout/2015-16/`) are the inputs;
rosters are read when present and checked but not needed. A game read from any `heldout/` path is held out
whatever its id. `--out` defaults to DATA_ROOT; the Mac command is `--out "<Drive>/nba_harvest"`. Fixture runs print and write "fixture output,
not a result".

Fixture flags (every ghosting CLI): `--fixture-games 4`, `--fixture-possessions 10` (at least 4),
`--fixture-heldout-every 0`; `--games`, `--possessions` and `--heldout-every` are kept as aliases. The fixtures
are `analytics.fixtures`; production code never imports `tests/`. `--threads N` is passed to
`analytics.eval.fit_threads(N)`, which pins BLAS and OpenMP for every fit.
