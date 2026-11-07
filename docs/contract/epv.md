# EPV contract: expected possession value from 1 Hz states (`analytics/epv`)

Unit: one state at 1 Hz inside a possession (docs/DATA_CONTRACT.md section 20). Target: `points_to_end`, the
points the offence scores from the state to the end of the possession (0 to 4), modelled as a regression of
expected points with the classification view P(points > 0) for calibration. Held-out games of
`sportvu/splits.json` never fit or tune anything.

## Inputs

| table | where (DATA_ROOT layout) | used for |
| --- | --- | --- |
| states_1hz (section 20) | `analytics/states_1hz/<season>/<game>.parquet`, held-out under `analytics/states_1hz/heldout/<season>/` | the unit, the geometric features, the targets |
| frames (section 4) and possessions (section 19) | `sportvu_contract/<game>.parquet`, `analytics/possessions/<season>/<game>.parquet` | building states_1hz when it is absent (`python -m analytics.epv.states --data DATA_ROOT`) |
| play-by-play (section 5) | `sportvu_contract/<game>_pbp.parquet` | optional: refines `points_to_end` for an and-one (points logged before the state) |
| shots (sections 10 and 10a) | `collect/bref_shots/shotchart/<season>.parquet` (held-out under `collect/bref_shots/heldout/shotchart/`), `collect/shots/shotchartdetail/` unioned when present | the handler's shooting prior by zone and the location-only xFG fallback |
| rosters (section 16) | `analytics/rosters/<season>/<game>.parquet` | the shooter id of a basketball-reference shot (`player_slug` to `bref_slug`) |

`analytics.epv.validate` runs `analytics.schema.validate` on every contract table, logs null counts, out-of-range
counts and duplicate keys, drops duplicate keys (first row kept), and drops a states row whose target is outside
0 to 4, whose elapsed time is negative or whose non-nullable column is null. Coordinates outside the court by
more than 5 ft are logged, not dropped. Every table carries a bool `heldout` column from the directory it came
from, or-ed with the ids of `sportvu/splits.json`.

## states_1hz as built by `analytics.epv.states` (when the Phase 2 table is absent)

The section 20 columns and dtypes (`analytics.schema.cast`). Decisions where the contract is silent:

- elapsed time is measured on the game clock (`start_clock_s - game_clock_s`); the state is the span frame whose
  clock is nearest each whole second and is kept only within 0.5 s of it (a frame gap is skipped, not interpolated;
  a stalled clock adds no state);
- the ten players split by `team_id` against `offense_team_id`; a frame without exactly five and five is skipped;
  a possession whose `offense_team_id` is null is skipped whole; a frame without a ball entity is skipped (the
  ball columns are non-nullable);
- `handler_id` is the offensive player nearest the ball in the floor plane;
- `points_to_end` is the possession's points less the offence's points logged before the state (events with
  `clock_s` strictly greater than the state's clock inside the possession's event span, the and-one); without a
  play-by-play table it is the possession's points for every state.

All skip counts are printed per game and kept in the frame's `attrs`.

## Features (`analytics.epv.features.FEATURES`, time-forward)

The only column source is `analytics.schema.feature_allowlist("states_1hz")`; `points_to_end` and
`possession_end_kind` never enter; every feature frame passes `analytics.schema.assert_time_forward`.

| feature | meaning |
| --- | --- |
| `ball_x`, `ball_y`, `ball_z`, `ball_dist_basket` | ball position and its floor distance to the attacked basket |
| `handler_dist_basket`, `handler_angle` | the handler's distance to the basket and angle from the rim-to-rim axis (NaN without a handler) |
| `handler_nearest_def_dist_ft` | as in the states table |
| `off_dist_1` to `off_dist_5` | the five offensive players' distances to the basket, sorted ascending |
| `off_def_1` to `off_def_5` | their distances to their nearest defender, sorted ascending |
| `off_hull_area` | convex-hull area of the five offensive players, square feet |
| `shot_clock_s`, `elapsed_in_possession_s` | from the table (shot clock NaN when the source has none) |
| `ball_dist_min_so_far`, `n_handler_switches` | the possession's past up to t: the smallest ball distance to the basket so far, handler changes so far |
| `handler_fg_prior` | the handler's smoothed make probability in the zone of the handler's current distance, from the shots of the training games only, out-of-fold (a training game's own shots are left out of its zone counts), shrunk with 20 pseudo-attempts towards the location-only xFG (`analytics.features_xfg`, a four-parameter league model fitted once on the whole training fold, the row's own game included) at the handler's position |
| `handler_zone_value` | 2 or 3, the value of a shot from the handler's distance (22 ft and beyond is 3) |
| `lineup_tov_prior` | the offensive lineup's smoothed turnover rate per possession from the training games, out-of-fold the same way, 30 pseudo-possessions towards the training mean |

Prior fitting detail: the xFG shrinkage target and the lineup turnover mean (`tov_rate`) are fitted once on the whole
outer training fold (league-wide quantities with a handful of parameters), while the per-player zone counts and the
per-lineup counts are leave-one-game-out inside the fold. Leave-one-game-out target encoding carries a mild negative
bias of order 1/(number of training games - 1): a game's own outcomes are subtracted from the total, so with very
few games the prior anti-correlates with the game's target. Negligible at hundreds of games, visible on fixtures.

Attacked basket: the contract does not carry it, so it is the basket nearer the expanding mean of the offensive
centroid over the possession's states up to t (no later frame is read). Zones by distance: rim under 4 ft, paint
4 to 14, mid 14 to 22, three 22 and beyond. The gated sequence model reads `features.lagged`: the same features
at t, t-1, ..., t-4 of the same possession (padded with the row's own values at the start of the possession).

## Protocol (`analytics.epv.train`)

1. Held-out rows set aside; `analytics.eval.refuse_heldout` on the training game ids (states and shots).
2. Grouped folds by `sportvu_game` (fewer games than folds: one fold per game; one game: in-sample, labelled).
   Priors are rebuilt inside each fold from its training games.
3. Candidates: constant (training mean), ridge (median imputation, standardisation, alpha 1),
   `HistGradientBoostingRegressor` (100 iterations, learning rate 0.05, 8 leaves, 50 per leaf, fixed in advance),
   each with an isotonic calibration of P(points > 0) fitted on inner out-of-fold predictions (3 grouped folds);
   under the gate, `MLPRegressor` (one hidden layer of 32) over the last 5 states' features (the default,
   `--deep mlp`); with `--deep torch` the torch sequence model below fills that slot instead.
4. Gate: the sequence model is trained only if the gradient-boosting calibration slope by shot-clock bucket
   (0 to 7, 7 to 14, 14 to 24 s) is outside 0.9 to 1.1 in any bucket or its MSE lift over the constant is under 5
   percent, and kept only if it beats gradient boosting under paired bootstrap by game.
5. Selection: the lowest out-of-fold MSE among the three is the base; the sequence model replaces it only when it
   beats the base under paired bootstrap by game with a lower out-of-fold MSE. The report gives the in-sample MSE next
   to the out-of-fold MSE (overfit is visible), a one-sentence reason for the selection, the Brier score of the calibrated
   P(points > 0) and the MSE by elapsed bucket (0 to 4, 4 to 10, 10 to 24 s, later states in the last bucket) with
   bootstrap intervals by game, the calibration slope per bucket (target 0.9 to 1.1, reported only), the
   reliability tables, the shuffled-target check (possession points permuted across possessions, the lift must sit
   at chance), and the held-out games scored once by the selected candidate refit on every training row, with the
   constant kept as a labelled floor; no other candidate ever sees a held-out row (point estimates only with fewer
   than 5 held-out games).

Every fit and predict of a candidate runs inside `analytics.eval.fit_threads()`, which pins the BLAS and OpenMP
pools to one thread after loading sklearn's gradient-boosting runtime (on the shared box that runtime hung with its
default thread count). Nothing is set in the environment at import. The per-bucket calibration slopes use
`analytics.eval.calibration_slope_fast` (the same regression, bounded on nearly separable buckets).

## Deep model (stage 6): the torch sequence model behind `--deep torch`

Code: `analytics/epv/deep.py` (the model) and `analytics/deep_torch.py` (shared plumbing); tests:
`tests/test_deep_models.py`.

| item | definition |
| --- | --- |
| option | `--deep {mlp,torch}` on `analytics.epv.train`, default `mlp` (the `MLPRegressor` stand-in, byte for byte the same candidates and decision) |
| slot | with `--deep torch` the GRU replaces the MLP as the gated sequence model, still named `mlp` in the code and the tables (`epv_mlp`, `SEQUENCE`), so one sequence model per run |
| inputs | the same `features.lagged_columns(5)`: the state's features at t and at the four earlier states of the same possession (padded with the row's own values at the possession start), reshaped to (n, 5, n_features) and reversed on the time axis, so the GRU runs from t minus 4 to t; no state after t is read (`features.lagged` reads earlier states only, and `assert_time_forward` runs on the frame) |
| target | the same `points_to_end` regression; `train.Calibrated` fits the same isotonic map to P(points > 0) on inner out-of-fold predictions |
| architecture | median imputation and standardisation fit on the training rows (as the stand-in's pipeline), target standardised; `GRU(n_features -> 16, one layer)`, last hidden state, `Linear(16 -> 1)` |
| training | Adam (learning rate 0.002, weight decay 0.0001), batch 256, at most 100 epochs, MSE loss; early stopping after 8 epochs without a lower validation MSE, the best epoch's weights kept |
| early-stopping split | from the rows the fit receives (the training games of the current outer fold, or of the inner calibration fold): whole games when there are at least 5, else a seeded 20 percent row split; never a held-out row, never the fold being scored |
| determinism | `torch.manual_seed(--seed)` before the net is built, a seeded shuffling generator, CPU only, `torch.set_num_threads(1)` for the fit and the predictions (restored after) |
| gate and keep rule | unchanged and fixed in advance: trained only when `gate_decision` opens it; kept only by `select_model` (paired bootstrap of MSE(base) minus MSE(sequence) with its interval above 0 and a lower out-of-fold MSE) |
| held-out | the same code path as the MLP (`split_tables` and `refuse_heldout` first; the held-out games are scored once by the selected candidate); `deep.fit_sequence` raises ValueError on a true `heldout` flag or a held-out game id before torch is touched |
| fallback | `--deep torch` without an importable torch logs "torch not available; MLP stand-in used", runs the stand-in, and the report says so; torch is imported lazily inside the fit |
| report | with `--deep torch` a "Sequence model (--deep torch)" line gives these settings and the gate line names the torch GRU; the default report is unchanged |

Stage 6 command (only when this lane's gate opened; `--skip-heldout` is the stage 4 flag every real run carries):

    python -m analytics.epv.train --data "$DATA_ROOT" --report reports/epv.md --folds 5 --seed 0 --bootstrap 1000 --out "$DATA_ROOT" --deep torch --skip-heldout

## Outputs

Written under `OUT_ROOT` only (never the repo); the layout the decisions and playcalling lanes read.

### epv_states: `OUT_ROOT/analytics/epv/<season>/<game>.parquet` (held-out: `OUT_ROOT/analytics/epv/heldout/<season>/<game>.parquet`)

One row per state of the states table that reached the trainer, the selected model's prediction.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 20 frame of the state (join key for `merge_asof` on frame; int32 as in the states table) |
| `elapsed_in_possession_s` | float32 | seconds since the possession's start frame |
| `epv_points` | float64 | expected points from this state to the end of the possession (the selected model) |
| `p_score` | float64 | calibrated P(points > 0) |
| `points_to_end` | int16 | the target, carried for evaluation (never a feature downstream either) |
| `fold` | int32 | the grouped fold the row was predicted out of (>= 0); -1 for held-out rows; -2 for in-sample rows (the one-training-game mode, where no grouped fold is possible and the report says IN-SAMPLE) |
| `heldout` | bool | the game is held out (scored once by the pipeline refit on all training rows) |

Out-of-fold rows (training games) carry predictions from a model that never saw their game; held-out rows carry
the refit model's predictions. A consumer must not fit on held-out rows.

### epv_possessions: `OUT_ROOT/analytics/epv/<season>/epv_possessions.parquet` (held-out: `OUT_ROOT/analytics/epv/heldout/<season>/epv_possessions.parquet`)

One row per possession, one file per season (a later run merges its games into the file).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `epv_at_start` | float64 | `epv_points` of the possession's first state (smallest elapsed time) |
| `points` | int16 | the possession's points (the target at that state) |

### Report: `reports/epv.md`

Inputs, protocol, the candidate table by elapsed bucket, calibration tables, the shot-clock slopes and the gate
decision, the shuffled check, the held-out games once, the Mac command. A fixture run says "fixture output, not a
result" and its numbers are never presented as results.

### Feature frames: `OUT_ROOT/analytics/epv_features/<season>/<game>.parquet` (held-out: `OUT_ROOT/analytics/epv_features/heldout/<season>/<game>.parquet`)

Written by `python -m analytics.epv.features`: `sportvu_game`, `possession`, `frame` and the geometric features of
FEATURES, one row per state; the three prior columns are NaN here because they are fold-dependent and the trainer
builds them inside each fold. Also written: `OUT_ROOT/analytics/epv_validate.md` by `python -m analytics.epv.validate`.

## Where the CLIs read from

`analytics.epv.validate.load_root(DATA_ROOT, out_root=OUT_ROOT)` reads each Phase 0 table (`analytics/states_1hz`,
`possessions`, `rosters`, `shot_frames`, with their `heldout/` variants) from OUT_ROOT when OUT_ROOT holds any parquet
for that table, else from DATA_ROOT; the shots tables always come from `DATA_ROOT/collect/`. So
`states --data X --out Y` followed by `features` or `train --data X --out Y` finds the states under Y and the other
tables under X (`make ext-epv DATA=X OUT=Y`).

## Commands

    python -m analytics.epv.states   --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--tol 0.5]
    python -m analytics.epv.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.epv.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.epv.train    --fixtures | --data DATA_ROOT --report reports/epv.md [--folds 5] [--seed 0] [--bootstrap 200] [--out OUT_ROOT] [--deep mlp|torch]

`--fixtures` builds the synthetic season of `analytics.fixtures` (one league: teams and rosters recur) in memory
through `analytics.epv.validate.fixture_season`; every CLI sizes it with `--fixture-games` (alias `--games`; 4, or 8
for train), `--fixture-possessions` (40) and `--fixture-heldout-every` (4, the first held-out game being
12.16.2015.PHX.at.GSW). A fixture report carries "fixture output, not a result" on its banner and on every section
heading, and records its command with the value of `--out` written as `OUT_ROOT`. `<season>` in every path is
`analytics.schema.season_of_game` (the season turns over in July); held-out ids are `analytics.eval.heldout_mask`.

On the Mac: `python -m analytics.epv.states --data DATA_ROOT` once (when `analytics/states_1hz` is absent), then
`python -m analytics.epv.train --data DATA_ROOT --report reports/epv.md --folds 5 --seed 0 --bootstrap 1000 --out DATA_ROOT`.

## Tests

`tests/test_epv.py` (the season of `analytics.fixtures`, through `analytics.epv.validate` and the test-only
adapter `tests/_fx_epv.py`): the states builder against the possessions table, validation, the
feature guard (a target column raises, the allow-list is the only source), the priors out of fold, the end-to-end
trainer with the report and the output tables, the leakage guard (a shuffled target shows no lift; a held-out id
in the training set stops the trainer), the CLIs on a temporary DATA_ROOT.
