# xFG upgrade: the shot-quality tables and protocol (`analytics/xfg`)

Lane XFG of the analytics build. Unit: one field-goal attempt of the shots table (docs/DATA_CONTRACT.md sections
10 and 10a). Target: `made` (1/0). Code: `analytics/xfg/{validate,features,train,eda}.py`; tests:
`tests/test_xfg_upgrade.py` on the synthetic season of `tests/_fx_xfg.py` (which prefers `analytics.fixtures`).
Nothing in this lane reads the network or writes raw data into the repository.

## 1. Inputs and the join

| input | contract | key |
| --- | --- | --- |
| shots | sections 10 and 10a | `sportvu_game`, `event` (the play-by-play event index) |
| shot_frames | section 17 | `sportvu_game`, `event` |
| frames | section 4 | `game`, `frame`, `player_id` |
| rosters, possessions | sections 16, 19 | validated and counted, not used as features |

Join rule:

1. `shots` to `shot_frames` by (`sportvu_game`, `event`). A shots table without an `event` column (basketball-reference
   rows carry a null `event_id`) is matched by (`sportvu_game`, `period`) and nearest `clock_s` within 1.0 s, each
   shot_frames row used at most once; unmatched attempts get no contest features. The count is logged.
2. The matched row gives `release_frame` and the shooter (`shooter_id`, else the shots `player_id`, else
   `shooter_at_release_id`). An attempt with `release_frame` -1, `confidence_flag` `unmatched`, no shot_frames row, or a
   shooter absent from the release frame keeps its location features and gets null contest features with
   `has_contest` 0.
3. The release frame and the frames before it (10 for speeds, up to 250 for the touch) are read from the frames table
   of the same game, one game at a time (a season of frames does not fit in memory).

Nothing from `shot_frames` that is a target or post-outcome column (`made`, `apex_frame`, `outcome_frame`,
`pbp_shooter_name`, `identity_match`, `confidence`, `confidence_flag`) enters the feature frame;
`analytics.schema.assert_time_forward` runs on the frame and `analytics.xfg.features.assert_no_leak` on every design
matrix (`FORBIDDEN` = `schema.ALL_EXCLUDED` plus `made`, `points`, `score`, `outcome_frame`, `apex_frame`,
`identity_match`, `confidence`, `confidence_flag`).

## 2. Validation (`analytics.xfg.validate`)

`schema.validate` on shot_frames, rosters and possessions; column and range checks on frames, play-by-play and shots:
court 0 to 94 by 0 to 50 ft (`x`, `y`, `ball_x`, `ball_y`), clocks 0 to 720 s, shot clock 0 to 24 s or null, shot
location inside the half court (`loc_x` -250 to 250, `loc_y` -55 to 420 tenths of a foot). Rows failing a range check
or a non-nullable column are dropped; duplicates (frames by `game, frame, player_id`; shots and shot_frames by
`sportvu_game, event`; rosters by `sportvu_game, player_id`; possessions by `sportvu_game, possession`) keep the first
row. Rows in and out and every drop are logged and written to `OUT_ROOT/analytics/xfg/<season>/validate.json`.

## 3. Features (`analytics.xfg.features.FEATURES`, all measured at or before the release)

| feature | definition |
| --- | --- |
| `distance_ft`, `angle_rad`, `is_three`, `is_corner` | the location-only baseline (`analytics.features_xfg`) |
| `closest_defender_dist_ft` | distance from the shooter to the nearest opponent at the release frame |
| `closest_defender_closing_speed_ftps` | (that defender's distance 10 frames before release minus at release) / 0.4 s; positive when closing |
| `n_defenders_within_6ft` | opponents within 6 ft of the shooter at release |
| `shooter_speed_ftps` | the shooter's displacement over the 10 frames before release / 0.4 s |
| `touch_time_s` | frames (walking back from the release, same period, consecutive) in which the ball stayed within 2.5 ft horizontally of the shooter, / 25, capped at 10 s |
| `dribbles_proxy` | local minima of ball z under 2 ft strictly inside the touch window |
| `shot_clock_s` | shot clock at the release frame (null when the source has none) |
| `has_contest` | 1 when the contest features were measured, else 0 |
| `period`, `clock_s` | the attempt's period and logged game clock (shots table) |
| `shooter_prior` (candidate 3 only) | the shooter's smoothed make rate in the attempt's zone, computed only from fitting rows, out-of-fold by game for the rows being fitted: `(makes + 20 * zone_rate) / (n + 20)`; zones rim (under 4 ft), paint (under 14 ft), mid, corner3, above3 |

Frame rate 25 Hz; a window that crosses a period, has a frame gap or fewer than 2 frames gives null speeds.

## 4. Candidates and protocol (`analytics.xfg.train`)

| candidate | model |
| --- | --- |
| `location_logistic` | `analytics.train_xfg.fit_models` on the four location features (the Phase 1 baseline) |
| `contest_logistic` | median imputation, standardisation, logistic regression (C = 1) on `FEATURES` |
| `hgb_contest_prior` | `HistGradientBoostingClassifier` (100 iterations, depth 3, learning rate 0.1, min leaf 10, L2 1.0) on `FEATURES` plus `shooter_prior`; isotonic calibration fitted on out-of-fold predictions inside the fitting rows (3 inner grouped folds, nested priors) |

Protocol: `analytics.eval.grouped_folds` by `sportvu_game`, the same folds for every candidate (`grouped_cv`);
selection by out-of-fold log loss; Brier, ECE (10 bins, target under 0.02) and calibration slope with bootstrap
intervals by game (`bootstrap_by_game`; the slope resamples use a Newton solve of the same regression); paired
bootstrap by game of candidate 3 against 1 and 2 against 1 on log loss (`paired_bootstrap_by_game`). Rows whose
`heldout` flag is true, whose game is under `heldout/`, or whose game id carries a held-out token of
`sportvu/splits.json` (or the CLE arena) are set aside before any fit; `analytics.eval.refuse_heldout` runs on the
training game ids and again inside every fit. Held-out rows are scored once by the candidates refit on every
training row; the reported held-out score is the selected candidate's. Fewer games than folds: leave-one-game-out;
one training game: the trainer stops.

## 5. Output table: `xfg_predictions`

`OUT_ROOT/analytics/xfg/<season>/xfg_predictions.parquet`, one row per attempt of the input (training rows and
held-out rows).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int64 | play-by-play event index of the attempt (join key to shots and shot_frames) |
| `player_id` | Int64, nullable | the shooter |
| `p_location` | float32 | probability from `location_logistic` |
| `p_contest` | float32 | probability from `contest_logistic` |
| `p_selected` | float32 | probability from the selected candidate (named in the report) |
| `made` | bool | the target, carried for scoring downstream (never a feature) |
| `fold` | int16 | the out-of-fold index of a training row; -1 for a held-out row |
| `heldout` | bool | true for a held-out row (scored once by the refit candidates, never fitted on) |

For a training row every `p_*` is out-of-fold. For a held-out row every `p_*` comes from the candidate refit on all
training rows. `OUT_ROOT` defaults to `DATA_ROOT` under `--data`; under `--fixtures` the table is written only with
`--out`.

Also written: `OUT_ROOT/analytics/xfg/<season>/xfg_features.parquet` (by `python -m analytics.xfg.features --out`),
the feature frame of section 3 with the keys, `release_frame` and `zone`, and no label.

## 6. Commands

    python -m analytics.xfg.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.xfg.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.xfg.train    --fixtures | --data DATA_ROOT --report reports/xfg.md [--folds 5] [--seed 0] [--bootstrap 200] [--out OUT_ROOT]
    python -m analytics.xfg.eda      --fixtures | --data DATA_ROOT

DATA_ROOT layout: `sportvu_contract/<game>.parquet` and `<game>_pbp.parquet` (held-out games under `heldout/`),
`collect/bref_shots/shotchart/<season>.parquet` (+ `heldout/shotchart/`), `analytics/<table>/<season>/<game>.parquet`
(+ `heldout/<season>/`) for shot_frames, rosters and possessions. `--fixtures` runs on the synthetic season; every
number it produces is fixture output, not a result.

## 7. What needs real data

The report `reports/xfg.md` in the repository is a fixture run. The real numbers need, on the Mac: the 2015-16 shots
parquet, the shot_frames of every game with frames and play-by-play (the `analytics.shot_frames` driver over the
Drive manifest) and the frames themselves. Open questions only real data can answer: the share of attempts whose
release frame localises (70 percent on the one smoke game), the match rate of basketball-reference attempts to
shot_frames rows by clock, and whether the contest features and the prior beat the location baseline under the
paired bootstrap.
