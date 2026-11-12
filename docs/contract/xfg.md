# xFG upgrade: the shot-quality tables and protocol (`analytics/xfg`)

Lane XFG of the analytics build. Unit: one field-goal attempt of the shots table (docs/DATA_CONTRACT.md sections
10 and 10a). Target: `made` (1/0). Code: `analytics/xfg/{validate,features,train,eda}.py`; tests:
`tests/test_xfg_upgrade.py` on the synthetic season of `analytics.fixtures` (through the test-only adapter
`tests/_fx_xfg.py`; production code reads `analytics.fixtures` directly and never imports `tests/`). Fixture runs take
`--fixture-games`, `--fixture-possessions` and `--fixture-heldout-every` on every CLI of the lane.
Nothing in this lane reads the network or writes raw data into the repository.

## 1. Inputs and the join

| input | contract | key |
| --- | --- | --- |
| shots | sections 10 and 10a | `sportvu_game`, `event` (the play-by-play event index) |
| shot_frames | section 17 | `sportvu_game`, `event` |
| frames | section 4 | `game`, `frame`, `player_id` |
| rosters, possessions | sections 16, 19 | validated and counted, not used as features |

Join rule:

1. `shots` to `shot_frames` by (`sportvu_game`, `event`) when the shots table carries the section 5 `event` index. A
   shots table without one (basketball-reference rows, which is every real run) is matched by clock: every shot_frames
   row of the same (`sportvu_game`, `period`) with |`clock_s` difference| <= 1.0 s is a candidate, candidates are
   assigned in order of clock distance, each attempt and each shot_frames row at most once (an attempt and its putback
   0.8 s later each get their own row). An attempt with no match carries `event` -1. NBA.com `event_id`
   (GAME_EVENT_ID) is not the section 5 page-order index and is never used as one. The match count is logged.
2. The matched row gives `release_frame` and the shooter (`shooter_id`, else the shots `player_id`, else
   `shooter_at_release_id`). An attempt with `release_frame` -1 (section 17: that is what `confidence_flag`
   `unmatched` means; the flag itself is not read), no shot_frames row (`event` -1), or a shooter absent from the
   release frame keeps its location features and gets null contest features with `has_contest` 0.
3. The release frame and the frames before it (10 for speeds, up to 250 for the touch) are read from the frames table
   of the same game, one game at a time (a season of frames does not fit in memory).

Nothing from `shot_frames` that is a target or post-outcome column (`made`, `apex_frame`, `outcome_frame`,
`pbp_shooter_name`, `identity_match`, `confidence`, `confidence_flag`) enters the feature frame;
`analytics.schema.assert_time_forward` runs on the frame and `analytics.xfg.features.assert_no_leak` on every design
matrix (`FORBIDDEN` = `schema.ALL_EXCLUDED` plus `made`, `points`, `score`, `outcome_frame`, `apex_frame`,
`identity_match`, `confidence`, `confidence_flag`).

## 2. Validation (`analytics.xfg.validate`)

`schema.validate` on shot_frames, rosters and possessions; column and range checks on frames, play-by-play and shots:
the ball inside the court (`ball_x` 0 to 94, `ball_y` 0 to 50 ft), players within a 5 ft margin of it (they step out
of bounds), clocks 0 to 720 s, shot clock 0 to 24 s or null, shot
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
| `period` | the attempt's period (shots table) |
| `clock_s` | the game clock at the release frame when `has_contest` is 1, else the logged clock of the shots table (about 1 s after the release) |
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
intervals by game (`bootstrap_by_game`; the slope resamples use `analytics.eval.calibration_slope_fast`, a Newton
solve of the same regression); fits run under `analytics.eval.fit_threads(1)`; paired
bootstrap by game of candidate 3 against 1 and 2 against 1 on log loss (`paired_bootstrap_by_game`). Rows whose
`heldout` flag is true, whose game is under `heldout/`, or whose game id carries a held-out token of
`sportvu/splits.json` (or the CLE arena; `analytics.eval.heldout_mask`) are set aside before any fit; `analytics.eval.refuse_heldout` runs on the
training game ids and again inside every fit. Held-out rows are scored once, by the selected candidate alone, refit
on every training row; no other candidate is refit or scores them. Fewer games than folds: leave-one-game-out;
one training game: the trainer stops.

## 5. Output table: `xfg_predictions`

`OUT_ROOT/analytics/xfg/<season>/xfg_predictions.parquet`, one row per attempt of the input (training rows and
held-out rows).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int64 | play-by-play event index of the attempt (join key to shots and shot_frames); -1 for an attempt that matched no shot_frames row (its location features and predictions are still present) |
| `player_id` | Int64, nullable | the shooter |
| `p_location` | float32, nullable | out-of-fold probability from `location_logistic` on a training row; on a held-out row NaN unless `location_logistic` is the selected candidate (then equal to `p_selected`) |
| `p_contest` | float32, nullable | the same for `contest_logistic` |
| `p_selected` | float32 | probability from the selected candidate (named in the report): out-of-fold on a training row, from the one refit on a held-out row |
| `made` | bool | the target, carried for scoring downstream (never a feature) |
| `fold` | int16 | the out-of-fold index (>= 0) of a training row; -1 for a held-out row; -2 is reserved for an in-sample row (this trainer has no one-game in-sample mode and stops with fewer than 2 training games, so -2 is never written here; the convention is shared with analytics/train_xfg.py's in-sample mode) |
| `heldout` | bool | true for a held-out row (scored once by the selected candidate refit on all training rows, never fitted on) |

Held-out rows are scored exactly once: only the selected candidate is refit, so the other two columns are NaN there.
`OUT_ROOT` defaults to `DATA_ROOT` under `--data`; under `--fixtures` the table is written only with `--out`.

Held-out membership and the `heldout` column (interface with the matchups lane and any consumer): a row is held out
when its game is under `heldout/` on disk, when the collector wrote `heldout` true, or when its game id carries a
held-out token of `sportvu/splits.json` or ends in the CLE arena (`.at.CLE`), the by-id rule of
`analytics.eval.is_heldout_game`. The three are reconciled when the
inputs are loaded, so the `heldout` column of this table is authoritative: a consumer that decides training rows
must read it (and `fold` -1) rather than re-derive membership from the id alone. For that to be consistent, every
held-out game must also be expressible by id: either a splits.json token or the CLE arena, which is how
`sportvu/splits.json` defines the held-out sets today; a game held out only by flag or directory would be invisible to
an id-only check, and the contract requires that such a game never exists.

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

The report `reports/xfg.md` in the repository is the real-data run of stage 5 (training folds only, held-out not
scored); the frozen settings are at the end of this page. The real numbers need, on the Mac: the 2015-16 shots
parquet, the shot_frames of every game with frames and play-by-play (the `analytics.shot_frames` driver over the
Drive manifest) and the frames themselves. Open questions only real data can answer: the share of attempts whose
release frame localises (70 percent on the one smoke game), the match rate of basketball-reference attempts to
shot_frames rows by clock, and whether the contest features and the prior beat the location baseline under the
paired bootstrap.

## Frozen for the held-out run (freeze-v1)

Frozen from the code on branch track-fix and the committed reports/xfg.md and reports/xfg_baseline.md. Every number
below is out-of-fold on the training games, from runs with `--skip-heldout`; no held-out number has been read.

**Selected model:** `hgb_contest_prior` (lowest out-of-fold log loss of the three candidates).

| setting (analytics/xfg/train.py, features.py, validate.py) | frozen value |
| --- | --- |
| booster | `HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, min_samples_leaf=10, l2_regularization=1.0, early_stopping=False, random_state=seed)` on `FEATURES` plus `shooter_prior` |
| calibration | isotonic on inner out-of-fold predictions, `INNER_FOLDS = 3`, output bounded to [0.02, 0.98] (`ISO_CLIP`) |
| shooter prior | `PRIOR_M = 20` pseudo-attempts toward the zone rate (rim under 4 ft, paint under 14 ft, mid, corner3, above3), out of fold by game |
| other candidates | `location_logistic` (analytics.train_xfg), `contest_logistic` (median imputation, standardisation, C = 1, max_iter 2000) |
| features | 25 Hz; speed over 10 frames (0.4 s); touch cap 250 frames; touch radius 2.5 ft; dribble ball height 2 ft; contest radius 6 ft; a basketball-reference attempt joins shot_frames by period and nearest clock within 1 s |
| cleaning | ball inside 0 to 94 by 0 to 50 ft, players within 5 ft of it; game clock 0 to 720 s; shot clock 0 to 24 s or null; shots loc_x -250 to 250 and loc_y -55 to 420 tenths of a foot, period at least 1, made not null; duplicates dropped |

Rules the held-out run applies: selection by lowest out-of-fold log loss (ties to the earlier candidate; ECE under
0.02 is reported, never a selection rule); 5 grouped folds by `sportvu_game`, seed 0; 200 game bootstrap resamples,
seed 0 (run_all passes no `--bootstrap`). Held out: a game under `heldout/`, flagged `heldout`, or carrying a
held-out token of sportvu/splits.json, which covers every game at CLE and 12.16.2015.PHX.at.GSW (18 games, 2939
attempts set aside in the training run; the tracking lanes count 17 because 01.23.2016.CHI.at.CLE has
basketball-reference shots but no usable tracking, its SportVU source is broken, reports/sportvu_gaps.md). Only the selected candidate is refit on every training row and scores
them once; with two or more held-out games its held-out block carries bootstrap intervals by game.

Out-of-fold headline (101853 attempts, 607 games): log loss 0.6294 [0.6277, 0.6316], Brier 0.2209 [0.2201, 0.2219],
ECE 0.0029 [0.0021, 0.0061], calibration slope 1.0001 [0.9760, 1.0249]; paired against `location_logistic`
-0.0370 [-0.0385, -0.0354].

**xFG baseline (analytics/train_xfg.py).** Selected: `logistic`, fixed (the constant is reported, never scored on
held-out rows): StandardScaler then `LogisticRegression(C=1.0, max_iter=1000)` on distance_ft, angle_rad, is_three
(shot_type 3PT, else 22 ft or more), is_corner (|loc_x| over 220 and loc_y under 90 tenths). 5 grouped folds by
`sportvu_game`, seed 0, 200 game resamples. Held out: `heldout` true or a held-out token of sportvu/splits.json
(CLE arena included); scored once by the logistic model refit on every training row, point estimates only.
Out-of-fold headline (102024 attempts, 607 games): log loss 0.6665 [0.6652, 0.6681], Brier 0.2369 [0.2363, 0.2377],
ECE 0.0400 [0.0369, 0.0429], slope 0.9979 [0.9643, 1.0234]; constant 0.6879 [0.6872, 0.6885]. That run read the
training shotchart only (0 held-out rows); stage 8 adds the held-out file.

Stage 8 commands:

    python3 run_all.py --data "$DATA_ROOT" --ext xfg rebounds matchups gravity ghosting embeddings epv decisions lineups winprob load --continue-on-error
    python3 -m analytics.train_xfg --shots "$DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet" "$DATA_ROOT/collect/bref_shots/heldout/shotchart/2015-16.parquet" --report reports/xfg_baseline.md

run_all.step_argv gives `analytics.xfg.train` the argv `--data DATA_ROOT --seed 0 --report reports/xfg.md` (DATA_ROOT
resolved to an absolute path, the report under the working directory); validate and features get
`--data DATA_ROOT --seed 0`.
