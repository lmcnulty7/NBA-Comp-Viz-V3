# EPV: expected possession value from 1 Hz states (analytics/epv)

Written 2026-11-07T08:37:37Z by `python -m analytics.epv.train --fixtures --fixture-games 8 --folds 5 --bootstrap 200 --report reports/epv.md` (fixture output, not a result).

**Every number below is fixture output, not a result**: the inputs are the synthetic season of analytics.fixtures (synth_season, one league), built to exercise the pipeline, and say nothing about basketball.

## Inputs (fixture output, not a result)

| table | rows in | schema problems | duplicates dropped | rows dropped | rows out |
| --- | --- | --- | --- | --- | --- |
| possessions | 320 | 0 | 0 | 0 | 320 |
| states_1hz | 4014 | 0 | 0 | 0 | 4014 |
| shot_frames | 326 | 0 | 0 | 0 | 326 |
| rosters | 112 | 0 | 0 | 0 | 112 |
| shots | 326 | 0 | 0 | 0 | 326 |

Training: 3026 states, 240 possessions, 6 game(s) (02.10.2016.QCY.at.FRS, 03.08.2016.GRN.at.HLX, 11.28.2015.HLX.at.FRS, 12.01.2015.DUL.at.HLX, 12.09.2015.PEO.at.DUL, 12.10.2015.GRN.at.DUL). Held-out, set aside before any fit: 988 states from 2 game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked the 6 training game ids against sportvu/splits.json; `analytics.schema.assert_time_forward` passed on every feature frame (the only column source is `schema.feature_allowlist("states_1hz")`; points_to_end and possession_end_kind never enter).
Training mean points per state: 1.2406; P(points > 0): 0.5585.

## Protocol (fixture output, not a result)

Features (25): ball_x, ball_y, ball_z, ball_dist_basket, handler_dist_basket, handler_angle, handler_nearest_def_dist_ft, off_dist_1, off_dist_2, off_dist_3, off_dist_4, off_dist_5, off_def_1, off_def_2, off_def_3, off_def_4, off_def_5, off_hull_area, shot_clock_s, elapsed_in_possession_s, ball_dist_min_so_far, n_handler_switches, handler_fg_prior, handler_zone_value, lineup_tov_prior.
Priors come from the training games of each fold only. The per-player zone counts and the per-lineup counts are out-of-fold for the training rows (a game's own shots and possessions are left out of its prior); the two league-wide quantities they shrink towards, the location-only xFG (analytics.features_xfg, four parameters) at the handler's position and the training turnover rate, are fitted once on the whole training fold, the row's own game included.
Candidates: constant (training mean), ridge (median imputation, standardisation, alpha 1), HistGradientBoostingRegressor (100 iterations, learning rate 0.05, 8 leaves, min 50 per leaf, fixed in advance, never tuned) with isotonic calibration of P(points > 0) fitted on inner out-of-fold predictions (3 grouped folds), and under the gate an MLPRegressor over the last 5 states' features.
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates by elapsed-time bucket (out-of-fold) (fixture output, not a result)

| bucket (s) | model | n | Brier P(points > 0) | MSE expected points | calibration slope |
| --- | --- | --- | --- | --- | --- |
| all | constant | 3026 | 0.2516 [0.2423, 0.2619] | 1.3341 [1.2365, 1.4472] | n/a |
| all | ridge | 3026 | 0.2541 [0.2448, 0.2661] | 1.3746 [1.2232, 1.5410] | 0.002 |
| all | hgb | 3026 | 0.2543 [0.2428, 0.2688] | 1.4563 [1.3144, 1.6102] | -0.814 |
| all | mlp | 3026 | 0.2519 [0.2427, 0.2635] | 2.2112 [1.9892, 2.3961] | -0.008 |
| 0-4 | constant | 956 | 0.2560 [0.2445, 0.2677] | 1.3618 [1.2846, 1.4541] | n/a |
| 0-4 | ridge | 956 | 0.2592 [0.2478, 0.2725] | 1.4148 [1.2966, 1.5436] | -1.362 |
| 0-4 | hgb | 956 | 0.2576 [0.2447, 0.2714] | 1.4780 [1.3771, 1.6035] | -1.304 |
| 0-4 | mlp | 956 | 0.2550 [0.2442, 0.2662] | 2.0646 [1.9392, 2.1890] | -0.023 |
| 4-10 | constant | 1329 | 0.2525 [0.2424, 0.2627] | 1.3573 [1.2815, 1.4602] | n/a |
| 4-10 | ridge | 1329 | 0.2553 [0.2466, 0.2660] | 1.3933 [1.2750, 1.5176] | -1.711 |
| 4-10 | hgb | 1329 | 0.2573 [0.2440, 0.2712] | 1.5127 [1.3720, 1.6528] | -0.501 |
| 4-10 | mlp | 1329 | 0.2525 [0.2429, 0.2623] | 2.1299 [1.9170, 2.3627] | -0.447 |
| 10-24 | constant | 741 | 0.2440 [0.2309, 0.2622] | 1.2566 [1.1043, 1.4524] | n/a |
| 10-24 | ridge | 741 | 0.2455 [0.2277, 0.2684] | 1.2890 [1.0327, 1.5905] | 0.132 |
| 10-24 | hgb | 741 | 0.2445 [0.2321, 0.2639] | 1.3273 [1.1415, 1.5602] | -1.238 |
| 10-24 | mlp | 741 | 0.2470 [0.2331, 0.2663] | 2.5459 [2.0318, 3.1165] | 0.053 |

| model | in-sample MSE (mean over folds) | out-of-fold MSE |
| --- | --- | --- |
| constant | 1.3088 | 1.3341 |
| ridge | 1.2594 | 1.3746 |
| hgb | 0.9074 | 1.4563 |
| mlp | 0.3794 | 2.2112 |

Selected: **constant** (lowest out-of-fold MSE among the three candidates).
Why the constant: no learner beats the constant out of fold (gradient boosting lift -9.2 percent); the gradient-boosting calibration slopes sit outside 0.9 to 1.1 in 14-24, 7-14; the real lift (-9.2 percent) is no better than the shuffled-target lift (-8.7 percent); the learners overfit (in-sample MSE well under out-of-fold for hgb, mlp: the target is one draw per possession while min_samples_leaf counts states, so a leaf can hold a handful of possessions).

## Calibration of P(points > 0), constant model (out-of-fold) (fixture output, not a result)

| bucket (s) | bin | n | mean p | observed rate |
| --- | --- | --- | --- | --- |
| 0-4 | 0.4 to 0.6 | 637 | 0.550 | 0.579 |
| 0-4 | 0.6 to 0.8 | 319 | 0.603 | 0.426 |
| 4-10 | 0.4 to 0.6 | 893 | 0.549 | 0.601 |
| 4-10 | 0.6 to 0.8 | 436 | 0.603 | 0.454 |
| 10-24 | 0.4 to 0.6 | 482 | 0.550 | 0.639 |
| 10-24 | 0.6 to 0.8 | 259 | 0.603 | 0.548 |

## Gradient boosting by shot-clock bucket and the gate (fixture output, not a result)

| shot clock (s) | calibration slope |
| --- | --- |
| 14-24 | -0.727 |
| 7-14 | -1.449 |

MSE lift of gradient boosting over the constant: -9.2 percent (1.4563 vs 1.3341). Slopes outside 0.9 to 1.1 (n/a counts as outside): 14-24, 7-14.
Gate decision: the sequence model (MLPRegressor over the last 5 states, standing in for a GRU) was TRAINED.
Paired bootstrap by game of MSE(constant) - MSE(mlp), constant being the lowest-MSE candidate: -0.8771 [-1.0400, -0.7132]; the sequence model is DROPPED (no significant gain over the base candidate).

## Shuffled-target check (fixture output, not a result)

Possession points permuted across possessions of the training games, gradient boosting refit out-of-fold: MSE 1.4598 vs constant 1.3428, lift -8.7 percent (chance is at or below 0; the real lift above is -9.2 percent).

## Held-out games, scored once (never fitted or tuned on) (fixture output, not a result)

988 states from 2 held-out game(s): 11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW; the pipeline was refit on all 3026 training states. Only the selected candidate (constant) is scored here; the other candidates never see a held-out row.
Calibration slope per bucket against the 0.9 to 1.1 target is reported, not tuned on. Point estimates only: fewer than 5 held-out games, so no bootstrap interval.

| bucket (s) | model | n | Brier P(points > 0) | MSE expected points | calibration slope |
| --- | --- | --- | --- | --- | --- |
| all | constant | 988 | 0.2426 | 1.2191 | n/a |
| 0-4 | constant | 320 | 0.2432 | 1.2256 | n/a |
| 4-10 | constant | 450 | 0.2425 | 1.2367 | n/a |
| 10-24 | constant | 218 | 0.2422 | 1.1732 | n/a |

## Status (fixture output, not a result)

Fixture output, not a result. The real run of this protocol on the owner's Mac, where the full data lives, is

    python -m analytics.epv.train --data /path/to/DATA_ROOT --report reports/epv.md --folds 5 --seed 0 --bootstrap 1000 --out /path/to/DATA_ROOT

which expects DATA_ROOT/analytics/states_1hz/<season>/<game>.parquet (build it first with `python -m analytics.epv.states --data DATA_ROOT` when absent), DATA_ROOT/analytics/possessions, rosters, and DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet, and writes the epv_states and epv_possessions tables of docs/contract/epv.md under OUT_ROOT/analytics/epv/<season>/.
