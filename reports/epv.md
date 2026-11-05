# EPV: expected possession value from 1 Hz states (analytics/epv)

Written 2026-11-05T08:59:43Z by `python -m analytics.epv.train --fixtures --games 8 --folds 5 --bootstrap 200 --report reports/epv.md --out /tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/epv_out` (fixture output, not a result).

**Every number below is fixture output, not a result**: the inputs are the synthetic season of tests/_fx_epv.py, built to exercise the pipeline, and say nothing about basketball.

## Inputs

| table | rows in | schema problems | duplicates dropped | rows dropped | rows out |
| --- | --- | --- | --- | --- | --- |
| possessions | 320 | 0 | 0 | 0 | 320 |
| states_1hz | 3995 | 0 | 0 | 0 | 3995 |
| shot_frames | 317 | 0 | 0 | 0 | 317 |
| rosters | 112 | 0 | 0 | 0 | 112 |
| shots | 317 | 0 | 0 | 0 | 317 |

Training: 2969 states, 240 possessions, 6 game(s) (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held-out, set aside before any fit: 1026 states from 2 game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked the 6 training game ids against sportvu/splits.json; `analytics.schema.assert_time_forward` passed on every feature frame (the only column source is `schema.feature_allowlist("states_1hz")`; points_to_end and possession_end_kind never enter).
Training mean points per state: 1.2529; P(points > 0): 0.5699.

## Protocol

Features (25): ball_x, ball_y, ball_z, ball_dist_basket, handler_dist_basket, handler_angle, handler_nearest_def_dist_ft, off_dist_1, off_dist_2, off_dist_3, off_dist_4, off_dist_5, off_def_1, off_def_2, off_def_3, off_def_4, off_def_5, off_hull_area, shot_clock_s, elapsed_in_possession_s, ball_dist_min_so_far, n_handler_switches, handler_fg_prior, handler_zone_value, lineup_tov_prior.
Priors come from the training games of each fold only and are out-of-fold for the training rows (a game's own shots and possessions are left out of its prior); the location-only xFG (analytics.features_xfg) at the handler's position is the shrinkage target of the shooter prior.
Candidates: constant (training mean), ridge (median imputation, standardisation, alpha 1), HistGradientBoostingRegressor (100 iterations, learning rate 0.05, 8 leaves, min 50 per leaf, fixed in advance, never tuned) with isotonic calibration of P(points > 0) fitted on inner out-of-fold predictions (3 grouped folds), and under the gate an MLPRegressor over the last 5 states' features.
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates by elapsed-time bucket (out-of-fold)

| bucket (s) | model | n | Brier P(points > 0) | MSE expected points | calibration slope |
| --- | --- | --- | --- | --- | --- |
| all | constant | 2969 | 0.2462 [0.2426, 0.2497] | 1.2802 [1.2373, 1.3299] | n/a |
| all | ridge | 2969 | 0.2476 [0.2447, 0.2506] | 1.2976 [1.2573, 1.3308] | -1.147 |
| all | hgb | 2969 | 0.2468 [0.2429, 0.2506] | 1.3525 [1.3243, 1.3847] | -0.249 |
| all | mlp | 2969 | 0.2473 [0.2435, 0.2505] | 2.2361 [2.0533, 2.4140] | 0.011 |
| 0-4 | constant | 961 | 0.2489 [0.2442, 0.2544] | 1.2855 [1.2546, 1.3316] | n/a |
| 0-4 | ridge | 961 | 0.2504 [0.2459, 0.2552] | 1.3162 [1.2604, 1.3590] | -1.565 |
| 0-4 | hgb | 961 | 0.2482 [0.2437, 0.2541] | 1.3379 [1.2918, 1.3671] | 0.164 |
| 0-4 | mlp | 961 | 0.2520 [0.2462, 0.2569] | 2.4111 [2.0382, 2.8034] | -1.373 |
| 4-10 | constant | 1337 | 0.2462 [0.2418, 0.2509] | 1.2864 [1.2493, 1.3352] | n/a |
| 4-10 | ridge | 1337 | 0.2476 [0.2438, 0.2515] | 1.3086 [1.2734, 1.3396] | -1.641 |
| 4-10 | hgb | 1337 | 0.2483 [0.2432, 0.2537] | 1.3690 [1.3392, 1.4042] | -0.657 |
| 4-10 | mlp | 1337 | 0.2478 [0.2429, 0.2528] | 2.1378 [1.9931, 2.2951] | -1.327 |
| 10-24 | constant | 671 | 0.2424 [0.2387, 0.2466] | 1.2602 [1.1744, 1.3415] | n/a |
| 10-24 | ridge | 671 | 0.2438 [0.2391, 0.2504] | 1.2490 [1.1950, 1.3115] | -0.743 |
| 10-24 | hgb | 671 | 0.2417 [0.2387, 0.2466] | 1.3404 [1.2896, 1.3907] | 0.501 |
| 10-24 | mlp | 671 | 0.2397 [0.2371, 0.2426] | 2.1812 [1.9110, 2.3826] | 0.980 |

Selected: **constant** (lowest out-of-fold MSE among the three candidates).

## Calibration of P(points > 0), constant model (out-of-fold)

| bucket (s) | bin | n | mean p | observed rate |
| --- | --- | --- | --- | --- |
| 0-4 | 0.4 to 0.6 | 961 | 0.566 | 0.549 |
| 4-10 | 0.4 to 0.6 | 1337 | 0.565 | 0.571 |
| 10-24 | 0.4 to 0.6 | 671 | 0.565 | 0.596 |

## Gradient boosting by shot-clock bucket and the gate

| shot clock (s) | calibration slope |
| --- | --- |
| 14-24 | -0.461 |
| 7-14 | 1.453 |
| 0-7 | n/a |

MSE lift of gradient boosting over the constant: -5.6 percent (1.3525 vs 1.2802). Slopes outside 0.9 to 1.1: 14-24, 7-14, 0-7.
Gate decision: the sequence model (MLPRegressor over the last 5 states, standing in for a GRU) was TRAINED.
Paired bootstrap by game of MSE(hgb) - MSE(mlp): -0.8836 [-1.0629, -0.6749]; the sequence model is DROPPED (no significant gain over gradient boosting).

## Shuffled-target check

Possession points permuted across possessions of the training games, gradient boosting refit out-of-fold: MSE 1.3277 vs constant 1.2811, lift -3.6 percent (chance is at or below 0; the real lift above is -5.6 percent).

## Held-out games, scored once (never fitted or tuned on)

1026 states from 2 held-out game(s): 11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW; the pipeline was refit on all 2969 training states.
Calibration slope per bucket against the 0.9 to 1.1 target is reported, not tuned on.

| bucket (s) | model | n | Brier P(points > 0) | MSE expected points | calibration slope |
| --- | --- | --- | --- | --- | --- |
| all | constant | 1026 | 0.2605 [0.2579, 0.2632] | 1.3791 [1.3057, 1.4584] | n/a |
| all | ridge | 1026 | 0.2645 [0.2637, 0.2654] | 1.4291 [1.3679, 1.4952] | 0.039 |
| all | hgb | 1026 | 0.2642 [0.2601, 0.2680] | 1.4641 [1.4459, 1.4838] | -0.221 |
| all | mlp | 1026 | 0.2625 [0.2619, 0.2632] | 2.1367 [2.1354, 2.1382] | -0.334 |
| 0-4 | constant | 319 | 0.2599 [0.2544, 0.2654] | 1.3819 [1.3118, 1.4516] | n/a |
| 0-4 | ridge | 319 | 0.2588 [0.2536, 0.2640] | 1.3767 [1.2675, 1.4852] | 0.103 |
| 0-4 | hgb | 319 | 0.2605 [0.2577, 0.2633] | 1.4109 [1.3894, 1.4325] | -0.183 |
| 0-4 | mlp | 319 | 0.2601 [0.2533, 0.2669] | 2.0745 [1.8093, 2.3380] | -1.235 |
| 4-10 | constant | 447 | 0.2585 [0.2536, 0.2633] | 1.3802 [1.2999, 1.4608] | n/a |
| 4-10 | ridge | 447 | 0.2603 [0.2582, 0.2624] | 1.4108 [1.3710, 1.4508] | -0.720 |
| 4-10 | hgb | 447 | 0.2673 [0.2626, 0.2719] | 1.5335 [1.5315, 1.5356] | -0.565 |
| 4-10 | mlp | 447 | 0.2590 [0.2542, 0.2637] | 2.0576 [1.9444, 2.1713] | -0.574 |
| 10-24 | constant | 260 | 0.2646 [0.2600, 0.2679] | 1.3738 [1.3081, 1.4636] | n/a |
| 10-24 | ridge | 260 | 0.2789 [0.2738, 0.2826] | 1.5248 [1.4698, 1.5999] | 0.093 |
| 10-24 | hgb | 260 | 0.2634 [0.2585, 0.2670] | 1.4100 [1.3260, 1.5245] | 0.770 |
| 10-24 | mlp | 260 | 0.2717 [0.2568, 0.2826] | 2.3491 [1.7804, 2.7662] | -0.224 |

## Status

Fixture output, not a result. The real run of this protocol on the owner's Mac, where the full data lives, is

    python -m analytics.epv.train --data /path/to/DATA_ROOT --report reports/epv.md --folds 5 --seed 0 --bootstrap 1000 --out /path/to/DATA_ROOT

which expects DATA_ROOT/analytics/states_1hz/<season>/<game>.parquet (build it first with `python -m analytics.epv.states --data DATA_ROOT` when absent), DATA_ROOT/analytics/possessions, rosters, and DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet, and writes the epv_states and epv_possessions tables of docs/contract/epv.md under OUT_ROOT/analytics/epv/<season>/.
