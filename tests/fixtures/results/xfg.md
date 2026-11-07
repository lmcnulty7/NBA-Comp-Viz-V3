# xFG upgrade: shot quality with contest features (analytics/xfg) (fixture output, not a result)

Written 2026-11-07T08:40:33Z by `python -m analytics.xfg.train --fixtures --report reports/xfg.md --folds 5 --seed 0 --bootstrap 200`.

**Fixture output, not a result.** Every number below comes from the synthetic season of analytics.fixtures (no real tracking, play-by-play or shot data). The real run is the command in the last section.

## Inputs and rows (fixture output, not a result)

| table | rows in | rows out | dropped missing | dropped range | dropped duplicate | schema problems |
| --- | --- | --- | --- | --- | --- | --- |
| shot_frames | 326 | 326 | 0 | 0 | 0 | 0 |
| rosters | 112 | 112 | 0 | 0 | 0 | 0 |
| possessions | 320 | 320 | 0 | 0 | 0 | 0 |
| shots | 326 | 326 | 0 | 0 | 0 | 0 |
| events | 566 | 566 | 0 | 0 | 0 | 0 |
| frames | 971060 | 971060 | 0 | 0 | 0 | 0 |

Training rows: 243 attempts from 6 game(s), 243 with contest features (has_contest 1), make rate 0.5226.
Held-out rows set aside before any fit: 83 attempts from 2 game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids against sportvu/splits.json, and runs again on every fit; `schema.assert_time_forward` passed on the feature frame; `assert_no_leak` on every design matrix.

## Protocol (fixture output, not a result)

Features: distance_ft, angle_rad, is_three, is_corner, closest_defender_dist_ft, closest_defender_closing_speed_ftps, n_defenders_within_6ft, shooter_speed_ftps, touch_time_s, dribbles_proxy, shot_clock_s, has_contest, period, clock_s; the boosting candidate adds `shooter_prior`.
Candidates: location_logistic (analytics.train_xfg, four location features), contest_logistic (median imputation, standardisation, C = 1), hgb_contest_prior (HistGradientBoostingClassifier {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 10, 'l2_regularization': 1.0, 'early_stopping': False}, isotonic calibration on inner out-of-fold predictions, 3 inner folds, output bounded to [0.02, 0.98]).
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0; the slope resamples use the Newton solve `analytics.eval.calibration_slope_fast`, the point estimate `analytics.eval.calibration_slope`). Selection: lowest out-of-fold log loss. ECE target: under 0.02.

## Candidate table (out-of-fold, 243 attempts, 6 games) (fixture output, not a result)

| candidate | log loss | Brier | ECE (10 bins) | ECE under 0.02 | calibration slope | selected |
| --- | --- | --- | --- | --- | --- | --- |
| location_logistic | 0.6843 [0.6653, 0.7095] | 0.2458 [0.2365, 0.2581] | 0.0525 [0.0321, 0.1496] | no | 0.6735 [0.1577, 1.2453] | yes |
| contest_logistic | 0.7136 [0.6751, 0.7624] | 0.2585 [0.2411, 0.2805] | 0.0876 [0.0548, 0.1785] | no | 0.2728 [-0.0334, 0.7217] |  |
| hgb_contest_prior | 0.7253 [0.6893, 0.7878] | 0.2583 [0.2476, 0.2760] | 0.1115 [0.0534, 0.1947] | no | 0.1191 [-0.1101, 0.5335] |  |

Selected: **location_logistic** (lowest out-of-fold log loss).

Paired bootstrap by game, log loss difference (negative favours the first):

| pair | diff | lo | hi | interval excludes 0 |
| --- | --- | --- | --- | --- |
| hgb_contest_prior vs location_logistic | 0.0409 | 0.0084 | 0.0779 | yes |
| contest_logistic vs location_logistic | 0.0292 | 0.0105 | 0.0548 | yes |

## Calibration (fixture output, not a result)

### location_logistic (out-of-fold, selected) (fixture output, not a result)

243 rows, 6 games, base rate 0.5226, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6843 | 0.6653 | 0.7095 |
| Brier | 0.2458 | 0.2365 | 0.2581 |
| ECE | 0.0525 | 0.0321 | 0.1496 |
| calibration slope | 0.6735 | 0.1576 | 1.2441 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 2 | 0.2757 | 0.0000 |
| 3 | 0.30 | 0.40 | 28 | 0.3588 | 0.3929 |
| 4 | 0.40 | 0.50 | 58 | 0.4473 | 0.5172 |
| 5 | 0.50 | 0.60 | 79 | 0.5566 | 0.5063 |
| 6 | 0.60 | 0.70 | 69 | 0.6404 | 0.5942 |
| 7 | 0.70 | 0.80 | 7 | 0.7116 | 0.7143 |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

### contest_logistic (out-of-fold) (fixture output, not a result)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 1 | 0.0274 | 0.0000 |
| 1 | 0.10 | 0.20 | 1 | 0.1053 | 1.0000 |
| 2 | 0.20 | 0.30 | 8 | 0.2479 | 0.2500 |
| 3 | 0.30 | 0.40 | 34 | 0.3564 | 0.5294 |
| 4 | 0.40 | 0.50 | 59 | 0.4527 | 0.4915 |
| 5 | 0.50 | 0.60 | 53 | 0.5544 | 0.5283 |
| 6 | 0.60 | 0.70 | 54 | 0.6428 | 0.5741 |
| 7 | 0.70 | 0.80 | 27 | 0.7471 | 0.5185 |
| 8 | 0.80 | 0.90 | 6 | 0.8161 | 0.6667 |
| 9 | 0.90 | 1.00 | 0 | - | - |

### hgb_contest_prior (out-of-fold) (fixture output, not a result)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 1 | 0.0622 | 1.0000 |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 3 | 0.2820 | 0.6667 |
| 3 | 0.30 | 0.40 | 9 | 0.3502 | 0.4444 |
| 4 | 0.40 | 0.50 | 63 | 0.4442 | 0.5714 |
| 5 | 0.50 | 0.60 | 109 | 0.5512 | 0.4771 |
| 6 | 0.60 | 0.70 | 50 | 0.6337 | 0.5200 |
| 7 | 0.70 | 0.80 | 2 | 0.7500 | 1.0000 |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 6 | 0.9800 | 0.6667 |

## Held-out score (scored once by the selected candidate refit on all 243 training rows) (fixture output, not a result)

83 attempts from 2 held-out game(s): 11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW. Never fitted or tuned on.

| candidate | log loss | Brier | ECE (10 bins) | calibration slope |
| --- | --- | --- | --- | --- |
| location_logistic | 0.6770 | 0.2415 | 0.0748 | 0.7243 |

Point estimates, with the game bootstrap below.

### location_logistic (held-out, scored once) (fixture output, not a result)

83 rows, 2 games, base rate 0.5663, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6770 | 0.6554 | 0.6991 |
| Brier | 0.2415 | 0.2315 | 0.2518 |
| ECE | 0.0748 | 0.0748 | 0.1806 |
| calibration slope | 0.7243 | 0.3990 | 1.2271 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 1 | 0.2947 | 1.0000 |
| 3 | 0.30 | 0.40 | 8 | 0.3696 | 0.2500 |
| 4 | 0.40 | 0.50 | 19 | 0.4581 | 0.4737 |
| 5 | 0.50 | 0.60 | 27 | 0.5476 | 0.6667 |
| 6 | 0.60 | 0.70 | 28 | 0.6439 | 0.6071 |
| 7 | 0.70 | 0.80 | 0 | - | - |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

## Output (fixture output, not a result)

xfg_predictions (docs/contract/xfg.md): not written (no --out)

## Assumptions (fixture output, not a result)

- The unit is a row of the shots table; shot_frames rows without a shots row are not attempts here.
- shots rows are joined to shot_frames by (sportvu_game, event); a shots table without an event column (basketball-reference rows) is matched by (sportvu_game, period) and nearest clock_s within 1 s, each shot_frames row used once.
- Frame rate 25 Hz: 0.4 s is 10 frames, the touch cap of 10 s is 250 frames; speeds use frame counts, not the clock.
- An attempt with release_frame -1, confidence_flag unmatched, no shot_frames row or a shooter absent from the release frame keeps its location features and gets NaN contest features with has_contest 0; the logistic candidate imputes the training median, the boosting candidate takes NaN natively.
- The shooter prior is the smoothed (m = 20) make rate by (player_id, zone) from the fitting rows only, out-of-fold by game for the rows being fitted; zones are rim (under 4 ft), paint (under 14 ft), mid, corner3, above3.
- Selection is by out-of-fold log loss alone; ECE under 0.02 is a target reported, not a selection rule.
- The isotonic map of the boosting candidate is bounded to [0.02, 0.98] so a sparse calibration set cannot emit exact 0 or 1.
- Held-out rows are scored once, by the selected candidate refit on all training rows; in xfg_predictions their p_location and p_contest are NaN unless that candidate is the selected one.
- Bootstrap intervals are 2.5 and 97.5 percentiles over whole-game resamples; with one held-out game no interval is meaningful.

## The real run on the Mac (fixture output, not a result)

With DATA_ROOT holding sportvu_contract/, collect/bref_shots/ and analytics/{shot_frames,rosters,possessions}/ (docs/contract/xfg.md):

    python -m analytics.xfg.train --data "$DATA_ROOT" --report reports/xfg.md --folds 5 --seed 0 --bootstrap 1000

Validate and build the features first with `python -m analytics.xfg.validate --data "$DATA_ROOT"` and `python -m analytics.xfg.features --data "$DATA_ROOT" --out "$DATA_ROOT"`.

fixture output, not a result
