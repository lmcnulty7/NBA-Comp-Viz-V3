# xFG upgrade: shot quality with contest features (analytics/xfg) (fixture output, not a result)

Written 2026-11-05T09:04:51Z by `python -m analytics.xfg.train --fixtures --report reports/xfg.md --folds 5 --seed 0 --bootstrap 200`.

**Fixture output, not a result.** Every number below comes from the synthetic season of tests/_fx_xfg.py (no real tracking, play-by-play or shot data). The real run is the command in the last section.

## Inputs and rows

| table | rows in | rows out | dropped missing | dropped range | dropped duplicate | schema problems |
| --- | --- | --- | --- | --- | --- | --- |
| shot_frames | 317 | 317 | 0 | 0 | 0 | 0 |
| rosters | 112 | 112 | 0 | 0 | 0 | 0 |
| possessions | 320 | 320 | 0 | 0 | 0 | 0 |
| shots | 317 | 317 | 0 | 0 | 0 | 0 |
| events | 547 | 547 | 0 | 0 | 0 | 0 |
| frames | 966540 | 966540 | 0 | 0 | 0 | 0 |

Training rows: 231 attempts from 6 game(s), 231 with contest features (has_contest 1), make rate 0.5714.
Held-out rows set aside before any fit: 86 attempts from 2 game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids against sportvu/splits.json, and runs again on every fit; `schema.assert_time_forward` passed on the feature frame; `assert_no_leak` on every design matrix.

## Protocol

Features: distance_ft, angle_rad, is_three, is_corner, closest_defender_dist_ft, closest_defender_closing_speed_ftps, n_defenders_within_6ft, shooter_speed_ftps, touch_time_s, dribbles_proxy, shot_clock_s, has_contest, period, clock_s; the boosting candidate adds `shooter_prior`.
Candidates: location_logistic (analytics.train_xfg, four location features), contest_logistic (median imputation, standardisation, C = 1), hgb_contest_prior (HistGradientBoostingClassifier {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 10, 'l2_regularization': 1.0, 'early_stopping': False}, isotonic calibration on inner out-of-fold predictions, 3 inner folds, output bounded to [0.02, 0.98]).
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0; the slope resamples use the Newton solve `fast_slope`, the point estimate `analytics.eval.calibration_slope`). Selection: lowest out-of-fold log loss. ECE target: under 0.02.

## Candidate table (out-of-fold, 231 attempts, 6 games) (fixture output, not a result)

| candidate | log loss | Brier | ECE (10 bins) | ECE under 0.02 | calibration slope | selected |
| --- | --- | --- | --- | --- | --- | --- |
| location_logistic | 0.6678 [0.6410, 0.7035] | 0.2373 [0.2245, 0.2544] | 0.0484 [0.0361, 0.1004] | no | 0.8326 [0.2079, 1.8906] | yes |
| contest_logistic | 0.6946 [0.6591, 0.7269] | 0.2496 [0.2329, 0.2648] | 0.0762 [0.0749, 0.1562] | no | 0.3888 [0.0935, 0.8062] |  |
| hgb_contest_prior | 0.7299 [0.6949, 0.7886] | 0.2616 [0.2512, 0.2764] | 0.0770 [0.0615, 0.1672] | no | 0.0639 [-0.0092, 0.2589] |  |

Selected: **location_logistic** (lowest out-of-fold log loss).

Paired bootstrap by game, log loss difference (negative favours the first):

| pair | diff | lo | hi | interval excludes 0 |
| --- | --- | --- | --- | --- |
| hgb_contest_prior vs location_logistic | 0.0621 | 0.0098 | 0.1409 | yes |
| contest_logistic vs location_logistic | 0.0268 | -0.0015 | 0.0612 | no |

## Calibration (fixture output, not a result)

### location_logistic (out-of-fold, selected)

231 rows, 6 games, base rate 0.5714, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6678 | 0.6410 | 0.7035 |
| Brier | 0.2373 | 0.2245 | 0.2544 |
| ECE | 0.0484 | 0.0361 | 0.1004 |
| calibration slope | 0.8326 | 0.2079 | 1.8906 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 0 | - | - |
| 3 | 0.30 | 0.40 | 4 | 0.3495 | 0.7500 |
| 4 | 0.40 | 0.50 | 78 | 0.4457 | 0.4359 |
| 5 | 0.50 | 0.60 | 23 | 0.5339 | 0.4348 |
| 6 | 0.60 | 0.70 | 120 | 0.6558 | 0.6917 |
| 7 | 0.70 | 0.80 | 6 | 0.7066 | 0.3333 |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

### contest_logistic (out-of-fold)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 1 | 0.0873 | 0.0000 |
| 1 | 0.10 | 0.20 | 1 | 0.1825 | 0.0000 |
| 2 | 0.20 | 0.30 | 13 | 0.2566 | 0.7692 |
| 3 | 0.30 | 0.40 | 29 | 0.3566 | 0.3793 |
| 4 | 0.40 | 0.50 | 29 | 0.4568 | 0.4828 |
| 5 | 0.50 | 0.60 | 48 | 0.5547 | 0.5417 |
| 6 | 0.60 | 0.70 | 55 | 0.6508 | 0.6545 |
| 7 | 0.70 | 0.80 | 45 | 0.7477 | 0.5778 |
| 8 | 0.80 | 0.90 | 10 | 0.8228 | 0.9000 |
| 9 | 0.90 | 1.00 | 0 | - | - |

### hgb_contest_prior (out-of-fold)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 4 | 0.0200 | 0.2500 |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 0 | - | - |
| 3 | 0.30 | 0.40 | 8 | 0.3900 | 0.6250 |
| 4 | 0.40 | 0.50 | 34 | 0.4568 | 0.6765 |
| 5 | 0.50 | 0.60 | 128 | 0.5551 | 0.5547 |
| 6 | 0.60 | 0.70 | 32 | 0.6318 | 0.5312 |
| 7 | 0.70 | 0.80 | 22 | 0.7413 | 0.5909 |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 3 | 0.9800 | 0.6667 |

## Held-out score (scored once by the selected candidate refit on all 231 training rows) (fixture output, not a result)

86 attempts from 2 held-out game(s): 11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW. Never fitted or tuned on.

| candidate | log loss | Brier | ECE (10 bins) | calibration slope |
| --- | --- | --- | --- | --- |
| location_logistic | 0.7084 | 0.2568 | 0.1554 | 0.7924 |

Point estimates, with the game bootstrap below.

### location_logistic (held-out, scored once)

86 rows, 2 games, base rate 0.4302, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.7084 | 0.6907 | 0.7277 |
| Brier | 0.2568 | 0.2485 | 0.2660 |
| ECE | 0.1554 | 0.1225 | 0.1953 |
| calibration slope | 0.7924 | 0.4839 | 1.0764 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 0 | - | - |
| 3 | 0.30 | 0.40 | 6 | 0.3701 | 0.1667 |
| 4 | 0.40 | 0.50 | 29 | 0.4601 | 0.3448 |
| 5 | 0.50 | 0.60 | 6 | 0.5304 | 0.6667 |
| 6 | 0.60 | 0.70 | 45 | 0.6664 | 0.4889 |
| 7 | 0.70 | 0.80 | 0 | - | - |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

## Output

xfg_predictions (docs/contract/xfg.md): not written (no --out)

## Assumptions

- The unit is a row of the shots table; shot_frames rows without a shots row are not attempts here.
- shots rows are joined to shot_frames by (sportvu_game, event); a shots table without an event column (basketball-reference rows) is matched by (sportvu_game, period) and nearest clock_s within 1 s, each shot_frames row used once.
- Frame rate 25 Hz: 0.4 s is 10 frames, the touch cap of 10 s is 250 frames; speeds use frame counts, not the clock.
- An attempt with release_frame -1, confidence_flag unmatched, no shot_frames row or a shooter absent from the release frame keeps its location features and gets NaN contest features with has_contest 0; the logistic candidate imputes the training median, the boosting candidate takes NaN natively.
- The shooter prior is the smoothed (m = 20) make rate by (player_id, zone) from the fitting rows only, out-of-fold by game for the rows being fitted; zones are rim (under 4 ft), paint (under 14 ft), mid, corner3, above3.
- Selection is by out-of-fold log loss alone; ECE under 0.02 is a target reported, not a selection rule.
- The isotonic map of the boosting candidate is bounded to [0.02, 0.98] so a sparse calibration set cannot emit exact 0 or 1.
- Held-out rows are scored once, by the selected candidate refit on all training rows; in xfg_predictions their p_location and p_contest are NaN unless that candidate is the selected one.
- Bootstrap intervals are 2.5 and 97.5 percentiles over whole-game resamples; with one held-out game no interval is meaningful.

## The real run on the Mac

With DATA_ROOT holding sportvu_contract/, collect/bref_shots/ and analytics/{shot_frames,rosters,possessions}/ (docs/contract/xfg.md):

    python -m analytics.xfg.train --data "$DATA_ROOT" --report reports/xfg.md --folds 5 --seed 0 --bootstrap 1000

Validate and build the features first with `python -m analytics.xfg.validate --data "$DATA_ROOT"` and `python -m analytics.xfg.features --data "$DATA_ROOT" --out "$DATA_ROOT"`.

fixture output, not a result
