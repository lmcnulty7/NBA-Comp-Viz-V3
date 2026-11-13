# xFG upgrade: shot quality with contest features (analytics/xfg)

Written 2026-11-12T21:43:04Z by `python -m analytics.xfg.train --data $DATA_ROOT --seed 0 --report reports/xfg.md`.

**Real data, stage 8 held-out run of group 1 (tag freeze-v1, 2026-11-12): the held-out games were scored exactly once, by the model frozen at freeze-v1.** Every out-of-fold number below is identical to the frozen training-fold run; nothing was refit, tuned or reselected after the held-out numbers were seen.

## Stage 8: held-out next to out-of-fold (freeze-v1)

Selected model hgb_contest_prior, refit on all 101853 training attempts (607 games), scored once on 2939 attempts
from 18 held-out games. Intervals: bootstrap by game (200 resamples, seed 0).

| metric | out-of-fold (607 games) | held-out (18 games) |
| --- | --- | --- |
| log loss | 0.6294 [0.6277, 0.6316] | 0.6232 [0.6132, 0.6327] |
| Brier | 0.2209 [0.2201, 0.2219] | 0.2187 [0.2142, 0.2233] |
| ECE (10 bins) | 0.0029 [0.0021, 0.0061] | 0.0189 [0.0143, 0.0347] |
| calibration slope | 1.0001 [0.9757, 1.0245] | 1.0896 [0.9850, 1.1978] |

Where held-out is worse: calibration. The held-out ECE is about six times the out-of-fold ECE and its interval lies
entirely above the out-of-fold one; the slope of 1.09 says the model is under-confident on these games, although its
interval still contains 1. Log loss and Brier are slightly better on held-out (the held-out make rate is 0.460 against
0.449 in training), with intervals that overlap the out-of-fold ones.

## Inputs and rows

| table | rows in | rows out | dropped missing | dropped range | dropped duplicate | schema problems |
| --- | --- | --- | --- | --- | --- | --- |
| shot_frames | 104574 | 104574 | 0 | 0 | 0 | 0 |
| rosters | 13274 | 13272 | 2 | 0 | 0 | 1 |
| possessions | 119161 | 119160 | 0 | 1 | 0 | 0 |
| shots | 104967 | 104792 | 0 | 92 | 83 | 0 |
| events | 279955 | 279955 | 0 | 0 | 0 | 0 |

Training rows: 101853 attempts from 607 game(s), 62387 with contest features (has_contest 1), make rate 0.4487.
Held-out rows set aside before any fit: 2939 attempts from 18 game(s) (01.02.2016.ORL.at.CLE, 01.04.2016.TOR.at.CLE, 01.18.2016.GSW.at.CLE, 01.23.2016.CHI.at.CLE, 10.30.2015.MIA.at.CLE, 11.04.2015.NYK.at.CLE, 11.06.2015.PHI.at.CLE, 11.08.2015.IND.at.CLE, 11.10.2015.UTA.at.CLE, 11.19.2015.MIL.at.CLE, 11.21.2015.ATL.at.CLE, 11.23.2015.ORL.at.CLE, 12.01.2015.WAS.at.CLE, 12.08.2015.POR.at.CLE, 12.16.2015.PHX.at.GSW, 12.17.2015.OKC.at.CLE, 12.20.2015.PHI.at.CLE, 12.23.2015.NYK.at.CLE).
Guard: `analytics.eval.refuse_heldout` checked 607 training game ids against sportvu/splits.json, and runs again on every fit; `schema.assert_time_forward` passed on the feature frame; `assert_no_leak` on every design matrix.

## Protocol

Features: distance_ft, angle_rad, is_three, is_corner, closest_defender_dist_ft, closest_defender_closing_speed_ftps, n_defenders_within_6ft, shooter_speed_ftps, touch_time_s, dribbles_proxy, shot_clock_s, has_contest, period, clock_s; the boosting candidate adds `shooter_prior`.
Candidates: location_logistic (analytics.train_xfg, four location features), contest_logistic (median imputation, standardisation, C = 1), hgb_contest_prior (HistGradientBoostingClassifier {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 10, 'l2_regularization': 1.0, 'early_stopping': False}, isotonic calibration on inner out-of-fold predictions, 3 inner folds, output bounded to [0.02, 0.98]).
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0; the slope resamples use the Newton solve `analytics.eval.calibration_slope_fast`, the point estimate `analytics.eval.calibration_slope`). Selection: lowest out-of-fold log loss. ECE target: under 0.02.

## Candidate table (out-of-fold, 101853 attempts, 607 games)

| candidate | log loss | Brier | ECE (10 bins) | ECE under 0.02 | calibration slope | selected |
| --- | --- | --- | --- | --- | --- | --- |
| location_logistic | 0.6665 [0.6651, 0.6681] | 0.2369 [0.2362, 0.2377] | 0.0400 [0.0368, 0.0429] | no | 0.9979 [0.9634, 1.0236] |  |
| contest_logistic | 0.6552 [0.6538, 0.6571] | 0.2316 [0.2309, 0.2325] | 0.0334 [0.0302, 0.0362] | no | 0.9972 [0.9685, 1.0200] |  |
| hgb_contest_prior | 0.6294 [0.6277, 0.6316] | 0.2209 [0.2201, 0.2219] | 0.0029 [0.0021, 0.0061] | yes | 1.0001 [0.9760, 1.0249] | yes |

Selected: **hgb_contest_prior** (lowest out-of-fold log loss).

Paired bootstrap by game, log loss difference (negative favours the first):

| pair | diff | lo | hi | interval excludes 0 |
| --- | --- | --- | --- | --- |
| hgb_contest_prior vs location_logistic | -0.0370 | -0.0385 | -0.0354 | yes |
| contest_logistic vs location_logistic | -0.0113 | -0.0122 | -0.0104 | yes |

## Calibration

### location_logistic (out-of-fold)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 710 | 0.2868 | 0.3183 |
| 3 | 0.30 | 0.40 | 42193 | 0.3506 | 0.3702 |
| 4 | 0.40 | 0.50 | 26077 | 0.4546 | 0.4337 |
| 5 | 0.50 | 0.60 | 22820 | 0.5450 | 0.4797 |
| 6 | 0.60 | 0.70 | 10053 | 0.6374 | 0.7557 |
| 7 | 0.70 | 0.80 | 0 | - | - |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

### contest_logistic (out-of-fold)

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 124 | 0.1761 | 0.1210 |
| 2 | 0.20 | 0.30 | 9425 | 0.2788 | 0.3132 |
| 3 | 0.30 | 0.40 | 33341 | 0.3504 | 0.3677 |
| 4 | 0.40 | 0.50 | 25823 | 0.4460 | 0.4132 |
| 5 | 0.50 | 0.60 | 19504 | 0.5440 | 0.5012 |
| 6 | 0.60 | 0.70 | 10379 | 0.6476 | 0.7006 |
| 7 | 0.70 | 0.80 | 2456 | 0.7437 | 0.8359 |
| 8 | 0.80 | 0.90 | 670 | 0.8386 | 0.8776 |
| 9 | 0.90 | 1.00 | 131 | 0.9305 | 0.8702 |

### hgb_contest_prior (out-of-fold, selected)

101853 rows, 607 games, base rate 0.4487, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6294 | 0.6277 | 0.6316 |
| Brier | 0.2209 | 0.2201 | 0.2219 |
| ECE | 0.0029 | 0.0021 | 0.0061 |
| calibration slope | 1.0001 | 0.9757 | 1.0245 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 257 | 0.0710 | 0.0973 |
| 1 | 0.10 | 0.20 | 1879 | 0.1520 | 0.1474 |
| 2 | 0.20 | 0.30 | 6566 | 0.2703 | 0.2740 |
| 3 | 0.30 | 0.40 | 41662 | 0.3646 | 0.3631 |
| 4 | 0.40 | 0.50 | 28514 | 0.4377 | 0.4402 |
| 5 | 0.50 | 0.60 | 8611 | 0.5472 | 0.5397 |
| 6 | 0.60 | 0.70 | 6173 | 0.6449 | 0.6528 |
| 7 | 0.70 | 0.80 | 1154 | 0.7475 | 0.7374 |
| 8 | 0.80 | 0.90 | 3357 | 0.8660 | 0.8668 |
| 9 | 0.90 | 1.00 | 3680 | 0.9454 | 0.9451 |

## Held-out score (scored once by the selected candidate refit on all 101853 training rows)

2939 attempts from 18 held-out game(s): 01.02.2016.ORL.at.CLE, 01.04.2016.TOR.at.CLE, 01.18.2016.GSW.at.CLE, 01.23.2016.CHI.at.CLE, 10.30.2015.MIA.at.CLE, 11.04.2015.NYK.at.CLE, 11.06.2015.PHI.at.CLE, 11.08.2015.IND.at.CLE, 11.10.2015.UTA.at.CLE, 11.19.2015.MIL.at.CLE, 11.21.2015.ATL.at.CLE, 11.23.2015.ORL.at.CLE, 12.01.2015.WAS.at.CLE, 12.08.2015.POR.at.CLE, 12.16.2015.PHX.at.GSW, 12.17.2015.OKC.at.CLE, 12.20.2015.PHI.at.CLE, 12.23.2015.NYK.at.CLE. Never fitted or tuned on.

| candidate | log loss | Brier | ECE (10 bins) | calibration slope |
| --- | --- | --- | --- | --- |
| hgb_contest_prior | 0.6232 | 0.2187 | 0.0189 | 1.0896 |

Point estimates, with the game bootstrap below.

### hgb_contest_prior (held-out, scored once)

2939 rows, 18 games, base rate 0.4597, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6232 | 0.6132 | 0.6327 |
| Brier | 0.2187 | 0.2142 | 0.2233 |
| ECE | 0.0189 | 0.0143 | 0.0347 |
| calibration slope | 1.0896 | 0.9850 | 1.1978 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 5 | 0.0599 | 0.0000 |
| 1 | 0.10 | 0.20 | 47 | 0.1609 | 0.0851 |
| 2 | 0.20 | 0.30 | 118 | 0.2655 | 0.2373 |
| 3 | 0.30 | 0.40 | 1128 | 0.3653 | 0.3759 |
| 4 | 0.40 | 0.50 | 858 | 0.4291 | 0.4126 |
| 5 | 0.50 | 0.60 | 282 | 0.5375 | 0.5106 |
| 6 | 0.60 | 0.70 | 248 | 0.6424 | 0.6653 |
| 7 | 0.70 | 0.80 | 42 | 0.7500 | 0.8095 |
| 8 | 0.80 | 0.90 | 83 | 0.8845 | 0.9518 |
| 9 | 0.90 | 1.00 | 128 | 0.9362 | 0.9297 |

## Output

xfg_predictions (docs/contract/xfg.md): OUT_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet

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
