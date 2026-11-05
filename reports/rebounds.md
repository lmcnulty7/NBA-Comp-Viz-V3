# Rebound probability (analytics/rebounds/train.py)

Written 2026-11-05T08:59:28Z by `python -m analytics.rebounds.train --fixtures --report reports/rebounds.md --bootstrap 200`.

**Fixture output, not a result.** Every number below comes from synthetic games (analytics.fixtures); no real tracking data was read. The real run is the Mac command at the end.

## Inputs and rows

| game | held out | frames in | frames out | misses in | kept | team rebounds (none) | dropped | candidate rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | False | 115020 | 115020 | 15 | 15 | 0 | 0 | 165 |
| 01.18.2016.HLX.at.LRD | False | 123210 | 123210 | 20 | 20 | 0 | 0 | 220 |
| 02.04.2016.GRN.at.QCY | False | 123230 | 123230 | 13 | 13 | 0 | 0 | 143 |
| 12.16.2015.PHX.at.GSW | True | 128550 | 128550 | 25 | 25 | 0 | 0 | 275 |
| 12.15.2015.ABQ.at.GRN | False | 117110 | 117110 | 16 | 16 | 0 | 0 | 176 |
| 11.27.2015.GRN.at.QCY | False | 119730 | 119730 | 17 | 17 | 0 | 0 | 187 |
| 12.12.2015.RNO.at.MOB | False | 121170 | 121170 | 18 | 18 | 0 | 0 | 198 |
| 11.11.2015.HLX.at.CLE | True | 118520 | 118520 | 24 | 24 | 0 | 0 | 264 |

Dropped misses by reason (all games): no_rim_frame 0, no_rebound_event 0, unmatched_named_rebounder 0, offense_unknown 0, rim_frame_not_in_frames 0, too_few_players 0, rebounder_not_on_court 0.
Kept misses by kind: team_rebounds_kept 0, player_rebounds_kept 148, identity_match_null_kept 0, identity_match_false_kept 0, unmatched_flag_kept 0.
Validation (analytics.rebounds.validate): frames 966540 rows in, 966540 out; events 547 rows in, 547 out; rebound_frames 148 rows in, 148 out; shot_frames 317 rows in, 317 out; rosters 112 rows in, 112 out; possessions 320 rows in, 320 out.

Training: 1089 candidate rows, 99 misses, 6 game(s) (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held out, set aside before any fit: 539 rows, 49 misses, 2 game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (11 columns).

## Protocol

Unit: a missed field-goal attempt; one candidate row per player on court at the rim frame plus a none row (team rebound).
Features: distance_to_rim_ft, distance_to_ball_landing_ft, angle_to_rim, speed_toward_rim_ftps, is_offense, nearest_opponent_dist_ft, n_opponents_closer_to_rim, ball_apex_ft, shot_distance_ft, is_none, prior_rate.
Models: prior (smoothed player rate by is_offense, normalised within the miss), logit (binary-logistic approximation of the conditional logit: logistic regression on the candidate rows with the binary target, then a softmax of the linear scores within the miss; the within-miss likelihood itself is not maximised), hgb (HistGradientBoostingClassifier, normalised within the miss).
Priors for the logit and hgb training rows: out of fold over 4 to 5 inner grouped folds of each outer fold's training games (0 means in-sample: one game); test rows take the prior of their fold's training games.
Selection: lowest out-of-fold per-miss multinomial log loss. Top-1 accuracy reported against the plan target of 40 percent.
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates (out-of-fold, 99 misses, 6 game(s))

| model | per-miss log loss | top-1 accuracy |
| --- | --- | --- |
| prior | 2.1141 [2.0150, 2.2294] | 0.1636 [0.1489, 0.1786] |
| logit | 1.9630 [1.7647, 2.1603] | 0.2727 [0.1333, 0.4357] |
| hgb (selected) | 1.8108 [1.5781, 2.1397] | 0.3939 [0.2989, 0.4661] |
| uniform reference | 2.3979 | 0.0909 |

Paired bootstrap by game, hgb minus prior per-miss log loss: -0.3033 [-0.5296, 0.0290]; interval excludes zero: False (negative favours hgb).
Selected: hgb.

## Calibration of the top candidate's probability (hgb, out-of-fold)

Per miss: y is whether the top candidate secured the rebound (a tie at the maximum counts as yes), p is the top candidate's probability.

### top candidate, hgb

99 rows, 6 games, base rate 0.3939, bootstrap by game (n=200, seed=0, 95 percent percentile interval)

| metric | point | lo | hi |
| --- | --- | --- | --- |
| log loss | 0.6637 | 0.6263 | 0.6931 |
| Brier | 0.2355 | 0.2175 | 0.2494 |
| ECE | 0.0793 | 0.0728 | 0.1504 |
| calibration slope | 0.6254 | 0.0296 | 1.0345 |

| bin | lo | hi | n | mean_p | mean_y |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 0.10 | 0 | - | - |
| 1 | 0.10 | 0.20 | 0 | - | - |
| 2 | 0.20 | 0.30 | 28 | 0.2620 | 0.3571 |
| 3 | 0.30 | 0.40 | 33 | 0.3460 | 0.3939 |
| 4 | 0.40 | 0.50 | 20 | 0.4336 | 0.3000 |
| 5 | 0.50 | 0.60 | 14 | 0.5410 | 0.5000 |
| 6 | 0.60 | 0.70 | 3 | 0.6335 | 0.6667 |
| 7 | 0.70 | 0.80 | 1 | 0.7429 | 1.0000 |
| 8 | 0.80 | 0.90 | 0 | - | - |
| 9 | 0.90 | 1.00 | 0 | - | - |

## Held-out games (scored once by the models refit on all 1089 training rows; never fitted or tuned on)

49 misses from 2 held-out game(s): 11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW.

| model | per-miss log loss | top-1 accuracy |
| --- | --- | --- |
| prior | 2.2190 | 0.1469 |
| logit | 2.1964 | 0.2653 |
| hgb | 2.1918 | 0.1633 |

Held-out numbers are point estimates over 2 game(s), scored exactly once. The selection (p_selected = hgb) was fixed on the out-of-fold training folds above before any held-out row was scored; the other rows are shown for the record and select nothing.

## Assumptions taken without asking

- Team rebounds are kept as the none class when a rebound event follows the miss and names no player; every other null identity_match is dropped and counted (the lane spec's literal rule would drop every team rebound, since its identity_match is null by construction).
- The offence of a miss is the section 5 team of the miss event, else the possession's offense_team_id; a miss with neither is dropped.
- The 0.4 s look-back is the frame of the same period nearest 0.4 s of game clock before the rim frame, within 0.2 s; otherwise the speed is NaN (imputed by the median in the logit, native in hgb).
- shot_distance_ft uses the section 17 release_frame when present, else the ball's horizontal distance from the basket 1 s before the rim frame.
- The prior is smoothed with 10 pseudo-opportunities toward the per-side base rate; the none rate is the Laplace team-rebound rate. Nothing was tuned.
- Top-1 accuracy counts a tie at the maximum as 1 over the number tied (expected accuracy under random tie-breaking); the calibration block counts it as a hit.
- Model settings fixed in advance: logistic regression C = 1 after median imputation and standardisation; HistGradientBoostingClassifier with 80 iterations, depth 3, learning rate 0.1, 10 rows per leaf.

## Predictions

No --out given: the rebound_predictions table was not written.

## Status

Fixture output, not a result: a synthetic run that exercises the pipeline end to end. The real run on the Mac, where the full data lives, is

    python -m analytics.rebounds.train --data "<Drive>/nba_harvest" --out "<Drive>/nba_harvest" --report reports/rebounds.md --folds 5 --seed 0 --bootstrap 1000
