# Gravity and spacing (analytics/gravity) (fixture output, not a result)

Written 2026-11-05T08:46:56Z by `python -m analytics.gravity.train --fixtures --report reports/gravity.md --folds 5 --seed 0 --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are synthetic games from `analytics.fixtures` with two planted high-gravity players per game, used to exercise the pipeline. The real run is the Mac command at the end.

## Inputs

| games | training | held-out | player-frames (train) | spacing frames (train) | shots (train) |
| --- | --- | --- | --- | --- | --- |
| 8 | 6 | 2 | 14595 | 2919 | 231 |

Training games: 01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN. Held-out games set aside before any fit: 11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW.
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `schema.assert_time_forward` passed on the feature frame.
Sampling: 2919 frames sampled at 1 Hz inside possessions, 2919 kept (not ten players: 0, no ball: 0, degenerate hull: 0).
Validation counts: 0 schema problem(s), dropped rows {"frames_duplicate_frame_player": 0, "frames_null_xy": 0, "possessions_duplicate": 0, "rosters_duplicate_player": 0}.

## Protocol

Unit: player-frame at 1 Hz inside possessions. Target: nearest-defender distance (ft). Features: x_ft, y_ft, dist_basket_ft, angle_rad, dist_ball_ft, has_ball, ball_dist_basket_ft, skill_zone, skill_three, game_clock_s, shot_clock_s, shot_clock_missing.
Shooting prior: smoothed make rate by zone (K = 25) from training-fold shots only, matched to SportVU ids by name. Gravity sign: `gravity_ft = expected_ft - nearest_def_dist_ft` (positive: defenders closer than location predicts).
Folds: out-of-fold under 5 grouped folds by sportvu_game (seed 0). Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates (out-of-fold MSE, 14595 player-frames, 6 games; variance of y 7.391)

| model | MSE | R2 (1 - MSE / var y) |
| --- | --- | --- |
| constant | 7.394 [6.853, 7.821] | -0.000 |
| ridge | 4.272 [3.927, 4.537] | 0.422 |
| hgb | 2.213 [2.055, 2.378] | 0.701 |

Paired bootstrap by game, MSE(hgb) - MSE(ridge): -2.0595 [-2.1826, -1.8760], excludes zero: True. Selected: **hgb** (HGB only when the interval excludes zero on HGB's side).

## Players (gate: 300 possessions; 0 gated of 84) (fixture output, not a result)

| rank | player_id | possessions | gravity_ft | lo | hi |
| --- | --- | --- | --- | --- | --- |

### Top-decile check (fixture output, not a result)

Known high-gravity players [9000, 9007, 9020, 9027, 9040, 9047, 9080, 9087, 9100, 9107, 9120, 9127]; ranks among 0 gated players {9000: None, 9007: None, 9020: None, 9027: None, 9040: None, 9047: None, 9080: None, 9087: None, 9100: None, 9107: None, 9120: None, 9127: None}; decile cut 0; passed: False.

Within each training game (the check that applies when rosters do not persist across games): 6 of 6 games pass; passed: True.

| game | players | decile cut | ranks of the known players |
| --- | --- | --- | --- |
| 01.18.2016.HLX.at.LRD | 14 | 2 | {9020: 1, 9027: 2} |
| 02.04.2016.GRN.at.QCY | 14 | 2 | {9040: 1, 9047: 2} |
| 02.07.2016.OMA.at.DUL | 14 | 2 | {9000: 1, 9007: 2} |
| 11.27.2015.GRN.at.QCY | 14 | 2 | {9100: 1, 9107: 2} |
| 12.12.2015.RNO.at.MOB | 14 | 2 | {9120: 1, 9127: 2} |
| 12.15.2015.ABQ.at.GRN | 14 | 2 | {9080: 1, 9087: 2} |

Mean out-of-fold gravity (ft) of the known players under each candidate (the skill prior nearly identifies a player, so a flexible model can absorb part of a player's pull into its expectation; the fixture plants a pull of 2 ft):

| model | 9000 | 9007 | 9020 | 9027 | 9040 | 9047 | 9080 | 9087 | 9100 | 9107 | 9120 | 9127 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 2.364 | 2.326 | 2.369 | 2.305 | 2.307 | 2.426 | 2.384 | 2.455 | 2.332 | 2.371 | 2.550 | 2.523 |
| ridge | 1.206 | 1.218 | 1.219 | 1.208 | 1.325 | 1.274 | 1.344 | 1.272 | 1.294 | 1.319 | 1.539 | 1.428 |
| hgb | 0.795 | 0.631 | 0.815 | 0.554 | 0.813 | 0.719 | 0.868 | 0.751 | 0.683 | 0.670 | 0.907 | 0.903 |

### Sign test: teammates' location-only xFG with a high-gravity player on the floor (fixture output, not a result)

Candidates [9000, 9007, 9020, 9027, 9040, 9047, 9080, 9087, 9100, 9107, 9120, 9127] (the fixture's planted high-gravity players). Games with a difference: 0; positive: 0; one-sided binomial p = n/a; mean difference n/a [n/a, n/a] (bootstrap by game).

## Spacing (fixture output, not a result)

Mean hull area 338.2 ft2 (sd 156.3), mean nearest-defender distance 4.47 ft over 2919 training possession-frames.

## Held-out, scored once (fixture output, not a result)

2 held-out game(s), 5010 player-frames, scored once by the hgb model refit on all 14595 training rows: MSE 2.389 (constant 7.667, ridge 4.671, hgb 2.389). Never fitted, tuned or ranked on.

## Outputs

No tables written (pass --out OUT_ROOT).

## Assumptions

- The spec gives gravity as y minus the expectation and also says positive means defenders stay closer; these conflict, and the stated meaning wins: gravity_ft = expected_ft - nearest_def_dist_ft.
- A possession's frames run from start_frame to the next possession's start_frame (exclusive) in the same period, not to end_frame, which the contract marks post-outcome; analytics.possessions makes possessions contiguous so the span is the same.
- The attacked basket per (offense team, period) is the one nearer the mean x of the sampled offensive positions.
- has_ball marks the offensive player nearest the ball (the section 20 handler), with no distance threshold.
- The shooting prior uses zones by distance only (rim, paint, short mid, long mid, three) and K = 25 shots of smoothing; shooters of basketball-reference rows (null player_id) are matched to SportVU ids by name against the game's roster.
- shot_clock_s null is filled with 24 and flagged; frames without a ball, or without exactly five offensive and five defensive players, are dropped and counted.
- A player's interval resamples that player's games; a player seen in one game gets no interval. The player table is built on training frames only.
- The sign test's candidates are the fixture's planted players on fixtures and the top ten gated players on real data; the xFG probability is the location-only logistic of analytics.train_xfg fitted out of fold by game on the training games' shots.
- On-court at a shot means the player has a row in the frame of the same period nearest the shot's clock.
- Fixture runs on analytics.fixtures: every game has a fresh roster (player ids depend on the game seed), so no player reaches the 300-possession gate and the season-wide top-decile check cannot pass by construction; the per-game check is the one that applies there. Its star plays every period, so the on/off sign test finds no off-court shots; the fallback generator (tests/_fx_gravity.py) keeps one roster across the season and sits its star, and both checks run on it.

## Mac command (the real run)

    python -m analytics.gravity.train --data DATA_ROOT --report reports/gravity.md --folds 5 --seed 0 --bootstrap 1000 --min-possessions 300 --out DATA_ROOT

DATA_ROOT holds sportvu_contract/, analytics/possessions/2015-16/ (and heldout/2015-16/), analytics/rosters/, analytics/shot_frames/ and collect/bref_shots/shotchart/2015-16.parquet; the tables land under DATA_ROOT/analytics/gravity/.
