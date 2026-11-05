# Gravity and spacing (analytics/gravity) (fixture output, not a result)

Written 2026-11-05T09:17:51Z by `python -m analytics.gravity.train --fixtures --report reports/gravity.md --folds 5 --seed 0 --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are synthetic games from `analytics.fixtures` with two planted high-gravity players per game, used to exercise the pipeline. The real run is the Mac command at the end.

## Protocol

Unit: player-frame at 1 Hz inside possessions. Target: nearest-defender distance (ft). Features: x_ft, y_ft, dist_basket_ft, angle_rad, dist_ball_ft, has_ball, ball_dist_basket_ft, zone_make_rate, game_clock_s, shot_clock_s, shot_clock_missing.
Shooting prior: the league make rate by zone of the player's position (zones rim, paint, short_mid, long_mid, three), from training-fold shots only; nothing in the features identifies a player. Gravity sign: `gravity_ft = expected_ft - nearest_def_dist_ft` (positive: defenders closer than location predicts). Shipped expectation: ridge (see the candidate table for why).

## Inputs (fixture output, not a result)

| games | training | held-out | player-frames (train) | spacing frames (train) | shots (train) |
| --- | --- | --- | --- | --- | --- |
| 8 | 6 | 2 | 14795 | 2959 | 243 |

Training games: 02.10.2016.QCY.at.FRS, 03.08.2016.GRN.at.HLX, 11.28.2015.HLX.at.FRS, 12.01.2015.DUL.at.HLX, 12.09.2015.PEO.at.DUL, 12.10.2015.GRN.at.DUL. Held-out games set aside before any fit: 11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW.
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `schema.assert_time_forward` passed on the feature frame.
Sampling: 2959 frames sampled at 1 Hz inside possessions, 2959 kept (not ten players: 0, no ball: 0, degenerate hull: 0).
Validation counts: 0 schema problem(s), dropped rows {"frames_duplicate_frame_player": 0, "frames_null_xy": 0, "possessions_duplicate": 0, "rosters_duplicate_player": 0}.

## Candidates (out-of-fold MSE, 14795 player-frames, 6 games; variance of y 7.213) (fixture output, not a result)

Folds: out-of-fold under 5 grouped folds by sportvu_game (seed 0). Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

| model | MSE | R2 (1 - MSE / var y) | role |
| --- | --- | --- | --- |
| constant | 7.215 [6.975, 7.429] | -0.000 | floor |
| ridge | 4.299 [4.050, 4.496] | 0.404 | SHIPPED expectation (the plan's model) |
| hgb | 2.254 [2.021, 2.401] | 0.687 | diagnostic only |

Paired bootstrap by game, MSE(hgb) - MSE(ridge): -2.0446 [-2.1452, -1.9410], excludes zero: True (HGB lower: True). MSE is not the selection criterion: gravity is the residual of the expectation, and a flexible model lowers MSE partly by fitting what should stay in the residual, so the gravity tables use ridge and the HGB row says how much structure the location features leave unexplained.

## Players (gate: 300 possessions; 0 gated of 42) (fixture output, not a result)

| rank | player_id | possessions | gravity_ft | lo | hi |
| --- | --- | --- | --- | --- | --- |
| - | - | - | no player reaches the gate | - | - |

### Top-decile check (fixture output, not a result)

Known high-gravity players [500030, 500050, 500060, 500070, 500150, 500160]; season-wide ranks among 0 gated players {500030: None, 500050: None, 500060: None, 500070: None, 500150: None, 500160: None}; decile cut 0; passed: False.

Within each training game (the check that applies when rosters do not persist across games): 6 of 6 games pass; passed: True.

| game | players | decile cut | ranks of the known players |
| --- | --- | --- | --- |
| 02.10.2016.QCY.at.FRS | 14 | 2 | {500160: 1, 500050: 2} |
| 03.08.2016.GRN.at.HLX | 14 | 2 | {500070: 1, 500060: 2} |
| 11.28.2015.HLX.at.FRS | 14 | 2 | {500070: 1, 500050: 2} |
| 12.01.2015.DUL.at.HLX | 14 | 2 | {500030: 1, 500070: 2} |
| 12.09.2015.PEO.at.DUL | 14 | 2 | {500150: 1, 500030: 2} |
| 12.10.2015.GRN.at.DUL | 14 | 2 | {500030: 1, 500060: 2} |

Mean out-of-fold gravity (ft) of the known players under each candidate (ridge is the shipped one):

| model | 500030 | 500050 | 500060 | 500070 | 500150 | 500160 |
| --- | --- | --- | --- | --- | --- | --- |
| constant | 2.386 | 2.389 | 2.447 | 2.458 | 2.437 | 2.459 |
| ridge | 1.385 | 1.285 | 1.352 | 1.548 | 1.325 | 1.376 |
| hgb | 0.913 | 0.711 | 0.825 | 0.862 | 0.741 | 0.784 |

### Sign test: teammates' location-only xFG with a high-gravity player on the floor (fixture output, not a result)

Candidates [500030, 500050, 500060, 500070, 500150, 500160] (the fixture's planted high-gravity players). Games with a difference: 0; positive: 0; one-sided binomial p = n/a; mean difference n/a [n/a, n/a] (bootstrap by game).

## Spacing (fixture output, not a result)

Mean hull area 340.5 ft2 (sd 154.9), mean nearest-defender distance 4.48 ft over 2959 training possession-frames.

## Held-out, scored once (fixture output, not a result)

2 held-out game(s), 4860 player-frames, scored once by the shipped ridge refit on all 14795 training rows: MSE 4.892 (constant floor 8.032, not a selection input). Never fitted, tuned or ranked on.

## Second run: the lane's fallback generator (tests/_fx_gravity.py) (fixture output, not a result)

The default generator above gives every game a fresh roster and plays its star every period, so the season-wide top-decile check and the on/off sign test cannot run there. The fallback keeps the same fourteen players across the season, sits its star on about 40 percent of possessions and plants a known pull, so both checks run here.

### Inputs (fixture output, not a result)

| games | training | held-out | player-frames (train) | spacing frames (train) | shots (train) |
| --- | --- | --- | --- | --- | --- |
| 8 | 6 | 2 | 11780 | 2356 | 216 |

Training games: 01.01.2016.AAA.at.HHH, 01.02.2016.AAA.at.HHH, 01.03.2016.AAA.at.HHH, 01.05.2016.AAA.at.HHH, 01.06.2016.AAA.at.HHH, 01.07.2016.AAA.at.HHH. Held-out games set aside before any fit: 02.01.2016.AAA.at.CLE, 12.16.2015.PHX.at.GSW.
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `schema.assert_time_forward` passed on the feature frame.
Sampling: 2356 frames sampled at 1 Hz inside possessions, 2356 kept (not ten players: 0, no ball: 0, degenerate hull: 0).
Validation counts: 0 schema problem(s), dropped rows {"frames_duplicate_frame_player": 0, "frames_null_xy": 0, "possessions_duplicate": 0, "rosters_duplicate_player": 0}.

### Candidates (out-of-fold MSE, 11780 player-frames, 6 games; variance of y 5.082) (fixture output, not a result)

Folds: out-of-fold under 5 grouped folds by sportvu_game (seed 0). Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

| model | MSE | R2 (1 - MSE / var y) | role |
| --- | --- | --- | --- |
| constant | 5.085 [4.985, 5.214] | -0.001 | floor |
| ridge | 2.047 [1.985, 2.126] | 0.597 | SHIPPED expectation (the plan's model) |
| hgb | 1.897 [1.824, 2.000] | 0.627 | diagnostic only |

Paired bootstrap by game, MSE(hgb) - MSE(ridge): -0.1497 [-0.1756, -0.1226], excludes zero: True (HGB lower: True). MSE is not the selection criterion: gravity is the residual of the expectation, and a flexible model lowers MSE partly by fitting what should stay in the residual, so the gravity tables use ridge and the HGB row says how much structure the location features leave unexplained.

### Players (gate: 300 possessions; 0 gated of 14) (fixture output, not a result)

| rank | player_id | possessions | gravity_ft | lo | hi |
| --- | --- | --- | --- | --- | --- |
| - | - | - | no player reaches the gate | - | - |

#### Top-decile check (fixture output, not a result)

Known high-gravity players [1001, 2001]; season-wide ranks among 0 gated players {1001: None, 2001: None}; decile cut 0; passed: False.

Within each training game (the check that applies when rosters do not persist across games): 6 of 6 games pass; passed: True.

| game | players | decile cut | ranks of the known players |
| --- | --- | --- | --- |
| 01.01.2016.AAA.at.HHH | 14 | 2 | {2001: 1, 1001: 2} |
| 01.02.2016.AAA.at.HHH | 14 | 2 | {1001: 1, 2001: 2} |
| 01.03.2016.AAA.at.HHH | 14 | 2 | {1001: 1, 2001: 2} |
| 01.05.2016.AAA.at.HHH | 14 | 2 | {2001: 1, 1001: 2} |
| 01.06.2016.AAA.at.HHH | 14 | 2 | {1001: 1, 2001: 2} |
| 01.07.2016.AAA.at.HHH | 14 | 2 | {1001: 1, 2001: 2} |

Mean out-of-fold gravity (ft) of the known players under each candidate (ridge is the shipped one):

| model | 1001 | 2001 |
| --- | --- | --- |
| constant | 1.159 | 1.718 |
| ridge | 1.427 | 1.376 |
| hgb | 1.420 | 1.259 |

Recovered pull under the shipped expectation against the generator's effective pull (the planted pull in the target's own terms, see tests/_fx_gravity.py): 1001: recovered 1.427 ft, effective 1.536 ft; 2001: recovered 1.376 ft, effective 1.545 ft.

#### Sign test: teammates' location-only xFG with a high-gravity player on the floor (fixture output, not a result)

Candidates [1001, 2001] (the fixture's planted high-gravity players). Games with a difference: 6; positive: 6; one-sided binomial p = 0.0156; mean difference 0.1474 [0.1197, 0.1712] (bootstrap by game).

### Spacing (fixture output, not a result)

Mean hull area 243.1 ft2 (sd 126.3), mean nearest-defender distance 5.83 ft over 2356 training possession-frames.

### Held-out, scored once (fixture output, not a result)

2 held-out game(s), 3940 player-frames, scored once by the shipped ridge refit on all 11780 training rows: MSE 2.129 (constant floor 5.242, not a selection input). Never fitted, tuned or ranked on.

## Outputs

No tables written (pass --out OUT_ROOT).

## Assumptions

- The spec gives gravity as y minus the expectation and also says positive means defenders stay closer; these conflict, and the stated meaning wins: gravity_ft = expected_ft - nearest_def_dist_ft.
- The shipped expectation is ridge regardless of MSE: a residual quantity needs an expectation that does not fit the residual, and HGB on the same features is reported as a diagnostic only.
- The shooting prior is the league make rate by zone (rim, paint, short mid, long mid, three) of the player's position, from training-fold shots; a per-player prior was dropped because it identifies the player and let a flexible model absorb the pull.
- A possession's frames run from start_frame to the next possession's start_frame (exclusive) in the same period, which on analytics.possessions output is [start_frame, end_frame - 1]; end_frame itself is post-outcome and is not read. Frames over a stopped clock collapse to one sample.
- The attacked basket per (offense team, period) is the one nearer the mean x of the sampled offensive positions.
- has_ball marks the offensive player nearest the ball (the section 20 handler), with no distance threshold.
- shot_clock_s null is filled with 24 and flagged; frames without a ball, or without exactly five offensive and five defensive players, are dropped and counted.
- A player's interval resamples that player's games; a player seen in one game gets no interval. The player table is built on training frames only.
- The sign test's candidates are the fixture's planted players on fixtures and the top ten gated players on real data; the xFG probability is the location-only logistic of analytics.train_xfg fitted out of fold by game on the training games' shots; shots whose shooter could not be matched to a roster id are left out of the teammates' set.
- On-court at a shot means the player has a row in the frame of the same period nearest the shot's clock.
- Fixture runs on analytics.fixtures: every game has a fresh roster (player ids depend on the game seed), so no player reaches the 300-possession gate and the season-wide top-decile check cannot pass by construction; the per-game check is the one that applies there. Its star plays every period, so the on/off sign test finds no off-court shots; the fallback generator (tests/_fx_gravity.py) keeps one roster across the season and sits its star, and both checks run on it (the second section of a fixture report).

## Mac command (the real run)

    python -m analytics.gravity.train --data DATA_ROOT --report reports/gravity.md --folds 5 --seed 0 --bootstrap 1000 --min-possessions 300 --out DATA_ROOT

DATA_ROOT holds sportvu_contract/, analytics/possessions/2015-16/ (and heldout/2015-16/), analytics/rosters/, analytics/shot_frames/ and collect/bref_shots/shotchart/2015-16.parquet; the tables land under DATA_ROOT/analytics/gravity/.
