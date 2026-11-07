# Load and fatigue baselines (analytics/load/train.py)

Written 2026-11-07T08:42:06Z by `python -m analytics.load.train --fixtures --report reports/load.md --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are a synthetic season (fixture (analytics.fixtures synth_season)). The real run is the Mac command at the end.

## Inputs (fixture output, not a result)

| games loaded | training games | held-out games | stints | player-games | players | aggregates rows |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 6 | 2 | 320 | 112 | 63 | 3528 |

Validation totals over the loaded frames: rows_in 971060, rows_missing_key_or_position 0, rows_duplicate_frame_player 0, rows_outside_court 0, rows_speed_over_30_ftps 2, players_distance_over_ceiling 0, rows_out 971058.

injuries table (section 15): not present; it covers 2021-22 onward only and never overlaps the 2015-16 tracking era, so absence here is read from the rosters, not from injury reports.

Guard: `analytics.eval.refuse_heldout` checked 6 training game ids against sportvu/splits.json; `schema.assert_time_forward` and `assert_no_leak` passed on the design matrix (days_rest, back_to_back, games_last_7, minutes_last_3, distance_last_3_ft, prev_decline_ftps, prev_avg_speed_ftps).

## Target definitions and their limits

- Unit: the player-stint, a continuous on-court span of one player in the frames (a new stint when the period changes or the game clock jumps by more than 5 s between consecutive rows of the player); positions smoothed over 5 rows; steps count when their wall-clock dt is in (0, 1] s. Dead-ball movement counts.
- y_decline (`decline_ftps`): average speed of the player's last stint minus the first, over the stints of at least 1 minute, in ft per s; NaN with fewer than two such stints. `slope_ftps_per_stint` is the least-squares slope over those stints. Limits: average speed mixes role and game state (a blowout fourth quarter, a closing lineup) with fatigue; a player whose late stints are short defensive stands reads as declining; one game is noisy.
- `absent_next`: the team's next manifest game has a loaded roster and the player is not on it. Absence is not injury: rest, a coach's decision, a suspension, a trade or a tracking miss read the same. The section 15 injuries table starts in 2021-22 and never overlaps the tracking era, so no injury label exists here.
- Features come from the manifest schedule and from previous games only (minutes_last_3, distance_last_3_ft, prev_decline_ftps, prev_avg_speed_ftps); same-game totals and later stints are never features. A held-out game is never read for a feature (the window is NaN instead).

## Candidate 1: mean decline by rest bucket (fixture output, not a result)

Buckets by days_rest: back_to_back (1 or fewer), one_day (2), two_plus (3 or more, including a team's first listed game, which takes the cap of 7). Bootstrap intervals by game (n=200, seed=0); an interval needs at least two games in the bucket.

| bucket | n | games | mean_decline_ftps | lo | hi |
| --- | --- | --- | --- | --- | --- |
| back_to_back | 5 | 1 | -0.712 |  |  |
| one_day | 0 | 0 |  |  |  |
| two_plus | 55 | 6 | -0.141 | -0.490 | 0.319 |

Per-player summary (load_summary): 42 players, 6 with at least 10 stints (the others are gated out of any ranking, `insufficient`). The 6 largest mean declines among the gated-in players:

| player_id | n_games | n_stints | mean_decline | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| 500032 | 3 | 12 | -0.878 | -1.394 | -0.579 | False |
| 500030 | 3 | 12 | -0.876 | -1.548 | -0.008 | False |
| 500031 | 3 | 12 | -0.264 | -1.249 | 0.419 | False |
| 500072 | 3 | 12 | 0.063 | -0.523 | 0.963 | False |
| 500071 | 3 | 12 | 0.216 | -0.215 | 0.688 | False |
| 500070 | 3 | 12 | 0.276 | -0.106 | 0.637 | False |

## Candidate 2: ridge regression of y_decline (fixture output, not a result)

out-of-fold under 5 grouped folds by sportvu_game (seed 0). 60 player-games with a decline from 6 games. Ridge alpha 1.0, median imputation and standardisation fitted inside the training folds.

| model | MSE (ft per s)^2 |
| --- | --- |
| mean (training-fold mean) | 0.9133 [0.4863, 1.4505] |
| ridge on FEATURES | 0.8584 [0.5032, 1.3310] |

Paired bootstrap by game of MSE(ridge) - MSE(mean): -0.0549 [-0.2815, 0.1987], excludes zero: False.

## Candidate 3: absence from the next game (fixture output, not a result)

out-of-fold under 4 grouped folds by sportvu_game (seed 0); fewer games (4) than the 5 folds asked, so n_folds = number of games. 42 player-games with a known absent_next from 4 games, base rate 0.0000. Logistic C 1.0, same preprocessing. Absence is not injury.

| model | log loss | Brier | ECE (10 bins) |
| --- | --- | --- | --- |
| base rate (training-fold rate) | 0.0010 [0.0010, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.0010 [0.0010, 0.0010] |
| logistic on FEATURES | 0.0010 [0.0010, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.0010 [0.0010, 0.0010] |

Paired bootstrap by game of log loss(logistic) - log loss(base): 0.0000 [0.0000, 0.0000], excludes zero: False. Calibration slope of the logistic model: n/a (one fit, no interval; 1 is calibrated).

One class only in absent_next: no calibration table.

## Cross-check against the tracking aggregates (section 11) (fixture output, not a result)

42 players matched over 252 aggregate rows.

| quantity | Spearman (ours vs aggregate) | ours mean | aggregate mean |
| --- | --- | --- | --- |
| distance per game, miles | 0.184 | 0.380 | 2.264 |
| average speed, mph | -0.047 | 4.085 | 4.273 |

Our distance counts dead-ball movement at the wall clock and only frames that passed validation; the aggregate is NBA.com's own, so a gap in level with a high rank correlation is the expected picture.

## Held-out games, scored once (fixture output, not a result)

28 player-games from 2 held-out game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW). One model per pair was selected on its out-of-fold score and scored once here, refit on all training rows; never fitted or tuned on. The constant baseline appears only as a labelled floor under a selected candidate. Point estimates only.

| quantity | value |
| --- | --- |
| decline rows | 20 |
| MSE, selected model (ridge) | 0.6271 |
| MSE, floor (training mean, not selected) | 0.5085 |
| absence rows | 7 |
| log loss, selected model (logistic) | 0.0010 |
| log loss, floor (base rate, not selected) | 0.0010 |

## Status and the Mac command

Fixture output, not a result. The pipeline is tested (tests/test_load.py); the real run on the owner's Mac, where DATA_ROOT holds sportvu_contract/, analytics/rosters/ and, when present, collect/tracking/aggregates/, is

    python -m analytics.load.train --data "$DATA_ROOT" --report reports/load.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

Tables written under OUT_ROOT/analytics/load/<season>/ (held-out rows under heldout/<season>/): player_stints, player_games, load_summary (docs/contract/load.md).
