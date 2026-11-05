# Load and fatigue baselines (analytics/load/train.py)

Written 2026-11-05T08:59:15Z by `python -m analytics.load.train --fixtures --local-fixtures --report reports/load.md --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are the synthetic season of tests/_fx_load.py (fixture (local tests/_fx_load.py)). The real run is the Mac command at the end.

Produced with --local-fixtures: the shared analytics.fixtures season gives every game distinct teams and players, so the schedule history and the absence target are undefined on it; the local generator repeats four teams over a schedule.

## Inputs

| games loaded | training games | held-out games | stints | player-games | players | aggregates rows |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 6 | 2 | 448 | 152 | 40 | 160 |

Validation totals over the loaded frames: rows_in 960000, rows_missing_key_or_position 0, rows_duplicate_frame_player 0, rows_outside_court 0, rows_speed_over_30_ftps 0, players_distance_over_ceiling 0, rows_out 960000.

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
| back_to_back | 9 | 1 | -0.434 | n/a | n/a |
| one_day | 9 | 1 | -0.466 | n/a | n/a |
| two_plus | 95 | 6 | -0.376 | -0.423 | -0.343 |

Per-player summary (load_summary): 40 players, 11 with at least 10 stints (the others are gated out of any ranking, `insufficient`). The 10 largest mean declines among the gated-in players:

| player_id | n_games | n_stints | mean_decline | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| 304 | 3 | 12 | -0.639 | -0.836 | -0.393 | False |
| 103 | 4 | 16 | -0.625 | -0.997 | -0.293 | False |
| 203 | 3 | 12 | -0.524 | -0.742 | -0.277 | False |
| 105 | 4 | 12 | -0.393 | -0.477 | -0.286 | False |
| 206 | 3 | 11 | -0.385 | -0.529 | -0.242 | False |
| 301 | 3 | 11 | -0.378 | -0.423 | -0.334 | False |
| 303 | 3 | 12 | -0.372 | -0.427 | -0.262 | False |
| 101 | 4 | 14 | -0.318 | -0.422 | -0.147 | False |
| 202 | 3 | 12 | -0.209 | -0.213 | -0.206 | False |
| 102 | 4 | 16 | -0.160 | -0.211 | -0.079 | False |

## Candidate 2: ridge regression of y_decline (fixture output, not a result)

out-of-fold under 5 grouped folds by sportvu_game (seed 0). 113 player-games with a decline from 6 games. Ridge alpha 1.0, median imputation and standardisation fitted inside the training folds.

| model | MSE (ft per s)^2 |
| --- | --- |
| mean (training-fold mean) | 0.0428 [0.0304, 0.0547] |
| ridge on FEATURES | 0.0489 [0.0396, 0.0557] |

Paired bootstrap by game of MSE(ridge) - MSE(mean): 0.0061 [-0.0051, 0.0187], excludes zero: False.

## Candidate 3: absence from the next game (fixture output, not a result)

out-of-fold under 5 grouped folds by sportvu_game (seed 0). 57 player-games with a known absent_next from 5 games, base rate 0.0526. Logistic C 1.0, same preprocessing. Absence is not injury.

| model | log loss | Brier | ECE (10 bins) |
| --- | --- | --- | --- |
| base rate (training-fold rate) | 0.2149 [0.1266, 0.3273] | 0.0507 [0.0227, 0.0842] | 0.0006 [0.0006, 0.0637] |
| logistic on FEATURES | 0.4749 [0.0915, 0.8148] | 0.1073 [0.0202, 0.1856] | 0.0937 [0.0006, 0.1847] |

Paired bootstrap by game of log loss(logistic) - log loss(base): 0.2600 [-0.0361, 0.5717], excludes zero: False. Calibration slope of the logistic model: -0.3890 (one fit, no interval; 1 is calibrated).

Calibration table of the logistic model (equal-width bins of p):

| bin | lo | hi | n | mean p | mean absent |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 51 | 0.028 | 0.059 |
| 1 | 0.1 | 0.2 | 2 | 0.132 | 0.000 |
| 2 | 0.2 | 0.3 | 0 | n/a | n/a |
| 3 | 0.3 | 0.4 | 0 | n/a | n/a |
| 4 | 0.4 | 0.5 | 0 | n/a | n/a |
| 5 | 0.5 | 0.6 | 0 | n/a | n/a |
| 6 | 0.6 | 0.7 | 1 | 0.628 | 0.000 |
| 7 | 0.7 | 0.8 | 0 | n/a | n/a |
| 8 | 0.8 | 0.9 | 1 | 0.887 | 0.000 |
| 9 | 0.9 | 1.0 | 2 | 0.987 | 0.000 |

## Cross-check against the tracking aggregates (section 11) (fixture output, not a result)

40 players matched over 160 aggregate rows.

| quantity | Spearman (ours vs aggregate) | ours mean | aggregate mean |
| --- | --- | --- | --- |
| distance per game, miles | 0.628 | 0.251 | 1.612 |
| average speed, mph | 0.984 | 3.559 | 3.529 |

Our distance counts dead-ball movement at the wall clock and only frames that passed validation; the aggregate is NBA.com's own, so a gap in level with a high rank correlation is the expected picture.

## Held-out games, scored once (fixture output, not a result)

39 player-games from 2 held-out game(s) (11.07.2015.BBB.at.DDD, 11.14.2015.CCC.at.DDD), scored once by the models refit on all training rows; never fitted or tuned on. Point estimates only.

| quantity | value |
| --- | --- |
| decline rows | 39 |
| MSE mean | 0.0534 |
| MSE ridge | 0.0485 |
| absence rows | 20 |
| log loss base | 0.3431 |
| log loss logistic | 0.3414 |

## Status and the Mac command

Fixture output, not a result. The pipeline is tested (tests/test_load.py); the real run on the owner's Mac, where DATA_ROOT holds sportvu_contract/, analytics/rosters/ and, when present, collect/tracking/aggregates/, is

    python -m analytics.load.train --data "$DATA_ROOT" --report reports/load.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

Tables written under OUT_ROOT/analytics/load/<season>/ (held-out rows under heldout/<season>/): player_stints, player_games, load_summary (docs/contract/load.md).
