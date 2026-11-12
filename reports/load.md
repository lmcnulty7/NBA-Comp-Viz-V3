# Load and fatigue baselines (analytics/load/train.py)

**Real data, training folds only; held-out not scored.** Stage 5 phase 5 (docs/mac_runbook.md), run with --skip-heldout on 2026-11-11.

Written 2026-11-12T07:11:59Z by `python -m analytics.load.train --data $DATA_ROOT --seed 0 --report reports/load.md --skip-heldout`.

## Inputs

| games loaded | training games | held-out games | stints | player-games | players | aggregates rows |
| --- | --- | --- | --- | --- | --- | --- |
| 621 | 604 | 17 | 98102 | 13274 | 448 | 0 |

Validation totals over the loaded frames: rows_in 482166487, rows_missing_key_or_position 0, rows_duplicate_frame_player 2, rows_outside_court 27919, rows_speed_over_30_ftps 68569, players_distance_over_ceiling 0, rows_out 482069997.

injuries table (section 15): not present; it covers 2021-22 onward only and never overlaps the 2015-16 tracking era, so absence here is read from the rosters, not from injury reports.

Guard: `analytics.eval.refuse_heldout` checked 604 training game ids against sportvu/splits.json; `schema.assert_time_forward` and `assert_no_leak` passed on the design matrix (days_rest, back_to_back, games_last_7, minutes_last_3, distance_last_3_ft, prev_decline_ftps, prev_avg_speed_ftps).

## Target definitions and their limits

- Unit: the player-stint, a continuous on-court span of one player in the frames (a new stint when the period changes or the game clock jumps by more than 5 s between consecutive rows of the player); positions smoothed over 5 rows; steps count when their wall-clock dt is in (0, 1] s. Dead-ball movement counts.
- y_decline (`decline_ftps`): average speed of the player's last stint minus the first, over the stints of at least 1 minute, in ft per s; NaN with fewer than two such stints. `slope_ftps_per_stint` is the least-squares slope over those stints. Limits: average speed mixes role and game state (a blowout fourth quarter, a closing lineup) with fatigue; a player whose late stints are short defensive stands reads as declining; one game is noisy.
- `absent_next`: the team's next manifest game has a loaded roster and the player is not on it. Absence is not injury: rest, a coach's decision, a suspension, a trade or a tracking miss read the same. The section 15 injuries table starts in 2021-22 and never overlaps the tracking era, so no injury label exists here.
- Features come from the manifest schedule and from previous games only (minutes_last_3, distance_last_3_ft, prev_decline_ftps, prev_avg_speed_ftps); same-game totals and later stints are never features. A held-out game is never read for a feature (the window is NaN instead).

## Candidate 1: mean decline by rest bucket

Buckets by days_rest: back_to_back (1 or fewer), one_day (2), two_plus (3 or more, including a team's first listed game, which takes the cap of 7). Bootstrap intervals by game (n=200, seed=0); an interval needs at least two games in the bucket.

| bucket | n | games | mean_decline_ftps | lo | hi |
| --- | --- | --- | --- | --- | --- |
| back_to_back | 2434 | 214 | -0.477 | -0.520 | -0.439 |
| one_day | 6643 | 454 | -0.437 | -0.466 | -0.411 |
| two_plus | 2817 | 229 | -0.457 | -0.495 | -0.419 |

Per-player summary (load_summary): 448 players, 422 with at least 10 stints (the others are gated out of any ranking, `insufficient`). The 10 largest mean declines among the gated-in players:

| player_id | n_games | n_stints | mean_decline | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| 1626146 | 2 | 15 | -1.259 | -1.616 | -0.903 | False |
| 203955 | 2 | 17 | -1.228 | -1.240 | -1.217 | False |
| 201942 | 40 | 339 | -1.051 | -1.268 | -0.875 | False |
| 200755 | 35 | 267 | -1.000 | -1.288 | -0.741 | False |
| 203484 | 42 | 447 | -0.970 | -1.180 | -0.740 | False |
| 201960 | 22 | 157 | -0.968 | -1.202 | -0.717 | False |
| 201936 | 24 | 277 | -0.965 | -1.276 | -0.677 | False |
| 201167 | 35 | 279 | -0.963 | -1.120 | -0.789 | False |
| 2772 | 40 | 528 | -0.962 | -1.157 | -0.717 | False |
| 202694 | 41 | 432 | -0.934 | -1.089 | -0.781 | False |

## Candidate 2: ridge regression of y_decline

out-of-fold under 5 grouped folds by sportvu_game (seed 0). 11894 player-games with a decline from 604 games. Ridge alpha 1.0, median imputation and standardisation fitted inside the training folds.

| model | MSE (ft per s)^2 |
| --- | --- |
| mean (training-fold mean) | 0.5086 [0.4884, 0.5352] |
| ridge on FEATURES | 0.4929 [0.4740, 0.5178] |

Paired bootstrap by game of MSE(ridge) - MSE(mean): -0.0157 [-0.0197, -0.0118], excludes zero: True.

Selected candidate: ridge on FEATURES.

## Candidate 3: absence from the next game

out-of-fold under 5 grouped folds by sportvu_game (seed 0). 12239 player-games with a known absent_next from 592 games, base rate 0.0997. Logistic C 1.0, same preprocessing. Absence is not injury.

| model | log loss | Brier | ECE (10 bins) |
| --- | --- | --- | --- |
| base rate (training-fold rate) | 0.3244 [0.3118, 0.3379] | 0.0898 [0.0852, 0.0946] | 0.0042 [0.0009, 0.0095] |
| logistic on FEATURES | 0.2764 [0.2644, 0.2883] | 0.0777 [0.0742, 0.0815] | 0.0146 [0.0118, 0.0202] |

Paired bootstrap by game of log loss(logistic) - log loss(base): -0.0481 [-0.0553, -0.0417], excludes zero: True. Calibration slope of the logistic model: 0.9908 (one fit, no interval; 1 is calibrated).

Selected candidate: logistic on FEATURES.

Calibration table of the logistic model (equal-width bins of p):

| bin | lo | hi | n | mean p | mean absent |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 8268 | 0.043 | 0.045 |
| 1 | 0.1 | 0.2 | 2227 | 0.142 | 0.101 |
| 2 | 0.2 | 0.3 | 827 | 0.246 | 0.261 |
| 3 | 0.3 | 0.4 | 655 | 0.351 | 0.440 |
| 4 | 0.4 | 0.5 | 262 | 0.437 | 0.447 |
| 5 | 0.5 | 0.6 | 0 | n/a | n/a |
| 6 | 0.6 | 0.7 | 0 | n/a | n/a |
| 7 | 0.7 | 0.8 | 0 | n/a | n/a |
| 8 | 0.8 | 0.9 | 0 | n/a | n/a |
| 9 | 0.9 | 1.0 | 0 | n/a | n/a |

## Cross-check against the tracking aggregates (section 11)

No leaguedashptstats SpeedDistance aggregates were present; the cross-check did not run.

## Held-out games, not scored

17 held-out game(s) set aside before any fit and never fitted, tuned or scored: held-out games not scored (--skip-heldout).

## Status and the Mac command

The run of this protocol was

    python -m analytics.load.train --data "$DATA_ROOT" --report reports/load.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

Tables written under OUT_ROOT/analytics/load/<season>/ (held-out rows under heldout/<season>/): player_stints, player_games, load_summary (docs/contract/load.md).
