# Matchups: individual defender assessment (analytics/matchups)

Written 2026-11-05T08:58:06Z by `python -m analytics.matchups.train --fixtures --report reports/matchups.md --seed 0 --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are the synthetic season of tests/_fx_matchups.py (or analytics.fixtures), drawn from a seeded generator. The real run is the Mac command at the end.

## Inputs (fixture output, not a result)

| game | held out | possessions | sampled frames | matchup rows | attempts |
| --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | no | 40 | 478 | 293 | 37 |
| 01.18.2016.HLX.at.LRD | no | 40 | 506 | 290 | 44 |
| 02.04.2016.GRN.at.QCY | no | 40 | 508 | 277 | 37 |
| 12.15.2015.ABQ.at.GRN | no | 40 | 486 | 284 | 35 |
| 11.27.2015.GRN.at.QCY | no | 40 | 497 | 290 | 40 |
| 12.12.2015.RNO.at.MOB | no | 40 | 500 | 301 | 38 |
| 12.16.2015.PHX.at.GSW | yes | 40 | 534 | 282 | 45 |
| 11.11.2015.HLX.at.CLE | yes | 40 | 494 | 283 | 41 |

Training: 6 game(s) (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held out, set aside before any fit: 2 game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
xFG probabilities: location-only logistic, out-of-fold under 5 grouped folds by game.
Skipped: 0 possession(s) without an offence id, 0 without frames, 0 sampled frame(s) without both squads.

## Protocol

Assignment at 1 Hz by `scipy.optimize.linear_sum_assignment` on distance to the offensive player plus 0.5 times distance to the midpoint between that player and the attacked basket (one basket per offense team and period, by majority over the possessions' last-third offensive centroids); positions only. Share = fraction of the possession's sampled frames assigned. Points enter only as targets; `schema.assert_time_forward` passed on the feature frame (partial_possession, switches); the ridge design is built from partial_possession only.
Gate: 300 distinct possessions per defender (`analytics.eval.gated`) before any ranking. Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Assignment stability (fixture output, not a result)

Share of defender sampled frames whose assignment equals the previous sampled frame: 0.933 [0.930, 0.937] over 13675 defender-frames (6 games).
Switches per possession per defender: mean 0.77, share of defender-possessions with no switch 0.607.

## Defended FG% consistency (fixture output, not a result)

| assigned defender | attempts | FG% |
| --- | --- | --- |
| within 4 ft | 179 | 0.609 [0.536, 0.686] |
| beyond 4 ft | 52 | 0.442 [0.280, 0.600] |

Beyond minus within: -0.167 [-0.366, 0.064] (paired by game; excludes zero: False).

## Baseline comparison: possession points, out-of-fold under 5 grouped folds by game (seed 0) (fixture output, not a result)

Ridge alpha 30.0 chosen from [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] by out-of-fold MSE (0.3: 1.2903, 1.0: 1.2898, 3.0: 1.2888, 10.0: 1.2876, 30.0: 1.2873, 100.0: 1.2875, 300.0: 1.2877).

| model | out-of-fold MSE |
| --- | --- |
| constant | 1.2878 [1.2528, 1.3345] |
| raw | 1.2878 [1.2528, 1.3345] |
| ridge | 1.2873 [1.2542, 1.3345] |
| ridge minus raw | -0.0005 [-0.0042, 0.0035] (excludes zero: False) |
| ridge minus constant | -0.0005 [-0.0042, 0.0035] (excludes zero: False) |

The ridge MSE is at the alpha chosen on the same folds, so it is slightly optimistic; the raw and constant models carry no such choice.
Every defender is in one game on this input: raw and ridge out-of-fold equal the constant, so the comparison above carries no information about the defenders.

## Defenders (fixture output, not a result)

84 defenders, 0 pass the 300-possession gate. Sorted by diff_per100 (points allowed minus expected per 100 matchup possessions; positive = worse); ridge_value is the ridge coefficient per 100 possessions (positive = worse).

No defender passes the gate, so there is no ranking. The ungated rows, every one flagged insufficient, for inspection only:

| def_player_id | n_possessions | points_allowed_per100 | expected_per100 | diff_per100 | ridge_value | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9013 | 5 | 3.1 | 51.5 | -48.4 | -2.4 | n/a | n/a | yes |
| 9026 | 5 | 8.4 | 56.3 | -47.9 | 1.0 | n/a | n/a | yes |
| 9126 | 5 | 24.6 | 56.9 | -32.3 | -2.1 | n/a | n/a | yes |
| 9133 | 5 | 30.0 | 60.8 | -30.8 | -4.2 | n/a | n/a | yes |
| 9105 | 5 | 35.0 | 65.2 | -30.2 | -5.1 | n/a | n/a | yes |
| 9112 | 5 | 29.3 | 51.2 | -21.8 | -2.1 | n/a | n/a | yes |
| 9052 | 5 | 2.5 | 23.5 | -21.0 | 6.9 | n/a | n/a | yes |
| 9010 | 15 | 13.1 | 33.7 | -20.6 | 0.3 | n/a | n/a | yes |
| 9023 | 15 | 16.4 | 35.3 | -18.9 | 1.9 | n/a | n/a | yes |
| 9028 | 20 | 14.9 | 33.6 | -18.7 | -0.8 | n/a | n/a | yes |

## Held-out game(s), scored once (fixture output, not a result)

11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW: 80 possessions, 565 matchup rows; never fitted or tuned on; point estimates only.

| model | MSE |
| --- | --- |
| constant | 1.3653 |
| raw | 1.3653 |
| ridge | 1.3720 |

Assignment stability 0.947; defended FG% within 4 ft 0.433 (60 attempts) vs beyond 0.423 (26 attempts).

## Status

Fixture output, not a result. The real run, on the Mac with the full DATA_ROOT (frames, play-by-play, possessions, shot_frames, rosters for 2015-16) and the xFG lane's predictions under OUT_ROOT/analytics/xfg/2015-16/, is

    python -m analytics.matchups.train --data "<DATA_ROOT>" --out "<OUT_ROOT>" --report reports/matchups.md --seed 0 --bootstrap 1000 --folds 5 --n-min 300

## Assumptions

- The attacked basket is the one nearer the offensive centroid over the possession's sampled frames (positions only).
- A possession's points go to the rows of the offensive player who took its last field-goal attempt; possessions without an attempt (turnovers, free throws without a shot) count 0 points and 0 expected for every matchup.
- Shot value is 3 when the play-by-play text says 3-pt, else by 22 ft from the basket at the release frame.
- switches counts, per defender and possession, the sampled frames at which its assignment changed; the count repeats on each of that defender's pair rows.
- The gate counts distinct possessions in which the defender was on the floor; n_possessions is that count, the per-100 denominators are the summed shares.
- Unseen players in the held-out game have no ridge column; the intercept carries them.
- No NBA.com matchup table exists for 2015-16, so validation is internal (stability, defended FG% consistency, held-out MSE); nothing here is checked against an external matchup record.
