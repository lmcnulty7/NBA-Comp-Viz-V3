# Matchups: individual defender assessment (analytics/matchups) (fixture output, not a result)

Written 2026-11-07T08:40:31Z by `python -m analytics.matchups.train --fixtures --report reports/matchups.md --seed 0 --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are the synthetic season of analytics.fixtures, drawn from a seeded generator. The real run is the Mac command at the end.

## Inputs (fixture output, not a result)

| game | held out | possessions | sampled frames | matchup rows | attempts |
| --- | --- | --- | --- | --- | --- |
| 03.08.2016.GRN.at.HLX | no | 40 | 509 | 285 | 41 |
| 12.10.2015.GRN.at.DUL | no | 40 | 527 | 295 | 45 |
| 02.10.2016.QCY.at.FRS | no | 40 | 481 | 290 | 37 |
| 12.01.2015.DUL.at.HLX | no | 40 | 506 | 309 | 40 |
| 11.28.2015.HLX.at.FRS | no | 40 | 531 | 286 | 46 |
| 12.09.2015.PEO.at.DUL | no | 40 | 477 | 285 | 34 |
| 12.16.2015.PHX.at.GSW | yes | 40 | 490 | 300 | 41 |
| 11.26.2015.DUL.at.CLE | yes | 40 | 500 | 300 | 42 |

Training: 6 game(s) (02.10.2016.QCY.at.FRS, 03.08.2016.GRN.at.HLX, 11.28.2015.HLX.at.FRS, 12.01.2015.DUL.at.HLX, 12.09.2015.PEO.at.DUL, 12.10.2015.GRN.at.DUL). Held out, set aside before any fit: 2 game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
xFG probabilities: location-only logistic, out-of-fold under 5 grouped folds by game.
Skipped: 0 possession(s) without an offence id, 0 without frames, 0 sampled frame(s) without both squads.

## Protocol (fixture output, not a result)

Assignment at 1 Hz by `scipy.optimize.linear_sum_assignment` on distance to the offensive player plus 0.5 times distance to the midpoint between that player and the attacked basket (one basket per offense team and period, by majority over the possessions' last-third offensive centroids); positions only. Share = fraction of the possession's sampled frames assigned. Points enter only as targets; `schema.assert_time_forward` passed on the feature frame (partial_possession, switches); the ridge design is built from partial_possession only.
Gate: 300 distinct possessions per defender (`analytics.eval.gated`) before any ranking. Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Assignment stability (fixture output, not a result)

Share of defender sampled frames whose assignment equals the previous sampled frame: 0.936 [0.932, 0.940] over 13955 defender-frames (6 games).
Switches per possession per defender: mean 0.74, share of defender-possessions with no switch 0.595.

## Defended FG% consistency (fixture output, not a result)

| assigned defender | attempts | FG% |
| --- | --- | --- |
| within 4 ft | 187 | 0.545 [0.463, 0.661] |
| beyond 4 ft | 56 | 0.446 [0.293, 0.525] |

Beyond minus within: -0.099 [-0.265, 0.004] (paired by game; excludes zero: False).

## Baseline comparison: possession points, out-of-fold under 5 grouped folds by game (seed 0) (fixture output, not a result)

Ridge alpha nested: inside each outer fold an inner grouped CV over that fold's training games chose from [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] (per-fold alphas 300.0, 100.0, 300.0, 300.0, 300.0); the final model's alpha 300.0 was chosen by grouped CV over all training games (0.3: 1.6698, 1.0: 1.6424, 3.0: 1.5983, 10.0: 1.5339, 30.0: 1.4681, 100.0: 1.4058, 300.0: 1.3746).

| model | out-of-fold MSE |
| --- | --- |
| constant | 1.3580 [1.3008, 1.4412] |
| raw | 1.5105 [1.4062, 1.5897] |
| ridge | 1.3854 [1.3381, 1.4578] |
| ridge minus raw | -0.1251 [-0.1995, -0.0363] (excludes zero: True) |
| ridge minus constant | 0.0273 [-0.0004, 0.0743] (excludes zero: False) |

The out-of-fold MSE of every model carries no choice made on the scored rows (the ridge alpha of each fold comes from an inner CV over that fold's training games).
28 of 42 defenders appear in two or more training games (only those can move an out-of-fold prediction).

## Defenders (fixture output, not a result)

42 defenders, 0 pass the 300-possession gate. Sorted by diff_per100 (points allowed minus expected per 100 matchup possessions; positive = worse); ridge_value is the ridge coefficient per 100 possessions (positive = worse).

No defender passes the gate, so there is no ranking. The ungated rows, every one flagged insufficient, for inspection only:

| def_player_id | n_possessions | points_allowed_per100 | expected_per100 | diff_per100 | ridge_value | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500066 | 10 | 6.1 | 41.4 | -35.4 | -0.1 | -48.8 | -22.0 | yes |
| 500063 | 30 | 12.5 | 38.9 | -26.4 | -1.2 | -27.9 | -24.8 | yes |
| 500074 | 45 | 7.4 | 26.2 | -18.8 | -2.3 | -34.5 | -2.7 | yes |
| 500035 | 15 | 27.2 | 38.6 | -11.4 | -0.4 | -32.3 | 26.4 | yes |
| 500071 | 60 | 22.9 | 34.2 | -11.4 | -1.9 | -23.6 | 2.8 | yes |
| 500033 | 45 | 19.9 | 30.6 | -10.7 | 1.6 | -19.5 | -4.0 | yes |
| 500056 | 10 | 1.3 | 10.0 | -8.6 | -0.5 | -17.3 | 0.0 | yes |
| 500031 | 60 | 15.4 | 22.3 | -7.0 | 1.6 | -17.0 | 0.8 | yes |
| 500152 | 20 | 3.3 | 8.5 | -5.2 | -0.2 | n/a | n/a | yes |
| 500073 | 45 | 18.0 | 23.0 | -5.1 | -1.4 | -19.2 | 7.8 | yes |

## Held-out game(s), scored once (fixture output, not a result)

11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW: 80 possessions, 600 matchup rows; never fitted or tuned on; point estimates only.

Selected candidate: ridge (the lower out-of-fold MSE on the training folds; the rule was fixed before scoring). Only the candidate and the constant floor are scored here.

| model | MSE |
| --- | --- |
| constant (floor) | 1.2336 |
| ridge (selected) | 1.2287 |

Assignment stability 0.922; defended FG% within 4 ft 0.569 (65 attempts) vs beyond 0.556 (18 attempts).

## Status (fixture output, not a result)

Fixture output, not a result. The real run, on the Mac with the full DATA_ROOT (frames, play-by-play, possessions, shot_frames, rosters for 2015-16) and the xFG lane's predictions under OUT_ROOT/analytics/xfg/2015-16/, is

    python -m analytics.matchups.train --data "<DATA_ROOT>" --out "<OUT_ROOT>" --report reports/matchups.md --seed 0 --bootstrap 1000 --folds 5 --n-min 300

## Assumptions (fixture output, not a result)

- The attacked basket is the one nearer the offensive centroid over the possession's sampled frames (positions only).
- A possession's points go to the rows of the offensive player who took its last field-goal attempt; possessions without an attempt (turnovers, free throws without a shot) count 0 points and 0 expected for every matchup.
- Shot value is 3 when the play-by-play text says 3-pt, else by 22 ft from the basket at the release frame.
- switches counts, per defender and possession, the sampled frames at which its assignment changed; the count repeats on each of that defender's pair rows.
- The gate counts distinct possessions in which the defender was on the floor; n_possessions is that count, the per-100 denominators are the summed shares.
- Unseen players in the held-out game have no ridge column; the intercept carries them.
- No NBA.com matchup table exists for 2015-16, so validation is internal (stability, defended FG% consistency, held-out MSE); nothing here is checked against an external matchup record.
