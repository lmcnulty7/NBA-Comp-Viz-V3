# Matchups: individual defender assessment (analytics/matchups)

Written 2026-11-05T08:41:31Z by `python -m analytics.matchups.train --fixtures --report reports/matchups.md --seed 0 --bootstrap 200`.

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
| 11.11.2015.HLX.at.GRN | no | 40 | 494 | 283 | 41 |
| 12.16.2015.PHX.at.GSW | yes | 40 | 534 | 282 | 45 |

Training: 7 game(s) (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.11.2015.HLX.at.GRN, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held out, set aside before any fit: 1 game(s) (12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
xFG probabilities: location-only logistic, out-of-fold under 5 grouped folds by game.
Skipped: 0 possession(s) without an offence id, 0 without frames, 0 sampled frame(s) without both squads.

## Protocol

Assignment at 1 Hz by `scipy.optimize.linear_sum_assignment` on distance to the offensive player plus 0.5 times distance to the midpoint between that player and the attacked basket (the basket nearer the offensive centroid); positions only. Share = fraction of the possession's sampled frames assigned. Points enter only as targets; `schema.assert_time_forward` passed on the feature frame (partial_possession, switches) and on the ridge design columns.
Gate: 300 distinct possessions per defender (`analytics.eval.gated`) before any ranking. Intervals: 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Assignment stability (fixture output, not a result)

Share of defender sampled frames whose assignment equals the previous sampled frame: 0.934 [0.930, 0.938] over 15945 defender-frames (7 games).
Switches per possession per defender: mean 0.75, share of defender-possessions with no switch 0.609.

## Defended FG% consistency (fixture output, not a result)

| assigned defender | attempts | FG% |
| --- | --- | --- |
| within 4 ft | 213 | 0.582 [0.511, 0.657] |
| beyond 4 ft | 59 | 0.424 [0.286, 0.566] |

Beyond minus within: -0.158 [-0.327, 0.023] (paired by game; excludes zero: False).

## Baseline comparison: possession points, out-of-fold under 5 grouped folds by game (seed 0) (fixture output, not a result)

Ridge alpha 30.0 chosen from [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] by out-of-fold MSE (0.3: 1.3126, 1.0: 1.3121, 3.0: 1.3113, 10.0: 1.3103, 30.0: 1.3099, 100.0: 1.3100, 300.0: 1.3101).

| model | out-of-fold MSE |
| --- | --- |
| constant | 1.3102 [1.2639, 1.3586] |
| raw | 1.3102 [1.2639, 1.3586] |
| ridge | 1.3099 [1.2617, 1.3621] |
| ridge minus raw | -0.0003 [-0.0037, 0.0037] (excludes zero: False) |
| ridge minus constant | -0.0003 [-0.0037, 0.0037] (excludes zero: False) |

The ridge MSE is at the alpha chosen on the same folds, so it is slightly optimistic; the raw and constant models carry no such choice.

## Defenders (fixture output, not a result)

98 defenders, 0 pass the 300-possession gate. Sorted by diff_per100 (points allowed minus expected per 100 matchup possessions; positive = worse); ridge_value is the ridge coefficient per 100 possessions (positive = worse).

No defender passes the gate, so there is no ranking. The ungated rows, every one flagged insufficient, for inspection only:

| def_player_id | n_possessions | points_allowed_per100 | expected_per100 | diff_per100 | ridge_value | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9013 | 5 | 3.1 | 50.4 | -47.3 | -2.3 | n/a | n/a | yes |
| 9026 | 5 | 8.4 | 55.1 | -46.6 | 1.1 | n/a | n/a | yes |
| 9146 | 5 | 0.0 | 43.3 | -43.3 | -1.5 | n/a | n/a | yes |
| 9133 | 5 | 30.0 | 60.5 | -30.5 | -4.1 | n/a | n/a | yes |
| 9105 | 5 | 35.0 | 64.7 | -29.7 | -5.0 | n/a | n/a | yes |
| 9126 | 5 | 24.6 | 53.6 | -29.0 | -1.9 | n/a | n/a | yes |
| 9141 | 20 | 1.4 | 28.1 | -26.7 | -5.7 | n/a | n/a | yes |
| 9112 | 5 | 29.3 | 51.2 | -21.8 | -1.9 | n/a | n/a | yes |
| 9046 | 5 | 0.0 | 20.5 | -20.5 | -7.8 | n/a | n/a | yes |
| 9010 | 15 | 13.1 | 32.6 | -19.5 | 0.6 | n/a | n/a | yes |

## Held-out game(s), scored once (fixture output, not a result)

12.16.2015.PHX.at.GSW: 40 possessions, 282 matchup rows; never fitted or tuned on; point estimates only.

| model | MSE |
| --- | --- |
| constant | 1.2956 |
| raw | 1.2956 |
| ridge | 1.2985 |

Assignment stability 0.952; defended FG% within 4 ft 0.423 (26 attempts) vs beyond 0.474 (19 attempts).

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
