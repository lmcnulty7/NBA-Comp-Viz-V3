# Live win probability (analytics/winprob/train.py)

Written 2026-11-05T08:57:44Z by `python -m analytics.winprob.train --fixtures --report reports/winprob.md --bootstrap 200`.

**Fixture run: every number below is fixture output, not a result.** The synthetic play-by-play of tests/_fx_winprob.py stands in for the five seasons on the owner's Mac.

## Inputs

| input | value |
| --- | --- |
| files | <fixtures seed 0> |
| events (states) | 14331 |
| games | 60 |
| seasons | 2020-21, 2021-22, 2024-25 |
| training seasons | 2020-21, 2021-22 (9640 events, 40 games) |
| held-out season | 2024-25 (4691 events, 20 games), set aside before any fit |
| rows flagged heldout (sportvu/splits.json games) dropped | 0 |
| guard | `analytics.eval.refuse_heldout` checked 40 training game ids; `analytics.schema.assert_time_forward` passed on the design matrix |
| closing spread coverage (training games) | 1.000 |
| possession known (training events) | 0.990 |
| training home win rate (by game) | 0.525 |

## Protocol

Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): margin, seconds_remaining, overtime, period, possession_home, spread_home, spread_missing, margin_x_sqrt_time.
The final score never enters a feature; `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k).
Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag) and margin_x_sqrt_time (standardised, C=1); HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.
Selection: out-of-fold under 5 grouped folds by game within the training seasons (seed 0). The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). The held-out season is scored once, by the selected candidate refit on all training rows.

## Candidates by time bucket (fixture output, not a result)

Brier score (lower is better), out-of-fold within the training seasons.

| model | 48-36 | 36-24 | 24-12 | 12-4 | 4-0 | OT | selection (mean of 5) | log loss overall | Brier overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 0.2645 [0.2489, 0.2793] | 0.2656 [0.2498, 0.2808] | 0.2653 [0.2496, 0.2813] | 0.2650 [0.2496, 0.2795] | 0.2643 [0.2490, 0.2796] | n/a | 0.2650 | 0.7236 [0.6922, 0.7541] | 0.2650 [0.2496, 0.2801] |
| logistic (selected) | 0.2427 [0.2120, 0.2793] | 0.1923 [0.1305, 0.2741] | 0.1457 [0.0968, 0.2287] | 0.0729 [0.0382, 0.1239] | 0.0502 [0.0212, 0.0935] | n/a | 0.1408 | 0.4814 [0.3887, 0.6279] | 0.1611 [0.1278, 0.2124] |
| hgb_iso | 0.3263 [0.2663, 0.3810] | 0.2937 [0.2359, 0.3508] | 0.2654 [0.2087, 0.3264] | 0.2256 [0.1724, 0.2767] | 0.2176 [0.1677, 0.2653] | n/a | 0.2657 | 0.9637 [0.6708, 1.4094] | 0.2768 [0.2202, 0.3238] |

Rows per bucket: 48-36 2395, 36-24 2403, 24-12 2415, 12-4 1558, 4-0 869, OT 0.

Selected: **logistic** (lowest mean bucket Brier).

## Calibration by time bucket (fixture output, not a result)

ECE (10 bins) and calibration slope per bucket, out-of-fold.

| model | bucket | n | ECE | slope |
| --- | --- | --- | --- | --- |
| constant | 48-36 | 2395 | 0.1089 [0.0299, 0.2605] | n/a |
| constant | 36-24 | 2403 | 0.1153 [0.0365, 0.2673] | n/a |
| constant | 24-12 | 2415 | 0.1166 [0.0375, 0.2654] | n/a |
| constant | 12-4 | 1558 | 0.1018 [0.0361, 0.2579] | n/a |
| constant | 4-0 | 869 | 0.0908 [0.0245, 0.2482] | n/a |
| logistic | 48-36 | 2395 | 0.0918 [0.0622, 0.1972] | 0.5369 [-0.0815, 1.4773] |
| logistic | 36-24 | 2403 | 0.0857 [0.0613, 0.2212] | 0.6591 [0.2190, 1.5428] |
| logistic | 24-12 | 2415 | 0.0884 [0.0526, 0.2159] | 0.7557 [0.4455, 1.2691] |
| logistic | 12-4 | 1558 | 0.0299 [0.0319, 0.1012] | 1.1957 [0.7845, 2.3461] |
| logistic | 4-0 | 869 | 0.0308 [0.0320, 0.0910] | 1.4018 [0.9455, 4.4817] |
| hgb_iso | 48-36 | 2395 | 0.2436 [0.1748, 0.3939] | -0.0473 [-0.9413, 0.1343] |
| hgb_iso | 36-24 | 2403 | 0.2220 [0.1683, 0.3755] | 0.0972 [-0.2634, 0.3866] |
| hgb_iso | 24-12 | 2415 | 0.1826 [0.1304, 0.3275] | 0.0688 [-0.0731, 0.4418] |
| hgb_iso | 12-4 | 1558 | 0.1238 [0.0936, 0.2714] | 0.1779 [0.0327, 1.2077] |
| hgb_iso | 4-0 | 869 | 0.1340 [0.0934, 0.2791] | 0.1893 [0.0512, 1.4260] |

The constant model has no calibration slope by design.

### Reliability table of the selected candidate (out-of-fold) (fixture output, not a result)

| bin | lo | hi | n | mean p | mean home win |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 1588 | 0.034 | 0.089 |
| 1 | 0.1 | 0.2 | 659 | 0.148 | 0.203 |
| 2 | 0.2 | 0.3 | 572 | 0.252 | 0.245 |
| 3 | 0.3 | 0.4 | 662 | 0.352 | 0.296 |
| 4 | 0.4 | 0.5 | 845 | 0.455 | 0.530 |
| 5 | 0.5 | 0.6 | 1197 | 0.545 | 0.565 |
| 6 | 0.6 | 0.7 | 898 | 0.647 | 0.604 |
| 7 | 0.7 | 0.8 | 761 | 0.749 | 0.693 |
| 8 | 0.8 | 0.9 | 689 | 0.849 | 0.817 |
| 9 | 0.9 | 1.0 | 1769 | 0.966 | 0.951 |

### winprob_calibration rows of the selected candidate (training seasons, out-of-fold) (fixture output, not a result)

| season | time_bucket | n | mean_p | mean_y | brier | slope |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-21 | 48-36 | 1172 | 0.547 | 0.435 | 0.2512 | 0.494 |
| 2020-21 | 36-24 | 1173 | 0.526 | 0.443 | 0.1910 | 0.824 |
| 2020-21 | 24-12 | 1212 | 0.522 | 0.432 | 0.1966 | 0.616 |
| 2020-21 | 12-4 | 756 | 0.504 | 0.460 | 0.1045 | 0.818 |
| 2020-21 | 4-0 | 437 | 0.514 | 0.467 | 0.0724 | 1.131 |
| 2021-22 | 48-36 | 1223 | 0.522 | 0.608 | 0.2346 | 0.714 |
| 2021-22 | 36-24 | 1230 | 0.459 | 0.602 | 0.1936 | 0.713 |
| 2021-22 | 24-12 | 1203 | 0.541 | 0.598 | 0.0944 | 1.008 |
| 2021-22 | 12-4 | 802 | 0.529 | 0.605 | 0.0432 | 4.529 |
| 2021-22 | 4-0 | 432 | 0.550 | 0.597 | 0.0277 | 3.487 |

## Held-out season 2024-25, scored once (fixture output, not a result)

4691 events from 20 games of 2024-25, never fitted or tuned on; the selected candidate (logistic) refit on all 9640 training rows scored them once.

| bucket | n | Brier | ECE | slope | benchmark (NBA.com WinProbabilityPBP) |
| --- | --- | --- | --- | --- | --- |
| 48-36 | 1150 | 0.1755 [0.1383, 0.2282] | 0.1046 [0.0666, 0.2221] | 2.1399 [0.9459, 5.2963] |  |
| 36-24 | 1156 | 0.1450 [0.0783, 0.2442] | 0.0948 [0.0712, 0.2452] | 0.9938 [0.5125, 2.8468] |  |
| 24-12 | 1181 | 0.1130 [0.0380, 0.2224] | 0.1074 [0.0567, 0.2510] | 0.9000 [0.5974, 3.5502] |  |
| 12-4 | 766 | 0.1159 [0.0416, 0.2382] | 0.1194 [0.0361, 0.2459] | 0.7857 [0.3880, 25.4474] |  |
| 4-0 | 438 | 0.0449 [0.0055, 0.1215] | 0.0753 [0.0292, 0.1690] | 2.5561 [2.0298, 45.8795] |  |

Overall: log loss 0.3984 [0.2510, 0.6233], Brier 0.1303 [0.0756, 0.2072], mean bucket Brier 0.1188.

## Benchmark

The NBA.com WinProbabilityPBP endpoint is unreachable from here, so the benchmark column is empty until it exists. When it does, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier.

## Outputs

- no tables written (no --out)

## Mac command

The real run over the five seasons under DATA_ROOT (collect/stints/playbyplay/<season>.parquet, collect/extras/odds/<season>.parquet when present):

    python -m analytics.winprob.train --data "$DATA_ROOT" --out "$DATA_ROOT" --report reports/winprob.md --seed 0 --bootstrap 1000 --folds 5 --heldout-season 2024-25

## Assumptions

- A whole season is held out (default 2024-25) and scored once; folds inside the training seasons are grouped by game.
- 2015-16 games of sportvu/splits.json never fit anything: rows flagged heldout are dropped, the 2015-16 season is refused as a held-out season, and refuse_heldout runs on the training ids.
- The closing spread is the median over books of the is_close spreads market for the home side, joined on game_date and the home team name (analytics.winprob.features.TEAM_NAMES); null with a flag when absent.
- The possession indicator is read from event kinds and is null when the events so far do not say.
- The interaction is margin times sqrt(seconds remaining), as specified; the logistic model also sees margin and time on their own.
- On the fixtures the boosted candidate loses to the logistic one because the spread is nearly unique per game among a few dozen games and the trees memorise games through it; the selection rule picks the logistic candidate. This is a fixture artefact to re-check on the real seasons, where many games share a spread value.
- The calibration slope is the logistic regression of home_win on logit(p) of analytics.eval.calibration_slope, solved here by a bounded Newton method (analytics.winprob.train.slope) because the eval solver runs to its iteration cap on the nearly separable late-game buckets; Brier, ECE, log loss, the reliability table and the game bootstrap are analytics.eval's.
