# Live win probability (analytics/winprob/train.py)

Written 2026-11-13T00:41:57Z by `python -m analytics.winprob.train --data $DATA_ROOT --seed 0 --report reports/winprob.md`.

**Real data, stage 8 held-out run of group 1 (tag freeze-v1, 2026-11-12): the held-out games were scored exactly once, by the model frozen at freeze-v1.** Every out-of-fold number below is identical to the frozen training-fold run; nothing was refit, tuned or reselected after the held-out numbers were seen.

## Stage 8: held-out next to out-of-fold (freeze-v1)

Selected model: logistic with margin_over_sqrt_time (improvement search variant 1), refit on all 2292413 training
events (2020-21 to 2023-24) and scored once on 596840 events from 1304 games of the held-out 2024-25 season. Intervals:
bootstrap by game. The NBA.com benchmark column stays empty: NBA.com tables were skipped (gates/nbacom-skip).

| metric | out-of-fold (training seasons) | held-out 2024-25 |
| --- | --- | --- |
| Brier 48-36 | 0.2306 [0.2278, 0.2332] | 0.2318 [0.2261, 0.2377] |
| Brier 36-24 | 0.1942 [0.1905, 0.1998] | 0.1982 [0.1881, 0.2065] |
| Brier 24-12 | 0.1525 [0.1476, 0.1588] | 0.1502 [0.1397, 0.1607] |
| Brier 12-4 | 0.1038 [0.0986, 0.1089] | 0.1043 [0.0953, 0.1134] |
| Brier 4-0 | 0.0530 [0.0496, 0.0564] | 0.0582 [0.0504, 0.0666] |
| Brier OT (reported only) | 0.1538 [0.1379, 0.1707] | 0.1452 [0.1194, 0.1745] |
| selection metric (mean of the 5 regulation buckets) | 0.1468 | 0.1486 |
| log loss overall | 0.4845 [0.4767, 0.4945] | 0.4879 [0.4700, 0.5077] |
| Brier overall | 0.1645 [0.1616, 0.1683] | 0.1658 [0.1590, 0.1732] |
| calibration slope, 5 regulation buckets | 1.014, 1.004, 0.966, 1.017, 1.065 | 1.001, 0.955, 0.959, 0.957, 0.955 |
| ECE, 5 regulation buckets | 0.0049, 0.0085, 0.0118, 0.0112, 0.0051 | 0.0145, 0.0208, 0.0129, 0.0124, 0.0079 |

Where held-out is worse: the selection metric (0.1486 against 0.1468), four of the five regulation buckets (all but
24-12), most in the last 4 minutes (0.0582 against 0.0530), and calibration: ECE is higher in every bucket and the
slopes drop to about 0.96 (slightly over-confident) where they were 0.97 to 1.06. Every held-out Brier interval
overlaps its out-of-fold interval.

## Inputs

| input | value |
| --- | --- |
| files | $DATA_ROOT/collect/stints/playbyplay/2020-21.parquet, $DATA_ROOT/collect/stints/playbyplay/2021-22.parquet, $DATA_ROOT/collect/stints/playbyplay/2022-23.parquet, $DATA_ROOT/collect/stints/playbyplay/2023-24.parquet, $DATA_ROOT/collect/stints/playbyplay/2024-25.parquet |
| events (states) | 2889253 |
| games | 6363 |
| seasons | 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 |
| training seasons | 2020-21, 2021-22, 2022-23, 2023-24 (2292413 events, 5059 games) |
| held-out season | 2024-25 (596840 events, 1304 games), set aside before any fit |
| rows flagged heldout (sportvu/splits.json games) dropped | 0 |
| clock rule: dropped before any feature (all seasons) | 24584 of 2913837 events, 53 of 6416 games: 53 games (24584 events) with a gap in the period numbers, 0 games (0 events) with an overtime clock above 300 s, 0 single events with a null or out-of-range clock |
| seasons later than the held-out season, ignored (never fitted or scored) | none |
| guard | `analytics.eval.refuse_heldout` checked 5059 training game ids; `analytics.schema.assert_time_forward` passed on the design matrix |
| closing spread coverage (training games) | 0.000 |
| possession known (training events) | 0.781 |
| training home win rate (by game) | 0.555 |

## Protocol

Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): margin, seconds_remaining, overtime, period, possession_home, spread_home, spread_missing, margin_x_sqrt_time, margin_over_sqrt_time.
The final score never enters a feature; `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k).
Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag), margin_x_sqrt_time and margin_over_sqrt_time (standardised, C=1); HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.
Selection: out-of-fold under 5 grouped folds by game within the training seasons (seed 0). The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). The held-out season is scored once, by the selected candidate refit on all training rows.

## Candidates by time bucket

Brier score (lower is better), out-of-fold within the training seasons.

| model | 48-36 | 36-24 | 24-12 | 12-4 | 4-0 | OT | selection (mean of 5) | log loss overall | Brier overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 0.2471 [0.2454, 0.2487] | 0.2471 [0.2453, 0.2487] | 0.2470 [0.2453, 0.2486] | 0.2469 [0.2451, 0.2485] | 0.2477 [0.2460, 0.2493] | 0.2504 [0.2442, 0.2572] | 0.2472 | 0.6874 [0.6839, 0.6906] | 0.2471 [0.2454, 0.2487] |
| logistic (selected) | 0.2306 [0.2278, 0.2332] | 0.1942 [0.1905, 0.1998] | 0.1525 [0.1476, 0.1588] | 0.1038 [0.0986, 0.1089] | 0.0530 [0.0496, 0.0564] | 0.1538 [0.1379, 0.1707] | 0.1468 | 0.4845 [0.4767, 0.4945] | 0.1645 [0.1616, 0.1683] |
| hgb_iso | 0.2308 [0.2279, 0.2335] | 0.1944 [0.1906, 0.1999] | 0.1525 [0.1477, 0.1587] | 0.1039 [0.0989, 0.1087] | 0.0533 [0.0499, 0.0567] | 0.1548 [0.1384, 0.1718] | 0.1470 | 0.4848 [0.4770, 0.4951] | 0.1646 [0.1618, 0.1685] |

Rows per bucket: 48-36 549307, 36-24 581066, 24-12 561851, 12-4 367834, 4-0 218146, OT 14209.

Selected: **logistic** (lowest mean bucket Brier).

## Calibration by time bucket

ECE (10 bins) and calibration slope per bucket, out-of-fold.

| model | bucket | n | ECE | slope |
| --- | --- | --- | --- | --- |
| constant | 48-36 | 549307 | 0.0003 [0.0003, 0.0161] | n/a |
| constant | 36-24 | 581066 | 0.0003 [0.0002, 0.0168] | n/a |
| constant | 24-12 | 561851 | 0.0011 [0.0001, 0.0167] | n/a |
| constant | 12-4 | 367834 | 0.0021 [0.0002, 0.0178] | n/a |
| constant | 4-0 | 218146 | 0.0049 [0.0004, 0.0189] | n/a |
| constant | OT | 14209 | 0.0253 [0.0021, 0.0853] | n/a |
| logistic | 48-36 | 549307 | 0.0049 [0.0047, 0.0187] | 1.0140 [0.9309, 1.0911] |
| logistic | 36-24 | 581066 | 0.0085 [0.0071, 0.0189] | 1.0036 [0.9461, 1.0563] |
| logistic | 24-12 | 561851 | 0.0118 [0.0077, 0.0211] | 0.9656 [0.9131, 1.0104] |
| logistic | 12-4 | 367834 | 0.0112 [0.0083, 0.0179] | 1.0168 [0.9723, 1.0651] |
| logistic | 4-0 | 218146 | 0.0051 [0.0045, 0.0104] | 1.0648 [1.0134, 1.1169] |
| logistic | OT | 14209 | 0.0210 [0.0173, 0.0566] | 0.9305 [0.7871, 1.1196] |
| hgb_iso | 48-36 | 549307 | 0.0084 [0.0045, 0.0203] | 0.9386 [0.8622, 1.0118] |
| hgb_iso | 36-24 | 581066 | 0.0053 [0.0056, 0.0171] | 1.0002 [0.9390, 1.0540] |
| hgb_iso | 24-12 | 561851 | 0.0038 [0.0049, 0.0140] | 1.0047 [0.9441, 1.0567] |
| hgb_iso | 12-4 | 367834 | 0.0068 [0.0055, 0.0156] | 1.0270 [0.9750, 1.0827] |
| hgb_iso | 4-0 | 218146 | 0.0063 [0.0048, 0.0113] | 0.9756 [0.9206, 1.0331] |
| hgb_iso | OT | 14209 | 0.0234 [0.0185, 0.0593] | 0.8759 [0.7429, 1.0621] |

The constant model has no calibration slope by design.

### Reliability table of the selected candidate (out-of-fold)

| bin | lo | hi | n | mean p | mean home win |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 219767 | 0.033 | 0.028 |
| 1 | 0.1 | 0.2 | 121525 | 0.150 | 0.165 |
| 2 | 0.2 | 0.3 | 138852 | 0.252 | 0.266 |
| 3 | 0.3 | 0.4 | 175135 | 0.352 | 0.352 |
| 4 | 0.4 | 0.5 | 245618 | 0.453 | 0.448 |
| 5 | 0.5 | 0.6 | 350686 | 0.551 | 0.554 |
| 6 | 0.6 | 0.7 | 280221 | 0.649 | 0.647 |
| 7 | 0.7 | 0.8 | 221525 | 0.748 | 0.736 |
| 8 | 0.8 | 0.9 | 191988 | 0.850 | 0.846 |
| 9 | 0.9 | 1.0 | 347096 | 0.968 | 0.972 |

### winprob_calibration rows of the selected candidate (training seasons, out-of-fold)

| season | time_bucket | n | mean_p | mean_y | brier | slope |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-21 | 48-36 | 124987 | 0.552 | 0.543 | 0.2320 | 1.003 |
| 2020-21 | 36-24 | 132454 | 0.547 | 0.544 | 0.1920 | 1.068 |
| 2020-21 | 24-12 | 127107 | 0.541 | 0.545 | 0.1473 | 1.007 |
| 2020-21 | 12-4 | 84064 | 0.541 | 0.548 | 0.1012 | 1.020 |
| 2020-21 | 4-0 | 50898 | 0.541 | 0.544 | 0.0523 | 1.046 |
| 2020-21 | OT | 3221 | 0.503 | 0.486 | 0.1384 | 1.014 |
| 2021-22 | 48-36 | 140981 | 0.553 | 0.547 | 0.2306 | 1.052 |
| 2021-22 | 36-24 | 150099 | 0.555 | 0.546 | 0.1958 | 0.982 |
| 2021-22 | 24-12 | 144433 | 0.551 | 0.545 | 0.1573 | 0.909 |
| 2021-22 | 12-4 | 94057 | 0.553 | 0.544 | 0.1042 | 1.003 |
| 2021-22 | 4-0 | 55586 | 0.548 | 0.539 | 0.0507 | 1.102 |
| 2021-22 | OT | 3205 | 0.504 | 0.443 | 0.1760 | 0.840 |
| 2022-23 | 48-36 | 142622 | 0.558 | 0.583 | 0.2271 | 1.046 |
| 2022-23 | 36-24 | 150764 | 0.562 | 0.583 | 0.1968 | 0.955 |
| 2022-23 | 24-12 | 145203 | 0.566 | 0.586 | 0.1613 | 0.938 |
| 2022-23 | 12-4 | 95126 | 0.577 | 0.585 | 0.1113 | 1.013 |
| 2022-23 | 4-0 | 55949 | 0.577 | 0.578 | 0.0580 | 1.048 |
| 2022-23 | OT | 4391 | 0.508 | 0.546 | 0.1527 | 0.955 |
| 2023-24 | 48-36 | 140717 | 0.558 | 0.546 | 0.2329 | 0.955 |
| 2023-24 | 36-24 | 147749 | 0.561 | 0.548 | 0.1921 | 1.022 |
| 2023-24 | 24-12 | 145108 | 0.556 | 0.549 | 0.1435 | 1.019 |
| 2023-24 | 12-4 | 94587 | 0.563 | 0.551 | 0.0980 | 1.036 |
| 2023-24 | 4-0 | 55713 | 0.543 | 0.541 | 0.0508 | 1.069 |
| 2023-24 | OT | 3392 | 0.577 | 0.635 | 0.1489 | 0.935 |

## Held-out season 2024-25, scored once

596840 events from 1304 games of 2024-25, never fitted or tuned on; the selected candidate (logistic) refit on all 2292413 training rows scored them once.

| bucket | n | Brier | ECE | slope | benchmark (NBA.com WinProbabilityPBP) |
| --- | --- | --- | --- | --- | --- |
| 48-36 | 142433 | 0.2318 [0.2261, 0.2377] | 0.0145 [0.0087, 0.0415] | 1.0005 [0.8237, 1.1977] |  |
| 36-24 | 151539 | 0.1982 [0.1881, 0.2065] | 0.0208 [0.0142, 0.0413] | 0.9547 [0.8574, 1.0972] |  |
| 24-12 | 147390 | 0.1502 [0.1397, 0.1607] | 0.0129 [0.0094, 0.0337] | 0.9594 [0.8706, 1.0648] |  |
| 12-4 | 96071 | 0.1043 [0.0953, 0.1134] | 0.0124 [0.0099, 0.0301] | 0.9569 [0.8677, 1.0509] |  |
| 4-0 | 55910 | 0.0582 [0.0504, 0.0666] | 0.0079 [0.0063, 0.0181] | 0.9552 [0.8615, 1.0952] |  |
| OT | 3497 | 0.1452 [0.1194, 0.1745] | 0.0268 [0.0261, 0.1003] | 1.0499 [0.7729, 1.4783] |  |

Overall: log loss 0.4879 [0.4700, 0.5077], Brier 0.1658 [0.1590, 0.1732], mean bucket Brier 0.1486.

## Benchmark

NBA.com tables were skipped by the owner (gates/nbacom-skip, 2026-11-11), so the NBA.com WinProbabilityPBP benchmark was not collected and the benchmark column is empty. If it is collected later, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier.

## Outputs

- $DATA_ROOT/analytics/winprob/2020-21/winprob_states.parquet
- $DATA_ROOT/analytics/winprob/2020-21/winprob_calibration.parquet
- $DATA_ROOT/analytics/winprob/2021-22/winprob_states.parquet
- $DATA_ROOT/analytics/winprob/2021-22/winprob_calibration.parquet
- $DATA_ROOT/analytics/winprob/2022-23/winprob_states.parquet
- $DATA_ROOT/analytics/winprob/2022-23/winprob_calibration.parquet
- $DATA_ROOT/analytics/winprob/2023-24/winprob_states.parquet
- $DATA_ROOT/analytics/winprob/2023-24/winprob_calibration.parquet
- $DATA_ROOT/analytics/winprob/2024-25/winprob_states.parquet
- $DATA_ROOT/analytics/winprob/2024-25/winprob_calibration.parquet

## Mac command

The real run over the five seasons under DATA_ROOT (collect/stints/playbyplay/<season>.parquet, collect/extras/odds/<season>.parquet when present):

    python -m analytics.winprob.train --data "$DATA_ROOT" --out "$DATA_ROOT" --report reports/winprob.md --seed 0 --bootstrap 1000 --folds 5 --heldout-season 2024-25

## Assumptions

- A whole season is held out (default 2024-25) and scored once; folds inside the training seasons are grouped by game; seasons later than the held-out season are ignored entirely (listed under Inputs).
- 2015-16 games of sportvu/splits.json never fit anything: rows flagged heldout are dropped, the 2015-16 season is refused as a held-out season, and refuse_heldout runs on the training ids.
- The closing spread is the median over books of the is_close spreads market for the home side, joined on game_date and the home team name (analytics.winprob.features.TEAM_NAMES); null with a flag when absent.
- The possession indicator is read from event kinds and is null when the events so far do not say.
- The interaction is margin times sqrt(seconds remaining), as specified; the logistic model also sees margin and time on their own.
- On the fixtures the boosted candidate loses to the logistic one because the spread is nearly unique per game among a few dozen games and the trees memorise games through it; the selection rule picks the logistic candidate. This is a fixture artefact to re-check on the real seasons, where many games share a spread value.
- The HGB hyperparameters (depth 3, 50 rounds, 200 rows per leaf, L2 1.0) were set after seeing the fixture memorise games through the spread (a first setting of 15 leaves and 40 rows per leaf scored worse than the constant out of fold). They were chosen on fixture data, not on any real season, and the held-out season was not involved.
- The closing spread join is analytics.betting.validate's (per-book latest pre-tip is_close snapshot, median over books, both team keys, tip date shifted 8 h to US local with one day of tolerance); the games table's spread_home is used only when no odds table exists.
- The calibration slope is analytics.eval.calibration_slope_fast: the logistic regression of home_win on logit(p) of analytics.eval.calibration_slope, solved by a bounded Newton method because the lbfgs solver runs to its iteration cap on the nearly separable late-game buckets; Brier, ECE, log loss, the reliability table and the game bootstrap are analytics.eval's too.
