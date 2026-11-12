# Live win probability (analytics/winprob/train.py)

**Real data, training folds only; held-out not scored.** Stage 5 phase 5 (docs/mac_runbook.md), run with --skip-heldout on 2026-11-12 (train rerun after the OUT_ROOT fix, same numbers, tables written); the 2024-25 season is held out; no odds (betting skipped, gates/odds-skip), so the closing spread is missing on every game; NBA.com skipped (gates/nbacom-skip), so the benchmark column is empty.

Written 2026-11-12T11:32:39Z by `python -m analytics.winprob.train --data $DATA_ROOT --seed 0 --report reports/winprob.md --skip-heldout`.

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

Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): margin, seconds_remaining, overtime, period, possession_home, spread_home, spread_missing, margin_x_sqrt_time.
The final score never enters a feature; `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k).
Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag) and margin_x_sqrt_time (standardised, C=1); HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.
Selection: out-of-fold under 5 grouped folds by game within the training seasons (seed 0). The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). The held-out season is set aside and never scored (--skip-heldout).

## Candidates by time bucket

Brier score (lower is better), out-of-fold within the training seasons.

| model | 48-36 | 36-24 | 24-12 | 12-4 | 4-0 | OT | selection (mean of 5) | log loss overall | Brier overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 0.2471 [0.2454, 0.2487] | 0.2471 [0.2453, 0.2487] | 0.2470 [0.2453, 0.2486] | 0.2469 [0.2451, 0.2485] | 0.2477 [0.2460, 0.2493] | 0.2504 [0.2442, 0.2572] | 0.2472 | 0.6874 [0.6839, 0.6906] | 0.2471 [0.2454, 0.2487] |
| logistic (selected) | 0.2311 [0.2285, 0.2335] | 0.1943 [0.1906, 0.1999] | 0.1532 [0.1480, 0.1597] | 0.1040 [0.0987, 0.1092] | 0.0590 [0.0559, 0.0621] | 0.1626 [0.1492, 0.1765] | 0.1483 | 0.4884 [0.4804, 0.4989] | 0.1655 [0.1625, 0.1693] |
| hgb_iso | 0.2316 [0.2284, 0.2347] | 0.1946 [0.1906, 0.2003] | 0.1526 [0.1479, 0.1588] | 0.1040 [0.0990, 0.1089] | 0.0603 [0.0572, 0.0633] | 0.1653 [0.1514, 0.1790] | 0.1486 | 0.4897 [0.4818, 0.5004] | 0.1656 [0.1627, 0.1695] |

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
| logistic | 48-36 | 549307 | 0.0140 [0.0102, 0.0245] | 1.2064 [1.1118, 1.3023] |
| logistic | 36-24 | 581066 | 0.0082 [0.0072, 0.0186] | 0.9820 [0.9252, 1.0339] |
| logistic | 24-12 | 561851 | 0.0252 [0.0178, 0.0337] | 0.8703 [0.8231, 0.9108] |
| logistic | 12-4 | 367834 | 0.0119 [0.0085, 0.0186] | 0.9804 [0.9376, 1.0270] |
| logistic | 4-0 | 218146 | 0.0504 [0.0461, 0.0546] | 1.6891 [1.6046, 1.7799] |
| logistic | OT | 14209 | 0.0596 [0.0497, 0.0918] | 1.6041 [1.3572, 1.9182] |
| hgb_iso | 48-36 | 549307 | 0.0247 [0.0182, 0.0330] | 0.7788 [0.7130, 0.8395] |
| hgb_iso | 36-24 | 581066 | 0.0097 [0.0075, 0.0215] | 0.9222 [0.8630, 0.9734] |
| hgb_iso | 24-12 | 561851 | 0.0071 [0.0057, 0.0171] | 0.9632 [0.9047, 1.0156] |
| hgb_iso | 12-4 | 367834 | 0.0072 [0.0055, 0.0149] | 1.0273 [0.9751, 1.0826] |
| hgb_iso | 4-0 | 218146 | 0.0549 [0.0509, 0.0592] | 1.7307 [1.6554, 1.8136] |
| hgb_iso | OT | 14209 | 0.0567 [0.0456, 0.0901] | 1.5293 [1.3136, 1.7904] |

The constant model has no calibration slope by design.

### Reliability table of the selected candidate (out-of-fold)

| bin | lo | hi | n | mean p | mean home win |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 217517 | 0.036 | 0.034 |
| 1 | 0.1 | 0.2 | 124640 | 0.150 | 0.170 |
| 2 | 0.2 | 0.3 | 134562 | 0.251 | 0.260 |
| 3 | 0.3 | 0.4 | 164242 | 0.352 | 0.345 |
| 4 | 0.4 | 0.5 | 239293 | 0.454 | 0.437 |
| 5 | 0.5 | 0.6 | 385738 | 0.551 | 0.553 |
| 6 | 0.6 | 0.7 | 283366 | 0.647 | 0.655 |
| 7 | 0.7 | 0.8 | 207349 | 0.748 | 0.745 |
| 8 | 0.8 | 0.9 | 188133 | 0.850 | 0.843 |
| 9 | 0.9 | 1.0 | 347573 | 0.966 | 0.967 |

### winprob_calibration rows of the selected candidate (training seasons, out-of-fold)

| season | time_bucket | n | mean_p | mean_y | brier | slope |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-21 | 48-36 | 124987 | 0.553 | 0.543 | 0.2325 | 1.193 |
| 2020-21 | 36-24 | 132454 | 0.547 | 0.544 | 0.1919 | 1.046 |
| 2020-21 | 24-12 | 127107 | 0.540 | 0.545 | 0.1477 | 0.908 |
| 2020-21 | 12-4 | 84064 | 0.540 | 0.548 | 0.1015 | 0.984 |
| 2020-21 | 4-0 | 50898 | 0.542 | 0.544 | 0.0582 | 1.664 |
| 2020-21 | OT | 3221 | 0.513 | 0.486 | 0.1487 | 1.726 |
| 2021-22 | 48-36 | 140981 | 0.554 | 0.547 | 0.2313 | 1.246 |
| 2021-22 | 36-24 | 150099 | 0.555 | 0.546 | 0.1959 | 0.960 |
| 2021-22 | 24-12 | 144433 | 0.550 | 0.545 | 0.1584 | 0.819 |
| 2021-22 | 12-4 | 94057 | 0.552 | 0.544 | 0.1044 | 0.965 |
| 2021-22 | 4-0 | 55586 | 0.550 | 0.539 | 0.0573 | 1.741 |
| 2021-22 | OT | 3205 | 0.514 | 0.443 | 0.1858 | 1.384 |
| 2022-23 | 48-36 | 142622 | 0.558 | 0.583 | 0.2276 | 1.250 |
| 2022-23 | 36-24 | 150764 | 0.562 | 0.583 | 0.1970 | 0.933 |
| 2022-23 | 24-12 | 145203 | 0.566 | 0.586 | 0.1624 | 0.845 |
| 2022-23 | 12-4 | 95126 | 0.576 | 0.585 | 0.1115 | 0.980 |
| 2022-23 | 4-0 | 55949 | 0.577 | 0.578 | 0.0642 | 1.666 |
| 2022-23 | OT | 4391 | 0.512 | 0.546 | 0.1596 | 1.698 |
| 2023-24 | 48-36 | 140717 | 0.558 | 0.546 | 0.2331 | 1.136 |
| 2023-24 | 36-24 | 147749 | 0.560 | 0.548 | 0.1921 | 1.002 |
| 2023-24 | 24-12 | 145108 | 0.556 | 0.549 | 0.1436 | 0.918 |
| 2023-24 | 12-4 | 94587 | 0.563 | 0.551 | 0.0981 | 0.998 |
| 2023-24 | 4-0 | 55713 | 0.545 | 0.541 | 0.0561 | 1.695 |
| 2023-24 | OT | 3392 | 0.567 | 0.635 | 0.1577 | 1.641 |

## Held-out season 2024-25, not scored

1304 held-out game(s) set aside before any fit and never fitted, tuned or scored: held-out games not scored (--skip-heldout).

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

## Mac command

The real run over the five seasons under DATA_ROOT (collect/stints/playbyplay/<season>.parquet, collect/extras/odds/<season>.parquet when present):

    python -m analytics.winprob.train --data "$DATA_ROOT" --out "$DATA_ROOT" --report reports/winprob.md --seed 0 --bootstrap 1000 --folds 5 --heldout-season 2024-25

## Assumptions

- A whole season is held out (default 2024-25) and set aside unscored in this run (--skip-heldout); folds inside the training seasons are grouped by game; seasons later than the held-out season are ignored entirely (listed under Inputs).
- 2015-16 games of sportvu/splits.json never fit anything: rows flagged heldout are dropped, the 2015-16 season is refused as a held-out season, and refuse_heldout runs on the training ids.
- The closing spread is the median over books of the is_close spreads market for the home side, joined on game_date and the home team name (analytics.winprob.features.TEAM_NAMES); null with a flag when absent.
- The possession indicator is read from event kinds and is null when the events so far do not say.
- The interaction is margin times sqrt(seconds remaining), as specified; the logistic model also sees margin and time on their own.
- On the fixtures the boosted candidate loses to the logistic one because the spread is nearly unique per game among a few dozen games and the trees memorise games through it; the selection rule picks the logistic candidate. This is a fixture artefact to re-check on the real seasons, where many games share a spread value.
- The HGB hyperparameters (depth 3, 50 rounds, 200 rows per leaf, L2 1.0) were set after seeing the fixture memorise games through the spread (a first setting of 15 leaves and 40 rows per leaf scored worse than the constant out of fold). They were chosen on fixture data, not on any real season, and the held-out season was not involved.
- The closing spread join is analytics.betting.validate's (per-book latest pre-tip is_close snapshot, median over books, both team keys, tip date shifted 8 h to US local with one day of tolerance); the games table's spread_home is used only when no odds table exists.
- The calibration slope is analytics.eval.calibration_slope_fast: the logistic regression of home_win on logit(p) of analytics.eval.calibration_slope, solved by a bounded Newton method because the lbfgs solver runs to its iteration cap on the nearly separable late-game buckets; Brier, ECE, log loss, the reliability table and the game bootstrap are analytics.eval's too.
