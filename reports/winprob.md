# Live win probability (analytics/winprob/train.py)

Written 2026-11-05T09:42:11Z by `python -m analytics.winprob.train --fixtures --report reports/winprob.md --bootstrap 200`.

**Fixture run: every number below is fixture output, not a result.** The synthetic play-by-play of tests/_fx_winprob.py stands in for the five seasons on the owner's Mac.

## Inputs

| input | value |
| --- | --- |
| files | <fixtures seed 0> |
| events (states) | 15195 |
| games | 60 |
| seasons | 2020-21, 2021-22, 2024-25 |
| training seasons | 2020-21, 2021-22 (10181 events, 40 games) |
| held-out season | 2024-25 (5014 events, 20 games), set aside before any fit |
| rows flagged heldout (sportvu/splits.json games) dropped | 0 |
| guard | `analytics.eval.refuse_heldout` checked 40 training game ids; `analytics.schema.assert_time_forward` passed on the design matrix |
| closing spread coverage (training games) | 0.975 |
| possession known (training events) | 0.990 |
| training home win rate (by game) | 0.475 |

## Protocol

Unit: the game state at each play-by-play event; target: the home team wins. Features (time-forward, `analytics.winprob.features`): margin, seconds_remaining, overtime, period, possession_home, spread_home, spread_missing, margin_x_sqrt_time.
The final score never enters a feature; `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k).
Candidates: constant (training-fold home win rate); logistic regression on margin, seconds_remaining, possession_home (with a missing flag) and margin_x_sqrt_time (standardised, C=1); HistGradientBoostingClassifier on the same plus period, overtime, spread_home and spread_missing, calibrated by isotonic regression on inner out-of-fold predictions (3 folds by game). No deep model.
Selection: out-of-fold under 5 grouped folds by game within the training seasons (seed 0). The selection metric is the unweighted mean of the Brier score over the five regulation time buckets (minutes remaining 48-36, 36-24, 24-12, 12-4, 4-0; OT reported only). Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). The held-out season is scored once, by the selected candidate refit on all training rows.

## Candidates by time bucket (fixture output, not a result)

Brier score (lower is better), out-of-fold within the training seasons.

| model | 48-36 | 36-24 | 24-12 | 12-4 | 4-0 | OT | selection (mean of 5) | log loss overall | Brier overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | 0.2581 [0.2491, 0.2683] | 0.2583 [0.2499, 0.2689] | 0.2573 [0.2490, 0.2677] | 0.2586 [0.2502, 0.2690] | 0.2585 [0.2501, 0.2684] | n/a | 0.2582 | 0.7094 [0.6926, 0.7303] | 0.2581 [0.2497, 0.2685] |
| logistic (selected) | 0.2370 [0.1964, 0.2753] | 0.1957 [0.1242, 0.2582] | 0.0939 [0.0486, 0.1409] | 0.0784 [0.0282, 0.1279] | 0.0341 [0.0106, 0.0638] | n/a | 0.1278 | 0.4329 [0.3289, 0.5393] | 0.1473 [0.1078, 0.1882] |
| hgb_iso | 0.2990 [0.2512, 0.3417] | 0.2590 [0.2078, 0.3188] | 0.1926 [0.1407, 0.2354] | 0.1705 [0.1166, 0.2164] | 0.1509 [0.1088, 0.1898] | n/a | 0.2144 | 0.6506 [0.5415, 0.7583] | 0.2284 [0.1849, 0.2681] |

Rows per bucket: 48-36 2525, 36-24 2555, 24-12 2535, 12-4 1701, 4-0 865, OT 0.

Selected: **logistic** (lowest mean bucket Brier).

## Calibration by time bucket (fixture output, not a result)

ECE (10 bins) and calibration slope per bucket, out-of-fold.

| model | bucket | n | ECE | slope |
| --- | --- | --- | --- | --- |
| constant | 48-36 | 2525 | 0.1083 [0.0203, 0.2526] | n/a |
| constant | 36-24 | 2555 | 0.1144 [0.0303, 0.2606] | n/a |
| constant | 24-12 | 2535 | 0.0978 [0.0196, 0.2544] | n/a |
| constant | 12-4 | 1701 | 0.1125 [0.0192, 0.2614] | n/a |
| constant | 4-0 | 865 | 0.1166 [0.0277, 0.2597] | n/a |
| logistic | 48-36 | 2525 | 0.0895 [0.0543, 0.1988] | 0.6753 [0.0251, 1.5792] |
| logistic | 36-24 | 2555 | 0.1038 [0.0669, 0.2250] | 0.6484 [0.3519, 1.4976] |
| logistic | 24-12 | 2535 | 0.0453 [0.0371, 0.1212] | 1.1384 [0.7444, 2.5598] |
| logistic | 12-4 | 1701 | 0.0425 [0.0292, 0.1173] | 1.0744 [0.8303, 2.1698] |
| logistic | 4-0 | 865 | 0.0505 [0.0279, 0.0982] | 2.6010 [1.8378, 7.4726] |
| hgb_iso | 48-36 | 2525 | 0.2144 [0.1298, 0.3507] | -0.0471 [-0.7472, 0.3286] |
| hgb_iso | 36-24 | 2555 | 0.1823 [0.1129, 0.3246] | 0.2792 [-0.1821, 0.9024] |
| hgb_iso | 24-12 | 2535 | 0.1198 [0.0785, 0.2367] | 0.8820 [0.3703, 2.0929] |
| hgb_iso | 12-4 | 1701 | 0.0864 [0.0791, 0.2105] | 1.3020 [0.6107, 4.7151] |
| hgb_iso | 4-0 | 865 | 0.1070 [0.0695, 0.2176] | 2.2131 [1.4673, 4.0748] |

The constant model has no calibration slope by design.

### Reliability table of the selected candidate (out-of-fold) (fixture output, not a result)

| bin | lo | hi | n | mean p | mean home win |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 1999 | 0.025 | 0.033 |
| 1 | 0.1 | 0.2 | 735 | 0.146 | 0.200 |
| 2 | 0.2 | 0.3 | 746 | 0.251 | 0.307 |
| 3 | 0.3 | 0.4 | 975 | 0.354 | 0.414 |
| 4 | 0.4 | 0.5 | 1222 | 0.449 | 0.459 |
| 5 | 0.5 | 0.6 | 853 | 0.544 | 0.445 |
| 6 | 0.6 | 0.7 | 607 | 0.648 | 0.619 |
| 7 | 0.7 | 0.8 | 539 | 0.752 | 0.740 |
| 8 | 0.8 | 0.9 | 677 | 0.850 | 0.780 |
| 9 | 0.9 | 1.0 | 1828 | 0.977 | 0.981 |

### winprob_calibration rows of the selected candidate (training seasons, out-of-fold) (fixture output, not a result)

| season | time_bucket | n | mean_p | mean_y | brier | slope |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-21 | 48-36 | 1236 | 0.439 | 0.403 | 0.1963 | 1.438 |
| 2020-21 | 36-24 | 1236 | 0.431 | 0.394 | 0.1590 | 0.896 |
| 2020-21 | 24-12 | 1276 | 0.440 | 0.381 | 0.0804 | 1.500 |
| 2020-21 | 12-4 | 859 | 0.397 | 0.386 | 0.0597 | 1.239 |
| 2020-21 | 4-0 | 420 | 0.415 | 0.419 | 0.0243 | 2.630 |
| 2021-22 | 48-36 | 1289 | 0.480 | 0.569 | 0.2761 | 0.159 |
| 2021-22 | 36-24 | 1319 | 0.573 | 0.551 | 0.2300 | 0.469 |
| 2021-22 | 24-12 | 1259 | 0.556 | 0.566 | 0.1076 | 0.947 |
| 2021-22 | 12-4 | 842 | 0.499 | 0.577 | 0.0975 | 1.080 |
| 2021-22 | 4-0 | 445 | 0.487 | 0.546 | 0.0434 | 2.877 |

## Held-out season 2024-25, scored once (fixture output, not a result)

5014 events from 20 games of 2024-25, never fitted or tuned on; the selected candidate (logistic) refit on all 10181 training rows scored them once.

| bucket | n | Brier | ECE | slope | benchmark (NBA.com WinProbabilityPBP) |
| --- | --- | --- | --- | --- | --- |
| 48-36 | 1260 | 0.2344 [0.1933, 0.2882] | 0.1836 [0.0905, 0.4057] | 1.8904 [0.6624, 3.2003] |  |
| 36-24 | 1257 | 0.1947 [0.1295, 0.2798] | 0.1911 [0.0887, 0.3956] | 1.4153 [0.8428, 3.0320] |  |
| 24-12 | 1215 | 0.1685 [0.0817, 0.2955] | 0.1019 [0.0531, 0.2970] | 0.5894 [0.3427, 1.7108] |  |
| 12-4 | 821 | 0.1140 [0.0406, 0.2119] | 0.0640 [0.0428, 0.1946] | 0.6863 [0.3887, 2.4497] |  |
| 4-0 | 461 | 0.0538 [0.0122, 0.1171] | 0.0607 [0.0315, 0.1461] | 0.8951 [0.6039, 142.7178] |  |

Overall: log loss 0.5038 [0.3698, 0.7094], Brier 0.1721 [0.1236, 0.2483], mean bucket Brier 0.1531.

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
- The HGB hyperparameters (depth 3, 50 rounds, 200 rows per leaf, L2 1.0) were set after seeing the fixture memorise games through the spread (a first setting of 15 leaves and 40 rows per leaf scored worse than the constant out of fold). They were chosen on fixture data, not on any real season, and the held-out season was not involved.
- The closing spread join is analytics.betting.validate's (per-book latest pre-tip is_close snapshot, median over books, both team keys, tip date shifted 8 h to US local with one day of tolerance); the games table's spread_home is used only when no odds table exists.
- The calibration slope is the logistic regression of home_win on logit(p) of analytics.eval.calibration_slope, solved here by a bounded Newton method (analytics.winprob.train.slope) because the eval solver runs to its iteration cap on the nearly separable late-game buckets; Brier, ECE, log loss, the reliability table and the game bootstrap are analytics.eval's.
