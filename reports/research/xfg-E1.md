# xfg-E1: cross Venn-Abers calibration (the research pilot)

Written 2026-11-16. Pre-registration: docs/research/xfg.md, section "xfg-E1"; commitments before the test score:
PLAN.md D80. Code: tools/research_xfg.py. Every number below is copied from reports/research/xfg-E1-dev.json and
reports/research/xfg-E1-final.json; the single test score is in reports/research/scoring_ledger.json.

**Verdict: not adopted.** CVAP passed the threshold on the development games but not on the test set. Read as
pre-committed (D80): the test interval straddles the threshold, so the result is inconclusive at 10 games, not
evidence that CVAP is worse. The frozen isotonic step stays. The site rule holds on both sets (the 95th percentile of
the per-shot interval width is under 0.01), so xFG is shown as a point estimate whose calibration step moves it by
under 0.01.

## What was tested

The frozen xFG model (hgb_contest_prior) calibrates its booster with an isotonic map fitted on 3 inner folds. The
experiment replaced that map with a cross Venn-Abers predictor (Vovk, Petej, Fedorova, NeurIPS 2015) on the same 3
inner folds, merged by the log-loss minimax rule. Hypothesis: same log loss, plus an interval (p0, p1) for every shot.
Threshold, fixed before any run: the 95 percent upper bound of the paired log loss difference (CVAP minus isotonic)
is at most +0.0005, on the development games and on the test set. Ablation: a single inner split (IVAP).

## Gate

The isotonic arm, refit with the frozen functions on all 607 training games under the frozen folds, reproduced the
stored out-of-fold predictions of all 101853 training shots: largest difference 2.97e-8 (cap 1e-6), no fold
mismatch.

## Development games (607 games, 101853 shots, 5 grouped outer folds)

| Arm | Log loss [95% interval] | Brier | Calibration slope | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| isotonic (frozen) | 0.6294 [0.6273, 0.6316] | 0.2209 | 1.000 | 0.0031 |
| CVAP | 0.6289 [0.6268, 0.6310] | 0.2207 | 1.020 | 0.0029 |
| IVAP (ablation) | 0.6301 | 0.2212 | 1.007 | 0.0045 |

- CVAP minus isotonic: -0.00052 [-0.00081, -0.00024] (shot-weighted, bootstrap of games, 2000 resamples). Passes.
- IVAP minus isotonic: +0.00067 [+0.00027, +0.00109]. The gain comes from the cross-over, not from Venn-Abers alone.
- Interval width GM(p1) - GM(p0): median 0.0009, 95th percentile 0.0077 (per fold 0.0083, 0.0069, 0.0073).
- The noise floor of the 10-bin ECE at this sample size is 0.0037 [0.0024, 0.0056]; both arms sit under it.

## Test set (Set A: the 10 untouched 2015-12-05 games, 1668 shots, scored once)

| Arm | Log loss [95% interval] | Brier | Calibration slope [95% interval] | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| isotonic (frozen) | 0.6428 [0.6261, 0.6597] | 0.2265 | 0.892 [0.705, 1.165] | 0.0299 |
| CVAP | 0.6434 [0.6277, 0.6599] | 0.2271 | 0.902 [0.718, 1.173] | 0.0309 |
| IVAP (ablation) | 0.6422 [0.6273, 0.6581] | 0.2269 | 0.948 | 0.0239 |

- CVAP minus isotonic: +0.00067 [-0.00133, +0.00229] (bootstrap of 10 games). Upper bound above +0.0005: fails.
- The pre-committed sensitivity check, a t interval on the 10 per-game means: +0.00068 [-0.00158, +0.00293]. It
  agrees with the bootstrap.
- Interval width: median 0.0007, 95th percentile 0.0062. Under 0.01, as on the development games.
- Contest features exist for 56.1 percent of test shots against 61.3 percent of development shots.
- The noise floor of the 10-bin ECE at 1668 shots is 0.0291 [0.0177, 0.0421]; both arms sit inside it.

## What this means

- The calibration step is not where the uncertainty of an xFG value lives: the Venn-Abers interval is under 0.01
  wide for 95 percent of shots on both sets. The honest per-shot statement is a point estimate, as the site already
  says, now backed by a measurement.
- CVAP's development gain (0.0005 in log loss) is real on 607 games but too small to confirm on 10. A larger untouched
  set would be needed, and none exists for 2015-16 (docs/research/DATA.md).
- The isotonic arm is the frozen model (refit on every training game). On the 10 untouched games its log loss is 0.6428
  [0.6261, 0.6597], against 0.6294 out of fold: the out-of-fold value lies just inside that interval, and calibration
  is within the noise of 1668 shots. This is the frozen model's first score on games no part of V3 ever read.
- Set B (the ACB games) was not scored: the frozen features cannot be built from that data without a new adapter
  (D80).

## What would change the verdict

A test set of at least 100 untouched games (the power estimate in PLAN.md D77 and the development interval put the
needed size near there for a 0.0005 effect). Until then CVAP stays a development-only finding.
