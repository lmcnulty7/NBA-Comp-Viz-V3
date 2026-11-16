# xfg-E1: cross Venn-Abers calibration (the research pilot)

Written 2026-11-16. Pre-registration: docs/research/xfg.md, section "xfg-E1"; commitments before the test score:
PLAN.md D80. Scoring code: tools/research_xfg.py at commit b2e5ca34 (later commits change file paths and guards only).
Every number below is copied from reports/research/xfg-E1-dev.json or reports/research/xfg-E1-final.json; the single
test score is the one entry of reports/research/scoring_ledger.json. Intervals are 95 percent percentile intervals of
a 2000-resample bootstrap of games.

**Verdict: not adopted.** CVAP passed the threshold on the development games but not on the test set. Read as
pre-committed (D80): the test interval straddles the threshold, so the result is inconclusive at 10 games, not
evidence that CVAP is worse. The frozen isotonic step stays. The site rule holds on both sets: for 95 percent of shots
the per-shot interval is under 0.01 wide (the widest single shot: 0.134 on development, 0.082 on the test set), so xFG
is shown as a point estimate whose calibration step moves it by under 0.01 for 95 percent of shots.

## What was tested

The frozen xFG model (hgb_contest_prior) calibrates its booster with an isotonic map fitted on 3 inner folds. The
experiment replaced that map with a cross Venn-Abers predictor (Vovk, Petej, Fedorova, NeurIPS 2015) on the same 3
inner folds, merged by the log-loss minimax rule. Hypothesis (non-inferiority): no worse log loss, plus an interval
(p0, p1) for every shot. Threshold, fixed before any run: the 95 percent upper bound of the paired log loss difference
(CVAP minus isotonic) is at most +0.0005, on the development games and on the test set. Ablation: a single inner
split (IVAP).

The control arm is the frozen recipe refit with the frozen functions and seed 0 on all 607 training games, the same
rows as the frozen fit; the gate below checks it at the out-of-fold level.

## Gate

The isotonic arm, refit with the frozen functions on all 607 training games under the frozen folds, reproduced the
stored out-of-fold predictions of all 101853 training shots: largest difference 2.97e-8 (cap 1e-6), no fold
mismatch.

## Development games (607 games, 101853 shots, 5 grouped outer folds)

| Arm | Log loss | Brier | Calibration slope | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| isotonic (control) | 0.6294 [0.6273, 0.6316] | 0.2209 [0.2200, 0.2219] | 1.000 [0.976, 1.025] | 0.0031 [0.0027, 0.0070] |
| CVAP | 0.6289 [0.6268, 0.6310] | 0.2207 [0.2198, 0.2217] | 1.020 [0.996, 1.046] | 0.0028 [0.0028, 0.0069] |
| IVAP (ablation) | 0.6301 [0.6280, 0.6322] | 0.2212 [0.2203, 0.2221] | 1.007 [0.983, 1.032] | 0.0045 [0.0035, 0.0083] |

- CVAP minus isotonic: -0.00052 [-0.00081, -0.00024] (shot-weighted). Passes.
- IVAP minus isotonic: +0.00067 [+0.00027, +0.00109]. On these games the three-fold version beats the single split;
  the single split also calibrates on one third of the data, so this mixes the cross-over with calibration-set size.
- Interval width GM(p1) - GM(p0): median 0.0009, 95th percentile 0.0077 (per fold 0.0083, 0.0069, 0.0073).
- The noise floor of the 10-bin ECE at this sample size is 0.0037 [0.0024, 0.0056]; both arms sit under it.

## Test set (Set A: the 10 untouched 2015-12-05 games, 1668 shots, scored once)

| Arm | Log loss | Brier | Calibration slope | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| isotonic (control) | 0.6428 [0.6261, 0.6597] | 0.2265 [0.2197, 0.2334] | 0.892 [0.705, 1.165] | 0.0299 [0.0229, 0.0586] |
| CVAP | 0.6434 [0.6277, 0.6599] | 0.2271 [0.2205, 0.2339] | 0.902 [0.718, 1.173] | 0.0309 [0.0275, 0.0771] |
| IVAP (ablation) | 0.6422 [0.6273, 0.6581] | 0.2269 [0.2205, 0.2338] | 0.948 [0.764, 1.206] | 0.0239 [0.0237, 0.0635] |

- CVAP minus isotonic: +0.00067 [-0.00133, +0.00229]. Upper bound above +0.0005: fails.
- The pre-committed sensitivity check, a t interval on the 10 per-game means: +0.00068 [-0.00158, +0.00293]. It
  agrees with the bootstrap.
- IVAP minus isotonic: -0.00056 [-0.00428, +0.00251]. The development ordering of the ablation is not confirmed on 10
  games.
- Interval width: median 0.0007, 95th percentile 0.0062 (per fold 0.0060, 0.0047, 0.0060). Under 0.01, as on the
  development games.
- Contest features exist for 56.1 percent of test shots against 61.3 percent of development shots.
- The noise floor of the 10-bin ECE at 1668 shots is 0.0291 [0.0177, 0.0421]; every arm sits inside it.
- Binned ECE is biased upward under resampling, so its bootstrap intervals sit above the point estimate; read them as
  a range of plausible values, not as centred intervals.

## What this means

- The calibration step is not where the uncertainty of an xFG value lives: the Venn-Abers interval is under 0.01 wide
  for 95 percent of shots on both sets. The per-shot statement on the site stays a point estimate, now backed by a
  measurement.
- CVAP's development interval excludes zero (in its favour), but the effect, 0.0005 in log loss, is too small to
  confirm on 10 games. None of the untouched 2015-16 data is left to enlarge the test (docs/research/DATA.md).
- The control arm of xfg-E1 on Set A: log loss 0.6428 [0.6261, 0.6597], against 0.6294 out of fold; the out-of-fold
  value lies inside that interval, and calibration is within the noise of 1668 shots. This is a 10-game control-arm
  result of one experiment, not a new score of the frozen lane: it does not replace or sit beside the frozen held-out
  result in reports/results.md.
- Set B (the ACB games) was not scored: the frozen features cannot be built from that data without a new adapter
  (D80).

## What would change the verdict

A larger untouched test set. From the development interval (a standard error of about 0.00014 at 607 games), about
100 untouched games would give 80 percent power at a 0.001 margin. No such set exists for 2015-16, so CVAP stays a
development-only finding.
