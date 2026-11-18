# winprob-E2: a forward-in-time temperature for the over-confidence

Written 2026-11-18. Pre-registration: docs/research/winprob.md, section 6, "winprob-E2"
(written 2026-11-16, before any winprob result). Data rules: PLAN.md D79, D89. Code: tools/research_winprob_e2.py,
reusing tools/research_winprob.py and tools/research_winprob_elo.py unchanged. Every number below is copied from
reports/research/winprob-E2-dev.json unless it says "computed". Intervals are 95 percent percentile intervals of a
2000-resample bootstrap of games (seed 0); paired differences use the same resampled games for both arms. Slopes are
point estimates.

**Verdict: not adopted on development. The test season was not scored.** On the development seasons, E1 with the
forward temperature has a worse mean bucket ECE than E1 alone, 0.0038 [-0.0021, 0.0068] (e2 minus e1), so the
pre-registered rule fails. The finding is the temperatures themselves: fitted season by season on forward
predictions, the logit multiplier is 0.894 (2021-22), 0.955 (2022-23) and 1.046 (2023-24). The model's over- or
under-confidence does not carry forward from one season to the next as one temperature. The lead decided not to
spend the one test score on a method that misses its bar on development (as for any stage that fails there), so
2025-26 was not read for winprob-E2 and the scoring ledger has no winprob-E2 entry. winprob-E1's result and
status do not change; its site use still waits for a D line.

## What is tested, and on top of what

winprob-E1 was adopted as a research result, so, as the pre-registration says, E2 runs on top of the E1 model: the
no-temperature arm is E1 (the frozen logistic's recipe plus the Elo drift term), not the frozen logistic.

The temperature is one number a that multiplies the logit of E1's probability, p' = sigmoid(a * logit(p)), with no
intercept, fitted by log loss. a above 1 sharpens the forecast, a below 1 softens it. Arms:

- e1: the E1 model, no temperature (the control).
- e2: E1 with the forward temperature, fitted on the predictions of a model fitted on 2020-21 to 2022-23 for
  2023-24 (the pre-registered split).
- e2_grouped (ablation): E1 with a temperature fitted on out-of-fold predictions under the frozen protocol (5 grouped
  folds by game inside the development seasons, seed 0).

Metric: mean ECE over the five regulation buckets (10 equal-width bins, the frozen analytics.eval.ece), primary; mean
bucket Brier as a guard; a pathwise peak-on-loss check, descriptive. Adoption: the paired 95 percent interval of e2
minus e1 mean bucket ECE lies below zero, and the upper end of the paired interval of e2 minus e1 mean bucket Brier is
at most +0.0005.

The peak-on-loss check is in the spirit of Pipping-Gamon and Wyner (2026), not their exact benchmark: for a level x
(0.9 and 0.95), at the first event where a side's win probability reaches x, a calibrated forecast says that side
then loses with probability 1 - p. Observed losses after reaching x are compared with that expected count (observed
over expected; 1 means calibrated at the top, below 1 means the forecast was too cautious there, above 1 too bold).

## Gates

- The frozen logistic gate of winprob-E1 still holds (all 2292413 stored out-of-fold predictions reproduced, largest
  difference 2.98e-8).
- The E1 arm refit here reproduces the E1 fits that winprob-E1 recorded: largest coefficient difference 0.0 for the
  three forward folds and for the fit on all development rows.
- Elo parameters equal winprob-E1's (K 15, home advantage 50, pull to the mean 0.25).
- Every development season run through the scoring path: no game dropped, no non-finite value, schedule agreement
  1.0, no Elo prefix violation.

## Temperatures (development seasons only)

| fitted on | a (logit multiplier) | log loss before | log loss after |
| --- | --- | --- | --- |
| forward predictions of 2021-22 (model fitted on 2020-21) | 0.8943 | 0.4660 | 0.4646 |
| forward predictions of 2022-23 (model fitted on 2020-21 to 2021-22) | 0.9550 | 0.4842 | 0.4840 |
| forward predictions of 2023-24 (model fitted on 2020-21 to 2022-23): **the pre-registered final e2 temperature** | 1.0459 | 0.4422 | 0.4420 |
| grouped folds, seasons before 2022-23 | 0.9950 | | |
| grouped folds, seasons before 2023-24 | 0.9951 | | |
| grouped folds, all development seasons: **the final e2_grouped temperature** | 0.9987 | | |

The forward temperature moves from softening (0.894) to sharpening (1.046) across three seasons. The grouped
temperature is near 1 every time (|log a| at most 0.02, a rule fixed in the code before this run).

## Development result (nested forward chaining, 2605 games of 2022-23 and 2023-24)

Each check season takes e2's temperature from the season before it (2022-23 from 2021-22, a 0.8943; 2023-24 from
2022-23, a 0.9550) and e2_grouped's from grouped folds over the seasons before it; the E1 model is the one fitted on
the seasons before the check season. 2021-22 has no earlier forward season, so it is not a check season here.

| arm | mean bucket ECE | mean bucket Brier | log loss |
| --- | --- | --- | --- |
| e1 | 0.0105 [0.0103, 0.0215] | 0.1393 [0.1340, 0.1449] | 0.4634 [0.4496, 0.4783] |
| e2 | 0.0144 [0.0121, 0.0249] | 0.1393 [0.1342, 0.1447] | 0.4638 [0.4507, 0.4779] |
| e2_grouped | 0.0106 [0.0104, 0.0214] | 0.1393 [0.1340, 0.1449] | 0.4634 [0.4496, 0.4783] |

| difference | mean bucket ECE | mean bucket Brier |
| --- | --- | --- |
| e2 minus e1 | 0.0038 [-0.0021, 0.0068] | 0.00004 [-0.00022, 0.00029] |
| e2_grouped minus e1 | 0.0001 [-0.0003, 0.0004] | -0.00000 [-0.00002, 0.00001] |
| e2 minus e2_grouped | 0.0037 [-0.0018, 0.0065] | 0.00004 [-0.00020, 0.00028] |

- By the pre-registered rule e2 **fails on development**: the ECE interval includes zero and the point is worse. The
  Brier guard holds.
- Each season alone, e2 minus e1 mean bucket ECE: 0.0036 [-0.0040, 0.0068] (2022-23), 0.0023 [-0.0010, 0.0044]
  (2023-24). Both points are worse.
- Slopes by bucket, 48-36 to 4-0 (point estimates, pooled): e1 0.967, 0.992, 0.980, 1.028, 1.033; e2 1.047, 1.074,
  1.060, 1.113, 1.120. In 2022-23, softening with 2021-22's temperature moved the 48-36 slope from 0.855 to 0.956
  but pushed the later buckets past 1 (12-4: 1.016 to 1.136). In 2023-24, where E1 was already under-confident
  (48-36 slope 1.094), softening with 2022-23's temperature pointed the wrong way (1.146). E1's calibration changes
  direction from season to season, so last season's correction does not carry over.
- Peak-on-loss at 0.9, observed over expected: e1 1.037 [0.930, 1.147], e2 0.870 [0.769, 0.975], e2_grouped 1.030
  [0.922, 1.141]. At 0.95: e1 0.924 [0.757, 1.101], e2 0.681 [0.535, 0.841], e2_grouped 0.907 [0.741, 1.078]. On
  this check E1 alone is consistent with calibration at the top; the softened e2 is too cautious there.
- Binned ECE is biased upward under resampling, so the percentile intervals of an arm's ECE sit at or above its point
  (e1: 0.0105 [0.0103, 0.0215]). The paired difference is read as pre-registered, with that caveat.

## Commitments before the test score

**Not used: the test season was not scored, because the development rule failed (the lead's decision; the scoring ledger has no winprob-E2 entry).** The section below is kept as it was written before that decision.

Fixed now, before score_once is called, after the development result above. The test season is 2025-26, 1316 games
(PLAN.md D89).

1. **The verdict is the pre-registered rule, read once on the test season**, on e2 minus e1: the paired interval of
   mean bucket ECE below zero, and the upper end of the paired Brier interval at most +0.0005. The final e2
   temperature is the pre-registered one, a 1.0459 (sharpening), and e2_grouped's is 0.9987.
2. **The development result stands beside it.** The rule fails on development. If the test season passes, the report
   says "passes on the test season, fails on development"; the forward temperature changed direction across the
   three development seasons, so a pass is read as one season agreeing with one temperature, not as a correction that
   transfers. Either way E2 is not proposed for the site without a new D line, and E1's site use is not decided by
   E2.
3. **Power.** Scaled from the development interval (half-width 0.0044 at 2605 games), the test half-width of the ECE
   difference is about 0.0062 at 1316 games (computed in the dev json). That is as large as E1's whole mean bucket
   ECE on development (0.0105), so an interval that includes zero is the likely outcome and is read as "no detectable
   change", not as evidence either way.
4. **Ablation.** The grouped temperature is near 1 (0.9987), so by the pre-registration's own reading, any effect of
   E2 comes from the forward split, not from temperature scaling as such. e2 minus e2_grouped is reported; no verdict
   rests on it.
5. **Disclosure: this season has been read before.** 2025-26 was scored once by winprob-E1 and once by lineups-E1.
   E2's temperatures are fitted on development seasons only. winprob-E1's test calibration slopes (point estimates,
   published in reports/research/winprob-E1.md: 1.018, 1.024, 1.012, 0.996, 1.111) were known before E2's
   development run. No choice in E2 depends on them: the split, the one-parameter temperature, the metric and the
   threshold are the pre-registration's, written before any winprob result; the choices made here (no intercept, a
   log-loss fit, the nested forward development stage, the peak-on-loss levels and the near-one rule) were fixed in
   code before the development run. A reader should still know that slopes a little above 1 on the test season were
   public before the score, and that the pre-registered temperature (1.046) sharpens.
6. **Slopes and peak-on-loss** on the test season are reported beside the verdict as point estimates and a
   descriptive check; neither changes the verdict.
7. **Data faults are reported, not fixed** (as in winprob-E1): games listed, without play-by-play, dropped by the
   builders, scored, non-finite counts, schedule agreement, and the test-season Elo prefix check. A non-finite value
   or an Elo prefix violation refuses the verdict; a schedule disagreement is reported beside it with no rescore. A
   crash uses up the score.
8. **The ECE caveat** above applies to the test season too; the verdict is read on the paired interval as
   pre-registered.

## What this means

- E1's calibration on development moves from over-confident early in 2022-23 (48-36 slope 0.855) to under-confident
  in 2023-24 (1.094). A single temperature fitted on one season and carried to the next makes ECE worse on both
  check seasons (0.0036 [-0.0040, 0.0068] in 2022-23, 0.0023 [-0.0010, 0.0044] in 2023-24, e2 minus e1).
- Inside the development seasons, with grouped folds, the temperature is about 1 (0.9987): pooled over seasons, E1 is
  calibrated on average. The miscalibration is season to season, which a fixed temperature cannot follow.
- On the pathwise check E1 alone is consistent with calibration at the top (observed over expected losses after
  reaching 0.9: 1.037 [0.930, 1.147]); the softened e2 is too cautious there (0.870 [0.769, 0.975]).
- The held-out over-confidence that motivated E2 (slopes about 0.96 for the frozen model on 2024-25) is one season of
  is consistent with this season-to-season movement; the development seasons give no sign of a fixed
  forward-in-time shift.
- A correction that could follow it would need information from the current season (for example a temperature
  updated game by game); that would be a new pre-registration, not a rereading of this one.
