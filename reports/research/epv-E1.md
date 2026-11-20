# epv-E1: train on expected shot value instead of the make or miss

Written 2026-11-19. Pre-registration: docs/research/epv.md, section 6, "epv-E1", with its D117 amendment (PLAN.md
D116, D117). Scoring code: tools/research_epv.py at commit 135bea28; every piece was computed at that commit and the
dev json was assembled there (it records both). No log of the assembly run was kept; the json pins code_commit
135bea28. The json's stage string says "no Set A or held-out row read"; more exactly, no held-out row was fitted or
scored (the frozen loader reads those rows and sets them aside) and no Set A game was read. The json is left as
written. Every number below is copied from reports/research/epv-E1-dev.json, except where marked: the frozen lane's
numbers are from reports/epv.md and reports/xfg.md, the per-fold differences (in "Development games") were computed
after the result from the stored pieces, and the known fault's rate is the post-result review's measurement. Intervals are 95 percent
percentile intervals of a 2000-resample bootstrap of games (seed 0), with all arms paired on the same states and the
same resampled games.

**Verdict: not adopted, on development (D117).** Training the frozen GRU on the expected value of a possession's final
shot instead of its make or miss makes the out-of-fold error against realized points worse, not better:
MSE(baseline) minus MSE(e1) is -0.02247 [-0.02433, -0.02053], wholly below zero. The 0 to 4 s calibration slope of e1
(0.922) is inside 0.9 to 1.1, but the first condition fails. Set A is not scored (D117), and this module has no Set A
path at all.

## What is tested

The frozen EPV model, a GRU over the last 5 states of a possession, learns expected points to the end of the
possession from realized points, so every shot's make or miss enters its training label. epv-E1 replaces, for
possessions that end on their final field-goal attempt, the realized points of that attempt by its expected points
(xFG probability times 2 or 3) in the label of the states before the attempt; free-throw points stay realized.
Evaluation is always against realized points.

- Arms: baseline (the frozen calibrated GRU on the frozen label, refit out of fold); e1 (the smoothed label); the
  pre-registered ablation three_only (only 3-point attempts smoothed).
- Leakage guard (pre-registered): the xFG probability of a training row's shot comes from the frozen xFG recipe refit
  out of fold inside each EPV fold's training games, so it never saw that game nor any test game of the EPV fold.
- The calibration step: for e1 and three_only the regressor trains on the label while the frozen isotonic map to
  P(points > 0) is fitted on the realized points > 0 of the same rows (under a smoothed label "points > 0" is almost
  always true). This was fixed in the code's docstring before any number. The baseline is the frozen fit unchanged.
- Threshold (pre-registered, D117): MSE(baseline) minus MSE(e1) with its interval above 0, and e1's 0 to 4 s slope
  inside 0.9 to 1.1, out of fold under the frozen 5 grouped folds by game.

## Gates

- Gate A: the baseline arm, refit with frozen code under the frozen folds, reproduced the stored out-of-fold
  predictions of all 1,739,119 training states exactly (largest difference 0.0, no fold mismatch). So the baseline
  here is the frozen model, and its MSE 1.21554 equals the frozen report's 1.2155.
- Gate B: the nested xFG refits' log loss in each EPV fold: 0.62929, 0.62982, 0.62979, 0.62967 and 0.62912, all inside
  the frozen xFG out-of-fold interval [0.6277, 0.6316].
- Smoothed rows per fold: about 935,000 states (five-fold mean 934,839.2) in about 66,300 possessions for e1 (about
  266,000 states in about 18,900 possessions for three_only), of about 1.39 million training states a fold (computed
  from the counts).
- Known fault of 135bea28 (found by the post-result review; the scored code is not changed): when a possession's final
  attempt has no xFG row, the code can pick an earlier attempt of the possession as its "final" one. The reviewer
  measured it in 1 of 3,476 possessions over 25 sampled fold-0 games (01.01.2016.DAL.at.MIA, possession 47), with 4
  states left unsmoothed. That is about 0.03 percent of possessions against a loss of about 1.8 percent of the
  baseline MSE (computed), so it cannot explain the result.

## Development games (604 games, 1,739,119 states, out of fold)

| Arm | MSE against realized points | Brier P(points > 0) | Calibration slope, all (point estimate) | Slope, 0 to 4 s (point estimate) |
| --- | --- | --- | --- | --- |
| baseline (frozen) | 1.21554 [1.20789, 1.22265] | 0.23500 [0.23453, 0.23547] | 0.964 | 0.916 |
| e1 | 1.23801 [1.22935, 1.24589] | 0.23631 [0.23586, 0.23675] | 0.988 | 0.922 |
| three_only (ablation) | 1.22071 [1.21271, 1.22817] | 0.23488 [0.23442, 0.23533] | 1.025 | 0.936 |

- MSE(baseline) minus MSE(e1): -0.02247 [-0.02433, -0.02053]. e1 is worse.
- MSE(baseline) minus MSE(three_only): -0.00517 [-0.00614, -0.00415]. Smoothing only the threes is also worse, by less.
- Per fold (computed after the result from the stored pieces with a one-off script, read by no rule): MSE(baseline)
  minus MSE(e1) is -0.02834, -0.01719, -0.02135, -0.02178 and -0.02371 in folds 0 to 4 (about 345,000 to 351,000
  states each); MSE(baseline) minus MSE(three_only) is -0.00428, -0.00560, -0.00527, -0.00635 and -0.00434. Every fold
  shows the loss, so it is not the noise of one refit.
- By time elapsed in the possession, MSE(baseline) minus MSE(e1): 0 to 4 s -0.00859 [-0.01056, -0.00656] (443,333
  states); 4 to 10 s -0.01743 [-0.01942, -0.01548] (585,059); 10 to 24 s -0.03527 [-0.03789, -0.03258] (710,727; this
  bucket also holds the states past 24 s). The loss grows with time elapsed in the possession.
- Calibration slopes (point estimates) move toward 1 under e1 overall (0.964 to 0.988), at 0 to 4 s (0.916 to 0.922)
  and at 10 to 24 s (0.952 to 0.984), and away from it at 4 to 10 s (1.005 to 1.016); Brier is slightly worse overall.
  Per-bucket Brier values in the dev json are point estimates.

## Reading

The smoothed label moves most calibration slopes toward 1 (not at 4 to 10 s), but the model it trains predicts
realized points worse, most of all late in the possession. A plausible reading, not tested
here: the frozen GRU trained on realized points learns more about a shot's chance than xFG holds (handler and lineup
priors, the last 5 states), and the smoothed label replaces that with xFG's own view, errors included. The
pre-registration expected a tiny gain at most; the measured effect is a clear loss, about 1.8 percent of the
baseline MSE (computed: 0.02247 / 1.21554).

## Set A

Not scored (PLAN.md D117): Set A could not decide this test, and it already has three uses. The module has no
scorer, no final step and no test-data read; a test checks that.
