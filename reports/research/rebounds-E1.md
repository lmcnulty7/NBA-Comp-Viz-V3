# rebounds-E1: a temperature and an offset per candidate type

Written 2026-11-19. Pre-registration: docs/research/rebounds.md, section 6, "rebounds-E1", with its D125 amendment
(PLAN.md D124, D125, D126). Scoring code: tools/research_rebounds.py. The five fold pieces were computed at commit
58ef8a9b; the gate was rerun and the dev json assembled at e8909861 (the dev json records it as code_commit and each
fold's commit in per_fold), which differs from 58ef8a9b only in the gate tolerance (D126). Every number below is copied
from reports/research/rebounds-E1-dev.json (the frozen 1.6009 from its gate record), except the first gate run's
numbers, which are from its log (fit1.log in the work folder). Intervals are 95 percent percentile
intervals of a 2000-resample bootstrap of games (seed 0), with all arms paired on the same misses and the same
resampled games.

**Verdict: not adopted, on development (D125).** Separate sharpening per candidate type changes the per-miss log loss by
-0.004628 [-0.005472, -0.003816]: the interval excludes zero, but the point does not reach the pre-registered bar of
-0.005. Almost all of E1's gain comes from a separate slope per type, not from the offsets. The site keeps the frozen
single tau. Set A is not scored (D125).

## What is tested

The frozen rebound model (gradient boosting on candidate rows) sharpens its within-miss probabilities with one
exponent, tau, fitted on inner out-of-fold predictions: p proportional to q to the power tau. rebounds-E1 replaces it
with p proportional to exp(a_g log q + b_g), g the candidate type: three slopes and two free offsets (offense 0),
fitted the same way on each outer fold's inner out-of-fold rows.

- Baseline: the frozen hgb with its single tau, refit with frozen code on the development games.
- Ablation (pre-registered): one shared slope and the two type offsets.
- Metric: out-of-fold per-miss multinomial log loss (lower is better).
- Threshold (pre-registered, D125): E1 minus baseline at most -0.005 with its 95 percent interval excluding zero, on
  the development games.

## Inputs and the gate

- Candidate rows: the frozen run stored no candidate-row files (the pre-registration assumed it had), so the rows of
  every development game were rebuilt with the frozen code, one game's frames at a time. They equal the frozen record:
  537,175 rows, 49,288 misses, 604 games.
- Gate A, first run (refused): all five folds refit with the frozen code matched every stored row (537,175 matched, no
  fold mismatch) and gave the per-miss log loss 1.600912 (the frozen 1.6009), but the largest difference from the
  stored probabilities was 2.98e-8 against a tolerance of 1e-9 (from the first gate's log). The stored p_selected is a
  32-bit float, so half an ulp in [0.5, 1) is 2^-25, about 2.98e-8: no refit could meet 1e-9.
- PLAN.md D126 set the tolerance to 2^-24 (5.96e-8), with the log loss check unchanged. The gate was rerun on the saved
  fold outputs, with no refit, and passed: largest difference 2.98e-8 against 5.96e-8, every row matched, log loss
  1.600912. So the baseline here is the frozen model.

## Development games (604 games, 49,288 misses, out of fold)

| Arm | Per-miss log loss | Top-1 | Favourite calibration slope (point estimate) | ECE, 10 equal-mass bins (point estimate) |
| --- | --- | --- | --- | --- |
| baseline (frozen single tau) | 1.60091 [1.58873, 1.61273] | 0.47968 [0.47474, 0.48483] | 1.368 | 0.0411 |
| E1 (slope and offset per type) | 1.59628 [1.58423, 1.60808] | 0.47735 [0.47235, 0.48232] | 1.300 | 0.0321 |
| ablation (shared slope, offsets) | 1.60033 [1.58810, 1.61238] | 0.48212 [0.47703, 0.48712] | 1.346 | 0.0416 |

- E1 minus baseline: -0.004628 [-0.005472, -0.003816]. Below zero, but above the -0.005 bar: the rule fails.
- Ablation minus baseline: -0.000581 [-0.000913, -0.000255]. E1 minus ablation: -0.004047 [-0.004836, -0.003281]. Almost
  all of E1's gain comes from a separate slope per type, not from the offsets.
- The favourite's calibration slope moves toward 1 under E1 (1.368 to 1.300) but stays far above it; ECE falls from
  0.0411 to 0.0321. Top-1 is slightly lower under E1 (point estimates).
- Team accuracy on misses won by a player (offense against defense by summed probability, point estimates): baseline
  0.7749, E1 0.7734, ablation 0.7771.
- Fitted parameters per fold (a offense, a defense, a none, b defense, b none; point estimates): the offense and
  defense slopes are stable (1.113 to 1.123 and 1.346 to 1.353), the defense offset 0.387 to 0.425; the none slope and
  offset vary more (2.833 to 3.823, 3.658 to 5.855). Every fit converged. The frozen tau is 1.2548 to 1.2566.
- Projected Set A half-width (D125, from the development per-game spread of the difference; point estimates): 0.00656
  (random 10-game sets) and 0.00646 (development interval scaled).

## Reading

Separate sharpening per candidate type is a real but small improvement: its interval lies well below zero, and nearly
all of it comes from the slopes, the defense candidates sharper than the offense ones and the team-rebound row much
sharper. The pre-registered bar asked for more than this (at least 0.005 nats), so it is not adopted. The favourite
stays under-confident under E1 (slope 1.300): the hypothesis's calibration part moves the right way but not far.

## Set A

Not scored (PLAN.md D125): development did not adopt, so Set A is not scored. No Set A path was written.
