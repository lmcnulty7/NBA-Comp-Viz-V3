# rebounds-E2: train on the whole miss (within-miss softmax objective)

Written 2026-11-20; the Set A section added the same day after the single score (05:24 UTC, the ledger's time). Pre-registration:
docs/research/rebounds.md, "rebounds-E2", with its amendment (PLAN.md D127) and the commitments before the Set A score,
all in the dossier. Code: tools/research_rebounds_e2.py; the development stage ran on commit d12b130c. The code did
not check that the tree was clean for that run. The evidence that it ran on committed code: d12b130c was committed at
20:36:02, the run was started by the same command right after the commit, every fold piece records d12b130c, and the
five fold files were written at 20:36:18, 20:36:33, 20:36:47, 20:37:01 and 20:37:15 (2026-11-19, local time). Every number
below is copied from reports/research/rebounds-E2-dev.json or reports/research/rebounds-E2-final.json, whose
summary is the one rebounds-E2 entry of reports/research/scoring_ledger.json; the score ran on commit 72c261b9 with a
clean tree. Intervals are 95 percent percentile intervals of a
2000-resample bootstrap of the 604 development games (seed 0); paired differences use the same resampled games.
Set A intervals: the same bootstrap over its 10 games. Calibration slopes are point estimates.

**Verdict: adopted as a research result on development; Set A, scored once, reads "confirms" under D127.** Trained
on the whole miss, the softmax model lowers per-miss log loss against the frozen rebound model by 0.019327 on the
development games (-0.019327 [-0.021138, -0.017642], sign flipped, computed) and by 0.015837 on the 10 Set A games
(-0.015837 [-0.028443, -0.003126]); the bootstrap interval lies wholly below zero, so the confirmation reads
"confirms". Beside it, plainly: the report-only t interval on the 10 per-game means, [-0.031839, +0.000153], just
reaches above zero and would read "inconclusive at 10 games". The rule reads the bootstrap, as committed; with 10
games, read the confirmation as modest. The site's rebound model is unchanged.

## What is tested

The frozen rebound model is a boosted binary classifier on each candidate row (who could get the ball), normalised
within the miss and sharpened by one temperature, tau. Its training never sees that the candidates of a miss
compete. rebounds-E2 trains a boosted model on the whole miss instead: the softmax over a miss's candidates, with the
cross entropy of the true rebounder as the loss (Cao et al. 2007, one winner per list).

- Softmax arm: XGBoost 3.2 with a custom objective (gradient p - y, hessian p (1 - p) within each miss), depth 3,
  learning rate 0.1, 80 trees, as the frozen booster; fixed, not tuned.
- Baseline: the frozen hgb with tau. rebounds-E1 was not adopted on development, so the baseline is the frozen model,
  as pre-registered. Its out-of-fold probabilities are rebounds-E1's refit, reused: gate A matched all 537,175 stored
  candidate rows, with per-miss log loss 1.600912 against the frozen 1.6009.
- Ablation: the same XGBoost settings with the binary objective and the frozen tau procedure, so that a gain cannot be
  a library difference.
- Decision (D127, development games): softmax minus baseline per-miss log loss at most -0.010, with the 95 percent
  interval excluding zero. Same five outer folds as the frozen lane and rebounds-E1.

## Development result (604 games, 49,288 misses, 537,175 candidate rows)

| arm | per-miss log loss | top-1 | favourite's calibration slope |
| --- | --- | --- | --- |
| baseline (frozen hgb with tau) | 1.6009 [1.5887, 1.6127] | 0.4797 [0.4747, 0.4848] | 1.368 |
| softmax (whole miss) | 1.5816 [1.5702, 1.5929] | 0.4775 [0.4724, 0.4826] | 1.130 |
| binary ablation (XGBoost with tau) | 1.6056 [1.5936, 1.6174] | 0.4765 [0.4714, 0.4814] | 1.352 |

| paired difference (per-miss log loss) | value |
| --- | --- |
| softmax minus baseline | -0.019327 [-0.021138, -0.017642] |
| binary minus baseline | +0.004680 [+0.003939, +0.005423] |
| softmax minus binary | -0.024007 [-0.025810, -0.022213] |

- By the rule, the softmax arm is adopted on development: -0.019327 is below -0.010 and its interval lies wholly
  below zero.
- The ablation shows the gain is the objective, not the library: XGBoost with the binary loss and the frozen tau is
  slightly worse than the frozen model (+0.004680), and the softmax loss beats it by 0.024007.
- Top-1 does not improve (0.4775 against 0.4797, intervals overlapping); the gain is in the probabilities, not in
  which candidate ranks first.
- The favourite's calibration slope moves from 1.368 to 1.130, toward 1. Both models are still too cautious about
  their favourite, the softmax arm much less so.
- The binary arm's tau was 1.2732, 1.2742, 1.2751, 1.2759 and 1.2759 by fold.

## Before the Set A score

The projected Set A half-width, from the development per-game spread of softmax minus baseline, is 0.013754 (random
10-game sets) and 0.013572 (the development bootstrap scaled). Set A is scored once as a D108 confirmation, with the
readings and refusals written in the dossier before any Set A read.

## Review fixes before the score

The pre-score review of 807bdc2c passed with fixes, made in one commit (72c261b9) before the rehearsal and the score,
none changing the method: the final baseline is fitted on the development rows in their build order, as rebounds-E1's
gated fold fits were (only the two new arms see the rows sorted by miss); the scorer records the frozen build's drop
counts per game; a block or sum guard failure inside the scorer returns a refusal on record instead of crashing; tests
that the rehearsal check refuses another commit or a failed record and that the final command refuses a dirty tree;
and D108's sentence on a straddling interval in the commitments. The rehearsal at 72c261b9 rebuilt the 8,736
candidate rows of 10 development stand-in games exactly (every one of 19 shared columns equal, no tolerance).

## Set A (D127 confirmation, scored once)

The three arms were fitted on all 604 development games; Set A's rebound frames and candidate rows were built inside
score_once with the frozen builder. 10 games, 846 misses (about 820 were projected; docs/research/rebounds.md, PLAN.md D125), 9,218 candidate rows.

| arm | per-miss log loss | top-1 | favourite's calibration slope |
| --- | --- | --- | --- |
| baseline (frozen hgb with tau) | 1.6548 [1.5932, 1.7068] | 0.4421 [0.4097, 0.4743] | 1.438 |
| softmax (whole miss) | 1.6389 [1.5849, 1.6924] | 0.4551 [0.4219, 0.4904] | 1.110 |
| binary ablation | 1.6551 [1.5954, 1.7088] | 0.4409 [0.4044, 0.4781] | 1.422 |

| paired difference (per-miss log loss) | bootstrap of the 10 games |
| --- | --- |
| softmax minus baseline | -0.015837 [-0.028443, -0.003126] |
| binary minus baseline | +0.000311 [-0.006283, +0.006730] |
| softmax minus binary | -0.016148 [-0.028543, -0.002986] |

- Reading (D108, as committed): "confirms". The bootstrap interval of softmax minus baseline lies wholly below zero.
- Report only, read by no rule: the 95 percent t interval on the 10 per-game mean differences is -0.015843
  [-0.031839, +0.000153]. It just reaches above zero and would read "inconclusive at 10 games", so the two intervals
  disagree. The rule reads the bootstrap, as committed before the score. With 10 games the confirmation is modest.
- Report only, read by no rule and not leaned on: under the original threshold of 2026-11-16 (at most -0.010 with
  the interval excluding zero), the Set A point -0.015837 and its bootstrap interval would also have met it.
- The Set A effect (-0.015837) is smaller than development's (-0.019327); both points clear -0.010. The binary
  ablation is near zero on Set A (+0.000311), as the gain is the objective's, not the library's. The calibration
  slopes are point estimates; top-1 has intervals above.
- The frozen build's counts over the 10 games: 996 misses in, 846 kept, 9,218 candidate rows; drops: no rim frame
  137, rebounder not on court 8, and the D94/D95 drop of a named rebounder that cannot be matched 5; every other drop
  0. Past the first filter, before the 8 not-on-court drops: 89 team rebounds and 765 player rebounds.
- The projected half-width before the score was 0.013754 (random 10-game sets); the bootstrap half-width here is
  0.012658 (computed).

## What this means

- Training the rebound model on the whole miss, so that it learns that candidates compete, beats the frozen binary
  model plus a temperature on the training games by a margin well past the pre-registered bar, and the untouched games
  agree in direction and size, if modestly at 10 games.
- The gain is in the probabilities: top-1 barely moves, and the favourite's calibration slope comes closer to 1 (1.110
  against 1.438 on Set A; 1.130 against 1.368 on the training games).
- It is a research result. The site's rebound model is unchanged; using the softmax model on the site would need a D
  line.
