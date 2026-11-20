# rebounds-E2: train on the whole miss (within-miss softmax objective)

Draft written 2026-11-20 (development stage; the Set A confirmation is not scored yet). Pre-registration:
docs/research/rebounds.md, "rebounds-E2", with its amendment (PLAN.md D127) and the commitments before the Set A score,
all in the dossier. Code: tools/research_rebounds_e2.py; the development stage ran on commit d12b130c. The code did
not check that the tree was clean for that run. The evidence that it ran on committed code: d12b130c was committed at
20:36:02, the run was started by the same command right after the commit, every fold piece records d12b130c, and the
five fold files were written at 20:36:18, 20:36:33, 20:36:47, 20:37:01 and 20:37:15 (2026-11-19, local time). Every number
below is copied from reports/research/rebounds-E2-dev.json. Intervals are 95 percent percentile intervals of a
2000-resample bootstrap of the 604 development games (seed 0); paired differences use the same resampled games.
Calibration slopes are point estimates.

**Status: development adopts; the Set A confirmation (D127) is not scored yet.**

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

## Set A (D127 confirmation)

Not scored yet.
