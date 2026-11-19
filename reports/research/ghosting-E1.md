# ghosting-E1: rollout training of the same network

Written 2026-11-19 (UTC). Stage: **development done, adopted; Set A not scored yet** (it waits for the lead's go,
PLAN.md D113). Pre-registration: docs/research/ghosting.md, "ghosting-E1", and its amendment (D113), written and
committed (f8c5349d) before any code ran. Code: tools/research_ghosting.py, committed (a6c2cba7) before the frame
pass, with one gate fix (fe259133) before any result was read. The development numbers below come from
reports/research/ghosting-E1-dev.json, run on fe259133; the gate numbers from reports/research/ghosting-E1-gate.json;
0.437, 3.657 and 346 / 258 from reports/ghosting.md. Numbers marked computed are arithmetic on those.

**Verdict: adopted on development, on all three pre-registered parts.** Trained on its own 5-step rollout, the same
64x64 window network puts the 1.0 s ghost 1.295 ft closer to where the defender went than the frozen one-step
stand-in: 2.362 ft against 3.657 ft out of fold (paired difference -1.295 [-1.322, -1.269] ft, 604 games). The 0.2 s
error is lower too (paired -0.054 [-0.055, -0.053]), so the guard holds. The rollout ghost's deviation tracks points
allowed more often than not in 414 games against 190 (one-sided p = 2e-20), against 346 / 258 for the stand-in. The
gain comes from the rollout objective, not from the framework: the same network in torch trained on one step is
slightly worse than the stand-in (+0.117 [+0.090, +0.143] ft at 1.0 s). Set A, a pre-declared consistency check
(D113), has not been scored.

## What was tested

The ghosting lane predicts where an average defender would have been, from the last 1 s of everyone's positions.
Its model is trained on the one-step displacement, but the 1.0 s ghost is a 5-step rollout of it with its own
position fed back. Error grows from 0.437 ft at 0.2 s to 3.657 ft at 1.0 s. Both ghosting papers train on rollouts
for this reason.

- Baseline: the frozen MLP stand-in (sklearn, 64x64), refit with the frozen code out of fold.
- Candidate: the same network shape in torch, trained on its own rollout with the horizon grown 1 to 5 steps (two
  epochs each, 10 epochs, Adam 0.001, batch 1024), its own features recomputed along the rollout and everyone else
  actual, exactly as the frozen roll_forward scores it (a test checks that the training rollout gives the frozen
  roll_forward's ghosts).
- Ablation: the same torch network and budget trained on the one-step loss.
- Decision (D113, development games): 1.0 s rollout minus stand-in at most -0.10 ft with the interval below zero;
  the 0.2 s upper bound under +0.02 ft; the deviation sign test positive with one-sided p < 0.05.

## Gate (the baseline)

The frozen stand-in was refit on each fold's training games with the frozen code (fit_mlp, seed 0, one thread) on
the frozen 1 Hz fit rows, and rolled forward by the frozen roll_forward on every 5 Hz origin of its test games.

- Its ghosts equal the stored ghost_frames on all 604 games: 68935605 rows against 68935605 stored, none unmatched,
  largest difference 7.4e-6 ft (cap 1e-3 ft, D49), the stored fold of every row equal.
- Its mean errors reproduce the printed ones: 0.437 at 0.2 s and 3.657 at 1.0 s; its sign test reproduces 346 / 258.

One fault was found and fixed before any result was read. On fold 0 the comparison left the 0.2 s rows unmatched,
because the stored horizon is float32 (0.2 in float32 is not 0.2). The merge now rounds the horizon on both sides,
with a test. Fold 0 was rescored with the same stand-in (refit, deterministic) and its saved network weights; its
first outputs were moved aside, not deleted.

No held-out game's frames, possessions or ghosts were read. The 17 spent games stay unscored.

## Development games (604 games; 35500180 origins at 0.2 s, 33435425 at 1.0 s)

Mean position error (ft), out of fold, pooled over origins; 95 percent intervals from a bootstrap of games (2000
resamples, seed 0):

| ghost | 0.2 s | 1.0 s |
| --- | --- | --- |
| persistence (reference) | 1.2644 [1.2585, 1.2703] | 5.4083 [5.3910, 5.4251] |
| constant velocity (reference) | 0.5771 [0.5696, 0.5840] | 3.6175 [3.5934, 3.6391] |
| frozen MLP stand-in (baseline) | 0.4374 [0.4327, 0.4421] | 3.6572 [3.6307, 3.6839] |
| torch, one-step loss (ablation) | 0.4523 [0.4476, 0.4565] | 3.7743 [3.7620, 3.7874] |
| **torch, rollout loss (candidate)** | **0.3837 [0.3788, 0.3880]** | **2.3623 [2.3528, 2.3717]** |

Paired differences (ft; the lane's paired bootstrap by game of pooled means, 2000 resamples, seed 0; a resample of
games with fixed per-row errors adds no noise to any value):

| pair | 0.2 s | 1.0 s |
| --- | --- | --- |
| **rollout minus stand-in** | **-0.0537 [-0.0549, -0.0525]** | **-1.2949 [-1.3219, -1.2691]** |
| one-step torch minus stand-in | +0.0149 [+0.0132, +0.0165] | +0.1171 [+0.0896, +0.1429] |
| rollout minus one-step torch | -0.0686 [-0.0697, -0.0677] | -1.4120 [-1.4231, -1.4018] |
| stand-in minus persistence | -0.8270 [-0.8308, -0.8232] | -1.7511 [-1.7803, -1.7220] |
| stand-in minus constant velocity | -0.1397 [-0.1427, -0.1365] | +0.0397 [+0.0120, +0.0699] |

Deviation sign test (per-game Spearman of a possession's mean 1.0 s deviation against the points scored, one-sided):
rollout ghost 414 positive, 190 negative, p = 2.1e-20; stand-in 346 / 258, p = 0.0002.

Decision (D113): 1.0 s, -1.295 at most -0.10 with the interval below zero (yes); 0.2 s upper bound -0.053 under
+0.02 (yes); sign test 414 > 190 with p < 0.05 (yes). **Adopted on development.**

## What this means

- The measured weakness was error growth from 0.2 s to 1.0 s. Training on the rollout cuts the 1.0 s error by about
  35 percent (computed: 1.295 / 3.657), the largest change any ghosting variant has shown here.
- The objective is the cause. The torch network trained on one step is a little worse than the sklearn stand-in.
  Trained on the rollout with the same shape, optimiser and budget, it is 1.41 ft better than that one-step version.
- At 1.0 s the frozen stand-in was no better than the constant-velocity floor (+0.040 [+0.012, +0.070] ft worse),
  the floor the dossier said was never reported. The rollout ghost beats that floor by 1.255 ft (computed:
  3.6175 - 2.3623).
- The rollout ghost's deviation is more strongly tied to points allowed (414 / 190 against 346 / 258). This is the
  lane's own validation, read on development games.
- Not measured, the dossier's risk: rollout training can pull a ghost's motion toward the average, which lowers the
  error but can make paths less lifelike. No path-shape check was pre-registered, so none is claimed.
- Leakage checks done by reading the code, not by a statistic: every model is scored by the same frozen roll_forward
  on the same origins. Training reads only the training folds' fit rows, and the stored folds match. The actual
  future positions enter only the loss. At every rollout step the inputs use everyone else's positions one step
  before the target, as at scoring.

## Compute

Five folds, each its own lock job on the system Python (torch 2.15 on MPS, watermarks 0.6 / 0.5): 877 s, 1473 s, 1292
s, 1445 s and 1273 s, peak RSS 5.0 to 6.9 GB. Fold 0's figure is its rescore, which reused the saved network weights;
its first run took 1561 s (the superseded fold record, kept in the artifacts folder). The frame pass over the 604 games took 326 s. Every fold's network weights are saved with
their hashes in the development record. Training on MPS is not bit-reproducible, so the numbers trace to those
weights.

## Before the score: commitments (Set A, D113)

Development adopts, so under D113 Set A is scored once, as a consistency check, on the lead's go:

1. e1-final runs on a clean tree, only if the gate passed and development adopted. It first refits the stand-in and
   the rollout network on every development fit row with the same settings (no Set A data is read for that). It then
   scores the 10 test games once through score_once, recording the commit, the gate and development records'
   hashes and the rollout weights' hash, with the summary nested under "scored".
2. It reports the 1.0 s and 0.2 s paired differences (rollout minus stand-in) with the lane's bootstrap of the 10
   games (2000 resamples), and a report-only t interval on the per-game differences (as D80).
3. The reading uses the 1.0 s bootstrap interval, in D113's order: "inconsistent" if it excludes the development
   estimate (-1.295); otherwise "confirms" if it excludes zero on that side; otherwise "consistent". Only
   "inconsistent" stops the adoption.
4. Refusals: the scorer gives a reason and no number unless every test game has frames, possessions and at least one
   1.0 s origin; a summary with any non-finite number is a refused verdict, never a NaN.
5. Power, from the amendment: at 10 games the 1.0 s half-width is about 0.20 ft, so the check separates the
   development estimate from zero easily; whether it also falls inside the test interval is the open question.
6. The scorer was rehearsed through score_once on a synthetic root with a temporary test set and ledger, including a
   refusal (tests/test_research_ghosting.py).
