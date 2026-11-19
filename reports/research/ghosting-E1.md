# ghosting-E1: rollout training of the same network

Written 2026-11-19 (UTC); the Set A section added the same day after the single score. Stage: **finished: adopted
on development, then stopped by Set A's D113 reading ("inconsistent"), so not adopted.** Pre-registration:
docs/research/ghosting.md, "ghosting-E1", and its amendment (D113), written and committed (f8c5349d) before any code
ran. Code: tools/research_ghosting.py, committed (a6c2cba7) before the frame pass, with one gate fix (fe259133) before
any result was read; the score ran on eae3cb6f with a clean tree. Sources: the development numbers from
reports/research/ghosting-E1-dev.json, run on fe259133; the gate numbers from reports/research/ghosting-E1-gate.json;
0.437, 3.657 and 346 / 258 from reports/ghosting.md; the per-fold numbers from reports/research/ghosting-E1-folds.json
(a pre-score description, PLAN.md D115, read by no rule); the Set A numbers from reports/research/ghosting-E1-final.json,
whose summary is the one ghosting-E1 entry of reports/research/scoring_ledger.json. Numbers marked computed are
arithmetic on those.

**Verdict: not adopted.** On development the rollout-trained network met all three pre-registered parts: at 1.0 s
it puts the ghost 1.295 ft closer than the frozen one-step stand-in (2.362 against 3.657 ft; paired -1.295 [-1.322,
-1.269], 604 games), it is better at 0.2 s (-0.054 [-0.055, -0.053]), and its deviation tracks points allowed in 414
games against 190 (p = 2e-20). On Set A, scored once, the rollout ghost is again much better at 1.0 s: -1.117 [-1.154,
-1.085] ft (10 games). That interval excludes the development estimate (-1.295), so D113's reading is "inconsistent",
and under D113 that stops the adoption. Beside it, report only and fixed before the score (D115): the interval's upper
end, -1.085, is below -0.10 ft, so on the test games the gain is at least the pre-registered bar. The report said
before the score why the reading could be "inconsistent": the development estimate pools five stand-in refits that
differ from one another (fold 3's at 4.266 ft), while Set A scores one. On Set A the rollout ghost's error (2.354 ft)
sits inside its five development folds (2.33 to 2.44). The stand-in refit's error (3.471 ft) sits inside the other
four folds' stand-ins (3.44 to 3.55). The gap to -1.295 is consistent with the pooled development estimate including
fold 3's outlying stand-in. The rule stands as written.

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
  Trained on the rollout with the same shape, the same epochs and optimiser steps, it is 1.41 ft better than that
  one-step version.
- At 1.0 s the frozen stand-in was no better than the constant-velocity floor (+0.040 [+0.012, +0.070] ft worse),
  the floor the dossier said was never reported. The rollout ghost beats that floor by 1.255 ft (computed:
  3.6175 - 2.3623).
- The rollout ghost's deviation is more strongly tied to points allowed (414 / 190 against 346 / 258). This is the
  lane's own validation, read on development games.
- Not measured, the dossier's risk: rollout training can pull a ghost's motion toward the average, which lowers the
  error but can make paths less lifelike. A lower mean error does not show the paths are more lifelike. No
  path-shape check was pre-registered, so none is claimed.
- Leakage checks done by reading the code, not by a statistic: every model is scored by the same frozen roll_forward
  on the same origins. Training reads only the training folds' fit rows, and the stored folds match. The actual
  future positions enter only the loss. At every rollout step the inputs use everyone else's positions one step
  before the target, as at scoring. By the lane's definition the ghost reads the actual other nine players up to
  0.2 s before each step; that is available to every model alike, the stand-in included.
- Per fold (pre-score description): rollout minus stand-in at 1.0 s is -1.191, -1.161, -1.192, -1.932 and -1.002 ft,
  and the rollout's 1.0 s error is 2.33 to 2.44 ft in every fold. The spread comes from the stand-in: fold 3's
  stand-in has 4.266 ft against 3.44 to 3.55 in the others. The gain does not depend on fold 3: against constant
  velocity the rollout gains 1.19 to 1.32 ft and against the one-step torch net 1.28 to 1.56 ft in every fold.

## Compute

Five folds, each its own lock job on the system Python (torch 2.15 on MPS, watermarks 0.6 / 0.5): 877 s, 1473 s, 1292
s, 1445 s and 1273 s, peak RSS 5.0 to 6.9 GB. Fold 0's figure is its rescore, which reused the saved network weights;
its first run took 1561 s, and its training epochs are recorded in that first run's fold record
(fold0/superseded_gate_bug/fold.json in the artifacts folder). The frame pass over the 604 games took 326 s (its log,
research_ghosting_pass.log, in the artifacts folder). Every fold's network weights are saved with their hashes in the
development record. Training on MPS is not bit-reproducible, so the numbers trace to those
weights.

The section below was written before the score and is kept as written.

## Before the score: commitments (Set A, D113)

Development adopts, so under D113 Set A is scored once, as a consistency check, on the lead's go:

1. e1-final runs on a clean tree, only if the gate passed and development adopted. It first refits the stand-in and
   the rollout network on every development fit row with the same settings (no Set A data is read for that). It then
   scores the 10 test games once through score_once, recording the commit, the gate and development records'
   hashes and the rollout weights' hash, with the summary nested under "scored".
2. It reports the 1.0 s and 0.2 s paired differences (rollout minus stand-in) with the lane's bootstrap of the 10
   games (2000 resamples), and a report-only t interval on the per-game differences (as D80; report_only_t).
3. The reading uses the 1.0 s bootstrap interval, in D113's order: "inconsistent" if it excludes the development
   estimate (-1.295); otherwise "confirms" if it excludes zero on that side; otherwise "consistent". Only
   "inconsistent" stops the adoption.
4. Refusals: the scorer gives a reason and no number unless every test game has frames, possessions and at least one
   1.0 s origin, and unless both ghosts cover the same origins in every game and horizon; a summary with any
   non-finite number is a refused verdict, never a NaN. Before score_once, e1-final also refuses (and never retrains
   to fit) when a refit weight is not finite or the rollout's final-epoch loss lies outside the folds' range (7.68 to
   7.81, ghosting-E1-folds.json).
5. What the reading can be expected to say (D115, written before the score). The amendment's projection of a 0.20 ft
   half-width came from a different pair of models. Within each development fold the per-game SD of the rollout's
   difference is 0.054 to 0.088 ft, so at 10 games the 1.0 s half-width should be about 0.034 to 0.055 ft (computed:
   1.96 x SD / sqrt(10), ghosting-E1-folds.json). The development estimate -1.295 pools five fold pairs whose
   differences run from -1.00 to -1.93, because the stand-in varies from refit to refit (fold 3's at 4.266 ft). Set A
   scores one new refit pair. Its interval may therefore exclude -1.295 because of where that one stand-in refit
   lands, not because rollout training failed. An "inconsistent" reading of that kind is explained here in advance.
   D113's reading still decides.
6. Beside it, report only and fixed before the score (D115): whether the Set A 1.0 s interval's upper end is below
   -0.10 ft, that is, whether the rollout gain is at least the pre-registered bar on the test games.
7. The scorer was rehearsed through score_once on a synthetic root with a temporary test set and ledger, including a
   refusal (tests/test_research_ghosting.py).

## The refit before the score (PLAN.md D119)

The first e1-final run refit both models on every development fit row. Its pre-score check then refused before
score_once was called, so nothing was spent. The rollout network's final-epoch training loss was 7.6810, 0.0033 below
the five folds' range of 7.6843 to 7.8066 (ghosting-E1-folds.json; computed: 7.6843 - 7.6810). The check was there to catch a broken refit:
non-finite weights or a diverged loss. A slightly lower training loss on about 25 percent more rows (computed: 7147100 fit rows against
5704715 to 5730095 in the folds' training sets, 1.247 to 1.253 times as many) is the expected direction, not a failure, and a training loss says nothing about Set A. D119, decided knowing the value: the check
became one-sided (finite weights and a final-epoch loss not above 7.8066), and the saved rollout refit was scored as
it is: D119's "no retraining, no second refit" applies to the rollout network. Its epoch losses were transcribed from
the log of the run that trained it (rollout_epochs.json, with the log's hash). The stand-in was refit once more, in
the scored run, because the refused run kept none of it; it was saved this time. Its refit is deterministic (seed 0,
one thread), which the gate supports (the out-of-fold refits matched the stored ghosts within 7.35e-6 ft on 68935605
rows), but this refit was not hash-verified against the refused run's. The hashes of both models' weights, the epoch record and the 7.6810 loss are in the
final record.

## Set A (the 10 games of 2015-12-05, scored once)

e1-final ran once on eae3cb6f with a clean tree, under the lock (1401 s, peak 5.6 GB, from the lock's line in the run
log, research_ghosting_final2.log in the artifacts folder). No refusal fired. Both ghosts
covered the same origins in every game: 522180 at 1.0 s and 556040 at 0.2 s.

| quantity | 1.0 s | 0.2 s |
| --- | --- | --- |
| stand-in, mean error (ft) | 3.4711 | 0.4213 |
| rollout network, mean error (ft) | 2.3542 | 0.3875 |
| **rollout minus stand-in (bootstrap of the 10 games)** | **-1.1169 [-1.1538, -1.0851]** | -0.0338 [-0.0392, -0.0303] |
| report-only t interval on the per-game differences | -1.1207 [-1.1639, -1.0775] | -0.0344 [-0.0407, -0.0281] |

- D113's reading, from the 1.0 s bootstrap interval: it excludes the development estimate (-1.2949), so
  "inconsistent". It also excludes zero, but the fixed order reads "inconsistent" first. The adoption stops.
- Report-only size reading (D115): the upper end, -1.0851, is below -0.10 ft. On the test games the gain is at least
  the pre-registered bar.
- The 0.2 s difference is negative on Set A too.
- The interval's half-width came out 0.034 ft (computed: 0.0343), inside the 0.034 to 0.055 ft predicted before the
  score.

## What Set A means

- The rollout ghost behaves on unseen games as it did on development folds: 2.354 ft at 1.0 s against 2.33 to 2.44.
  What moved is the baseline. The gap is consistent with the pooled development estimate including fold 3's outlying
  stand-in, and one refit pair is not expected to match a pooled estimate to within a 0.034 ft half-width. That is
  the case described before the score.
- The pre-registered rule reads Set A against the development estimate, not against the bar, so the method is not
  adopted. The size reading, which reads no rule, says the gain on the test games is about 11 times the 0.10 ft bar
  (computed: 1.117 / 0.10).
- A lower mean error does not show the paths are more lifelike; that was not measured on either set.
