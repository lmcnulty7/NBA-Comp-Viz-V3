# gravity-E1: does gravity open shots for teammates?

Written 2026-11-19 (UTC). Stage: **development done, not adopted; Set A not scored** (it waits for the lead's
decision, PLAN.md D107). Pre-registration: docs/research/gravity.md, "gravity-E1", and its amendment (D107), written
and committed (22928efc) before any code ran. Code: tools/research_gravity.py, committed (8d1b8ee4) before the frame
pass. Every number below is copied from reports/research/gravity-E1-dev.json.

**Verdict: not adopted on development.** With the shot's location and the shooter accounted for, a shot is not
measurably more open when the shooter's on-floor teammates have more gravity. Per standard deviation of the
teammates' gravity sum, the nearest defender at the release is 0.014 ft farther (95 percent interval -0.066 to
0.102; 62358 attempts in 604 games). The pre-registered effect of 0.2 ft per SD lies above the whole interval. The
out-of-fold MSE drops by 0.0076 ft squared [-0.0152, -0.0002], below zero but tiny next to a baseline MSE of 16.09.
The adoption rule needs both, so gravity is not shown to leave teammates more open. In the dossier's words for a
fail: gravity was tested against teammates' openness and showed no effect of that size.

## What was tested

The gravity lane measures how close defenders stay to a player beyond what the player's location explains. Its only
link to outcomes was small (+0.0042 in teammates' location-only xFG, top ten players, training games). This
experiment tests the mechanism directly: are a teammate's shots more open?

- Outcome: the nearest-defender distance at the release (the frozen xFG lane's contest feature), on every
  development attempt that has it.
- Baseline: a ridge on the four location features of the frozen location-only xFG model plus a shooter term.
  Candidate: the baseline plus the sum and the maximum of the 4 on-floor teammates' gravity. A teammate's gravity is
  their mean over development games outside the attempt's fold, so no game supplies both.
- Decision (D107, development games): the out-of-fold MSE difference (candidate minus baseline) with its interval
  wholly below zero, and an effect of at least 0.2 ft per SD of the gravity sum.
- Set A (D107): a pre-declared consistency check, scored once, that can stop an adoption only by reading
  "inconsistent".

The amendment's nine commitments (the gravity rebuild and gate, attempts, teammates, folds, models, the cross-fitted
effect, what is reported beside, the Set A refusals) are in the dossier, dated before any run.

## Gate (the baseline)

The frozen gravity lane's own functions (validate_game, build_game) ran on all 604 development games, one game at a
time (94 s under the lock). The frozen ridge expectation was refit out of fold under the lane's folds. Then:

- gravity_frames: all 7983660 training player-frames reproduced, keys and folds identical, expected_ft identical
  (largest difference 0.0 ft).
- gravity_players: all 447 player means identical (0.0 ft).
- The ridge out-of-fold MSE is 28.119, as reports/gravity.md prints.

The HGB diagnostic candidate of the lane was not refit; nothing here reads it. The same pass read each attempt's
release frame. All 62387 attempts with a release distance had the shooter and exactly 4 teammates on the floor.
29 were dropped because a teammate had no gravity outside the attempt's fold. No held-out game was read, and shot
rows were read with a filter on the development games.

## Development games (604 games, 62358 attempts)

Intervals: a paired bootstrap by game of the fixed per-shot values (MSE, log loss) or of the fixed residuals (the
effect), 2000 resamples. A game resample of fixed values adds no noise to any value (unlike the reliability
statistics of decisions-E1, D86).

| quantity | value [95 percent interval] |
| --- | --- |
| MSE, baseline (ft squared) | 16.0942 [15.1712, 17.0628] |
| MSE, candidate (gravity) | 16.0866 [15.1611, 17.0561] |
| **MSE, candidate minus baseline** | **-0.0076 [-0.0152, -0.0002]** |
| **effect per SD of teammates' gravity sum (ft)** | **0.0145 [-0.0660, 0.1016]** |
| effect per SD of teammates' gravity maximum (ft, beside) | 0.1237 [0.0473, 0.1979] |
| MSE, ablation (raw mean defender distance instead of gravity) | 16.0910 [15.1677, 17.0604] |
| ablation, MSE minus baseline | -0.0032 [-0.0089, 0.0023] |
| ablation, effect per SD of the raw-distance sum (ft) | -0.1107 [-0.1728, -0.0462] |
| ablation, effect per SD of the raw-distance maximum (ft) | 0.0616 [-0.0002, 0.1222] |
| log loss of make, location-only xFG offset alone | 0.6655 [0.6636, 0.6675] |
| log loss of make, offset plus gravity | 0.6655 [0.6635, 0.6675] |
| log loss change | -0.0000 [-0.0001, 0.0001] |

One SD of teammates' gravity sum is 1.139 ft. The effects are partial slopes, cross-fitted: the release distance and
both gravity features are first adjusted for location and shooter by ridges fit on the other folds (commitment 7).

Decision (D107): the MSE interval is below zero (yes), and the effect is at least 0.2 ft per SD (no: 0.0145).
**Not adopted.**

## What this means

- The test is decisive at the pre-registered size. The effect's whole interval (-0.066 to 0.102) lies below 0.2,
  so an effect of that size is ruled out on 62358 development attempts, not just unseen.
- The MSE gain is real in sign but very small: 0.0076 ft squared on a baseline of 16.09, about 0.05 percent.
- The maximum teammate's gravity carries a small partial effect (0.124 ft per SD, interval 0.047 to 0.198). This
  is a hint that one strong pull matters more than the total, but it is also under 0.2 and reads no rule.
- Location-only xFG with an offset gains nothing from gravity in log loss of make.
- The ablation's raw closeness behaves differently (its sum's partial slope is negative), so the location
  expectation does change what the measure carries. Neither version opens shots by the pre-registered amount.
- Lineups and team schemes may confound gravity with teammates' skill (the dossier's risk); with no effect of size
  found, that confounding has no positive result to explain away.

## Before any score: commitments (Set A, D107)

Development does not adopt, so under D107 the Set A reading cannot change the decision: only "inconsistent" can
stop an adoption, and there is none to stop. Whether to spend the experiment's single ledger entry on the
consistency check anyway is the lead's decision. If it runs:

1. It is scored once through score_once, on a clean tree, recording the commit, the gate files' hashes and the
   development record's hash, with the summary nested under "scored" beside its setup.
2. Every model is fit on all development attempts; teammates' gravity is each player's mean over every development
   frame; the Set A attempts come from the frozen xFG features of the test games and the Set A frames at their
   release frames, under the same rules (shooter and exactly 4 teammates, all with gravity).
3. It reports the test effect per SD and the test MSE difference, each with a bootstrap of the 10 games (2000
   resamples), and a report-only t interval on the per-game values (as D80). The reading, in D107's fixed order,
   uses the bootstrap interval of the test effect: "inconsistent" if it excludes the development estimate (0.0145);
   otherwise "confirms" if it excludes zero on that side; otherwise "consistent".
4. Refusals: the scorer refuses, with its reason and no number, unless every test game has frames, the frozen xFG
   features build for every test game, and every test game has at least one used attempt. A summary with any
   non-finite number (for one, a t interval over fewer than 2 games) is a refused verdict, never a NaN.
5. Expected power, from the amendment: at about 935 attempts the test effect's half-width is about 0.23 to 0.27 ft,
   so it cannot separate 0.0145 from zero. "Confirms" is unlikely and "consistent" is the expected reading if
   nothing differs.
6. The scorer was rehearsed through score_once on a synthetic root with a temporary test set and ledger, including a
   refusal (tests/test_research_gravity.py).
