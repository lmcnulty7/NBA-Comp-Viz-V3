# gravity-E1: does gravity open shots for teammates?

Written 2026-11-19 (UTC). Stage: **finished as a development result: not adopted; Set A not scored** (PLAN.md
D109). Pre-registration: docs/research/gravity.md, "gravity-E1", and its amendment (D107), written
and committed (22928efc) before any code ran. Code: tools/research_gravity.py, committed (8d1b8ee4) before the frame
pass. Sources of the numbers below: the development numbers from reports/research/gravity-E1-dev.json; the gate
numbers from reports/research/gravity-E1-gate.json (a copy of the gate's check record); 28.119 and +0.0042 from
reports/gravity.md; 935 attempts and the 0.23 to 0.27 ft half-width from the amendment (docs/research/gravity.md);
the sum-only slope, marked post hoc, from reports/research/gravity-E1-posthoc.json. Numbers marked computed are
arithmetic on those.

**Verdict: not adopted on development.** With the shot's location and the shooter accounted for, a shot is not
more open by the pre-registered 0.2 ft per SD of the teammates' gravity sum, with the teammates' gravity maximum
held fixed (commitment 7's estimand). That effect is 0.014 ft per SD (95 percent interval -0.066 to 0.102; 62358
attempts in 604 games), so 0.2 lies above the whole interval: for this estimand, an effect of that size is ruled
out. The gravity features do lower the out-of-fold MSE slightly: candidate minus baseline is -0.0076 ft squared
[-0.0152, -0.0002], an interval below zero, against a baseline MSE of 16.09. The maximum's partial slope is
positive, 0.124 [0.047, 0.198] ft per SD, but under 0.2. Post hoc and read by no rule: the sum alone, with the
maximum not held fixed, gives 0.109 [0.053, 0.167] ft per SD on the same development folds, so a total effect of the
sum does not reach 0.2 either; "ruled out" is said of the committed estimand only. The adoption rule needs both
parts, so gravity is not shown to leave teammates more open by the pre-registered amount. In the dossier's words for a
fail: gravity was tested against teammates' openness and showed no effect of that size. Set A was not scored
(PLAN.md D109): under D107 its reading could only stop an adoption, and there is none to stop; at its size it could
not separate 0.0145 from zero either. The scoring ledger has no gravity-E1 entry.

## What was tested

The gravity lane measures how close defenders stay to a player beyond what the player's location explains. Its only
link to outcomes was small (+0.0042 in teammates' location-only xFG, top ten players, training games). This
experiment tests the mechanism directly: are a teammate's shots more open?

- Outcome: the nearest-defender distance at the release (the frozen xFG lane's contest feature), on every
  development attempt that has it.
- Baseline: a ridge on the four location features of the frozen location-only xFG model plus a shooter term.
  Candidate: the baseline plus the sum and the maximum of the 4 on-floor teammates' gravity. A teammate's gravity is
  their mean over development games outside the attempt's fold, so no game supplies both. Stated, not repaired
  (commitment 4): the frozen out-of-fold expectations of those frames come from ridge fits that saw the attempt's
  fold; and fold f's frames enter the teammate-gravity means used by training rows of the other folds' fits
  (development only, diluted).
- Decision (D107, development games): the out-of-fold MSE difference (candidate minus baseline) with its interval
  wholly below zero, and an effect of at least 0.2 ft per SD of the gravity sum.
- Set A (D107): a pre-declared consistency check, scored once, that can stop an adoption only by reading
  "inconsistent".

The amendment's nine commitments (the gravity rebuild and gate, attempts, teammates, folds, models, the cross-fitted
effect, what is reported beside, the Set A refusals) are in the dossier, dated before any run.

## Gate (the baseline)

The frozen gravity lane's own functions (validate_game, build_game) ran on all 604 development games, one game at a
time, under the lock. The frozen ridge expectation was refit out of fold under the lane's folds. Then:

- gravity_frames: all 7983660 training player-frames reproduced, keys and folds identical, expected_ft identical
  (largest difference 0.0 ft).
- gravity_players: all 447 player means identical (0.0 ft).
- The ridge out-of-fold MSE is 28.119, as reports/gravity.md prints.

The HGB diagnostic candidate of the lane was not refit; nothing here reads it. The same pass read each attempt's
release frame. All 62387 attempts with a release distance had the shooter and exactly 4 teammates on the floor.
29 were dropped because a teammate had no gravity outside the attempt's fold. No held-out row was used; the stored
frozen tables were read whole and their held-out rows dropped on load. Shot rows were read with a filter on the
development games, and no held-out game's frames were read.

## Development games (604 games, 62358 attempts)

Intervals: a paired bootstrap by game of the fixed per-shot values (MSE, log loss; seed sequence [0, 1]) or of the
fixed residuals (the gravity effects, [0, 2]; the ablation's effects, [0, 3]; the post hoc sum-only slope, [0, 4]),
2000 resamples each. A game resample of fixed values adds no noise to any value (unlike the reliability statistics
of decisions-E1, D86). Every MSE change is written one way: candidate (or ablation) minus baseline.

| quantity | value [95 percent interval] |
| --- | --- |
| MSE, baseline (ft squared) | 16.0942 [15.1712, 17.0628] |
| MSE, candidate (gravity) | 16.0866 [15.1611, 17.0561] |
| **MSE, candidate minus baseline** | **-0.0076 [-0.0152, -0.0002]** |
| **effect per SD of teammates' gravity sum (ft)** | **0.0145 [-0.0660, 0.1016]** |
| effect per SD of teammates' gravity maximum (ft; beside, read by no rule) | 0.1237 [0.0473, 0.1979] |
| post hoc: sum-only effect per SD, maximum not held fixed (ft; read by no rule) | 0.1091 [0.0533, 0.1669] |
| MSE, ablation (raw mean defender distance instead of gravity) | 16.0910 [15.1677, 17.0604] |
| ablation, MSE minus baseline | -0.0032 [-0.0089, 0.0023] |
| ablation, effect per SD of the raw-distance sum (ft) | -0.1107 [-0.1728, -0.0462] |
| ablation, effect per SD of the raw-distance maximum (ft) | 0.0616 [-0.0002, 0.1222] |
| log loss of make, location-only xFG offset alone | 0.6655 [0.6636, 0.6675] |
| log loss of make, offset plus gravity | 0.6655 [0.6635, 0.6675] |
| log loss change (offset plus gravity minus offset alone) | -0.00003 [-0.00013, +0.00005] |

One SD of teammates' gravity sum is 1.139 ft. The effects are partial slopes, cross-fitted: the release distance and
both gravity features are first adjusted for location and shooter by ridges fit on the other folds (commitment 7).

Decision (D107): the MSE interval is below zero (yes), and the effect is at least 0.2 ft per SD (no: 0.0145).
**Not adopted.**

## What this means

- The test is decisive at the pre-registered size, for the committed estimand (the sum with the maximum held
  fixed). Its whole interval (-0.066 to 0.102) lies below 0.2, so an effect of that size is ruled out on 62358
  development attempts, not just unseen. The post hoc sum-only slope (0.109 [0.053, 0.167]) also stays under 0.2,
  but it was not pre-registered and reads no rule.
- The development half-width of that effect is about 0.084 ft (computed from the interval), wider than the about
  0.03 ft that D107 projected. The projection treated attempts as independent and the effect as a slope on one
  feature; game clustering and the second, correlated gravity feature likely widen it (not measured). The 0.2
  threshold is still outside.
- The MSE change is below zero but very small: candidate minus baseline -0.0076 ft squared against a baseline of
  16.09, about 0.05 percent (computed: 0.0076 / 16.0942 = 0.047 percent).
- Read by no rule: the maximum teammate's gravity carries a small positive partial effect (0.124 ft per SD,
  interval 0.047 to 0.198). This is a hint that one strong pull matters more than the total, but it is also under
  0.2.
- Location-only xFG with an offset gains nothing from gravity in log loss of make.
- The ablation's raw closeness behaves differently (its sum's partial slope is negative), so the location
  expectation does change what the measure carries. Neither version opens shots by the pre-registered amount.
- Lineups and team schemes may confound gravity with teammates' skill (the dossier's risk); with no effect of size
  found, that confounding has no positive result to explain away.

## Before any score: commitments (Set A, D107)

The test was not scored (PLAN.md D109). Development does not adopt, so under D107 the Set A reading could not change
the decision: only "inconsistent" can stop an adoption, and there is none to stop. The lead decided not to spend the
experiment's single ledger entry on it, as for winprob-E2. Set A was not read for gravity-E1, and e1-final was never
run. The commitments below stay as they were written before that decision, for the record. Had it run:

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
