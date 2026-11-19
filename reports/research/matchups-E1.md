# matchups-E1: split the outcome into shot frequency and shot efficiency

Draft written 2026-11-18, before the Set A score. Pre-registration: docs/research/matchups.md, section 6,
"matchups-E1", with its D108 amendment (PLAN.md D106, D108). Scoring code: tools/research_matchups.py; the
development numbers come from commit 8bd3ccfe (recorded in the dev json as code_commit). Every number below is copied
from reports/research/matchups-E1-dev.json, except where marked: the frozen lane's numbers are from
reports/matchups.md, and the first development run's numbers (item 6) are from its saved json in the work folder.
Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with both models of a
part paired on the same resampled games.

**Verdict: development decided (D108); Set A not scored yet.** Part (a), who takes the shot: passes on the
development games. Part (b), make given xFG: fails on the development games.

## What is tested

The frozen matchups lane found no defender signal in possession points. Following Franks et al. (2015), this
experiment splits the outcome into two lower-noise parts and asks whether defender identity carries out-of-sample
signal for either.

- Part (a), who shoots. One row per (possession, attacker) for the possessions whose last field-goal attempt has a
  known shooter; target: this attacker took it. Columns: the attacker, the defenders' shares of the possession's
  sampled frames spent guarding him (the frozen assignment), and his court-zone mix over the possession (rim, paint,
  mid, three). Compared with the same model without the defender columns.
- Part (b), make given xFG. One row per attempt with an assigned defender at release; target: make. Logit of xFG as an
  offset, the assigned defender as a column. Compared with the offset and an intercept alone.
- Both: penalized logistic regression, the penalty chosen by inner grouped CV on each outer fold's training games;
  out-of-fold log loss under 5 grouped folds by game (seed 0).
- Threshold (pre-registered): for each part, defender model minus no-defender model below zero with its interval
  wholly below zero. D108: the development games decide each part; Set A (10 games of one date) is scored once
  afterwards as a confirmation.
- Ablation: the frozen possession design on possession points minus xFG-expected points, with and without defender
  columns. Reported, not a selection input: split-half stability of the defender coefficients.

## Gate

- The frozen assignment, rerun on all 604 development games, rebuilt the stored matchup table exactly: 1,488,475 rows
  in both, no value different (largest difference 0.0 in every column).
- The frozen ridge refit on it chose alpha 300 in every fold and overall, reproduced all 447 stored defender ridge
  values (largest difference 0.0) and the out-of-fold MSEs printed in reports/matchups.md (constant 1.2842, raw
  1.3244, ridge 1.2854).
- The lane's original target, as the reference: ridge minus constant on possession points +0.00119 [+0.00017,
  +0.00217], worse than the constant, as frozen.

## Development games (604 games, out of fold)

| Part | Rows | Log loss, defender model | Log loss, no defender | Defender minus no defender | Reading |
| --- | --- | --- | --- | --- | --- |
| (a) who shoots | 430,555 (84,381 possessions) | 0.45885 [0.45768, 0.45997] | 0.46342 [0.46231, 0.46450] | -0.004574 [-0.004904, -0.004230] | passes |
| (b) make given xFG | 95,538 attempts | 0.63322 [0.63109, 0.63540] | 0.63323 [0.63113, 0.63541] | -0.000003 [-0.000140, +0.000121] | fails |

- Part (a) penalties: the defender model chose 0.004 to 0.037 per fold and the no-defender model 0.001 to 0.111; both
  are light (the grid grew downward past 1, see item 6). Part (b): the defender model chose 100 or 300; the
  no-defender model has no penalized column.
- Context, not pre-registered, reads no verdict: part (a) with the defender shares taken only from frames at least 2 s
  before the shot (81,878 possessions): -0.002162 [-0.002423, -0.001879]. About half of part (a)'s gain survives
  when the moment of the shot is left out; the other half comes from who reaches the shooter at the shot.
- Ablation (possession points minus xFG-expected points): defender minus no-defender MSE +0.00110 [+0.00033,
  +0.00187]. Defender columns make this target worse, as with raw possession points: the gain of part (a) comes from
  the decomposition, not from conditioning on shot quality.
- Stability, reported only: part (a) defender coefficients on odd against even games, Spearman 0.874 over 297
  defenders with at least 300 possessions in each half; part (b) 0.165 over 194 defenders with at least 100 attempts
  in each half.
- Projected Set A half-widths (D108, from the development per-game variance): part (a) about 0.0026 to 0.0027
  against a development effect of -0.0046; part (b) about 0.0010 against an effect of about zero.

## Commitments before the Set A score

1. **Development decides (D108).** Part (a) passes; part (b) fails. Set A cannot change either decision; it is read
   only as a confirmation.
2. **Set A readings, per part.** Interval wholly below zero: "confirms". Interval straddling zero: "inconclusive at
   10 games", not evidence against. Interval wholly above zero: "contradiction", reported beside the development
   result, and it blocks any site defender table for that part. Part (b)'s Set A reading cannot overturn its
   development fail, whatever it is. A t interval on the 10 per-game means is reported beside each bootstrap interval
   (report only, as D80); if the two disagree, the report says so. No per-defender claim is made from Set A (about 8
   attempts per defender there).
3. **What part (a)'s pass supports.** Defender identity, as the frozen assignment places it over the possession,
   adds out-of-sample information on which attacker takes the possession's last attempt, beyond the attacker
   himself and his court-zone mix. About half of it depends on the moment of the shot (the pre-shot context keeps
   -0.0022 of -0.0046): a defender who rotates to the shooter at the release gains share on him. So a part (a)
   defender table, if one is ever shown, must say it measures who a defender ends up guarding when shots are taken,
   partly at the shot itself; it is not proof that a defender keeps his man from shooting. It says nothing about
   making shots (part (b) fails) or about points (the ablation and the frozen target fail). The very high stability
   (0.874) fits a structural trait (role, position, scheme) better than a skill that varies game to game; the report
   says so.
4. **What part (b)'s fail means.** With xFG as an offset, the assigned defender at release adds no detectable
   information on whether the shot goes in (interval -0.00014 to +0.00012 nats per shot). xFG already holds contest
   features, so this does not show that defenders have no effect on makes; it shows none beyond what xFG measures.
5. **Projected power (D108).** Set A's half-width for part (a) (about 0.0026) is a little over half the development
   effect, so Set A will likely confirm if the effect holds; for part (b) "inconclusive at 10 games" is the expected
   reading.
6. **Changes after a development number, disclosed.**
   - The first development run (commit 219411696; numbers in its saved json in the work folder) put part (a)'s penalty
     at the bottom of the grid (1.0) in every fold. The grid now grows past whichever end wins (it had grown only
     upward); that run's part (a) difference was -0.0040, this run's -0.0046. Part (b) was unchanged.
   - The pre-shot context was added after that run, labelled not pre-registered, and reads no verdict.
   - A second run on the extended grid was stopped by me at 23 minutes for the 30-minute lock budget, with no number
     used; the stage now runs in cached pieces with warm-started fits (the optimum does not depend on the start).
7. **Choices the pre-registration leaves open, fixed in the code's docstring before any number.** Part (a) rows are
   the possessions whose last attempt has a known shooter (84,594; 213 dropped because the shooter was not tracked);
   possessions without an attempt are left out. The attacker enters as an indicator; the zone mix columns are not
   penalized, with "three" as the reference. Part (b) rows need an assigned defender at release and an xFG value (422
   attempts lack an out-of-fold xFG). On development games xFG is the stored out-of-fold prediction; on Set A it is
   the frozen xFG model fitted on the training games, which never read Set A (the same path decisions-E1 used).
8. **Refusals.** The scorer refuses before computing anything if a test game lacks frames, possessions, shot rows or a
   roster, if any test game gives no rows to either part, or if any value is non-finite; a refusal is reported as
   such, never as a fail.

## Set A

Not scored yet.
