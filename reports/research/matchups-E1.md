# matchups-E1: split the outcome into shot frequency and shot efficiency

Written 2026-11-18 and 2026-11-19. Everything above the section "Set A" was committed before the score (last at
bb49534a); only these two sentences, the Set A source sentence, the verdict line and that section were written after
it. Pre-registration: docs/research/matchups.md, section 6,
"matchups-E1", with its D108 amendment (PLAN.md D106, D108, D110). Scoring code: tools/research_matchups.py; the
development numbers come from commit 8bd3ccfe (recorded in the dev json as code_commit). The pre-score review fixes
(commit e2901a40) changed only the gate check, the Set A readings, the scorer's xFG loading and the dry run, none of the
code that made the development numbers. Every number below is copied from reports/research/matchups-E1-dev.json,
except where marked: the frozen lane's numbers are from reports/matchups.md, the first development run's numbers
(item 6) from reports/research/matchups-E1-dev-first-run.json, and the dry run's counts from its printed output.
The Set A numbers are copied from reports/research/matchups-E1-final.json; the single
score is the matchups-E1 entry of reports/research/scoring_ledger.json, whose sportvu_2015_16_untouched entries
list every use of Set A (on this branch before this score: xfg-E1 and embeddings-E1). Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with both
models of a part paired on the same resampled games.

**Verdict: part (a) passes and Set A confirms it; part (b) fails, and Set A is inconclusive at 10 games (D108).**
Part (a), who takes the possession's last shot: defender identity adds out-of-sample information, -0.004574
[-0.004904, -0.004230] nats per attacker row on the development games and -0.007284 [-0.009136, -0.005454] on Set A.
Part (b), make given xFG: no detectable defender information beyond xFG, -0.000003 [-0.000140, +0.000121] on development and
-0.000122 [-0.000521, +0.000295] on Set A. Beside part (a), as committed: about half of it comes from frames at and
after the release (the pre-shot context keeps -0.003525 of -0.007284 on Set A), and it is confounded by role
(opponents put their best defender on the first option), so it is not defensive skill and not proof that a defender
keeps his man from shooting. The lane's original target, possession points, still shows no defender signal.

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
- The frozen ridge refit on it chose alpha 300 in every fold and overall, refit the same 447 defenders as the stored
  table, reproduced every stored ridge value (largest difference 0.0) and the out-of-fold MSEs printed in
  reports/matchups.md (constant 1.2842, raw 1.3244, ridge 1.2854).
- The lane's original target, as the reference: ridge minus constant on possession points +0.00119 [+0.00017,
  +0.00217], worse than the constant, as frozen. The interval differs slightly from the frozen report's [0.0002,
  0.0020] because it uses 2000 game resamples, the frozen report 200.

## Development games (604 games, out of fold)

| Part | Rows | Log loss, defender model | Log loss, no defender | Defender minus no defender | Reading |
| --- | --- | --- | --- | --- | --- |
| (a) who shoots | 430,555 (84,381 possessions) | 0.45885 [0.45768, 0.45997] | 0.46342 [0.46231, 0.46450] | -0.004574 [-0.004904, -0.004230] | passes |
| (b) make given xFG | 95,538 attempts | 0.63322 [0.63109, 0.63540] | 0.63323 [0.63113, 0.63541] | -0.000003 [-0.000140, +0.000121] | fails |

- Part (a) penalties: the defender model chose 0.004 to 0.037 per fold and the no-defender model 0.001 to 0.111; both
  are light (the grid grew downward past 1, see item 6). In fold 0 the no-defender model stopped at the floor of the
  extended grid (1/729, the last step above 0.001), so its penalty there is a bound, not an interior minimum. Part (b):
  the defender model chose 100 or 300; the no-defender model has no penalized column.
- Context, not pre-registered, reads no verdict: part (a) with the defender shares taken only from sampled frames at
  least 2 sampled frames (about 2 s) before the sampled frame at or before the shot's release (81,878 possessions):
  -0.002162 [-0.002423, -0.001879]. About half of part (a)'s gain survives
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
   development fail, whatever it is: a Set A interval wholly below zero for part (b) reads "wholly below zero; does
   not overturn the development fail (D108)", never "confirms". The pre-shot context arm reads "context, reads no
   verdict" and gets no D108 reading. A t interval on the 10 per-game means is reported beside each bootstrap interval
   (report only, as D80); if the two disagree, the report says so. No per-defender claim is made from Set A (about 8
   attempts per defender there).
3. **What part (a)'s pass supports.** Defender identity, as the frozen assignment places it over the possession,
   adds out-of-sample information on which attacker takes the possession's last attempt, beyond the attacker
   himself and his court-zone mix. About half of it depends on frames at and after the release (the pre-shot context
   keeps -0.0022 of -0.0046): a defender who rotates to the shooter at the release gains share on him. It is also
   confounded by role: opponents put their best defender on the first option, so who guards an attacker partly encodes
   the attacker's role that night. So a part (a) defender table, if one is ever shown, must say it measures who a
   defender ends up guarding when shots are taken, partly at the shot itself; it is not defensive skill and not proof
   that a defender keeps his man from shooting. It says nothing about
   making shots (part (b) fails) or about points (the ablation and the frozen target fail). The very high stability
   (0.874) fits a structural trait (role, position, scheme) better than a skill that varies game to game; the report
   says so.
4. **What part (b)'s fail means.** With xFG as an offset, the assigned defender at release adds no detectable
   information on whether the shot goes in (interval -0.00014 to +0.00012 nats per shot). xFG already holds contest
   features, so this does not show that defenders have no effect on makes; it shows none beyond what xFG measures. The
   offset is the stored out-of-fold xFG on the development games and the frozen xFG model fitted on all training games
   on Set A (a full fit, not an out-of-fold one; the frozen model never read Set A).
5. **Projected power (D108).** Set A's half-width for part (a) (about 0.0026) is a little over half the development
   effect, so Set A will likely confirm if the effect holds; for part (b) "inconclusive at 10 games" is the expected
   reading.
6. **Changes after a development number, disclosed.**
   - The first development run (commit 219411696; reports/research/matchups-E1-dev-first-run.json) put part (a)'s
     penalty at the bottom of the grid (1.0) in every fold, for both models. The grid now grows past whichever end wins (it had grown only
     upward); that run's part (a) difference was -0.004030 [-0.004314, -0.003726], this run's -0.004574 [-0.004904,
     -0.004230]. Part (b) was unchanged. Recorded in PLAN.md D110.
   - The pre-shot context was added after that run, labelled not pre-registered, and reads no verdict.
   - A second run on the extended grid was stopped by me at 23 minutes for the 30-minute lock budget, with no number
     used; the stage now runs in cached pieces with warm-started fits (the optimum does not depend on the start).
7. **Choices the pre-registration leaves open, fixed in the code's docstring before any number.** Part (a) rows are
   the possessions whose last attempt has a known shooter (84,594; 213 dropped because the shooter was not tracked);
   possessions without an attempt are left out. The attacker enters as an indicator; the zone mix columns are not
   penalized, with "three" as the reference. Part (b) rows need an assigned defender at release and an xFG value (422
   attempts lack an out-of-fold xFG). On development games xFG is the stored out-of-fold prediction; on Set A it is
   the frozen xFG model fitted on the training games, which never read Set A (the same path decisions-E1 used).
8. **Site table (the lead's decision, before the score).** No site defender table is decided now. If one is ever
   proposed, it shows both the primary part (a) number and the pre-shot context, labelled, under its own D line.
9. **Dry run, before the score.** The scorer's own path (load, frozen assignment, the frozen xFG model, rows, losses)
   ran on 3 development games with the environment and arguments planned for e1-final: 6,915 matchup rows, 515
   attempts, 389 part (a) possessions (1,995 rows), 453 part (b) rows (at least 129 a game), 385 pre-shot possessions,
   every game with rows for both parts, no part (b) defender without a development coefficient, and an empty list of
   non-finite values. Rerunning the assignment on 2 of those games gave parts equal to the stored ones (the zone mix
   included). Mechanics only: these games are in the development fit.
10. **Refusals.** The scorer refuses before computing anything if a test game lacks frames, possessions, shot rows or a
   roster, if any test game gives no rows to either part, or if any value is non-finite; a refusal is reported as
   such, never as a fail.

## Set A (the 10 untouched games of 2015-12-05, scored once)

The models are those fixed above, fitted on every development game (penalties: part (a) 0.037 with defenders and
0.111 without, part (b) 300, the pre-shot context 0.111 for both). The frozen xFG model was loaded before the score.

- Coverage: 10 games, 24,835 matchup rows, 1,677 attempts. Part (a): 1,416 possessions with a known last shooter,
  1,415 kept (1 shooter not tracked), 7,228 attacker rows. Part (b): 1,611 attempts with an assigned defender, 10
  without an xFG value, 1,601 rows (at least 139 a game); every defender in them has a development coefficient. Every
  game gave rows to both parts; no value was non-finite; nothing was refused.

| Part | Rows | Log loss, defender model | Log loss, no defender | Defender minus no defender | t interval, 10 game means | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| (a) who shoots | 7,228 | 0.46081 [0.45185, 0.46930] | 0.46809 [0.45965, 0.47631] | -0.007284 [-0.009136, -0.005454] | -0.007285 [-0.009678, -0.004891] | confirms |
| (b) make given xFG | 1,601 | 0.64616 [0.63100, 0.66185] | 0.64628 [0.63129, 0.66177] | -0.000122 [-0.000521, +0.000295] | -0.000120 [-0.000633, +0.000392] | inconclusive at 10 games |

- Part (a): the interval lies wholly below zero, so Set A confirms the development pass. The t interval on the 10
  per-game means agrees. The Set A difference is larger than the development one (-0.007284 against -0.004574): the
  development point lies just outside the Set A interval; the gap (0.0027, computed) is about one projected
  half-width.
- Part (b): the interval straddles zero: inconclusive at 10 games, as projected (half-width about 0.0010). It does not
  change the development fail.
- Context, reads no verdict: part (a) with pre-shot shares only (1,362 possessions, 6,955 rows): -0.003525 [-0.004867,
  -0.002326] (t interval -0.003517 [-0.005176, -0.001858]). As on the development games, about half of the part (a)
  gain survives when frames at and after the release are left out.
- No per-defender claim is made from Set A.

What this means for the lane. The frozen verdict, no defender signal in possession points, stands (the reference and
the ablation both confirm it). Split into its parts, defender identity carries information on who takes the
shot, partly through who guards whom at the shot and partly through role, and not, beyond xFG, on whether the shot
goes in. No site defender
table follows from this; any future one needs its own D line and shows the pre-shot number beside the primary one.
