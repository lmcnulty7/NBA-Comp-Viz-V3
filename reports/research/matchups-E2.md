# matchups-E2: hidden Markov assignment with fitted weights

Written 2026-11-19. Pre-registration: docs/research/matchups.md, section 6, "matchups-E2", with its D112 amendment
(PLAN.md D111, D112, D114). The sections from "What is tested" to the commitments were drafted before PLAN.md D114
decided that Set A is not scored; after it, only the verdict line, the defended FG% figures (item 6), the header and
the section "Set A" changed. Scoring code: tools/research_matchups_e2.py, reusing tools/research_matchups.py
(matchups-E1). The development pieces were computed at commit 09432573; the dev json was assembled at b72fe4e7 (it
records both, as pieces_commit and code_commit). Between the two only the assembly's loader of the cached game ids, the
Set A scorer (never run) and the defended FG% statistic changed; the reported piece (switches and defended FG%) was
recomputed at b72fe4e7, every other piece is the cached one of 09432573. The json's pieces_commit field does not
record that recompute; this header does. Every number below is copied from reports/research/matchups-E2-dev.json,
except where marked: Franks et al.'s weights are quoted from the paper (docs/research/matchups.md), the old defended
FG% values (item 6) from the first committed dev json (550212bb), and ratios marked computed. Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games
(seed 0), with every assignment paired on the same rows and the same resampled games.

**Verdict: not adopted, on development (D112, D114); Set A not scored (D114).** The HMM assignment carries less
defender signal than the frozen Hungarian assignment: the E1 part (a) gain falls from 0.004574 to 0.001418, a
difference of -0.003156 [-0.003417, -0.002893]. It fails. The HMM does switch far less (0.73 against 2.44 per
defender-possession). The site keeps the frozen assignment.

## What is tested

matchups-E1 part (a) found that defender identity, as the frozen Hungarian assignment places it, carries
out-of-sample information on which attacker takes the possession's last shot. E2 replaces the hand-set Hungarian
cost with the hidden Markov model of Franks et al. (2015) and asks whether that assignment carries more of it.

- The model: each defender's position at a 1 Hz sampled frame is normal around a convex mix of his attacker, the ball
  and the basket; which attacker he guards follows a Markov chain (stay with probability rho). Weights, variance and
  rho are fitted by EM on the development games only. The HMM assignment is the Viterbi path; doubles are allowed.
- Ablation (pre-registered): the same fitted emission without the transition (the most likely attacker at each frame).
- Metric (pre-registered): the E1 part (a) gain (no-defender minus defender log loss, out of fold) under each
  assignment. The rows and the no-defender model are the same under every assignment, so HMM minus Hungarian is the
  per-row Hungarian defender-model loss minus the HMM defender-model loss. The pre-registration writes the E1 gain the
  other way round (defender minus no-defender log loss, where negative is better); here a larger gain means more
  defender signal. This sign convention was fixed in the code's docstring before any number.
- Threshold (pre-registered): the HMM's gain larger, the difference above zero with its interval wholly above zero.
  D112: the development games decide; Set A is scored once as a confirmation.
- Context (D112, reads no verdict): the same difference with pre-shot shares (assignments from frames at least 2
  sampled frames before the release only; the HMM is decoded on the cut chains, so no later frame enters it).

## Inputs and checks

- The Hungarian arm is matchups-E1's gated assignment (the frozen assignment rebuilt on all 604 development games,
  equal to the stored table). Its gain here, 0.004574 [0.004230, 0.004904], equals matchups-E1's development number.
- The HMM cache: 8,725,037 defender-frames in 632,682 chains (a chain breaks when the five attackers change; cells
  without the ball's position are left out).
- EM: 8 iterations, converged. Weights: attacker 0.652, ball 0.106, basket 0.242 (Franks et al.: 0.62, 0.11, 0.27).
  Stay probability 0.937 per sampled frame (about one second); variance 27.6 square feet.
- Unit tests check forward-backward and Viterbi against brute force, the convex weight step, and EM recovering known
  parameters on synthetic chains.

## Development games (604 games, 430,555 attacker rows, out of fold)

| Assignment | E1 part (a) gain | Switches per defender-possession | No-switch share | Defended FG%, beyond 4 ft minus within |
| --- | --- | --- | --- | --- |
| Hungarian (frozen) | 0.004574 [0.004230, 0.004904] | 2.44 | 0.184 | 0.0267 [0.0196, 0.0344] |
| HMM | 0.001418 [0.001164, 0.001666] | 0.73 | 0.470 | 0.0235 [0.0164, 0.0314] |
| Emission only (ablation) | 0.002248 [0.001976, 0.002507] | 3.04 | 0.122 | 0.0062 [-0.0014, +0.0142] |

- HMM minus Hungarian: -0.003156 [-0.003417, -0.002893]. Wholly below zero: the HMM loses about two thirds of the
  defender signal (computed: 1 - 0.001418 / 0.004574 = 0.69). Fails.
- Ablation: emission only minus Hungarian -0.002325 [-0.002548, -0.002094]; HMM minus emission only -0.000830
  [-0.000975, -0.000693]. Both steps lose signal: the fitted emission (attacker, ball and basket) carries less than the
  hand-set cost, and the temporal smoothing loses a further 0.000830. The emission step changes two things at once: the
  fitted mean, and the one-to-one constraint of the Hungarian matching, which it drops (doubles allowed).
- Context, reads no verdict: with pre-shot shares, Hungarian gain 0.002162 [0.001879, 0.002423], HMM 0.000766
  [0.000553, 0.000984], difference -0.001396 [-0.001621, -0.001167]. The HMM also loses before the shot, so its loss is
  not only the late switch onto the shooter.
- The HMM does what it was built for: far fewer switches (0.73 against 2.44 per defender-possession, 47 percent of
  defender-possessions without a switch against 18 percent). Fewer switches did not bring more defender signal. The
  defended FG% gap is about the same under the HMM and much weaker under the emission alone. (This table's Hungarian
  gap uses the nearest assigned defender at frames in the HMM cache, so it differs slightly from the frozen report's
  0.027, which counts every assigned attempt.)
- Penalties (part (a) defender models): Hungarian 0.004 to 0.037 per fold, HMM 0.11 to 0.33, emission only 0.04 to
  0.33; the final fits use 0.037, 0.333 and 0.333.
- Projected Set A half-width (D112, from the development per-game variance of the difference): about 0.0020, against a
  development difference of -0.0032.

## Commitments before the Set A score

1. **Development decides (D112): the HMM fails.** Set A cannot change that.
2. **Set A readings, fixed now, given the development fail.** Interval wholly below zero: "confirms the development
   fail". Interval straddling zero: "inconclusive at 10 games", not evidence either way. Interval wholly above zero:
   "does not overturn the development fail (D112)". (Had development passed, wholly above would have read "confirms" and
   wholly below "contradiction".) The ablation and the pre-shot context read no verdict on Set A either. A t interval on
   the 10 per-game means is reported beside each bootstrap interval (report only, as D80).
3. **What the fail supports.** Under the E1 part (a) test, the fitted HMM assignment carries less information on who
   takes the shot than the hand-set Hungarian cost. About half of the Hungarian gain comes from frames at and after the
   release (matchups-E1's pre-shot context), so the E1 gain alone cannot tell a better assignment from one that removes
   that shot-moment signal; the pre-shot arm shows the HMM also loses before the shot (-0.001396 [-0.001621,
   -0.001167]). It does not show the HMM's matchups are wrong: there is no
   ground-truth matchup record, and this metric judges an assignment only by how much defender signal it carries. The
   HMM is smoother (fewer switches) and keeps the defended FG% gap; it is a different, not a better-tested, assignment.
   The site keeps the frozen assignment.
4. **Projected power.** At a half-width of about 0.0020 Set A will most likely read "confirms the development fail" if
   the development effect holds (it is about 1.6 half-widths; computed: 0.003156 / 0.0020).
5. **Choices fixed in the code's docstring before any development number.** Chains break when the attackers change;
   frames without five attackers or without the ball are left out; the initial state is uniform; EM starts from
   Franks et al.'s weights, rho 0.9, and stops at a relative log-likelihood gain under 1e-7 (it took 8 iterations);
   shares use every five-attacker frame of the possession as the denominator; the pre-shot HMM is decoded on cut chains.
6. **Disclosed process facts.** The first full development run computed every piece at 09432573, then its assembly
   crashed loading the cached game ids; the loader was fixed (4ae5e744) and the same pieces assembled, with both commits
   recorded. One assembly attempt with a mistyped commit was stopped while it waited for the lock, before it computed
   anything. After that assembly, writing this report found a bug in the defended FG% statistic (reported, read by no
   verdict): an attempt whose defender position was missing (a frame without the ball) counted as within 4 ft. Fixed in
   b72fe4e7 and only that piece recomputed, with the lead's agreement. The Hungarian gap moved from 0.0256 [0.0187,
   0.0333] (26,025 attempts within 4 ft) to 0.0267 [0.0196, 0.0344] (25,696); the HMM and emission-only figures did not
   change, because their assignments exist only at frames in the cache. The old piece is kept aside in the work
   folder. The frame cache (e2-cache) was built by working-tree code between the amendment commit (708b831b) and the
   code commit (09432573): positions only, no metric; its function frame_rows did not change before that commit.
7. **Refusals.** The scorer refuses before computing anything if a test game lacks frames, possessions, shot rows or a
   roster, if any test game gives no part (a) row or no HMM cell, or if any value is non-finite; a refusal is reported
   as such, never as a fail.

## Set A

Not scored (PLAN.md D114). Development decided the experiment, and under D112 a Set A reading could only confirm the
development fail or fail to overturn it, so it cannot change the verdict (as D109 for gravity-E1). The ledger keeps
Set A unused by matchups-E2. The scorer, the dry run and the final step are in the code with their tests (a rehearsal
through score_once on a synthetic root passes), but were never run; the readings of commitment 2 stay as written
before the decision.
