# lineups-E2: lineup ratings shrunk toward their players (L-RAPM)

Draft written 2026-11-18, before the test score. Pre-registration: docs/research/lineups.md, section 6, "lineups-E2";
PLAN.md D102. Scoring code: tools/research_lineups_e2.py. Every number below is copied from
reports/research/lineups-E2-dev.json, except where marked: the paper's settings (penalties 4000 and 6000, unseen players
at -1, possessions excluded for unseen lineups) are quoted from Petridis and Pelechrinis (2026). The test season
(2025-26) has not been scored by this experiment. It has been scored once each by lineups-E1 and winprob-E1 (the two
entries of reports/research/scoring_ledger.json for that source); this is its third use, as its own ledger entry.
Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with every arm paired on
the same resampled games.

**Verdict: not scored yet.**

## What is tested

Inside one season, each week's lineup matchups are predicted from the season's earlier weeks only. The question is
whether lineup offensive and defensive ratings, shrunk toward the sum of their players' ratings from earlier seasons,
predict next week's points per possession better than raw lineup ratings, most of all for lineups with few
possessions.

- Observations: each kept stint gives two, the home lineup on offense against the away lineup on defense and the
  reverse, with y the points scored per possession and the stint's possessions (duration / 28.8, frozen) as weight.
- Weeks: from the season's first game; weeks 1 to 4 only train; each later week is predicted from the weeks before it.
- Arms:
  - raw (the baseline): the offense lineup's points scored per possession plus the defense lineup's points allowed
    per possession, minus the league average, all from earlier weeks; a lineup not yet seen counts at league average.
  - lrapm: y = b0 + O(offense lineup) - D(defense lineup), a possession-weighted ridge over earlier weeks with each
    rating shrunk toward the sum of its five players' offensive (defensive) ratings. A lineup not yet seen is
    predicted by that sum.
  - shrink_zero (the ablation): the same ridge shrunk toward 0 (league average).
  - rapm_synergy (the second pre-registered baseline, reads no verdict): the frozen candidate refit on the development
    seasons, its stint margin split evenly between the two sides.
  - lrapm_net_prior (context, not pre-registered, reads no verdict): lrapm with the prior from the frozen net RAPM,
    each player giving half to offense and half to defense.
- Metric: the weekly RMSE of points per possession averaged over the predicted weeks, and d = (RMSE(lrapm) -
  RMSE(raw)) / RMSE(raw) (the paper's equation 1). Bins by the possessions, in earlier weeks, of the less-seen of the
  two lineups: under 25, 25 to 100, over 100.
- Threshold (pre-registered): d at most -0.01 with the paired interval excluding zero, and no bin worse than raw beyond
  its interval.

## Inputs and gates

- The development stints are lineups-E1's gated tables (the frozen builder made exactly the stored stints; the frozen
  RAPM was reproduced exactly). Each side's points are rebuilt with the frozen builder and joined back; every kept
  stint matches its stored points difference (32774, 37435, 37114 and 35367 kept stints).
- Synergy gate: the frozen rapm_synergy refit on 2020-21 to 2023-24 reproduces the stored synergy of all 15320
  lineups of the 2023-24 table (253 gated lineup terms, alpha 3000); the largest difference is 0.0 (tolerance 1e-6).
- The player prior (offensive and defensive RAPM) is new here, not frozen. On 2020-21 to 2022-23 its grouped CV chose
  penalties 1000 (offense) and 3000 (defense); offense plus defense correlates 0.957 with the frozen net RAPM of the
  same players, so the sign convention holds on real data (it is also unit-tested).

## Development checks

### Pre-registered: the weekly windows inside 2023-24 (priors from 2020-21 to 2022-23)

Every penalty was chosen on these windows, so they are in sample for the penalty. 1106 games in 30 predicted weeks,
59242 observations. Chosen penalty: 400 for lrapm, shrink_zero and lrapm_net_prior alike (an interior minimum; the curve
is flat from 200 to 800). 54.7 percent of predicted possessions involve a lineup not seen in earlier weeks.

| Arm | Mean weekly RMSE (points per possession) | d against raw |
| --- | --- | --- |
| raw | 1.1915 [1.0627, 1.2753] | (reference) |
| lrapm | 0.7258 [0.7113, 0.7393] | -0.3908 [-0.4309, -0.3184] |
| shrink_zero (ablation) | 0.7267 [0.7121, 0.7403] | -0.3901 [-0.4299, -0.3172] |
| rapm_synergy | 0.7265 [0.7119, 0.7399] | -0.3903 [-0.4303, -0.3173] |
| lrapm_net_prior (context) | 0.7260 [0.7115, 0.7396] | -0.3907 [-0.4306, -0.3179] |

- lrapm minus raw: -0.4657 [-0.5485, -0.3385]. Passes as written; no bin is worse.
- lrapm minus shrink_zero: -0.0009 [-0.0016, -0.0002].
- Bins, d of lrapm against raw: under 25 (52660 observations) -0.4634 [-0.5356, -0.3710]; 25 to 100 (4798) -0.0464
  [-0.0545, -0.0383]; over 100 (1784) -0.0168 [-0.0246, -0.0086].

### Not pre-registered, context and power: 2022-23 with priors from 2020-21 and 2021-22 and the 2023-24 penalties

The penalties are out of sample here. 1098 games in 30 predicted weeks.

- Mean weekly RMSE: raw 1.1886 [1.0603, 1.2671], lrapm 0.7371 [0.7243, 0.7498], shrink_zero 0.7378 [0.7250, 0.7505],
  rapm_synergy 0.7373 [0.7248, 0.7500].
- d of lrapm against raw: -0.3798 [-0.4183, -0.3043]. lrapm minus shrink_zero: -0.0007 [-0.0012, -0.0002].
- Bins, d of lrapm against raw: under 25 -0.4467 [-0.5175, -0.3580]; 25 to 100 -0.0473 [-0.0574, -0.0376]; over 100
  -0.0246 [-0.0333, -0.0179].

### The test fit

Priors fitted on every development season (5059 games, 836 players, penalties 1000 and 3000), the frozen
rapm_synergy refit on the same seasons, and the penalties chosen above (400 each).

## Commitments before the test score

1. **The verdict is the pre-registered rule against raw, nothing else.** The ablation, rapm_synergy and
   lrapm_net_prior read no verdict.
2. **The raw baseline is weak, so a pass says little.** Possessions here are estimated from duration, so a lineup
   seen for a few seconds has a fraction of a possession, and its raw rating can be several points per possession.
   In both development checks raw loses to every other arm by about 38 to 39 percent, mostly in the under-25 bin. A pass is
   close to certain. The informative numbers, reported beside the verdict and reading none:
   - lrapm minus shrink_zero (the pre-registered ablation, what player information adds over pure shrinkage): an
     interval wholly below zero is read as "player priors add a small, detectable gain"; an interval that touches zero
     as "no detectable gain from player information". In development it was -0.0009 and -0.0007, about 0.1 percent.
   - lrapm against raw in the 25-to-100 and over-100 bins, where raw ratings are not dominated by fractions of a
     possession (about -4.6 and -1.7 to -2.5 percent in development).
   - lrapm minus rapm_synergy.
3. **What a pass supports.** Inside a season, lineup ratings shrunk toward their players predict next week's points
   per possession better than raw lineup ratings, given who is on the floor. It is not a game-margin claim, not a
   season-ahead lineup rank claim (lineups-E1's lineup ranks stay inconclusive), and not a claim about any one
   lineup's rating.
4. **How each outcome is read.** All three conditions: adopted as a research result. d above -0.01 with the
   difference interval below zero: fails (gain under 1 percent). Difference interval touching zero: fails,
   inconclusive at this sample size. A bin worse than raw beyond its interval: fails, and the bin is named. A
   non-finite value refuses the verdict; a bin with no observations is reported as empty (null), not as worse.
5. **Choices made here, disclosed.**
   - The prior is an offensive and defensive player RAPM fitted on the development seasons (the paper's construction),
     not the frozen net RAPM, because the hypothesis is about offensive and defensive ratings. The frozen net prior is
     the context arm lrapm_net_prior.
   - A player never seen counts 0 (frozen behaviour; the paper uses -1). The league average in the paper's priors
     cancels in O - D and is left out.
   - RMSE is over stint sides weighted by possessions, not over single possessions: it leaves out a within-stint term
     no prediction changes, so relative changes are larger than possession-level ones and are not comparable to the
     paper's.
   - Bins use the less-seen of the two lineups. Lineups not yet seen are in the under-25 bin (the paper reports them
     apart); their share of possessions is reported.
   - rapm_synergy is fixed from the development seasons for the whole season, as the pre-registration says ("refit
     with frozen code on the development seasons"); the league average it is centred on comes from earlier weeks.
   - The playoffs and the NBA Cup final are in the season's weeks (the paper includes playoffs).
   - Each ridge arm's penalty is its own, chosen on the 2023-24 windows (the same rule for all).
   - The first development run used a direct sparse solver, which was too slow (stopped after 10 minutes with no
     result); the solver is now conjugate gradients, tested equal to a direct solve. No number was produced before
     the change.
6. **Power.** Scaled from development to about 1300 test games, the half-width of d's interval is about 0.05; d against
   raw will not be near its threshold. The ablation difference had half-widths of 0.0007 and 0.0005 in development, so
   a gain of the development size (0.0007 to 0.0009) is near the edge of detection.
7. **Possible data faults, reported beside the verdict, no rescore.** The whole-game stint margin agreeing with the
   2025-26 schedule in under every game (lineups-E1 found 1298 of 1300), or more than 30 percent of player slots
   unseen by the prior fit (8.1 and 7.3 percent in development; the test has the skipped season 2024-25 between the
   fit and the scored season, so a higher share is expected).
8. **The test season's third use.** lineups-E1 and winprob-E1 scored it once each; their reports show game-margin and
   win-probability results, nothing about lineup points per possession. Every choice above was made on development
   seasons only; the choices in item 5 were made after lineups-E1's test result was known, and none of them uses a
   test-season number.

## Test season

Not scored yet.
