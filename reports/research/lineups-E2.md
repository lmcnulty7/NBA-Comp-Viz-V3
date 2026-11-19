# lineups-E2: lineup ratings shrunk toward their players (L-RAPM)

Written 2026-11-18. Pre-registration: docs/research/lineups.md, section 6, "lineups-E2"; PLAN.md D102 and D104. The
sections up to and including "Commitments before the test score" were committed before the score (commits dbb4e885,
7b2d30d1, 12ae4935); only the intro paragraph's tense, the verdict line and the section "Test season" were written
after it. Scoring code:
tools/research_lineups_e2.py; the development numbers come from commit 7b2d30d1 and the test score from 12ae4935 (each
recorded as code_commit in its json). Every number below is copied from reports/research/lineups-E2-dev.json or
reports/research/lineups-E2-final.json, except where marked: the paper's settings (penalties 4000 and 6000, unseen players at -1, unseen lineups reported
apart) are quoted from Petridis and Pelechrinis (2026); the projected half-widths in commitment 6 are computed from the
dev json's intervals; 99 percent and 30 percent are thresholds set here before the score. Before this score the test season (2025-26) had been
scored once each by lineups-E1 and winprob-E1; this was its third use, as its own entry of
reports/research/scoring_ledger.json. Intervals are 95
percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with every arm paired on the same
resampled games.

**Verdict: fails as pre-registered (PLAN.md D104, both baselines).** On the 2025-26 season (1095 games in 30
predicted weeks), shrunk lineup ratings (lrapm) show no detectable gain overall over the frozen rapm_synergy ratings: d -0.0005
[-0.0015, +0.0002], far from the -0.01 bar, with an interval that touches zero; only the under-25 bin shows a tiny
gain, -0.0004 [-0.0007, -0.0001], far under the 1 percent bar. Player priors show no detectable gain
over pure shrinkage either: lrapm minus shrink_zero -0.0002 [-0.0009, +0.0004]. Beside it, not as the headline: the
rule holds against the raw baseline (d -0.3204 [-0.3607, -0.2619]), which is weak by construction here (commitment 2).
No data-fault line is crossed. The lineups lane gains a clean negative result: inside a season, lineup ratings shrunk
toward their players predict next week's points per possession no better than ratings that already pool players
(rapm_synergy) or shrink toward the league average.

## What is tested

Inside one season, each week's lineup matchups are predicted from the season's earlier weeks only. The question is
whether lineup offensive and defensive ratings, shrunk toward the sum of their players' ratings from earlier seasons,
predict next week's points per possession better than the two pre-registered baselines, most of all for lineups with
few possessions.

- Observations: each kept stint gives two, the home lineup on offense against the away lineup on defense and the
  reverse, with y the points scored per possession and the stint's possessions (duration / 28.8, frozen) as weight.
- Weeks: from the season's first game; weeks 1 to 4 only train; each later week is predicted from the weeks before it.
- Arms:
  - lrapm: y = b0 + O(offense lineup) - D(defense lineup), a possession-weighted ridge over earlier weeks with each
    rating shrunk toward the sum of its five players' offensive (defensive) ratings. A lineup not yet seen is
    predicted by that sum.
  - raw (baseline 1): the offense lineup's points scored per possession plus the defense lineup's points allowed per
    possession, minus the league average, all from earlier weeks; a lineup not yet seen counts at league average.
  - rapm_synergy (baseline 2): the frozen candidate refit on the development seasons, its stint margin split evenly
    between the two sides. It is handicapped by construction: fixed from the development seasons, no in-season
    update, its margin split evenly.
  - shrink_zero (the ablation): the lrapm ridge shrunk toward 0 (league average), which isolates what player
    information adds.
  - lrapm_net_prior (context, not pre-registered, reads no verdict): lrapm with the prior from the frozen net RAPM,
    each player giving half to offense and half to defense.
- Metric: the weekly RMSE of points per possession averaged over the predicted weeks, and d = (RMSE(lrapm) -
  RMSE(baseline)) / RMSE(baseline) (the paper's equation 1). Bins by the possessions, in earlier weeks, of the
  less-seen of the two lineups: under 25, 25 to 100 (100 included), over 100; a bin's RMSE is pooled over all predicted
  weeks.
- Threshold (pre-registered, read against both named baselines, PLAN.md D104): against raw and against rapm_synergy
  each, d at most -0.01, the paired interval of the RMSE difference wholly below zero, and no bin worse than that
  baseline beyond its interval.

## Inputs and gates

- The development stints are lineups-E1's gated tables (the frozen builder made exactly the stored stints; the frozen
  RAPM was reproduced exactly). Each side's points are rebuilt with the frozen builder and joined back; every kept
  stint matches its stored points difference (32774, 37435, 37114 and 35367 kept stints).
- Synergy gate: the frozen rapm_synergy refit on 2020-21 to 2023-24 has 253 gated lineup terms (alpha 3000). The stored
  2023-24 lineup table holds 15320 lineups, 88 of them with a non-zero synergy coefficient; the refit equals all 88
  (and the zeros of the rest); the largest difference is 0.0 (tolerance 1e-6).
- The player prior (offensive and defensive RAPM) is new here, not frozen. On 2020-21 to 2022-23 its grouped CV chose
  penalties 1000 (offense) and 3000 (defense); offense plus defense correlates 0.957 with the frozen net RAPM of the
  same players, so the sign convention holds on real data (it is also unit-tested).

## Development checks

### Pre-registered: the weekly windows inside 2023-24 (priors from 2020-21 to 2022-23)

Every penalty was chosen on these windows, so they are in sample for the penalty. 1106 games in 30 predicted weeks,
59242 observations. Chosen penalty: 400 for lrapm, shrink_zero and lrapm_net_prior alike (an interior minimum; the curve
is flat from 200 to 800). 54.7 percent of predicted possessions involve a lineup not seen in earlier weeks.

Against rapm_synergy (baseline 2):

- Mean weekly RMSE: lrapm 0.7258 [0.7113, 0.7393], rapm_synergy 0.7265 [0.7119, 0.7399].
- d: -0.0009 [-0.0017, -0.0002]; RMSE difference -0.0007 [-0.0012, -0.0002]. The interval is below zero but d is far
  above -0.01: the rule does not hold.
- Bins, lrapm minus rapm_synergy: under 25 -0.0013 [-0.0017, -0.0008]; 25 to 100 -0.0021 [-0.0038, -0.0004]; over
  100 -0.0008 [-0.0038, +0.0023].

Against raw (baseline 1): d -0.3908 [-0.4309, -0.3184]; RMSE difference -0.4657 [-0.5485, -0.3385]; no bin worse. The
rule holds, but see commitment 2: raw is weak by construction here.

Ablation: lrapm minus shrink_zero -0.0009 [-0.0016, -0.0002].

As the verdict reads now (both baselines), this development check fails.

### Not pre-registered, context and power: 2022-23 with priors from 2020-21 and 2021-22 and the 2023-24 penalties

The penalties are out of sample here. 1098 games in 30 predicted weeks.

- Against rapm_synergy: lrapm 0.7371 [0.7243, 0.7498], rapm_synergy 0.7373 [0.7248, 0.7500]; d -0.0003 [-0.0011,
  +0.0005]; RMSE difference -0.0002 [-0.0008, +0.0003]. Over 100: lrapm minus rapm_synergy +0.0024 [+0.0005, +0.0046],
  worse beyond its interval.
- Against raw: d -0.3798 [-0.4183, -0.3043].
- Ablation: lrapm minus shrink_zero -0.0007 [-0.0012, -0.0002].
- Fails as the verdict reads now.

### The test fit

Priors fitted on every development season (5059 games, 836 players, penalties 1000 and 3000), the frozen
rapm_synergy refit on the same seasons, and the penalties chosen above (400 each).

## Commitments before the test score

1. **The verdict is the pre-registered rule against both baselines (PLAN.md D104).** It passes only if the rule holds
   against raw and against rapm_synergy. lrapm_net_prior reads no verdict. The verdict json keeps each baseline's
   reading apart (against_raw, against_rapm_synergy).
2. **The raw baseline is weak by construction, and its 38 percent is never the headline.** Possessions are estimated
   from duration, so a lineup seen for a few seconds has a fraction of a possession and its raw rating can be several
   points per possession; raw loses to every other arm by 38 to 39 percent in development, almost all in the under-25
   bin. That number is reported in the tables but never leads a summary. The comparison with rapm_synergy and the
   ablation carry the information.
3. **What a pass supports.** Only "shrunk lineup ratings beat raw and rapm_synergy ratings" for next week's points per
   possession inside a season, given who is on the floor. "Toward their players" is added only if the interval of
   lrapm minus shrink_zero is wholly below zero. It is not a game-margin claim, not a season-ahead lineup rank claim
   (lineups-E1's lineup ranks stay inconclusive), and not a claim about any one lineup's rating.
4. **How each outcome is read.**
   - All conditions against both baselines: adopted as a research result, with the claim of item 3.
   - The rule holds against raw but not against rapm_synergy: fails. If the rapm_synergy difference interval is
     wholly below zero but d is above -0.01, it is read as "a small gain over rapm_synergy, under the pre-registered 1
     percent"; if the interval touches zero, as "no detectable gain over rapm_synergy".
   - A bin worse than a baseline beyond its interval: fails, and the bin and baseline are named.
   - The ablation reads apart, as in item 3: wholly below zero, "player priors add a small, detectable gain over pure
     shrinkage"; touching zero, "no detectable gain from player information".
   - A non-finite value refuses the verdict; a bin with no observations is reported as empty (null), not as worse.
5. **Departures from the pre-registration and the paper (recorded in PLAN.md D104).**
   - RMSE is over stint sides weighted by possessions, not over single possessions. It drops a within-stint term that
     is the same for every arm, so every relative change is larger than at possession level and the 1 percent bar is
     easier to clear than in the paper.
   - The prior is an offensive and defensive player RAPM fitted on the development seasons (the paper's construction),
     not the frozen net RAPM, because the hypothesis is about offensive and defensive ratings. The frozen-net prior is
     the context arm lrapm_net_prior.
   - A player never seen counts 0 (frozen behaviour; the paper uses -1). The league average in the paper's priors
     cancels in O - D and is left out.
   - Bins use the less-seen of the two lineups; lineups not yet seen are in the under-25 bin (the paper reports them
     apart; their share of possessions is reported). Bin RMSEs are pooled over all predicted weeks, while the overall
     metric averages weekly RMSEs. The 25-to-100 bin includes 100.
   - Weeks 1 to 4 only train (the paper's warm-up).
   - The playoffs and the NBA Cup final are in the season's weeks (the test season's selection, PLAN.md D89; the paper
     includes playoffs).
   - One penalty rule for every ridge arm, each choosing its own penalty on the 2023-24 windows.
   - Conjugate gradients solve the ridge (tested equal to a direct solve). The first development run used a direct
     sparse solver, which stalled; it was stopped after 10 minutes with no number produced.
   - rapm_synergy is fixed from the development seasons for the whole season, as written ("refit with frozen code on the
     development seasons"); the league average it is centred on comes from earlier weeks.
6. **Power.** Scaled to about 1300 test games by the square root of the game counts (computed from the dev json), the
   half-width of d against rapm_synergy is about 0.0007, so d cannot reach -0.01 unless the test effect is about ten
   times the development one; d against raw has a half-width of about 0.05. The ablation difference had half-widths of
   0.0007 and 0.0005 in development, near the size of its effect.
7. **Possible data faults, reported beside the verdict, no rescore.** The whole-game stint margin agreeing with the
   2025-26 schedule in fewer than 99 percent of checked games (lineups-E1 found 1298 of 1300), or more than 30 percent
   of player slots unseen by the prior fit (8.1 and 7.3 percent in development; the test has the skipped season 2024-25
   between the fit and the scored season, so a higher share is expected).
8. **The test season's third use.** lineups-E1 and winprob-E1 scored it once each; their reports show game-margin and
   win-probability results, nothing about lineup points per possession. Every choice above was made on development
   seasons only; the choices in item 5 were made after lineups-E1's test result was known, and none of them uses a
   test-season number. The reading against both baselines (item 1) was fixed after the development numbers, by D104,
   in the direction that makes a pass harder.

## Test season (2025-26, scored once)

The fit is the one fixed above (priors on 5059 games, 836 players, penalties 1000 and 3000; rapm_synergy with 253
lineup terms; penalty 400 for every ridge arm).

- Games: 1316 listed, all with play-by-play. The frozen builder dropped 16 whole (shifted period labels); no listed
  game lacked play-by-play, no game was built without stints, no game lacked kept stints. 41799 stints built, 40276
  kept. The season has 34 weeks; the 205 games of the first 4 weeks only train, so 1095 games in 30 weeks are
  predicted (67144 observations). No value is non-finite.
- Data checks: the whole-game stint margin equals the schedule's final margin in 1298 of 1300 checked games (share
  0.998, above the 99 percent line). 20.6 percent of player slots belong to players the prior fit never saw (under the
  30 percent line; 8.1 and 7.3 percent in development, higher as expected after the skipped season). 59.8 percent of
  predicted possessions involve a lineup not seen in earlier weeks.

Against rapm_synergy (baseline 2):

- Mean weekly RMSE (points per possession): lrapm 0.7891 [0.7715, 0.8034], rapm_synergy 0.7895 [0.7720, 0.8038].
- d: -0.0005 [-0.0015, +0.0002]; RMSE difference -0.0004 [-0.0012, +0.0001]. d is above -0.01 and the interval touches
  zero: the rule does not hold. Read as committed (item 4): no detectable gain over rapm_synergy.
- Bins, lrapm minus rapm_synergy: under 25 (61608 observations) -0.0004 [-0.0007, -0.0001]; 25 to 100 (4402) -0.0010
  [-0.0024, +0.0004]; over 100 (1134) -0.0007 [-0.0045, +0.0028]. No bin is worse beyond its interval.

Against raw (baseline 1), beside the verdict and never its headline: d -0.3204 [-0.3607, -0.2619]; RMSE difference
-0.3720 [-0.4432, -0.2790]; bins, d of lrapm against raw: under 25 -0.4007 [-0.4738, -0.3178], 25 to 100 -0.0469
[-0.0558, -0.0386], over 100 -0.0168 [-0.0278, -0.0074]. The rule holds against raw.

Ablation and context, reading no verdict:

- lrapm minus shrink_zero: -0.0002 [-0.0009, +0.0004]. Read as committed: no detectable gain from player information
  over pure shrinkage, so "toward their players" is not claimed. In development it was -0.0009 and -0.0007 with
  intervals below zero; that small gain does not carry to the test season.
- Mean weekly RMSE of the other arms: shrink_zero 0.7893 [0.7718, 0.8033], lrapm_net_prior 0.7888 [0.7711, 0.8031].
  All the shrunk and pooled arms sit within 0.001 of each other; raw is the only arm far away (1.1611 [1.0664,
  1.2308]).

What this means for the lane. lineups-E1 showed that player RAPM carries game-margin information across seasons. This
experiment shows that, inside a season, shrinking lineup ratings toward those player ratings adds no detectable gain
overall over pooling players (rapm_synergy) or plain shrinkage, at the stint-side resolution used here; only the
under-25 bin shows a tiny gain over rapm_synergy, -0.0004 [-0.0007, -0.0001], far under the 1 percent bar. The large gap to raw
ratings is a property of tiny estimated possession counts, not evidence for the player prior. Lineup ranks stay
inconclusive (lineups-E1).
