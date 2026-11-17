# lineups-E1: RAPM scored on every game's margin

Draft written 2026-11-17, before the test score. Pre-registration: docs/research/lineups.md, section 6, "lineups-E1".
Scoring code: tools/research_lineups.py. Every number below is copied from reports/research/lineups-E1-dev.json. The
test season (2025-26) has not been scored; the section "Test season" is filled after the single score.
Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with every arm paired on
the same resampled games.

**Verdict: not scored yet.**

## What is tested

The frozen lineups lane was inconclusive because it was scored on about 20 recurring lineups a season. This
experiment scores the same frozen ratings on every game's margin, as Sill (2010) did: a stint's predicted margin is
its possessions / 100 times its predicted net rating, and a game's prediction is the sum over its stints.

Arms, each fitted on the development seasons 2020-21 to 2023-24 (2024-25 is spent and never read, PLAN.md D79):

- floor: home court only (the possession-weighted mean net rating of the fit stints).
- plus_minus: the frozen candidate, each five-man lineup's raw net rating pooled over the fit seasons. A stint gets
  the floor's home court plus the home lineup's rating minus the away lineup's. A lineup never seen counts 0.
- rapm: the frozen candidate, alpha 3000, its own home court plus the home players' ratings minus the away players'.
  A player never seen counts 0 (frozen behaviour).
- rapm_replacement (the ablation): players under 10 minutes per game played lose their own rating; two count
  columns (how many such players are on the floor for each side) get their own ratings, with their penalty chosen by
  grouped CV by game. Every player without a rating counts as one of them at prediction. The 10-minute cut is the value
  Watts, Pipping-Gamon and Wyner (2026) selected; it is fixed, not tuned here.
- player_plus_minus (context arm, not pre-registered, reads no verdict; added by the lead before the test score so a
  reader sees RAPM against a raw baseline that is not a strawman): each player's raw on-court net rating per 100
  possessions over the fit stints. A stint gets the floor's home court plus the mean of the five home players' ratings
  minus the mean of the five away players'. The mean, not the sum: each raw rating already measures the player's
  whole lineup, so a sum over five would count it five times. A player never seen counts 0, as in rapm.

Metric: game-margin RMSE per arm, and R squared against the floor, 1 - SSE(arm) / SSE(floor). The whole-game margin is
primary: every stint the frozen builder makes, with the stints the frozen validation drops (not five against five)
predicted by each arm's home court alone. The margin over the kept stints only is a sensitivity check and reads no
verdict.

Threshold (pre-registered): rapm is validated when the interval of RMSE(rapm) - RMSE(plus_minus) lies wholly below
zero, and rapm's R squared against the floor is above zero with its interval above zero. Otherwise the lane stays
inconclusive and says so.

## Gate

- Stints: the frozen builder, rerun here on the development play-by-play, made exactly the stored stints of every
  development season (34285, 38896, 38867 and 38336 rows; no cell differs).
- Ratings: the frozen candidates refit on those stints chose alpha 3000 and reproduced all 836 stored player ratings
  of the fit on 2020-21 to 2023-24; the largest difference is 0.0 (tolerance 1e-6).
- Data check: in every development season the whole-game stint margin equals the final margin of the frozen schedule
  table in every game (share 1.0). The frozen builder drops 9, 11, 12 and 10 games whole (shifted period labels);
  those games have no stints and are not scored.

## Development checks

### Dry run inside 2023-24 (pre-registered): fit October to February (879 games), score March and April (378 games)

| Arm | RMSE, whole game | R squared vs floor |
| --- | --- | --- |
| floor | 16.06 [14.80, 17.30] | (reference) |
| plus_minus | 28.73 [25.43, 32.13] | -2.201 [-3.154, -1.492] |
| rapm | 14.59 [13.37, 15.78] | 0.175 [0.132, 0.217] |
| rapm_replacement | 14.37 [13.17, 15.54] | 0.200 [0.150, 0.248] |
| player_plus_minus (context) | 14.20 [13.01, 15.36] | 0.218 [0.156, 0.279] |

- rapm minus player_plus_minus (context): +0.39 [+0.05, +0.76]. Inside one season, the raw player rating predicts
  better than RAPM.
- rapm minus plus_minus: -14.15 [-17.56, -10.96]. rapm minus floor: -1.47 [-1.85, -1.08]. Passes as written.
- rapm_replacement minus rapm: -0.22 [-0.45, +0.02]; the CV chose a penalty ratio of 0.
- Kept stints only (91.1 percent of possessions): rapm 14.45 [13.21, 15.66], R squared 0.172 [0.128, 0.215]. Same
  reading.
- This design (within season, March and April scored) is the published one. Sill (2010) reports R squared 0.169 and
  Watts et al. (2026) 0.173 to 0.178; our 0.175 [0.132, 0.217] sits with them. RMSE is not comparable across eras:
  the spread of our margins is larger (whole-game standard deviation 16.07 in these games), and our March and April
  include the first playoff games.

### Season-ahead check (not pre-registered; context and power only): fit 2020-21 and 2021-22, score every 2023-24 game

This has the test's shape: one season skipped between the fit and the scored season (the test fit ends with 2023-24
and skips the spent 2024-25). 1303 games.

| Arm | RMSE, whole game | R squared vs floor |
| --- | --- | --- |
| floor | 15.73 [15.07, 16.39] | (reference) |
| plus_minus | 18.41 [16.24, 21.02] | -0.370 [-0.782, -0.081] |
| rapm | 15.12 [14.45, 15.79] | 0.076 [0.045, 0.106] |
| rapm_replacement | 15.13 [14.49, 15.77] | 0.075 [0.032, 0.116] |
| player_plus_minus (context) | 15.51 [14.85, 16.17] | 0.027 [0.007, 0.047] |

- rapm minus plus_minus: -3.29 [-5.89, -1.22]. rapm minus floor: -0.61 [-0.86, -0.35].
- rapm_replacement minus rapm: +0.01 [-0.17, +0.18].
- rapm minus player_plus_minus (context): -0.39 [-0.57, -0.21]. With a season between fit and score, RAPM predicts
  better than the raw player rating: the order of the dry run reverses.
- Coverage (counts only): 16.6 percent of player possessions belong to players the fit never saw; only 2.3 percent of
  lineup possessions have a lineup rating from the fit.
- The gain is under half the within-season gain: ratings two seasons old know less about the rosters.

### The test fit (every development season, the fit e1-final uses)

5059 games, 142690 kept stints, 836 players, alpha 3000 (frozen CV). Ablation: 601 players keep their own rating;
the penalty ratio chosen by grouped CV is 10 (penalty 30000). The CV curve is flat: out-of-fold game RMSE moves by
under 0.01 points over the whole grid.

## Commitments before the test score

Fixed here, in the spirit of PLAN.md D80, before score_once is called.

1. **The verdict is the pre-registered rule on the whole-game margin, nothing else.** The kept-stints margin and the
   ablation read no verdict; if the kept-stints reading disagrees with the primary one, the report says so.
2. **What a pass supports.** Player ratings fitted on 2020-21 to 2023-24, summed over who is on the floor, predict
   2025-26 game margins better than the frozen lineup plus-minus and better than home court alone, given the actual
   stints of each game, which a pregame forecast would not know. That is a claim about game margin. It does not validate lineup ranks, rapm_synergy, the rating of any one lineup, any one player's
   rating, or the gap between two players. The lane's lineup-rank result (season-ahead Spearman over about 20
   lineups) stays inconclusive whatever happens here.
3. **How each outcome is read.**
   - Pass (both conditions): "tested: RAPM player ratings predict game margin out of sample", with the R squared and
     its interval.
   - rapm minus plus_minus wholly below zero, but rapm's R squared interval touches or crosses zero: fails. Read as
     "beats the frozen lineup baseline, not shown better than home court". Not validated.
   - rapm's R squared interval wholly below zero: fails, and read as a negative result: ratings this old make game
     predictions worse than home court alone.
   - rapm minus plus_minus straddles zero: fails, read as inconclusive at this sample size, not as evidence that RAPM
     is no better. Wholly above zero: fails, plus_minus is better on margin.
4. **The baseline is weak, so the second condition decides.** "Raw plus-minus ratings" is read as the frozen
   plus_minus candidate that the Baseline line names: raw net ratings of five-man lineups, not shrunk, so a lineup
   seen for a few possessions in the fit carries an extreme rating. In both development checks it loses to home court
   alone. A pass on the first condition is therefore close to certain and says little; the informative condition is R
   squared against the floor. Two choices made here: the floor's home court is added to plus_minus (without it, home
   court alone would beat it by construction), and a lineup never seen counts 0. A player-level raw plus-minus is not
   in the pre-registration and is not added.
5. **Power.** Scaled from the season-ahead check to the 1316 listed test games, the expected half-widths are about
   0.030 for rapm's R squared and 0.26 points for rapm minus floor (about 0.023 and 0.21 scaled from the dry run). A
   true R squared the size of the season-ahead check (0.076) would be detected; one under about 0.03 would likely end
   with an interval that touches zero, which item 3 reads as a fail of the threshold, not as no effect.
6. **Disclosed change after a development result.** The first development run chose the top of the penalty grid for
   the ablation (ratio 10). The grid was then given the rule Watts et al. use (extend past the top while the top wins);
   the next value (20) is worse, so the choice stays 10. This touches the ablation only and no test data.
7. **The context arm.** player_plus_minus is reported beside the others, computed in the same single score, and
   reads no verdict whatever its numbers. rapm minus player_plus_minus is reported with its interval and called a
   context comparison; neither a win nor a loss changes the verdict of item 1.
8. **Games.** Every test game the frozen builder can build is scored; games it drops whole (shifted period labels)
   are counted and reported, not scored. All arms are scored on the same games. Leaving out the builder-dropped games
   is a deviation from the pre-registration's "all test games", stated here; on the development seasons it is 9 to
   12 games a season. Listed games without play-by-play, games built without stints and games with a non-finite value
   are counted too; a non-finite value refuses the verdict instead of reading it as a fail.
9. **Possible data faults.** If the whole-game stint margin agrees with the 2025-26 schedule table in under every
   game (share below 1.0, against 1.0 in every development season), or the share of player possessions by players the
   fit never saw is far above the season-ahead check's 16.6 percent (above 30 percent), it is reported beside the
   verdict as a possible data fault. There is no rescore.

## Test season

Not scored yet.
