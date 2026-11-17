# lineups-E1: RAPM scored on every game's margin

Written 2026-11-17. Pre-registration: docs/research/lineups.md, section 6, "lineups-E1". The sections up to and
including "Commitments before the test score" were committed before the score (commits 3e48eb4c, 2518111f, 8c638fab);
only the verdict line and the section "Test season" were written after it. Scoring code: tools/research_lineups.py at
commit 8c638fab (recorded in the final json as code_commit). Every number below is copied from
reports/research/lineups-E1-dev.json or reports/research/lineups-E1-final.json; the single test score is the
lineups-E1 entry of reports/research/scoring_ledger.json.
Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of games (seed 0), with every arm paired on
the same resampled games.

**Verdict: passes as pre-registered.** On the 2025-26 season (1300 games scored), RAPM player ratings fitted on
2020-21 to 2023-24 predict game margins better than the frozen lineup plus-minus (RMSE difference -1.15
[-1.59, -0.77] points) and better than home court alone (R squared 0.098 [0.065, 0.132]). Read as pre-committed: this
is a tested claim about game margin, given the actual stints of each game, which a pregame forecast would not know. It
does not validate lineup ranks, synergy, any one lineup or player, or the gap between two players; the lane's
lineup-rank result stays inconclusive. One possible data fault is reported beside it (commitment 9): the stint margin
agrees with the schedule table in 1298 of 1300 games, not all.

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

## Test season (2025-26, scored once)

The fit is the one fixed above (5059 games, 836 players, alpha 3000, replacement penalty ratio 10).

- Games: 1316 listed, all with play-by-play. The frozen builder dropped 16 whole (shifted period labels), so 1300 are
  scored; no listed game lacked play-by-play, no game was built without stints, and no game carries a non-finite
  value. 41799 stints were built and 40276 kept (96.1 percent of possessions). (The scorer's top-level kept_stints
  count was overwritten in the json by the results block of the same name; the count is coverage.stints.)
- Data checks: the whole-game stint margin equals the schedule table's final margin in 1298 of 1300 games (share
  0.998, mean gap 0.003 points a game). Below 1.0, so reported as a possible data fault (commitment 9), no rescore.
  Players the fit never saw hold 20.0 percent of player possessions (the season-ahead check: 16.6), under the 30
  percent line. 1.2 percent of lineup possessions have a lineup rating from the fit.
- Margins: whole-game mean 1.72 points for the home side, standard deviation 16.50.

| Arm | RMSE, whole game | R squared vs floor |
| --- | --- | --- |
| floor | 16.50 [15.83, 17.16] | (reference) |
| plus_minus | 16.81 [16.10, 17.54] | -0.039 [-0.079, -0.010] |
| rapm | 15.66 [15.06, 16.30] | 0.098 [0.065, 0.132] |
| rapm_replacement (ablation) | 15.72 [15.11, 16.37] | 0.092 [0.050, 0.134] |
| player_plus_minus (context) | 15.89 [15.26, 16.56] | 0.072 [0.051, 0.092] |

- rapm minus plus_minus: -1.15 [-1.59, -0.77]. Wholly below zero: condition 1 holds.
- rapm R squared against the floor: 0.098 [0.065, 0.132]. Above zero with its interval above zero: condition 2 holds.
- rapm minus floor: -0.83 [-1.13, -0.54]. plus_minus minus floor: +0.32 [+0.08, +0.64]; the frozen lineup baseline
  is worse than home court alone, as in both development checks.
- Ablation, reads no verdict: rapm_replacement minus rapm +0.05 [-0.14, +0.25]. Pooling low-minute and unseen players
  into a shared replacement rating made no detectable difference, as in the season-ahead check.
- Context arm, reads no verdict: rapm minus player_plus_minus -0.23 [-0.46, -0.02]. Across the gap season, RAPM beats
  the raw player rating, the order the season-ahead check showed (inside one season the order was the reverse).
- Sensitivity, kept stints only, reads no verdict: rapm 15.37 [14.77, 16.01], R squared 0.096 [0.062, 0.130], rapm
  minus plus_minus -1.15 [-1.59, -0.76]. Same reading as the primary.

Power and comparability. The R squared interval is 0.067 wide, near the 0.06 the season-ahead check projected. The
test R squared (0.098) is about half the published within-season values (Sill 0.169; Watts et al. 0.173 to 0.178) and
our own within-season dry run (0.175 [0.132, 0.217]): the test fit ends a season before the skipped 2024-25, so the
ratings are older and a fifth of player possessions belong to players they never saw. RMSE is not comparable across
eras; the spread of our margins (standard deviation 16.50) is larger than in Sill's seasons.
