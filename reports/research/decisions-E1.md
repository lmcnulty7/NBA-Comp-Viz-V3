# decisions-E1: is the target reliable, and is the metric more than decision mix?

Written 2026-11-17. Pre-registration: docs/research/decisions.md, section 6, "decisions-E1" (PLAN.md D83, D84).
Code: tools/research_decisions.py. Every definition the pre-registration left open was fixed in that file's docstring
and committed (93a90d81, 03:05 local time) before the refit ran (gate record written 03:08) and before any statistic
existed (development result written 03:10). Every number below is copied from reports/research/decisions-E1-dev.json,
except the three marked post hoc, which come from reports/research/decisions-E1-posthoc.json. Intervals are 95 percent
percentile intervals from a bootstrap of games, 2000 resamples, seed 0.

**Verdict.** H1 fails: the target, a player's on-court offensive rating, is reliable over half a season (split-half
r 0.420). As pre-registered, the failed verdict of the decisions lane is therefore read as evidence against the decision
metric, not as a target too noisy to detect anything. H2 holds: the frozen per-player mean quality is very stable
(0.932). H3 fails: kind shares explain about a third of its spread between players (R squared 0.342), not half. The
kind-adjusted quality is **not adopted**: it is stable (0.961) but more tied to kind shares (R squared 0.537), not less.
The frozen player table stays as it is. Set A was not scored: none of the pre-registered statistics can exist on it
(see "Test set").

## What was tested

The frozen decisions lane values each shot, pass and drive against the best option at that moment; quality is the
regret (at most 0). Its validation, a per-game Spearman between player mean quality and on-court offensive rating,
found nothing (0.012 [-0.008, 0.029] out of fold, reports/decisions.md). This experiment asks two questions first
(Franks et al. 2016: reliability before relevance). Is the target reliable enough that a real signal could have shown
up? Is the metric more than the player's mix of decision kinds?

- H1: the target has split-half reliability below 0.2 across the development games.
- H2: the frozen per-player mean quality has a Franks stability of at least 0.5.
- H3: at least half of the between-player variance of the frozen mean quality is explained by kind shares.
- Adoption of a kind-adjusted quality (regret minus the development mean regret of the same kind and court zone):
  stability at least 0.5 with its interval above 0.3, and R squared on kind shares below 0.2.

Definitions fixed before the run (module docstring): halves by date, odd/even games for split-half and first half of
the season against the second for stability; player sets of at least 300 decisions in each half (split-half,
stability) or in all games (discrimination, R squared); the target value of a player is the mean of their per-game
on-court rating over games with at least 3 decisions (the frozen validation's unit); court zone is the frozen EPV
lane's zone of the decision-frame distance (rim, paint, mid, three). Stability uses variances with ddof 1. The
paper's own sample moment (1/S_p) is biased upward with only two periods, so it is reported beside but never read.
Hypotheses are read on the point value, with the interval reported beside it.

## Gate (the baseline)

The frozen decisions lane (analytics.decisions.train.run, unchanged) was rerun in one process on the 604 development
games (every training game; the D71 test games were never in the frozen data, and the 17 spent held-out games were not
loaded), seed 0, 5 grouped folds, reading the stored EPV and xFG outputs. Nothing was written under DATA_ROOT or
reports/ by the refit, and the lane's command line was not run.

- decision_events: all 518421 decisions reproduced, keys identical, largest difference in any value 0.0 points
  (tolerance 1e-6, fixed before the run).
- decision_players: all 446 players reproduced exactly.
- The validation rows printed in reports/decisions.md were recomputed and match to the printed digit: xfg_selected
  0.012 [-0.008, 0.029], 317 of 603 positive, sign p 0.222, shuffled -0.013; and the location row. This also checks
  the per-game on-court ratings, which the frozen run never stored.
- Prior selected by the frozen rule: xfg_selected (coverage 0.99).

The gate record has the same form as tools/frozen_common's check records: it vouches for its files by hash. It is
written by this module, because frozen_common's tolerance table (D49) has no entry for the decisions lane.

## Development games (604 games, 518421 decisions, 12769 player-games)

Players: 344 with at least 300 decisions; 251 with 300 in each odd/even half; 240 in each chronological half (302
games per half; the first half ends on 2015-12-10).

| measure | split-half r (1) | Spearman-Brown | discrimination (2) | stability (2) | R squared on kind shares (3) |
| --- | --- | --- | --- | --- | --- |
| target: on-court points per 100 | 0.420 [0.132, 0.414] | 0.591 [0.234, 0.586] | 0.571 [0.635, 0.751] | 0.788 [0.187, 0.600] | not computed |
| frozen mean quality | 0.927 [0.843, 0.893] | 0.962 [0.915, 0.943] | 0.956 [0.954, 0.961] | 0.932 [0.841, 0.902] | 0.342 [0.288, 0.357] |
| kind and zone adjusted (4) | 0.954 [0.895, 0.930] | 0.977 [0.944, 0.964] | 0.973 [0.971, 0.975] | 0.961 [0.902, 0.941] | 0.537 [0.493, 0.539] |
| kind adjusted only (ablation) | 0.941 [0.868, 0.912] | 0.970 [0.929, 0.954] | 0.964 [0.961, 0.967] | 0.945 [0.869, 0.921] | 0.419 [0.369, 0.424] |

- Split-half Spearman: target 0.364 [0.117, 0.377], frozen 0.918 [0.832, 0.886], adjusted 0.920 [0.831, 0.888],
  kind only 0.920 [0.820, 0.881].
- Paired test, stability adjusted minus frozen: +0.029 [+0.032, +0.069]; kind only minus frozen: +0.013 [+0.011,
  +0.035].
- Spread between players (standard deviation): target 6.01 points per 100 [6.55, 7.84]; frozen mean quality 0.057
  [0.057, 0.059]; adjusted 0.061 [0.061, 0.063]; kind only 0.060 [0.060, 0.063].
- Stability by the paper's printed estimator (biased with two periods, not read): target 1.382, frozen 0.9995,
  adjusted 1.001, kind only 1.000. A value above 1 shows the bias.

**Read the intervals with care.** For every statistic that depends on how much noise sits in a player's value
(split-half r, stability, discrimination, spread), the interval does not straddle the point value. A game-level
resample repeats some games and drops others, which adds about one more sampling variance to each player's value.
Reliabilities therefore come out lower in the resamples than in the data, and the spread of the noisy target comes
out higher. For example, the frozen split-half r is 0.927 in the data; doubling its noise share gives about 0.86,
the middle of its interval. This is a known weakness of the percentile bootstrap for reliability statistics, not a
fault in the data. The method was fixed before the run, so it is kept and reported as computed. For the adoption rule
it is conservative (the interval sits low), and the rule failed on R squared anyway. The paired difference has the
same shift.

Mean regret by kind and zone (development games; the cells of the adjustment):

| kind | rim | paint | mid | three | all zones |
| --- | --- | --- | --- | --- | --- |
| shot | -0.211 (6892) | -0.236 (17460) | -0.215 (12046) | -0.062 (17802) | -0.171 |
| pass | -0.395 (5164) | -0.211 (30526) | -0.205 (49730) | -0.451 (219040) | -0.386 |
| drive | none (0) | -0.163 (3078) | -0.256 (17916) | -0.601 (138767) | -0.554 |

Decision counts in brackets. No decision had a missing distance. The cell means are descriptive and use every
development game, so they carry no interval.

## Hypotheses and protocol

| | statement | value | 95 percent interval | result |
| --- | --- | --- | --- | --- |
| H1 | target split-half r below 0.2 | 0.420 | 0.132 to 0.414 | fails (the interval crosses 0.2; it sits low, see above) |
| H2 | frozen stability at least 0.5 | 0.932 | 0.841 to 0.902 | holds |
| H3 | frozen R squared on kind shares at least 0.5 | 0.342 | 0.288 to 0.357 | fails |
| adoption | adjusted stability at least 0.5, interval above 0.3, R squared below 0.2 | 0.961; 0.902; 0.537 | | not adopted (R squared) |

Protocol, as pre-registered: H1 fails, so the target is reliable, and the failed verdict of the lane is read as
evidence against the metric. The kind-adjusted quality does not replace the frozen mean in the player table.

## What this means

- The frozen player mean measures something lasting: its stability (0.932) and discrimination (0.956) are high. About
  a third of its spread between players is the mix of shots, passes and drives (0.342). So the player table partly
  ranks role, but most of it is not role alone.
- Adjusting each decision by the average of its kind and zone does not remove the link to role; it raises it (0.537;
  0.419 with kind alone). Inside each kind and zone, the size of the regret also moves with a player's mix. So a
  simple centring cannot turn the table into a skill table, and the experiment does not adopt it.
- The target is reliable over a half season (r 0.420). The lane's validation, though, ranks players within one game.
  Post hoc, not pre-registered, and not used for any verdict: the players of the split had a mean of 18.0 qualifying games per
  half (median 18), and the Spearman-Brown formula stepped down to one game gives a per-game reliability of about
  0.039. A single game's on-court rating is almost all noise, so a per-game Spearman against it has little room to
  move whatever the metric does. This does not change the protocol's reading above. It bears on how much weight the
  failed per-game statistic can carry, which is the lead's call, not this experiment's.
- On-court rating is shared by every teammate on the floor, so part of its half-season reliability may be the team's
  offence rather than the player. This was not measured here.

## Test set (Set A): not scored

The pre-registration asks for the same statistics once on the D71 test set. They cannot exist there, so the test set
was not touched and the scoring ledger has no decisions-E1 entry.

- Set A is the 10 games of a single date, 2015-12-05 (PLAN.md D77; tests/test_research_testset.py checks the date).
  On one date every team plays once, so every player has at most one game. No player can sit in both halves, so
  split-half reliability and stability have no player at all.
- Discrimination and R squared need 300 decisions in the set. Post hoc: the most decisions any player made in one of
  the 12769 development player-games is 229 (99th percentile 153). No player can reach 300 in one game.
- Scoring would read the test games' outcomes and use up the experiment's single ledger entry to produce only empty
  values. Set B (ACB) has no path to the frozen features (D80), as for xfg-E1.

The scoring code (e1-final) is written and guarded, but it was not run. A test-set check of this question would need
a test set where players have several games, or a gate set for one game; either is a new pre-registration.

## Proposed wording for reports/decisions.md

Not applied: the lane report is frozen output. Proposed text, to sit next to the unchanged failed verdict:

> A later check (decisions-E1, reports/research/decisions-E1.md) asked whether the target was too noisy to detect
> anything. It was not: a player's on-court offensive rating over half a season is fairly reliable (split-half r
> 0.42, 95 percent interval 0.13 to 0.41; the interval sits low because resampling games adds noise). So the failed
> validation counts as evidence against this decision metric. The player means are stable (stability 0.93, interval
> 0.84 to 0.90), but about a third of their spread is the player's mix of shots, passes and drives (R squared 0.34,
> interval 0.29 to 0.36), so the table partly ranks role. Adjusting for kind and court zone did not remove that link,
> so the table is unchanged.

## What would change the verdict

A target with a per-game reliability well above 0.04, or a validation that pools each player over many games, would
let the relevance test mean more. decisions-E2 changes the question instead: it checks the chosen option's value
against the points that followed, decision by decision.
