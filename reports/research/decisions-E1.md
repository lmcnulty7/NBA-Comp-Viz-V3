# decisions-E1: is the target reliable, and is the metric more than decision mix?

Written 2026-11-17, revised the same day after the review (ml-reviewer PASS; the fixes below are the lead's list).
Pre-registration: docs/research/decisions.md, section 6, "decisions-E1" (PLAN.md D83, D84). Code:
tools/research_decisions.py. Every definition the pre-registration left open was fixed in that file's docstring and
committed (93a90d81, 03:05 local time) before the refit ran (gate record written 03:08) and before any statistic
existed (development result written 03:10). Pre-registered numbers come from reports/research/decisions-E1-dev.json.
Numbers marked post hoc come from reports/research/decisions-E1-posthoc.json. They were computed after the result and
its review, and no verdict reads them.

**Verdict.** H1 fails: the target, a player's on-court offensive rating, is reliable over half a season (split-half
r 0.420). As pre-registered, the failed verdict of the decisions lane is therefore read as evidence against the decision
metric. Post hoc, not pre-registered: stepped down to one game, the target's reliability is about 0.04, which would meet
H1's intent; part of the half-season reliability may be team offence shared by teammates (not measured). H2 holds:
the frozen per-player mean quality is very stable (0.932). H3 fails: kind shares explain about a third of its spread
between players (R squared 0.342), not half. The kind-adjusted quality is **not adopted**: it is stable (0.961) but
more tied to kind shares (R squared 0.537), not less. The frozen player table stays as it is. Set A was not scored:
none of the pre-registered statistics can exist on it (see "Test set").

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
Hypotheses are read on the point value. A statistic that is not a finite number reads "undefined", never "holds" or
"fails" (added in review; no statistic here is undefined).

A note on H1's unit. The pre-registration ties H1 to the per-game Spearman of the lane's validation, but metric (1),
as written and as fixed before the run, is the odd/even half-season correlation. The protocol reading follows metric
(1) as pre-registered. The lead records the ambiguity as a PLAN.md D line.

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

## About the intervals

The pre-registration asked for 95 percent intervals from a bootstrap of games. Those ranges are kept below as
computed, but **they are not valid confidence intervals.** A game resample repeats some games and drops others. That
adds about one more sampling variance to every player's value, so every statistic here that depends on that noise
comes out shifted in the resamples. That covers the reliabilities, discrimination, stability, the spread between
players, R squared, and the differences between them. The range often does not even contain the estimate. Each
statistic in the JSON carries `interval_valid_as_confidence_interval: false` and a note saying this.

Post hoc, a coverage check on synthetic seasons with a known answer (100 seasons each, 150 players, 40 games; one
season tuned so split-half r is about 0.93, one about 0.42) gave, for a nominal 95 percent:

| statistic | game percentile (pre-registered) | game basic, 2 theta - q | bootstrap by player |
| --- | --- | --- | --- |
| split-half r, reliable / noisy | 0.01 / 0.23 | 0.01 / 0.43 | 0.96 / 0.91 |
| discrimination, reliable / noisy | 0.62 / 0.07 | 0.44 / 0.18 | 0.89 / 0.94 |
| stability, reliable / noisy | 0.01 / 0.00 | 0.09 / 0.14 | 0.96 / 0.97 |

The basic interval does not repair the game bootstrap, so it is not used. The bootstrap by player does: it resamples
the players over whom every one of these statistics is an average, takes each player's values as measured, and draws
the same players for every measure, so differences are paired. Its intervals are the valid ones below, labelled post
hoc. They read close to nominal, slightly under for discrimination in the reliable season (0.89). In the noisy season
the stability estimate itself runs low (mean 0.876 against a true 0.932), so a stability from noisy values errs
toward too low. Every pre-registered range had all 2000 resamples finite (finite share 1.0), so none is undefined.

## Development games (604 games, 518421 decisions, 12769 player-games)

Players: 344 with at least 300 decisions; 251 with 300 in each odd/even half; 240 in each chronological half (302
games per half; the first half ends on 2015-12-10).

Point estimates with the post hoc valid interval (bootstrap by player, 2000 resamples):

| measure | split-half r (1) | Spearman-Brown | discrimination (2) | stability (2) | R squared on kind shares (3) |
| --- | --- | --- | --- | --- | --- |
| target: on-court points per 100 | 0.420 [0.295, 0.519] | 0.591 [0.456, 0.683] | 0.571 [0.462, 0.648] | 0.788 [0.580, 0.991] | not computed |
| frozen mean quality | 0.927 [0.904, 0.946] | 0.962 [0.950, 0.972] | 0.956 [0.948, 0.963] | 0.932 [0.887, 0.961] | 0.342 [0.258, 0.434] |
| kind and zone adjusted (4) | 0.954 [0.939, 0.966] | 0.977 [0.969, 0.983] | 0.973 [0.968, 0.977] | 0.961 [0.936, 0.979] | 0.537 [0.478, 0.596] |
| kind adjusted only (ablation) | 0.941 [0.921, 0.958] | 0.970 [0.959, 0.978] | 0.964 [0.956, 0.969] | 0.945 [0.909, 0.969] | 0.419 [0.341, 0.497] |

The same point estimates with the pre-registered game-bootstrap range (not a valid confidence interval, see above):

| measure | split-half r | Spearman-Brown | discrimination | stability | R squared on kind shares |
| --- | --- | --- | --- | --- | --- |
| target | 0.420 (0.132 to 0.414) | 0.591 (0.234 to 0.586) | 0.571 (0.635 to 0.751) | 0.788 (0.187 to 0.600) | not computed |
| frozen | 0.927 (0.843 to 0.893) | 0.962 (0.915 to 0.943) | 0.956 (0.954 to 0.961) | 0.932 (0.841 to 0.902) | 0.342 (0.288 to 0.357) |
| adjusted | 0.954 (0.895 to 0.930) | 0.977 (0.944 to 0.964) | 0.973 (0.971 to 0.975) | 0.961 (0.902 to 0.941) | 0.537 (0.493 to 0.539) |
| kind only | 0.941 (0.868 to 0.912) | 0.970 (0.929 to 0.954) | 0.964 (0.961 to 0.967) | 0.945 (0.869 to 0.921) | 0.419 (0.369 to 0.424) |

- Split-half Spearman: target 0.364 [0.231, 0.477], frozen 0.918 [0.891, 0.939], adjusted 0.920 [0.891, 0.941], kind
  only 0.920 [0.890, 0.941] (post hoc valid intervals).
- Paired test, stability adjusted minus frozen: +0.029, post hoc valid interval [+0.009, +0.056] (pre-registered
  game range +0.032 to +0.069, which does not contain the estimate). Kind only minus frozen: +0.013 [-0.001, +0.029]
  (game range +0.011 to +0.035).
- Spread between players (standard deviation): target 6.01 points per 100 [5.36, 6.61]; frozen mean quality 0.057
  [0.052, 0.061]; adjusted 0.061 [0.056, 0.066]; kind only 0.060 [0.055, 0.065] (post hoc valid intervals).
- Stability by the paper's printed estimator (biased with two periods, not read): target 1.382, frozen 0.9995,
  adjusted 1.001, kind only 1.000. A value above 1 shows the bias.

Mean regret by kind and zone (development games; the cells of the adjustment):

| kind | rim | paint | mid | three | all zones |
| --- | --- | --- | --- | --- | --- |
| shot | -0.211 (6892) | -0.236 (17460) | -0.215 (12046) | -0.062 (17802) | -0.171 |
| pass | -0.395 (5164) | -0.211 (30526) | -0.205 (49730) | -0.451 (219040) | -0.386 |
| drive | none (0) | -0.163 (3078) | -0.256 (17916) | -0.601 (138767) | -0.554 |

Decision counts in brackets. No decision had a missing distance. The cell means are point estimates over every
development game.

## Hypotheses and protocol

| | statement | value | post hoc valid interval | pre-registered game range (not valid) | result |
| --- | --- | --- | --- | --- | --- |
| H1 | target split-half r below 0.2 | 0.420 | 0.295 to 0.519 | 0.132 to 0.414 | fails |
| H2 | frozen stability at least 0.5 | 0.932 | 0.887 to 0.961 | 0.841 to 0.902 | holds |
| H3 | frozen R squared on kind shares at least 0.5 | 0.342 | 0.258 to 0.434 | 0.288 to 0.357 | fails |
| adoption | adjusted stability at least 0.5, interval above 0.3, R squared below 0.2 | 0.961; R squared 0.537 | 0.936 to 0.979; 0.478 to 0.596 | 0.902 to 0.941 | not adopted (R squared) |

Results are read from the point values, as pre-registered. The JSON flag that says whether H1's range crosses 0.2
reads the invalid game range. It is named `interval_crosses_threshold_reads_invalid_interval`, renamed from
`interval_crosses_threshold` with the same values. The post hoc valid interval for H1 lies wholly above 0.2. The
adoption rule's interval test reads the pre-registered range (lower end 0.902); the valid lower end, 0.936, also
passes, and the rule fails on R squared either way.

Protocol, as pre-registered: H1 fails, so the target is reliable, and the failed verdict of the lane is read as
evidence against the metric. The kind-adjusted quality does not replace the frozen mean in the player table.

## What this means

- The frozen player mean measures something lasting: its stability (0.932) and discrimination (0.956) are high. About
  a third of its spread between players is the mix of shots, passes and drives (0.342). So the player table partly
  ranks role, but most of it is not role alone.
- Adjusting each decision by the average of its kind and zone does not remove the link to role; it raises it (0.537;
  0.419 with kind alone). This suggests that inside each kind and zone, the size of the regret also moves with a
  player's mix. So a simple centring cannot turn the table into a skill table, and the experiment does not adopt it.
- The target is reliable over a half season (r 0.420). The lane's validation, though, ranks players within one game.
  Post hoc, not pre-registered, and not used for any verdict: the players of the split had a mean of 18.0 qualifying
  games per half (median 18). The Spearman-Brown formula, stepped down to one game, gives a per-game reliability of
  about 0.039. A single game's on-court rating is almost all noise, so a per-game Spearman against it has little room
  to move whatever the metric does. That would meet H1's intent. It does not change the protocol's reading above.
- On-court rating is shared by every teammate on the floor, so part of its half-season reliability may be the team's
  offence rather than the player. This was not measured here.

## Test set (Set A): not scored

The pre-registration asks for the same statistics once on the D71 test set. They cannot exist there, so the test set
was not touched and the scoring ledger has no decisions-E1 entry. The reason is structural, so `e1-final` now refuses
before `score_once` is reached and reads no test data (tested in tests/test_research_decisions.py).

- Set A is the 10 games of a single date, 2015-12-05 (PLAN.md D77; tests/test_research_testset.py checks the date).
  On one date every team plays once, so every player has at most one game. No player can sit in both halves, so
  split-half reliability and stability have no player at all.
- Discrimination and R squared need 300 decisions in the set. Post hoc: the most decisions any player made in one of
  the 12769 development player-games is 229 (99th percentile 153). No player can reach 300 in one game.
- Scoring would read the test games' outcomes and use up the experiment's single ledger entry to produce only empty
  values. Set B (ACB) has no path to the frozen features (D80), as for xfg-E1.

A test-set check of this question would need a test set where players have several games, or a gate set for one
game; either is a new pre-registration.

## Proposed wording for reports/decisions.md

Not applied: the lane report is frozen output. Proposed text, to sit next to the unchanged failed verdict. Intervals
are the post hoc valid ones (bootstrap by player):

> A later check (decisions-E1, reports/research/decisions-E1.md) asked whether the target was too noisy to detect
> anything. Over half a season it is not: a player's on-court offensive rating is fairly reliable (split-half r 0.42,
> 95 percent interval 0.29 to 0.52). As set out in advance, the failed validation therefore counts as evidence
> against this decision metric. Post hoc, not pre-registered: stepped down to one game, the target's reliability is
> about 0.04, which would meet H1's intent; part of the half-season reliability may be team offence shared by
> teammates (not measured). The player means are stable (stability 0.93, interval 0.89 to 0.96), but about a third
> of their spread is the player's mix of shots, passes and drives (R squared 0.34, interval 0.26 to 0.43), so the
> table partly ranks role. Adjusting for kind and court zone did not remove that link, so the table is unchanged.

## What would change the verdict

A target with a per-game reliability well above 0.04, or a validation that pools each player over many games, would
let the relevance test mean more. decisions-E2 changes the question instead: it checks the chosen option's value
against the points that followed, decision by decision.
