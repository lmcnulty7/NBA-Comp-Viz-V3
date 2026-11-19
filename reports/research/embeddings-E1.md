# embeddings-E1: does the style embedding find the same player in other games?

Written 2026-11-19 (UTC); the test section added the same day after the single score (04:47 UTC, the ledger's time). Pre-registration:
docs/research/embeddings.md, "embeddings-E1", and its amendment (PLAN.md D103), written and committed (131d0ec3)
before any code ran. Code: tools/research_embeddings.py, committed (d848e20a) before the frame pass; the score ran on
57fcc7b3 with a clean tree. Development numbers are copied from reports/research/embeddings-E1-dev.json; test numbers
from reports/research/embeddings-E1-final.json, whose summary is the one embeddings-E1 entry of
reports/research/scoring_ledger.json.

**Verdict: passes.** On the 10 Set A games, scored once, the frozen PCA embedding built from a single game finds the
same player's full-season development embedding among its 5 nearest neighbours 47.4 percent of the time (0.474
[0.384, 0.558], 171 players; chance 0.0125; the position-only retriever 0.037 [0.036, 0.038]). The lower end, 0.384,
is above 0.10 and above 0.038, so the frozen embedding passed the D103 one-game test (self-retrieval). This is the
embeddings lane's first held-out score. The pre-score sensitivity interval agrees (0.476 [0.372, 0.580]). The 19
standardised features without PCA retrieve better on the same test players (0.561 [0.491, 0.631]; paired difference
-0.088 [-0.140, -0.040]), as they did on the development games, and the movement features carry most of a player's
identity.

The claim is narrow: from one held-out game, the frozen embedding finds the same player far above chance. It is not
a test of what the style map means, of whether a player's listed neighbours play alike, or of any one player.

## What is tested

The embeddings lane had no held-out score before this experiment: the frozen check is 5-NN position accuracy on the fitted players. This
experiment asks whether the embedding identifies a player's style across games it was not built from, with no labels.

- Hypothesis (dossier): a player's embedding has their own embedding from other games among its 5 nearest neighbours
  far more often than chance.
- Development (as pre-registered): each player's development games are sorted by date and alternated into two
  halves; each half's features are embedded with the development fit and retrieve the other half. Comparators: the 19
  standardised features without PCA, and a position-only retriever. Ablation: movement features only against shot
  features only.
- The one-game test (D103, which replaced the original test because Set A is one date and no player can reach 300
  possessions in one game): a Set A player with at least 50 possessions in their test game retrieves among the 399
  full-season development embeddings. It passes if the lower end of its 95 percent interval (bootstrap of the 10 test
  games) is above 0.10 and above the upper end of the position-only retriever's interval.
- Matched development comparator (D103): leave one game out. Each development player's one-game embedding retrieves
  their rest-of-season embedding among every player's rest-of-season embedding. The development fit saw every
  development game, including the left-out one; stated, not repaired.

The amendment's commitments (queries, the retrieval rule, the exact position retriever, the comparators, the halves,
and every interval) are in the dossier, dated before any run.

## Gate (the baseline)

The frozen per-game partial sums (analytics.embeddings.features.tracking_partials, unchanged) were computed for all
604 development games, one game of frames at a time (130 s under the lock). Then:

- The frozen functions (combine_partials, player_season_features) rebuilt the features of all 448 player-seasons.
  They equal the stored frozen table of tools.frozen_models (D45) exactly (largest difference 0.0), with the same
  possession counts and the same 49 insufficient rows.
- The experiment's own path (sums of partials, so any subset of games can be rebuilt) equals the frozen path to
  7e-14.
- The frozen pipeline (median imputation, standardisation, PCA to 8) refit on the 399 sufficient rows equals the
  stored embedding exactly (0.0), and its 5-NN position accuracy is 0.768, as reports/embeddings.md prints.

The 17 spent held-out games were never read. Shot rows were read with a filter on the development games, so no other
game's row was loaded. Nothing was written under DATA_ROOT or reports/ except the research results.

## Development games (604 games; 358 players in the split-half check; 10491 one-game queries)

Top-5 self-retrieval rates. Split-half: 358 players with at least 300 possessions in each half, 716 queries (both
directions), chance 5/358 = 0.014. One game: every development game, players with at least 50 possessions in it and
at least 300 in the rest of the season; 10491 queries from 391 players; pool 397 to 399; chance 0.0125.

| retriever | split-half (player bootstrap) | split-half, pre-registered game range (not valid) | one game, leave one game out (game bootstrap) |
| --- | --- | --- | --- |
| PCA embedding (frozen) | 0.920 [0.894, 0.944] | 0.793 to 0.848 | 0.472 [0.460, 0.482] |
| 19 standardised features, no PCA | 0.950 [0.929, 0.968] | 0.824 to 0.876 | 0.544 [0.532, 0.555] |
| movement block only (11) | 0.937 [0.912, 0.959] | 0.771 to 0.837 | 0.443 [0.431, 0.455] |
| shot block only (8) | 0.628 [0.582, 0.675] | 0.376 to 0.459 | 0.152 [0.145, 0.159] |
| position-only retriever | 0.042 [0.040, 0.044] | 0.042 to 0.042 | 0.037 [0.037, 0.037] |

| paired difference | split-half (player bootstrap) | split-half game range (not valid) | one game (game bootstrap) |
| --- | --- | --- | --- |
| PCA minus 19 standardised features, no PCA | -0.029 [-0.045, -0.015] | -0.046 to -0.010 | -0.073 [-0.080, -0.066] |
| PCA minus movement block only (11) | -0.017 [-0.041, +0.006] | -0.015 to +0.052 | +0.029 [+0.017, +0.039] |
| PCA minus shot block only (8) | +0.292 [+0.250, +0.335] | +0.363 to +0.444 | +0.319 [+0.309, +0.330] |
| PCA minus position-only retriever | +0.878 [+0.852, +0.902] | +0.751 to +0.806 | +0.435 [+0.423, +0.445] |

About the intervals, as fixed in the amendment (commitment 6):
- The split-half check uses a bootstrap of the queries by player, with the pool held fixed.
- The pre-registered game bootstrap rebuilds both halves from resampled games. As decisions-E1 found (D86), that adds
  sampling noise to every player's value. Here the range sits wholly below the estimate (0.793 to 0.848 against
  0.920), so it is shown but not read.
- The one-game intervals resample the queries by game. A repeated game leaves its players' one-game features
  unchanged, so this bootstrap adds no noise. It has 604 games here, so it is narrow.

Franks discrimination per component of the full-season embedding (399 players; sampling variance from 2000 game
resamples; interval by player bootstrap):

| component | explained variance | discrimination [player bootstrap] |
| --- | --- | --- |
| 1 | 0.306 | 0.989 [0.987, 0.991] |
| 2 | 0.235 | 0.983 [0.979, 0.985] |
| 3 | 0.105 | 0.971 [0.965, 0.975] |
| 4 | 0.086 | 0.939 [0.925, 0.949] |
| 5 | 0.063 | 0.938 [0.925, 0.948] |
| 6 | 0.059 | 0.911 [0.892, 0.926] |
| 7 | 0.039 | 0.798 [0.732, 0.842] |
| 8 | 0.032 | 0.812 [0.767, 0.848] |

## What this means (development only)

- The embedding is a reliable fingerprint over half a season: 92 percent top-5 self-retrieval against 1.4 percent by
  chance and 4.2 percent for the position-only retriever. Position labels explain almost none of it.
- One game carries much of the style already: 47 percent from a single game, against 1.25 percent by chance.
- PCA loses some identity. The 19 standardised features retrieve better than the 8 components in both checks
  (by 0.029 on halves, 0.073 on one game). The embedding is a compression with a measurable cost, not a gain.
- The movement block (speed, distance, court zones, ball share, spacing) carries the identity. Movement alone matches
  the full embedding on halves (0.937 against 0.920) and nearly on one game (0.443 against 0.472). The shot block
  alone is much weaker (0.628 and 0.152). Its zone shares are noisy at small attempt counts.
- Not measured, the dossier's risk: some of what is retrieved may be the team's system rather than the player.
  Teammates share the context but are different pool entries, so a hit is the player's own row, not a teammate's.
- Every component except the last two has discrimination above 0.9: the per-component spread between players is
  mostly not sampling noise.

## Before the score: commitments

These held for the one-game test and were fixed before any test data was read (when they were written, the Set A
data had not been opened by this experiment and the scoring ledger had no embeddings-E1 entry).

1. The rule is D103's, unchanged: queries with at least 50 possessions in their Set A game and a development
   embedding. The pass needs the lower 95 percent end (bootstrap of the 10 test games, 2000 resamples) above 0.10,
   and above the upper end of the position-only retriever's interval from the same resamples. The development one-game
   rate (0.472) does not move the threshold.
2. One game per player: each test query is one player-game. The 10 games are resampled as clusters, so the players of
   a game move together. A repeated game leaves its players' features unchanged, so the interval has no added noise.
   With 10 games it is wide. Players without a development embedding, and those under 50 possessions, are counted and
   reported, not scored.
3. Reported beside the rule, read by no rule: the raw-feature, movement-only and shot-only rates, and the paired
   differences against the embedding.
4. Pre-score sensitivity (report only), kept by the lead before the score: a 95 percent t interval on the 10 per-game
   mean hit rates, beside the bootstrap, as D80 did for xfg-E1, because a percentile bootstrap of 10 games
   under-covers. Its centre is the mean of the per-game means; the bootstrap's value is the pooled per-query mean, so
   the two centres differ when games have different numbers of queries. It reads no rule; if it and the bootstrap
   disagree on the rule, the report says so. Code: game_mean_t_interval, added after the development result and
   before any test read.
5. Set B (ACB) is not scored: the frozen features need SportVU frames and NBA shot charts, which the ACB data does not
   have (as D80 found for xfg-E1).
6. Refusals, decided before the score (pre-score review). Before any value is computed, and without reading an
   outcome column (shots are read with the shooter, zone and shot-type columns only), the scorer refuses unless every
   test game has frames, possessions, a roster and at least one shot row, and no shot row is flagged held out. A
   summary with no qualifying query, or with any non-finite number (a t interval over fewer than 2 games, for one),
   is returned as a refused verdict with its reason. A refusal still uses up the one ledger entry, but the ledger
   never holds a NaN.
7. Provenance: the score runs only on a clean working tree. The result records the commit, the hashes of the gate's
   files and the hash of the development record. The final file nests the scored summary under "scored" beside its
   setup (the gate record and the D103 rule). The scorer was rehearsed through score_once on a synthetic root, with a
   temporary test set and ledger (tests/test_research_embeddings.py).
8. The test pool and fit are the frozen fit, unchanged. The test games are on 2015-12-05, inside the development
   season. The full-season development embeddings include games before and after that date, but not the test games.


## Test set (Set A, the 10 games of 2015-12-05, scored once)

The lead gave the go after the pre-score review. e1-final ran once, on 57fcc7b3, under the lock. No refusal
fired. Test players: 204 player-games in the 10 games, 171 with at least 50 possessions; all 171 have a development
embedding, so 171 queries from 10 games, against the 399 full-season development embeddings (chance 5/399 = 0.0125).

| retriever | test, top-5 self-retrieval (bootstrap of the 10 games) | pre-score sensitivity t interval (mean of per-game means, report only) | development, one game (for comparison) |
| --- | --- | --- | --- |
| PCA embedding (frozen) | 0.474 [0.384, 0.558] | 0.476 [0.372, 0.580] | 0.472 [0.460, 0.482] |
| 19 standardised features, no PCA | 0.561 [0.491, 0.631] | 0.562 [0.477, 0.647] | 0.544 [0.532, 0.555] |
| movement block only (11) | 0.456 [0.349, 0.544] | 0.458 [0.338, 0.578] | 0.443 [0.431, 0.455] |
| shot block only (8) | 0.158 [0.118, 0.202] | 0.158 [0.104, 0.212] | 0.152 [0.145, 0.159] |
| position-only retriever | 0.037 [0.036, 0.038] | 0.037 [0.036, 0.038] | 0.037 [0.037, 0.037] |

| paired difference (test) | bootstrap of the 10 games | pre-score sensitivity t interval |
| --- | --- | --- |
| PCA minus 19 standardised features | -0.088 [-0.140, -0.040] | -0.086 [-0.149, -0.022] |
| PCA minus movement block | +0.018 [-0.066, +0.108] | +0.018 [-0.090, +0.125] |
| PCA minus shot block | +0.316 [+0.229, +0.405] | +0.318 [+0.213, +0.423] |
| PCA minus position-only retriever | +0.437 [+0.347, +0.521] | +0.439 [+0.335, +0.543] |

The rule (D103), read exactly: the lower end of the embedding's 95 percent interval is 0.384. That is above 0.10 and
above 0.038, the upper end of the position-only retriever's interval, so the one-game test passes. The pre-score
sensitivity t interval reads no rule. Its lower end, 0.372, is on the same side of both numbers, so the two agree.
The bootstrap value (0.474) is the pooled mean over the 171 queries, and the t interval's centre (0.476) is the mean
of the 10 per-game means.

## What the test means

- Self-retrieval now has a tested hit rate on games the fit never saw. From one game, the player's own
  season row is in the top 5 about 47 percent of the time, against about 1 percent by chance and 4 percent for a
  same-position guess. The test rate is close to the development estimate. With 10 games, the test cannot rule out a
  gap of up to about 0.09 (computed: the development estimate 0.472 minus the test's lower end 0.384 is 0.088, and the
  test's upper end 0.558 minus 0.472 is 0.086).
- Position labels, the lane's only check until now, explain almost none of this, as Golovkine and Gunning's
  finding (dossier, section 5) suggests.
- PCA costs identity on the test set too: the standardised features without PCA retrieve better by about 0.09.
  Whether to show neighbours in the 19-feature space is a lane decision; this experiment tested the frozen embedding
  (it passed the D103 one-game test), not the alternative.
- Movement features carry the identity, and shot shares alone are weak from one game.
- The test is one date of 10 games, so the interval is wide (0.384 to 0.558); the pass does not depend on that width,
  since the lower end is far above both thresholds. Set B was not scored (commitment 5).
