# Research dossier: embeddings (player style)

- Lane: embeddings (analytics/embeddings, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: descriptive and only weakly validated. PCA on 19 tracking and shot features passes its 0.60 gate on 5-NN
  position accuracy (0.768), but there is no held-out score by design, and position labels are a coarse check of
  style.

## 1. What we built

- Question: which players move and shoot alike (reports/embeddings.md).
- Data: 448 player-seasons from the 604 training games of the 2015-16 SportVU season, 399 with at least 300
  possessions; tracking and shot features only, since NBA.com aggregates were skipped (reports/embeddings.md, Inputs).
- Features (19): speed, distance per possession, six court-zone shares, ball share, teammate and ball distances, six
  shot-zone shares, three-point share, attempts per possession (reports/embeddings.md, Inputs).
- Model: median imputation, standardisation, PCA to 8 components (explained variance 0.31, 0.24, 0.10, 0.09, 0.06,
  0.06, 0.04, 0.03); the autoencoder was not trained because PCA passed the gate (reports/embeddings.md, Candidates).
- Numbers: leave-one-out 5-NN position accuracy 0.768 [0.720, 0.809] against a majority baseline of 0.413 [0.395,
  0.461] (G, F, C; 397 labelled player-seasons). Held-out: absent. Next-season R squared: n/a, no next season of
  tracking (reports/embeddings.md; reports/results.md, embeddings table).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Miller, A., Bornn, L., Adams, R. and Goldsberry, K. (2014). Factorized Point Process Intensities: A Spatial Analysis of Professional Basketball. Proceedings of the 31st International Conference on Machine Learning, PMLR 32(1): 235-243. | [proceedings page](https://proceedings.mlr.press/v32/miller14.html), [arXiv:1401.0942](https://arxiv.org/abs/1401.0942) | A low-dimensional basis of player shooting habits by non-negative matrix factorization of smoothed shot intensities. We use coarse zone shares plus movement features and a linear PCA instead of a spatial point-process basis. |
| Franks, A., D'Amour, A., Cervone, D. and Bornn, L. (2016). Meta-analytics: tools for understanding the statistical properties of sports metrics. Journal of Quantitative Analysis in Sports 12(4). | [arXiv:1609.09830](https://arxiv.org/abs/1609.09830), [doi:10.1515/jqas-2016-0098](https://doi.org/10.1515/jqas-2016-0098) | Discrimination (share of between-player variance that is not sampling noise, game bootstrap) and stability (share that persists within a player). These are the checks a descriptive metric can pass without a supervised target; we have not computed them. |
| Yamada, K. and Fujii, K. (2024). Offensive Lineup Analysis in Basketball with Clustering Players Based on Shooting Style and Offensive Role. arXiv preprint. | [arXiv:2403.13821](https://arxiv.org/abs/2403.13821) | Shooting-style clusters from the same 2015-16 SportVU season (630 games, 41160 shots, Section 2.1), with Wasserstein distances between shot distributions. Closest data to ours; they cluster and link styles to lineup efficiency, we embed and list neighbours. |
| Golovkine, S. and Gunning, E. (2025, revised 2026). Is Stephen Curry really a guard? A new perspective on player typologies using functional data analysis. arXiv preprint. | [arXiv:2504.21761](https://arxiv.org/abs/2504.21761) | Functional PCA of shot charts (2018-19 to 2023-24, 173 players) and k-medoids clusters; their clusters barely agree with NBA position labels. It questions position accuracy as the validation target we use. |

## 3. Benchmark

| ours | published | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| 5-NN position accuracy 0.768 [0.720, 0.809], 3 classes, majority 0.413 (reports/embeddings.md) | Adjusted Rand Index between NBA position groups and shot-chart clusters 0.05 (equal weights) and 0.04 (variance weights) (Golovkine and Gunning, Section 4.2, v4) | Golovkine and Gunning | Metric (k-NN accuracy vs clustering ARI), features (shots only vs shots plus movement), labels (3 vs 5 position groups), seasons (2015-16 vs 2018-19 to 2023-24) | not directly comparable: different metric and label set |

No paper we found reports a k-NN position accuracy, a held-out score, or a stability score for a tracking-based style
embedding, so there is no published number to put beside ours on the same metric.

## 4. Measured weaknesses (from our reports only)

- No held-out score: held-out games are dropped before any feature, so the embedding exists for training games only
  (reports/embeddings.md, Stage 8).
- The only validation is position accuracy, and the gate (0.60) decided the model: PCA passed, so the autoencoder was
  never trained and nothing was compared (reports/embeddings.md, Decision).
- No next-season check: one season of tracking, so the next-season R squared is n/a (reports/embeddings.md).
- Embedding and scaler are fit on all sufficient rows, unsupervised, so only the k-NN vote is out of sample
  (reports/embeddings.md, Candidates protocol).
- 49 of 448 player-seasons are under 300 possessions and are never embedded (reports/embeddings.md).

## 5. Recent work that targets these weaknesses

- Validation without a label: Franks et al. (2016) define discrimination and stability by a game bootstrap within a
  season. With one season, a split of each player's games into two disjoint halves gives the same idea: a style
  embedding is reliable if a player's two halves are each other's nearest neighbours. This attacks the first two
  weaknesses: it is a score on games the embedding did not see, and it does not depend on position labels.
- Position labels as a weak target: Golovkine and Gunning (2025, revised 2026) find almost no agreement (ARI 0.05)
  between shot-chart style and NBA positions, which says a high position accuracy is neither necessary nor sufficient
  for a good style embedding.
- Finer spatial representation: Miller et al. (2014) and Yamada and Fujii (2024) represent shooting as a spatial
  distribution rather than six zone shares. Whether that gains reliability is testable with the same split-half score.

## 6. Pre-registered experiments

Shared rules: development games are the 2015-16 training games that D71 leaves for development. The D71 test set
(being fixed separately; for tracking it may be games carved from 2015-16 if no new tracking source serves the lane)
is scored once per experiment. Intervals: bootstrap by game, 2000 resamples, 95 percent percentile intervals. Frames
are read one game at a time under the heavy-job lock (CLAUDE.md rule 7). Nothing here fits under `export/` or `web/`.

### embeddings-E1: a held-out style score, split-half and test-set self-retrieval

- Hypothesis: the frozen PCA embedding identifies a player's style across disjoint games. Built from one set of games,
  a player's embedding has their own embedding from other games among its 5 nearest neighbours far more often than
  chance.
- Baseline: the frozen pipeline (imputer, scaler, PCA to 8 components) refit with frozen code on the development
  games. Comparators: the standardised 19 raw features without PCA, and a position-only retriever (a random player of
  the same position), which is what the current validation would reward.
- Metric: top-5 self-retrieval rate. Development: each player's development games split into two halves by
  alternating game date, features built per half, embedded with the development fit, retrieved half to half. Test: a
  player's D71 test-set embedding retrieves among all development embeddings. Secondary: Franks et al. discrimination
  per component, by a game bootstrap. Why: it needs no label and it scores games the embedding never saw.
- Adoption threshold: the embedding is reported as validated on held-out games if the test-set top-5 self-retrieval
  rate has the lower end of its 95 percent interval above 0.25 and above the position-only retriever's upper end.
  Reason for 0.25: with about 400 players, chance for top-5 is about 0.0125, so 0.25 is 20 times chance, and a
  neighbour list that finds the same player only rarely is not trustworthy for a scout.
- Paired test: bootstrap by game (resample games, rebuild per-player features, re-embed with the fixed fit), 2000
  resamples, 95 percent interval of the difference between the embedding and each comparator.
- Ablation: movement features only (drop the eight shot features) against shot features only, to show which block
  carries identity.
- Data: development games for the fit and the split-half check; D71 test set scored once, only for players with at
  least 300 possessions in it (if few qualify, the interval will say so).
- Compute: per-game, per-player feature sums need one pass over the development frames, one game at a time; estimate 2
  to 4 hours on the 16 GB Mac, CPU, under the heavy-job lock (the load lane read 482 million rows in one pass,
  reports/load.md). Retrieval and bootstraps on the sums take minutes.
- Risks: a player's role inside one season is partly set by their team, so retrieval may find teammates or system
  rather than style; a small test set leaves few qualifying players; the embedding fit uses all development rows,
  which is fine for an unsupervised fit but must be stated.
- What a coach gains: a "plays like" list with a tested hit rate on unseen games, instead of a list checked only
  against position labels.
- Backlog: expected gain 4 (it gives the lane its first held-out score and replaces a weak gate); effort 3 (one heavy
  pass over frames to build per-game sums).

#### Amendment (PLAN.md D103, 2026-11-18, written before any development number or run)

Why: the test above scores only players with at least 300 possessions in the test set. The D71 test set for tracking
(Set A, PLAN.md D77) is the 10 games of one date, so no player has more than one game in it, and no development game
gives any player more than 268 possessions (the most possessions in any of the 604 development games, both teams). The
test set would be empty. D103 keeps the development stage as written above and replaces the test with a one-game test,
fixed before any development number exists.

- One-game test (D103): a Set A player with at least 50 possessions in their test game gets a one-game embedding (the
  19 features of that game alone, through the development fit: imputer, scaler, PCA). It retrieves among the
  full-season development embeddings (the 399 sufficient player-seasons of the frozen fit); metric top-5
  self-retrieval. Interval: a bootstrap of the 10 test games, 2000 resamples. The one-game test passes if the lower
  end of its 95 percent interval is above 0.10 and above the upper end of the position-only retriever's interval.
- What one game means. The test embedding is built from one game, so it is much noisier than a full-season embedding.
  The test asks whether a single game already carries a player's style, which is a harder and different question
  from the split-half check. Its interval resamples the 10 games, so the players of one game move together. A
  duplicated game leaves its players' one-game features unchanged, so this bootstrap only resamples the queries, by
  game; it does not add noise to any player's value. With 10 games the interval is wide, and the result is read as
  such.
- Matched development comparator (D103), computed and reported before the score, with no change to the threshold:
  leave one game out. For each development game g, every player's rest-of-season features (all development games but
  g) are embedded with the development fit; the pool is the players with at least 300 rest-of-season possessions. A
  player with at least 50 possessions in g and a pool entry queries with their one-game embedding from g. Interval: a
  bootstrap of the development games, by game, 2000 resamples. The development fit itself saw g; this is stated, not
  repaired.

Commitments fixed with this amendment (code: tools/research_embeddings.py), so that nothing below is chosen after a
number exists:

1. Queries on the test set are the Set A players with at least 50 possessions in their test game who have a
   development embedding (one of the 399). A player with no development embedding cannot retrieve themself; they are
   counted and reported, not scored.
2. Retrieval is Euclidean in the embedding space, as the lane's neighbour table is. A query hits when fewer than 5
   pool entries are strictly closer than its own entry.
3. The position-only retriever returns 5 random pool entries of the query's position, from the frozen positions table
   (G, F, C). Its hit rate is computed exactly: for a labelled query, min(5, n) / n, with n the pool entries of that
   position (their own included); for an unlabelled query, 5 over the pool size. Its interval comes from the same game
   resamples as the embedding's.
4. Comparators and ablation are scored on the same queries: the 19 standardised features without PCA (the development
   imputer and scaler), and movement-only (the 11 tracking features) and shot-only (the 8 shot features), each with
   the frozen pipeline (median imputation, standardisation, PCA to 8 components) fitted on the development games for
   its block.
5. Development split-half: each player's development games are sorted by date and alternated into two halves; a
   player takes part with at least 300 possessions in each half (the lane's own sufficiency rule); each half
   queries the other half's pool, both directions, and the rate is the mean over the 2P queries.
6. Intervals of the split-half rates. The pre-registered game bootstrap (resample games, rebuild the halves'
   features, re-embed with the fixed fit) is computed. decisions-E1 showed (PLAN.md D86) that a game resample adds
   sampling noise to every multi-game player value, so this range is expected to sit below the estimate and is not
   read as a confidence interval. Fixed now as the valid interval: a bootstrap of the queries by player, 2000
   resamples, with the pool held fixed.
7. Franks discrimination per component: a player's full-season component value against its variance over 2000 game
   resamples (rebuild, re-embed with the fixed fit). Its interval: a bootstrap by player, 2000 resamples (the valid
   method in D86). No verdict reads it.
8. The development stage has no adoption rule; the one-game test is the only rule, as D103 fixes it.

### embeddings-E2: a spatial shot basis by non-negative matrix factorization

- Hypothesis: replacing the six shot-zone shares and six court-zone shares with weights on a non-negative spatial basis
  of shot and touch locations (Miller et al. 2014, about 10 basis maps on a 1 ft grid, fitted on development games)
  raises top-5 self-retrieval from E1's design by at least 0.05.
- Baseline: the frozen PCA embedding on the 19 features, refit with frozen code on development games, scored with the
  E1 metric.
- Metric: E1's top-5 self-retrieval rate (split-half on development, test-set retrieval once). Why: the measured
  weakness is reliability, and this is the reliability score.
- Adoption threshold: paired difference (NMF minus frozen) at least +0.05 with its 95 percent interval above zero on
  the D71 test set.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval of the paired difference.
- Ablation: NMF basis on shots only against shots plus off-ball court positions.
- Data: development games for the basis and the PCA; D71 test set scored once. Runs only after E1, which builds the
  metric.
- Compute: the per-player location histograms come from the same pass as E1; NMF on about 400 by 2000 matrices takes
  minutes on the 16 GB Mac, CPU.
- Risks: finer maps need more shots per player, and half a season may be too few for bench players; NMF has no unique
  solution, so the basis is fitted with fixed seeds and reported with that caveat; gains may be within noise.
- What a coach gains: style neighbours that respect where on the floor a player actually shoots and stands, shown as
  readable basis maps.
- Backlog: expected gain 2 (plausible but unproven gain in reliability); effort 2 (reuses E1's pass and metric).

## 7. Sources checked (all fetched 2026-11-16)

- Miller, Bornn, Adams and Goldsberry 2014: [arXiv abstract](https://arxiv.org/abs/1401.0942), [PMLR proceedings page](https://proceedings.mlr.press/v32/miller14.html).
- Franks, D'Amour, Cervone and Bornn 2016: [arXiv abstract](https://arxiv.org/abs/1609.09830), [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1609.09830) (Sections 2 and 3), [Crossref record](https://api.crossref.org/works?query.bibliographic=Meta-analytics+tools+for+understanding+the+statistical+properties+of+sports+metrics&rows=2) for the journal, volume and DOI.
- Yamada and Fujii 2024: [arXiv abstract](https://arxiv.org/abs/2403.13821), [PDF](https://arxiv.org/pdf/2403.13821) (Section 2.1, data).
- Golovkine and Gunning 2025, revised 2026: [arXiv abstract](https://arxiv.org/abs/2504.21761), [PDF v4](https://arxiv.org/pdf/2504.21761) (data section and the ARI passage). The abstract search record lists it in JQAS 2026, but the arXiv page lists no journal reference, so it is cited as a preprint.
