# embeddings-E2: does a spatial NMF basis make the style embedding more reliable?

Written 2026-11-19. Pre-registration: docs/research/embeddings.md, "embeddings-E2", and its amendment (PLAN.md D118),
written and committed (f5bb5b51) before any code ran. Code: tools/research_embeddings_e2.py, building on
tools/research_embeddings.py (embeddings-E1). The development stages ran on commit bb888bc4; that commit is not
written inside the development json (its writer now records it, for future runs; this json is not rewritten). No stage
log was kept, so the timeline rests on the file times and on the cache, whose files are hash-checked against the gate
record. Every number below is copied from
reports/research/embeddings-E2-dev.json; a number marked "computed" is worked out from the copied ones. Intervals are
95 percent percentile intervals of 2000 resamples.

**Verdict: not adopted on development. Set A was not scored.** On the split-half check of the development games,
replacing the twelve zone shares with weights on two NMF spatial bases (shots and offensive court positions) raises
top-5 self-retrieval from 0.920 to 0.953. The paired difference is +0.032 [+0.006, +0.060]: the interval lies above
zero, but the difference is smaller than the pre-registered +0.05. The ceiling stated before any number held: a pass needed
0.970 or better. From one game (leave one game out, reported, deciding nothing) the NMF features retrieve slightly
worse than the frozen embedding (-0.016 [-0.028, -0.004]). The NMF bases and arm fits saw every development game,
including both halves of every split-half player and every left-out game, as the frozen fit did; any bias from that
favours NMF, so it cannot change "not adopted". The player bootstrap holds the fits fixed. As D118 fixes, Set A was not scored, and the scoring ledger
has no embeddings-E2 entry.

## What was tested

The frozen style embedding (PCA to 8 components of 19 tracking and shot features) describes where a player shoots and
stands with twelve coarse zone shares: six shot zones and a 3 by 2 grid of the attacked half. Miller et al. (2014)
describe shooting with weights on a non-negative spatial basis. The experiment asks whether such a basis makes the
embedding a more reliable fingerprint of a player.

- Arms. Frozen (the control): embeddings-E1's gated rebuild of the frozen embedding, unchanged. NMF: the six shot-zone
  shares become weights on a 10-map shot basis and the six court-zone shares weights on a 10-map court basis (27
  features); the frozen pipeline (median imputation, standardisation, PCA to 8) is refit on the same 399 player-seasons.
  Ablation: the shot basis only (23 features).
- Bases: 1 ft cells on the 47 by 50 half court; sklearn NMF with Kullback-Leibler loss, multiplicative updates,
  nndsvda initialisation, seed 0, fitted on the full-season development rates. Weights for any set of games come from
  that set's rates with the basis held fixed, from the same constant start for every row, divided by their sum. No
  smoothing: a stated deviation from Miller et al.
- Decision (D118, development games): split-half top-5 self-retrieval, NMF minus frozen, paired on the same queries,
  at least +0.05 with the lower end of its player-bootstrap interval above zero.
- Not part of the rule: leave one game out (D103's matched comparator, NMF minus frozen) and the pre-registered game
  bootstrap of the split-half rates, shown as a range and not read (D86).

The deciding interval resamples the split-half queries by player, 2000 resamples, with the pool held fixed.

## Gates

- The histogram pass read each of the 604 development games' frames once. Recounting the frozen six zone cells from
  its own mirrored positions, it equals embeddings-E1's gated frozen partials for all 12897 player-games (0 zone count
  mismatches, 0 rows on one side only). Every court histogram sums to its six zone counts (0 exceptions), and every
  shot histogram sums to the frozen attempts (0 exceptions; 101058 shots).
- The frozen arm here reproduces embeddings-E1's published development rates exactly: split-half 0.9203910614525139
  and leave one game out 0.47164236011819655 in both files.
- Both bases converged: the court basis in 290 iterations on 399 player-seasons, the shot basis in 120 on 397. The
  fixed-basis weights agree with sklearn's own transform to within 0.019 (court) and 0.018 (shot) in shares, with no
  row left unconverged.
- The 17 spent held-out games were never read, and Set A was never opened.

## Development result (604 games; split-half: 358 players, 716 queries)

Top-5 self-retrieval, split-half (player bootstrap; chance 0.014):

| retriever | rate |
| --- | --- |
| frozen embedding (control) | 0.920 [0.894, 0.944] |
| NMF, shot and court bases | 0.953 [0.932, 0.971] |
| ablation: shot basis only | 0.948 [0.927, 0.966] |
| position-only retriever | 0.042 [0.040, 0.044] |

| paired difference | split-half (player bootstrap) | split-half game range (not read) | one game, leave one game out (game bootstrap) |
| --- | --- | --- | --- |
| NMF minus frozen | +0.032 [+0.006, +0.060] | -0.003 to +0.063 | -0.016 [-0.028, -0.004] |
| shot basis only minus frozen | +0.028 [+0.001, +0.057] | -0.017 to +0.047 | -0.042 [-0.052, -0.031] |
| NMF minus shot basis only | +0.004 [-0.011, +0.018] | -0.011 to +0.042 | +0.026 [+0.016, +0.035] |

One game, leave one game out (10491 queries from 391 players, 604 games; pool 397 to 399): frozen 0.472 [0.460,
0.482], NMF 0.455 [0.445, 0.465], shot basis only 0.430 [0.420, 0.440], position-only 0.037 [0.037, 0.037].

- By the rule, NMF is not adopted on development: +0.032 is below +0.05, though its interval lies above zero.
- The pre-registered game bootstrap rebuilds both halves from resampled games; as in embeddings-E1 (D86), that adds
  sampling noise to every player's value, and its range for the frozen arm (0.793 to 0.848) sits wholly below the
  estimate (0.920). It is shown and not read.

## What this means

- Over half a season, a finer spatial description helps a little. The shot basis alone also gains, +0.028 [+0.001,
  +0.057]; the court basis adds +0.004 [-0.011, +0.018] on top of it.
- From one game it hurts. A single game has a handful of shots, and ten weights fitted to them are noisier than six
  zone shares. The shot basis alone loses 0.042 against the frozen embedding on one game; the court basis, built from
  thousands of frames even in one game, wins back 0.026.
- The gain the pre-registration asked for (+0.05) is larger than what was left: the frozen embedding already finds the
  same player in 92 percent of halves, so the most any change could add was 0.080 (computed), and the 19 raw features
  without PCA had reached only 0.950 (embeddings-E1).
- The frozen embedding stays as it is. A coach-facing map of where a player shoots could still use these bases as a
  description, but this experiment does not show that they make the "plays like" list more reliable at the size asked.

## Limits

- The transform's convergence record was kept for the full-season fit rows (no row unconverged), not for the
  half-season and one-game rows of the split-half and leave-one-game-out stages. For future runs the code now records
  it from every transform and writes it with each stage.
- One development attempt (player 203921, 11.10.2015.NYK.at.TOR, cell 1353) fell in a shot cell where every map of
  the shot basis is zero, which would give a non-finite weight row for any set of games holding it. That player has
  89 development possessions, so is in no fit, query or pool row, and no number above is affected. No court cell was
  affected. For future runs the code now refuses such a row instead of imputing it.

## Set A

Not scored (D118, as D109 for gravity-E1): under D118, Set A could only check an adoption, and there is none to check.
Set A was read once before, by embeddings-E1, whose frozen arm is this experiment's control.
