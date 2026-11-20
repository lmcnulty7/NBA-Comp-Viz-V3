# Research dossier: xfg (shot quality)

Lane: xfg, the frozen `hgb_contest_prior` model and the location-only baseline. Written 2026-11-16 (PLAN.md D70).

Verdict: the model works and beats the best published 2015-16 SportVU number we could check; the reported held-out
calibration gap is about the size that sampling noise alone produces at 2939 shots; what is really missing is a
per-shot interval, which the site rule asks for.

## 1. What we built

- Model: gradient boosting (sklearn, 100 trees, depth 3) on location and contest features plus a shooter-by-zone
  prior, isotonic calibration on inner out-of-fold scores, clipped to [0.02, 0.98] (reports/xfg.md, Protocol).
- Data: 101853 training attempts, 607 games of 2015-16 SportVU, 5 folds by game; 2939 held-out attempts, 18 games.
- Out of fold: log loss 0.6294 [0.6277, 0.6316], Brier 0.2209, ECE 0.0029, slope 1.000. Held out: log loss 0.6232
  [0.6132, 0.6327], Brier 0.2187, ECE 0.0189 [0.0143, 0.0347], slope 1.090 [0.985, 1.198] (reports/xfg.md).
- Location-only baseline: 0.6665 out of fold, 0.6735 held out (reports/xfg_baseline.md); contest features gain
  0.037 out of fold and 0.050 held out (reports/results.md, xfg).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Chang, Maheswaran, Su, Kwok, Levy, Wexler, Squire. Quantifying Shot Quality in the NBA. MIT Sloan Sports Analytics Conference 2014 (research paper competition). | Archived conference PDF: http://web.archive.org/web/20140311083800/http://www.sloansportsconference.com/wp-content/uploads/2014/02/2014-SSAC-Quantifying-Shot-Quality-in-the-NBA.pdf (the original sloansportsconference.com URL now returns 404) | Shot quality as the make probability of a league-average shooter from SportVU features (location, defender distance, more), scored by MSE (Brier) under cross-validation. We use the same idea and the same feature family, but our model adds a shooter prior, so it is not a league-average-player quantity, and we select by log loss with folds grouped by game. |
| Zadrozny, Elkan. Transforming classifier scores into accurate multiclass probability estimates. KDD 2002, pp. 694-699. | https://doi.org/10.1145/775047.775151 (full text: https://cseweb.ucsd.edu/~elkan/kdd2002.pdf) | Isotonic regression (pool adjacent violators) as a calibrator for any ranking score; the paper notes PAV "overfits slightly the training data" but beats sigmoid fitting (Section 5 results). Ours fits it on inner out-of-fold scores grouped by game and adds an ad hoc clip to [0.02, 0.98]. |
| Vovk, Petej. Venn-Abers predictors. UAI 2014. | https://arxiv.org/abs/1211.0025 | Venn predictors built on isotonic regression: each test object gets two probabilities (p0, p1), and under IID data one of them, selected by the true label, is perfectly calibrated (Theorem 1). Not used yet; it is the basis of xfg-E1. |
| Vovk, Petej, Fedorova. Large-scale probabilistic predictors with and without guarantees of validity. NeurIPS 2015. | https://arxiv.org/abs/1511.00213 | Inductive and cross Venn-Abers predictors (IVAP, CVAP) and the log-loss minimax merge p = GM(p1) / (GM(1 - p0) + GM(p1)) (Section 4). Section 5.2 shows plain isotonic regression can give infinite log loss when a test score falls outside the calibration scores; our clip is the kind of ad hoc fix the authors decline to use. Not used yet; xfg-E1. |

## 3. Benchmark

Brier skill is 1 minus Brier over the constant model's Brier, computed here from numbers read in each source.

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| OOF log loss 0.6294, Brier 0.2209 (hgb_contest_prior, reports/xfg.md); held out 0.6232, 0.2187 | GNN trained with Brier loss: log loss 0.6522, Brier 0.2287; their gradient-boosted classifier: 0.6696, 0.2384 (Table 1, p. 81) | Schmid, Schöpf, Kolbinger 2025 | Same season (2015-16) and the same SportVU source. They test on the 2015-16 playoffs after training on the regular season; we use 5 folds grouped by game in the regular season. They have 32272 shots, we have 101853. Their inputs add defender height and arm span and the second defender; ours add a shooter prior. | roughly comparable: same data source, season and metrics, different split and sample, and a playoff test may be harder |
| OOF ECE 0.0029, 10 bins | ECE 0.0263 for the GNN trained with Brier loss, 30 bins (Table 1) | Schmid, Schöpf, Kolbinger 2025 | Bin count and sample size differ, and binned ECE grows as the sample shrinks and the bin count rises (section 4) | not directly comparable: ECE depends on bins and sample size |
| OOF Brier 0.2209, Brier skill 0.107 (hgb with shooter prior); contest_logistic 0.2316, skill 0.064 (reports/xfg.md) | M5, all variables: MSE 0.2322, skill 0.067; M4, shot and defender distance: 0.2330, skill 0.064 (Section 3 table; constant M2 0.2488 at make rate 0.466) | Chang et al. 2014 | Season 2013-14 (the paper applies M5 to shots through early January 2014; the training sample size is not stated). 10-fold CV, grouping not stated. Their target is a league-average shooter; our hgb includes the shooter, so contest_logistic is the closer match. | not directly comparable for hgb (shooter prior, other season); roughly comparable for contest_logistic against M4 and M5 (same feature family, other season and split) |
| Location-only logistic: OOF Brier 0.2369, skill 0.042 (reports/xfg_baseline.md) | M3, shot distance only: MSE 0.2373, skill 0.046 (Section 3 table) | Chang et al. 2014 | Season and split as above; ours uses distance, angle, three and corner flags | roughly comparable |

Reading: our boosting model has lower log loss and Brier than every published 2015-16 SportVU number we could
check, and our location-only baseline (0.6665 out of fold) is already close to Schmid et al.'s feature-engineered models on their playoff test. Part of our lead is
the shooter prior, which a league-average shot quality leaves out by design.

## 4. Measured weaknesses (from our reports)

1. Held-out calibration reads worse: ECE 0.0189 against 0.0029 out of fold, slope 1.090, under-confident
   (reports/xfg.md, Stage 8). Two readings temper this. The slope interval [0.985, 1.198] contains 1. And binned ECE
   has a floor set by sample size: a Monte Carlo with the out-of-fold bin counts and bin means of reports/xfg.md
   (training games only; independent shots, each bin's outcomes drawn at its own mean, so the model is perfectly
   calibrated by construction) gives an expected ECE of 0.0029 at 101853 shots, the out-of-fold value itself, and
   0.0170 (95 percent range 0.0082 to 0.0286) at 2939 shots. The held-out 0.0189 lies inside that range. Grouping by
   game would widen it further. This is a reading of existing report numbers, not a new held-out score. It means the
   held-out "calibration gap" is not evidence of shift; it also means the out-of-fold ECE says little beyond "no
   gross miscalibration".
2. No per-shot uncertainty. The site shows per-shot xFG in the possession review, and every site value must show an
   interval or say it is a point estimate (CLAUDE.md, Words). Today the xFG is a point estimate.
3. Ad hoc bounding: the isotonic map is clipped to [0.02, 0.98] so a sparse calibration set cannot give 0 or 1
   (reports/xfg.md, Assumptions). The held-out lowest bin (n = 5) and second bin (n = 47) over-predict (0.160
   predicted against 0.085 made), the sparse tail where an isotonic step is least supported.
4. Thin defender context: one defender's distance and closing speed, a count within 6 ft. No second defender, no
   contest angle, no defender size (the rosters table has no height; docs/contract/xfg.md section 3).
5. The baseline's held-out numbers are point estimates only (reports/xfg_baseline.md).
6. The lane takes the shooter of an attempt from the play-by-play name matched to the tracking roster, and the names
   that fail are not random (two players with one short name, a later name, mojibake, a suffix). When the name does
   not resolve, the lane takes the player nearest the ball at release as the shooter (analytics/xfg/features.py, lines
   30 to 31 and 273 to 275). In the training games 5477 out-of-fold attempts did not resolve by name: 5079 took the
   player nearest the ball, for the shooter prior and the contest features, and 398 had no shooter and so no contest
   features. Another 611 tracked attempts carry a name that fits two teammates and went to the first of them
   (reports/export/name_joins.json). The lane was not refit (PLAN.md D94, D95); the site withholds the shot-making
   values those joins touch.

## 5. Recent work that targets these weaknesses

| weakness | work | what it offers | fit for us |
| --- | --- | --- | --- |
| 1, measuring calibration | Roelofs, Cain, Shlens, Mozer. Mitigating Bias in Calibration Error Estimation. AISTATS 2022. https://arxiv.org/abs/2012.08668 | Binned ECE is a biased estimate; equal-mass bins have lower bias than equal-width bins; a debiased estimator (Brocker 2012; Ferro and Fricker 2012) and their ECE_sweep are reliable (abstract) | Report calibration with equal-mass bins and a debiased estimator next to the noise floor, so the D71 test score is not misread the way the held-out ECE was. Cheap; part of xfg-E1. |
| 1, measuring calibration | Kumar, Liang, Ma. Verified Uncertainty Calibration. NeurIPS 2019 (spotlight). https://arxiv.org/abs/1909.10155 | An estimator of calibration error from meteorology needs fewer samples than the plug-in one (abstract) | Older than 2022, kept because it is the standard reference for the estimator problem. |
| 2, per-shot interval | Vovk, Petej, Fedorova 2015 (above) | CVAP gives every shot a pair (p0, p1) whose gap shows how much the calibration step depends on one label; merged probabilities were better than isotonic regression in their experiments, though for Brier loss on large data Isotonic and IVAP were "very close" (Section 6) | The direct drop-in for our isotonic step: same inputs, adds an interval, removes the clip. xfg-E1. |
| 2, per-shot interval | Gupta, Podkopaev, Ramdas. Distribution-free binary classification: prediction sets, confidence intervals and calibration. NeurIPS 2020 (spotlight), arXiv v4 2022. https://arxiv.org/abs/2006.10564 | Distribution-free confidence intervals for binned probabilities (fixed-width and uniform-mass binning); distribution-free calibration needs a binned map, which Platt scaling is not; extensions to covariate shift (abstract) | A true confidence interval for P(make given bin), at the price of a coarse, binned xFG. The comparator if E1's Venn-Abers interval proves too narrow to mean anything to a reader. |
| 2, per-shot interval | van der Laan, Alaa. Self-Calibrating Conformal Prediction. NeurIPS 2024. https://arxiv.org/abs/2402.07307 | Venn-Abers calibration combined with conformal prediction, intervals valid conditional on the calibrated prediction (abstract) | It extends Venn-Abers from binary outcomes to regression, so it matters for continuous targets such as EPV; for a binary make, E1 already tests the original procedure. |
| 4, defender context | Schmid, Schöpf, Kolbinger. Getting NBA Shots in Context: Analysing Basketball Shots with Graph Embeddings. IJCSS 24(1), 2025. https://doi.org/10.2478/ijcss-2025-0005 | First and second closest defenders' positions, speeds and accelerations, plus height and arm span, as inputs; a GNN on raw tracking beats feature models on 2015-16 playoffs (Table 1) | Their feature models score worse than ours, so the gain is not proven to transfer; the second defender and the contest geometry are cheap to add. xfg-E2. |
| 4, defender context | Daly-Grafstein, Bornn. Using in-game shot trajectories to better understand defensive impact in the NBA. Journal of Sports Analytics 6(4), 235-242, 2020. https://doi.org/10.3233/JSA-200400 (preprint https://arxiv.org/abs/1905.00822) | On 50916 threes of 2014-15: tight contests raise variance in depth (56 percent) and left-right distance (38 percent); taller defenders raise entry angle given defender distance; no effect of the defender's side (left or right) on left-right error (Sections 3 and 4 of the preprint) | Supports contest distance and suggests defender side alone adds little; height would need a data source we do not have. |
| player-level intervals (not a measured weakness today) | Bajons, Kook. Rethinking player evaluation in sports: Goals above expectation and beyond. arXiv 2509.20083, 2025 (preprint, revised 2026) | Residualized "above expectation" metrics with valid frequentist inference, applied to NBA 2022-23 shooting (appendix) | Backlog only: matters if the site ever shows shooting above xFG per player. |

## 6. Pre-registered experiments

### xfg-E1: cross Venn-Abers calibration for a per-shot interval (the pilot)

**Result (2026-11-16, reports/research/xfg-E1.md): not adopted.** Development games: CVAP minus isotonic -0.00052
[-0.00081, -0.00024], passes. Test set (10 untouched games, scored once): +0.00067 [-0.00133, +0.00229], fails the
+0.0005 bound; inconclusive at 10 games (PLAN.md D80). Interval width p95 0.0077 and 0.0062, so xFG stays a point
estimate on the site. The pre-registration below is unchanged.

- Hypothesis. Replacing the frozen inner isotonic step of hgb_contest_prior by a cross Venn-Abers predictor (CVAP)
  on the same three inner folds keeps log loss (non-inferior) and gives every shot an interval (p0, p1); the
  log-minimax merge replaces the [0.02, 0.98] clip.
- Baseline. The frozen hgb_contest_prior (isotonic, clipped), refit with frozen code on the development games.
- What it needs from the frozen model. Nothing stored suffices: xfg_predictions.parquet holds only the calibrated
  p_selected. The run needs, per outer fold, from a tools/ refit that calls the frozen functions unchanged
  (analytics.xfg.train: grouped_folds, oof_prior, prior_table, apply_prior, feature_columns, _design, HGB_PARAMS):
  (a) the feature frame (the cached tools.research_xfg features file, FEATURES plus keys and `made`);
  (b) for each of the 3 inner folds, the raw booster probability (before isotonic) of its inner out-of-fold rows,
  with `made` and the inner fold id: the calibration sets;
  (c) the raw booster probability of every outer-test row from each of the 3 inner boosters (the frozen code
  discards these models; the refit keeps them), with the outer-test prior from that inner model's prior table;
  (d) the frozen arm's isotonic-calibrated p for the same rows, `sportvu_game`, outer fold and `made`.
  Gate: before the development refit counts, the isotonic arm refit on all 607 training games must reproduce the
  stored out-of-fold p_selected (fold >= 0) through tools.frozen_common.require_passed.
- Metric and why. Out-of-fold log loss on development games, the lane's selection metric; it punishes the
  confident tail misses that the clip exists to prevent. Reported next to it: Brier; calibration slope; ECE with 10
  equal-mass bins, the debiased estimator of Roelofs et al., and the Monte Carlo noise floor at the same n; and the
  distribution of the interval width p1 - p0 (median, 95th percentile, by decile of p).
- Adoption threshold (fixed now). Adopt CVAP as the calibrator if the paired difference in log loss (CVAP merged
  minus frozen isotonic) has a 95 percent upper bound of at most +0.0005 on development games, and again on the D71
  test set when scored once. +0.0005 is under 1.5 percent of the 0.037 contest gain. Separately, the site copy rule
  is fixed now: if the 95th percentile of p1 - p0 is under 0.01, the site states xFG as a point estimate whose
  calibration step moves it by under 0.01; otherwise each shot shows its (p0, p1).
- Paired test. Bootstrap by game, 2000 resamples, 95 percent percentile interval of the per-game log loss
  difference.
- Ablation. IVAP with a single inner split (one inner fold as calibration set, its one booster scoring the test
  rows), to separate the cross-over from the Venn-Abers regularization, the split Vovk et al. discuss in Section 6.
- Data. Development games only; scored once on the D71 test set. No held-out game is read.
- Compute. 16 GB Mac, CPU, one thread, under the heavy-job lock: 5 outer folds times (3 inner + 1 full) = 20 booster
  fits of the frozen size on at most about 80000 rows, plus the 607-game reproduction (another 20 fits), CVAP by the
  O(n log n) algorithm of Vovk et al. 2015, and a 2000-resample bootstrap on per-game sums. Estimate under 30 minutes
  wall time and under 2 GB memory, reading only the cached feature frame (no tracking frames).
- Risks and failure modes. The Venn-Abers guarantee assumes IID shots; shots in a game are dependent and the D71
  set may be another season, so the guarantee is approximate. (p0, p1) is not a confidence interval for the true
  make probability: it bounds how much one label could move the calibration, so with tens of thousands of
  calibration shots it may be very narrow except in the tails; the site copy rule above handles that outcome. A tie
  or bug in the CVAP code would look like a gain; the isotonic arm's reproduction gate and a unit test against a
  brute-force Venn-Abers on a small sample guard it.
- What a coach gains. Each shot's xFG comes with an honest statement of how firm it is, and the rare very low or very
  high xFG values stop being artefacts of a clip.
- Backlog scores. Expected gain 3: it closes the site's interval gap for the most-shown number, but Vovk et al.
  report Isotonic and IVAP close on large data, so little accuracy change is expected. Effort 2: a tools/ wrapper
  around frozen functions and a small, well-specified algorithm; no tracking frames.

### xfg-E2: second defender and contest geometry

- Hypothesis. Adding the second closest defender's distance, the angle between the shooter-to-basket line and the
  shooter-to-closest-defender line (contest from the front or from the side), and the closest defender's speed
  toward the shooter's line of shot, all at the release frame, lowers out-of-fold log loss.
- Baseline. Frozen hgb_contest_prior refit with frozen code on development games (the E1 baseline).
- Metric and why. Out-of-fold log loss (the selection metric); Brier and calibration reported.
- Adoption threshold (fixed now). Paired log loss difference (new minus baseline) at most -0.001 with its 95 percent
  interval excluding 0, on development games and again on the D71 test set scored once. Chang et al. found 0.0008
  Brier between shot-and-defender distance and all variables, so a smaller change would not show in a coach's view.
- Paired test. Bootstrap by game, 2000 resamples, 95 percent interval.
- Ablation. Second defender distance alone, without the angle and speed features.
- Data. Development games only; the new features are built from tracking frames at the stored release frame,
  one game at a time; scored once on the D71 test set. Defender height is left out: no height source exists in our
  tables.
- Compute. 16 GB Mac, CPU, under the lock: one pass over about 600 games of frames (one game in memory at a time),
  estimated 1 to 3 hours, then the same 20 booster fits as E1 (minutes).
- Risks and failure modes. The current model may already capture most of this through the defender count and
  closing speed, giving a null result; Daly-Grafstein and Bornn found no effect of the defender's side on left-right
  error. Release frame errors move the angle more than the distance.
- What a coach gains. A shot's xFG that tells a hand-in-the-face contest from a late closeout from the side.
- Backlog scores. Expected gain 2: Chang et al.'s step from defender distance to all variables was small, and our
  model already has more contest features than their M4. Effort 3: new frame-reading feature code under tools/, a
  full pass over the tracking frames.

#### xfg-E2 commitments, fixed before any development number (2026-11-19, PLAN.md D121)

- Feasibility, read before any development number from out-of-fold training rows only: Set A (10 games, about 1,668
  shots) has a projected half-width of about 0.002 (xfg-E1's own Set A interval, two near-identical calibrators) to
  about 0.010 (10 random training games, 300 draws: hgb minus location-only 0.0117, hgb minus contest_logistic
  0.0099), against a threshold of 0.001. It cannot decide; development (about 600 games, about 0.0003 to 0.0016)
  is borderline at the threshold and decisive from about 0.002.
- Decision (D121). Development decides: adopt when the paired difference (new minus baseline) is at most -0.001 and
  its 95 percent interval excludes 0. If development adopts, Set A is scored once (score_once, entry "xfg-E2") as a
  confirmation read as D108: "confirms" when the interval lies wholly below 0, "inconclusive at 10 games" when it
  straddles 0, "contradiction" when it lies wholly above 0, which stops the adoption; a t interval on the 10
  per-game mean differences is reported beside it (D80), and the projected test half-width from the development
  per-game variance is reported before the score. If development does not adopt, Set A is not scored (D109).
- The three features, measured at the stored release frame (shot_frames, as the frozen features use it) for the
  resolved shooter of the frozen feature frame (its player_id), only where the frozen contest features exist
  (has_contest 1); NaN elsewhere, which the booster handles as it does the frozen contest features:
  - second_defender_dist_ft: distance from the shooter to the second nearest opponent at release (NaN with fewer
    than 2 opponents tracked).
  - contest_angle_deg: the angle at the shooter between the line to the target rim and the line to the closest
    defender at release, 0 to 180 degrees (0: the defender is straight between shooter and rim; 90: at the side).
  - defender_speed_to_line_ftps: the closest defender's (closest at release) speed toward the line through the
    shooter's release position and the target rim, (distance to that line 0.4 s before release minus at release)
    over 0.4 s, positive when moving toward it; NaN under the frozen closing-speed rule (window across a period,
    not 10 consecutive frames, or a missing position).
  - The target rim is the rim the shooter's team attacks in that period by the frozen vote
    (analytics.decisions.features.baskets_by_team_period on the game's possessions), else the rim nearer the
    shooter at release.
- Arms: the baseline is the frozen hgb_contest_prior refit with the frozen functions (gate: the refit on all 607
  training games under the frozen folds must reproduce the stored out-of-fold p_selected, as in xfg-E1); the new arm
  adds the three columns to the same design, every other setting unchanged (HGB_PARAMS, the shooter prior, the
  isotonic step on 3 inner folds, seed 0); the ablation adds second_defender_dist_ft only.
- Development: every development game (tools.research_testset.dev_games), 5 grouped outer folds, seed 0, the
  calibrated p of each arm; paired per-shot log loss difference with a 2000-resample bootstrap by game; Brier,
  calibration slope and 10-bin equal-mass ECE per arm with game-bootstrap intervals, as in xfg-E1.
- Compute: features from training-game frames, one game at a time under the lock, in resumable jobs of at most
  about 30 minutes; no held-out game is read (the frozen guards run on every game id); Set A frames are read only
  inside the score_once scorer.

#### xfg-E2 Set A confirmation, fixed after the development result and before any score (2026-11-19)

- Development adopts (reports/research/xfg-E2-dev.json: new minus baseline -0.00182 [-0.00222, -0.00141] over 607
  games), so Set A is scored once as D121 says. The projected 10-game half-width from the development per-game
  spread is 0.0034 (normal approximation, the spread of 607 per-game mean differences, 0.0055).
- The scorer fits the baseline and the new arm on every development game (the same fit as development, seed 0), then,
  inside score_once("xfg-E2", Set A), builds the Set A rows with the frozen feature code from the research data root
  (as xfg-E1's scorer) and the three new features from the same root's frames and possessions with the code that
  built the development features, and scores both arms once.
- Readings (D121, D108): the paired per-shot log loss difference (new minus baseline) with a 2000-resample bootstrap
  of the 10 games: "confirms" if its 95 percent interval lies wholly below 0, "inconclusive at 10 games" if it
  straddles 0, "contradiction" if it lies wholly above 0, which stops the adoption. Beside it, reported and read by
  no rule: the t interval on the 10 per-game mean differences (D80), per-arm log loss, Brier, calibration slope and
  ECE with game-bootstrap intervals, the share of Set A rows with contest features, and the ablation is not scored
  on Set A.
- Before the score the scorer is rehearsed: the same function on 10 development games drawn with seed 0 in place of
  Set A, with both arms fitted on the other development games and the rows built from DATA_ROOT by the scorer's own
  path; its new features must equal the cached development features of those games. The rehearsal's numbers are not
  a result and are not reported as one.

## 7. Sources checked (all fetched 2026-11-16)

| source | link | what was read |
| --- | --- | --- |
| Chang et al. 2014, Sloan conference PDF (Internet Archive capture of 2014-03-11) | http://web.archive.org/web/20140311083800/http://www.sloansportsconference.com/wp-content/uploads/2014/02/2014-SSAC-Quantifying-Shot-Quality-in-the-NBA.pdf | full text; Section 3 model table |
| Zadrozny and Elkan 2002, DOI metadata (Crossref) | https://doi.org/10.1145/775047.775151 | title, venue, pages 694-699 |
| Zadrozny and Elkan 2002, full text on the author's site | https://cseweb.ucsd.edu/~elkan/kdd2002.pdf | abstract, isotonic regression section, results |
| Vovk and Petej 2014 | https://arxiv.org/abs/1211.0025 | abstract page and PDF, Theorem 1 |
| Vovk, Petej, Fedorova 2015 | https://arxiv.org/abs/1511.00213 | abstract page and PDF, Sections 4 to 6 |
| Roelofs et al. 2022 | https://arxiv.org/abs/2012.08668 | abstract page |
| Kumar, Liang, Ma 2019 | https://arxiv.org/abs/1909.10155 | abstract page |
| Gupta, Podkopaev, Ramdas 2020 | https://arxiv.org/abs/2006.10564 | abstract page |
| van der Laan and Alaa 2024 | https://arxiv.org/abs/2402.07307 | abstract page |
| Schmid, Schöpf, Kolbinger 2025 | https://doi.org/10.2478/ijcss-2025-0005 (PDF at https://reference-global.com/download/article/10.2478/ijcss-2025-0005.pdf) | full PDF; data section and Table 1 |
| Daly-Grafstein and Bornn 2020, DOI metadata (Crossref) and preprint | https://doi.org/10.3233/JSA-200400 and https://arxiv.org/abs/1905.00822 | metadata; preprint full text |
| Bajons and Kook 2025 | https://arxiv.org/abs/2509.20083 | abstract page; basketball appendix of the PDF |
| Kambhamettu, Shrivastava, Gwilliam 2024, MMSports workshop | https://doi.org/10.1145/3689061.3689068 | abstract only (Consensus and OpenAlex); the full text returned 403, so it is not used (see below) |

Dropped or not used: Kambhamettu et al. 2024 report 81.7 to 96.3 percent accuracy on 2015-16 shots in the abstract,
but the full text could not be read, so the split, the time window of the inputs and any calibration number are
unknown; it is left out of the benchmark. No DOI was found on the Sloan PDF of Chang et al.; the conference archive
copy is cited instead. The Brocker 2012 and Ferro and Fricker 2012 estimators are named only as Roelofs et al.
describe them; their papers were not fetched.
