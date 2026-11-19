# Research dossier: matchups

- Lane: matchups (who guards whom, and what a defender allows), reports/matchups.md.
- Date: 2026-11-16.
- Verdict: the assignment is stable and passes its consistency check, but the defender model failed: ridge on
  possession points does not beat a constant out of fold. The literature measures defense through lower-noise parts
  of the outcome (who shoots, where, and make or miss), which is the change we propose to test.

## 1. What we built

- Question: who guards whom on each possession, and how many points a defender allows against expectation
  (reports/matchups.md).
- Data: 2015-16 SportVU, 604 training games; 17 held-out games with 3162 possessions (reports/matchups.md).
- Assignment: at 1 Hz, Hungarian method on distance to the attacker plus 0.5 times the distance to the midpoint of
  attacker and basket; share = fraction of a possession's sampled frames assigned (reports/matchups.md, Protocol).
- Model: ridge on defender and attacker share columns, target possession points; alpha by nested grouped CV (300 in
  every fold) (reports/matchups.md).
- Numbers: possession-points MSE out of fold ridge 1.2854 [1.2788, 1.2916], constant 1.2842 [1.2776, 1.2903], ridge
  minus constant +0.0012 [0.0002, 0.0020]; held out 1.3195 against 1.3251 (point estimates). Assignment stability
  0.823; defended FG% 0.427 within 4 ft against 0.454 beyond, gap 0.027 [0.020, 0.035] (reports/matchups.md;
  reports/results.md, matchups).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Franks, A., Miller, A., Bornn, L., Goldsberry, K. (2015). Characterizing the spatial structure of defensive skill in professional basketball. Annals of Applied Statistics 9(1), 94-121. | doi:10.1214/14-AOAS799; arXiv:1405.0231 | Two things. (1) Matchups: a hidden Markov model where the defender's position is normal around a convex mix of attacker, ball and hoop, mean = 0.62 O + 0.11 B + 0.27 H estimated by EM, stable across 100 games fitted on 30 possessions each, with transition parameter 0.96 to 0.99 (section 3). Ours fixes the weights by hand (attacker plus half the attacker-basket midpoint) and has no transition model. (2) Defensive skill: outcomes split into shot frequency by court region and shot efficiency, with hierarchical shrinkage; 2013-14 SportVU, half-court possessions of at least 5 s, up to the first shot. Ours regresses possession points directly. |
| Franks, A. M., D'Amour, A., Cervone, D., Bornn, L. (2016). Meta-analytics: tools for understanding the statistical properties of sports metrics. Journal of Quantitative Analysis in Sports 12(4), 151-165. | doi:10.1515/jqas-2016-0098; arXiv:1609.09830 | Stability (does the metric measure the same thing over time), discrimination (does it separate players) and independence. Our defender table was never checked for stability; this paper gives the frame for that check. |
| Keshri, S., Oh, M., Zhang, S., Iyengar, G. (2019). Automatic event detection in basketball using HMM with energy based defensive assignment. Journal of Quantitative Analysis in Sports 15(2), 141-153. | doi:10.1515/jqas-2017-0126 | A player- and location-dependent attraction model for assignment, with HMMs, Gaussian processes and a "bond breaking" model for switches. The abstract reports improvements over existing assignment benchmarks; we could not read the numbers at the source, so none is quoted. |

## 3. Benchmark

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Ridge minus constant, possession-points MSE: +0.0012 [0.0002, 0.0020] out of fold (defender information does not help) (reports/matchups.md) | Out-of-sample log-likelihood, full model against the model with no defense: shooter -25,474.93 against -25,725.17; efficiency -3202.09 against -3239.12 (Table 3, 10-fold CV) | Franks et al. 2015 | Different target (who shoots and from where, and make or miss, against possession points); different metric (log-likelihood against MSE); 2013-14 against 2015-16; their sample is half-court possessions of at least 5 s up to the first shot, with fouls and stoppages excluded; their full model also adds defender types and spatial shrinkage. | Not directly comparable. It shows only the direction: in their decomposition, defense information improved out-of-sample prediction, and in ours it did not. |
| Assignment stability 0.823; defended FG% gap 0.027 (reports/matchups.md) | none found | none | Franks et al. show assignment quality by figures and animations; Keshri et al. report gains whose numbers we could not read at the source. | No published comparable number. |
| Hand-set assignment weights (attacker, plus 0.5 times the attacker-basket midpoint) | Fitted mean weights (0.62, 0.11, 0.27) for attacker, ball and hoop (section 3) | Franks et al. 2015 | Ours is a cost for a one-to-one matching, theirs is the mean of an emission distribution in an HMM that allows doubles. | Not a performance number; listed only as a design difference. |

## 4. Measured weaknesses (from our reports)

1. No defender signal on possession points: ridge is worse than the constant out of fold, +0.0012 [0.0002, 0.0020]
   (reports/matchups.md). The held-out ridge is below the constant by 0.0056, but that is a point estimate over 17
   games and is not evidence (reports/matchups.md, Stage 8).
2. The target is noisy and coarse: possession points go to the rows of the last shooter; possessions without an
   attempt count 0 points and 0 expected (reports/matchups.md, Assumptions). The constant floor MSE is 1.2842.
3. Ridge picks the largest alpha on the grid (300) in every fold (reports/matchups.md), so it shrinks to near the
   constant; the grid may stop too early, which is a sign that the data hold little defender signal for this target.
4. Assignment switches often: 2.44 switches per possession per defender on average, and only 18.4 percent of
   defender-possessions have no switch (reports/matchups.md, Assignment stability). With no transition model, some
   of these switches may be noise; no ground truth exists to tell.
5. The defender table has no stability check across games (reports/matchups.md); reports/results.md says it "ranks
   noise around a small effect".
6. No external matchup record exists for 2015-16, so validation is internal only (reports/matchups.md).

## 5. Recent work aimed at these weaknesses

- Lower-noise targets (weaknesses 1 to 3): Franks et al. 2015 split the outcome into shot frequency by region and
  shot efficiency, and their defense terms improve out-of-sample likelihood in both (Table 3). This is the direct
  answer to a defender effect hidden in possession points.
- Reliability (weakness 5): Franks et al. 2016 give stability and discrimination meta-metrics; we use them as a
  reported check, not a selection rule.
- Assignment with a transition model (weakness 4): the HMM of Franks et al. 2015 and the attraction model of Keshri
  et al. 2019.
- Recent work: our searches found no 2022 to 2026 paper on individual NBA defender ratings from tracking that we
  could check at the source and that targets these weaknesses better than the work above. Recent defensive papers
  we saw classify schemes (switches, traps), which does not attack a measured weakness here, so none is proposed.

## 6. Pre-registered experiments

Shared rules: development games only (the 2015-16 training games that remain after the D71 test set is fixed); 5
grouped folds by game; penalties chosen by inner grouped CV on each fold's training games, as in the frozen lane;
paired bootstrap by game, 2000 resamples, 95 percent percentile interval; the result is scored once on the D71 test
set after the decision; the 17 spent held-out games are never used. Shot probabilities come from the xFG lane's
out-of-fold predictions only (fold >= 0).

### matchups-E1: split the outcome into shot frequency and shot efficiency

- Hypothesis: defender identity carries out-of-sample signal for (a) whether the attacker a defender guards takes the
  possession's field-goal attempt, and (b) whether that attempt goes in given its xFG, even though it carries none
  for possession points.
- Models: (a) penalized logistic regression on matchup rows (attacker, possession), target "this attacker took the
  attempt", with attacker share columns, defender share columns and the attacker's court zone mix; (b) penalized
  logistic regression on attempts, target make, with logit(xFG) as an offset and the assigned defender at release
  as columns. Each is compared with the same model without defender columns.
- Baseline: the model without defender columns (the "no defense" model of Franks et al.). The frozen ridge on
  possession points, refit with frozen code on development games, is reported beside it as the reference for the
  original target.
- Metric: out-of-fold log loss for (a) and for (b). A proper score for a binary outcome; the targets have far less
  noise than possession points, and it is the comparison Franks et al. make.
- Adoption threshold: for each part, defender model minus no-defender model log loss below zero with the paired
  interval excluding zero. A defender table is shown only for a part that passes, labelled with that part. If
  neither passes, the lane keeps its "no defender signal" verdict, now for three targets.
- Ablation: the same defender model on possession points minus xFG-expected points (keep the noisy target, add the
  expectation). It separates the gain of the decomposition from the gain of conditioning on shot quality.
- Reported, not a selection input: split-half stability of the gated defender effects (Spearman between odd and
  even development games), as in Franks et al. 2016.
- Compute: Mac CPU, sparse designs of about 900 columns and a few million rows; under 1 h under the heavy-job lock.
  Estimate, not measured.
- Risks and failure modes: (a) is driven by offensive roles and usage, so defender terms may add little once
  attacker terms are in; (b) has 104792 attempts with an out-of-fold xFG (reports/matchups.md, Inputs), which may be too few for 447 defenders; xFG already
  holds contest features, so (b) may double count defender distance; if the D71 test set is carved from 2015-16
  training games, the frozen xFG model was fitted on those games and its predictions there are not clean.
- What a coach gains: a defender rating that says which part of defense it measures (keeping a man from shooting, or
  making his shots worse) and that has passed an out-of-sample test, instead of a table of noise.
- Backlog: expected gain 4 (turns the lane's failed model into a test that the literature has passed before).
  Effort 3 (new targets and joins to xFG out-of-fold rows; the assignment and folds are reused).
- Amendment (PLAN.md D108, 2026-11-18, written before any development number): the D71 test set is Set A, 10 games
  of one date (about 1,530 attempts with a defender at release and about 9,250 attacker-possession rows, about 8
  attempts per defender), which very likely cannot decide part (b) and is marginal for part (a); the 604
  development games can decide both. As the shared rule reads ("scored once on the D71 test set after the
  decision"), the development out-of-fold result decides each part. Before the test score, the projected test
  half-width from the development per-game variance is reported. Set A is then scored once as a confirmation, read
  as PLAN.md D80: an interval that straddles zero is "inconclusive at 10 games", not evidence against; an interval
  wholly on the wrong side is reported as a contradiction and blocks any site defender table for that part. No
  per-defender claim is made from Set A. On Set A the xFG offset comes from the frozen xFG model refit on the
  development games (the frozen model never read these games); on development games it is the stored out-of-fold
  p_selected (fold >= 0).

### matchups-E2: hidden Markov assignment with fitted weights

- Hypothesis: replacing the hand-set Hungarian cost with the Franks et al. HMM (emission mean = convex mix of
  attacker, ball and hoop, weights and switch probability fitted by EM on development games) gives fewer spurious
  switches and a larger defender signal in E1.
- Baseline: the frozen Hungarian assignment, rebuilt with frozen code on development games, run through E1.
- Metric: the E1 gain (defender minus no-defender log loss) for part (a), computed under each assignment. It judges
  an assignment by whether it carries more defender signal, since no ground-truth matchups exist.
- Adoption threshold: the E1 gain under the HMM larger than under the Hungarian assignment, paired interval
  excluding zero. Reported beside it: switches per possession and the defended FG% gap.
- Ablation: the same emission model with no transition model (per-frame most likely attacker). It separates the
  fitted weights from the temporal smoothing.
- Compute: EM on 1 Hz frames, one game at a time; estimate 2 to 4 h on the Mac CPU under the lock.
- Risks and failure modes: E1 may fail, which leaves E2 with nothing to measure (run E2 only if E1 passes part
  (a)); HMM smoothing can hide real switches on screens; the HMM allows doubles, which changes the share columns.
- What a coach gains: matchup lines that follow real switches and not noise.
- Backlog: expected gain 2 (the assignment already passes its checks; the gain only shows through E1). Effort 3 (an
  EM implementation and a second pass of E1).

## 7. Sources checked (all fetched 2026-11-16)

- Franks et al. 2015: https://arxiv.org/abs/1405.0231 and the PDF https://arxiv.org/pdf/1405.0231v3 (data description,
  section 3, section 5 and Table 3 read); DOI 10.1214/14-AOAS799 as given on the arXiv page.
- Franks et al. 2016: https://arxiv.org/abs/1609.09830 (abstract) and
  https://ideas.repec.org/a/bpj/jqsprt/v12y2016i4p151-165n3.html (journal, volume, pages, DOI).
- Keshri et al. 2019: https://econpapers.repec.org/RePEc:bpj:jqsprt:v:15:y:2019:i:2:p:141-153:n:1 (abstract, journal,
  DOI). Full text not read; no number from it is used.
- Consensus search for defensive matchup inference with tracking data (2026-11-16): returned the papers above and
  scheme classifiers, none of which targets a measured weakness here.
- Dropped: preliminary assignment accuracies for Keshri et al. seen only in a search summary of conference slides;
  they were not read at the source and are not used.
