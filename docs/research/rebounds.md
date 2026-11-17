# Research dossier: rebounds (who secures a missed shot)

Lane: rebounds, the frozen hgb model with a within-miss temperature. Written 2026-11-16 (PLAN.md D70).

Verdict: the model works, far above the uniform floor, but it is trained as a yes or no model per player and only
then normalised within the miss. That leaves the favourite under-confident, and the held-out games score worse by
more than game-to-game noise explains. No published player-level number on NBA tracking could be checked, so the
benchmark is ours against our own floor.

## 1. What we built

- Unit: a miss; one row per player on court at the rim frame plus a "none" row (team rebound); features at the rim
  frame (distances, speed toward the rim, side, nearest opponent, ball apex, player rate) (reports/rebounds.md).
- Model: sklearn gradient boosting (80 trees, depth 3) with a binary target, normalised within the miss as p
  proportional to q^tau, tau about 1.256 fitted on inner folds (reports/archive/rebounds.md, variant 1).
- Data: 49288 misses, 604 training games, 5 folds by game; 1227 misses from 17 held-out games.
- Out of fold: per-miss log loss 1.6009 [1.5882, 1.6103], top-1 0.4797; uniform floor 2.3881. Held out (point
  estimates): 1.7115, top-1 0.4474 (reports/rebounds.md, Stage 8).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Maheswaran, Chang, Henehan, Danesis. Deconstructing the Rebound with Optical Tracking Data. MIT Sloan Sports Analytics Conference 2012. | https://www.sloansportsconference.com/research-papers/deconstructing-the-rebound-with-optical-tracking-data (PDF linked there) | SportVU positions of players and ball around a miss; offensive rebound rate depends non-linearly on shot location; which team rebounds becomes more predictable as the ball falls (Section 7). We predict the player, not the team, at one moment (the rim frame). |
| Maheswaran, Chang, Su, Kwok, Levy, Wexler, Hollingsworth. The Three Dimensions of Rebounding. MIT Sloan Sports Analytics Conference 2014. | https://www.sloansportsconference.com/research-papers/the-three-dimensions-of-rebounding (PDF linked there) | Rebounding as positioning at the shot, crash to the rim, then conversion; the value of a player's Voronoi cell, weighted by where similar misses land, is re-evaluated at the moment the ball is closest to the rim (Sections 3 and 4). Our rim frame matches their crash endpoint; we learn the probability from features instead of integrating a landing map over Voronoi cells. |
| Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017. | https://arxiv.org/abs/1706.04599 | Temperature scaling, a one-parameter recalibration that keeps the ranking. Our tau is that idea applied to log q within each miss; top-1 is unchanged by it (reports/archive/rebounds.md). |
| Cao, Qin, Liu, Tsai, Li. Learning to rank: from pairwise approach to listwise approach. ICML 2007, pp. 129-136. | https://doi.org/10.1145/1273496.1273513 (abstract: https://www.microsoft.com/en-us/research/?p=160449) | A listwise loss built from probability models over the whole list of objects. Our model does not use it: it trains on single candidate rows. rebounds-E2 trains on the whole miss in this listwise way. |
| Chen, Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016. | https://arxiv.org/abs/1603.02754 | Tree boosting on first and second order gradient statistics of the loss (Section 2.2), so any twice-differentiable loss, including a within-miss softmax, can be boosted. The tool for rebounds-E2. |

## 3. Benchmark

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Player-level top-1 0.4797 out of fold, 0.4474 held out; per-miss log loss 1.6009 and 1.7115 (reports/rebounds.md) | Team-level accuracy: about 77 percent at the release, 87.5 percent when the ball has fallen to 8 ft (Section 7 and its figure; SVM, better than logistic regression by several points) | Maheswaran et al. 2012 | They predict which team, we predict which player (or none); their moments are the release and fixed ball heights after the rim, ours is the rim frame; over 11000 misses from an earlier SportVU sample (season not stated), split not stated; we do not report a team-level accuracy | not directly comparable: different target, moment and split |
| same | no player-level number on NBA tracking found that could be read | none | Hojo, Fujii, Kawahara 2019 report 70.3 and 71.1 percent "predicted who obtained rebounds" in the abstract, but the full text is closed and what that accuracy counts (players, misses, balance) could not be checked | no comparable published number |

Our own references stand in: the uniform floor (2.3881 out of fold) and the prior-only model (2.1380), both in
reports/rebounds.md; hgb beats the prior by -0.5371 [-0.5477, -0.5283].

## 4. Measured weaknesses (from our reports)

1. Held out is worse beyond noise: 1.7115 against 1.6009 [1.5882, 1.6103], top-1 0.4474 against 0.4797, with the same
   number of candidates per miss (reports/rebounds.md, Stage 8). A rough reading: the out-of-fold interval spans
   about 0.011 on each side over 604 games; scaled by the square root of 604/17 (about 6) that is about 0.066 for 17
   games, and the 0.111 gap exceeds it. For top-1, the same scaling gives about 0.027 against a gap of 0.032. This
   reads existing numbers; no held-out metric is recomputed. 16 of the 17 held-out games are at one arena, so arena,
   tracking installation and opponent mix cannot be separated.
2. The favourite is under-confident even after the temperature: top-candidate calibration slope 1.367 [1.340, 1.402],
   ECE 0.041; in the 0.7 to 0.8 bin the favourite is predicted 0.747 and secures 0.868 (reports/rebounds.md,
   Calibration). Before the temperature the slope was 1.69 (reports/archive/rebounds.md).
3. The cause named in the archive: "a binary model normalised within the miss flattens the distribution"
   (reports/archive/rebounds.md). One exponent for every candidate cannot fix a flattening that differs between
   offense, defense and the none row. The logit candidate is also only a binary approximation of the conditional
   logit (reports/rebounds.md, Protocol).
4. One moment only: features at the rim frame and the 0.4 s before it; nothing about who boxes out whom beyond the
   nearest opponent distance; no player size (no height source in our tables).

## 5. Recent work that targets these weaknesses

| weakness | work | what it offers | fit for us |
| --- | --- | --- | --- |
| 3, the objective | Wang, Veličković, Hennes and others. TacticAI: an AI assistant for football tactics. Nature Communications 15, 1906 (2024). https://doi.org/10.1038/s41467-024-45965-x (preprint https://arxiv.org/abs/2310.10553) | Receiver prediction for corner kicks as one softmax over the 22 players, trained with softmax cross entropy, from a graph attention network; top-3 test accuracy 0.782 plus or minus 0.039 on 7176 Premier League corners (Results; Methods) | Same shape as our task (which of the players gets the ball, one softmax per event). Supports a within-event softmax objective; their graph network is the heavier follow-up if E2 shows the objective matters. Not comparable as a number (other sport and task). |
| 3, the objective | Cao et al. 2007 (above) | Listwise loss over a whole list | The loss for rebounds-E2. Older than 2022, kept because it is the source of the listwise idea. |
| 2, calibration | Guo et al. 2017 (above) | Temperature scaling, one parameter that keeps the ranking (abstract) | Our archive shows one shared exponent left the favourite under-confident (slope 1.37), which motivates rebounds-E1, a temperature and an offset per candidate type. |
| player ability, not a measured weakness today | Kiriazis, Genest, Leblanc. A Bayesian two-stage framework for lineup-independent assessment of individual rebounding ability in the NBA. Journal of Quantitative Analysis in Sports 21(4), 303-326 (issued 2024 per Crossref). https://doi.org/10.1515/jqas-2023-0097 | Rebounding ability separated from the lineup on the floor, predicting 2021-22 from 2020-21 (abstract) | Our smoothed player rate ignores teammates; backlog only, if the site shows a per-player rebounding rating. |
| broadcast stand-in, not this lane's measured weakness | Barrera Roy, Clapés Sintes. See It Before You Grab It: Deep Learning-based Action Anticipation in Basketball. arXiv 2512.15386, 2025. https://arxiv.org/abs/2512.15386 | Which team gains possession, anticipated from broadcast video; about 100000 clips, over 2000 annotated rebounds (abstract) | Relevant to the broadcast demo, which has no ball or identity; team-level only. |

## 6. Pre-registered experiments

### rebounds-E1: a temperature and an offset per candidate type

- Hypothesis. Replacing the single exponent by p proportional to exp(a_g log q + b_g) within the miss, with g one of
  offense player, defense player and none (three slopes, two free offsets), fitted the same way as the frozen tau on
  inner out-of-fold predictions, lowers per-miss log loss and brings the favourite's calibration slope toward 1.
- Baseline. The frozen hgb with its single tau, refit with frozen code on the development games.
- What it needs from the frozen model. A tools/ refit that calls the frozen functions unchanged (analytics.rebounds.
  train: oof_prior, design, the HGB settings, the inner grouped folds of inner_temperature) and keeps, per outer
  fold: the raw hgb q (before normalisation) of the inner out-of-fold rows, and of the outer-test rows from the full
  fit; with miss key, is_offense, is_none, the label, sportvu_game and the fold ids. The candidate-row feature files
  already on DATA_ROOT suffice; no tracking frames. Gate: the frozen single-tau arm refit on all 604 training games
  must reproduce the stored out-of-fold per-miss log loss 1.6009 (tools.frozen_common.require_passed) before the
  development refit counts.
- Metric and why. Out-of-fold per-miss multinomial log loss, the lane's selection metric, which scores the whole
  distribution over candidates. Also reported: top-1 (unchanged only if the type offsets never reorder candidates,
  which they can), the favourite's calibration slope and ECE (equal-mass bins), and team-level accuracy where the
  label names a team, so a later comparison with Maheswaran et al. 2012 is possible.
- Adoption threshold (fixed now). Paired difference (new minus baseline) at most -0.005 with its 95 percent interval
  excluding 0, on development games and again on the D71 test set scored once. -0.005 is under a quarter of the gain
  the temperature itself bought (-0.0225).
- Paired test. Bootstrap by game, 2000 resamples, 95 percent percentile interval.
- Ablation. One shared slope (the frozen tau) with the two type offsets only, to tell whether the gain is in the
  offsets or in separate sharpening per type.
- Data. Development games only; scored once on the D71 test set.
- Compute. 16 GB Mac, CPU, under the heavy-job lock: per outer fold 5 inner and 1 full booster fits on up to about
  430000 candidate rows (30 fits), the same again for the reproduction gate, and a 5-parameter likelihood fit.
  Estimate under 40 minutes and under 3 GB.
- Risks and failure modes. Five parameters fitted on inner folds can overfit little, but the gain may be small if
  the flattening is not type-specific; the D71 test set may have few games, so its interval may be wide and the
  adoption may fail on power alone (the result is then reported as inconclusive, not as a rejection).
- What a coach gains. Rebound chances that mean what they say: a 75 percent favourite who wins it 87 percent of the
  time makes every "rebounds over expected" number look better than it is.
- Backlog scores. Expected gain 2: a calibration repair with a few parameters; the temperature showed this kind of
  step is worth about 0.02 nats, and part of that is already taken. Effort 1: a wrapper and one small likelihood
  fit, no new features, no frames.

### rebounds-E2: train on the whole miss (within-miss softmax objective)

- Hypothesis. A boosted model trained with the within-miss softmax cross entropy (the listwise loss of Cao et al.
  2007 for one winner per list), on the same features, beats the frozen binary model plus temperature in per-miss
  log loss, because it learns that candidates compete.
- Baseline. The frozen hgb with tau refit with frozen code on the development games (as in E1).
- Model. XGBoost (installed, version 3.2) with a custom objective: per miss, gradient p_i - y_i and hessian
  p_i (1 - p_i) of the softmax over that miss's candidate rows; the same depth (3), learning rate (0.1) and number of
  trees (80) as the frozen booster, fixed now, no tuning. A unit test checks the gradient against a brute-force
  softmax on a small sample.
- Metric and why. Out-of-fold per-miss log loss (the selection metric); top-1 and the favourite's calibration slope
  reported.
- Adoption threshold (fixed now). Paired difference at most -0.010 with its 95 percent interval excluding 0, on
  development games and again on the D71 test set scored once. The bar is higher than E1 because it replaces the
  model, not a calibration step.
- Paired test. Bootstrap by game, 2000 resamples, 95 percent interval.
- Ablation. The same XGBoost settings with the binary objective and the frozen tau procedure, so a gain is not a
  library difference.
- Data. Development games only; scored once on the D71 test set.
- Compute. 16 GB Mac, CPU, under the lock: 5 outer folds times 2 arms (softmax and binary ablation) on up to about
  430000 rows, 80 trees, depth 3. Estimate under 1 hour and under 4 GB.
- Risks and failure modes. With only 80 depth-3 trees the softmax model may underfit the none row, which has its own
  feature pattern; the objective needs every miss's rows contiguous (a sort bug would silently mix misses, guarded by
  the unit test and a check that each miss's probabilities sum to 1). If E1 is adopted first, E2's baseline becomes
  the E1 model, decided now so the comparison is fixed.
- What a coach gains. A rebound model that weighs who is fighting whom, which is what box-out evaluation needs.
- Backlog scores. Expected gain 3: it attacks the cause the archive names, and the jump from the binary-logistic
  approximation (1.7558) to boosting (1.6009) shows how much model form matters here. Effort 3: custom objective
  code, tests, a second library in the experiment path.

## 7. Sources checked (all fetched 2026-11-16)

| source | link | what was read |
| --- | --- | --- |
| Maheswaran et al. 2012 | https://www.sloansportsconference.com/research-papers/deconstructing-the-rebound-with-optical-tracking-data and the PDF it links | full text; data description; Section 7 |
| Maheswaran et al. 2014 | https://www.sloansportsconference.com/research-papers/the-three-dimensions-of-rebounding and the PDF it links | full text, Sections 1 to 4 |
| Guo et al. 2017 | https://arxiv.org/abs/1706.04599 | abstract page |
| Cao et al. 2007, DOI metadata (Crossref) and abstract | https://doi.org/10.1145/1273496.1273513 and https://www.microsoft.com/en-us/research/?p=160449 | metadata (pages 129-136); abstract |
| Chen and Guestrin 2016 | https://arxiv.org/abs/1603.02754 | abstract page and PDF, Section 2.2 |
| Wang et al. 2024 (TacticAI), DOI metadata (Crossref) and preprint | https://doi.org/10.1038/s41467-024-45965-x and https://arxiv.org/abs/2310.10553 | metadata (volume 15, article 1906); preprint Results and Methods |
| Kiriazis, Genest, Leblanc, DOI metadata (Crossref) and abstract (Consensus) | https://doi.org/10.1515/jqas-2023-0097 | metadata; abstract |
| Barrera Roy and Clapés Sintes 2025 | https://arxiv.org/abs/2512.15386 | abstract page |
| Hojo, Fujii, Kawahara 2019, DOI metadata (Crossref) and abstract (OpenAlex) | https://doi.org/10.1080/24748668.2019.1582892 | metadata; abstract; full text closed, so not used as a benchmark |

Dropped or not used: Hojo et al. 2019's 70.3 and 71.1 percent (abstract only; the meaning of the accuracy could not
be checked). Okubo et al.'s rebound-position simulations (Procedia Engineering 2013, 2014) appeared in the
search but were not fetched, so they are not cited. No published player-level NBA tracking benchmark was found.
