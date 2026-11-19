# Research dossier: gravity

- Lane: gravity (how close defenders stay to a player beyond what location explains), reports/gravity.md.
- Date: 2026-11-16.
- Verdict: the location expectation holds on held-out games, but gravity itself, the residual, has no test of its
  own, and its only link to outcomes is small (+0.0042 location-only xFG for teammates, training games). Recent work
  removes the pull of teammates' positions before calling something gravity; ours does not.

## 1. What we built

- Question: how far defenders stay from a player beyond what the player's location explains (reports/gravity.md).
- Data: 2015-16 SportVU, 7983660 player-frames at 1 Hz from 604 training games; 195015 from 17 held-out games
  (reports/gravity.md).
- Model: ridge expectation of nearest-defender distance on the player's location, ball distance, has_ball, zone make
  rate and clocks; gravity_ft = expected minus actual (positive: defenders closer than location predicts). Gradient
  boosting fits better (MSE 22.712) and is a diagnostic only (reports/gravity.md, Candidates).
- Numbers: ridge MSE 28.119 [27.800, 28.452], skill over the constant floor 0.416, out of fold; 29.913, skill 0.434,
  held out. Top-ten gravity players on the floor: teammates' location-only xFG higher by 0.0042 [0.0009, 0.0074],
  163 of 290 games positive, p = 0.0198, training games (reports/gravity.md; reports/results.md, gravity).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Franks, A., Miller, A., Bornn, L., Goldsberry, K. (2015). Characterizing the spatial structure of defensive skill in professional basketball. Annals of Applied Statistics 9(1), 94-121. | doi:10.1214/14-AOAS799; arXiv:1405.0231 | The canonical defender position as a convex mix of attacker, ball and hoop (section 3): the idea that a defender's distance has an expected part set by the ball and the basket. Ours models the nearest defender's distance with no assignment and a linear expectation. |
| Keshri, S., Oh, M., Zhang, S., Iyengar, G. (2019). Automatic event detection in basketball using HMM with energy based defensive assignment. Journal of Quantitative Analysis in Sports 15(2), 141-153. | doi:10.1515/jqas-2017-0126 | A player- and location-dependent attraction model, which is a model-based form of gravity built into the assignment. Ours is a residual of a location-only expectation. Full text not read; nothing beyond the abstract is used. |
| Barron, B., Sitaraman, N., Arias, T. (2025). Analyzing NBA player positions and interactions with density-functional fluctuation theory. Scientific Reports 15, 19830. | doi:10.1038/s41598-025-04953-x | Gravity as the change in local defensive density a specific player causes, compared with a typical player at the same spot and with the same teammate positions, so the pull of the other four attackers is removed ("Player gravity" section). Gravity is location-specific (eight half-court spots, on and off ball). Data: Second Spectrum, 2022-23 up to 20 January 2023, positions within 3 s before a shot. Ours does not condition on teammates and gives one number per player. |

## 3. Benchmark

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| Expectation skill 0.416 out of fold, 0.434 held out (reports/gravity.md) | none found | none | Barron et al. report density-based "ball equivalents" per location and a location prediction rate for players, not an error for nearest-defender distance. | No published comparable number. |
| Teammates' location-only xFG +0.0042 [0.0009, 0.0074] with a top-ten gravity player on the floor (reports/gravity.md) | none found | none | Barron et al. check gravity against known scoring patterns of named players, not against an outcome; Keshri et al. give no outcome number we could read. | No published comparable number. |

We found no peer-reviewed paper that ties a tracking-based gravity measure to teammates' shot outcomes with a number
we could check. The idea of gravity in public writing (ESPN 2014, and NBA and WNBA gravity stats from 2025) is not a
paper and gives no method we could verify, so it is not used.

## 4. Measured weaknesses (from our reports)

1. Gravity has no accuracy or validity test of its own: only its expectation is scored (reports/gravity.md, Stage 8:
   "Gravity itself is the residual of this expectation and has no held-out accuracy of its own").
2. The outcome link is small and narrow: +0.0042 in teammates' location-only xFG, measured only for the top ten
   players, by an on/off sign test, on training games (reports/gravity.md, Sign test). Location-only xFG measures
   where teammates shoot, not how open they are.
3. The expectation leaves structure: gradient boosting on the same features lowers MSE by 5.407 [5.280, 5.555]
   (reports/gravity.md). Part of what we call gravity is misfit of a linear expectation.
4. The expectation sees only the player's own location and the ball (reports/gravity.md, Protocol), not the other
   attackers' positions, so a player's number can carry the pull of his teammates.
5. Ranks carry no claim: "No known high-gravity player on this input" (reports/gravity.md, Top-decile check).

## 5. Recent work aimed at these weaknesses

- Teammate confounding (weakness 4): Barron et al. 2025 compare a player-specific model with a typical-player
  model on the same offensive positions, so the difference is not due to where teammates stand. This is the
  recent method most directly aimed at a measured design gap of ours.
- Location dependence (weakness 3): Barron et al. find gravity strongly depends on court location; Keshri et al.
  2019 make attraction player- and location-dependent. A single per-player number with a linear expectation can
  misread location effects as player effects.
- Outcome link (weakness 2): no checked paper links gravity to teammates' openness; the experiment below builds
  that test from our own data rather than from a published design.

## 6. Pre-registered experiments

Shared rules: development games only (the 2015-16 training games that remain after the D71 test set is fixed); 5
grouped folds by game; paired bootstrap by game, 2000 resamples, 95 percent percentile interval; the result is
scored once on the D71 test set after the decision; the 17 spent held-out games are never used. A player's gravity
used to explain a shot is computed without that shot's game (from the other folds), so no game supplies both. No
player-identifying feature enters the expectation.

### gravity-E1: does gravity open shots for teammates?

- Hypothesis: a teammate's shot is more open (larger nearest-defender distance at release) when the shooter's
  on-floor teammates have higher gravity, beyond the shot's location and the shooter.
- Baseline: openness modelled by ridge on shot location features and a shooter term (the gravity-free model); the
  gravity input comes from the frozen ridge expectation refit with frozen code on development games.
- Candidate: the baseline plus the sum and the maximum of the on-floor teammates' gravity (out of fold by game).
- Metric: out-of-fold MSE of nearest-defender distance at release (ft squared). Openness is the mechanism gravity
  claims, it is measured on every attempt, and it is less noisy than make or miss. Reported beside it: the change in
  out-of-fold log loss of make with logit(location-only xFG) as an offset.
- Adoption threshold: candidate minus baseline MSE below zero with the paired interval excluding zero, and an effect
  of at least 0.2 ft of openness per standard deviation of teammates' gravity sum. If it fails, the site says gravity
  was tested against teammates' openness and showed no effect of that size.
- Ablation: the same candidate with each player's raw mean nearest-defender distance in place of gravity (no
  expectation). It tests whether the location expectation adds anything over raw closeness.
- Compute: shots on development games joined to release-frame defender distance (the xFG lane's contest features)
  and to per-player gravity from the gravity tables; Mac CPU, under 1 h under the lock. Estimate, not measured.
- Risks and failure modes: lineups confound gravity with teammates' skill and team scheme (a shooter term helps, a
  lineup term may be too sparse); garbage time and transition shots can dominate openness; shots whose shooter has
  no roster match are dropped (as in the frozen lane); if D71 carves test games from 2015-16, the frozen tables were
  built on those games and the lead must rebuild them without the test games.
- What a coach gains: a gravity number that is shown, or shown not, to leave teammates more open, which is the use a
  coach has for it.
- Backlog: expected gain 3 (turns weakness 1 into a real test with a sharper outcome than location-only xFG). Effort
  2 (joins of existing tables and two ridge fits).

#### Amendment (PLAN.md D107, 2026-11-19 UTC, written before any development number or run)

Why: Set A can be scored, but it holds about 935 attempts with a release-frame defender distance (1668 shots, 56.1
percent with contest features, reports/research/xfg-E1-final.json). At the threshold effect (0.2 ft per SD) the chance
that its MSE interval excludes zero is about 0.11 to 0.13, and its own estimate of the effect has a 95 percent
half-width of about 0.23 to 0.27 ft. Development (62387 attempts) is well powered. The shared rule says the result is
scored once on the D71 test set after the decision. D107 therefore fixes: development decides; Set A is a
pre-declared consistency check, scored once.

- Decision (development games only): adopt if the out-of-fold MSE difference (candidate minus baseline) has its 95
  percent interval wholly below zero and the effect is at least 0.2 ft of openness per SD of teammates' gravity sum,
  in the direction the hypothesis states (more open with more gravity).
- Set A (D107), scored once: the test effect per SD and the test MSE difference, each with a bootstrap of the 10 games
  (2000 resamples) and a report-only t interval on the per-game values (as D80). Readings, from the bootstrap
  interval of the test effect, in this order: "inconsistent" if it excludes the development estimate; otherwise
  "confirms" if it excludes zero on the development estimate's side; otherwise "consistent". Only "inconsistent"
  stops an adoption. The order is part of this commitment: a test interval that excludes both zero and the
  development estimate reads "inconsistent", as D107's wording has it.

Commitments fixed with this amendment (code: tools/research_gravity.py), before any number:

1. Gravity input: the frozen gravity lane rerun with its own functions on the development games (validate_game and
   build_game one game at a time, the frozen ridge expectation out of fold under its 5 grouped folds, seed 0; the
   HGB diagnostic candidate is not refit, since nothing here reads it). Gate: the rebuilt gravity_frames equal the
   stored frozen table's training rows (keys identical, expected_ft within 1e-4 ft), the stored gravity_players
   means are reproduced (within 1e-6 ft), and the ridge out-of-fold MSE equals the printed 28.119.
2. Shots: the frozen xFG feature frame of the training rows (built by tools.research_xfg with the frozen code; its
   keys must equal the stored xFG predictions' training rows), development games only, attempts with a release-frame
   defender distance (has_contest). Outcome: closest_defender_dist_ft at release.
3. Teammates: the cleaned frames (frozen clean_frames) at the release frame; the shooter's team is the team of the
   shooter's row in that frame. An attempt is used only when the shooter is in the frame and exactly 4 teammates are,
   each with a gravity value; every other attempt is dropped and counted.
4. A teammate's gravity for an attempt is the mean gravity_ft of the teammate's frames in development games outside
   the attempt's fold (the experiment uses the frozen lane's 5 grouped folds by game, seed 0), so no game supplies
   both. Stated, not repaired: the frozen out-of-fold expectations of those frames come from ridge fits that saw the
   attempt's fold. For Set A, the mean over every development frame.
5. Features: the four location features of the frozen location-only xFG model (distance_ft, angle_rad, is_three,
   is_corner) standardised, plus a shooter term (one indicator per shooter seen in the training folds; an unseen
   shooter has none). Baseline: Ridge (alpha 1, the frozen lane's) on those. Candidate: the baseline plus the sum and
   the maximum of the 4 teammates' gravity, standardised. Folds: the 5 grouped folds of commitment 4.
6. Metric: out-of-fold MSE of the release distance (ft squared), candidate minus baseline, shot-weighted; interval: a
   paired bootstrap by game of the fixed per-shot squared errors, 2000 resamples, seed 0. Resampling games of fixed
   errors adds no noise to any value (unlike the decisions-E1 reliability statistics, D86).
7. Effect per SD: cross-fitted partial coefficient. In each fold, ridges with the baseline design, fit on the other
   folds, predict the release distance and the two standardised gravity features; on the fold, the residuals are
   formed; the pooled least-squares slope of the distance residual on the two gravity residuals (with an intercept)
   gives the effect per SD of the gravity sum. SD: the standard deviation of the teammates' gravity sum over the
   development attempts. Interval: a bootstrap by game of the fixed residuals, 2000 resamples. The decision reads the
   point estimate. On Set A the same slope is formed with the nuisance ridges fit on every development attempt.
8. Reported beside, read by no rule: the change in out-of-fold log loss of make (logistic with the out-of-fold
   location-only xFG logit, the frozen analytics.train_xfg model, as an offset, plus the two gravity features;
   against the offset alone); the ablation (each player's raw mean nearest-defender distance in place of gravity,
   same folds); the coefficient on the gravity maximum.
9. Set A refusals, decided now: the scorer refuses, with its reason and no number, unless every test game has frames
   and at least one used attempt; a summary with any non-finite number (for one, a t interval over fewer than 2
   games) is a refused verdict, never a NaN in the ledger. The score runs on a clean tree; the final file records the
   commit, the gate files' hashes and the development record's hash, and nests the summary under "scored".

### gravity-E2: an expectation that knows where the teammates are

- Hypothesis: adding the other four attackers' positions to the expectation (relative offsets sorted by distance,
  plus spacing such as hull area; still no identity), in the spirit of Barron et al., gives a gravity that predicts
  teammates' openness better than the frozen gravity.
- Baseline: the frozen ridge expectation refit with frozen code on development games, run through E1.
- Metric: the E1 effect, computed with each gravity. It judges a gravity by the outcome it claims to explain; the
  expectation's own MSE is reported as a diagnostic only, since a lower MSE can come from fitting the pull itself.
- Adoption threshold: E1's MSE improvement with the new gravity larger than with the frozen gravity, paired interval
  excluding zero; and the new gravity must pass E1's own threshold. Run only after E1.
- Ablation: the same features in gradient boosting. It tests the frozen lane's claim that a flexible expectation
  absorbs the pull: if boosting gives a gravity with a smaller E1 effect, the claim holds.
- Compute: ridge on about 8M player-frames with about 20 more columns, one game at a time; 1 to 2 h on the Mac CPU
  under the lock; boosting ablation 2 to 4 h. Estimates.
- Risks and failure modes: teammate features can also absorb part of a star's pull (when a star draws two defenders,
  the help shows in teammates' positions); sorted slots are a crude stand-in for the density model of Barron et al.;
  the gain may only show if E1 already finds an effect.
- What a coach gains: a gravity that does not credit a player for the spacing of the four players around him.
- Backlog: expected gain 2 (aimed at a design gap that recent work names, but the size of the confounding in our
  data is unknown). Effort 3 (new expectation features and a second pass of E1).

## 7. Sources checked (all fetched 2026-11-16)

- Barron et al. 2025: full text via Europe PMC, https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12141578/fullTextXML
  (Data and Player gravity sections read; DOI, volume and article number from the same record). PMC page
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12141578/ returned a browser check and was not read.
- Franks et al. 2015: https://arxiv.org/abs/1405.0231 and https://arxiv.org/pdf/1405.0231v3 (section 3 read).
- Keshri et al. 2019: https://econpapers.repec.org/RePEc:bpj:jqsprt:v:15:y:2019:i:2:p:141-153:n:1 (abstract only).
- Consensus search for off-ball gravity and defensive attention with tracking data (2026-11-16): returned Barron et
  al., Franks et al., a 2026 systematic review of spatial indicators (Liu et al., Journal of Sports Science and
  Medicine) and a dissertation; the review and the dissertation were not read at the source and are not cited.
- Web search for gravity in basketball (2026-11-16): returned journalism and league stat pages, not papers; not used.
- Dropped: a student executive summary on barycentric gravity (2014-15 data) and a Sloan-style soccer space paper
  (Fernandez and Bornn 2018) seen only through third-party copies; neither was read at an official source.
