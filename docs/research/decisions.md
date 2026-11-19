# Research dossier: decisions (decision quality)

- Lane: decisions (analytics/decisions, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: failed. Mean decision quality does not track on-court offensive rating in either sample (mean per-game
  Spearman 0.012 [-0.008, 0.029] out of fold, -0.014 [-0.116, 0.100] held out). This dossier asks whether the method,
  the validation target, or both are at fault. The failed verdict stays as it is.

## 1. What we built

- Question: did the ball handler choose the best available option, in expected points (reports/decisions.md).
- Data: 518421 decisions from 604 training games (304460 passes, 159761 drives, 54200 shots); 11373 from 17
  held-out games (reports/decisions.md, Event counts and Held-out sections).
- Option values (reports/decisions.md, Value-function assumptions): shot = points times xFG (the xFG lane's
  p_selected, chosen by a coverage rule fixed in advance); pass = points times location-only xFG at the teammate times
  an openness factor times a pass-risk factor; drive = the shot value 8 ft nearer the rim times 0.90; hold = the EPV
  GRU's value. Quality = chosen value minus best value, so it is regret, at most 0.
- Validation: per game, Spearman between a player's mean quality and their on-court points per 100 possessions;
  mean over games, bootstrap by game (reports/decisions.md, Validation).
- Numbers: 0.012 [-0.008, 0.029], 317 of 603 games positive, sign p 0.222 out of fold; -0.014 [-0.116, 0.100], 7 of
  17, held out. Shuffled: -0.013 and -0.069 (reports/decisions.md).

## 2. Foundations

| paper | link | what we take from it | how ours differs |
| --- | --- | --- | --- |
| Cervone, D'Amour, Bornn, Goldsberry (2014). POINTWISE: Predicting Points and Valuing Decisions in Real Time with NBA Optical Tracking Data. MIT Sloan Sports Analytics Conference. | PDF: https://raw.githubusercontent.com/papers-we-love/papers-we-love/refs/heads/main/sports_analytics/2014-ssac-pointwise-predicting-points-and-valuing-decisions-in-real-time.pdf | Every option is valued on one scale: the EPV of the possession after that option (Figure 1: "EPV after pass", "EPV of shot", "change in EPV"). | Only our hold option uses EPV. Our pass, drive and shot values are one-step formulas with hand-set factors (openness, 0.95 pass risk, 0.90 drive factor), so options sit on different scales. |
| Cervone, D'Amour, Bornn, Goldsberry (2016). A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes. JASA 111(514). | DOI 10.1080/01621459.2016.1141685; arXiv:1408.0777 | Shot satisfaction: a shot's value against the most reasonable pass at the moment of the shot (Section 5.4); EPV-Added (Section 5.3, Eq. 19). They give no quantitative validation of either against other statistics and say EPV analysis is best paired with film. | Our "best option" uses hand-set values; theirs comes from fitted transition models. Neither has a validated decision target. |
| Jutamulia, Hosoi (2025). Leveraging expected action value to measure on-court decision-making skills in basketball. Journal of Sports Analytics 11. | DOI 10.1177/22150218251325018 | Expected Action Value: value only the immediate pass and shot options of the ball handler, then count missed opportunities and their cost (abstract). | Closest published design to ours. Full text was blocked (HTTP 403), so no method detail or number is taken from it. |
| Decroos, Bransen, Van Haaren, Davis (2019). Actions Speak Louder Than Goals: Valuing Player Actions in Soccer. KDD 2019 (DOI 10.1145/3292500.3330758). | arXiv:1802.07127 | Value an action as the change in scoring (minus conceding) probability between the states before and after it; evaluate the probability models (Brier and AUC, Table 3), not the action values, because action values have no ground truth. | We value unchosen options, not only the chosen action, and we validate player means against a team outcome. VAEP reports no validation of its player ratings against outcomes. |
| Fernandez, Bornn, Cervone (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning (DOI 10.1007/s10994-021-05989-6). | arXiv:2011.09426 | Option values from one decomposition: P(success of the action) times the EPV after success, plus P(failure) times the value after failure; shots through expected goals (Section 3.2, Eq. 2 to 4). | Our pass value uses a fixed risk factor and a location-only xFG at the receiver instead of a learned completion probability and the EPV at the receiver. |
| Franks, D'Amour, Cervone, Bornn (2016). Meta-analytics: tools for understanding the statistical properties of sports metrics. Journal of Quantitative Analysis in Sports 12(4), 151 to 165. | DOI 10.1515/jqas-2016-0098; arXiv:1609.09830 | Before asking whether a metric is relevant, check that it is reliable: discrimination (between-player variance against sampling variance, Section 2.1, Eq. 2) and stability (Section 2.2, Eq. 4), with sampling variance from a bootstrap over games (Section 3). They say their meta-metrics measure internal quality only, not relevance. | We went straight to relevance (a correlation with offensive rating) and never measured whether either side of that correlation is reliable. |
| Liu, Schulte (2018). Deep Reinforcement Learning in Ice Hockey for Context-Aware Player Evaluation. IJCAI 2018 (DOI 10.24963/ijcai.2018/478). | arXiv:1805.11088 | Validate a learned action-value metric with three checks: correlation with season success totals (Table 5), round-by-round consistency (Section 7.3), and future salary (Table 6). | Their target is season totals across players; ours is a per-game Spearman of a per-decision mean. |
| Sandholtz, Bornn (2020). Markov Decision Processes with Dynamic Transition Probabilities: An Analysis of Shooting Strategies in Basketball. Annals of Applied Statistics 14(3), 1122 to 1145. | DOI 10.1214/20-AOAS1348; arXiv:1812.05170 | Judge a decision policy by simulating the possession under the observed and an altered policy, on 2015-16 STATS tracking data (the same season), with the simulator checked against observed transition counts. | We judge single decisions by a one-step regret. They judge policies by simulated points per play, and do not validate against an outside outcome either. |
| Fujii, Takeuchi, Kuribayashi, Takeishi, Kawahara, Takeda (2022 to 2024). Estimating counterfactual treatment outcomes over time in complex multiagent scenarios. arXiv preprint. | arXiv:2206.01900 | Counterfactual passes in shot scenarios estimated with a recurrent multi-agent model, on 2015-16 STATS SportVU data (630 games), outcome = a binary "effective attack" label. | Treats the pass as a treatment with a causal estimator. Its table could not be read with confidence, so no number is taken from it. |

## 3. Benchmark

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| mean per-game Spearman 0.012 [-0.008, 0.029] out of fold, -0.014 [-0.116, 0.100] held out (reports/decisions.md) | none: no quantitative validation of EPV-Added or shot satisfaction against other statistics | Cervone et al. 2016 (Sections 5.3, 5.4) | 2013-14 data; different value model | not directly comparable: no published number |
| same | GIM correlation with season points 0.930, with points per game 0.749 (Table 5) | Liu and Schulte 2018 | NHL play-by-play, not tracking; season totals across players, so volume drives part of the correlation; our statistic is a within-game rank correlation of a per-decision mean that removes volume on purpose | not directly comparable: different sport, target, aggregation and statistic |
| same | none: VAEP player ratings are compared with goal and assist lists and market values descriptively, with no correlation reported | Decroos et al. 2019 | soccer event data | not directly comparable: no published number |
| same | not read: full text blocked | Jutamulia and Hosoi 2025 | unknown | cannot be compared; dropped |

No basketball paper we could check reports a validated correlation between a decision-quality metric and an outcome.
We do not invent one.

## 4. Measured weaknesses (from our reports only)

1. No validated signal: 0.012 [-0.008, 0.029] out of fold and -0.014 [-0.116, 0.100] held out; the shuffled statistic
   is of the same size (-0.013 and -0.069) (reports/decisions.md).
2. Options are on different scales. Shot and pass values are one-step expected points with hand-set factors, the
   drive value is a shot 8 ft nearer times 0.90, and the hold value is the EPV of the whole remaining possession
   (reports/decisions.md, Value-function assumptions). A regret between a one-step value and a whole-possession value
   is not a like-for-like comparison.
3. Decision kind may drive the player ranking. Held-out mean quality by kind is shot -0.159, pass -0.376, drive -0.528
   (reports/decisions.md, Held-out section): a gap of 0.37 between kinds. The 25 ranked players span only -0.256 to
   -0.329 in training (reports/decisions.md, Player table). So a player's mean may mostly reflect how often they shoot
   versus drive, not how well they choose. This is an inference from the report, not yet measured; decisions-E1
   measures it.
4. The validation target is noisy and shared. On-court offensive rating in one game is shared by every teammate on the
   floor at the same time, and the statistic is a within-game Spearman over players with as few as 3 decisions
   (reports/decisions.md, Validation). Its reliability was never measured.
5. The EPV used for the hold option is itself weak in the first seconds (reports/epv.md: 0 to 4 s MSE 1.2859
   against the constant's 1.2909), so early-possession hold values are close to a constant.

## 5. Recent work that targets these weaknesses

- Weakness 2 (different scales). POINTWISE (2014, Figure 1) and Fernandez et al. (2021, Eq. 2 to 4) value every option
  as success probability times the possession value after it, on one scale. VAEP (2019) values an action by the change
  in a state value before and after. Jutamulia and Hosoi (2025) do this for immediate pass and shot options in
  basketball (abstract only). HoopEval (MIT Sloan 2026, abstract only) learns action values by deep reinforcement
  learning on tracking data. This is decisions-E2.
- Weakness 3 and 4 (target and reliability). Franks et al. (2016) give the tools: discrimination and stability measure
  whether a metric separates players reliably before any relevance test, with game-level bootstrap variance. Liu and
  Schulte (2018) add a within-season consistency check. Neither supports per-game on-court rating as a target; Franks
  et al. say reliability is a precondition, not a validation. The literature we checked does not supply a
  better-founded outcome target for decision quality at the player level. It does support a decision-level check: an
  action-value model is validated through the calibration of its probability models (VAEP, Table 3; Fernandez et
  al., component calibration), which in our setting means: does the value of the option actually chosen predict the
  points that followed. Reliability and kind mix are decisions-E1; the decision-level check is the primary metric of
  decisions-E2.
- Counterfactual caution. Values of options not taken are only identified where similar states saw those options
  taken; off-policy evaluation surveys (Uehara, Shi, Kallus, arXiv:2212.06355, abstract only) treat this as the core
  difficulty. decisions-E2 therefore validates on chosen options and reports overlap, and does not claim the unchosen
  values are validated.

## 6. Pre-registered experiments

### decisions-E1: is the target reliable, and is the metric more than decision mix?

- Hypotheses, stated before any run: (H1) per-player on-court offensive rating, measured per game, has split-half
  reliability below 0.2 across the development games, so a per-game Spearman with it is close to zero whatever the
  decision metric does; (H2) the frozen per-player mean quality has a stability (Franks et al., Eq. 4, game-level
  bootstrap) of at least 0.5; (H3) at least half of the between-player variance of the frozen mean quality is
  explained by the player's shares of shot, pass and drive decisions.
- Baseline: the frozen decisions output, refit with frozen code (tools/, D45) on the development games. Nothing is fit
  in this experiment beyond a linear regression of player means on kind shares, which is descriptive.
- Metrics: (1) split-half reliability (odd versus even development games, players with at least 300 decisions in each
  half) of the frozen target and of the frozen metric; (2) Franks discrimination and stability, sampling variance by
  a bootstrap of games; (3) R squared of player mean quality on kind shares; (4) the same statistics for a
  kind-adjusted quality: regret minus the development-game mean regret of decisions of the same kind and court zone.
  These fit because they are the published preconditions for any relevance test (Franks et al., Sections 2 and 3).
- Adoption threshold (a protocol, not a model): if H1 holds, the lane report adds, next to the unchanged failed
  verdict, that the frozen validation could not have detected a signal; the kind-adjusted quality replaces the frozen
  mean in the player table only if its stability is at least 0.5 with the 95 percent interval above 0.3 and its R
  squared on kind shares is below 0.2. If H1 fails (the target is reliable), the failed verdict is read as evidence
  against the metric, and that is written down.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval, for the difference in stability between the
  kind-adjusted and the frozen metric.
- Ablation: kind-adjusted without the court-zone term.
- Data: development games for every statistic; the same statistics computed once on the D71 test set and reported next
  to them, flagged as low-powered if the set holds fewer than about 60 games.
- Compute and wall time (estimate, not measured): CPU, the Mac, under the heavy-job lock only for the frozen refit;
  the statistics run on the decision_events and decision_players tables in minutes.
- Risks and failure modes: split halves may differ in opponents; 300-decision gates leave few players per half
  (102 of 446 are under the gate for the whole season already, reports/decisions.md); a kind adjustment may remove
  real skill (a player who rightly shoots more). The test cannot rescue the failed verdict and is not allowed to.
- What a coach gains: knowing whether a decision table measures a player's choices or their role, before anyone uses
  it to judge a player.
- Backlog: expected gain 3 of 5 (it settles whether the failure says anything about the method, and protects the site
  from showing a role table as a skill table). Effort 1 of 5 (descriptive statistics on existing tables plus one
  frozen refit).

### decisions-E2: one value scale for every option, validated on the decisions actually made

- Hypothesis: valuing every option on the EPV scale (POINTWISE, Fernandez et al.) gives values whose chosen-option
  value predicts the points that followed, which the frozen hand-set values do less well. Option values: shot = points
  times the out-of-fold xFG probability; pass to teammate j = P(completion to j given the decision frame) times the EPV
  of a state with j holding the ball at the decision frame, plus P(failure) times 0; drive = P(drive keeps the ball)
  times the EPV of the state 1 s later on the drive, using the observed drive path only for the chosen drive and a
  model of that state for an unchosen one; hold = the EPV as now. Completion and drive models are fitted out of fold by
  game, under tools/.
- Baseline: the frozen option values, refit with frozen code on the development games.
- Primary metric: decision-level calibration. Regress realized points to end of possession on the chosen option's
  value, out of fold by game; report the slope (target 1) and the coefficient on regret in points_to_end on hold EPV
  plus regret. It fits because the chosen option's value is the only option value with an observed outcome, and this
  is the check the literature uses for action-value models (probability-model calibration in VAEP and Fernandez et
  al.). Secondary: decisions-E1's stability and kind R squared for the new metric; the frozen per-game Spearman,
  reported for continuity only and not gated.
- Adoption threshold: the regret coefficient's 95 percent interval above 0, the paired difference (new minus frozen)
  in that coefficient with interval above 0, and the chosen-value calibration slope inside 0.8 to 1.2, all on
  development games. Then scored once on the D71 test set.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval.
- Ablation: VAEP-style values for the chosen action only (EPV after minus EPV before), with no counterfactual options,
  to measure what the counterfactual part adds.
- Overlap report (pre-registered): the share of decisions where the best-valued option has an estimated choice
  probability below 0.05; unchosen-option values in those decisions are flagged as extrapolated.
- Data: development games only; frames read one game at a time from DATA_ROOT; scored once on the D71 test set.
- Compute and wall time (estimate, not measured): CPU on the Mac under the heavy-job lock, one game at a time; the
  frame pass of the frozen decisions run plus 5 folds of two small classifiers and EPV scoring of counterfactual
  states. Expect one to two working days of machine time. Depends on epv-E1 only if E1 is adopted (then the new EPV
  is used for every option, fixed before the run).
- Risks and failure modes: the counterfactual state for a pass (receiver holding the ball at the decision frame)
  ignores the ball's travel time, as Cervone et al. note for their own pass model (Section 3.3); EPV is weak early in
  the possession, so early decisions stay flat; confounding by unobserved play calls; if the regret coefficient is
  positive but tiny, the player table may still not be reliable (checked with E1's statistics).
- What a coach gains: a pass or a drive valued in the same points as the shot it replaced, checked against what
  actually happened after real decisions, before it is used to rate anyone.
- Backlog: expected gain 4 of 5 (it fixes the measured scale problem and gives the lane a validation it can pass or
  fail on hundreds of thousands of decisions instead of 17 or 603 game-level correlations). Effort 4 of 5 (new option
  models, counterfactual EPV states, a frame pass over all development games).

#### Amendment (PLAN.md D122, 2026-11-19 UTC, written before any development number or run)

Why: Set A (about 8,600 decisions in about 1,700 possessions) gives the regret coefficient a 95 percent half-width
of about 0.13 to 0.16 and barely reads the 0.8 to 1.2 slope band; development (half-width about 0.02) decides, as
this pre-registration already says. D122 fixes the Set A reading.

- Decision (development games only, as written above): the regret coefficient's interval above 0, the paired
  difference (new minus frozen) in that coefficient with its interval above 0, and the chosen-value calibration slope
  inside 0.8 to 1.2.
- Set A (D122), scored once only if development adopts, read against the bar and not against the development
  estimate: "confirms" if the regret coefficient's interval (bootstrap of the 10 games) lies above zero;
  "contradicts" if it lies below zero, which stops the adoption; otherwise "inconclusive at 10 games". Beside it,
  report only: the slope, the paired difference, and a t interval on per-game values. If development does not
  adopt, Set A is not scored.

Commitments fixed with this amendment (code: tools/research_decisions_e2.py), before any number. Items 3, 5, 6 and 7
read the pre-registration's words in a definite way; item 4 departs from them, and says why.

1. Decisions and baseline: the frozen decisions of the development games and their frozen option values, from the
   decisions-E1 gate (tools/research_decisions.py: the frozen lane refit with frozen code, reproduced exactly). Both
   arms value the same decisions. The frozen detector finds completed passes only (a teammate's holding segment
   starts within 1.5 s), so pass decisions are selected on completion in both arms; this is stated, not repaired.
2. Outcome: points_to_end of the possession's last 1 Hz state at or before the decision frame (the frozen EPV
   target, which subtracts points logged before the state); a decision with no such state is left out of both arms.
3. EPV, out of fold: the frozen EPV sequence model refit on each of the lane's 5 grouped folds of the development
   games (seed 0) with the frozen functions (fold_features, lagged, fit_sequence), its inner isotonic map left out
   (it changes P(points > 0), not expected points). Gate per fold: its expected points on the fold's stored states
   equal the stored out-of-fold epv_points within 1e-5. Every EPV of a decision in game g uses the fold model that
   did not see g.
4. One value scale, every option valued at the decision frame from what is known then:
   - hold: the EPV of the actual state built at the decision frame (the state row of the frozen builder, at that
     frame), scored as the last state of a sequence made of the possession's stored 1 Hz states before it;
   - pass to teammate j: P(completion to j) times the EPV of the same state with j as handler and the ball at j's
     position (Cervone et al.'s travel-time caveat applies);
   - drive: P(drive keeps the ball) times the EPV of a modelled state 1 s later: the handler and the ball moved 8 ft
     toward the basket along the line to it (the frozen drive geometry, never nearer than 3 ft), everyone else where
     they are, the clocks advanced 1 s. The chosen drive uses the same modelled state, not its observed path: the
     observed path would carry 1 s of the outcome into the chosen value that the primary metric regresses on the
     outcome. This departs from the pre-registration's wording for the chosen drive;
   - shot: the frozen shot value (points times the xFG lane's out-of-fold probability for a taken shot; for a
     decision that is not a shot, the frozen location-only value with its contest term, since the xFG lane predicts
     taken shots only).
   Chosen value = the value of the option taken; regret = chosen minus the best of shot, every pass, hold and drive.
5. Completion model: pass attempts detected from frames (any player's holding segment, the frozen thresholds 2.5 ft,
   8 ft, 3 frames): an offensive holder's segment followed within 1.5 s, with no shot release, by a teammate's
   (completed) or a defender's (failed; the intended receiver is the offensive teammate nearest the ball where the
   defender took it). Attempts with no next holder within 1.5 s are left out and counted. A logistic regression
   (standardised) on pass length, the receiver's nearest-defender distance, the smallest defender distance to the
   passing lane, the number of defenders within 3 ft of the lane and the receiver's distance to the basket, fitted
   on the training folds' attempts.
6. Drive model: the frozen drive decisions, labelled kept unless a defender holds the ball or the possession ends
   with a turnover within 1 s; a logistic regression on the handler's speed toward the basket, distance to the
   basket, nearest-defender distance, defenders within 6 ft between the handler and the basket, and the shot clock,
   fitted on the training folds' drives and applied to every decision.
7. Overlap report (report only): a multinomial logistic choice model of the decision kind (shot, pass, drive) on the
   decision-frame features above; the share of decisions whose best option's kind has a choice probability below
   0.05. Hold is not a decision kind and has no choice probability; decisions whose best option is hold are counted
   apart.
8. Primary metric, per arm, on the decisions with every value finite in both arms: the slope of points_to_end on
   the chosen value (least squares with an intercept); the coefficient on regret in points_to_end on the arm's hold
   value plus regret. Intervals: a bootstrap of games, 2000 resamples, seed 0, of the fixed per-decision values
   (no value is refit in a resample). The paired difference resamples the same games for both arms.
9. Ablation (pre-registered, report only): the change in EPV over the chosen action, the EPV of the actual state 1 s
   after the decision frame minus the EPV at it (same fold model); its coefficient in points_to_end on hold plus
   that change. It uses 1 s of the future by design.
10. Compute: frames read one game at a time from DATA_ROOT; development games only; every job under the lock,
    resumable, under 30 minutes; one fold measured first. Set A, if scored: the EPV, completion, drive and choice
    models refit on every development game and loaded before score_once; a clean tree; the commit and the hashes of
    the gate records, the development record and the models recorded; refusals (a test game without frames,
    possessions, states or a decision; any non-finite value) give a reason and no number.

#### Amendment to commitment 5 (PLAN.md D123, 2026-11-19 UTC, before any development number)

The smoke run on two development games (01.01.2016.CHA.at.TOR and 01.01.2016.DAL.at.MIA) showed that the failed
label of commitment 5 does not mark failed passes. A defender's holding segment after an offensive holder's labelled
177 and 167 attempts failed (a completion rate near 75 percent). In those attempts the defender was the player
nearest the ball for a median 0.28 s, starting a median 0.08 s after the holder's last frame (0.48 s for completed
passes); the offence held the ball again later in the possession in 57 and 62 percent of them; and 18 and 9 of them
ended the possession in a turnover with no offensive hold again. They are tight on-ball defence or a dribble near a
defender, not failed passes. No outcome metric had been computed.

Commitment 5 now reads: an attempt is completed when the next holder within 1.5 s is a teammate; failed when the
next holder is a defender, the offence never holds the ball again in the possession, and the possession ends in a
turnover; every other attempt is left out and counted, by game. The report states the completion rate this label
gives on development and the excluded count. The frame pass records the raw fields; the label is applied before the
completion model is fitted. Nothing else changes.

## 7. Sources checked (all fetched 2026-11-16)

- Cervone et al. 2014, POINTWISE, PDF (pages 1 to 3 read): https://raw.githubusercontent.com/papers-we-love/papers-we-love/refs/heads/main/sports_analytics/2014-ssac-pointwise-predicting-points-and-valuing-decisions-in-real-time.pdf
- Cervone et al. 2016: https://arxiv.org/abs/1408.0777 ; full text https://ar5iv.arxiv.org/html/1408.0777
- Jutamulia and Hosoi 2025, record: https://lida.sport-iat.de/dbb/Record/4094899?lng=en ; DOI landing https://doi.org/10.1177/22150218251325018 (publisher returned HTTP 403; full text not read)
- Jutamulia 2021, MIT thesis record (predecessor, abstract only): https://dspace.mit.edu/handle/1721.1/139205
- Decroos et al. 2019, VAEP: https://arxiv.org/abs/1802.07127 ; full text https://ar5iv.arxiv.org/html/1802.07127
- Fernandez et al. 2021: https://arxiv.org/pdf/2011.09426 (Sections 1 to 3 read)
- Franks et al. 2016: https://arxiv.org/abs/1609.09830 ; full text https://ar5iv.arxiv.org/html/1609.09830 ; bibliographic record https://econpapers.repec.org/RePEc:bpj:jqsprt:v:12:y:2016:i:4:p:151-165:n:3
- Liu and Schulte 2018: https://arxiv.org/abs/1805.11088 ; full text https://ar5iv.arxiv.org/html/1805.11088
- Sandholtz and Bornn 2020: https://arxiv.org/abs/1812.05170 ; full text https://ar5iv.arxiv.org/html/1812.05170
- Fujii et al.: https://arxiv.org/html/2206.01900v4 (basketball section read; venue not confirmed in the paper)
- Uehara, Shi, Kallus, A Review of Off-Policy Evaluation in Reinforcement Learning: https://arxiv.org/abs/2212.06355 (abstract only)
- Wang et al. 2026, HoopEval: https://www.sloansportsconference.com/research-papers/hoopeval-individual-player-action-evaluation-via-deep-reinforcement-learning (abstract only)

Dropped because they could not be checked at the source: every number and method detail of Jutamulia and Hosoi 2025
(publisher blocked); Q-Ball (Yanai et al., AAAI 2022), whose PDF was blocked by a bot wall, so it is not cited; the
numbers of Fujii et al. Table II, whose column alignment could not be read reliably.
