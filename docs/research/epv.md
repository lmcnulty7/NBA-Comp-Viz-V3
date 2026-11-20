# Research dossier: epv (expected possession value)

- Lane: epv (analytics/epv, frozen at tag freeze-v1)
- Date: 2026-11-16
- Verdict: works, but narrowly. The GRU beats a constant by 5.2 percent out of fold and 7 percent on held-out games, and
  the first 4 seconds of a possession are under-confident on held-out games (slope 1.23). Most of what a possession
  scores is not visible in our 1 Hz state summary.

## 1. What we built

- Question: expected points from a 1 Hz possession state to the end of the possession (reports/epv.md).
- Data: 2015-16 SportVU, 1739119 states from 604 training games, 44703 states from 17 held-out games (reports/epv.md).
- Inputs: 25 hand-built features per state (ball and handler position, five offence-defence distances, hull area,
  shot clock, elapsed time, handler and lineup priors), over the last 5 states (reports/epv.md, Protocol).
- Model: torch GRU (hidden 16) over the last 5 states, kept over gradient boosting by 0.0212 [0.0201, 0.0228] MSE
  and over the MLP stand-in (reports/epv.md; reports/epv_mlp_standin.md, MLP stand-in MSE 1.2218).
- Out of fold: MSE 1.2155 [1.2080, 1.2226] against the constant's 1.2822; Brier P(points > 0) 0.2350; slope 0.964.
- Held out: MSE 1.2362 [1.1899, 1.2812] against the constant's 1.3319; Brier 0.2318; slope 1.024 (reports/epv.md).
- Shuffled-target check: lift 0.0 percent, so the real lift is not chance (reports/epv.md).

## 2. Foundations

| paper | link | what we take from it | how ours differs |
| --- | --- | --- | --- |
| Cervone, D'Amour, Bornn, Goldsberry (2014). POINTWISE: Predicting Points and Valuing Decisions in Real Time with NBA Optical Tracking Data. MIT Sloan Sports Analytics Conference, Research Paper Competition. | PDF (Papers We Love archive): https://raw.githubusercontent.com/papers-we-love/papers-we-love/refs/heads/main/sports_analytics/2014-ssac-pointwise-predicting-points-and-valuing-decisions-in-real-time.pdf | The definition we use: EPV = E[points to end of possession given the state] (page 2). | They model the ball handler's next action (pass, shot, turnover as macrotransitions, movement as microtransitions) and sum over futures (page 3, Eq. 1); we regress realized points directly on a state summary. They used 2012-13 data. |
| Cervone, D'Amour, Bornn, Goldsberry (2016). A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes. Journal of the American Statistical Association 111(514). | DOI 10.1080/01621459.2016.1141685; arXiv:1408.0777 | The full model: competing-risk hazards for six macrotransitions, hierarchical spatial shrinkage across players, 25 Hz STATS data from 2013-14, 90/10 split (Sections 2.2, 3.2, 5.1). | Our model is a direct regression with no transition structure and no player-level spatial effects beyond two priors. They report that without shrinkage the model can do worse than one with no spatial effects (Section 5.1). |
| Sicilia, Pelechrinis, Goldsberry (2019). DeepHoops: Evaluating Micro-Actions in Basketball Using Deep Feature Representations of Spatio-Temporal Data. arXiv working paper (secondary indexes list it at KDD 2019; the ACM page was not checked). | arXiv:1902.08081 | Learn from raw positions: a stacked LSTM over 128 moments (about 5 s) of all 10 players and the ball, predicting the terminal action class (shot attempt, shooting foul, non-shooting foul, turnover, null), with expected points as the probability-weighted value of the classes (Sections 3, 4). | We feed 25 hand features at 1 Hz and predict points directly; they predict terminal classes and attach league-average point values. 2016-17 data, about 750 games. |
| Fernandez, Bornn, Cervone (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning (DOI 10.1007/s10994-021-05989-6). | arXiv:2011.09426 | EPV decomposed by the law of total expectation into action selection times expected value given the action, with the shot term equal to an expected-goals model (Section 3.2, Eq. 1 to 4). | Soccer, not basketball; we use no decomposition and train on realized outcomes, including the make or miss of every shot. |
| Alcorn, Nguyen (2021). baller2vec: A Multi-Entity Transformer For Multi-Agent Spatiotemporal Modeling. arXiv preprint. | arXiv:2102.03291 | A transformer that attends across all players and time steps with no fixed player order; same 2015-16 SportVU season (631 games, 5 Hz, 4 s inputs). Ablations: all 10 players cut trajectory NLL by 18.0 percent over one player, player identity by a further 4.4 percent (Table 3). | It models trajectories, not possession value. It shows what raw multi-agent input adds over a summary, which is the question our narrow lift raises. |

## 3. Benchmark

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| MSE 1.2155 out of fold, 1.2362 held out (reports/epv.md) | none: no EPV MSE or calibration is reported; Table 1 gives out-of-sample log-likelihoods of the macrotransition models only (Section 5.1) | Cervone et al. 2016 | 2013-14 STATS data, 25 Hz, random 90/10 split; they score transition models because EPV is never observed (Section 5.1) | not directly comparable: no published number for the same quantity |
| Brier skill of P(points > 0) over the constant: 0.054 out of fold, 0.069 held out (derived from reports/epv.md: 1 - 0.2350/0.2483 and 1 - 0.2318/0.2491) | Brier skill score 0.2686 at K = 2 (Table 2) | Sicilia et al. 2019 (DeepHoops) | 5-class terminal action in the next moments with a 0.64 s blind spot, against our binary "points > 0 by the end of the possession"; training rows downsampled (K null windows per possession), which changes the base rate; 2016-17, about 750 games, 75/10/15 split | not directly comparable: different target, horizon, base rate and season |
| MSE lift over the constant 5.2 percent out of fold (reports/epv.md) | none | Fernandez et al. 2021 | soccer, goal within possession, component-wise calibration only | not directly comparable |
| none | trajectory NLL 0.492 (Table 2) | Alcorn and Nguyen 2021 | trajectory modelling, not value | not directly comparable: different task; listed only because it uses the same season |

No paper we could check reports an expected-points error for basketball possessions. We do not invent one.

## 4. Measured weaknesses (from our reports only)

1. Narrow gain: MSE 1.2155 against the constant's 1.2822 out of fold, a 5.2 percent lift; 7.2 percent on held-out games
   (1.2362 against 1.3319) (reports/epv.md). Gradient boosting alone gets 3.5 percent.
2. The first seconds carry almost no signal: in the 0 to 4 s bucket the GRU's MSE is 1.2859 against the constant's
   1.2909 out of fold; 434252 of 443333 states sit in the 0.4 to 0.6 probability bin (reports/epv.md, calibration
   table).
3. First-seconds calibration differs between samples: slope 0.916 out of fold, 1.232 held out, outside the 0.9 to 1.1
   target (reports/epv.md, Stage 8 table). The held-out set is 16 games at one arena plus one other game, so this may be
   a team effect; with 17 games it is a point estimate.
4. Held-out MSE is higher in every elapsed-time bucket (reports/epv.md). Every held-out interval contains the
   out-of-fold point.
5. The training target is the realized points of the possession, so every shot's make or miss enters the label
   (reports/epv.md, Protocol: target points_to_end).

## 5. Recent work that targets these weaknesses

- Weakness 1 and 5 (noisy target, small lift). The decomposition in Fernandez et al. (2021, Section 3.2) and the
  terminal-class design of DeepHoops (2019, Section 3) both value a shot through a shot-quality model rather than
  through its make or miss. In our setting this replaces about one coin flip per shot in the training label with
  the xFG lane's out-of-fold probability, a lower-variance target with the same expectation if xFG is calibrated.
  xFG is calibrated out of fold (ECE 0.0029, slope 1.000, reports/results.md, xfg section). This is experiment epv-E1.
- Weakness 1 and 2 (state summary may discard information). baller2vec (2021, Table 3) shows that attending over all
  10 players with identities beats a one-player input by a wide margin for trajectories on our exact season.
  DeepHoops learns its features from raw positions of all players. HoopEval (Wang et al., MIT Sloan 2026) uses deep
  reinforcement learning on tracking data to value actions; only its abstract was checked, so nothing beyond the
  approach is taken from it. This is experiment epv-E2.
- Not proposed: a full Cervone-style multiresolution model. It attacks the same weakness but its fit took up to 18
  hours per player on 461 processors (Section 4.3); it does not fit a 16 GB Mac.

## 6. Pre-registered experiments

### epv-E1: train on expected shot value instead of the make or miss

- Hypothesis: replacing the realized points of possessions that end in a field-goal attempt with the expected points of
  that attempt (out-of-fold xFG probability times 2 or 3; any free-throw points the frozen target counts stay
  realized) lowers the out-of-fold MSE of the GRU against realized points, and moves the 0 to 4 s calibration slope
  toward 1.
- Baseline: the frozen GRU, refit with frozen code (tools/, D45) on the development games, same features, folds and
  seeds.
- Change: the training label only. Features, architecture, epochs rule and isotonic step are frozen. Evaluation is
  always against realized points, never against the smoothed label.
- Metric: MSE of expected points against realized points to end of possession, out of fold, 5 grouped folds by game.
  It fits because it is the frozen lane's metric and the quantity a coach reads. Secondary, reported not gated:
  Brier P(points > 0), calibration slope per elapsed-time bucket.
- Adoption threshold: paired difference MSE(baseline) minus MSE(E1) with its 95 percent interval above 0 on the
  development games, and the 0 to 4 s slope inside 0.9 to 1.1 out of fold. Then scored once on the D71 test set and
  reported whatever the result.
- Paired test: bootstrap by game, 2000 resamples, 95 percent percentile interval.
- Ablation: smooth only 3-point attempts (the noisiest label), keep 2-point makes and misses as realized.
- Leakage guard (pre-registered): the xFG probability used as a label for a training row must come from an xFG model
  that never saw that row's game and never saw any game of the EPV outer test fold. The frozen xFG out-of-fold folds
  are not the EPV folds, so xFG is refit inside each EPV outer fold with frozen xFG code (tools/), training games
  only. If the refit does not reproduce the frozen xFG out-of-fold log loss within its interval
  (tools.frozen_common.require_passed), the experiment stops.
- Data: development games only for fitting and selection; scored once on the D71 test set.
- Compute and wall time (estimate, not measured): CPU on the 16 GB Mac under the heavy-job lock. 5 xFG refits plus 5
  GRU fits plus the baseline refit; the inputs are the existing states_1hz table, read one game at a time. Expect
  under one working day.
- Risks and failure modes: xFG miscalibration on some shot types moves bias into the label; possessions with fouls or
  offensive rebounds after a miss need a rule fixed before the run (pre-registered: smooth only the final field-goal
  attempt of the possession, and only when the possession ends on that attempt); the gain may be real but tiny, since
  the evaluation target keeps all its noise.
- What a coach gains: a possession value that moves less with lucky or unlucky shots, so a state is rated by the shot it
  tends to produce, not by whether that shot fell.
- Backlog: expected gain 2 of 5 (the shuffled check and the 0 to 4 s table say most outcome noise is irreducible; a
  label with less noise can only help estimation, not remove that noise). Effort 2 of 5 (one new label column, refits
  with frozen code, no new data).
- Amendment (PLAN.md D117, 2026-11-19, written before any development number): the D71 test set is Set A, 10 games of
  one date (about 25,000 to 30,000 states), which cannot decide the paired MSE difference (projected half-width about
  0.010 against a plausible effect of a few thousandths) or the 0 to 4 s calibration slope; the 604 development games
  can (paired half-width about 0.0013). Development decides, as pre-registered: the paired difference MSE(baseline)
  minus MSE(E1) with its 95 percent interval above 0, and the 0 to 4 s slope inside 0.9 to 1.1, out of fold, with the
  full pre-registered ablation. Two gates come first: the baseline GRU refit out of fold reproduces the frozen
  out-of-fold record, and the xFG refit inside each EPV fold reproduces the frozen xFG out-of-fold log loss within its
  interval. Set A is not scored whatever development shows; no Set A path is written.

### epv-E2: multi-entity attention over raw positions

- Hypothesis: a small permutation-invariant transformer over the raw positions of all 10 players and the ball for the
  last 4 s at 5 Hz (baller2vec-style attention across players and time, players unordered within a team), with the
  25 frozen features appended, lowers out-of-fold MSE against realized points versus the frozen GRU, with most of the
  gain in the 0 to 4 s bucket.
- Baseline: the frozen GRU refit with frozen code on the development games.
- Metric: MSE against realized points, out of fold, grouped by game; secondary per-bucket MSE and slope.
- Adoption threshold: paired MSE difference interval above 0 on development games, and a gain of at least 0.0062,
  the GRU's own margin over the MLP stand-in (reports/epv.md), so that a new architecture must beat the frozen one by
  at least the step that justified the GRU. Then scored once on the D71 test set.
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval.
- Ablation: the same transformer without player identity embeddings (baller2vec's identity ablation, Table 3).
- Data: development games only; the raw frames come from DATA_ROOT one game at a time and are exported as per-game
  shards; scored once on the D71 test set.
- Compute and wall time (estimate, not measured): export on the Mac under the lock, one game at a time; training on a
  Colab GPU. About 1.7 million samples of 20 time steps by 11 entities by 3 coordinates is about 4.6 GB in float32, so
  shards, not one array. A model of about 1 million parameters (baller2vec reports about 19 million parameters and about 7 days on one
  GTX 1080 Ti) is planned at 5 folds of a few GPU hours each. Colab uploads follow the per-game tar rule.
- Risks and failure modes: overfitting player identity to 604 games; the gain may come only from features the
  baseline could have had (to check: the ablation and a GRU given the same raw inputs); Colab data loss on upload;
  long wall time makes the single test-set score expensive to redo, so the run is gated on E1 finishing first.
- What a coach gains: value that reacts to off-ball spacing and cuts in the first seconds, where the frozen model is
  close to a constant.
- Backlog: expected gain 3 of 5 (the 0 to 4 s bucket is where a richer state could matter; there is no published
  basketball EPV number to say how much). Effort 5 of 5 (new data export, GPU training, new model code under tools/, the Colab pipeline).

## 7. Sources checked (all fetched 2026-11-16)

- Cervone et al. 2016, arXiv abstract page: https://arxiv.org/abs/1408.0777
- Cervone et al. 2016, full text (ar5iv HTML of arXiv:1408.0777): https://ar5iv.arxiv.org/html/1408.0777
- Cervone et al. 2014, POINTWISE, PDF: https://raw.githubusercontent.com/papers-we-love/papers-we-love/refs/heads/main/sports_analytics/2014-ssac-pointwise-predicting-points-and-valuing-decisions-in-real-time.pdf
- POINTWISE record: https://paperswelove.org/papers/pointwise-predicting-points-and-valuing-decisions--0b9314e8/
- Sicilia et al. 2019, DeepHoops, abstract: https://arxiv.org/abs/1902.08081 ; full text: https://ar5iv.arxiv.org/html/1902.08081
- Fernandez et al. 2021, arXiv PDF: https://arxiv.org/pdf/2011.09426 (Sections 1 to 3 read); DOI 10.1007/s10994-021-05989-6 (publisher page asked for a login, not read)
- Alcorn and Nguyen 2021, baller2vec, abstract: https://arxiv.org/abs/2102.03291 ; full text: https://ar5iv.arxiv.org/html/2102.03291
- Alcorn and Nguyen 2021, baller2vec++, abstract: https://arxiv.org/abs/2104.11980 (abstract only; not used for any number)
- Wang et al. 2026, HoopEval, conference page: https://www.sloansportsconference.com/research-papers/hoopeval-individual-player-action-evaluation-via-deep-reinforcement-learning (abstract only)
- Sa-Freire et al. 2025, GraphEIV, MLSA 2024 (CCIS 2460, pp. 29 to 40, DOI 10.1007/978-3-031-86692-0_3), record: https://lida.sport-iat.de/dbb/Record/4095579?lng=en (record only; a GNN decomposition into xFG, next action, receiver and turnover models; no numbers taken)

Dropped: no claim relies on GraphEIV, HoopEval or baller2vec++ beyond their abstracts; no EPV accuracy number is quoted
from any paper because none reports one for the same quantity.
