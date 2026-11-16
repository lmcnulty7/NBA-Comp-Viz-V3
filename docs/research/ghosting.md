# Research dossier: ghosting

- Lane: ghosting (where an average defender would have been), reports/ghosting.md and reports/ghosting_torch.md.
- Date: 2026-11-16.
- Verdict: the stand-in works and holds on held-out games, but it is a one-step-trained MLP with one second of
  history, while the literature trains on rolled-out trajectories with longer history; no published number is
  directly comparable to ours.

## 1. What we built

- Question: a defender's position 0.2 s and 1.0 s ahead, from the last five sampled frames (5 Hz, so 1 s of history)
  of everyone on the floor (reports/ghosting.md, Protocol).
- Data: 2015-16 SportVU, 604 training games; fitted at 1 Hz on 7147100 rows (every 5th step), scored at 5 Hz on
  35500180 defender-steps; 17 held-out games, 896885 defender-steps (reports/ghosting.md).
- Model: sklearn MLP 64x64 on the 5-step window of 31 features, target the one-step displacement; the 1.0 s ghost
  is rolled forward five steps with its own position fed back and everyone else actual (reports/ghosting.md).
- Numbers: 1.0 s error 3.657 [3.635, 3.682] ft out of fold, 3.417 held out; persistence 5.408 and 5.250; 0.2 s
  error 0.437 [0.433, 0.442] and 0.377 (reports/ghosting.md, Stage 8 table; reports/results.md, ghosting).
- The torch GRU (hidden 32) on the same window was worse than the MLP by 0.111 ft [0.085, 0.138] at 1.0 s and was
  not selected (reports/ghosting_torch.md).
- Validation: deviation from the ghost against points allowed, sign test 346 of 604 games positive (p = 0.0002)
  out of fold, 13 of 17 (p = 0.0245) held out (reports/ghosting.md).

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Le, H. M., Carr, P., Yue, Y., Lucey, P. (2017). Data-Driven Ghosting using Deep Imitation Learning. MIT Sloan Sports Analytics Conference, Boston, 3 to 4 March 2017. | https://authors.library.caltech.edu/records/74xh3-ysx85 (PDF: files/1671-2.pdf) | The idea of a league-average ghost and of team-specific ghosts by fine-tuning. Theirs: soccer, 100 games, 10 Hz, about 17400 sequences, a two-layer LSTM with 512 units trained with imitation learning to control drift over long rollouts. Ours: basketball, an MLP on 1 s of history, trained on one step only, scored over a 1 s rollout. |
| Le, H. M., Yue, Y., Carr, P., Lucey, P. (2017). Coordinated Multi-Agent Imitation Learning. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1995-2003. | arXiv:1703.03121; https://proceedings.mlr.press/v70/le17a/le17a.pdf | The training procedure: roll the policy out over a horizon that grows from 1 to 10 steps so that training sees its own errors (section 4.2), and a latent role model for coordination. Theirs: soccer, 45 games, all ten outfield defenders rolled out jointly. Ours: one defender, others actual, no rollout in training. |
| Seidl, T., Cherukumudi, A., Hartnett, A., Carr, P., Lucey, P. (2018). Bhostgusters: Realtime Interactive Play Sketching with Synthesized NBA Defenses. MIT Sloan Sports Analytics Conference. | https://www.sloansportsconference.com/research-papers/bhostgusters-realtime-interactive-play-sketching-with-synthesized-nba-defenses | The basketball version of ghosting: five role LSTMs, trained in three phases (one-step pre-training, single-policy rollout with other players actual, then joint rollout), and ensembles of 10 runs, which had lower error than single models (section 2.3). Ours stops at the first phase. |
| Felsen, P., Lucey, P., Ganguly, S. (2018). Where Will They Go? Predicting Fine-Grained Adversarial Multi-Agent Motion using Conditional Variational Autoencoders. ECCV 2018, pp. 732-747. | https://openaccess.thecvf.com/content_ECCV_2018/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf | The closest basketball forecasting setup: predict defenders given the offense's actual future motion, 2015-16 STATS data, 5 Hz. Also: velocity extrapolation is the strong simple baseline, and role alignment helps. Theirs: CVAE, 4 s history, all five defenders jointly. Ours: deterministic MLP, 1 s history, one defender. |

## 3. Benchmark

Our numbers converted to metres where the paper uses metres: 3.657 ft = 1.115 m, 3.417 ft = 1.042 m.

| our number | published number | paper | what differs | comparability |
| --- | --- | --- | --- | --- |
| 1.0 s error 3.657 ft out of fold, 3.417 ft held out (reports/ghosting.md) | defense only, 1 s horizon, CVAE-role-team: 2.08 ft average distance (Table 2b) | Felsen et al. 2018 | Theirs averages the error over every step of the 1 s horizon, ours is the error at the 1.0 s point; they predict all five defenders jointly, we predict one with the other four actual; 4 s history against 1 s; 1247 games of 2015-16 STATS data against our 604 games of the public release; their sequences hold one team on offense and end in a shot, turnover or foul; their test split is not ours. | Not directly comparable: different error definition (average over the horizon against endpoint) and different conditioning. |
| 1.0 s error 3.657 ft out of fold | league-average ghost, error over the first 2 s: 8.22 ft per player; San Antonio team ghost 6.59 ft (section 2.3) | Seidl et al. 2018 | 2016-17 season; joint rollout of five defenders that feed each other's predictions; 2 s horizon; scored only on 93 test possessions with San Antonio on defense. | Not directly comparable: longer horizon, joint rollout, one team's possessions. |
| 1.0 s error 1.115 m out of fold | GroupNet + CVAE at 1.0 s: minADE20 0.34 m, minFDE20 0.48 m (Table 2) | Xu et al. 2022 (section 6) | Best of 20 sampled futures against one deterministic prediction; all 11 entities forecast with no future context against one defender given everyone else actual; 2 s of history at 2.5 Hz (5 steps for 2 s); different games and split. | Not directly comparable: best-of-20 metric and different conditioning. |

No paper we found reports a single deterministic defender error at a 1.0 s endpoint on SportVU with the other nine
players and the ball given, so there is no directly comparable published number. Le et al. (Sloan 2017) report an
average deviation of about 4 m in soccer over long rollouts (section 3.1); that is a different sport and horizon and is
not used as a benchmark.

## 4. Measured weaknesses (from our reports)

1. The model is trained on the one-step displacement only, yet scored after a 5-step rollout: error grows from
   0.437 ft at 0.2 s to 3.657 ft at 1.0 s (reports/ghosting.md). Both Le et al. (ICML 2017) and Seidl et al. (2018)
   train on rollouts for this reason.
2. History is five sampled frames, 1 s. A recurrent model over the same five steps did not help (GRU worse by
   0.111 ft [0.085, 0.138] at 1.0 s, reports/ghosting_torch.md), so the limit may be the input, not the cell type.
3. The deep model is fitted on every 5th step (1 Hz, 7147100 of 35500180 rows) for memory (reports/ghosting.md).
   The torch run needed 55 min and a 23 GB peak (reports/ghosting_torch.md), above the Mac's 16 GB.
4. The floor is static persistence (5.408 ft); no constant-velocity floor is reported, though Felsen et al. found
   velocity extrapolation the strongest simple baseline. Our gain over a moving floor is not measured.
5. The link to outcomes is a sign test over games (346 of 604, 13 of 17); its effect size is not reported
   (reports/ghosting.md).
6. Only one defender is rolled out; the other defenders stay actual (reports/ghosting.md, Assumptions), so the
   ghost is not a whole average defense. Its assignment (nearest attacker at the first frame) differs from the
   matchups lane's Hungarian assignment (reports/matchups.md, Protocol).

## 5. Recent work aimed at these weaknesses

- Rollout training (weakness 1): Le et al. 2017 (ICML, section 4.2) grow the rollout horizon from 1 to 10 steps;
  Seidl et al. 2018 (section 2.2) train single-policy rollouts with the other players actual, which is exactly our
  scoring setup. This is the cheapest fix and it is aimed at the measured error growth.
- More history and all-agent context (weakness 2): baller2vec (Alcorn and Nguyen, arXiv:2102.03291, 2021), a
  multi-entity transformer on the same 2015-16 SportVU season at 5 Hz, reports that using all 10 players improved
  NLL by 18.0 percent over one player, and that full sequences were 70.6 percent better than the first frame only
  (Table 3, section 4.2). Felsen et al. (Table 2a) found 4 s of history best, though only marginally. TranSPORTmer
  (Capellera et al., ACCV 2024, arXiv:2410.17785) uses set attention blocks so the model is permutation
  equivariant over agents, which removes our fixed slot order of offence and defence features. baller2vec also
  gains 4.4 percent from player identity; we would not use defender identity, since the ghost must stay an
  average defense.
- Joint rollout (weakness 6): Seidl et al. 2018 and baller2vec++ (arXiv:2104.11980) model coordinated agents. Not
  proposed now: it changes what the ghost means and needs weaknesses 1 and 2 settled first.
- Broadcast imputation (TranSPORTmer, imputation tasks) targets the broadcast pipeline, not this SportVU lane, and is
  not proposed here.

## 6. Pre-registered experiments

Shared rules: development games only (the 2015-16 training games that remain after the D71 test set is fixed); 5
grouped folds by game for out-of-fold numbers; paired bootstrap by game, 2000 resamples, 95 percent percentile
interval, on the per-game mean error difference; the result is scored once on the D71 test set after the decision;
nothing is fitted on the 17 spent held-out games, which are never scored again. No player identity is an input.

### ghosting-E1: rollout training of the same network

- Hypothesis: training the 64x64 window network on the error of its own 5-step rollout (horizon grown 1, 2, 3, 4, 5
  steps; own prediction fed back, everyone else actual, as at scoring time) lowers the 1.0 s error against the same
  network trained on one step.
- Baseline: the frozen MLP stand-in, refit with frozen code on the development games (tools/, D45 rule).
- Metric: mean position error at 1.0 s (ft), out of fold. It is the lane's selection metric and the quantity the
  rollout objective targets. Guard: the 0.2 s error. Reference only: a constant-velocity floor and persistence.
- Adoption threshold: 1.0 s error lower by at least 0.10 ft with the paired interval below zero; the 0.2 s error not
  worse by more than 0.02 ft (upper interval bound below +0.02); the deviation sign test still positive with
  one-sided p < 0.05 on development games. 0.10 ft is close to the GRU gap that decided stage 6 (0.111 ft) and
  about four times the half-width of the MLP's interval.
- Ablation: the same network in torch trained on the one-step loss with the same optimizer and budget. It
  separates the rollout objective from any framework or optimizer effect.
- Compute: torch on CPU, the 1 Hz fit rows as float32 (about 7.1M rows x 166 columns, about 4.7 GB), streamed one
  game at a time under the heavy-job lock. Estimate: 1 to 2 h per fold on the Mac, 6 to 10 h in all; or about 2 h
  on a Colab GPU. Estimates, not measured.
- Risks and failure modes: with only 5 steps the compounding may be small and the gain below threshold; rollout
  training can shrink motion toward the mean (lower error, less lifelike paths); a ghost that tracks the actual
  defender more closely shrinks the deviation and can weaken the sign test; memory (stage 6 peaked at 23 GB).
- What a coach gains: the 1 s ghost path sits closer to where an average defender goes, so the "should have been
  here" marker flags fewer false misses.
- Backlog: expected gain 3 (aimed at the measured 0.2 s to 1.0 s error growth, and the standard fix in both
  ghosting papers; the short horizon limits the size). Effort 2 (same features, network and scoring; the new piece
  is a differentiable rollout that recomputes the defender's own relative features).

### ghosting-E2: multi-entity attention with 4 s of history

- Hypothesis: a permutation-equivariant attention model over the 10 players and the ball with 4 s of history (20
  steps at 5 Hz) lowers the 1.0 s error against the frozen MLP with 1 s of history.
- Baseline: the frozen MLP stand-in refit with frozen code on the development games.
- Metric: mean position error at 1.0 s (ft), out of fold, as in E1; 0.2 s error as guard.
- Adoption threshold: as E1 (at least 0.10 ft lower with the interval below zero; 0.2 s not worse by more than
  0.02 ft; sign test still positive at p < 0.05). If E1 is adopted first, E2 must also beat E1 by the same rule.
- Ablation: the same attention model with 1 s of history (5 steps). It separates the gain of history from the gain
  of the architecture; the stage 6 GRU result says the architecture alone may give nothing.
- Compute: Colab GPU, one fixed configuration chosen before the run (2 to 4 layers, width 64 to 128, early stopping
  on an inner split of development games); estimate 4 to 8 h. Not suited to the Mac CPU.
- Risks and failure modes: many possessions have less than 4 s before a step, so history must be masked, and
  results may differ by possession phase; a tuning budget that grows after results are seen would void the test;
  more context can let the model copy the actual defender's habits, which again can weaken the sign test.
- What a coach gains: ghosts that react to how the play developed (a screen 3 s ago, a drive), not only to the last
  second.
- Backlog: expected gain 3 (history and all-agent input both helped in baller2vec and Felsen et al., but our GRU on
  the same short window did not). Effort 4 (new 20-step windows, a new model, GPU runs and masking).

## 7. Sources checked (all fetched 2026-11-16)

- Le et al. 2017, Data-Driven Ghosting: https://authors.library.caltech.edu/records/74xh3-ysx85 and its PDF
  https://authors.library.caltech.edu/records/74xh3-ysx85/files/1671-2.pdf (sections 2 and 3.1 read).
- Le et al. 2017, Coordinated Multi-Agent Imitation Learning: https://arxiv.org/abs/1703.03121 and
  https://proceedings.mlr.press/v70/le17a/le17a.pdf (section 4.2 read; its results are in Figure 5 only, so no
  number is quoted).
- Seidl et al. 2018, Bhostgusters: https://www.sloansportsconference.com/research-papers/bhostgusters-realtime-interactive-play-sketching-with-synthesized-nba-defenses,
  the PDF linked there, and the library record https://lida.sport-iat.de/dbb/Record/4047517 (authors and year).
- Felsen et al. 2018: https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Panna_Felsen_Where_Will_They_ECCV_2018_paper.php
  and https://openaccess.thecvf.com/content_ECCV_2018/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf
  (sections 3, 5 and Tables 1 and 2 read).
- Xu, C., Li, M., Ni, Z., Zhang, Y., Chen, S. (2022). GroupNet: Multiscale Hypergraph Neural Networks for
  Trajectory Prediction with Relational Reasoning. CVPR 2022. https://arxiv.org/abs/2204.08770 (Table 2 read).
- Alcorn, M. A., Nguyen, A. (2021). baller2vec: A Multi-Entity Transformer For Multi-Agent Spatiotemporal
  Modeling. arXiv:2102.03291, https://arxiv.org/pdf/2102.03291 (sections 3 and 4, Table 3 read).
- Alcorn, M. A., Nguyen, A. (2021). baller2vec++: A Look-Ahead Multi-Entity Transformer For Modeling Coordinated
  Agents. arXiv:2104.11980, https://arxiv.org/abs/2104.11980 (abstract only).
- Capellera, G., Ferraz, L., Rubio, A., Agudo, A., Moreno-Noguer, F. (2024). TranSPORTmer: A Holistic Approach to
  Trajectory Understanding in Multi-Agent Sports. ACCV 2024. https://arxiv.org/abs/2410.17785 (abstract and
  dataset section read).
- Checked, not used: Teoh, W. Z. (2025). Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj.
  arXiv:2511.18248 (a preprint whose metrics are best-of-20 like GroupNet's; it adds nothing comparable).
