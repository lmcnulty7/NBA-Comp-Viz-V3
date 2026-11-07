# Results

<!-- results:begin -->

**Fixture output, not a result.** 16 of 16 lane reports below are fixture output (synthetic analytics.fixtures runs, or a smoke run on fixture pages); every number copied from them is fixture output, not a result.

Generated 2026-11-07T10:22:15Z by `python3 -m analytics.results --reports reports --out reports/results.md` (analytics/results.py; docs/contract/results.md). Every value is copied from the report named in its row; nothing here is computed. Regeneration rewrites only the text between the results markers.

### Sources

| lane | report | status | written | command |
| --- | --- | --- | --- | --- |
| xfg | reports/xfg.md | fixture output, not a result | 2026-11-07T08:40:33Z | `python -m analytics.xfg.train --fixtures --report reports/xfg.md --folds 5 --seed 0 --bootstrap 200` |
| rebounds | reports/rebounds.md | fixture output, not a result | 2026-11-07T08:40:32Z | `python -m analytics.rebounds.train --fixtures --report reports/rebounds.md --bootstrap 200` |
| matchups | reports/matchups.md | fixture output, not a result | 2026-11-07T08:40:31Z | `python -m analytics.matchups.train --fixtures --report reports/matchups.md --seed 0 --bootstrap 200` |
| gravity | reports/gravity.md | fixture output, not a result | 2026-11-07T08:42:42Z | `python -m analytics.gravity.train --fixtures --report reports/gravity.md --folds 5 --seed 0 --bootstrap 200` |
| ghosting | reports/ghosting.md | fixture output, not a result | 2026-11-07T08:42:18Z | `python -m analytics.ghosting.train --fixtures --games 4 --possessions 6 --heldout-every 4 --folds 5 --seed 0 --bootstrap 200 --report reports/ghosting.md` |
| embeddings | reports/embeddings.md | fixture output, not a result | 2026-11-07T08:42:24Z | `python -m analytics.embeddings.train --fixtures --report reports/embeddings.md --seed 0 --bootstrap 200` |
| epv | reports/epv.md | fixture output, not a result | 2026-11-07T08:37:37Z | `python -m analytics.epv.train --fixtures --fixture-games 8 --folds 5 --bootstrap 200 --report reports/epv.md` |
| actions | reports/actions.md | fixture output, not a result | 2026-11-07T08:40:37Z | `python -m analytics.actions.train --fixtures --report reports/actions.md --bootstrap 200` |
| coverage | reports/coverage.md | fixture output, not a result | 2026-11-07T08:42:54Z | `python -m analytics.coverage.train --fixtures --fixture-games 8 --fixture-heldout-every 4 --fixture-possessions 40 --report reports/coverage.md --folds 5 --seed 0 --bootstrap 200` |
| decisions | reports/decisions.md | fixture output, not a result | 2026-11-07T08:32:25Z | `python -m analytics.decisions.train --fixtures --report reports/decisions.md` |
| playcalling | reports/playcalling.md | fixture output, not a result | 2026-11-07T08:35:44Z | `python -m analytics.playcalling.train --fixtures --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 200` |
| lineups | reports/lineups.md | fixture output, not a result | 2026-11-07T08:38:54Z | `python -m analytics.lineups.train --fixtures --fixture-generator local --n-games 40 --report reports/lineups.md --bootstrap 200` |
| winprob | reports/winprob.md | fixture output, not a result | 2026-11-07T08:42:18Z | `python -m analytics.winprob.train --fixtures --report reports/winprob.md --bootstrap 200` |
| load | reports/load.md | fixture output, not a result | 2026-11-07T08:42:47Z | `python -m analytics.load.train --fixtures --report reports/load.md --bootstrap 200` |
| betting | reports/betting.md | fixture output, not a result | 2026-11-07T08:42:01Z | `python -m analytics.betting.train --fixtures --report reports/betting.md` |
| xfg_baseline | reports/xfg_baseline.md | smoke run on fixture pages, not a result | 2026-11-05T09:44:35Z | `python -m analytics.train_xfg --shots /tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/bref_smoke_shots.parquet --report reports/xfg_baseline.md --folds 5 --seed 0 --bootstrap 200 --smoke` |

### Overview

| lane | status | selected model | out-of-fold metric [interval] | held-out | gates and checks fired |
| --- | --- | --- | --- | --- | --- |
| xfg | fixture output, not a result | location_logistic | log loss 0.6843 [0.6653, 0.7095] | log loss 0.6770 [0.6554, 0.6991] | ECE under 0.02 (a target, not a selection rule): no; paired bootstrap, log loss, hgb_contest_prior vs location_logistic: excludes 0: yes; paired bootstrap, log loss, contest_logistic vs location_logistic: excludes 0: yes |
| rebounds | fixture output, not a result | prior | per-miss log loss 1.7981 [1.6972, 1.8927] | per-miss log loss 2.1135 (point estimate) | paired bootstrap: excludes zero: False |
| matchups | fixture output, not a result | ridge | possession points MSE 1.3854 [1.3381, 1.4578] | possession points MSE 1.2287 (point estimate) | paired bootstrap, ridge minus raw: excludes zero: True; paired bootstrap, ridge minus constant: excludes zero: False; defender gate: 0 pass the 300-possession gate |
| gravity | fixture output, not a result | ridge (SHIPPED expectation (the plan's model)) | MSE of nearest-defender distance (ft^2) 4.299 [4.050, 4.496] | MSE of nearest-defender distance (ft^2) 4.892 (point estimate) | paired bootstrap: excludes zero: True; player gate: 13 gated of 42; top-decile check: passed: True; sign test: p = n/a |
| ghosting | fixture output, not a result | 2 ridge on the lag-1 features (linear baseline) | 1.0 s position error (ft) 3.874 [3.356, 4.304] | 1.0 s position error (ft) 3.623 (point estimate) | deep candidate (MLP, GRU stand-in): dropped; validation sign test: p = 0.7500 |
| embeddings | fixture output, not a result | pca | 5-NN position accuracy (leave-one-out) 0.921 [0.841, 1.000] | absent | autoencoder gate: the autoencoder is not trained |
| epv | fixture output, not a result | constant | MSE expected points 1.3341 [1.2365, 1.4472] | MSE expected points 1.2191 (point estimate) | sequence model gate: TRAINED; sequence model kept or dropped: DROPPED; shuffled-target check |
| actions | fixture output, not a result | hgb | F1 ball_screen 1.000 [1.000, 1.000] | F1 ball_screen 1.000 (point estimate) | plan target: a fixture cannot meet or miss it; detections |
| coverage | fixture output, not a result | hgb | macro F1 0.944 [0.902, 0.979] | macro F1 0.835 (point estimate) | target: beat majority (met) |
| decisions | fixture output, not a result | location | mean Spearman, decision quality vs on-court rating (by game) 0.285 [0.125, 0.421] | mean Spearman, decision quality vs on-court rating (by game) -0.144 [-0.167, -0.121] | prior rule; player gate: nobody is ranked |
| playcalling | fixture output, not a result | ridge | possession points MSE 1.3674 [1.3195, 1.4340] | possession points MSE 1.2359 (point estimate) | paired bootstrap, ridge minus mean: excludes zero: no; paired bootstrap, ridge minus by_action: excludes zero: no; recommendation gate: no recommendation is made |
| lineups | fixture output, not a result | plus_minus | out-of-season Spearman 0.624 [-0.113, 0.950] | Spearman 0.881 [0.342, 1.000] | selection; paired bootstrap, rapm Spearman minus plus_minus (paired); paired bootstrap, rapm_synergy Spearman minus plus_minus (paired) |
| winprob | fixture output, not a result | logistic | mean bucket Brier (the selection metric) 0.1278 (no interval stated) | mean bucket Brier 0.1531 (no interval stated) | prefix invariance check: 0 violation(s) |
| load | fixture output, not a result | ridge (decline); logistic (absence) | y_decline MSE (ft/s)^2 0.8584 [0.5032, 1.3310] | y_decline MSE (ft/s)^2 0.6271 (point estimate) | paired bootstrap, decline: excludes zero: False; paired bootstrap, absence: excludes zero: False; absence target: One class only |
| betting | fixture output, not a result | cover model market, margin model market | cover log loss 0.6931 [0.6931, 0.6931] | cover log loss 0.6931 (point estimate, season 2021-22) | none stated |
| xfg_baseline | smoke run on fixture pages, not a result | logistic | log loss (in-sample) 0.6741 (no interval stated) | not run yet (set aside and NOT scored: a smoke run never touches them) | metrics in-sample |

### xfg (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/xfg.md, written 2026-11-07T08:40:33Z |
| selected model | location_logistic |
| out-of-fold metric | log loss 0.6843 [0.6653, 0.7095] |
| out-of-fold, selected | Brier 0.2458 [0.2365, 0.2581] |
| out-of-fold, selected | ECE (10 bins) 0.0525 [0.0321, 0.1496] |
| out-of-fold, selected | calibration slope 0.6735 [0.1577, 1.2453] |
| candidates, out-of-fold | location_logistic: log loss 0.6843 [0.6653, 0.7095]; contest_logistic: log loss 0.7136 [0.6751, 0.7624]; hgb_contest_prior: log loss 0.7253 [0.6893, 0.7878] |
| held-out | log loss 0.6770 [0.6554, 0.6991] |
| held-out, selected | Brier 0.2415 [0.2315, 0.2518] |
| ECE under 0.02 (a target, not a selection rule) | selected candidate: no |
| paired bootstrap, log loss, hgb_contest_prior vs location_logistic | 0.0409 [0.0084, 0.0779], interval excludes 0: yes |
| paired bootstrap, log loss, contest_logistic vs location_logistic | 0.0292 [0.0105, 0.0548], interval excludes 0: yes |

### rebounds (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/rebounds.md, written 2026-11-07T08:40:32Z |
| selected model | prior |
| out-of-fold metric | per-miss log loss 1.7981 [1.6972, 1.8927] |
| out-of-fold, selected | top-1 accuracy 0.3379 [0.2835, 0.4085] |
| candidates, out-of-fold | prior: per-miss log loss 1.7981 [1.6972, 1.8927]; logit: per-miss log loss 1.8517 [1.7540, 1.9911]; hgb: per-miss log loss 1.8307 [1.6344, 2.0684]; uniform reference: per-miss log loss 2.3979 |
| held-out | per-miss log loss 2.1135 (point estimate) |
| held-out, selected | top-1 accuracy 0.2167 (point estimate) |
| paired bootstrap | Paired bootstrap by game, hgb minus prior per-miss log loss: 0.0326 [-0.1358, 0.2302]; interval excludes zero: False (negative favours hgb). |

### matchups (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/matchups.md, written 2026-11-07T08:40:31Z |
| selected model | ridge |
| out-of-fold metric | possession points MSE 1.3854 [1.3381, 1.4578] |
| candidates, out-of-fold | constant: MSE 1.3580 [1.3008, 1.4412]; raw: MSE 1.5105 [1.4062, 1.5897]; ridge: MSE 1.3854 [1.3381, 1.4578] |
| held-out | possession points MSE 1.2287 (point estimate) |
| paired bootstrap, ridge minus raw | -0.1251 [-0.1995, -0.0363] (excludes zero: True) |
| paired bootstrap, ridge minus constant | 0.0273 [-0.0004, 0.0743] (excludes zero: False) |
| defender gate | 42 defenders, 0 pass the 300-possession gate. |

### gravity (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/gravity.md, written 2026-11-07T08:42:42Z |
| selected model | ridge (SHIPPED expectation (the plan's model)) |
| out-of-fold metric | MSE of nearest-defender distance (ft^2) 4.299 [4.050, 4.496] |
| out-of-fold, selected | R2 0.404 |
| candidates, out-of-fold | constant: MSE 7.215 [6.975, 7.429]; ridge: MSE 4.299 [4.050, 4.496]; hgb: MSE 2.254 [2.021, 2.401] |
| held-out | MSE of nearest-defender distance (ft^2) 4.892 (point estimate) |
| held-out, selected | constant floor MSE 8.032 (not a selection input) |
| paired bootstrap | Paired bootstrap by game, MSE(hgb) - MSE(ridge): -2.0446 [-2.1452, -1.9410], excludes zero: True (HGB lower: True). |
| player gate | Players (gate: 40 possessions, the fixture gate; real data gates at 300; 13 gated of 42) |
| top-decile check | Known high-gravity players [500030, 500050, 500060, 500070, 500150, 500160]; season-wide ranks among 13 gated players {500030: 2, 500050: None, 500060: 3, 500070: 1, 500150: None, 500160: None} (None: under the gate, not ranked); 3 of the known players are gated; cut 3 (the top decile, or the top n_known when the planted players outnumber a decile); passed: True. |
| top-decile check | Within each training game (the check that applies when rosters do not persist across games): 6 of 6 games pass; passed: True. |
| sign test | Candidates [500030, 500050, 500060, 500070, 500150, 500160] (the fixture's planted high-gravity players). Games with a difference: 0; positive: 0; one-sided binomial p = n/a; mean difference n/a [n/a, n/a] (bootstrap by game). |
| note | the fixture report carries a second run on the planted-pull generator; this summary copies the first run |

### ghosting (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/ghosting.md, written 2026-11-07T08:42:18Z |
| selected model | 2 ridge on the lag-1 features (linear baseline) |
| out-of-fold metric | 1.0 s position error (ft) 3.874 [3.356, 4.304] |
| out-of-fold, selected | 0.2 s position error (ft) 0.556 [0.535, 0.585] |
| candidates, out-of-fold | 1 persistence (position at t minus 1): 1.0 s error 4.015 [3.425, 4.672]; 2 ridge on the lag-1 features (linear baseline): 1.0 s error 3.874 [3.356, 4.304]; 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in): 1.0 s error 4.968 [4.781, 5.153] |
| held-out | 1.0 s position error (ft) 3.623 (point estimate) |
| held-out, selected | 0.2 s position error (ft) 0.524 (point estimate) |
| deep candidate (MLP, GRU stand-in) | The deep candidate (3) is dropped: its 1.0 s error minus the linear baseline's is 1.094 with paired interval [0.849, 1.425], which excludes zero. |
| validation sign test | Sign test: 1 positive, 1 negative, 0 zero; one-sided p = 0.7500. |

### embeddings (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/embeddings.md, written 2026-11-07T08:42:24Z |
| selected model | pca |
| out-of-fold metric | 5-NN position accuracy (leave-one-out) 0.921 [0.841, 1.000] |
| out-of-fold, selected | majority baseline 0.421 [0.342, 0.579] |
| out-of-fold, selected | next-season R squared (ridge, out-of-fold) 0.859 [0.797, 0.910] |
| candidates, out-of-fold | pca: 5-NN accuracy 0.921 [0.841, 1.000]; autoencoder: 5-NN accuracy not trained |
| held-out | absent |
| autoencoder gate | PCA 5-NN accuracy 0.921 is at or above the 0.60 gate: the autoencoder is not trained |
| note | Player-seasons: 42, of which 38 sufficient (at least 20 possessions) and 4 insufficient; the contract gate is 300, lowered for the short fixture games. |
| note | the lane scores no held-out rows: held-out material is set aside before any feature |

### epv (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/epv.md, written 2026-11-07T08:37:37Z |
| selected model | constant |
| out-of-fold metric | MSE expected points 1.3341 [1.2365, 1.4472] |
| out-of-fold, selected | Brier P(points > 0) 0.2516 [0.2423, 0.2619] |
| candidates, out-of-fold | constant: MSE 1.3341 [1.2365, 1.4472]; ridge: MSE 1.3746 [1.2232, 1.5410]; hgb: MSE 1.4563 [1.3144, 1.6102]; mlp: MSE 2.2112 [1.9892, 2.3961] |
| held-out | MSE expected points 1.2191 (point estimate) |
| held-out, selected | Brier P(points > 0) 0.2426 (point estimate) |
| sequence model gate | Gate decision: the sequence model (MLPRegressor over the last 5 states, standing in for a GRU) was TRAINED. |
| sequence model kept or dropped | Paired bootstrap by game of MSE(constant) - MSE(mlp), constant being the lowest-MSE candidate: -0.8771 [-1.0400, -0.7132]; the sequence model is DROPPED (no significant gain over the base candidate). |
| shuffled-target check | Possession points permuted across possessions of the training games, gradient boosting refit out-of-fold: MSE 1.4598 vs constant 1.3428, lift -8.7 percent (chance is at or below 0; the real lift above is -9.2 percent). |
| note | Why the constant: no learner beats the constant out of fold (gradient boosting lift -9.2 percent); the gradient-boosting calibration slopes sit outside 0.9 to 1.1 in 14-24, 7-14; the real lift (-9.2 percent) is no better than the shuffled-target lift (-8.7 percent); the learners overfit (in-sample MSE well under out-of-fold for hgb, mlp: the target is one draw per possession while min_samples_leaf counts states, so a leaf can hold a handful of possessions). |

### actions (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/actions.md, written 2026-11-07T08:40:37Z |
| selected model | hgb |
| out-of-fold metric | F1 ball_screen 1.000 [1.000, 1.000] |
| out-of-fold, selected | macro F1 0.930 |
| candidates, out-of-fold | rule: F1 ball_screen 0.643 [0.560, 0.718]; hgb: F1 ball_screen 1.000 [1.000, 1.000] |
| held-out | F1 ball_screen 1.000 (point estimate) |
| held-out, selected | macro F1 1.000 (point estimate) |
| plan target | plan target: ball_screen F1 over 0.8 on a held-out game; a fixture cannot meet or miss it |
| detections | Detected actions (selected class other than none with probability above 0.5): 54 of 219 training candidates; 0 ball_screen detection(s) refused for lack of a screener. |

### coverage (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/coverage.md, written 2026-11-07T08:42:54Z |
| selected model | hgb |
| out-of-fold metric | macro F1 0.944 [0.902, 0.979] |
| out-of-fold, selected | accuracy 0.945 [0.908, 0.979] |
| candidates, out-of-fold | majority: macro F1 0.040 [0.013, 0.055]; logistic: macro F1 0.763 [0.620, 0.898]; hgb: macro F1 0.944 [0.902, 0.979] |
| held-out | macro F1 0.835 (point estimate) |
| held-out, selected | accuracy 0.852 (point estimate) |
| target | Fixture target: beat majority (met). |

### decisions (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/decisions.md, written 2026-11-07T08:32:25Z |
| selected model | location |
| out-of-fold metric | mean Spearman, decision quality vs on-court rating (by game) 0.285 [0.125, 0.421] |
| out-of-fold, selected | positive / signed games 5 / 6 |
| out-of-fold, selected | sign p 0.219 |
| out-of-fold, selected | shuffled mean rho -0.039 |
| held-out | mean Spearman, decision quality vs on-court rating (by game) -0.144 [-0.167, -0.121] |
| prior rule | candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least 0.50. Selected: location (no xFG table present). |
| player gate | No player passes the 300-decision gate: nobody is ranked. |

### playcalling (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/playcalling.md, written 2026-11-07T08:35:44Z |
| selected model | ridge |
| out-of-fold metric | possession points MSE 1.3674 [1.3195, 1.4340] |
| candidates, out-of-fold | mean: MSE 1.3580 [1.3008, 1.4412]; by_action: MSE 1.3608 [1.2657, 1.4517]; ridge: MSE 1.3674 [1.3195, 1.4340] |
| held-out | possession points MSE 1.2359 (point estimate) |
| paired bootstrap, ridge minus mean | 0.0093 [-0.0070, 0.0270], interval excludes zero: no |
| paired bootstrap, ridge minus by_action | 0.0065 [-0.0581, 0.0521], interval excludes zero: no |
| recommendation gate | No coverage has a cell with at least 300 rows, so no recommendation is made. |
| note | association only, not causal (the report's title) |

### lineups (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/lineups.md, written 2026-11-07T08:38:54Z |
| selected model | plus_minus |
| out-of-fold metric | out-of-season Spearman 0.624 [-0.113, 0.950] |
| out-of-fold, selected | out-of-season MSE 712.776 [221.103, 1266.230] |
| candidates, out-of-fold | plus_minus: Spearman 0.624 [-0.113, 0.950]; rapm: Spearman 0.576 [-0.273, 0.923]; rapm_synergy: Spearman 0.576 [-0.273, 0.923] |
| held-out | Spearman 0.881 [0.342, 1.000] |
| held-out, selected | MSE 198.677 [39.772, 383.567] |
| selection | Mean validation Spearman: plus_minus 0.624, rapm 0.576, rapm_synergy 0.576 |
| paired bootstrap, rapm Spearman minus plus_minus (paired) | -0.048 [-0.297, 0.151] |
| paired bootstrap, rapm_synergy Spearman minus plus_minus (paired) | -0.048 [-0.297, 0.151] |

### winprob (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/winprob.md, written 2026-11-07T08:42:18Z |
| selected model | logistic |
| out-of-fold metric | mean bucket Brier (the selection metric) 0.1278 (no interval stated) |
| out-of-fold, selected | Brier overall 0.1473 [0.1078, 0.1882] |
| out-of-fold, selected | log loss overall 0.4329 [0.3289, 0.5393] |
| candidates, out-of-fold | constant: mean bucket Brier 0.2582; logistic: mean bucket Brier 0.1278; hgb_iso: mean bucket Brier 0.2144 |
| held-out | mean bucket Brier 0.1531 (no interval stated) |
| held-out, selected | Brier overall 0.1721 [0.1236, 0.2483] |
| held-out, selected | log loss overall 0.5038 [0.3698, 0.7094] |
| prefix invariance check | `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k) |
| note | The NBA.com WinProbabilityPBP endpoint is unreachable from here, so the benchmark column is empty until it exists. When it does, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier. |

### load (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/load.md, written 2026-11-07T08:42:47Z |
| selected model | ridge (decline); logistic (absence) |
| out-of-fold metric | y_decline MSE (ft/s)^2 0.8584 [0.5032, 1.3310] |
| out-of-fold, selected | absent_next log loss 0.0010 [0.0010, 0.0010] |
| candidates, out-of-fold | mean (training-fold mean): MSE 0.9133 [0.4863, 1.4505]; ridge on FEATURES: MSE 0.8584 [0.5032, 1.3310] |
| held-out | y_decline MSE (ft/s)^2 0.6271 (point estimate) |
| held-out, selected | absent_next log loss 0.0010 (point estimate) |
| paired bootstrap, decline | Paired bootstrap by game of MSE(ridge) - MSE(mean): -0.0549 [-0.2815, 0.1987], excludes zero: False. |
| paired bootstrap, absence | Paired bootstrap by game of log loss(logistic) - log loss(base): 0.0000 [0.0000, 0.0000], excludes zero: False. |
| absence target | One class only in absent_next: no calibration table, and the logistic model reduces to the base rate. |

### betting (fixture output, not a result)

| field | value |
| --- | --- |
| status | fixture output, not a result |
| report | reports/betting.md, written 2026-11-07T08:42:01Z |
| selected model | cover model market, margin model market |
| out-of-fold metric | cover log loss 0.6931 [0.6931, 0.6931] |
| out-of-fold, selected | margin MSE 983.9 [618.8, 1345.3] |
| out-of-fold, selected | bets 0 |
| out-of-fold, selected | ROI n/a |
| candidates, out-of-fold | market: cover log loss 0.6931 [0.6931, 0.6931]; logistic: cover log loss 0.9645 [0.7472, 1.2319]; ridge: cover log loss 1.0231 [0.7217, 1.4786] |
| held-out | cover log loss 0.6931 (point estimate, season 2021-22) |
| held-out, selected | margin MSE 1000.8 |
| held-out, selected | bets 0 |
| held-out, selected | ROI n/a |
| gates and checks | none stated |
| note | Standing caveat: odds history is a paid feed not yet collected; until it is, every number here is fixture output. |

### xfg_baseline (fixture output, not a result)

| field | value |
| --- | --- |
| status | smoke run on fixture pages, not a result |
| report | reports/xfg_baseline.md, written 2026-11-05T09:44:35Z |
| selected model | logistic |
| out-of-fold metric | log loss (in-sample) 0.6741 (no interval stated) |
| out-of-fold, selected | Brier (in-sample) 0.2406 |
| out-of-fold, selected | calibration slope (in-sample) 1.0329 |
| candidates, out-of-fold | constant: log loss (in-sample) 0.6920; logistic: log loss (in-sample) 0.6741 |
| held-out | not run yet (set aside and NOT scored: a smoke run never touches them) |
| metrics in-sample | Metrics: one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0); no interval with a single game. |
| note | Phase 1 baseline (analytics/train_xfg.py): a constant floor and the logistic location model, no selection rule |

<!-- results:end -->

<!-- Hand-written sections (stage 9 of docs/mac_runbook.md) go below this line. analytics.results rewrites only the text between the results markers. -->
