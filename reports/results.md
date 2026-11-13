# Results

Group 1 of the analytics build: eleven lanes and the xFG baseline, frozen at tag `freeze-v1` and scored on held-out
data exactly once (stage 8, 2026-11-12). Every number below is copied from the lane report named in its section;
the generated tables at the end (between the results markers, written by `analytics.results`) repeat them field by
field. Intervals are 95 percent percentile intervals of a bootstrap by game unless a section says otherwise; a value
with no interval is a point estimate.

## Summary

**Data.** Tracking: the 2015-16 SportVU season, 621 games with usable frames, of which 604 train and 17 are held out
(every game at CLE plus 12.16.2015.PHX.at.GSW, fixed in sportvu/splits.json before any model was built). Shots:
basketball-reference shot charts for 625 of those games (607 train, 18 held out; the extra games have shots but broken
tracking). Play-by-play: 2020-21 to 2024-25, 6363 games; 2024-25 (1304 games) is the held-out season for the two
multi-season lanes. NBA.com tables and betting odds were skipped by the owner, so nothing here is benchmarked against
NBA.com's own models.

**Protocol.** Each lane chose its model on training data only, by a rule fixed before it ran (grouped folds by game,
paired bootstrap by game). Two improvement-search variants were accepted on training data (the rebounds temperature
and a winprob late-game feature, reports/archive/). The models were frozen, then each lane scored its held-out games
once. Out-of-fold numbers were reproduced exactly by the held-out run.

| lane | question | selected model | out-of-fold | held-out | verdict |
| --- | --- | --- | --- | --- | --- |
| xfg | will this shot go in? | gradient boosting with contest features and a shooter prior | log loss 0.6294 [0.6277, 0.6316] | 0.6232 [0.6132, 0.6327] | works; beats the location-only baseline by 0.050 on held-out; held-out calibration worse (ECE 0.019) |
| rebounds | who gets this rebound? | gradient boosting plus a fitted temperature | per-miss log loss 1.6009 [1.5882, 1.6103] | 1.7115 | works, far above the 2.39 floor; held-out worse by 0.11 |
| matchups | who guards whom, and what do they allow? | ridge on defender and attacker shares | possession-points MSE 1.2854 vs constant 1.2842 | 1.3195 vs constant 1.3251 | assignment is stable (0.82); no defender signal beats the constant out of fold |
| gravity | how far do defenders stay from a player, beyond what location explains? | ridge expectation, gravity as its residual | MSE 28.12 (skill 0.416) | 29.91 (skill 0.434) | expectation holds on held-out; teammates of top-gravity players shoot slightly better (+0.004, training games) |
| ghosting | where should a defender have been? | MLP on the last 5 frames (GRU stand-in) | 1.0 s error 3.657 [3.635, 3.682] ft | 3.417 ft | works, best on held-out; a deviation sign test agrees on 13 of 17 held-out games |
| embeddings | which players move and shoot alike? | PCA on 19 style features | 5-NN position accuracy 0.768 [0.720, 0.809] | none by design | passes its 0.60 gate; no held-out score exists |
| epv | how many points will this possession score, from here? | torch GRU over the last 5 states | MSE 1.2155 [1.2080, 1.2226] | 1.2362 [1.1899, 1.2812] | works; 5 percent better than the constant out of fold, 7 percent on held-out |
| decisions | was this pass, drive or shot the best option? | EPV option values with the xFG prior | Spearman vs on-court rating 0.012 [-0.008, 0.029] | -0.014 [-0.116, 0.100] | **failed**: no validated signal in either sample |
| lineups | which lineups and players are good? | RAPM (ridge, alpha 3000) | season-ahead Spearman mean 0.109 | 0.065 [-0.488, 0.482] | **inconclusive**: about 20 recurring lineups a season, every interval spans zero |
| winprob | who wins from this game state? | logistic with a margin over root time feature | mean bucket Brier 0.1468 | 0.1486; log loss 0.4879 [0.4700, 0.5077] | works; held-out slightly worse and slightly over-confident |
| load | does a player slow down, and will they sit next game? | ridge (decline), logistic (absence) | decline MSE 0.4929 [0.4740, 0.5178]; absence log loss 0.2764 [0.2644, 0.2883] | 0.5164; 0.2776 | absence works; decline barely beats the mean on held-out |

**Where held-out is worse than out of fold.** xFG calibration, the xFG baseline, rebounds, EPV MSE (and its first-4
seconds calibration), decisions, lineups, win probability and the load decline model. Ghosting is better on held-out;
matchups and gravity have higher absolute error but the floor rises more. Held-out games are few (17 or 18 for the
tracking lanes), so most held-out numbers are point estimates and a difference of this size is within what the games
themselves vary by.

**What failed.** Decision quality does not track on-court offensive rating, in training or held out. Matchups' ridge
does not beat a constant on possession points out of fold, so the defender table ranks noise around a small effect.
Lineup ratings cannot be validated with this many recurring lineups. The load decline model is barely above the mean
on held-out. None of these was rescued after the held-out numbers were seen.

**Not in this write-up.** Group 2 (actions, coverage, playcalling) waits for hand labels and gets its own freeze
(`freeze-v2`) and its own single held-out run; its reports are still fixture output. Betting was skipped (no odds);
its report is fixture output too.

## By extension

### xfg: shot quality (reports/xfg.md, reports/xfg_baseline.md)

- **Question.** The probability that a field-goal attempt goes in, from where and how it was taken.
- **Data.** 101853 training attempts from 607 games (basketball-reference shots matched to tracking release frames),
  2939 held-out attempts from 18 games.
- **Model.** hgb_contest_prior: gradient boosting on location and contest features (closest defender distance and
  closing speed, defenders within 6 ft, shooter speed, touch time, shot clock) plus a shooter prior, with isotonic
  calibration; chosen over location-only and contest logistic models by out-of-fold log loss.
- **Numbers.** Out of fold: log loss 0.6294 [0.6277, 0.6316], Brier 0.2209 [0.2201, 0.2219], ECE 0.0029, slope 1.000.
  Held out: log loss 0.6232 [0.6132, 0.6327], Brier 0.2187 [0.2142, 0.2233], ECE 0.0189 [0.0143, 0.0347], slope 1.090
  [0.985, 1.198]. The location-only baseline: 0.6665 [0.6652, 0.6681] out of fold, 0.6735 held out.
- **What failed and what it means.** Held-out calibration is worse (ECE six times higher, slope above 1: the model is
  under-confident on these games). Ranking and overall accuracy hold, and the gain over location alone grows on
  held-out (0.050 against 0.037), so contest information is the real improvement. The baseline is worse on every
  held-out metric.

### rebounds: who secures a missed shot (reports/rebounds.md)

- **Question.** For a missed shot, the probability that each player on the floor gets the rebound, or that it is a
  team rebound.
- **Data.** 49288 misses from 604 training games, 1227 from 17 held-out games, positions at the rim frame.
- **Model.** Gradient boosting on candidate rows, normalised within the miss with a temperature fitted on inner folds
  (improvement-search variant 1, accepted with a paired difference of -0.0225 [-0.0246, -0.0205]).
- **Numbers.** Out of fold: per-miss log loss 1.6009 [1.5882, 1.6103], top-1 0.4797 [0.4756, 0.4848]. Held out: 1.7115,
  top-1 0.4474. Uniform floor 2.388 and 2.386.
- **What failed and what it means.** Held-out is worse on both metrics, beyond the out-of-fold intervals, with the
  same number of candidates per miss. The model still cuts the floor's loss by about 0.67 nats. The top candidate stays
  under-confident out of fold (slope 1.37) even after the temperature.

### matchups: individual defense (reports/matchups.md)

- **Question.** Who guards whom on each possession, and how many points a defender allows against expectation.
- **Data.** 604 training games, 3162 held-out possessions from 17 games; assignment at 1 Hz by the Hungarian method on
  distance to the attacker plus half the distance to the attacker-basket midpoint.
- **Model.** Ridge on defender and attacker share columns (the lower out-of-fold MSE of raw and ridge).
- **Numbers.** Possession-points MSE out of fold: ridge 1.2854 [1.2788, 1.2916], constant 1.2842 [1.2776, 1.2903],
  ridge minus constant 0.0012 [0.0002, 0.0020]. Held out: ridge 1.3195, constant 1.3251. Assignment stability 0.823 in
  both; defended FG% 0.427 within 4 ft against 0.454 beyond (gap 0.027 [0.020, 0.035]), 0.432 against 0.465 held out.
- **What failed and what it means.** Ridge is worse than a constant out of fold: possession points carry too little
  defender signal for this model to find. The assignment itself is sound (stable, and closer defense means a lower FG%
  in both samples), so the matchup tables are usable; the defender ratings are not evidence of skill.

### gravity: spacing pull (reports/gravity.md)

- **Question.** How far defenders stay from a player beyond what the player's location explains.
- **Data.** 7983660 player-frames from 604 training games, 195015 from 17 held-out games.
- **Model.** Ridge expectation of nearest-defender distance on location features; gravity is the residual. Gradient
  boosting fits better (MSE 22.71) but is a diagnostic only, since a flexible expectation absorbs the residual.
- **Numbers.** Ridge MSE 28.119 [27.800, 28.452] (skill over the floor 0.416) out of fold; 29.913 (skill 0.434) held
  out. With one of the top ten gravity players on the floor, teammates' location-only xFG is higher by 0.0042 [0.0009,
  0.0074] (163 of 290 games, p = 0.0198, training games).
- **What failed and what it means.** Absolute error is higher on held-out, but the floor rises more, so the expectation
  transfers. The teammate effect is small and measured on training games only.

### ghosting: where a defender should have been (reports/ghosting.md)

- **Question.** A defender's position 0.2 s and 1.0 s ahead from the last five sampled frames of everyone on the floor.
- **Data.** All 604 training games, fitted at 1 Hz (every 5th step, the owner's option 4) and scored at 5 Hz; 896885
  held-out defender-steps from 17 games.
- **Model.** MLP 64x64 on the 5-step window. The stage 6 torch GRU was worse by 0.111 ft [0.085, 0.138] at 1.0 s and
  was not selected.
- **Numbers.** 1.0 s error 3.657 [3.635, 3.682] ft out of fold, 3.417 held out; persistence 5.408 and 5.250. 0.2 s
  error 0.437 [0.433, 0.442] and 0.377. Deviation from the ghost against points allowed: 346 of 604 games positive
  (p = 0.0002) out of fold, 13 of 17 (p = 0.0245) held out.
- **What failed and what it means.** Nothing got worse on held-out. The stand-in is the selected model; the contract's
  GRU did not earn its place.

### embeddings: player style (reports/embeddings.md)

- **Question.** Which players move and shoot alike.
- **Data.** 448 player-seasons (399 with at least 300 possessions) from the 604 training games; tracking and shot
  features only, since the NBA.com aggregates were skipped.
- **Model.** PCA on 19 features. Its 5-NN position accuracy passed the 0.60 gate, so the autoencoder was not trained.
- **Numbers.** 5-NN position accuracy 0.768 [0.720, 0.809] (leave one out).
- **What failed and what it means.** No held-out score exists by design: held-out games are dropped before any
  feature. The embedding is descriptive.

### epv: expected possession value (reports/epv.md)

- **Question.** Expected points from a 1 Hz possession state to the end of the possession.
- **Data.** 1739119 states from 604 training games, 44703 from 17 held-out games.
- **Model.** The torch GRU over the last 5 states (stage 6), kept over the MLP stand-in by 0.0062 [0.0054, 0.0071] and
  over gradient boosting by 0.0212 [0.0201, 0.0228].
- **Numbers.** MSE 1.2155 [1.2080, 1.2226] out of fold against the constant's 1.2822; 1.2362 [1.1899, 1.2812] held out
  against 1.3319. Brier P(points > 0) 0.2350 and 0.2318. Calibration slope 0.964 and 1.024.
- **What failed and what it means.** Held-out MSE is higher in every elapsed-time bucket, and the first 4 seconds are
  under-confident on held-out (slope 1.23). Relative to the constant the model is better on held-out (7 percent against
  5 percent). A shuffled-target check gives no lift, so the gain is real but small: most of a possession's outcome is
  not visible in its state.

### decisions: decision quality (reports/decisions.md)

- **Question.** Whether the ball handler chose the best available option, in expected points.
- **Data.** Decisions from 604 training games, 11373 from 17 held-out games, valued by the frozen EPV GRU with the xFG
  prior on shots.
- **Model.** Option values with the xfg_selected prior (fixed in advance by coverage).
- **Numbers.** Mean per-game Spearman between a player's mean decision quality and their on-court offensive rating:
  0.012 [-0.008, 0.029] out of fold (317 of 603 games positive, p = 0.222), -0.014 [-0.116, 0.100] held out (7 of 17).
- **What failed and what it means.** This lane failed its validation: decision quality as measured does not track
  offensive rating in either sample. Either the EPV values are too flat to rank options or on-court rating is too noisy
  to check them; the decision tables should not be used to rate players.

### lineups: lineup and player ratings (reports/lineups.md)

- **Question.** Net rating of lineups and players, adjusted for teammates and opponents.
- **Data.** Play-by-play stints, 2020-21 to 2023-24 for fitting, 2024-25 held out.
- **Model.** RAPM: possession-weighted ridge on player indicators, alpha 3000 by grouped CV.
- **Numbers.** Season-ahead Spearman for lineups present in both seasons: 0.065, 0.222, 0.041 (mean 0.109) in the
  three validation seasons, 0.065 [-0.488, 0.482] on 2024-25; plus-minus 0.105 mean and -0.101 held out. RAPM MSE
  169.2 [76.6, 279.3] against plus-minus 230.7 on held-out.
- **What failed and what it means.** Inconclusive: only 18 to 27 lineups recur from one season to the next, and every
  Spearman interval spans zero. RAPM is never worse than raw plus-minus on MSE, which is the only claim this supports.

### winprob: live win probability (reports/winprob.md)

- **Question.** The home team's chance of winning from the game state at each play-by-play event.
- **Data.** 2292413 events from 5059 games (2020-21 to 2023-24) for training; 596840 events from 1304 games of
  2024-25 held out. No closing spreads (odds skipped).
- **Model.** Logistic on margin, time, possession and two margin-time interactions, including margin over root time
  (improvement-search variant 1, paired gain 0.00149 [0.00132, 0.00169]).
- **Numbers.** Mean bucket Brier 0.1468 out of fold, 0.1486 held out; log loss 0.4845 [0.4767, 0.4945] and 0.4879
  [0.4700, 0.5077]; last-4-minutes Brier 0.0530 [0.0496, 0.0564] and 0.0582 [0.0504, 0.0666].
- **What failed and what it means.** Held-out is slightly worse in four of five buckets and slightly over-confident
  (slopes about 0.96), but every interval overlaps. The NBA.com benchmark column is empty because those tables were
  skipped, so there is no external comparison.

### load: fatigue and absence (reports/load.md)

- **Question.** Whether a player's speed declines across a game, and whether they miss the team's next game.
- **Data.** 11894 player-games with a decline from 604 training games (absence on its own rows); 377 player-games from
  17 held-out games. Absence is not injury (rest, coach's decision and trades read the same).
- **Model.** Ridge on schedule and previous-game load features (decline); logistic on the same (absence).
- **Numbers.** Decline MSE 0.4929 [0.4740, 0.5178] against the mean's 0.5086 out of fold; 0.5164 against 0.5180 held out.
  Absence log loss 0.2764 [0.2644, 0.2883] against the base rate's 0.3244; 0.2776 against 0.3085 held out.
- **What failed and what it means.** The decline model barely beats the mean on held-out: in-game speed decline is
  mostly role and game state, not something past load predicts. The absence model holds up.

## Appendix: tables generated from the reports


<!-- results:begin -->

**Fixture output, not a result.** 4 of 16 lane reports below are fixture output (synthetic analytics.fixtures runs, or a smoke run on fixture pages); every number copied from them is fixture output, not a result.

Generated 2026-11-13T00:53:01Z by `python3 -m analytics.results --reports reports --out reports/results.md` (analytics/results.py; docs/contract/results.md). Every value is copied from the report named in its row; nothing here is computed. Regeneration rewrites only the text between the results markers.

### Sources

| lane | report | status | written | command |
| --- | --- | --- | --- | --- |
| xfg | reports/xfg.md | real data, held-out scored once | 2026-11-12T21:43:04Z | `python -m analytics.xfg.train --data $DATA_ROOT --seed 0 --report reports/xfg.md` |
| rebounds | reports/rebounds.md | real data, held-out scored once | 2026-11-12T21:47:24Z | `python -m analytics.rebounds.train --data $DATA_ROOT --seed 0 --report reports/rebounds.md` |
| matchups | reports/matchups.md | real data, held-out scored once | 2026-11-12T21:53:45Z | `python -m analytics.matchups.train --data $DATA_ROOT --seed 0 --report reports/matchups.md` |
| gravity | reports/gravity.md | real data, held-out scored once | 2026-11-12T22:00:30Z | `python -m analytics.gravity.train --data $DATA_ROOT --seed 0 --report reports/gravity.md` |
| ghosting | reports/ghosting.md | real data, held-out scored once | 2026-11-12T23:09:28Z | `python -m analytics.ghosting.train --data $DATA_ROOT --seed 0 --report reports/ghosting.md` |
| embeddings | reports/embeddings.md | real data | 2026-11-12T23:14:04Z | `python -m analytics.embeddings.train --data $DATA_ROOT --seed 0 --report reports/embeddings.md` |
| epv | reports/epv.md | real data, held-out scored once | 2026-11-13T00:24:51Z | `python -m analytics.epv.train --data $DATA_ROOT --seed 0 --report reports/epv.md --deep torch` |
| actions | reports/actions.md | fixture output, not a result | 2026-11-07T08:40:37Z | `python -m analytics.actions.train --fixtures --report reports/actions.md --bootstrap 200` |
| coverage | reports/coverage.md | fixture output, not a result | 2026-11-07T08:42:54Z | `python -m analytics.coverage.train --fixtures --fixture-games 8 --fixture-heldout-every 4 --fixture-possessions 40 --report reports/coverage.md --folds 5 --seed 0 --bootstrap 200` |
| decisions | reports/decisions.md | real data, held-out scored once | 2026-11-13T00:30:51Z | `python -m analytics.decisions.train --data $DATA_ROOT --seed 0 --report reports/decisions.md` |
| playcalling | reports/playcalling.md | fixture output, not a result | 2026-11-07T08:35:44Z | `python -m analytics.playcalling.train --fixtures --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 200` |
| lineups | reports/lineups.md | real data, held-out scored once | 2026-11-13T00:33:18Z | `python -m analytics.lineups.train --data $DATA_ROOT --seed 0 --report reports/lineups.md` |
| winprob | reports/winprob.md | real data, held-out scored once | 2026-11-13T00:41:57Z | `python -m analytics.winprob.train --data $DATA_ROOT --seed 0 --report reports/winprob.md` |
| load | reports/load.md | real data, held-out scored once | 2026-11-13T00:48:38Z | `python -m analytics.load.train --data $DATA_ROOT --seed 0 --report reports/load.md` |
| betting | reports/betting.md | fixture output, not a result | 2026-11-07T08:42:01Z | `python -m analytics.betting.train --fixtures --report reports/betting.md` |
| xfg_baseline | reports/xfg_baseline.md | real data, held-out scored once | 2026-11-13T00:48:43Z | `python -m analytics.train_xfg --shots $DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet $DATA_ROOT/collect/bref_shots/heldout/shotchart/2015-16.parquet --report reports/xfg_baseline.md` |

### Overview

| lane | status | selected model | out-of-fold metric [interval] | held-out | gates and checks fired |
| --- | --- | --- | --- | --- | --- |
| xfg | real data, held-out scored once | hgb_contest_prior | log loss 0.6294 [0.6277, 0.6316] | log loss 0.6232 [0.6132, 0.6327] | ECE under 0.02 (a target, not a selection rule): yes; paired bootstrap, log loss, hgb_contest_prior vs location_logistic: excludes 0: yes; paired bootstrap, log loss, contest_logistic vs location_logistic: excludes 0: yes |
| rebounds | real data, held-out scored once | hgb | per-miss log loss 1.6009 [1.5882, 1.6103] | per-miss log loss 1.7115 (point estimate) | paired bootstrap: excludes zero: True |
| matchups | real data, held-out scored once | ridge | possession points MSE 1.2854 [1.2788, 1.2916] | possession points MSE 1.3195 (point estimate) | paired bootstrap, ridge minus raw: excludes zero: True; paired bootstrap, ridge minus constant: excludes zero: True; defender gate: 368 pass the 300-possession gate |
| gravity | real data, held-out scored once | ridge (SHIPPED expectation (the plan's model)) | MSE of nearest-defender distance (ft^2) 28.119 [27.800, 28.452] | MSE of nearest-defender distance (ft^2) 29.913 (point estimate) | paired bootstrap: excludes zero: True; player gate: 362 gated of 447; sign test: p = 0.0198 |
| ghosting | real data, held-out scored once | 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in) | 1.0 s position error (ft) 3.657 [3.635, 3.682] | 1.0 s position error (ft) 3.417 (point estimate) | deep candidate (MLP, GRU stand-in): kept; validation sign test: p = 0.0002 |
| embeddings | real data | pca | 5-NN position accuracy (leave-one-out) 0.768 [0.720, 0.809] | absent | autoencoder gate: the autoencoder is not trained |
| epv | real data, held-out scored once | mlp | MSE expected points 1.2155 [1.2080, 1.2226] | MSE expected points 1.2362 [1.1899, 1.2812] | sequence model gate: TRAINED; sequence model kept or dropped: KEPT; shuffled-target check |
| actions | fixture output, not a result | hgb | F1 ball_screen 1.000 [1.000, 1.000] | F1 ball_screen 1.000 (point estimate) | plan target: a fixture cannot meet or miss it; detections |
| coverage | fixture output, not a result | hgb | macro F1 0.944 [0.902, 0.979] | macro F1 0.835 (point estimate) | target: beat majority (met) |
| decisions | real data, held-out scored once | xfg_selected | mean Spearman, decision quality vs on-court rating (by game) 0.012 [-0.008, 0.029] | mean Spearman, decision quality vs on-court rating (by game) -0.014 [-0.116, 0.100] | prior rule; player gate |
| playcalling | fixture output, not a result | ridge | possession points MSE 1.3674 [1.3195, 1.4340] | possession points MSE 1.2359 (point estimate) | paired bootstrap, ridge minus mean: excludes zero: no; paired bootstrap, ridge minus by_action: excludes zero: no; recommendation gate: no recommendation is made |
| lineups | real data, held-out scored once | rapm | mean out-of-season Spearman over validation pairs 0.109 (no interval stated) | Spearman 0.065 [-0.488, 0.482] | paired Spearman, rapm minus plus_minus (fit 2020-21, predict 2021-22); paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, predict 2021-22); paired Spearman, rapm minus plus_minus (fit 2020-21, 2021-22, predict 2022-23); paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, 2021-22, predict 2022-23); paired Spearman, rapm minus plus_minus (fit 2020-21, 2021-22, 2022-23, predict 2023-24); paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, 2021-22, 2022-23, predict 2023-24); selection |
| winprob | real data, held-out scored once | logistic | mean bucket Brier (the selection metric) 0.1468 (no interval stated) | mean bucket Brier 0.1486 (no interval stated) | prefix invariance check: 0 violation(s) |
| load | real data, held-out scored once | ridge on FEATURES (decline); logistic on FEATURES (absence) | y_decline MSE (ft/s)^2 0.4929 [0.4740, 0.5178] | y_decline MSE (ft/s)^2 0.5164 (point estimate) | paired bootstrap, decline: excludes zero: True; paired bootstrap, absence: excludes zero: True |
| betting | fixture output, not a result | cover model market, margin model market | cover log loss 0.6931 [0.6931, 0.6931] | cover log loss 0.6931 (point estimate, season 2021-22) | none stated |
| xfg_baseline | real data, held-out scored once | logistic | log loss (out-of-fold) 0.6665 [0.6652, 0.6681] | log loss 0.6735 (point estimate) | none stated |

### xfg

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/xfg.md, written 2026-11-12T21:43:04Z |
| selected model | hgb_contest_prior |
| out-of-fold metric | log loss 0.6294 [0.6277, 0.6316] |
| out-of-fold, selected | Brier 0.2209 [0.2201, 0.2219] |
| out-of-fold, selected | ECE (10 bins) 0.0029 [0.0021, 0.0061] |
| out-of-fold, selected | calibration slope 1.0001 [0.9760, 1.0249] |
| candidates, out-of-fold | location_logistic: log loss 0.6665 [0.6651, 0.6681]; contest_logistic: log loss 0.6552 [0.6538, 0.6571]; hgb_contest_prior: log loss 0.6294 [0.6277, 0.6316] |
| held-out | log loss 0.6232 [0.6132, 0.6327] |
| held-out, selected | Brier 0.2187 [0.2142, 0.2233] |
| ECE under 0.02 (a target, not a selection rule) | selected candidate: yes |
| paired bootstrap, log loss, hgb_contest_prior vs location_logistic | -0.0370 [-0.0385, -0.0354], interval excludes 0: yes |
| paired bootstrap, log loss, contest_logistic vs location_logistic | -0.0113 [-0.0122, -0.0104], interval excludes 0: yes |

### rebounds

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/rebounds.md, written 2026-11-12T21:47:24Z |
| selected model | hgb |
| out-of-fold metric | per-miss log loss 1.6009 [1.5882, 1.6103] |
| out-of-fold, selected | top-1 accuracy 0.4797 [0.4756, 0.4848] |
| candidates, out-of-fold | prior: per-miss log loss 2.1380 [2.1292, 2.1436]; logit: per-miss log loss 1.7558 [1.7418, 1.7660]; hgb: per-miss log loss 1.6009 [1.5882, 1.6103]; uniform reference: per-miss log loss 2.3881 |
| held-out | per-miss log loss 1.7115 (point estimate) |
| held-out, selected | top-1 accuracy 0.4474 (point estimate) |
| paired bootstrap | Paired bootstrap by game, hgb minus prior per-miss log loss: -0.5371 [-0.5477, -0.5283]; interval excludes zero: True (negative favours hgb). |

### matchups

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/matchups.md, written 2026-11-12T21:53:45Z |
| selected model | ridge |
| out-of-fold metric | possession points MSE 1.2854 [1.2788, 1.2916] |
| candidates, out-of-fold | constant: MSE 1.2842 [1.2776, 1.2903]; raw: MSE 1.3244 [1.3153, 1.3324]; ridge: MSE 1.2854 [1.2788, 1.2916] |
| held-out | possession points MSE 1.3195 (point estimate) |
| paired bootstrap, ridge minus raw | -0.0390 [-0.0416, -0.0362] (excludes zero: True) |
| paired bootstrap, ridge minus constant | 0.0012 [0.0002, 0.0020] (excludes zero: True) |
| defender gate | 447 defenders, 368 pass the 300-possession gate. |

### gravity

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/gravity.md, written 2026-11-12T22:00:30Z |
| selected model | ridge (SHIPPED expectation (the plan's model)) |
| out-of-fold metric | MSE of nearest-defender distance (ft^2) 28.119 [27.800, 28.452] |
| out-of-fold, selected | R2 0.416 |
| candidates, out-of-fold | constant: MSE 48.155 [47.801, 48.563]; ridge: MSE 28.119 [27.800, 28.452]; hgb: MSE 22.712 [22.460, 22.942] |
| held-out | MSE of nearest-defender distance (ft^2) 29.913 (point estimate) |
| held-out, selected | constant floor MSE 52.822 (not a selection input) |
| paired bootstrap | Paired bootstrap by game, MSE(hgb) - MSE(ridge): -5.4070 [-5.5550, -5.2801], excludes zero: True (HGB lower: True). |
| player gate | Players (gate: 300 possessions; 362 gated of 447) |
| sign test | Candidates [2594, 201939, 202691, 2744, 2546, 203382, 977, 202331, 200755, 1717] (the top ten gated players by gravity, no claim attached). Games with a difference: 290; positive: 163; one-sided binomial p = 0.0198; mean difference 0.0042 [0.0009, 0.0074] (bootstrap by game). |

### ghosting

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/ghosting.md, written 2026-11-12T23:09:28Z |
| selected model | 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in) |
| out-of-fold metric | 1.0 s position error (ft) 3.657 [3.635, 3.682] |
| out-of-fold, selected | 0.2 s position error (ft) 0.437 [0.433, 0.442] |
| candidates, out-of-fold | 1 persistence (position at t minus 1): 1.0 s error 5.408 [5.392, 5.425]; 2 ridge on the lag-1 features (linear baseline): 1.0 s error 4.541 [4.526, 4.555]; 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in): 1.0 s error 3.657 [3.635, 3.682] |
| held-out | 1.0 s position error (ft) 3.417 (point estimate) |
| held-out, selected | 0.2 s position error (ft) 0.377 (point estimate) |
| deep candidate (MLP, GRU stand-in) | The deep candidate (3) is kept: its 1.0 s error minus the linear baseline's is -0.884 with paired interval [-0.909, -0.858], which excludes zero. |
| validation sign test | Sign test: 346 positive, 258 negative, 0 zero; one-sided p = 0.0002. |

### embeddings

| field | value |
| --- | --- |
| status | real data |
| report | reports/embeddings.md, written 2026-11-12T23:14:04Z |
| selected model | pca |
| out-of-fold metric | 5-NN position accuracy (leave-one-out) 0.768 [0.720, 0.809] |
| out-of-fold, selected | majority baseline 0.413 [0.395, 0.461] |
| out-of-fold, selected | next-season R squared (ridge, out-of-fold) n/a |
| candidates, out-of-fold | pca: 5-NN accuracy 0.768 [0.720, 0.809]; autoencoder: 5-NN accuracy not trained |
| held-out | absent |
| autoencoder gate | PCA 5-NN accuracy 0.768 is at or above the 0.60 gate: the autoencoder is not trained |
| note | Player-seasons: 448, of which 399 sufficient (at least 300 possessions) and 49 insufficient. |
| note | the lane scores no held-out rows: held-out material is set aside before any feature |

### epv

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/epv.md, written 2026-11-13T00:24:51Z |
| selected model | mlp |
| out-of-fold metric | MSE expected points 1.2155 [1.2080, 1.2226] |
| out-of-fold, selected | Brier P(points > 0) 0.2350 [0.2345, 0.2355] |
| candidates, out-of-fold | constant: MSE 1.2822 [1.2745, 1.2901]; ridge: MSE 1.2766 [1.2683, 1.2843]; hgb: MSE 1.2368 [1.2287, 1.2446]; mlp: MSE 1.2155 [1.2080, 1.2226] |
| held-out | MSE expected points 1.2362 [1.1899, 1.2812] |
| held-out, selected | Brier P(points > 0) 0.2318 [0.2298, 0.2340] |
| sequence model gate | Gate decision: the sequence model (torch GRU (hidden 16) over the last 5 states) was TRAINED. |
| sequence model kept or dropped | Paired bootstrap by game of MSE(hgb) - MSE(mlp), hgb being the lowest-MSE candidate: 0.0212 [0.0201, 0.0228]; the sequence model is KEPT (interval above 0 and a lower out-of-fold MSE). |
| shuffled-target check | Possession points permuted across possessions of the training games, gradient boosting refit out-of-fold: MSE 1.2837 vs constant 1.2834, lift -0.0 percent (chance is at or below 0; the real lift above is 3.5 percent). |
| note | Why: mlp has the lowest out-of-fold MSE (1.2155 against the constant's 1.2822, lift 5.2 percent). |

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

### decisions

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/decisions.md, written 2026-11-13T00:30:51Z |
| selected model | xfg_selected |
| out-of-fold metric | mean Spearman, decision quality vs on-court rating (by game) 0.012 [-0.008, 0.029] |
| out-of-fold, selected | positive / signed games 317 / 603 |
| out-of-fold, selected | sign p 0.222 |
| out-of-fold, selected | shuffled mean rho -0.013 |
| held-out | mean Spearman, decision quality vs on-court rating (by game) -0.014 [-0.116, 0.100] |
| prior rule | candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least 0.50. Selected: xfg_selected. |
| player gate | 102 of 446 players are under the 300-decision gate and carry no rank (listed unranked, by id, first 10): |

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

### lineups

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/lineups.md, written 2026-11-13T00:33:18Z |
| selected model | rapm |
| out-of-fold metric | mean out-of-season Spearman over validation pairs 0.109 (no interval stated) |
| out-of-fold, selected | fit 2020-21, predict 2021-22: Spearman 0.065 [-0.472, 0.527] |
| out-of-fold, selected | fit 2020-21, predict 2021-22: MSE 143.132 [73.673, 213.029] |
| out-of-fold, selected | fit 2020-21, 2021-22, predict 2022-23: Spearman 0.222 [-0.170, 0.547] |
| out-of-fold, selected | fit 2020-21, 2021-22, predict 2022-23: MSE 196.717 [64.432, 415.504] |
| out-of-fold, selected | fit 2020-21, 2021-22, 2022-23, predict 2023-24: Spearman 0.041 [-0.428, 0.519] |
| out-of-fold, selected | fit 2020-21, 2021-22, 2022-23, predict 2023-24: MSE 154.511 [85.839, 240.148] |
| candidates, out-of-fold | plus_minus: mean Spearman 0.105; rapm: mean Spearman 0.109; rapm_synergy: mean Spearman 0.087 |
| held-out | Spearman 0.065 [-0.488, 0.482] |
| held-out, selected | MSE 169.239 [76.614, 279.349] |
| paired Spearman, rapm minus plus_minus (fit 2020-21, predict 2021-22) | -0.017 [-0.672, 0.725] |
| paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, predict 2021-22) | -0.062 [-0.713, 0.611] |
| paired Spearman, rapm minus plus_minus (fit 2020-21, 2021-22, predict 2022-23) | 0.149 [-0.290, 0.584] |
| paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, 2021-22, predict 2022-23) | 0.162 [-0.265, 0.540] |
| paired Spearman, rapm minus plus_minus (fit 2020-21, 2021-22, 2022-23, predict 2023-24) | -0.119 [-0.635, 0.268] |
| paired Spearman, rapm_synergy minus plus_minus (fit 2020-21, 2021-22, 2022-23, predict 2023-24) | -0.154 [-0.629, 0.227] |
| selection | Mean validation Spearman: plus_minus 0.105, rapm 0.109, rapm_synergy 0.087 |
| note | 3 validation pair(s); the selection metric is the mean over them |

### winprob

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/winprob.md, written 2026-11-13T00:41:57Z |
| selected model | logistic |
| out-of-fold metric | mean bucket Brier (the selection metric) 0.1468 (no interval stated) |
| out-of-fold, selected | Brier overall 0.1645 [0.1616, 0.1683] |
| out-of-fold, selected | log loss overall 0.4845 [0.4767, 0.4945] |
| candidates, out-of-fold | constant: mean bucket Brier 0.2472; logistic: mean bucket Brier 0.1468; hgb_iso: mean bucket Brier 0.1470 |
| held-out | mean bucket Brier 0.1486 (no interval stated) |
| held-out, selected | Brier overall 0.1658 [0.1590, 0.1732] |
| held-out, selected | log loss overall 0.4879 [0.4700, 0.5077] |
| prefix invariance check | `prefix_invariance_violations` found 0 violation(s) on 2 probed game(s) (a feature at event k is unchanged when the play-by-play is cut after k) |
| note | NBA.com tables were skipped by the owner (gates/nbacom-skip, 2026-11-11), so the NBA.com WinProbabilityPBP benchmark was not collected and the benchmark column is empty. If it is collected later, its probability at each event joins winprob_states on (game_id, event) and is scored with the same bucket Brier. |

### load

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/load.md, written 2026-11-13T00:48:38Z |
| selected model | ridge on FEATURES (decline); logistic on FEATURES (absence) |
| out-of-fold metric | y_decline MSE (ft/s)^2 0.4929 [0.4740, 0.5178] |
| out-of-fold, selected | absent_next log loss 0.2764 [0.2644, 0.2883] |
| candidates, out-of-fold | mean (training-fold mean): MSE 0.5086 [0.4884, 0.5352]; ridge on FEATURES: MSE 0.4929 [0.4740, 0.5178] |
| held-out | y_decline MSE (ft/s)^2 0.5164 (point estimate) |
| held-out, selected | absent_next log loss 0.2776 (point estimate) |
| paired bootstrap, decline | Paired bootstrap by game of MSE(ridge) - MSE(mean): -0.0157 [-0.0197, -0.0118], excludes zero: True. |
| paired bootstrap, absence | Paired bootstrap by game of log loss(logistic) - log loss(base): -0.0481 [-0.0553, -0.0417], excludes zero: True. |

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

### xfg_baseline

| field | value |
| --- | --- |
| status | real data, held-out scored once |
| report | reports/xfg_baseline.md, written 2026-11-13T00:48:43Z |
| selected model | logistic |
| out-of-fold metric | log loss (out-of-fold) 0.6665 [0.6652, 0.6681] |
| out-of-fold, selected | Brier (out-of-fold) 0.2369 [0.2363, 0.2377] |
| out-of-fold, selected | calibration slope (out-of-fold) 0.9979 [0.9643, 1.0234] |
| candidates, out-of-fold | constant: log loss (out-of-fold) 0.6879 [0.6872, 0.6885]; logistic: log loss (out-of-fold) 0.6665 [0.6652, 0.6681] |
| held-out | log loss 0.6735 (point estimate) |
| held-out, selected | Brier 0.2403 (point estimate) |
| gates and checks | none stated |
| note | Phase 1 baseline (analytics/train_xfg.py): a constant floor and the logistic location model, no selection rule |

<!-- results:end -->
