# Ghosting: defender position from the past five sampled frames (analytics/ghosting/train.py) (fixture output, not a result)

Written 2026-11-05T09:44:28Z by `python -m analytics.ghosting.train --fixtures --games 4 --possessions 6 --heldout-every 4 --folds 5 --seed 0 --bootstrap 200 --report reports/ghosting.md`.

**Fixture output, not a result.** Every number below comes from synthetic games (tests/_fx_ghosting.py or analytics.fixtures); it shows the pipeline runs end to end and nothing about NBA defenders.

## Inputs

| game | held out | possessions | used | skipped (gap, no offence, short, lineup) | sampled steps |
| --- | --- | --- | --- | --- | --- |
| 03.08.2016.GRN.at.HLX | False | 6 | 5 | 1, 0, 0, 0 | 379 |
| 12.10.2015.GRN.at.DUL | False | 6 | 5 | 1, 0, 0, 0 | 302 |
| 02.10.2016.QCY.at.FRS | False | 6 | 5 | 1, 0, 0, 0 | 264 |
| 12.16.2015.PHX.at.GSW | True | 6 | 5 | 1, 0, 0, 0 | 399 |

Training rows: 4500 defender-steps from 3 game(s) (03.08.2016.GRN.at.HLX, 12.10.2015.GRN.at.DUL, 02.10.2016.QCY.at.FRS). Held-out rows set aside before any fit: 1920 from 1 game(s) (12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 3 training game ids against sportvu/splits.json; `schema.assert_time_forward` passed on the feature frame (166 columns, 31 lag-1 features, 155 window columns).

## Protocol

Unit: defender at a sampled frame (5 Hz, every 5 frames of the 25 Hz tracking) inside a possession; assignment = nearest offensive player at the possession's first sampled frame. Features at t minus 1 (and t minus 2 to t minus 5 for the window): rel_asg_dx, rel_asg_dy, rel_off1_dx, rel_off1_dy, rel_off2_dx, rel_off2_dy, rel_off3_dx, rel_off3_dy, rel_off4_dx, rel_off4_dy, rel_off5_dx, rel_off5_dy, rel_ball_dx, rel_ball_dy, rel_def1_dx, rel_def1_dy, rel_def2_dx, rel_def2_dy, rel_def3_dx, rel_def3_dy, rel_def4_dx, rel_def4_dy, ball_basket_dist_ft, def_basket_dist_ft, def_vx, def_vy, def_x, def_y, shot_clock_s, elapsed_s, attack_right.
Target: displacement of the defender from t minus 1 to t. The 1.0 s horizon rolls the ghost forward five steps with its own position fed back and everyone else actual.
Candidates: 1 persistence (position at t minus 1); 2 ridge on the lag-1 features (linear baseline); 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in).
Metrics: out-of-fold under 3 grouped folds by sportvu_game (seed 0); fewer games (3) than the 5 folds asked, so n_folds = number of games (leave-one-game-out); mean position error in feet; intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates by horizon (out-of-fold, feet)

| candidate | 0.2 s error | rows | 1.0 s error | rows |
| --- | --- | --- | --- | --- |
| 1 persistence (position at t minus 1) | 0.952 [0.835, 1.100] | 4500 | 4.015 [3.425, 4.672] | 4200 |
| 2 ridge on the lag-1 features (linear baseline) | 0.556 [0.535, 0.585] | 4500 | 3.874 [3.356, 4.304] | 4200 |
| 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in) | 0.864 [0.790, 0.908] | 4500 | 4.968 [4.781, 5.153] | 4200 |

## Paired bootstrap by game (difference of mean error, feet; negative favours the first)

| pair | horizon | diff | lo | hi | excludes zero |
| --- | --- | --- | --- | --- | --- |
| mlp vs ridge | 0.2 s | 0.308 | 0.253 | 0.338 | True |
| mlp vs ridge | 1.0 s | 1.094 | 0.849 | 1.425 | True |
| ridge vs persistence | 0.2 s | -0.396 | -0.563 | -0.250 | True |
| ridge vs persistence | 1.0 s | -0.142 | -1.316 | 0.879 | False |

## Decision

The deep candidate (3) is **dropped**: its 1.0 s error minus the linear baseline's is 1.094 with paired interval [0.849, 1.425], which excludes zero. The rule is fixed in advance: keep 3 only when the difference is negative and the interval excludes zero. Selected ghost: 2 ridge on the lag-1 features (linear baseline) (fixture output, not a result).

## Validation: graded deviation against points allowed (out-of-fold rows, selected ghost, 1.0 s horizon)

deviation_ft = distance(actual, assignment) minus distance(ghost, assignment); per possession the mean over its 1.0 s rows; per game the Spearman correlation of that mean with the points the offence scored; then a sign test over games (one-sided, more positive than negative correlations).

| game | possessions | Spearman |
| --- | --- | --- |
| 02.10.2016.QCY.at.FRS | 5 | -0.289 |
| 03.08.2016.GRN.at.HLX | 5 | 0.447 |
| 12.10.2015.GRN.at.DUL | 5 | n/a |

Sign test: 1 positive, 1 negative, 0 zero; one-sided p = 0.7500 (fixture output, not a result).

## Held-out games (the selected candidate refit on all 4500 training rows, scored once; never fitted or tuned on)

1920 defender-steps from 1 held-out game(s): 12.16.2015.PHX.at.GSW. Only the selected candidate and the persistence floor are scored here; the dropped candidate never sees the held-out game.

| candidate | 0.2 s error | 1.0 s error |
| --- | --- | --- |
| 1 persistence (position at t minus 1) | 0.870 | 3.584 |
| 2 ridge on the lag-1 features (linear baseline) | 0.524 | 3.623 |

Held-out numbers are point estimates (no bootstrap over 1 game(s)). Held-out sign test: 1 positive, 0 negative, p = 0.5000.

## Tables

ghost_frames: 12440 rows; ghost_possessions: 20 rows (docs/contract/ghosting.md).

## Assumptions and what needs real data

- Assignment is the nearest offensive player at the possession's first sampled frame (defender-centric, no one-to-one matching); the attacking side is the half the ball is in at t minus 1; a null shot clock is max(24 minus elapsed, 0); possessions with a clock gap or no offence are skipped; a sequence is cut at the first sampled frame without the first frame's ten players or without a ball.
- The 1.0 s roll-forward keeps the offence, the ball and the other defenders actual and feeds only this defender's ghost back.
- Fits run single-threaded (THREADS = 1) because the build container shares 4 cores across 21 lanes; `--threads` raises it on the Mac.
- Real data: the Step 1 frames and the section 19 possessions of every tracked game on the owner's Mac (DATA_ROOT layout); the held-out games are scored once, after the decision, by the candidates refit on every training row. The sign test needs dozens of games to say anything; with the fixtures it is a pipeline check.

## Next on the Mac

    python -m analytics.ghosting.train --data "<Drive>/nba_harvest" --report reports/ghosting.md --folds 5 --seed 0 --bootstrap 1000 --out "<Drive>/nba_harvest"

Torch GRU note: the deep candidate here is sklearn's MLPRegressor over the flattened 5-step window because torch is not installed in the build container. The window columns are ordered lag 1 first, so `X.reshape(n, 5, 31)` is the (batch, time, features) tensor a `torch.nn.GRU` consumes (reverse the time axis for oldest-first); it replaces `fit_mlp` with no change to the features, the roll-forward or the decision rule, and is kept under the same paired-bootstrap test.
