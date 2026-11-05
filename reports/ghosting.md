# Ghosting: defender position from the past five sampled frames (analytics/ghosting/train.py) (fixture output, not a result)

Written 2026-11-05T08:44:13Z by `python -m analytics.ghosting.train --fixtures --games 4 --possessions 6 --heldout-every 4 --folds 5 --seed 0 --bootstrap 200 --report reports/ghosting.md`.

**Fixture output, not a result.** Every number below comes from synthetic games (tests/_fx_ghosting.py or analytics.fixtures); it shows the pipeline runs end to end and nothing about NBA defenders.

## Inputs

| game | held out | possessions | used | skipped (gap, no offence, short, lineup) | sampled steps |
| --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | False | 6 | 5 | 1, 0, 0, 0 | 245 |
| 01.18.2016.HLX.at.LRD | False | 6 | 5 | 1, 0, 0, 0 | 316 |
| 02.04.2016.GRN.at.QCY | False | 6 | 5 | 1, 0, 0, 0 | 294 |
| 12.16.2015.PHX.at.GSW | True | 6 | 5 | 1, 0, 0, 0 | 282 |

Training rows: 4050 defender-steps from 3 game(s) (02.07.2016.OMA.at.DUL, 01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY). Held-out rows set aside before any fit: 1335 from 1 game(s) (12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 3 training game ids against sportvu/splits.json; `schema.assert_time_forward` passed on the feature frame (166 columns, 31 lag-1 features, 155 window columns).

## Protocol

Unit: defender at a sampled frame (5 Hz, every 5 frames of the 25 Hz tracking) inside a possession; assignment = nearest offensive player at the possession's first sampled frame. Features at t minus 1 (and t minus 2 to t minus 5 for the window): rel_asg_dx, rel_asg_dy, rel_off1_dx, rel_off1_dy, rel_off2_dx, rel_off2_dy, rel_off3_dx, rel_off3_dy, rel_off4_dx, rel_off4_dy, rel_off5_dx, rel_off5_dy, rel_ball_dx, rel_ball_dy, rel_def1_dx, rel_def1_dy, rel_def2_dx, rel_def2_dy, rel_def3_dx, rel_def3_dy, rel_def4_dx, rel_def4_dy, ball_basket_dist_ft, def_basket_dist_ft, def_vx, def_vy, def_x, def_y, shot_clock_s, elapsed_s, attack_right.
Target: displacement of the defender from t minus 1 to t. The 1.0 s horizon rolls the ghost forward five steps with its own position fed back and everyone else actual.
Candidates: 1 persistence (position at t minus 1); 2 ridge on the lag-1 features (linear baseline); 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in).
Metrics: out-of-fold under 3 grouped folds by sportvu_game (seed 0); fewer games (3) than the 5 folds asked, so n_folds = number of games (leave-one-game-out); mean position error in feet; intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0).

## Candidates by horizon (out-of-fold, feet)

| candidate | 0.2 s error | rows | 1.0 s error | rows |
| --- | --- | --- | --- | --- |
| 1 persistence (position at t minus 1) | 1.054 [1.002, 1.173] | 4050 | 4.565 [4.289, 5.143] | 3750 |
| 2 ridge on the lag-1 features (linear baseline) | 0.600 [0.532, 0.711] | 4050 | 3.852 [3.296, 4.730] | 3750 |
| 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in) | 0.802 [0.685, 0.864] | 4050 | 4.413 [3.708, 4.814] | 3750 |

## Paired bootstrap by game (difference of mean error, feet; negative favours the first)

| pair | horizon | diff | lo | hi | excludes zero |
| --- | --- | --- | --- | --- | --- |
| mlp vs ridge | 0.2 s | 0.202 | 0.123 | 0.332 | True |
| mlp vs ridge | 1.0 s | 0.561 | 0.084 | 1.199 | True |
| ridge vs persistence | 0.2 s | -0.454 | -0.635 | -0.291 | True |
| ridge vs persistence | 1.0 s | -0.713 | -1.848 | 0.441 | False |

## Decision

The deep candidate (3) is **dropped**: its 1.0 s error minus the linear baseline's is 0.561 with paired interval [0.084, 1.199], which excludes zero. The rule is fixed in advance: keep 3 only when the difference is negative and the interval excludes zero. Selected ghost: 2 ridge on the lag-1 features (linear baseline) (fixture output, not a result).

## Validation: graded deviation against points allowed (out-of-fold rows, selected ghost, 1.0 s horizon)

deviation_ft = distance(actual, assignment) minus distance(ghost, assignment); per possession the mean over its 1.0 s rows; per game the Spearman correlation of that mean with the points the offence scored; then a sign test over games (one-sided, more positive than negative correlations).

| game | possessions | Spearman |
| --- | --- | --- |
| 01.18.2016.HLX.at.LRD | 5 | 0.354 |
| 02.04.2016.GRN.at.QCY | 5 | 0.000 |
| 02.07.2016.OMA.at.DUL | 5 | 0.000 |

Sign test: 1 positive, 0 negative, 2 zero; one-sided p = 0.5000 (fixture output, not a result).

## Held-out games (scored once by the candidates refit on all 4050 training rows; never fitted or tuned on)

1335 defender-steps from 1 held-out game(s): 12.16.2015.PHX.at.GSW.

| candidate | 0.2 s error | 1.0 s error |
| --- | --- | --- |
| 1 persistence (position at t minus 1) | 1.065 | 4.568 |
| 2 ridge on the lag-1 features (linear baseline) | 0.629 | 4.680 |
| 3 MLPRegressor 64x64 on the 5-step window (GRU stand-in) | 0.948 | 5.380 |

Held-out numbers are point estimates (no bootstrap over 1 game(s)). Held-out sign test: 1 positive, 0 negative, p = 0.5000.

## Tables

ghost_frames: 10370 rows; ghost_possessions: 20 rows (docs/contract/ghosting.md).

## Next on the Mac

    python -m analytics.ghosting.train --data "<Drive>/nba_harvest" --report reports/ghosting.md --folds 5 --seed 0 --bootstrap 1000 --out "<Drive>/nba_harvest/analytics"

Torch GRU note: the deep candidate here is sklearn's MLPRegressor over the flattened 5-step window because torch is not installed in the build container. The window columns are ordered lag 1 first, so `X.reshape(n, 5, 31)` is the (batch, time, features) tensor a `torch.nn.GRU` consumes (reverse the time axis for oldest-first); it replaces `fit_mlp` with no change to the features, the roll-forward or the decision rule, and is kept under the same paired-bootstrap test.
