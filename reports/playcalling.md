# Play calling: Association only, not causal: every number below is an observational comparison of what followed each action against each coverage, with a bootstrap interval by game and a sample gate. Nothing here estimates what would happen if a different play were called.

Written 2026-11-05T08:57:35Z by `python -m analytics.playcalling.train --fixtures --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 200`. FIXTURE OUTPUT, NOT A RESULT: the inputs are synthetic.

## Inputs

- frames: rows=966540, games=8, duplicates_dropped=0, shot_clock_out_of_range=0, shot_clock_null=0, period_out_of_range=0
- possessions: rows=320, games=8, duplicates_dropped=0, points_out_of_range=0, start_after_end=0, null_offense_team=0, unknown_end_kind={}, null_end_kind=0
- screens: rows=136, games=8, duplicates_dropped=0, unknown_coverage={}, null_coverage=0
- coverage_predictions: rows=136, games=8, duplicates_dropped=0, unknown_predicted={}, null_predicted=0, p_out_of_range=0, p_not_summing_to_one=0
- epv_states: rows=1600, games=8, duplicates_dropped=0, epv_not_finite=0, epv_out_of_range=0
- actions: rows=320, games=8, duplicates_dropped=0, unknown_action_type={}, null_action_type=0, unknown_label_source={}, null_label_source=0, start_after_end=0, null_ball_handler_id=0, null_screener_id=134, possession_missing=0, outside_possession_span=0
- heldout_games: ["11.11.2015.HLX.at.CLE", "12.16.2015.PHX.at.GSW"]

Build: possessions=320, no_first_frame=0, not_five_v_five=0, no_offense_team=0, actions_in=320, actions_without_possession=0, rows=320, epv_gain_rows=320, coverage_source={"none": 184, "hand": 136}, null_lineups=0.
Training rows: 240 actions from 6 game(s). Held-out rows set aside before any fit: 80 actions from 2 game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (action_type, coverage, off_lineup, def_lineup, score_margin, shot_clock_s, period).

## The caveat

The unit is an action event; the outcomes are the containing possession's points (PPP) and, when epv_states exist, the EPV gained across the action. Which action gets called depends on the defence, the lineup, the clock and the score, so a difference in PPP between action types is an association, not the effect of the call. Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0); a group with fewer than 300 rows is marked insufficient and its number is not to be quoted.

## 1. PPP by action type (fixture output, not a result)

| action_type | n | ppp | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- |
| off_ball_screen | 19 | 1.737 | 1.154 | 2.058 | yes |
| cut | 22 | 1.364 | 0.998 | 2.000 | yes |
| handoff | 19 | 1.316 | 0.733 | 1.714 | yes |
| drive | 22 | 1.273 | 1.000 | 1.500 | yes |
| isolation | 28 | 1.179 | 0.642 | 1.668 | yes |
| ball_screen | 109 | 1.110 | 0.990 | 1.250 | yes |
| post_up | 21 | 0.952 | 0.667 | 1.143 | yes |

## 2. PPP by action type x coverage (fixture output, not a result)

| coverage | action_type | n | ppp | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 22 | 1.091 | 0.688 | 1.324 | yes |
| drop | ball_screen | 16 | 0.938 | 0.363 | 1.468 | yes |
| hedge_show | ball_screen | 14 | 1.000 | 0.426 | 1.412 | yes |
| ice | ball_screen | 17 | 1.176 | 0.397 | 1.885 | yes |
| none | off_ball_screen | 19 | 1.737 | 1.154 | 2.058 | yes |
| none | cut | 22 | 1.364 | 0.998 | 2.000 | yes |
| none | handoff | 19 | 1.316 | 0.733 | 1.714 | yes |
| none | drive | 22 | 1.273 | 1.000 | 1.500 | yes |
| none | isolation | 28 | 1.179 | 0.642 | 1.668 | yes |
| none | post_up | 21 | 0.952 | 0.667 | 1.143 | yes |
| over | ball_screen | 14 | 0.571 | 0.000 | 1.200 | yes |
| switch | ball_screen | 12 | 1.750 | 1.624 | 2.000 | yes |
| under | ball_screen | 14 | 1.357 | 0.692 | 1.814 | yes |

## 2b. EPV gained across the action, by action type and by action type x coverage (fixture output, not a result)

epv_gain = epv_points at the action's end frame minus at its start frame (EPV lane output); 240 rows carried it.

| action_type | n | epv_gain | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- |
| cut | 22 | 0.077 | 0.029 | 0.145 | yes |
| off_ball_screen | 19 | 0.058 | -0.050 | 0.109 | yes |
| post_up | 21 | 0.050 | -0.050 | 0.112 | yes |
| ball_screen | 109 | 0.034 | -0.005 | 0.069 | yes |
| isolation | 28 | 0.017 | -0.085 | 0.093 | yes |
| drive | 22 | 0.016 | -0.009 | 0.037 | yes |
| handoff | 19 | 0.012 | -0.095 | 0.110 | yes |

| coverage | action_type | n | epv_gain | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 22 | 0.019 | -0.046 | 0.081 | yes |
| drop | ball_screen | 16 | -0.017 | -0.103 | 0.069 | yes |
| hedge_show | ball_screen | 14 | -0.011 | -0.136 | 0.079 | yes |
| ice | ball_screen | 17 | 0.005 | -0.130 | 0.116 | yes |
| none | cut | 22 | 0.077 | 0.029 | 0.145 | yes |
| none | off_ball_screen | 19 | 0.058 | -0.050 | 0.109 | yes |
| none | post_up | 21 | 0.050 | -0.050 | 0.112 | yes |
| none | isolation | 28 | 0.017 | -0.085 | 0.093 | yes |
| none | drive | 22 | 0.016 | -0.009 | 0.037 | yes |
| none | handoff | 19 | 0.012 | -0.095 | 0.110 | yes |
| over | ball_screen | 14 | -0.054 | -0.259 | 0.128 | yes |
| switch | ball_screen | 12 | 0.226 | 0.165 | 0.264 | yes |
| under | ball_screen | 14 | 0.120 | -0.037 | 0.244 | yes |

## 3. Ridge adjustment (fixture output, not a result)

Possession points regressed on indicator columns of action type, coverage, action type x coverage, offensive lineup, defensive lineup (lineups from the possession's first frame) and the binned score margin at the action start (trailing by more than 5, within 5, leading by more than 5), the last three groups being controls averaged out of the adjusted PPP. Alpha grid [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]; out-of-fold under 5 grouped folds by sportvu_game (seed 0), ridge alpha chosen by 3 inner grouped folds. Outer-fold alphas: [100.0, 300.0, 300.0, 100.0, 300.0]. Final alpha on all training rows: 100.0 (grouped-CV MSE by alpha: 0.3: 1.4433, 1.0: 1.3993, 3.0: 1.3523, 10.0: 1.3074, 30.0: 1.2795, 100.0: 1.2735, 300.0: 1.2791, 1000.0: 1.2844).

| model | out-of-fold MSE | lo | hi |
| --- | --- | --- | --- |
| mean | 1.2878 | 1.2528 | 1.3345 |
| by_action | 1.2915 | 1.2436 | 1.3533 |
| ridge | 1.2782 | 1.2493 | 1.3246 |

Paired bootstrap by game of the MSE difference (ridge minus baseline; negative favours the ridge):

- ridge minus mean: -0.0096 [-0.0240, 0.0061], interval excludes zero: no
- ridge minus by_action: -0.0133 [-0.0425, 0.0128], interval excludes zero: no

## 4. Recommendation table (association) (fixture output, not a result)

Per coverage, the adjusted PPP of each action type: the ridge prediction with the lineup effects averaged over the training rows. recommended marks the highest adjusted PPP among the cells with at least 300 rows; a coverage with no sufficient cell has no recommendation. This is the action type that was followed by the most points against that coverage after a linear lineup adjustment, not the action that would score most if called.

| coverage | action_type | adjusted_ppp | lo | hi | n | insufficient | recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 1.150 | 1.018 | 1.252 | 22 | yes | no |
| drop | ball_screen | 1.113 | 0.924 | 1.297 | 16 | yes | no |
| hedge_show | ball_screen | 1.129 | 0.949 | 1.259 | 14 | yes | no |
| ice | ball_screen | 1.175 | 0.979 | 1.400 | 17 | yes | no |
| none | off_ball_screen | 1.384 | 1.206 | 1.531 | 19 | yes | no |
| none | cut | 1.282 | 1.107 | 1.436 | 22 | yes | no |
| none | handoff | 1.260 | 1.055 | 1.420 | 19 | yes | no |
| none | drive | 1.250 | 1.151 | 1.324 | 22 | yes | no |
| none | isolation | 1.212 | 1.006 | 1.398 | 28 | yes | no |
| none | post_up | 1.157 | 1.044 | 1.242 | 21 | yes | no |
| over | ball_screen | 1.045 | 0.916 | 1.198 | 14 | yes | no |
| switch | ball_screen | 1.284 | 1.179 | 1.378 | 12 | yes | no |
| under | ball_screen | 1.208 | 1.060 | 1.375 | 14 | yes | no |

No coverage has a cell with at least 300 rows, so no recommendation is made.

## Held-out rows (fixture output, not a result)

80 actions from 2 held-out game(s) (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW), scored ONCE by the final models refit on all 240 training rows; never fitted or tuned on.

| model | held-out MSE |
| --- | --- |
| mean | 1.3653 |
| by_action | 1.3488 |
| ridge | 1.3788 |

Point estimates only (no interval over 2 game(s)).

## Status

This is a fixture run on synthetic inputs: fixture output, not a result. The real run awaits the actions, coverage and EPV lane outputs and the possessions tables on the owner's Mac, with the exact command

    python -m analytics.playcalling.train --data DATA_ROOT --out OUT_ROOT --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 1000 --n-min 300

No tables written (no --out).
