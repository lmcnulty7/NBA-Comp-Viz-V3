# Play calling (fixture output, not a result): Association only, not causal: every number below is an observational comparison of what followed each action against each coverage, with a bootstrap interval by game and a sample gate. Nothing here estimates what would happen if a different play were called.

Written 2026-11-07T08:35:44Z by `python -m analytics.playcalling.train --fixtures --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 200`.

**Fixture output, not a result.** Every number below comes from synthetic inputs (analytics/playcalling/synthetic.py on the shared generator); the real run is the command under Status.

## Inputs (fixture output, not a result)

- frames: rows=971060, games=8, duplicates_dropped=0, shot_clock_out_of_range=0, shot_clock_null=0, period_out_of_range=0
- possessions: rows=320, games=8, duplicates_dropped=0, points_out_of_range=0, start_after_end=0, null_offense_team=0, unknown_end_kind={}, null_end_kind=0
- screens: rows=142, games=8, duplicates_dropped=0, unknown_coverage={}, null_coverage=0
- coverage_predictions: rows=142, games=8, duplicates_dropped=0, unknown_predicted={}, null_predicted=0, p_out_of_range=0, p_not_summing_to_one=0
- epv_states: rows=1600, games=8, duplicates_dropped=0, epv_not_finite=0, epv_out_of_range=0
- actions: rows=320, games=8, duplicates_dropped=0, unknown_action_type={}, null_action_type=0, unknown_label_source={}, null_label_source=0, start_after_end=0, null_ball_handler_id=0, null_screener_id=116, possession_missing=0, outside_possession_span=0
- heldout_games: ["11.26.2015.DUL.at.CLE", "12.16.2015.PHX.at.GSW"]

Build: possessions=320, no_first_frame=0, not_five_v_five=0, no_offense_team=0, actions_in=320, actions_without_possession=0, rows=320, epv_gain_rows=320, coverage_source={"none": 178, "hand": 142}, null_lineups=0.
Training rows: 240 actions from 6 game(s). Held-out rows set aside before any fit: 80 actions from 2 game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW).
Guard: `analytics.eval.refuse_heldout` checked 6 training game ids; `analytics.schema.assert_time_forward` passed on the feature frame (action_type, coverage, off_lineup, def_lineup, score_margin, shot_clock_s, period).

## The caveat (fixture output, not a result)

The unit is an action event; the outcomes are the containing possession's points (PPP) and, when epv_states exist, the EPV gained across the action. Which action gets called depends on the defence, the lineup, the clock and the score, so a difference in PPP between action types is an association, not the effect of the call. Intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0); a group with fewer than 300 rows is marked insufficient and its number is not to be quoted.

## 1. PPP by action type (fixture output, not a result)

| action_type | n | ppp | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- |
| cut | 23 | 1.391 | 1.111 | 1.714 | yes |
| ball_screen | 103 | 1.330 | 1.183 | 1.512 | yes |
| isolation | 34 | 1.265 | 0.860 | 1.758 | yes |
| drive | 20 | 1.000 | 0.598 | 1.601 | yes |
| post_up | 14 | 1.000 | 0.428 | 1.600 | yes |
| off_ball_screen | 27 | 0.815 | 0.531 | 1.148 | yes |
| handoff | 19 | 0.789 | 0.478 | 1.105 | yes |

## 2. PPP by action type x coverage (fixture output, not a result)

| coverage | action_type | n | ppp | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 9 | 1.778 | 1.249 | 2.335 | yes |
| drop | ball_screen | 19 | 1.211 | 0.913 | 1.563 | yes |
| hedge_show | ball_screen | 10 | 1.100 | 0.500 | 1.751 | yes |
| ice | ball_screen | 12 | 0.917 | 0.000 | 1.813 | yes |
| none | cut | 23 | 1.391 | 1.111 | 1.714 | yes |
| none | isolation | 34 | 1.265 | 0.860 | 1.758 | yes |
| none | drive | 20 | 1.000 | 0.598 | 1.601 | yes |
| none | post_up | 14 | 1.000 | 0.428 | 1.600 | yes |
| none | off_ball_screen | 27 | 0.815 | 0.531 | 1.148 | yes |
| none | handoff | 19 | 0.789 | 0.478 | 1.105 | yes |
| over | ball_screen | 20 | 1.200 | 0.768 | 1.551 | yes |
| switch | ball_screen | 13 | 1.231 | 0.333 | 1.938 | yes |
| under | ball_screen | 20 | 1.800 | 1.276 | 2.300 | yes |

## 2b. EPV gained across the action, by action type and by action type x coverage (fixture output, not a result)

epv_gain = epv_points at the action's end frame minus at its start frame (EPV lane output); 240 rows carried it.

| action_type | n | epv_gain | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- |
| ball_screen | 103 | 0.068 | 0.035 | 0.099 | yes |
| cut | 23 | 0.060 | 0.021 | 0.096 | yes |
| post_up | 14 | 0.031 | -0.104 | 0.173 | yes |
| isolation | 34 | 0.014 | -0.048 | 0.081 | yes |
| drive | 20 | -0.017 | -0.053 | 0.018 | yes |
| handoff | 19 | -0.031 | -0.097 | 0.033 | yes |
| off_ball_screen | 27 | -0.056 | -0.115 | 0.024 | yes |

| coverage | action_type | n | epv_gain | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 9 | 0.112 | 0.026 | 0.206 | yes |
| drop | ball_screen | 19 | 0.070 | -0.026 | 0.184 | yes |
| hedge_show | ball_screen | 10 | 0.053 | -0.052 | 0.152 | yes |
| ice | ball_screen | 12 | -0.051 | -0.225 | 0.133 | yes |
| none | cut | 23 | 0.060 | 0.021 | 0.096 | yes |
| none | post_up | 14 | 0.031 | -0.104 | 0.173 | yes |
| none | isolation | 34 | 0.014 | -0.048 | 0.081 | yes |
| none | drive | 20 | -0.017 | -0.053 | 0.018 | yes |
| none | handoff | 19 | -0.031 | -0.097 | 0.033 | yes |
| none | off_ball_screen | 27 | -0.056 | -0.115 | 0.024 | yes |
| over | ball_screen | 20 | 0.107 | 0.019 | 0.179 | yes |
| switch | ball_screen | 13 | 0.027 | -0.177 | 0.168 | yes |
| under | ball_screen | 20 | 0.112 | 0.003 | 0.217 | yes |

## 3. Ridge adjustment (fixture output, not a result)

Possession points regressed on indicator columns of action type, coverage, action type x coverage, offensive lineup, defensive lineup (lineups from the possession's first frame) and the binned score margin at the action start (trailing by more than 5, within 5, leading by more than 5), the last three groups being controls averaged out of the adjusted PPP. Alpha grid [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]; out-of-fold under 5 grouped folds by sportvu_game (seed 0), ridge alpha chosen by 3 inner grouped folds. Outer-fold alphas: [300.0, 100.0, 300.0, 100.0, 100.0]. Final alpha on all training rows: 300.0 (grouped-CV MSE by alpha: 0.3: 1.9052, 1.0: 1.7847, 3.0: 1.6352, 10.0: 1.4880, 30.0: 1.4027, 100.0: 1.3635, 300.0: 1.3560, 1000.0: 1.3565).

| model | out-of-fold MSE | lo | hi |
| --- | --- | --- | --- |
| mean | 1.3580 | 1.3008 | 1.4412 |
| by_action | 1.3608 | 1.2657 | 1.4517 |
| ridge | 1.3674 | 1.3195 | 1.4340 |

Paired bootstrap by game of the MSE difference (ridge minus baseline; negative favours the ridge):

- ridge minus mean: 0.0093 [-0.0070, 0.0270], interval excludes zero: no
- ridge minus by_action: 0.0065 [-0.0581, 0.0521], interval excludes zero: no

## 4. Recommendation table (association) (fixture output, not a result)

Per coverage, the adjusted PPP of each action type: the ridge prediction with the lineup effects averaged over the training rows. recommended marks the highest adjusted PPP among the cells with at least 300 rows; a coverage with no sufficient cell has no recommendation. This is the action type that was followed by the most points against that coverage after a linear lineup adjustment, not the action that would score most if called.

| coverage | action_type | adjusted_ppp | lo | hi | n | insufficient | recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blitz_trap | ball_screen | 1.250 | 1.096 | 1.408 | 9 | yes | no |
| drop | ball_screen | 1.217 | 1.118 | 1.333 | 19 | yes | no |
| hedge_show | ball_screen | 1.211 | 1.052 | 1.407 | 10 | yes | no |
| ice | ball_screen | 1.196 | 0.998 | 1.417 | 12 | yes | no |
| none | cut | 1.180 | 1.018 | 1.335 | 23 | yes | no |
| none | isolation | 1.171 | 1.027 | 1.305 | 34 | yes | no |
| none | post_up | 1.137 | 0.967 | 1.306 | 14 | yes | no |
| none | drive | 1.131 | 0.944 | 1.307 | 20 | yes | no |
| none | handoff | 1.110 | 0.937 | 1.271 | 19 | yes | no |
| none | off_ball_screen | 1.099 | 0.954 | 1.242 | 27 | yes | no |
| over | ball_screen | 1.216 | 1.064 | 1.396 | 20 | yes | no |
| switch | ball_screen | 1.220 | 1.088 | 1.362 | 13 | yes | no |
| under | ball_screen | 1.287 | 1.114 | 1.465 | 20 | yes | no |

No coverage has a cell with at least 300 rows, so no recommendation is made.

## Held-out rows (fixture output, not a result)

80 actions from 2 held-out game(s) (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW), scored ONCE by the selected model (the ridge) refit on all 240 training rows; never fitted or tuned on. The training mean is shown as a labelled floor only; no other candidate is scored on held-out rows.

| model | held-out MSE |
| --- | --- |
| ridge (selected) | 1.2359 |
| training mean (floor) | 1.2336 |

Point estimates only (no interval over 2 game(s)).

## Status (fixture output, not a result)

This is a fixture run on synthetic inputs: fixture output, not a result. The real run awaits the actions, coverage and EPV lane outputs and the possessions tables on the owner's Mac, with the exact command

    python -m analytics.playcalling.train --data DATA_ROOT --out OUT_ROOT --report reports/playcalling.md --folds 5 --seed 0 --bootstrap 1000 --n-min 300

No tables written (no --out).
