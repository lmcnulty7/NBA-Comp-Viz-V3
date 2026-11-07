# Coverage recognition: pick-and-roll defensive coverage (analytics/coverage)

Written 2026-11-07T08:42:54Z by `python -m analytics.coverage.train --fixtures --fixture-games 8 --fixture-heldout-every 4 --fixture-possessions 40 --report reports/coverage.md --folds 5 --seed 0 --bootstrap 200`.

## Status (fixture output, not a result)

No hand labels exist yet (STATUS.md): this run read the synthetic fixture season (shared generator), so every number below is fixture output, not a result. The generator scripts the two defenders' paths by coverage class, so the classes are separable by construction; the F1 here says nothing about hand labels. The real run awaits the hand-label tables of lane ACTIONS under DATA_ROOT/labels/ on the owner's Mac.

## Inputs (fixture output, not a result)

| item | count |
| --- | --- |
| games with frames | 8 (held out: 2) |
| games with labels but no frames | 0 |
| label reader; rows of another labeller pass that reached the trainer, must be 0 (actions, screens) | fixture; 0, 0 |
| action rows read / kept | 320 / 320 |
| ball_screen actions | 136 |
| screen rows read / kept | 136 / 136 |
| screens dropped: missing value, non-integer, unknown coverage, unknown confidence, duplicate id, no ball_screen action, bad defender ids | 0, 0, 0, 0, 0, 0, 0 |
| ball_screen actions without a screen label (scored, never fitted) | 0 |
| detected screens (label_source model; scored, never fitted) | 0 |
| screens dropped at feature time: no frames, roles untracked, defenders not inferable | 0, 0, 0 |
| training rows (sure-labelled, training games) | 109 from 6 game(s) |
| unsure-labelled rows of training games set aside (scored, never fitted) | 0 |
| held-out rows set aside before any fit | 27 from 2 game(s) |

Guard: `analytics.eval.refuse_heldout` checked 6 training game ids against sportvu/splits.json; `analytics.schema.assert_time_forward` passed on the feature frame.

## Label counts (training rows) (fixture output, not a result)

| class | n | share |
| --- | --- | --- |
| switch | 12 | 0.110 |
| drop | 16 | 0.147 |
| hedge_show | 14 | 0.128 |
| blitz_trap | 22 | 0.202 |
| ice | 17 | 0.156 |
| under | 14 | 0.128 |
| over | 14 | 0.128 |

Confidence over every labelled screen read: 136 sure, 0 unsure. Only sure rows fit (docs/contract/labels.md leaves unsure out of the first training set); unsure rows are scored and the flag never enters a feature.

## Protocol (fixture output, not a result)

Unit: one ball screen. Window: start_frame to end_frame plus 1.5 s (38 frames at 25 Hz), declared as the object being classified; no later frame enters. Features (37): screen_basket_dist_ft, handler_depth_t0_ft, handler_lat_t0_ft, d1_depth_t0_ft, d1_lat_t0_ft, d2_depth_t0_ft, d2_lat_t0_ft, d1_handler_dist_t0_ft, d2_handler_dist_t0_ft, d2_basket_dist_t0_ft, d2_nearest_end, def_dist_diff_end_ft, d1_handler_dist_end_ft, d2_handler_dist_end_ft, d2_depth_end_ft, d2_basket_dist_min_ft, d2_handler_dist_min_early_ft, d2_handler_dist_mean_early_ft, d2_handler_dist_mean_post_ft, d2_handler_dist_late_ft, d2_handler_dist_min_post_ft, d2_handler_dist_min_frac, d2_handler_dist_after_min_ft, d1_depth_rel_screener_pass_ft, d1_lat_rel_handler_end_ft, handler_lat_end_ft, handler_depth_end_ft, both_within_6ft_frac, both_within_6ft_any, d2_screen_dist_end_ft, d2_screen_dist_max_ft, handler_speed_mean_ftps, handler_speed_max_ftps, d1_speed_mean_ftps, d2_speed_mean_ftps, d1_handler_dist_mean_post_ft, n_window_frames.
Candidates: majority (training-fold class frequencies), multinomial logistic regression (median imputation, standardisation, C = 1), HistGradientBoostingClassifier (100 trees, depth 3, learning rate 0.1, min 5 rows per leaf).
Metrics: out-of-fold under 5 grouped folds by sportvu_game (seed 0); macro F1 over the classes present in the truth; intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). Selection: highest out-of-fold macro F1, ties to the simpler model.

## Candidates (out-of-fold, 109 rows, 6 game(s)) (fixture output, not a result)

| model | macro F1 | accuracy | selected |
| --- | --- | --- | --- |
| majority | 0.040 [0.013, 0.055] | 0.101 [0.037, 0.181] |  |
| logistic | 0.763 [0.620, 0.898] | 0.761 [0.652, 0.888] |  |
| hgb | 0.944 [0.902, 0.979] | 0.945 [0.908, 0.979] | yes |

Selected: hgb. Fixture target: beat majority (met).

## Per-class F1 (hgb, out-of-fold) (fixture output, not a result)

| class | F1 |
| --- | --- |
| switch | 0.957 |
| drop | 0.903 |
| hedge_show | 0.966 |
| blitz_trap | 0.957 |
| ice | 0.970 |
| under | 0.929 |
| over | 0.929 |

## Confusion (hgb, out-of-fold): rows true, columns predicted (fixture output, not a result)

| true \ predicted | switch | drop | hedge_show | blitz_trap | ice | under | over |
| --- | --- | --- | --- | --- | --- | --- | --- |
| switch | 11 | 0 | 0 | 1 | 0 | 0 | 0 |
| drop | 0 | 14 | 1 | 1 | 0 | 0 | 0 |
| hedge_show | 0 | 0 | 14 | 0 | 0 | 0 | 0 |
| blitz_trap | 0 | 0 | 0 | 22 | 0 | 0 | 0 |
| ice | 0 | 1 | 0 | 0 | 16 | 0 | 0 |
| under | 0 | 0 | 0 | 0 | 0 | 13 | 1 |
| over | 0 | 0 | 0 | 0 | 0 | 1 | 13 |

## Team rates (predicted class shares per defending team and season, training games; insufficient under 300 screens) (fixture output, not a result)

Season 2015-16:

| team_id | n_screens | share_switch | share_drop | share_hedge_show | share_blitz_trap | share_ice | share_under | share_over | insufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1610612701 | 10 | 0.200 | 0.300 | 0.300 | 0.100 | 0.000 | 0.100 | 0.000 | yes |
| 1610612702 | 8 | 0.000 | 0.125 | 0.000 | 0.375 | 0.250 | 0.125 | 0.125 | yes |
| 1610612703 | 12 | 0.000 | 0.167 | 0.250 | 0.250 | 0.000 | 0.167 | 0.167 | yes |
| 1610612704 | 5 | 0.000 | 0.000 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 | yes |
| 1610612705 | 11 | 0.182 | 0.182 | 0.091 | 0.000 | 0.182 | 0.182 | 0.182 | yes |
| 1610612706 | 4 | 0.000 | 0.250 | 0.500 | 0.000 | 0.250 | 0.000 | 0.000 | yes |
| 1610612709 | 10 | 0.000 | 0.000 | 0.000 | 0.400 | 0.300 | 0.100 | 0.200 | yes |
| 1610612710 | 8 | 0.000 | 0.000 | 0.125 | 0.625 | 0.125 | 0.125 | 0.000 | yes |
| 1610612711 | 9 | 0.000 | 0.111 | 0.222 | 0.111 | 0.222 | 0.111 | 0.222 | yes |
| 1610612712 | 5 | 0.200 | 0.200 | 0.000 | 0.400 | 0.200 | 0.000 | 0.000 | yes |
| 1610612713 | 14 | 0.357 | 0.071 | 0.071 | 0.214 | 0.000 | 0.071 | 0.214 | yes |
| 1610612714 | 13 | 0.077 | 0.231 | 0.077 | 0.077 | 0.231 | 0.231 | 0.077 | yes |

A team marked insufficient has fewer than 300 screens; its shares are shown for completeness and are not to be read.

## Held-out rows (scored once by hgb refit on all 109 training rows; never fitted or tuned on) (fixture output, not a result)

27 screens from 2 held-out game(s): 01.04.2016.F03.at.CLE, 02.08.2016.F07.at.CLE.

| model | macro F1 | accuracy | labelled rows |
| --- | --- | --- | --- |
| hgb | 0.835 | 0.852 | 27 |

Only the selected model is scored on the held-out rows (one look); the numbers are point estimates over 2 game(s), no interval.

## Outputs (under OUT_ROOT) (fixture output, not a result)

- `analytics/coverage/2015-16/coverage_predictions.parquet`
- `analytics/coverage/heldout/2015-16/coverage_predictions.parquet`
- `analytics/coverage/2015-16/coverage_team_rates.parquet`
- `reports/coverage.md (this report, in the repo)`

## On the Mac (where the labels and frames will be) (fixture output, not a result)

    python -m analytics.coverage.train --data "<Drive>/nba_harvest" --out "<Drive>/nba_harvest" --report reports/coverage.md --folds 5 --seed 0 --bootstrap 1000

## Assumptions (fixture output, not a result)

- action_id is the row's position within its game's actions file when the table carries no action_id column.
- The basket a screen attacks is the one nearer the handler at start_frame.
- Detected screens (no defender ids) take the opponent nearest the handler as on-ball defender and the opponent nearest the screener, other than that one, as the screener's defender.
- A held-out game is one with a sportvu/splits.json token, the `.at.CLE` arena rule, or frames under heldout/.
- Team rates use the predicted class of every training-game screen (out-of-fold for labelled rows) so labelled and unlabelled screens are counted the same way; the defending team is the on-ball defender's team.
- The feature frame is checked with `schema.assert_time_forward(frame, "states_1hz")`: the screen table is not one of the five contract tables, so the nearest per-frame table name is passed; the check is against the union of every table's excluded columns whatever the name.
- The features assume start_frame is the screen moment. In the shared fixture (analytics.fixtures) a ball_screen action opens with a 1 s approach and a 1 s hold before the drive, so the `_t0` features describe the approach there; the window-wide features carry the class. The hand-label convention is confirmed once labels exist.
