# Action recognition: ball-screen detection (analytics/actions/train.py)

Written 2026-11-05T08:54:30Z by `python -m analytics.actions.train --fixtures --report reports/actions.md --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are synthetic games from analytics.fixtures (shared, lane I1) with scripted screens, drives and isolations. The real run awaits the SportVU frames, possessions and hand labels on the owner's Mac (command under Status).

## Inputs (fixture output, not a result)

| game | held-out | frames | possessions | action labels | screen labels | validation |
| --- | --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | no | 7044 | 24 | 24 | 12 | ok |
| 01.18.2016.HLX.at.LRD | no | 8043 | 24 | 24 | 9 | ok |
| 12.16.2015.PHX.at.GSW | yes | 7558 | 24 | 24 | 13 | ok |
| 02.04.2016.GRN.at.QCY | no | 7259 | 24 | 24 | 8 | ok |
| 12.15.2015.ABQ.at.GRN | no | 7304 | 24 | 24 | 9 | ok |
| 11.27.2015.GRN.at.CLE | yes | 6950 | 24 | 24 | 10 | ok |

Label counts by action_type over every game (hand-label schema; only ball_screen, drive and isolation are detected): ball_screen 61, cut 7, drive 16, handoff 16, isolation 17, off_ball_screen 12, post_up 15.
Label counts by label_source: synthetic 144.
Training games: 4 (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 12.15.2015.ABQ.at.GRN). Held-out games set aside before any fit: 2 (11.27.2015.GRN.at.CLE, 12.16.2015.PHX.at.GSW). Skipped possessions (null offense or no rows): 0. Games skipped by validation: none.
Guard: `analytics.eval.refuse_heldout` checked 4 training game ids against sportvu/splits.json; `analytics.schema.assert_time_forward` and `features.assert_no_leak` passed on the training feature frame (197 rows, 21 features).

## Candidate generation (fixture output, not a result)

Rules (5 Hz): ball_screen = 1 s window, teammate within 4 ft of the handler's defender, under 3 ft per s, handler within 12 ft; drive = 0.6 s window, handler closing on the basket over 8 ft per s; isolation = 2 s window, same handler with the ball, no teammate within 10 ft. Target: IoU over frames with a label of at least 0.50.

Candidates: 197 over 96 possessions (2.05 per possession); 16 overlap a label of a type this detector does not model (counted as none).

Rule class (rows) against label class (columns), training games:

| rule_class | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 38 | 0 | 0 | 36 |
| drive | 0 | 11 | 0 | 78 |
| isolation | 0 | 0 | 10 | 24 |

Candidate recall against the labels (training games), IoU >= 0.50:

| class | labels | hit_any | hit_same_class | recall_any | recall_same_class |
| --- | --- | --- | --- | --- | --- |
| ball_screen | 38 | 38 | 38 | 1.000 | 1.000 |
| drive | 11 | 11 | 11 | 1.000 | 1.000 |
| isolation | 11 | 10 | 10 | 0.909 | 0.909 |

## Protocol

Features (21, positions and velocities inside the window): scr_def_dist_min, scr_def_dist_mean, scr_speed_mean, scr_speed_max, handler_speed_mean, handler_speed_max, handler_dir_change_rad, handler_radial_speed_mean, handler_basket_dist_start, handler_basket_dist_end, def_handler_dist_mean, def_recovery_ft, ball_z_mean, ball_z_max, shot_clock_start, n_within_8ft_mean, teammate_min_dist_mean, window_s, rule_is_screen, rule_is_drive, rule_is_iso. `def_recovery_ft` is the one declared post-window feature (defender to handler distance over the 1 s after the window minus inside it); no feature reads the possession outcome and no label column is a feature.
Models: rule (the generator's own class) and hgb (HistGradientBoostingClassifier, {'max_iter': 100, 'learning_rate': 0.1, 'max_leaf_nodes': 15, 'min_samples_leaf': 5, 'l2_regularization': 0.1}) over ['ball_screen', 'drive', 'isolation', 'none'].
Metrics: out-of-fold under 4 grouped folds by sportvu_game (seed 0); fewer games (4) than the 5 folds asked, so n_folds = number of games (leave-one-game-out); the ball_screen F1 interval is the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). Selection: out-of-fold ball_screen F1, ties to the rule baseline.

## Metrics (out-of-fold, 197 candidates, 4 training game(s)) (fixture output, not a result)

| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.679 [0.581, 0.800] | 0.220 | 0.455 | 0.000 | 0.338 | 0.299 |
| hgb | 0.974 [0.941, 1.000] | 0.900 | 0.947 | 0.982 | 0.951 | 0.975 |

Selected model: **hgb** (plan target: ball_screen F1 over 0.8 on a held-out game; a fixture cannot meet or miss it).

Confusion of the selected model (rows true, columns predicted, out-of-fold):

| true \ pred | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 37 | 0 | 0 | 1 |
| drive | 0 | 9 | 0 | 2 |
| isolation | 0 | 0 | 9 | 1 |
| none | 1 | 0 | 0 | 137 |

Detected actions (selected class other than none with probability above 0.5): 56 of 197 training candidates.

## Held-out games (scored once by the models refit on all 197 training candidates; never fitted or tuned on) (fixture output, not a result)

104 candidates from 2 held-out game(s): 11.27.2015.GRN.at.CLE, 12.16.2015.PHX.at.GSW.

| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.730 | 0.185 | 0.500 | 0.000 | 0.354 | 0.317 |
| hgb | 0.978 | 1.000 | 1.000 | 0.993 | 0.993 | 0.990 |

Held-out numbers are point estimates (no bootstrap over 2 game(s)). Confusion of the selected model on the held-out candidates:

| true \ pred | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 22 | 0 | 0 | 1 |
| drive | 0 | 5 | 0 | 0 |
| isolation | 0 | 0 | 5 | 0 |
| none | 0 | 0 | 0 | 71 |

## Outputs

No table written (no --out). With --out OUT_ROOT the action_candidates and actions_detected tables of docs/contract/actions.md go under OUT_ROOT/analytics/actions/<season>/.

## Status

Fixture output, not a result: this run shows the pipeline works end to end on scripted motion. The real run, on the owner's Mac where the SportVU frames, the possessions tables and the hand labels (DATA_ROOT/labels/actions/<game>.jsonl, lane I4's tool) live, is

    python -m analytics.actions.train --data "$DATA_ROOT" --report reports/actions.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

Needs real data before any number means anything: hand-labelled ball screens, drives and isolations on training games (the plan asks for over 0.8 ball_screen F1 on a held-out game, judged once on the held-out block above), and the possessions table of every labelled game.
