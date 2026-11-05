# Action recognition: ball-screen detection (analytics/actions/train.py)

Written 2026-11-05T09:44:36Z by `python -m analytics.actions.train --fixtures --report reports/actions.md --bootstrap 200`.

**Every number below is fixture output, not a result**: the inputs are synthetic games from analytics.fixtures (shared, lane I1) with scripted screens, drives and isolations. The real run awaits the SportVU frames, possessions and hand labels on the owner's Mac (command under Status).

## Inputs (fixture output, not a result)

| game | held-out | frames | possessions | visited (pass 1) | action labels | screen labels | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 03.08.2016.GRN.at.HLX | no | 7900 | 24 | 24 | 24 | 10 | ok |
| 12.10.2015.GRN.at.DUL | no | 6466 | 24 | 24 | 24 | 6 | ok |
| 12.16.2015.PHX.at.GSW | yes | 7382 | 24 | 24 | 24 | 9 | ok |
| 02.10.2016.QCY.at.FRS | no | 6624 | 24 | 24 | 24 | 8 | ok |
| 12.01.2015.DUL.at.HLX | no | 7576 | 24 | 24 | 24 | 12 | ok |
| 11.28.2015.HLX.at.CLE | yes | 6868 | 24 | 24 | 24 | 15 | ok |

Label counts by action_type over every game (hand-label schema; only ball_screen, drive and isolation are detected): ball_screen 60, cut 14, drive 17, handoff 10, isolation 16, off_ball_screen 14, post_up 13.
Label counts by label_source: synthetic 144.
Training games: 4 (02.10.2016.QCY.at.FRS, 03.08.2016.GRN.at.HLX, 12.01.2015.DUL.at.HLX, 12.10.2015.GRN.at.DUL). Held-out games set aside before any fit: 2 (11.28.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW). Skipped possessions (null offense or no rows): 0. Games skipped by validation: none.
Label scope: only candidates of possessions the labeller's pass 1 visited (labels/visited, analytics.labels.read_visited) are fitted or scored. Candidates outside the visited set: 0 (in 0 possessions, scored as inference only); games with labels but no visited file (every possession counted as visited): none; games with neither labels nor visited records (inference only, never fitted or scored): none; inference-only candidates in the tables: 0. Action labels with label_source model dropped before matching: 0.
Guard: `analytics.eval.refuse_heldout` checked 4 training game ids against sportvu/splits.json; `analytics.schema.assert_time_forward` and `features.assert_no_leak` passed on the training feature frame (219 rows, 21 features).

## Candidate generation (fixture output, not a result)

Rules (5 Hz): ball_screen = window of at least 0.8 s, teammate within 4 ft of the handler's defender (one of the 2 nearest, within 9 ft of the handler, not the teammate's own man), under 3 ft per s, handler within 12 ft, extent padded 1 s before and 1 s after the set; drive = 0.6 s window, handler starting within 35 ft of the basket closing on it over 8 ft per s; isolation = 2 s window, same handler with the ball under 8 ft per s, starting within 47 ft, no teammate within 10 ft on at least 70 percent of the frames. Target: IoU over frames with a label of at least 0.50.

Candidates: 219 over 96 visited possessions of 96 (2.28 per visited possession); 16 overlap a label of a type this detector does not model (counted as none); 0 are duplicates (they overlap a label at IoU >= 0.50 that a higher-IoU candidate took in the one-to-one matching; counted as none).

Rule class (rows) against label class (columns), training games:

| rule_class | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 36 | 0 | 0 | 40 |
| drive | 0 | 13 | 0 | 86 |
| isolation | 0 | 0 | 12 | 32 |

Candidate recall against the labels (training games), IoU >= 0.50:

| class | labels | hit_any | hit_same_class | recall_any | recall_same_class |
| --- | --- | --- | --- | --- | --- |
| ball_screen | 36 | 36 | 36 | 1.000 | 1.000 |
| drive | 13 | 13 | 13 | 1.000 | 1.000 |
| isolation | 12 | 12 | 12 | 1.000 | 1.000 |

Label length, candidate length (frames at the contract's 25 Hz) and IoU of the matched pairs, training games:

| class | labels | label_len_p10 | label_len_median | label_len_p90 | candidates | cand_len_p10 | cand_len_median | cand_len_p90 | matched | iou_p10 | iou_median | iou_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball_screen | 36 | 100.00 | 100.00 | 100.00 | 76 | 68.50 | 76.00 | 106.00 | 36 | 0.71 | 0.72 | 0.76 |
| drive | 13 | 37.00 | 37.00 | 37.00 | 99 | 16.00 | 21.00 | 51.00 | 13 | 0.67 | 0.77 | 0.81 |
| isolation | 12 | 62.00 | 62.00 | 62.00 | 44 | 51.00 | 66.00 | 83.00 | 12 | 0.73 | 0.94 | 0.95 |

## Protocol

Features (21, positions and velocities inside the window): scr_def_dist_min, scr_def_dist_mean, scr_speed_mean, scr_speed_max, handler_speed_mean, handler_speed_max, handler_dir_change_rad, handler_radial_speed_mean, handler_basket_dist_start, handler_basket_dist_end, def_handler_dist_mean, def_recovery_ft, ball_z_mean, ball_z_max, shot_clock_start, n_within_8ft_mean, teammate_min_dist_mean, window_s, rule_is_screen, rule_is_drive, rule_is_iso. `def_recovery_ft` is the one declared post-window feature (defender to handler distance over the 1 s after the window minus inside it); no feature reads the possession outcome and no label column is a feature.
Models: rule (the generator's own class) and hgb (HistGradientBoostingClassifier, {'max_iter': 100, 'learning_rate': 0.1, 'max_leaf_nodes': 15, 'min_samples_leaf': 5, 'l2_regularization': 0.1}) over ['ball_screen', 'drive', 'isolation', 'none'].
Metrics: out-of-fold under 4 grouped folds by sportvu_game (seed 0); fewer games (4) than the 5 folds asked, so n_folds = number of games (leave-one-game-out); the ball_screen F1 interval is the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0). Selection: out-of-fold ball_screen F1, ties to the rule baseline.

## Metrics (out-of-fold, 219 candidates, 4 training game(s)) (fixture output, not a result)

| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| rule | 0.643 [0.560, 0.718] | 0.232 | 0.429 | 0.000 | 0.326 | 0.279 |
| hgb | 1.000 [1.000, 1.000] | 1.000 | 0.737 | 0.984 | 0.930 | 0.977 |

Selected model: **hgb** (plan target: ball_screen F1 over 0.8 on a held-out game; a fixture cannot meet or miss it).

Confusion of the selected model (rows true, columns predicted, out-of-fold):

| true \ pred | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 36 | 0 | 0 | 0 |
| drive | 0 | 13 | 0 | 0 |
| isolation | 0 | 0 | 7 | 5 |
| none | 0 | 0 | 0 | 158 |

Detected actions (selected class other than none with probability above 0.5): 54 of 219 training candidates; 0 ball_screen detection(s) refused for lack of a screener.

## Held-out games (scored once by the selected model refit on all 219 training candidates; never fitted or tuned on) (fixture output, not a result)

111 candidates from 2 held-out game(s): 11.28.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW.

| model | F1 ball_screen | F1 drive | F1 isolation | F1 none | macro F1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| hgb | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Held-out numbers are point estimates (no bootstrap over 2 game(s)); only the selected model is scored there. Confusion on the held-out candidates:

| true \ pred | ball_screen | drive | isolation | none |
| --- | --- | --- | --- | --- |
| ball_screen | 24 | 0 | 0 | 0 |
| drive | 0 | 4 | 0 | 0 |
| isolation | 0 | 0 | 3 | 0 |
| none | 0 | 0 | 0 | 80 |

## Outputs

No table written (no --out). With --out OUT_ROOT the action_candidates and actions_detected tables of docs/contract/actions.md go under OUT_ROOT/analytics/actions/<season>/.

## Assumptions (recorded because nobody could be asked)

- Hand labels are read only through `analytics.labels.read_labels(DATA_ROOT, game, labeller_pass=1)`: the last line per id wins, tombstoned ids are dropped, pass-2 relabels never reach a fit; `label_source` hand is the accepted value (synthetic on fixtures).
- A labelled ball screen spans approach, set and use, so the candidate extent pads the detected set by 1 s before and 1 s after (`SCREEN_PAD_PRE_S`, `SCREEN_PAD_POST_S`; set both to 0 if lane I4's labels mark the set alone). The plan's 1 s set is read as at least 0.8 s because a 1 s set lands on 4 or 5 intervals of the 5 Hz grid.
- The screened defender is one of the 2 defenders nearest the handler at the window start (a switch, ice or under puts him second), within 9 ft of the handler, and not the screener's own matchup on the frames before the window on which they were apart.
- A drive starts within 35 ft of the attacked basket (itself the basket nearest the offence over the last half of the possession), not in the transition sprint. An isolation is a 2 s hold (the plan said 3 s; 3 s missed the fixture's 2.4 s holds and the model, not the window, decides the class) under 8 ft per s starting in the front court, with no teammate within 10 ft on at least 70 percent of the frames (a clear-out still in progress). These constants were set so that candidate recall on the synthetic fixtures is high; the recall table above is the first thing to re-check on real labels.
- The ball has no handler when it is over 6 ft from every offensive player (in flight); features come from the rule window (the set), the padded extent serves the IoU only; `def_recovery_ft` is the one declared post-window feature.
- A candidate overlapping a label of a type the detector does not model (handoff, off_ball_screen, post_up, cut) is class none and counted above. Matching is one-to-one (each label to its highest-IoU free candidate); leftover overlapping candidates are none and counted as duplicates.
- An unvisited possession is unlabelled, not negative: only possessions in the pass-1 visited record enter a fit or a metric; a game with neither labels nor a visited file is inference only. Action labels with label_source model are dropped before matching.
- actions_detected is an offline table: its confidence is the selected model's probability, which reads def_recovery_ft 1 s after the window ends, so it is not available at start_frame. A ball_screen detection without a screener (candidate or proxy) is refused and counted.
- The hgb model fits on one thread (threadpoolctl): on 4 cores shared by many processes its OpenMP loops stalled for minutes.
- Fewer games than folds gives leave-one-game-out; one game gives in-sample metrics labelled as such; ties in the selection metric go to the rule baseline.

## Status

Fixture output, not a result: this run shows the pipeline works end to end on scripted motion. The real run, on the owner's Mac where the SportVU frames, the possessions tables and the hand labels (DATA_ROOT/labels/actions/<game>.jsonl and labels/screens/<game>.jsonl, lane I4's tool, read with labeller_pass 1) live, is

    python -m analytics.actions.train --data "$DATA_ROOT" --report reports/actions.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

Needs real data before any number means anything: hand-labelled ball screens, drives and isolations on training games (the plan asks for over 0.8 ball_screen F1 on a held-out game, judged once on the held-out block above), and the possessions table of every labelled game.
