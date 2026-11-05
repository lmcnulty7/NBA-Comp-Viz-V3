# Contract: action recognition tables (`analytics/actions/`)

Phase 2 lane ACTIONS. Unit: a candidate window inside a possession. Target: the candidate overlaps a
labelled action at IoU over frames of at least 0.5 (`analytics.actions.IOU_MIN`). The detector's classes
are `ball_screen`, `drive`, `isolation` and `none` (`analytics.actions.CLASSES`); the hand-label schema
knows more action types (handoff, off_ball_screen, post_up, cut), which the detector counts as `none`
and reports. Frame ids are the section 4 ids (25 Hz) even though the rules run at 5 Hz.

## Inputs

| input | where | contract |
| --- | --- | --- |
| frames | `DATA_ROOT/sportvu_contract/<game>.parquet` (held-out under `heldout/`) | DATA_CONTRACT section 4 |
| possessions | `DATA_ROOT/analytics/possessions/<season>/<game>.parquet` (held-out under `heldout/<season>/`) | section 19 |
| rosters (optional) | `DATA_ROOT/analytics/rosters/<season>/<game>.parquet` | section 16 |
| hand labels | `DATA_ROOT/labels/actions/<game>.jsonl` | below |

### Hand labels (lane I4 writes the tool; this lane codes against the schema)

One JSON object per line. A line is an `actions` row when it carries `action_type` and a `screens` row
when it carries `coverage`; a `table` key (`actions` or `screens`), when present, decides.

actions: `sportvu_game` string, `possession` int32, `action_type` in {ball_screen, handoff,
off_ball_screen, isolation, post_up, drive, cut}, `start_frame` int32, `end_frame` int32,
`ball_handler_id` int32, `screener_id` int32 nullable, `label_source` in {hand, model, synthetic},
`labeller_pass` (string or int), `action_id` int32 (optional; defaults to the row's position among the
file's actions rows).

screens: `sportvu_game`, `possession`, `action_id` (an actions row that is a ball_screen), `coverage`
string (vocabulary not fixed; values outside `analytics.actions.COVERAGES` are counted as warnings),
`on_ball_defender_id` int32, `screener_defender_id` int32, `label_source`, `confidence` (a number in
[0, 1] or a text grade; never null).

Validation (`analytics.actions.validate`): enum checks, `start_frame <= end_frame`, labels outside their
possession span (warning), duplicates (warning, kept), overlapping labels of one type in one possession
(warning, kept), screens whose `action_id` is unknown or not a ball_screen (problem), handler or screener
ids missing from the roster (problem). A game with a problem is skipped by the trainer and listed.

## Candidate rules (5 Hz, `analytics/actions/features.py`)

The handler at a frame is the offensive player nearest the ball, none when the ball is over 6 ft from
every offensive player. Constants are module-level in `features.py`.

| rule_class | window | condition on every frame of the window |
| --- | --- | --- |
| ball_screen | at least 0.8 s | a teammate within 4 ft of the handler's defender (either of the two defenders nearest the handler at the window start, within 9 ft of him, and not the teammate's own matchup before the window), moving under 3 ft per s, handler within 12 ft of him; merged windows of one (handler, screener) are the set, padded by 1 s before and 1 s after (approach and use) |
| drive | 0.6 s | handler starts within 30 ft of the basket and closes on it faster than 8 ft per s |
| isolation | 2 s | handler starts within 30 ft, keeps the ball (within 4 ft) under 8 ft per s, no teammate within 10 ft |

Overlapping windows of one rule and handler merge into one candidate. Candidates are numbered 0..k-1
within the possession in start order.

## Features (`features.FEATURES`, all inside the rule window unless stated; for a ball screen the set, not the padded extent)

`scr_def_dist_min`, `scr_def_dist_mean` (screener to the screened defender; for a drive or isolation the
screener is the teammate nearest the handler's defender at the window start), `scr_speed_mean`,
`scr_speed_max`, `handler_speed_mean`, `handler_speed_max`, `handler_dir_change_rad`,
`handler_radial_speed_mean`, `handler_basket_dist_start`, `handler_basket_dist_end`,
`def_handler_dist_mean`, `def_recovery_ft` (POST-WINDOW, declared: defender to handler distance over the
1 s after the window minus inside it; allowed for a detector, never anything later), `ball_z_mean`,
`ball_z_max`, `shot_clock_start`, `n_within_8ft_mean`, `teammate_min_dist_mean`, `window_s`,
`rule_is_screen`, `rule_is_drive`, `rule_is_iso`.

Leakage: the feature frame holds only `sportvu_game`, `possession`, `candidate` and the features above;
`analytics.schema.assert_time_forward(frame, "possessions")` and `features.assert_no_leak` run on it
(the latter also refuses `label_class`, `label_iou`, `label_action_type`, `rule_class`, `start_frame`,
`end_frame`, `pred_class`, `p_*` and every label-schema column). `analytics.eval.refuse_heldout` runs on
the training game ids before any fit.

## Output tables

Written under `OUT_ROOT/analytics/actions/<season>/` (held-out games under
`OUT_ROOT/analytics/actions/heldout/<season>/`), one file per game.

### `action_candidates/<game>.parquet`

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `candidate` | int32 | 0..k-1 within the possession |
| `start_frame`, `end_frame` | int32 | the candidate window in section 4 frame ids |
| `rule_class` | string | ball_screen, drive, isolation (the generator's class) |
| `ball_handler_id` | int32 | handler at the window start |
| `screener_id` | int32, nullable | the screener of a ball_screen candidate; null otherwise |
| `label_class` | string | ball_screen, drive, isolation, none (TARGET: best label at IoU >= 0.5) |
| `label_iou` | float | best IoU with any label of the possession (TARGET; 0 without labels) |
| `label_action_type` | string, nullable | type of the best-matched label at IoU >= 0.5, whatever it is (TARGET) |
| `pred_class` | string | the selected model's argmax class |
| `p_ball_screen` | float | the selected model's probability of ball_screen |
| `p_selected_class` | float | the selected model's probability of `pred_class` |
| `fold` | int32 | out-of-fold index for training games; -1 for held-out games (scored once by the refit) |
| `heldout` | bool | the game is held out (sportvu/splits.json, the heldout/ layout, or the CLE arena) |

### `actions_detected/<game>.parquet` and `<game>.jsonl` (the canonical detected-actions file)

`OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` (held-out games under
`OUT_ROOT/analytics/actions/heldout/<season>/actions_detected/`) is what the playcalling and coverage
lanes read. One row per candidate whose `pred_class` is not none with `p_selected_class` above 0.5, in
the hand-label actions schema:

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `action_id` | int32 | row position within the game (0..n-1 in possession, start_frame order); the detector has no hand id |
| `action_type` | string | = `pred_class`: ball_screen, drive or isolation |
| `start_frame`, `end_frame` | int32 | the candidate window |
| `ball_handler_id` | int32 | handler at the window start |
| `screener_id` | int32, nullable | the screener of a ball screen; null otherwise |
| `label_source` | string | `model` for every detected row (`hand` marks the hand labels of `DATA_ROOT/labels/`, same schema, read through `analytics.labels.read_labels`) |
| `labeller_pass` | string | trainer, selected model, seed and input kind, e.g. `actions.train hgb seed0 data` |
| `confidence` | float | = `p_selected_class` |

The JSONL beside it has the same rows with `table: actions`, so `validate.read_labels_jsonl` reads it
back; it is not a hand-label file and never goes under `DATA_ROOT/labels/`.

## Evaluation (`analytics/actions/train.py`)

Models: `rule` (the generator's class, the baseline) and `hgb` (HistGradientBoostingClassifier,
multi-class). Grouped folds by `sportvu_game` (n_folds = number of games when fewer; in-sample and
labelled so with one game). Selection metric: out-of-fold F1 for ball_screen (plan target over 0.8 on a
held-out game), ties to the rule baseline. Also per-class F1, macro F1, accuracy, the confusion table, the
ball_screen F1 interval from bootstrap resamples of whole games, candidate recall against the labels
(any candidate, same-class candidate). Held-out games are scored once by the models refit on all training
rows and reported in their own block. A fixture run says "fixture output, not a result" on every table.
