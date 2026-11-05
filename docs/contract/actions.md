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
| hand labels | `DATA_ROOT/labels/actions/<game>.jsonl`, `labels/screens/<game>.jsonl`, `labels/visited/<game>.jsonl` | below; read only through `analytics.labels.read_labels(root, game, labeller_pass=1)` and `read_visited` |

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

Label scope. Hand labels are a sample of possessions (about 50 per game). A possession the labeller's pass 1
visited (`labels/visited`, `analytics.labels.read_visited`, filtered to `labeller_pass == 1`) is labelled, even
when it holds no action; any other possession is unlabelled, not negative. For a game with a visited file the
trainer fits and scores only candidates of visited possessions; a game with labels but no visited file (older
roots) counts every possession as visited and is listed; a game with neither labels nor visited records is
inference only: its candidates are scored by the refit selected model and written to the tables with
`labelled` false, never fitted or measured. Action rows whose `label_source` is `model` are dropped before
matching and counted, so a model never trains on its own detections.

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
| drive | 0.6 s | handler starts within 35 ft of the attacked basket (a step beyond the arc, not the transition sprint) and closes on it faster than 8 ft per s |
| isolation | 2 s | handler starts in the front court, keeps the ball (within 4 ft) under 8 ft per s, no teammate within 10 ft on at least 70 percent of the frames |

The attacked basket is the one nearest the offence's mean position over the last half of the possession.

Overlapping windows of one rule and handler merge into one candidate. Candidates are numbered 0..k-1
within the possession in start order. Time inside a possession is frame-based, `(frame - frame0) / hz`, with
`hz` the inverse of the median clock step (25 Hz on the contract's frames; a zero step is logged and 25 Hz
assumed); the clock serves gap detection only.

Label matching is one-to-one and greedy: every (label, candidate) pair at IoU >= 0.5 is taken in descending
IoU order while both sides are free. A label matches at most one candidate; a candidate left over with an
overlap is class `none` and flagged `label_duplicate` (counted in the report).

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
| `screener_proxy_id` | int32, nullable | the teammate nearest the handler's defender at the window start (the screener of a ball_screen candidate itself); the screener of a drive or isolation candidate that the model calls a ball screen |
| `label_class` | string, nullable | ball_screen, drive, isolation, none (TARGET: the one-to-one matched label at IoU >= 0.5); null for an inference-only row |
| `label_iou` | float | best IoU with any label of the possession (TARGET; 0 without labels, NaN for inference-only rows) |
| `label_action_type` | string, nullable | type of the matched label, whatever it is (TARGET) |
| `label_duplicate` | bool | the candidate overlaps a label at IoU >= 0.5 that another candidate matched (TARGET) |
| `pred_class` | string | the selected model's argmax class |
| `p_ball_screen` | float | the selected model's probability of ball_screen |
| `p_selected_class` | float | the selected model's probability of `pred_class` |
| `fold` | int32 | out-of-fold index for training games; -1 for held-out and inference-only rows (scored once by the refit) |
| `heldout` | bool | the game is held out (sportvu/splits.json, the heldout/ layout, or the CLE arena) |
| `labelled` | bool | the possession was visited by pass 1 (or the game has labels and no visited file): the row entered a fit or a metric; false for inference-only rows |

### `actions_detected/<game>.parquet` and `<game>.jsonl` (the canonical detected-actions file)

`OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` (held-out games under
`OUT_ROOT/analytics/actions/heldout/<season>/actions_detected/`) is what the playcalling and coverage
lanes read. One row per candidate whose `pred_class` is not none with `p_selected_class` above 0.5, cast
with `analytics.labels.cast(df, "actions")` (so it is a valid actions frame for `analytics.labels`), plus
two columns of this lane. It is an offline table: `confidence` is the selected model's probability, which
reads `def_recovery_ft` 1 s after the window ends, so neither the row nor its confidence exists at
`start_frame`. A ball_screen prediction without a screener (candidate or proxy) is refused and counted.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `action_id` | int32 | row position within the game (0..n-1 in possession, start_frame order); the detector has no hand id |
| `action_type` | string | = `pred_class`: ball_screen, drive or isolation |
| `start_frame`, `end_frame` | int32 | the candidate window |
| `ball_handler_id` | int32 | handler at the window start |
| `screener_id` | int32, nullable | the screener of a ball screen (the candidate's, else its screener proxy); null for a drive or isolation |
| `label_source` | string | `model` for every detected row (`hand` marks the hand labels of `DATA_ROOT/labels/`, same schema, read through `analytics.labels.read_labels`) |
| `labeller_pass` | int16 | 1 (the schema's training pass) |
| `labelled_at` | string | UTC time of the run |
| `model` | string | provenance: trainer, selected model, seed and input kind, e.g. `actions.train hgb seed0 data` |
| `confidence` | float | = `p_selected_class` (offline, see above) |

The JSONL beside it has the same rows with `table: actions`, so `validate.read_labels_jsonl` reads it
back; it is not a hand-label file and never goes under `DATA_ROOT/labels/`.

### `features/<game>.parquet`

`python -m analytics.actions.features --out OUT_ROOT` writes the feature frame per game under
`OUT_ROOT/analytics/actions/<season>/features/<game>.parquet` (held-out under `heldout/<season>/features/`):
`sportvu_game`, `possession`, `candidate` and the FEATURES columns, nothing else (the frame
`assert_time_forward` and `assert_no_leak` passed on).

## Evaluation (`analytics/actions/train.py`)

Models: `rule` (the generator's class, the baseline) and `hgb` (HistGradientBoostingClassifier,
multi-class, one thread). Grouped folds by `sportvu_game` (n_folds = number of games when fewer; in-sample
and labelled so with one game). Selection metric: out-of-fold F1 for ball_screen (plan target over 0.8 on a
held-out game), ties to the rule baseline. Also per-class F1, macro F1, accuracy, the confusion table, the
ball_screen F1 interval from bootstrap resamples of whole games, candidate recall against the labels (any
candidate, same-class candidate), the distribution of label length, candidate length and matched IoU per
class, duplicate counts. Held-out games are scored once by the selected model refit on all training rows
and reported in their own block. A fixture run says "fixture output, not a result" on every table.
