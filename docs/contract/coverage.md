# Coverage recognition: tables and guarantees (analytics/coverage)

Unit: one ball-screen event. Target: the defensive coverage class, one of seven: `switch`, `drop`,
`hedge_show`, `blitz_trap`, `ice`, `under`, `over`. Status: no hand labels exist yet (STATUS.md), so every
number the package has produced is fixture output, not a result.

## Inputs consumed (interface of lane ACTIONS, read as given)

| table | where | columns |
| --- | --- | --- |
| actions | `DATA_ROOT/labels/actions/<game>.jsonl` | `sportvu_game`, `possession`, `action_type`, `start_frame`, `end_frame`, `ball_handler_id`, `screener_id`, `label_source`, `labeller_pass`, `labelled_at` (`action_id` when present, else the row's position within the game's file) |
| screens | `DATA_ROOT/labels/screens/<game>.jsonl` | `sportvu_game`, `possession`, `screen_id`, `action_id`, `coverage` (7 classes), `on_ball_defender_id`, `screener_defender_id`, `label_source`, `confidence` in {`sure`, `unsure`}, `labeller_pass`, `labelled_at` |
| actions_detected | `OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` (held-out under `heldout/<season>/actions_detected/`; the older flat `<season>/<game>.parquet` layout is a fallback) | the actions columns with `label_source` `model`; no defender ids, no coverage |
| frames | `DATA_ROOT/sportvu_contract/<game>.parquet` (held-out under `heldout/`) | docs/DATA_CONTRACT.md section 4 (`frame`, `period`, `game_clock_s`, `team_id`, `player_id`, `x`, `y` are read) |
| rosters, possessions | `DATA_ROOT/analytics/<table>/<season>/<game>.parquet` (held-out under `heldout/<season>/`) | sections 16 and 19; rosters give the defending team, possessions are used only to log screens outside their span |

Validation (`analytics.coverage.validate`): required columns; `coverage` and `confidence` enums; a screen must
reference a `ball_screen` action of the same game (else dropped); the action's possession wins over the
screen's; frames non-negative with `start_frame <= end_frame`; integer ids with handler != screener and the two
defenders distinct; duplicate `(sportvu_game, action_id)` keeps the first row; rows with a missing required
value are dropped. Every count is logged and lands in the report. `label_source` outside {`hand`, `model`,
`synthetic`} (the values analytics/labels.py writes) is logged, not dropped.

Label files are append-only (analytics/labels.py): the last line about an id wins and a tombstone line
(`deleted: true`) removes the id. They are read through `analytics.labels.read_labels` when the files carry its
id fields, else resolved here by the same rule. `read_labels` is called with `labeller_pass=1`, so only
first-pass labels arrive; the lane keeps its own `labeller_pass == 1` filter as a check (it must count 0 and
warns otherwise) and as the filter for the fallback reader, so a pass-2 relabel never duplicates a training label.

## Feature window (declared)

The object classified is the screen and the play right after it: frames `start_frame` to
`end_frame + 38` (1.5 s at 25 Hz). No later frame enters; the label, `confidence`, `label_source`,
`labeller_pass` and every column of `analytics.schema.ALL_EXCLUDED` are refused on a design matrix
(`features.assert_no_leak`), and `analytics.schema.assert_time_forward` runs on the feature frame.
Coordinates: origin at the screener's position at `start_frame`, depth axis toward the basket nearer the
handler, lateral axis toward the nearer sideline. The 37 features are listed by `features.feature_columns()`;
the families are positions of the four roles at the screen frame, which defender ends nearest the handler,
the screener's defender's depth and distance to the rim, his distance to the handler early, at its minimum,
after the minimum and late, the on-ball defender's path side relative to the screener, the handler's lateral
displacement, the fraction of frames with both defenders within 6 ft of the handler, the big's distance to the
screen point, and speeds. Detected screens get their defenders by nearest opponent at `start_frame`.

## Tables written

Under `OUT_ROOT/analytics/coverage/<season>/` (held-out games under `OUT_ROOT/analytics/coverage/heldout/<season>/`);
`<season>` is derived from the game id date (`MM.DD.YYYY`, September or later starts the season).

### coverage_predictions (`coverage_predictions.parquet`, one row per screen)

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession (the action's) |
| `action_id` | int32 | the action the screen belongs to |
| `label_source` | string | `hand`, `synthetic` or `model`; with `sportvu_game` and `action_id` it keys the row (detected ids are positional and could otherwise collide with hand ids; detected screens of a game with any hand label are dropped) |
| `coverage_label` | string, nullable | the hand label; null for an unlabelled or detected screen |
| `p_switch` .. `p_over` | float32 | the selected model's class probabilities (7 columns, sum to 1) |
| `predicted` | string | argmax class |
| `fold` | int8 | >= 0: the grouped fold whose model scored the row (out-of-fold); -1: scored by the model refit on every training row (held-out, unlabelled, unsure and detected screens); -2: in-sample (a run with a single training game, where no grouped fold exists) |
| `heldout` | bool | the game is held out (sportvu/splits.json token, `.at.CLE` arena, or frames under `heldout/`) |

### coverage_team_rates (`coverage_team_rates.parquet`, one row per defending team, training games only)

| column | type | meaning |
| --- | --- | --- |
| `team_id` | int | the on-ball defender's team (rosters, else frames) |
| `n_screens` | int | screens of that team in training games |
| `share_switch` .. `share_over` | float | share of the predicted class (7 columns) |
| `insufficient` | bool | fewer than 300 screens (`analytics.eval.gated`); the shares are not to be read |

Shares use the predicted class of every training-game screen (out-of-fold for labelled rows) so labelled,
unlabelled and detected screens are counted the same way. One file per season, computed from that season's rows.

### coverage_features (`coverage_features.parquet`, one row per screen; written by `analytics.coverage.features`)

The feature frame: `sportvu_game`, `possession`, `action_id`, `screen_frame`, `screen_end_frame`, `heldout`,
`label_source`, `coverage_label` (nullable), `label_confidence` (nullable), `on_ball_defender_id`,
`screener_defender_id` (inferred for detected screens), then the 37 features of `features.feature_columns()`
as float64. Held-out games go under `heldout/<season>/`. The label, confidence and source columns are
bookkeeping for the trainer and are refused on any design matrix.

## Protocol

Training rows are the sure-labelled screens of training games; screens labelled `unsure` are scored, never
fitted (docs/contract/labels.md leaves them out of the first training set). Held-out rows are scored by the
selected model only, once.

Candidates: majority (training-fold class frequencies), multinomial logistic regression (median imputation,
standardisation, C = 1), HistGradientBoostingClassifier (100 trees, depth 3, learning rate 0.1, min 5 rows per
leaf). Grouped folds by `sportvu_game` (`analytics.eval.grouped_folds`); selection by out-of-fold macro F1
over the classes present in the truth, ties to the simpler model; accuracy, per-class F1 and a confusion table
for the selected model; intervals by bootstrap of whole games (`analytics.eval.bootstrap_by_game`). Held-out
games are set aside before any fit, `analytics.eval.refuse_heldout` checks the training ids, and the held-out
rows are scored once by the selected model refit on every training row. No deep model.

## Commands

    python -m analytics.coverage.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.coverage.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.coverage.train --fixtures | --data DATA_ROOT [--out OUT_ROOT] --report reports/coverage.md [--folds 5] [--seed 0] [--bootstrap 200]

`--fixtures` builds the season in memory from `analytics.fixtures` (lane I1) when it imports, else from the
local scripted generator of `tests/_fx_coverage.py`; `--fixture-games`, `--fixture-heldout-every` and
`--fixture-possessions` size it. Without `--out` the tables go to a temporary directory that the CLI prints.
