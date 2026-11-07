# Hand labels: actions and ball-screen coverages

Contract for the two label tables that `analytics/labels.py` reads and writes and `analytics/label_tool.py`
collects. Companion to `docs/labeling_guide.md` (definitions, keys, sampling, consistency procedure).

Code: `analytics.labels.ACTIONS_SCHEMA`, `analytics.labels.SCREENS_SCHEMA` (pyarrow schemas),
`read_labels`, `write_labels`, `validate_labels`, `sample_possessions`, `kappa`.

## 1. Actions table

One row per labelled action inside a possession. `action_id` is unique within a game (the tool assigns
max + 1); it is the key the screens table references.

| column | type | nullable | meaning |
| --- | --- | --- | --- |
| `sportvu_game` | string | no | Step 1 game id (`MM.DD.YYYY.AWAY.at.HOME`) |
| `possession` | int32 | no | possession number of the section 19 table |
| `action_id` | int32 | no | id of the action within the game |
| `action_type` | string | no | `ball_screen`, `handoff`, `off_ball_screen`, `isolation`, `post_up`, `drive`, `cut` |
| `start_frame` | int32 | no | first frame of the action (inside the possession span) |
| `end_frame` | int32 | no | last frame of the action (inside the possession span, at or after `start_frame`) |
| `ball_handler_id` | int32 | no | NBA player id of the ball handler (the cutter for `cut`, the receiver for `handoff`, the player being screened for `off_ball_screen`) |
| `screener_id` | int32 | yes | the screener (`ball_screen`, `off_ball_screen`) or the giver (`handoff`); required for those three types, null otherwise |
| `label_source` | string | no | `hand`, `model`, `synthetic` |
| `labeller_pass` | int16 | no | 1 for the first pass, 2 for the relabel a week later |
| `labelled_at` | string | no | ISO 8601 UTC time the line was written, e.g. `2026-11-05T14:03:00Z` |

## 2. Screens table

One row per coverage label. A screen label attaches to a `ball_screen` action of the same game and
possession; the tool refuses any other `action_id`.

| column | type | nullable | meaning |
| --- | --- | --- | --- |
| `sportvu_game` | string | no | as above |
| `possession` | int32 | no | as above |
| `screen_id` | int32 | no | id of the label within the game |
| `action_id` | int32 | no | the `ball_screen` action this coverage describes |
| `coverage` | string | no | `switch`, `drop`, `hedge_show`, `blitz_trap`, `ice`, `under`, `over` |
| `on_ball_defender_id` | int32 | no | NBA player id of the ball handler's defender at the screen |
| `screener_defender_id` | int32 | no | NBA player id of the screener's defender (differs from the on-ball defender) |
| `label_source` | string | no | `hand`, `model`, `synthetic` |
| `confidence` | string | no | `sure`, `unsure` |
| `labeller_pass` | int16 | no | as above |
| `labelled_at` | string | no | as above |

## 3. JSONL layout (append-only)

```
DATA_ROOT/labels/actions/<sportvu_game>.jsonl
DATA_ROOT/labels/screens/<sportvu_game>.jsonl
```

One JSON object per line, written with `json.dumps` (no spaces, UTF-8). A label line holds the columns
of its table in table order; a tombstone line holds only the id, `"deleted": true` and `labelled_at`.
Files are only ever appended to:

- a new label is one new line, stamped with `labelled_at` and `labeller_pass`;
- a deletion is a tombstone line `{"action_id": 7, "deleted": true, "labelled_at": "..."}`
  (`screen_id` for screens); deleting an action also tombstones its screens;
- `read_labels` resolves a file in order (the last line about an id wins, deleted ids are dropped) and
  returns pandas frames with the contract dtypes (`screener_id` as `Int32`); a label line without an id
  (a root written before ids existed) takes the running 0-based position of the non-tombstone lines of
  its file;
- `write_labels(data_root, game, actions_df, screens_df)` casts the rows to the contract before writing
  anything, fills `labelled_at` with the current UTC time where the frame lacks it and defaults
  `labeller_pass` to 1 and `label_source` to `hand`.

**Sanctioned readers.** The only readers of these files are `analytics.labels.read_labels` and the
parquet that `python -m analytics.label_tool --export` writes from it. A consumer that reads the raw
lines anyway must drop every line carrying `"deleted"` and keep the last line per id, or it will count
deleted labels as live and ingest tombstones as rows.

**Which pass training reads.** Both passes of the consistency check live in the same file, told apart by
`labeller_pass`. Training consumers use `labeller_pass == 1` only (`read_labels(root, game,
labeller_pass=1)`); pass 2 is the relabel for the kappa check and would otherwise land as duplicate
labels. `python -m analytics.label_tool --data DATA_ROOT --game G --export OUT.parquet` writes the pass 1
actions to `OUT.parquet` and the pass 1 screens to `OUT_screens.parquet` with the pyarrow schemas
(`--export-pass 2` or `all` for the others). The first training set also leaves out screens with
`confidence == "unsure"`.

**label_source.** A person at the tool writes `hand`; `model` marks a model's proposal and `synthetic`
a fixture. Consumers must accept `hand` (not `human`) as the hand-labelled value.

**Visited record.** `labels/visited/<sportvu_game>.jsonl` holds one line
`{"sportvu_game", "possession", "labeller_pass", "visited_at"}` per possession a pass finished (the page
appends it on next/previous, `mark_visited` from code). It makes "this pass looked and found nothing"
explicit: `--kappa` compares only possessions both passes visited, so an action one pass labelled and
the other did not counts as `none` instead of vanishing.

## 4. Validation (`validate_labels(actions_df, screens_df, possessions_df)`)

Returns a list of problem strings; empty means the frames conform. Checked:

1. contract columns present and castable; nulls only in `screener_id`; no duplicate ids;
2. enum columns hold only the listed values;
3. `start_frame <= end_frame`, both inside `[start_frame, end_frame]` of the possession (section 19);
   `labeller_pass` is 1 or 2;
4. `screener_id` present for `ball_screen`, `handoff`, `off_ball_screen` and different from the handler;
5. two actions of the same type in one possession never overlap in frames (different types may);
6. every screen's `action_id` is a `ball_screen` of the same game and possession, and its two
   defenders differ;
7. with the frames at hand (the tool always has them), the handler and screener belong to the
   possession's offence and the two defenders to its defence.

The tool runs this on every POST and answers 400 with the problems instead of writing.

## 5. Sampling and consistency

`sample_possessions(possessions_df, frames_df, n, seed, heldout_tokens)` returns `n` possessions with
`half_court`, `screen_candidate` and `sample_rank` columns: training games only (any id carrying a
held-out token is refused and counted in `attrs["n_heldout_refused"]`), half-court possessions only
(the ball crosses x = 47 ft and then spends at least 2 s in the new half), the quota split evenly across
offensive teams (a team short of eligible possessions hands the shortfall to the others; what cannot be
placed is `attrs["n_short"]`), at least half of each team's quota from screen-candidate possessions (a non-handler
offensive player within 4 ft of the handler's nearest defender for 1 s of consecutive frames; the
handler is the offensive player nearest the ball).

`kappa(pass1, pass2)` pairs the two passes' actions inside each possession by temporal overlap (an
action only one pass has counts as `none`) and returns Cohen's kappa for `action_type` and, over paired
ball screens, for `coverage` (`sure` labels only). `label_tool --kappa` restricts both passes to the
possessions both visited and skips held-out games. The acceptance bar is 0.7 on both
(`docs/labeling_guide.md`).
