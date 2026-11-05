# Play calling tables (analytics/playcalling)

Every table here is an observational association with a bootstrap interval by game and the 300-row gate of
`analytics.eval.gated`; none of them is a causal estimate of what calling a play does. Files are parquet under
`OUT_ROOT/analytics/playcalling/<season>/`; rows of held-out games (sportvu/splits.json tokens, the CLE arena, or
a `heldout/` folder in the inputs) are written to `OUT_ROOT/analytics/playcalling/heldout/<season>/action_outcomes.parquet`
only and never reach `ppp_by_action` or `playcalling_adjusted`. Written by `python -m analytics.playcalling.train`
(`action_outcomes` also by `python -m analytics.playcalling.features`).

## action_outcomes (one row per action event)

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id (docs/DATA_CONTRACT.md section 16) |
| `possession` | int32 | section 19 possession |
| `action_id` | int32 | the action's id from the actions input; when absent, the row's 0-based position within its game (jsonl line order, parquet row order: the convention of analytics/coverage and analytics/fixtures) |
| `action_type` | string | ACTION_TYPES of `analytics/playcalling/__init__.py`, the ACTIONS lane vocabulary (`ball_screen`, `handoff`, `off_ball_screen`, `isolation`, `post_up`, `drive`, `cut`); an unknown upstream value is kept as its own category and counted |
| `coverage` | string | the hand screen label (`labels/screens`), else the coverage lane's `predicted`, else `none`; COVERAGES are the COVERAGE lane classes `switch`, `drop`, `hedge_show`, `blitz_trap`, `ice`, `under`, `over` plus `none` |
| `off_lineup` | string, nullable | the five offensive player ids of the possession's first frame, sorted ascending, joined with `-`; null when the first frame is missing or does not split five against five |
| `def_lineup` | string, nullable | the same for the defence |
| `ppp` | float32 | OUTCOME: the possession's `points` (section 19) |
| `epv_gain` | float32, nullable | OUTCOME: `epv_points` at the action's `end_frame` minus at its `start_frame` (nearest epv_states frame of the same possession at or before each); null without epv_states |
| `heldout` | bool | the game is held out (never fitted or tuned on) |
| `score_margin` | float32, nullable | offence minus defence points scored on the earlier possessions of the game (known at the action's start); null when `offense_team_id` is null |
| `shot_clock_s` | float32, nullable | section 4 `shot_clock_s` at `start_frame` |
| `period` | int8 | period of the possession |
| `label_source` | string | as the actions input gives it (`hand`, `human`, `model`, `detected`, `synthetic`; unknown values counted and kept) |
| `coverage_source` | string | `hand`, `predicted` or `none` (where `coverage` came from) |
| `start_frame`, `end_frame` | int32 | the action's span (id-like; not features) |

Feature columns (`analytics.playcalling.features.FEATURES`): `action_type`, `coverage`, `off_lineup`, `def_lineup`,
`score_margin`, `shot_clock_s`, `period`. The outcomes `ppp` and `epv_gain` are never features;
`features.features` calls `analytics.schema.assert_time_forward` on the feature frame and refuses them.

## ppp_by_action (tables 1 and 2 of the report, one file)

| column | type | meaning |
| --- | --- | --- |
| `action_type` | string | as above |
| `coverage` | string, nullable | null for the marginal rows (table 1: all coverages); the coverage for table 2 rows |
| `n` | int32 | action rows in the group; each row is one possession bearing that action (a possession with two actions counts once in each of their groups) |
| `ppp` | float32 | mean `ppp` of the group's training rows |
| `lo`, `hi` | float32, nullable | the 2.5 and 97.5 percentiles of the mean over bootstrap resamples of whole games; null when the group sits in one game |
| `insufficient` | bool | `n` under the gate (300 by default, `--n-min`); the row's number is not to be quoted |

The same two tables for `epv_gain` appear in the report only (section 2b) when epv_states were given.

## playcalling_adjusted (table 4 of the report)

| column | type | meaning |
| --- | --- | --- |
| `coverage` | string | as above |
| `action_type` | string | as above |
| `adjusted_ppp` | float32 | the ridge prediction for the cell with the lineup effects averaged over the training rows: intercept + mean lineup contribution + action + coverage + action x coverage coefficients |
| `lo`, `hi` | float32, nullable | percentiles of `adjusted_ppp` over ridge refits on bootstrap resamples of whole games; null with one game or `--bootstrap 0` |
| `n` | int32 | training rows in the (coverage, action_type) cell; cells never observed have no row |
| `insufficient` | bool | `n` under the gate |
| `recommended` | bool | the highest `adjusted_ppp` among the sufficient cells of that coverage (at most one per coverage; none when every cell is insufficient); an association label, not advice |

Ridge: `sklearn.linear_model.Ridge` on indicator columns of action type, coverage, action type x coverage,
offensive lineup and defensive lineup (unknown lineup levels at prediction time get no column). Alpha from
`(0.3, 1, 3, 10, 30, 100)` by grouped cross-validation by game; the reported out-of-fold MSE is nested (alpha
chosen inside each outer fold). Baselines: the training-fold mean and the training-fold PPP by action type.

## Inputs this lane consumes (assumed interfaces; counted and kept when they differ)

- actions (hand labels `DATA_ROOT/labels/actions/<game>.jsonl`, or the ACTIONS lane's
  `OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` for games without hand labels; files under
  `action_candidates/` are not read): `sportvu_game`, `possession`, `action_type`, `start_frame`, `end_frame`,
  `ball_handler_id`, `screener_id`, `label_source`, optional `action_id`.
- screens (`DATA_ROOT/labels/screens/<game>.jsonl`): `sportvu_game`, `possession`, `action_id`, `coverage`.
- coverage_predictions (`OUT_ROOT/analytics/coverage/<season>/coverage_predictions.parquet`, COVERAGE lane; the
  team-rates file beside it is not read): the same keys, `predicted` (or `coverage`) and `p_<class>` columns in
  [0, 1] summing to about 1.
- epv_states (every parquet under `OUT_ROOT/analytics/epv/` that holds the columns, EPV lane; this table was not yet
  produced by that lane when this was written, so the name and columns are this lane's assumption):
  `sportvu_game`, `possession`, `frame`, `epv_points`.
- A file lacking a table's required columns is skipped with a count; a path holding `heldout` marks its games held out.
- possessions (section 19) and frames (section 4), the latter for the first-frame lineups and the shot clock.

Validation (`python -m analytics.playcalling.validate`): required columns (hard failure), duplicates on the key of
each table dropped with a count, enum values outside the vocabularies counted and kept, ranges (points 0..8,
shot clock 0..24, epv_points in [-1, 4], p columns in [0, 1]), actions whose possession is missing or whose span
leaves the possession span counted.
