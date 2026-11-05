# Decision quality: contract (analytics/decisions)

Unit: a decision event of the ball handler (a shot, a pass or a drive start). Target: the value of the chosen
option minus the best available option, in expected points, at the decision frame. Code:
`analytics/decisions/{validate,features,train,eda}.py`; tests `tests/test_decisions.py`; report
`reports/decisions.md`; notebook `notebooks/eda_decisions.ipynb`.

## Inputs

| input | where | used for |
| --- | --- | --- |
| frames (section 4) | `DATA_ROOT/sportvu_contract/<game>.parquet` (held-out under `heldout/`) | the ball path, positions, on-court players |
| play-by-play (section 5) | `DATA_ROOT/sportvu_contract/<game>_pbp.parquet` | counted only |
| possessions (section 19) | `DATA_ROOT/analytics/possessions/<season>/<game>.parquet` | spans, offence, points (validation only) |
| shot_frames (section 17) | `DATA_ROOT/analytics/shot_frames/<season>/<game>.parquet` | release frames (shot decisions); `made` for the value-model fit only |
| states_1hz, rosters, rebound_frames | same layout | validated, not consumed |
| epv_states (lane EPV) | `OUT_ROOT/analytics/epv/<season>/<game>.parquet` | `epv_points` as the hold option; optional |
| xfg_predictions (lane XFG) | `OUT_ROOT/analytics/xfg/<season>/*.parquet` | `p_selected` on shot events; optional |

A game listed under `heldout/` or whose id carries a held-out token of `sportvu/splits.json` is held out: it is
valued once by the model refit on all training games and never fits or tunes anything
(`analytics.eval.refuse_heldout` on the training ids).

## Event detection (features.py)

| kind | rule | decision frame | player |
| --- | --- | --- | --- |
| holder | offensive player within 2.5 ft (horizontal) of the ball while the ball is under 8 ft; nearest when several; a run shorter than 3 frames is ignored | | |
| pass | holder A's run ends at f, teammate B's run starts at g, (g - f) / fps <= 1.5 s, no release frame in [f - 2, g] | f | A, receiver B |
| shot | shot_frames row with `release_frame` inside the possession span | release_frame | `shooter_at_release_id`, else `shooter_id` |
| drive | the holder's speed toward the basket (frame to frame) above 8 ft/s for at least 0.6 s inside one holder run; one per run | first frame of the run | the holder |

Possessions without an offence, with `gap` true, or without both sides on the floor are skipped and counted.
The attacked basket is the end of the court where the offence's mean x over the span lies (assumption: the
frames do not state the attacked basket).

## Option values at the decision frame (features.option_values)

| option | value |
| --- | --- |
| shot | points(dist) * max(p - 0.03 * max(0, 6 - d_def), 0.01), points = 3 at 22 ft or more else 2, p = the value model's make probability at the position (location-only xFG), d_def = nearest-defender distance; with the xFG lane's `p_selected` joined on the shot event, p is replaced and the contest term is not applied |
| pass to teammate t | t's shot value * openness(d_def(t)) * risk(length), openness = 0.5 + 0.5 * min(d, 6) / 6, risk = 0.95 - 0.01 * max(0, length - 15 ft) |
| hold | `epv_points` of the last EPV state at or before the frame in the same possession; absent without the EPV table |
| drive (chosen only) | shot value at the position 1 s ahead along the decision-frame velocity, never nearer than 3 ft to the basket, with the decision-frame d_def, times 0.9 |

`chosen_value` is the value of the taken option; `best_value` the max over shot, the pass options and hold (a
drive is not in the option set, so a drive's quality may exceed 0); `decision_quality = chosen_value -
best_value`. The contest term, openness, pass risk and drive value are assumptions, not fitted; the prior of
the option values (location-only xFG, or the xFG lane's `p_selected`) is selected by a rule fixed in advance
(coverage of shot decisions at least 0.5) and reported, not fitted.

## The value model (train.py)

The location-only xFG of `analytics/features_xfg.py` (distance_ft, angle_rad, is_three, is_corner): a logistic
regression with standardisation (C=1) fit on the shot_frames rows with a release frame, at the shooter's
position at the release relative to the basket of the possession, target `made`. Fit under grouped folds by
game (`analytics.eval.grouped_folds`): the decisions of a game are valued by the model fit on the other folds,
so a shot's own outcome never informs its value. The design matrix passes `schema.assert_time_forward` and
`features_xfg.assert_no_leak`. One training game: in-sample, labelled.

## Outputs, under `OUT_ROOT/analytics/decisions/<season>/`

### decision_events: `decision_events/<game>.parquet` (held-out games under `heldout/<season>/decision_events/`)

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the decision frame |
| `player_id` | int32 | the handler |
| `decision_kind` | string | `shot`, `pass`, `drive` |
| `chosen_value` | float | expected points of the taken option |
| `best_value` | float | max over the option set |
| `best_option` | string | `shot`, `pass_to_<rank>` (rank 1 = the most valuable pass), `hold` |
| `decision_quality` | float | `chosen_value - best_value` |
| `heldout` | bool | the game is held out |
| `event` | int32 | play-by-play event of a shot decision, -1 otherwise |
| `receiver_id` | int32 | the receiver of a pass, -1 otherwise |
| `shot_value` | float | the handler's own shot option |
| `hold_value` | float | the EPV hold option, NaN when absent |
| `prior` | string | `location` or `xfg_selected` |
| `fold` | int32 | the grouped fold that valued the game; -1 for held-out games |

### decision_players: `decision_players.parquet` (training games only)

| column | type | meaning |
| --- | --- | --- |
| `player_id` | int32 | |
| `n_decisions` | int32 | decisions in the training games |
| `mean_quality` | float | mean decision quality |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of a bootstrap over the player's games; NaN with one game |
| `insufficient` | bool | under 300 decisions (`analytics.eval.gated`); such rows are not ranked |
| `n_games`, `sum_quality` | | diagnostics |

## Validation (train.validate_quality)

Per game, the players' summed decision quality against their on-court offensive rating (points per 100
offensive possessions on the floor: the offence's players at the first tracked frame of each possession,
from possessions and frames). Spearman per game over players with at least 3 decisions and at least 3 such
players; a two-sided sign test over the games' signs; a bootstrap by game of the mean Spearman; and the same
with quality permuted across the players of each game (the shuffled check).

## Leakage rules

Option values use the decision frame and earlier frames only; no outcome column (`made`, `points`,
`points_to_end`, `end_kind`) enters a context or a value; `schema.assert_time_forward` runs on every context
frame and on the FEATURES subset; `features.assert_decisions_before_outcome` refuses a decision frame outside
its possession span or, for a shot, after the release frame (tested by shifting a frame after the outcome).
Fixture numbers are never results: `reports/decisions.md` from `--fixtures` says "fixture output, not a result".
