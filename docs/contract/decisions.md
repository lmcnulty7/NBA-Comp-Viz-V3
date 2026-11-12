# Decision quality: contract (analytics/decisions)

Unit: a decision event of the ball handler (a shot, a pass or a drive start). Target: the value of the chosen
option minus the best available option, in expected points, at the decision frame. Code:
`analytics/decisions/{validate,features,train,eda}.py`; tests `tests/test_decisions.py`,
`tests/test_decisions_streaming.py`; report
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
| xfg_predictions (lane XFG) | `OUT_ROOT/analytics/xfg/<season>/*.parquet` (one file, held-out rows flagged `heldout=True, fold=-1`; a `heldout/<season>/` directory is read too when present), filtered on `sportvu_game` | `p_selected` on shot events; optional |

OUT_ROOT defaults to DATA_ROOT, and both `OUT_ROOT/analytics` and `DATA_ROOT/analytics` are searched, so on a
real-data run without `--out` (as run_all makes it) the xFG and EPV outputs under `DATA_ROOT/analytics` are read.

With `--data`, frames are read one game at a time (`validate.stream_bundles`): each game's frames are loaded,
checked, turned into its decision contexts, on-court ratings and value-model fit rows, then dropped before the
next game is read. Only those per-game tables are kept (with the small Phase 0 tables and upstream rows), so
memory does not grow with the season's frames. The xFG predictions file is read once, not once per game.
Tests: `tests/test_decisions_streaming.py`.

A game listed under `heldout/` or whose id carries a held-out token of `sportvu/splits.json`
(`analytics.eval.is_heldout_game`, which includes the held-out arena) is held out: it is
valued once by the model refit on all training games and never fits or tunes anything
(`analytics.eval.refuse_heldout` on the training ids).

## Event detection (features.py)

| kind | rule | decision frame | player |
| --- | --- | --- | --- |
| holder | offensive player within 2.5 ft (horizontal) of the ball while the ball is under 8 ft; nearest when several; a run shorter than 3 frames is ignored | | |
| pass | holder A's run ends at f, teammate B's run starts at g, (g - f) / fps <= 1.5 s, no release frame in [f - 2, g] | f | A, receiver B |
| shot | shot_frames row with `release_frame` inside the possession span | release_frame | `shooter_at_release_id`, else `shooter_id` |
| drive | the holder's speed toward the basket (frame to frame) above 8 ft/s for at least 0.6 s inside one holder run; one per run | the first frame at which the speed toward the basket is measured above 8 ft/s in that run | the holder |

Possessions without an offence, with `gap` true, or without both sides on the floor are skipped and counted.
Every release frame inside the span blocks a pass, matched shooter or not. The attacked basket is inferred per
(offense_team_id, period) by majority vote over that team's possessions in the period, each voting with the end
where its offence's mean x lies (a transition possession starting in the backcourt is outvoted; a tie goes to the
frame-weighted mean x). The receiver of a pass always gets a teammate slot; teammates beyond four are dropped and
counted (`teammates_truncated`).

## Option values at the decision frame (features.option_values)

| option | value |
| --- | --- |
| shot | points(dist) * max(p - 0.03 * max(0, 6 - d_def), 0.01), points = 3 at 22 ft or more else 2, p = the value model's make probability at the position (location-only xFG), d_def = nearest-defender distance; with the xFG lane's `p_selected` joined on the shot event, p is replaced and the contest term is not applied |
| pass to teammate t | points * p(t's position) * openness(d_def(t)) * risk(length), with no contest term (openness already carries the defender distance), openness = 0.5 + 0.5 * min(d, 6) / 6, risk = 0.95 - 0.01 * max(0, length - 15 ft) |
| hold | `epv_points` of the last EPV state at or before the frame in the same possession; absent without the EPV table |
| drive | shot value at the position 8 ft nearer the rim along the line to the basket (never nearer than 3 ft), with the decision-frame d_def, times 0.9; in every decision's option set, and the chosen value of a drive decision |

`chosen_value` is the value of the taken option; `best_value` the max over shot, the pass options, hold and
drive; `decision_quality = chosen_value - best_value`, at most 0 for every kind (regret). The contest term,
openness, pass risk and drive value are assumptions, not fitted; the prior of the option values (location-only
xFG, or the xFG lane's `p_selected`) is selected by a rule fixed in advance (pooled coverage of the training
games' shot decisions at least 0.5) and reported, not fitted. When any game carries an xFG table, the
`xfg_selected` events are computed for every game (rows without a joined `p_selected` fall back to the
location prior), so no game vanishes under partial coverage. Upstream EPV and xFG rows are filtered: a training
game refuses rows with `fold == -1`, a held-out game refuses rows not flagged `heldout`; refused counts are
logged in the report.

## The value model (train.py)

The location-only xFG of `analytics/features_xfg.py` (distance_ft, angle_rad, is_three, is_corner): a logistic
regression with standardisation (C=1) fit on the shot_frames rows with a release frame, at the shooter's
position at the release relative to the basket nearer the shooter, target `made`. Fit under grouped folds by
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
| `best_option` | string | `shot`, `pass_to_<rank>` (rank 1 = the most valuable pass), `hold`, `drive` |
| `decision_quality` | float | `chosen_value - best_value`, at most 0 |
| `heldout` | bool | the game is held out |
| `event` | int32 | play-by-play event of a shot decision, -1 otherwise |
| `receiver_id` | int32 | the receiver of a pass, -1 otherwise |
| `shot_value` | float | the handler's own shot option |
| `hold_value` | float | the EPV hold option, NaN when absent |
| `drive_value` | float | the drive option |
| `prior` | string | `location` or `xfg_selected` (the held-out table carries the prior actually selected) |
| `fold` | int32 | the grouped fold that valued the game (>= 0, out-of-fold); -1 for held-out games (valued by the full refit); -2 for in-sample rows (one training game, no fold possible) |

Upstream tables follow the same convention: for a training game only rows with `fold >= 0` are accepted
(-1 held-out, -2 in-sample and a table without a `fold` column, which counts as -1, are refused); for a
held-out game only rows flagged `heldout`.

`possession`, `frame`, `player_id`, `event`, `receiver_id` and `fold` are written as int32; the values as float64.

### decision_players: `decision_players.parquet` (training games only)

| column | type | meaning |
| --- | --- | --- |
| `player_id` | int32 | |
| `n_decisions` | int32 | decisions in the training games |
| `mean_quality` | float | mean decision quality |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of a bootstrap over the player's games; NaN with one game |
| `insufficient` | bool | under 300 decisions (`analytics.eval.gated`); such rows carry no rank and are listed after the ranked rows, by id |
| `rank` | float | 1 = best mean quality among the players past the gate; NaN when insufficient |
| `mean_quality_shot`, `mean_quality_pass`, `mean_quality_drive` | float | mean quality by kind (each is regret, at most 0); NaN without decisions of that kind |
| `n_shot`, `n_pass`, `n_drive` | int32 | decisions by kind |
| `n_games`, `sum_quality` | | diagnostics |

## Validation (train.validate_quality)

Per game, the players' mean decision quality (not the sum: quality is regret, so a sum would rank by
decision count) against their on-court offensive rating (points per 100 offensive possessions on the floor:
the offence's players at the first tracked frame of each possession, from possessions and frames). Spearman
per game over players with at least 3 decisions and at least 3 such players; a two-sided sign test over the
games' signs; a bootstrap by game of the mean Spearman; and the same with mean quality permuted across those
same players within each game (the shuffled check, filtered before shuffling).

## Leakage rules

Option values use the decision frame and earlier frames only; no outcome column (`made`, `points`,
`points_to_end`, `end_kind`) enters a context or a value; `schema.assert_time_forward` runs on every context
frame and on the FEATURES subset; `features.assert_decisions_before_outcome` refuses a decision frame outside
its possession span or, for a shot, after the release frame (tested by shifting a frame after the outcome).
Fixture numbers are never results: `reports/decisions.md` from `--fixtures` says "fixture output, not a result" in
its banner and on every section heading, and lists the written files relative to OUT_ROOT. The fixture games come
from `analytics.fixtures` (knobs `--fixture-games`, `--fixture-possessions`, `--fixture-heldout-every`; the older
`--n-games`, `--n-possessions`, `--heldout-every` remain as aliases); production code never imports `tests/`. Every
value-model fit runs inside `analytics.eval.fit_threads()`.
