# Matchups tables (analytics/matchups)

Individual matchup assessment from the Phase 0 tables. Unit: defender x offensive player partial possession.
Code: `analytics/matchups/features.py` (assignment, `matchup_possessions`), `analytics/matchups/train.py`
(`matchup_defenders`, ridge adjustment, validation), `analytics/matchups/validate.py` (inputs). Report:
`reports/matchups.md`. No NBA.com matchup table exists for 2015-16 (they start in 2017-18), so the procedure is
validated internally.

## Inputs

Per game: frames (DATA_CONTRACT section 4), play-by-play (section 5), possessions (19), shot_frames (17),
rosters (16), from the DATA_ROOT layout

    DATA_ROOT/sportvu_contract/<game>.parquet, <game>_pbp.parquet          (held-out games under heldout/)
    DATA_ROOT/analytics/<table>/<season>/<game>.parquet                    (held-out under analytics/<table>/heldout/<season>/)

and, optionally, the xFG lane's predictions `OUT_ROOT/analytics/xfg/<season>/*.parquet` with columns
`sportvu_game, event, player_id, p_selected, fold`. `fold` must be present and non-negative on every
training row (out-of-fold); otherwise the file is refused and the location-only fallback of
`analytics.features_xfg` (a logistic regression out-of-fold by game) gives p.

## Assignment

Inside each possession the frames are sampled at 1 Hz (the frame whose elapsed time since `start_frame`
is nearest each whole second). At each sampled frame every defender (rows of the team that is not
`offense_team_id`) is assigned to one offensive player by `scipy.optimize.linear_sum_assignment` on

    cost[d, o] = dist(d, o) + 0.5 * dist(d, midpoint(o, basket))

with one basket per (offense team, period): each possession votes for the basket nearer its offensive
centroid over the last third of its sampled frames, and the majority over that team's possessions in the
period applies to all of them. Positions only: nothing about the outcome enters the assignment.

## Table `matchup_possessions`

One row per (possession, defender, offensive player) pair with at least one assigned sampled frame.
Written by `python -m analytics.matchups.features --data DATA_ROOT --out OUT_ROOT` to
`OUT_ROOT/analytics/matchups/<season>/<game>.parquet` (held-out under `heldout/<season>/`).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `def_player_id` | int32 | the defender |
| `off_player_id` | int32 | the offensive player guarded |
| `partial_possession` | float32 | 0 to 1: fraction of the possession's sampled frames at which the defender was assigned to this player |
| `points_allowed_share` | float32 | the possession's `points` times the share, when the offensive player took the possession's last field-goal attempt; else 0 (TARGET) |
| `xfg_expected_share` | float32 | sum over that player's attempts in the possession of p(make) times the shot value (2 or 3), times the share (TARGET) |
| `switches` | int16 | sampled frames (after the first) at which this defender's assignment changed, per defender and possession, repeated on each of the defender's rows |

Feature columns (`features.FEATURES`): `partial_possession`, `switches`. Targets: `points_allowed_share`,
`xfg_expected_share`. `features.feature_frame` returns ids plus features and runs
`schema.assert_time_forward` on it. Per defender and possession the shares sum to at most 1 (exactly 1 when
the defender was assigned at every sampled frame).

Auxiliary tables (in memory, not written by default): `matchup_frames` (one row per defender and sampled
frame: `sportvu_game, possession, k, frame, def_player_id, off_player_id, dist_ft, changed`) feeds the
stability check; `shots` (one row per attempt inside a possession: shooter, value, p, assigned defender and
its distance at the sampled frame at or before the release, and `k`, that frame's index in `matchup_frames`,
-1 when it holds no assignment) feeds the defended FG% check. The possession's points go to the shooter of
its last attempt; when that shooter is unmatched (null `shooter_id`) nothing is attributed.

## Table `matchup_defenders`

One row per defender over the training games, written by `python -m analytics.matchups.train ... --out OUT_ROOT`
to `OUT_ROOT/analytics/matchups/tables/<season>/matchup_defenders.parquet` (next to
`matchup_possessions_train.parquet`, the training games' matchup_possessions in one file; both live apart from
the per-game directory so a glob over `<season>/*.parquet` sees each game once).

| column | type | meaning |
| --- | --- | --- |
| `def_player_id` | int | the defender |
| `n_possessions` | int | distinct possessions in which the defender was assigned at least once |
| `points_allowed_per100` | float | 100 times the sum of `points_allowed_share` over the sum of `partial_possession` |
| `expected_per100` | float | 100 times the sum of `xfg_expected_share` over the sum of `partial_possession` |
| `diff_per100` | float | `points_allowed_per100 - expected_per100`; positive = more points allowed than expected (worse) |
| `ridge_value` | float | 100 times the defender's coefficient in the ridge regression of possession points on defender and offensive-player indicator columns (final alpha by grouped CV by game over all training games; the out-of-fold comparison chooses each outer fold's alpha by an inner CV over that fold's training games); positive = worse |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of `diff_per100` over bootstrap resamples of whole games; NaN with one game |
| `insufficient` | bool | fewer than 300 distinct possessions (`analytics.eval.gated`); such rows are never ranked |

The ridge design has one row per possession, a column per defender holding its share of the possession's
sampled frames and a column per offensive player holding the share of frames it was guarded; the target is
the possession's `points`. The baseline comparison (out-of-fold MSE of possession points under grouped folds
by game) is between the constant model, the raw model (the sum over the defenders on the floor of their
training-fold points allowed per matchup possession times their share) and the ridge.

## Validation (internal)

- Assignment stability: share of defender sampled frames whose assignment equals the previous one, with a
  bootstrap interval by game.
- Defended FG% consistency: FG% of attempts whose assigned defender is within 4 ft at the sampled frame at
  or before the release versus beyond, with intervals and a paired difference.
- The held-out game(s) of `sportvu/splits.json`: matchups built with the training-fitted xFG and scored once
  by the selected candidate (raw or ridge, whichever had the lower out-of-fold MSE on the training folds; the
  rule is fixed before scoring) next to the constant floor, both fit on all training games; never fitted or
  tuned on.
- xFG `fold` convention: `>= 0` out-of-fold, `-1` held-out, `-2` in-sample; `load_xfg` refuses a file whose
  training rows carry a missing or negative fold.
- Held-out membership: a `heldout/` directory, the fixture flag, or the game id by `analytics.eval.is_heldout_game`
  (sportvu/splits.json tokens plus the held-out arena); an xFG table's own `heldout` column wins when present, else
  `analytics.eval.heldout_mask` by id. Fits (the location-only xFG fallback, the ridge) run under
  `analytics.eval.fit_threads(1)`.
- Fixture runs read `analytics.fixtures` directly (production code never imports `tests/`); every CLI of the lane
  takes `--fixture-games`, `--fixture-possessions` and `--fixture-heldout-every`, with the older `--n-games`,
  `--n-possessions` and `--heldout-every` kept as aliases.

## Leakage rules

Assignments use positions only. `points` and `made` enter only as targets. `schema.assert_time_forward`
runs on the feature frame and on the ridge design columns. `analytics.eval.refuse_heldout` runs on the
training game ids before any fit. The xFG `p_selected` is used only with a verified out-of-fold `fold`.
