# Gravity and spacing tables (analytics/gravity)

Phase 2, lane GRAVITY. Three tables written by `python -m analytics.gravity.train ... --out OUT_ROOT` under
`OUT_ROOT/analytics/gravity/<table>.parquet`. Inputs are the frames (docs/DATA_CONTRACT.md section 4),
possessions (section 19), rosters (section 16) and the basketball-reference shots (section 10a). Column
names follow the contract's conventions: `sportvu_game` is the Step 1 game id, feet are the court frame of
section 1, nullable columns are marked.

Unit: the player-frame sampled at 1 Hz inside possessions. For every possession with `gap` false and a known
`offense_team_id`, the span is the frames of the same period from `start_frame` to the next possession's
`start_frame` (exclusive; `end_frame` is post-outcome and is not read), or to the period's last frame. Elapsed
time is `start_clock_s - game_clock_s`; the frame nearest each whole second is the sample. A sample is kept
only with exactly five offensive and five defensive players and a ball position (counts of the rest are logged).

Target: `nearest_def_dist_ft`, the distance from the offensive player to the nearest defender.

Features (`analytics.gravity.features.FEATURES`, all time-forward, all about location): `x_ft`, `y_ft`,
`dist_basket_ft`, `angle_rad`, `dist_ball_ft`, `has_ball`, `ball_dist_basket_ft`, `skill_zone`, `skill_three`
(the player's smoothed make rate by zone, from training-fold shots only), `game_clock_s`, `shot_clock_s`,
`shot_clock_missing`. `schema.assert_time_forward` runs on the feature frame; no target or post-outcome column
of sections 16 to 20 is in it.

Sign convention: `gravity_ft = expected_ft - nearest_def_dist_ft`. Positive means the nearest defender stays
closer than the player's location alone predicts (the spec's stated meaning; its "y minus expectation" formula
has the opposite sign and was not kept).

## gravity_frames

One row per sampled player-frame. Training rows carry the out-of-fold expectation of the selected model
(fold = the grouped fold whose test set held the game); held-out rows (fold -1, heldout true) are scored once
by the model refit on every training row and never rank anything.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame the sample is read from |
| `player_id` | int32 | the offensive player |
| `nearest_def_dist_ft` | float32 | distance to the nearest defender (the target) |
| `expected_ft` | float32 | the selected model's expectation from the location features |
| `gravity_ft` | float32 | `expected_ft - nearest_def_dist_ft` |
| `has_ball` | bool | the player is the offensive player nearest the ball |
| `fold` | int16 | grouped fold by game whose test set held this row; -1 for held-out rows |
| `heldout` | bool | the game is held out in sportvu/splits.json (or flagged so by the fixture) |

## gravity_players

One row per player seen on offence in a training game, sorted with the gated players first by gravity.

| column | type | meaning |
| --- | --- | --- |
| `player_id` | int32 | NBA player id |
| `n_possessions` | int32 | distinct (sportvu_game, possession) with the player on offence |
| `n_frames` | int32 | sampled player-frames |
| `n_games` | int32 | training games with the player |
| `gravity_ft` | float64 | mean `gravity_ft` over the player's frames |
| `lo`, `hi` | float64, nullable | 2.5 and 97.5 percentiles of the mean over resamples of the player's games; null with one game |
| `insufficient` | bool | fewer than `--min-possessions` (default 300) possessions (`analytics.eval.gated`); such a player is listed but never ranked |

## spacing_frames

One row per sampled possession-frame.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame |
| `hull_area_ft2` | float64, nullable | convex hull area of the five offensive players (`scipy.spatial.ConvexHull`, its `volume` in 2-D); null when the five points are collinear |
| `mean_nearest_def_ft` | float64 | mean over the five offensive players of the nearest-defender distance |
| `heldout` | bool | as above |

## Selection and validation (reports/gravity.md)

Candidates: constant (floor), ridge regression (the plan's model) and `HistGradientBoostingRegressor` on the
same features. Selection metric: out-of-fold MSE under grouped folds by `sportvu_game`, bootstrap intervals
by game; HGB is kept only when the paired bootstrap by game of MSE(HGB) - MSE(ridge) excludes zero on HGB's
side. Validations: the known high-gravity players rank in the top decile of the gated players (on fixtures the
generator's planted players; on real data the top ten are listed without a claim), and a sign test by game of
teammates' location-only xFG probability (`analytics.features_xfg` features, `analytics.train_xfg` logistic,
out of fold) with a high-gravity player on the floor, with the mean difference and a bootstrap interval by game.
Held-out games are scored exactly once. Numbers from `--fixtures` are fixture output, not a result.

## CLI

    python -m analytics.gravity.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.gravity.features --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.gravity.train --fixtures | --data DATA_ROOT --report reports/gravity.md [--folds 5] [--seed 0] [--bootstrap 200] [--min-possessions 300] [--out OUT_ROOT]
