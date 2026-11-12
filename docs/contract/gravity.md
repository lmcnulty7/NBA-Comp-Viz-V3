# Gravity and spacing tables (analytics/gravity)

Phase 2, lane GRAVITY. Three tables written by `python -m analytics.gravity.train ... --out OUT_ROOT` under
`OUT_ROOT/analytics/gravity/<table>.parquet`. Inputs are the frames (docs/DATA_CONTRACT.md section 4),
possessions (section 19), rosters (section 16) and the basketball-reference shots (section 10a). Column
names follow the contract's conventions: `sportvu_game` is the Step 1 game id, feet are the court frame of
section 1, nullable columns are marked.

Unit: the player-frame sampled at 1 Hz inside possessions. For every possession with `gap` false and a known
`offense_team_id`, the span is the frames of the same period from `start_frame` to the next possession's
`start_frame` (exclusive), or to the period's last frame; on `analytics.possessions` output the next possession
starts at the previous one's end frame, so the span equals `[start_frame, end_frame - 1]` (`end_frame` itself is
post-outcome and is not read). Elapsed time is `start_clock_s - game_clock_s`; the frame nearest each whole
second is the sample, and frames over a stopped clock share one elapsed value and collapse to one sample. A
sample is kept only with exactly five offensive and five defensive players and a ball position (counts of the
rest are logged).

Target: `nearest_def_dist_ft`, the distance from the offensive player to the nearest defender.

Features (`analytics.gravity.features.FEATURES`, all time-forward, all about location): `x_ft`, `y_ft`,
`dist_basket_ft`, `angle_rad`, `dist_ball_ft`, `has_ball`, `ball_dist_basket_ft`, `zone_make_rate` (the league
make rate in the zone the player stands in, from training-fold shots only: rim, paint, short mid, long mid,
three), `game_clock_s`, `shot_clock_s`, `shot_clock_missing`. Nothing in the features identifies a player: a
per-player shooting prior let a flexible model absorb a player's pull into the expectation, which is the
quantity being measured. `schema.assert_time_forward` runs on the feature frame; no target or post-outcome
column of sections 16 to 20 is in it.

Expectation: ridge regression (the plan's model) on the features above, out of fold by game. Sign convention:
`gravity_ft = expected_ft - nearest_def_dist_ft`. Positive means the nearest defender stays closer than the
player's location alone predicts (the spec's stated meaning; its "y minus expectation" formula has the opposite
sign and was not kept).

## gravity_frames

One row per sampled player-frame. Training rows carry the out-of-fold ridge expectation (fold = the grouped
fold whose test set held the game); held-out rows (fold -1, heldout true) are scored once by ridge refit on
every training row and never rank anything.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame the sample is read from |
| `player_id` | int32 | the offensive player |
| `nearest_def_dist_ft` | float32 | distance to the nearest defender (the target) |
| `expected_ft` | float32 | the ridge expectation from the location features |
| `gravity_ft` | float32 | `expected_ft - nearest_def_dist_ft` |
| `has_ball` | bool | the player is the offensive player nearest the ball |
| `fold` | int16 | grouped fold by game whose test set held this row (>= 0, out of fold); -1 for held-out rows; -2 for in-sample rows (a single training game, where no grouped fold is possible and the report says in-sample) |
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
| `insufficient` | bool | fewer than `--min-possessions` possessions (`analytics.eval.gated`; default 300 on data, 40 on `--fixtures` whose seasons are a few small games); such a player is listed but never ranked |

## spacing_frames

One row per sampled possession-frame (training and held-out, flagged).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `possession` | int32 | section 19 possession |
| `frame` | int32 | the section 4 frame |
| `hull_area_ft2` | float64, nullable | convex hull area of the five offensive players (`scipy.spatial.ConvexHull`, its `volume` in 2-D); null when the five points are collinear |
| `mean_nearest_def_ft` | float64 | mean over the five offensive players of the nearest-defender distance |
| `heldout` | bool | as above |

`python -m analytics.gravity.features --out OUT_ROOT` writes the same `spacing_frames.parquet` plus
`player_frames.parquet` (the feature frame with the target, `zone_make_rate` left null).

## Candidates and validation (reports/gravity.md)

Candidates: constant (floor), ridge (the plan's model, the shipped expectation) and
`HistGradientBoostingRegressor` on the same features, with out-of-fold MSE under grouped folds by
`sportvu_game`, bootstrap intervals by game and the paired bootstrap of MSE(HGB) - MSE(ridge). MSE is not the
selection criterion: gravity is the residual of the expectation, and a flexible model lowers MSE partly by
fitting what should stay in the residual (on the planted-pull fixture a 2 ft pull came out at 0.6 ft under
HGB with a player-identifying prior); HGB is reported as a diagnostic of how much structure the location
features leave unexplained.

Validations: (a) the known high-gravity players rank within the top decile of the gated players, or within the
top `n_known` when the planted players outnumber a decile (two stars among fourteen), season-wide and within
each game (on fixtures the generator's planted players; on real data the top ten are listed without a claim);
(b) a sign test by game of teammates' location-only xFG probability (`analytics.features_xfg` features,
`analytics.train_xfg` logistic, out of fold) with a high-gravity player on the floor, with the mean difference
and a bootstrap interval by game; (c) on the lane's planted-pull fixture, the recovered pull against the
generator's effective pull (tolerance stated in tests/test_gravity.py). Held-out games are scored exactly once.
Numbers from `--fixtures` are fixture output, not a result. The default fixture is `analytics.fixtures`; a
fixture report on it carries a second section from the lane's synthetic planted-pull generator
(`analytics/gravity/fixture_pull.py`: its star sits on part of the possessions and its pull is known), where the
sign test and the pull recovery run. Production code never imports `tests/`.

## CLI

    python -m analytics.gravity.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT] [FIXTURE FLAGS]
    python -m analytics.gravity.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [FIXTURE FLAGS]
    python -m analytics.gravity.train --fixtures | --data DATA_ROOT --report reports/gravity.md [--folds 5] [--seed 0] [--bootstrap 200] [--min-possessions 300] [--out OUT_ROOT] [--no-pull-section] [FIXTURE FLAGS]

FIXTURE FLAGS: `--fixture-games 8`, `--fixture-possessions 40`, `--fixture-heldout-every K` (default 4 with at
least four games), `--fixture-generator shared|pull` (`--n-games`, `--n-possessions` and the values `auto`,
`fallback` are kept as aliases). Fits run under `analytics.eval.fit_threads()`.

With `--data`, frames are read one game at a time: each game's frames are loaded, checked, turned into its
player-frame and spacing rows (and, in train, the players on the floor at each of its shots), then dropped
before the next game is read. Only those per-game rows are kept, so memory does not grow with the season's frames.
