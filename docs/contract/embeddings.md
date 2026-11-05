# Player style embeddings: tables and protocol (analytics/embeddings)

Unit: player-season. Unsupervised. Code: `analytics/embeddings/{validate,features,train,eda}.py`; tests
`tests/test_embeddings.py`; report `reports/embeddings.md`; notebook `notebooks/eda_embeddings.ipynb`.

The tracking and shot aggregates below are built from TRAINING games only; the NBA.com block is season-level as the
source publishes it and covers every game of the season, held-out ones included. Held-out games of `sportvu/splits.json` (the ids, the
CLE arena, files under `heldout/`, shot rows flagged `heldout`) are set aside by the loader before any table is
read into a feature; the training game ids pass `analytics.eval.refuse_heldout`, the feature frame passes
`analytics.schema.assert_time_forward`.

## Inputs (DATA_ROOT layout on the owner's Mac)

| input | path | role |
| --- | --- | --- |
| tracking frames (section 4) | `DATA_ROOT/sportvu_contract/<game>.parquet` (+ `heldout/`) | tracking features; streamed one game at a time |
| possessions (section 19) | `DATA_ROOT/analytics/possessions/<season>/<game>.parquet` (+ `heldout/<season>/`) | which frames belong to which possession and offence; `n_possessions` |
| rosters (section 16) | `DATA_ROOT/analytics/rosters/<season>/<game>.parquet` | resolves a null basketball-reference shooter id by `player_slug = bref_slug` |
| shots (section 10a or 10) | `DATA_ROOT/collect/bref_shots/shotchart/2015-16.parquet` or `collect/shots/shotchartdetail/2015-16.parquet` | shot mix; restricted to the tracked training games |
| NBA.com aggregates (section 11) | `DATA_ROOT/collect/tracking/aggregates/<season>.parquet` | optional: drives, touches, passes, play types, hustle; the 2016-17 file supplies the next-season target only |
| positions | `DATA_ROOT/collect/positions.parquet` (collect/positions.py: `player_slug, player, season, position`) or `collect/tracking/positions/<season>.parquet` (`player_id, season, position`) | optional, a NEEDED INPUT for the 5-NN position metric; without it the metric is n/a and the autoencoder gate cannot be evaluated |

Position rule: a table without `player_id` is joined to the section 16 rosters by `collect.positions.join_rosters`
(`bref_slug = player_slug`, season from `sportvu_game`), then reduced to one row per (player_id, season). The
basketball-reference position collapses to G/F/C by the part before the first hyphen: PG, SG, G to G; SF, PF, F to F;
C to C (so SF-SG is F and C-PF is C); anything else is dropped with a logged count.

A game without a possessions table is skipped and counted. Shots of games without tracking frames are not used
(attempts per possession needs the tracking possessions).

## Table player_season_features

One row per player-season with tracking frames in the training games. Written to
`OUT_ROOT/analytics/embeddings/<season>/player_season_features.parquet`.

| column | type | meaning |
| --- | --- | --- |
| `player_id` | int | NBA player id (section 4) |
| `season` | string | `YYYY-YY` from the game date |
| FEATURES | float, NaN allowed | the 38 columns of `analytics.embeddings.features.FEATURES`, in this order: `avg_speed_ftps`, `dist_per_poss_ft`, `zone_near_left`, `zone_near_right`, `zone_mid_left`, `zone_mid_right`, `zone_far_left`, `zone_far_right`, `ball_share`, `nearest_teammate_ft`, `offball_ball_dist_ft`, `shot_share_restricted_area`, `shot_share_paint_non_ra`, `shot_share_mid_range`, `shot_share_corner_3`, `shot_share_above_break_3`, `shot_share_backcourt`, `shot_share_3pt`, `attempts_per_poss`, `agg_drives`, `agg_touches`, `agg_passes_made`, `agg_catch_shoot_fga`, `agg_pull_up_fga`, `agg_deflections`, `agg_contested_shots`, `agg_screen_assists`, `pt_<play_type>_freq` for the 11 Synergy play types (isolation, transition, prballhandler, prrollman, postup, spotup, handoff, cut, offscreen, offrebound, misc) |
| `n_possessions` | int | distinct (game, possession) spans the player has frames in, training games only |
| `insufficient` | bool | `n_possessions < 300`; such rows are written but neither fitted nor evaluated (a fixture run lowers the gate to 20 and says so) |

Feature definitions (details in the `features.py` docstring):

- tracking: mean speed over consecutive frames of the same game and period with `0 < dt <= 2 s` on `ts_ms` (the gap
  rule on the wall clock, not `game_clock_s`; steps over 30 ft/s dropped; frames outside a possession span are included,
  so dead-ball movement counts); distance over those steps per possession; share of the player's offensive frames in each cell of a
  3 by 2 grid of the attacked half (x bands of 47/3 ft from the attacked baseline, y halves; positions mirrored so the
  attacked rim is at (5.25, 25); the attacked basket of a (game, period, team) is the side where the ball spends that
  team's offensive frames; backcourt frames count in the denominator only); share of offensive frames as the handler
  (nearest to the ball, within 3 ft, ball under 8 ft); mean distance to the nearest teammate over all frames; mean
  distance to the ball over offensive frames as a non-handler;
- shots: share of attempts by `zone_basic` (NBA.com names mapped; derived from `court_x, court_y, shot_type` for a
  table without the column), share of 3PT, attempts per tracking possession;
- aggregates: section 11 long table of the SAME season pivoted wide; a metric is matched by name under the endpoints
  listed in `features.AGG_METRICS`; play-type frequency is `poss_pct` (or `freq`, or a `poss` count normalised over
  the play types), offensive grouping; a missing play type is 0, a missing other metric is NaN. On a machine without
  aggregates the block is NaN on every row and the trainer uses the tracking and shot features only.

Standardisation (median imputation then z-scores) is fit on the sufficient rows inside the trainer; the table
stores raw values.

## Table player_embeddings

`OUT_ROOT/analytics/embeddings/<season>/player_embeddings.parquet`. Sufficient rows only; both methods when both
were fitted (the chosen method's rows first).

| column | type | meaning |
| --- | --- | --- |
| `player_id`, `season` | int, string | as above |
| `method` | string | `pca` or `autoencoder` |
| `e1` .. `e8` | float | the 8 embedding coordinates (PCA scores, or the autoencoder bottleneck activations in [-1, 1]) |

## Table embedding_neighbours

`OUT_ROOT/analytics/embeddings/<season>/embedding_neighbours.parquet`, the chosen method, Euclidean distance in
the embedding.

| column | type | meaning |
| --- | --- | --- |
| `player_id`, `season` | int, string | the query player-season |
| `method` | string | the chosen method (added to the spec columns so the table says which space it was built in) |
| `neighbour_rank` | int | 1 to 5, 1 is the nearest |
| `neighbour_player_id` | int | the neighbour |
| `neighbour_season` | string | the neighbour's season (added: with several tracking seasons a player would be ambiguous) |
| `distance` | float | Euclidean distance |

## Candidates, gate and evaluation

1. PCA to 8 components, always.
2. Autoencoder: sklearn `MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation=tanh)` reconstructing the
   standardised features; the bottleneck is the second hidden layer's activation. Trained ONLY when PCA's 5-NN
   position accuracy is under 0.60; kept ONLY when it beats PCA on that metric. Without positions the gate cannot be
   evaluated and PCA stands.
3. 5-NN position accuracy: leave-one-out over the sufficient player-seasons with a position; majority of the 5
   nearest player-seasons' positions, ties to the nearest; baseline the majority class share.
4. Next-season R squared: ridge (alpha 1) from the 2015-16 embedding to a 2016-17 aggregate stat (the first present
   of `leaguedashptstats.drives`, `touches`, `passes_made`, `leaguehustlestatsplayer.contested_shots`), out-of-fold
   under grouped K-fold by `player_id` (5 folds, or the number of players when fewer); baseline the fold mean.
   The 2016-17 value is a target only: `aggregate_features` reads one season, so no next-season value enters the
   embedding.
5. Intervals: 2.5 and 97.5 percentiles of bootstrap resamples of players (`analytics.eval.bootstrap_by_game` with
   `player_id` as the group) over the per-player out-of-sample outcomes (correct flag, out-of-fold prediction).
   The imputer, scaler and embedding (PCA or autoencoder) are fit once on all sufficient rows, unsupervised; only the
   ridge fit and the k-NN vote are out-of-sample.
6. The gate's comparison metric is the reported 5-NN accuracy (leave-one-out point estimate). A fifth branch of the
   rule: an autoencoder trained but with undefined accuracy leaves PCA chosen.

## Leakage rules

- No held-out game row in any aggregate (loader, `refuse_heldout`, a second check on the shot rows).
- No target or post-outcome column in the feature frame (`schema.assert_time_forward`); the possessions table's
  `end_frame` is used only to assign frames to spans, never as a feature.
- No position label and no next-season value among the features (`features.assert_no_leak`).

## Commands

    python -m analytics.embeddings.validate --fixtures | --data DATA_ROOT [--out OUT_ROOT]
    python -m analytics.embeddings.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--n-min 300]
    python -m analytics.embeddings.train --fixtures [--local-fixtures] | --data DATA_ROOT --report reports/embeddings.md [--out OUT_ROOT] [--seed 0] [--bootstrap 200]

A fixture run writes its tables under a temporary directory unless `--out` is given; nothing under `--fixtures`
is a result.
