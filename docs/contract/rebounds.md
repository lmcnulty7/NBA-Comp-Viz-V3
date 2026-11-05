# Rebound probability: contract of the rebound lane (analytics/rebounds)

Unit: one missed field-goal attempt (a row of the section 18 `rebound_frames` table with a rim frame).
Target: which of the players on court at the rim frame secures the rebound, or none (team rebound).
Inputs: `rebound_frames` (section 18), the frames of section 4 at the rim frame and the 0.4 s before it,
the play-by-play of section 5 (the `team` of the miss gives the offence), `shot_frames` (section 17,
`release_frame` only, for the shot distance) and `possessions` (section 19, fallback for the offence).
Nothing after the rim frame enters a feature; `analytics.schema.assert_time_forward` runs on every
feature frame and `analytics.eval.refuse_heldout` on every training game id.

## Candidate rows (`analytics.rebounds.features.build`, written by `python -m analytics.rebounds.features`)

One row per player on court at the rim frame plus one none row per kept miss, written per game to
`OUT_ROOT/analytics/rebounds/features/<season>/<sportvu_game>.parquet` (held-out games under
`features/heldout/<season>/`).

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index of the missed shot (section 18 `event`) |
| `candidate_player_id` | int32 | NBA player id of the candidate; `-1` on the none row |
| `period` | int16 | period of the rim frame |
| `clock_s` | float | game clock at the rim frame (buckets only, never a feature) |
| `heldout` | bool | the game is held out (sportvu/splits.json or a `heldout/` path) |
| `n_candidates` | int16 | rows of this miss (players on court plus one) |
| `is_rebounder` | bool | TARGET: the play-by-play rebounder (`pbp_rebounder_id`), or the none row on a team rebound |
| `distance_to_rim_ft` | float | candidate to the basket nearest the ball at the rim frame |
| `distance_to_ball_landing_ft` | float | candidate to the ball's horizontal position at the rim frame |
| `angle_to_rim` | float | absolute angle in radians of the candidate seen from the basket; 0 straight in front |
| `speed_toward_rim_ftps` | float | decrease of `distance_to_rim_ft` over the 0.4 s before the rim frame, per second; NaN without a frame within 0.2 s of that time |
| `is_offense` | int8 | the candidate's team shot the ball |
| `nearest_opponent_dist_ft` | float | box-out proxy: distance to the nearest player of the other team at the rim frame |
| `n_opponents_closer_to_rim` | int16 | opponents nearer the basket than the candidate |
| `ball_apex_ft` | float | highest ball z in the 2 s of game clock before the rim frame (miss-level) |
| `shot_distance_ft` | float | ball to basket at the section 17 `release_frame`; else the ball's distance 1 s before the rim frame (miss-level) |
| `is_none` | int8 | 1 on the none row, whose player features are 0 and miss-level features copied |
| `prior_rate` | float | NaN on disk: the candidate's smoothed rebound rate by player and `is_offense`, filled by the trainer from training folds only |

Which misses are kept: `pbp_rebounder_id` not null and `confidence_flag` not `unmatched` (label the player),
or a rim frame and a rebound event with no named rebounder (label none). Dropped and counted: no rim frame,
no rebound event, a named rebounder the roster join could not match, every other null `identity_match`,
unknown offence, fewer than two players in the rim frame, a labelled rebounder not on court in the rim frame.
Assumption: the lane spec drops every null `identity_match`, which would also drop every team rebound
(null by construction); team rebounds are kept as the none class instead.

## Predictions (`rebound_predictions`, written by `python -m analytics.rebounds.train --out OUT_ROOT`)

`OUT_ROOT/analytics/rebounds/<season>/<sportvu_game>.parquet`, held-out games under
`OUT_ROOT/analytics/rebounds/heldout/<season>/`. One row per candidate row above.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index of the missed shot |
| `candidate_player_id` | int32 | NBA player id, `-1` for none |
| `p_prior` | float32 | player-prior baseline: smoothed training-fold rebound rate by player and `is_offense`, normalised within the miss |
| `p_logit` | float32 | conditional logit: logistic regression on the candidate rows, softmax of the linear scores within the miss |
| `p_selected` | float32 | the model selected by out-of-fold per-miss log loss (prior, logit or hgb, named in the report) |
| `is_rebounder` | bool | the target |
| `fold` | int16 | grouped fold (by `sportvu_game`) that scored the row out of fold; `-1` on held-out rows |
| `heldout` | bool | scored once by the models refit on every training row; never fitted or tuned on |

Probabilities of a miss sum to one for each model. Selection metric: per-miss multinomial log loss, the mean
over misses of minus the log probability of the true candidate; also reported: top-1 accuracy (a tie at the
maximum counts 1 over the number tied), bootstrap intervals by game, the paired bootstrap of hgb against prior,
and the calibration of the top candidate's probability (`analytics.eval.report_block`). Numbers from
`--fixtures` are fixture output, not a result.
