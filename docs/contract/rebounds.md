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

Which misses are kept (symmetric between the two classes; the frame-level localisation of the rebound is
never required because the label is the play-by-play rebounder): a rim frame, a rebound event and
`pbp_rebounder_id` not null (label the player, whatever `identity_match` or `confidence_flag` say; the
report counts `identity_match_null_kept`, `identity_match_false_kept`, `unmatched_flag_kept`), or a rim frame
and a rebound event with no named rebounder (label none). Dropped and counted: no rim frame, no rebound
event, a named rebounder the roster join could not match (name present, id null), unknown offence, fewer
than two players in the rim frame, a labelled rebounder not on court in the rim frame.
Assumption: the lane spec drops every null `identity_match` or `unmatched` flag, which would drop every team
rebound (null by construction) and every player rebound whose frame the localiser missed; both are kept
since neither affects the label or the features. The offence of a miss is the section 5 `team` of the miss
event, else the `offense_team_id` of the possession with the largest `start_frame` at or before the rim frame.

## Predictions (`rebound_predictions`, written by `python -m analytics.rebounds.train --out OUT_ROOT`; OUT_ROOT defaults to DATA_ROOT on real data, as run_all passes no --out)

`OUT_ROOT/analytics/rebounds/<season>/<sportvu_game>.parquet`, held-out games under
`OUT_ROOT/analytics/rebounds/heldout/<season>/`. One row per candidate row above.

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `event` | int32 | play-by-play event index of the missed shot |
| `candidate_player_id` | int32 | NBA player id, `-1` for none |
| `p_prior` | float32 | player-prior baseline: smoothed training-fold rebound rate by player and `is_offense`, normalised within the miss; NaN on held-out rows unless prior is the selected model |
| `p_logit` | float32 | binary-logistic approximation of the conditional logit: logistic regression on the candidate rows with the binary target, softmax of the linear scores within the miss; NaN on held-out rows unless logit is the selected model |
| `p_selected` | float32 | the model selected by out-of-fold per-miss log loss (prior, logit or hgb, named in the report) |
| `is_rebounder` | bool | the target |
| `fold` | int16 | grouped fold (by `sportvu_game`) that scored the row out of fold; `-1` on held-out rows; `-2` when one training game forced an in-sample fit (`train.IN_SAMPLE_FOLD`) |
| `heldout` | bool | scored once by the selected candidate alone, refit on every training row; never fitted or tuned on; the non-selected models are not scored on held-out rows (their columns are NaN there) |

Probabilities of a miss sum to one for each model. The hgb candidate's probabilities are sharpened within the miss by
a temperature (improvement search variant 1, reports/archive/rebounds.md): p proportional to q^tau, where q is the
binary model's probability and tau (bounded 0.25 to 8) minimises the per-miss log loss of hgb's inner out-of-fold
predictions over 5 inner grouped folds of the fit's own training games; the ranking within a miss is unchanged.
Selection metric: per-miss multinomial log loss, the mean
over misses of minus the log probability of the true candidate; also reported: top-1 accuracy (a tie at the
maximum counts 1 over the number tied), bootstrap intervals by game, the paired bootstrap of hgb against prior
and the calibration of the top candidate's probability (`analytics.eval.report_block`), all on out-of-fold
training rows. Held-out rows get one score of the selected candidate next to a uniform floor (1 over the
candidates of the miss, no fit); no bootstrap or paired comparison is computed on them. Numbers from
`--fixtures` are fixture output, not a result.

Held-out membership: the bundle's `heldout/` path or fixture flag, or the game id by
`analytics.eval.is_heldout_game` / `heldout_mask` (sportvu/splits.json tokens plus the held-out arena);
`analytics.eval.refuse_heldout` runs again inside every fit. Fits run under `analytics.eval.fit_threads(1)`.
Fixture runs read `analytics.fixtures` directly (production code never imports `tests/`) and every CLI of the lane
takes `--fixture-games`, `--fixture-possessions` and `--fixture-heldout-every` (the older `--games` and
`--possessions` remain as aliases).

## Frozen for the held-out run (freeze-v1)

Frozen from the code on branch track-fix, reports/rebounds.md (rerun 2026-11-12T11:50:53Z with the variant merged)
and reports/archive/rebounds.md. Every number below is out-of-fold on the training games, from runs with
`--skip-heldout`; no held-out number has been read.

**Selected model:** `hgb` with the accepted improvement-search variant 1 (temperature on the within-miss
normalisation: p proportional to q^tau, tau fitted on inner out-of-fold hgb predictions; tau 1.255 to 1.257 across
the 5 outer folds in the archive run).

| setting (analytics/rebounds/train.py, features.py) | frozen value |
| --- | --- |
| booster | `HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=3, min_samples_leaf=10, early_stopping=False, random_state=seed)` on the candidate rows, normalised within the miss |
| temperature | `TEMPERATURE = True`, tau bounded to `TAU_BOUNDS = (0.25, 8.0)` (bounded scalar search, xatol 0.001), fitted over `INNER_FOLDS = 5` grouped folds of each fit's training games |
| prior feature | `SMOOTH = 10` pseudo-opportunities toward the per-side base rate, none row at the Laplace team-rebound rate; out of fold over 5 inner grouped folds |
| other candidates | `prior` (normalised smoothed rates), `logit` (median imputation, standardisation, C = 1, max_iter 2000, softmax within the miss) |
| features | look-back 0.4 s within 0.2 s; apex window 2 s; shot distance fallback 1 s before the rim frame |
| drop rules | a miss is dropped for no_rim_frame, no_rebound_event, unmatched_named_rebounder, offense_unknown, rim_frame_not_in_frames, too_few_players or rebounder_not_on_court; team rebounds are kept as the none class |

Rules the held-out run applies: selection by lowest out-of-fold per-miss multinomial log loss; top-1 accuracy
reported against the 40 percent plan target; 5 grouped folds by `sportvu_game`, seed 0; 200 game bootstrap
resamples, seed 0. Held out: a game under `heldout/`, flagged, or matching sportvu/splits.json by id (every game at
CLE and 12.16.2015.PHX.at.GSW; 17 games, 1227 misses in the training run). Only the selected candidate is refit on
every training row and scores them once, next to a uniform floor; point estimates, no bootstrap or paired test.

Out-of-fold headline (49288 misses, 604 games): hgb per-miss log loss 1.6009 [1.5882, 1.6103], top-1 accuracy
0.4797 [0.4756, 0.4848]; logit 1.7558 [1.7418, 1.7660]; uniform reference 2.3881. Variant 1 against the incumbent
hgb (archive, 1000 game resamples): -0.0225 [-0.0246, -0.0205]. Top-candidate calibration slope 1.3667
[1.3396, 1.4015], ECE 0.0414 [0.0379, 0.0454].

Stage 8 command:

    python3 run_all.py --data "$DATA_ROOT" --ext xfg rebounds matchups gravity ghosting embeddings epv decisions lineups winprob load --continue-on-error

run_all.step_argv gives `analytics.rebounds.train` the argv `--data DATA_ROOT --seed 0 --report reports/rebounds.md`;
validate and features get `--data DATA_ROOT --seed 0`. OUT_ROOT defaults to DATA_ROOT, so the held-out predictions go
under `analytics/rebounds/heldout/<season>/`.
