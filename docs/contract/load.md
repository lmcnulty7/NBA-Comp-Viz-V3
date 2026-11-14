# Load and fatigue tables: `analytics/load/<season>/`

Extension of the Phase 0 contract (docs/DATA_CONTRACT.md sections 4, 6, 11, 15, 16) built by the `analytics/load`
package. Three parquet tables per season, written under `OUT_ROOT/analytics/load/<season>/`; the rows of a held-out
game of `sportvu/splits.json` go under `OUT_ROOT/analytics/load/heldout/<season>/` instead and never fit or tune
anything (every row also carries `heldout`). `sportvu_game` is the Step 1 game id (`MM.DD.YYYY.AWAY.at.HOME`).

Inputs: the section 4 frames (one game at a time), the section 6 manifest (dates and teams: the schedule), the
section 16 rosters (built from the frames when the table is absent), and, when present, the section 11 aggregates
(`leaguedashptstats` SpeedDistance, a cross-check only). The section 15 injuries table starts in 2021-22 and never
overlaps the 2015-16 tracking era: it is listed by `validate.py` and nothing joins to it.

## Units

- **Player-stint**: a continuous on-court span of one player. The player's rows are sorted by period and frame; a
  new stint starts when the period changes or the game clock jumps by more than `CONTINUOUS_S` (5 s) between
  consecutive rows (a substitution leaves a hole in the clock, a stopped clock does not). A step between
  consecutive rows counts when its wall-clock dt (`ts_ms`, else the frame index at 25 Hz) is in (0, 1] s.
  Positions are smoothed by a centred 5-row mean that restarts at every uncounted step (a dropout of 1 to 5 s
  inside a stint is never bridged; the two rows at each end of a segment stay raw). Dead-ball movement counts
  because the wall clock runs: a known limit.
- **Player-game**: one row per player with frames in a game.

## Validation (`analytics/load/validate.py`), counts logged per game

Required frame columns; rows with a missing key or position dropped; duplicate `(frame, player_id)` rows dropped;
positions outside x in [-5, 99], y in [-5, 55] ft dropped; a step faster than 30 ft per s (within a continuous span)
flags the arriving row, which is dropped. The check is one pass, aimed at one-frame glitches: a glitch flags two rows
(the jump out and the jump back); a longer excursion loses only its first and last row and keeps its interior rows
(a limit, logged by the count). A player-game distance above 35,000 ft is flagged only. Nothing is interpolated.

## `player_stints`

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `player_id` | int32 | NBA player id (section 4) |
| `stint` | int32 | 0..k-1 within the player's game, in time order |
| `start_frame` | int32 | first frame of the span |
| `end_frame` | int32 | last frame of the span (a span bound of a descriptive unit; the table is never fed to a model) |
| `minutes` | float | game clock elapsed over the stint, minutes |
| `distance_ft` | float | sum of counted step lengths, feet |
| `avg_speed_ftps` | float | `distance_ft` over counted seconds; NaN with no counted step |
| `high_intensity_share` | float | counted seconds above 12 ft per s over counted seconds |
| `accelerations` | int | episodes where the 5-row smoothed speed rises faster than 6 ft per s^2 |
| `heldout` | bool | the game is held out |

## `player_games`

| column | type | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id |
| `game_date` | string | `YYYY-MM-DD` from the manifest |
| `player_id` | int32 | NBA player id |
| `team` | string | SportVU abbreviation from the roster (extra column: the schedule key) |
| `minutes`, `distance_ft` | float | sums over the player's stints (SAME-GAME: never features) |
| `avg_speed_ftps` | float | `distance_ft` over counted seconds of the game (SAME-GAME) |
| `decline_ftps` | float | TARGET y_decline: avg speed of the last stint of at least 1 minute minus the first; NaN with fewer than two such stints |
| `slope_ftps_per_stint` | float | TARGET: least-squares slope of avg speed on the index of those stints |
| `days_rest` | float | days since the team's previous manifest game, capped at 7; the team's first listed game takes 7 |
| `back_to_back` | bool | `days_rest <= 1` |
| `games_last_7` | float | the team's manifest games dated 7 days or fewer before the game (the game itself excluded) |
| `minutes_last_3` | float | the player's minutes over the team's previous three games (0 when absent from one; NaN when one is held out or not loaded) |
| `distance_last_3_ft` | float | same window, distance |
| `prev_decline_ftps` | float | the player's `decline_ftps` in the team's previous game; NaN when absent, held out or not loaded |
| `prev_avg_speed_ftps` | float | the player's `avg_speed_ftps` in the team's previous game; NaN likewise |
| `absent_next` | boolean, nullable | TARGET: the team's next manifest game has a loaded, non-held-out roster and the player is not on it; NA otherwise. Absence is not injury |
| `heldout` | bool | the game is held out |

`FEATURES` (analytics.load.features): `days_rest`, `back_to_back`, `games_last_7`, `minutes_last_3`,
`distance_last_3_ft`, `prev_decline_ftps`, `prev_avg_speed_ftps`. All come from the manifest schedule or from previous
games; a held-out game is never read for a feature. The same-game columns and the targets are `FORBIDDEN`:
`assert_no_leak` refuses a design matrix that carries one and `schema.assert_time_forward` runs on every feature frame.

## `load_summary`

| column | type | meaning |
| --- | --- | --- |
| `player_id` | int32 | NBA player id |
| `n_games` | int | non-held-out player-games with a `decline_ftps` |
| `n_stints` | int | non-held-out stints of the player (the gate) |
| `mean_decline` | float | mean `decline_ftps` over those games |
| `lo`, `hi` | float | 2.5 and 97.5 percentiles of bootstrap resamples of the player's games (NaN with one game) |
| `insufficient` | bool | fewer than 10 stints: never ranked |

## Rest buckets (candidate 1 and the EDA)

By `days_rest`: `back_to_back` (1 or fewer), `one_day` (2), `two_plus` (3 or more, including a team's first listed
game, which takes the cap of 7).

## Calibration of the absence baseline

Log loss, Brier and ECE carry bootstrap intervals by game. The calibration slope is one fit on all out-of-fold rows
(`analytics.eval.calibration_slope_fast`, the bounded Newton solution of the harness's logistic regression of the
target on logit p, which stays finite on near-separable probabilities) and carries no interval; at the fixture scale
the slope is meaningless anyway, and with one class in `absent_next` it is not computed.

## Limits of the targets

`decline_ftps` mixes fatigue with role and game state (closing lineups, a blowout fourth quarter, short defensive
stands late); one game is noisy, so the per-player summary is gated and carries an interval. `absent_next` reads
roster presence only: rest, a coach's decision, a suspension, a trade or a tracking miss read the same, and the
section 15 injuries table never overlaps the tracking era, so no injury label exists for these seasons. The
schedule is the manifest, so a game missing from the manifest overstates rest. No deep model: three baselines
(`analytics/load/train.py`): rest-bucket means with bootstrap intervals by game, ridge on `decline_ftps` against
the mean, logistic on `absent_next` against the base rate, all under grouped folds by game, held-out games scored
once.

Commands (`--fixtures` runs the synthetic season of `analytics.fixtures`, one league with fixed team ids and rosters,
sized by `--fixture-games`, `--fixture-possessions` and `--fixture-heldout-every` (`--n-games` and `--heldout-every`
are kept as aliases); fixture output is never a result. On that season every listed player plays every game, so
`absent_next` has one class there and the absence baseline reduces to the base rate):

    python -m analytics.load.validate --data "$DATA_ROOT" --out "$DATA_ROOT"
    python -m analytics.load.features --data "$DATA_ROOT" --out "$DATA_ROOT"
    python -m analytics.load.train --data "$DATA_ROOT" --report reports/load.md --out "$DATA_ROOT" --folds 5 --seed 0 --bootstrap 1000

## Frozen for the held-out run (freeze-v1)

Frozen from the code on branch track-fix and the committed reports/load.md. Every number below is out-of-fold on the
training games, from a run with `--skip-heldout`; no held-out number has been read.

**Selected models (one per pair):** `ridge on FEATURES` for decline_ftps and `logistic on FEATURES` for absent_next.

| setting (analytics/load/train.py, features.py, validate.py) | frozen value |
| --- | --- |
| ridge | median imputation, standardisation, `Ridge(alpha=1.0)` (`RIDGE_ALPHA`) |
| logistic | the same preprocessing, `LogisticRegression(C=1.0, max_iter=1000)` (`LOGISTIC_C`) |
| features | days_rest, back_to_back, games_last_7, minutes_last_3, distance_last_3_ft, prev_decline_ftps, prev_avg_speed_ftps; days_rest capped at 7 |
| stints and targets | a new stint at a period change or a game clock jump over 5 s; positions smoothed over 5 rows; steps with dt in (0, 1] s; decline over stints of at least 1 minute; ranking gated at 10 stints |
| cleaning | rows outside x -5 to 99 ft or y -5 to 55 ft and steps over 30 ft/s dropped and counted; duplicates by frame and player dropped; player-games over 35000 ft flagged, never dropped |

Rules the held-out run applies: per pair, the model is selected on its out-of-fold score (ridge when its MSE is at
most the training-fold mean's; logistic when its log loss is at most the base rate's); the constant is scored only
as a labelled floor. 5 grouped folds by `sportvu_game`, seed 0; 200 game bootstrap resamples, seed 0; rest buckets
back_to_back (1 day or fewer), one_day (2), two_plus (3 or more). Held out: the `heldout` flag or
`analytics.eval.heldout_mask` (every game at CLE and 12.16.2015.PHX.at.GSW; 17 games in the training run); a
held-out game is never read for a feature window. The selected models, refit on every training row, score the
held-out player-games once (point estimates) next to their floors.

Out-of-fold headline (604 games): decline MSE ridge 0.4929 [0.4740, 0.5178] against the mean 0.5086
[0.4884, 0.5352], paired -0.0157 [-0.0197, -0.0118]; absence log loss logistic 0.2764 [0.2644, 0.2883] against
the base rate 0.3244 [0.3118, 0.3379], paired -0.0481 [-0.0553, -0.0417]; logistic Brier 0.0777 [0.0742, 0.0815],
calibration slope 0.9908 (one fit, no interval).

Stage 8 command:

    python3 run_all.py --data "$DATA_ROOT" --ext xfg rebounds matchups gravity ghosting embeddings epv decisions lineups winprob load --continue-on-error

run_all.step_argv gives `analytics.load.train` the argv `--data DATA_ROOT --seed 0 --report reports/load.md`;
validate and features get `--data DATA_ROOT --seed 0`. OUT_ROOT defaults to DATA_ROOT on real data
(`out_root_for`, set before freeze-v1), so the tables are written there; the held-out scores are in the report.
