# Cross-lane interfaces (analytics/interfaces.py, lane I6)

An interface is a parquet table one extension lane writes under `OUT_ROOT` and other lanes read. The registry
`analytics.interfaces.INTERFACES` records the five of them; `check_frame(name, df)` raises an `AssertionError`
naming every missing required column and every column of the wrong dtype class; `interface_path(out_root,
name, season, game)` gives the file the producer writes; `find_outputs(out_root, name)` lists what is actually
there; `discover_lanes()` lists the extension packages that are complete enough to test.

Dtype classes: `string` (object or pandas string), `int` (any integer dtype, nullable `Int64` included), `float`
(float, or an integer dtype), `bool` (bool or nullable boolean). Optional columns are checked for dtype when
present and noted when absent. Duplicate rows on the key are a note, not a failure (consumers de-duplicate on
their key; a producer may write out-of-fold rows per fold).

## Fold convention

Every interface or lane table that carries a `fold` column uses one convention (`analytics.interfaces.FOLD_*`):
`fold >= 0` is an out-of-fold row (scored by the model of the grouped CV fold that held it out), `fold == -1` is a
held-out game scored once by the model refit on every training row, and `fold == -2` is an in-sample row (scored by
a model fitted on it; descriptive tables only, never a reported score). `check_frame(name, df, path=...)` returns a
note for `fold == -2` rows in a file outside `heldout/` (a training folder) and for any fold below -2, so a consumer
sees which rows are not out-of-fold scores.

## Held-out isolation check

`analytics.fixtures.write_fixture_root` puts the real held-out id `12.16.2015.PHX.at.GSW` in its first held-out
slot. The shared DATA_ROOT part asserts, for every lane, that no file named after a held-out token (that id and
every token of `sportvu/splits.json`) sits outside `heldout/` in the lane's output folders, that any rows naming
such a game in a single-file table outside `heldout/` carry `heldout == True` and `fold == -1`, and that no
"Training games" line of the lane's report under `OUT_ROOT/reports/` names it. Nothing depends on a manifest.

Every number produced by these lanes so far is fixture output: no real SportVU, play-by-play, NBA.com or odds
row is in the repository, and nothing below is a result.

## Decisions and assumptions

The paths and column sets below are the coordinator's decisions for the build, reconciled with what the
producer lanes document (docs/contract/<ext>.md) and what the consumers code against. Where the two differ,
the decision wins and the entry's `note` in `analytics/interfaces.py` says so.

- `xfg_predictions`: any `OUT_ROOT/analytics/xfg/<season>/*.parquet` with `sportvu_game, event, player_id,
  p_selected, fold`; `analytics/xfg/train.py` writes `xfg_predictions.parquet` with `p_location, p_contest,
  made, heldout` as well (optional here). `xfg_features.parquet` is the lane's feature frame and is skipped.
  decisions joins `p_selected` by `(sportvu_game, event)` and matchups reads it for defender credit; gravity and
  epv refit `analytics.train_xfg` in process and never read the file, so they are not consumers.
- `epv_states`: `OUT_ROOT/analytics/epv/<season>/<game>.parquet` (held-out under `heldout/<season>/`) with
  `sportvu_game, possession, frame, epv_points`. `epv_possessions.parquet` in the same folder is the lane's
  possession table and is skipped. The lane's `states_1hz` output (`analytics/states_1hz/`, contract section
  20) is an input table, not this interface.
- `actions_detected`: `OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` with
  `sportvu_game, possession, action_id, action_type, start_frame, end_frame, ball_handler_id, screener_id`
  (nullable), `label_source`. Hand labels are read only through `analytics.labels.read_labels` (labeller_pass
  1, label_source `hand`). The lane's `action_candidates/` and `features/` folders are not matched.
- `coverage_predictions`: as documented in docs/contract/coverage.md (`coverage_predictions.parquet` per
  season with `coverage_label` nullable, `p_switch` to `p_over`, `predicted`, `fold`, `heldout`). playcalling
  reads `predicted` and the `p_<class>` columns. `coverage_features.parquet` and `coverage_team_rates.parquet`
  are not matched by the glob.
- `player_rapm`: as documented in docs/contract/lineups.md: `season, player_id, rapm, lo, hi, fit_seasons`.
  The decision was no `as_of` and no `team_id` (betting treats its RAPM features as absent), but
  `analytics/lineups/train.py` now also writes `team_id`, `as_of` and `in_sample`, which betting joins when
  present; they are optional columns here.

## xfg_predictions

Producer `xfg`; consumers `decisions`, `matchups` (gravity and epv refit `analytics.train_xfg` in process and never read the file). Path `OUT_ROOT/analytics/xfg/<season>/*.parquet`
(`xfg_predictions.parquet` from the trainer; one file per season, held-out rows flagged, not separated).
Key `(sportvu_game, event)`.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id (contract section 16) |
| `event` | int | the shot's play-by-play event id (section 10) |
| `player_id` | int, nullable | shooter id |
| `p_selected` | float | the selected candidate's probability (the shot-quality column consumers join) |
| `fold` | int | fold convention: >= 0 out-of-fold, -1 held-out scored once, -2 in-sample |
| `p_location` | float, optional | the location-only candidate's out-of-fold make probability |
| `p_contest` | float, optional | the contest logistic candidate's out-of-fold make probability |
| `made` | bool, optional | the outcome (never a feature downstream) |
| `heldout` | bool, optional | the row's game is held out |

## epv_states

Producer `epv`; consumers `decisions`, `playcalling`. Path `OUT_ROOT/analytics/epv/<season>/<game>.parquet`
(held-out under `heldout/<season>/`). Key `(sportvu_game, possession, frame)`.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | game id |
| `possession` | int | section 19 possession |
| `frame` | int | the 1 Hz state's frame (section 20) |
| `epv_points` | float | expected points of the possession from this state, out of fold or held-out scored once |
| `heldout` | bool, optional | the game is held out |
| `fold` | int, optional | fold convention: >= 0 out-of-fold, -1 held-out, -2 in-sample |

## actions_detected

Producer `actions`; consumers `coverage`, `playcalling`. Path
`OUT_ROOT/analytics/actions/<season>/actions_detected/<game>.parquet` (held-out under `heldout/<season>/`).
Key `(sportvu_game, possession, action_id)`, unique only per `label_source`: a game may carry hand rows and
detected rows with the same ids, and both consumers drop the detected rows of a game that has hand labels.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | game id |
| `possession` | int | section 19 possession |
| `action_id` | int | id inside the game |
| `action_type` | string | `ball_screen`, `drive`, `isolation` (the detector's classes) or another ACTION_TYPES value |
| `start_frame`, `end_frame` | int | the action's span in 25 Hz frame ids |
| `ball_handler_id` | int | the handler |
| `screener_id` | int, nullable | the screener for a ball screen |
| `label_source` | string | `model` for detected rows (`hand` for labels passed through) |
| `labeller_pass` | int, optional (nullable) | the labelling pass of a hand label (1 for the first pass); null for detected rows |
| `heldout` | bool, optional | the game is held out |

## coverage_predictions

Producer `coverage`; consumer `playcalling`. Path `OUT_ROOT/analytics/coverage/<season>/coverage_predictions.parquet`
(held-out under `heldout/<season>/`). Key `(sportvu_game, possession, action_id)`.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | game id |
| `possession` | int | possession |
| `action_id` | int | the ball screen's action id (actions interface) |
| `predicted` | string | the selected model's class, one of `switch, drop, hedge_show, blitz_trap, ice, under, over` |
| `p_<class>` | float, optional | class probabilities in [0, 1], summing to 1 |
| `coverage_label` | string, optional (nullable) | the hand label; null for an unlabelled or detected screen |
| `heldout` | bool, optional | the game is held out |
| `fold` | int, optional | fold convention: >= 0 out-of-fold, -1 held-out, -2 in-sample |

## player_rapm

Producer `lineups`; consumer `betting` (matchups never reads it). Path `OUT_ROOT/analytics/lineups/<season>/player_rapm.parquet`
(one file per season). Key `(season, player_id)`.

| column | dtype | meaning |
| --- | --- | --- |
| `season` | string | `YYYY-YY` |
| `player_id` | int | NBA.com player id |
| `rapm` | float | regularised adjusted plus-minus per 100 possessions from the fit on seasons up to and including `season` |
| `lo`, `hi` | float | bootstrap-by-game interval |
| `fit_seasons` | string, optional | comma-joined seasons of the fit |
| `team_id` | int, optional (nullable) | the player's team in that season (written by lineups today) |
| `as_of` | string, optional | date the value is known (written by lineups today; betting uses values dated before each game only) |
| `in_sample` | bool, optional | the fit included this season |

## Discovery convention

`discover_lanes()` returns every `analytics/<ext>/` directory that holds `__init__.py`, `validate.py`,
`features.py` and `train.py`. `missing_lanes()` gives the reason for every expected lane not discovered
(`EXPECTED_LANES`: xfg, rebounds, matchups, gravity, ghosting, embeddings, epv, actions, coverage, decisions,
playcalling, lineups, winprob, load, betting). Every step module exposes `main(argv)` and accepts `--fixtures`;
the train module carries `refuse_heldout` or `assert_time_forward`; a fixture generator is `tests/_fx_<ext>.py`
or the shared `analytics/fixtures.py`.

## Running

    python3 -m pytest -q tests/test_integration.py                            # fast part (per present lane:
                                                                              # imports, main(argv), guard,
                                                                              # fixture module, validate --fixtures)
    INTEGRATION_SLOW=1 python3 -m pytest -q tests/test_integration.py -m slow # features and train --fixtures
                                                                              # into one OUT_ROOT, check_frame
                                                                              # on every interface output,
                                                                              # manifest held-out check

    INTEGRATION_SLOW=1 INTEGRATION_SHARED=1 python3 -m pytest -q tests/test_integration.py -m slow
                                                                              # also: one DATA_ROOT from
                                                                              # `python -m analytics.fixtures`,
                                                                              # then validate, features, train
                                                                              # with --data per lane in
                                                                              # dependency order, interface
                                                                              # checks on what they wrote

The slow part also runs `python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT` once and
asserts every interface file exists under that OUT_ROOT and passes `check_frame` (run_all.py passes `--out`
and `--seed` to every step).

A lane that is not present is a skip with the reason. Every subprocess has a timeout (120 s for validate,
600 s per features or train step with --fixtures, 300 s per step on the shared root). The fast part runs its
subprocesses four at a time (`WORKERS`). Results of the latest run: reports/integration.md.
