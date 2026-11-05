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

Every number produced by these lanes so far is fixture output: no real SportVU, play-by-play, NBA.com or odds
row is in the repository, and nothing below is a result.

## Assumptions

Where the producer has documented its table (docs/contract/<ext>.md, or the docstring and constants of its
train module while the contract page is pending), the entry matches that: `xfg_predictions` matches
`analytics/xfg/train.py` `PREDICTION_COLUMNS`; `player_rapm` matches the `analytics/lineups/train.py`
docstring. Where it has not (no `docs/contract/{epv,actions,coverage}.md` and no train module at the time of
writing), the entry records the columns the consumers already code against:

- `epv_states`: decisions and playcalling read `OUT_ROOT/analytics/epv/<season>/<game>.parquet` (held-out
  games under `heldout/<season>/`) with `sportvu_game, possession, frame, epv_points`. The lane's `states_1hz`
  output (`analytics/states_1hz/`, contract section 20) is an input table, not this interface.
- `actions_detected`: coverage (`ACTION_COLUMNS`) and playcalling read `OUT_ROOT/analytics/actions/<season>/
  <game>.parquet` with the hand-label columns; `action_id` is optional and defaults to the row's position in
  the file; `screener_id` is nullable. The lane's feature frames under `analytics/actions/<season>/features/`
  are skipped by `find_outputs`.
- `coverage_predictions`: playcalling reads `OUT_ROOT/analytics/coverage/**/*.parquet` with
  `sportvu_game, possession, action_id, predicted` and `p_<class>` columns in [0, 1]. The file name
  `coverage_predictions.parquet` per season is the assumption; `coverage_features.parquet` (the lane's feature
  frame, already written) is not matched by the glob.
- `xfg_predictions`: gravity and epv use the location-only baseline of `analytics.train_xfg` in their own code
  today; they are listed as consumers of the lane's shot-quality column (`p_selected`) per the registry note
  in `analytics/registry.py`, and decisions joins `p_selected` by `(sportvu_game, event)`.
- `player_rapm`: betting reads `team_id` and `as_of` when present and keeps its RAPM features null otherwise;
  matchups has no RAPM read in its code yet. Both are optional columns here.

When a producer lane documents something different, the registry entry changes to match; the consumers are
told through reports/integration.md.

## xfg_predictions

Producer `xfg`; consumers `decisions`, `gravity`, `epv`. Path `OUT_ROOT/analytics/xfg/<season>/xfg_predictions.parquet`
(one file per season; held-out rows flagged, not separated). Key `(sportvu_game, event)`.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | Step 1 game id (contract section 16) |
| `event` | int | the shot's play-by-play event id (section 10) |
| `player_id` | int, optional | shooter id |
| `p_location` | float, optional | the location-only candidate's out-of-fold make probability |
| `p_contest` | float, optional | the contest logistic candidate's out-of-fold make probability |
| `p_selected` | float | the selected candidate's probability (the shot-quality column consumers join) |
| `made` | bool | the outcome (never a feature downstream) |
| `fold` | int | grouped CV fold; -1 for held-out rows scored once |
| `heldout` | bool | the row's game is held out |

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
| `fold` | int, optional | grouped CV fold |

## actions_detected

Producer `actions`; consumers `coverage`, `playcalling`. Path `OUT_ROOT/analytics/actions/<season>/<game>.parquet`
(held-out under `heldout/<season>/`). Key `(sportvu_game, possession, start_frame)`.

| column | dtype | meaning |
| --- | --- | --- |
| `sportvu_game` | string | game id |
| `possession` | int | section 19 possession |
| `action_type` | string | `ball_screen`, `drive`, `isolation` (the detector's classes) or another ACTION_TYPES value |
| `start_frame`, `end_frame` | int | the action's span in 25 Hz frame ids |
| `ball_handler_id` | int | the handler |
| `screener_id` | int, optional (nullable) | the screener for a ball screen |
| `label_source` | string | `model` for detected rows (`hand` for labels passed through) |
| `action_id` | int, optional | id inside the game; defaults to the row's position when absent |
| `labeller_pass` | string, optional | carried from hand labels |
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
| `p_<class>` | float, optional | class probabilities in [0, 1] |
| `heldout` | bool, optional | the game is held out |
| `fold` | int, optional | grouped CV fold |

## player_rapm

Producer `lineups`; consumers `matchups`, `betting`. Path `OUT_ROOT/analytics/lineups/<season>/player_rapm.parquet`
(one file per season). Key `(season, player_id)`.

| column | dtype | meaning |
| --- | --- | --- |
| `season` | string | `YYYY-YY` |
| `player_id` | int | NBA.com player id |
| `rapm` | float | regularised adjusted plus-minus per 100 possessions from the fit on seasons up to and including `season` |
| `lo`, `hi` | float | bootstrap-by-game interval |
| `fit_seasons` | string, optional | comma-joined seasons of the fit |
| `team_id` | int, optional | the player's team (betting sums by team when present) |
| `as_of` | string, optional | date the value is known (betting uses values dated before each game only) |

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

A lane that is not present is a skip with the reason. Every subprocess has a timeout (120 s for validate,
600 s per features or train step with --fixtures, 300 s per step on the shared root). The fast part runs its
subprocesses four at a time (`WORKERS`). Results of the latest run: reports/integration.md.
