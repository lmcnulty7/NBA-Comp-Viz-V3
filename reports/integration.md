# Integration report (lane I6)

Latest run: 2026-11-07 08:43 UTC. **Every number and outcome here is fixture output** (synthetic data from analytics/fixtures.py or tests/_fx_<ext>.py), not a result: no real SportVU, play-by-play, NBA.com or odds row is involved. Produced by `python3 -m pytest -q tests/test_integration.py -m "not slow"` (fixture output: 18 s wall) and `python3 -m pytest -q tests/test_integration.py -m slow` (fixture output: 361 s wall). Paths are relative to the run's OUT_ROOT or DATA_ROOT.

Fixture output, test counts: fast part 65 passed, 0 failed, 0 errors, 0 skipped; slow part 61 passed, 1 failed, 0 errors, 0 skipped.

Lanes present (15 of 15): actions, betting, coverage, decisions, embeddings, epv, ghosting, gravity, lineups, load, matchups, playcalling, rebounds, winprob, xfg

## Fast part, per lane (fixture output)

| lane | imports+main | guard | fixture module | validate --fixtures |
| --- | --- | --- | --- | --- |
| `xfg` | pass | pass | pass | pass |
| `rebounds` | pass | pass | pass | pass |
| `matchups` | pass | pass | pass | pass |
| `gravity` | pass | pass | pass | pass |
| `ghosting` | pass | pass | pass | pass |
| `embeddings` | pass | pass | pass | pass |
| `epv` | pass | pass | pass | pass |
| `actions` | pass | pass | pass | pass |
| `coverage` | pass | pass | pass | pass |
| `decisions` | pass | pass | pass | pass |
| `playcalling` | pass | pass | pass | pass |
| `lineups` | pass | pass | pass | pass |
| `winprob` | pass | pass | pass | pass |
| `load` | pass | pass | pass | pass |
| `betting` | pass | pass | pass | pass |

Registry checks: test_registry_is_consistent pass, test_check_frame_reports_missing_and_mistyped pass, test_check_frame_fold_convention_notes pass, test_consumers_match_the_review pass, test_discovery_lists_present_lanes pass

## Slow part 1: features and train --fixtures into one OUT_ROOT (fixture output)

| lane | features+train --fixtures --out |
| --- | --- |
| `xfg` | pass |
| `rebounds` | pass |
| `matchups` | pass |
| `gravity` | pass |
| `ghosting` | pass |
| `embeddings` | pass |
| `epv` | pass |
| `actions` | pass |
| `coverage` | pass |
| `decisions` | pass |
| `playcalling` | pass |
| `lineups` | pass |
| `winprob` | pass |
| `load` | pass |
| `betting` | pass |

- `xfg_predictions` (producer xfg, consumers decisions, matchups): pass; 1 file(s): `OUT_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet`
- `epv_states` (producer epv, consumers decisions, playcalling): pass; 8 file(s): `OUT_ROOT/analytics/epv/2015-16/02.10.2016.QCY.at.FRS.parquet`, `OUT_ROOT/analytics/epv/2015-16/03.08.2016.GRN.at.HLX.parquet`, `OUT_ROOT/analytics/epv/2015-16/11.28.2015.HLX.at.FRS.parquet`, `OUT_ROOT/analytics/epv/2015-16/12.01.2015.DUL.at.HLX.parquet` ...
- `actions_detected` (producer actions, consumers coverage, playcalling): pass; 6 file(s): `OUT_ROOT/analytics/actions/2015-16/actions_detected/02.10.2016.QCY.at.FRS.parquet`, `OUT_ROOT/analytics/actions/2015-16/actions_detected/03.08.2016.GRN.at.HLX.parquet`, `OUT_ROOT/analytics/actions/2015-16/actions_detected/12.01.2015.DUL.at.HLX.parquet`, `OUT_ROOT/analytics/actions/2015-16/actions_detected/12.10.2015.GRN.at.DUL.parquet` ...
- `coverage_predictions` (producer coverage, consumers playcalling): pass; 1 file(s): `OUT_ROOT/analytics/coverage/2015-16/coverage_predictions.parquet`
- `player_rapm` (producer lineups, consumers betting): pass; 4 file(s): `OUT_ROOT/analytics/lineups/2020-21/player_rapm.parquet`, `OUT_ROOT/analytics/lineups/2021-22/player_rapm.parquet`, `OUT_ROOT/analytics/lineups/2021-22/player_rapm_prior.parquet`, `OUT_ROOT/analytics/lineups/2022-23/player_rapm.parquet`

## Slow part 2: shared DATA_ROOT (fixture output)

One DATA_ROOT from `python -m analytics.fixtures --out DATA_ROOT --seed 0 --games 6 --heldout-every 3`, then validate, features and train with `--data DATA_ROOT --out OUT_ROOT` per lane in dependency order. The held-out column checks that the real held-out id the fixture root puts in its first held-out slot (`12.16.2015.PHX.at.GSW`) sits only under `heldout/` in the lane's output folders (or in single-file tables with heldout True and fold -1), that no other training-folder file names it, and that no report lists it as a training game. No manifest file is read.

- fixture root written, held-out id under `heldout/`: pass

| lane | validate+features+train --data | held-out isolation | files of the held-out id under heldout/ | training-folder tables with its rows flagged |
| --- | --- | --- | --- | --- |
| `xfg` | pass | pass | 0 | 2 |
| `rebounds` | pass | pass | 2 | 1 |
| `matchups` | pass | pass | 1 | 1 |
| `gravity` | pass | FAIL | 0 | 3 |
| `ghosting` | pass | pass | 3 | 0 |
| `embeddings` | pass | pass | 0 | 0 |
| `epv` | pass | pass | 3 | 0 |
| `actions` | pass | pass | 4 | 1 |
| `coverage` | pass | pass | 2 | 0 |
| `decisions` | pass | pass | 1 | 0 |
| `playcalling` | pass | pass | 1 | 0 |
| `lineups` | pass | pass | 0 | 0 |
| `winprob` | pass | pass | 0 | 0 |
| `load` | pass | pass | 2 | 1 |
| `betting` | pass | pass | 0 | 0 |

- `xfg_predictions` under the shared root: pass; 1 file(s), 0 under `heldout/`
- `epv_states` under the shared root: pass; 6 file(s), 2 under `heldout/`
- `actions_detected` under the shared root: pass; 6 file(s), 2 under `heldout/`
- `coverage_predictions` under the shared root: pass; 2 file(s), 1 under `heldout/`
- `player_rapm` under the shared root: pass; 1 file(s), 0 under `heldout/`

## Slow part 3: run_all.py (fixture output)

`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT --bootstrap 20` once, then every interface file must exist under that OUT_ROOT and pass check_frame.

- run_all exit 0: pass
- `xfg_predictions` after run_all: pass; 1 file(s)
- `epv_states` after run_all: pass; 4 file(s)
- `actions_detected` after run_all: pass; 4 file(s)
- `coverage_predictions` after run_all: pass; 2 file(s)
- `player_rapm` after run_all: pass; 1 file(s)

## Failures (fixture output)

- `test_shared_root_heldout_isolation[gravity]` FAIL:
    - `AssertionError: held-out game 12.16.2015.PHX.at.GSW in gravity training outputs:`
    - `OUT_ROOT/analytics/gravity/validate.json: names the held-out game outside a held-out record at /per_game/5/sportvu_game`
    - `assert not ['OUT_ROOT/analytics/gravity/validate.json: names the held-out game outside a held-out record at /per_game/5/sportvu_game']`


## Lane findings (fixture output, not results)

- `gravity`: `python -m analytics.gravity.validate --data DATA_ROOT --out OUT_ROOT` on the shared fixture root writes
  `OUT_ROOT/analytics/gravity/validate.json` whose `per_game` record for the held-out game `12.16.2015.PHX.at.GSW`
  carries no `heldout: true` flag, in a folder outside `heldout/`. The other lanes' validation summaries (rebounds,
  matchups, actions, load) flag that record `heldout: true`. Fix in the lane: add the flag to the per-game record
  (or write the held-out game's record under `heldout/`). Reproduce with
  `python3 -m pytest -q tests/test_integration.py -m slow -k "heldout_isolation and gravity"`.
- Earlier in this pass, `analytics/epv/features.py` and `analytics/epv/train.py` failed to import
  (`cannot import name 'season_of' from 'analytics.epv'`) while the epv lane was being edited; the final runs above
  import and pass.
