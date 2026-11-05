# Integration report (lane I6)

Latest run: 2026-11-05 08:54 UTC. Every number here is fixture output (synthetic data from tests/_fx_<ext>.py or analytics/fixtures.py), not a result. Produced by `python3 -m pytest -q tests/test_integration.py` (fast part, 182 s of test time) and the slow part with INTEGRATION_SLOW=1 (199 s).

Lanes present (15 of 15): actions, betting, coverage, decisions, embeddings, epv, ghosting, gravity, lineups, load, matchups, playcalling, rebounds, winprob, xfg

Lanes missing (0): none

## Fast part, per lane

| lane | imports+main | guard | fixture module | validate --fixtures | features+train --fixtures |
| --- | --- | --- | --- | --- | --- |
| `xfg` | pass | pass | pass | pass | pass |
| `rebounds` | pass | pass | pass | pass | pass |
| `matchups` | pass | pass | pass | pass | pass |
| `gravity` | pass | pass | pass | pass | pass |
| `ghosting` | pass | pass | pass | pass | pass |
| `embeddings` | pass | pass | pass | pass | FAIL |
| `epv` | pass | pass | pass | pass | pass |
| `actions` | pass | pass | pass | pass | pass |
| `coverage` | pass | pass | pass | pass | pass |
| `decisions` | pass | pass | pass | pass | pass |
| `playcalling` | pass | pass | pass | pass | pass |
| `lineups` | pass | pass | pass | pass | pass |
| `winprob` | pass | pass | pass | pass | pass |
| `load` | pass | pass | pass | pass | pass |
| `betting` | pass | pass | pass | pass | pass |

Fast part totals: 63 passed, 0 failed, 0 skipped (skips are absent lanes).

## Slow part

- `xfg_predictions` (producer xfg, consumers decisions, gravity, epv): pass
- `epv_states` (producer epv, consumers decisions, playcalling): skip: epv wrote no epv_states under OUT_ROOT/epv with --fixtures (pattern analytics/epv/<season>/<game>.parquet)
- `actions_detected` (producer actions, consumers coverage, playcalling): FAIL
- `coverage_predictions` (producer coverage, consumers playcalling): pass
- `player_rapm` (producer lineups, consumers matchups, betting): pass
- held-out tokens in training manifests: skip: no trainer wrote a manifest under OUT_ROOT with --fixtures

## Shared DATA_ROOT part (INTEGRATION_SHARED=1)

One DATA_ROOT from `python -m analytics.fixtures --out DATA_ROOT --seed 0 --games 6 --heldout-every 3` (lane I1), then validate, features and train with `--data DATA_ROOT --out OUT_ROOT` per lane in dependency order.

- fixture root written: skip: shared-root part: set INTEGRATION_SHARED=1

## run_all.py part

`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT --bootstrap 20` once (lane I5), then every interface file must exist under that OUT_ROOT and pass check_frame.

- run_all exit 0: n/a
- `xfg_predictions` after run_all: n/a
- `epv_states` after run_all: n/a
- `actions_detected` after run_all: n/a
- `coverage_predictions` after run_all: n/a
- `player_rapm` after run_all: n/a

## Failures

- `embeddings` features+train:
    - `AssertionError: python -m analytics.embeddings.train --fixtures --out OUT_ROOT exited 2:`
    - `train.py: error: unrecognized arguments: --folds 3`
    - `assert 2 == 0`
- `actions` interface actions_detected:
    - `AssertionError: /tmp/pytest-of-root/pytest-200/out_root0/analytics/actions/2015-16/action_candidates/01.18.2016.HLX.at.LRD.parquet: interface actions_detected (producer actions, consumers coverage, playcalling): missing `
    - `E       AssertionError: /tmp/pytest-of-root/pytest-200/out_root0/analytics/actions/2015-16/action_candidates/01.18.2016.HLX.at.LRD.parquet: interface actions_detected (producer actions, consumers coverage, playcalling): `
    - `analytics/interfaces.py:223: AssertionError`

## Registry checks

- test_registry_is_consistent: pass
- test_check_frame_reports_missing_and_mistyped: pass
- test_discovery_lists_present_lanes: pass

## Run history

- run 1 (fast only): 62 passed, 1 failed (rebounds imports+main), 15 of 15 lanes present, 196 s
- run 2 (fast only, subprocesses 4 at a time): 63 passed, 0 failed, 15 of 15 lanes present, 183 s wall while the slow part and other lanes shared the 4 cores
- run 3 (slow part, 15 lanes, 200 s): 17 passed, 2 failed (embeddings train rejects --folds; actions_detected files not found under the decided path, action_candidates matched the old glob), epv wrote no epv_states; the registry was then updated to the coordinator's path decisions

## Assumptions

The interface columns of epv_states, actions_detected and coverage_predictions are taken from the consumers' code because the producer lanes have not documented them (docs/contract/interfaces.md, Assumptions). The slow part passes `--folds 3 --seed 0 --bootstrap 20 --out OUT_ROOT --report OUT_ROOT/reports/<ext>.md` to every trainer, as the standard CLI promises; a trainer that rejects one of these fails the step.
