# Integration report (lane I6)

Latest run: 2026-11-05 08:46 UTC. Every number here is fixture output (synthetic data from tests/_fx_<ext>.py or analytics/fixtures.py), not a result. Produced by `python3 -m pytest -q tests/test_integration.py` (fast part, 182 s of test time); the slow part has not been run in this report.

Lanes present (15 of 15): actions, betting, coverage, decisions, embeddings, epv, ghosting, gravity, lineups, load, matchups, playcalling, rebounds, winprob, xfg

Lanes missing (0): none

## Fast part, per lane

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

Fast part totals: 63 passed, 0 failed, 0 skipped (skips are absent lanes).

## Failures

None in the latest run.

## Registry checks

- test_registry_is_consistent: pass
- test_check_frame_reports_missing_and_mistyped: pass
- test_discovery_lists_present_lanes: pass

## Run history

- run 1 (fast only): 62 passed, 1 failed (rebounds imports+main), 15 of 15 lanes present, 196 s
- run 2 (fast only, subprocesses 4 at a time): 63 passed, 0 failed, 15 of 15 lanes present, 183 s wall while the slow part and other lanes shared the 4 cores

## Assumptions

The interface columns of epv_states, actions_detected and coverage_predictions are taken from the consumers' code because the producer lanes have not documented them (docs/contract/interfaces.md, Assumptions). The slow part passes `--folds 3 --seed 0 --bootstrap 20 --out OUT_ROOT --report OUT_ROOT/reports/<ext>.md` to every trainer, as the standard CLI promises; a trainer that rejects one of these fails the step.
