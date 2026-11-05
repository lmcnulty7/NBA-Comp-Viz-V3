# Integration report (lane I6)

Latest run: 2026-11-05 09:38 UTC. Every number here is fixture output (synthetic data from tests/_fx_<ext>.py or analytics/fixtures.py), not a result. Produced by `python3 -m pytest -q tests/test_integration.py` (fast part, 575 s of test time) and the slow part with INTEGRATION_SLOW=1 (575 s).

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
| `embeddings` | pass | pass | pass | pass | pass |
| `epv` | pass | pass | pass | pass | pass |
| `actions` | pass | pass | pass | pass | pass |
| `coverage` | pass | pass | pass | pass | pass |
| `decisions` | pass | pass | pass | pass | pass |
| `playcalling` | pass | pass | pass | pass | pass |
| `lineups` | pass | pass | pass | pass | pass |
| `winprob` | pass | pass | pass | pass | pass |
| `load` | pass | pass | pass | pass | pass |
| `betting` | pass | pass | pass | pass | pass |

Totals over both parts in this report: 108 passed, 2 failed, 1 skipped (skips are absent lanes and gated parts).

## Slow part

- `xfg_predictions` (producer xfg, consumers decisions, gravity, epv): pass
- `epv_states` (producer epv, consumers decisions, playcalling): pass
- `actions_detected` (producer actions, consumers coverage, playcalling): pass
- `coverage_predictions` (producer coverage, consumers playcalling): pass
- `player_rapm` (producer lineups, consumers matchups, betting): pass
- held-out tokens in training manifests: skip: no trainer wrote a manifest under OUT_ROOT with --fixtures

## Shared DATA_ROOT part (INTEGRATION_SHARED=1)

One DATA_ROOT from `python -m analytics.fixtures --out DATA_ROOT --seed 0 --games 6 --heldout-every 3` (lane I1), then validate, features and train with `--data DATA_ROOT --out OUT_ROOT` per lane in dependency order.

- fixture root written: pass

| lane | validate+features+train --data |
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

- `xfg_predictions` under the shared root: pass
- `epv_states` under the shared root: pass
- `actions_detected` under the shared root: pass
- `coverage_predictions` under the shared root: pass
- `player_rapm` under the shared root: pass

## run_all.py part

`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT --bootstrap 20` once (lane I5), then every interface file must exist under that OUT_ROOT and pass check_frame.

- run_all exit 0: FAIL: AssertionError: run_all.py --fixtures --all --continue-on-error exited 1:
- `xfg_predictions` after run_all: pass
- `epv_states` after run_all: pass
- `actions_detected` after run_all: pass
- `coverage_predictions` after run_all: FAIL: AssertionError: run_all.py --fixtures --all wrote no coverage_predictions under OUT_ROOT (pattern analytics/coverage/<season>/coverage_predictions.parquet, producer coverage)
- `player_rapm` after run_all: pass

## Failures

- `run_all (I5)` run_all --fixtures --all:
    - `betting      train     failed     0.75s  /tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/bt/run_all_out0/logs/run_all/betting_train.log  ValueError: betting.features: feature 'hom`
    - `AssertionError: run_all.py --fixtures --all --continue-on-error exited 1:`
    - `>       assert r.returncode == 0, "run_all.py --fixtures --all --continue-on-error exited %d:\n%s" % (r.returncode, _tail(r.stderr) or _tail(r.stdout))`
- `coverage` run_all interface coverage_predictions:
    - `AssertionError: run_all.py --fixtures --all wrote no coverage_predictions under OUT_ROOT (pattern analytics/coverage/<season>/coverage_predictions.parquet, producer coverage)`

## Registry checks

- test_registry_is_consistent: pass
- test_check_frame_reports_missing_and_mistyped: pass
- test_discovery_lists_present_lanes: pass

## Run history

- run 1 (fast only): 62 passed, 1 failed (rebounds imports+main), 15 of 15 lanes present, 196 s
- run 2 (fast only, subprocesses 4 at a time): 63 passed, 0 failed, 15 of 15 lanes present, 183 s wall while the slow part and other lanes shared the 4 cores
- run 3 (slow part, 15 lanes, 200 s): 17 passed, 2 failed (embeddings train rejects --folds; actions_detected files not found under the decided path, action_candidates matched the old glob), epv wrote no epv_states; the registry was then updated to the coordinator's path decisions
- run 4 (fast and slow, registry on the decided paths, 246 s): 82 passed, 1 failed (betting train: NaN in LogisticRegression when lineups' player_rapm with as_of sits in the same OUT_ROOT), 28 skipped (shared-root part off, run_all part waiting on I5's --out fix, no manifests written); all five interface files found and valid, embeddings accepts --folds now
- run 5 (fast, slow and shared DATA_ROOT parts, 438 s): 102 passed, 2 failed, 7 skipped. Betting passes again. The two failures were the registry's own error (labeller_pass typed string; the actions lane writes the pass number as int16), fixed in analytics/interfaces.py after the run; every interface file of run 5 passes check_frame with the corrected registry. All 15 lanes ran validate, features and train with --data on the shared fixture root and exited 0. Skips: run_all part (run_all.py still omits --out on train), no training manifests written.
- run 7, final (fast, slow, shared DATA_ROOT and run_all parts, 576 s): 108 passed, 2 failed, 1 skipped. Fast part 63 of 63; slow part: every lane's features and train --fixtures --out exit 0 and all five interface files pass check_frame; shared root: all 15 lanes exit 0 on validate, features and train --data and the interface files pass. run_all.py --fixtures --all --continue-on-error --out exits 1: coverage validate rejects run_all's --seed (analytics/coverage/validate.py argparse, so playcalling is skipped as its dependent and coverage_predictions is missing under run_all's OUT_ROOT), winprob train fails its own cut-play-by-play self-check on run_all's fixture root (spread_home changes for game 0022000017), betting train trips its leak guard on run_all's fixture root (home_pf_std correlates 0.999 with home_margin). The other 12 lanes pass every run_all step; xfg_predictions, epv_states, actions_detected and player_rapm exist under run_all's OUT_ROOT and pass check_frame. Skip: no trainer writes a training manifest, so the held-out token check has nothing to read.

## Assumptions

The interface columns of epv_states, actions_detected and coverage_predictions are taken from the consumers' code because the producer lanes have not documented them (docs/contract/interfaces.md, Assumptions). The slow part passes `--folds 3 --seed 0 --bootstrap 20 --out OUT_ROOT --report OUT_ROOT/reports/<ext>.md` to every trainer, as the standard CLI promises; a trainer that rejects one of these fails the step.
## run_all step failures in the final run (fixture output, not results)

`python run_all.py --fixtures --all --continue-on-error --out OUT_ROOT --bootstrap 20` writes its own fixture
root (analytics.fixtures, seed 0, 4 games, held-out every 4) and runs every step with `--data ROOT --out OUT_ROOT
--seed 0`. Twelve lanes pass every step; three do not:

- `coverage` validate: exit 2, `run_all.py: error: unrecognized arguments: --seed 0` (analytics/coverage/validate.py
  argparse has no `--seed`; run_all passes it to every step). Consequence: coverage features and train do not run,
  `playcalling` is skipped as its dependent, and `coverage_predictions` is missing under run_all's OUT_ROOT.
- `winprob` train: `SystemExit: winprob.train: a feature changed when the play-by-play was cut:
  [('0022000017', 66, 'spread_home'), ('0022000017', 66, 'spread_missing'), ...]` (the lane's own time-forward
  self-check, on run_all's 4-game fixture root; it passes on the 6-game shared root of the INTEGRATION_SHARED part).
- `betting` train: `ValueError: betting.features: feature 'home_pf_std' correlates 0.999 with the final margin
  (over 0.99): it is built from the outcome` (the lane's leak guard, on run_all's 4-game fixture root; it passes on
  the 6-game shared root).

Under run_all's OUT_ROOT the other four interface files (xfg_predictions, epv_states, actions_detected,
player_rapm) exist and pass `check_frame`.
