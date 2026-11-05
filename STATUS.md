# STATUS.md: Phase 0 build, 2026-11-05 (unattended, about 30 minutes of cloud credit)

Goal: Phase 0 of the execution plan (data foundation and evaluation harness) to a tested state, then the
location-only xFG baseline of Phase 1. Integration branch `analytics-phase0`, based on `track-fix`.
Eight lanes ran in parallel worktrees; each is listed below with its outcome once it reported.

## Assumptions taken without asking

- The five Phase 0 table schemas were fixed by the lead at launch (docs/DATA_CONTRACT.md sections 16 to 20)
  so no lane waited on another; lane A wrote them into the contract and `analytics/schema.py`.
- The only real game with both tracking frames and play-by-play in this container is the held-out game
  12.16.2015.PHX.at.GSW. Lanes D, E and F used it once as a smoke test with every threshold fixed in advance
  and no parameter adjusted after looking at it. A deterministic name lookup (lane C) is not fitting.
- The training game 11.14.2015.BKN.at.GSW has frames here but no play-by-play file, so it served only the
  roster extraction.
- The xFG baseline (lane H) trained on the one real shot-chart page that is neither tracked nor held out
  (GSW at OKC, 2016-02-27) and scored the held-out PHX page once; its numbers are a smoke run, labelled as such.
- A day of effort in the plan is 5 to 6 hours; nothing in Phase 2 or later was started.

## Inputs present and missing

| input | status here | where it lives |
| --- | --- | --- |
| tracking frames, 2 games (BKN at GSW train, PHX at GSW held out) | present in the session scratchpad, never copied into the repo | full set of about 621 games on Drive, `nba_harvest/sportvu_contract/` |
| play-by-play for PHX at GSW | present: `data/pbp/201512160GSW.json` | the 621 `_pbp.parquet` files on Drive, filled by `collect/pbp.py --br-only` |
| Step 1 manifest with gaps over 2 s | 3-game scratchpad manifest only | full manifest on Drive |
| shots table, 2015-16 (104,967 rows) | missing; two real shot-chart pages in `tests/fixtures/bref/` | `data/collect/bref_shots/shotchart/2015-16.parquet` on the Mac, copied to Drive |
| NBA.com tables | not pulled (site blocked) | not available |

## Lanes

| lane | files | outcome |
| --- | --- | --- |
| A schema | docs/DATA_CONTRACT.md 16 to 20, analytics/schema.py, tests/test_schema.py | merged; reviewer found three leakage holes (post-outcome fields in the allowlist, shooter name reconstructing the identity target, guard checking one table only); fixed, 33 tests |
| B gaps | analytics/gaps.py, tests/test_gaps.py, reports/sportvu_gaps.md | merged, reviewer PASS; 8 tests; real 3-game report: 1 game with a gap over 10 s (BKN at GSW, longest 22.3 s) |
| C rosters | analytics/rosters.py, tests/test_rosters.py, reports/rosters_unmatched.md | merged, review running at cutoff; 9 tests; PHX at GSW: 22 of 22 shot-chart names and 25 of 25 play-by-play names matched |
| D shot frames | analytics/shot_frames.py, tests/test_shot_frames.py, reports/localisation_shots.md | merged, reviewer PASS; 8 tests; PHX at GSW smoke: release found for 125 of 179 shots (70%), shooter identity agreement 97% where known, 71 high-confidence rows |
| E rebound frames | analytics/rebound_frames.py, tests/test_rebound_frames.py, reports/localisation_rebounds.md | merged, reviewer PASS (advisory: confidence columns encoded identity_match; fixed by marking them targets in the schema); 7 tests; PHX at GSW smoke: rebound frame for 83 of 95 misses (87%), rebounder identity agreement 72.5% where known |
| F possessions | analytics/possessions.py, tests/test_possessions.py | merged, reviewer PASS with one medium finding (free throws by the non-offense team credited to nobody), fix merged (free throws by the other team now split the possession); 10 tests; pre-fix PHX at GSW smoke: 219 possessions, points sum 231 equals the final score |
| G evaluation | analytics/eval.py, tests/test_eval.py, tests/test_splits.py | merged, reviewer PASS; 17 tests; the guard test fails on any analytics/train_*.py without refuse_heldout and on any file that fits without the guard |
| H xFG baseline | analytics/features_xfg.py, analytics/train_xfg.py, tests/test_xfg.py, reports/xfg_baseline.md | merged, reviewer PASS (low: fit_models guards on the flag only; an id guard was tried and reverted because it tripped the synthetic test, left as a follow-up); 7 tests; smoke run only: one game in-sample (191 attempts) log loss 0.674 logistic vs 0.692 constant; held-out PHX (179 attempts, scored once) 0.664 vs 0.693; no interval over one game |

## Where it stands

97 tests pass on `analytics-phase0` (`python3 -m pytest tests/test_schema.py tests/test_gaps.py tests/test_rosters.py tests/test_shot_frames.py tests/test_rebound_frames.py tests/test_possessions.py tests/test_eval.py tests/test_splits.py tests/test_xfg.py`). Phase 0 is built and tested on synthetic fixtures plus one real held-out game used as a smoke test. The Phase 1 baseline exists as a tested pipeline; its numbers are a smoke run, not a result. Open items: the lane C review verdict, an id-based guard inside fit_models, pinning four unspecified shot-localisation constants in the contract (apex search 2 s, rim radius 6 ft, flight 25 frames, high-confidence 0.6), and a driver that runs the three localisers over every game of the Drive manifest.

## Next on the Mac (where the full data is)

1. `python -m analytics.gaps --manifest "<Drive>/nba_harvest/sportvu_contract/manifest.json" --out reports/sportvu_gaps.md`
2. For every game with frames and play-by-play: `python -m analytics.shot_frames ...`, `python -m analytics.rebound_frames ...`,
   `python -m analytics.possessions ...` (a driver over the manifest is the first follow-up task).
3. `python -m analytics.train_xfg --shots data/collect/bref_shots/shotchart/2015-16.parquet --report reports/xfg_baseline.md`
