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
| A schema | docs/DATA_CONTRACT.md 16 to 20, analytics/schema.py, tests/test_schema.py | pending |
| B gaps | analytics/gaps.py, tests/test_gaps.py, reports/sportvu_gaps.md | pending |
| C rosters | analytics/rosters.py, tests/test_rosters.py, reports/rosters_unmatched.md | pending |
| D shot frames | analytics/shot_frames.py, tests/test_shot_frames.py, reports/localisation_shots.md | pending |
| E rebound frames | analytics/rebound_frames.py, tests/test_rebound_frames.py, reports/localisation_rebounds.md | pending |
| F possessions | analytics/possessions.py, tests/test_possessions.py | pending |
| G evaluation | analytics/eval.py, tests/test_eval.py, tests/test_splits.py | pending |
| H xFG baseline | analytics/features_xfg.py, analytics/train_xfg.py, tests/test_xfg.py, reports/xfg_baseline.md | pending |

## Next on the Mac (where the full data is)

1. `python -m analytics.gaps --manifest "<Drive>/nba_harvest/sportvu_contract/manifest.json" --out reports/sportvu_gaps.md`
2. For every game with frames and play-by-play: `python -m analytics.shot_frames ...`, `python -m analytics.rebound_frames ...`,
   `python -m analytics.possessions ...` (a driver over the manifest is the first follow-up task).
3. `python -m analytics.train_xfg --shots data/collect/bref_shots/shotchart/2015-16.parquet --report reports/xfg_baseline.md`
