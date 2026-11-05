# Phase 0 driver summary

Fixture output: synthetic games written by tests/_fx_phase0.py (or analytics/fixtures.py), not real tracking data. The numbers only show the driver works end to end.

Data root: `/tmp/phase0_fixtures_twzgzn7r/sportvu_contract`. Output root: `/tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/fxrun2`. Written 2026-11-05 08:34:49 by `python -m analytics.run_phase0`.
Held-out games (sportvu/splits.json) are written under `heldout/` by the modules' own routing and are never fit or tuned on; every localiser is deterministic with its thresholds fixed in the contract.

## Games

| metric | value |
|---|---|
| games in the Step 1 manifest | 6 |
| games processed (status ok, pbp joined) | 5 |
| games ok (every step) | 5 |
| games with a failed step | 0 |
| held-out games among the processed | 1 |
| games skipped | 1 |

## Per-step totals

| step | games ok | games failed | rows | seconds |
|---|---|---|---|---|
| rosters | 5 | 0 | 70 | 1.1 |
| shot_frames | 5 | 0 | 198 | 2.9 |
| rebound_frames | 5 | 0 | 89 | 2.9 |
| possessions | 5 | 0 | 200 | 0.6 |

### rosters

| metric | value |
|---|---|
| players | 70 |
| play-by-play names matched / total | 70 / 70 |
| share_pbp_matched | 1.0000 |
| match_method counts | exact 70, fuzzy 0, lastname 0, none 0 |

### shot_frames

| metric | value |
|---|---|
| n_shots | 198 |
| n_release_found | 198 |
| share_release_found | 1.0000 |
| n_identity_known | 198 |
| n_identity_agree | 198 |
| identity_agreement (over all shots with both ids) | 1.0000 |
| n_high | 198 |
| share_high | 1.0000 |

### rebound_frames

| metric | value |
|---|---|
| n_misses | 89 |
| n_rebound_found | 89 |
| share_rebound_found | 1.0000 |
| n_identity_known | 89 |
| n_identity_agree | 89 |
| identity_agreement (over all misses with both ids) | 1.0000 |
| n_high | 89 |
| share_high | 1.0000 |

### possessions

| metric | value |
|---|---|
| possessions | 200 |
| points_total | 241 |
| possessions with a clock gap | 5 |
| dropped (shorter than one frame) | 0 |
| by end_kind | shot_made 109, shot_missed 70, turnover 21 |
| points-sum check: games with a final score in the events | 5 |
| points-sum check: possession points equal the final score | 5 |
| points-sum mismatches | none |

## Clock gaps (analytics.gaps over the Step 1 manifest)

| metric | value |
|---|---|
| games in manifest | 6 |
| games with status ok | 6 |
| games with any gap over 2 s | 6 |
| games with any gap over 10 s | 0 |
| gaps over 2 s | 6 |
| gaps over 10 s | 0 |
| total gap seconds | 18 |

| split | games | ok | with gap over 2 s | with gap over threshold | gap seconds |
|---|---|---|---|---|---|
| heldout_game | 2 | 2 | 2 | 0 | 6 |
| train | 4 | 4 | 4 | 0 | 12 |

## Skipped games

| game | reason |
|---|---|
| 11.27.2015.GRN.at.QCY | manifest says heldout True but sportvu/splits.json routes it to the training folder: refusing to run a game whose held-out status is in doubt |

## Per game

| game | split | rosters | shot_frames | rebound_frames | possessions | n_shots | share_release_found | n_misses | share_rebound_found | possessions | points / final | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01.18.2016.HLX.at.LRD | train | ok | ok | ok | ok | 44 | 1.0000 | 20 | 1.0000 | 40 | 53 / 53 | 1.7 |
| 02.04.2016.GRN.at.QCY | train | ok | ok | ok | ok | 45 | 1.0000 | 25 | 1.0000 | 40 | 44 / 44 | 2.2 |
| 02.07.2016.OMA.at.DUL | train | ok | ok | ok | ok | 37 | 1.0000 | 15 | 1.0000 | 40 | 48 / 48 | 1.4 |
| 12.15.2015.ABQ.at.GRN | train | ok | ok | ok | ok | 35 | 1.0000 | 16 | 1.0000 | 40 | 43 / 43 | 1.1 |
| 12.16.2015.PHX.at.GSW | held-out | ok | ok | ok | ok | 37 | 1.0000 | 13 | 1.0000 | 40 | 53 / 53 | 1.2 |

RUN OK
