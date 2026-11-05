# Phase 0 driver summary

Fixture output: synthetic games written by tests/_fx_phase0.py (or analytics/fixtures.py), not real tracking data. The numbers only show the driver works end to end.

Data root: `/tmp/phase0_fixtures_8t4z_qwv`. Output root: `/tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/fxrun`. Written 2026-11-05 08:27:34 by `python -m analytics.run_phase0`.
Held-out games (sportvu/splits.json) are written under `heldout/` by the modules' own routing and are never fit or tuned on; every localiser is deterministic with its thresholds fixed in the contract.

## Games

| metric | value |
|---|---|
| games in the Step 1 manifest | 6 |
| games processed (status ok, pbp joined) | 6 |
| games ok (every step) | 6 |
| games with a failed step | 0 |
| held-out games among the processed | 2 |
| games skipped | 0 |

## Per-step totals

| step | games ok | games failed | rows | seconds |
|---|---|---|---|---|
| rosters | 6 | 0 | 24 | 0.4 |
| shot_frames | 6 | 0 | 36 | 0.4 |
| rebound_frames | 6 | 0 | 19 | 0.5 |
| possessions | 6 | 0 | 48 | 0.2 |

### rosters

| metric | value |
|---|---|
| players | 24 |
| play-by-play names matched / total | 24 / 24 |
| share_pbp_matched | 1.0000 |
| match_method counts | exact 24, fuzzy 0, lastname 0, none 0 |

### shot_frames

| metric | value |
|---|---|
| n_shots | 36 |
| n_release_found | 36 |
| share_release_found | 1.0000 |
| n_identity_known | 36 |
| n_identity_agree | 36 |
| identity_agreement (over all shots with both ids) | 1.0000 |
| n_high | 36 |
| share_high | 1.0000 |

### rebound_frames

| metric | value |
|---|---|
| n_misses | 19 |
| n_rebound_found | 19 |
| share_rebound_found | 1.0000 |
| n_identity_known | 19 |
| n_identity_agree | 19 |
| identity_agreement (over all misses with both ids) | 1.0000 |
| n_high | 19 |
| share_high | 1.0000 |

### possessions

| metric | value |
|---|---|
| possessions | 48 |
| points_total | 38 |
| possessions with a clock gap | 4 |
| dropped (shorter than one frame) | 0 |
| by end_kind | period_end 6, shot_made 17, shot_missed 19, turnover 6 |
| points-sum check: games with a final score in the events | 6 |
| points-sum check: possession points equal the final score | 6 |
| points-sum mismatches | none |

## Clock gaps (analytics.gaps over the Step 1 manifest)

| metric | value |
|---|---|
| games in manifest | 6 |
| games with status ok | 6 |
| games with any gap over 2 s | 4 |
| games with any gap over 10 s | 2 |
| gaps over 2 s | 4 |
| gaps over 10 s | 2 |
| total gap seconds | 30.2 |

| split | games | ok | with gap over 2 s | with gap over threshold | gap seconds |
|---|---|---|---|---|---|
| None | 4 | 4 | 2 | 2 | 24.1 |
| heldout_arena | 2 | 2 | 2 | 0 | 6.1 |

## Skipped games

None.

## Per game

| game | split | rosters | shot_frames | rebound_frames | possessions | n_shots | share_release_found | n_misses | share_rebound_found | possessions | points / final | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11.01.2015.BKN.at.GSW | train | ok | ok | ok | ok | 6 | 1.0000 | 3 | 1.0000 | 8 | 7 / 7 | 0.4 |
| 11.04.2015.SAC.at.GSW | train | ok | ok | ok | ok | 6 | 1.0000 | 4 | 1.0000 | 8 | 4 / 4 | 0.2 |
| 11.07.2015.IND.at.CLE | held-out | ok | ok | ok | ok | 6 | 1.0000 | 3 | 1.0000 | 8 | 7 / 7 | 0.2 |
| 11.10.2015.CHA.at.GSW | train | ok | ok | ok | ok | 6 | 1.0000 | 2 | 1.0000 | 8 | 10 / 10 | 0.2 |
| 11.13.2015.PHX.at.GSW | train | ok | ok | ok | ok | 6 | 1.0000 | 5 | 1.0000 | 8 | 2 / 2 | 0.2 |
| 11.16.2015.NYK.at.CLE | held-out | ok | ok | ok | ok | 6 | 1.0000 | 2 | 1.0000 | 8 | 8 / 8 | 0.3 |

RUN OK
