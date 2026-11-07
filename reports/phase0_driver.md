# Phase 0 driver summary

Fixture output: synthetic games written by tests/_fx_phase0.py (analytics.fixtures.write_fixture_root when present), not real tracking data. The numbers only show the driver works end to end.

Data root: `<temporary synthetic DATA_ROOT>`. Output root: `<temporary OUT_ROOT>`. Written 2026-11-05 08:56:23 by `python -m analytics.run_phase0`.
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
| rosters | 6 | 0 | 84 | 0.3 |
| shot_frames | 6 | 0 | 238 | 0.7 |
| rebound_frames | 6 | 0 | 106 | 0.9 |
| possessions | 6 | 0 | 240 | 0.2 |

### rosters

| metric | value |
|---|---|
| players | 84 |
| play-by-play names matched / total | 84 / 84 |
| share_pbp_matched | 1.0000 |
| shot chart names matched / total | 0 / 0 |
| share_bref_matched | n/a |
| players with bref_slug filled | 0 |
| games with a shot chart page (bref_slug, bref_name filled) | 0 |
| games without one (bref_slug, bref_name null; pass --bref-dir) | 6 |
| match_method counts | exact 84, fuzzy 0, lastname 0, none 0 |

### shot_frames

| metric | value |
|---|---|
| n_shots | 238 |
| n_release_found | 238 |
| share_release_found | 1.0000 |
| n_identity_known | 238 |
| n_identity_agree | 238 |
| identity_agreement (over all shots with both ids) | 1.0000 |
| n_high | 238 |
| share_high | 1.0000 |

### rebound_frames

| metric | value |
|---|---|
| n_misses | 106 |
| n_rebound_found | 106 |
| share_rebound_found | 1.0000 |
| n_identity_known | 106 |
| n_identity_agree | 106 |
| identity_agreement (over all misses with both ids) | 1.0000 |
| n_high | 106 |
| share_high | 1.0000 |

### possessions

| metric | value |
|---|---|
| possessions | 240 |
| points_total | 291 |
| possessions with a clock gap | 6 |
| dropped (shorter than one frame) | 0 |
| by end_kind | shot_made 132, shot_missed 83, turnover 25 |
| points-sum check: games with a final score in the events | 6 |
| points-sum check: possession points equal the final score | 6 |
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
| heldout_arena | 1 | 1 | 1 | 0 | 3 |
| heldout_game | 1 | 1 | 1 | 0 | 3 |
| train | 4 | 4 | 4 | 0 | 12 |

## Skipped games

None.

## Per game

| game | split | rosters | shot_frames | rebound_frames | possessions | n_shots | share_release_found | n_misses | share_rebound_found | possessions | points / final | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01.18.2016.HLX.at.LRD | train | ok | ok | ok | ok | 44 | 1.0000 | 20 | 1.0000 | 40 | 53 / 53 | 0.4 |
| 02.04.2016.GRN.at.QCY | train | ok | ok | ok | ok | 45 | 1.0000 | 25 | 1.0000 | 40 | 44 / 44 | 0.4 |
| 02.07.2016.OMA.at.DUL | train | ok | ok | ok | ok | 37 | 1.0000 | 15 | 1.0000 | 40 | 48 / 48 | 0.3 |
| 11.27.2015.GRN.at.CLE | held-out | ok | ok | ok | ok | 40 | 1.0000 | 17 | 1.0000 | 40 | 50 / 50 | 0.4 |
| 12.15.2015.ABQ.at.GRN | train | ok | ok | ok | ok | 35 | 1.0000 | 16 | 1.0000 | 40 | 43 / 43 | 0.4 |
| 12.16.2015.PHX.at.GSW | held-out | ok | ok | ok | ok | 37 | 1.0000 | 13 | 1.0000 | 40 | 53 / 53 | 0.3 |

RUN OK
