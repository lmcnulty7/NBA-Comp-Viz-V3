# SportVU clock gaps

Manifest: `/tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/out/manifest.json` (3 games, updated 2026-11-04 23:52:59, contract v1).

Scope: this report covers only the games listed in that manifest. A gap is a hole of more than 2 s in the tracking clock of one quarter, as recorded by the Step 1 converter (`gaps_over_2s` in the manifest); the clock counts down, so a gap runs from the larger clock to the smaller one. Held-out games are counted here and nothing is fit or tuned.

Coverage note: this is the 3-game manifest of the local Step 1 scratchpad run. The full Step 1 manifest (621 games) lives on the owner's Google Drive and was not available to this run, so the season-wide gap picture is not in this report; rerun the same command against that manifest.json to produce it.

## Summary

| metric | value |
|---|---|
| games in manifest | 3 |
| games with status ok | 2 |
| games with any gap over 2 s | 2 |
| games with any gap over 10 s | 1 |
| gaps over 2 s | 4 |
| gaps over 10 s | 2 |
| total gap seconds (all gaps over 2 s) | 45.4 |

## By split

| split | games | ok | with gap over 2 s | with gap over 10 s | gap seconds |
|---|---|---|---|---|---|
| heldout_arena | 1 | 0 | 0 | 0 | 0 |
| heldout_game | 1 | 1 | 1 | 0 | 8.4 |
| train | 1 | 1 | 1 | 1 | 37 |

## Games with a gap over 10 s

| game | split | gaps over 10 s | longest gap (s) | seconds over 10 s |
|---|---|---|---|---|
| 11.14.2015.BKN.at.GSW | train | 2 | 22.3 | 37 |

Games without gap data (status not ok): 01.23.2016.CHI.at.CLE (source_broken).
