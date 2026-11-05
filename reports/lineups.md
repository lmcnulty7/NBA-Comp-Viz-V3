# Lineup synergy and stints (analytics/lineups)

Written 2026-11-05T09:05:19Z by `python -m analytics.lineups.train --fixtures --fixture-generator local --n-games 40 --report reports/lineups.md --bootstrap 200`.

**This run used the synthetic fixture generator tests._fx_lineups._fallback_synth_stints. Every number below is fixture output, not a result.** The real run is the Mac command at the end.

## Inputs (fixture output, not a result)

| item | value |
| --- | --- |
| source | fixture generator tests._fx_lineups._fallback_synth_stints |
| seasons | 2020-21, 2021-22, 2022-23 |
| stints in | 2256 |
| stints after validate | 2256 |
| games | 120 |
| lineup_ok share | 1.000 |
| dropped not 5v5 | 0 |
| dropped duration | 0 |
| dropped points | 0 |
| dropped duplicate | 0 |
| dropped heldout | 0 |
| possessions | 12000 |
| feature frame rows | 2256 |

## Protocol

Unit: stint; target: (points_home - points_away) / possessions x 100 with possessions = duration_s / 28.8 when the table has no count (100 per team per 48 minutes). Candidates: plus_minus (raw net rating pooled over the fitting seasons), rapm (possession-weighted ridge on player indicators, alpha by grouped CV by game over [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]), rapm_synergy (rapm plus ridge-penalised indicators for five-man lineups with at least 300 possessions in the fitting seasons; the lineup coefficient is the synergy). Selection metric: out-of-season Spearman and MSE of the predicted against the raw next-season lineup net rating for lineups with at least 100 possessions on both sides, intervals from 200 bootstrap resamples over lineups (seed 0), paired against plus_minus. The candidate with the highest mean validation Spearman is selected and the held-out season 2022-23 is scored once by the candidates refitted on 2020-21, 2021-22. Guards: assert_season_split on every fit, refuse_heldout on the training game ids, assert_time_forward on the feature frame.

## Candidate table: out-of-season validation within the training seasons (fixture output, not a result)

Fit on 2020-21, predict 2021-22: 10 lineups present in both (119 in the fit seasons, 111 in the target), 4 gated lineups, 36 players, alpha rapm 300, alpha synergy 300.

| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |
| --- | --- | --- | --- |
| plus_minus | 0.624 [-0.113, 0.950] | 712.776 [221.103, 1266.230] | baseline |
| rapm | 0.576 [-0.273, 0.923] | 703.568 [220.711, 1210.533] | -0.048 [-0.297, 0.151] |
| rapm_synergy | 0.576 [-0.273, 0.923] | 739.105 [204.853, 1302.922] | -0.048 [-0.297, 0.151] |

Mean validation Spearman: plus_minus 0.624, rapm 0.576, rapm_synergy 0.576. Selected: **plus_minus**.

## Out-of-season correlation of raw lineup net rating (fixture output, not a result)

| from | to | lineups | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| 2020-21 | 2021-22 | 10 | 0.606 | 0.624 |
| 2021-22 | 2022-23 (held-out season: descriptive, scored once below, not used for selection) | 6 | 0.831 | 0.829 |

Seasons later than the held-out season are not in this table; the held-out row is labelled.

## Held-out season, scored once (fixture output, not a result)

Season 2022-23 scored once by the candidates refitted on 2020-21, 2021-22: 8 lineups present in both, 4 gated lineups, 36 players, alpha rapm 1000, alpha synergy 3000, home court (rapm intercept) 1.43 per 100.

| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |
| --- | --- | --- | --- |
| plus_minus (selected) | 0.881 [0.342, 1.000] | 198.677 [39.772, 383.567] | baseline |
| rapm | 0.857 [0.341, 1.000] | 167.705 [50.270, 313.989] | -0.024 [-0.159, 0.000] |
| rapm_synergy | 0.786 [0.215, 0.976] | 183.477 [50.150, 341.626] | -0.095 [-0.450, 0.000] |

## Outputs

No tables written (pass --out OUT_ROOT).

## Status and the Mac command

Fixture run: the pipeline is exercised end to end on synthetic stints with hidden ratings and a lineup interaction; nothing here is a result.

    python -m analytics.lineups.train --data "$DATA_ROOT" --report reports/lineups.md --out "$DATA_ROOT" --seed 0 --bootstrap 1000 --heldout-season 2024-25

Needs on the Mac: DATA_ROOT/collect/stints/{stints,playbyplay}/<season>.parquet for 2020-21 to 2024-25 (collect/stints.py, or `python -m analytics.lineups.stints --data DATA_ROOT` from the play-by-play), games/<season>.parquet for team ids, and the optional starters/<season>.parquet. Nothing under collect/stints/heldout/ is read.
