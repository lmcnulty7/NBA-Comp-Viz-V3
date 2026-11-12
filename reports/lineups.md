# Lineup synergy and stints (analytics/lineups)

**Real data, training folds only; held-out not scored.** Stage 5 phase 4 (docs/mac_runbook.md), run with --skip-heldout on 2026-11-12; the 2024-25 season is held out; 53 games with shifted period labels dropped before stints.

Written 2026-11-12T09:20:54Z by `python -m analytics.lineups.train --data $DATA_ROOT --seed 0 --report reports/lineups.md --skip-heldout`.

## Inputs

| item | value |
| --- | --- |
| source | DATA_ROOT $DATA_ROOT |
| seasons | 2020-21, 2021-22, 2022-23, 2023-24, 2024-25 |
| stints in | 189784 |
| stints after validate | 180243 |
| games | 6363 |
| lineup_ok share | 0.950 |
| dropped not 5v5 | 9541 |
| dropped duration | 0 |
| dropped points | 0 |
| dropped duplicate | 0 |
| dropped heldout | 0 |
| possessions | 607295 |
| feature frame rows | 180243 |

## Protocol

Unit: stint; target: (points_home - points_away) / possessions x 100 with possessions = duration_s / 28.8 when the table has no count (100 per team per 48 minutes). Candidates: plus_minus (raw net rating pooled over the fitting seasons), rapm (possession-weighted ridge on player indicators, alpha by grouped CV by game over [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]), rapm_synergy (rapm plus ridge-penalised indicators for five-man lineups with at least 300 possessions in the fitting seasons; the lineup coefficient is the synergy). Selection metric: out-of-season Spearman and MSE of the predicted against the raw next-season lineup net rating for lineups with at least 100 possessions on both sides, intervals from 200 bootstrap resamples over lineups (seed 0), paired against plus_minus. The candidate with the highest mean validation Spearman is selected and the held-out season 2024-25 is set aside and never scored (--skip-heldout). Guards: assert_season_split on every fit, refuse_heldout on the training game ids, assert_time_forward on the feature frame.

## Candidate table: out-of-season validation within the training seasons

Fit on 2020-21, predict 2021-22: 18 lineups present in both (14865 in the fit seasons, 17224 in the target), 50 gated lineups, 517 players, alpha rapm 3000, alpha synergy 3000.

| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |
| --- | --- | --- | --- |
| plus_minus | 0.082 [-0.445, 0.442] | 199.496 [108.675, 316.330] | baseline |
| rapm | 0.065 [-0.472, 0.527] | 143.132 [73.673, 213.029] | -0.017 [-0.672, 0.725] |
| rapm_synergy | 0.020 [-0.528, 0.475] | 144.476 [73.816, 211.090] | -0.062 [-0.713, 0.611] |

Fit on 2020-21, 2021-22, predict 2022-23: 27 lineups present in both (31502 in the fit seasons, 16592 in the target), 110 gated lineups, 662 players, alpha rapm 3000, alpha synergy 3000.

| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |
| --- | --- | --- | --- |
| plus_minus | 0.073 [-0.267, 0.380] | 384.548 [229.966, 623.179] | baseline |
| rapm | 0.222 [-0.170, 0.547] | 196.717 [64.432, 415.504] | 0.149 [-0.290, 0.584] |
| rapm_synergy | 0.235 [-0.129, 0.541] | 199.359 [60.215, 427.355] | 0.162 [-0.265, 0.540] |

Fit on 2020-21, 2021-22, 2022-23, predict 2023-24: 26 lineups present in both (47040 in the fit seasons, 15320 in the target), 186 gated lineups, 744 players, alpha rapm 3000, alpha synergy 3000.

| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |
| --- | --- | --- | --- |
| plus_minus | 0.160 [-0.229, 0.589] | 231.900 [114.607, 360.293] | baseline |
| rapm | 0.041 [-0.428, 0.519] | 154.511 [85.839, 240.148] | -0.119 [-0.635, 0.268] |
| rapm_synergy | 0.006 [-0.433, 0.476] | 164.984 [87.798, 260.210] | -0.154 [-0.629, 0.227] |

Mean validation Spearman: plus_minus 0.105, rapm 0.109, rapm_synergy 0.087. Selected: **rapm**.

## Out-of-season correlation of raw lineup net rating

| from | to | lineups | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| 2020-21 | 2021-22 | 18 | 0.120 | 0.082 |
| 2021-22 | 2022-23 | 21 | 0.102 | 0.177 |
| 2022-23 | 2023-24 | 23 | 0.253 | 0.261 |

Seasons later than the held-out season are not in this table; the held-out season is left out of it (--skip-heldout).

## Held-out season, not scored

1304 held-out game(s) set aside before any fit and never fitted, tuned or scored: held-out games not scored (--skip-heldout).
Season 2024-25 is set aside before any fit; it is left out of the correlation table above and no output table is written for it.

## Outputs (relative to OUT_ROOT)

- analytics/lineups/2021-22/player_rapm_prior.parquet
- analytics/lineups/2022-23/player_rapm_prior.parquet
- analytics/lineups/2023-24/player_rapm_prior.parquet
- analytics/lineups/2020-21/lineup_ratings.parquet
- analytics/lineups/2020-21/player_rapm.parquet
- analytics/lineups/2021-22/lineup_ratings.parquet
- analytics/lineups/2021-22/player_rapm.parquet
- analytics/lineups/2022-23/lineup_ratings.parquet
- analytics/lineups/2022-23/player_rapm.parquet
- analytics/lineups/2023-24/lineup_ratings.parquet
- analytics/lineups/2023-24/player_rapm.parquet

## Status and the Mac command

Real data run over 2020-21, 2021-22, 2022-23, 2023-24, 2024-25.

    python -m analytics.lineups.train --data "$DATA_ROOT" --report reports/lineups.md --out "$DATA_ROOT" --seed 0 --bootstrap 1000 --heldout-season 2024-25

Needs on the Mac: DATA_ROOT/collect/stints/{stints,playbyplay}/<season>.parquet for 2020-21 to 2024-25 (collect/stints.py, or `python -m analytics.lineups.stints --data DATA_ROOT` from the play-by-play), games/<season>.parquet for team ids, and the optional starters/<season>.parquet. Nothing under collect/stints/heldout/ is read.
