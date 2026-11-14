# xFG baseline: location only (analytics/train_xfg.py)

Written 2026-11-05T09:44:35Z by `python -m analytics.train_xfg --shots /tmp/claude-0/-home-user-NBA-Comp-Viz-V3/3957b6d2-ff62-5cf9-86ad-09705e0cfdf5/scratchpad/bref_smoke_shots.parquet --report reports/xfg_baseline.md --folds 5 --seed 0 --bootstrap 200 --smoke`.

## Inputs

| file | rows | games | held-out rows | held-out games |
| --- | --- | --- | --- | --- |
| bref_smoke_shots.parquet | 370 | 2 | 179 | 1 |

Training rows: 191 attempts from 1 game(s) (02.27.2016.GSW.at.OKC). Held-out rows set aside before any fit: 179 attempts from 1 game(s) (12.16.2015.PHX.at.GSW).
Guard: `sportvu.splits.refuse_heldout` checked 1 training game ids against sportvu/splits.json.
Training make rate: 0.4764 over 191 attempts.

## Protocol

Features: distance_ft, angle_rad, is_three, is_corner (period and clock_s kept for buckets, not fed to the model; `assert_no_leak` passed on the design matrix).
Models: constant (training-fold make rate) and logistic regression on the four location features with standardisation (C=1).
Metrics: one training game: no grouped fold is possible, so the metrics are IN-SAMPLE (fit and scored on the same rows); intervals are the 2.5 and 97.5 percentiles of 200 bootstrap resamples of whole games (seed 0); no interval with a single game.

## Metrics (in-sample, 191 attempts, 1 game(s))

| model | log loss | Brier | ECE (10 bins) | calibration slope |
| --- | --- | --- | --- | --- |
| constant | 0.6920 | 0.2494 | 0.0000 | n/a |
| logistic | 0.6741 | 0.2406 | 0.0485 | 1.0329 |

The constant model's calibration slope is n/a by design: its predictions vary only by fold make rate, so a slope is not defined.

## Calibration table, logistic model (in-sample)

| bin | lo | hi | n | mean p | mean made |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.1 | 0 | n/a | n/a |
| 1 | 0.1 | 0.2 | 0 | n/a | n/a |
| 2 | 0.2 | 0.3 | 0 | n/a | n/a |
| 3 | 0.3 | 0.4 | 49 | 0.377 | 0.388 |
| 4 | 0.4 | 0.5 | 63 | 0.429 | 0.413 |
| 5 | 0.5 | 0.6 | 55 | 0.557 | 0.491 |
| 6 | 0.6 | 0.7 | 24 | 0.621 | 0.792 |
| 7 | 0.7 | 0.8 | 0 | n/a | n/a |
| 8 | 0.8 | 0.9 | 0 | n/a | n/a |
| 9 | 0.9 | 1.0 | 0 | n/a | n/a |

## Held-out rows

179 attempts from 1 held-out game(s) (12.16.2015.PHX.at.GSW) were set aside and NOT scored: a smoke run never touches them. The held-out game is reserved for the one final scoring on the Mac, by the real run below.

## Status

This is a smoke run on the saved basketball-reference fixture pages, not the baseline: the real run awaits the full 2015-16 shots parquet (104,967 attempts, 625 games, on the owner's Mac) with the exact command

    python -m analytics.train_xfg --shots collect/shots/shotchartdetail/2015-16.parquet collect/shots/heldout/shotchartdetail/2015-16.parquet --report reports/xfg_baseline.md --folds 5 --seed 0 --bootstrap 1000
