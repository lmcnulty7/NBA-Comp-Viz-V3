# P8 step 0, Track B: training-game baseline of scorecard v2

Written 2026-11-21, extended 2026-11-22 under D140, for docs/research/P8_step0.md (Track B go or no-go: "a baseline
of the scorecard rows that can be measured on training games"). Scorer: scorecard v2 (`tools/p8_scorecard.py`,
adopted and pinned by D88; the pin test passed before the runs), unchanged. Build: r23 (the production court model,
detector and team model) in the P8 workspace. Truth: camera. No held-out or D85 game, no threshold or setting chosen,
nothing trained.

## Games and how their truth was made

| Game | Mac camera truth | Accepted frames (camera fit reprojection) | Scored |
| --- | --- | --- | --- |
| gsw_bkn_2015 | night 1 (2026-11-17) | 2,075 (6.52 px) | pooled |
| gsw_cha_2016 | sections s00 to s05 on night 1; s06 to s09, the fit and the r23 build under D140 | 1,366 (7.65 px) | pooled |
| gsw_sac_2015 | sections s00 to s03 on night 1; s04 to s10, the fit and the r23 build under D140 | 2,151 (5.47 px) | pooled |
| gsw_ind_2016 | night 1 | 970 (18.35 px) | apart (D98) |

Accepted frames and reprojection are from `labels/<game>_result.json` in the P8 workspace. The D140 run used the
night runner unchanged (`V3_CV_PYTHON=<frozen CV python> python3 -m tools.p8_night run --games
gsw_cha_2016,gsw_sac_2015`) with the device settings of night 1 (P8_DEVICE_DETECTOR unset), so the workspace mixes no
device settings. Every heavy job took the heavy-job lock for its own duration, taking turns with the p8-events pilot.
All 42 jobs ended ok. The two section failures of night 1 (gsw_cha_2016_s06 and gsw_sac_2015_s04, one window build
each) were gate-empty windows, which D97's fix now marks and skips. No Colab session and no D100 decision was needed:
the Mac rule is the one gsw_bkn_2015 was built with.

Lock hours used by D140: 3.35 h held by its 40 heavy jobs (the sum of the lock wrapper's "done in" times in this run's
per-job logs), 3.47 h of job time including waits for the lock; the run started at 21:49 on 2026-11-21 and ended
before 01:18 on 2026-11-22. The pooled combine held the lock 45 s more.

## Pooled rows, the three R1.3 games

`python3 -m tools.p8_scorecard combine --build r23 --truth camera --tags gsw_bkn_2015,gsw_cha_2016,gsw_sac_2015`
pools position and missed players with a game and a window bootstrap (`combined__r23__camera.json`).
`python3 -m tools.p8_baseline` adds the ghost and team rows, pooled from each game's frozen counts with the same game
bootstrap (`baseline_r13__r23__camera.json`). Both files are in the P8 workspace's scorecards folder. Bootstrap: 1,000
resamples, seed 0, 95 percent percentile interval. A game bootstrap over 3 games has only 10 distinct resamples, so
its intervals are coarse.

| Row | Pooled | Game bootstrap | Window bootstrap | Target |
| --- | --- | --- | --- | --- |
| Missed players | 0.2654 | [0.233, 0.325] | [0.254, 0.277] | <= 5% |
| Ghost boxes (non-referee) | 0.1287 (6,234 of 48,429) | [0.108, 0.178] | none: the scorer keeps no per-window non-referee count | <= 5% |
| Team labels | 0.8248 (36,655 of 44,442) | [0.773, 0.868] | none: no per-window team counts | >= 95% |
| Position error p50 | 1.531 ft | [1.514, 1.556] | [1.489, 1.58] | <= 2.0 ft |
| Position error p90 | 3.28 ft | [3.154, 3.341] | [3.171, 3.388] | <= 5.0 ft |

## Per game

Window-cluster bootstrap intervals (the scorer's, 1,000 resamples, seed 0).

| Row | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 | gsw_ind_2016 (D98, apart) |
| --- | --- | --- | --- | --- |
| Live wide frames with accepted truth (share) | 2,073 of 17,531 (0.118) | 1,353 of 15,499 (0.087) | 2,151 of 15,734 (0.137) | 970 of 20,988 (0.046) |
| Missed players | 0.26 [0.246, 0.278], n 20,457 | 0.325 [0.311, 0.344], n 13,284 | 0.233 [0.218, 0.249], n 21,144 | 0.421 [0.406, 0.438], n 9,577 |
| Ghost boxes (non-referee) | 0.118 [0.103, 0.134], n 18,190 | 0.178 [0.162, 0.196], n 11,948 | 0.108 [0.096, 0.123], n 19,278 | 0.286 [0.267, 0.307], n 8,439 |
| Team labels | 0.868, n 16,479 | 0.773, n 10,558 | 0.815, n 17,405 | 0.634, n 6,305 |
| Position error p50 | 1.56 ft [1.503, 1.635] | 1.52 ft [1.431, 1.631] | 1.52 ft [1.437, 1.596] | 1.52 ft [1.44, 1.62] |
| Position error p90 | 3.36 ft [3.213, 3.512] | 3.16 ft [2.991, 3.41] | 3.29 ft [3.124, 3.463] | 3.55 ft [3.321, 3.797] |
| Near-field bias (near, far third) | 0.03, 1.57 ft | 0.33, 1.14 ft | 0.22, 1.34 ft | 0.34, 1.34 ft |
| Coverage | 0.998 | 0.993 | 0.994 | 1.0 |
| Identity (switches a possession) | 13.73 (75 possessions) | 14.55 (51) | 12.94 (81) | 22.05 (40) |
| Truth feet reprojection p50, p90 | 16.087, 45.909 px | 18.65, 53.12 px | 14.667, 44.595 px | 19.123, 52.041 px |

Not measurable on r23, as scorecard v2 reports: ball in the image, ball carrier, identity (v2), identity (reported),
shots, rebounds, shooter and rebounder position (the build writes no ball, carrier, roster, jersey or event output),
and the swap test (outside this scorer).

## Caveats

- In sample: the r23 court model was trained on these games' R1.3 labels, so the position, bias and coverage rows are
  not a held-out measure (the scorer's caveat).
- The Mac truth accepts fewer frames than Colab's R1.3 run on the same games (gsw_cha_2016 1,366 against 3,531;
  gsw_sac_2015 2,151 against 3,305; reports/sportvu_labels_manifest.json), for the reasons of D99 (MPS boxes, the clock
  offset solve). The truth share is small and favours frames where the pipeline already works.
- The team row is the frozen best A/B mapping per window, an upper bound on a fixed mapping; pooled it has a game
  bootstrap, but no window bootstrap.
- A missed player here is a SportVU player the frozen scorer matches to no box under the camera truth; the teacher
  follow-up in docs/research/P8_teachers.md counts misses with its own 2.5 ft rule on 400 sampled frames (0.2656 on
  gsw_bkn_2015), consistent with that game's 0.26.
