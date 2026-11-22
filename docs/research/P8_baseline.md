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

## New arenas (D146): hou_cle_2016 scored, nop_gsw_2015 stopped

The two Track A games (identity confirmed automatically under D130 and D138; downloaded under D146,
reports/research/p8_d146_fetch.json) were registered as confirmed development games in
reports/research/p8_registry.json and run through the same night runner, MPS settings and lock as D140, after a
symbolic link `video/<tag>.mp4` to each downloaded file (no re-encode).

Frame rate, checked before any build: both files run at about 60 fps (60/1 and 60000/1001), as do the four R1.3
sections already used for Mac truth (60000/1001; nyk_bos_2016, okc_sac_2015 and por_okc_2016 are 30000/1001). Every
Phase A step reads by time, not by frame count: the build stride is round(fps / 10), 10 Hz (harvest_driver.build_stride);
the clock OCR steps round(fps / 1.0) frames, 1 Hz (sportvu/local_sync.py READ_HZ); windows are cut in seconds
(sportvu/rebuild.py); the clock offset is searched in seconds (sportvu/truth.py OFFSET_MAX 1.5, OFFSET_STEP 0.04). So
60 fps changes nothing the pipeline reads, and the HOU time maps show stride 6 and about one read a second.

| | hou_cle_2016 | nop_gsw_2015 |
| --- | --- | --- |
| Scorebug layout (frozen automatic check, sportvu.prepare scoring) | abc_finals16_720 (plausible 0.775, consistent 0.725) | none passes (best: fso15_720 plausible 0.389, consistent 0.167) |
| Direction | identity, vote share 1.0 over 11 windows | not reached |
| Camera fit | 4,427 accepted frames, reprojection 6.4 px | not reached |
| Scored | yes | stopped at the first sync section |

hou_cle_2016, r23, camera truth (window-cluster bootstrap, 1,000 resamples, seed 0). r23's court model was not
trained on this game, so unlike the R1.3 rows these are not in sample. The GSW gold-key paint check does not exist
in Houston.

| Row | hou_cle_2016 |
| --- | --- |
| Live wide frames with accepted truth (share) | 4,425 of 20,310 (0.218) |
| Missed players | 0.233 [0.221, 0.247], n 43,030 |
| Ghost boxes (non-referee) | 0.083 [0.074, 0.094], n 37,135 |
| Team labels | 0.734, n 33,428 |
| Position error p50 | 1.53 ft [1.47, 1.602], n 36,115 |
| Position error p90 | 3.34 ft [3.19, 3.481] |
| Near-field bias (near, far third) | 0.48, 1.15 ft |
| Coverage | 1.0 |
| Identity (switches a possession) | 13.88 (120 possessions) |
| Truth feet reprojection p50, p90 | 14.418, 37.095 px |

It is one channel game, so it is not pooled with the R1.3 games (the scorer never pools the two groups) and has no game
bootstrap.

nop_gsw_2015 stopped: its broadcast is the Pelicans' home feed (a full-width lower strip), and no registered scorebug
layout passes the frozen automatic check, so the sync step refuses to run ("has no scorebug layout yet"). The registry's
existing layouts for new broadcasters (msg15_720, csn_nw15_720) were measured by hand on a frame; doing that here would
be a hand step, which D128 rules out, so it waits for the lead's decision.

Lock hours used by D146: 4.26 h held by its 55 heavy jobs (the lock wrapper's "done in" times in this run's per-job
logs), 5.48 h of job time including waits for the lock (the shots2526 pilot held it in between); the run started at
01:31 and stopped at 07:00 on 2026-11-22.

## D152: the automatic layout finder failed validation, so nop_gsw_2015 is dropped

tools/p8_layout_find.py proposes a scorebug layout from whole-frame OCR of 40 sampled frames (clock and period tokens
grouped by position across frames) and accepts a proposal only through the frozen automatic check
(tools.p8_frozen.step_layout, the proposal as the only candidate, bar 0.25). Its rule and the validation bar were
committed before any run (1188725f): on every development video whose layout already exists, IoU at least 0.5 with the
existing clock box and with the existing period box, and the frozen check passing. The existing layout went through the
same check as a control. Output: layout_find/validation.json in the P8 workspace.

| Video | Existing layout | Proposal | Clock IoU | Period IoU | Check, consistent share (control) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| gsw_bkn_2015 | csn_ba16_720 | ok | 0.604 | 0.402 | passes, 0.528 (0.583) | fail: period IoU |
| gsw_cha_2016 | csn_ba16_720 | ok | 0.604 | 0.425 | passes, 0.75 (0.7) | fail: period IoU |
| gsw_ind_2016 | abc_finals16_720 | ok | 0.77 | 0.688 | passes, 0.583 (0.611) | pass |
| gsw_sac_2015 | csn_ba16_720 | ok | 0.604 | 0.402 | passes, 0.694 (0.667) | fail: period IoU |
| nyk_bos_2016 | msg15_720 | no period on the clock's row | | | not run (0.583) | fail |
| okc_sac_2015 | fso15_720 | ok | 0.599 | 0.53 | passes, 0.65 (0.35) | pass |
| por_okc_2016 | csn_nw15_720 | no clock (seen in 9 of 36 frames, under the 25 percent share) | | | not run (control below the bar) | fail |
| hou_cle_2016 | abc_finals16_720 | ok | 0.77 | 0.688 | passes, 0.6 (0.725) | pass |

3 of 8 pass, so under D152 nop_gsw_2015 is dropped and the finder was never run on it. Nothing was tuned after
these numbers.

What the failures are:

- The three CSN Bay Area 2016 videos: the proposal reads the clock about as well as the hand layout under the frozen
  check (consistent 0.75 against 0.7 and 0.694 against 0.667; 0.528 against 0.583 on gsw_bkn_2015), but its period
  box sits further right than the hand box (x 1012 to 1076 against 1002 to 1052), so the IoU bar fails.
- nyk_bos_2016 (MSG): the period sits under the clock, not on its row, and the rule looks only on the clock's row.
- por_okc_2016 (CSN Northwest): whole-frame OCR found the clock in 9 of 36 frames. The hand layout itself does not pass
  the frozen check there (its plausible share is under 0.25), because that check reads the period as digits and this
  broadcast writes it as a word.

On D151: the finder proposes a clock crop from OCR alone and could serve the shot-check scorebug crop in principle,
but it has not passed its own validation, so it is not offered for that use.

Lock hours used by D152: 0.51 h (1,846 s held by the one validation job; it waited for the shots2526 pilot first).
