# P8 step 0, Track B: training-game baseline of scorecard v2

Written 2026-11-21 for docs/research/P8_step0.md (Track B go or no-go: "a baseline of the scorecard rows that can be
measured on training games"). Scorer: scorecard v2 (`tools/p8_scorecard.py`, adopted and pinned by D88; the pin test
passed before the runs), unchanged. Build: r23 (the production court model, detector and team model) in the P8
workspace. Truth: camera. Each game scored once, CPU interpreter of the frozen CV code, inside the heavy-job lock:

    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_scorecard game TAG --build r23 --truth camera

No held-out or D85 game, no threshold or setting chosen, nothing trained.

## Which games could be scored

| Game | Camera truth in the P8 workspace | Scored |
| --- | --- | --- |
| gsw_bkn_2015 | yes (2,075 accepted frames) | yes: the pooled row is this game alone |
| gsw_cha_2016 | no: phase A labelling covers sections s00 to s05 of 10, no camera fit, no r23 build (meta/done) | stopped |
| gsw_sac_2015 | no: phase A labelling covers sections s00 to s03 of 11, no camera fit, no r23 build (meta/done) | stopped |
| gsw_ind_2016 | yes (970 accepted frames) | yes, reported apart (D98) |

The main checkout holds Colab R1.3 camera truth for all four games, but scorecard v2 reads the P8 workspace, and
mixing Colab truth with Mac builds is the platform question of docs/research/P8_PLATFORM.md and D100. So
gsw_cha_2016 and gsw_sac_2015 wait. Finishing them on the Mac needs their remaining relabel sections, the camera fit
and an r23 rebuild (tools/p8_frozen, the night runner). The lead decides whether to run that.

With one pooled game there is no game bootstrap (it needs at least two games). The intervals below are the scorer's
window-cluster bootstrap (1,000 resamples, seed 0, 95 percent percentile interval).

## Rows

| Row | gsw_bkn_2015 | gsw_ind_2016 (D98, apart) |
| --- | --- | --- |
| Live wide frames with accepted truth (share) | 2,073 of 17,531 (0.118) | 970 of 20,988 (0.046) |
| Missed players | 0.26 [0.246, 0.278], n = 20,457 | 0.421 [0.406, 0.438], n = 9,577 |
| Ghost boxes (non-referee) | 0.118 [0.103, 0.134], n = 18,190 | 0.286 [0.267, 0.307], n = 8,439 |
| Team labels | 0.868, n = 16,479 (no interval: see below) | 0.634, n = 6,305 |
| Position error p50 | 1.56 ft [1.503, 1.635], n = 17,167 | 1.52 ft [1.44, 1.62], n = 6,986 |
| Position error p90 | 3.36 ft [3.213, 3.512] | 3.55 ft [3.321, 3.797] |
| Near-field bias (near, far third) | 0.03 ft, 1.57 ft | 0.34 ft, 1.34 ft |
| Coverage | 0.998 | 1.0 |
| Identity | 13.73 switches a possession, 75 possessions | 22.05, 40 possessions |
| Truth feet reprojection | p50 16.087 px, p90 45.909 px, n = 17,167 | p50 19.123 px, p90 52.041 px, n = 6,986 |

Not measurable on r23, as scorecard v2 reports: ball in the image, ball carrier, identity (v2), identity (reported),
shots, rebounds, shooter and rebounder position (the build writes no ball, carrier, roster, jersey or event output),
and the swap test (outside this scorer).

Stopped rows:

- Team labels have no interval: the pinned scorer reports the frozen team row (best A/B mapping per window, an upper
  bound) without a bootstrap and keeps no per-window team counts. An interval needs a scorer change (a PLAN.md line
  under D88) or new code that recomputes the frozen team rows per window. Not done.
- Every row lacks a game bootstrap and a pooled value over three games until gsw_cha_2016 and gsw_sac_2015 have
  camera truth.

Caveats (printed by the scorer):

- In sample: the r23 court model was trained on these games' R1.3 labels, so the position, bias and coverage rows
  here are not a held-out measure.
- Camera-truth frames are accepted only where V3's court mapping and boxes let the ICP matching start, so the truth
  share (0.118 and 0.046) is small and favours frames where the pipeline already works.
- A missed player here is a SportVU player the frozen scorer matches to no box under the camera truth; the
  teacher follow-up in docs/research/P8_teachers.md counts misses with its own 2.5 ft rule on 400 sampled frames
  (0.2656 on gsw_bkn_2015), consistent with this row's 0.26.
