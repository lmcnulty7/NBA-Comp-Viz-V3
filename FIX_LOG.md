# FIX_LOG

Pass-by-pass record for FIX_PLAN.md (newest at the bottom; summary at the top on stop).

2026-10-03: plan written; gates 1 and 2 ticked by Lucien (SportVU download OK, branch track-fix).

## Pass 1 (2026-10-03): A0 frame sidecar
- build_trajectories now writes <clip>_frames.json per build: per processed frame the tracker state (TRACK / LINE_TRACK / HELD / LOST), H, quality_ft, n_inliers, snap res_px and n_match (new read-only attributes on CourtTracker), hull points, and every box with raw and stabilized foot. Output only; no logic path touched.
- Verified: 55/55 triage renders have a sidecar (3,721 frames: 3,234 TRACK, 360 HELD, 119 LOST, 8 LINE_TRACK); trajectories byte-identical to the previous render for all 55 clips; pytest 52 passed.
- Note: a first single-clip determinism test looked nondeterministic; it was a max-frames mismatch (90 vs 79). The build is deterministic. Already visible for A2: 360 HELD frames (9.7%) currently emit positions.

## Pass 2 (2026-10-03): A1 line-overlay score, BLOCKED after 4 variants
- Built qc/track_qc.py + track_qc.py (--triage, --sidecar/--video). Final A1 = ridge-side distance: median px from detected line pixels (outside boxes) to the projected template, share within 6 px, template-side support. reports/qc/<clip>.json for all 55 renders, reports/qc/_ranking.json worst first.
- Variants tried: 3 px template hit rate (every clip ~0.02), template-to-ridge distance at 40 px (every clip ~17 px), ridge-side distance (separates gross failures: the 4 Cleveland + 2 OKC flagged clips rank 1..7 of 55), ridge-side on line-like components only (noisy). Measurable "11 flagged clips in the bottom half" reached 7/11 under every clip summary (median, p90, share_far).
- Finding that matters more than the score: the ridge field sees the painted lines well, and the projected template is 20..40 px off them on nearly EVERY clip (corpus median 42.7 px, best clip 27 px), anchored on the far sideline. "Court off" is a continuum here. A moderate case (splash62_s01 frame 120: real key 100+ px from the template) still reads 38 px because the near sideline drawn through the crowd "explains" crowd and wordmark ridges. The pre-screen is one still per clip and one of its flags (nyk_s04_f9135) is a bad single frame in a clip that is otherwise the corpus's second best, so the yardstick is weak too. B6 must calibrate the score against SportVU; until then it is a worst-first ranking, not a pass/fail.
- Side notes for A2: the maple HSV mask (gate/hsv_baseline.py) is arena-specific (OKC blue floor, Cleveland wine key) and cannot be the floor mask as-is; the solver's own n_match is 14..48 of ~450 template samples with residual ~2.5 px on every clip, which is what random matches in a 6 px window look like, so the snap's "evidence" on 720p footage is weak everywhere. pytest 52 passed.

## Pass 3 (2026-10-03): A2 geometry checks
- Added geometry() to qc/track_qc.py and the per-game roll-up to track_qc.py (reports/qc/_by_game.json). Live rules: scorebug crossing (far sideline / baselines vs the layout's clock+period boxes padded 20 px) and off-court feet (> 1 box mapping outside court +-3 ft). Over 3,721 triage frames: scorebug 18, offcourt_feet 95, no_H 119. Sparse: worst game 18% of frames (xmas16: mostly no_H + off-court feet), the held-out phx 0%.
- Rule (b) floor mask demoted to a measurement after three variants: maple HSV band fails on bright Oracle floors (V saturates at 255, S above the band) and OKC blue; ridge texture reads the sideline itself as texture (good frame far_in 0.31); adaptive per-frame colour from the court interior still puts "2 ft inside the near sideline" in the crowd for the best H's, because of the near-field overshoot, and painted aprons vary by arena. Floor shares stay in the per-frame json for B6.
- Reading: the live A2 rules catch gross failures only when the court leaves the floor; the typical 20..40 px offset passes them all. pytest 52 passed.

## Pass 4 (2026-10-03): A3 physics checks
- physics() in qc/track_qc.py, merged per frame into reports/qc/<clip>.json (track_qc.py --checks physics reuses the image-check results). Rules: cleaned speed > 30 ft/s, > 5 boxes of one team, kept id jumping > 15 ft across a camera cut (the pipeline's own vanish rule replayed from the sidecar).
- Baseline, 3,721 frames: speed fails 583 (16%; on RAW positions 49%, which matches the pipeline's own 14% impossible-step metric), team_count 893 (24%; worst game xmas16 144/242), cuts 4 with 0 kept-id jumps. The near-sideline overshoot and foot jitter through perspective are the likely sources; B5 will say which.
- Correction on record: the 2026-10-03 pre-screen proxy "gliding: essentially none" used a time step 3x too long (frame index / 10 instead of stride / fps); withdrawn in data/triage/prescreen.json. pytest 52 passed.

## Pass 5 (2026-10-03): A4 second-tracker disagreement
- qc/second_tracker.py runs COCO yolov8m person + ByteTrack over the sidecar frames (cache data/triage/side/<clip>_second.json), track_qc.py --checks second merges a per-frame disagreement (1 minus mutual IoU>=0.5 matches / max(n1, n2); fail > 0.30), per-clip median in the summary, counts in reports/qc/_by_game.json.
- Baseline, 3,721 frames: median disagreement 0.40 in every game, 70% of frames over 0.30; 7.5 pipeline vs 6.6 second boxes per frame. Loosening to IoU 0.3 (0.38) or a 25 px foot match (0.43) changes nothing, so the two detectors box different people about 40% of the time on 720p harvest footage; both are below 10 players. One inspected frame (sas_2016_s02 frame 9) shows the pipeline boxing a bench person and the referee while missing two players, the second tracker boxing a coach. SportVU (B5) decides who is right per box.
- Caveat for all triage numbers: snippets start cold, so the first second of each render is tracker warm-up; production has the same warm-up after each pregate interval and cut reset, so it is kept, and the effect measured here is small (0.38 vs 0.40). pytest 52 passed.

## Pass 6 (2026-10-03): A5 worst-frame contact sheets + QC summary
- track_qc.py --sheets: 55 clip sheets + 14 game sheets in reports/qc_sheets/ (gitignored, 28 MB); --summary: reports/qc_summary.{json,txt}. Tiles are the review render's broadcast half with second-tracker-only boxes in red, pipeline-only boxes in magenta, and a caption (frame, state, line distance, max speed, per-team counts, disagreement, failed rules).
- Glance test on game_201612250CLE.jpg: the failure mode is readable in one look: court template rotated (lines 106..135 px), teams collapsed to one label ("teams 9"), speeds to 471 ft/s, the pipeline boxing a bench person while missing the referee and two players that the second tracker finds. Lucien should look at the other 13 game sheets and note the ones that look wrong here.
- Per game (frames failing any live rule, uncalibrated): xmas16 0.94, splash62 0.91, sas_2016 0.89, hou_duel 0.88 ... phx (held out) 0.75, nyk 0.64. The live rules fire on most frames everywhere, which says the thresholds are not yet meaningful, not that 80% of production is broken; B6 sets them. pytest 52 passed.

## Pass 7 (2026-10-03): A6 QC in the triage report + Phase A baseline
- triage_sheet.py --report now prints the per-clip QC indicators (lines px, fail rate, second-tracker disagreement) next to any human label, worst first; reports/triage.json carries them under "qc".
- PHASE A BASELINE (label-free, pre-truth, 55 renders / 3,721 frames, thresholds uncalibrated): template-to-lines median 42.7 px per clip (best 27, worst 94); second-tracker disagreement median 0.40; cleaned speed > 30 ft/s on 16% of frames; > 5 per team on 24%; off-court feet 95 frames, scorebug crossing 18, no H 119 (HELD 360 frames = 9.7% still emit positions); frames failing any live rule 82% overall, held-out phx 75%. These numbers describe the indicators, not the pipeline, until B6 says which indicators track truth.
- Phase A is complete except A1 [BLOCKED]. Next pass starts Phase B (SportVU fetch).

## Pass 8 (2026-10-03): B1 SportVU fetch and parse
- sportvu/fetch.py downloads, extracts (py7zr, now in requirements.txt) and flattens one game; data under data/sportvu/ (gitignored, 100 MB raw + 30 MB moments for one game). Report: reports/sportvu_fetch_12.16.2015.PHX.at.GSW.json.
- Verified on PHX at GSW: JSON layout as the plan assumed; 25 Hz; 81,654 moments after dedupe; four quarters each spanning 720 to 0 s; every moment has 10 players; only two clock gaps over 2 s (Q3: 3.1 s and 5.3 s); coordinates in the project's corner-origin court frame (x 0..94, y 0..50; 99.9% of player positions inside +-3 ft). Roster 26 players, teams 1610612756 PHX / 1610612744 GSW. pytest 52 passed.

## Pass 9 (2026-10-03): B2 section time map
- sportvu/sync.py builds production-frame -> (period, clock) maps from the existing OCR anchors: running spans only (clock delta vs video delta within 1 s), linear interpolation, everything else excluded with a reason. Per section json under data/sportvu/sync/ (gitignored), report reports/sportvu_sync_gsw_phx_2016.json.
- phx: 10,337 of 34,425 processed frames mapped (30%), 1,056 s of running clock in 137 spans across all 10 sections; 3,552 s excluded (stopped clock between possessions, replays, 2 period changes). Coverage is lowest in s04/s06 (35 s and 32 s), highest in s02 (218 s). This is enough frames for B5 (about 10k frames x 10 players) and stays in production frame space, so B3 can run on the production trajectories without any local rebuild. pytest 52 passed.
