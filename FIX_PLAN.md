# Tracking fix plan (one item per loop pass, in order)

> **2026-10-03: Phases C and D are superseded by `ROADMAP.md`.** Phases A and B stay as the record of
> done work and their tools (`qc/`, `sportvu/`) carry over. Do not run C or D items from this file.

Goal: make the court projection and the player boxes right on the 14 production games,
measure it against real tracking data instead of hand-labeled frames, and only then
retrain. Loop instructions live in `FIX_LOOP.md`; the pass-by-pass record in `FIX_LOG.md`.
Branch: `track-fix`. Written 2026-10-03 after reading the code; nothing here is a result yet.

## Diagnosis checked against the code (read this before any pass)

| Observation (gsw_bos_2016 triage render) | What the code says | Verdict |
|---|---|---|
| A plausible but wrong H passes because `h_sane` only checks fold + 6..29 vertices on frame | `court/snap_track.py:131` `h_sane`: vertex count in (5, 30) and a Jacobian sign test at the four frame corners. Nothing else. On the model path (`update`, line 243) a sane H is accepted with NO check on the snap result: `_snap` refines within 8 then 5 px, reverts if corners move > 40 px, and the residual it returns is only stored as `quality`. An H that is off by more than the search radius is kept as is. | Confirmed, and worse than stated: there is no pixel-evidence acceptance test on the model path at all. The LINE_TRACK path does have one (`MIN_TRACK_MATCH` 50, `MAX_TRACK_RES` 2.5 px). |
| Frame 9 has no tracking history, so nothing could reject it | The tracker is forward only, and a sane model H re-anchors the track on every frame regardless of history (line 244 runs before the propagate branch). History would not have saved frame 9; a wrong model H wins on any frame. | Corrected: history is not the cause. The missing piece is an acceptance test, not memory. A backward pass still helps the frames BEFORE the first good H. |
| Oracle's center logo and wordmark add paint the template lacks | `ridge_field` zeroes texture where ridge density > 0.20 (crowd) but logo edges are sparse ridges and survive. The template (`court33_segments`, `court33_curves`) has no logo chords, so logo ridges inside the search radius can pull sideline and circle samples. | Plausible, unmeasured. Phase A reports overlay score and rejection counts per arena so this becomes a number. |
| Player positions wrong, inherited from the H | `build_trajectories.py:224` records any projection inside court +-25 ft (`X_LO/X_HI`). `CourtMapper.is_inside_court` is only consulted by `split_tracks`, which the build loop never calls. | Confirmed. Off-court feet are kept up to 25 ft out. |
| HELD frames | `snap_track.py:283` returns an H with quality 9.9 on HELD; `mapper.has_homography` is True for it, so positions are recorded. They are flagged `extrap` (hull is empty on HELD), and that flag is written to trajectories but read by nothing downstream (`segment_possessions.py`, `matchup_metrics.py` never touch it). | Not in the diagnosis; added. "Rejected frame = LOST, no positions" is currently false for up to 20 consecutive frames (`COURT_TRACK_MAX_HELD`). |
| Near sideline (my addition from the 55-still pre-screen, `data/triage/prescreen.json`) | In nearly every clip the far sideline and the key sit on the paint while the near sideline is drawn into the crowd. The yellow hull (points supporting H) is always in the far half; the near side is extrapolation. | New systematic error, outside the evidence hull. Shows up as a y-dependent position bias. SportVU will measure it. |
| Boxes merge a player and a referee; crouching player missed; bench and fans boxed | Detector is `models/player_detector.pt` (basketball set, class 0 player, class 1 referee; the tracker takes class 0 only). Eval on hand boxes: P .89 / R .87 (PIPELINE row 2). No pose model anywhere in the pipeline. `FootPointStabilizer` corrects clipped boxes by height median, nothing about two people in one box. | Confirmed as detector errors, not tracker. Pose split and ankle feet are new capability, Phase C6. |
| gsw_phx_2016 is a held-out game never used for training | Court grid model: trained on Roboflow nbacourt only (data/court_pose_grid, per the checkpoint's train_args; corrected 2026-10-04 at ROADMAP R2.3: the label-factory frames from 8Ap3jYl0nAA and qAeUUwn-A8s went into court_pose_grid_v2, which this model did not use), not phx. Detector: external set only. Teams and identity: prototype clips. BUT gate v2 (`gate_ship_v2.py`, 2026-10-02) was refit on all 6,159 harvest frames, 242 of them from gsw_phx_2016. | Corrected: held out for court, detection, teams, identity; NOT for gate v2. The gate is not what SportVU evaluates, so phx stays the held-out game, with that caveat on record. Any future gate refit excludes phx. |
| SportVU covers 12.16.2015 PHX at GSW | GitHub API listing of `linouk23/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs`: 636 games, `12.16.2015.PHX.at.GSW.7z` present, 20 games at GSW (incl. `12.25.2015.CLE.at.GSW.7z`), 17 CLE home, 25 OKC home, 22 NYK home. Of the other 13 production games none fall in the window (gsw_sas_2016 is 2016-01-25, two days late). | Confirmed. |

Also on record: the production harvest gated at v1 0.70 (DEVLOG 10-02e), the triage renders
reproduce that (`triage_sheet.py`, `--pregate`), and local section files need the clock-solved
frame offset (`data/triage/offsets.json`) before any production frame number is used locally.

## Ground rules (every pass)
- CLAIMS.md discipline: a number exists only with its artifact path and its caveat. Nothing is a
  claim until measured on gsw_phx_2016 against SportVU; before that it is a "label-free indicator".
- Never train, tune, calibrate or pick thresholds on gsw_phx_2016. Thresholds for the automatic
  checks are chosen on the other 13 games or on prototype clips, then reported on phx once.
- Every acceptance rule reports rejection counts per rule per game. A fix that drops more than
  25% of frames that previously had positions in any game fails unless SportVU shows those frames
  were wrong.
- Run locally with `/opt/anaconda3/bin/python` and `PYTORCH_ENABLE_MPS_FALLBACK=1`; GPU steps
  (any model training, full-game rebuilds) go to Colab through a repo script behind a one-cell
  notebook, per `colab_run.py` conventions. Local section files are AV1 (ffmpeg decodes, cv2
  via the existing two-step open); production frame numbers only index the h264 sections.
- Third-party data (SportVU JSON, Roboflow, SportsMOT, DeepSportRadar) lives under
  `data/external/` or `data/sportvu/`, gitignored. Download scripts and manifests are committed;
  data never is.
- No em dashes in any written output.

## Phase A: automatic quality checks (replace frame-by-frame review)
Decision: a new module `qc/track_qc.py` with pure functions over the build outputs
(`*_trajectories.json`, per-frame H, boxes) plus a CLI `track_qc.py`. `triage_sheet.py --report`
and `harvest_driver` call into it; `triage_sheet.py` stays the renderer and human reviewer.
Reason: the checks must run on harvest sections on Colab, not only on triage renders.
Artifacts: `reports/qc/<clip>.json`, `reports/qc_summary.{json,txt}`, `reports/qc_sheets/<clip>.jpg`.

- [x] A0. Make the build keep what QC needs: `build_trajectories.py` writes a sidecar
      `<clip>_frames.json` with, per processed frame: tracker state (TRACK / LINE_TRACK / HELD / LOST),
      H (3x3), snap residual px, match count, hull points, boxes with track id and raw foot pixel.
      Measurable: sidecar present for all 55 triage renders after a re-render; byte-identical
      trajectories to the committed ones (no behaviour change). Note: this is the one Phase A item
      that touches pipeline code; it adds output only.
- [ ] [BLOCKED: 2026-10-03, 4 variants; clip ranking puts 7/11 pre-screen flags in the bottom half,
      see FIX_LOG pass 2; the score ships as a ranking tool and is calibrated or dropped in B6]
      A1. Line-overlay score per frame: implemented in `qc/track_qc.py` as the ridge-side distance
      (median px from detected line pixels outside player boxes to the projected template; score =
      share within 6 px; template-side support as a secondary number). Reports in `reports/qc/`.
      Original spec (3 px hit rate on template samples) saturated at 0.02 for every clip because
      production H's are 10..40 px off the paint nearly everywhere.
- [x] A2. Geometry checks per frame (`qc/track_qc.py: geometry`): (a) far sideline or either baseline
      crosses the scorebug rectangle (layout clock+period boxes, 20 px pad); (c) more than one box whose
      foot maps outside court +-3 ft. Rule (b) floor mask is MEASURED but does not fail a frame: three
      variants (maple HSV band, ridge-texture, adaptive per-frame colour) could not separate right from
      wrong H's on the 55 renders (FIX_LOG pass 3); B6 decides if any floor rule survives. Counts per rule
      per clip in `reports/qc/<clip>.json`, per game in `reports/qc/_by_game.json`.
- [x] A3. Physics checks (`qc/track_qc.py: physics`): speed > 30 ft/s between consecutive processed
      frames on the CLEANED positions (failing rule; raw reported as a measurement), more than 5 boxes
      of one team per frame, kept-id jump > 15 ft across a replayed camera cut. Counts per clip and per
      game in `reports/qc/`. Baseline on the 55 renders: cleaned speed fails on 16% of frames (raw 49%),
      team_count on 24%, 4 cuts with 0 jumps.
- [x] A4. Second-tracker disagreement (`qc/second_tracker.py`): COCO `yolov8m.pt` person + `bytetrack.yaml`
      on the same frames, cached as `<clip>_second.json`; per frame 1 minus mutual IoU>=0.5 matches over
      max(n_pipeline, n_second); fails above 0.30. Baseline on the 55 renders: median disagreement 0.40 on
      every game (0.38 at IoU 0.3, 0.43 on a 25 px foot match, so it is different people, not box extents);
      7.5 vs 6.6 boxes per frame; 70% of frames above 0.30. Not a warm-up effect (first second 0.38 vs 0.40).
      Which tracker is wrong is B5's question. Cost: ~10 min for 3,721 frames on MPS.
- [x] A5. Worst-frame contact sheets: `track_qc.py --sheets` writes `reports/qc_sheets/<clip>.jpg` (6 worst
      frames by failed-check count, then line distance, then disagreement; review-render tile + red boxes
      only the second tracker has + magenta boxes only the pipeline has + caption) and `game_<id>.jpg` (6
      worst across the game); gitignored, regenerable. `track_qc.py --summary` writes `reports/qc_summary.*`
      (per clip and per game, worst first). Lucien: the 14 game sheets are the glance; note which look wrong
      in FIX_LOG.
- [x] A6. `triage_sheet.py --report` lists the label-free QC per clip (line distance, fail rate, second-tracker
      disagreement) beside any human label; `reports/triage.*`. Phase A baseline logged in FIX_LOG pass 7.

## Phase B: ground truth from SportVU on the held-out game
Artifacts: `sportvu/` package (`fetch.py`, `sync.py`, `eval.py`), `data/sportvu/` (gitignored),
`reports/sportvu_phx.{json,txt}`, `reports/sportvu_check_validation.{json,txt}`.

- [x] B1. `sportvu/fetch.py` (`python -m sportvu.fetch <game>`): downloads the .7z from the GitHub raw URL,
      extracts with py7zr (added to requirements), parses and dedupes moments to `data/sportvu/<game>_moments.json`
      (gitignored), reports per quarter to `reports/sportvu_fetch_<game>.json`. Verified on 12.16.2015.PHX.at.GSW:
      layout as expected (events -> moments -> [q, ts, clock, shot, None, 11 entities]), 25 Hz, 81,654 deduped
      moments, every quarter 720 -> 0 s with 10 players per moment, two gaps > 2 s (Q3 626.0..622.9 and
      178.4..173.1), player x in -5..99 ft and y in -3..53 ft (99.9% inside court +-3 ft).
- [x] B2. `sportvu/sync.py` (`python -m sportvu.sync gsw_phx_2016`): per section, consecutive OCR anchors
      (from `data/pbp/<section>_outcomes.json`, production frame numbers) whose clock delta matches the video
      delta within 1 s define running spans; processed frames (those with raw positions in the production
      trajectories) inside them get an interpolated (period, clock). Everything else excluded and counted.
      phx: 232 anchors, 34,425 processed frames, 10,337 mapped (1,056 s in 137 running spans), 3,552 s excluded
      (88 spans: stopped clock or disagreement, 2 period changes). `data/sportvu/sync/<section>_timemap.json`,
      `reports/sportvu_sync_gsw_phx_2016.json`.
- [x] B3. Direction and offset (`python -m sportvu.sync gsw_phx_2016 --direction 12.16.2015.PHX.at.GSW`):
      four mirrors x offsets in 0.04 s steps over +-1 s, scored by the median nearest-SportVU-player distance
      of the production positions over mapped frames. phx, all 10 sections: identity wins; flip_x is 30+ ft
      worse; flip_y is only 0.4..1.7 ft worse (under the 2 ft rule, so each section reads "ambiguous"), and
      still only 1..3 ft worse on positions > 12 ft from the mid-line. Resolved at game level: 10/10 sections
      pick identity under both statistics. Offsets per section -0.60..+0.84 s. First real number: median
      nearest-neighbour distance 4.6..6.9 ft per section, and the weak y-flip signal means the pipeline's y
      coordinates are compressed toward mid-court (consistent with the near-sideline overshoot). B5 quantifies.
- [x] B4. Per-frame truth H (`sportvu/truth.py`), on a LOCAL windowed rebuild because production never
      saved pixel feet and local sections are not frame-aligned with production (anchors re-read 4..9 s off,
      drifting). Chain: `sportvu.local_sync` (OCR clock at 1 Hz per section, running spans) ->
      `sportvu.rebuild` (25 windows, 1,034 s, longest spans up to 100 s per section, ffmpeg cut +
      build_trajectories --pregate at the harvest stride, 27 min on MPS) -> `sportvu.local_sync --windows`
      (OCR on the window files themselves: an ffmpeg -ss cut starts a keyframe early, +4.5..+7 s measured)
      -> `sportvu.truth` (residual offset per window within +-1.5 s, then ICP-style Hungarian match at gates
      20/12/8/5 ft with RANSAC refits, final RANSAC 1.5 ft, >= 6 inliers). phx: 9,500 window frames; 1,447 (15%)
      with a truth H (24/25 windows, truth residual median 0.33 ft); untestable: 3,961 too few inliers,
      1,155 too few matches, 1,835 fewer than 6 boxes, 33 no H; 1,064 outside the mapped spans.
      Caveat on record: truth passes through the pipeline's own boxes and starts from its H.
- [x] B5. `sportvu/eval.py` on gsw_phx_2016 (`reports/sportvu_phx.{json,txt}`; CLAIMS.md row A8 PROPOSED with
      every caveat): 1,447 testable frames, 11,721 matched players: position error p50 7.31 ft / p90 15.38 ft,
      frame H error p50 7.44 ft; by region far 7.92 / middle 7.44 / near 6.85 ft with bias toward the camera
      +1.69 (far) / +0.58 / -1.04 (near) ft, i.e. compression toward mid-court; missed players 25.1% of those
      in frame; ghost boxes 15.3% (referees count as ghosts); team accuracy 81.8% (n=11,240); nearest-defender
      proxy 48.1% agreement (160 of 224 defenders mapped, via segment_possessions + matchup_metrics on the windows).
- [x] B6. Validate the Phase A checks against truth (`sportvu/validate_checks.py`, `reports/sportvu_check_validation.*`,
      1,447 testable phx frames). Degenerate under the plan's definition: wrong_any = 1.00 (H > 3 ft on 95%,
      a missed player on 95%, a ghost on 74%), so precision is trivially 1.0 and the pre-set rule keeps
      A1 at 30/40 px and A4 at 0.30 only because they flag most frames; A2 scorebug (2 flags), A2 off-court
      feet (26), A3 speed, A3 team count, A3 cut jump (0) and A4 at 0.50 are DROPPED by recall. Graded view:
      at H > 10 ft (base rate 0.29) the best check, A1 >= 50 px, reaches precision 0.43 / recall 0.47; A1's
      rank correlation with H error is 0.27; A4 has no lift against H error. Consequence for Phase C: no
      label-free check can carry an acceptance rule on its own; the SportVU p50/p90 is the arbiter and A1 is
      a weak prior. A2/A3 stay as reported measurements, not rules.
- [ ] [HUMAN] Lucien reads `reports/sportvu_phx.txt` and the validation, and confirms the Phase C order.

## Phase C: fixes, each adopted only if it improves Phase B on gsw_phx_2016
Acceptance for every item: SportVU position error p50 and p90 do not get worse, the share of
frames with positions does not drop more than 25% in any game unless those frames were wrong,
the rejection counts per rule per game are logged, and `align_outcomes` on gsw_phx_2016 yields
the same or more aligned possessions. Each fix is a flag in config (default off) until adopted.

- [ ] C1. Stricter H acceptance on the model path in `snap_track.update`: after `_snap`, require
      the surviving Phase A rules (B6: only A1 line distance survived, and weakly; plus the solver's own
      residual <= 1.1 px at 640 scale per `label_factory` convention and a minimum match count). Judge
      by SportVU p50/p90 on phx, not by the checks. Failing H is treated as "model failed" and falls to the
      propagate branch. Report per game: frames by outcome (accepted, relocked, held, lost).
- [ ] C2. Rejected means LOST: `build_trajectories` records no position when the tracker state is
      HELD (quality 9.9). Check `segment_possessions.py` tolerates the gaps (it already handles
      missing frames between pregate intervals; confirm with the tests). Measure on phx: positions
      removed, SportVU error of the removed positions vs the kept ones (removed should be worse).
- [ ] C3. Mirrored-template test: score the accepted H and its left-right mirror with the overlay
      score; if the mirror scores within 10%, the frame is ambiguous: keep the one consistent with
      the previous accepted frame, else LOST. Report ambiguity rate per arena.
- [ ] C4. Backward pass: run the tracker backward from the first accepted frame of each pregate
      interval over the frames before it, keep positions only where the backward H passes C1.
      Cost: a second tracker pass over at most `COURT_TRACK_MAX_HELD` frames per interval.
- [ ] C5. Drop boxes whose feet map more than 3 ft outside the court under an accepted H (replaces
      the +-25 ft window), and never record a position for them. Report drops per game and the
      share that SportVU says were real players (should be near zero).
- [ ] C6. Pose feet: `yolov8n-pose.pt` on each box crop; if both ankles have conf >= 0.5 the foot
      is the ankle midpoint, otherwise the stabilized bottom-centre as now. A box with two skeletons
      whose hips are > 25% of the box width apart is split into two boxes. Flag off by default.
      Measure on phx: position error p50/p90, missed and ghost rates. GPU helps; MPS with fallback works.
- [ ] C7. Near-sideline bias: report SportVU error by court y (near third vs far third) before and
      after C1..C6. If the near-third bias remains > 2 ft, record it as the open item that Phase D's
      SportVU grid labels are meant to fix, and say so in PIPELINE.md row 3.
- [ ] C8. Re-run the full phx section builds with adopted flags on Colab, then the PBP canary for
      phx and the 13 other games. Canary must not fall. Update CLAIMS.md and PIPELINE.md rows 2..4.
- [ ] [HUMAN] Lucien approves which C flags become defaults.

## Phase D: retraining data (only after C)
- [ ] D1. Game availability pass: for the 19 other GSW home games in the SportVU window, plus the
      CLE, OKC and NYK home games, check (a) a full broadcast exists via `yt-dlp` (avc1, 720p, per
      `harvest_driver` fetch rules), (b) the SportVU file has no gaps > 10 s in any quarter. Write
      `data/sportvu/manifest.json` with the verdicts. No downloads of video yet beyond the probe.
      Priority list: 12.25.2015 CLE at GSW first, then the other Oracle games, then CLE, OKC, NYK home.
- [ ] [HUMAN] Lucien picks which games to download and names ONE extra held-out game (an Oracle
      game other than phx) that no training step may touch.
- [ ] D2. Fetch, register (`data/harvest/games.json`), section-split and clock-calibrate the chosen
      games with the existing harvest tooling (Colab). Sync them with B2/B3.
- [ ] D3. Auto-labels from SportVU: per synced wide frame, H_truth (B4) projects the 13x7 grid
      (`court/grid.py`) to pixel keypoints, giving court-grid labels with near-side points the line
      factory never had; SportVU-confirmed boxes (foot within 2.5 ft of a true player) become
      detector labels, misses become review tiles. Accept only frames passing the Phase A checks.
      Report accepted frames per game and per court region.
- [ ] D4. Label-factory harvest on the 14 production games with the Phase A rules as the accept
      filter (replacing the fixed thresholds in `label_factory.py`); exclude gsw_phx_2016 and the
      extra held-out game. Report accept rates per game.
- [ ] D5. External sets as a minority mix (<= 30% of training frames): download scripts and
      license notes only for Roboflow basketball-player-detection-3 and its jersey-number set,
      Roboflow Universe court-keypoint sets (filtered through the line-snap accept gate),
      SportsMOT basketball (CC BY-NC), DeepSportRadar (CC BY-NC-SA). Nothing committed.
- [ ] D6. Retrain the court grid model on Colab (existing notebook convention, `DATASET` switch);
      evaluate on the 279 original val frames, the 28 held-out frames, AND SportVU error on phx
      and the extra held-out game. Adopt only if all improve or hold (LABEL_SCHEMA rule 4).
- [ ] D7. Retrain the detector on Colab (`colab_train_player.py`); evaluate on `box_truth`
      (P .89 / R .87 baseline) AND SportVU missed and ghost rates on both held-out games.
- [ ] [HUMAN] Lucien adopts or rejects each retrained model.
- [ ] D8. Re-harvest the 14 games under adopted models (and gate v2), re-run the PBP canary and
      the bias audit, update CLAIMS.md, PIPELINE.md, DEVLOG.md, the paper sections 4 and 9, and
      the portfolio numbers.

## Waiting on Lucien
- [x] [HUMAN] OK to download SportVU data from the GitHub repo into `data/sportvu/` (gitignored).
- [x] [HUMAN] Confirm `track-fix` as the branch name and that main stays untouched until C8.
