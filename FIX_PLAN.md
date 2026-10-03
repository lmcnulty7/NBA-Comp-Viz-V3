# Tracking fix plan (one item per loop pass, in order)

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
| gsw_phx_2016 is a held-out game never used for training | Court grid model: trained on Roboflow nbacourt + label-factory frames from two YouTube games (ids 8Ap3jYl0nAA, qAeUUwn-A8s), not phx. Detector: external set only. Teams and identity: prototype clips. BUT gate v2 (`gate_ship_v2.py`, 2026-10-02) was refit on all 6,159 harvest frames, 242 of them from gsw_phx_2016. | Corrected: held out for court, detection, teams, identity; NOT for gate v2. The gate is not what SportVU evaluates, so phx stays the held-out game, with that caveat on record. Any future gate refit excludes phx. |
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
- [ ] A4. Second-tracker disagreement: run COCO `yolov8m.pt` person class with `bytetrack.yaml`
      (both already available through ultralytics, no new dependency) on the same frames; frame
      disagreement = 1 minus the mutual IoU >= 0.5 match rate; flag frames above 0.30. Report per
      clip. Cost note: doubles detection time; run on triage renders locally, on sections in Colab.
- [ ] A5. Worst-frame contact sheets: for each clip the 6 frames with the most failed checks,
      overlay plus a caption listing which checks failed, one jpg per clip, one summary sheet per
      game (`reports/qc_sheets/`). Then `track_qc.py --summary` writes `reports/qc_summary.*`
      with per-clip and per-game scores. Measurable: a glance at the 14 game sheets replaces labeling;
      record in FIX_LOG which sheets look wrong to you.
- [ ] A6. Wire `triage_sheet.py --report` to include the QC scores next to any human labels, and
      log the QC numbers for all 55 renders as the Phase A baseline (label-free, pre-truth).

## Phase B: ground truth from SportVU on the held-out game
Artifacts: `sportvu/` package (`fetch.py`, `sync.py`, `eval.py`), `data/sportvu/` (gitignored),
`reports/sportvu_phx.{json,txt}`, `reports/sportvu_check_validation.{json,txt}`.

- [ ] B1. `sportvu/fetch.py`: download one game archive from the GitHub raw URL, extract (7z: use
      `py7zr`, add to requirements), parse to a flat table of moments: quarter, game_clock_s,
      shot_clock, 10 player rows (team_id, player_id, x_ft, y_ft) plus ball. Dedupe repeated moments
      across events on (quarter, game_clock_s, first player position). Verify on first pass: the JSON
      layout is events -> moments -> [quarter, timestamp_ms, game_clock, shot_clock, None, entities],
      25 Hz, court coordinates 0..94 by 0..50 ft. Report per quarter: moments, coverage seconds, gaps
      > 2 s. Run on 12.16.2015.PHX.at.GSW.
- [ ] B2. `sportvu/sync.py`: section time map from the existing clock anchors
      (`data/pbp/<section>_outcomes.json` anchors: frame, period, clock_s). Between two anchors
      whose clock delta matches the frame delta within 1 s the clock is running: interpolate.
      Otherwise mark the span stopped and exclude it. Output per section: list of (frame, period,
      game_clock_s) for processed frames, plus how many seconds were excluded and why.
- [ ] B3. Direction and offset per section: for each of the four mirror candidates (identity, flip x,
      flip y, both) and each sub-second offset in 0.04 s steps over +-1.0 s, project pipeline feet
      through the pipeline H and compute the median nearest-neighbour distance to SportVU players.
      Keep the best; require the runner-up candidate to be worse by >= 2 ft or mark the section
      ambiguous. Report chosen offsets and ambiguities. gsw_phx_2016 has 10 sections; expect all to resolve.
- [ ] B4. Per-frame truth H: Hungarian-match pipeline feet (pixels) to SportVU players through the
      pipeline H, then RANSAC an H_truth from pixel feet to SportVU feet at a 1.5 ft threshold; accept
      with >= 6 inliers. Frames with fewer are "untestable" and counted. Caveat on record: H_truth
      passes through the pipeline's own boxes, so a frame where every box is wrong has no truth.
- [ ] B5. `sportvu/eval.py` on gsw_phx_2016: position error in ft per matched player (p50, p90, per
      court region: near third, middle, far third); frame H error = median over matched players;
      missed players = SportVU players projecting inside the frame with no box foot within 2.5 ft;
      ghost boxes = boxes whose foot is > 3 ft from every SportVU player (refs are not in SportVU:
      report ghosts split by the detector's referee class); team accuracy = pipeline A/B vs SportVU
      team_id under the majority mapping per section; nearest-defender agreement = for each
      offensive player, the pipeline's assigned defender vs the SportVU nearest opponent (this is a
      proxy for CLAIMS C1, labeled as such, not the human-labeled matchup). Write `reports/sportvu_phx.*`
      and a CLAIMS.md Tier A row "A8 (proposed)" with every caveat above.
- [ ] B6. Validate the Phase A checks against truth: per check, precision and recall for flagging
      frames whose SportVU H error > 3 ft, or with a missed or ghost player. Thresholds fixed BEFORE
      looking: drop a check with recall < 0.50 or precision < 0.30; keep the rest. Report in
      `reports/sportvu_check_validation.*`; update Phase C rules to use only surviving checks.
      Thresholds for the surviving checks are the ones chosen in Phase A on the other games; do not
      re-tune them on phx.
- [ ] [HUMAN] Lucien reads `reports/sportvu_phx.txt` and the validation, and confirms the Phase C order.

## Phase C: fixes, each adopted only if it improves Phase B on gsw_phx_2016
Acceptance for every item: SportVU position error p50 and p90 do not get worse, the share of
frames with positions does not drop more than 25% in any game unless those frames were wrong,
the rejection counts per rule per game are logged, and `align_outcomes` on gsw_phx_2016 yields
the same or more aligned possessions. Each fix is a flag in config (default off) until adopted.

- [ ] C1. Stricter H acceptance on the model path in `snap_track.update`: after `_snap`, require
      the surviving Phase A rules (overlay score, floor mask, scorebug, residual <= 1.1 px at 640
      scale per `label_factory` convention). Failing H is treated as "model failed" and falls to the
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
