# ROADMAP.md: broadcast tracking, rebuilt on a real scoreboard

Written 2026-10-03. Replaces FIX_PLAN.md Phases C and D (A and B stay as done work). Run it with
FIX_LOOP.md: same rules, with ROADMAP.md as the plan and ROADMAP_LOG.md as the log. One item per
pass, measurable before the next, [HUMAN] gates where Lucien decides. Branch: `track-fix` until
Phase 5 says otherwise. No em dashes in anything written.

## North star

Turn one TV broadcast into a Hawk-Eye-style feed: every visible player's court position, team
and identity, frame by frame, with the error measured against real tracking data and printed on
the product. Stage 1 asks "how good is broadcast tracking". Stage 2 (team defense, matchups) and
Stage 3 (shot quality, scheme quality models) start only when Stage 1 hits its targets. The
coursework angle lives inside Stage 1 as methods (foundation-model embeddings for teams and
identity, vision-language models for jerseys, synthetic data for occlusion), each adopted only if
it beats the baseline on the scoreboard.

## Where the project stands (measured 2026-10-03, FIX_PLAN Phase B)

Held-out game gsw_phx_2016 vs SportVU, on the 15% of frames where truth could be fitted at all:
position error p50 7.3 ft / p90 15.4 ft, compressed toward mid-court (far third +1.7 ft toward the
camera, near third -1.0 ft away); 25% of in-frame players missed; 15% of boxes ghosts (referees
included); team labels 82%; nearest-defender agreement 48%. The earlier stage numbers (0.30 ft
court median, 98.7% gate, P.89/R.87 detection) were measured on the 7 prototype clips the models
were tuned on; the 87.8% play-by-play canary only checks which team scored, so it never saw
position error. Court mapping is the dominant error and is not solved: the template sits 20..40 px
off the paint on nearly every production clip, anchored on the far sideline, with the near side
extrapolated from keypoints that only ever lie in the far half.

## Definition of done for Stage 1 (targets, measured on held-out footage)

| Metric | How measured | Target |
|---|---|---|
| Court line error | px between projected template and painted lines, from the SportVU truth H (B4) and the line field | <= 3 px (about 0.5 ft) on 95% of accepted wide frames |
| Position error, visible players | pipeline court position vs SportVU, all matched players | p50 <= 2.0 ft, p90 <= 5.0 ft |
| Near-field bias | median error toward the camera, near third vs far third | within +-0.5 ft |
| Coverage | share of live wide-shot seconds with accepted positions | reported; >= 50% of live play |
| Missed players | SportVU players in frame with no box within 2.5 ft | <= 5% |
| Ghost boxes | boxes > 3 ft from any player, referees excluded by class | <= 5% |
| Team labels | vs SportVU team of the matched player | >= 95% |
| Identity | id switches per possession vs SportVU player ids; jersey-read rate | reported; no target yet (720p ceiling) |
| Generalisation | the same metrics on an arena never trained on, and an era never trained on | within 1.5x of the held-out game |

Physical ceiling for one 720p camera: about 1.5 ft median. Hawk-Eye (14 cameras, about 1 inch)
is matched in output format and honesty, not in accuracy; say so on the product.

## What stays, what goes

Keep: the measurement discipline (CLAIMS.md, audits, corrections), the SportVU scoreboard built in
FIX_PLAN Phase B (`sportvu/`: fetch, sync, local_sync, rebuild, truth, eval, validate_checks), the
label-free QC (`qc/`, a ranking tool only), the play-by-play alignment, the possession and stats
layer (reattached in Phase 5), the harvest tooling and game registry.
Freeze: V3's court solver, detector training set, team classifier, and the stats on top. Tag it;
nothing new is built on it. Extend nothing in the stats layer until Phase 5.
Replace: the foundation (court, detection and tracking, teams, identity) as one clean package,
each stage measured on the scoreboard before and after, starting from an off-the-shelf baseline.

## Data: how much, and why it is free

SportVU gives 10 labelled players per frame for free once a game is synced to its broadcast.
One synced game yields about 1,500..3,000 wide frames with court positions, team and identity
truth; the court truth H turns each into a court keypoint label with near-side points. Sizing:
about 10 synced Oracle games (20 exist) gives ~30k truth frames, enough to train the court model and
detector for that camera and measure honestly; other arenas need their own games (SportVU 2015-16
covers all 30 arenas' home games: CLE 17, OKC 25, NYK 22 in the window). Hold out whole arenas, not
only games. Third-party data stays under `data/sportvu/` and `data/external/` (gitignored);
download scripts, manifests and licence notes are committed.

## Rules (every pass)
- Never train, tune or pick a threshold on gsw_phx_2016, on the held-out arena, or on the held-out
  era. Report them once per adoption decision.
- Every number ships with its artifact path and caveat (CLAIMS.md discipline). Nothing is a claim
  until measured on held-out footage; before that it is an indicator.
- Every acceptance rule logs rejections per rule per game; rejected means no positions, never
  wrong positions.
- A stage is adopted only if it beats the current scoreboard on ALL target metrics it touches and
  does not worsen the others.
- Training runs on Colab through a repo script behind a one-cell notebook; everything else local
  with `/opt/anaconda3/bin/python` and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Known traps (FIX_LOG passes 11..13): local section files are not frame-aligned with production
  (OCR the clock, never trust frame numbers across copies); an ffmpeg `-ss` cut can start a keyframe
  early (OCR the cut file itself); gate v2 saw phx frames (court, detector, teams, identity did not);
  the clock OCR resolves 1 s, so solve a residual offset against SportVU per window.

## Phase 0: freeze and reframe
- [x] R0.1 Tag V3 (`v3-frozen` on main) and add a README section: what V3 is, what is reused, where the
      scoreboard lives. No code change.
- [x] R0.2 Write the targets table above into CLAIMS.md as "Stage 1 definition of done", with the
      measurement protocol (held-out game, held-out arena, held-out era) and the single scorecard path.
- [ ] R0.3 Scorecard CLI `python -m sportvu.bench <build_dir>`: one json + txt with every target metric
      on gsw_phx_2016. Reuse eval.py; add coverage (accepted wide seconds / live wide seconds from the
      local time map), referee split (run the detector's referee class on the sidecar boxes), identity
      switches per possession (track to player mapping from the truth pairs), court line px error.
      Measurable: the V3 scorecard printed, every metric filled or marked not measurable with a reason.
- [ ] [HUMAN] Lucien confirms the targets or changes them. Nothing below starts before this.

## Phase 1: SportVU-synced data
- [ ] R1.1 Availability manifest: for the 20 Oracle games plus CLE, OKC and NYK home games in the SportVU
      window, probe a full broadcast with yt-dlp (avc1, 720p, per harvest_driver rules) and the SportVU
      file for gaps > 10 s per quarter. `data/sportvu/manifest.json` with verdicts and priority
      (12.25.2015 CLE at GSW first, then Oracle, then the other arenas). No video downloads beyond probes.
- [ ] [HUMAN] Lucien picks the games to download, names the held-out ARENA (one of CLE, OKC, NYK) and
      the held-out ERA (the 2013 NYK production game stays untouched by training), and confirms
      gsw_phx_2016 stays the held-out game.
- [ ] R1.2 Fetch, register (`data/harvest/games.json`), split and clock-calibrate the chosen games with
      the harvest tooling; OCR time maps per section (`sportvu.local_sync`); SportVU moments fetched.
      Measurable: per game, mapped running seconds and the mirror/offset resolution (`sportvu.sync`).
- [ ] R1.3 Auto-labels: on every synced wide frame (gate v2, frames outside the held-out sets), fit the
      truth H with the ICP matcher (`sportvu.truth`, bootstrapped from the current detector's feet),
      reject frames with truth residual > 0.5 ft or < 6 inliers, then write (a) court keypoints: the
      13x7 grid and 1 ft line samples projected through H_truth, (b) SportVU-confirmed boxes (foot within
      2.5 ft of a player) with team id and player id, (c) a contact sheet per game of projected lines
      over the paint for a glance check. Measurable: `data/sportvu/labels/manifest.json` with frame
      counts per arena, per court region, and the rejection counts per rule.
- [ ] R1.4 Splits written to `data/sportvu/splits.json`: train arenas, held-out arena, held-out game,
      held-out era, with per-split frame counts. Every training script reads this file and refuses
      to run on held-out ids.

## Phase 2: court (first priority; nothing else is meaningful until this holds)
- [ ] R2.1 Baseline scorecard of the V3 solver on gsw_phx_2016 (the R0.3 output) recorded as the number
      to beat; also run it on 20 frames of the held-out arena from the production clips for a
      generalisation baseline.
- [ ] R2.2 Off-the-shelf baseline: run one public court/field registration model (candidates: Roboflow
      Universe basketball court keypoints; a sports field registration network) on the same frames.
      Scorecard. The better of V3 and the public model is the starting point. Measurable: two
      scorecards side by side.
- [ ] R2.3 Train the line/keypoint detector on the R1.3 labels (Colab, held-out splits enforced).
      Scorecard on the held-out game AND the held-out arena. Adopt by the rule.
- [ ] R2.4 Camera model: per game, estimate the camera position once from the first minutes (self-
      calibration from detected lines), then fit pan, tilt, zoom per frame; the homography is derived,
      never free. Scorecard; must cut the near-field bias and the near-third error. Adopt by the rule.
- [ ] R2.5 Acceptance and LOST: a frame is accepted only when the line residual and the line-overlay
      score pass thresholds chosen on train arenas; otherwise no positions. Report coverage and
      rejections per rule per game. Measurable: p90 falls and coverage is reported, on held-out footage.
- [ ] R2.6 Temporal smoothing of camera parameters across frames (pan, tilt, zoom are smooth; cuts
      reset). Scorecard.
- [ ] R2.7 Generalisation report: the court metrics on the held-out arena and the held-out era, printed
      per arena. If either is worse than 1.5x the held-out game, add that arena's games to R1 (not the
      held-out one) and repeat R2.3.
- [ ] [HUMAN] Lucien accepts the court stage when the court rows of the targets table are met.

## Phase 3: detection and tracking
- [ ] R3.1 Baseline scorecard: missed, ghosts split by referee class, id switches per possession, on the
      held-out game under the accepted court stage.
- [ ] R3.2 Off-the-shelf comparison: current player_detector.pt (trained on Roboflow
      basketball-player-detection-3) vs one modern detector on the same frames (candidates: RF-DETR or
      YOLO11 fine-tuned on the same set). Scorecard. The better one is the starting point.
- [ ] R3.3 Train the detector on SportVU-confirmed boxes (R1.3) plus the external set as a minority
      mix (<= 30%); referee as its own class. Scorecard on held-out game and arena. Adopt by the rule.
- [ ] R3.4 Foot point: pose ankles (yolov8n-pose) vs box bottom; position error on held-out. Adopt if
      p50 improves without raising missed.
- [ ] R3.5 Tracking: BoT-SORT vs ByteTrack vs with appearance ReID, measured by id switches per
      possession against SportVU ids (track to player mapping from the truth pairs). Adopt by the rule.
- [ ] [HUMAN] Lucien accepts the detection rows of the targets table.

## Phase 4: teams and identity
- [ ] R4.1 Teams: train the kit classifier on SportVU team labels from R1.3 (crops to team), compare
      with a foundation-model embedding + linear head. Scorecard >= 95% on held-out. Adopt by the rule.
- [ ] R4.2 Identity: SportVU player ids as labels; jersey OCR (and a vision-language reader as the
      coursework method) plus ReID across possessions. Report the read rate and switches honestly; the
      720p ceiling is part of the product text.
- [ ] [HUMAN] Lucien accepts the team and identity rows.

## Phase 5: product and reattachment
- [ ] R5.1 Demo: broadcast | top-down feed | SportVU truth side by side for gsw_phx_2016, with the
      running position error printed on the frame. This is the honest Hawk-Eye comparison.
- [ ] R5.2 Re-harvest the 14 production games with the new foundation (Colab); reattach possessions,
      play-by-play alignment and the stats layer; re-run the canary and the bias audit.
- [ ] R5.3 Update CLAIMS.md (A8 becomes a shipped claim with the final numbers; A2's stage numbers
      replaced by held-out ones), PIPELINE.md, the paper (sections 4 and 9), the portfolio numbers.
- [ ] R5.4 Clean the repo root: the 70 one-off scripts move under `tools/` or `archive/` with a one-line
      index; the package layout is `court/ detect/ sportvu/ qc/` plus the stage CLIs.
- [ ] [HUMAN] Lucien approves publishing; Stage 2 planning starts in a new document.

## Waiting on Lucien
- [ ] [HUMAN] Confirm the Stage 1 targets (gate after R0.3).
- [ ] [HUMAN] Pick games and held-out arena/era (gate after R1.1).
