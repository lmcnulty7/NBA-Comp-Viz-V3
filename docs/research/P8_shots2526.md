# P8 Level 4: shooter position on NBA 2025-26 broadcasts against the public shot coordinates

**Status (2026-11-23).** D157: the 0022501048 pilot is recorded as a placement failure (section 12), and the cut screen of section 13 now comes before any search pass. Earlier, (2026-11-22): Held from 2026-11-21 22:04 under D147, resumed under D149 (Lucien): D79 binds the play-by-play lanes only. The 10 games are a tracking evaluation set: nothing is tuned on them, each pipeline version is scored once, and the pilot measures cost and placement only. 0022501165 and 0042500171 stay out for good.

Branch r-p8-shots2526 from v4-research aa344cbe, written 2026-11-21 under PLAN.md D137 and D144,
docs/research/TRACKING_LEVELS.md (Level 4, step 2), docs/research/P8_level4.md (sections 3, 12 and 13) and
docs/research/P8_FEASIBILITY.md section 6.2. This file is step 1: the pre-registration, written before any build,
any OCR read of these videos, or any position number. Step 2 (a one-game pilot) waits for the lead's go.

## 1. What was seen before this was written

Truth-side checks only, on the 12 downloaded games (`level4/nba2526/fetch.json` in the P8 workspace) and the mirror
files in `_p8-data/nba_shots/`. No frame of any of the 12 videos was read and nothing was built.

- Field-goal attempts per game in `shotdetail_2025` and `shotdetail_po_2025`: 159 to 179, 2,061 in all. Every row
  joins `cdnnba_2025` or `cdnnba_po_2025` on (`GAME_ID`, `GAME_EVENT_ID`) = (`gameId`, `actionNumber`) (2,061 of
  2,061). The join's `side` is `left` on 1,039 attempts and `right` on 1,022.
- The court frame of section 4.1 below puts LOC_X and LOC_Y within 0.04 ft (median) of the mirror's own full-court
  `x`, `y` (percent of 94 by 50 ft): 99th percentile 0.07 ft, largest 3.26 ft, on 2,061 attempts.
- 182 of the 2,061 attempts have a clock with tenths (the last minute of a period). No rebound action carries a
  position (0 of 1,238 with `x`).
- Home teams: NYK, GSW, TOR, WAS, DET, ORL, BOS, PHI, ATL, OKC, LAL, HOU.

## 2. Games

The 12 downloaded games of P8_level4.md section 13 (identity rule calibrated on 2015-16 broadcasts only, D142).

- **Kept out:** 0022501165 (MIA at TOR, home TOR) and 0042500171 (HOU at LAL, home LAL). Both are played in arenas of
  the D85 broadcast CV test set (Toronto, Staples Center), and D85 lets no P8 development code touch a game played
  in a test arena. The tool passes each game's home team to `refuse_cv_test(..., homes=[HTM])`, so these two stop
  it. Whether a Level 4 evaluation may use them, given that it tunes nothing, is a question for the lead and Lucien.
  They are not built or read here.
- **The check set:** the other 10 games: 0022501048, 0022501154, 0022501172, 0042500102, 0042500103, 0042500112,
  0042500113, 0042500123, 0042500142, 0042500173.
- **Pilot (step 2):** the shortest video of the 10, which is 0042500112 (PHI at BOS, L2u8tsE_zi0, 2,423.8 s, 179
  attempts). It is chosen for cost, by a rule that reads no result.
- These games are also in the `pbp_2025_26` scoring set of the play-by-play lanes (winprob, lineups). This check
  reads the NBA mirror, not that basketball-reference set, fits nothing, and feeds no play-by-play lane.
- Every game id and video id passes `refuse_cv_test` and `tools.p8_common.guard` before any file is read.
- The mirror's rows are NBA Statistics: private, non-commercial use with attribution to NBA.com (P8_level4.md
  section 3). Publishing derived results is Lucien's decision.

## 3. Placing a logged shot in the video by reading the scorebug

There is no time map for these games: local_sync's clock_reader has no layout calibrated for 2025-26 scorebugs.
Each shot is therefore placed by reading the scorebug directly. The parser is `tools.p8_identity.parse_tokens`. The
reader is `tools.p8_identity.ocr_dir`'s easyocr on the same two bands (bottom quarter and top sixth, 2x upscale,
confidence at least 0.2). Both are imported unchanged, as calibrated in D138.

- **A frame reads a moment (p, c)** when, on that frame, `tools.p8_identity.frame_matches` holds against the
  mirror's score at (p, c), and the parsed clocks include c (whole seconds; tenths under one minute) and the parsed
  periods include p, or no period was read. So the period, clock and both scores agree, as in the identity rule.
- **Anchors.** One frame every 20 s of the whole video. Each frame that reads some moment of the game is an anchor,
  (video second, period, clock).
- **Search.** For a shot at (p, c), the estimate t* comes from the last anchor before it in game time, assuming
  the clock ran from that anchor to the shot. Frames are read at 1 per second from t* - 5 s, for at most 60 reads.
  The search stops at the first frame that reads (p, c) after a frame that read (p, c') with c' greater than c in
  the same period. The shot's moment is the first frame that shows c.
- **Refinement.** Two bisection reads between the last frame showing c' and the first showing c. The placed frame
  is the midpoint of the final bracket. Its timing error is then at most about 0.125 s, before the scorebug's own
  rounding.
- **Not placed.** A shot is not placed when the search ends without that bracket, for example during a replay, a
  cut-away or an edited upload. It is counted, never guessed.

The logged clock need not be the release: in 2015-16 the basketball-reference clock trailed the SportVU release by a
median of 2.05 s (docs/research/P8_E2.md, row C). So every position row is read at the placed frame and also as the
best frame of the window [placed - 3.0 s, placed + 0.5 s]. The 3.0 s is the frozen shot_frames release window
(analytics/shot_frames.py, WINDOW_AFTER_S). The second of these is labelled an oracle over time.

## 4. Positions compared

### 4.1 The truth and the court frame

The truth is the shot location, LOC_X and LOC_Y in tenths of a foot from the basket being attacked, with the
attacked basket from the joined `side`. It is converted to a full court 94 by 50 ft, x along the length, y across:

- left basket (5.25, 25): X = 5.25 + LOC_Y / 10, Y = 25 - LOC_X / 10
- right basket (88.75, 25): X = 88.75 - LOC_Y / 10, Y = 25 + LOC_X / 10

(Checked against the mirror's own x, y in section 1.)

### 4.2 Our position

These are r23's tracked players, the `cleaned` points of `<window>_trajectories.json` (court feet), on frames whose
sidecar row is accepted (`sportvu.bench.accepted`).

**r23 does not name the shooter.** It has no identity (E4) and no ball (E7), so section 6.2's shooter row ("needs
identity (E4) or the ball (E7)") cannot be measured as written. The position compared is that of the r23 track
nearest the truth location. That selection uses the truth, so the distance is a lower bound on the shooter
position error of any assignment from this build: an **oracle lower bound**.

It can show that the row fails: if even the nearest track misses the targets, so does every assignment. It can
never show that the row passes.

### 4.3 Orientation

The r23 court frame and the mirror's frame may differ by one of the four mirrors of `sportvu.sync.MIRRORS`
(identity, x flip, y flip, both). Without SportVU, the frozen direction step cannot solve this. Rule, fixed here:

- Per game, the mirror is the one with the smallest median windowed oracle distance over that game's period 1
  shots. Those shots are then left out of every row of that game.
- If the best mirror's median does not beat the runner-up's by at least 2.0 ft, the game is "orientation
  unresolved" and reported apart.
- The four medians are printed per game.

## 5. Rows and targets (section 6.2, copied unchanged)

| Row | Target (section 6.2) | Here |
| --- | --- | --- |
| Shooter position | p50 <= 3.0 ft, p90 <= 6.0 ft | measured only as the oracle lower bound, at the placed frame (primary) and windowed; can show a fail, never a pass |
| Shots | >= 95 percent found within 1.0 s of the release, <= 5 percent false | not measurable: the mirror's clock is this check's input, and no independent video time of the release exists |
| Rebounds | >= 90 percent found within 1.0 s | not measurable: no time truth |
| Rebounder position | p50 <= 4.0 ft | not measurable: the mirror gives no rebound position |
| Ball, ball carrier, identity | F1 >= 0.75; 85 percent; 80 percent | not measurable: no truth in the mirror, and r23 has no ball or identity |

Reported beside these, with no target:

- placed share: placed attempts of all attempts outside period 1;
- court share: placed attempts whose comparison frame is accepted;
- spot share: attempts with an r23 track within 3.0 ft and within 6.0 ft of the truth location, at the placed frame
  and windowed;
- the windowed oracle's time offset (best frame minus placed frame), median and quartiles;
- everything per game, and by regular season against playoffs.

**Level 4 gate** (TRACKING_LEVELS.md, step 2: "past a pre-set margin over the 2015-16 error"). The margin is set now:
the pooled windowed oracle must have p50 at most R0's p50 + 1.0 ft and p90 at most R0's p90 + 2.0 ft. R0 is the same
metric on 2015-16 (below). Past the margin, the gate's next step (retraining on modern frames) is Lucien's and is
not started. The lead may change these margins before step 2 runs, never after.

**R0, the 2015-16 reference.** The same metric on gsw_bkn_2015, the Phase A baseline (Mac r23 build on MPS):

- Shots are placed as in P8_E2.md M0 (`builds/e2` events).
- The truth is the SportVU shooter's position at the frozen release frame (shot_frames `release_frame` and
  `shooter_id`, contract frames x, y), with the game's mirror (sportvu_direction report).
- The window is the same.

r23 was trained on this game's R1.3 labels, so R0 is in sample and optimistic. That makes the gate stricter, not
looser.

## 6. Intervals

- Game bootstrap over the games that carry a row (at most 10), 1000 resamples, seed 0, 95 percent percentile
  interval: `tools.p8_scorecard.boot_windows` over games.
- The shot-window bootstrap is printed beside it.
- Pass and fail are read on the point estimate, as for every section 6.2 row.

## 7. What the Mac r23 build needs for a 2025-26 game, and its cost

**Needs**, all already present except new code under tools/:

1. The video as one section of the P8 workspace: a link `video/nba2526_<game>_s00.mp4` to the downloaded file.
   `sportvu.rebuild.cut` (through `tools.p8_frozen.safe_cut`) reads sections by stem.
2. A window plan in the frozen `plan_windows` form (`window`, `section`, `fps`, `f_start`, `f0`, `f_end`,
   `period`), written from the placed shots instead of from time maps. One window runs from placed - 3.0 s -
   PAD_S (1.5 s warm-up) to placed + 1.0 s. Overlapping windows are merged. New code: `tools/p8_shots2526.py`,
   with `refuse_cv_test(` and tests.
3. `tools.p8_frozen.safe_build` with r23's weights (`court_weights("r23")`, sha256 checked; `COURT_GRID_WEIGHTS`),
   `sportvu.autolabel.gate_args()` and the default device (MPS, as in the Phase A baseline; D100 leaves the CPU
   switch open).
   - Stride: `build_stride(30)` = 3, the pipeline's 10 Hz.
   - The gate head, player detector and thresholds come from the main checkout's models folder, as in Phase A.
4. The CV Python of D87 (V3_CV_PYTHON, with ultralytics and easyocr), for the build and the OCR, under the
   heavy-job lock, in jobs of at most 30 minutes (anchors in slices of the video, searches and builds in batches of
   shots).
5. Nothing game-specific beyond that: no SportVU, no clock layout, no truth fit.
   - The court model has seen only 2015-16 GSW arenas. Court acceptance on 2025-26 courts is unknown and is
     measured as the court share.
   - Some uploads may be edited: the pilot's 2,424 s is short for a full game. Edits show up in the placed share.

**Estimated Mac hours per game** (estimates, from measured rates):

| Part | Basis | Typical game (3,500 s, 175 attempts) | Pilot (2,424 s, 179 attempts) |
| --- | --- | --- | --- |
| OCR anchors | 1 frame per 20 s | 175 reads | 121 reads |
| OCR search and refinement | 4 to 8 reads per attempt | 700 to 1,400 reads | 716 to 1,432 reads |
| OCR time | 5 to 7 s per read (P8 identity batch: 51 to 59 s per job of 8 new frames, model load included) | 1.2 to 3.1 h | 1.2 to 3.0 h |
| r23 build (MPS) | about 5.5 s of clip per attempt; 1.54 and 1.72 s per video second (Phase A r23 rebuilds: gsw_bkn_2015 3,684 s for 2,393.5 s of windows, gsw_ind_2016 4,532 s for 2,639.5 s) | 0.41 to 0.46 h | 0.42 to 0.47 h |
| r23 build, detector on CPU | about 2.4 times MPS (D100) | 1.0 to 1.1 h | 1.0 to 1.1 h |
| Clip cuts | about 175 clips of 5.5 s, re-encoded (not measured) | under 0.1 h | under 0.1 h |
| Total (MPS) | | about 1.7 to 3.7 h | about 1.7 to 3.6 h |

So 10 games take about 17 to 37 lock hours on MPS. OCR dominates. A crop to the scorebug's box, located on the anchor
frames, would cut the read time, but it changes the reader's input from the calibrated bands, so it is not used
unless the lead decides it before the pilot. The pilot measures the real rates, and the remaining 9 games are planned
from those.

## 8. Fixed before any number

The game set and the two exclusions, the pilot rule, the placement rule (anchor step, search window, stop rule,
refinement), the 3.0 s and 0.5 s window, the court frame, the orientation rule and its 2.0 ft margin, the
period 1 exclusion, the rows, the gate margins (1.0 ft on p50, 2.0 ft on p90), R0's definition and the bootstrap.
Nothing is changed after a 2025-26 result is seen; anything later is reported as post hoc.

## 9. Implementation details fixed with the code (dee4c113), before any placement or position was seen

The lead's D145 kept the reader input as registered (same bands, upscale, confidence floor and parse_tokens) and
moved every read of a job into one long-lived easyocr process; each read's extract and OCR seconds are recorded in
`reads.json`. Written with tools/p8_shots2526.py, before any shot was placed (only 74 anchor reads existed, unparsed
into moments, when this was written):

- **Which moments a frame reads**: every (period, clock) of the parsed sets for which an ordered pair of the
  frame's integers equals the mirror's score (tools.p8_scorebug.matches); no period read means periods 1 to 4, as
  in frame_matches. An anchor is an anchor read with exactly one such moment.
- **The stop rule, read for tenths**: a frame "shows c" when its moments in the shot's period include a clock at
  most c and none above it. For whole-second clocks this is the first frame showing c, as registered; under one
  minute (tenths, 182 of the 2,061 attempts) the scorebug rarely shows exactly c, so "at most" is the same rule.
- **The bracket**: the last 1 Hz read showing a larger clock and the first showing at most c must be at most
  2.0 s apart (one unreadable read between), and the clock may not fall by more than the video time between them
  plus 1.0 s (a cut in the upload). Otherwise the shot is not placed ("gap at the moment", "clock jumped").
  A search whose first readable frame already shows at most c is "passed before a bracket".
- **Frames** are extracted with ffmpeg's input seek (`-ss` before `-i`), the seek the frozen clip cut uses, so a
  placed video second and a clip frame name the same frame. The pilot video's start_time is 0.
- **Windows** closer than 10 s are built as one window (capped at 120 s), because each frozen build loads its models
  again; this changes which frames are warm-up, not the rule for the compared frame.
- **The compared frame** is the accepted built frame nearest the placed frame, used only within 0.06 s (the build
  runs at 10 Hz); the windowed oracle uses every accepted built frame in [placed - 3.0 s, placed + 0.5 s].
- **The bootstrap** for one game is over its built windows; over games, `tools.p8_scorecard.boot_windows`.

## 10. Pilot results (0042500112, PHI at BOS), D149: cost and placement, nothing tuned

Run 2026-11-21 to 2026-11-22, all jobs under the heavy-job lock with caffeinate. Placement was paused under D147
after job 3 and resumed under D149 from the cached reads, with the rules unchanged. Numbers are copied from
`place.json`, `build/build_log.json` and `score.json` in the P8 workspace and from `reports/research/p8_shots2526.json`.
Under D149 these 10 games are a tracking evaluation set and the pilot measures cost and placement only: no rule,
threshold or parameter changes because of it.

**Lock time: 8,642 s (2.40 h).**

| Part | Lock time | Detail |
| --- | --- | --- |
| Placement | 8,434 s | 7 jobs of 504, 534, 1,525, 1,560, 1,512, 1,550 and 1,249 s |
| Build | 205 s | |
| Score and reference | 3 s | |

The lock went to other teammates' jobs for several hours in between. That waiting is not counted.

**Reads: 1,280.** OCR per read p50 6.274 s, p90 7.41 s; frame extraction p50 0.098 s. There were 7 model loads, one
per job. One long-lived worker removes the model load, not the per-read cost.

**Placement: 14 of 179 attempts placed (7.8 percent).** Searches used 1,805 reads in all, some shared between
shots.

| Status | Attempts |
| --- | --- |
| placed | 14 |
| passed before a bracket | 72 |
| gap at the moment | 58 |
| no anchor | 23 |
| no bracket in the search | 7 |
| clock jumped | 5 |

There were 27 anchors among the 122 anchor reads. Placed by period: 3, 3, 6 and 2. Outside period 1, 11 of 130 were
placed (placed share 0.085).

Why (diagnostic, from the cached reads, post hoc):

1. **The upload is edited.** It is 2,423.8 s long, under the 2,880 s of game clock. Between anchors of one period
   the clock often runs faster than the video: 120 s of video for 200 s of clock, and 40 s for 76 s and for 85 s.
   Live play is cut, which gives "passed before a bracket" and "clock jumped". D134's condensed class fits it
   better than a full game.
2. **The scorebug reads poorly.** Of the first 379 reads, 113 parse a clock and 72 a score-consistent moment. The
   registered bracket (two readable 1 Hz reads at most 2 s apart) is therefore rare, which gives "gap at the
   moment".

**Build:** 14 windows, all built (0 gate-empty, 0 failed): 204.1 s of wall time for 77.1 s of clips, or 2.65 s per
video second. Each window reloads its models, so short windows cost more per second than Phase A's 1.54 to
1.72 s.

**Orientation: unresolved.** The period 1 medians of the windowed oracle (3 placed shots) are identity 41.673 ft,
flip_x 2.758 ft, flip_y 40.783 ft and flip_xy 1.358 ft. flip_xy beats flip_x by 1.40 ft, under the 2.0 ft margin, so
the game is reported apart as "orientation unresolved". The rows below are under flip_xy and carry that flag.

**Shooter position (nearest track, oracle; can show failure only), outside period 1, flip_xy, unresolved:**

| | n | p50 (window CI) | p90 (window CI) | within 3 ft | within 6 ft |
| --- | --- | --- | --- | --- | --- |
| At the placed frame | 8 (court share 0.727) | 14.236 ft (10.29, 16.963) | 24.097 ft (15.034, 38.327) | 0.0 | 0.0 |
| Windowed [placed - 3.0 s, placed + 0.5 s] | 11 | 2.885 ft (1.609, 9.192) | 15.609 ft (4.278, 27.106) | 0.545 | 0.636 |

- The windowed oracle's best frame is 1.506 s before the placed frame (median; quartiles -2.102 and -1.147 s), so
  the logged clock trails the shot here as in 2015-16.
- Against section 6.2 (p50 at most 3.0 ft, p90 at most 6.0 ft), the oracle fails on p90 at both frames and on p50
  at the placed frame. Since it is a lower bound, the row fails for any assignment on this game. That rests on 11
  shots of one edited video with unresolved orientation.

**R0, the 2015-16 reference (gsw_bkn_2015, Phase A r23 build on MPS, in sample for r23):** 60 shots with a release
frame and a placement.

| | n | p50 (window CI) | p90 (window CI) |
| --- | --- | --- | --- |
| At the placed frame | 58 | 4.743 ft | 10.34 ft |
| Windowed | 59 | 0.579 ft (0.393, 1.002) | 3.417 ft (1.929, 5.008) |

**Gate:** not computed (`gate: null`), because the gate pools only games with resolved orientation and the pilot is
unresolved. The registered limits are p50 at most 1.579 ft and p90 at most 5.417 ft (R0 + 1.0 and + 2.0 ft). The
pilot's flagged windowed values, 2.885 and 15.609 ft, are past both. That comparison is reported, not a gate
verdict.

**Failures and fixes:** two bugs were fixed in the scorer's input handling while the pilot ran, with no rule changed:

- R0 read a nullable shooter id with `s == s`; it now uses `pd.notna`.
- r23's cleaned trajectories hold NaN points, which made the at-frame oracle NaN. Non-finite points are now
  dropped, and a test covers it.

The scores above are from the corrected run.

**Cost per game at this rate:** placement scales with reads (about 1,280 to 1,800 reads, 2.2 to 3.1 h at 6.3 s).
The build scales with placed shots: at 2.65 s per clip second and about 5.5 s per shot, a fully placed game of 175
shots needs about 0.7 h. Placement coverage on edited uploads, not compute, is what limits this check.

## 11. Revision under D151, registered before anything runs on it

Written 2026-11-22, before any read of 0022501048. This section changes only what D151 allows: the scored set, the
pilot, and the reader's input for the search. The bracket rule, the stop rule, the windows, the oracle, the
orientation rule, the rows, R0, the gate margins and the bootstrap are as in sections 3 to 9. D149 still binds:
nothing is tuned on these games, each version is scored once, KEPT_OUT stays out, and no tracking output on a
2025-26 game goes into a play-by-play lane.

### 11.1 Full-broadcast screen (metadata only)

A video counts as a full broadcast only if its probed duration (`probe_duration_s` in `fetch.json`, the metadata
probe before download) is at least 2,880 s, the regulation game clock. Shorter uploads are D134's condensed class
and leave the scored set. Applied to the 10:

| Game | Video | Probe (s) | Full |
| --- | --- | --- | --- |
| 0022501048 | 4ZrE7OXRcuQ | 7,867 | yes |
| 0022501154 | IKPtnyERx4c | 4,109 | yes |
| 0022501172 | pZqKb08f6u0 | 3,456 | yes |
| 0042500102 | QpQCu6dnUIA | 3,198 | yes |
| 0042500103 | fWYk9XeB7bk | 3,563 | yes |
| 0042500112 | L2u8tsE_zi0 | 2,424 | no |
| 0042500113 | ngsPkigANU4 | 2,872 | no |
| 0042500123 | mCihOZ_0vjQ | 2,738 | no |
| 0042500142 | B-AcMUKEt2Y | 3,121 | yes |
| 0042500173 | NLYLQDfxu4s | 3,557 | yes |

**The scored set is 7 games.** 0042500112 leaves it for good: it shaped this revision, and it fails the screen too.
Its section 10 numbers remain the record of that pilot.

Passing the screen does not prove an upload is unedited: a full broadcast with dead time runs about 2 h. Six of the
seven are 3,121 to 4,109 s long, so they may also have cuts. That will show as "passed before a bracket" and
"clock jumped".

### 11.2 The new pilot

0022501048 (NOP at NYK, 2026-03-24, 173 attempts), the longest remaining video (7,867 s), chosen on duration only.

### 11.3 The scorebug crop

**Anchors are unchanged:** one frame every 20 s, read on the full bands by the registered reader. Each anchor read
now also stores easyocr's box for every kept text, in frame pixels.

**The crop box** comes from a fixed rule, written here. It uses only the anchor reads of the same video, never a
placement, a position or a shot outcome.

1. Take every anchor frame that reads exactly one moment (section 9).
2. In each such frame, keep the boxes of the texts that carry its matched clock, either of its matched scores, or a
   period.
3. Across those frames, the crop is:
   - x0 and y0: the 10th percentile of the kept boxes' left and top edges;
   - x1 and y1: the 90th percentile of their right and bottom edges;
   - widened by 16 px on every side, clipped to the frame.
4. With fewer than 5 such frames there is no crop. Then the search does not run, and the lead is told.

The automatic layout step (the frozen clock_reader check over the registered layouts) is not used. The registered
layouts are 2015-16 scorebugs, and D152's layout finder met its bar on only 3 of 8 videos. No layout is calibrated
by hand.

**The search reads the crop.** Search and bisection reads OCR the crop instead of the two bands:

- same 2x cubic upscale, same easyocr reader, same 0.2 confidence floor, same parse_tokens and moment rule;
- frames extracted by the same ffmpeg input seek;
- crop reads cached apart from band reads.

Recorded:

- per-read OCR seconds for band reads and for crop reads, p50 and p90;
- the read rate of each: the share of reads that parse a clock, and the share that read exactly one moment.

**Unchanged:** the bracket (two readable 1 Hz reads at most 2 s apart, no interpolation), the 2 bisection reads,
the 60-read search limit, and the "clock jumped" slack.

### 11.4 Lock and report

- Jobs run in turns with p8-teachers' D146 jobs, each under the lock with caffeinate.
- After the pilot the report gives: the placed share, the lock hours, OCR p50 and p90 for band and crop, the read
  rates, the scored set's size (7), and, if the build runs, the go or no-go against the registered gate. The gate
  is computed only for a game whose orientation is resolved (section 4.3).
- Then stop for the lead's go on the other 6.

## 12. The 0022501048 pilot (D151): a placement failure, recorded under D157

Run 2026-11-22 under the heavy-job lock with caffeinate, cost and placement only: no build, no position, no R0
for this game. Numbers are copied from the job logs (each job prints `place.json`'s summary and the lock's
`done in` line), from `place.json` and from `reads.json` in the P8 workspace.

**The crop** from section 11.3's fixed rule: [148, 25, 572, 587] (x0, y0, x1, y1 in frame pixels).

**At the stop D157 records (after job 4):**

- Jobs 1 to 4 took 1,507, 1,506, 1,634 and 1,785 s of lock time, 6,432 s (1.79 h). Job 1 was the anchor pass and
  stopped on its budget; job 2 finished the anchors and began the search.
- 60 of the 173 attempts had been searched and 4 were placed. The others: passed before a bracket 20, no anchor 13,
  clock jumped 11, no bracket in the search 7, gap at the moment 5.
- Band OCR (anchors): p50 6.828 s, p90 7.718 s over 394 reads; 0.995 of them parse a clock and 0.556 read exactly
  one moment.
- Crop OCR (search): p50 4.399 s, p90 5.364 s over 766 reads; 0.83 parse a clock and 0.679 read exactly one
  moment. The crop raised the read rate over the 0042500112 bands (section 10, 2.).

**What ran after that stop.** The cache holds six more jobs than D157's record. Jobs 5 to 7 (1,520, 1,597 and
1,520 s) continued the search, and three short jobs (136, 126 and 155 s) followed. All ten jobs took 11,486 s
(3.19 h) of lock time. After job 7, 132 of the 173 attempts had been searched and 6 were placed: passed before a
bracket 42, clock jumped 31, no bracket in the search 22, no anchor 19, gap at the moment 12. Crop OCR was then p50
4.272 s over 1,795 reads. The short jobs added 60 crop reads to `reads.json` (1,855 in all, 10 model loads) and
left no `place.json`. None of this changes the verdict: the search places about 1 attempt in 20.

**Why (post hoc, read from the cached anchors after the search had failed):**

1. **The video starts mid first quarter.** The first anchor, at 20 s of video, reads period 1 at 7:07, so about
   5 minutes of first-quarter play are not in the upload. The frame at 0 s shows a web page (a search box and a
   video title), not the broadcast. Those period 1 attempts are "no anchor".
2. **The clock runs faster than the video between anchors.** Live play is cut out, while long dead-ball stretches
   are kept, so the upload is 7,867 s long and still edited. Section 14 counts these cuts by the screen of
   section 13.

**A duration of 2,880 s or more does not prove a full broadcast.** Section 11.1's screen is necessary, not
sufficient: an edited upload with long dead-ball stretches passes it.

## 13. The cut screen (D157), registered before any new read

Written 2026-11-23, before any read of 0022501154, 0022501172, 0042500102, 0042500103, 0042500142 or 0042500173.
The rule comes from physics: the game clock cannot fall faster than the video runs. Its two numbers were set by the
lead in D157 from that, not from any video's data.

- **Anchors** are section 3's anchors from the anchor (band) pass only: one frame every 20 s of the whole video,
  read on the registered bands, kept when the frame reads exactly one moment (section 9). No crop, no search read
  and no placement enters the screen.
- **Pairs.** The anchors are ordered by video time. Two consecutive anchors form a pair when they read the same
  period. A pair across a period change is not counted.
- **A cut** is a pair whose clock drop exceeds the video time between them by more than 2 s:
  (clock of the first - clock of the second) - (video s of the second - video s of the first) > 2.0, rounded to
  0.001 s first. A pair where the clock stands still or rises is not a cut.
- **A video passes** when its cuts are 10% or less of its pairs. A video with no pair does not pass.
- **Order.** No search pass runs on a video until it passes; then only on the lead's go. `place` refuses a game
  whose `screen.json` does not show a pass.
- **Applied to 0022501048** from its cached anchor reads, with no new read. Its anchors were seen (sections 10 and
  12) before this rule was written, so its result is a record, not a test of the rule.
- **The anchor pass alone on the other six** of the scored set, in the order above, one lock job at a time with
  caffeinate, each job at most 25 minutes (a rerun continues from the cached reads). Never a crop, a search, a
  build, a position or R0 on any of them.
- **Expected cost:** 157 to 206 anchor reads per video, 1,053 in all, at about 6.8 s per read: about 2.0 lock
  hours.
- **What follows (D157):** if fewer than 3 of the 7 pass, the 2025-26 shot check closes as not feasible on public
  uploads. If 3 or more pass, the lead decides their search pass.

Code: `tools/p8_shots2526.py` (`cut_screen`, `screen_record`, the `anchors` and `screen` commands), tests in
`tests/test_p8_shots2526.py` on synthetic anchors. Each result is written to `screen.json` in the game's folder of
the P8 workspace.
