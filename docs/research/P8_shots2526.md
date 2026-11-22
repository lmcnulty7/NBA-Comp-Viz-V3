# P8 Level 4: shooter position on NBA 2025-26 broadcasts against the public shot coordinates

**Status (2026-11-22).** Held from 2026-11-21 22:04 under D147, resumed under D149 (Lucien): D79 binds the play-by-play lanes only. The 10 games are a tracking evaluation set: nothing is tuned on them, each pipeline version is scored once, and the pilot measures cost and placement only. 0022501165 and 0042500171 stay out for good.

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

## 10. Pilot status (0042500112): placement stopped for the lead's decision

The rules above are unchanged. Placement ran 3 jobs under the lock (504 s, 534 s and 1,525 s, one long-lived easyocr
worker each). Numbers from `place.json` and `reads.json` in the P8 workspace:

- **Reads:** 379 cached. Per-read OCR is p50 6.416 s and p90 7.598 s, and frame extraction is p50 0.101 s. The cost
  is the OCR itself: a long-lived worker removes the model load, not the per-read time.
- **Searched shots:** 27 of 179, with these statuses:

  | Status | Shots |
  | --- | --- |
  | placed | 2 |
  | passed before a bracket | 11 |
  | gap at the moment | 9 |
  | no anchor | 3 |
  | clock jumped | 2 |

Why (diagnostic, from the cached reads only, post hoc):

1. **The upload is edited.** It is 2,423.8 s long, shorter than the 2,880 s of game clock. Between anchors of one
   period the clock often runs faster than the video: 120 s of video for 200 s of clock (2nd quarter), 40 s for
   76 s and 40 s for 85 s (4th quarter). So live play is cut, and "passed before a bracket" and "clock jumped" are
   those cuts. D134's condensed class fits it better than a full game.
2. **The scorebug reads poorly.**
   - Of 379 reads, 113 parse a clock, 39 a period and 72 a moment that agrees with the score.
   - Of the 122 anchor frames, 46 parse a clock and 27 are anchors.
   - At about 1 read in 5, the registered bracket (two readable 1 Hz reads at most 2 s apart) is rare, which gives
     "gap at the moment".

The pilot rule (the shortest of the 10 videos) picked the one upload most likely to be edited. Going on would spend
about 2 more lock hours for a low placed share on this video. The pilot is stopped here and the choice goes to the
lead:

- (a) finish this game as registered;
- (b) take another game as the pilot, with a rule that reads no result, for example the longest video;
- (c) a revised placement rule (a wider bracket, or a scorebug crop that raises the read rate), registered before
  it runs on a game not yet searched.
