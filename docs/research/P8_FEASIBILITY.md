# P8 Step 0: feasibility study for broadcast tracking to full operation

Written 2026-11-16 on branch v4-research under PLAN.md P8, D75 and D76. Planning only: no training, no model run,
no heavy job, no lane code, no video downloaded. It builds on the tracking dossier (docs/research/tracking.md) and
the data survey (docs/research/DATA.md). Every number names the repository file it comes from or a source fetched
on 2026-11-16. Numbers marked "estimate" are estimates and say what they rest on.

## Verdict

**No-go for P8 as written. Go for a re-scoped P8, in two phases with a gate between them. Choose P8 over P5.**

P8 as written cannot be finished or scored. Its definition of done is "every Stage 1 row passes on the held-out
game and arena". Those games are among the 17 that D71 declares spent, and they may never be scored again. The
D71 test set for tracking lanes (Set A, D77) has no usable broadcast video: this study found none at 720p for any
of its 10 games. The scorecard is also not ready to drive the work. Its missed-player and ghost-box rows are
counted through a truth homography that is valid only near the fitted feet (CLAIMS.md correction, 2026-10-04).
They are counted on only 4.7 to 15 percent of window frames, and they disagree with two human box-truth checks of
the same detector (section 2.3).

There is a way forward. A public channel holds 720p broadcasts of at least 10 more 2015-16 games in the SportVU
window, in 9 arenas the pipeline has never seen. The repository already uses three videos from the same channel
(section 4). That gives a CV test set and new development games from the same era, which is exactly the data the
court scaling study asked for (ROADMAP_LOG.md, side study). The re-scoped P8 starts with measurement (Phase A,
mostly CPU). Three experiments follow only if Phase A's gate is met (Phase B).

## 1. What was read and done

- Read: PLAN.md (P5, P8, D3, D5, D16, D42, D43, D50, D52, D58, D71 to D77), docs/research/tracking.md,
  docs/research/DATA.md, docs/DATA_SOURCES.md, CLAIMS.md, README.md, ROADMAP.md, ROADMAP_LOG.md, DEVLOG.md
  (named passages), LABEL_SCHEMA.md, `sportvu/splits.json`, `sportvu/manifest.py`, `sportvu/truth.py`,
  `sportvu/bench.py`, and these named reports: every file in `reports/scorecard/`, `reports/sportvu_splits.txt`,
  `reports/sportvu_manifest.txt`, `reports/broadcast_search_away.txt` and `reports/r23_verification.txt`.
- Searched YouTube by metadata only (the R1.1 method of `sportvu/manifest.py`: yt-dlp flat search and format
  listing, nothing downloaded). One public channel's upload list was read the same way.
- Read the 2015-16 schedule table (`DATA_ROOT/collect/bref_schedule/games/2015-16.parquet`), the `games` keys of
  `DATA_ROOT/sportvu_contract/manifest.json`, and the columns of one training game's play-by-play table.
- Not done: no file under any `heldout/` folder was opened. The per-observation bench files of the held-out games
  (`data/sportvu/bench_obs/`) were listed, not read. A new breakdown of them would be a new held-out metric
  (CLAUDE.md rule 4, D71).

## 2. Error analysis of the scorecards

### 2.1 What the stored files hold

Each `reports/scorecard/<game>__<build>.json` holds the game totals of every row and position error split into
three camera thirds (near, middle, far). Per window, it holds testable frames, coverage seconds, missed players,
in-frame players, ghost boxes, boxes and referee boxes. It also counts untestable frames by cause. Team labels and
identity exist only as game totals. The three scored games are gsw_phx_2016 (held-out game) and cle_nyk_2015 and
cle_gsw_2016 (held-out arena). All of them are spent under D71. The ratios below are read from counts already in
the stored files. They describe; nothing is tuned on them. They only point at what Phase A must measure again on
development games.

### 2.2 Findings

1. **Distance to camera.** The far third is the worst third for p50 in every game. Its bias points toward the
   camera in every game: r23 far third p50 2.24, 3.10 and 3.10 ft, bias +1.52, +1.66 and +2.00 ft (gsw_phx, cle_nyk,
   cle_gsw). The near third has p50 1.90, 1.77 and 1.85 ft and bias -0.67, -0.63 and -0.12 ft. The middle third
   has p50 1.71, 2.11 and 1.94 ft (`*__r23.json`, `position_error.by_region`). p90 does not follow the thirds:
   on cle_nyk the middle third is worst (6.07 ft). Missed and ghost counts are not split by third.
2. **Missed players and ghost boxes are spread evenly, not clustered.** Take the gsw_phx windows with at least 20
   testable frames (16 windows). Missed players run from 19.7 to 31.3 percent (median 25.6). Ghost boxes (all
   boxes, referees included) run from 9.5 to 22.6 percent. The 7 such cle_nyk windows give 20.1 to 31.2 and 8.2 to
   24.0 percent. No small set of windows carries the rate. The cause is systematic: the detector, the matching
   rule, or the truth that scores them.
3. **The detection rows have never moved.** Builds v3, public, r23 and r23_cam have the same missed count (3,515
   on gsw_phx) and the same ghost count (1,938) (`compare__v3__public__r23.txt`, `compare__r23_replay__r23_cam.txt`).
   Every build so far changed the court model only. ROADMAP.md Phase 3 (R3.1 to R3.5, detection and tracking) and
   Phase 4 (teams, identity) are all unticked.
4. **The scored frames favour easy frames.** gsw_phx has 9,500 window frames. On 19.3 percent of them the build had
   fewer than 6 player boxes (`untestable_boxes`). On 41.7 percent, the truth fit lacked 6 inliers. On cle_nyk the
   box cause alone is 45.3 percent, and on cle_gsw it is 33.8 percent (plus 21.1 percent of frames with no SportVU
   moment). Missed players are counted only on frames where at least 6 boxes matched. Frames where detection fails
   badly never enter the row. Some of those frames are close-ups with few players in view, and the stored files do
   not say how many.
5. **Referees are a small part of the ghosts.** On gsw_phx, 72 of 1,938 ghosts are referee-class boxes (rate 15.3
   percent with referees, 15.1 percent without). 208 of 280 referee-class boxes sit within ghost distance of a
   player. The scorecard caveat reads this as a box that may be a player standing beside a referee.
6. **Team labels are worse in the new arena.** They are 81.9 percent on gsw_phx and 73.7 and 78.2 percent on the two
   CLE games, and each is an upper bound (best A/B mapping per window). No breakdown exists, so the stored files
   cannot say whether kits, lighting, broadcaster or crowding cause it.
7. **Identity.** There are 21.39, 31.07 and 14.29 id switches per possession, and 2.58, 3.47 and 2.10 tracks per
   player per possession. These rest on 56, 15 and 7 possessions. They are lower bounds, because switches are
   counted between sparse testable frames.
8. **Coverage fails in one broadcast.** On cle_gsw_2016 the share of live wide seconds accepted per window runs
   from 0.12 to 0.80 (39.7 percent overall). On gsw_phx it runs from 0.96 to 1.00. That is a court-acceptance
   failure on that broadcast (layout `tnt_playoffs16_720` in `data/harvest/games.json`). It is not a detection
   problem.

### 2.3 A cross-check that changes the plan

The same detector (`models/player_detector.pt`) was scored twice against human-labelled boxes in the image, on
production footage:

- Recall 0.87, precision 0.89 (DEVLOG.md, 2026-06-26).
- Recall 0.967, precision 0.956 on 50 frames (DEVLOG.md, 2026-07-18).

The scorecard counts players and boxes on the court, through the truth homography, and gets 25.1 percent missed
and 15.1 percent ghosts. Missing 3 to 13 percent in the image does not square with missing 25 percent on the court.
The CLAIMS.md correction gives the likely reason: the truth homography is fitted to 6 to 10 bunched feet and sits
about as far from the painted lines as V3's own. A player far from the fitted cluster lands in the wrong place. He
then counts once as missed, and his box counts once as a ghost.

This is not proven. The frames differ, the human sets are small, and image IoU is not court distance. But it means
**no detector experiment should be ranked or judged until the missed and ghost rows are measured with a better
truth.** The camera-model truth of R1.3 (`sportvu/camera.py`, one camera position per game, validated against the
paint) is that better truth for any synced game.

### 2.4 What the stored files do not allow

The files cannot break any row down by crowding, player speed, distance to a camera cut, or court zone (beyond the
three thirds). They cannot split missed or ghost counts by third, or team errors by anything. They cannot do any
breakdown per player. They hold no row for the ball or events, and no jersey read rate (no crops were saved).
These breakdowns must be built into the new scorer (section 7, prerequisite 4) and run on development games only.

## 3. Published single-camera results as ceilings

The rows reuse the dossier's verified citations (docs/research/tracking.md, sections 3 and 7). Two event rows are
new and were fetched at the source on 2026-11-16. None of these rows measures our metric on our data, so every row
is "not directly comparable". They show where published systems stand. They are not targets to beat.

| Row | Best published number found | Source | Why not directly comparable |
| --- | --- | --- | --- |
| Position | Median error 1.55 to 1.84 m on broadcast TV, 1.13 to 1.20 m on a tactical camera | Theiner et al., WACV 2022, Table 3 | Soccer, about 105 x 68 m pitch, different aggregation, reference error about 1 m |
| Position (commercial) | RMSE 0.44 to 1.14 m when detected; 4.62 to 12.23 m when not | Crang et al. 2026, J. Sports Analytics | One soccer match, RMSE, vendors fill undetected players |
| Position (commercial basketball, self-reported) | 76 to 83 percent of player positions detected; mean expected error 1.5 to 1.9 ft; extrapolated positions 4.4 to 5.2 ft | SkillCorner basketball open data README (docs/research/DATA.md, question 3) | The vendor's own model of its error, not a measurement against sensor truth; ACB, 2025-26 |
| Team labels | 94 percent after one frame, 97 percent within 500 frames | Koshkina et al., arXiv 2021, abstract | Hockey |
| Team labels | Micro accuracy 0.93 | Theiner et al., WACV 2022, Section 4.3 | Soccer, annotated boxes |
| Tracking | Basketball HOTA 60.8, IDF1 67.8 | Cui et al. (SportsMOT), ICCV 2023, Table 7 | Image boxes, NCAA clips with no viewpoint cuts |
| Jersey identity | 87.4 percent tracklet accuracy | Koshkina and Elder, CVPRW 2024, abstract | Soccer (SoccerNet) |
| Jersey identity | 91.0 to 95.0 percent for open VLMs, four-way multiple choice | Hu et al., arXiv 2026, Table 2 | Multiple choice, not open reading |
| Ball | F1 82.6 at 4 px, basketball split | Tarashima et al. (WASB), BMVC 2023, Table 2 | 2D ball in the image on short clips |
| Events (video only) | Event detection mAP 0.435 over 11 classes (3-point, 2-point, layup, dunk, free throw by success, steal), untrimmed video | Ramanathan et al., "Detecting events and key actors in multi-person videos", CVPR 2016 (arXiv:1511.02917), Table 3 | NCAA, 2016 methods, video only; we would read the play-by-play |
| Events (rebounds) | Rebound spotting average mAP 0.46 (DREB 0.58, OREB 0.34) over tolerances of 1 to 3 s, validation set; DREB vs OREB accuracy 0.881 | Barrera-Roy and Clapés, arXiv:2512.15386 (December 2025, not peer reviewed), Tables II and III | Short 720p clips scraped from NBA Stats, data not public |
| Fill of unseen players | GC-VRNN 5.39 to 5.86 ft; TranSPORTmer 3.08 to 3.60 ft | Xu et al., CVPR 2023, Table 1; Capellera et al., ACCV 2024, Table 3 | Simulated camera on SportVU |

What this says for the targets: position is already in the published range for detected players. Team labels at
95 percent sit at the level of published learned-embedding methods, not colour clustering. Identity, ball and
events have no basketball broadcast result scored against sensor truth. Their targets must be set from what the
analyses need, and are said to be.

ROADMAP.md states a "physical ceiling for one 720p camera: about 1.5 ft median". That line cites no measurement
and is a planning figure, not a ceiling.

## 4. Labelled games with video

SportVU (players with ids and jerseys, ball x y z, 25 Hz) plus a synced broadcast gives labels for position,
team, identity and ball. The contract's play-by-play table for each game maps every event to a SportVU frame. In
`11.14.2015.BKN.at.GSW_pbp.parquet`, 452 events (113 rebounds, 195 field-goal attempts) carry a frame within
0.04 s of their clock. So events have truth too.

| Group | Games | Video | Status for P8 |
| --- | --- | --- | --- |
| Training (R1.3 labels) | gsw_bkn_2015, gsw_cha_2016, gsw_ind_2016, gsw_sac_2015 | 720p, synced; 11,506 labelled frames kept by the paint check (`reports/sportvu_splits.txt`) | Development. One arena (Oracle), two broadcasters |
| Held-out game | gsw_phx_2016 | synced | Spent (D71). Never scored again |
| Held-out arena | cle_nyk_2015, cle_gsw_2016 | synced | Spent (D71) |
| Set A (D77) | the 10 games of 2015-12-05 | **None usable.** Flat searches, 2 to 3 queries per game, found only highlights for 9 games. For IND at UTA: one full-length upload (5,086 s, posted 2015-12-06), 360p only, which is the manifest's `low_res` verdict. One "Cavaliers at Heat 2015 / 2016" full game exists at 720p, but a same-length upload states 2016-03-19, which is outside the SportVU window | Not a CV test set. Video-free lanes only |
| Other GSW home games | 11.02 MEM, 01.02 DEN (ambiguous); 11.06 DEN, 11.24 LAL, 12.18 MIL (short) | `reports/sportvu_manifest.txt` | Possible development games after a scorebug check |
| Road leads | 11.12.2015 at MIN (69.5 min, Spanish feed), 12.30.2015 at DAL (ambiguous) | `reports/broadcast_search_away.txt` | Possible, after a check by eye |
| **New: one public channel** | 10 more games in the SportVU window, 9 arenas (table below) | 720p avc1 confirmed by format listing for all 11 videos probed | Never registered. Candidate CV test set and new development games |

That channel's uploads are titled "Nba HD Full Game <away> at <home> 2015 / 2016". It also hosts three videos
the repository already uses: f8lAcHg6kk0 (gsw_phx_2016), hoPgMa02zB4 and qzJbC4tmJeA. The manifest search
(`sportvu/manifest.py`) looked only at GSW, CLE, OKC and NYK home games and read a bare season string as
ambiguous. So it could not have found most of these. Each title below is resolved against the 2015-16 schedule.
"SV" means the archive is in the SportVU contract manifest:

| Video id | Title matchup | Meetings at that venue in 2015-16 (* = in the SportVU window) | Date certain without a scorebug check? |
| --- | --- | --- | --- |
| ut-EcJRm_XA | Pelicans at Heat (Christmas) | 2015-12-25* SV | yes |
| GqVLOlBOj0E | Clippers at Lakers (Christmas) | 2015-12-25* SV, 2016-04-06 | yes (Christmas title) |
| MVSEj_jQxJk | Thunder at Heat | 2015-12-03* SV | yes |
| olDo7SEOwj0 | Spurs at Raptors | 2015-12-09* SV | yes |
| wtGuZJAkm08 | Bulls at Pacers | 2015-11-27* SV, 2016-03-29 | no |
| QIEVYvOfeaM | Pacers at Magic | 2016-01-06* SV, 2016-02-21 | no |
| UiAdZkI4tb0 | Thunder at Trail Blazers | 2016-01-10* SV, 2016-04-06 | no |
| Td8GojYg1FM | Pacers at Wizards | 2015-11-24* SV, 2016-03-05 | no |
| CVFO7eUrTL8 | Celtics at Knicks | 2016-01-12* SV, 2016-02-02 | no |
| tqDqi5XqUKU | Kings at Thunder | 2015-12-06* SV, 2016-01-04* SV | no (both in window) |

Not counted: 5ehjglPdY2g (Nets at Warriors, likely a re-upload of gsw_bkn_2015, same 6,364 s), Knicks at
Grizzlies (2016-01-16, one of the 26 games with no archive, DATA.md), and Celtics at Warriors (2016-04-01, outside
the window).

Caveats: a title is not proof of the game. The manifest's rule holds: identity is confirmed only by a scorebug
frame at R1.2. None of these is a CLE home game, so none is held-out under `sportvu/splits.json`. Their SportVU
data trained the frozen analytics models. That does not matter for CV, because no CV model or design choice has
seen their video. It does matter for any swap test on them (section 6). These are third-party uploads with the
same rights status as the videos already used (CLAIMS.md standing caveat 4). Using more of them is Lucien's call.

## 5. GPU hours

### 5.1 What has been measured in this repository

| Basis | Measured | Source |
| --- | --- | --- |
| b1 Detector fine-tune, YOLOv8m, 654 images | about 0.5 h on a Colab T4 (local MPS about 22 h) | DEVLOG.md 2026-06-26; PROJECT_SUMMARY.md |
| b2 Camera-truth labelling, 4 games, 9,787 s of video (about 2,450 s a game) | 2.7 h on an A100; this Mac 2.15 s per video second (about 6 h) | ROADMAP_LOG.md, decision of 2026-10-04 and pass 9 |
| b3 Court model training arms | about 52 min each on an A100, two at a time (short recipe); r23 recipe up to 100 epochs, stopped at 66 | ROADMAP_LOG.md side study and pass 14 |
| b4 Scoring a court model on three games' windows | about 20 min by court replay, about 1 h by full rebuild | ROADMAP_LOG.md side study |
| b5 Teacher proposals (Grounding DINO, YOLOv8x-pose and the detector) on 6,197 frames | 197 min on a T4 | DEVLOG.md 2026-07-18 |
| Seed spread on a new arena (court, r23 recipe, 3 seeds) | CLE p50 2.24 to 2.57 ft, p90 5.47 to 6.74 ft | ROADMAP_LOG.md side study, third pass |

Every Colab run so far needed Lucien to open the notebook under the right account (ROADMAP_LOG.md, "Waiting on
Lucien"). GPU time is therefore also Lucien's time.

### 5.2 Estimates per experiment

All of these are estimates. A100 times for detection are not measured here. Detection times scale b1 linearly in
images at the same epochs.

| Work | GPU (estimate) | Mac CPU under the lock (estimate) | Basis |
| --- | --- | --- | --- |
| Phase A: camera truth for about 10 new games | about 0.7 A100-h a game, about 7 A100-h in all | or about 1.5 h a game on the Mac (2,450 s x 2.15 s) | b2 |
| Phase A: baseline r23 rebuild on development games | none | about 0.3 h a game | b4 |
| P8-E1 detector on teacher boxes, 3 seeds | teacher pass on 6k to 12k frames: 3 to 6 T4-h; training 6k to 12k images: 5 to 9 T4-h a seed, 15 to 27 T4-h for 3 seeds | about 0.3 h per game per seed to score | b1, b4, b5 |
| P8-E2 events from play-by-play and the clock | none | under 1 h a game (no basis measured; the clock time maps already exist for rebuilt windows) | none |
| P8-E3 team embeddings (tracking-E1) | none | 4 to 8 h in all | docs/research/tracking.md |
| P8-E4 jersey identity with a VLM | no repository basis; measure on one window first | about 0.3 h a game for a rebuild that saves crops | b4 |
| P8-E5 court on new-arena labels, 3 seeds | 1 to 2 A100-h a seed, 3 to 6 in all | about 20 min per model by court replay | b3, b4 |
| P8-E6 learned fill (tracking-E2) | a few T4 or A100 hours | under 1 h | docs/research/tracking.md |
| P8-E7 ball detector in the image | no repository basis and no source checked; measure with a one-epoch pilot | about 0.3 h a game to score | none |

Phase A plus the top three experiments (E1 to E3): about 25 to 40 GPU-hours, 20 to 40 Mac hours under the lock,
and 3 to 5 Colab sessions run by Lucien. Phase A alone fits on the Mac if Lucien prefers no Colab session.

## 6. Proposed targets, set before any work

### 6.1 Existing Stage 1 rows

These targets are Lucien's (ROADMAP.md, confirmed 2026-10-03) and stay as they are. They are scored on the new CV
test set instead of the spent games, with the amended adoption rule (D5). The evidence says how likely each is:

- Position, p50 2.0 ft and p90 5.0 ft. Met on gsw_phx on the point estimate only. On a new arena the seed spread
  alone (p90 5.47 to 6.74 ft) is larger than the gap. Likely only with new-arena labels (E5) and 3 seeds.
- Near-field bias within 0.5 ft. The far third is +1.5 to +2.0 ft. r23_cam reaches +1.27 ft. Unlikely without a
  camera model.
- Missed players and ghost boxes, 5 percent each. Cannot be judged until Phase A remeasures them (section 2.3).
- Team labels, 95 percent. At the published level of learned embeddings. E3 alone is not expected to reach it
  (the dossier's own expectation).
- Coverage, 50 percent. A per-broadcast failure (cle_gsw). It needs the court stage, not detection.

Lucien should decide now whether "done" means every row passes (likely not reachable) or the targets below plus
improvement under the adoption rule on every Stage 1 row.

### 6.2 New rows (proposed by this study; the lead and Lucien confirm them before work)

The truth for every new row is SportVU and the contract's play-by-play table, projected through the camera-model
truth. Every new row is reported per arena. Pass is judged on the point estimate, with a game-bootstrap interval
printed beside it, as the existing rows are.

| Row | Metric | Target | Basis |
| --- | --- | --- | --- |
| Ball, in the image | F1 on live wide frames where the SportVU ball projects inside the frame. A hit is within a pixel tolerance fixed on development games before any ball model runs: twice the camera truth's median reprojection error on SportVU feet | F1 >= 0.75 | WASB reports F1 82.6 at 4 px on its basketball split, not directly comparable. We set lower because our labels are a projected sensor ball with truth error, and because every live wide frame counts, occlusion included |
| Ball carrier | Share of live frames with a SportVU holder (one player within 3 ft of the ball, ball below 6 ft; both numbers fixed now and never tuned) where the pipeline names the same tracked player | >= 85 percent | No published basketball broadcast number found. This is a judgement of what the shot and EPV lanes need, stated as such |
| Identity | Share of matched player-frames whose assigned roster player equals the SportVU player | >= 80 percent (pass or fail) | Koshkina and Elder 87.4 percent tracklet accuracy in soccer (not directly comparable). Our 720p OCR read rate of 30.8 percent (CLAIMS.md A4) means reading alone cannot get there; voting over a tracklet and the roster and team constraints must |
| Identity, reported | Jersey tracklet accuracy on tracklets with at least 8 crops; id switches per possession | reported; goal 5 or fewer switches per possession (today 14.29 to 31.07, lower bounds) | Hu et al. use 8 crops per player |
| Shots | Share of field-goal attempts in live wide windows found within 1.0 s of the SportVU release frame; false events per true event | >= 95 percent found, <= 5 percent false | The clock OCR resolves 1 s (ROADMAP.md, known traps). The play-by-play lists every attempt, so the floor is the sync |
| Rebounds | Same, at the moment the ball is secured | >= 90 percent found within 1.0 s | Barrera-Roy and Clapés reach spotting mAP 0.46 from video alone (not directly comparable: we read the play-by-play) |
| Shooter position | Position of the shooter's track at release vs SportVU | p50 <= 3.0 ft, p90 <= 6.0 ft | The position row (2 and 5 ft) plus up to 1 s of timing error on a moving player. Needs identity (E4) or the ball (E7) |
| Rebounder position | Position at the rebound vs SportVU | p50 <= 4.0 ft | As above; rebounds happen in crowds. Needs identity or the ball |
| Swap test (D3) | For each lane fed, the lane's output on broadcast inputs vs on SportVU inputs, as in `demo/swap.json` | Per lane, set by that lane's owner before the run | D3. On the new games the analytics models were fitted on their SportVU, so a swap test there measures input substitution only, never model accuracy |

## 7. Prerequisites

1. **A CV test set, fixed and hashed before any experiment (new D line).** Set A has no usable video. Proposed
   rule: the test set is whole arenas, chosen now by a rule that reads no result. The arenas are the ones whose
   games are dated without a scorebug check: Miami (Pelicans at Heat, Thunder at Heat), Toronto (Spurs at
   Raptors) and Staples Center (Clippers at Lakers). That is 4 games in 3 arenas. Video ids and SportVU archive
   blob hashes go into the D77 ledger file through `tools/research_testset.py`, under one more guarded key. Each
   game is scored once per experiment. The other channel games are for development after their scorebug check.
2. **Restate P8's definition of done against that set (new D line).** The held-out game and arena are spent.
3. **A training-only development split.** The 4 R1.3 games plus the confirmed new games from other arenas.
   Bootstrap by game (with 4 games only 35 resamples exist, as the dossier notes), with the window bootstrap
   printed beside it. Any trained model needs at least 3 seeds before a new-arena comparison (ROADMAP_LOG.md side
   study).
4. **Scorecard v2, read-only to the team.** `sportvu/` is frozen (CLAUDE.md rule 3), so it is new code under
   `tools/` that imports the frozen scorer's functions and changes none. It scores every live wide frame where the
   camera truth is accepted, not only the B4 testable set. It adds the ball, identity and event rows. It logs, per
   observation, the crowding (distance to the nearest other SportVU player), the player's speed, the court zone
   and the time since the last cut, so section 2.4's breakdowns exist on development games. A guard test fails if a
   teammate's branch changes the scorer or its constants, like `tests/test_research_testset.py`.
5. **A baseline before any change.** r23 rebuilt with frozen code on the development games and scored by scorecard
   v2. Outputs go outside DATA_ROOT and outside `reports/` (CLAUDE.md rule 6), under the heavy-job lock.
6. **Labels.** Positions, team, SportVU ids, jerseys and ball come from the synced logs. Boxes for missed players
   come only from an outside teacher prompted at the projected SportVU feet, and the teacher must first beat the
   detector on the human box truth (LABEL_SCHEMA.md, anti-inbreeding rules). The play-by-play in the contract comes
   from basketball-reference. Its terms forbid using its content to train AI models (DATA.md, question 4). Using
   it as scoring truth is what the frozen lanes already do. Using it as training labels needs Lucien's decision.
7. **GPU budget and Lucien's time.** About 25 to 40 GPU-hours and 3 to 5 Colab sessions for Phase A and Phase B
   (section 5.2), approved before Phase B starts.
8. **Rights.** Lucien decides whether to register 10 more third-party uploads under CLAIMS.md caveat 4.

## 8. Ranked experiments

Gain and effort are on a 1 to 5 scale. Gain counts the failing rows the experiment moves and the lanes it unlocks.
Effort counts new code, labels, GPU and Lucien's time. Every experiment is scored on development games under
scorecard v2. Each is adopted under the D5 rule with 3 seeds where it trains, then scored once on the CV test set
and reported whatever the result.

| Rank | Experiment | Rows it moves | Gain | Effort | Cost (estimate) |
| --- | --- | --- | --- | --- | --- |
| 1 | E2 events from the play-by-play and the clock | shots, rebounds (timing) | 4 | 2 | no GPU; a few Mac hours; 3 to 4 agent days |
| 2 | E1 detector trained on teacher boxes at SportVU feet, referee as its own class | missed, ghosts; feeds team, identity, coverage, fill | 5 | 3 | 18 to 33 T4-h, 2 Colab sessions, 5 to 7 agent days |
| 3 | E3 team labels from tracklet embeddings (tracking-E1) | team | 3 | 2 | no GPU; 4 to 8 Mac-h; 1 to 2 agent days |
| 4 | E4 jersey identity (VLM, tracklet voting, roster and team constraints) | identity, shooter and rebounder position | 4 | 3 | GPU unknown until a pilot; crops need a rebuild |
| 5 | E5 court model on new-arena labels, 3 seeds | position (arena), bias, generalisation | 3 | 3 | 3 to 6 A100-h plus 0.7 A100-h a game of labels |
| 6 | E6 learned fill of missed player-frames (tracking-E2) | the demo's sensor-fill share | 4 | 4 | a few GPU hours; 3 to 5 agent days |
| 7 | E7 ball detector in the image (WASB-style) | ball rows | 4 | 5 | unknown until a pilot; the largest |

E2 ranks first. It turns a free, already-aligned source (the play-by-play with SportVU frames) into the event rows
at almost no cost, and it is the step every shot and rebound lane needs. It does not place the shooter: that part
waits for E4 or E7, and the row says so. E1 has the largest gain, but it ranks second because its size is unknown
until Phase A remeasures missed and ghosts (section 2.3). If scorecard v2 puts missed players under 10 percent,
E1's gain drops to 3 and E4 moves up. E3 is cheap and pre-registered already (docs/research/tracking.md). A P5-style
automated loop could be used inside a stage for knob-level search, such as tracker thresholds, scored on
development games only.

## 9. Team plan

- **Lead** (main session): owns PLAN.md, the D lines of section 7, merges, and every test-set score.
- **Judge** (read-only): runs scorecard v2 the same way for every candidate on development games, writes the
  ledger, and never edits code. The ml-reviewer and verifier close each milestone (D38).
- **Measurement teammate** (Phase A): scorebug checks of the new games (one frame each), sync, camera truth,
  scorecard v2 and its guard test, the baseline, and the error analysis on development games.
- **Detection and appearance teammate** (Phase B): E1, then E3, then E4 (one worktree, because all three read the
  same boxes and crops).
- **Events and ball teammate** (Phase B): E2, then a pilot for E7.

That is three teammates, not P8's five. CLAUDE.md rule 7 allows one heavy job at a time on the Mac. Five teammates
would mostly wait on the lock, and every GPU run waits on Lucien anyway. Court and camera (E5) and fill (E6) start
only after Phase B's first adoption decision.

**Gate between Phase A and Phase B.** Phase B starts when all of these hold:

- Scorecard v2 reproduces r23's position rows on development games within the window-bootstrap interval of the
  old truth.
- The human box truth and scorecard v2 agree on the detector's miss rate within a margin written down before the
  check.
- The CV test set is hashed.
- Lucien has approved the GPU budget.

If the scorecard cannot be made trustworthy, P8 stops there, and the measured scorecard is still a result.

## 10. Go or no-go, and P5 or P8

**No-go for P8 as written.** Its done condition is scored on spent games (D71). Set A gives it no video. Two of its
rows (missed players, ghosts) are not measured well enough to steer work. Five teammates against one heavy-job
lock and Lucien-run GPU sessions would mostly wait.

**Go for re-scoped P8.** Phase A comes first: measurement, the new CV test set, scorecard v2 and the baseline,
mostly CPU, roughly 1 to 2 weeks of agent work (estimate). Then Phase B runs E2, E1 and E3 under the gate. Why:

- The measurement work is needed whatever is chosen, P5 included.
- The new channel games answer the scaling study's open question (new games in new arenas of the same era) and
  give the CV lane its first untouched multi-arena test.
- The top three experiments cost about 25 to 40 GPU-hours in all.
- The stop point after Phase A limits the loss if the truth cannot be fixed.

**P8 rather than P5.** P5 is an automated improvement loop on training games. It would optimise today's
scorecard, whose detection rows are in doubt and which scores 4.7 to 15 percent of frames. It would search over
four development games from one arena, which is where an automated search overfits fastest. And it cannot add what
is missing: a ball, identity and events need new stages, labels and rows, not a search over settings. P5's held-out
score is spent too. P5's loop stays useful as a tool inside a P8 stage once scorecard v2 exists.

## 11. Sources checked (2026-11-16)

| Source | Link | Used for |
| --- | --- | --- |
| Tracking dossier and its sources | docs/research/tracking.md, sections 3 and 7 | Ceilings (reused, not refetched) |
| Data survey | docs/research/DATA.md | Set A, SkillCorner quality, play-by-play terms |
| Ramanathan et al., CVPR 2016, arXiv abstract and PDF | https://arxiv.org/abs/1511.02917 | Events ceiling, Tables 1 to 3 |
| Barrera-Roy and Clapés, arXiv HTML | https://arxiv.org/html/2512.15386v1 | Rebound ceiling, Tables II and III |
| YouTube metadata (flat search, format listing, one channel's upload list; nothing downloaded) | https://www.youtube.com | Section 4 |
