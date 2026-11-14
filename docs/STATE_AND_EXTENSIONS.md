# STATE_AND_EXTENSIONS.md: where the project is and what it can carry

Audit date 2026-10-19. State as of commit `0ca0e4c` (2026-10-06), branch `track-fix`. Sources: this
repository only. Where a fact is not in a file here the text says "not in repo". Source hierarchy
(higher wins): measured artifacts under `reports/`, `data/pbp/tier2_credit.json`,
`data/harvest/games.json`, `sportvu/splits.json`; then CLAIMS.md; then ROADMAP.md and ROADMAP_LOG.md
(newest pass wins); then PIPELINE.md, PROJECT_SUMMARY.md, README.md, FIX_PLAN.md, FIX_LOG.md; then
DEVLOG.md; then git log for chronology. Every number carries its path, its date and its CLAIMS status.
Date convention: the `date` field inside the file where one exists (the scorecards and
`reports/sportvu_manifest.json` have one), otherwise the file's last git commit date, otherwise its
modification time (marked mtime). Numbers measured on the 7 prototype clips are marked PROTOTYPE;
numbers measured against SportVU on held-out footage are marked HELD-OUT; harvest-domain human
labels are marked HARVEST. The three kinds never share a sentence.

## 1. State as of commit 0ca0e4c (2026-10-06)

Finding: Stage 1 is inside Phase 2 (court). R0.1 to R2.3 are ticked, R2.4 is measured but not
ticked (waiting on a [HUMAN] decision), R2.5 to R5.4 are open; Phases 3, 4 and 5 have no started item.

| Item | Value | Source |
|---|---|---|
| Branch | `track-fix`, in sync with `origin/track-fix` (0 unpushed commits), 51 commits ahead of `main` (`f726066`) | `git status -sb`, `git log main..HEAD` |
| HEAD | `0ca0e4c` 2026-10-06 "R2.4 camera model measured: near bias within 0.5 ft on all held-out games, phx meets position targets; adoption pending" | `git log -1` |
| Frozen tag | `v3-frozen` (tag object `544aa34`) points at `07bf4e6` on `main`; pushed 2026-10-03 after a history rewrite that removed 7 clips, `paper/` and 2 raw frames | `git show-ref --tags -d`; ROADMAP_LOG.md "Publication (2026-10-03)" |
| Commits | 136 on `track-fix` (first commit 2026-07-05, V3 first commit) | `git log --oneline` |
| Tests | 87 collected (`pytest --collect-only -q tests`, run 2026-10-19); ROADMAP_LOG pass 15 says "pytest 87 passed" | tests/ |
| Untracked at audit | `data/gate_sheet/emb_cache_clip.pkl`, `data/matchup_eval/labels.json`, `data/matchup_eval/labels_discarded_20260716_2343.json`, `logs/`, `portfolio/story/`, `reports/broadcast_wishlist.csv`; `docs/` did not exist before this file | `git status --short` |
| Production default | V3 grid model (`models/court_grid_snapped.pt`); r23 adopted as the court baseline for R2.4 onward but switched in only through `COURT_GRID_WEIGHTS` | CLAIMS.md "Court stage candidate r23"; ROADMAP.md R2.3 |

### 1.1 ROADMAP.md phases (ticks as written in ROADMAP.md at 0ca0e4c)

| Phase | Ticked | Unticked |
|---|---|---|
| 0 Freeze and reframe | R0.1 tag; R0.2 targets into CLAIMS.md; R0.3 scorecard CLI; [HUMAN] targets confirmed 2026-10-03 with two changes (court line row reported only; held-out era label-free only) | none |
| 1 SportVU-synced data | R1.1 availability manifest (84 games); [HUMAN] games and held-out arena CLE (2026-10-04); R1.2 fetch, split, clock-calibrate, sync; R1.3 auto-labels (13,975 camera-truth labels, 11,506 kept by the paint check); R1.4 `sportvu/splits.json` and a held-out guard in every training script | none |
| 2 Court | R2.1 V3 baselines on phx and CLE; R2.2 public court model baseline (V3 stays the starting point); R2.3 r23 grid model ADOPTED 2026-10-04 under the amended rule | R2.4 camera model (measured in pass 15, 2026-10-06, not ticked: adoption decision for Lucien); R2.5 acceptance and LOST; R2.6 temporal smoothing; R2.7 generalisation report (amended 2026-10-06: add new-arena games while the held-out arena misses the court targets, not only above 1.5x); [HUMAN] court-stage acceptance |
| 3 Detection and tracking | none | R3.1 baseline scorecard; R3.2 off-the-shelf detector comparison; R3.3 retrain on SportVU-confirmed boxes; R3.4 pose feet; R3.5 tracker comparison; [HUMAN] |
| 4 Teams and identity | none | R4.1 kit classifier vs foundation-model embedding; R4.2 identity (jersey OCR, VLM reader, ReID); [HUMAN] |
| 5 Product and reattachment | none | R5.1 side-by-side demo; R5.2 re-harvest 14 production games; R5.3 update CLAIMS/PIPELINE/paper/portfolio; R5.4 repo clean-up; [HUMAN] publishing, Stage 2 planning |
| Waiting on Lucien | targets gate; games and arena gate; tag push decision | [HUMAN] non-gating: retrain gate v2 without the 882 held-out frames or keep it with the CLAIMS caveat |

FIX_PLAN.md (superseded for Phases C and D on 2026-10-03): Phase A done except A1 [BLOCKED] (kept as
a ranking tool); Phase B B1 to B6 done; its [HUMAN] "confirm the Phase C order" gate is unticked and
moot. Side study (ROADMAP_LOG.md 2026-10-05/06, not a plan item): data-scaling experiment, 3 passes,
22 arms; decision recorded: r23 stays the court baseline, no scaling-study model adopted.

### 1.2 Disagreements between files (both paths given; the higher source wins)

| Topic | Lower source says | Higher source says | Resolution |
|---|---|---|---|
| Training games | ROADMAP.md Phase 1 [HUMAN] gate text: "Training (Oracle): 12.25.2015 CLE (QJLso8Uwklc), 11.14.2015 BKN, 01.22.2016 IND, 12.28.2015 SAC, 01.04.2016 CHA" (5 games); ROADMAP_LOG.md "Gate (2026-10-04)": "training = 5 Oracle games" | `sportvu/splits.json` (2026-10-04): train = 4 games (bkn, cha, ind, sac); `data/harvest/games.json` `gsw_cle_xmas15` has `excluded: NBA 2K16 simulation` | 4 training games. ROADMAP.md R1.3 and ROADMAP_LOG pass 5 already say 4; the gate text was not updated |
| V3 position error on phx | CLAIMS.md A8 and README.md Status: p50 7.3 / p90 15.4 ft (B5, `reports/sportvu_phx.txt`, 2026-10-03: 7.31 / 15.38, 11,721 matched players) | `reports/scorecard/gsw_phx_2016__v3.txt` (2026-10-04): 7.33 / 15.35 ft, n = 11,292 players | Scorecard wins; CLAIMS.md itself says the bench re-matches feet through H_truth at 5 ft and reproduces B5 within 0.03 ft. The B5 numbers are superseded |
| Court line row | ROADMAP_LOG.md pass 3 and STOP pass 3: V3 "court line 0.0% <= 3 px (p50 26.7 px)" | CLAIMS.md Stage 1 table (corrected 2026-10-04) and every scorecard txt: "not measurable (truth H fits the feet, not the court)"; `reports/sportvu_truth_line_check.txt` | Withdrawn; appears only as superseded below |
| Stage numbers for gate, detection, homography, teams | PIPELINE.md (header "as of 2026-07-08") rows 1 to 5 and PROJECT_SUMMARY.md (last commit 2026-10-02) present 98.7%, P .89 / R .87, 0.30 ft median, 87.1% as validated results with no held-out tracking number | ROADMAP.md "Where the project stands" (2026-10-03): those were measured on the 7 prototype clips the models were tuned on; the held-out SportVU scorecards give position p50 7.33 ft and team labels 81.8% for V3 | PROTOTYPE numbers stand as prototype measurements (CLAIMS A2 still SHIPPED); the public documents lack the held-out context. See section 6 |
| Test count | PROJECT_SUMMARY.md lines 21 and 241: "52 tests"; `portfolio/numbers.json` hand_typed_sources: "52 collected (2026-09-12)" | `pytest --collect-only -q tests` on 2026-10-19: 87 collected; ROADMAP_LOG pass 15: "pytest 87 passed" | 87 |
| Game count | README.md Status: "the 14-game production harvest"; PROJECT_SUMMARY.md: "641 joined possessions across 16 games"; `reports/tier2_bias_audit.txt`: "16 games" | `data/pbp/` holds 16 game files; `data/harvest/games.json` holds 14 production games plus 6 Stage 1 games plus 1 excluded. `data/pbp/tier2_join.json` sources include `clip_*` and `curry_q1_clip` (prototype clips from 2 further games) | Both are right with different denominators: 14 harvested games, 16 games with play-by-play once the prototype clips' two games are counted |
| Season span | `portfolio/numbers.json` hand_typed_sources: "PROJECT_SUMMARY.md and paper say 2013-2017, wrong by a season" | PROJECT_SUMMARY.md now says "2012-2017"; CLAIMS.md kill list says "2012-2017 pooled" | The portfolio note is stale |
| `reports/tier2_bias_audit.txt` date | last git commit 2026-07-16 | file mtime 2026-09-11 (regenerated with the 2026-09-11 corpus correction, same values) | Cited as 2026-07-16 (committed), content unchanged |

## 2. Scoreboard

Finding: on the held-out game the best measured build (r23_cam, not adopted) meets position error
and coverage; on the held-out arena no build meets position error; missed players, ghost boxes and
team labels are unchanged by the court stage and all fail.

Targets from CLAIMS.md "Stage 1 definition of done" (CONFIRMED by Lucien 2026-10-03). All
HELD-OUT, measured by `python -m sportvu.bench` on the B4 testable frames only (phx 1,447 of 9,500
window frames, 15%; cle_nyk 485 of 6,393; cle_gsw 158 of 3,380). Builds: v3 = production models;
r23 = adopted court baseline (2026-10-04, CLAIMS "adopted under the amended rule"); r23_cam = R2.4
camera model on r23 weights (2026-10-06, measured, adoption pending). Artifact paths:
`reports/scorecard/<game>__<build>.txt`, dates from the `date` field of the matching `.json`.

### 2.1 Held-out game gsw_phx_2016 (2015-12-16 PHX at GSW, Oracle)

| Row | Target | V3 (2026-10-04) | r23 adopted (2026-10-04) | r23_cam best (2026-10-06) | Met by r23_cam | Superseded |
|---|---|---|---|---|---|---|
| Court line error | reported; goal <= 3 px on 95% once truth residual <= 1.5 px | not measurable (diagnostic p50 26.74 px) | not measurable (16.68 px) | not measurable (16.43 px) | reported, not judged | "0.0% <= 3 px, p50 26.7 px" (ROADMAP_LOG pass 3) withdrawn 2026-10-04 |
| Position error p50 / p90 ft | <= 2.0 / <= 5.0 | 7.33 / 15.35 (n = 11,292) | 1.92 / 4.98; p90 CI [4.42, 6.02], P(p90 <= 5.0) = 0.51 (`reports/r23_verification.txt`) | 1.84 / 4.95 | MET (point estimate; no CI for r23_cam's own pass probability in repo) | B5 7.31 / 15.38 (`reports/sportvu_phx.txt`, 2026-10-03) |
| Near-field bias near / far ft | within +-0.5 | -1.02 / +1.67 | -0.67 / +1.52 | -0.27 / +1.27 | not met (near met, far not) | |
| Coverage | floor >= 50% of live wide s | 98.6% (838.1 of 849.8 s) | 99.3% | 99.3% (844.0 of 849.8 s) | MET | |
| Missed players | <= 5% | 25.1% (n = 14,020) | 25.1% | 25.1% | not met | |
| Ghost boxes | <= 5% non-referee | 15.1% (15.3% with referees; n = 12,637) | 15.1% | 15.1% | not met | |
| Team labels | >= 95% | 81.8% (n = 10,829) | 81.9% (n = 10,830) | 81.8% | not met | A8 "81.8% (n = 11,240)" from B5 |
| Identity | reported | 21.68 id switches / possession (n = 56); jersey not measurable | 21.39 | 21.68 | reported | |

### 2.2 Held-out arena CLE (12.23.2015 NYK at CLE; 01.18.2016 GSW at CLE)

| Row | cle_nyk V3 | cle_nyk r23 | cle_nyk r23_cam | cle_gsw V3 | cle_gsw r23 | cle_gsw r23_cam | CLE pooled r23 to r23_cam (`reports/camera_track.txt`, 2026-10-06) |
|---|---|---|---|---|---|---|---|
| Position p50 / p90 ft | 11.04 / 18.40 (n = 3,862) | 2.25 / 5.46 | 2.17 / 5.49 | 9.51 / 17.14 (n = 1,178) | 2.22 / 5.50 | 2.26 / 5.45 | 2.24 / 5.47 to 2.19 / 5.49; not met |
| Near / far bias ft | +2.87 / +7.61 | -0.63 / +1.66 | -0.47 / +1.30 | +1.60 / +4.50 | -0.12 / +2.00 | +0.26 / +1.89 | abs 0.53 / 1.71 to 0.35 / 1.41; near met, far not |
| Coverage | 90.5% | 92.3% | 92.3% | 38.5% | 39.7% | 39.7% | cle_gsw below the 50% floor for every build |
| Missed / ghosts / team | 24.4% / 14.8% / 74.7% | 24.4% / 14.8% / 73.7% | 24.4% / 14.8% / 74.7% | 26.2% / 14.8% / 78.4% | 26.2% / 14.8% / 78.2% | 26.2% / 14.8% / 78.4% | all not met |
| Identity | 31.0 switches / possession (n = 15) | 31.07 | 31.0 | 14.29 (n = 7) | 14.29 | 14.29 | reported |
| Generalisation (ratio to phx, p50 / p90) | 1.51x / 1.20x not met | 1.17x / 1.10x MET | 1.18x / 1.11x MET | 1.30x / 1.12x MET | 1.16x / 1.10x MET | 1.23x / 1.10x MET | within 1.5x; the absolute court targets are still missed (R2.7 amendment) |

Paired changes r23 minus r23_cam, window-cluster bootstrap 95% (`reports/scorecard/compare__r23_replay__r23_cam.txt`, 2026-10-06; positive = camera better): phx p50 +0.08 [+0.03, +0.10], p90 +0.03 [-0.01, +0.10], abs_near +0.40 [+0.32, +0.45], abs_far +0.25 [+0.17, +0.33]; cle_nyk p50 +0.08 [+0.03, +0.11], abs_far +0.36 [+0.10, +0.48]; cle_gsw p50 -0.04 [-0.19, +0.11], abs_near -0.14 [-0.39, +0.66], every cle_gsw interval crosses zero.

Caveats printed on every scorecard (`reports/scorecard/*.txt`): testable frames are the B4 set where V3's H matched >= 6 players, so the set leans toward frames V3 handled; the truth H is fitted through V3's own boxes and is valid only near those feet (so missed and ghost counts carry that error; position error and bias stand); local windowed rebuild, not the production artifacts; wide reference is gate v2, which saw 294 phx frames in training; referee recall never measured; id switches are a lower bound; team accuracy uses the best per-window A/B mapping (upper bound). r23 adds (CLAIMS.md): phx shares arena and season with the training games, so CLE is the cross-arena evidence; V3's Roboflow grid set holds 56 frames of 2023-24 Cleveland home games, so CLE is unseen games and season in a building seen in 2023-24 footage; about 2 ft may be near the truth's own floor.

### 2.3 PROTOTYPE and HARVEST numbers (not held-out tracking; CLAIMS A1 to A7 SHIPPED)

| Number | Kind | n | Path | Date | CLAIMS |
|---|---|---|---|---|---|
| Gate accuracy 0.987, macro-F1 0.986, 1 FN / 1 FP, threshold 0.7004546103892618 | PROTOTYPE test split | 158 frames (99 live / 59 dead) | `reports/metrics.txt` | mtime 2026-06-23 (never committed) | A2 SHIPPED |
| Gate on harvest frames at 0.70: precision 0.988, recall 0.859, FN 500, FP 37; at 0.35 precision 0.912, FP 343; objective picks 0.42 (recall 0.998, precision 0.933); max accuracy 0.975 at 0.55 | HARVEST, human-verified | 6,159 frames, 3,544 wide | `reports/gate_sheet.txt` | 2026-10-02 | Standing caveat 1 |
| Gate v2 (refit) LOSO acc 0.970, FP-rate 0.068 at recall 0.998, worst source 0.948, prototype test 0.975 at 0.33 | HARVEST | 6,159 frames / 21 sources | `reports/gate_retrain_experiment.txt` | 2026-10-02 | Standing caveat 1 (shipped for future harvesting; published sample still v1 at 0.70) |
| Detection P 0.887 / R 0.865 / F1 0.876 at IoU 0.5 | PROTOTYPE | 50 frames, 453 GT boxes | `reports/tracking_metrics.txt` | mtime 2026-06-26 (never committed) | A2 SHIPPED |
| Homography success 100%, reprojection median 0.302 ft, p90 0.511 ft; keypoint px error mean 71.15, median 12.46 | PROTOTYPE val | 28 frames | `reports/homography_metrics.txt` | mtime 2026-06-24 (never committed) | A2 SHIPPED |
| Teams 83.9% crop (118) / 87.1% track (31 tracks) | PROTOTYPE, one clip | 160 crops | `reports/teams_eval_curry_q1_clip.txt` | 2026-07-06 | A2 SHIPPED |
| Possessions: attacked basket 0.965, offense 0.912 | PROTOTYPE, 4 clips | 57 scorable of 73 | `reports/poss_eval.txt` | 2026-07-06 | A2 SHIPPED |
| Clock 55 of 56 exact, 1 human-unreadable, 0 confident-wrong | HARVEST layouts | 56 | `reports/clock_eval.json` | 2026-07-07 | A2 SHIPPED |
| Play-by-play agreement 0.878 (869 / 990); non-duplicate 0.896 (n = 904); overlap >= 0.80 0.88 (n = 933); anchor failures 248 | HARVEST corpus | 151 clips, 991 aligned | `reports/tier2_crossval_corpus.txt` | 2026-09-11 | A1 SHIPPED (corrected from 91.1%) |
| Funnel selection offset +8.8 pts/100; joined 641 of population 3,191 (halfcourt 1,768) | HARVEST | 16 games | `reports/tier2_bias_audit.txt` | 2026-07-16 (mtime 2026-09-11) | B3 SHIPPED |
| GSW credit -7.4 per 100, CI [-21.6, 6.3], n = 330 possessions, 15 games; 1 of 12 rows meaningful (min bucket 300) | HARVEST | 641 joined | `data/pbp/tier2_credit.json` | mtime 2026-09-11 (last commit 2026-07-16) | B1, B2 SHIPPED; B4 STANDING |
| Auto-label audit: agreement band 0.95 [0.888, 0.978] (n = 100); judge bands 0.067 to 0.2 (n = 200) | HARVEST corpus | 300 | `reports/label_audit.txt` | 2026-09-12 | A5, A6 SHIPPED |
| shot_type: closeup 0.98 [0.895, 0.996] (n = 50); wide_broadcast 0.688 [0.579, 0.778] (n = 80) | HARVEST corpus | 200 | `reports/shot_type_audit.txt` | 2026-10-02 | A7 SHIPPED |
| Nearest-defender proxy for C1: 0.481 on 160 of 224 defenders | HELD-OUT (proxy, not human labels) | 160 | `reports/sportvu_phx.txt` | 2026-10-03 | A8 PROPOSED; C1 OPEN |

## 3. What the pipeline outputs today

Finding: the production chain (V3) writes 156 trajectory files and 151 possession, matchup and
outcome files, joined into 641 credit rows; the Stage 1 rebuild writes per-window frame sidecars,
truth files, camera-truth labels and scorecards for 8 SportVU games. No file carries a ball position
from video, a player name, or a referee class on a box.

Schemas read from the named file. Counts are over the whole repository on 2026-10-19.

| Output | Stage (code) | Fields (read from the file) | Count and coverage | File read |
|---|---|---|---|---|
| `data/tracking/<clip>_trajectories.json` | 2 to 5, `build_trajectories.py` | dict `track_id` -> {`team` int or null, `raw` [[frame, x_ft, y_ft]], `cleaned` [[frame, x_ft, y_ft, extrap_bool]]} | 156 files, 27,842 tracks, 1,636,106 raw points, 1,915,931 cleaned points (the cleaned series interpolates gaps; PIPELINE.md: mask presence with `raw`) | `data/tracking/gsw_hou_duel_s01_trajectories.json` |
| `data/tracking/<clip>_identity.json` | 4 to 5, `detect/reid.py`, `detect/teams.py` | `physics_raw` {steps, impossible_step_pct, step_p50_ft, step_p99_ft, off_court_pct}; `identity` {fragments, canonical_tracks, merges, ambiguous_refused, team_vetoed, median_players_per_frame, id_churn_before, id_churn_after, footpoint_clip_fixes, camera_cut_resets, teams{silhouette, tracks_team_A, tracks_team_B, tracks_abstained, frame_balance_median}}; `merges[]`, `refused[]`, `idmap`, `team_by_track`, `team_colors_bgr`, `light_team` | 156 files | `data/tracking/gsw_hou_duel_s01_identity.json` |
| `<build>/<window>_frames.json` sidecar | 2 to 3, `build_trajectories.py` (FIX_PLAN A0) | {source, fps, stride, frames[{frame, state TRACK / LINE_TRACK / HELD / LOST, H 3x3, quality_ft, n_inliers, res_px, n_match, hull, boxes[{tid, bbox, foot_raw, foot_stab, foot_corrected}]}]} | 71 windows in each of `data/sportvu/build`, `build_r23`, `replay/r23_cam`; 55 triage renders summarised in `reports/qc/` | `data/sportvu/build_r23/gsw_phx_2016_s00_w00_frames.json` (646 frames) |
| `data/tracking/<clip>_possessions.json` | 6, `segment_possessions.py` | {source, stride, fps, spans[{kind halfcourt or transition, start_frame, set_start_frame, end_frame, core_start_frame, core_end_frame, core_s, metrics_eligible, duration_s, approach_s, set_s, attacked_basket, occupancy_x_med, defense_team, offense_team, confidence, mean_dist_to_basket_ft}]} | 151 files, 6,376 spans (2,034 halfcourt, 4,342 transition); `metrics_eligible` exists on halfcourt spans only, true on 1,239 of 2,034 | `data/tracking/gsw_hou_duel_s01_possessions.json` |
| `data/tracking/<clip>_matchups.json` | 7, `matchup_metrics.py` | {clip, possessions[{start_frame, set_start_frame, end_frame, core_start_frame, core_end_frame, core_s, attacked_basket, offense_team, defense_team, confidence, coverage_pairs_per_frame, frames_used, frames_excluded_team_gt5, pct_excluded, degraded, team_count_distribution, spacing_conceded_ft2{median, p25, p75}, defenders[{defender, primary_man, time_assigned_s, men{}, matchup_dist_mean_ft, matchup_dist_median_ft}]}], skipped_spans[{set_start_frame, reason}]} | 151 files, 1,183 possession records, 168 degraded; skipped: core_too_short 795, low_conf 45, offense_unknown 11 | `data/tracking/gsw_hou_duel_s01_matchups.json` |
| `data/pbp/<clip>_outcomes.json` | 8, `align_outcomes.py` | list of {set_start_frame, core_end_frame, attacked_basket, offense_team, confidence, offense_real, defense_real, status, period, anchors[{frame, period, clock_s, raw}], pbp_possession{team, clock, points}, overlap_frac_of_span, offense_points, terminating_event{clock_s, team, kind, desc}, pbp_offense_real, offense_agrees, orientation_flipped, offense_source, orientation_conf} | 151 files, 1,239 rows; status aligned 991, anchor_failed 103, no_pbp_overlap 89, clock_stopped_span 37, anchor_inconsistent 19; offense_agrees true 869 / false 121 / null 249 | `data/pbp/gsw_hou_duel_s01_outcomes.json` |
| `data/pbp/<clip>_orientation.json` | 8 | {set_start_frame: {offense_team, defense_team, orientation_conf}} | 126 files | `data/pbp/gsw_hou_duel_s01_orientation.json` |
| `data/pbp/<game_code>.json` | 8, `fetch_pbp.py` (basketball-reference) | {meta{away, home, date}, events[]} | 16 games (2012-06-07 to 2017-06-12); `201505210GSW.json` has 416 events | `data/pbp/201505210GSW.json` |
| `data/pbp/tier2_join.json` | Tier 2 join, `tier2_join.py` | {checks{joined, excluded, every_join_has_outcome_and_defenders, no_shared_pbp_possession, team_consistency_all, note}, sources{}, joined[{clip, game, set_start_frame, period, pbp_clock, offense_real, defense_real, team_consistency_ok, outcome{points, terminating}, defenders[{fragment, primary_man, time_assigned_s, matchup_dist_median_ft}], primary_defender_fragment, _span_len}], excluded[{clip, set_start_frame, reason}]} | joined 641, excluded 598 (matchup_degraded 114, anchor_failed 103, no_pbp_overlap 89, offense_disagrees_pbp 63, weak_pbp_match 47, clock_stopped_span 37, no_defenders 20, no_matchup_record 19, anchor_inconsistent 19, duplicates 87); all three checks true; 151 source clips | `data/pbp/tier2_join.json` (last commit 2026-07-16) |
| `data/pbp/tier2_credit.json` | credit aggregation, `tier2_credit.py` | {rows[{defense, n_possessions, n_games, ppp_allowed, baseline_ppp, credit_per_100, meaningful, ci95, ci_excludes_zero, baseline_naive_ppp, credit_rel_per_100}], min_bucket, joined_total, note, baseline, ci, credit_rel} | 12 rows, 1 meaningful (GSW n = 330); 7 rows have `ci95` null (1 game each) | `data/pbp/tier2_credit.json` |
| `data/harvest/games.json` | registry, `harvest_driver.py` | per game {video_id, game_code, away, home, date, layout, note}; Stage 1 games add {sportvu, split, layout_check}; one adds {excluded} | 21 games: 14 production (no `split`), 7 Stage 1 (5 tagged `split: train`, of which `gsw_cle_xmas15` also carries `excluded`; 2 `heldout_arena`) | `data/harvest/games.json` (mtime 2026-10-04) |
| `sportvu/splits.json` | R1.4, `sportvu/splits.py` | {version, item, rule, train{arenas, games{arena, sportvu, video_id, frames, labels_before_paint_check}, frames_rule, frames}, heldout_game, heldout_arena, heldout_era, excluded} | train 4 games 11,506 frames; heldout_game 1,447 frames; heldout_arena 3 games (2 synced: 2,454.0 s and 1,870.5 s); heldout_era 1 game | `sportvu/splits.json` (2026-10-04) |
| `data/sportvu/<game>_moments.json` | B1, `sportvu/fetch.py` (gitignored) | {game, gameid, date, teams, players{pid: {name, jersey, team}}, moments[{q, clock, shot, ts, ball [x, y, z], players[[team_id, pid, x, y] x 10]}]} | 8 games (26 to 40 MB each); phx 81,654 moments at 25 Hz. Ball position is present in SportVU, never in any pipeline output | `data/sportvu/12.16.2015.PHX.at.GSW_moments.json` |
| `data/sportvu/truth/<window>_truth.json` | B4, `sportvu/truth.py` | {window{window, section, fps, f_start, f0, f_end, period}, offset_s, offset_median_ft, frames[{status ok or untestable_*, n_boxes, n_matched, inliers, iter, frame, q, clock, moment, state}]} | 71 windows; phx 1,447 ok frames of 9,500 | `data/sportvu/truth/gsw_phx_2016_s00_w00_truth.json` |
| `data/sportvu/labels/<game>/<window>.jsonl` (gitignored) | R1.3, `sportvu/autolabel.py` | rows {game, window, video, frame, section, section_frame, q, clock, offset_s, H_truth, truth = "camera", camera_C, camera_f, feet_resid_ft, line_matches, grid[[idx, x, y]], lines[[x, y]], boxes[{bbox, foot, pid, team, jersey, d_ft}], region, image} | 13,975 rows in 329 window files across the 4 games; `<game>_paint.jsonl` {window, frame, region, status, n, edge_px, edge_px_p90, reject, keep} keeps 11,506 | `data/sportvu/labels/gsw_bkn_2015/gsw_bkn_2015_s00_w00.jsonl`; `gsw_bkn_2015_paint.jsonl` |
| `reports/scorecard/<game>__<build>.json` | R0.3, `sportvu/bench.py` | {game, sportvu_game, build, build_dir, mirror, date, targets{9}, truth_frames, testable, untestable_share, untestable_by_cause, metrics{court_line_error, position_error{matched_players, p50_ft, p90_ft, mean_ft, by_region, pass}, near_field_bias, coverage{live_s, live_wide_s, accepted_live_wide_s, coverage, wide_reference, pass}, missed_players, ghost_boxes, team_labels, identity{possessions, player_observations, id_switches, id_switches_per_possession, jersey_read_rate null}, generalisation}, per_window, method, caveats[7], split} | 13 scorecards (3 games x v3, public, r23, r23_cam, plus gsw_phx_2016__public_conf005) and 3 compare files | `reports/scorecard/gsw_phx_2016__r23_cam.json` |
| `data/sportvu/bench_obs/<game>__<build>.json` | side study, `sportvu/bench.py` | per window: list of 5-element rows, sample [54, 4, 0.8633, 0.1769, "middle"]; field names not in the file | 26 builds x 3 games | `data/sportvu/bench_obs/gsw_phx_2016__r23_cam.json` |
| `reports/qc/<clip>.json` | FIX_PLAN A1 to A4, `qc/track_qc.py` | {clip, video, summary{frames, frames_with_H, frames_scored, rule_counts, frames_failing_any, second_disagreement_median, dist_px_median, dist_px_p90, score_median, support_median, share_far}, frames[]} | 80 clip files plus `_by_game.json`, `_ranking.json`; `reports/sportvu_check_validation.txt` (2026-10-03): no check separates right from wrong frames (A1 rho 0.272 vs H error) | `reports/qc/gsw_phx_2016_s00_w00.json` |
| `<build>/<window>_timemap.json` | B4, `sportvu/local_sync.py` | {section, fps, stride, reads, reads_ok, running_spans, mapped_video_s, mapped_frames, spans[], frames[[...]], ocr_seconds} | one per window | `data/sportvu/build/gsw_phx_2016_s00_w00_timemap.json` |

Not produced anywhere: `<window>_jersey.json` (every scorecard: "jersey OCR (CLAIMS C2) did not run on these windows"); ball positions from video (CLAIMS kill list); player names (identity = fragment, CLAIMS standing caveat 3); shot events (ARCHITECTURE_DIFF.md: "Shot mapping ... missing").

## 4. Extension feasibility map

Finding: every extension on the list is gated by at least one Stage 1 row that is unmet on the
held-out arena, and the six that need the ball are also blocked by the CLAIMS.md kill list ("Ball
tracking, shot-quality models"). The only recommendable work is the open Phase 2 item (R2.4 decision,
then R2.5 to R2.7) and, independent of vision quality, the matchup label set that CLAIMS C1 needs.

Gate shorthand (status from section 2, best build r23_cam, 2026-10-06): POS = position p50 / p90
(met on phx, not met on CLE); BIAS = near-field bias (near met, far not met anywhere); COV = coverage
floor (met phx and cle_nyk, failed cle_gsw 39.7%); MISS = missed players 25.1% (not met); GHOST =
15.1% (not met); TEAM = 81.8% vs 95% (not met); ID = identity (reported; 21.68 switches per
possession on phx, no target); GEN = held-out arena within 1.5x (met). C1 = matchup accuracy vs human
labels (OPEN, 0 usable labels); C2 = jersey OCR read rate (OPEN); BALL = ball tracking (kill list,
no plan item). COURT-GATE = the unticked [HUMAN] court-stage acceptance. Effort S / M / L is this
audit's estimate, not a repo figure.

| Extension | Stage 1 rows that gate it | Gate met | Pipeline outputs consumed | Missing | Repo data that supports it (counts) | Depends on | Public data named in repo | Effort |
|---|---|---|---|---|---|---|---|---|
| Foundation: expected possession value (EPV) | POS, MISS, TEAM, ID, BALL | no (MISS 25.1%, TEAM 81.8%, BALL killed) | trajectories, possessions, outcomes points | all 10 players per frame, ball, player identity | SportVU moments for 8 games with ball and 10 players (phx 81,654 moments); 1,239 outcome rows with points | identity (R4.2), ball tracking (no plan item), action recognition for state features | SportVU 2015-16 logs (linouk23/NBA-Player-Movements, 636 games listed in FIX_PLAN.md; GSW 20, CLE 17, OKC 25, NYK 22 home games in `reports/sportvu_manifest.txt`) | L |
| Foundation: action / play recognition (pick-and-roll, handoffs, screens, cuts) | POS, MISS, ID, BALL | no | trajectories (raw), possessions spans | per-player continuity within a possession (21.68 id switches per possession), ball handler, labels for plays | none: no play labels in repo (not in repo) | court stage, detection (R3), identity (R4.2) | Roboflow basketball-player-detection-3 action classes (jump-shot, layup-dunk, shot-block) noted in ARCHITECTURE_DIFF.md, collapsed in the current remap; no play dataset named (not in repo) | L |
| Foundation: defensive coverage recognition (switch, drop, hedge, blitz, zone) | POS, MISS, TEAM, ID, C1, BALL | no | matchups.json defenders and men{}, trajectories | correct primary defender (C1 proxy 0.481, n = 160), screen detection, ball | 1,183 matchup records with per-defender `men{}` weights | action recognition, C1, TEAM | none (not in repo) | L |
| Player: shot quality (xFG) | POS, MISS, C1, BALL; also on the CLAIMS kill list ("shot-quality models") | no, and excluded by CLAIMS | outcomes terminating_event (desc carries shot type and distance text, e.g. "makes 3-pt jump shot from 27 ft"), trajectories | shooter identity and release location from video, defender distance at release, shot event detection ("missing", ARCHITECTURE_DIFF.md section 4) | 991 aligned outcome rows with terminating events; SportVU moments with ball and `shot` clock | shot mapping (ARCHITECTURE_DIFF.md section 5C: detector retrain keeping action classes), C1 | basketball-reference play-by-play; Roboflow player-detection-3 action classes | L |
| Player: individual matchup assessment | POS, TEAM, ID, C1, COURT-GATE | no (C1 OPEN; TEAM not met) | matchups.json (1,183 records), tier2_join.json defenders (641 rows) | a measured primary-defender accuracy (C1 asks n >= 150); player names (C2) | `data/matchup_eval/sample.json` frozen at 200 items (LABEL_PLAN.md 3.1); 38 labels discarded and 33 untrusted (`portfolio/numbers.json` matchup_labeling, read from untracked `data/matchup_eval/*.json`) | LABEL_PLAN item 5 (250 verdicts, about 1.5 h once the on-video overlay is fixed); C2 for names | none beyond SportVU (the nearest-opponent proxy) | M for the label set; L for named players |
| Player: gravity / spacing | POS, BIAS, MISS, BALL (gravity) | spacing: POS met on phx only, MISS not met; gravity: BALL killed | matchups.json `spacing_conceded_ft2` {median, p25, p75} already per possession | far-third bias +1.27 ft and 25.1% missed players distort any hull; ball handler for gravity | 1,183 spacing records | detection (R3) for the hull; ball for gravity | none | S to report existing spacing with caveats; L for gravity |
| Player: ghosting | POS, MISS, TEAM, ID, BALL | no | trajectories | full 10-on-10 tracks with identity, ball | SportVU 8 games would train a ghosting model, but that is SportVU, not pipeline output | EPV-level inputs | SportVU logs | L |
| Player: decision quality | POS, MISS, ID, BALL | no | outcomes | passes, shots, ball handler, EPV | none from video | EPV, action recognition | none | L |
| Player: rebound probability | POS, MISS, BALL | no | trajectories, outcomes | ball, shot location, all players | SportVU moments (ball z present) | ball tracking, shot mapping | SportVU logs | L |
| Player: player style embeddings | ID, C2, C3 | no (C2 OPEN: OCR read rate 30.8% at 720p, 31.0% at 1080p, CLAIMS A4; 4,694 jersey numbers withdrawn with the judge-adjudicated band, PROJECT_SUMMARY.md "Delivered dataset" and DEVLOG 2026-09-12; CLAIMS A5 names the attributes without the count) | trajectories, identity.json | player-level identity across possessions; per-player n (C3: GSW only, n about 20 to 50) | rosters `data/rosters/{GSW_2015, GSW_2016, SAC_2015, OKC_2016}.json`; `data/external/basketball-jersey-numbers-ocr.v7i.jsonl.zip` (3,615 crops per ARCHITECTURE_DIFF.md) | R4.2 identity, C2, C3 | Roboflow basketball-jersey-numbers-ocr v7 | L |
| Team: lineup synergy | TEAM, ID, C2 | no | tier2_join.json, credit rows | identity of all 10 players; sample size (641 joined, GSW 330) | 16 play-by-play games (lineup content of the events not verified in repo) | C2, R5.2 re-harvest | basketball-reference play-by-play | L |
| Team: optimal play calling | everything action recognition and EPV need | no | none today | play labels, EPV | none | action recognition, EPV | none | L |
| Game: live win probability | none of the Stage 1 rows (vision-free) | n/a; blocked by scope: ROADMAP.md "Extend nothing in the stats layer until Phase 5"; CLAIMS.md "anything that serves no claim is out of scope" | outcomes (period, clock, points), clock reader (55 of 56) | a score-and-clock model needs far more than 16 games | 16 play-by-play games, 1,239 outcome rows | none in Stage 1; out of ROADMAP scope until Phase 5 | basketball-reference play-by-play | M |
| Game: load / fatigue | POS, MISS, ID, COV | no | trajectories (speeds), identity | per-player distance over a game: identity (fragments), full-game coverage (windows only), reliable speeds (FIX_LOG pass 4: cleaned speed > 30 ft/s on 16% of frames) | none | R4.2, R5.2 | none | L |
| Game: betting market modeling | none of the Stage 1 rows | n/a; no market data in repo (not in repo); CLAIMS kill list forbids league-wide conclusions | none | market data, many games | none | none | none (not in repo) | L |
| Named in repo: team-defense credit (Stage 2, CLAIMS Tier B) | POS, TEAM, C1 for player level; R5.2 for refresh | B1 to B4 SHIPPED on V3 output; refresh gated by R5.2 | tier2_credit.json | re-harvest under the new foundation; canary and bias audit re-run | 641 joined, GSW -7.4 [-21.6, 6.3] | Phases 2 to 4, R5.2 | basketball-reference play-by-play | M (R5.2 is a Colab re-run of 14 games) |
| Named in repo: jersey OCR identity (C2, R4.2) | ID | OPEN | identity.json fragments | OCR on windows (no crops saved), VLM reader, ReID | 8 SportVU games give player ids as labels (R1.3 boxes carry pid and jersey) | R3 detection, COURT-GATE | Roboflow jersey-number set; "vision-language models for jerseys" (ROADMAP.md north star) | L |
| Named in repo: player-level credit tables (C3) | C1, C2, Tier B | OPEN | tier2_join defenders | C1 and C2 | 641 rows, fragment defenders | C1, C2 | none | M after C1 and C2 |
| Named in repo: Hawk-Eye side-by-side demo (R5.1) | COURT-GATE | no (gate unticked) | build sidecars, truth files, scorecards for phx | an accepted court stage | 25 phx windows with truth and SportVU moments | R2.4 to R2.7 decision | SportVU logs | S once the gate is ticked |
| Named in repo: re-harvest 14 games (R5.2) | all Stage 1 rows via the phase gates | no | harvest tooling, games.json | adopted detection, teams, identity stages | 14 production games registered | Phases 2 to 4 | YouTube broadcasts (CLAIMS caveat 4) | M |
| Named in repo: coursework methods (foundation-model team embedding R4.1, VLM jersey reader R4.2, synthetic occlusion data) | TEAM, ID, MISS | no | crops from trajectories | the R4 and R3 items | R1.3 boxes with team and pid (13,975 label rows) | COURT-GATE, R3 | Roboflow sets; SportVU | M each |
| Named in repo: scheme quality models (Stage 3) | all | no | none | everything above | none | Stage 2 | none | L |
| Named in repo: shot mapping, trajectory correction, top-down court view (ARCHITECTURE_DIFF.md) | POS, MISS, BALL for made/miss | no | trajectories `cleaned` (jump handling already exists per identity.json physics), sidecars | shot-event classes (collapsed in the detector remap) | detector external set `data/external/player_det3*` | R3.3 retrain decision on classes | Roboflow basketball-player-detection-3 (654 images per PROJECT_SUMMARY.md) | M |
| Kill list (CLAIMS.md): transition defense, help-position metrics, closeout headline, league-wide or era-normalised conclusions | n/a | excluded by decision | n/a | n/a | n/a | n/a | n/a | n/a |

### 4.1 Build order (dependency list; nothing below item 1 is recommended now because the court [HUMAN] gate is unmet)

1. Court stage: Lucien's R2.4 decision (section 5), then R2.5 acceptance and LOST (fixes cle_gsw coverage reporting), R2.6 smoothing, R2.7 generalisation report with new-arena games (0 usable found, 2 leads). Gate: POS and BIAS on the held-out arena. Blocks everything else.
2. Detection and tracking (R3.1 to R3.5): MISS 25.1%, GHOST 15.1%, ID. Blocks spacing, matchups, any per-frame model.
3. Teams (R4.1) then identity (R4.2, C2): TEAM 81.8%, ID. Blocks named players, style embeddings, lineups, load.
4. R5.1 demo (S) and R5.2 re-harvest (M), then R5.3 updates the public numbers. Refreshes Tier B credit.
5. C1 matchup label set (LABEL_PLAN item 5, M): can be labelled on V3 output once the on-video overlay is fixed, but the number it produces describes V3 and must be re-measured after step 2.
6. Spacing report on existing `spacing_conceded_ft2` (S) after step 2; gravity needs the ball.
7. Ball-dependent extensions (EPV, xFG, rebound probability, decision quality, ghosting, gravity): need a [HUMAN] reversal of the CLAIMS kill list and a ball-tracking stage with its own scorecard row; no plan item exists.
8. Action recognition, then defensive coverage recognition, then play calling: need steps 1 to 3, item 7, and play labels that are not in the repo.
9. Vision-free models (win probability, betting): not gated by Stage 1 rows; blocked by ROADMAP scope (stats layer frozen until Phase 5) and by data (16 games; no market data in repo).

## 5. Decisions waiting on Lucien

Finding: one gating decision (R2.4 adoption) and three non-gating ones are open; the court-stage
acceptance gate cannot be ticked on the current numbers because CLE misses position and far-bias
targets.

| Decision | Measured facts behind it (path, date) | What the plan says |
|---|---|---|
| 1. Adopt the R2.4 camera model (r23_cam) or not | phx: p50 1.92 to 1.84, p90 4.98 to 4.95, near -0.67 to -0.27, far +1.52 to +1.27 ft; paired r23 minus camera p50 +0.08 [+0.03, +0.10], abs_near +0.40 [+0.32, +0.45], abs_far +0.25 [+0.17, +0.33]. cle_nyk: p50 2.25 to 2.17, p90 5.46 to 5.49 (+/-0 within [-0.18, +0.21]), abs_far +0.36 [+0.10, +0.48]. cle_gsw: p50 2.22 to 2.26 (-0.04 [-0.19, +0.11]), near 0.12 to 0.26 abs (-0.14 [-0.39, +0.66]); every cle_gsw interval crosses zero. CLE pooled 2.24 / 5.47 to 2.19 / 5.49, abs_far +0.31 [+0.06, +0.45]. Training games bkn, sac: camera position within 0.3 to 0.8 ft of the truth camera; foot error unchanged. (`reports/camera_track.txt`, `reports/scorecard/compare__r23_replay__r23_cam.txt`, `reports/scorecard/*__r23_cam.txt`, all 2026-10-06) | ROADMAP_LOG pass 15: under the amended rule nothing is worse, but cle_gsw's p50 and near-bias point estimates do not improve, so "improves every row" holds only if the held-out arena is judged pooled. Decision for Lucien. ROADMAP.md R2.4 stays unticked |
| 2. Court-stage [HUMAN] acceptance | Court rows on the held-out arena with the best build: cle_nyk 2.17 / 5.49 ft, cle_gsw 2.26 / 5.45 ft (targets 2.0 / 5.0); far bias +1.30 and +1.89 ft (target +-0.5); cle_gsw coverage 39.7% (floor 50%, failed by V3 too). phx meets position (1.84 / 4.95) and near bias (-0.27) but not far bias (+1.27). (`reports/scorecard/cle_*__r23_cam.txt`, `gsw_phx_2016__r23_cam.txt`, 2026-10-06) | ROADMAP.md: "Lucien accepts the court stage when the court rows of the targets table are met (position error and near-field bias)". Not met on the arena, so R2.5 to R2.7 come first; R2.7 amended 2026-10-06 to add new-arena games while the arena misses the absolute targets |
| 3. New-arena games for R2.7 (which leads to check by eye, whether to download) | GSW road games 2015-16: usable 0; leads 2015-11-12 at MIN `JVQhDSZXyKI` (short_broadcast, 4,172 s, 720p avc1, uploaded the game day) and 2015-12-30 at DAL `ZkwH9zTcLhw` (ambiguous, 5,517 s, 720p, uploaded 2020) (`reports/broadcast_search_away.txt`, 2026-10-06). Home manifest: GSW usable 4 of 20 plus held-out phx, CLE 2 of 17, OKC 0 of 25, NYK 0 of 22; short GSW uploads 11.06 DEN 3,970 s, 11.24 LAL 4,317 s, 12.18 MIL 3,991 s (`reports/sportvu_manifest.txt`, 2026-10-04). Scaling study: new labelled data should come from new games in new arenas of the same era; any new-arena comparison needs >= 3 seeds (`reports/scaling/scaling_first_pass.txt`, ROADMAP_LOG 2026-10-06) | ROADMAP.md R2.7: add 2015-16 games from other non-Oracle arenas, never the held-out arena, repeat R2.3 with >= 3 seeds. `sportvu/manifest_overrides.json` is the place for human-confirmed ids |
| 4. Gate v2 retrain (non-gating) | gate v2's training set holds 882 frames of held-out games: gsw_phx_2016 294, gsw_nyk_curry54 299, gsw_cle_xmas16 289 (`reports/sportvu_splits.txt`, 2026-10-04). It picks the wide frames the scorecard scores; builds are compared on the same cached frames, so adoption verdicts are unaffected (CLAIMS.md caveat). `gate_ship_v2.py` now refuses to run | ROADMAP.md Waiting on Lucien: retrain without them (gate v3 by the gate adoption rule, then re-cache the bench's wide-frame scores) or keep v2 with the caveat |
| 5. FIX_PLAN.md Phase B [HUMAN] "confirm the Phase C order" | unticked; FIX_PLAN.md header says Phases C and D are superseded by ROADMAP.md (2026-10-03) | Housekeeping only: tick or strike it so the loop has one open gate list |

## 6. Public-claim check

Finding: README.md carries the SportVU result with its caveats; PROJECT_SUMMARY.md, PIPELINE.md and
the portfolio page carry only the PROTOTYPE stage numbers and the HARVEST canary, with no held-out
tracking number, and two counts are stale. No published figure contradicts its own artifact; the
gaps are omissions and staleness.

| Where stated | Stated value | Current value (artifact, date) | Path of the statement |
|---|---|---|---|
| PROJECT_SUMMARY.md tech stack and practices | "pytest (52 tests)", "52 tests" | 87 collected (`pytest --collect-only -q tests`, 2026-10-19) | PROJECT_SUMMARY.md lines 21, 241 |
| PROJECT_SUMMARY.md Stage 3 result, Quantified highlights, Resume bullets | "100% homography success ... median 0.30 ft (~3.6 in) reprojection error (p50 1.7 px over 279 validation frames)" presented as the court result; no held-out tracking number anywhere in the file | PROTOTYPE 0.302 ft on 28 val frames stands (`reports/homography_metrics.txt`, mtime 2026-06-24); HELD-OUT position error of the same V3 models p50 7.33 / p90 15.35 ft (`reports/scorecard/gsw_phx_2016__v3.txt`, 2026-10-04); ROADMAP.md (2026-10-03): "Court mapping is the dominant error and is not solved" | PROJECT_SUMMARY.md sections "Stage 3", "Quantified highlights", "Geometry" bullet |
| PROJECT_SUMMARY.md Stage 2 | detector table row "basketball-trained": precision 0.89, recall 0.87, F1 0.88 as the detection result | PROTOTYPE 0.887 / 0.865 stands (`reports/tracking_metrics.txt`); HELD-OUT missed players 25.1%, ghost boxes 15.1% (`gsw_phx_2016__v3.txt`, 2026-10-04) not mentioned | PROJECT_SUMMARY.md "Stage 2" |
| PROJECT_SUMMARY.md Stage 5 | teams "87.1% track-level / 83.9% crop-level" | PROTOTYPE stands (one clip, 31 tracks); HELD-OUT team labels 81.8% phx, 74.7% cle_nyk, 78.4% cle_gsw (scorecards, 2026-10-04) not mentioned | PROJECT_SUMMARY.md "Stage 5" |
| PROJECT_SUMMARY.md Quantified highlights | "Models trained: 7+ (..., gate-v2 pending)" | gate v2 shipped 2026-10-02 (CLAIMS.md standing caveat 1); r23 grid model adopted 2026-10-04 (`models/court_grid_r23.pt`, CLAIMS.md) | PROJECT_SUMMARY.md "Quantified highlights" |
| PROJECT_SUMMARY.md Stage 7 | "38 labels discarded, a second attempt reaching 33" | matches `portfolio/numbers.json` matchup_labeling (38 discarded; 33 with 23 negative, 6 positive, 4 unsure) read from untracked `data/matchup_eval/*.json`; CLAIMS C1 OPEN | consistent; noted because the files are untracked |
| PIPELINE.md header and rows 1 to 5 | "as of 2026-07-08"; row 3 "H-err p50 1.7 px, 279 val frames; unseen games 66-89% sane-H"; row 5 "87.1% track" | PROTOTYPE numbers stand; HELD-OUT V3 7.33 ft and 81.8% absent; PIPELINE.md is scheduled for rewrite at R5.3 | PIPELINE.md table |
| README.md Status (2026-10-03) | "position error p50 7.3 ft, p90 15.4 ft ... 25.1% ... 15.3% of boxes are ghosts (referees included); team labels 81.8%" citing `reports/sportvu_phx.txt` and A8 | scorecard supersedes by 0.03 ft: 7.33 / 15.35, ghosts 15.1% non-referee (15.3% with referees) (`gsw_phx_2016__v3.txt`, 2026-10-04); no mention of r23 (1.92 / 4.98, adopted 2026-10-04) or r23_cam (1.84 / 4.95, 2026-10-06) | README.md "Status (2026-10-03)" |
| README.md Status | V3 stage numbers "were measured on the 7 prototype clips the models were tuned on" | agrees with ROADMAP.md; this is the only public document that says so | README.md "Status" |
| README.md gate section | "the label factory uses 0.35" | consistent with the 2026-10-02 provenance correction (CLAIMS.md caveat 1: 0.35 belongs to `label_factory.py`, the harvest ran 0.70) | README.md "Component: court-visibility gate" |
| portfolio/template.html, portfolio/numbers.json (generated 2026-10-03T01:12 UTC), built page `portfolio/build/court-ledger.html` (2026-10-02) | gate 98.7% (158), detection F1 0.88, homography 0.30 ft on 28 frames and "p50 1.7 px on 279 validation frames" as a separate benchmark, teams 87.1%, 641 joined / 16 games, 87.8% (n = 990), +8.8 per 100, GSW -7.4 [-21.6, +6.3], 95.0% [88.8, 97.8] | all match their artifacts (`reports/metrics.json`, `tracking_metrics.json`, `homography_metrics.json`, `teams_eval_curry_q1_clip.json`, `tier2_crossval_corpus.json`, `tier2_bias_audit.json`, `data/pbp/tier2_credit.json`, `label_audit.json`); the built page contains 0 occurrences of "SportVU": the held-out tracking result (first measured 2026-10-03) is absent from the recruiter-facing page | `portfolio/numbers.json`, `portfolio/build/court-ledger.html` |
| portfolio/numbers.json hand_typed_sources | "PROJECT_SUMMARY.md and paper ... say 2013-2017, wrong by a season"; "pytest ... 52 collected (2026-09-12)" | PROJECT_SUMMARY.md says "2012-2017"; tests 87 | `portfolio/numbers.json` hand_typed_sources |
| CLAIMS.md A8 | "p50 7.3 ft, p90 15.4 ft ... team label accuracy 81.8% (n=11,240)" | scorecard 7.33 / 15.35, n = 10,829 (CLAIMS.md explains the re-matching; A8 keeps B5 numbers until R5.3) | CLAIMS.md Tier A row A8 |
| CLAIMS.md A1, A5, A7, standing caveat 1 | 87.8% (869/990); 95.0% [88.8, 97.8] (n=100), 6.7 to 20.0% (n=200); 68.8% [57.9, 77.8]; precision 0.988, recall 0.859, 500 FN, 37 FP, 0.42, 0.975 at 0.55 | match `reports/tier2_crossval_corpus.txt`, `label_audit.txt`, `shot_type_audit.txt`, `gate_sheet.txt` | consistent |
| paper/ | referenced by PROJECT_SUMMARY.md ("Report in progress: paper/") and the portfolio ledger (paper sections 4 to 6, 9, 10) | not in repo (private repository NBA-Comp-Viz-V3-paper per ROADMAP_LOG.md "Publication (2026-10-03)") | PROJECT_SUMMARY.md line 15 |

## 7. Verification

Finding: after the recheck, every cited number matches its file; the pass found and fixed seven
errors in the first written version of this document.

Method (2026-10-19): every file cited above was re-opened; 269 cited tokens across 48 files were
string-matched against their files (`grep -F` equivalent), and every count in section 3 was recomputed
from the files. Four tokens failed only because the test string spanned a line break or named the
wrong file; the fixes below are the errors that were in the document itself.

| Where | First version said | Corrected to | Evidence |
|---|---|---|---|
| Section 3, scorecard row | 15 scorecards | 13 scorecards (3 games x 4 builds plus `gsw_phx_2016__public_conf005`) and 3 compare files | `ls reports/scorecard/*.json` |
| Section 3, `tier2_join.json` row | 40 source clips | 151 source clips | `len(sources)` in the file; the earlier key dump was truncated |
| Section 3, labels row | 13,975 rows in 333 window files (84 / 73 / 93 / 83) | 329 `.jsonl` window files; the earlier per-game counts included non-label entries in the directories | `glob data/sportvu/labels/<game>/*.jsonl` |
| Section 3, possessions row | 1,239 metrics_eligible | `metrics_eligible` exists on halfcourt spans only; true on 1,239 of 2,034 | key missing on all 4,342 transition spans |
| Section 3, `games.json` row | 6 Stage 1 games (4 train, 2 heldout_arena), 1 excluded | 7 Stage 1 entries: 5 tagged `split: train` (one of them, `gsw_cle_xmas15`, also `excluded`), 2 `heldout_arena` | `Counter(split)` over the file |
| Section 4, style embeddings row | "4,694 jersey reads withdrawn by the audit, CLAIMS A5" | the 4,694 figure is in PROJECT_SUMMARY.md line 188 and DEVLOG 2026-09-12; CLAIMS A5 names the attributes without the count | `grep 4,694 CLAIMS.md` finds nothing |
| Section 6, PROJECT_SUMMARY Stage 2 row | quoted a "P 0.89 / R 0.87" string | PROJECT_SUMMARY.md states it as a table row (0.89 / 0.87 / 0.88); the "P .89 / R .87" string is in README.md and PIPELINE.md | `grep` of PROJECT_SUMMARY.md |

Dating notes recorded, not corrections: `reports/metrics.txt`, `homography_metrics.txt` and
`tracking_metrics.txt` were never committed (dates are mtimes); `reports/tier2_bias_audit.txt` was last
committed 2026-07-16 and regenerated 2026-09-11 with the same values; `data/pbp/tier2_credit.json`
was last committed 2026-07-16 and its mtime is 2026-09-11.
