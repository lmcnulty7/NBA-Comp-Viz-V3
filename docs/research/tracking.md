# Dossier: broadcast tracking pipeline

Lane: broadcast tracking (the measured stand-in for vendor tracking, scored against SportVU). Not a row of
`reports/results.md`. Date: 2026-11-16. Branch: v4-research. Written under PLAN.md D70 to D74.

**Verdict.** For the players it finds, build r23 is as close to SportVU as published single-camera systems get to
their references. It also misses about a quarter of the players in view, keeps no identity, gets about one team
label in five wrong and has no ball. Those failures, not position error, are what keep it out of the analytics.

## 1. What we built

- Method: CLIP-head frame gate, then a YOLOv8 detector fine-tuned on 654 Roboflow images with BoT-SORT tracking
  (appearance re-ID from the detector's own features), a grid-keypoint court model with RANSAC homography, line
  snap and tracking across frames (r23: fine-tuned on SportVU-derived labels of 4 Oracle Arena training games),
  an offline fragment linker, and k-means (k = 2) per clip on the median Lab colour of each track's torso
  (README.md, "The broadcast pipeline, stage by stage"; `detect/teams.py`; `botsort_reid.yaml`).
- Data: 25 SportVU-synced windows of the held-out game gsw_phx_2016 (9,500 frames, 1,447 testable) and two
  held-out-arena games (cle_nyk_2015, 485 testable; cle_gsw_2016, 158 testable) (`reports/scorecard/*__r23.txt`).
- Numbers, r23 on gsw_phx_2016 (`reports/scorecard/gsw_phx_2016__r23.txt`): position p50 1.92 ft, p90 4.98 ft
  (met on the point estimate only; p90 interval [4.42, 6.02], `reports/r23_verification.txt`); near / far bias
  -0.67 / +1.52 ft (not met); coverage 99.3%; missed players 25.1%; ghost boxes 15.1%; team labels 81.9%;
  21.39 id switches per possession; jersey read rate not measurable.
- Arena games: position 2.25 / 5.46 ft and 2.22 / 5.5 ft (not met), team labels 73.7% and 78.2%
  (`reports/scorecard/cle_nyk_2015__r23.txt`, `reports/scorecard/cle_gsw_2016__r23.txt`).
- Demo (illustration, D16): 48.7 percent of player-frames had to be filled from sensor data because the tracker
  had no matched box there (PLAN.md morning summary; the fill share is written to the site's `demo/swap.json`).

## 2. Foundations

| Paper | Link | What we take, and how ours differs |
| --- | --- | --- |
| Jocher, G., Qiu, J., Chaurasia, A. Ultralytics YOLO, version 8.0.0 (software), 2023. AGPL-3.0. | https://github.com/ultralytics/ultralytics (CITATION.cff) | Our detector is a YOLOv8 model from this package (the Colab training script starts from yolov8m). Ours is fine-tuned on 654 Roboflow basketball images only. |
| Aharon, N., Orfaig, R., Bobrovsky, B. BoT-SORT: Robust Associations Multi-Pedestrian Tracking. arXiv preprint, 2022. | arXiv:2206.14651 | Our tracker is BoT-SORT as shipped in Ultralytics, with camera-motion compensation and re-ID. Ours takes the re-ID embedding from the detector's head instead of a separate re-ID network. |
| Zhang, Y., Sun, P., Jiang, Y., et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. arXiv preprint, 2021. | arXiv:2110.06864 | BoT-SORT builds on its two-stage matching of high- and low-score boxes, which our tracker config uses (`track_high_thresh`, `track_low_thresh`). |
| Nie, X., Chen, S., Hamid, R. A Robust and Efficient Framework for Sports-Field Registration. WACV 2021, pp. 1936-1944. | https://mlanthology.org/wacv/2021/nie2021wacv-robust/ | The closest published design to our court model: a uniform grid of keypoints over the whole court, then a homography. Ours adds line snap and frame-to-frame tracking, and r23 was fine-tuned on labels derived from SportVU, not hand labels. |
| Lu, W.-L., Ting, J.-A., Little, J. J., Murphy, K. P. Learning to Track and Identify Players from Broadcast Sports Videos. IEEE TPAMI 35(7), 2013, pp. 1704-1716. | doi:10.1109/TPAMI.2012.242 | The basketball broadcast ancestor of our whole pipeline: detect, track, court homography, team, then identity, on NBA broadcast (2010 Finals). Ours has no identity stage; theirs used weak labels from play-by-play text. |
| Koshkina, M., Pidaparthy, H., Elder, J. H. Contrastive Learning for Sports Video: Unsupervised Player Classification. arXiv preprint, 2021. | arXiv:2104.10068 | Unsupervised team split without known jersey colours, as ours. Ours clusters a hand-made colour feature; theirs learns an embedding (hockey). |
| Cui, Y., Zeng, C., Zhao, X., Yang, Y., Wu, G., Wang, L. SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes. ICCV 2023, pp. 9921-9931. | doi:10.1109/ICCV51070.2023.00910; arXiv:2304.05170 | The standard sports MOT benchmark, with a basketball split. We score against SportVU on the court instead of boxes in the image, and our footage keeps its camera cuts. |
| Somers, V., Joos, V., Cioppa, A., et al. SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap. CVPR Workshops 2024. | arXiv:2404.11335 | The closest task definition to ours: positions on a court map, with team and jersey, scored by GS-HOTA (5 m tolerance). Ours is basketball, scored row by row, not in one number. |
| Van Zandycke, G., Somers, V., Istasse, M., Del Don, C., Zambrano, D. DeepSportradar-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations. MMSports 2022 (ACM MM workshop). | arXiv:2208.08190 | Basketball tasks we lack (ball 3D localisation, calibration, re-ID). Its footage is a fixed automated camera in European arenas, not NBA broadcast. |

## 3. Benchmark

No paper reports a number on our metric, data and split. Every row below is "not directly comparable" at best;
the rows are here so a reader can place our numbers, not rank them. Metres converted from feet by us (1 ft =
0.3048 m).

| Our number | Published number | Paper | What differs | Comparability |
| --- | --- | --- | --- | --- |
| Position p50 1.92 ft (0.59 m), p90 4.98 ft (1.52 m), matched players, gsw_phx_2016 | Median error 1.55 to 1.84 m on broadcast TV (TV12, TV14), 1.13 to 1.20 m on a tactical camera (TC14) | Theiner et al., WACV 2022, Table 3 | Soccer, about 105 x 68 m pitch; per-frame mean of the best 80% of Hungarian matches, then median over frames; their reference carries about 1 m error; ours is a median over player observations on testable frames only (15% of window frames) | Not directly comparable: different sport, field scale, aggregation and reference |
| Same, plus 48.7% of demo player-frames had no matched box | Position RMSE 0.44 to 1.14 m when the player is detected (providers 1 and 2); 4.62 to 12.23 m when not detected; 1.68 to 16.39 m overall | Crang et al., J. Sports Analytics 2026, Results and Discussion | One 2022 World Cup match, commercial systems, TRACAB Gen 5 reference, RMSE not median, undetected players filled by the vendors' models | Not directly comparable (metric, sport, systems). The same pattern holds: good when detected, poor when filled |
| 21.39 id switches per possession (lower bound, sparse samples); no HOTA computed | Basketball HOTA 60.8, IDF1 67.8, AssA 46.8 (MixSort-Byte) | Cui et al., ICCV 2023, Table 7 | Image-plane boxes on NCAA clips, 720p, no viewpoint switches inside a clip; ours counts switches against SportVU ids on court, across cuts | Not directly comparable: we do not compute HOTA or IDF1 |
| Team labels 81.9% (upper bound, best A/B mapping per window) | 94% after one frame, 97% within 500 frames | Koshkina et al., arXiv 2021, abstract | Hockey, their own dataset and labels | Not directly comparable |
| Team labels 81.9% | Micro accuracy 0.93 for the two team classes | Theiner et al., WACV 2022, Section 4.3 | Soccer, annotated boxes, not matched to tracking | Not directly comparable |
| No ball | F1 82.6, AP 77.1 at a 4 px tolerance (WASB, step 1), Basketball split | Tarashima et al., BMVC 2023, Table 2 | Basketball broadcast clips adapted from a group-activity dataset (their reference 91), 2D ball in the image | Nothing of ours to compare |
| Jersey read rate not measurable on the windows; 30.8% read rate at 720p in an earlier controlled probe (CLAIMS.md row A4, prototype era) | 87.4% tracklet accuracy on SoccerNet; 91.4% image accuracy on hockey | Koshkina and Elder, CVPRW 2024, abstract | Read rate is not accuracy; soccer and hockey | Not directly comparable |
| Same | 91.0 to 95.0% for open VLMs on four-way multiple-choice jersey questions (Q9, 300 items) | Hu et al., arXiv 2026, Table 2 | Multiple choice from 4 options, 2025-26 NBA clips, 8 crops per player; not open-set reading | Not directly comparable |
| Broadcast-only fill: no number (the demo fills from sensor beyond 0.5 s, D50) | Imputation error of players outside a simulated camera view (theta 10 / 20 / 30 degrees): linear fit 12.78 / 11.47 / 11.26 ft, GC-VRNN 5.86 / 5.56 / 5.39 ft | Xu et al., CVPR 2023, Table 1 | 2012 NBA SportVU, simulated camera view that follows the ball | Nothing of ours yet; tracking-E2 would produce it |
| Same | TranSPORTmer I-ADE 3.60 / 3.26 / 3.08 ft on the same three settings (their own runs, in parentheses in the table) | Capellera et al., ACCV 2024, Table 3 | Same data and masks as above | Nothing of ours yet |

## 4. Measured weaknesses (our reports only)

1. Missed players 25.1%, 24.4%, 26.2% and ghost boxes 15.1%, 14.8%, 14.8% on the three scored games, against a
   5% target (`reports/scorecard/*__r23.txt`). Both counts project through a truth homography valid only near
   the fitted feet, so they carry that error (scorecard caveats).
2. Team labels 81.9%, 73.7%, 78.2% against 95%, and these are upper bounds (best A/B mapping per window).
3. Identity: 21.39, 31.07 and 14.29 id switches per possession (lower bounds); jersey reading did not run on the
   windows.
4. No ball (README.md, top section; CLAIMS.md out-of-scope list).
5. Near-field bias not met on any game: far third +1.52, +1.66, +2.00 ft (r23); r23_cam was not adopted (D42).
6. Position target not met on either arena game (p90 5.46 and 5.5 ft), and met on the held-out game on the
   point estimate only.
7. Coverage 39.7% on cle_gsw_2016, below the 50% floor.
8. In the demo, 48.7 percent of player-frames came from sensor data, not the tracker (PLAN.md summary).
9. Scored frames are only the testable set (15%, 7.6% and 4.7% of window frames), which leans toward frames
   the pipeline already handled.

## 5. Recent work aimed at these weaknesses

- Team labels (weakness 2). The SoccerNet game state baseline (Somers et al. 2024, Section 6.3) averages
  learned re-ID embeddings over each tracklet and splits them with k-means into two teams. PRTreID (Mansourian
  et al., MMSports 2023, arXiv:2401.09942) trains one embedding for re-ID, team and role together. Koshkina et
  al. (2021) learn a team embedding with no labels. Istasse et al. (CVPR Workshops 2019, arXiv:1907.01058) learn
  pixel embeddings for team discrimination on panoramic basketball footage. All replace a colour feature with a
  learned embedding, which is the change tracking-E1 tests.
- Identity and jersey (weakness 3). Koshkina and Elder (CVPRW 2024) localise the number with body pose and read
  it with a scene-text model, then vote over the tracklet. Hu et al. (arXiv:2608.23435, August 2026, not peer
  reviewed) read jerseys with an open VLM over 8 crops per track. PaliGemma (Beyer et al., arXiv:2407.07726) is
  the VLM family whose format our Roboflow jersey set already uses (docs/DATA_SOURCES.md). Lu et al. (2013)
  show that play-by-play text can stand in for most identity labels. Basketball-SORT (Hu, Scott, Yeung, Fujii,
  arXiv:2406.19655) targets occlusions of three or more look-alike basketball players, but on a fixed camera
  (HOTA 63.48% on 5v5, abstract).
- Ball (weakness 4). WASB (Tarashima et al., BMVC 2023, arXiv:2311.05237) is a heatmap detector with temporal
  inference that reports results on a basketball broadcast ball split it annotated (Tables 1 and 2). Kim et al. (KDD 2023, arXiv:2306.08206)
  infer the ball from player trajectories alone, in soccer (position error 3.69 m for their best model, Table 2).
  TranSPORTmer (2024) does the same with set attention. These matter because our broadcast output has no ball
  and the frozen models read one.
- Fill for missed and off-screen players (weaknesses 1 and 8). Omidshafiei et al. (Scientific Reports 2022,
  doi:10.1038/s41598-022-12547-0) impute football players outside a simulated broadcast camera from those in
  view. GC-VRNN (Xu et al., CVPR 2023, arXiv:2303.16005) and TranSPORTmer (Capellera et al., ACCV 2024,
  arXiv:2410.17785) do it on NBA SportVU with a simulated camera, and both beat a linear fit by a wide margin
  (benchmark rows above). Bassek et al. (Science and Medicine in Football 2025, 9(4), 467-477,
  doi:10.1080/24733938.2025.2533808) find that less than 50% of player activity is visible on soccer TV
  broadcast. Crang et al. (2026) find position RMSE of 0.44 to 1.14 m for detected players and 4.62 to 12.23 m for players the software did not detect (providers 1 and 2).
- Detection (weakness 1). SportsMOT's basketball split (720p, boxes on athletes) is in-domain training data for
  the detector, but it is CC BY-NC 4.0 and may not be redistributed (SportsMOT GitHub page), the concern already
  logged in docs/DATA_SOURCES.md.

## 6. Pre-registered experiments

Prerequisites for both (no result yet, so nothing is chosen by looking at one):

- Development games: the 4 Oracle Arena training games with synced video and SportVU (gsw_bkn_2015,
  gsw_cha_2016, gsw_ind_2016, gsw_sac_2015; `reports/sportvu_splits.txt`). r23's court model was fine-tuned on
  labels from these games, so their court rows (position, bias) are in-sample. Both experiments leave the court
  stage untouched, so only paired differences on these games mean anything; absolute dev-game numbers are not
  claims.
- Baseline: build r23 rebuilt with frozen code on the 4 dev games and scored by `sportvu.bench` with an explicit
  `--card-dir` outside `reports/` (CLAUDE.md rule 6), under the heavy-job lock. No such dev-game scorecard
  exists in `reports/scorecard/` today.
- Final score: once, on the D71 test set. For this lane the D71 set must hold broadcast video synced to SportVU
  (or another tracking source) in games not used anywhere so far; whether such games exist is for the data
  survey (D74).
- Paired test: bootstrap by game, 2000 resamples, 95 percent interval. With 4 dev games the game bootstrap has
  only 35 distinct resamples and is coarse, so the scorecard's existing window-cluster bootstrap
  (`sportvu/compare.py --paired`) is reported beside it, and adoption needs both intervals to exclude zero.

### tracking-E1: team labels from tracklet embeddings

- Hypothesis: replacing the median-Lab colour feature with a frozen pretrained image embedding (CLIP ViT-B/32,
  the backbone the frame gate already loads, `gate/backbones.py`; Radford et al., arXiv:2103.00020), averaged
  over each track's crops and split by the same per-clip k-means (k = 2), raises the team-label row on the dev
  games. No labels are used: as in the baseline, it is fitted per clip with no supervision.
- Baseline: r23 with `detect/teams.py` as frozen. Boxes, tracks and court are identical in both arms; only the
  team feature changes.
- Abstention: the baseline abstains by a distance-ratio rule. The new arm sets its threshold per clip so that its
  abstention share equals the baseline's on that clip (rank-based, label-free), so accuracy is compared at equal
  coverage.
- Metric: the scorecard's team-label accuracy (vs the SportVU team of the matched player), plus the same with one
  fixed A/B mapping per clip (closer to how labels are used), plus the abstention share. It fits because it is a
  failing row, and team gates the linker's merges and the offense/defense split.
- Adoption threshold: pooled dev-game team accuracy up by 5.0 points or more, both intervals above zero, no dev
  game down by more than 1 point, fixed-mapping accuracy not lower. Then one D71 score, reported whatever it is.
  A pass still leaves the row under its 95% target and is reported that way.
- Ablation: DINOv2 small (Oquab et al., arXiv:2304.07193; also already in `gate/backbones.py`) in place of CLIP,
  same rule. If both or neither beat colour, the gain is the embedding, not the model.
- Compute (estimate, not measured): one replay of the team stage per game on stored tracks, 8 crops per track,
  CLIP ViT-B/32 on CPU. About 1 to 2 days of work (a replay tool under `tools/`, like `sportvu/court_replay.py`
  for the court) and about 4 to 8 hours on the 16 GB Mac under the lock. No GPU needed.
- Risks: the embedding may encode pose and background more than the jersey; all 4 dev games are Warriors home
  games, so the home kit is the same in every one and the gain may not carry to other arenas (only D71 tells);
  referees in grey may fall into a team cluster; the row is an upper bound in both arms.
- What a coach gains: offense and defense assigned to the right players more often, which every matchup and
  spacing figure needs, and fewer cross-team merges in the linker.
- Backlog: expected gain 3 (one failing row, and it feeds identity and possessions, but even +10 points stays
  under 95%). Effort 2 (no training, existing backbone, CPU only).

#### tracking-E1 under D153 (written 2026-11-22, with the replay tool, before any scored run)

The experiment above runs as P8 Track B's E3 on the Mac (PLAN.md D153). Arms, abstention matching, metrics and the
adoption threshold are as written above. Two substitutions, fixed before any run:

1. Development games: the 3 R1.3 games with Mac camera truth (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015), pooled with
   the game bootstrap. gsw_ind_2016 is reported apart (D98). hou_cle_2016 is reported apart as the new-arena check and
   does not enter the adoption rule.
2. "One D71 score" becomes D129: the D85 broadcast set, scored once at the end of P8. It is not scored now.

How it runs (tools/p8_e3.py, committed with this text):

- Replay on stored r23 tracks: the frozen TorsoCropCollector (every 3rd processed frame, at most 40 crops per
  fragment) is run on r23's own `<window>_frames.json` boxes and fragment ids, frame by frame from the window clip.
  Boxes, fragments, links (idmap) and court are r23's. The frozen colour classifier is run on the same crops as a
  reproduction check against r23's stored labels (agreement reported); the baseline arm is r23's stored
  team_by_track itself.
- Embedding arms: CLIP ViT-B/32 (openai/clip-vit-base-patch32) and the ablation DINOv2 small (facebook/dinov2-small;
  gate/backbones.py configures dinov2-base, so the small model is loaded directly through the same transformers
  classes, its pooled CLS output L2-normed). 8 crops per fragment, evenly spaced over its collected crops; a
  fragment's feature is the L2-normed mean of its crop embeddings; per window, k-means (k = 2, n_init 10, the project
  seed) on the fragments with at least TEAM_MIN_CROPS crops, as detect/teams.py fits on fragments; each canonical track
  gets the mean over its member fragments' crops and the nearest centroid. Same canonical track set as r23
  (the keys of its team_by_track).
- Abstention: per window, the arm abstains on exactly as many canonical tracks as r23 did (tracks with no feature
  first, then the largest near/far distance ratio), never fewer than its tracks with no feature.
- Metric: the pinned scorer (tools/p8_scorecard.py score_window, truth camera, D88) per window and arm; team accuracy
  with the frozen majority mapping per window; fixed-mapping accuracy with the best one-to-one A/B mapping per window;
  abstention share at track level.
- Intervals: paired difference of pooled accuracy (arm minus r23), bootstrap by game (3 games, 2000 resamples, seed 0;
  only 10 distinct resamples) and by window (2000, seed 0). Adoption: pooled gain at least 5.0 points, both intervals
  above zero, no pooled game down more than 1 point, pooled fixed-mapping accuracy not lower than r23's.
- Guards: refuse_cv_test and tools.p8_frozen.game_of on every tag before any frame is read; each game's replay and
  score run under the heavy-job lock, one game's crops in memory at a time (one window at a time). No E-BARD colour
  labels or any other labels are used.

### tracking-E2: broadcast-only fill of missed player-frames by learned imputation

- Hypothesis: for player-frames with no matched box (missed in frame or off screen), a set-attention imputation
  model in the TranSPORTmer design, conditioned only on the matched broadcast positions of the other players
  and of the same player before and after the gap, gives lower position error than linear interpolation across
  the gap. This is what the demo's 48.7% sensor fill would have to become for a broadcast-only pipeline.
- Baseline: r23 on the dev games with the D50 rule (linear between samples at most 0.5 s apart) extended to
  every gap by linear interpolation, holding the last or first position at the ends of a window. No fit.
- Model data: trained on SportVU player trajectories of training games only, excluding the 4 dev games and every
  game under a heldout/ folder. Masks are drawn before training from two rules fixed now: a simulated camera
  view that follows the ball (as in Xu et al. 2023 and Omidshafiei et al. 2022) and random in-frame drop-out at
  the measured 25% missed-player rate. The model never reads the SportVU ball at test time, because broadcast
  has none; a test must hold that masked players' SportVU positions are never read at prediction time.
- Metric: median and p90 error (ft) against SportVU of filled player-frames on the dev games, by gap length
  (0.5 to 2 s, 2 to 5 s, over 5 s), and the share of player-frames still needing sensor fill. Primary: median
  error over all gaps over 0.5 s. It fits because it scores exactly the frames the demo now fills from sensor.
- Identity caveat: imputing "player k" across a gap needs to know who k is, which the pipeline cannot do today
  (weakness 3). E2 therefore uses the SportVU id of each matched box, an oracle identity, and is an upper bound
  on fill quality. It is labelled that way and is not a claim about the full pipeline.
- Adoption threshold: primary median error at least 25% below the linear baseline, both intervals excluding
  zero, p90 not worse in any gap band. Then one D71 score.
- Ablation: the same model with the SportVU ball given as an input (an oracle). The gap between the two arms
  sizes what a ball tracker would add to the fill, which ranks the ball item in the backlog.
- Compute (estimate, not measured): about 3 to 5 days of work (data loader from the contract frames, masks,
  model, scorer); training a few hours on a Colab T4 or A100 (Xu et al. trained on an A100); scoring on the Mac
  CPU under the lock in under an hour.
- Risks: simulated masks differ from real misses, which cluster in occlusions near the ball; long gaps are
  close to forecasting and errors grow fast; oracle identity flatters the result; the fill is a model output
  and must never be shown as tracking (the site would label filled frames).
- What a coach gains: all ten players on the court for every frame from video alone, so the frozen models can
  run on games with no vendor tracking, with each filled frame marked and its error known.
- Backlog: expected gain 4 (attacks the 48.7% directly and is the step a broadcast-only demo needs; capped by
  the identity caveat). Effort 4 (a new model, GPU training, new scoring code).

### Not proposed now (backlog lines, no run planned)

- Detector fine-tune on in-domain boxes (SportsMOT basketball) for missed players and ghost boxes. Gain 4,
  effort 3. Blocked by the CC BY-NC licence question, and by the caveat that both rows project through a truth
  homography valid only near the feet, so a better truth may be needed before the rows can show a gain.
- Ball detection in the image (WASB-style), trained on dev-game frames with the SportVU ball projected through
  the R1.3 camera truth. Gain 4, effort 5 (3D projection labels, small fast object at 720p, GPU training).
- Jersey reading with a VLM plus tracklet voting and roster constraint, scored as the share of matched
  tracklets whose read number equals the SportVU jersey. Gain 4, effort 3; needs saved crops, which the
  windows do not have today.
- Weak identity labels from play-by-play (Lu et al. 2013). Gain 3, effort 4.
- A GS-HOTA-style single number on the court (Somers et al. 2024) added to the scorecard, so a published
  benchmark could be matched in form. A scorer change, not an experiment; it needs the lead's decision because
  the scorecard is read-only to experiments.

## 7. Sources checked

All fetched 2026-11-16. Numbers were read in the PDF or on the page named.

| Source | Link | Used for |
| --- | --- | --- |
| BoT-SORT, arXiv abstract | https://arxiv.org/abs/2206.14651 | Foundation |
| ByteTrack, arXiv abstract | https://arxiv.org/abs/2110.06864 | Foundation |
| Ultralytics YOLO, GitHub page and CITATION.cff | https://github.com/ultralytics/ultralytics | Foundation |
| Nie et al., WACV 2021, ML Anthology page | https://mlanthology.org/wacv/2021/nie2021wacv-robust/ | Foundation |
| Lu et al., TPAMI 2013, authors' PDF and Crossref record | https://www.cs.ubc.ca/~murphyk/Papers/weilwun-pami12.pdf ; https://doi.org/10.1109/TPAMI.2012.242 | Foundation, Section 6 numbers |
| Koshkina et al. 2021, arXiv abstract | https://arxiv.org/abs/2104.10068 | Foundation, benchmark |
| SportsMOT, arXiv PDF, ML Anthology page, GitHub page | https://arxiv.org/abs/2304.05170 ; https://mlanthology.org/iccv/2023/cui2023iccv-sportsmot ; https://github.com/MCG-NJU/SportsMOT | Foundation, Tables 3, 4, 7, licence |
| SoccerNet Game State Reconstruction, arXiv PDF | https://arxiv.org/abs/2404.11335 | Foundation, Table 1, Section 6.3 |
| DeepSportradar-v1, arXiv PDF | https://arxiv.org/abs/2208.08190 | Foundation |
| Koshkina and Elder 2024, arXiv PDF and ML Anthology page | https://arxiv.org/abs/2405.13896 ; https://mlanthology.org/cvprw/2024/koshkina2024cvprw-general | Benchmark, recent work |
| WASB, arXiv PDF | https://arxiv.org/abs/2311.05237 | Benchmark (Table 2), recent work |
| Kim et al., KDD 2023, arXiv PDF | https://arxiv.org/abs/2306.08206 | Recent work (Table 2) |
| TranSPORTmer, arXiv PDF | https://arxiv.org/abs/2410.17785 | Benchmark (Table 3), recent work, E2 |
| GC-VRNN, arXiv PDF | https://arxiv.org/abs/2303.16005 | Benchmark (Table 1), E2 |
| Omidshafiei et al. 2022, PubMed Central | https://pmc.ncbi.nlm.nih.gov/articles/PMC9126960/ | Recent work |
| Theiner et al., WACV 2022, arXiv PDF | https://arxiv.org/abs/2110.11107 | Benchmark (Table 3, Section 4.3) |
| Crang et al. 2026, arXiv PDF and Crossref record | https://arxiv.org/abs/2508.19477 ; https://doi.org/10.1177/22150218261445834 | Benchmark |
| Bassek et al. 2025, LIDA index record (the publisher page refused automated access) | https://lida.sport-iat.de/twm/Record/4096118 ; https://doi.org/10.1080/24733938.2025.2533808 | Recent work |
| Mansourian et al. (PRTreID), arXiv abstract | https://arxiv.org/abs/2401.09942 | Recent work |
| Istasse et al. 2019, arXiv abstract | https://arxiv.org/abs/1907.01058 | Recent work |
| Hu et al. (BasketballBench), arXiv PDF | https://arxiv.org/abs/2608.23435 | Benchmark (Table 2), recent work |
| Basketball-SORT, arXiv abstract | https://arxiv.org/abs/2406.19655 | Recent work |
| PaliGemma, arXiv abstract | https://arxiv.org/abs/2407.07726 | Recent work |
| CLIP, arXiv abstract | https://arxiv.org/abs/2103.00020 | E1 |
| DINOv2, arXiv abstract | https://arxiv.org/abs/2304.07193 | E1 ablation |
| TOTNet, arXiv abstract | https://arxiv.org/abs/2508.09650 | Checked and not used: no basketball data |
| Consensus search, "broadcast video tracking validity vs optical tracking" | https://consensus.app | Found Crang, Bassek, Theiner |

Dropped because they could not be checked at the source: a vendor blog's 56% jersey accuracy for a small VLM
(a blog, not a paper); the journal version and page numbers of Basketball-SORT (seen only in a search snippet);
units for the L2 errors of Omidshafiei et al. (the paper does not state them, so no number is quoted); a second
page range for SportsMOT given by one index (the CVF and ML Anthology range is used).
