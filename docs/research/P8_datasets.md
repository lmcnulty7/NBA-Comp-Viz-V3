# P8: outside labelled basketball datasets (sweep of 2026-11-22)

Asked by Lucien on 2026-11-22: find more labelled data like SportsMOT, search and collect, and say what needs him.
Five read-only researchers searched by label type (players and tracks; team, jersey and identity; ball and events;
court and camera; synced tracking plus the dataset hubs). Each fact below was read at its source in that sweep; the
full reports, with every quote and URL, are kept in the lead's scratchpad for this session and summarised here. The
lead re-read the Hugging Face cards of E-BARD (3 repos), SHOT and saveerjain/basketball-events on 2026-11-22 and
confirmed their license tags and sizes. Nothing was downloaded, signed up for, or requested.

## Headline

- No public dataset pairs basketball broadcast video with tracking ground truth. The binding limit of P8 (broadcast
  video synced to tracking truth) is not loosened by any dataset found. The TrackID3x3 paper (arXiv 2503.18282) says
  the only large 5-on-5 tracking set is the 2015-16 NBA one we already use.
- Labelled frames, by contrast, exist for every weak row except court fit in new NBA arenas: player and referee
  boxes, uniform colour, jersey numbers, ball positions and frame-timed events, much of it on NBA broadcast footage.
- Roboflow Universe refuses automated reads (HTTP 403 on every page tried, about 50 projects). Those projects are
  listed in the scratchpad reports from search snippets only and count as unverified.

## Ranked: what to get, and the row each moves

| Rank | Dataset | Labels (as stated at source) | Footage | License (source) | Access | Rows it can move |
|---|---|---|---|---|---|---|
| 1 | E-BARD (Univ. of Brescia), 4 Hugging Face repos | 1,800 frames with 22,210 manual boxes (15,296 players, 3,853 referees, 1,565 hoops, 1,496 balls); 15,295 uniform colour labels; 3,633 hand-labelled 16-frame player tracklets with jersey numbers (300 "not visible") | 60 games, NBA 2024-25, all 30 teams, nba.com event clips | cc-by-4.0 (card metadata, re-read by the lead) | direct download | missed players, referee versus player, team, identity, ball |
| 2 | SportsMOT (MCG-NJU) | player boxes and track ids; referees and coaches not labelled; 45 train, 45 val, 150 test videos | YouTube: Olympics, NCAA, NBA; 720p, 25 fps | CC BY-NC 4.0, no redistribution | CodaLab sign-up (Lucien) | missed players, identity |
| 3 | WASB ball labels (NTT, BMVC 2023) | frame-wise manual 2D ball positions, 275,328 frames, 3,824 clips of 81 games | NBA broadcast (NUST-NBA181 clips), mostly 1080p | repository MIT; whether MIT covers the label zip: not stated | labels on Google Drive; the videos need an email request to the NUST-NBA181 author (Lucien) | ball in the image (the ball is untracked today); ships pretrained basketball weights |
| 4 | PL-NBA (arXiv 2608.19646) | frame-timed events by three annotators with consensus: dribble, pass, shots made and missed, offensive and defensive rebound, steal; no boxes | NBA 2022 to 2025 replays, main camera only, 1080p | CC BY-NC 4.0 | annotations on GitHub; videos self-download or Baidu | ball carrier, shots, rebounds |
| 5 | SHOT (ACM MM 2025) | manual keyframe boxes for all 10 players, offense ids 1 to 5 and defense 6 to 10, roles (holding, shooting); tracks, pose, gaze by model | NBA highlights and replays from Tencent Sports; seasons not named, clip names like ATLvsNJ and CLEvsGS | cc-by-nc-4.0 (card, re-read) | direct download | team (offense versus defense), identity, ball carrier |
| 6 | SoccerNet Game State and Jersey 2023 | soccer: role (with referee), team and jersey per person; 2,853 jersey tracklets with a "not visible" class | soccer broadcast | data license not stated (code GPL-3.0) | pip SoccerNet | identity, referee versus player (transfer only) |
| 7 | TeamTrack, basketball part | MOT boxes and ids, court trajectories; about 1.3M basketball boxes | university games, fisheye and drone, not broadcast | MIT (Kaggle and GitHub) | Kaggle or Google Drive | identity switches (association tuning) |
| 8 | SportCenter (EPFL CVLab) | per-image camera pose, intrinsics and homography; player ground positions on a subset; 50,127 images | one amateur match, two moving phones | "free of use for research purposes" | direct (Google Drive) | camera model and foot to court mapping (test bed, one venue) |
| 9 | TrackID3x3 (MMSports 2025) | boxes of all 6 players every frame, 10 keypoints on some frames, court coordinates | 3x3 university games, fixed cameras and drone | README: "licensed under ... CC BY 4.0" (read in one report; another found no data license: to check) | Google Drive | foot contact point (ankles against box bottom), identity |
| 10 | BARD (Univ. of Brescia) | clip-level actions, results, jersey number and colour from play-by-play, 14,676 clips | NBA 2024-25 | cc-by-4.0 (card) | direct | shots, rebounds (clip level) |

By request only, each worth one email if the rows above prove the value:

- NUST-NBA181 (ECCV 2020): 181 NBA broadcast games, 9,172 shot outcome clips; "available upon request and only for
  academic research"; the project page is dead (archived copy read). It is also the video source for WASB.
- SportsJumpMotion (CVPRW 2025): hand-labelled jersey numbers and legibility on SportsMOT basketball tracklets;
  email request, no license stated.
- Sha et al. 2020 (CVPR): 13 NCAA games in 13 venues, 640 images with homography from professional annotators; no
  release found. The only multi-venue broadcast calibration truth in print.
- BasketLiDAR (MMSports 2025): 3D positions and ids of B.League players at practice, LiDAR plus 3 cameras; by form.

Checks only, not training data:

- saveerjain/basketball-events: 4 NBA 2015-16 games, 897 events; "For research purposes only"; labelling method
  not stated. Its games are 01.12.2016 BOS at NYK and 12.06.2015 SAC at OKC (both our development videos) and two
  games outside the SportVU window (2016-01-25 GSW and SAS, 2016-03-05 WAS and IND). tools.p8_common.guard passes on
  all four (run by the lead). Usable as a same-era check of the shot and rebound rows on development games.

Not available: NBA Rebounds (arXiv 2512.15386), "not publicly available as explicit permission from the NBA has not
yet been granted".

## Rules before anything trains on these (Lucien, D149 and D150, 2026-11-22)

1. Outside labelled datasets may train P8 models (D150).
2. Seasons: D79 binds the play-by-play lanes only (D149), so E-BARD, BARD and PL-NBA (2022 to 2025) may train a
   detector; no tracking output on a 2024-25 or 2025-26 game flows into a play-by-play lane.
3. Screening before any training (D150): drop every clip of a spent held-out game and of a home game of MIA, TOR, LAC
   or LAL in any season (D85, D129). E-BARD covers all 30 teams, so its MIA, TOR, LAC and LAL home games come out.
   A set whose games cannot be identified (SportsMOT) is used with that residual risk disclosed.
4. Licenses: a row in docs/DATA_SOURCES.md before download; non-commercial is acceptable (D128); nothing is
   redistributed. Every training run records the datasets and versions it used.

## Expected effect, honestly

These sets move detection, referee, team colour, jersey and ball rows. They do not move the parts of position error
and missed players that come from the camera model and foot placement (docs/research/P8_step0_trackB.md), and they
add no synced tracking truth. Every gain is measured by the pinned scorecard on training games first, and on the D85
test set once at the end of P8.
