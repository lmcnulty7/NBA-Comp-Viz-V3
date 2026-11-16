# DATA.md: data survey for the D71 test set

Written 2026-11-16 by the data surveyor (PLAN.md D74). Every source below was fetched on 2026-11-16 (GitHub REST
API with a generic client, raw file hosts, WebFetch, WebSearch, the Hugging Face Hub). No dataset or archive was
downloaded; sizes come from the GitHub contents API or from HTTP HEAD responses. No file under any `heldout/` folder
was read. Repository facts cite the file they come from. "Not confirmed" means the source did not say it.

## Verdict

There is truly new, untouched tracking data, in two forms. First, the 2015-16 SportVU release itself holds 10 games
(all played 2015-12-05, 59.6 MB of archives) that the frozen models never saw: their file names carry a stray
`2016.NBA.Raw.SportVU.Game.Logs` prefix, the frozen name parser rejects them, and they appear in no manifest,
lane table or cache of this repository. They are the best untouched test set for every tracking lane (xfg,
rebounds, matchups, gravity, ghosting, embeddings, epv, decisions, load), but they are in distribution (same season,
players and provider, dated inside the training span) and only 10 games, so intervals will be wide. Second,
SkillCorner's open release (September 2026) gives 10 games of Spanish Liga ACB 2025-26 broadcast tracking at 25 fps
with players, ball (with height) and an event layer (shots with outcome, rebounds, possessions, passes, matchups).
It is a real out-of-distribution test for the tracking lanes, with large caveats: another league and rule set,
positions derived from broadcast video (17 to 24 percent extrapolated), other ids, and no SportVU-style event ids.
For the play-by-play lanes (lineups, winprob), the 2025-26 NBA season is complete on basketball-reference.com, the
source the repository used, and is a forward-in-time untouched season. What cannot be had: the 5 broken 2015-16
games are broken at the original upload in every copy found (no intact copy exists), 26 scheduled games of the
release window were never uploaded anywhere found, no public NBA frame-level tracking exists for any season after
2015-16 as far as the search reached, and the research trajectory sets (Sports-Traj, UniTraj, Zhan et al.) come from
the same STATS era without game ids, so they cannot be shown untouched.

## Sources

| name | link | what it holds | frame-level | players / ball | coverage | license | checked on | usable for | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linouk23/NBA-Player-Movements | https://github.com/linouk23/NBA-Player-Movements | 636 `.7z` SportVU game logs | yes, 25 Hz | 10 players, ball x y z | 2015-10-27 to 2016-01-23 | none stated (API `license: null`); NBA and STATS rights not stated | 2026-11-16 | test (10 untouched games); train (already used) | 10 archives under a misnamed file were never ingested (question 1) |
| neilmj/BasketballData (upstream) | https://github.com/neilmj/BasketballData | the original daily uploads; today its SportVU folder holds 1 game and a Readme | yes | as above | 1 game now; 636 in history | none stated | 2026-11-16 | none | history shows the broken files were broken at upload (question 2) |
| rajshah4/BasketballData | https://github.com/rajshah4/BasketballData | fork of neilmj, last push 2016-08-25 | yes | as above | the same 636 archives | none stated | 2026-11-16 | none | identical names and sizes to linouk23 |
| sealneaward/nba-movement-data | https://github.com/sealneaward/nba-movement-data | the same 636 archives (the 10 misnamed ones renamed correctly), 632 play-by-play CSVs, `shots.csv` | yes | as above | as above | MIT on the repository (cannot license the underlying data) | 2026-11-16 | none | identical sizes; nothing new |
| dcayton/nba_tracking_data_15_16 | https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16 | a loader script that downloads the linouk23 archives and a 2015-16 play-by-play CSV | yes (via linouk23) | as above | the linouk23 listing | not stated | 2026-11-16 | none | holds no data of its own |
| SkillCorner open data, basketball | https://github.com/SkillCorner/opendata-basketball | tracking for 10 ACB games, event files for those games, season aggregates for 293 games | yes, 25 fps | home and away players (x, y; z always 0), ball x y z, detection flag and expected error per position | 10 games, 2025-10-11 to 2026-05-29, Liga ACB 2025-26 | MIT License file in the repository; README asks for credit to SkillCorner | 2026-11-16 | test (out of distribution) | the only new public frame-level 5-on-5 tracking with ball found |
| Sports-Traj / UniTraj "Basketball" | https://arxiv.org/html/2405.17680 , https://github.com/colorfulfuture/UniTraj-pytorch | 50-step sequences of 10 players and ball | yes, 6.25 Hz | yes / yes | 93,490 train and 11,543 test sequences; season not stated | data license not stated; code Apache-2.0 | 2026-11-16 | none | STATS data, season and game ids not stated, so overlap with our training games cannot be ruled out |
| Zhan et al. STATS basketball | https://ar5iv.arxiv.org/html/1803.07612 , https://github.com/ezhan94/calibratable-style-consistency | offensive half-court sequences | yes, 6 Hz | 5 offensive players; "we ignore the defensive players and the ball" | 107,146 train, 13,845 test; season not stated | not stated (Google Drive link) | 2026-11-16 | none | same reason; no defense, no ball |
| TeamTrack | https://atomscott.github.io/TeamTrack/ , https://github.com/AtomScott/TeamTrack | MOT boxes and ids; side-view sets projected to pitch coordinates | yes (video) | players; ball not mentioned | university games: indoor fisheye, outdoor drone side and top | repository MIT; dataset license not stated on the page, Kaggle page not readable | 2026-11-16 | labels (player tracking) | not NBA, no ball, no events |
| TrackID3x3 | https://github.com/open-starlab/TrackID3x3 , https://arxiv.org/abs/2503.18282 | boxes for 6 players in every frame, pose for some frames, identity and team attributes | yes (video) | players; ball not mentioned | 3x3 games: indoor, outdoor, drone | dataset CC BY 4.0 (README) | 2026-11-16 | labels (identity, team) | 3x3, fixed cameras, not broadcast |
| BasketLiDAR | https://arxiv.org/abs/2508.15299 | LiDAR and multi-view camera with 3D player positions and ids | yes | players; ball not mentioned | 4,445 frames, professional 5-on-5 and 3-on-3 | not stated; "available upon request" | 2026-11-16 | none for now | needs a request (no sign-up allowed in this survey) |
| SportsMOT | https://github.com/MCG-NJU/SportsMOT | player boxes and track ids in 240 clips (basketball, football, volleyball) from Olympics, NCAA and NBA on YouTube, 720p, 25 FPS | yes (video) | players | basketball clips average 422 frames | CC BY-NC 4.0 (README) | 2026-11-16 | labels (player MOT) | non-commercial |
| DeepSportRadar basketball instants | https://github.com/DeepSportradar/instance-segmentation-challenge , https://github.com/DeepSportradar/ball-3d-localization-challenge | Keemotion images with calibration, player segmentation, ball for 3D localization, re-id crops | no (instants) | players, ball | not stated | segmentation data CC BY-NC-ND 4.0 (README); ball and re-id data license not confirmed | 2026-11-16 | labels (ball, calibration, segmentation) | non-commercial, no derivatives |
| WASB-SBDT basketball ball annotations | https://github.com/nttcom/WASB-SBDT | ball annotations on the "NBA_data" videos of the SAM paper | yes (video) | ball | not stated | code MIT; the video source page returned 404, data terms not confirmed | 2026-11-16 | labels (ball), after terms are confirmed | terms unknown |
| basketball-reference.com play-by-play | https://www.basketball-reference.com/leagues/NBA_2026.html | event text per game, with substitutions | no | n/a | 2025-26 complete (champion listed; teams at 82 games) | Sports Reference data use terms (question 4) | 2026-11-16 | test (lineups, winprob) | same source and parser as the frozen lanes |
| basketball-reference.com shot charts | https://www.basketball-reference.com/boxscores/shot-chart/202606130SAS.html | every field-goal attempt as a positioned marker with period, clock, player, made or missed, distance | no | n/a | 2015-16 (repo) and 2025-26 (checked) | as above | 2026-11-16 | test (xfg location part) | same source as `collect/bref_shots.py` |
| shufinskiy/nba_data | https://github.com/shufinskiy/nba_data | NBA.com play-by-play (stats v2, v3, cdn), pbpstats, shot details with x y, matchups, as tar.xz per season | no | n/a | 1996-97 to 2025-26, regular season and playoffs | repository Apache-2.0; NBA.com terms not reviewed | 2026-11-16 | test (xfg location part; play-by-play cross-check) | full 2025-26 confirmed by commit messages |
| DomSamangy/NBA_Shots_04_25 | https://github.com/DomSamangy/NBA_Shots_04_25 | NBA.com shot locations (LOC_X, LOC_Y), outcome, zone | no | n/a | 2003-04 to 2024-25 regular season | not stated in README | 2026-11-16 | none (shufinskiy covers more) | no 2025-26; license unknown |
| NBA.com (stats.nba.com, cdn.nba.com) | https://stats.nba.com/stats/ , https://cdn.nba.com/static/json/liveData/playbyplay/ | play-by-play, shot charts | no | n/a | not reachable | not reviewed | 2026-11-16 | none from this machine | stats.nba.com timed out at 15 s; cdn.nba.com returned HTTP 403 |
| BasketballBench | https://arxiv.org/abs/2608.23435 | question answering over 2,501 possession clips of the 2025-26 NBA season | no positions | none | 2025-26 | not stated for data; no download link | 2026-11-16 | none | no tracking |
| neilmj NCAAM.Player.Tracking | https://github.com/neilmj/BasketballData/tree/master/NCAAM.Player.Tracking | one CSV of per player-game aggregates (distance, touches, rebound chances) | no | n/a | NCAA 2014-15 | none stated | 2026-11-16 | none | aggregates, not frames |

Dropped because not confirmed: a public college SportVU frame-level release (arXiv 1806.06696 mentions a college
tracking dataset but no release); the SAM "NBA_data" video set page (404).

## Question 1: the 636 versus 626 gap

Evidence. The GitHub contents API listing of `linouk23/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs`
(the `LIST_URL` in `sportvu/manifest.py`) returns 636 entries, all `.7z`. The `games` keys of
`DATA_ROOT/sportvu_contract/manifest.json` number 626. Every manifest key is in the listing. The 10 listing entries
missing from the manifest are the 10 games of 2015-12-05, stored under names that start with the folder name:

| game | file name in the listing | size (bytes) | git blob sha |
| --- | --- | --- | --- |
| 12.05.2015.BOS.at.SAS | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.BOS.at.SAS.7z | 5,393,125 | cc37c5c2cba022c64c5e11a1bab3dd6f63c02d48 |
| 12.05.2015.CHA.at.CHI | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.CHA.at.CHI.7z | 5,714,149 | 19823e860fc32b36d43ebe997c03fad533ff4722 |
| 12.05.2015.CLE.at.MIA | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.CLE.at.MIA.7z | 6,154,105 | 44c0f1287082ece4e9f82e7a5412c2389ea427fc |
| 12.05.2015.DEN.at.PHI | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.DEN.at.PHI.7z | 6,499,003 | e753686364ccf497583f35fc116f35486e9d17f9 |
| 12.05.2015.GSW.at.TOR | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.GSW.at.TOR.7z | 6,320,419 | ca535614ef883ceaf7346d42673da3f1b58f1365 |
| 12.05.2015.IND.at.UTA | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.IND.at.UTA.7z | 6,888,899 | 74ef1051ede87a78f091bdda8f35eb97f9b8669f |
| 12.05.2015.NYK.at.MIL | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.NYK.at.MIL.7z | 5,198,010 | 91183499a95838a5387e073227a034de5958e9ca |
| 12.05.2015.ORL.at.LAC | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.ORL.at.LAC.7z | 5,815,756 | 3bbb6fd176cab93e7a284ec4c89f69969b6e7161 |
| 12.05.2015.POR.at.MIN | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN.7z | 5,840,157 | ba31bb86be48a75b18c33fc798d65776190907e3 |
| 12.05.2015.SAC.at.HOU | 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.SAC.at.HOU.7z | 5,759,727 | 13e7d16ad20a1483bdb5ebffef32dea357e88271 |

- Why they were missed: `sportvu/manifest.py` line 46, `NAME_RE = ^(\d\d)\.(\d\d)\.(\d{4})\.([A-Z]{3})\.at\.([A-Z]{3})\.7z$`,
  is anchored at the start, so `parse_name` returns None for the prefixed names and `list_all` drops them. They are
  not duplicates, not renamed copies of other games, not preseason and not too large: sizes sit in the normal range
  (listing median 6,076,965 bytes, max 8,574,753). The misnaming is in the upstream upload (neilmj history shows
  the same names) and was added to linouk23 in commit 59b354a6 ("Added match data, pt. 4", 2016-09-19).
- They are real games: the repository's own schedule table (`DATA_ROOT/collect/bref_schedule/games/2015-16.parquet`,
  basketball-reference) lists exactly these 10 games on 2015-12-05.
- They are untouched: "12.05.2015" or "20151205" occurs 0 times in `sportvu_contract/manifest.json`,
  `collect/bref_shots/manifest.json`, `collect/bref_seasons/manifest.json` and `analytics/phase0_manifest.json`;
  0 of the 604 files in `analytics/possessions/2015-16/` and 0 entries under `DATA_ROOT/sportvu`, `DATA_ROOT/tracking`
  and the local `data/sportvu` carry the date. None is a game at CLE, so none falls under the held-out rule of
  `sportvu/splits.json`.
- Both raw URLs answer: an HTTP HEAD on the prefixed raw URL returned 200 with content-length 5,393,125
  (BOS at SAS) and 5,759,727 (SAC at HOU). Contents were not opened (no download in this survey), so whether every
  archive parses is not confirmed.
- Shots and play-by-play exist for them: the basketball-reference shot chart of 2015-12-05 BOS at SAS
  (`boxscores/shot-chart/201512050SAS.html`) holds 170 shot markers; the page format ("1st quarter", class
  `tooltip q-1 p-<slug> make`) is matched by the regexes in `collect/bref_shots.py` (lines 79 to 81 accept both
  "Qtr" and "quarter"); class order against the parser was not tested.
- Expected size of the set: at the training average of 167.8 attempts a game (101,853 attempts over 607 games,
  reports/results.md), about 1,700 shots. This is an estimate, not a count.

Other gaps in the release, for the record:

- The 5 broken games: 01.14.2016.CHI.at.PHI, 01.14.2016.CLE.at.SAS, 01.14.2016.DET.at.MEM and 01.23.2016.UTA.at.WAS
  are 32 bytes each and share one blob sha (4f8fc9c9d1695f244efa249f2215dca4fa92bf2a), so they are the same empty
  file; 01.23.2016.CHI.at.CLE is 2,350 bytes (blob e589b7189b463b737e6c16c1c6fe330d242ca203). UTA at WAS was never
  played that day: the repository's schedule table dates UTA at WAS on 2016-02-18 (a postponed game).
- Of 661 regular-season games scheduled from 2015-10-27 to 2016-01-23 in that schedule table, 26 have no archive at
  all: 10.28.2015 CHI.at.BKN, CHA.at.MIA, DAL.at.PHX; 10.30.2015 GSW.at.HOU; 11.09.2015 CHI.at.PHI;
  11.27.2015 NOP.at.LAC; 11.28.2015 BKN.at.CLE; 12.02.2015 IND.at.LAC; 12.03.2015 BOS.at.SAC; 01.14.2016 TOR.at.ORL
  (London); all 8 games of 2016-01-16 and all 5 of 2016-01-17; 01.21.2016 LAC.at.CLE, ATL.at.SAC, SAS.at.PHX.
  No copy found holds any of them.
- Arithmetic check: 636 archives minus 5 broken gives the 631 games the literature cites (for example the
  Sports-Traj and Hugging Face descriptions); 631 minus the 10 misnamed gives the 621 usable games of the manifest.

Note for the lead: `sportvu/manifest.py` is frozen. Ingesting these 10 games needs new code under `tools/` or
`export/` (or a renamed local copy), never a change to `NAME_RE`.

## Question 2: other copies of the release

Found by search (WebSearch for SportVU raw game logs mirrors) and confirmed through the GitHub API:

- neilmj/BasketballData, the upstream. Today its `2016.NBA.Raw.SportVU.Game.Logs` folder holds one archive
  (10.27.2015.DET.at.ATL, 6,183,079 bytes) and a Readme ("daily uploads of full game raw SportVU game logs").
  At the commit that added the 1/14 games (c1f42cc9db, 2016-01-18) the three 01.14.2016 broken files were already
  32 bytes; at the commit that added the 1/23 games (b46f92a943, 2016-01-24, 637 entries) CHI at CLE was 2,350 bytes
  and UTA at WAS 32 bytes. The files were removed later in commit d0bf575a05 ("update", 2016-08-22). So the 5 games
  were broken from the first upload; there was never an intact version upstream.
- rajshah4/BasketballData, a fork of neilmj (last push 2016-08-25): 636 archives plus the Readme; every name and
  every size equals linouk23, the 5 broken files included.
- sealneaward/nba-movement-data (MIT, last push 2018-02-08): 636 archives with the 10 misnamed ones renamed to
  `12.05.2015.<AWAY>.at.<HOME>.7z`; every size equals linouk23; the 5 broken files are the same sizes. It adds
  `data/events/` (632 play-by-play CSVs named by NBA game id, for example `0021500001.csv`) and `data/shots/`
  (`shots.csv` 40,592,226 bytes, `shots_fixed.csv` 18,951,683 bytes).
- dcayton/nba_tracking_data_15_16 (Hugging Face): no data files; its loader downloads from the linouk23 listing.

Result: no intact copy of any of the 5 broken games, and no game outside the linouk23 listing, in any copy found.

## Question 3: other public frame-level tracking

Searches (WebSearch, extended mode, several phrasings; Hugging Face Hub) found one new public 5-on-5 source with
players and ball, and several that are not usable as tests.

SkillCorner open data, basketball (repository created 2026-09-07, last push 2026-09-10; announced as part of
SkillCorner's Analytics Cup 2.0):

- Holds: for 10 Liga ACB 2025-26 games, `{gameId}_tracking_data.jsonl.gz` (one line per video frame, 25 fps,
  Git LFS, "roughly 30-46 MB per game"; the API shows a 133-byte LFS pointer), `{gameId}_dynamic_events.json`
  (20 tables: possessions, chances, chance_players, matchups, shots, free_throws, rebounds, turnovers, fouls,
  timeouts, passes, touches, dribbles, picks, handoffs, off_ball_screens, drives, isolations, posts, closeouts;
  3,526,294 bytes for game 114243) and `{gameId}_game_data.json` (rosters). Season aggregates of shots, drives and
  picks cover 293 of 327 games, offense only.
- Frames: players with `xyz` in feet from court centre (z always 0), `speed`, `jersey`, `playerId`, `isDetected`,
  `predError`; ball with `xyz` (real height), `speed`, `isDetected`, `predError`; `gameClock`, `shotClock`,
  `period`. About half the frames are dead time with empty player and ball fields.
- Quality, from the README: 76 to 83 percent of player positions detected, the rest extrapolated; mean expected
  error 1.5 to 1.9 ft (95th percentile 4.9 to 6.3 ft); extrapolated positions 4.4 to 5.2 ft; "ball calibration is
  not as accurate as player calibration".
- Shots carry `outcome` (made or missed), `closestDefDist`, `closestDefId`, `contestLevel`, `distance`.
- License: an MIT License file covers the repository; the README asks users to credit SkillCorner. No separate data
  license.
- Differences from 2015-16 SportVU: another league and rule set (FIBA rules; court size, three-point distance and
  game length differ and must be read from the FIBA rules before use, not checked here); positions come from
  broadcast video, not arena cameras, with extrapolation when players are off screen; players have no height;
  SkillCorner player ids, not NBA ids; events come from SkillCorner's models, not an official play-by-play;
  coordinates centred on the court with a per-event attack direction. Ten years later in style of play.
- Use for the tracking lanes as an out-of-distribution test: xfg (features recomputed from their frames; outcome
  from `shots`), rebounds (`rebounds` plus ball height), gravity and ghosting (defender positions, restricted to
  detected frames), epv and decisions (possessions, passes, drives), matchups (their `matchups` table is another
  model's output, so it is a comparison, not ground truth), embeddings (other players and league, so only a
  descriptive transfer check), load (speeds exist, but absence needs ACB rosters, not in the release). Each needs a
  coordinate and rules adapter under `export/` or `tools/`, written before the set is scored.

Not usable as an untouched test:

- Sports-Traj and UniTraj "Basketball": 93,490 train and 11,543 test sequences of 10 players and ball at 6.25 Hz
  ("The base dataset is available from Stats Perform"); Zhan et al. 2018: 107,146 train and 13,845 test sequences at
  6 Hz, offense only ("We ignore the defensive players and the ball"). Season and game ids not stated, so overlap
  with our 604 training games cannot be excluded; no shot outcomes or events.
- TeamTrack, TrackID3x3, BasketLiDAR: frame-level, but university, 3x3 or LiDAR captures with players only (ball not
  mentioned), no events; useful as vision labels at most (question 6).
- BasketballBench (2025-26 NBA clips): no positions.
- No public NBA frame-level tracking for any season other than 2015-16 was found. The Hugging Face card of
  dcayton states "2015-16 was the last season with publically available tracking data"; that is the card's claim,
  consistent with what the searches returned.

## Question 4: the 2025-26 play-by-play season

How the repository fetched play-by-play: `collect/bref_seasons.py` (basketball-reference `boxscores/pbp/<code>.html`,
parsed by `fetch_pbp.parse_events`, at most 17 requests a minute, stop on the first 429 or 403), compacted to
`DATA_ROOT/collect/stints/playbyplay/<season>.parquet`. The 2023-24 table read for this survey has one `source`
value, `basketball-reference.com`, over 592,078 rows and 1,313 games. Starters come from `collect/bref_boxscores.py`
(box score pages) into `stints/starters/<season>.parquet`. The load lane does not use play-by-play: it reads the
2015-16 SportVU frames (reports/load.md, Inputs).

2025-26 exists and is complete at that source:

- `leagues/NBA_2026.html` lists the champion (New York Knicks) and teams at 82 games (W plus L checked for
  OKC, DET, BOS, UTA, WAS; the team table lists G = 82 for every team).
- `leagues/NBA_2026_games-june.html` lists the five Finals games with final scores, last on 2026-06-13.
- `boxscores/pbp/202606130SAS.html` holds a full four-quarter play-by-play with substitution lines ("enters the game
  for"), from the opening jump ball to "End of 4th quarter".
- April page: Play-In games are labelled "Play-In Game"; the schedule parser must keep them out of the regular
  season as it did for the frozen seasons.

What a fetch would take (no fetch now): the schedule (one page a month), one play-by-play page a game and one box
score page a game for starters: about 1,230 regular-season games, so about 2,500 page requests plus postseason if
wanted; at the repository's 17 requests a minute, about 2.5 hours, under the heavy-job lock. Sports Reference terms
(https://www.sports-reference.com/bot-traffic.html, https://www.sports-reference.com/data_use.html, read 2026-11-16):
no more than 20 requests a minute ("your session will be in jail for up to a day"); the data use page forbids using
site content "for purposes of training, fine-tuning, prompting, or instructing artificial intelligence models" and
says users "should not create websites or tools based on data you scrape". These terms bear on the frozen lanes as
much as on a new season; the decision belongs to Lucien.

An alternative copy from NBA.com: shufinskiy/nba_data (Apache-2.0 repository) holds `nbastatsv3_2025.tar.xz`
(8,967,164 bytes), `cdnnba_2025.tar.xz` (19,290,552), `nbastatsv3_po_2025.tar.xz` (614,224) and `matchups_2025`;
its README names seasons by start year (`seq(1996, 2024)` for 1996/97 to 2024/25), and its commits read
"upd nba data: full regular season 2025/26" (2026-04-25) and "add full nba PO 2025/26" (2026-07-17). It is a
different source with different event codes from the frozen lanes, so it suits a cross-check, not the main test.

## Question 5: shot charts with x, y

- basketball-reference shot chart pages: one per game, every field-goal attempt as a positioned marker with period,
  clock, shooter, made or missed and stated distance. Confirmed for 2015-16 (the repository collected 625 games;
  checked again on 201512050SAS: 170 markers) and 2025-26 (202606130SAS: 173 markers, equal to 87 plus 86 field-goal
  attempts). Same source and parser as the frozen xfg lane, so it is the first choice.
- shufinskiy/nba_data `shotdetail_<year>`: NBA.com ShotChartDetail rows for 1996-97 to 2025-26
  (`shotdetail_2025.tar.xz` 2,825,192 bytes, `shotdetail_2015.tar.xz` 2,609,520 bytes). NBA.com coordinates, a
  different convention from the bref pixel mapping (`collect/bref_shots.py` notes the loc_x sign is unverified).
- DomSamangy/NBA_Shots_04_25: NBA.com shots with LOC_X, LOC_Y for 2003-04 to 2024-25 regular season, Google Drive
  link, no license stated. No 2025-26.
- sealneaward `shots.csv` and `shots_fixed.csv`: shots of the same 2015-16 SportVU games; nothing new.
- NBA.com directly: stats.nba.com timed out at 15 s; cdn.nba.com live play-by-play returned 403 to a non-browser
  client.
- Use: a later season tests only the location part of xfg (the location-only baseline). The contest features need
  tracking, which no later NBA season has publicly.

## Question 6: labels for the broadcast tracking pipeline

| source | ball | identity | team | license (as stated at the source) |
| --- | --- | --- | --- | --- |
| SportsMOT (basketball part, NBA, NCAA, Olympics clips, 720p, 25 FPS) | no (not mentioned) | player track ids | no | CC BY-NC 4.0 |
| DeepSportRadar basketball instants (Keemotion images) | ball for 3D localization, camera calibration | re-id crops (train 436 ids, 8,569 images) | no | segmentation data CC BY-NC-ND 4.0; ball and re-id data license not confirmed (README lists a License section that is empty) |
| TrackID3x3 | no | jersey and player ids | team attributes | CC BY 4.0 |
| TeamTrack (basketball fisheye and drone) | no | track ids | not stated | repository MIT; data license not confirmed |
| WASB-SBDT ball annotations on "NBA_data" | yes | no | no | code MIT; video source page 404, terms not confirmed |
| BasketLiDAR | no | 3D player ids | not stated | on request |
| SkillCorner ACB | ball positions, but no video to pair them with | jersey per player per frame | home or away | MIT file; video not included |

The repository's Roboflow sets (`docs/DATA_SOURCES.md`) remain the only labels already in hand. Every set above with
NC or ND terms is fine for a research measurement but needs Lucien's decision before any use in a product.

## Recommendation for D71

Two test sets, each fixed and hashed before any experiment and scored once per experiment: A, the 10 untouched
2015-12-05 SportVU games (in distribution; hash the archive bytes after download and keep the git blob shas above);
B, the 10 SkillCorner ACB 2025-26 games (out of distribution). The play-by-play lanes use the 2025-26 NBA season.
A carve-out from the 604 training games is not untouched (the frozen models fit on them) and should be used only if
set A turns out not to parse.

| lane | best test source | caveat |
| --- | --- | --- |
| xfg | A (shots from basketball-reference shot charts, about 1,700 attempts); B as an out-of-distribution check; 2025-26 bref shot charts for the location-only baseline | 10 games give wide intervals; B needs FIBA court geometry and recomputed contest features |
| rebounds | A; B as a check | B rebound events come from SkillCorner's models, and its ball calibration is weaker |
| matchups | A | B's `matchups` table is another model's output, a comparison only; the lane already shows no defender signal |
| gravity | A; B as a check | B positions are 17 to 24 percent extrapolated; restrict to detected frames |
| ghosting | A; B as a check | B defender positions off screen are extrapolated, which biases the deviation |
| embeddings | A (new player-games for a stability check) | no held-out score by design; B players and league differ, descriptive only |
| epv | A; B as a check | B possession outcomes and rules (game length, shot clock) differ |
| decisions | A | failed lane; any score stays descriptive |
| load | A (speeds from frames; absence from the 2015-16 rosters and schedule) | 10 games of about 22 player-games each; B has no next-game rosters |
| lineups | 2025-26 NBA play-by-play from basketball-reference (forward in time) | about 20 recurring lineups a season, so it stays underpowered; Sports Reference terms need Lucien's decision |
| winprob | 2025-26 NBA play-by-play from basketball-reference | no odds, so `spread_missing` stays on as in the frozen run; same terms caveat |
