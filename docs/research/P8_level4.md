# P8 Level 4: what truth exists for current seasons

Written 2026-11-21 and 2026-11-22 under PLAN.md D137 (continue toward Level 4 of `docs/research/TRACKING_LEVELS.md`)
and D128's constraints. Measurement only: no training, no hand labelling, no League Pass. Sections 1 to 5 downloaded
no video; sections 6 to 9 (second pass, the lead's decisions after the first) downloaded identity sections only
(8-second pieces). Every number below was read from output produced in this work; every quote was fetched at its
source on the date given.

Level 4 (TRACKING_LEVELS.md) asks for current-season broadcasts and a check of the pipeline against partial truth:
"the public shot coordinates against where the pipeline puts the shooter at the shot, and play-by-play consistency".
This file measures the two sources of partial truth that exist without a browser: SkillCorner's open basketball
tracking (Liga ACB 2025-26) and NBA shot coordinates for 2025-26.

## 1. SkillCorner open basketball data

### Source and license

- Repository: github.com/SkillCorner/opendata-basketball, commit `4bbed2e35e8fdd2cf083e8c8b280b3e5381a2c66`
  (2026-09-10). GitHub API: created 2026-09-07T09:03:52Z, last push 2026-09-10T14:21:31Z, license `MIT`.
- `LICENSE`: "The MIT License / Copyright (c) 2026 SkillCorner." It grants permission "to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies", on the condition that "The above copyright notice and this
  permission notice shall be included in all copies or substantial portions of the Software."
- `README.md` asks for credit: "If you use the data, we kindly ask that you credit SkillCorner and notify us on
  Twitter so we can follow the work being done with it."
- Downloaded to the P8 data workspace outside the repository (`_p8-data/skillcorner/`, beside the worktrees): the
  repository without Git LFS content, then the 10 LFS tracking files from media.githubusercontent.com. Each file's
  size and sha256 equal its LFS pointer (10 of 10).

### The 10 games

From `data/matches.json`. Its `date_time` "is the official ACB scheduled tip-off in UTC" (README). The release has
no venue field (`game_data.md` lists none); the venue is the home team's, not named in the data.

| gameId | Tip-off (UTC) | Home | Away | Score |
|---|---|---|---|---|
| 114243 | 2025-10-11 19:00 | BAXI Manresa | Coviran Granada | 83-68 |
| 114234 | 2025-10-18 19:00 | Casademont Zaragoza | Leche Rio Breogan | 84-88 |
| 114169 | 2025-12-14 11:30 | CB UCAM Murcia | Club Joventut Badalona | 80-77 |
| 114099 | 2026-01-31 20:00 | Basquet Girona | San Pablo Burgos | 77-71 |
| 114086 | 2026-02-08 17:00 | Dreamland Gran Canaria | Bitci Baskonia | 75-97 |
| 178442 | 2026-03-21 18:00 | BC MoraBanc Andorra | Bilbao Basket | 98-102 |
| 179612 | 2026-04-12 10:30 | Malaga | Valencia BC | 89-96 |
| 184439 | 2026-04-19 17:00 | Real Madrid | Lenovo Tenerife | 90-95 |
| 188630 | 2026-05-03 17:00 | BC Barcelona | Dreamland Gran Canaria | 91-69 |
| 191313 | 2026-05-29 18:30 | Leche Rio Breogan | Casademont Zaragoza | 94-95 |

### Fields (from `docs/data_dictionary/tracking_data.md` and `docs/PRIMER.md`)

- One JSON line per video frame at 25 fps: `frameIdx`, `wallClock` (ms "since the start of the video"),
  `gameClock`, `gameClockStopped`, `period`, `shotClock`, `homePlayers`, `awayPlayers`, `ball`.
- Player: `xyz` in feet from court centre, "z is height, currently always 0"; `speed` (ft/s), `jersey`,
  `playerId`, `isDetected` ("`1` if the player is on screen and detected by computer vision, `0` if the position is
  extrapolated"), `predError` ("Expected error for the player's coordinate, in feet. In 90% of cases the tracking
  is within `predError` feet of ground truth").
- Ball: `xyz` where "z (height) is a real height and can go slightly negative near the floor (calibration noise)";
  `speed`, `isDetected` (a float), `predError`; "Ball calibration is not as accurate as player calibration."
- "roughly half of all frames in a game (dead ball, warmups, breaks) carry empty `homePlayers`/`awayPlayers` arrays
  and an empty `ball` object".
- Events (`{gameId}_dynamic_events.json`, 20 tables) come from "a Game Intelligence model [that] reads the tracking
  data" (PRIMER). In `shots`, `location` is the "x, y coordinate of the shooter at the release of the shot" and is
  not play-by-play derived; `outcome`, `three`, `shooterId` are play-by-play derived (`shots.md`, "PBP derived").
  `docs/KNOWN_ISSUES.md`: "All 10 published games match the official final score exactly".

### Measured per game

One pass over each tracking file under the heavy-job lock (one game at a time). Frames: all lines; live: frames
with players (every live frame held 10.0 players on average and a ball); duration: last `wallClock`. Player columns
are over every live player position. `shots` counts rows of the shots table (it "includes fouled misses").

| gameId | frames | live frames | duration (min) | periods | players detected | player predError mean (ft) | detected | extrapolated | ball predError mean (ft) | ball z median / p99 / min (ft) | shots |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 114243 | 146,870 | 77,023 | 99.7 | 4 | 0.826 | 1.56 | 0.91 | 4.63 | 1.85 | 3.52 / 14.27 / -5.97 | 159 |
| 114234 | 158,923 | 72,132 | 107.7 | 4 | 0.828 | 1.63 | 0.89 | 5.15 | 1.76 | 3.48 / 14.52 / -6.42 | 144 |
| 114169 | 172,257 | 106,176 | 118.7 | 4 | 0.762 | 1.86 | 0.90 | 4.92 | 1.94 | 3.59 / 13.69 / -4.81 | 148 |
| 114099 | 147,892 | 75,655 | 100.8 | 4 | 0.817 | 1.58 | 0.88 | 4.72 | 1.82 | 3.58 / 14.84 / -4.99 | 150 |
| 114086 | 124,796 | 92,188 | 84.4 | 4 | 0.807 | 1.52 | 0.84 | 4.36 | 1.83 | 3.52 / 14.07 / -5.42 | 139 |
| 178442 | 162,479 | 89,642 | 108.3 | 5 | 0.803 | 1.60 | 0.88 | 4.57 | 1.88 | 3.57 / 14.90 / -5.29 | 158 |
| 179612 | 140,856 | 83,437 | 93.9 | 4 | 0.789 | 1.72 | 0.91 | 4.76 | 1.83 | 3.63 / 14.96 / -4.94 | 145 |
| 184439 | 154,021 | 77,428 | 102.7 | 4 | 0.815 | 1.61 | 0.88 | 4.85 | 1.87 | 3.55 / 14.49 / -5.70 | 142 |
| 188630 | 121,793 | 73,138 | 81.2 | 4 | 0.806 | 1.57 | 0.86 | 4.52 | 1.80 | 3.48 / 14.37 / -5.65 | 133 |
| 191313 | 143,045 | 82,885 | 95.4 | 4 | 0.793 | 1.75 | 0.88 | 5.08 | 1.90 | 3.53 / 14.64 / -5.68 | 141 |

These agree with the README's ranges (76 to 83 percent detected, mean predError 1.5 to 1.9 ft, extrapolated 4.4 to
5.2 ft). Two findings the README does not state:

- The ball's `isDetected` is 0.0 on every ball frame of all 10 games (game 114243: 77,023 of 77,023 ball objects,
  all with keys `isDetected`, `predError`, `speed`, `xyz`). The ball detection flag carries no information in this
  release, so detected and extrapolated ball positions cannot be told apart.
- Ball height goes as low as -6.42 ft (game 114234), beyond "slightly negative".

SkillCorner's positions are themselves a broadcast estimate (its own `predError`), so they are partial truth: a
comparison with our pipeline measures agreement, and that agreement cannot be read more finely than SkillCorner's
own error.

## 2. Broadcast video for the 10 games

### Method

`tools/p8_level4_search.py`, metadata only (`yt-dlp --flat-playlist` for YouTube search, `--skip-download --print`
with `bilisearch5` for Bilibili, `--skip-download -j` for each probe). Raw output: `_p8-data/level4/` (the first pass
`search_results.json`, the second pass `pass2_a.json` and `pass2_b.json`, merged in `merged.json`).

- First pass, 5 forms per game: "{home} vs {away} partido completo", "{home} - {away} ({score}) Liga Endesa",
  "{home} vs {away} {yyyy-mm-dd} full game", "Liga Endesa {home} {away} {dd/mm/yyyy}" (YouTube, 10 results each) and
  "{home} {away} ACB" (Bilibili, 5 results). Both dated forms returned 0 results on every game (one reproduced by
  hand: 0 results, exit 0); cause not established.
- Second pass, the 5 forms again plus 3 with the date in words: "{home} {away} partido completo {d} {mes} {yyyy}",
  "{home} vs {away} full game {Month} {d} {yyyy}", "{home} {away} Liga Endesa 2025 26 completo". Every result
  listed at 480 s or more, or with no listed length, was probed.
- A result is kept when its title names both teams. Length classes fixed before the search: full at least 3,600 s,
  condensed 480 to 3,600 s, highlights under 480 s.
- A probed video is a candidate for the game only when it is at least 480 s long and was uploaded on the game day or
  up to 7 days after. Rejected by title: "Live Score" ticker streams, simulations, youth leagues (Liga U, Minicopa),
  other seasons.
- 130 queries in all (13 per game). 12 of the 20 Bilibili queries failed, each with "HTTP Error 412: Precondition
  Failed"; no Bilibili result named both teams. No YouTube query failed.

### ACB's own channel

The ACB channel (YouTube `@ACBCOM`, channel id `UCjZh7FzztNo3up7mD75DAiQ`) posts a RESUMEN or HIGHLIGHTS video for
each Liga Endesa game. Its search for "partido completo" returns full games only for youth competitions (Liga U
2025-26), Minicopa streams and historic games (for example "PARTIDO COMPLETO | Final Copa del Rey Málaga 2014");
none of the 20 listed was a 2025-26 Liga Endesa game. Its streams tab lists Minicopa Endesa games only.

### Per game

| gameId | named both (unique) | candidates (at least 480 s, day 0 to 7) | ACB highlights (id, s, max height) |
|---|---|---|---|
| 114243 | 3 | none: `0gyYBPIapcs` (801 s, "Análisis y resumen", an analysis video) | `KxOXoriFwDM`, 184, 1080 |
| 114234 | 10 | none | `ziPCMtLczBU`, 186, 1080 |
| 114169 | 17 | none | `APBbUdF7O5U` (RESUMEN), 191; `5gRKroQRgoY` (HIGHLIGHTS), 216 (listing only) |
| 114099 | 9 | none: `BDFnvbVpSHw` is a "Live Score" ticker | `sJQH1Gl43DY`, 214, 1080 |
| 114086 | 9 | none | `UJvELKi9mi4`, 186, 1080 |
| 178442 | 5 | none: `2xOj9dSJf8w` is a "Live Score" ticker | `IQYIKzWU0-U`, 223, 1080 |
| 179612 | 20 | **`00k9_ueP2bY`**: "UNICAJA MÁLAGA - VALENCIA BASKET \| Liga Endesa 25/26 J26 \| Partido completo", 7,454 s, uploaded 2026-04-12, max 720p (avc1 available), channel "Partidos Unicaja ACB", description "Partido completo Unicaja - Valencia 12/04/2026". Also `Sg9nA-p79Hg` (754 s, a "Videocrónica") and `eStct1H44mc` ("Live Score") | `daTJjIE_4nw` (RESUMEN), 213; `5-Pe4dQrOKg` (GAME HIGHLIGHTS), 199 (listing only) |
| 184439 | 22 | none: `rYaLcizFEjk` is a "Live Score" ticker | `SKKNCRJ_LYQ` (RESUMEN), 192; `ja3i7hZtbb4` (HIGHLIGHTS), 230 (listing only) |
| 188630 | 15 | unverified: `lefFmF2gEvA`, "Liga Endesa EN VIVO \| Barça Basket vs Dreamland Gran Canaria \| Jornada 29 \| Narración en directo", 8,813 s, a past live stream (`was_live`) of a fan commentary channel (PickandRollTV), max 1080p; nothing in its metadata says it shows the broadcast picture. `MTIRlnzsb-Q` is a "Live Score" ticker | `qGLpdHrPVws`, 214, 1080 |
| 191313 | 11 | none: `0v7SXJ7_cD0` is a "Live Score" ticker; `AmdCMw-Dv80` (1,594 s) is the club's post-game show ("POST") | `Fv70wnA3yOM`, 253, 1080 |

Result: 1 game of 10 (179612) has a full-game candidate whose metadata names the game and its date; 1 (188630) has
a long live-commentary stream whose picture is unknown; all 10 have ACB highlights of 184 to 253 s at 1080p. No
candidate is confirmed as the game: that needs an automatic identity check in the spirit of D130 (OCR of period,
clock and score at 3 or more moments, matched to SkillCorner's `gameClock` and the score from `shots` and
`free_throws`), which this task did not run (no video downloaded). Uploads by third parties carry the standing
caveats of the YouTube row in `docs/DATA_SOURCES.md` (research and education, no video redistributed).

## 3. NBA shot coordinates for 2025-26

### Direct NBA endpoints (2026-11-21, UTC 09:54, user agent "Mozilla/5.0 (generic research client)", no other
custom header)

- `stats.nba.com/robots.txt` redirects (301) to `www.nba.com/stats/robots.txt`, which returns 404. `www.nba.com`'s
  robots.txt has "User-Agent: *" with "Disallow: /api/*" and "Disallow: /stats/events/".
- `stats.nba.com/stats/shotchartdetail` (GameID 0022500001, Season 2025-26): no response in 40 s ("Operation timed
  out after 40006 milliseconds with 0 bytes received"). The repository already records that the site "times out for
  non-browser clients" (`fetch_pbp.py` line 5). Not retried with browser-imitating headers: that would work around
  the server's refusal of non-browser clients.
- `cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022500001.json`: HTTP 403 "Access Denied". A request for
  `cdn.nba.com/static/json/staticData/scheduleLeagueV2.json` (to tell a missing file from a block) also returned
  403 "Access Denied", so the CDN refuses this client. Stopped there: 3 data requests in all.

### Terms (www.nba.com/termsofuse, "LAST UPDATED: July 13, 2026", fetched 2026-11-21)

- Scope: they "apply to your access and use of the digital platforms of the National Basketball Association ...
  including League websites (including, but not limited to nba.com ...), apps ... and online content offerings
  (collectively, the "Services")".
- "NBA STATISTICS ... By using such NBA Statistics, you agree that: (i) any use, display, or publication of the NBA
  Statistics shall include a prominent attribution to NBA.com in connection with such use, display, or publication;
  (ii) the NBA Statistics may only be used, displayed, or published for legitimate news reporting or private,
  non-commercial purposes; (iii) the NBA Statistics may not be used in connection with any sponsorship or commercial
  identification; (iv) the NBA Statistics may not be used in connection with any gambling activity (including legal
  gambling activity); (v) the NBA Statistics may not be used in connection with any fantasy game or other commercial
  product or service; (vi) the NBA Statistics may not be used in connection with any product or service that
  presents a live, near-live, or other real-time or archived play-by-play account or depiction of any NBA game; and
  (vii) the NBA Statistics may not be used in connection with any website, product, or service that features a
  database (in any medium or format) of comprehensive, regularly updated statistics from NBA, WNBA or G-League games,
  competitions, or events without the Operator's express prior consent."
- "No Basketball Content from the Services may be reproduced, republished, uploaded, posted, modified, reused,
  transmitted, reproduced, distributed, copied, publicly displayed, linked to, or otherwise used except as provided
  in these Terms of Use without the written permission of the Operator." Basketball Content includes "statistics".
- "you may download material displayed on the Services to any single computer only for your personal,
  noncommercial use".
- No clause found on scraping, robots or automated access (searched for scrap, robot, spider, automated, crawl,
  harvest).

### A public mirror that needs no browser

github.com/shufinskiy/nba_data (GitHub API: license `Apache-2.0`, last push 2026-11-06T07:40:57Z) holds
`datasets/shotdetail_2025.tar.xz` (2,825,192 bytes; last commit to it 2026-04-25, "upd nba data: full regular
season 2025/26"), `shotdetail_po_2025.tar.xz` and `cdnnba_2025.tar.xz`. Its README: "Dataset contains play-by-play
data from three sources: stats.nba.com, data.nba.com and pbpstats.com and also shots details." Downloaded to
`_p8-data/nba_shots/`:

- `shotdetail_2025.csv`: 219,160 rows, 1,230 games, 2025-10-21 to 2026-04-12; columns include `GAME_ID`,
  `GAME_EVENT_ID`, `PLAYER_ID`, `PERIOD`, `MINUTES_REMAINING`, `SECONDS_REMAINING`, `SHOT_DISTANCE`, `LOC_X`, `LOC_Y`,
  `SHOT_MADE_FLAG`, `HTM`, `VTM`. `LOC_X` spans -250 to 250 and `LOC_Y` -51 to 734. On every row,
  sqrt(LOC_X^2 + LOC_Y^2) / 10 is within 1 ft of `SHOT_DISTANCE` (median gap 0.48 ft, 99th percentile 0.985 ft), so
  the units are tenths of a foot from the basket.
- `shotdetail_po_2025.csv`: 14,473 rows, 85 games, 2026-04-18 to 2026-06-13.
- `cdnnba_2025.csv`: 708,268 actions, 1,230 games, 219,158 field goals all with `x`, `y` (0 to 100, a percentage of
  the court) and `timeActual` (UTC wall time of each action, for example 2025-10-21T23:43:24.700Z).

The mirror's Apache-2.0 license covers the repository; the rows are NBA Statistics, so the NBA terms above still
govern their use. Using them for a private, non-commercial measurement with attribution to NBA.com fits clause
(ii); whether a public portfolio site may show results derived from them, given (ii) "private" and (vi) "archived
play-by-play account or depiction", is for Lucien to decide, not settled here.

## 4. What a Level 4 check could measure

1. ACB, against SkillCorner (out of distribution: another league, FIBA court and rules, ten years after 2015-16).
   On footage that passes an identity check, run the pipeline and compare, frame by frame after a clock alignment:
   player position error on frames where SkillCorner marks the player detected (`isDetected` 1, mean predError
   0.84 to 0.91 ft), with SkillCorner's own predError beside it; team assignment; jersey identity against
   SkillCorner's `jersey` and `playerId`; shooter position at release against `shots.location`. Ball checks are
   weak: the ball detection flag is always 0 and SkillCorner warns its ball calibration is weaker. Needed first: a
   FIBA court model in the pipeline (court size and markings from the FIBA rules: section 7) and a
   clock alignment of the video to SkillCorner's `gameClock` and `period` (the `wallClock` origin is SkillCorner's
   video, not ours).
   - Full game: only 179612 has a candidate (720p).
   - Sampled possessions: the 10 ACB highlight videos (184 to 253 s, 1080p) give short clips from every game;
     per TRACKING_LEVELS.md's Level 4 gate, claims would be limited to the sampled possessions.
2. NBA, against public shot coordinates. For each field goal of a 2025-26 game, compare where the pipeline puts the
   shooter at the release with (`LOC_X`, `LOC_Y`), and check the play-by-play facts (shooter, made or missed, period
   and clock). Shot coordinates exist for the whole 2025-26 season; what is missing is 2025-26 NBA broadcast
   footage, which this task did not search for (League Pass is excluded by D128).

## 5. Go or no-go

- SkillCorner as Level 4 truth: **go**, as partial truth, for one full game (179612) once an automatic identity
  check passes, and for highlight clips of all 10 games as sampled possessions. Blocked on decisions, not data:
  an identity rule for ACB footage (D130 is written for SportVU games), downloading a 2-hour video (D128 asked for
  approval beyond identity sections; D137 allows downloads as each step needs), and the FIBA court model.
- NBA shot coordinates as Level 4 truth: **go** for a private, non-commercial measurement from the public mirror,
  with attribution to NBA.com; **no-go** for direct requests to stats.nba.com or cdn.nba.com without a browser
  (timeout and 403). Publishing results derived from them needs Lucien's reading of clauses (ii) and (vi). The check
  itself waits on 2025-26 NBA footage, not yet searched.
- Overall: a Level 4 check is feasible on ACB first (truth and one full video exist now); the NBA check has truth
  but no footage.

## 6. ACB identity check (second pass)

Rule (the lead's adaptation of D130 and D135, fixed before the run): three 8-second sections at 25, 50 and 75
percent of the video, one retry 90 s later per moment, frames at 1 per second, easyocr on the bottom quarter and the
top sixth; a frame matches a game when a period, a game clock and both scores equal SkillCorner's state at that
clock (2 s slack, either team order); a moment needs 2 matching frames; confirmed only when all 3 moments match one
SkillCorner game and no other of the 10. Code: `tools/p8_acb_identity.py`, which imports tools.p8_identity's
download, frame, OCR, parsing, matching and verdict functions. SkillCorner's state is built from its `shots` (made,
2 or 3 points) and `free_throws` (made, 1 point) by `offTeamId`; the rebuilt final score equals the official score
in 10 of 10 games (`python3 -m tools.p8_acb_identity scores`: 78 to 112 scoring events per game).

One change to reading, not to the rule. The ACB scorebug prints the game clock and the shot clock together, and
easyocr read it as "09.40/24", "09.40124" or "08.4524"; tools.p8_identity's clock forms read none of these. A second
reading (`parse_acb`: minutes up to 10, FIBA Article 8.1, then seconds, then up to 3 characters of shot clock) was
written after the first run on 00k9_ueP2bY showed this, so it is not evidence by itself; every record keeps both
readings ("unchanged" and "acb_clock") and the verdict is reported for each. Records: `acb_identity.json` in each
video's identity folder of the P8 workspace.

| Video | Moment | Sections (s) | Frames with a clock, unchanged / acb_clock | Games matched, unchanged | Games matched, acb_clock |
|---|---|---|---|---|---|
| 00k9_ueP2bY (179612 candidate, 7,454 s) | 25% | 1863 | 0 / 8 | none | 179612 |
| | 50% | 3727, 3817 | 0 / 0, 0 / 0 | none | none |
| | 75% | 5590, 5680 | 0 / 0, 8 / 8 | 114099, 179612, 184439 | 114099, 179612, 184439 |
| lefFmF2gEvA (188630 stream, 8,813 s) | 25% | 2203, 2293 | 0 / 0, 0 / 0 | none | none |
| | 50% | 4406, 4496 | 6 / 6, 3 / 3 | none | none |
| | 75% | 6609, 6699 | 1 / 1, 8 / 8 | none | none |

- 00k9_ueP2bY: **rejected** under both readings. At 25 percent the OCR read "Unicaja 23 27 Valencia Basket 2
  09.40/24": SkillCorner's score for 179612 in period 2 at 9:40 is home 23, away 27, and only 179612 matched under
  the acb_clock reading. The 50 percent moment fell in half-time: neither section nor its retry 90 s later shows a
  scorebug (the OCR text is advertising). FIBA's half-time interval is 15 minutes ("There shall be a half-time
  interval of play of 15 minutes", Official Basketball Rules 2026 v1.1, Article 8.4), so a single 90 s retry cannot
  leave it. At 75 percent the OCR read "Unicaja 68 76 Valencia Basket 4 09.16 24" (SkillCorner, period 4 at 9:16:
  home 68, away 76), but small numbers on screen (the period digit, team fouls) also matched 114099 and 184439 at
  other periods. The rule as written cannot confirm a FIBA broadcast whose 50 percent point falls in half-time; a
  half-time retry rule is the lead's decision.
- lefFmF2gEvA: **rejected** under both readings; every frame's OCR is live-chat text and stream counters, and the
  clocks read are chat timestamps. One frame (6609 s, first frame) was looked at to answer whether the stream holds
  the broadcast picture, not to decide identity: it shows two webcam windows, a chat panel and a stream scoreboard
  ("BAR 61-41 DGC", "3C 02:56"), and no broadcast picture.

## 7. FIBA court against the pipeline's court model

Source: FIBA Official Basketball Rules 2026, v1.1, "Valid as of 1st October 2026" (assets.fiba.basketball, fetched
2026-11-22). The 2024 edition that governed the 2025-26 season was not found at FIBA's site (its old URL returns
404). FIBA's own list of 2026 changes ("THE FIBA OFFICIAL BASKETBALL RULES CHANGES, VALID AS OF 1 OCTOBER 2026",
v1.1) changes Article 2 only in line colours ("Increased flexibility in line colours"), so the dimensions below are
read as unchanged from 2024; that reading rests on the changes list, not on the 2024 text.

Quoted from Article 2: "dimensions of 28 m in length by 15 m in width measured from the inner edge of the boundary
line"; the centre circle "radius of 1.80 m"; the free-throw line "furthest edge 5.80 m from the inner edge of the
endline and shall be 3.60 m long"; restricted-area lines "their outer edges being 2.45 m from the mid-point of the
endlines"; the 3-point line "2 parallel lines ... with the outer edge 0.90 m from the inner edge of the sidelines"
and "An arc of radius 6.75 m measured from the point on the court beneath the exact centre of the opponents' basket
... The distance of the point on the court from the inner edge of the mid-point of the endline is 1.575 m"; the
no-charge semi-circle "radius of 1.30 m". Article 8.1: "4 quarters of 10 minutes each".

| Quantity | FIBA (m) | FIBA (ft) | NBA in `court/geometry.py` and `court/court33.py` (ft) |
|---|---|---|---|
| Court length by width | 28 by 15 | 91.86 by 49.21 | 94 by 50 |
| Basket centre from endline | 1.575 | 5.17 | 5.25 |
| 3-point arc radius | 6.75 | 22.15 | 23.75 |
| Corner 3-point line from sideline | 0.90 | 2.95 | 3 (corner line at 22 ft from the axis) |
| Arc meets corner line, from endline | 2.99 (computed) | 9.81 | 13.92 (court33 vertices 7, 8, 24, 25) |
| Top of the arc from endline | 8.325 (computed) | 27.31 | 29.01 (court33 vertices 13, 19) |
| Free-throw line from endline | 5.80 | 19.03 | 19 |
| Restricted area (lane) width | 4.90 | 16.08 | 16 |
| Centre circle radius | 1.80 | 5.91 | 6 |

SkillCorner's coordinates follow the FIBA court: in game 179612, putting the basket at x = -40.76 ft (half of 28 m
less 1.575 m) gives a median gap of 0.011 ft (90th percentile 0.014 ft, 143 shots) between the shooter's distance
to it and the shot's `distance`; at the NBA's -41.75 ft the median gap is 0.748 ft. Detected players' |x| has a 99th
percentile of 44.85 ft (FIBA half length 45.93 ft).

What the pipeline's court model would need (nothing changed here):

1. A FIBA vertex table beside `court/court33.py`'s `COURT_VERTICES_33` (NBA feet, from roboflow/sports' NBA court
   configuration), in the same index order. The source (github.com/roboflow/sports, MIT; branch `feat/basketball`,
   commit 415751384e34, `sports/basketball/config.py`) builds the 33 vertices from one preset per league and has a
   `League.FIBA` preset beside `League.NBA`. It shows that vertices 12, 14, 18 and 20 are the throw-in line marks
   on the sidelines (`baseline_to_throw_line`: 835 cm NBA, 830 cm FIBA; Article 2.5.6 says 8.325 m). Its FIBA
   preset agrees with Article 2 on court size (2800 by 1500 cm), arc radius (675), corner distance (90), lane (490
   by 580), centre circle (180) and basket (157), but puts the 3-point straight ends (vertices 7, 8, 24, 25) at
   330 cm from the endline, where Article 2.5.4's arc and lines meet at 2.99 m (computed above). A FIBA table should
   take those four vertices from Article 2, not from the preset.
2. `COURT_LENGTH_FT` and `COURT_WIDTH_FT` (94 and 50) wherever they bound or draw positions (for example
   `build_trajectories.py` imports them), as 91.86 and 49.21 for ACB games.
3. The court keypoint detector was trained on NBA courts; whether it finds the same 33 points on ACB floors (the
   lines have the same pattern, at other distances, and ACB floors carry other paint and advertising) is untested.
4. A transform from the pipeline's corner-origin feet to SkillCorner's centre-origin feet (x minus 45.93, y minus
   24.61), and SkillCorner's convention that the offensive hoop of an event is at negative x.

## 8. 2025-26 NBA broadcasts (metadata only)

Code: `tools/p8_nba2526_search.py`; output in the P8 workspace (`level4/nba2526/`). Games: the 1,230 regular-season
and 85 playoff games of the shot-chart mirror. Classes and matching rules are in the tool's docstring, fixed before
any listing was read, with two later additions stated there: a channel with any 2K or simulation title is rejected
whole (one channel posted NBA 2K26 games as "NBA LIVE: ... | NBA Regular Season", some titles without "2K"), and
one-quarter uploads became their own class after a channel listing showed games posted a quarter at a time.

Sources read: full upload lists of 8 YouTube channels (the league's channel, newest 6,000 uploads, the oldest a
game of 2025-03-23; 7 others found by search, 50 to 3,000 uploads each, the 3,000 a cap); YouTube searches for a seeded random sample of 200
games (186 regular season, 14 playoff); 5 Bilibili searches (1 failed with HTTP 412). 1,614 videos named one game
and had a class.

| Class | Regular season games (of 1,230) | Playoff games (of 85) | Source |
|---|---|---|---|
| Full game, any height | 12 | 14 | 6 YouTube channels and 2 Bilibili uploaders; 18 of the games from one channel |
| Full game, probed at 720p or more | 12 | 13 | every full-game candidate probed; the 2 Bilibili items failed to probe |
| League "EXTENDED ... FULL GAME HIGHLIGHTS" | 162 | 78 | the league's channel; 20 sampled: all 1080p, 1,459 to 3,825 s |
| League "FULL GAME HIGHLIGHTS" | 1,207 | 80 | the league's channel |
| One quarter uploaded alone | 1 | 15 | 2 YouTube channels; 8 playoff games with all 4 quarter labels |

Full games per team (regular season, usable): LAL 4; GSW, MIA, NOP, TOR 2; ATL, BOS, CHA, CLE, DAL, MIN, NYK,
OKC, ORL, SAC, UTA, WAS 1; the other 13 teams 0. Playoffs (usable): HOU and LAL 3; ATL, BOS, DET, NYK, ORL, PHI 2;
CLE, DEN, MIN, OKC, PHX, POR, SAS, TOR 1. League extended highlights per team range from 2 (WAS) to 20 (NYK) in the
regular season.

In the 200-game search sample, 6 games had a full-game upload, 40 an extended one and 196 the league's highlights.
The full-game uploads are third-party; 20 of the 25 probed run 2,424 to 4,317 s, shorter than a broadcast with
breaks, so many are cut; none is identity-checked. An NBA identity check would use the mirror's `cdnnba_2025` play-by-play
(`scoreHome`, `scoreAway`, `clock`, `period` on every action) in place of SkillCorner's state.

## 9. Go or no-go, second pass

- ACB against SkillCorner: still **go** as partial truth, but no ACB video is confirmed. 00k9_ueP2bY is rejected
  by the rule as written (half-time at its 50 percent point), although the acb_clock reading matches 179612 alone
  at 25 percent and 179612 among 3 at 75 percent. Decisions for the lead: whether the acb_clock reading stands
  (proposed check: ACB's own highlight videos, games in and out of the 10), and a half-time retry rule for FIBA
  footage. lefFmF2gEvA holds no broadcast picture and is out.
- FIBA court: the dimensions are settled at the source; the court model needs a FIBA vertex table and bounds, and
  the keypoint detector's transfer to ACB floors is untested.
- NBA 2025-26: the shot truth covers every game, but public full broadcasts exist for 25 games found at 720p or
  more (12 regular season, 13 playoff), all third-party uploads. The league's own extended highlights (240 games;
  the 20 sampled are 1080p and 24 to 64 minutes long) and highlights (1,287 games) are the broad public footage: a shot-location check
  on them is a sampled-possession check (TRACKING_LEVELS.md Level 4 gate), with shots matched by period and clock.


## 10. The acb_clock reading and the half-time retry (PLAN.md D139)

Written before any calibration video is downloaded.

Half-time retry, ACB tool only: when neither a moment's section nor its 90 s retry reads any clock (in either
reading), one more section 900 s after the first start. FIBA Official Basketball Rules 2026 v1.1, Article 8.4:
"There shall be a half-time interval of play of 15 minutes." The SportVU rule (tools.p8_identity; D130, D138) is
unchanged. The other rules stay as they are, including the 75 percent multi-match on small-number pairs: the
three-moment uniqueness is the guard.

Check of the acb_clock reading, fixed in `tools/p8_acb_identity.py` (`CALIBRATION_POSITIVES`,
`CALIBRATION_NEGATIVES`, `calibrate`) before it runs. Videos: ACB's own highlights ("RESUMEN", channel ACB), identity
sections only, each checked against all 10 SkillCorner games under the final rule above.

- Positives, the first 3 games of `matches.json` other than 179612 (the candidate under test): `KxOXoriFwDM`
  (114243), `ziPCMtLczBU` (114234), `APBbUdF7O5U` (114169). Each must be confirmed as its own game.
- Negatives, 2 Liga Endesa 2025-26 games outside the 10, each between the same two clubs as one of the 10:
  `paD9I26G7uA` (Recoletas Salud San Pablo Burgos - Bàsquet Girona, 97-79, 2025-10-04) and `rES4CvHaRcI` (Asisa
  Joventut - UCAM Murcia, 67-89, 2026-04-04). Each must confirm nothing.
- Bar: 3 of 3 positives confirmed and 0 negatives confirmed, under the acb_clock reading. If the bar is met, the
  acb_clock reading is accepted and 00k9_ueP2bY is checked again under the final rule; if not, the reading is not
  accepted and 00k9_ueP2bY's verdict stays with the unchanged reading. The unchanged reading is reported beside it
  either way.

### Results

(Added after the runs, below.)

## 11. NBA 2025-26 identity check (PLAN.md D141)

Written before any calibration video is downloaded.

Rule: tools.p8_identity's D130 rule as fixed in D138, imported unchanged (`tools/p8_nba_identity.py`), with the
mirror's play-by-play (`cdnnba_2025`, `cdnnba_po_2025`: period, clock, scoreHome and scoreAway on every action) in
place of the SportVU play-by-play. `python3 -m tools.p8_nba_identity events` (under the heavy-job lock) turned them
into 156,583 score changes for 1,290 games: all 1,230 regular-season games and 60 of the 85 playoff games (the
mirror's playoff file holds 60). Candidates for a video: every 2025-26 game between its two teams; a game without
play-by-play stays a candidate that can never match, and is listed with each record.

Calibration (fixed in `CALIBRATION` before it runs): 5 of the league channel's "EXTENDED: ... FULL GAME
HIGHLIGHTS" uploads of regular-season games, drawn with `random.Random(141).sample` from the 162 matched ones sorted
by video id; identity is the league's own title.

| Video | Titled game | Date | Games between the two teams in 2025-26 |
|---|---|---|---|
| qXCV4eissO4 | 0022500332, HEAT at MAVERICKS | 2025-12-03 | 2 |
| 1krUPLClXYA | 0022500714, TIMBERWOLVES at GRIZZLIES | 2026-02-02 | 4 |
| tXKAuBt2vjw | 0022500300, MAVERICKS at CLIPPERS | 2025-11-29 | 4 |
| nb402apDzh8 | 0022500881, NUGGETS at JAZZ | 2026-03-02 | 4 |
| GDpD5lL9c20 | 0022500227, HAWKS at JAZZ | 2025-11-13 | 2 |

Bar: 5 of 5 confirmed as the titled game against every game of the same two teams, and 0 confirmed as another game.
If the bar is met, the 25 probed full-game candidates (section 8, YouTube, 720p or more) are checked, identity
sections only; every candidate's play-by-play exists for its titled game. If it is not met, no candidate is checked
and the failure is reported.

### Results

(Added after the runs, below.)
