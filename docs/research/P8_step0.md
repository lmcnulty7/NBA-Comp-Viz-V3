# P8 step 0, second pass: feasibility of more synced footage and of labels without it

Started 2026-11-21 under PLAN.md D128 to D136. An addendum to `docs/research/P8_FEASIBILITY.md` (the first Step 0),
which stays the reference for everything this file does not change. Plan and measure only: no training, no model
change, no download beyond the identity sections.

Status: in progress. This version fixes the rules and bars below before any search candidate is identity-checked;
results are added under the sections marked "Results" when they exist.

## Constraints (Lucien, D128 to D132)

- No hand labelling of any kind. Existing human labels (the 50 box-truth files, the 2026-09 audit of 300 labels) may
  qualify teachers and check the scorecard; no new ones are made (D132).
- No League Pass recording: its terms forbid copying and circumventing access controls.
- The sportvu/splits.json held-out game and arena stay spent (D71). P8's single held-out score, at its end, is on the
  D85 broadcast test set (4 games in Miami, Toronto and Los Angeles); any game in those arenas joins the test side
  and never training (D129).
- Every new source records its license in `docs/DATA_SOURCES.md`; non-commercial licenses are acceptable.

## Rules fixed before any candidate is checked

### Track A: the search (tools/p8_search.py)

- Games: every game of the SportVU contract manifest (626: the 636 archives less the 10 misnamed 2015-12-05 games,
  which are the sealed Set A). Classes, by the existing guards: training (526), D85 test (78: every home game of MIA,
  TOR, LAC and LAL), spent held-out (17: the CLE home games and 12.16.2015.PHX.at.GSW), broken archive (5).
- Sources and queries: YouTube with the 3 frozen R1.1 nickname forms, a city-name form, the title form of the channel
  that holds the known full games, and 2 Chinese forms; then Bilibili with the first Chinese form (D136).
- Rules: the frozen rejection rules (video-game simulations, another date or year, postseason) and their Chinese
  forms; full broadcasts at least 2,400 s (the frozen minimum); condensed games (D134) at least 480 s with
  "condensed" in the title, as their own class; both teams named in the title or description; at most 720p
  avc1 (YouTube) or at most 720p (Bilibili), and at least 720p to count.
- D85 test and spent games are searched by metadata only and counted; no request reaches their videos.

### Track A: the identity check (tools/p8_identity.py, D130, D135)

- Three 8-second sections at 25, 50 and 75 percent of the video, one retry 90 s later per moment; frames at 1 per
  second; easyocr on the bottom quarter and the top sixth.
- A frame matches a game when a period, a game clock and two scores read in it equal the play-by-play score at that
  clock (tools.p8_scorebug.matches, 2 s slack), in either team order. A moment matches when at least 2 of its frames
  do. Confirmed when exactly one candidate game matches at all 3 moments; ambiguous when more than one; rejected
  otherwise. No person reads anything.
- Candidates for a video: every training-class game between its two teams in the SportVU window.

**Calibration bar, fixed here before the full run:** the 7 videos of known identity (the 4 R1.3 training games and
the 3 confirmed channel games, tools.p8_common.dev_games) must all be confirmed as their true game against their own
matchup's candidates, and none may be confirmed against another video's candidate list: 7 of 7 confirmed, 0 false
confirmations. If the bar fails, no search candidate counts as a game until the rule is fixed and the bar met again
on the same 7, and the failure is reported here.

### Go or no-go (fixed here before results)

Go for Phase B work if either holds; otherwise no-go, naming what is missing:

- Track A: at least 8 identity-confirmed full broadcasts usable for training (training class, at least 720p), from at
  least 4 arenas not already among the development arenas (GSW, NYK, OKC, POR).
- Track B: at least one outside teacher that beats our detector on the existing human box truth (recall at matched
  precision), and a baseline of the scorecard rows that can be measured on training games.

## Results

Not yet. Sections to follow: Track A yield (per team, per source, full and condensed, hit rate on the 84 earlier
games against 6, training against D85 test and spent), identity calibration, Track B sources and licenses, teacher
qualification design, pseudo-label rule (D131), the scoring protocol on training games, targets for every row, GPU
and Mac hours per experiment, ranked experiments, go or no-go.
