# Research backlog

Written 2026-11-16 (PLAN.md D70). Every pre-registered proposal of the lane dossiers, ranked by expected gain over
effort (both scored 1 to 5 in the dossier, with reasons there). Ties go to the higher gain, then to the proposal whose
test data is ready. Gain and effort are judgements written before any run; the wall times in the dossiers are
estimates. Test sets are those of PLAN.md D77 and D79: Set A (the 10 untouched 2015-16 games) for tracking lanes,
Set B (10 Liga ACB games) as an out-of-distribution check, and the 2025-26 NBA season for the play-by-play lanes.

| Rank | Id | Lane | What it tests | Gain | Effort | Ratio | Test data | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | decisions-E1 | decisions | is the target reliable, and is the metric more than decision mix | 3 | 1 | 3.0 | training tables, Set A | done: not adopted (H1 fails as pre-registered, H2 holds, H3 fails; reports/research/decisions-E1.md) |
| 2 | genai-E1 | genai | a template writer against the Haiku texts on the same checker | 3 | 1 | 3.0 | genai test players and training-game possessions (reports/research/genai-splits.json) | done: H1 and H2 hold (pinned checker: template 81 of 81 and 480 of 480, Haiku 0.840 [0.753, 0.914] snapshot, 0.963 [0.914, 1.000] cache arm; Haiku minus template frame variety [0.331, 0.439]; sensitivity checker: template 36 of 81, Haiku 65 of 81; reports/research/genai-E1.md) |
| 3 | load-E1 | load | a shrunk player history for decline | 3 | 1 | 3.0 | Set A (about 220 player-games, under its own 1000-row power floor) | proposed |
| 4 | lineups-E1 | lineups | score RAPM on every game's margin (Sill's construction) | 4 | 2 | 2.0 | 2025-26 season | done: passes (game margin, given the stints: RAPM minus plus_minus RMSE -1.15 [-1.59, -0.77], R squared vs home court 0.098 [0.065, 0.132]; lineup ranks stay inconclusive; possible data fault: stint margin equals the schedule in 1298 of 1300 games; reports/research/lineups-E1.md) |
| 5 | winprob-E1 | winprob | pregame strength from Elo as Stern's drift term, no odds | 4 | 2 | 2.0 | 2025-26 season | done: passes (mean bucket Brier, Elo drift term minus frozen logistic -0.0124 [-0.0161, -0.0086] on 1300 games of 2025-26, no bucket worse, overtime included; drift form beyond Elo alone -0.0005 [-0.0011, -0.0001]; early calibration is a point estimate, so any site use waits for winprob-E2 or a D line; reports/research/winprob-E1.md) |
| 6 | rebounds-E1 | rebounds | a temperature and an offset per candidate type | 2 | 1 | 2.0 | Set A | proposed |
| 7 | winprob-E2 | winprob | a forward-in-time temperature for the over-confidence | 2 | 1 | 2.0 | 2025-26 season | done: not adopted on development (E1 plus a forward temperature minus E1, mean bucket ECE 0.0038 [-0.0021, 0.0068] on 2605 development games; the forward temperature is 0.894, 0.955, 1.046 across 2021-22 to 2023-24, so the miscalibration does not carry forward as one temperature; the test season was not scored; reports/research/winprob-E2.md) |
| 8 | xfg-E1 | xfg | cross Venn-Abers in place of the inner isotonic step | 3 | 2 | 1.5 | Set A | done: not adopted (passed on development games, inconclusive on the 10-game test set; reports/research/xfg-E1.md) |
| 9 | ghosting-E1 | ghosting | rollout training of the same network | 3 | 2 | 1.5 | Set A | proposed |
| 10 | gravity-E1 | gravity | does gravity open shots for teammates | 3 | 2 | 1.5 | Set A | proposed |
| 11 | tracking-E1 | tracking | team labels from tracklet embeddings | 3 | 2 | 1.5 | needs synced video beyond the 4 development games (docs/research/P8_FEASIBILITY.md) | proposed |
| 12 | matchups-E1 | matchups | split the outcome into shot frequency and efficiency (Franks et al.) | 4 | 3 | 1.33 | Set A | proposed |
| 13 | embeddings-E1 | embeddings | a first held-out style score by self-retrieval | 4 | 3 | 1.33 | Set A | proposed |
| 14 | genai-E2 | genai | a small open model (Qwen2.5 0.5B, 1.5B), LoRA on the Mac: few-shot, distillation, rejection sampling against the checker, constrained decoding | 4 | 4 | 1.0 | genai test players and training-game possessions | done: H1 passes on both kinds at both sizes (A1 minus few-shot, scouting +0.741 [0.642, 0.840] at 0.5B and +0.901 [0.840, 0.963] at 1.5B); H2 inconclusive at 0.5B and hurts on scouting at 1.5B (-0.272 [-0.370, -0.173]); H3 inconclusive at 0.5B (+0.049 [-0.062, 0.160]) and non-inferior at 1.5B (+0.111 [0.012, 0.210]); H4 inconclusive; possession texts copy the template; reports/research/genai-E2.md and reports/research/genai-E2-1.5B.md |
| 15 | decisions-E2 | decisions | one value scale for every option, validated on decisions made | 4 | 4 | 1.0 | Set A | proposed |
| 16 | tracking-E2 | tracking | broadcast-only fill of missed player-frames (learned imputation) | 4 | 4 | 1.0 | SportVU training games, Set A | proposed |
| 17 | lineups-E2 | lineups | lineup ratings shrunk toward their players (L-RAPM) | 3 | 3 | 1.0 | 2025-26 season | proposed |
| 18 | rebounds-E2 | rebounds | train on the whole miss (within-miss softmax) | 3 | 3 | 1.0 | Set A | proposed |
| 19 | load-E2 | load | injury-labelled absence on the play-by-play seasons | 3 | 3 | 1.0 | 2025-26 season, if the injuries table covers it | proposed |
| 20 | epv-E1 | epv | train on expected shot value instead of the make or miss | 2 | 2 | 1.0 | Set A | proposed |
| 21 | embeddings-E2 | embeddings | a spatial shot basis by NMF (Miller et al.) | 2 | 2 | 1.0 | Set A | proposed |
| 22 | ghosting-E2 | ghosting | attention over 10 players and the ball, 4 s of history | 3 | 4 | 0.75 | Set A | proposed |
| 23 | xfg-E2 | xfg | second defender and contest geometry | 2 | 3 | 0.67 | Set A | proposed |
| 24 | matchups-E2 | matchups | hidden Markov assignment with fitted weights (runs only if matchups-E1(a) passes) | 2 | 3 | 0.67 | Set A | proposed |
| 25 | gravity-E2 | gravity | an expectation that knows where the teammates are | 2 | 3 | 0.67 | Set A | proposed |
| 26 | epv-E2 | epv | multi-entity attention over raw positions (runs after epv-E1) | 3 | 5 | 0.6 | Set A | proposed |

genai-E1 and genai-E2 were added on 2026-11-17 (PLAN.md D92, docs/research/genai.md), after the top three below
were argued; the ranks move by one or two and the argument below is unchanged (load-E1, now third by ratio).

Backlog lines in docs/research/tracking.md that are not yet proposals (detector fine-tune on SportsMOT, ball detection
from SportVU-projected labels, vision-language jersey reading, weak identity labels from play-by-play) are ranked by
P8 Step 0 (docs/research/P8_FEASIBILITY.md).

## The top three, argued

The ratio alone would put load-E1 second. It drops below the next two because its own pre-registration calls a test
set under about 1000 rows underpowered, and Set A gives about 220 player-games: it would end "underpowered" whatever
the model does. The pilot (xfg-E1) has run (not adopted) and is not counted.

1. **decisions-E1 (gain 3, effort 1).** The decisions lane is the one that failed outright. The dossier found that only
   the hold option is valued with EPV, and that the gap between kinds of decision (held-out mean regret: shot -0.159,
   pass -0.376, drive -0.528) is wider than the spread of the 25 ranked players (-0.256 to -0.329), so the player
   table may rank how often a player drives, not how well they choose. This experiment costs descriptive statistics on
   existing tables and settles two things before any larger work: whether the target (per-game on-court rating) is
   reliable enough to validate against (Franks et al. 2016), and whether the site may show the player table at all.
   It decides whether decisions-E2 (effort 4) is worth running.
2. **winprob-E1 (gain 4, effort 2).** The literature measures this gain directly: Maddox et al. 2022 report per-second
   Brier 0.1663 without a pregame prior and 0.1568 with one (2018-19), about 0.01, and our model has no pregame term
   (our overall Brier 0.1645 out of fold, roughly comparable). The drift term follows Stern 1994, whose root-time
   feature the frozen model already uses, and needs no odds. Its test is forward in time on a season no model has
   seen.
3. **lineups-E1 (gain 4, effort 2).** The lane is inconclusive because it was scored on about 20 recurring lineups a
   season. Scoring the frozen ratings on every test game's margin, as Sill 2010 did, uses every game and gives a
   published metric (game-margin RMSE) to stand next to. It shares the 2025-26 fetch with winprob-E1, so the two
   cost one data pull (about 2,500 basketball-reference pages, about 2.5 hours under the rate limit, PLAN.md D78).

Costs and expected gains of the top three, from the dossiers: decisions-E1, under an hour of CPU on existing tables
(estimate); winprob-E1 and lineups-E1, under an hour of CPU each after the shared 2.5-hour fetch (estimate). Expected
gains are the dossier scores above; no accuracy change is promised before a run.
