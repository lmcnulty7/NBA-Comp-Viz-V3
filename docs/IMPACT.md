# What this project measurably did

A one-page summary for a hiring manager. Every number below is copied from the committed file named after it, and
`tests/test_impact_doc.py` checks that each one is there. A value with a range in brackets is its interval from a
bootstrap; a value with no range is a point estimate. Nothing here counts visitors: the site has no visitor
measurement.

## What was built

- Data: the 2015-16 SportVU tracking season, with 604 games to train on and 17 games held out before any model was
  built. Source: `reports/results.md`
- Models: eleven model lanes, from shot quality to win chances, each frozen and scored once on the held-out games.
  Failed and inconclusive lanes stay in the results. Source: `reports/results.md`
- A broadcast video tracker, scored against the sensor data on a held-out game. For the players it finds, the
  middle position error is 1.92 ft and nine in ten are within 4.98 ft (point estimates). It misses 25.1% of the
  players in view and gets at most 81.9% of team labels right, so it stays out of the models. Source:
  `reports/scorecard/gsw_phx_2016__r23.txt`
- A site in two copies. The public copy has 21,176 files and no tracking frames; the private copy sits behind
  sign-in. It covers 3,162 possessions. Source: `PLAN.md`
- Tests: on the deployed code, 1624 Python tests passed, 552 browser tests passed on the public copy and 633 on the
  private copy. Source: `PLAN.md`

## Results that held up on sealed test sets

Each new method was set up in writing before it ran, then scored once on games set apart and hashed in advance. A
ledger allows one score per experiment. Source: `reports/research/scoring_ledger.json`

- Lineups (lineups-E1): passed. Player ratings predicted each game's margin better than raw lineup plus-minus on the
  test season of 1300 games: RMSE difference -1.15 [-1.59, -0.77] points, R squared over home court alone 0.098
  [0.065, 0.132]. The claim is narrow: game margin, given who played. Source: `reports/research/lineups-E1.md`
- Lineups, second test (lineups-E2): fails as set up. On 1095 games of the test season, lineup ratings shrunk
  toward their players showed no detectable gain over the frozen ratings that already pool players: d -0.0005
  [-0.0015, +0.0002], far from the -0.01 bar. Beside it, not as the headline: the rule holds against the raw
  baseline, which is weak by construction (d -0.3204 [-0.3607, -0.2619]). Source: `reports/research/lineups-E2.md`
- Player style (embeddings-E1): passes the one-game test. Built from a single held-out game, the frozen style map
  finds the same player among its 5 nearest neighbours 0.474 [0.384, 0.558] of the time (171 players; chance
  0.0125). The lower end is above the bar of 0.10 and above 0.038, the top of a retriever that knows only position.
  The claim is narrow: it shows the map finds the same player, not what the map means. The 19 raw features without
  the map retrieve better, 0.561 [0.491, 0.631]. Source: `reports/research/embeddings-E1.md`
- Win chances (winprob-E1): passed on both stages. A pregame strength term changed mean bucket Brier by -0.0124
  [-0.0161, -0.0086] on the test season (lower is better), with no time bucket worse. It is kept as a research result; the site did
  not change. Source: `reports/research/winprob-E1.md`
- Shot quality (xfg-E1): not adopted. It met its bar on the training games, -0.00052 [-0.00081, -0.00024], but
  not on the test games, +0.00067 [-0.00133, +0.00229] against a bar of +0.0005. With ten test games this reads as
  inconclusive. Source: `reports/research/xfg-E1.md`
- Win chances, second test (winprob-E2): not adopted on the training seasons. A forward temperature did not
  lower mean bucket ECE (difference 0.0038 [-0.0021, 0.0068]), since the fitted temperature swung from 0.894 to
  0.955 to 1.046 across seasons. Its test season was not scored, so its one test score was kept unused. Source:
  `reports/research/winprob-E2.md`
- Decision quality (decisions-E1): not adopted. The target proved reliable (r 0.420 [0.295, 0.519]), so the lane's
  earlier failure counts against the metric. Kind of decision explains about a third of the player spread (R
  squared 0.342 [0.258, 0.434]). Source: `reports/research/decisions-E1.md`

## Honesty work

- Failed results stay visible. The decisions lane failed its check, -0.014 [-0.116, 0.100] on the held-out games,
  and it is shown as failed. Source: `reports/results.md`
- Bad name joins in the frozen code dropped 1032 training misses by name. The site now withholds 36 rebound rates
  that would otherwise show, rather than show values those joins distort. Source: `reports/export/name_joins.json`
- One shot-making value was withheld for the same reason, and shown rebound rates went from 346 to 322. Source:
  `PLAN.md`
- Machine-written texts must pass a checker that reads every number. Of 3,162 possession texts, 2,743 passed and
  ship. The scouting texts are held back after a hand read found errors the checker missed in 3 of 15. Source:
  `PLAN.md`

## The machine-written text lane

- genai-E1 put a fixed template with no model against the Haiku model on the same checker. The template passed 81
  of 81 test players and 480 of 480 possessions; Haiku passed 0.840 [0.753, 0.914]. Haiku wrote with far more
  variety: Haiku minus template in sentence frame variety is [0.331, 0.439]. So the checker alone cannot show what a model adds. A
  stricter checker turned the order round: 36 of 81 template texts against 65 of 81 for Haiku. Source:
  `reports/research/genai-E1.md`
- genai-E2 tuned a small open model (Qwen2.5 0.5B) on this laptop and scored it once on the same test items.
  Training lifted its pass rate on scouting from 0.111 to 0.852, a gain of +0.741 [0.642, 0.840], and on
  possessions from 0.665 to 1.000, a gain of +0.335 [0.296, 0.375], so H1 passes. Against Haiku the test is
  inconclusive: 0.852 against 0.802, with a range of [-0.062, 0.160] that does not rule out a real loss. Read
  beside that pass rate: on possessions the model copies the template (0.992 of its texts are the same), and on
  scouting its sentence frame variety is 0.106 against Haiku's 0.612. It costs no API spend and 0.76 s per text on
  the Mac. Rejection sampling and constrained decoding were both inconclusive. Source:
  `reports/research/genai-E2.md`
- The larger 1.5B model, tuned the same way and scored once, gets its own reading; no test compares the two sizes.
  Training lifted its scouting pass rate from 0.012 to 0.914, a gain of +0.901 [0.840, 0.963], so H1 passes.
  Against Haiku it reads non-inferior: 0.914 against 0.802, a difference of +0.111 [0.012, 0.210]. Every
  possession text it writes is the same as the template's, and its scouting frame variety is 0.106 against
  Haiku's 0.612, as with the smaller model. Rejection sampling hurt it: -0.272 [-0.370, -0.173]. It costs no API
  spend and 1.20 s per text on the Mac with one attempt. Source: `reports/research/genai-E2-1.5B.md`

## Engineering

- The site polish loop proposed 5 changes, scored each by machine checks only, and kept 1. The kept change cut the
  copy's reading grade from 4.062 to 3.791. Sources: `PLAN.md`, `docs/POLISH_LOOP.md`
- The loop's scorer was fixed before its runs counted. Before the fix, it charged the private home page 12.9 MB
  where 1.4 MB arrived. Source: `docs/POLISH_LOOP.md`
- One data contract joins the Python side and the site. Every number on the methods page is checked against its
  data file, and every data file against the contract, before a build can pass. Sources:
  `web/scripts/check-methods.mjs`, `export/schemas.py`

## What is not measured yet

- Visitors. The site counts no one. A free Cloudflare Web Analytics beacon, which sets no cookie, would count them.
  It needs Lucien's yes.
- The reader test: a real person reading the site, planned for the interview stage (P7).
- The hand read of the scouting texts, which waits for the next prompt and Lucien's yes.
