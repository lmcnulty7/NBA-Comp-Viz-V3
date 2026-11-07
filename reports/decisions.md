# Decision quality (analytics/decisions) (fixture output, not a result)

Written 2026-11-07T08:32:25Z by `python -m analytics.decisions.train --fixtures --report reports/decisions.md`.

**Fixture output, not a result.** Every number below comes from synthetic games (analytics/fixtures.py); the real run is the Mac command at the end.

## Inputs (fixture output, not a result)

| game | held out | frames | possessions | shots | schema problems | duplicates dropped | epv rows | xfg rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03.08.2016.GRN.at.HLX | no | 123500 | 40 | 41 | 0 | 0 | 0 | 0 |
| 12.10.2015.GRN.at.DUL | no | 126350 | 40 | 45 | 0 | 0 | 0 | 0 |
| 02.10.2016.QCY.at.FRS | no | 114850 | 40 | 37 | 0 | 0 | 0 | 0 |
| 12.16.2015.PHX.at.GSW | yes | 118000 | 40 | 41 | 0 | 0 | 0 | 0 |
| 12.01.2015.DUL.at.HLX | no | 122180 | 40 | 40 | 0 | 0 | 0 | 0 |
| 11.28.2015.HLX.at.FRS | no | 128610 | 40 | 46 | 0 | 0 | 0 | 0 |
| 12.09.2015.PEO.at.DUL | no | 115790 | 40 | 34 | 0 | 0 | 0 | 0 |
| 11.26.2015.DUL.at.CLE | yes | 121780 | 40 | 42 | 0 | 0 | 0 | 0 |

Training games: 6 (02.10.2016.QCY.at.FRS, 03.08.2016.GRN.at.HLX, 11.28.2015.HLX.at.FRS, 12.01.2015.DUL.at.HLX, 12.09.2015.PEO.at.DUL, 12.10.2015.GRN.at.DUL). Held-out games set aside before any fit: 2 (11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
Value model fit rows (localised shots of the training games): 243. Protocol: out-of-fold under 5 grouped folds by sportvu_game (seed 0).

## Event counts (training games) (fixture output, not a result)

| game | shots | passes | drives | possessions used | skipped |
| --- | --- | --- | --- | --- | --- |
| 02.10.2016.QCY.at.FRS | 36 | 58 | 50 | 39 | {'possessions_gap': 1} |
| 03.08.2016.GRN.at.HLX | 39 | 62 | 53 | 39 | {'possessions_gap': 1} |
| 11.28.2015.HLX.at.FRS | 44 | 63 | 48 | 39 | {'possessions_gap': 1} |
| 12.01.2015.DUL.at.HLX | 40 | 65 | 50 | 39 | {'possessions_gap': 1} |
| 12.09.2015.PEO.at.DUL | 33 | 69 | 51 | 39 | {'possessions_gap': 1} |
| 12.10.2015.GRN.at.DUL | 44 | 59 | 55 | 39 | {'possessions_gap': 1} |

Total: 919 decisions ({'pass': 376, 'drive': 307, 'shot': 236}).
Hold option (EPV lane) present on 0 decisions; xFG lane coverage of shot decisions 0.00.

## Value-function assumptions (fixture output, not a result)

- Shot option: points (3 at 22 ft or more, else 2) times the location-only xFG probability at the player's position, minus 0.03 per foot of nearest-defender distance below 6 ft when no contest model exists (floor 0.01).
- Pass option: points times the location-only probability at the teammate's position (no contest term: openness already carries the defender distance) times openness (1 at 6 ft or more of nearest-defender distance, linearly down to 0.5 at 0 ft) times pass risk (0.95 minus 0.01 per foot of pass length over 15 ft).
- Hold option: the EPV lane's epv_points of the last state at or before the decision frame; absent when the EPV table is absent.
- Drive option: the shot value 8 ft nearer the rim along the line to the basket (never nearer than 3 ft), with the decision-frame defender distance, times 0.90; it is in every decision's option set and is the chosen value of a drive, so every quality is regret (at most 0) and best_option is one of shot, pass_to_<rank>, hold, drive.
- Attacked basket: per (offence team, period) by majority vote over that team's possessions, each voting with the end where its offence's mean x lies; the value-model fit uses the basket nearer the shooter at the release.
- Event detection: holder within 2.5 ft of the ball under 8 ft; pass arrives within 1.5 s with no release in between; drive = the first frame at which the holder's speed toward the basket is measured above 8 ft/s in a run lasting 0.6 s.
- Upstream rows: a training game refuses EPV or xFG rows with fold == -1, a held-out game refuses rows not flagged heldout (refused counts: none).
- Leakage: option values use the decision frame and earlier frames only; the value model is out-of-fold by game; `schema.assert_time_forward` ran on every context frame; `assert_decisions_before_outcome` on every game.
- Prior of the option values: candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least 0.50. Selected: **location** (no xFG table present).

## Player table (training games, gated at 300 decisions; ranked on mean quality, by-kind means are regret at most 0) (fixture output, not a result)

No player passes the 300-decision gate: nobody is ranked.

42 of 42 players are under the 300-decision gate and carry no rank (listed unranked, by id, first 10):

| player_id | n_decisions | n_games | mean quality | shot | pass | drive |
| --- | --- | --- | --- | --- | --- | --- |
| 500030 | 40 | 3 | -0.472 | -0.413 | -0.380 | -0.676 |
| 500031 | 93 | 3 | -0.274 | -0.223 | -0.419 | -0.154 |
| 500032 | 23 | 3 | -0.326 | -0.242 | -0.361 | -0.379 |
| 500033 | 36 | 3 | -0.295 | -0.307 | -0.492 | -0.104 |
| 500034 | 29 | 3 | -0.291 | -0.172 | -0.417 | -0.178 |
| 500035 | 8 | 3 | -0.517 | -0.725 | -0.610 | -0.124 |
| 500036 | 10 | 3 | -0.288 | 0.000 | -0.692 | -0.027 |
| 500050 | 12 | 2 | -0.350 | -0.166 | -0.389 | -0.599 |
| 500051 | 57 | 2 | -0.265 | -0.320 | -0.376 | -0.137 |
| 500052 | 15 | 2 | -0.327 | -0.239 | -0.369 | -0.425 |

## Validation: mean decision quality per player-game against on-court offensive rating (fixture output, not a result)

Spearman per game between the players' mean decision quality (a sum would rank by decision count, since quality is regret) and their points per 100 offensive possessions on the floor, over players with at least 3 decisions in the game; sign test over games (two sided); bootstrap of the mean Spearman by game (n=200, seed 0); shuffled = the same statistic with mean quality permuted across those players within each game.

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location | 6 | 0.285 | 0.125 | 0.421 | 5 / 6 | 0.219 | -0.039 |

## Held-out games (valued once by the model refit on all training games; never fitted or tuned on) (fixture output, not a result)

303 decisions from 2 held-out game(s): 11.26.2015.DUL.at.CLE, 12.16.2015.PHX.at.GSW.

| kind | n | mean quality |
| --- | --- | --- |
| drive | 102 | -0.210 |
| pass | 120 | -0.441 |
| shot | 81 | -0.235 |

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location (held out) | 2 | -0.144 | -0.167 | -0.121 | 0 / 2 | 0.500 | 0.273 |

## Outputs (relative to OUT_ROOT) (fixture output, not a result)

- analytics/decisions/fixture/decision_events/02.10.2016.QCY.at.FRS.parquet
- analytics/decisions/fixture/decision_events/03.08.2016.GRN.at.HLX.parquet
- analytics/decisions/fixture/decision_events/11.28.2015.HLX.at.FRS.parquet
- analytics/decisions/fixture/decision_events/12.01.2015.DUL.at.HLX.parquet
- analytics/decisions/fixture/decision_events/12.09.2015.PEO.at.DUL.parquet
- analytics/decisions/fixture/decision_events/12.10.2015.GRN.at.DUL.parquet
- analytics/decisions/heldout/fixture/decision_events/11.26.2015.DUL.at.CLE.parquet
- analytics/decisions/heldout/fixture/decision_events/12.16.2015.PHX.at.GSW.parquet
- analytics/decisions/fixture/decision_players.parquet

## Status (fixture output, not a result)

Fixture output, not a result. The real run, on the Mac where the full data lives:

    python -m analytics.decisions.train --data "$DATA_ROOT" --out "$OUT_ROOT" --season 2015-16 --report reports/decisions.md --folds 5 --seed 0 --bootstrap 1000

What needs real data: the Phase 0 tables of every game (possessions, shot_frames, states_1hz under DATA_ROOT/analytics), the frames and play-by-play under DATA_ROOT/sportvu_contract, and, optionally, the EPV and xFG lanes' outputs under OUT_ROOT.
