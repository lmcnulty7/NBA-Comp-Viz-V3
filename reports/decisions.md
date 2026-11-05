# Decision quality (analytics/decisions) (fixture output, not a result)

Written 2026-11-05T08:37:12Z by `python -m analytics.decisions.train --fixtures --report reports/decisions.md`.

**Fixture output, not a result.** Every number below comes from synthetic games (tests/_fx_decisions.py or analytics/fixtures.py); the real run is the Mac command at the end.

## Inputs

| game | held out | frames | possessions | shots | schema problems | duplicates dropped | epv rows | xfg rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | no | 115020 | 40 | 37 | 0 | 1 | 0 | 0 |
| 01.18.2016.HLX.at.LRD | no | 123210 | 40 | 44 | 0 | 1 | 0 | 0 |
| 02.04.2016.GRN.at.QCY | no | 123230 | 40 | 37 | 0 | 1 | 0 | 0 |
| 12.16.2015.PHX.at.GSW | yes | 128550 | 40 | 45 | 0 | 1 | 0 | 0 |
| 12.15.2015.ABQ.at.GRN | no | 117110 | 40 | 35 | 0 | 1 | 0 | 0 |
| 11.27.2015.GRN.at.QCY | no | 119730 | 40 | 40 | 0 | 1 | 0 | 0 |
| 12.12.2015.RNO.at.MOB | no | 121170 | 40 | 38 | 0 | 1 | 0 | 0 |
| 11.11.2015.HLX.at.GRN | yes | 118520 | 40 | 41 | 0 | 1 | 0 | 0 |

Training games: 6 (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held-out games set aside before any fit: 2 (11.11.2015.HLX.at.GRN, 12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
Value model fit rows (localised shots of the training games): 231. Protocol: out-of-fold under 5 grouped folds by sportvu_game (seed 0).

## Event counts (training games)

| game | shots | passes | drives | possessions used | skipped |
| --- | --- | --- | --- | --- | --- |
| 01.18.2016.HLX.at.LRD | 43 | 58 | 53 | 39 | {'possessions_gap': 1} |
| 02.04.2016.GRN.at.QCY | 37 | 71 | 49 | 39 | {'possessions_gap': 1} |
| 02.07.2016.OMA.at.DUL | 37 | 59 | 55 | 39 | {'possessions_gap': 1} |
| 11.27.2015.GRN.at.QCY | 39 | 69 | 58 | 39 | {'possessions_gap': 1} |
| 12.12.2015.RNO.at.MOB | 37 | 45 | 50 | 39 | {'possessions_gap': 1} |
| 12.15.2015.ABQ.at.GRN | 34 | 61 | 58 | 39 | {'possessions_gap': 1} |

Total: 913 decisions ({'pass': 363, 'drive': 323, 'shot': 227}).
Hold option (EPV lane) present on 0 decisions; xFG lane coverage of shot decisions 0.00.

## Value-function assumptions

- Shot option: points (3 at 22 ft or more, else 2) times the location-only xFG probability at the player's position, minus 0.03 per foot of nearest-defender distance below 6 ft when no contest model exists (floor 0.01).
- Pass option: the teammate's shot value times openness (1 at 6 ft or more of nearest-defender distance, linearly down to 0.5 at 0 ft) times pass risk (0.95 minus 0.01 per foot of pass length over 15 ft).
- Hold option: the EPV lane's epv_points of the last state at or before the decision frame; absent when the EPV table is absent.
- Drive (chosen value only): the shot value at the position 1 s ahead along the decision-frame velocity, never nearer than 3 ft, times 0.90; a drive is not in the option set, so its quality can exceed 0.
- Attacked basket: the end of the court where the offence's mean x over the possession lies.
- Event detection: holder within 2.5 ft of the ball under 8 ft; pass arrives within 1.5 s with no release in between; drive at over 8 ft/s toward the basket for 0.6 s.
- Leakage: option values use the decision frame and earlier frames only; the value model is out-of-fold by game; `schema.assert_time_forward` ran on every context frame; `assert_decisions_before_outcome` on every game.
- Prior of the option values: candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least 0.50. Selected: **location** (no xFG table present).

## Player table (training games, gated at 300 decisions; insufficient rows are not ranked)

| player_id | n_decisions | n_games | mean quality | lo | hi | insufficient |
| --- | --- | --- | --- | --- | --- | --- |
| 9033 | 8 | 1 | -0.0140 | n/a | n/a | yes |
| 9032 | 2 | 1 | -0.0453 | n/a | n/a | yes |
| 9025 | 2 | 1 | -0.0559 | n/a | n/a | yes |
| 9120 | 4 | 1 | -0.0667 | n/a | n/a | yes |
| 9006 | 4 | 1 | -0.0810 | n/a | n/a | yes |
| 9045 | 2 | 1 | -0.0819 | n/a | n/a | yes |
| 9053 | 4 | 1 | -0.0825 | n/a | n/a | yes |
| 9029 | 7 | 1 | -0.0966 | n/a | n/a | yes |
| 9093 | 5 | 1 | -0.1143 | n/a | n/a | yes |
| 9030 | 11 | 1 | -0.1382 | n/a | n/a | yes |
| 9011 | 13 | 1 | -0.1404 | n/a | n/a | yes |
| 9083 | 12 | 1 | -0.1418 | n/a | n/a | yes |
| 9022 | 9 | 1 | -0.1667 | n/a | n/a | yes |
| 9042 | 3 | 1 | -0.1674 | n/a | n/a | yes |
| 9001 | 34 | 1 | -0.1685 | n/a | n/a | yes |
| 9031 | 14 | 1 | -0.1716 | n/a | n/a | yes |
| 9110 | 4 | 1 | -0.1719 | n/a | n/a | yes |
| 9008 | 28 | 1 | -0.1800 | n/a | n/a | yes |
| 9041 | 29 | 1 | -0.1826 | n/a | n/a | yes |
| 9084 | 13 | 1 | -0.1858 | n/a | n/a | yes |
| 9081 | 25 | 1 | -0.1895 | n/a | n/a | yes |
| 9028 | 24 | 1 | -0.2031 | n/a | n/a | yes |
| 9088 | 18 | 1 | -0.2038 | n/a | n/a | yes |
| 9121 | 26 | 1 | -0.2057 | n/a | n/a | yes |
| 9003 | 13 | 1 | -0.2083 | n/a | n/a | yes |

82 of 82 players are under the 300-decision gate.

## Validation: summed decision quality per player-game against on-court offensive rating

Spearman per game over players with at least 3 decisions in the game; sign test over games (two sided); bootstrap of the mean Spearman by game (n=200, seed 0); shuffled = quality permuted across the players of each game.

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location | 6 | -0.0114 | -0.2389 | 0.2118 | 3 / 6 | 1.0000 | 0.0326 |

## Held-out games (valued once by the model refit on all training games; never fitted or tuned on)

323 decisions from 2 held-out game(s): 11.11.2015.HLX.at.GRN, 12.16.2015.PHX.at.GSW.

| kind | n | mean quality |
| --- | --- | --- |
| drive | 106 | -0.1054 |
| pass | 133 | -0.3951 |
| shot | 84 | -0.1571 |

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location (held out) | 2 | 0.1445 | -0.1134 | 0.4023 | 1 / 2 | 1.0000 | 0.0057 |

## Outputs

- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/01.18.2016.HLX.at.LRD.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/02.04.2016.GRN.at.QCY.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/02.07.2016.OMA.at.DUL.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/11.27.2015.GRN.at.QCY.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/12.12.2015.RNO.at.MOB.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_events/12.15.2015.ABQ.at.GRN.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/heldout/fixture/decision_events/11.11.2015.HLX.at.GRN.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/heldout/fixture/decision_events/12.16.2015.PHX.at.GSW.parquet
- /tmp/decisions_fixture_pskkt5av/analytics/decisions/fixture/decision_players.parquet

## Status

Fixture output, not a result. The real run, on the Mac where the full data lives:

    python -m analytics.decisions.train --data "$DATA_ROOT" --out "$OUT_ROOT" --season 2015-16 --report reports/decisions.md --folds 5 --seed 0 --bootstrap 1000

What needs real data: the Phase 0 tables of every game (possessions, shot_frames, states_1hz under DATA_ROOT/analytics), the frames and play-by-play under DATA_ROOT/sportvu_contract, and, optionally, the EPV and xFG lanes' outputs under OUT_ROOT.
