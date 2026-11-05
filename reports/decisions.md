# Decision quality (analytics/decisions) (fixture output, not a result)

Written 2026-11-05T09:00:48Z by `python -m analytics.decisions.train --fixtures --report reports/decisions.md`.

**Fixture output, not a result.** Every number below comes from synthetic games (tests/_fx_decisions.py or analytics/fixtures.py); the real run is the Mac command at the end.

## Inputs

| game | held out | frames | possessions | shots | schema problems | duplicates dropped | epv rows | xfg rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02.07.2016.OMA.at.DUL | no | 115020 | 40 | 37 | 0 | 0 | 0 | 0 |
| 01.18.2016.HLX.at.LRD | no | 123210 | 40 | 44 | 0 | 0 | 0 | 0 |
| 02.04.2016.GRN.at.QCY | no | 123230 | 40 | 37 | 0 | 0 | 0 | 0 |
| 12.16.2015.PHX.at.GSW | yes | 128550 | 40 | 45 | 0 | 0 | 0 | 0 |
| 12.15.2015.ABQ.at.GRN | no | 117110 | 40 | 35 | 0 | 0 | 0 | 0 |
| 11.27.2015.GRN.at.QCY | no | 119730 | 40 | 40 | 0 | 0 | 0 | 0 |
| 12.12.2015.RNO.at.MOB | no | 121170 | 40 | 38 | 0 | 0 | 0 | 0 |
| 11.11.2015.HLX.at.CLE | yes | 118520 | 40 | 41 | 0 | 0 | 0 | 0 |

Training games: 6 (01.18.2016.HLX.at.LRD, 02.04.2016.GRN.at.QCY, 02.07.2016.OMA.at.DUL, 11.27.2015.GRN.at.QCY, 12.12.2015.RNO.at.MOB, 12.15.2015.ABQ.at.GRN). Held-out games set aside before any fit: 2 (11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW). Guard: `analytics.eval.refuse_heldout` checked the training ids against sportvu/splits.json.
Value model fit rows (localised shots of the training games): 231. Protocol: out-of-fold under 5 grouped folds by sportvu_game (seed 0).

## Event counts (training games)

| game | shots | passes | drives | possessions used | skipped |
| --- | --- | --- | --- | --- | --- |
| 01.18.2016.HLX.at.LRD | 43 | 58 | 53 | 39 | {'possessions_gap': 1} |
| 02.04.2016.GRN.at.QCY | 37 | 71 | 51 | 39 | {'possessions_gap': 1} |
| 02.07.2016.OMA.at.DUL | 37 | 59 | 57 | 39 | {'possessions_gap': 1} |
| 11.27.2015.GRN.at.QCY | 39 | 69 | 58 | 39 | {'possessions_gap': 1} |
| 12.12.2015.RNO.at.MOB | 37 | 45 | 50 | 39 | {'possessions_gap': 1} |
| 12.15.2015.ABQ.at.GRN | 34 | 61 | 58 | 39 | {'possessions_gap': 1} |

Total: 917 decisions ({'pass': 363, 'drive': 327, 'shot': 227}).
Hold option (EPV lane) present on 0 decisions; xFG lane coverage of shot decisions 0.00.

## Value-function assumptions

- Shot option: points (3 at 22 ft or more, else 2) times the location-only xFG probability at the player's position, minus 0.03 per foot of nearest-defender distance below 6 ft when no contest model exists (floor 0.01).
- Pass option: points times the location-only probability at the teammate's position (no contest term: openness already carries the defender distance) times openness (1 at 6 ft or more of nearest-defender distance, linearly down to 0.5 at 0 ft) times pass risk (0.95 minus 0.01 per foot of pass length over 15 ft).
- Hold option: the EPV lane's epv_points of the last state at or before the decision frame; absent when the EPV table is absent.
- Drive option: the shot value 8 ft nearer the rim along the line to the basket (never nearer than 3 ft), with the decision-frame defender distance, times 0.90; it is in every decision's option set and is the chosen value of a drive, so every quality is regret (at most 0) and best_option is one of shot, pass_to_<rank>, hold, drive.
- Attacked basket: per (offence team, period) by majority vote over that team's possessions, each voting with the end where its offence's mean x lies; the value-model fit uses the basket nearer the shooter at the release.
- Event detection: holder within 2.5 ft of the ball under 8 ft; pass arrives within 1.5 s with no release in between; drive = the first frame at which the holder's speed toward the basket is measured above 8 ft/s in a run lasting 0.6 s.
- Upstream rows: a training game refuses EPV or xFG rows with fold == -1, a held-out game refuses rows not flagged heldout (refused counts: none).
- Leakage: option values use the decision frame and earlier frames only; the value model is out-of-fold by game; `schema.assert_time_forward` ran on every context frame; `assert_decisions_before_outcome` on every game.
- Prior of the option values: candidate 1 = location-only xFG, candidate 2 = the xFG lane's p_selected on shot events. Rule fixed in advance: candidate 2 when its coverage of shot decisions is at least 0.50. Selected: **location** (no xFG table present).

## Player table (training games, gated at 300 decisions; ranked on mean quality, by-kind means are regret at most 0)

No player passes the 300-decision gate: nobody is ranked.

82 of 82 players are under the 300-decision gate and carry no rank (listed unranked, by id, first 10):

| player_id | n_decisions | n_games | mean quality | shot | pass | drive |
| --- | --- | --- | --- | --- | --- | --- |
| 9000 | 8 | 1 | -0.3006 | -0.3026 | -0.2105 | -0.5627 |
| 9001 | 35 | 1 | -0.2048 | -0.1241 | -0.2769 | -0.1733 |
| 9002 | 5 | 1 | -0.2895 | -0.2974 | -0.2906 | -0.2785 |
| 9003 | 13 | 1 | -0.2512 | -0.0482 | -0.5096 | -0.2068 |
| 9004 | 6 | 1 | -0.2758 | 0.0000 | -0.4084 | -0.0215 |
| 9005 | 3 | 1 | -0.1284 | -0.1018 | -0.1951 | -0.0883 |
| 9006 | 4 | 1 | -0.1434 | -0.0040 | n/a | -0.2827 |
| 9007 | 6 | 1 | -0.3866 | -0.4871 | -0.3736 | -0.2991 |
| 9008 | 28 | 1 | -0.1792 | -0.0730 | -0.2724 | -0.1047 |
| 9009 | 11 | 1 | -0.3789 | -0.0778 | -0.6722 | -0.5450 |

## Validation: mean decision quality per player-game against on-court offensive rating

Spearman per game between the players' mean decision quality (a sum would rank by decision count, since quality is regret) and their points per 100 offensive possessions on the floor, over players with at least 3 decisions in the game; sign test over games (two sided); bootstrap of the mean Spearman by game (n=200, seed 0); shuffled = the same statistic with mean quality permuted across those players within each game.

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location | 6 | 0.0801 | -0.1646 | 0.2831 | 4 / 5 | 0.3750 | -0.0949 |

## Held-out games (valued once by the model refit on all training games; never fitted or tuned on)

322 decisions from 2 held-out game(s): 11.11.2015.HLX.at.CLE, 12.16.2015.PHX.at.GSW.

| kind | n | mean quality |
| --- | --- | --- |
| drive | 106 | -0.1979 |
| pass | 132 | -0.3119 |
| shot | 84 | -0.1466 |

| prior | games | mean rho | lo | hi | positive / signed | sign p | shuffled mean rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| location (held out) | 2 | -0.2329 | -0.3185 | -0.1474 | 0 / 2 | 0.5000 | 0.2545 |

## Outputs

- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/01.18.2016.HLX.at.LRD.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/02.04.2016.GRN.at.QCY.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/02.07.2016.OMA.at.DUL.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/11.27.2015.GRN.at.QCY.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/12.12.2015.RNO.at.MOB.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_events/12.15.2015.ABQ.at.GRN.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/heldout/fixture/decision_events/11.11.2015.HLX.at.CLE.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/heldout/fixture/decision_events/12.16.2015.PHX.at.GSW.parquet
- /tmp/decisions_fixture_fcjxm2sz/analytics/decisions/fixture/decision_players.parquet

## Status

Fixture output, not a result. The real run, on the Mac where the full data lives:

    python -m analytics.decisions.train --data "$DATA_ROOT" --out "$OUT_ROOT" --season 2015-16 --report reports/decisions.md --folds 5 --seed 0 --bootstrap 1000

What needs real data: the Phase 0 tables of every game (possessions, shot_frames, states_1hz under DATA_ROOT/analytics), the frames and play-by-play under DATA_ROOT/sportvu_contract, and, optionally, the EPV and xFG lanes' outputs under OUT_ROOT.
