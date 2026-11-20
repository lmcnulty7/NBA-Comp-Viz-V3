# decisions-E2: one value scale for every option, validated on the decisions made

Written 2026-11-19 (UTC). Stage: **finished as a development result: not adopted; Set A not scored** (PLAN.md
D122: Set A is scored only if development adopts). Pre-registration: docs/research/decisions.md, "decisions-E2",
with its amendment (D122, committed ac7ce030 before any run) and the amendment to commitment 5 (D123, be7dfc0a,
before any development number). Code: tools/research_decisions_e2.py (first committed 8faf8182, before any run; the
fixes after it are listed under "Faults found and fixed"). Source of every number below:
reports/research/decisions-E2-dev.json, written by `e2-dev` at a682d9ad. Numbers marked computed are arithmetic on
it.

**Verdict: not adopted on development.** Of the three adoption checks, the regret coefficient alone passes; the
paired difference and the slope fail.

- Regret coefficient on the EPV scale: 0.0700 [0.0405, 0.1005], above zero. Passes.
- Paired difference in that coefficient, new minus frozen: -0.0956 [-0.1289, -0.0624], wholly below zero. Fails:
  the frozen regret carries more signal (0.1655).
- Chosen-value slope: 0.3972 [0.3710, 0.4236], outside 0.8 to 1.2. Fails (the frozen slope is 0.1760).

These are on 486448 decisions in 592 games. 12 development games are left out entirely: their frames have no shot
clock, which the committed drive model needs, so none of their decisions has a drive value.

The first e2-dev run's numbers were seen before a fault in its rows was fixed. That run kept a decision when its
chosen, hold and regret values were finite, so a decision with no drive value kept a regret over fewer options.
Commitment 8 says every value finite in both arms; the fix (e2c7b624) restores commitment 8 as written and is not a
new choice. The corrected run above is the primary reading. The first run, on 509852 decisions, gave regret
coefficient 0.1306 [0.1017, 0.1595], paired difference -0.0393 [-0.0710, -0.0084] and slope 0.4163 [0.3910,
0.4408]: the same verdict, not adopted, with the same two checks failing.

Report only, point estimates: the new metric depends more on decision mix than the frozen one (R squared of the
player value on kind shares 0.5712 against 0.3350). Set A was not read, the scoring ledger has no decisions-E2
entry, and no Set A scorer was written.

## What was tested

The frozen decisions lane values each option of a decision (shot, pass to each teammate, hold, drive) with hand-set
formulas, and only hold uses the EPV model; decisions-E1 found the resulting player table tracks decision mix more
than decision quality. This experiment puts every option on the EPV scale and checks the values against what
happened after real decisions.

- Decisions: the frozen decisions of the 604 development games (518421), reproduced exactly by the decisions-E1
  gate. Both arms value the same decisions. The frozen detector finds completed passes only, so pass decisions are
  selected on completion in both arms (stated, not repaired).
- New values (commitment 4), each at the decision frame:
  - hold: the EPV of the actual state, scored as the last state of the possession's sequence;
  - pass to teammate j: P(completion to j) times the EPV of the same state with j holding the ball;
  - drive: P(drive keeps the ball) times the EPV of a modelled state 1 s later (handler and ball 8 ft toward the
    basket, everyone else still); the chosen drive uses the same modelled state, not its observed path;
  - shot: the frozen shot value.
- EPV: the frozen EPV sequence model refit on each of the lane's 5 folds with the frozen functions, so every EPV of a
  decision comes from a fold model that never saw its game.
- Completion model (commitment 5 as amended by D123) and drive model: logistic regressions fitted out of fold by game.
- Outcome: points_to_end of the possession's last stored state at or before the decision frame.
- Decision (D122, development games): the regret coefficient's interval above 0, the paired difference (new minus
  frozen) in that coefficient with its interval above 0, and the chosen-value slope inside 0.8 to 1.2.

## Gates and data

- EPV fold refits: on every fold, the refit model's expected points on the fold's stored states equal the stored
  out-of-fold epv_points exactly (largest difference 0.0 against a tolerance of 1e-5), with every stored state matched
  and every stored fold equal to the fold refit.

| fold | training games | test games | stored states matched | GRU fit (s) |
| --- | --- | --- | --- | --- |
| 0 | 483 | 121 | 347153 | 215.6 |
| 1 | 483 | 121 | 348423 | 209.7 |
| 2 | 483 | 121 | 350600 | 144.5 |
| 3 | 483 | 121 | 348560 | 264.6 |
| 4 | 484 | 120 | 344383 | 160.9 |

- State builder: the state rows built at any frame reproduce the stored states_1hz rows on the 13 games checked
  (37415 rows; positions within 1.9e-6 ft, no id mismatch; elapsed differs by at most 0.5 s by design, continuous
  time against the whole-second grid).
- Frame pass: 604 games, 518421 decisions, 3622555 state rows scored. 757 decisions have no stored state at or
  before their frame (for example a possession of about 1 s of game clock that has no state at all), so no outcome.
  12 decisions have no usable state at the decision frame (recorded as "decision player not on the floor"; this run
  checked that reason before a frame's own reason, so the 12 are not split by cause).
- Pass attempts (D123): 343447 attempts, 273330 completed, 9007 failed, 61110 left out; completion rate 0.968 on the
  labelled attempts. Also not attempts: 76525 releases with no next holder within 1.5 s and 11622 with a shot between.
- No held-out game and no Set A game was read. Development games come only through dev_games.

## What each estimate leaves out

- **The completion model estimates completion among attempts that resolved cleanly.** The 61110 attempts the D123
  label leaves out (17.79 percent) are the contested ones: a defender held the ball, then the offence held it again
  or the possession did not end in a turnover. Left-out share by fold: 17.83, 17.31, 17.85, 17.98, 18.01 percent; by
  game, 10.0 to 35.5 percent (median 16.86).
- **The drive value needs the shot clock.** The drive model uses the shot clock; 31356 decisions are in frames with
  no shot clock, 12 games have none at all, so 31366 decisions have no drive value (6.05 percent; by fold 7.07,
  6.17, 6.28, 4.86, 5.85 percent). Shot values are missing for none; hold and pass values for 12.
- **Primary rows** (commitment 8: every value finite in both arms): 486448 decisions in 592 games. Left out: 31973
  (6.17 percent); by fold 7.17, 6.29, 6.40, 5.02, 5.93 percent; by kind drive 5.00, pass 6.62, shot 7.03 percent;
  by game median 3.96 percent, with the 12 games without a shot clock left out whole.

## Development games (592 games, 486448 decisions)

Intervals: a bootstrap of games, 2000 resamples, seed 0, of the fixed per-decision values (no value refit in a
resample, so the resample adds no noise, unlike the reliability statistics of decisions-E1, D86). The paired
difference resamples the same games for both arms.

| quantity | new (EPV scale) | frozen |
| --- | --- | --- |
| **slope of points_to_end on the chosen value** | **0.3972 [0.3710, 0.4236]** | 0.1760 [0.1638, 0.1876] |
| **coefficient on regret** (points_to_end on hold value plus regret) | **0.0700 [0.0405, 0.1005]** | 0.1655 [0.1520, 0.1781] |
| mean chosen value (points) | 1.0013 | 0.6435 |
| mean regret (points) | -0.0928 | -0.4065 |

- **Paired difference in the regret coefficient, new minus frozen: -0.0956 [-0.1289, -0.0624].**
- Ablation (pre-registered, report only; it uses 1 s of the future by design): the change in EPV over the chosen
  action, EPV 1 s after the decision minus EPV at it, has coefficient 0.3917 [0.3706, 0.4129] in points_to_end on
  hold plus that change (481252 decisions with a state 1 s later).

Decision (D122): regret coefficient above zero (yes); paired difference above zero (no: below zero); slope inside
0.8 to 1.2 (no: 0.397). **Not adopted.** Set A not scored.

## Report only

- Overlap (commitment 7): the best option is hold in 94423 decisions (19.41 percent). Of the 392025 decisions whose
  best option has a rated kind, 69312 (17.68 percent) have a choice probability below 0.05 for that kind, so their
  best value is extrapolated.
- Secondary readings (decisions-E1's statistics on the same decisions, point estimates): stability 0.9653 new
  against 0.8895 frozen; R squared of the player value on kind shares 0.5712 new against 0.3350 frozen; split-half r
  0.8703 new against 0.9212 frozen.
- The first e2-dev run used a looser filter than commitment 8 (see below): 509852 decisions, slope 0.4163 [0.3910,
  0.4408], regret coefficient 0.1306 [0.1017, 0.1595], paired difference -0.0393 [-0.0710, -0.0084]. Same decision.

## What this means

- One EPV scale does make the chosen value more predictive: the slope rises from 0.176 to 0.397. But 0.397 is far
  from 1: a 1-point difference in the chosen value goes with about 0.4 points at the end of the possession. Noise in
  the values (an errors-in-variables pull toward zero) and the simplified states (no ball travel time for a pass,
  defenders that do not move on a drive) can both do this; this run does not separate them.
- The regret on the EPV scale carries less signal than the frozen regret. The frozen regret coefficient is larger and
  the paired interval is wholly below zero. The hand-set values, uncalibrated as they are, rank good and bad choices
  against what followed better than the EPV values do.
- Report only, point estimates: the new metric is more a function of decision mix, not less. At the player level, kind shares explain 0.571 of its
  variance against 0.335 for the frozen one. This experiment does not fix decisions-E1's finding; it sharpens it.
- About one decision in five has hold as its best option, and about one in six of the others has a best option the
  player would almost never choose there (choice probability under 0.05). Those values are extrapolated.
- The ablation, which looks 1 s ahead, carries far more signal (0.392) than either regret. It is not a decision
  value: it knows the outcome of the next second.
- Caveats (the dossier's risks, unrepaired): completed passes only; completion estimated among cleanly resolved
  attempts; the shot value for a decision that is not a shot is the frozen location value; play calls are
  unobserved.

## Faults found and fixed

All before the development statistics existed, except the last.

1. The first failure label (commitment 5) marked tight defence as failed passes on 2 smoke games (completion near 75
   percent). The lead fixed the label in D123; the amendment is in the dossier.
2. The join of the frozen decisions to their contexts aligned on a table index and matched no row after the first
   game (96cfa9b3).
3. A possession with no stored state stopped the frame pass; such a decision now has no outcome and is counted
   (4af01429).
4. The fold model could not be pickled whole; it is now saved as the frozen reader keeps it and checked bit for bit
   (b3f96020). The gate result of that first fold 0 run was lost; the rerun's gate is the one above.
5. The first e2-dev run's rows needed only the chosen, hold and regret values finite, not every value as commitment 8
   says, so a decision with no drive value kept a regret over fewer options. Fixed (e2c7b624) after that run's
   numbers were seen; they are kept above as a labelled reading, and the decision is the same under both.

## Set A

Not scored (PLAN.md D122): the test set is scored only if development adopts, and it does not. No Set A file was
read, no scorer was written, and the ledger has no decisions-E2 entry.
