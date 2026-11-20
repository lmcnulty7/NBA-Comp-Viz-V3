# xfg-E2: second defender and contest geometry

Written 2026-11-19. Pre-registration: docs/research/xfg.md, section "xfg-E2", with its commitments fixed before any
development number (83a67474) and the Set A confirmation commitments fixed after the development result and before
the score (cbdf9881); decision rule PLAN.md D121. Scoring code: tools/research_xfg2.py at b526abf0, the code_commit of
reports/research/xfg-E2-final.json. Every development number below is copied from reports/research/xfg-E2-dev.json
and every test number from reports/research/xfg-E2-final.json (the one xfg-E2 entry of
reports/research/scoring_ledger.json); a number computed here says so. Intervals are 95 percent percentile intervals
of a 2000-resample bootstrap of games, seed 0.

**Verdict: adopted as a research result on development; Set A reads "inconclusive at 10 games", which is not a
contradiction, so the adoption stands (D121). The frozen xFG model on the site is unchanged.** On the 607
development games, adding the three release-frame features lowers out-of-fold log loss by 0.00182 [0.00141,
0.00222], past the pre-registered -0.001 with the interval excluding 0. On the 10 untouched Set A games the
difference is -0.00224 [-0.00657, +0.00192]: the interval straddles 0, as the projected half-width (0.0034) said it
likely would.

## What was tested

- Baseline: the frozen hgb_contest_prior (location and contest features, shooter prior, isotonic step on 3 inner
  folds), refit with the frozen functions. New arm: the same design plus three features at the stored release frame:
  the second nearest defender's distance; the angle at the shooter between the line to the target rim and the line
  to the closest defender; the closest defender's speed toward the shooter-to-rim line over 0.4 s. Ablation: the
  baseline plus the second defender's distance only. Definitions in docs/research/xfg.md, xfg-E2 commitments.
- Gate: the baseline as this module fits it, refit on all 607 training games under the frozen outer folds,
  reproduced the stored out-of-fold p_selected of all 101,853 training shots (largest difference 2.97e-8, tolerance
  1e-6, no fold mismatch; the gate record carried in the final json).
- Decision (D121): development decides (difference at most -0.001, interval excluding 0). If it adopts, Set A is
  scored once as a confirmation: "confirms" wholly below 0, "inconclusive at 10 games" straddling 0,
  "contradiction" wholly above 0, which stops the adoption.

## Development games (607 games, 101,853 shots, 5 grouped outer folds)

| Arm | Log loss | Brier | Calibration slope | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| baseline | 0.62944 [0.62726, 0.63162] | 0.22090 [0.21995, 0.22185] | 1.000 [0.976, 1.025] | 0.0031 [0.0027, 0.0070] |
| new | 0.62762 [0.62539, 0.62982] | 0.22013 [0.21919, 0.22110] | 1.002 [0.979, 1.027] | 0.0030 [0.0027, 0.0068] |
| ablation | 0.62909 [0.62688, 0.63130] | 0.22074 [0.21979, 0.22167] | 0.996 [0.973, 1.021] | 0.0025 [0.0029, 0.0071] |

- New minus baseline: -0.00182 [-0.00222, -0.00141]. Adopts on development.
- Ablation minus baseline: -0.00034 [-0.00065, -0.00002]; new minus ablation: -0.00148 [-0.00185, -0.00112]. Most
  of the gain comes from the contest angle and the defender's speed toward the shot line, not from the second
  defender's distance.
- Feature coverage: 0.613 of shots have the second defender's distance (the shots with frozen contest features),
  0.609 the angle, 0.607 the speed (dev json, new_feature_coverage).
- The spread of the 607 per-game mean differences is 0.0055; the projected 10-game half-width from it is 0.0034
  (1.96 times 0.0055 over the square root of 10), reported before the score.
- Binned ECE is biased upward under resampling, so its intervals can sit above the point estimate (as in
  xfg-E1).

## Test set (Set A: the 10 untouched 2015-12-05 games, 1,668 shots, scored once)

| Arm | Log loss | Brier | Calibration slope | ECE, 10 equal-mass bins |
| --- | --- | --- | --- | --- |
| baseline | 0.64278 [0.62610, 0.65967] | 0.22653 [0.21973, 0.23340] | 0.892 [0.705, 1.165] | 0.0299 [0.0230, 0.0586] |
| new | 0.64054 [0.62555, 0.65662] | 0.22620 [0.22016, 0.23267] | 0.904 [0.751, 1.117] | 0.0264 [0.0232, 0.0585] |

- New minus baseline: -0.00224 [-0.00657, +0.00192]. Straddles 0: "inconclusive at 10 games". Not a
  contradiction, so the development adoption stands.
- Beside it, read by no rule: the t interval on the 10 per-game mean differences, -0.00218 [-0.00719, +0.00283]
  (D80).
- Report only, no rule reads it: the test estimate has the same sign as the development one and is of similar size.
- Coverage differs: 0.561 of Set A shots have the frozen contest features (and so the new ones), against 0.613 on
  development. The new features can move only those shots.
- The ablation was not scored on Set A, as committed.

## How the score was reached

- The dev run's commit was not recorded in xfg-E2-dev.json (the dev stage now records it for future runs). In its
  place the pre-score review checked that the arms, the features, the folds and the development code are unchanged
  byte for byte from 29c85546 to the scoring commit, and the rehearsal rebuilt the features exactly.
- Three rehearsals ran the scorer's own path on 10 development games in place of Set A, with both arms fitted on
  the other development games. The first refused: its row check compared the keys as strings, and a shot with no
  shooter id became a float NaN, which never equals itself; two diagnoses on the same games showed the rows, the keys
  and all three features identical (0 differing rows). The second passed at 710d134f with a typed, exact check. The
  third passed at the scoring commit b526abf0, after the pre-score review, with every shared column compared
  exactly; the final json holds its sha256 and the final stage refuses a rehearsal that did not pass or ran at
  another commit.
- No held-out game was read; Set A was read only inside the score_once scorer.

## What this means

- On development, a defender's angle and approach toward the shot line add measurable information to the frozen
  model's contest features, about four times the gain of the second defender's distance alone (computed: 0.00148
  over 0.00034, 4.4).
- Ten untouched games cannot confirm an effect of this size: the interval is about 0.0085 wide (computed: 0.00192
  minus -0.00657). No larger untouched 2015-16 set exists (docs/research/DATA.md).
- This is a research result. The site keeps the frozen xFG model; any change to it needs a D line.

## What would change the verdict

A larger untouched test set; from the development spread, about 100 games would bring the 10-game half-width of
0.0034 down to about 0.0011 (computed: 0.0034 times the square root of 10 over 100), still near the effect size. A
contradiction on Set A would have stopped the adoption; it did not occur.
