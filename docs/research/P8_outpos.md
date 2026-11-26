# P8 placement route: K1 on the output positions only (design, registered before any number)

Written 2026-11-26 after P8_footpos.md. The lead decided to take its proposed next step, a report-only scorer variant
beside the pinned rows. This document is the design. It was committed before the tool produced any number, and
before any hou_cle_2016 file was read for this arm. Tool: tools/p8_outpos.py (rules in its docstring, the same as
here).

## What changes and what does not

- The arm. The pipeline's output court position of a box is B(foot_stab), where B is the r23 court model's per-frame
  H. The arm moves that output point by K1 (P8_footbias.md, fit rule unchanged). K1 takes the median depth offset of
  the box's cell, where a cell is the box pixel height by the third of the output point under B's near side. It
  removes that offset along the unit vector from the build camera's ground point to the output point. The ground
  point comes from sportvu.camera.pose_from_h(B), and there is no shift when B has no pose.
- foot_stab is not touched. The truth-frame feet, the matching, missed and ghosts are therefore identical to the
  baseline by construction. The code asserts it for every window: missed, in-frame, non-referee ghost and box counts,
  and the matched keys, must be equal, or the tool stops.
- The pinned scorer (tools/p8_scorecard.py, D88) and its rows are unchanged and adopt nothing. The variant is report
  only and is shown beside the pinned rows.

## The report-only scorer variant

The pinned lines, from tools/p8_scorecard.py score_window:

    feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
    if B.accepted(r) and r.get("H") is not None:
        Hp = np.array(r["H"], np.float64).reshape(3, 3)
        cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
        for bi, j, _ in pairs:
            e = cp_[bi] - sv[j]
            err_of[j] = (float(np.hypot(*e)), float(e[1] if nh else -e[1]))

The variant runs the same lines. The only change is that between cp_ and e it replaces cp_[bi] with its K1-shifted
point. Everything else is unchanged: ct, pairs, nh, the region third_of(sv[j][1], nh), and the rows of
tools/p8_e1_score.py (rows, diffs, boot; 2,000 resamples, seed 0, windows of the game). Both arms score the same
matched players, so the paired rows have no denominator change.

## Fit and the frozen table

- Fit: the 3 pool games only (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015), through tools.p8_footbias.offset_rows and
  fit, unchanged.
- Pool rows: for each pool game, K1 is fitted on the other two and scored by the variant against the pinned rows.
  This is development evidence only. Its numbers are largely known already: P8_footpos.md scored the same K1 on the
  same games. The only difference there was that the foot pixel moved and matching was redone, so these rows will be
  close to P8_footpos.md's (p50 -0.25 to -0.28 ft, near-field bias -0.94 to -1.09 ft).
- Frozen table: K1 fitted on all three pool games, written by `pool` to reports/research/p8_outpos_table.json, with
  the fit rows per cell. `pool` reads no hou_cle_2016 file.
- `score-hou` refuses to run unless that table is tracked by git with no uncommitted change. It records the table's
  sha256 and the commit that last changed it.

## hou_cle_2016, scored once

`score-hou` refuses if its record (reports/research/p8_outpos_hou.json) already exists. It applies the frozen table
unchanged to hou_cle_2016's reference-platform r23 sidecars (builds/m9_r23_detcpu__hou_cle_2016). It checks the
pinned matched entries against the registered score file, as the pool does.

Decision rule (variant minus pinned, window intervals over hou_cle_2016's windows):

- PASS when all four hold:
  1. the position p50 difference is below zero, with its window interval wholly below zero;
  2. the near-field bias difference is below zero, with its window interval wholly below zero;
  3. the position p90 difference has a window interval upper end of at most +0.10 ft;
  4. missed and ghosts are identical (asserted; the tool stops otherwise).
- Anything else is FAIL.

Why the p90 margin is +0.10 ft:

- K1 shifts every output point in a cell, including players already placed well, so the tail must not grow.
- +0.10 ft is the p90 margin E1b registered. It is about 3 percent of r23's p90 on this game.
- It is larger than the change the detector device alone made to r23's p90 (M9: -0.02 to -0.04 ft) but small against
  the gain needed to matter.

## Forking paths, named

- Steps 1 and 1b of P8_placement.md, P8_footbias.md and P8_footpos.md: every number in them has been seen. That
  includes the choice of K1's form and cells, made after step 1 and step 1b.
- hou_cle_2016, what is seen:
  - its missed gains: -0.021 from step 1's pooled C1 and -0.0087 from P8_footbias.md's K1;
  - its step 1 and step 1b offsets under the truth camera;
  - its median depth offsets under the r23 court's H in P8_footbias.md: by third -0.71, -0.96, -0.84 ft, and by box
    height and camera distance;
  - its pinned (baseline) position rows, recorded in M9 of P8_E1.md (r23 on CPU: p50 1.516 ft, p90 3.286 ft,
    near-field bias 1.154 ft).
- hou_cle_2016, what is unseen: its variant position rows. The build-frame offset medians above bear directly on how
  far K1 will move its near-field bias. So the direction of the bias row on hou_cle_2016 is partly predictable from
  seen numbers, and the result is not a blind test of direction for that row.
- Platform: one arena outside GSW, one game, development data.

## What a pass would and would not mean

- A pass would mean that a K1 table frozen on three GSW games improves the output positions (p50 and near-field bias)
  on one non-GSW game, without widening the tail, with detection unchanged by construction.
- It would not mean:
  - that the correction generalises across arenas: one game is one arena and one broadcaster;
  - an adoption: nothing goes into the pipeline or the site without a further D line;
  - a Level 1 result: Level 1's sealed scoring stays at the end of P8 (D129), and hou_cle_2016 is a development game.
- A FAIL would mean that the pool table does not transfer to the new arena. Any arena-specific variant would need a
  new design.

## Order of work

1. This design, committed with the tool and tests, before any number (this commit).
2. `pool`, after the design commit: the pool rows and the frozen table, committed. It reads no hou_cle_2016 file.
3. Stop for the lead's independent review.
4. `score-hou`, once, only after the review.

## Pool results (computed 2026-11-26, after the design commit 3b777b99; no hou_cle_2016 file read)

Stored files only, no lock, about 45 seconds. Every check passed:

- load_frames' count check against the registered score files;
- every window's pinned matched entries equal to the registered score file's CPU side;
- on every window, the variant's missed, in-frame, ghost and box counts and matched keys equal to the pinned ones
  (missed and ghost differences are exactly 0.0000 with zero-width intervals).

Records: reports/research/p8_outpos_pool.json and the frozen table reports/research/p8_outpos_table.json.

Leave one game out, variant minus pinned, on all matched players (17,414, 11,593 and 18,883; the same set for both
arms):

| Row | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| Position p50 (ft) | 1.546 to 1.293; -0.253 [-0.287, -0.214] | 1.519 to 1.235; -0.284 [-0.319, -0.232] | 1.505 to 1.235; -0.269 [-0.311, -0.215] |
| Position p90 (ft) | 3.329 to 3.110; -0.219 [-0.294, -0.134] | 3.106 to 2.986; -0.120 [-0.201, -0.050] | 3.247 to 3.073; -0.174 [-0.258, -0.097] |
| Near-field bias (ft) | 1.576 to 0.642; -0.934 [-0.960, -0.912] | 1.150 to 0.050; -1.100 [-1.116, -0.939] | 1.330 to 0.231; -1.099 [-1.117, -1.040] |
| Missed, ghosts | identical | identical | identical |

As expected, the rows are close to P8_footpos.md's. On the pool, the hou_cle_2016 rule would pass on every game:
p50 and near-field bias below zero with intervals below zero, and p90 improving. The pinned single-arm values here
(for example gsw_bkn_2015 p50 1.546) differ from the registered paired rows (1.535 in M6) only because those were
computed on the players both of two builds matched.

Frozen table: the median depth offset per cell, from all three pool games (ft; negative means short). The cells:

| Box height | Near | Middle | Far |
| --- | --- | --- | --- |
| 150 to 200 px | -0.41 | -0.87 | -1.16 |
| 200 to 250 px | -0.34 | -0.87 | -0.88 |
| under 150 px | -0.21 | -0.88 | -1.10 |
| 250 px and over | -0.39 | -0.79 | 0.00 (1 fit row, under 50) |

The fit rows per cell are in the table file. hou_cle_2016 has not been read for this arm. `score-hou` waits for the
lead's independent review.

## Design version 2 (2026-11-26, after the independent review; registered before any of its numbers)

The independent review returned GO WITH FIXES. It found the variant scorer faithful and found no leakage, but the
version 1 hou_cle_2016 test was nearly a sure pass. Its per-cell build-frame medians were already committed and predict
the result, and the rule had no naive comparator. Version 2 replaces the version 1 decision rule and score-hou. The
version 1 design (3b777b99) and pool rows (18df8e7f) stay as committed. No hou_cle_2016 file has been read for this
arm, and score-hou has not run. tools/p8_outpos.py carries the same rules in its docstring; its version 1 score-hou now
refuses to run.

### 1. What is already seen about hou_cle_2016, and a forecast written before scoring

Seen, in reports/research/p8_footbias.json "cells" (median depth offset in ft under the r23 court's H; count):

| Box height | Near | Middle | Far |
| --- | --- | --- | --- |
| under 150 px | -0.50 (499) | -0.95 (1,847) | -0.72 (4,688) |
| 150 to 200 px | -0.67 (7,327) | -0.99 (13,126) | -0.97 (5,047) |
| 200 to 250 px | -0.78 (4,917) | -0.81 (2,160) | -0.42 (119) |
| 250 px and over | -0.69 (169) | +0.48 (35) | none (0) |

Also seen: hou_cle_2016's missed gains (-0.021, -0.0087), its step 1 and 1b offsets, its by-third, by-height and
by-distance medians in P8_footbias.md, and its pinned position rows in M9 of P8_E1.md.

Forecast, computed by `forms` from these committed numbers only (plus the frozen tables and the pool rows) and
committed before score-hou runs. The hou_cle_2016 result then tests the forecast. For a game g and a form F with table
t:

- R_g(F) is the count-weighted mean over cells of (|m| - |m - t|), with m the cell's median depth. It is the mean
  absolute cell bias the shift removes.
- Calibration per form, from the pool's leave-one-game-out rows: k_p50 = (sum of the pool games' p50 differences) /
  (sum of their R_g), and k_p90 likewise.
- Forecast for hou_cle_2016: p50 difference k_p50 x R_hou and p90 difference k_p90 x R_hou.
- Near-field bias, forecast in depth terms. Per third (near, far), before is the absolute count-weighted mean cell
  median, and after is the same once the shift is removed. The forecast reads the larger third.
- The pool's own predicted against observed p50 differences are reported as a calibration check.

### 2. Naive comparator K0 and the form decision

- K0 is one global depth shift: the median depth offset of the fit games' whole offset population, with no cells,
  removed along the same ray. It is frozen from the 3 pool games into reports/research/p8_outpos_table_k0.json.
- Form rule, on the pool only, leave one game out, through the same variant scorer: K1's cells are kept only if K1's
  position p50 difference is below K0's on every pool game. Otherwise the arm is K0.
- The chosen form and the K0 table's sha256 are then pinned in tools/p8_outpos.py by a commit before hou_cle_2016 is
  read.
- On hou_cle_2016, K0 and K1 are both scored in the single run and both are reported. A claim that the cells matter
  needs the chosen form to beat the other on hou_cle_2016 p50. That is development evidence.

### 3. Decision rule for hou_cle_2016 (replaces version 1's)

For the chosen form, variant minus pinned, window intervals over hou_cle_2016's windows, PASS when all hold:

1. The position p50 difference is at most -0.10 ft, with its window interval wholly below zero.
2. The position p90 difference has a window interval upper end of at most +0.10 ft. This is non-inferiority on the
   tail at about 3 percent of p90.
3. Near-field bias is a guard only: its value after must not exceed its value before. The third it reads (near or far)
   is reported before and after.
4. Missed and ghosts are identical (asserted in code).

Why 0.10 ft is the practical effect. It is about 7 percent of r23's p50 on these games (about 1.5 ft). It is also
about the smallest gain that would show at the 1.92 ft held-out median the README reports. A smaller gain would leave
that headline number nearly unchanged.

### 4. Claim limits

- Detection is identical by construction of this scorer variant, not of the pipeline. Anything downstream that uses
  the court positions is untested by this arm: tracks, team labels, the analytics features.
- The rows are per box, B(foot_stab) shifted, not the pipeline's smoothed trajectories.
- The intervals are window intervals within one game: one arena, one broadcaster, development data.
- The direction of every row was predictable from seen numbers (above). A pass would confirm a forecast, not discover
  an effect.
- Nothing is adopted into the pipeline or the site without a further D line. Level 1's sealed scoring stays at the
  end of P8 (D129).

### 5. Guards (in code, with tests)

- Once-only marker. score-hou writes placement/outpos_hou.started under the P8 workspace before it reads any
  hou_cle_2016 file, and refuses if that marker or the record (reports/research/p8_outpos_hou.json) exists. A crash
  after the marker is written therefore needs the lead's decision to retry.
- Pinned tables. The K1 table's sha256 is pinned in the tool (c420b0a642851104894aec03189cce3404c407703347b3fff29dfc96b7cb80ce,
  at 18df8e7f), and so will the K0 table's be. score-hou refuses on a mismatch, on an unpinned sha, or on a table that
  is untracked or carries an uncommitted change.
- Clean code. git HEAD is recorded. score-hou refuses if tools/p8_outpos.py, tools/p8_footbias.py,
  tools/p8_placement.py, tools/p8_scorecard.py or tools/p8_e1_score.py has an uncommitted change.
- Tests cover the refusals on an existing record or marker, on a mismatched or unpinned sha, and on a dirty table or
  dirty code.

### 6. Report only

- Position p90 by camera third and by box-height bin, pinned against variant.
- Per game, the count of frames where near_side(B) differs from near_side(truth).
- After the hou_cle_2016 decision, as post-run diagnosis: K1 fitted on hou_cle_2016 in sample (an upper bound) and the
  transfer ratio (the frozen K1's p50 difference over the in-sample one).

### Order of work (version 2)

1. This version 2, committed with the tool and tests, before any of its numbers.
2. `forms` (pool only): K0 against K1, the form decision, the frozen K0 table and the forecast, committed.
3. A commit pinning the K0 table's sha256 and the chosen form in the tool.
4. Stop for the lead's short re-review.
5. score-hou, once, only after the lead's go.

### Version 2 results, pool only (computed 2026-11-26, after a10b35d2; no hou_cle_2016 file read)

Stored files only, no lock, about a minute. Every check passed on every pool game:

- the registered counts and matched entries reproduce;
- for both forms, the variant's detection rows equal the pinned ones on every window.

Records: reports/research/p8_outpos_forms.json and the frozen K0 table reports/research/p8_outpos_table_k0.json (one
global shift of -0.785 ft from 51,198 fit rows; sha256 21bc22d960410a5b1723bc20133cfd781ee6691ec3cf2be4a2914064fa7f47ba).

Leave one game out, variant minus pinned (window intervals):

| Row | Form | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- | --- |
| Position p50 (ft) | K1 | -0.253 [-0.287, -0.214] | -0.284 [-0.319, -0.232] | -0.269 [-0.311, -0.215] |
| Position p50 (ft) | K0 | -0.229 [-0.265, -0.180] | -0.267 [-0.305, -0.227] | -0.264 [-0.306, -0.207] |
| Position p90 (ft) | K1 | -0.219 [-0.294, -0.134] | -0.120 [-0.201, -0.050] | -0.174 [-0.258, -0.097] |
| Position p90 (ft) | K0 | -0.140 [-0.217, -0.070] | -0.103 [-0.189, -0.026] | -0.151 [-0.231, -0.076] |
| Near-field bias, before to after (third read) | K1 | 1.576 to 0.642 (far, far) | 1.150 to 0.050 (far, far) | 1.330 to 0.231 (far, far) |
| Near-field bias, before to after (third read) | K0 | 1.576 to 0.827 (far, far) | 1.150 to 0.402 (far, far) | 1.330 to 0.549 (far, near) |

Form decision by the rule: K1. Its p50 difference is below K0's on every pool game, by 0.025, 0.017 and 0.005 ft.
The margin on gsw_sac_2015 is small against the window intervals; the rule asks only for the order.

Report only:

- Frames where near_side(B) differs from near_side(truth): 0 of 2,073, 0 of 1,353 and 0 of 2,151.
- p90 by third, pinned to K1: the near third worsens on every pool game (3.11 to 3.23, 3.05 to 3.12, 3.13 to 3.26 ft).
  The middle and far thirds improve (far 3.66 to 3.14, 3.19 to 2.95, 3.49 to 3.07 ft). K0 worsens the near third
  more (to 3.42, 3.25 and 3.44 ft).
- p90 by box height, pinned to K1: improves in every bin on every game, except 200 to 250 px on gsw_cha_2016 (3.13 to
  3.15 ft).

Forecast for hou_cle_2016, from committed numbers only (written before score-hou):

| Form | R_hou (ft) | k_p50 | Forecast p50 difference | k_p90 | Forecast p90 difference | Near-field bias, depth terms, before to after |
| --- | --- | --- | --- | --- | --- | --- |
| K1 (chosen) | 0.627 | -0.419 | -0.263 | -0.266 | -0.167 | 0.84 (far) to 0.33 (near) |
| K0 | 0.715 | -0.472 | -0.337 | -0.245 | -0.175 | 0.84 (far) to 0.08 (near) |

Calibration check on the pool, predicted against observed p50 differences: K1 -0.268/-0.253, -0.264/-0.284,
-0.275/-0.269; K0 -0.230/-0.229, -0.268/-0.267, -0.262/-0.264.

What the forecast says, before any hou_cle_2016 file is read:

- The chosen form (K1) passes the p50 bar (-0.10 ft) with room, forecast -0.26 ft.
- The forecast puts K0 ahead of K1 on hou_cle_2016 p50 (-0.34 against -0.26 ft). Its cells are more uniform than the
  pool's, about -0.7 to -1.0 ft in every third, so K1's small near-third shifts leave more of its near-third bias in
  place. The forecast therefore expects the claim "the cells matter" to fail on hou_cle_2016.
- The near-field bias guard is forecast to hold for both forms.

Next: a commit pinning the K0 table's sha256 and the chosen form (K1) in tools/p8_outpos.py. Then stop for the lead's
re-review. score-hou has not run.


## Design version 3 (2026-11-26, after the re-review of version 2; registered before any of its numbers)

The re-review of version 2 came back GO WITH FIXES and left one decision to the lead. The lead decided that growth
of the near-third p90 counts as harm: the near third is where the Level 2 near-camera aim lives. On the pool, K1
raises near-third p90 on every game (3.11 to 3.23, 3.05 to 3.12, 3.13 to 3.26 ft). It also over-corrects the near
bias: on gsw_bkn_2015 the near third's scored bias goes from 0.01 to 0.40 ft. The build-frame near medians (0.29 to
0.48 ft) overstate the scored row's near bias (0.01 to 0.37 ft). hou_cle_2016 was therefore not spent on version 2:
its score-hou was never run and now refuses. Version 3 starts with a pool-only stage. No hou_cle_2016 raw file has
been read for this arm.

### Forms, named before computing

- Reference only, not eligible: K1 (version 1's cells) and K0 (version 2's global shift). Both are scored under the
  same rule so the reader can compare.
- Candidate K1m: K1 with the near third unshifted. The middle and far cells shift as in K1.
- Candidate K1t: K1's middle and far cells, plus one near shift s fitted on the scored row's own quantity, leave one
  game out.
  - The fit uses the fit games' matched players (truth pairs, error through B) whose output point lies in the near
    third under B.
  - s is the value on a grid from -1.50 to +0.50 ft, by 0.01, that brings their median toward_camera closest to zero
    after the shift along the ray. Ties go to the smaller |s|.
  - Why this second variant is justified: the over-correction comes from fitting the near shift on the build-frame
    depth offset while the scored row reads toward_camera on matched players. K1t fits the near third on the scored
    quantity itself.
- No other form will be tried in version 3.

### Pool rule (every pool game held out, each candidate)

1. Position p50 difference at most -0.10 ft, with its window interval wholly below zero.
2. Position p90 (the whole game) window interval upper end at most +0.10 ft.
3. Near-third p90 difference, paired on matched players whose SportVU third is near, with a window bootstrap
   (2,000 resamples, seed 0): upper end at most +0.05 ft. This is half the pooled margin, because a subgroup tail
   should not grow by more than noise. It was set from that reason, not from the data.
4. Near-field bias guard: the row after at most the row before.
5. Detection identical (asserted in code).

Among the candidates that pass on every pool game, the arm is the one with the lowest mean p50 difference over the
pool games. If no candidate passes, the arm stops and hou_cle_2016 is kept unspent. The hou_cle_2016 rule for the
chosen form is the same pool rule applied to hou_cle_2016 alone.

### Forking paths

This is the third look at the pool for this arm. Version 1's pool rows (18df8e7f) and version 2's K0 against K1 rows
(73a27713) are seen, including the near-third p90 and the near-field bias by third. The candidate forms were chosen
after seeing them. Everything listed under versions 1 and 2 about hou_cle_2016 is still seen: its per-cell medians and
counts, its missed gains, its offsets, its pinned position rows.

### Forecast (fixes review finding 2)

- k_p50 and k_p90 per form are calibrated leave one game out on the pool. Each pool game is predicted from the other
  two, and the prediction errors are reported. The hou_cle_2016 forecast uses k from all three pool games.
- Registered tolerance: the forecast holds if the chosen form's hou_cle_2016 p50 difference lies within plus or minus
  0.05 ft of it.
- The near-field bias forecast is in depth terms and is uncalibrated. On the pool, its near third runs low against the
  scored near bias, as shown by the build-frame medians against the scored near bias above. The run reports the
  pool's depth-term forecast beside its scored bias by third.

### Claim limits (fixes review finding 3)

- K1's order over K0 on the pool was below noise. The margin on gsw_sac_2015 was 0.005 ft, against window intervals
  about 0.1 ft wide.
- A claim that the cells matter rests only on the hou_cle_2016 comparison of the chosen form against K0, and that is
  development evidence.
- Every claim limit of version 2 still holds: detection is identical by construction of the scorer only; the rows are
  per box, not trajectories; the evidence is one game; the directions were predictable; nothing is adopted without a D
  line.

### Guards (fixes review finding 4)

- score-hou writes the decision record (reports/research/p8_outpos_hou.json) before it computes the in-sample
  diagnosis. The diagnosis goes to its own file, reports/research/p8_outpos_hou_diagnosis.json.
- It refuses unless `git status --porcelain -- tools sportvu` is empty, which covers untracked files too.
- It records the P8 workspace as "default". It refuses unless p8_root() is the default workspace, meaning
  V3_P8_ROOT is unset or equal to it.
- The once-only marker, the pinned shas of every form's table, the clean-code check and HEAD stay as in version 2.
- After the pool stage, the tool is pinned again with the frozen K1m and K1t tables' sha256 and the chosen form (or
  STOP). score-hou refuses unless a candidate is pinned.
- Tests cover the near-third bootstrap, the rule, the choice (reference forms never chosen; STOP when none passes),
  the default-workspace refusal and the porcelain refusal.

### Version 3 pool results (computed 2026-11-26, after cf82a23a; no hou_cle_2016 file read)

Stored files only, no lock, about two minutes. Every check passed on every pool game:

- the registered counts and matched entries reproduce;
- for every form, the variant's detection rows equal the pinned ones on every window.

Records: reports/research/p8_outpos_forms3.json and the frozen tables reports/research/p8_outpos_table_k1m.json and
reports/research/p8_outpos_table_k1t.json (K1t's near shift from all three pool games: -0.39 ft; per fold -0.48,
-0.33 and -0.37 ft for bkn, cha and sac held out).

Leave one game out, variant minus pinned:

| Form | Game | p50 [interval] | p90 upper end | Near-third p90 [interval] | Near-field bias before to after (third) | Rule |
| --- | --- | --- | --- | --- | --- | --- |
| K1m | gsw_bkn_2015 | -0.232 [-0.264, -0.203] | -0.148 | +0.036 [+0.012, +0.089] | 1.576 to 0.642 (far, far) | fails near p90 |
| K1m | gsw_cha_2016 | -0.249 [-0.285, -0.201] | -0.046 | +0.035 [+0.002, +0.073] | 1.150 to 0.367 (far, near) | fails near p90 |
| K1m | gsw_sac_2015 | -0.238 [-0.277, -0.190] | -0.111 | +0.036 [+0.013, +0.070] | 1.330 to 0.231 (far, far) | fails near p90 |
| K1t | gsw_bkn_2015 | -0.254 [-0.290, -0.213] | -0.130 | +0.148 [+0.077, +0.281] | 1.576 to 0.642 (far, far) | fails near p90 |
| K1t | gsw_cha_2016 | -0.284 [-0.321, -0.229] | -0.050 | +0.068 [-0.043, +0.148] | 1.150 to 0.059 (far, near) | fails near p90 |
| K1t | gsw_sac_2015 | -0.267 [-0.313, -0.214] | -0.098 | +0.138 [+0.041, +0.190] | 1.330 to 0.231 (far, far) | fails near p90 |
| K1 (reference) | bkn, cha, sac | -0.253, -0.284, -0.269 | below 0 | +0.116, +0.079, +0.133 (upper +0.247, +0.163, +0.194) | as version 2 | fails near p90 |
| K0 (reference) | bkn, cha, sac | -0.229, -0.267, -0.264 | below 0 | +0.306, +0.205, +0.309 (upper +0.441, +0.308, +0.407) | as version 2 | fails near p90 |

Every form passes the p50, p90 and bias-guard conditions on every pool game. Every form fails the near-third p90
condition on every pool game.

Scored near-third bias, before to after: K1m leaves it unchanged (0.010, 0.367 and 0.213 ft), since it does not shift
the near third. K1t moves it from 0.010 to 0.441 on gsw_bkn_2015, 0.367 to 0.059 on gsw_cha_2016 and 0.213 to 0.135
on gsw_sac_2015.

## Version 3 decision: STOP

No candidate passes the pool rule, so the arm stops, as registered. hou_cle_2016 is kept unspent: score-hou has not
run, no hou_cle_2016 raw file has been read for this arm, and the tool is pinned to STOP, so score-hou refuses.

What the numbers say (post hoc, not tested):

- K1m does not shift the near third, yet the near-third p90 still rises by about 0.035 ft on every game. The rule's
  near third is the player's SportVU third. The shift's cell comes from the output point's third under B. Players
  whose SportVU position is near but whose output point falls in the middle third receive the middle shift (about
  -0.87 ft, pushing them out), and they set the near tail. The margin of +0.05 ft is exceeded at the interval's upper
  end only (+0.070 to +0.089).
- K1t's near shift, fitted on the scored quantity of the two other games, is -0.33 to -0.48 ft. The scored near bias
  itself varies by game (0.010, 0.367, 0.213 ft before), so any near shift carried across games over-corrects a game
  whose near bias is small, as on gsw_bkn_2015. The near third's bias does not transfer across games. The middle and
  far thirds do (scored bias after: 0.012 to 0.191 middle, 0.050 to 0.642 far).
- Forecast check, leave one game out: the K0 calibration predicts the held-out game within 0.003 ft. K1, K1m and K1t
  miss by up to 0.022, 0.035 and 0.031 ft. The cell forms are less predictable from cell medians than the global
  shift. The bias forecast in depth terms is uncalibrated, as registered: its near third before (0.29, 0.47, 0.41 ft)
  runs above the scored near bias (0.010, 0.367, 0.213 ft).

If the lead wants another form, it must be a fourth look at the pool and needs its own registration. One candidate is
a shift that hands off smoothly between the near and middle thirds, so that near players whose output point falls just
inside the middle third do not get the full middle shift. That registration must name everything above.
