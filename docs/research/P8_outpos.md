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

### Version 2 results

Not computed yet.

