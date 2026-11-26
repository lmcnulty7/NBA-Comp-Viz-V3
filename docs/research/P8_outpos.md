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
