# P8 placement route: K1 on the position rows (registered before any number)

Written 2026-11-26 after P8_footbias.md found a consistent short-foot bias under the r23 court's H that a correction
could not turn into a lower missed row. Report only: no model runs, nothing is built, trained or changed, no unit.
Tool: tools/p8_footpos.py (rules in its docstring, the same as here); record: reports/research/p8_footpos.json. This
document and the tool were committed before the tool produced any number.

## Question

The scorer's position rows (position p50 and p90, near-field bias) are Level 1's target. They map the foot through the
build's own H, the r23 court model's per-frame H, which is where P8_footbias.md measured a consistent bias: the foot
lands 0.3 to 1.4 ft short of the player, toward the camera. Does P8_footbias.md's K1, unchanged and fitted leave one
game out, improve those rows on every pool game? And what does it cost in missed and ghosts?

## The position rows, quoted from the registered scorer

tools/p8_scorecard.py score_window, for each scored frame (truth mode camera):

    feet = np.array([b["foot_stab"] for b in boxes], np.float32).reshape(-1, 2)
    ct = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Ht).reshape(-1, 2)
    pairs = _match(ct, sv, B.MATCH_FT) if len(ct) else []
    if B.accepted(r) and r.get("H") is not None:
        Hp = np.array(r["H"], np.float64).reshape(3, 3)
        cp_ = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), Hp).reshape(-1, 2)
        for bi, j, _ in pairs:
            e = cp_[bi] - sv[j]
            err_of[j] = (float(np.hypot(*e)), float(e[1] if nh else -e[1]))
            rec["matched"].append({"err": err_of[j][0], "toward_camera": err_of[j][1],
                                   "region": third_of(sv[j][1], nh), "frame": int(f), "sv": int(j)})

What the lines do:

- Ht is the truth camera's H, and sv holds all 10 SportVU players, mirrored.
- _match is sportvu.truth._match: a Hungarian match of truth-frame feet to players within 5 ft.
- nh is sportvu.eval.near_side(Ht).
- Players are matched through the truth camera. Their position error is measured through the build's H.

The rows are computed by tools/p8_e1_score.py rows(), called unchanged:

- Position p50 and p90 are the percentiles of err over the matched players that both arms score (keys frame:sv in
  both).
- Near-field bias is the larger of the near and far thirds' absolute median toward_camera over those players.
- Missed and non-referee ghosts are scored as in P8_footbias.md, which is also how the scorer scores them.
- The difference is paired, corrected minus baseline. The window bootstrap is tools.p8_e1_score.boot (2,000
  resamples, seed 0) over each held-out game's windows.

## Registered definitions

- Games: the pool only (gsw_bkn_2015, gsw_cha_2016, gsw_sac_2015), each scored by the K1 fitted on the other two.
  hou_cle_2016 is not read: its position rows stay unseen for a later registered arm. Guards run on every game,
  SportVU id and window. No held-out game, no D85 test game, no hand labelling. Stored P8-workspace files only, so no
  lock is needed.
- Baseline: the r23 reference-platform sidecars as built. Check: every window's baseline matched entries (keys, err,
  toward, region) must equal the registered score file's CPU-side "matched" to 1e-4, and missed and ghost counts must
  equal it, or the tool stops.
- Corrected arm: K1 exactly as P8_footbias.md defines it.
  - Fit (tools.p8_footbias.fit): on the two other pool games' offset populations, over cells of box height by third.
    The shift is the cell's median depth, or 0 for a cell with fewer than 50 fit rows.
  - Apply (tools.p8_footbias.corrected_px): q = B(p) is moved along the unit vector from the build camera's ground
    point, then sent back to the image by B^-1.
  - The corrected pixels then go through the scorer's lines above unchanged. Matching, missed and ghosts use the
    truth H; err and toward_camera use the build's H.
- Check: K1's missed and ghost changes must reproduce P8_footbias.md's record to 1e-5, or the tool stops.

## Reading rule (fixed now)

- POSITION GAIN: on every pool game, the position p50 difference and the near-field bias difference are below zero,
  each with its window interval wholly below zero.
- NON-INFERIOR: on every pool game, the upper end of the window interval of the missed difference is at most +0.010,
  and that of the non-referee ghost difference is at most +0.010.
- PASS: both hold. p90 is reported, not ruled.

Why these margins:

- +0.010 on ghosts is the margin E1b used, about the size of the change the detector device alone made to r23's
  ghost row (M9: +0.003 to +0.010).
- +0.010 on missed mirrors the adoption bar for a missed gain (-0.010). A position fix may not cost the detection
  rows more than the size of gain a detection fix must show.

Stated before computing, as asked: P8_footbias.md already measured K1's missed and ghost changes on gsw_sac_2015. They
are +0.0084 [-0.0002, +0.0165] and +0.0083 [+0.0031, +0.0134], and both upper ends exceed +0.010. So NON-INFERIOR
cannot hold for K1 as fitted, and PASS is out of reach. What this measurement can still decide is POSITION GAIN.

## What each result leads to

- PASS: not reachable here (above).
- POSITION GAIN without NON-INFERIOR: the bias is a real, transferable position error. The next design is a
  correction that protects detection, for example one applied to the position output only. That cannot be scored by
  the pinned scorer, which derives positions from foot_stab, so it needs a scorer decision from the lead first. It
  would then be one registered arm, scored on hou_cle_2016 once. hou_cle_2016 must be named as already seen for
  missed (step 1's pooled C1 -0.021, P8_footbias.md's K1 -0.0087), though not for position.
- No POSITION GAIN: the build-frame bias does not transfer across games well enough to correct the position rows
  either, and the short-foot bias is closed as a correction target.

## Forking paths, named now

- P8_placement.md steps 1 and 1b and P8_footbias.md: every number in them has been seen.
- hou_cle_2016's gains have been seen: -0.021 (step 1, C1) and -0.0087 (P8_footbias.md, K1). Both are on missed.
- K1's form and fit were chosen in P8_footbias.md after step 1 and step 1b. Here they are used unchanged.

A wording note: the camera truth was initialised from, and fitted to, the V3 label build's H and feet, not the r23
court's H (correction recorded on branch r-p8-footbias). B here is the r23 court's H.

## Results

Not computed yet.
