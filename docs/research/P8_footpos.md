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

## Results (computed 2026-11-26, after the definitions above were committed in f8423be1)

Stored files only, no lock, about 45 seconds. Both checks passed: every window's baseline matched entries equal the
registered score file's CPU side, and K1's missed and ghost changes reproduce P8_footbias.md's record. Windows with
scored frames: 62, 53 and 60 (bkn, cha, sac). Record: reports/research/p8_footpos.json.

Rows on the matched players both arms score (baseline, corrected; difference [window interval]):

| Row | gsw_bkn_2015 | gsw_cha_2016 | gsw_sac_2015 |
| --- | --- | --- | --- |
| Position p50 (ft) | 1.539, 1.285; -0.255 [-0.288, -0.215] | 1.511, 1.227; -0.284 [-0.322, -0.235] | 1.496, 1.224; -0.271 [-0.315, -0.217] |
| Position p90 (ft) | 3.300, 3.068; -0.232 [-0.309, -0.154] | 3.082, 2.940; -0.142 [-0.214, -0.070] | 3.218, 3.009; -0.209 [-0.282, -0.119] |
| Near-field bias (ft) | 1.592, 0.653; -0.940 [-0.956, -0.912] | 1.157, 0.062; -1.094 [-1.116, -0.937] | 1.345, 0.255; -1.091 [-1.109, -1.056] |
| Missed | 0.2525, 0.2480; -0.0045 [-0.0110, +0.0021] | 0.2952, 0.2876; -0.0076 [-0.0165, +0.0014] | 0.2060, 0.2144; +0.0084 [-0.0002, +0.0165] |
| Ghosts (non-referee) | 0.1255, 0.1264; +0.0009 [-0.0024, +0.0045] | 0.1876, 0.1844; -0.0032 [-0.0109, +0.0035] | 0.1144, 0.1228; +0.0083 [+0.0031, +0.0134] |

Matched players (baseline, corrected, both): 17,414, 17,364, 17,110 on gsw_bkn_2015; 11,593, 11,534, 11,348 on
gsw_cha_2016; 18,883, 18,803, 18,512 on gsw_sac_2015. The position rows are on the "both" set, 98 percent of the
baseline's. That is why the baseline values here differ slightly from the M6 rows, which are paired with the M6 model
(gsw_bkn_2015's p50 1.539 here against 1.535 there); r23's p50 over all its 17,414 matched players is 1.546. The 2
percent left out cannot move a median by more than a few hundredths of a foot, against gains of 0.25 ft and more.

## Reading by the registered rule

POSITION GAIN holds. On every pool game, held out from its own fit, position p50 falls by 0.25 to 0.28 ft and
near-field bias by 0.94 to 1.09 ft, every window interval wholly below zero. p90 falls by 0.14 to 0.23 ft, which is
reported, not ruled. NON-INFERIOR does not hold, as stated before computing: gsw_sac_2015's missed upper end is
+0.0165 and its ghost upper end +0.0134, both above +0.010. PASS: no.

So the short-foot bias is a real, transferable position error. Post hoc, not ruled: fitted on two games, the correction moves the third
game's near-field bias from 1.16 to 1.59 ft down to 0.06 to 0.65 ft. The first draft's Level 2 aim is "near-camera bias
within 0.5 ft", and two of three games reach it on these leave-one-game-out pool readings; that is a development
reading of an arm that fails NON-INFERIOR, not a Level 2 score. The same shift applied to the foot pixel costs detection on one game,
because matching and missed are scored through the truth camera, where the bias is likely partly absorbed already
(P8_footbias.md, step 1b; not measured here).

## Next step (as registered for POSITION GAIN without NON-INFERIOR)

A correction that protects detection: apply K1 to the court positions the pipeline outputs, and leave foot_stab, so
matching, missed and ghosts are untouched by construction. The pinned scorer derives positions from foot_stab through
the build's H, so it cannot score that arm as it stands. It needs a decision from the lead first. One option is a
report-only scorer variant that maps foot_stab as now and then applies the same court-frame shift before computing err
and toward_camera, beside the pinned rows and never replacing them. The arm would then be registered once, with this
document's pool numbers named as seen, and scored once on hou_cle_2016. That game's position rows are still unseen;
its missed gains from steps 1 (-0.021) and P8_footbias.md (-0.0087) are seen.
