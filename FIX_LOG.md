# FIX_LOG

Pass-by-pass record for FIX_PLAN.md (newest at the bottom; summary at the top on stop).

2026-10-03: plan written; gates 1 and 2 ticked by Lucien (SportVU download OK, branch track-fix).

## Pass 1 (2026-10-03): A0 frame sidecar
- build_trajectories now writes <clip>_frames.json per build: per processed frame the tracker state (TRACK / LINE_TRACK / HELD / LOST), H, quality_ft, n_inliers, snap res_px and n_match (new read-only attributes on CourtTracker), hull points, and every box with raw and stabilized foot. Output only; no logic path touched.
- Verified: 55/55 triage renders have a sidecar (3,721 frames: 3,234 TRACK, 360 HELD, 119 LOST, 8 LINE_TRACK); trajectories byte-identical to the previous render for all 55 clips; pytest 52 passed.
- Note: a first single-clip determinism test looked nondeterministic; it was a max-frames mismatch (90 vs 79). The build is deterministic. Already visible for A2: 360 HELD frames (9.7%) currently emit positions.

## Pass 2 (2026-10-03): A1 line-overlay score, BLOCKED after 4 variants
- Built qc/track_qc.py + track_qc.py (--triage, --sidecar/--video). Final A1 = ridge-side distance: median px from detected line pixels (outside boxes) to the projected template, share within 6 px, template-side support. reports/qc/<clip>.json for all 55 renders, reports/qc/_ranking.json worst first.
- Variants tried: 3 px template hit rate (every clip ~0.02), template-to-ridge distance at 40 px (every clip ~17 px), ridge-side distance (separates gross failures: the 4 Cleveland + 2 OKC flagged clips rank 1..7 of 55), ridge-side on line-like components only (noisy). Measurable "11 flagged clips in the bottom half" reached 7/11 under every clip summary (median, p90, share_far).
- Finding that matters more than the score: the ridge field sees the painted lines well, and the projected template is 20..40 px off them on nearly EVERY clip (corpus median 42.7 px, best clip 27 px), anchored on the far sideline. "Court off" is a continuum here. A moderate case (splash62_s01 frame 120: real key 100+ px from the template) still reads 38 px because the near sideline drawn through the crowd "explains" crowd and wordmark ridges. The pre-screen is one still per clip and one of its flags (nyk_s04_f9135) is a bad single frame in a clip that is otherwise the corpus's second best, so the yardstick is weak too. B6 must calibrate the score against SportVU; until then it is a worst-first ranking, not a pass/fail.
- Side notes for A2: the maple HSV mask (gate/hsv_baseline.py) is arena-specific (OKC blue floor, Cleveland wine key) and cannot be the floor mask as-is; the solver's own n_match is 14..48 of ~450 template samples with residual ~2.5 px on every clip, which is what random matches in a 6 px window look like, so the snap's "evidence" on 720p footage is weak everywhere. pytest 52 passed.

## Pass 3 (2026-10-03): A2 geometry checks
- Added geometry() to qc/track_qc.py and the per-game roll-up to track_qc.py (reports/qc/_by_game.json). Live rules: scorebug crossing (far sideline / baselines vs the layout's clock+period boxes padded 20 px) and off-court feet (> 1 box mapping outside court +-3 ft). Over 3,721 triage frames: scorebug 19, offcourt_feet 97, no_H 119. Sparse: worst game 18% of frames (xmas16: mostly no_H + off-court feet), the held-out phx 0%.
- Rule (b) floor mask demoted to a measurement after three variants: maple HSV band fails on bright Oracle floors (V saturates at 255, S above the band) and OKC blue; ridge texture reads the sideline itself as texture (good frame far_in 0.31); adaptive per-frame colour from the court interior still puts "2 ft inside the near sideline" in the crowd for the best H's, because of the near-field overshoot, and painted aprons vary by arena. Floor shares stay in the per-frame json for B6.
- Reading: the live A2 rules catch gross failures only when the court leaves the floor; the typical 20..40 px offset passes them all. pytest 52 passed.
