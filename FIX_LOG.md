# FIX_LOG

Pass-by-pass record for FIX_PLAN.md (newest at the bottom; summary at the top on stop).

2026-10-03: plan written; gates 1 and 2 ticked by Lucien (SportVU download OK, branch track-fix).

## Pass 1 (2026-10-03): A0 frame sidecar
- build_trajectories now writes <clip>_frames.json per build: per processed frame the tracker state (TRACK / LINE_TRACK / HELD / LOST), H, quality_ft, n_inliers, snap res_px and n_match (new read-only attributes on CourtTracker), hull points, and every box with raw and stabilized foot. Output only; no logic path touched.
- Verified: 55/55 triage renders have a sidecar (3,721 frames: 3,234 TRACK, 360 HELD, 119 LOST, 8 LINE_TRACK); trajectories byte-identical to the previous render for all 55 clips; pytest 52 passed.
- Note: a first single-clip determinism test looked nondeterministic; it was a max-frames mismatch (90 vs 79). The build is deterministic. Already visible for A2: 360 HELD frames (9.7%) currently emit positions.
