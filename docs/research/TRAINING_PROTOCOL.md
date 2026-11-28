# Training protocol for this repository (PLAN.md D172)

Every training run here follows the user-scope standard `ml-training-rigor` (a Claude Code skill outside the
repository) and is reviewed before any compute by the user-scope `training-design-reviewer` agent, given the design
and the code, not the lead's conclusion. This file adds what is specific to this repository. It was written after E1
(D171), which spent about 15 Colab units and failed on its key row; the checks below would likely have stopped it
for about 1 to 2 units.

## The steps, with this repository's tools

1. **Design, committed before compute.** A doc under docs/research/ (as P8_E1.md, P8_truth_v2.md) and a PLAN.md D
   line: hypothesis and mechanism; an upper bound on the effect from the data; the decision metric (the pinned
   scorecard, tools/p8_scorecard.py, D88, through the experiment's scorer, as tools/p8_e1_score.py); the adoption rule
   (D5 as amended with D158: worse when the window or the game interval lies wholly on the bad side of zero; improves
   on the point estimate); kill criteria per stage; the noise floor; the budget per stage in units.
2. **Pre-flight on the Mac, no GPU.**
   - Guards in code: tools.p8_common.guard (spent held-out games) and tools.research_testset.refuse_cv_test (the D85
     test set) on every game list; folds split by game; validation from training games only.
   - Label audit against SportVU on the training frames: label recall by cause (court third, box size, image edge,
     occlusion), and what is left unlabelled and so taught as background.
   - Initialisation: r23 (models/player_detector.pt) unless the design shows a generic base is needed.
   - Baseline: the scorer reproduces r23's recorded rows on the same windows (as the E1 stub did).
   - Device: registered builds run the detector on CPU (D175). A new detector or library version first passes a
     device parity check on the corpus JPEGs (tools/p8_device_parity.py, D174); MPS on this Mac corrupts boxes.
   - Mechanics: the label files render the boxes the manifest says (counts per frame), a tiny-batch overfit, the loss
     falls on a short run.
3. **Ladder, gates written in the design.** Smoke (minutes, scored end to end), then a pilot (one fold, short
   schedule, scored on that fold's held-out game with the real scorer), then the full first seed, then the other seeds
   only if the first adopts. Extra seeds are never run when the adoption rule can no longer pass (D171).
4. **Instrumented training.** A truth-aligned validation number during training (recall and ghosts against SportVU on
   a few training-game windows) besides the framework's mAP; the checkpoint rule recorded (Ultralytics 8.4.48 picks
   best.pt by mAP50-95 alone, on the training labels; older versions weighted 0.1 mAP50 plus 0.9 mAP50-95).
5. **Statistics.** Paired against r23 on the same windows; window and game cluster bootstrap (2,000 resamples, seed
   0); with 3 games the game interval is close to the range of the game effects. The D85 test set is scored once, at
   the end of P8 (D129).
6. **Diagnosis, win or lose,** in the result doc: result against expectation, error analysis by cause, matched-set
   and denominator counts, the lesson added to TRACKING_LEVELS.md "Lessons and priors".
7. **Ledger.** One line per run below.

## Colab

Runbook: docs/research/P8_E1_colab.md section 5 (HTTP transfer with hash checks, a lock file against duplicate
launches, a check at least every 8 minutes). Measured cost: at most about 1.15 units a T4-hour (D171). Cap: 60 units
for Levels 1 and 2 (D169); Level 3 GPU work needs a cap proposal to Lucien first (D172).

## Ledger

| Date | Run | Stage | Compute | Outcome against its gate | Lesson |
| --- | --- | --- | --- | --- | --- |
| 2026-11-28 | posefoot (P8_posefoot.md v1.1 to 1.3 and section 15), a per-box foot from ViTPose-B ankle keypoints, the hybrid foot_stab + (lifted ankle pixel - foot_raw), report only | stage 0a (foot_raw), stage A (decode alignment, 151 of 156 windows), stage 1 pose smoke, stage 2 falsifier | about 2 lock minutes (0a), 21 (A), 4 (smoke), 67 (stage 2); no units | stage 2 fails rules 1, 2, 4 and 5 on both games (R - R_null +0.047 and +0.063 against 0.10; median under T -0.37 and -0.45 ft against -0.5; hits lost 0.42 and 0.41 against 0.05; est +0.28); rule 3 passes; the pose line stops; hou_cle_2016 unspent | post hoc: the ankle point carries a large frame-wide offset from the box bottom (hits move about 1.3 ft), not a per-box signal (gain over a same-frame null about 0.05); a falsifier against a same-run null and a hit control catches this; reproduce the build's own read path before trusting any decode |
| 2026-11-27 | footfix (P8_footfix.md v1.2), a foot-pixel correction toward the decided truth's image of the player, F1 depth by cell and F2 box-height ratio (control) | stage 0 (smoke, reproduction, audit, in-sample falsifier G0) | 474 lock seconds of scoring plus 71 s of report, stored files, no units | G0 FAIL, KILL: in sample F1 missed -0.0039 (gsw_cha_2016) and -0.0035 (gsw_bkn_2015) against -0.010; rule 7 (one-to-one missed at most -0.0075) failed on both (-0.0018, -0.0018); near-third p90 +0.097 ft on gsw_bkn_2015 against +0.05; no stage 1; hou_cle_2016 unspent | post hoc: median shifts of 0.09 to 0.87 ft move missed (under T) by about 0.004 even fitted in sample, while p50 (under B) falls about 0.15 ft; consistent with per-box scatter, or with a P-ii offset unlike the hits' (the cell medians are set by stage A hits, 86 percent of fit rows), or with ghost or neighbour boxes; not separated here; on these truths the two median forms did not reach the bar (about -0.004 in sample); other population shifts were not tested, and D188 bars tuning new ones on the pool; run the in-sample falsifier first, it cost 8 lock minutes |
| 2026-11-27 | reset-fix arm (P8_resetfix.md), pool builds and scoring, r23 with the tracker reset fixed | full (builds and score) | 1.80 lock hours of builds, 146 s of scoring (the score record), no units | no pass: no pool (fixed truth: cha unused, sac apart, bkn in use); bkn reported only: net error -0.049 [-0.087, -0.012], missed -0.060 [-0.107, -0.016], ghost row (per box) +0.0008 [-0.0024, +0.0033]; bracket (frames both truths accepted) net error -0.020 [-0.061, 0.000], missed -0.023 [-0.071, 0.000] | a truth rebuilt on the arm's own tracker must pass acceptance on every pool game first; on the one game in use the fix lowers missed and net error, the ghost share per box is unchanged while ghost boxes per in-frame player rise from 0.115 to 0.126 (point values) |
| 2026-11-26 | reset-fix arm (P8_resetfix.md), smoke retry 1 | smoke (retry 1 of 2) | 601 lock seconds, no units | passes every gate: boxes match a3input's fixed variant on 1,053 of 1,053 frames, 2 resets logged (frames 312 and 6,534, stripped 2), no-reset window identical to the baseline | confirms the double-import cause of smoke 1 |
| 2026-11-26 | reset-fix arm (P8_resetfix.md), smoke 1, r23 with the tracker reset fixed, reference platform | smoke (retry 0 of 2) | 598 lock seconds, no units | fails its gate: the arm's boxes match a3input's fixed variant on 1,053 of 1,053 frames (8,371 boxes), but gsw_cha_2016_s05_w00's reset log is empty against 2 resets in its identity file, failing the reset-count, record and prefix gates; cause (post hoc until the rerun): the build child ran the module as __main__ and p8_device imported it again, so two logs | a log written at exit must live in the canonical module, not in whatever runs as __main__; test the real entry path |
| 2026-11-26 | E1b transfer gate | design gate (section 2b) | none (stored files, seconds) | fails: perfect-imitation build ceiling -0.0199 pooled, -0.0079 on gsw_cha_2016 (needed -0.020 and -0.010 on every game); E1b stops before any unit | measure transfer to the scored row before sizing a label set; placement gains keep about 0.3 at build level |
| 2026-11-26 | E1b design | design and upper bound | none (stored files, seconds) | teacher-label ceiling -0.080 raw, 31 percent of misses, mostly placement | the reachable part of misses is mostly foot placement, which the build partly corrects already |
| 2026-11-26 | E1R, E1 seed 0 on the reference platform (CPU detector), evaluation only | smoke, Stage 1, Stage 2 | about 13.7 lock hours (smoke and scoring 0.1, Stage 1 4.7, Stage 2 8.7; workspace records e1r/rebuild__*.json), no units | smoke gates pass; Stage 1 passes (primary and blind; missed -0.0067 pooled, -0.0019 blind); Stage 2 fails its gate (2 of 3 folds on hou_cle_2016; fold gsw_sac_2015 missed +0.0014) | check device parity before scoring a model; seed 0, point estimates, post hoc: on the right platform E1's gains are ghosts and position (team weak), not recall |
| 2026-11-25 | E1 design, blind retro review | design review | none | NO-GO: ceiling about 0.02 on missed, no pilot, pseudo-label validation, COCO start | the reviewer catches this failure class before compute |
| 2026-11-24 | E1 seed 0, 3 folds | full (no pilot stage existed) | about 15 units (12.7 T4-h) plus about 2 lost | fails: missed +0.0278, both intervals above zero | validate against truth, pilot first, audit what labels teach as background, start from r23 |
