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
| 2026-11-25 | E1 design, blind retro review | design review | none | NO-GO: ceiling about 0.02 on missed, no pilot, pseudo-label validation, COCO start | the reviewer catches this failure class before compute |
| 2026-11-24 | E1 seed 0, 3 folds | full (no pilot stage existed) | about 15 units (12.7 T4-h) plus about 2 lost | fails: missed +0.0278, both intervals above zero | validate against truth, pilot first, audit what labels teach as background, start from r23 |
