# NBA Broadcast-Video Defensive-Stats Pipeline — Project Summary

A computer-vision pipeline that extracts advanced basketball stats (player positions,
tracking, court geometry) from **raw NBA broadcast video**, built to surface latent
**defensive** performance indicators that box scores miss (off-ball movement, spacing,
matchup distances). Built component-by-component as an independent ML/CV research project
on Apple Silicon, with cloud-GPU (Colab) training for the heavy models.

> This document describes what was designed, built, trained, evaluated, and the measured
> results — for distilling into resume bullets.

---

## Tech stack
**Languages/Libs:** Python, PyTorch, Ultralytics YOLOv8, OpenCV, CLIP (HuggingFace
`transformers`), scikit-learn, NumPy, Matplotlib.
**Models:** CLIP ViT-B/32, YOLOv8m (detection), YOLOv8m-pose (keypoints), BoT-SORT
(multi-object tracking), a custom U-Net (segmentation).
**Data/Infra:** custom annotation tooling, Roboflow datasets, Google Colab GPU (T4),
Apple MPS, reproducible seeded pipelines with saved splits.

## Pipeline architecture (per frame)
```
broadcast frame
   → [1] court-visibility gate   (process this frame, or skip dead-ball footage?)
   → [2] player detection + tracking   (who/where, persistent IDs)
   → [3] court keypoints → homography   (pixel → real court coordinates in feet)
   → (downstream: spatial defensive metrics — future)
```
Each component is a self-contained module with a stable interface and its own held-out
evaluation harness, so it drops into the larger pipeline without rework.

---

## Component 1 — Court-visibility gate (live game vs. dead ball)
**Problem:** broadcasts constantly cut to replays, close-ups, crowd, ads, timeouts.
Feeding those to the stats pipeline produces empty or garbage output. The gate is a
per-frame binary classifier that admits only usable live-court footage.

**Approach (transfer learning, low-data):**
- Froze a **CLIP ViT-B/32** image encoder and trained a lightweight **logistic-regression
  head** on its 512-D embeddings (a "linear probe") — no fine-tuning, so it generalizes
  from a few hundred labels without overfitting.
- Built **two baselines to beat**: CLIP **zero-shot** (text-prompt classification in CLIP's
  image–text space) and the previously-shipped **HSV floor-color heuristic**.

**Data & labeling integrity:**
- Hand-labeled **1,050 frames** (659 live / 391 dead) using a custom keypress labeling tool.
- Used CLIP zero-shot as **weak supervision** to pre-sort frames for faster human labeling.
- Enforced a strict **`truth/` (human-verified) vs. `predicted/` (model guesses)** split
  **in code** to prevent target leakage; deterministic stratified 70/15/15 split saved to disk.

**Evaluation (cost-sensitive):**
- Reported per-class precision/recall/F1, confusion matrix, threshold sweep, and PR curve.
- Labeled the two error types by **downstream cost** — false negatives = dropped live frames
  (the empty-clip bug), false positives = garbage events — and tuned the decision threshold on
  validation to minimize FN subject to an FP cap.
- Generated **visual error-analysis contact sheets** (correct / false-neg / false-pos).

**Result:** **98.7% test accuracy** (macro-F1 0.986, 1 FN / 1 FP on 158 held-out frames),
decisively beating CLIP zero-shot (0.886) and the HSV baseline (0.873).

---

## Component 2 — Player detection + multi-object tracking
**Approach:** YOLOv8 detection + **BoT-SORT** tracking in a single streaming pass, with
persistent IDs, broadcast **camera-cut detection/reset**, and an Apple-MPS numerical-stability
guard for the tracker's Kalman filter.

**Evaluation:** hand-labeled player bounding boxes on 50 live frames; computed detection
**precision/recall/F1 @ IoU** plus label-free tracking diagnostics (players/frame,
track-length distribution, ID-churn rate). Drove decisions with **visual error analysis**
rather than aggregate accuracy.

**Key result — external-dataset upgrade:** the generic COCO "person" detector scored
P 0.68 / R 0.74 / F1 0.71, with error analysis showing false positives were referees/crowd
and misses were occluded players. Integrated an external **Roboflow basketball dataset**
(654 images), **remapped 10 → 5 classes** (player/referee/ball/rim/number), and trained
**YOLOv8m on a Colab T4 GPU** (~30 min vs. ~22 h on local hardware). On held-out clips:

| detector | precision | recall | F1 |
|---|---|---|---|
| COCO person (baseline) | 0.68 | 0.74 | 0.71 |
| **basketball-trained** | **0.89** | **0.87** | **0.88** |

(per-class val mAP@50: player **0.97**, referee **0.99**, rim 0.99). This **fixed the
referee/crowd-as-player problem** and improved both precision and recall simultaneously.

---

## Component 3 — Court keypoints → homography (pixel → court feet)
**Problem (and the hard part):** mapping broadcast pixels to real court coordinates requires
a per-frame homography, which a continuously panning/zooming camera changes every frame. The
prior version of this project never got it working.

**Approach:**
- Built a **guided keypoint annotation tool** (one landmark at a time with an on-screen court
  diagram) and hand-labeled **400 frames** on a 20-point court scheme.
- Trained a **YOLOv8m-pose** keypoint detector; debugged **two stacked failures**: degenerate
  bounding-box targets and a **YOLOv8-pose gradient bug on Apple MPS** (the keypoint head
  wouldn't train) — diagnosed via loss/metric curves and resolved by training on CPU.
- Computed the homography with **RANSAC** point correspondences; built a homography-based
  **label-QA tool** that flags mislabels via leave-one-out reprojection.

**Result:** **100% homography success** on held-out frames with **median 0.30 ft (~3.6 in)
reprojection error** — the projected court template lands on the real court lines. (First time
this worked in the project's history.)

**Rigor / honest negatives:** built and **measured court-masking** (dropping off-court
detections) and reported it as a **net wash on single frames** — refining the claim from a
hopeful prediction to a data-backed limitation, and tracing it to homography precision degrading
away from the labeled keypoints.

**Advanced refinement (built):**
- **Point + line homography refinement** (ICP-style: project court lines, snap to detected
  line pixels, re-solve) with a **monotonic guard** that only accepts an improvement — so it
  can never degrade a frame.
- A **learned court-line segmentation U-Net** to replace a brittle classical line detector,
  with training **labels auto-generated for free** by projecting the court template through each
  frame's homography (Dice 0.587); ~3× better refinement than the classical baseline.

**In progress:** adopting an **850-image external court-keypoint dataset (33-point scheme)** —
reverse-engineered its exact court-coordinate mapping from the open-source library that produced
it — and training it on Colab GPU to further improve homography robustness.

---

## Cross-cutting engineering practices
- **Reproducibility:** fixed seeds everywhere, saved train/val/test splits and model+threshold
  artifacts, separable training vs. evaluation entry points.
- **Custom annotation tooling:** keypress classifiers, interactive bounding-box and guided
  keypoint labelers, plus automated **label-QA** (homography reprojection) and
  **weak-supervision pre-labeling** to accelerate human annotation.
- **Honest, cost-aware evaluation:** held-out test sets, per-class and downstream-cost metrics,
  PR curves, visual error-analysis sheets, and explicit reporting of **negative results**.
- **Hardware/infra:** diagnosed and worked around Apple-MPS limitations (pose-training gradient
  bug, `torchvision::nms` fallback, video-codec quirks); stood up a **Colab GPU training
  workflow** (pulling datasets via the Roboflow API) for all heavy models.
- **External-data integration:** mapped/remapped third-party Roboflow datasets into the project's
  schemas, and bootstrapped one model's labels from another model's outputs.

## Quantified highlights
- Court-visibility gate: **98.7%** accuracy (macro-F1 0.99) on held-out frames; beat two baselines.
- Player detection: **F1 0.71 → 0.88** (precision 0.68 → 0.89) by training a basketball-specific
  YOLOv8 on a cloud GPU; player/referee detection mAP@50 **0.97 / 0.99**.
- Court homography: **median 0.30 ft** reprojection error, **100%** success on held-out frames.
- Hand-labeled **~1,500 frames** across tasks with custom tooling; integrated 2 external datasets
  (~1,500 additional images); trained 5+ models (CLIP head, 2× YOLOv8, YOLOv8-pose, U-Net).

---

## Suggested resume bullets (edit to taste)
- Built an end-to-end **computer-vision pipeline** to extract player tracking and court geometry
  from raw NBA broadcast video (Python, PyTorch, YOLOv8, OpenCV, CLIP).
- Trained a **CLIP-embedding + logistic-regression** frame classifier to gate live vs. dead-ball
  footage at **98.7% accuracy**, beating zero-shot and heuristic baselines; enforced
  leakage-proof human-labeled splits and cost-sensitive thresholding.
- Improved player detection **F1 from 0.71 to 0.88** by integrating and **remapping an external
  basketball dataset** and training **YOLOv8 on a Colab GPU**, eliminating referee/crowd false
  positives; paired YOLOv8 with **BoT-SORT** for multi-object tracking with camera-cut handling.
- Implemented **camera homography** from a YOLOv8-pose **court-keypoint** model + RANSAC,
  achieving **~0.3 ft** median reprojection error; built a U-Net **court-line segmentation** model
  with **auto-generated geometric labels** to refine it.
- Developed **custom annotation and label-QA tooling** (weak-supervision pre-labeling, guided
  keypoint labeler, homography-based mislabel detection) and **reproducible, cost-aware evaluation
  harnesses**; debugged Apple-MPS training limitations and migrated heavy training to cloud GPU.
