# NBA Broadcast-Video Defensive-Stats Pipeline — Project Summary

A computer-vision measurement pipeline that converts **raw NBA broadcast video** into
**per-possession defensive matchup records** — who guarded whom, at what distance, and what
the possession produced — to surface defensive performance that box scores miss. Built as an
independent ML/CV research project on Apple Silicon, with cloud-GPU (Colab) training and
harvesting for the heavy stages.

The chain runs end to end today: **641 joined possessions across 16 games**, with possession
attributions agreeing with independent play-by-play ground truth at **87.8% (n=990)**.

> This document describes what was designed, built, trained, evaluated, and measured.
> Current-state references: `PIPELINE.md` (stage table + residuals), `CLAIMS.md` (what the
> project will and will not claim), `LABEL_SCHEMA.md` (annotation contract), `DEVLOG.md`
> (decisions and dead ends). Report in progress: `paper/`.

---

## Tech stack
**Languages/Libs:** Python, PyTorch, Ultralytics YOLOv8, OpenCV, CLIP (HuggingFace
`transformers`), scikit-learn, SciPy, easyocr, NumPy, Matplotlib, pytest (45 tests).
**Models:** CLIP ViT-B/32 (frozen + linear probe), YOLOv8m (detection), YOLOv8m-pose
(court keypoints), YOLOv8x-pose (person keypoints, teacher), BoT-SORT (tracking), a custom
U-Net (court-line segmentation), Grounding DINO (open-vocab teacher), Claude Haiku/Sonnet
via the Anthropic **Batches API** (label adjudication).
**Data/Infra:** custom annotation tooling, Roboflow datasets, basketball-reference
play-by-play, Google Colab GPU (T4) with Drive-persisted resumable runs, Apple MPS,
content-fingerprinted artifacts, reproducible seeded pipelines with saved splits.

## Pipeline architecture
```
broadcast video
  → [1] court-visibility gate        (live vs dead footage)
  → [2] detection + tracking         (boxes, fragment ids)
  → [3] court homography             (pixels → court feet)
  → [4] identity                     (foot-point stability, fragment linking)
  → [5] teams                        (A/B kit per track, linker veto)
  → [6] possessions                  (approach + set spans, offense/defense)
  → [7] matchups                     (Hungarian pairs on set cores)
  → [8] clock/PBP/outcomes           (real teams, points, terminating events)
  → tier-2 join                      (possession outcome × defenders)
  → credit aggregation               (bootstrapped CIs, context-matched baselines)
```
Each stage is a self-contained module with a stable interface and its own evaluation
harness, so it drops into the chain without rework.

---

## Stage 1 — Court-visibility gate (live game vs. dead ball)
**Problem:** broadcasts constantly cut to replays, close-ups, crowd, ads, timeouts. Feeding
those to the stats pipeline produces empty or garbage output. The gate is a per-frame binary
classifier admitting only usable live-court footage.

**Approach (transfer learning, low-data):** froze a **CLIP ViT-B/32** image encoder and
trained a **logistic-regression head** on its 512-D embeddings — no fine-tuning, so it
generalizes from a few hundred labels without overfitting. Built **two baselines to beat**:
CLIP **zero-shot** and the previously-shipped **HSV floor-color heuristic**.

**Labeling integrity:** hand-labeled **1,050 frames** (659 live / 391 dead) with a custom
keypress tool; CLIP zero-shot used as **weak supervision** to pre-sort for faster human
labeling; a strict **`truth/` vs `predicted/` split enforced in code** to prevent target
leakage; deterministic stratified 70/15/15 split saved to disk.

**Evaluation (cost-sensitive):** per-class P/R/F1, confusion matrix, threshold sweep, PR
curve. Error types labeled by **downstream cost** — false negatives = dropped live frames
(the empty-clip bug), false positives = garbage events — with the threshold tuned on
validation to minimize FN subject to an FP cap. Visual error-analysis contact sheets.

**Result: 98.7% test accuracy** (macro-F1 0.986, 1 FN / 1 FP on 158 held-out frames), beating
CLIP zero-shot (0.886) and HSV (0.873). **Standing caveat:** harvesting runs at threshold
**0.35**, not the validated 0.70 — unfamiliar broadcasts score 0.47–0.66 under domain shift.
The 98.7% belongs to the 0.70 in-domain eval; the 0.35 operating point is protected
downstream by possession structure and PBP cross-validation, not by a frame-level eval.

## Stage 2 — Player detection + multi-object tracking
**Approach:** YOLOv8 detection + **BoT-SORT** tracking in a single streaming pass, with
persistent IDs, broadcast **camera-cut detection/reset**, and an Apple-MPS numerical-stability
guard for the tracker's Kalman filter.

**Evaluation:** hand-labeled boxes on 50 live frames (453 boxes); detection **P/R/F1 @ IoU**
plus label-free tracking diagnostics (players/frame, track-length distribution, ID churn).
Driven by **visual error analysis** rather than aggregate accuracy.

**Key result — external-dataset upgrade:** the generic COCO "person" detector scored
P 0.68 / R 0.74 / F1 0.71, with FPs being referees/crowd and misses being occluded players.
Integrated an external **Roboflow basketball dataset** (654 images), **remapped 10 → 5
classes**, and trained **YOLOv8m on a Colab T4** (~30 min vs. ~22 h locally):

| detector | precision | recall | F1 |
|---|---|---|---|
| COCO person (baseline) | 0.68 | 0.74 | 0.71 |
| **basketball-trained** | **0.89** | **0.87** | **0.88** |

(per-class val mAP@50: player **0.97**, referee **0.99**, rim 0.99.) **Residual:** per-frame
recall ceiling ~0.74 on paint scrums — shown to be a model-capacity limit, not a tuning
problem (see Negative results). IDs are per-shot **fragments**, not players.

## Stage 3 — Court keypoints → homography (pixel → court feet)
**Problem (the hard part):** mapping broadcast pixels to real court coordinates requires a
per-frame homography, which a continuously panning/zooming camera changes every frame. The
prior version of this project never got it working.

**Approach:** built a **guided keypoint annotation tool** and hand-labeled **400 frames** on a
20-point court scheme; trained a **YOLOv8m-pose** keypoint detector; debugged **two stacked
failures** — degenerate bounding-box targets and a **YOLOv8-pose gradient bug on Apple MPS**
(the keypoint head wouldn't train), diagnosed via loss/metric curves and resolved by training
on CPU. Homography solved with **RANSAC** correspondences, plus a homography-based **label-QA
tool** flagging mislabels via leave-one-out reprojection.

**Result: 100% homography success** on held-out frames at **median 0.30 ft (~3.6 in)
reprojection error** (p50 1.7 px over 279 validation frames) — the first time this worked in
the project's history. On unseen games, 66–89% sane-H.

**Refinement (built):** **point + line homography refinement** (ICP-style: project court
lines, snap to detected line pixels, re-solve) with a **monotonic guard** that only accepts
improvements, plus a **learned court-line segmentation U-Net** whose training labels were
**auto-generated for free** by projecting the court template through each frame's homography
(Dice 0.587); ~3× better refinement than the classical baseline.

**Residual:** far-court extrapolation tail (physics p99 ~40 ft); era/floor-design sensitivity.

## Stage 4 — Identity (fragment linking)
Foot-point stability filtering and conservative re-identification to link track fragments.
Label-free metrics: physically **impossible steps 15.4% → 14.3%**, **ID churn 6.57 → 5.0**.
**Residual, stated everywhere:** identity is a **fragment, not a player**, upstream of jersey
OCR; churn remains ~5× by design (the linker refuses uncertain merges).

## Stage 5 — Teams (A/B kit assignment)
Per-track kit classification from crops with a linker veto. **87.1% track-level / 83.9%
crop-level** accuracy on 160 hand labels. **Residual:** ~13% track misassignment (crops
lacking jersey — occluders, face framing), which surfaces downstream as >5-team frames and is
hard-excluded there; A/B is not home/away by itself.

## Stage 6 — Possession segmentation
Trajectories → approach/set spans with basket and offense/defense assignment. **96.5% basket
/ 91.2% offense** on 73 hand labels, rising to **100% / 100% more than 2 s from span
boundaries** — which is why downstream metrics are computed on **set cores** rather than full
spans. **Residual:** ±1–2 s boundary fuzz; no within-occupancy possession change; free-throw
clusters read as sets.

## Stage 7 — Matchups (primary defender assignment)
Hungarian pairing of defenders to offensive players on set cores, yielding matchup distance,
spacing, and closeout measures. Coverage **2.5–4.0 pairs/frame** after a ghost-track audit.
**Honest status: structural checks only — no labeled ground truth.** The labeling attempt was
halted when positional fidelity proved too low to annotate reliably (38 labels discarded, a
second attempt reaching 33). This is `CLAIMS.md` **C1, OPEN**, and it gates all player-level
claims. Closeout is directional only; >5-team frames are hard-excluded (7–62% per possession,
`degraded` flag above 40%).

## Stage 8 — Clock, play-by-play, outcomes
**Scorebug OCR** (easyocr on sparse anchors, one-time clock-box calibration per broadcast
layout) → game clock; **basketball-reference play-by-play** fetch; **anchor alignment** mapping
video spans to real possessions with real teams, points, and terminating events.
Clock reading: **100% on readable crops, 0 confident-wrong** (56 labels). Alignment: **9/9
offense cross-validation, 100% overlap**; period-orientation assignment lifted cross-validation
**69.4% → 93.7%**. **Residual:** ~10% anchor failure (retryable); PBP era ≥ 2000-01;
light-kit=home assumption pre-2017.

## Tier-2 join — possession outcomes × defenders
Joins aligned outcomes to matchup records, producing per-possession credit rows with hard
dedupe and assertions. Correctness verified at **7/7 team-consistency across independent
pipelines**; scaled to **641 joined possessions / 16 games**. Composite funnel yield
**~58% span→join**. Exclusions are **hard filters with visible reasons**, never metadata-only.

---

## Data and labeling infrastructure
- **Human labels:** ~1,500 frames across tasks (gate 1,050; detection boxes 50/453; court
  keypoints 400; teams 160; possessions 73; clock 56) using purpose-built tools — keypress
  classifier, interactive box labeler, guided keypoint labeler, matchup review tool.
- **Label QA:** homography-reprojection mislabel detection; `--repredict` re-scores code
  changes against existing labels; predictions never shown while labeling.
- **Stratified corpus:** 6,197 time-stratified frames drawn from all 21 sources with **no
  model touching the sampling**.
- **External-teacher auto-labeling** (`LABEL_SCHEMA.md`): teachers propose (Grounding DINO
  open-vocab boxes + YOLO-pose keypoints), Claude adjudicates class/tightness/attributes and
  proposes court landmarks; geometry supplies precision via RANSAC through the court template.
  **In-house models are comparators only, never label sources** — self-training inbreeds
  exactly the scrum false negatives the retrain is meant to fix.
- **Delivered dataset (after audit):** 6,159 frames judged, 21 games, 41 failures (0.7%),
  $10.80 via the Batches API. The **300-label human audit (2026-09-12) split the output in
  two**: the **37,916-box agreement band measures 95.0% [88.8, 97.8]** and is the usable
  dataset; every **judge-adjudicated band measures 6.7–20.0%** (adj_player 11.3%, referee
  20.0%, rim 16.7%, scorebug 16.7%, ball 10.0%, backboard 6.7%) and is **NOT training data**,
  along with the attributes riding on it (kit, on_court, occlusion, 4,694 jersey numbers).
  Frame-level shot_type came from the same judge and was not audited — untested, not validated.
- **What the protocol bought:** knowing which half to keep BEFORE a retrain consumed the
  other half. That is what the audit rule exists for.

## Measurement layer (statistics)
- Team-defense credit reported **per 100 possessions with cluster-bootstrapped 95% CIs by
  game** — interval, never point estimate.
- Baselines are **leave-sample-out** and **context-matched**: each possession is scored against
  its own start-type norm (the audit showed 35% of the sample is live-start, not halfcourt).
- **Funnel selection bias quantified** rather than assumed away: a **+8.8 points/100** residual
  outcome-shape offset (free-throw-trip exclusion, made-shot enrichment), with period and
  clock-phase distributions verified clean. Cross-team comparison uses `credit_rel`
  (leave-bucket-out), which differences the common-mode offset out.
- Current headline is deliberately inconclusive: GSW **n=330, −7.4 [−21.6, +6.3]** — a CI that
  straddles zero, reported as such.

## Validation and reproducibility
- **Independent end-to-end arbiter:** possession attributions vs. play-by-play the pipeline
  never sees — **87.8% (n=990)**, with disagreements hard-excluded by construction. This
  doubles as the health canary: a drop means an upstream stage broke.
- **Stale-artifact consumption is structurally impossible** — content-fingerprint guards
  refused a real stale join on first deployment.
- **Byte-identical reproducibility:** a fresh VM re-derived all **151 outcome files
  identically** (fingerprint match 151/151); the alignment chain is deterministic given cached
  anchors.
- **Failure-aware run accounting:** a failed alignment stage now skips join and credit, so a
  broken run can never print plausible-looking tables from partial state; honesty banners fire
  when a queued game builds nothing.
- **Artifact verification by content, not by log line** — macOS AVFoundation silently fails to
  overwrite mp4s, so renders are pixel-scanned.

## Negative results and honest findings
Kept deliberately, because they are evidence the instrument works:
- **The qualification gate rejected its own teacher.** Grounding DINO scored P 0.70 / R 0.94
  against the in-house detector's P 0.956 / R 0.967 on identical human-labeled frames — even
  over-proposing, it missed more real players (27 vs. 15 FN). The gate existed to catch exactly
  this before a $30 adjudication and a retrain on labels worse than the pipeline's own output.
- **Court-masking was a net wash** on single frames — a hopeful prediction refined into a
  data-backed limitation, traced to homography precision degrading away from labeled keypoints.
- **The matchup eval was paused, not shipped**, when the annotator could not reliably correlate
  engine output with footage. Recorded as a finding with its implication (C1 unverifiable at
  current positional fidelity).
- **Detector levers that did nothing:** confidence sweep (+2% recall, 2× crowd FP), NMS IoU
  (no change), yolov8x (recall *drops* to 0.695) ⇒ the ~0.74 scrum-recall ceiling is model
  capacity, not tuning.
- **OCR resolution A/B:** read-rate saturates at 720p (30.8% → 31.0% at 1080p); the added
  resolution buys ~3× tracking length and team-call abstention 53% → 11% instead.
- **Apple-MPS YOLOv8-pose gradient bug:** the keypoint head wouldn't train at all; diagnosed
  from curves and resolved by training on CPU.

## Cross-cutting engineering practices
- **Reproducibility:** fixed seeds, saved splits, model + threshold artifacts, content
  fingerprints, separable training vs. evaluation entry points, 45 tests.
- **Custom annotation tooling:** keypress classifiers, interactive box and guided keypoint
  labelers, automated label QA, weak-supervision pre-labeling, VLM adjudication.
- **Honest, cost-aware evaluation:** held-out test sets, downstream-cost error framing, PR
  curves, visual error-analysis sheets, label-free canaries at harvest scale, and explicit
  reporting of negative results.
- **Claims discipline:** `CLAIMS.md` is the definition of done — every claim ships with its
  evidence artifact and caveats, and a kill list records what is out of scope and why.
- **Hardware/infra:** diagnosed and worked around Apple-MPS limits (pose gradient bug,
  `torchvision::nms` fallback, video-codec quirks); stood up a resumable Colab GPU harvest
  workflow with Drive-persisted state across ~15 multi-hour runs.
- **External-data integration:** remapped third-party Roboflow datasets into project schemas;
  reverse-engineered an external 33-point court scheme's coordinate mapping from the
  open-source library that produced it.

## Quantified highlights
- **End-to-end:** 641 joined possessions / 16 games; **87.8% agreement with independent
  play-by-play (n=990)**; 151/151 byte-identical re-derivation on a fresh Colab VM
  (environment-conditional — a local re-derivation diverges; see DEVLOG 09-11).
- **Gate:** 98.7% accuracy (macro-F1 0.986) on 158 held-out frames; beat two baselines.
- **Detection:** F1 0.71 → 0.88 (precision 0.68 → 0.89); player/referee mAP@50 0.97 / 0.99.
- **Homography:** median 0.30 ft reprojection error, 100% success on held-out frames.
- **Teams / possessions / clock:** 87.1% / 96.5%–91.2% / 100%-on-readable.
- **Data:** ~1,500 frames hand-labeled with custom tooling; 2 external datasets integrated;
  6,159 frames auto-labeled for $10.80, audited at n=300 → a **37,916-box corpus measured at
  95.0% [88.8, 97.8]**; the adjudicated remainder measured 6.7–20.0% and was discarded.
- **Statistics:** +8.8/100 selection bias quantified; all estimates reported with
  cluster-bootstrapped 95% CIs.
- **Models trained:** 7+ (CLIP head, 2× YOLOv8 detection, YOLOv8-pose court keypoints ×2
  schemes, U-Net segmentation, gate-v2 pending).

## Resume bullets — XYZ method (updated 2026-09-11)

*Accomplished [X], as measured by [Y], by doing [Z].* Pick 3–4 for a given role;
the system bullet is the anchor for any of them.

**System (anchor)**
- Built an end-to-end computer-vision pipeline converting raw NBA broadcast video into
  per-possession defensive matchup records, **validated at 87.8% agreement with independent
  play-by-play ground truth (n=990)** across 641 joined possessions, by chaining eight
  independently-evaluated stages — live-frame gating, detection/tracking, court homography,
  team assignment, possession segmentation, matchup pairing, scorebug OCR, and play-by-play
  alignment.

**Classification / transfer learning**
- Eliminated unusable dead-ball footage from the processing stream at **98.7% accuracy
  (macro-F1 0.986) on 158 held-out hand-labeled frames, beating CLIP zero-shot (88.6%) and
  the incumbent HSV heuristic (87.3%)**, by training a logistic-regression probe on frozen
  CLIP ViT-B/32 embeddings with leakage-proof label splits and cost-sensitive threshold
  tuning on validation only.

**Detection**
- Raised player-detection **F1 from 0.71 to 0.88 (precision 0.68 → 0.89)** and eliminated
  referee/crowd false positives, by integrating and remapping a 654-image external
  basketball dataset (10 → 5 classes) and fine-tuning YOLOv8m on a Colab T4 (~30 min vs.
  ~22 h on local hardware).

**Geometry**
- Achieved pixel-to-court coordinate mapping at **0.30 ft (~3.6 in) median reprojection
  error with 100% homography success on held-out frames** — a capability the project's prior
  version never reached — by hand-labeling 400 frames on a 20-point scheme, training a
  YOLOv8m-pose keypoint model, and solving with RANSAC plus ICP-style line-snapping
  refinement guarded to accept only improvements.

**Data engineering / LLM systems**
- Built a two-teacher auto-labeling pipeline (Grounding DINO + YOLO-pose proposals, Claude
  adjudication via the Batches API) that labeled 6,159 frames for $10.80, and **caught its own
  failure before it reached a model**: a 300-label stratified human audit measured the
  unjudged agreement band at **95.0% [88.8, 97.8] (n=100)** and every judge-adjudicated band at
  **6.7–20.0% (n=200)**, retaining a 37,916-box corpus and discarding the rest.

**Judgment / rigor**
- Prevented a corrupted model retrain by building a qualification gate that **rejected the
  candidate teacher on measured evidence (Grounding DINO P 0.70 / R 0.94 vs. the in-house
  detector's P 0.956 / R 0.967 on identical human-labeled frames)**, and made stale-artifact
  consumption structurally impossible via content-fingerprint guards that refused a real
  stale join on first deployment.

**Statistics**
- Quantified the pipeline's funnel selection bias at **+8.8 points per 100 possessions** and
  reported all team-defense estimates as cluster-bootstrapped 95% confidence intervals
  against leave-sample-out, context-matched baselines, by auditing the 58% span-to-join
  yield across period, clock-phase, start-type, and points-mix distributions.

**Infrastructure**
- Ran ~15 multi-hour cloud-GPU harvest jobs on a resumable, Drive-persisted Colab runner
  with failure-aware accounting, **verified by a fresh VM re-deriving all 151 outcome files
  byte-identically (fingerprint match 151/151)**.

## What this project does NOT claim (keep ready — saying it is the differentiator)

- **Matchup assignment has no validated accuracy number.** The labeling attempt was halted
  when positional fidelity proved too low to annotate reliably (38 labels discarded, a second
  attempt reaching 33). No player-level conclusions follow from it. (CLAIMS.md C1, OPEN.)
- **The team-level estimate is not yet conclusive** — the GSW credit/100 confidence interval
  straddles zero.
- **No league-wide or era-normalized claims at any n.** The sample is halfcourt set-cores from
  a nonrandom, GSW-centric slice of 2013–2017 games.

Full ledger with evidence artifacts and standing caveats: `CLAIMS.md`.
