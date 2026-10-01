# LABEL_PLAN.md — where the project needs more labels, how many, and in what order

Deep dive 2026-10-01. Companion to CLAIMS.md (what we may claim) and LABEL_SCHEMA.md
(how labels are made). This answers: *where exactly is the project label-starved,
how much more is needed before any retrain, and which retrains are warranted at all.*

## 1. The finding: the labels are in the wrong place

Nearly every human label the project owns was made on the **seven prototype clips**
from the old repo (`curry_q1_clip`, `curry_classic_clip`, `clip_10m00_18m00` … `clip_70m00_78m00`).
The **fourteen harvested games** that produce the 641 joined possessions, the 87.8%
cross-validation and every published number have almost no stage-level labels of their own.
The only harvest-domain checks are the clock reads (56, across 7 layouts) and the
play-by-play cross-validation, which validates possession attribution and nothing upstream.

| stage | labels today | source | covers the 14 harvest games? | precision of the number |
|---|---|---|---|---|
| 1 gate | 1,050 frames (158 test) | 7 prototype clips | **no** — hence harvesting at 0.35, not the validated 0.70 | 98.7% in-domain only; OOD set = 1 video |
| 2 detection | 50 frames / 453 boxes | 7 prototype clips | **no** | recall 0.865 ± 0.03 — but not on the data that matters |
| 3 homography | 1,858 external Roboflow frames (val 279); 28 held-out for the 0.30 ft | `nbacourt_*` only | **no** — 0 verified frames from any harvest arena | success 28/28 → Wilson lower bound ≈ 88%; unseen games 66–89% sane H |
| 4 identity | 0 | — | — | label-free diagnostics only |
| 5 teams | 160 crops / **31 tracks** | **one clip** (`curry_q1_clip`) | **no** — one kit pairing, one arena | 87.1% at n=31 → Wilson **[71, 95]** |
| 6 possessions | 73 frames (57 scorable) | 4 prototype clips | **no** | 96.5% basket → [88, 99] |
| 7 matchups | **0** usable (38 discarded, 33 untrusted) | — | — | unmeasured (CLAIMS C1 open) |
| 8 clock | 56 reads | 7 broadcast layouts | **yes** | 100% on readable |
| auto-label corpus | 37,916 agreement boxes (95% [88.8, 97.8], n=100 audited) | the 6,197-frame stratified corpus (21 sources) | yes | usable, but see §3.2 |

Two consequences follow. First, every per-stage number on the page is an **in-domain
number for the wrong domain**: it describes the prototype clips, and the 0.35 gate hack
is the visible symptom. Second, the weakest links are not where the smallest numbers are:
teams (n=31, one clip) and matchups (n=0) are the two stages whose errors flow directly
into the credit table, and neither has a harvest-domain evaluation.

## 2. Principles for what to label next

1. **Evaluation labels before training labels.** A retrain you cannot score on the
   target domain is a change, not an improvement. Every set below is first an eval set
   drawn from the 14 harvest games, **held out by game**, stratified so per-game
   diagnostics exist (the per-game cross-validation spread, 0.768–0.967, is the model).
2. **Size by the question.** A 95% Wilson interval of ±5 points around p ≈ 0.85 needs
   n ≈ 250; ±3 points needs n ≈ 650. Decide which you need per stage before labeling.
3. **The agreement corpus does not contain the hard cases.** Boxes where the teacher and
   the in-house detector *agreed* are, by construction, the easy ones. Retraining on it
   will not move paint-scrum recall. Hard cases need human labels, deliberately sampled.
4. **Verification beats free labeling** wherever the engine has a prediction to show
   (matchups, teams, possessions): judging a shown claim is ~10× faster and measures
   exactly the quantity the credit chain consumes.
5. **No retrain is adopted unless it beats the current model on the new harvest-domain
   held-out set, and the end-to-end play-by-play canary does not fall** (LABEL_SCHEMA rule 4,
   CLAIMS adoption rule).

## 3. Per-stage plan

### 3.1 Matchups (stage 7) — P0, the gap that blocks every player-level claim
- **Need:** a measured primary-defender accuracy. CLAIMS C1 asks for n ≥ 150; at p ≈ 0.8,
  n = 150 gives ±6.3, n = 250 gives ±5. **Target 250 verified possessions**, stratified by
  game (the frozen 200-item `data/matchup_eval/sample.json` plus 50 more).
- **Blocker is the tool, not the count.** Both prior attempts stopped because the annotator
  could not match engine dots to footage. Before relabeling: render the claimed pair *on the
  broadcast frame* (boxes highlighted on video, synced to the top-down diagram), with scrub
  and replay. `label_matchups.py --label` already shows both views; the fix is the fidelity of
  the on-video overlay (raw-observation masking, no phantom dots), not a new tool.
- **Effort:** ~15–20 s per item with a working overlay → ~1.5 h for 250.
- **Unlocks:** C1; the first honest reading of how much stage-7 error the credit table carries.
- **Retrain?** No — it is an assignment algorithm. The number may instead change what §7
  of the paper is allowed to say.

### 3.2 Detection (stage 2) — P1, two different label sets for two different jobs
- **Eval set, harvest domain:** 100 frames, ~7 per game, stratified by on-court density
  (oversample frames with ≥ 8 players in the half court — the scrum stratum). ≈ 900 boxes →
  recall ± 3. Tool: `label_boxes.py`, ~60–90 s per dense frame → **~2 h**.
- **Training set for the scrum ceiling:** the 0.74 per-frame recall in scrums is a capacity
  limit (conf, NMS and model size were all swept, §10 of the paper). The 654-image external
  set moved F1 0.71 → 0.88; the residual is occlusion, which the agreement corpus
  under-represents. **300–500 human-labeled scrum frames (~3,000–5,000 boxes)**, drawn from
  the harvest games by a density filter, with partial-occlusion boxes included. Hold out 20%
  by game. Tool: `label_boxes.py` → **~8–12 h**, the single largest labeling cost in this plan.
- **What to do with the 37,916 agreement boxes:** use them as the bulk of the retrain set
  (they are 95% accurate) *plus* the scrum set; never alone. Their value is volume on easy
  frames, which keeps the model from regressing while the scrum labels push the ceiling.
- **Adoption test:** recall on the harvest eval set's scrum stratum must rise with precision
  held; F1 on the existing 453-box set must not fall; cross-validation canary unchanged.

### 3.3 Court homography (stage 3) — P1, zero labels from the arenas we measure
- **Eval set, harvest domain:** 10 wide frames × 14 games = **140 human-verified frames**
  on the 33-point scheme. This replaces "28/28 on prototype frames" with per-arena success
  rates and residuals, and makes the "66–89% on unseen games" figure a measured one.
  Tool: `label_keypoints.py` for a seed, then `verify_projections.py` on snapped overlays
  (~10 s per verified frame once a solve exists) → **~1–2 h**.
- **Training labels:** `label_factory.py` auto-harvests snap-verified frames, but it only
  accepts frames the *current* model can already solve — so it yields least exactly where
  labels are needed (the 1998/2013-style floors, the arenas at 66–69% sane H). For those
  layouts, **30–50 hand-labeled frames each** are required to seed the model; for arenas the
  model already solves, the factory supplies 100+ per game at spot-check cost.
  Target: **400–800 verified harvest frames** across the 8 broadcast layouts, train only;
  val stays the original 279 + the new 140 so benchmarks remain comparable.
- **Adoption test:** sane-H rate and median residual on the 140-frame harvest eval, per
  arena; the 279-frame external val must not regress.

### 3.4 Teams (stage 5) — P1, the weakest evaluation on the page
- **Eval set:** **30 tracks × 14 games ≈ 400 tracks** (≈ 2,000 crops), covering every kit
  pairing in the corpus (dark-vs-dark alternates, the purple-#34 case). Verification-style:
  show the track's crops, press the kit. Tool: `evaluate_teams.py --label`, ~3 s per track
  → **~20–30 min**. Replaces n = 31 [71, 95] with n ≈ 400 (±3.5).
- **Training labels for a supervised kit classifier** (the plan the audit killed, since the
  judge's kit attributes were discarded): the same tracks propagate kit labels to their crops,
  giving ~2,000 labeled crops for free; add **1,000–2,000 more crops** from frames the
  k-means abstains on. 3-class problem; this is enough. **~1 h**.
- **Adoption test:** track-level accuracy on the 400-track harvest eval vs k-means; the
  >5-on-a-team exclusion rate must fall (it is the direct downstream cost of team error).

### 3.5 Possessions (stage 6) — P2
- **Eval set:** **15 frames × 14 games ≈ 200**, stratified over span_mid / span_start /
  pre_start / transition as today. Tool: `evaluate_possessions.py --label`, two keypresses
  per frame → **~20 min**. Replaces 57 scorable from 4 prototype clips.
- **Retrain?** No model — segmentation is rule-based. The labels may change the boundary
  rule or the 2 s trim; they will also give the "100% beyond 2 s" claim a harvest-domain n.

### 3.6 Gate (stage 1) — P2, a coverage problem, not a count problem
- **Eval set:** **30 frames × 14 games ≈ 420** harvest frames, stratified by shot type
  (wide / closeup / replay / graphic / split). Tool: `label_frames.py`, ~1.5 s per frame
  → **~15 min**. This is what lets the operating point be set *in-domain* instead of the
  unvalidated 0.35.
- **Free data, untested:** the judge's frame-level `shot_type` on 6,159 frames was never
  audited. A **200-frame human audit** (~5 min) decides whether 6k gate-v2 labels exist or
  not — the highest value-per-minute item in this plan.
- **Retrain?** Threshold recalibration first; a gate-v2 (5-class shot type) only if the
  audit passes.

### 3.7 Clock (stage 8) — fine
- 56 reads across 7 layouts at 100% on readable. Add **~10 reads per new layout** as games
  are added. No action now.

### 3.8 Identity (stage 4) and jersey OCR (C2) — defer
- Identity accuracy needs MOT-style labels (box + id per frame across cuts). That is the
  most expensive label type here and the label-free churn diagnostics are serving their
  purpose. **Do not label for IDF1 yet.**
- Jersey OCR lost its 4,694 training reads to the audit. Before any OCR work: **~500
  human-read number crops** from harvest tracks (~30 min) as an eval set, then decide.

## 4. Order, totals, and what each step unlocks

| # | set | items | human time | unlocks |
|---|---|---|---|---|
| 1 | gate shot_type audit | 200 verdicts | 5 min | whether 6k gate-v2 labels exist |
| 2 | teams eval | 400 tracks | 30 min | an honest stage-5 number; kit-classifier training data |
| 3 | possessions eval | 200 frames | 20 min | harvest-domain stage-6 number |
| 4 | gate eval | 420 frames | 15 min | an in-domain operating point replacing 0.35 |
| 5 | **matchups** (after the overlay fix) | 250 verdicts | 1.5 h | **C1** |
| 6 | court eval | 140 frames | 1–2 h | per-arena homography truth |
| 7 | detection eval | 100 frames / ~900 boxes | 2 h | harvest-domain stage-2 number, scrum stratum |
| 8 | court training seeds (weak arenas) + factory | 400–800 frames | 2–3 h human | court retrain |
| 9 | **scrum training boxes** | 300–500 frames | 8–12 h | detector retrain against the 0.74 ceiling |
| 10 | OCR number eval | 500 crops | 30 min | whether C2 is worth pursuing |

**Total human labeling: roughly 17–22 hours**, of which the scrum boxes are about half.
Items 1–7 (≈ 6 h) are evaluation and should be finished before any model is retrained;
they are also what turns every number on the page from "in-domain for the prototype clips"
into "measured on the games the results come from."

## 5. Which retrains are warranted
- **Detector:** yes, once items 7 and 9 exist — the ceiling is capacity and the fix is
  targeted occlusion data. Adopt only on the harvest scrum-stratum test.
- **Court model:** yes, once item 6 exists and the weak-arena seeds are labeled. Adopt
  only on per-arena sane-H and residual.
- **Gate:** recalibrate the threshold on item 4; retrain to 5 classes only if item 1 passes.
- **Kit classifier:** new model, small; train on item 2's crops; adopt only against k-means
  on the same held-out tracks.
- **Matchups, possessions, clock:** no retrain; these need evaluation, not models.
- **Identity, OCR:** not yet.

## 6. What not to spend labels on
- More labels on the seven prototype clips. They are already the best-measured domain in
  the project and they are not the domain the results come from.
- More agreement-band boxes. The audit showed they are accurate; it also showed they are
  the easy cases. Volume there does not move the scrum ceiling.
- MOT identity labels before the matchup number exists. The credit chain does not consume
  identity continuity directly; it consumes the primary-defender attribution.
