# Architecture Diff — My Pipeline vs. External Colab Project

Comparison of this project against the external Roboflow "Basketball AI" Colab notebooks
(`Extrapolate External Collab Project code/`), and an integration plan for the analytics
layer. **Design only — no implementation yet.**

---

## 0. What the external project is
A Roboflow-tutorial pipeline built on the **`sports`** library + **Roboflow-hosted inference**:

```
RF-DETR detection → SAM2 mask-propagation tracking → TeamClassifier
  → jersey-number OCR (VLM) + ConsecutiveValueTracker (temporal voting) + roster
  → 33-keypoint court model → ViewTransformer homography
  → clean_paths (trajectory correction) → shot events / made-miss
  → top-down court rendering (draw_court / draw_points / draw_paths)
```

It uses **the same two Roboflow datasets I adopted** (`basketball-player-detection-3`,
`basketball-court-detection-2`) — I'm on newer versions (court v22 vs their v14; player v18
vs their v4) — plus the **`basketball-jersey-numbers-ocr`** dataset (I now have v7 locally).

---

## 1. Stage-by-stage diff
| Stage | **Mine** | **Theirs** | Gap |
|---|---|---|---|
| Detection | YOLOv8m (local), 5 classes | RF-DETR (hosted), same dataset family | ~parity; I collapsed action classes (§5 gotcha) |
| **Tracking** | **BoT-SORT** (detect every frame + association) | **SAM2 real-time** (prompt frame 0, propagate masks) | different paradigm — §2 |
| Team classification | none | `TeamClassifier` (crop embeddings → clustering) | **missing** |
| Jersey # / identity | detects `number`, unused | jersey-OCR VLM + `ConsecutiveValueTracker` + roster | **missing** (have the data now) |
| Court keypoints | 33-pt YOLOv8-pose (local) | court-detection-2/14 (hosted) | ~parity (mine newer) |
| Homography | `CourtHomography` (RANSAC `findHomography`) | `ViewTransformer` (same 33 vertices) | parity in math — §3 |
| **Trajectory correction** | none | **`clean_paths`** (jump rejection + Savitzky-Golay) | **missing — key insight** |
| Shot mapping | none | jump-shot class → court loc; ball-in-basket → made/miss; `ShotEventTracker` | **missing** |
| Output | per-frame frame overlay | **top-down court diagram** (players/paths/shots) | **missing** |
| Infra | local MPS + Colab GPU | Colab GPU + hosted inference + `sports` lib | — |

---

## 2. Tracking — the biggest divergence
**Mine — BoT-SORT (tracking-by-detection):** YOLO every frame, then associate via Kalman
motion + appearance/ReID + camera-motion compensation. Handles players entering/leaving and
**broadcast camera cuts** (I built cut-detection + reset). **Weakness (measured):** ID switches
through paint occlusion (`id_churn ≈ 4.5`); no masks.

**Theirs — SAM2 prompt-and-propagate (tracking-by-segmentation):** detect **once** on frame 0
to seed SAM2 (one `obj_id` per player), then propagate segmentation masks via SAM2 video memory.
**Strengths:** stable IDs through occlusion, pixel-tight masks. **Weaknesses (structural):**
roster is **fixed at the prompt frame** (late entrants missed); a **camera cut breaks
propagation** (no re-seed loop); GPU-heavy; detector only re-run periodically for *numbers*.

**Takeaway:** complementary trade-offs. Mine is more robust to roster change + broadcast cuts
(essential for full games); theirs is more robust to occlusion/ID stability on a continuous shot.
**Keep BoT-SORT**, and borrow their **identity layer** to fix ID-churn from a different angle (§5B).

---

## 3. Their homography — and the insight I'm missing
Mechanically **same as mine**: 33-keypoint model → keep keypoints with conf > 0.5 → if ≥4, fit a
point homography (`ViewTransformer(source=frame_kpts, target=config.vertices[mask])`, target =
the same NBA court-feet vertices) → project player **BOTTOM_CENTER** anchor → court (x,y). Per
frame, no smoothing of `H`.

**The missing part is downstream:** `clean_paths(video_xy, jump_sigma=3.5, min_jump_dist=0.6,
max_jump_run=18, smooth_window=9, smooth_poly=2)` — a per-player **trajectory-correction** pass:
- **jump rejection** — flag physically-impossible teleports (jump > `min_jump_dist`, `jump_sigma`
  outlier, over runs up to `max_jump_run` frames) — exactly the symptom of a bad single-frame H;
- **Savitzky-Golay smoothing** of the surviving path.

**This reframes my homography struggle.** I've been fixing bad per-frame H **at the source** (line
refinement, learned line segmentation, 33-pt model, more keypoints). Their answer: accept noisy
per-frame H and **fix it at the trajectory level** — reject jumps, smooth the rest. Simpler, more
robust, and the standard way these systems handle the "nonsensical frame" problem I saw.
**Highest-value thing to borrow.**

---

## 4. What's missing in my pipeline (dependency order)
1. **Player position → court trajectory** as a wired stage (I have tracking + homography but never
   *joined* them into per-ID court-coordinate time series). Prerequisite for everything below.
2. **Trajectory correction** (`clean_paths`-style jump rejection + smoothing) — robustness layer for §3.
3. **Player identity persistence** — jersey-number OCR + temporal voting (I detect `number`, unused).
4. **Team classification** — home/away (ref already separated; team is the remaining split).
5. **Shot mapping** — shot-event detection + court location + made/miss.
6. **Top-down court visualization** — analyst-facing output (dots / paths / shot chart on a court diagram).

---

## 5. Integration plan (design only)

### A. Trajectory correction — *first* (highest value, lowest risk)
- **Slot:** new post-processing stage after tracking + homography. Per `track_id`, accumulate
  per-frame court (x,y), then jump-rejection + Savitzky-Golay smoothing over the series.
- **Why first:** fixes the homography-noise problem downstream and cheaply; makes every later
  metric trustworthy; subsumes much of the value of line-refinement (may not need to finish porting it).
- **Fit:** consumes `tracks.json` (already emitted) + `CourtMapper`; reimplement or adopt
  `sports.clean_paths`.

### B. Player identity persistence — *second* (also patches my tracking weakness)
- **Components:**
  1. **Jersey-number reader** — fine-tune a small **VLM (PaliGemma-2 or Florence-2-base)** on the
     **`basketball-jersey-numbers-ocr` v7** dataset I now have: **3,615 crops** in JSONL
     (`{"image", "prefix":"Read the number.", "suffix":"<number>"}` — the PaliGemma fine-tune
     format; identical prompt to their hosted model). GPU-heavy → **Colab GPU**, same flow as the
     player/court models. This is the from-scratch version of their `basketball-jersey-numbers-ocr/3`.
  2. **number → player matching** by box containment (the detector's `number` box inside a `player` box).
  3. **Temporal voting** (`ConsecutiveValueTracker(n_consecutive=3)` equivalent) — accept a number for
     a track only after N consistent reads.
  4. Optional **roster lookup** → player name.
- **Synergy:** a stable jersey number **re-links a player across a BoT-SORT ID switch** — identity-by-
  number *stitches my broken tracks*. Mine benefits more than theirs (SAM2 IDs are already stable).
- **Pipeline role:** `number` box → crop → VLM reads number → temporal voting → `track_id → {team, number, name}`.

### C. Shot mapping — *third* (mind the gotcha)
- **⚠️ Gotcha I introduced:** the player-dataset 10→5 remap **collapsed `player-jump-shot` /
  `player-layup-dunk` / `player-shot-block` into `player`** — but their shot mapping keys off those
  exact action classes (`JUMP_SHOT_CLASS_ID`, `LAYUP_DUNK_CLASS_ID`, `BALL_IN_BASKET_CLASS_ID`). My
  current detector **can't drive shot detection.**
- **Decision:** (a) **retrain the detector keeping the action variants** as separate classes (lose
  nothing for tracking — just don't collapse them; reuse the Colab GPU flow) — cleaner; or (b) keep
  the lean 5-class detector and detect shots separately.
- **Then:** shot event = jump-shot/layup detection; **made/miss** = `ball-in-basket` near the `rim`
  box; project the shooter's court position at the shot frame (from A/B) → shot chart. Reference:
  `ShotEventTracker` + the `make_or_miss_jumpshot_detection` notebook.

### Recommended sequence
**A (trajectory correction) → B (identity) → C (shot mapping).** A makes positions usable, B makes
them attributable to a named player, C produces the first real "stat." A is also the cheapest/most
reversible.

---

## 6. Asset inventory (analytics layer)
| Component | Data asset | Model type | Status |
|---|---|---|---|
| Trajectory correction | — (algorithmic) | — | ready to build |
| Identity: number reader | **jersey-ocr v7 (3,615 crops)** | VLM (PaliGemma / Florence-2) | data in hand |
| Identity: temporal voting | — (algorithmic) | — | ready to build |
| Team classification | player crops (have) | embedding + cluster | ready to build |
| Shot mapping | player-detection action classes | detection | needs un-collapsed retrain (§5C) |

---

## 7. Strategic choice: adopt `sports` vs. re-implement
The external project leans on the **`sports` library** (`ViewTransformer`, `clean_paths`,
`TeamClassifier`, `ShotEventTracker`, `draw_court`) + Roboflow-hosted inference. I've built these
**in-house** (own homography, own eval harnesses, local models). Per component:
- **Adopt `sports`:** `clean_paths` + court-drawing utilities (pure post-processing, low lock-in).
- **Keep mine:** detection / tracking / homography (invested, run locally, already evaluated).
- **Reference, reimplement in my style:** `TeamClassifier`, `ShotEventTracker`, the jersey VLM
  (train my own on the v7 dataset rather than call the hosted model).

---

## 8. Key takeaways
1. **Keep BoT-SORT** (broadcast cuts) — don't switch to SAM2; borrow their identity layer instead.
2. **Trajectory correction is the missing robustness layer** — it solves the bad-per-frame-homography
   problem downstream, more cheaply than the source-level fixes I'd been pursuing.
3. **The jersey-number VLM doubles as an ID-repair mechanism** for my tracking's main weakness.
4. **Un-collapse the detector's action classes** if I want shot mapping (a remap decision to revisit).
5. Build order: **trajectories → identity → shots**, all training on **Colab GPU**.
