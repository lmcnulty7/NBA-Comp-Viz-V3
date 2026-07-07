# NBA Broadcast Stats Pipeline

Broadcast video → per-player court-space trajectories → (eventually) defensive
stats. Started as just the court-visibility gate; now contains the full
perception stack (see `DEVLOG.md` for the narrative history):

- **Gate** (`gate/`) — live-play vs dead-ball CLIP classifier (its own section below).
- **Court** (`court/`) — grid-keypoint detector (`models/court_grid_snapped.pt`)
  + line-snap + temporal tracking (`court/snap_track.py`); sub-pixel homography
  on solved frames, tracks through camera pans.
- **Players** (`detect/`) — YOLO detection + BoT-SORT tracking.
- **Trajectories** (`build_trajectories.py`) — foot-points → court feet → cleaned
  per-player paths + synced review video (broadcast | minimap).
- **Label factory** (`label_factory.py`) — self-training harvester: turns any
  broadcast into new court-model training data with no hand labeling.

---

## Operating manual — improving the court model (push-button loop)

Every step is a command + your eyes; run from the project root with
`/opt/anaconda3/bin/python`.

```
1. HARVEST     PYTORCH_ENABLE_MPS_FALLBACK=1 python label_factory.py harvest <youtube-urls...>
               # downloads 6x4-min sections/game, auto-labels frames that the model
               # AND the line evidence solve confidently; prints accept/reject counts

2. SPOT-CHECK  python verify_projections.py data/harvest/<vid>/harvest_rank.tsv \
                   data/harvest/<vid>/review data/harvest/<vid>/harvest_verdicts.tsv
               # worst-first; SPACE = fine, x = bad (bad frames auto-excluded at build)

3. BUILD       python label_factory.py build
               # -> data/court_pose{33,_grid}_v2/ + court_pose*_v2_colab.zip
               # harvest goes to TRAIN only; val is FROZEN at the 279 verified
               # frames so benchmarks stay comparable across retrains

4. TRAIN       upload zips to My Drive root; open train_pose_snapped_colab.ipynb on a
               Colab GPU; set DATASET="court_pose_grid_v2"; run cells 2-4; download
               best.pt -> models/court_grid_snapped_v2.pt

5. BENCHMARK   add the new .pt to the MODELS list in benchmark_court_models.py, then
               PYTORCH_ENABLE_MPS_FALLBACK=1 python benchmark_court_models.py
               # adopt the new model only if H-err p50/p90 improve on the same val

6. DEPLOY      point COURT_GRID_WEIGHTS in config.py at the winning weights
```

Sanity levers: `COURT_TRACKER=0` runs the pipeline with the legacy 33-pt detector
(A/B lever); `unseen_benchmark.py` and `render_overlay_video.py` re-measure
generalization and render watchable overlay clips for any footage dropped in
`data/unseen_videos/`. Known failure modes and their fixes are in `DEVLOG.md`
(2026-07-01 → 07-05 entries).

---

## Component: court-visibility gate

A per-frame classifier that decides **"live game footage"** vs. **"dead ball"**
(replay / closeup / ad / crowd / timeout) — the pipeline's entry filter.

Why it matters: the old project's symptom was **clips returning all-empty
output**. That was a gate problem — live frames were thrown away (false
negatives) before any model saw them. The gate was rebuilt as a *measurable*
component to drive that error down on purpose. NOTE: the trained threshold
(0.70) is tuned for OUR clips; on unfamiliar broadcasts, live frames can score
0.4–0.7, so recall-sensitive uses need a lower threshold (the label factory
uses 0.35 — see DEVLOG 2026-07-05).

---

## Recovered from the old project (Phase 0)

The previous project (`Basketball_Defensive_Vision`, read-only reference) gated
frames with **`_is_court_visible`** at `src/pipeline/runner.py:479`. Key finding:

> **The shipped gate was NOT CLIP zero-shot — it was an HSV floor-color
> heuristic.** It converts BGR→HSV, counts "maple-floor-colored" pixels
> (`lo=(8,25,90)`, `hi=(38,210,245)`), and accepts the frame if **12%–55%** of
> it is floor (too low = crowd/ad/face closeup; too high = floor-level closeup).
> No model, no prompts, no learned threshold.

So **there were no CLIP prompts or thresholds to recover.** Rather than invent a
training story that never existed, we:
1. Reimplement that HSV rule fresh (`gate/hsv_baseline.py`) as the real
   **baseline-to-beat** (the thing that actually shipped).
2. Add a **CLIP zero-shot** baseline (`gate/zero_shot.py`) with sensible default
   prompts (in `config.py`) — the "Approach A" the spec asked for.
3. Train a **lightweight head on frozen CLIP embeddings** (`gate/trained_head.py`)
   — "Approach B".

`court_detector_yolov8n.pt` in the old repo is a **separate** model (a court
bounding-box detector used before keypoint detection) and was **never wired into
the gate**. Frame extraction here adapts the old `VideoReader` pattern (FFmpeg
backend forced to dodge the macOS VideoToolbox/Metal segfault).

---

## Workflow (run in order)

```
extract_frames.py  →  presort_frames.py  →  [YOU verify into truth/]  →  train_gate.py  →  evaluate_gate.py
```

All entry points use `/opt/anaconda3/bin/python` (torch+MPS, cv2, transformers).
Seed is fixed at **42** everywhere and saved into every artifact.

### 1. Extract frames
```bash
python extract_frames.py                       # all 7 clips, 1 frame / ~1.5s, cap 150/clip
python extract_frames.py --interval-sec 1.0 --max-per-clip 200
```
Writes JPEGs to `data/visibility/_unsorted/`, named `{clip}__f{frame_idx}.jpg`.

### 2. Presort with CLIP (labeling convenience only)
```bash
python presort_frames.py
```
Runs the CLIP zero-shot gate over `_unsorted/` and **copies** each frame into
`data/visibility/predicted/live` or `.../predicted/dead`, **prefixing the
filename with CLIP's P(live)** (e.g. `plive0.082__clip_..._f000045.jpg`). Sort by
name to surface low-confidence guesses (P(live) near 0.5) first.

> ⚠️ `predicted/` holds **CLIP's guesses, not labels.** It is **never** read by
> training or evaluation.

### 3. You verify into `truth/` (the only labels)
Eyeball `predicted/`, then **move** each frame you have personally confirmed into
`data/visibility/truth/live` or `data/visibility/truth/dead`. **A frame lands in
`truth/` only because your eyes confirmed it** — correct CLIP freely (a frame
CLIP called "live" can go to `truth/dead`). Aim for **≥ 250 per class**.

### 4. Train
```bash
python train_gate.py                 # logreg head on CLIP embeddings (default)
python train_gate.py --head mlp
python train_gate.py --backbone dinov2   # optional stronger encoder (no zero-shot)
```
- Loads labels **only** from `truth/` (prints per-class counts; warns if < 250).
- Deterministic stratified **70/15/15** split → saved to `models/split.json`.
- Trains the head on TRAIN; tunes **both** gates' thresholds on **VAL** (objective:
  minimize false negatives subject to an FP-rate cap, `--max-fp-rate`, default 0.10).
- Saves `models/trained_head.joblib`, `models/thresholds.json`, `models/split.json`,
  `models/config_record.json`. TEST is left untouched.

### 5. Evaluate (re-runnable without retraining)
```bash
python evaluate_gate.py
python evaluate_gate.py --no-viz
```
On the held-out **TEST** split (100% human-verified by construction), for the
trained head, the CLIP zero-shot baseline, **and** the HSV baseline.

---

## Labeling integrity (do not violate)

- `truth/` is the **only** source of labels for train / val / test.
- `predicted/` is **never** read by `train_gate.py` or `evaluate_gate.py` — this is
  asserted in code (`gate/labels.py`, `load_test_split`). Grading a model on
  `predicted/` would measure CLIP against its own guesses: circular, ~100%
  meaningless, target leakage.
- If `truth/` is empty (or far smaller than `predicted/`), training and evaluation
  **error out** rather than silently falling back.

---

## Reading the reports

`reports/metrics.json` (machine) and `reports/metrics.txt` (human) contain, on TEST:

- **Accuracy / precision / recall / F1**, per class and overall, + the 2×2 confusion matrix.
- The two error types **by downstream cost**:
  - **FALSE NEGATIVE** = a *live* frame called *dead* → frame skipped → **empty/under-counted clips** (the current bug).
  - **FALSE POSITIVE** = a *dead* frame called *live* → replay/closeup/ad processed → **garbage events** in the box score.
- A **threshold sweep** table (FN vs FP as the threshold moves) and a PR curve
  (`reports/viz/pr_curve.png`). The recommended threshold minimizes FN subject to
  an FP cap and is **tuned on VAL, never TEST**.
- A **side-by-side** of trained head vs. zero-shot vs. HSV on the same TEST frames.

`reports/viz/` contact sheets (green = correct, red = wrong; annotated true vs.
predicted + confidence):
- `correct_samples.png` — sanity check.
- `false_negatives.png` — **live frames the gate skipped** (the visual of the empty-clip bug).
- `false_positives.png` — dead balls the gate let through.

---

## Layout

```
config.py            seed, paths, CLIP model, prompts, HSV constants
gate/
  backbones.py       frozen CLIP (default) / DINOv2 (flag) encoders → embeddings
  zero_shot.py       Approach A: CLIP zero-shot gate
  trained_head.py    Approach B: logistic/MLP head on frozen embeddings
  hsv_baseline.py    recovered shipped HSV gate (baseline-to-beat)
  labels.py          truth/-only label loader + integrity guards
  common.py          image IO, embedding cache, metrics, threshold selection
extract_frames.py    Phase 1: mp4 → _unsorted/ frames
presort_frames.py    Phase 1: CLIP guesses → predicted/ (labeling convenience)
train_gate.py        Phase 3: split + train + tune thresholds
evaluate_gate.py     Phase 4 metrics + Phase 5 contact sheets
```

Both gates expose the same interface so either drops into a bigger pipeline
untouched:
```python
gate.score(frame_bgr)                       # -> P(live) in [0, 1]
gate.is_court_visible(frame_bgr, threshold) # -> bool
```

---

# Component 2 — Player detection + tracking

Detects players and assigns persistent IDs across frames, running on **contiguous**
video (IDs propagate frame-to-frame). YOLOv8m + **BoT-SORT** in one ultralytics
call; person class only (ref/player split is a later concern). Runs on gate-passed
frames in the full pipeline.

```
detect/
  tracker.py      PlayerTracker — streaming update(frame, idx, ts) -> list[Track]
  detector.py     stateless single-frame PlayerDetector (used for eval)
  camera_cut.py   reset IDs when the broadcast cuts (mass track vanish)
  types.py        Track (bbox, id, foot_point; team_id/court_pos slots for downstream)
run_tracking.py       run over a clip → overlay mp4 + tracks.json + diagnostics
label_boxes.py        draw GT player boxes on ~50 live frames (lightweight eval)
evaluate_tracking.py  detection precision/recall@IoU vs those boxes
```

### Run it
```bash
PY=/opt/anaconda3/bin/python
$PY run_tracking.py --source ".../curry_q1_clip.mp4" --max-frames 300   # demo overlay + diagnostics
$PY run_tracking.py --source ".../curry_q1_clip.mp4" --use-gate         # gate + track together
# lightweight eval:
$PY label_boxes.py            # hand-draw boxes on ~50 live frames (drag=box, n=next, u=undo, q=quit)
$PY evaluate_tracking.py      # precision/recall@IoU → reports/tracking_metrics.{json,txt} + viz
```

### Reading the tracking diagnostics
`run_tracking.py` prints **label-free** health metrics: `avg_players_per_frame`
(~8–12 on live NBA wide shots), `id_churn_ratio` (near 1 = stable IDs, high = ID
switches/churn), `camera_cut_resets`, track-length distribution.
`evaluate_tracking.py` adds quantitative **detection** P/R/F1 vs hand-labeled boxes
(FN = a real player missed; FP = a phantom detection). **ID-switch rate is not
measured** — that needs box+ID MOT labels we opted out of; use `id_churn_ratio` as
the proxy.

### Apple-Silicon notes
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `config.py` (YOLO's `nms` op isn't on MPS).
- Overlay videos use the `avc1` codec (the only one that writes readable .mp4 on this
  OpenCV build).

---

# Component B — Player identity (re-ID) + foot-point stability

After the court fix, trajectory teleports are dominated by **BoT-SORT ID
fragmentation** (one player = many track fragments across occlusions/cuts) and
**bbox foot-point jitter** (the box bottom moves, not the player, and perspective
amplifies it). Three independent levers, all wired into `build_trajectories.py`:

```
botsort_reid.yaml     native BoT-SORT appearance re-ID (detector features, ~free)
                      → within-shot ID stability.       Lever: TRACKER_REID=0 env
detect/footpoint.py   FootPointStabilizer — occlusion height-clip correction +
                      EMA, in pixel space BEFORE the homography.  Lever: --no-stab
detect/reid.py        offline fragment linker: CLIP torso embeddings (gate's
                      frozen backbone, reused) + court-space motion feasibility
                      (endpoints in FEET are comparable across camera cuts) +
                      ambiguity margin (teammates → refuse).      Lever: --no-reid
```

Also fixed here: track-ID collisions — ultralytics restarts IDs at 1 on every
tracker reset, so IDs from different shots used to alias different players in
anything keyed by `track_id`. `PlayerTracker` now offsets IDs to be globally unique.

### Run it / A-B it
```bash
PY=/opt/anaconda3/bin/python
V=".../Basketball_Defensive_Vision/data/raw/curry_q1_clip.mp4"
$PY build_trajectories.py --source "$V" --start 11520 --max-frames 200 --stride 3 --use-gate
# baseline (Component B off):
TRACKER_REID=0 $PY build_trajectories.py --source "$V" --start 11520 --max-frames 200 \
    --stride 3 --use-gate --no-reid --no-stab
```
Every run prints the label-free **physics report** (impossible-step %, p50/p99
step, off-court %) and the **identity report** (fragments → canonical tracks,
merges, ambiguity refusals, id-churn before/after), and writes
`data/tracking/<clip>_identity.json` — the audit trail with the evidence
(similarity / gap / distance / implied speed) for every accepted merge and every
refusal. A false merge poisons two players' stats, so the linker is conservative
by design: motion gates first, appearance breaks ties, ambiguity refuses.

---

# Component C1 — Team classification (unsupervised)

`detect/teams.py` — k-means (k=2) on **pooled Lab jersey color, one feature per
track**, from the torso crops re-ID already collects. Two measured dead ends
before this landed (see DEVLOG 2026-07-05c): CLIP embeddings cluster by scene
statistics on tiny broadcast crops (not jersey), and per-crop color features are
a coin flip because single crops are up to half floor/apron — pooling a track's
10–40 crops makes its own jersey the dominant pixel mass. Floor pixels are
masked with the old gate's maple-HSV band first. Fit is per clip;
nearest-centroid assignment with an **abstention rule** (near/far
centroid-distance ratio > 0.75 or <2 crops ⇒ team `None` — bench/crowd leakage
stays out of both teams). Cluster ids are team **A/B**, not home/away.

Wired into `build_trajectories.py` (lever: `--no-teams`):
- **linker veto** — candidate merges across teams are refused outright
  (`"reason": "team_veto"` in the audit json), which also shrinks candidate
  sets so the ambiguity margin refuses less;
- `"team"` on every track in `<clip>_trajectories.json`; team diagnostics
  (silhouette, per-team track counts, per-frame balance — expect ~5v5) in the
  identity report;
- the review video colors boxes + minimap dots by **median jersey color**;
- `reports/viz/team_clusters_<clip>.png` — contact sheet of crops grouped by
  assigned team, for eyeball verification of the clustering.

No labels needed to run. To **quantify** accuracy, use the human-label eval
(`evaluate_teams.py` — export ~200 stratified crops → keypress-label them
w/n/a → report):
```bash
$PY evaluate_teams.py --export --start 11520 --max-frames 200   # data/teams_eval/<clip>/
$PY evaluate_teams.py --label    # w=white n=navy a=no-team; u undo, s skip, q quit (resumable)
$PY evaluate_teams.py --report   # confusion, per-team P/R, crop- + track-level accuracy
```
Predictions are never shown while labeling (truth/predicted separation, as in the
gate). Crop-level accuracy is a lower bound (occlusion crops show the opponent);
the track-level majority number is the fairer read. Refs never reach the
classifier (detector tracks class 0 = player only; refs are class 1) — anything
that leaks through typically abstains, but by geometry, not by rule.
