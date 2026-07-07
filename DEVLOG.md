# Dev Log — NBA Comp Viz pipeline (started as the Court-Visibility Gate)

Narrative history of decisions, rationale, dead ends, and open threads. This is
**not** the how-to (see `README.md`) — it's the "why did we do it this way and
where were we" record for when you come back to this cold.

**How to maintain:** add a new dated entry at the TOP each working session. Keep
the *reasoning*, not just the *what* — future-you can read the code for the what.

---

## 2026-07-07c — Ghost-track audit: user caught >5 same-team dots; two mechanisms, both fixed/gated

User review of the matchup video found >5 same-color dots simultaneously —
basketball-impossible, Hungarian was matching against corrupted input. Audit
across all 4 clips: **all 10 eligible possessions affected** (25–100% of core
frames had a team >5). Two mechanisms, each with direct evidence:

1. **Interpolation ghosts (dominant, 20–30% of ALL points)**: clean_paths
   linearly interpolates a track's internal observation gaps and NOTHING marks
   those points — so an occluded fragment keeps emitting a gliding phantom
   alongside its re-detection (e.g. q1 frame 12051: ghost of track 86 — one of
   the known navy-as-white tracks — plus 6 real same-team tracks). FIX:
   raw-support filter in BOTH consumers (segment_possessions + matchup_metrics)
   — a cleaned point is used only at frames where the track appears in its RAW
   series. Smoothing kept; invented presence dropped.
2. **Team misassignment inflow (C1's measured ~13% track error)**: all-raw
   6v4-signature frames — a mislabeled player inflates one team while the
   other runs short. Can't be fixed per-frame; GATED: matchup engine now
   hard-excludes any frame where either team has >5 tracks (>5, not ≠5 —
   labeled coverage is legitimately 4v4-ish because abstained tracks can't
   play). Excluded frames stay visible: red banner in the review video,
   per-possession counts + team-count distribution in the JSON, and a
   `degraded` flag when >40% of the core failed (or <30 frames survived).

Downstream effects, re-validated end to end:
- **C2 IMPROVED with the ghost filter** (ghosts polluted occupancy too):
  attacked basket 93.0 → **96.5%** vs the same 73 labels; offense 91.2%
  (within one-sample noise). No regression anywhere.
- C3 rerun: 10 possessions, per-possession exclusion 7–62%; exactly 1 flagged
  degraded (q1 @right, 62% excluded — the window with the known team-label
  contamination). Coverage now honest at 2.5–4.0 pairs/frame — the previous
  4–5 was inflated by phantom pairs, so pre-audit Tier-1 numbers were
  optimistic and are superseded by this run.
- Lesson recorded: any consumer of trajectories JSON must distinguish OBSERVED
  from interpolated presence (use the raw series as the mask). The cleaned
  series alone overstates who was on the floor.

Tier 2 sequencing remains held per user until this audit is accepted.

---

## 2026-07-07b — Component C3 Tier 1: matchup metrics on certified set-cores

First component that computes an actual defensive stat. `matchup_metrics.py`
consumes trajectories + possessions and works ONLY on span cores (`core_*`
fields added to the span schema: set minus the last 2 s — the region the C2
eval certified at 100%; cores < 4 s ⇒ `metrics_eligible=false`, flagged not
dropped — 2/12 current spans). Additional gates, all visible in the output:
offense confidence ≥ 0.60 (dropped 1 span at 0.57), team-labeled tracks only.

Tier 1 metrics (per possession):
- **Matchup assignment**: Hungarian offense↔defense on court distance per
  core frame; per-defender aggregation by (defender, man) pair-time.
- **Matchup distance**: median/mean to assigned man; primary man = most time.
- **Spacing conceded**: offense convex-hull area, median + IQR.
- **Closeout tendency**: DIRECTIONAL only (closing/holding/retreating shares,
  median rate) on rolling-median-smoothed ≥1.5 s same-pair runs with a 1 s
  central difference. Explicitly the shakiest metric — it differentiates
  noisy positions; never read it as precise ft/s.

Review harness (built first): `<clip>_matchups.mp4` — top-down per-possession
animation with team-colored dots, offense hull, matchup lines + distances —
for eyeballing assignments before trusting aggregates.

First-run sanity (4 clips, 8 possessions): matchup distances basketball-
plausible (on-ball 2.4–5.5 ft, sag/off-ball 10–15 ft); coverage 4.0–5.1
pairs/frame (abstained tracks cap it below 5v5); one spacing outlier
(1013 ft² on a 4.2 s core — early-clock spread or a stray track; visible in
the review video).

Known limitations, stated:
1. **Defender identity = track fragment.** Churn splits one player across
   fragments within/between possessions (q1's first possession lists 10
   "defenders"). Per-possession attribution is sound; cross-possession player
   PROFILES need jersey-number OCR / roster mapping (future).
2. Coverage < 5v5 because abstained-team tracks can't be matched.
3. Closeout tendency inherits foot-point + homography noise; smoothed and
   bucketed, but still the first metric to distrust.

Deferred by scope decision (user): help-position distance (encodes an
unverifiable positioning assumption — no ground truth to check it against)
and off-ball attentiveness (needs real ball position; a camera-pan proxy is
not quantitatively trustworthy — deliberately NOT built).
**Tier 2 (designed, not built): possession-outcome-adjusted defensive credit**
— tag each core possession's outcome and attribute it to the primary defender
via the matchup assignment ("did possessions this defender guarded run worse
for the offense than average"). DEPENDS ON outcome tagging (scoreboard OCR or
a shot-quality model) which does not exist yet — sequence that first.

---

## 2026-07-07 — C2 human-eval verdict + boundary-semantics fix: 93%, and 100% away from boundaries

User labeled the 73 frames → first result: 86.8% basket / 86.8% offense (n=53),
pre_start 0%, 11/13 transition-preds hid a visible set. Diagnosis flipped every
finding into ONE defect:
- pre_start "0%" was n=2, both frames sitting inside the PREVIOUS span while
  the human read the forming set — a boundary artifact, not a heuristic error.
- 9/11 coverage misses were frames exactly 1 s before a span we DID find.
- ALL 5 "wrong-basket" span_mid errors were 1.2–2.3 s before their span's end
  (the sampled frame shows players streaming back after the possession ended —
  e.g. curry_q1 frame 12378: shot clock freshly reset to 22, floor in motion).
  The 4-vs-1 direction split is noise, not clip- or camera-specific.
**Every error was a LATE BOUNDARY; zero mid-set mistakes.** Trailing players
hold the all-player median past the ±8 ft band after play turns.

Fix = boundary semantics v2 (segment_possessions.py), matching basketball:
possession = APPROACH + SET. Span ends at RETREAT ONSET (sustained ≥4 ft/s
occupancy motion toward midcourt for ≥0.7 s); the outbound frames + following
transition become the NEXT possession's approach (`set_start_frame` recorded).
Offense/defense + confidence computed on SET frames only — during the approach
the offense LEADS, legitimately inverting the closer-team geometry. No new
spans can be created (the band still gates span existence) ⇒ no new false
positives, verified: span_start stayed 100%.

Re-scored against the SAME 73 labels (new --repredict): **93.0% basket / 93.0%
offense (n=57)**; pre_start 0→100%, transition-missed 11→7. Decomposition:
- **>2 s from any boundary: 32/32 = 100% on BOTH metrics.**
- ≤2 s from a boundary: 21/25 = 84% — the residual is ±1–2 s disagreement
  about the exact instant possession flips, which is genuinely fuzzy even for
  a human. C3 GUIDANCE: treat per-frame attributions within ±2 s of a span
  boundary as uncertain (use the set core; `set_start_frame` is provided).
- Remaining 7 coverage misses: 4 window-leading edges + 1 across a gate gap
  (approach can't carry over a dead-ball cut, by design) + 2 in curry_q1's
  13038–13236 tail — the ONE real segmentation loss (~7 s): occupancy hovers
  at the band edge (p50=53 vs band 39–55) during a messy stretch. Documented,
  not chased — it's 1 stretch in 150 s.

C2 is validated to the C1 standard. Next: C3 (matchup metrics) on set cores.

---

## 2026-07-06d — C2 validation prep: 4-clip diagnostics, failure-mode anatomy, human eval built

User-directed: C2 gets the C1 treatment before C3 depends on it. Batch-ran the
pipeline (new `--no-video` lever) on 4 clips / ~150 s of live footage → 12
halfcourt spans. Label-free findings:

- **Internal consistency 100%**: within every clip, each team always attacks
  the same basket and offense flips with the end — exactly the structure real
  halves have. Strongest label-free signal available.
- **Shot-clock sanity**: all 12 span durations 3.2–19.4 s, none > 24 s.
- **Coverage**: 76–98% of each window classified halfcourt; the rest is
  genuine transition + one fragmented tail (curry_q1 13122+).
- **Confidence anatomy** (all spans ≥ 0.68, majority never in doubt): the
  sub-1.00 spans decompose into three situations — span-START disagreement
  (fast-break lag: offense still ahead of the retreating defense), span-END
  (shot + rebound scramble), and the dominant one, MID-SPAN FLICKER: frames
  where the two teams' mean distances to the basket differ by ~2 ft (p90
  ≤ 3.2 ft, full 8–10-player lineups — not a sampling artifact). That's the
  heuristic at its decision boundary while the offense collapses toward the
  rim; harmless to the span-level majority call.
- **Found + guarded an ultralytics 8.4 bug**: native BoT-SORT re-ID
  (`model: auto`) intermittently crashes with `'numpy.ndarray' has no 'cpu'`
  in bot_sort.py's feature lambda — fires on frames right after a tracker
  reset (mixed feature state). PlayerTracker now catches it like the Kalman
  guard (reset + skip frame). This is what silently killed one batch clip.

Human eval built: `evaluate_possessions.py` (--export per clip → --label
blind, two keypresses per frame: attacked basket a/d/u + offense kit w/n/u →
--report). 73 frames exported, stratified over span_mid (46) / span_start (12)
/ pre_start (12) / transition_mid (3), shuffled label order. Report scores
attacked-basket and offense/defense SEPARATELY (different mechanisms), plus
missed-coverage on transition predictions. Awaiting labels → accuracy numbers.

---

## 2026-07-06c — Component C2 v1: possession segmentation (ball-free)

`segment_possessions.py` — consumes trajectories JSON (positions + teams), no
new perception. Three geometric arguments replace the ball: (1) in a halfcourt
set all ten players occupy one half → smoothed median-x of everyone says which
basket is attacked (±8 ft transition band around midcourt); (2) spans = runs of
one-sided occupancy ≥3 s, broken by frame gaps >3 s; (3) offense/defense = the
DEFENSE's mean distance to the attacked basket is systematically smaller
(defenders sit between man and basket), reported with per-frame-agreement
confidence, offense=None when teams are unlabeled.

First run (standard 20 s window): 10.9 s halfcourt @left (offense navy, conf
0.67) → 1.1 s transition → 5.8 s @right (offense white, conf 1.00); 84% of the
window classified, and the two spans are mutually consistent (offense flips
with the basket — a change of possession). Validation is currently
plausibility + the timeline strip (data/tracking/<clip>_possessions.png);
event-level labels on 2–3 games are the planned real eval. Known limits:
ball-free = no possession changes WITHIN one halfcourt occupancy (steal +
re-set on the same end), and free-throw clusters read as halfcourt sets.
Next: run on longer windows, consistency diagnostic (a team must attack the
same basket all half), then matchup metrics (C3) on top of these spans.

---

## 2026-07-06b — Team-contamination fixes: two shipped, two measured dead ends, veto defused

Goal (user-directed): stop the misassigned-team tracks from poisoning the
linker's cross-team veto before starting possession segmentation. Every
mechanism below was measured against the SAME 160 human labels via the new
`evaluate_teams.py --repredict` (re-runs collection+clustering, updates only
predictions, reuses labels — no relabeling).

### Shipped
1. **Median-of-per-crop-features** (was: raw pixel pooling). Caps each crop's
   influence at one vote. Track-level accuracy 83.9% → **87.1%** (crop-level
   85.3 → 83.9 — the old pooling was accidentally right on some noisy crops).
2. **Floor-fallback dropped**: a mostly-floor crop is now excluded from the
   pool (was: inject floor color, L≈135 — measured pushing washed navy white).
3. **Veto continuity override**: a cross-team veto is skipped when the link
   evidence is overwhelming (sim ≥ 0.97, gap ≤ 2 s, dist ≤ 8 ft) — the team
   signal is ~87% accurate and must not outvote near-certain continuity.

### Measured dead ends (tried, reverted/not shipped)
- **Occlusion-aware crop skipping** (drop crops whose box overlaps another
  track's >30%): track accuracy CRASHED 87→71% — in dense NBA play boxes
  overlap constantly; starving tracks of crops costs more than the
  contamination. Reverted.
- **Vote-consistency abstention** (abstain tracks with mixed per-crop votes):
  no separating threshold exists — correct tracks show minority-vote fractions
  0.35–0.41, same as the broken ones (occlusion crops make mixed votes normal).
  Never shipped.

### The three broken tracks — final status (montage evidence, reports/viz/track_*_crops.png)
- Track 80: 3/7 crops literally show its white OCCLUDER's jersey. Track 86:
  7/11 crops are face/background (close framing) with no jersey at all. **No
  color feature can classify crops that don't contain the jersey** — these two
  stay misassigned, now a characterized limitation (2/31 tracks). Track 150
  (washout) also stays.
- **But the damage is gone.** Post-fix pipeline: the previously-blocked pairs
  (150→225 sim 0.979 @1.4 ft, 4→80 sim 0.982 @6.5 ft) now pass the veto and
  are refused ONLY by the ambiguity margin — and inspection shows that refusal
  is correct: 3–4 navy fragments end simultaneously near each candidate (a
  whistle cluster); merging the top pair would risk swapping two navy players.
  Refusals now reflect genuine uncertainty, not a wrong team label. Canonical
  teams for tracks 3 and 4 flipped to correct (navy) with the new features.

### Where this leaves teams (eval, 160 labels): 83.9% crop / 87.1% track
Known failure mode: tracks whose crops lack jersey evidence (close-ups,
persistent occlusion) — the fix is upstream crop QUALITY (pose-guided torso
crops), not a smarter color feature. Deferred; possession segmentation next.

---

## 2026-07-06 — Human-label eval of team classification (evaluate_teams.py)

Built the labeled eval (`evaluate_teams.py`: --export ~200 stratified crops →
--label keypress w/n/a, predictions hidden → --report). User labeled 160 crops
on the standard window. Headline: **85.3% crop-level / 83.9% track-level
accuracy**, and the error analysis found a clean systematic story.

### The asymmetry is real and directional — and it isn't label difficulty
User flagged: cluster A (white) wrong on 14/73 identifiable crops vs B (navy)
wrong on 3/43 — opposite of what human legibility would predict (navy was
harder to read under blur). Diagnosis:
- **10 of the 14 A-errors are just 3 tracks (150, 86, 80), 100% navy by human
  label** — whole-track misassignments, not occlusion noise. The other 4 (and
  all 3 B-errors) are scattered occlusion crops in mixed/correct tracks — the
  expected crop-level artifact.
- **All 3 misassignments are navy→white; zero white→navy.** Mechanism: every
  contamination source in the crop pipeline is BRIGHT — the ≥75%-floor fallback
  injects orange floor (L≈135) into the pool, close-camera crops have
  crowd/stands/apron backgrounds the maple-floor mask can't touch, and washout
  brightens navy itself (misassigned tracks: median L 90.5 vs 72 for correct
  navy). Contamination monotonically pushes pooled features toward the LIGHT
  cluster, so dark kits absorb ~all the risk. Misassigned tracks' crops are
  BIGGER (median 1775 px² vs 1134) — close shots, where background ≠ floor.
- Evidence the pooling (not the color space) is the weak link: on the SAMPLED
  crops alone, tracks 86 and 80 compute NEARER NAVY — the full-track pixel pool
  flipped them white, i.e. a minority of contaminated crops outvoted the jersey.
  (Track 150 is nearer white even on samples — that one is true washout.)

### Coverage gap (28/71 B-crops labeled "none" vs 7/80 A) — user's read confirmed
B's none-labeled crops are smaller (median 1013 vs 1426 px²) and blurrier
(Laplacian var 1549 vs 2255) but their color stats match navy (L 71 vs 70.5 for
confirmed navy) — i.e. the human couldn't call them, while the machine's color
evidence is navy-consistent. Coverage loss is largely HUMAN-side at this
resolution; identified A vs B crops are the same size (1488 vs 1426), so no
crop-quality asymmetry between clusters. True B coverage is likely better than
the eval shows.

### Implications / fix candidates (deferred until after possession segmentation)
1. Misassigned teams poison the linker veto (the 3 navy-as-white tracks veto
   their own true-navy merges — 86/150/80 all appear in the merge log).
2. Fix directions, in order of likely payoff: (a) drop the floor-fallback
   (abstain a mostly-floor crop instead of injecting floor color); (b) robust
   pooling — median of per-crop features (or trimmed pixel pool) so a few
   contaminated crops can't outvote the jersey; (c) exploit the directional
   law — contamination only brightens, so weight the DARK quartile (L_p25).
3. Artifacts: reports/teams_eval_curry_q1_clip.{json,txt},
   reports/viz/team_errors_A_navy.png; labels committed in
   data/teams_eval/curry_q1_clip/{index,truth}.csv (crops regenerable).

---

## 2026-07-05c — Component C1: team classification + linker team veto

Next node on the dependency spine after Component B: which team is each track
on. Unsupervised (k-means, k=2, per clip — team colors are constant within a
game), zero labels, zero new models. Three consumers wired immediately: the
fragment linker (hard cross-team veto), `"team"` on every trajectory, and
team-colored review video. `--no-teams` is the A/B lever.

### Three iterations, each measured (the whole story is the feature)
1. **CLIP embeddings** (the re-ID crops' embeddings, hypothesis: "jersey color
   dominates CLIP, the re-ID weakness is the team signal") — **WRONG at crop
   scale**: split 25-vs-1 tracks, silhouette 0.247, contact sheet showed white
   and navy jerseys mixed in one cluster. On 30–70 px broadcast crops CLIP is
   dominated by scene statistics (floor, blur, lighting), not jersey. Jersey
   color must be measured DIRECTLY; CLIP stays re-ID-only.
2. **Per-crop median Lab** (+ maple-floor HSV masking, reusing the old gate's
   band): silhouette jumped to ~0.55 and the split became real (white vs dark),
   but balance broke (5-vs-12 tracks, 16 abstained) — single crops are up to
   ~half floor/apron/occluder, so any per-crop color is a coin flip and
   crop-level voting abstained half the tracks.
3. **Track-pooled color** (shipped): ONE feature per track — [L p25/p50/p75,
   a p50, b p50] over the pixels pooled from all 10–40 of the track's crops
   (floor-masked). Pooling makes the track's own jersey the dominant pixel
   mass. Nearest-centroid assignment; abstain when near/far centroid-distance
   ratio > 0.75. Result: 16 / 10 / 7-abstained tracks, silhouette 0.50, frame
   balance [4,2] (vs ideal 5v5), abstained crops 192 → 11. Contact sheet: clean
   white vs navy at track level (per-crop impurities inside a group are
   occlusion frames — the sheet groups by TRACK team now).

### Identity impact (same window: 11520, 200 fr, stride 3, gate)
| | B only (07-05b) | B + team veto |
|---|---|---|
| merges | 8 | **11** |
| ambiguous refusals | 58 | **20** |
| cross-team vetoes | — | 38 |
| canonical tracks | 38 | **35** |
| id churn | 5.43 | **5.00** |
The veto does the disambiguation the appearance signal couldn't: removing
other-team candidates BEFORE the ambiguity margin turns "two look-alike
candidates → refuse" into "one candidate per team → merge". Physics metrics
unchanged (14.3% impossible, p99 39.6 — teams don't touch geometry).

### Caveats / next
- Cluster ids are team A/B, not home/away; naming needs roster knowledge.
- No labeled eval yet — checks are the contact sheet + 5v5-balance + silhouette.
  Planned: label a few hundred crops for a real accuracy number.
- Balance [4,2] ≠ [5,5]: abstentions + short tracks without homography frames.
- Both-teams-dark matchups will compress the L-axis separation; silhouette is
  logged per run as the canary.
- Unlocked next: offense/defense assignment (needs possession direction),
  ref/crowd filtering via abstention, matchup metrics (Component C2).

---

## 2026-07-05b — Component B: player identity (re-ID) + foot-point stability

The 2026-07-05 A/B ended with "the bottleneck has moved to player tracking —
BoT-SORT ID switches and bbox foot-point jitter." This session built that
component. Three levers + one bug fix, all independently A/B-able.

### Bug found first: track-ID collisions across camera cuts
ultralytics' `BYTETracker.__init__` calls `reset_id()` — every `PlayerTracker.
reset()` (camera cut, Kalman guard) restarted BoT-SORT ids at 1. Anything keyed
by `track_id` downstream (i.e. `build_trajectories`' raw dict) was silently
merging DIFFERENT players from different shots under one id. Fixed with a
monotonic id offset across resets (`detect/tracker.py`). This was a prerequisite:
fragment linking is meaningless if fragment ids alias.

### What got built
1. **Native BoT-SORT re-ID** (`botsort_reid.yaml`, `TRACKER_REID=0` to revert):
   ultralytics ≥8.3 `with_reid: true, model: auto` reuses the detector's own
   Detect-head features — appearance matching within a shot for ~free.
   Honest result: on the A/B window (0 camera cuts) fragment count was 46
   with AND without it — no measurable effect here; kept because it's free and
   should help on cut-heavy footage. Needs a cut-heavy window to prove.
2. **FootPointStabilizer** (`detect/footpoint.py`, `--no-stab`): pixel-space,
   per track — (a) occlusion height-clip correction: box much shorter than the
   track's running median height ⇒ bottom is clipped ⇒ re-extend foot to
   top + median (median always fed raw heights, so short occlusions can't
   poison it and real zoom changes still adapt); (b) EMA (α=0.6) to kill
   ±2–4 px box-edge jitter that perspective amplifies to ~0.5+ ft.
3. **Offline fragment linker** (`detect/reid.py`, `--no-reid`): torso crops
   (inner upper box = jersey) → mean CLIP embedding per fragment (the gate's
   frozen ViT-B/32, REUSED — no new model) + court-space motion gate
   (endpoints in feet are comparable ACROSS shots — the payoff of the court
   work) → conservative greedy chaining (≤1 successor/predecessor, ambiguity
   margin refuses teammate-converging cases). Full audit trail per run →
   `data/tracking/<clip>_identity.json` (every merge with sim/gap/dist/speed
   evidence, every ambiguity refusal).

### The calibration lesson (measured, not guessed)
First linker pass made **0 merges — all 89 candidates refused as ambiguous**.
Audit showed why: CLIP torso sims cluster 0.89–0.98 across ALL pairs (jersey
color dominates; teammates are near-identical) ⇒ appearance cannot RANK
candidates, only gate them. But 19/26 fragments had a spatially CLEAR best
successor (>6 ft margin over the runner-up). Fix: score = sim − 0.3·dist/reach
(motion dominates, appearance tiebreaks), ambiguity margin applied to the
combined score. Result: 8 merges, all physically plausible (implied speeds
4–23 ft/s, incl. one 3-fragment chain), 58 still refused. This mirrors the
gate/court lessons: the cheap general-purpose embedding is a filter, the
geometry is the signal.

### A/B (build_trajectories, same window as 07-05: 11520, 200 fr, stride 3, gate)
| | baseline (B off) | Component B |
|---|---|---|
| impossible steps (>3 ft/0.1 s) | 15.4% | **14.3%** |
| step p50 | 1.23 ft | **1.15 ft** |
| step p99 | 39.8 ft | 39.6 ft (~tie) |
| fragments → canonical tracks | 46 → 46 | 46 → **38** (8 merges) |
| id churn (tracks / median 7 players) | 6.57 | **5.43** |
| foot clip-corrections fired | — | 68 |
Baseline exactly reproduces the 07-05 numbers (15.4 / 39.8 / 16.5% corrected),
so the new in-script physics report measures the same thing the old table did.

### Honest reads / caveats
- The p99 tail didn't move: steps are measured WITHIN a track id at stride
  gaps, so ID churn never polluted that metric — the tail is homography
  extrapolation on far-court/HELD frames, not identity. The stabilizer's win
  is the body of the distribution (impossible-rate, p50), as expected.
- Merge validation is plausibility-only (speeds, visual review video). There
  are NO identity GT labels; a MOT-style labeled clip would quantify ID
  precision/recall properly. Deferred until team classification exists —
  team id will also become a hard veto gate for the linker (never merge
  across teams), which should let the ambiguity margin relax.
- Churn 5.43 ≈ still 5×: remaining fragments are short/no-crop (unlinkable by
  design), gaps > 4 s, or ambiguity refusals. That's the conservative trade —
  a false merge poisons two players' stats; a missed merge is recoverable.

### Status / next
Component B v1 shipped: `detect/footpoint.py`, `detect/reid.py`, tracker id fix,
`botsort_reid.yaml`; wired into `build_trajectories.py` with `--no-stab/--no-reid`
levers + always-on physics/identity reports. Next candidates: (a) team
classification (jersey-color clustering on the same torso crops — they're already
collected) → linker veto gate + ref filtering; (b) a cut-heavy A/B window to
actually exercise cross-shot linking + native re-ID; (c) MOT-labeled mini-clip
for real ID metrics.

---

## 2026-07-05 — Grid+snap tracker INTEGRATED into the pipeline; label factory built

### Integration: `court/snap_track.py` (CourtTracker) behind CourtMapper
Per frame: grid model → RANSAC H → geometric sanity → guarded line-snap; when the
model fails (pans/blur/closeups), the previous H is propagated by the camera's
global shift (cv2.phaseCorrelate on downscaled grays) and re-locked onto visible
lines (wider search radii, stricter accept: ≥50 matches, ≤2.5 px, sane). States
TRACK / LINE_TRACK / HELD / LOST; HELD carries `quality=9.9` so the existing
masking-confidence gate distrusts it with zero changes to consumers —
`CourtHomography.from_matrix()` makes the tracker a drop-in. Hull for
`is_extrapolated` = whatever evidence supports the H (model kps on TRACK, matched
line points on LINE_TRACK). `COURT_TRACKER=0` env reverts to the legacy detector
(kept as the A/B lever). Smoke-test lesson: the first test window (frame 6000+)
was a TIMEOUT — the tracker correctly refused to invent a court; the gate scan
found the real gameplay run at 11520–14040 (the same window config.py's old
threshold sweeps reference).

### The decisive A/B (build_trajectories, identical args: 11520, 200 fr, stride 3, gate)
| | OLD (33-pt det) | NEW (grid+tracker) |
|---|---|---|
| frames with H | 200/200 | 174/200 + held |
| points needing correction | 26.5% | **16.5%** |
| impossible steps (>3 ft / 0.1 s) | 25.4% | **15.4%** |
| p99 step | 69.6 ft | **39.8 ft** |
| off-court projections | 1.1% | 0.7% |

Old's "200/200 with H" is a vice: at conf 0.05 it always scrapes an H and is
confidently wrong (its 70-ft teleports). Every physics metric improved ~40%.
**The bottleneck has moved to player tracking** — remaining teleports are
dominated by BoT-SORT ID switches and bbox foot-point jitter, which no court
model can fix. Component B (identity) is the next frontier.

### Label factory (`label_factory.py`) — self-training loop, operable without Claude
`harvest <urls|files>`: 6×4-min sections per game → CLIP gate → grid model +
ANCHORED line-snap → strict accept gate → image + labels (both formats, identical
to generate_labels) + worst-first review overlays + per-game rank TSV
(verify_projections-compatible). `build`: merge base + all harvests (minus user
`no_lineup` verdicts) → `*_v2` datasets + Colab zips; harvest frames go to TRAIN
ONLY and **val stays frozen at the 279 verified frames** so benchmarks remain
comparable across retrains forever. Trial on 2 sections: 52 accepted (~150–250
per full game expected), every rejection logged by reason.

Three calibration lessons (each measured on real footage, not guessed):
1. **The snap refit must keep anchor points.** Line matches alone (median 28 on
   YouTube 720p vs 60–150 on the offline dataset) let findHomography fit the
   sparse lines perfectly while warping the frame corners by 2,600 px — the
   4-point "locally exact, globally garbage" failure mode reborn. Keeping the
   model's grid kps in every refit pins the global solution (this is exactly what
   the offline snap's code-2 anchors did; dropping them was the bug).
2. **Residual thresholds must be resolution-normalized.** "Sub-pixel" was a
   640-px-frame number; the same physical accuracy on a 1280-wide frame reads
   ~2×. Gate: res ≤ 1.1 px × (width/640).
3. **The pipeline gate threshold (0.70) is wrong for harvesting.** It was tuned
   on our own clips; live frames of an unfamiliar 2024 broadcast score 0.47–0.66
   (non-court ~0.03). Harvest runs at 0.35 — false-live costs nothing (the solve
   gate rejects it), false-dead costs yield.

Operating manual → README. Deferred, deliberately: the 404 no_lineup frames from
court_review (the factory produces better frames more cheaply than rescuing
those), and the 8 SKIP_ swap candidates in swap_verify/.

---

## 2026-07-03→04 — Retrained detectors on projection labels; benchmarks in-domain + UNSEEN; real-time clips

### Label generation + training (the payoff of the verification pipeline)
`generate_labels.py` projects labels straight from the snapped H's — every point is
a projection, including the original download vertices (the snapped H, ~0.5 px, is
now more accurate than the box-center anchors it came from, so pure projection
beats mixing sources). TWO formats, same frames: 33-vertex (`data/court_labels_33/`,
median 14 visible kp/frame) and a 13×7 = 91-pt court grid (`data/court_labels_grid/`,
median 36). Rationale for the grid: the 33-pt scheme was forced by the downloaded
dataset; with projection labels we can label ANY court point for free, and more,
better-spread anchors = more redundant inference H fits. On-frame points get v=2
(v=1/v=2 train identically in ultralytics); off-frame `0 0 0`; bbox = visible court
region. `build_pose_datasets.py`: ONE shared split (1579/279, seed 42) for both
formats. Split is random-by-frame — the source datasets don't expose video ids, so
group-splitting wasn't possible; val metrics carry mild leakage optimism (noted below).

Training: started locally on MPS, killed — user needs the Mac; **all training goes
to Colab** (`train_pose_snapped_colab.ipynb` + `court_pose{33,_grid}_colab.zip`,
352 MB each). Recipe copied verbatim from the old `court_aug` run (yolov8m-pose,
300 ep, patience 30, heavy HSV, flip_idx) so old-vs-new isolates the labels.

### In-domain benchmark (`benchmark_court_models.py`, 279 val frames, GT = snapped H)
| model | valid H | H-err p50 | p90 | >10px | kp-err p50 | pts/frame |
|---|---|---|---|---|---|---|
| old-clean | 278/279 | 8.2px | 28.1px | 110 | 8.9px | 12 |
| old-aug | 275/279 | 7.1px | 30.9px | 90 | 8.1px | 12 |
| NEW-33 | 279/279 | 1.8px | 3.2px | 2 | 1.9px | 14 |
| NEW-grid | 279/279 | **1.7px** | **2.8px** | **1** | 2.6px | 37 |

Same architecture/recipe → the ~4× median gain and the collapsed tail (90 → 2
frames >10px) is **pure label quality**. Grid ties the median, wins the tail:
each grid point is individually WORSE (2.6 vs 1.9 px — featureless mid-paint
points are hard) but 37 pts/frame out-votes noise in RANSAC. Caveats, both ways:
old models trained on sources of these val frames (bias favors OLD → real gap is
larger); GT shares its generation pipeline with the new labels (but was human-
verified + snapped to actual image lines, so it's the best truth available).

### Unseen-footage benchmark (`unseen_benchmark.py`) — the generalization question
4 YouTube games (1998 CHI@UTA SD, 2013 SAS@MIA, 2019 TOR@PHI, 2024 DEN@MIN),
12 × 5-min sections via yt-dlp, frames every 2 s, HSV-gate filtered → 1,291 frames.
No GT → unsupervised scoring: sane-H rate (fold/collapse/explode checks), line
residual vs image evidence, match count. Sane-H (old-aug → NEW-grid): 1998 44→66%,
2013 39→69%, 2019 62→89%, 2024 72→71% (tie, but line matches 46→66 — new H's sit
on the lines far more). New labels generalize; the gain was NOT in-domain memorization.
NOTE denominators: HSV gate passes closeups/replays no model can solve, so absolute
rates understate everything equally; comparisons are the signal. Black 3-pt line on
1998's white floor is already handled — line evidence is a top-hat PAIR (bright+dark).

### Real-time overlay clips (`render_overlay_video.py`) — per-frame % vs deployment %
Longest gate-passing stretch per game, EVERY frame: grid model → sanity → line-snap
(guarded) → draw; on failure reuse last H ≤ 2 s (yellow). mp4v is broken in this
OpenCV/AVFoundation build (silently writes nothing) — use `avc1`. Results:
TRACK 99% (2019) / 98% (2013) / 90% (1998) / 79% (2024), LOST ≤1% everywhere.
This is the resolution of "aren't 66–71% pretty bad?": per-frame failures cluster
in closeups/cuts, not mid-possession, so time-coverage on live play is 90–99%.
User's viewing observations → next work items: (1) pans between half-courts fail
under freeze-hold → upgrade HELD to a snap-TRACKER (seed from previous frame's H
+ global-shift estimate + line-snap; pose model re-anchors when confident);
(2) closeups are a gate problem (swap crude HSV band for the trained CLIP head);
(3) then the LABEL FACTORY: user feeds game list → auto-harvest snap-verified
frames (strict quality gate) → projection labels → spot-check sample → retrain.

---

## 2026-07-01→03 — Court labels solved: triage → 2,262-frame human verification → line-snap

Goal shifted from "review 2,262 densified court frames by hand" to "make the
homography projection perfect for every frame from the original datasets."
Dataset: `data/court_review/` — NBA Court v4 downloads (box-center → point
conversions, code 2) + old-model fills (code 3).

### Deterministic triage of the downloads (pure code-2)
Seeded RANSAC H per frame. **Bug caught mid-way:** RANSAC threshold was divided
by 50 (0.06 ft) → RANSAC fit minimal subsets → 662 frames over-flagged and a bogus
"systematic index-0 mis-mapping" theory; fixed to 3.0 ft (dst units = feet) and the
analysis was retracted. Key user insight (proved right): every flagged point sits ON
a real intersection → all 41 gross (>20 ft) errors are **INDEX SWAPS**, zero position
errors. `fix_index_swaps.py` auto-fixed 33 with a guarded refit (moved point <4 ft,
median not worse); 8 ambiguous skipped for eyes. The downloaded dataset is
positionally clean.

### Dead end: fill-agreement scoring (internal consistency ≠ global correctness)
For the 134 exactly-4-point frames, scored H's by agreement with the (independent)
old-model fills → 24 "trustworthy". User's visual inspection overruled: most
projections wrong. Measured cause: the 4 downloads cluster in a thin strip (median
hull 7.3% of frame) → H locally exact, globally garbage, and the fills cluster in
the same strip so "agreement" only tested locally. Lesson burned in: never certify
an H from evidence that doesn't SPAN the frame.

### Ranked human verification (the workflow that worked — user's design)
`rank_projections.py` renders all 2,262 projection overlays worst-first. Scoring
took 3 iterations: raw top-hat (crowd texture = universal false support) → density
gate alone (salt-and-pepper speckle made 93% of pixels "dense", zeroing everything;
fixed with medianBlur(5) first) → final primary signal is H-GEOMETRY SANITY
(count of 33 court vertices landing on-frame via H⁻¹: collapsed ≥30, exploded ≤6;
fold via Jacobian det sign at 4 corners), line evidence only as secondary. Overlays
draw curves too (3-pt arcs fit through their dataset vertices, corner-3s, circles)
— `court33_curves()`. `verify_projections.py`: one keystroke per frame, crash-safe,
resumable. **User verdicted all 2,208 scored frames: 1,858 lineup / 350 no_lineup**
(+54 NO-H). All 63 folded/collapsed flags were confirmed bad — 100% precision.
Verdict bar: "if you can tell which painted feature each line is trying to be" —
structural failure = reject; snap-fixable offsets = pass.

### Line-snap (`snap_projections.py`) — verified H's polished to sub-pixel
ICP-style: sample template every 1 ft → search ±R along each sample's NORMAL in
density-gated top-hat ridge → sub-pixel parabola peak → RANSAC refit (+ anchors) ×4
rounds. Occlusion handling is structural: crowd zones produce no evidence (gate),
players are RANSAC outliers, and the always-occluded near sideline is positioned by
H extrapolation from visible lines. **Accept-metric lesson:** global DT-median is
occlusion-dominated and can't see improvement (first version "failed" on frames
that were visibly improving); the working metric is matched-offset residual at
fixed radius + match count must not drop >20% + anchor/corner guards.
Result: 1,136 snapped / 722 kept (mostly already at the anchor noise floor);
final residuals p50 0.67 px, p90 2.03 px, max 5.63 px. H's → `snapped_H.npz`.
Open: 350 no_lineup + 54 NO-H (drop vs hand-anchor undecided); 8 skipped swaps.

---

## 2026-06-27→30 — (reconstructed from artifacts) 33-pt model iterations on Colab

Per `train_court_colab.ipynb` + weights in `models/`: three external-dataset merge
attempts failed (out-of-domain data hurt: `merged3307`, `merged_dense`, `boxfix`);
the run that stuck was **clean original court_det2 (850 imgs) + heavy HSV
appearance aug** (hsv_h .03 / s .9 / v .7) for cross-arena robustness →
`court_kp33_aug.pt` (old-aug baseline above). `court_kp33.pt` = pre-merge clean run.

---

## 2026-06-26 — Component A: trajectory correction (analytics layer begins)

After an architecture diff vs an external Roboflow "Basketball AI" project (ARCHITECTURE_DIFF.md;
their stack: RF-DETR + SAM2 tracking + TeamClassifier + jersey-OCR VLM + ConsecutiveValueTracker
+ 33-pt ViewTransformer homography + clean_paths + shot events). Key borrowed insight: they fix
bad per-frame homography DOWNSTREAM via trajectory cleaning, not at the source.

Built Component A: `build_trajectories.py` runs a clip through BoT-SORT + 33-pt homography,
projects each player's foot-point → court ft per frame, accumulates per-track series, and cleans
them with `court/trajectories.py` `clean_paths` (vendored from roboflow/sports, Apache-2.0: robust
jump rejection via median+σ·MAD AND absolute-dist, remove short teleport runs, linear-interp gaps,
Savitzky-Golay smooth). Adapter `clean_trajectories` handles BoT-SORT's variable roster (per-track,
over each track's presence span). Top-down court render (court33) of raw vs cleaned paths. Fixed a
bug in the vendored savgol fallback (even-kernel off-by-one on short spans). First run (200 frames):
114 w/ homography, 17 tracks, 149 points, 30% corrected. Observation: **homography quality is the
bottleneck** — many frames project players off-court (dropped pre-correction), so trajectory density
is gated upstream. Levers: restrict to confident-H frames; tune clean_paths params. Also have the
jersey-OCR v7 dataset (3,615 crops, PaliGemma JSONL) staged for Component B (identity).

## 2026-06-26 — Adopted external court dataset (33-keypoint homography)

Integrated Roboflow **basketball-court-detection-2** (850 imgs, 33-keypoint pose). Found the
exact 33-vertex→court-feet mapping by lifting it from `roboflow/sports` `CourtConfiguration`
(NBA, FEET) — the lib that generated the dataset — and reproduced it exactly (`court/court33.py`,
corner-origin: x 0→94, y 0→50 ft). Trained yolov8m-pose on Colab GPU (train_court_colab.ipynb;
GPU avoids the MPS pose bug). Weights → `models/court_kp33.pt`. New detector `court/detector33.py`
(index→vertex), `homography_from_indexed_keypoints`. Head-to-head vs the old 20-pt model: on hard
gameplay frames H-success 60%→70%, ~12 vs 8 keypoints (better-conditioned), reproj 0.22→0.60 ft
(looser per-point but the old low value still produced nonsensical frames — global alignment is
what matters). **User reviewed side-by-side overlays and judged the 33-pt model the winner.**
SWITCHED CourtMapper to the 33-pt scheme (corner-origin polygon/court_pos) + visualizer draws
court33 edges. Pipeline runs end-to-end (gate → basketball detector → 33-pt homography → masking).
Note: with the basketball detector excluding refs/crowd, court-masking now drops ~0 (redundant,
as predicted). NOT YET PORTED to 33-pt: line refinement (refine.py still 20-pt) and
evaluate_homography (still 20-pt labels). Old 20-pt model/path left intact.

## 2026-06-26 — Basketball detector (external dataset) — big Component-2 win

Integrated Roboflow **basketball-player-detection-3** (654 imgs, 10 classes) → remapped to
5 (player/referee/ball/rim/number; `prepare_player_dataset.py`), trained yolov8m on a Colab
T4 GPU (~0.5 hr; local MPS was ~22 hr so training moved to Colab — `train_player_colab.ipynb`,
pulls from Roboflow API). Val mAP50: player 0.973, referee 0.994, rim 0.995, number 0.942,
ball 0.841. Weights → `models/player_detector.pt`; tracker + eval auto-prefer it (track the
`player` class only). **On our hand-labeled box_truth: precision 0.68→0.89, recall 0.74→0.87,
F1 0.71→0.88, FP 158→50, FN 116→61** — beats generic COCO person on BOTH axes. Fixes the
refs/fans-as-players problem and reduces the need for court-masking (detector excludes
non-players itself). Strategic: Colab GPU is now the home for all heavy training (also fixes
the keypoint-pose MPS bug). Next external dataset: **court-detection-2** (850 imgs, 33-kpt
scheme) for the homography. Files: prepare_player_dataset.py, train_player_detector.py (local),
train_player_colab.ipynb, colab_train_player.py; detect/tracker.py + detector.py auto-prefer
PLAYER_DETECTOR_WEIGHTS.

## 2026-06-25 — Line-based homography refinement + learned line-segmentation model

Goal: fix the homography being wrong/nonsensical on many frames. Two prongs (#1 more
keypoint data — user labeling; #2 learned court-line model — below).

**Point+line refinement** (`court/refine.py`): bootstrap H from keypoints, project the
court lines, snap to detected line pixels along the normal, re-solve (points+lines), ICP
×3. MONOTONIC guard: accept only if it lowers the court-line residual AND keeps keypoint
agreement → never degrades (worst case no-op). First built with a white top-hat line
detector = ~no-op on low-contrast wood (+0.14px, 3/12).

**Learned line segmentation** (the fix for the top-hat): labels generated FOR FREE by
projecting the court template through each keypoint-labeled frame's homography
(`make_line_dataset.py` → 117 image/mask pairs). Small U-Net (`court/line_seg.py`,
`train_line_seg.py`) trains on MPS (conv+BCE/Dice fine on MPS, unlike the pose model) →
**val Dice 0.587**. `refine.court_line_mask` auto-uses it when weights exist.
Result: refinement with the learned mask = **+0.40px residual (3× top-hat), 5/14, never
worse** → COURT_REFINE re-enabled (default True). Caveat: refinement only sharpens a
roughly-right H; truly-wrong frames need better keypoints (#1). The keypoint→linelabel→
linemodel loop compounds: more keypoints (#1) → better H → better auto line-labels → better
line model → better refinement. Files: make_line_dataset.py, train_line_seg.py,
court/line_seg.py; config LINE_DATASET/LINE_SEG_WEIGHTS/COURT_REFINE.

## 2026-06-25 — Homography wired into pipeline; court-masking TESTED (honest negative)

### Homography → pipeline (WORKS)
`court/mapper.py` `CourtMapper` ties the trained court-kp model → homography → court
coords + masking. `run_tracking.py --court` projects every tracked player's foot-point to
court feet, stores in tracks.json, labels on overlay. Verified: 149/149 players got valid,
sensible court positions (0 failures). This is the real homography payoff: spatial player
positions for defensive metrics.

### Court-masking (predicted to fix Component 2 precision) — TESTED, DOESN'T pan out
Validated on the 50 hand-labeled box frames (evaluate_tracking.py --court-mask):
- baseline: P 0.681 / R 0.744 / F1 0.711
- court-mask (confident-gated): P 0.699 / R 0.706 / F1 0.703  ← F1 slightly DOWN
The earlier prediction ("court-masking recovers precision") was TOO OPTIMISTIC — now disproven
with data. Why: (1) pixel→court projection blows up near the homography horizon (max 1.2M ft!);
fixed by masking in IMAGE space (project court boundary→pixels, point-in-polygon). (2) Even so,
on ~30-40% of fresh single frames the H is slightly wrong → court polygon misplaced → drops REAL
players. Confidence-gating (≥6 inliers, reproj ≤0.6 ft) limits damage but only ~20 of 158 FPs are
confidently-removable; the rest are bench/courtside people NEAR the boundary where H error ≈ the
margin. So masking removes ~as many real players as crowd → net wash.
**Conclusion:** single-frame court-masking isn't reliable enough. Real fix = TEMPORAL homography
smoothing in the video pipeline (stable averaged H over a window) + more kp data. Court-masking
kept OFF by default; `--court-mask` available but caveated. The 0.30 ft val reproj was measured AT
keypoints — projection accuracy AWAY from them (player positions) is worse; don't conflate them.

### Per-frame masking visualizer (2026-06-25)
`visualize_masking.py` — renders court-masking onto a clip for the USER to review (don't
interpret it for them — see [[feedback-viz-for-user]]). Per frame it draws: projected court
template (green=confident H / orange=low-conf), cyan mask-boundary polygon, red "off-court"
boxes for dropped detections (drawn, not hidden), kept boxes with #id + court (x,y) ft,
H-status (inliers + reproj), and a legend. Outputs a scrubable .mp4 + full-res PNGs in
data/tracking/<clip>_masking_frames/. (run_tracking.py also has --court/--court-mask/--save-frames
for the same overlay inside the tracking pipeline.)

---

## 2026-06-24 — Component 3: court keypoints + homography (foundation built; labeling pending)

### Phase 0 (the smoking gun)
The old project's homography MATH was complete and correct (`src/court/homography.py`:
RANSAC `findHomography` + projection). It never worked for ONE reason: the
court-keypoint dataset was **never labeled** — `data/court_kp_dataset` had 57+8
images but **0 label files**. So no pose model was trained → no keypoints → `H=–`
throughout. This component = fix exactly that (label → train → feed the existing math).

### Decisions
- **Learned keypoint model** (YOLOv8-pose), not classical/manual: the broadcast
  camera pans/zooms continuously, so H changes every frame → need a per-frame
  keypoint source. Classical Hough was the old fallback and was brittle.
- **Keypoint scheme = 16 crisp points**, finalized AFTER a user labeler test:
  - kept: 8 paint corners + 2 half-court∩sideline (half_bot/top).
  - user feedback: the 4 three-point ARC-BREAK points are fuzzy (no sharp corner)
    and half_bot is foreshortened → hard to label precisely.
  - SWAPPED the 4 fuzzy arc-break points → 4 CRISP corner-3∩baseline points
    (±47, ±22). ADDED 2 center-circle points (half line ∩ center circle, (0,±6),
    user's idea — crisp + central, conditions H well).
  - Rationale: keypoints demand PRECISION (a misplaced point bends H; unlike boxes
    where IoU is forgiving) and we have redundancy (need only ≥4), so all-crisp set.
  - flip_idx updated + verified (mirrors x, involution). Homography round-trip
    re-verified exact (16/16 inliers, 0.00000 ft).

### Built + verified
`court/geometry.py` (16-kp scheme + court drawing), `court/homography.py` (RANSAC,
exact round-trip verified), `court/detector.py` (YOLOv8-pose wrapper),
`label_keypoints.py` (guided one-kp-at-a-time labeler with court-diagram inset;
user-tested OK), `train_court_kp.py` (labels→YOLO-pose dataset→fine-tune; label
conversion verified), `evaluate_homography.py` (keypoint px error + H success rate +
reproj error + court-overlay visual proof). Apple-Silicon notes from Component 2
(MPS nms fallback, avc1) carry over.

### Keypoint set grew to 20; labeling done + QA'd (2026-06-24)
- After more user feedback, ADDED 4 court corners (baseline∩sideline, ±47±25) — crispest
  points, anchor the sidelines. Declined restricted-area arc points (curve-tangent fuzzy +
  clustered under-hoop + occluded). Set LOCKED at 20.
- User labeled 150 frames. Built `review_keypoints.py` — QAs labels via homography: fits H
  from each frame's labels, flags high reprojection error, and isolates GENUINE mislabels
  via leave-one-out refit (drop worst point; if rest snap clean, it was the culprit) vs
  geometric degeneracy.
- Result: **median reproj 0.087 ft (~1 inch), 0 mislabels** — excellent labeling. 112 clean,
  31 "degenerate" (points correct but too collinear to anchor H alone — fine for training;
  cause = frame's visible crisp points all on one line, e.g. baseline), 7 with <4 points
  (excluded). 143 frames feed training. Lesson logged: each frame needs points spanning BOTH
  axes (don't label only-baseline points).

### Training debugging (2026-06-24) — TWO stacked bugs, both fixed
First train run: "best at epoch 1", `train/pose_loss` FLAT ~11.4 all epochs, pose mAP=0,
box mAP degraded 0.41→0.06. Two root causes:
1. **Degenerate bounding boxes.** `to_yolo_line` set each court instance's box to the HULL
   of its visible keypoints → on collinear frames (baseline-only points) the box is a thin
   sliver → broken detection target → box loss can't learn → pose (conditioned on detection)
   never trains. FIX: **full-frame box** (`0.5 0.5 1 1`) for every frame; court fills the
   broadcast frame, detection becomes trivial, capacity goes to keypoints.
2. **YOLOv8-pose keypoint loss does not train on Apple MPS.** After fix #1, box mAP recovered
   but pose_loss STILL flat / pose mAP=0. Detection trains on MPS, the pose head's gradients
   don't. CONFIRMED by a CPU run: pose_loss dropped 11.5→7.7 over 11 epochs, box mAP→0.9.
   FIX: `train_court_kp.py` now defaults `device="cpu"` (this training only; inference + all
   other components still use MPS). Also surfaced + fixed a homography crash on garbage
   keypoints (0 RANSAC inliers → perspectiveTransform None) — now fails gracefully.

### ✅ DONE — homography WORKS (2026-06-24)
Trained 100 epochs on CPU: pose_loss 11.5→2.9, box mAP 0.99, **pose mAP50 0.96**. Eval on
28 held-out frames: **100% homography success, median reproj 0.30 ft (~3.6 in), p90 0.51 ft**,
green court template lands on the real court. THE milestone the old project never reached
(it had H=– throughout). One more tuning win: RANSAC reproj threshold (in FEET) was 5.0 = too
loose → let noisy paint-corner keypoints skew H; swept on val → **2.0 ft** (median 0.94→0.30 ft,
still 100% success). Set as default in config. Known weak spot: the model mis-localizes the
FT-line paint corners (br/tr) on some frames (200-300px) — RANSAC discards them as outliers,
which is why H stays clean. Improve later with more data / higher-imgsz train / temporal
smoothing in video. `--ransac-ft` and `--conf` flags added to evaluate_homography for tuning.

### Status / next
Components 1 (gate), 2 (player det+track), 3 (court homography) all DONE. Homography unlocks:
(a) project player foot-points → court coords (real spatial defensive metrics — the research
goal); (b) COURT-MASKING to drop off-court detections → should fix Component 2's precision.
Next likely: wire homography into the pipeline + court-masking, or team classification.
Originally: label ~150 frames with `label_keypoints.py --n 150` (resumable,
seeded; the 5 test frames were on the old scheme and were cleared). Then
`train_court_kp.py` → `evaluate_homography.py`. Success = reproj error < ~1.5 ft and
the projected court template lands on the real court lines. This ALSO unlocks the
court-masking that's predicted to fix Component 2's detection precision.

---

## 2026-06-23 (pm) — Component 2: player detection + tracking (built + verified)

### What it is
Second pipeline stage: detect players per frame and assign persistent IDs across
frames, running on contiguous video (IDs propagate frame-to-frame — unlike the
gate's 1.5s-spaced sampling). Person class only; ref/player split is a later
(team-classification) concern.

### Phase 0 recovery (from old repo)
- Old approach = **one ultralytics call**: `model.track(frame, classes=[0],
  persist=True, tracker=...)` does YOLO detection + association in a single pass.
  Defaults: YOLOv8m, conf 0.40, iou 0.45. (`old src/tracking/multi_tracker.py`)
- Carried over two robustness fixes: **camera-cut reset** (when >80% of ≥4 active
  tracks vanish in a frame → reset IDs, else a broadcast cut drags stale IDs onto
  new players) and the **MPS Kalman guard** (`np.linalg.LinAlgError` → reset, skip).
- `yolov8m.pt` reused from the old repo (52 MB); ultralytics auto-downloads if absent.

### Decisions
- **Tracker = BoT-SORT** (not the old default ByteTrack): its camera-motion
  compensation (GMC) is the key win for broadcast pans/zooms; ReID can be enabled
  later via a custom yaml. `config.TRACKER_CONFIG = "botsort.yaml"`.
- **Eval = lightweight**: hand-label boxes on ~50 live frames → detection P/R@IoU.
  Deliberately NOT full MOT (no IDF1/MOTA) — that needs box+ID labels across clips.
  ID stability is tracked via label-free proxies instead (see diagnostics).

### Gotchas solved (Apple Silicon)
- **`torchvision::nms` not implemented on MPS** → set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` at the top of `config.py` (before any torch
  import). Falls back to CPU for just that op.
- **Video codec**: this OpenCV build can't write `mp4v`/`XVID`; only **`avc1`**
  (H.264/AVFoundation) produces a readable .mp4. `run_tracking.py` uses avc1.

### What got built
`detect/` package: `tracker.py` (PlayerTracker, streaming `update()->list[Track]`),
`detector.py` (stateless single-frame PlayerDetector, for eval), `camera_cut.py`,
`types.py` (Track with foot_point + downstream team_id/court_pos placeholders).
Entry points: `run_tracking.py` (overlay mp4 + tracks.json + diagnostics),
`label_boxes.py` (interactive box labeler), `evaluate_tracking.py` (detection P/R).

### First run (curry_q1_clip, 120 frames @ stride 3 ≈ 12s, no gate)
- avg **8.08 players/frame** (median 8, max 10) — plausible for wide shots, a bit
  low (distant/occluded players missed at conf 0.40).
- **id_churn_ratio 4.5** (36 unique IDs for ~8–10 players, 0 camera cuts) →
  **ID stability is the weak spot** (switches through paint occlusion / similar
  jerseys). Expected; BoT-SORT alone doesn't fully fix it.
- Overlay renders correctly (boxes + #IDs on players). Detection eval verified in a
  sandbox (synthetic GT) — plumbing good; real numbers await hand-labeling.

### Detection eval results (2026-06-24) — 50 hand-labeled frames, 453 boxes (9.1/frame)
yolov8m, conf 0.40, IoU 0.5: **precision 0.681, recall 0.744, F1 0.711**.
Visual error analysis (reports/viz/detection_eval.png) was decisive — the two error
types mean very different things:
- **FP = crowd / bench / sideline people.** YOLO correctly detects them; we only
  labeled on-court players, so they score as false positives. So precision (0.68) is
  likely PESSIMISTIC. HYPOTHESIS (untested, no code yet): once homography exists we can
  mask off-court detections, which SHOULD raise precision — but court-masking is NOT
  built and this is unverified. Supporting evidence only: raising conf→0.55 cuts FP
  158→86, precision→0.77, confirming the FPs are low-conf crowd.
- **FN = players clustered/occluded in the paint** (e.g. a frame with 7/10 missed in
  an under-basket scrum). The real signal.

Lever sweeps (added --conf/--nms-iou/--weights to evaluate_tracking.py):
- conf 0.25/0.40/0.55 → recall 0.764/0.744/0.651. Lowering conf barely lifts recall
  (+2%) but doubles crowd FP ⇒ conf is the WRONG lever; keep 0.40.
- NMS IoU 0.45→0.70: ~no change ⇒ occluded players aren't NMS-suppressed, they're
  simply NOT DETECTED.
- yolov8x (bigger model): recall DROPS to 0.695 (more conservative). Off-the-shelf
  scaling doesn't fix occlusion.
**Conclusion: recall ceiling ~0.74 is a model-capacity limit on occluded paint
players, not tunable away.** Real fixes (later, if needed): a basketball-finetuned
detector, and/or lean on BoT-SORT temporal continuity to carry tracks through brief
per-frame misses (per-TRACK recall > per-frame recall). Verdict: good enough to
proceed — finds ~3/4 of players/frame; misses are occlusion; precision is EXPECTED
(not yet shown) to improve once court-masking is built.

### Status / next
Component 2 COMPLETE (built, eval'd, characterized). Default = yolov8m, conf 0.40,
BoT-SORT. Open improvements if ever needed: basketball-finetuned detector, ReID,
pixel-histogram scene-cut detection. Interface ready for downstream (Track carries
team_id/court_pos slots). Next component likely court homography (which would ALSO
let us build court-masking — predicted to lift the precision number, to be verified
then) or team classification.

---

## 2026-06-23 — Component complete: gate built, labeled, trained, validated

### What this project is (one paragraph)
Fresh restart of the NBA broadcast-stats pipeline, rebuilt one component at a
time. This repo is **only** the *court-visibility gate*: a per-frame classifier
that decides "live game footage" vs "dead ball" (replay/closeup/ad/crowd/timeout)
so the downstream stats pipeline only ever processes usable frames. Hard scope
boundary — no player/ball/court/team/identity/event code here. The motivating
bug: clips were returning all-empty output, i.e. the old gate was throwing away
live frames (false negatives).

### Environment / where things live (easy to forget)
- Build dir: `NBA Comp Viz New Project Version 3/` (NOT the sibling `NBA Comp Viz s`).
- Interpreter: **`/opt/anaconda3/bin/python`** (has torch 2.2.2 + MPS, cv2 4.13.0,
  transformers, sklearn). Plain `python3` does NOT have torch/cv2.
- Source video: old project `Basketball_Defensive_Vision/data/raw` (7 mp4 clips, 720p).
- Seed = 42 everywhere.

### Phase 0 — what we recovered from the old project (read-only)
- The old shipped gate `_is_court_visible` (`Basketball_Defensive_Vision/src/pipeline/
  runner.py:479`) was **HSV maple-floor-color thresholding** (accept a frame if
  12–55% of pixels are floor-colored), **NOT CLIP**. So there were **no zero-shot
  prompts or learned threshold to recover.** We said so explicitly instead of
  inventing a training story, kept the HSV rule as a baseline, and wrote fresh
  CLIP-based approaches.
- `court_detector_yolov8n.pt` in the old repo is a *separate* court-bbox model,
  never wired into the gate.
- Frame extraction adapts the old `VideoReader` pattern, incl. the macOS quirk:
  **force OpenCV's FFmpeg backend** (`cv2.CAP_FFMPEG`) to avoid a VideoToolbox/
  Metal segfault once torch is loaded.

### Key design decisions + rationale
- **Frozen backbone + tiny head, not train-from-scratch.** With ~1k labels,
  fine-tuning a vision net would overfit. We freeze CLIP ViT-B/32 (loaded via
  `transformers`; `open_clip` wasn't installed) as a fixed image→ℝ⁵¹² encoder and
  train only a logistic-regression head. Bet: "live vs dead" is already ~linearly
  separable in CLIP space.
- **Three gates, one interface** (`score()->P(live)`, `is_court_visible()`):
  CLIP zero-shot (Approach A, no training), logistic head on CLIP embeddings
  (Approach B, the real model), and the recovered HSV rule (baseline-to-beat).
  Same interface so any one drops into the larger pipeline untouched.
- **Labeling integrity is enforced in code, not by good intentions.**
  `truth/{live,dead}` (human-verified) is the ONLY label source; `predicted/`
  (CLIP's pre-sort guesses) is NEVER read by train/eval. Grading on CLIP's own
  guesses would be circular target leakage. `gate/labels.py` and
  `evaluate_gate.load_test_split` assert this and error out on empty/too-small truth.
- **Threshold tuned on VAL, never TEST.** The decision threshold is a hyperparameter;
  tuning it on test is back-door leakage. Objective: minimize false negatives
  (the empty-clip bug) subject to an FP-rate cap.
- **Errors reported by downstream cost, not just accuracy.**
  FN = live→dead = frame skipped = empty clips (the bug). FP = dead→live = replay/
  ad processed = garbage events in the box score.

### Labeling rules we settled on (the edge cases — keep consistent for future frames)
Decide every frame by ONE litmus: **"Can the tracker place these players on the
court?"** — i.e. is this a usable full-frame broadcast court shot — NOT "is the
game clock running."
- **LIVE:** play in motion; **free throws**; **inbounds**; **foul-lulls where the
  camera stays on the wide/standard court** with players visible.
  - Why foul-lulls are live: a still frame of players standing during a foul is
    visually indistinguishable from a live half-court set, so labeling them "dead"
    would put contradictory labels on identical-looking frames and poison training.
    Excluding dead-time-on-court is a job for a later game-state/possession filter
    that has temporal context, not for this single-frame gate.
- **DEAD:** replays; close-ups; **tight group shots of 3–4 players talking**
  (court geometry lost → not processable); crowd; bench/timeout/huddle; ads;
  studio desk; scoreboard/stat graphics; **split-screen / composited / squeezed-
  game frames** (even though real action is present — the game is geometrically
  distorted/boxed, so the pipeline can't use it).
- **When genuinely unsure** (mid-zoom, the half-second as the camera pushes in):
  hit `s` (skip). A skipped frame beats a noisy label.

### What got built
`config.py`; `gate/` (`backbones.py`, `zero_shot.py`, `trained_head.py`,
`hsv_baseline.py`, `labels.py`, `common.py`); entry points `extract_frames.py`,
`presort_frames.py`, `label_frames.py` (keypress labeler), `train_gate.py`,
`evaluate_gate.py`; `README.md`, `requirements.txt`, `.gitignore`.

### Dataset
1050 frames extracted (7 clips, ~1 frame/1.5s) → CLIP pre-sorted into `predicted/`
→ hand-labeled with `label_frames.py` into `truth/`: **659 live / 391 dead**.
Deterministic stratified 70/15/15 split saved to `models/split.json`
(train=734, val=158, test=158).

### Results (held-out TEST, 158 frames: 99 live / 59 dead)
| approach | acc | live R | dead R | FN | FP |
|---|---|---|---|---|---|
| **trained head** | **0.987** | 0.990 | 0.983 | **1** | **1** |
| zero-shot CLIP | 0.886 | 0.848 | 0.949 | 15 | 3 |
| HSV (old gate) | 0.873 | 1.000 | 0.661 | 0 | 20 |

Trained head decisively wins. Baselines fail in opposite directions: zero-shot is
too aggressive on "dead" (15 dropped live frames ≈ the empty-clip bug); the HSV
gate is too permissive (20 dead frames let through) and its color band is brittle
to arena/lighting. Threshold lever (TEST): `0.65 → FN=0/FP=2`, `0.70 → FN=1/FP=1`
(saved default = VAL-tuned 0.70). Artifacts: `models/trained_head.joblib`,
`models/thresholds.json`, `reports/metrics.{json,txt}`, `reports/viz/*.png`.

### Residual errors (both = the hard cases we predicted)
- **1 FN:** an unorthodox low/far, crowd-heavy angle (scored 0.65, just under
  threshold). CLIP's "live game" prototype is the canonical sideline-elevated view,
  so atypical angles score low. Fix: add more such angles to `truth/live`.
- **1 FP:** a split-screen/composited frame (ESPN logo + boxed game, scored 0.80).
  The live half dominates the pixels. A single-frame model can't cleanly know the
  game is composited — the principled fix is a later context/score-bug signal,
  not a better still-image classifier.

### Open threads / next time
1. Integrate: `TrainedHeadGate.load()` + threshold 0.70 → replace old `_is_court_visible`.
2. Harden FN: label more unorthodox-angle live frames, retrain (cheap — embeddings cached).
3. FP on split-screens/replays-without-graphics is a known ceiling of single-frame
   models; defer to a temporal/score-bug-aware stage downstream.
4. Possible: group-aware split (no two frames from the same clip across train/test)
   to remove mild optimism in the scores. Current split is plain stratified.
5. Then: move to the next pipeline component (still out of scope here).
