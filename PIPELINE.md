# PIPELINE.md — end-to-end reference (as of 2026-07-08)

One page per the whole chain: stage → input → output → **validated number** →
known residual → what to watch when scaling to hundreds of possessions.
Numbers are from held-out or human-labeled evals unless marked label-free;
provenance for every number is in the DEVLOG entry cited.

```
broadcast video
  → [1] court-visibility gate        (live vs dead footage)
  → [2] detection + tracking         (boxes, fragment ids)
  → [3] court homography             (pixels → court feet)
  → [4] identity                     (foot-point stability, fragment linking)
  → [5] teams                        (A/B kit per track, linker veto)
  → [6] possessions                  (approach+set spans, offense/defense)
  → [7] matchups                     (Hungarian pairs on set cores)
  → [8] clock/PBP/outcomes           (real teams, points, terminating events)
  → tier2 join                       (possession outcome × defenders)
```

| # | Stage (code) | Input → Output | Validated number | Residual limitation |
|---|---|---|---|---|
| 1 | Gate (`gate/`, CLIP probe) | frame → live/dead | **98.7%** acc, held-out in-domain (DEVLOG 06-23) | Domain shift: unfamiliar broadcasts score 0.47–0.66 vs 0.70 threshold — harvesting runs at 0.35 (07-05) |
| 2 | Detection+tracking (`detect/`, YOLOv8m + BoT-SORT) | frames → boxes + fragment ids | det **P .89 / R .87** @IoU vs hand labels (06-28) | Per-frame recall ceiling ~.74 on paint scrums; ids are per-shot fragments; ultralytics native-reID crash guarded (07-07) |
| 3 | Homography (`court/`, grid model + snap tracker) | frame → H (px→ft) | H-err **p50 1.7 px**, 279 val frames; unseen games 66–89% sane-H (07-03/04) | Far-court extrapolation tail (physics p99 ~40 ft); era/floor-design sensitivity on unseen footage |
| 4 | Identity (`detect/footpoint.py`, `detect/reid.py`) | fragments → canonical tracks | impossible steps 15.4→**14.3%**; churn 6.57→**5.0** (label-free, 07-05b/c) | Identity = fragment, NOT player (needs jersey OCR); churn still ~5× (conservative refusals by design) |
| 5 | Teams (`detect/teams.py`) | track crops → team A/B | **87.1% track** / 83.9% crop, 160 labels (07-06) | ~13% track misassignment (crops lacking jersey: occluders/face framing) — surfaces downstream as >5-team frames; A/B not home/away by itself |
| 6 | Possessions (`segment_possessions.py`) | trajectories → spans (basket, offense, set core) | **96.5% basket / 91.2% offense**, 73 labels; **100%/100% >2 s from boundaries** (07-07) | ±1–2 s boundary fuzz (hence set cores); no within-occupancy possession change; FT clusters read as sets; rare band-edge stretch unsegmented |
| 7 | Matchups (`matchup_metrics.py`) | spans + trajectories → defender/man pairs, distance, spacing, closeout | coverage **2.5–4.0 pairs/frame** (honest, post-ghost-audit 07-07c/d); structural checks only | No labeled matchup GT yet; closeout is DIRECTIONAL only; >5-team frames hard-excluded (7–62%/possession, `degraded` flag >40%) |
| 8 | Clock/PBP/outcomes (`clock_reader.py`, `fetch_pbp.py`, `align_outcomes.py`) | scorebug + bref PBP → real teams, points, terminating event | clock **100%** on readable, 0 confident-wrong (56 labels); alignment **9/9** offense cross-val, 100% overlap (07-08) | ~10% anchor failure (retryable); one-time clock-box calibration per broadcast layout; PBP era ≥2000-01; light-kit=home assumption pre-2017 |
| — | Tier 2 join (`tier2_join.py`) | outcomes × matchups → per-possession credit rows | **7/7 team-consistency** across independent pipelines; hard dedupe + assert (07-08b) | NO aggregates yet — n=7 is a correctness sample; defender bucket = fragment until jersey OCR |

## Cross-cutting invariants (violate at your peril)
- **Raw-support masking**: the trajectories `cleaned` series interpolates gaps
  unmarked (20–30% invented presence) — every consumer must mask presence with
  the `raw` series (07-07c).
- **Artifact verification**: macOS AVFoundation silently fails to overwrite
  mp4s — unlink before write, check `isOpened()`, and verify renders by
  content (pixel-scan), never by log line (07-07d).
- **Exclusions are hard filters with visible reasons** (>5-team gate,
  `degraded`, `duplicate_of_span` + assert), never metadata-only.
- Truth vs predicted separation in every eval; predictions never shown while
  labeling; `--repredict` re-scores code changes against existing labels.

## What "success" looks like per stage at harvest scale (label-free canaries)
1. Gate: live-fraction of broadcast ≈ 35–50%; sudden 0% = threshold/domain issue.
2. Tracking: median players/frame 7–10 on live spans; camera-cut resets present.
3. Homography: frames-with-H ≥ ~85% of live; physics impossible-steps ≤ ~25% raw.
4. Identity: churn-after ≤ ~2× churn-before reduction seen in-domain (≥25% merges).
5. Teams: silhouette ≥ ~0.45; frame balance median within [3,5]v[3,5]; abstained ≤ ~40% of tracks.
6. Possessions: 75–98% window coverage; all span durations ≤ 24 s + rebounds;
   within-clip basket/offense consistency 100% (a violation = team or court failure).
7. Matchups: excluded-frames ≤ ~40%/possession typical; degraded rate ≤ ~20% of spans.
8. Alignment: anchor success ≥ ~85%; **possession-team cross-val ≥ ~90%** —
   this is the end-to-end health metric; a drop means an upstream stage broke.
```
Joinable yield measured on the validated clips: spans → eligible 10/12 →
aligned 9/10 → joined 7/9  ⇒  ~58% span→join composite.
```
