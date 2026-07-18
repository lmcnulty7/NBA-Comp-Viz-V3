# LABEL_SCHEMA.md — per-frame label schema v2 (Foundation Refresh)

The contract for the external auto-labeling pipeline (2026-07-17). Labels come
from OUTSIDE teachers only — the in-house models participate as comparators
for qualification/triage, never as label sources (self-training inbreeds
false negatives: a model's systematic misses never surface for verification).

## Teachers
- **Pass 1 (Colab GPU, `colab_autolabel.py`)**: Grounding DINO (open-vocab
  boxes, LOW threshold — over-propose to surface the FN class) + YOLO-pose
  (person keypoints). The in-house detector runs alongside for the
  agree/teacher-only/pipeline-only diff — comparison output, not labels.
- **Pass 2 (local + Anthropic API, `adjudicate_labels.py`)**: Claude judges
  every proposal (class, tightness, attributes) + proposes court landmarks +
  frame type. VLM = judge/semantics; geometry = precision (court template
  projected through a RANSAC H fit over Claude's landmark proposals).

## Box classes (pass 1 proposes, pass 2 adjudicates)
| class | status in current pipeline | why pertinent |
|---|---|---|
| player | trained (654 imgs), recall ~0.74 in scrums | retrain target — the core |
| referee | trained | FP suppression negative |
| rim | trained (0.99 mAP), **unused** | far-court homography anchor |
| backboard | NOT equipped | second far-court anchor |
| scorebug | NOT equipped (hand-calibrated crops per layout) | kills per-broadcast manual calibration |
| ball | trained, weak, unused | collect (free from teacher); consumption stays kill-listed |

## Per-player-box attributes (pass 2 only — pipeline has nothing here)
| attribute | values | why pertinent |
|---|---|---|
| team_kit | light / dark / other | supervised replacement for per-game k-means (13% track error — the matchup polluter) |
| on_court | yes / no | bench players corrupt the >5-team gate |
| occlusion | visible / partial / heavy | scrum-robust training + crop selection for OCR/re-ID |
| number_visible + value | flag + 0–99 | targeted OCR (read the number region, not the whole torso) |

## Pose (pass 1)
17-pt COCO person keypoints per player box (ankles are the payoff: court
position currently = bbox-bottom-center, the dominant foot-point noise term).

## Court (pass 2 + geometry)
33-point court scheme: Claude proposes visible named landmarks → RANSAC H →
project template → all 33 points, residual-gated. Template is ground truth;
neither the VLM nor the old model sets precision.

## Frame-level (pass 2)
shot_type: wide_broadcast / closeup / replay / graphic / split_screen —
supersedes the binary gate label; the gate's known failure modes are classes.

## Explicitly NOT labeled
Court zones (derivable from H), player masks (pipeline consumes boxes),
cross-frame identity (jersey-OCR route), temporal events (kill list).

## Anti-inbreeding rules
1. Teacher must beat the in-house model on the HUMAN-labeled evals before its
   labels are trusted (`colab_autolabel.py --qualify` vs data/tracking/box_truth).
2. Teacher–pipeline agreements: auto-accept (sampled into human audit).
   Disagreements: adjudicated and INCLUDED — never dropped for being contested.
3. Human audit of ~300 random ACCEPTED labels → measured auto-label error rate
   (goes in the report; makes the dataset citable).
4. Retrained models adopted only if they beat the existing human-labeled
   held-out evals; end-to-end arbiter = PBP cross-val canary (91.1%).
