"""
config.py — central configuration for the court-visibility gate.

Scope reminder: this project builds ONLY the per-frame "live game footage vs.
dead ball" gate and its evaluation harness. Nothing downstream (player/ball/
court/team/identity/events) lives here.

Everything reproducible flows through here: the fixed random seed, all on-disk
paths, the CLIP backbone name, and the zero-shot prompt sets.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# Apple MPS lacks a few torchvision ops (notably torchvision::nms used by YOLO).
# Fall back to CPU for just those ops instead of crashing. MUST be set before torch
# is imported anywhere — config.py is imported first by every entry point, so here.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── This project's root ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── External READ-ONLY source recovered in Phase 0 ────────────────────────────
# The previous project. We read from it (frames) but never write to it and never
# copy its source files wholesale.
OLD_PROJECT_PATH = Path(
    "/Users/lucienmcnulty/Documents/Data Science projects/Basketball_Defensive_Vision"
)
# The only .mp4 clips on disk live in the old project's data/raw (7 clips).
VIDEO_DIR = OLD_PROJECT_PATH / "data" / "raw"

# ── Dataset layout ────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "visibility"
UNSORTED_DIR = DATA_DIR / "_unsorted"                  # raw extractor output

PREDICTED_DIR = DATA_DIR / "predicted"                 # CLIP's GUESSES — never labels
PREDICTED_LIVE = PREDICTED_DIR / "live"
PREDICTED_DEAD = PREDICTED_DIR / "dead"

TRUTH_DIR = DATA_DIR / "truth"                          # human-verified — the ONLY labels
TRUTH_LIVE = TRUTH_DIR / "live"
TRUTH_DEAD = TRUTH_DIR / "dead"

# ── Artifacts ─────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIZ_DIR = REPORTS_DIR / "viz"

SPLIT_PATH = MODELS_DIR / "split.json"
HEAD_PATH = MODELS_DIR / "trained_head.joblib"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
CONFIG_RECORD_PATH = MODELS_DIR / "config_record.json"
METRICS_JSON = REPORTS_DIR / "metrics.json"
METRICS_TXT = REPORTS_DIR / "metrics.txt"


def emb_cache_path(backbone: str) -> Path:
    """Per-backbone embedding cache (CLIP=512-d, DINOv2=768-d differ)."""
    return MODELS_DIR / f"emb_cache_{backbone}.pkl"


# ── Backbone ──────────────────────────────────────────────────────────────────
# open_clip is not installed; we load the identical model via transformers.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINOV2_MODEL_NAME = "facebook/dinov2-base"  # optional, behind --backbone dinov2

# ── Classes ───────────────────────────────────────────────────────────────────
# Positive class for all metrics is "live". Index convention: 0=dead, 1=live.
LIVE = "live"
DEAD = "dead"
CLASSES = [DEAD, LIVE]   # CLASSES[label_int]

# ── Labeling-integrity thresholds ─────────────────────────────────────────────
MIN_PER_CLASS_HARD = 10    # below this per class → hard error (can't split/train)
MIN_PER_CLASS_WARN = 250   # below this per class → warn (test won't be meaningful)

# ── Zero-shot prompts (Phase 0: NONE were recovered) ──────────────────────────
# The shipped gate used HSV floor-color thresholding, not CLIP, so there are no
# prior prompts to reuse. These are sensible defaults for the CLIP zero-shot
# baseline (Approach A). The HSV rule is preserved separately as a baseline.
LIVE_PROMPTS = [
    "a wide broadcast shot of a basketball game in progress on the court",
    "a basketball game seen from the side broadcast camera with players on the full court",
    "live basketball game play with several players spread across the court",
    "an elevated wide angle view of a basketball court during a game",
    "professional basketball players running on the court during live play",
    "a half court basketball possession with offense and defense visible",
]
DEAD_PROMPTS = [
    "a close-up of a single basketball player's face",
    "a slow motion instant replay of a basketball play",
    "the crowd and spectators in the stands at a basketball arena",
    "a television commercial advertisement",
    "basketball coaches and players on the bench during a timeout",
    "an on-screen scoreboard or broadcast graphic overlay",
    "studio analysts talking on a sports broadcast desk",
    "a tight close-up of two players near the basket",
    "a referee or official shown in close-up",
]

# ── HSV gate (recovered verbatim from old runner.py:479) ──────────────────────
HSV_LO = (8, 25, 90)
HSV_HI = (38, 210, 245)
HSV_MIN_FLOOR_FRACTION = 0.12
HSV_MAX_FLOOR_FRACTION = 0.55


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch. Called at the top of each entry point."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


def get_device() -> str:
    """CUDA (Colab harvest) > MPS (this Mac) > CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — Player detection + tracking
# ══════════════════════════════════════════════════════════════════════════════
# Detection + tracking in one ultralytics call (YOLO .track()). Person class only;
# referee/player separation is a later (team-classification) concern.
# Reuse the old project's YOLOv8m weights if present; else ultralytics auto-downloads.
_OLD_YOLO = OLD_PROJECT_PATH / "yolov8m.pt"
YOLO_WEIGHTS = str(_OLD_YOLO if _OLD_YOLO.exists() else "yolov8m.pt")
PLAYER_CONF = 0.40
PLAYER_IOU = 0.45
PERSON_CLASS = 0

# BoT-SORT: motion + appearance ReID + camera-motion compensation (GMC), which is
# the key win for broadcast pans/zooms. "botsort.yaml" = ultralytics' bundled config.
TRACKER_CONFIG = "botsort.yaml"

# Camera-cut handling: ByteTrack/BoT-SORT keep IDs indefinitely, so a broadcast cut
# would drag stale IDs onto new players. Flag a cut when > this fraction of active
# tracks (>= min) vanish in one frame, then reset the tracker.
CAMERA_CUT_VANISH_FRAC = 0.80
CAMERA_CUT_MIN_TRACKS = 4

# Basketball-specific detector (trained on player-detection-3 → replaces generic COCO person).
# Classes: 0=player 1=referee 2=ball 3=rim 4=number. The tracker tracks PLAYER_CLASS_ID only,
# which excludes referees AND fans (a basketball-trained model won't fire "player" on crowd).
PLAYER_DETECTOR_WEIGHTS = MODELS_DIR / "player_detector.pt"
PLAYER_DETECTOR_DATA = PROJECT_ROOT / "data" / "external" / "player_det3_remap" / "data.yaml"
PLAYER_DETECTOR_BASE = "yolov8m.pt"
PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 1

# Outputs
TRACKING_DIR = PROJECT_ROOT / "data" / "tracking"          # tracks JSON + overlay mp4
TRACK_BOX_TRUTH = PROJECT_ROOT / "data" / "tracking" / "box_truth"  # hand-labeled boxes (eval)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — Court keypoints + homography
# ══════════════════════════════════════════════════════════════════════════════
# The old project's homography MATH was complete; it failed only because the
# keypoint dataset was never labeled. So this component is a keypoint-detection
# problem: label court landmarks → train YOLOv8-pose → feed homography.
COURT_DIR = PROJECT_ROOT / "data" / "court"
COURT_KP_DATASET = COURT_DIR / "kp_dataset"          # YOLO-pose dataset (images/ labels/ train/ val/)
COURT_KP_LABELS = COURT_DIR / "kp_labels"            # raw per-frame keypoint JSON from the labeler
COURT_KP_WEIGHTS = MODELS_DIR / "court_kp_yolo.pt"   # trained pose weights (legacy 20-pt scheme)
COURT_KP_BASE = "yolov8n-pose.pt"                    # base pose model to fine-tune
# 33-point scheme (external court-detection-2 dataset, trained on Colab GPU) — court/court33.py.
COURT_KP33_WEIGHTS = MODELS_DIR / "court_kp33.pt"
COURT_KP_DATASET_YAML = COURT_DIR / "court_kp.yaml"

# Snapped-label generation (July 2026): models trained on projection labels from the
# human-verified + line-snapped homographies (see DEVLOG 2026-07-01→04). The grid
# model + snap-tracker (court/snap_track.py) is the pipeline's court solver now.
COURT_KP33_SNAPPED_WEIGHTS = MODELS_DIR / "court_kp33_snapped.pt"   # 33-pt retrain (benchmark)
COURT_GRID_WEIGHTS = MODELS_DIR / "court_grid_snapped.pt"           # 13×7 grid model (deployed)
COURT_USE_GRID_TRACKER = os.environ.get("COURT_TRACKER", "1") != "0"
                                 # CourtMapper uses CourtTracker (grid+snap+temporal);
                                 # set COURT_TRACKER=0 to A/B against the legacy 33-pt detector
COURT_GRID_KP_CONF = 0.5         # grid model is well-calibrated; no need for the 0.05 crutch
COURT_TRACK_MAX_HELD = 20        # consecutive unconfirmed (HELD) updates before LOST
                                 # (at stride 3 / 30 fps ≈ 2 s of coasting)

RANSAC_REPROJ_THRESHOLD = 2.0    # FEET (dst space), for cv2.findHomography RANSAC.
                                 # Swept on val: 2.0 → median 0.30 ft reproj, 100% success
                                 # (was 5.0 = too loose, let noisy paint-corner kps skew H).
MIN_KEYPOINTS_FOR_H = 4          # ≥4 non-collinear correspondences → valid H
COURT_KP_CONF = 0.05             # min confidence to accept a predicted keypoint.
                                 # Swept on the 11445-13965 gameplay window: 0.30→0.05 nearly
                                 # doubles landmarks (12→23/frame) and grows keypoint coverage
                                 # 25%→42% of the frame, with reproj err ~unchanged (0.60→0.65 ft,
                                 # RANSAC@2px rejects the noisy ones). Less extrapolation = the
                                 # fix for off-court projections on the weak (far) side / in transition.

# Court-masking: a detection is kept only if its foot-point projects to within the
# court + this margin (ft). Margin keeps inbounders / near-out-of-bounds players while
# dropping bench (~10+ ft off sideline) and crowd (way off). Court is 94×50 ft.
COURT_MARGIN_FT = 5.0
# Only trust a homography for MASKING when it's well-determined; otherwise a slightly
# wrong H misplaces the court polygon and drops real players. On low-confidence frames
# we keep all detections. (court_pos for stats is still computed regardless.)
COURT_MASK_MIN_INLIERS = 6
COURT_MASK_MAX_REPROJ_FT = 0.6

# Line-based homography refinement: bootstrap H from keypoints, then snap the projected
# court LINES onto detected court-line pixels and re-solve (points + lines).
# ON: with the LEARNED line-segmentation mask it's net-positive (~+0.4 px residual, never
# worse thanks to the monotonic guard). It sharpens a roughly-right H; it can't rescue a
# wrong one (that needs better keypoints — #1). Falls back to top-hat if the seg model is absent.
COURT_REFINE = True
COURT_REFINE_ITERS = 3          # ICP-style: re-project → re-match → re-solve
COURT_REFINE_SEARCH_PX = 20     # perpendicular search window for the nearest line pixel
COURT_REFINE_STEP_FT = 2.0      # sampling interval along each court line
COURT_LINE_TOPHAT_K = 9         # white top-hat kernel (> line width) to isolate thin court lines
COURT_LINE_THRESH = 8           # top-hat brightness threshold (low: NBA lines are low-contrast on wood)

# Learned court-line segmentation (replaces top-hat). Labels are bootstrapped for free by
# projecting the court template through each keypoint-labeled frame's homography.
LINE_DATASET = COURT_DIR / "line_dataset"        # images/ + masks/ (train/val)
LINE_SEG_WEIGHTS = MODELS_DIR / "court_line_seg.pt"
LINE_MASK_THICKNESS = 3          # px width to rasterize projected court lines into the label mask
LINE_SEG_IMGSZ = 512             # segmentation input size


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT B — Player identity (re-ID) + foot-point stability
# ══════════════════════════════════════════════════════════════════════════════
# After the court fix (DEVLOG 2026-07-05) the remaining trajectory teleports are
# player-tracking artifacts: BoT-SORT ID fragmentation and bbox foot-point jitter.
# Three levers, each independently A/B-able:
#   1. Native BoT-SORT appearance re-ID (within-shot) — botsort_reid.yaml.
#      TRACKER_REID=0 env reverts to the bundled motion-only botsort.yaml.
#   2. FootPointStabilizer (detect/footpoint.py) — pixel-space foot correction
#      BEFORE the homography amplifies it. --no-stab in build_trajectories.
#   3. Offline fragment linker (detect/reid.py) — CLIP torso embeddings + court-
#      space motion gating merges the fragments of one player. --no-reid.
TRACKER_REID_YAML = PROJECT_ROOT / "botsort_reid.yaml"
# Native BoT-SORT re-ID: DEFAULT OFF (2026-07-08c). Verdict history: kept at first
# because it was free with no measured fragment benefit (07-05b) — then it crash-
# LOOPED at harvest scale (ultralytics numpy-no-cpu bug re-firing on every post-
# reset frame, zero tracks for whole stretches). Identity value lives in the
# offline fragment linker, not here. TRACKER_REID=1 re-enables for A/B only.
TRACKER_NATIVE_REID = os.environ.get("TRACKER_REID", "0") == "1"
if TRACKER_NATIVE_REID and TRACKER_REID_YAML.exists():
    TRACKER_CONFIG = str(TRACKER_REID_YAML)      # overrides the Component-2 default above

# Foot-point stabilizer. Units are PROCESSED frames (i.e. after stride), not video frames.
FOOT_HEIGHT_WIN = 15        # bbox-height history per track; median ≈ standing height at current zoom
FOOT_OCCLUSION_FRAC = 0.85  # box shorter than this × median ⇒ bottom clipped (occlusion) → re-extend.
                            # 0.85 tolerates crouching (~10-15% shorter) but catches leg occlusion.
FOOT_EMA_ALPHA = 0.6        # EMA weight of the CURRENT foot obs (1.0 = no smoothing).
                            # Kills ±2-4 px box-edge jitter (≈0.5+ ft at the far court) with ~1-frame lag.
FOOT_MAX_GAP = 10           # unseen longer than this ⇒ reset that track's filter state

# Fragment linker (offline, after the streaming pass).
REID_CROP_EVERY = 3         # collect a torso crop every Nth processed frame per track
REID_MAX_CROPS = 40         # cap crops per fragment (memory + embed cost)
REID_MIN_CROPS = 2          # fragments with fewer crops have no reliable appearance → never linked
REID_MAX_GAP_SEC = 4.0      # max real-time gap between fragment end and candidate start
REID_MAX_SPEED_FTS = 30.0   # NBA sprint ≈ 21 mph ≈ 30 ft/s — motion-feasibility gate (court space,
                            # comparable ACROSS camera cuts thanks to the homography)
REID_DIST_SLACK_FT = 6.0    # allowance for foot-point + homography noise at the endpoints
REID_SIM_MIN = 0.82         # CLIP cosine floor. Measured on the A/B window: sims cluster 0.89-0.98
                            # (jersey color dominates), so appearance is a FLOOR + TIEBREAK only.
REID_DIST_WEIGHT = 0.3      # candidate score = sim − w · dist/(max_speed·gap + slack).
                            # Measured: spatial margins are decisive (19/26 fragments had a >6 ft
                            # clear winner) where sim margins (<0.02) never were — so the motion
                            # term must dominate the ranking, appearance breaks ties.
REID_AMBIG_MARGIN = 0.02    # top-two candidates within this COMBINED-score margin ⇒ DON'T merge.
                            # A false merge poisons two players' stats; a missed merge is recoverable.
# Team-veto continuity override: the team signal (~87% track accuracy) must not
# outvote overwhelming continuity evidence. A cross-team candidate this close in
# appearance, space, and time is the same player with a misassigned team.
REID_VETO_OVERRIDE_SIM = 0.97
REID_VETO_OVERRIDE_GAP_S = 2.0
REID_VETO_OVERRIDE_DIST_FT = 8.0

# ── Component C1 — team classification (detect/teams.py, unsupervised per clip) ──
# k-means over median-Lab jersey color of the re-ID torso crops. NOT CLIP embs —
# measured: on 30-70px crops CLIP is dominated by scene stats, split 25v1 tracks
# at silhouette 0.247 (Lab color: clean ~5v5, see DEVLOG 2026-07-05c).
TEAM_N_CLUSTERS = 2         # team A / team B; refs are already excluded by the detector class
TEAM_ABSTAIN_RATIO = 0.75   # near-centroid dist / far-centroid dist above this ⇒ abstain (None):
                            # the track's pooled color sits between the kits — bench/crowd leakage
                            # and hopelessly contaminated tracks stay out of BOTH teams
TEAM_MIN_CROPS = 2          # tracks with fewer VALID (non-floor) crop features abstain
                            # (one crop is not evidence; mostly-floor crops carry none)
TEAMS_EVAL_DIR = PROJECT_ROOT / "data" / "teams_eval"   # human-label eval (evaluate_teams.py)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT C2 — Possession segmentation (segment_possessions.py)
# ══════════════════════════════════════════════════════════════════════════════
# Segments trajectories into halfcourt possession spans (which basket is attacked)
# and assigns offense/defense per span. Ball-free v1: occupancy geometry only.
POSS_HALF_MARGIN_FT = 8.0   # occupancy median must be this deep past midcourt (47 ft)
                            # to call a halfcourt set; the band between = transition
POSS_SMOOTH_SEC = 1.0       # rolling-median window on the occupancy signal
POSS_MIN_SPAN_SEC = 3.0     # halfcourt runs shorter than this stay "transition"
POSS_MAX_GAP_SEC = 3.0      # frame gaps (gate skips / lost H) longer than this break a span
POSS_MIN_PLAYERS = 4        # frames with fewer positioned players don't vote on occupancy
BASKET_LEFT = (5.25, 25.0)  # rim centers, corner-origin court feet (court/court33.py)
BASKET_RIGHT = (88.75, 25.0)
# Boundary semantics (eval 2026-07-06: ALL basket errors were late boundaries, zero
# mid-set errors): a possession = approach (transition) + halfcourt set. The span
# ends at RETREAT ONSET (sustained occupancy motion back toward midcourt) and the
# outbound frames + following transition belong to the NEXT possession's approach.
# Offense/defense is computed on the SET portion only — during the approach the
# OFFENSE leads the defense downcourt, so the closer-team geometry inverts there.
POSS_RETREAT_FTS = 4.0      # sustained occupancy speed toward midcourt ⇒ outbound
POSS_RETREAT_SUSTAIN_S = 0.7  # velocity must be sustained this long (drives/kicks spike briefly)
POSS_CORE_TRIM_S = 2.0      # metrics core = [set_start, end − this]: the human eval measured
                            # 100% accuracy >2 s from a boundary, 84% inside — so the core is
                            # the certified region and boundary-adjacent frames are excluded
POSS_MIN_CORE_S = 4.0       # spans with a shorter core ⇒ metrics_eligible=false (flagged, kept)


# Harvest-scale pre-gate (build_trajectories --pregate): a coarse gate-only pass
# finds live segments so the expensive chain never scans dead footage frame by
# frame. Padding guarantees the fine pass sees everything near a coarse live hit;
# accuracy-neutrality vs the fine-only path verified on curry_q1 (DEVLOG 07-08c).
PREGATE_STRIDE_SEC = 0.5    # coarse sampling — a live stretch shorter than
                            # this is below POSS_MIN_SPAN_SEC anyway
PREGATE_PAD_SEC = 1.5       # ± padding around each coarse live hit


def pregate_params(fps: float) -> tuple[int, int]:
    """(stride_frames, pad_frames) at this fps. Time-based, not frame-based —
    the validated 30fps values were 15/45; a 720p60 game at those FRAME counts
    scanned twice as often and padded half as much in time (run-9 postmortem)."""
    return (max(1, round(PREGATE_STRIDE_SEC * fps)),
            max(1, round(PREGATE_PAD_SEC * fps)))


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT C3 — Matchup metrics (matchup_metrics.py), Tier 1
# ══════════════════════════════════════════════════════════════════════════════
C3_MIN_SPAN_CONF = 0.60     # offense/defense call must clear this before computing matchups
                            # (a wrong offense call corrupts every metric in the possession)
C3_CLOSEOUT_SMOOTH_S = 0.7  # rolling-median on each matchup-distance series before any rate
C3_CLOSEOUT_WIN_S = 1.0     # closing-rate = central difference over this window — heavily
                            # smoothed and reported DIRECTIONALLY (closing/holding/retreating),
                            # never as precise ft/s: it rides on frame-to-frame tracking noise
C3_MIN_PAIR_RUN_S = 1.5     # a (defender, man) assignment run shorter than this contributes
                            # distance stats but no closing-rate samples (too short to smooth)
