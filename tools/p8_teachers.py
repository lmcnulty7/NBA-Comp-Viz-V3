"""tools/p8_teachers.py: qualify outside teacher detectors on the existing human box truth (P8 step 0, Track B;
PLAN.md D131, D132; LABEL_SCHEMA.md anti-inbreeding rule 1).

No new human labels of any kind: the only truth is the 50 hand-drawn files of config.TRACK_BOX_TRUTH (one class, no
class field). Nothing is trained, tuned or calibrated.
Each teacher runs once with its published inference settings; only its confidence threshold is swept, and the sweep
is part of the rule below, scored inside every bootstrap replicate.

    python3 -m tools.p8_teachers frames                 # guard every box-truth file, write the frame list
    python3 -m export.heavy --name "p8 teacher X" -- <venv python> -m tools.p8_teachers run --model X
    python3 -m tools.p8_teachers score                  # numpy only: every number of the report

`run` needs the teacher venv (torch, ultralytics, rfdetr; never the system Python, CLAUDE.md environment). `frames`
and `score` run under /usr/local/bin/python3. Weights, predictions and results live in the P8 teacher workspace
(default <home>/Developer/nba-v3-worktrees/_p8-data/teachers, or V3_P8_TEACHERS), outside the repository and
DATA_ROOT.

## Rules fixed before any teacher is run (2026-11-21)

Matching (colab_autolabel.py --qualify, pr_vs_gt): per frame, predictions in descending confidence take, one by one,
the unmatched truth box of highest IoU; a match needs IoU >= 0.5. A prediction is a true positive when it matches.
Precision and recall are pooled over frames.

Truth and classes: the box truth holds players only. Our detector (models/player_detector.pt, config.PLAYER_CONF
0.40, config.PLAYER_IOU 0.45) is scored on its player boxes (the primary comparator, the 2026-07-18 qualification's
class choice); its player and referee boxes together are a report-only row. A COCO teacher is scored on its person
class, so a referee or an off-court person it boxes counts against it, as it would as a player label; a report-only
row also removes the teacher boxes that overlap one of our referee boxes (IoU >= 0.5). The SportsMOT teacher is scored
on its single class.

Change made before any teacher ran (2026-11-21, commit after e0e942f3): the first version of this rule read
label_boxes.py's docstring ("include refs") and made player plus referee boxes the primary comparator. Our
detector's own run then showed 130 referee boxes (median confidence 0.89, 7 of them on a player box) of which only 4
match a truth box, and the frames show the referees unboxed: the files hold players only. The primary comparator
became the player boxes; no teacher number existed when it changed.

Device: CPU for every model, our detector included (docs/research/P8_PLATFORM.md: MPS finds fewer player boxes).

Teacher sweep: each teacher runs once at a floor of 0.05 confidence. Its recall at our detector's precision is the
largest pooled recall over thresholds t >= 0.05 whose pooled precision is at least our detector's; the threshold that
gives it is reported. A teacher that never reaches our precision has recall 0 there.

Interval: 2,000 frame bootstrap replicates (seed 20261121), the same resamples for every model. In each replicate our
detector's precision and recall, the teacher's threshold and its recall at that precision are all recomputed; the
difference is teacher recall at matched precision minus our recall; the interval is its 2.5 and 97.5 percentiles.
A clip bootstrap (the 7 source clips resampled whole) is printed beside it, report-only.

Qualification: a teacher qualifies when its recall at our detector's precision exceeds our detector's recall (the
point estimate of the difference is above zero) and the frame bootstrap interval of the difference lies wholly
above zero. Otherwise it does not qualify, whatever else it scores.

Teacher-only band (step 4): at its matched-precision threshold, the teacher boxes with IoU below 0.5 to every box of
our detector. Their precision is the share that match, greedily at IoU >= 0.5, a truth box our detector did not
match. The same for the boxes two teachers agree on (IoU >= 0.5, the mean of the pair) inside both teacher-only
bands: the boxes D131's two-teacher rule would add.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WHO = "tools.p8_teachers"

IOU = 0.5
FLOOR = 0.05
B = 2000
SEED = 20261121
DET_CONF = 0.40            # config.PLAYER_CONF (tests assert they agree)
DET_NMS_IOU = 0.45         # config.PLAYER_IOU
DET_PEOPLE = ("player", "referee")

# The teachers, fixed before any run. Each keeps its published inference settings; only the threshold is swept.
TEACHERS = {
    "rfdetr_large": {
        "what": "RF-DETR Large (COCO), rfdetr 1.12.1, rf-detr-large-2026.pth, default resolution 704",
        "weights": "rf-detr-large-2026.pth",
        "url": "https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth",
        "license": "Apache-2.0 (code and weights, Nano to Large; github.com/roboflow/rf-detr)",
        "default_conf": 0.5,
        "classes": ("person",),
    },
    "yolox_x_sportsmot": {
        "what": "YOLOX-X trained on SportsMOT (MixSort model zoo, yolox_x_sports_train.pth.tar), test size 800x1440, "
                "NMS 0.7 (exps/example/mot/yolox_x_sportsmot.py)",
        "weights": "yolox_x_sports_train.pth.tar",
        "url": "https://drive.google.com/drive/folders/1pQs1gFC_jG0TlGIUMgf3E0I3OztCvgxI (MixSort model zoo)",
        "license": "MIT code (github.com/MCG-NJU/MixSort); weights trained on SportsMOT, CC BY-NC 4.0: non-commercial",
        "default_conf": 0.1,
        "classes": ("player",),
    },
    "yolo11x": {
        "what": "Ultralytics YOLO11x (COCO), ultralytics 8.4.48, image size 640, NMS 0.7, person class",
        "weights": "yolo11x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "license": "AGPL-3.0 (Ultralytics code and weights)",
        "default_conf": 0.25,
        "classes": ("person",),
    },
}
OURS = "ours"


# ── paths ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def workspace() -> Path:
    from tools.p8_common import check_outside
    root = Path(os.environ.get("V3_P8_TEACHERS") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data" / "teachers")
    return check_outside(root)


def truth_dir() -> Path:
    from tools.p8_common import main_checkout
    env = os.environ.get("V3_BOX_TRUTH")
    if env:
        return Path(env)
    main = main_checkout()
    if main is None:
        raise SystemExit(f"{WHO}: no main checkout found; set V3_BOX_TRUTH")
    return main / "data" / "tracking" / "box_truth"


def detector_weights() -> Path:
    from tools.p8_common import main_checkout
    return Path(os.environ.get("V3_PLAYER_DETECTOR") or main_checkout() / "models" / "player_detector.pt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── matching (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def by_score(preds):
    """Predictions [x1, y1, x2, y2, score, ...] in descending score, ties in input order."""
    return sorted(preds, key=lambda p: -p[4])


def greedy(preds, gts, thr: float = IOU, skip=()):
    """colab_autolabel.pr_vs_gt's matching: each prediction, in the given order, takes the unmatched truth box of highest
    IoU; a match needs IoU >= thr. Truth indices in skip are never matched. -> (tp flag per prediction, matched truth
    indices)."""
    used = set(skip)
    flags = []
    for p in preds:
        best, bi = 0.0, None
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bi = v, i
        hit = bi is not None and best >= thr
        if hit:
            used.add(bi)
        flags.append(hit)
    return flags, used - set(skip)


def prf(tp, n_pred, n_gt) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


# ── curves and the rule (pure, numpy) ───────────────────────────────────────────────────────────────────────────────
class Curve:
    """A teacher's pooled precision-recall curve over a global threshold, for any frame weights. Built from per-frame
    (scores, tp flags) of the greedy match in score order, so the predictions above any threshold are a prefix of each
    frame's list and their flags do not change (the prefix property)."""

    def __init__(self, frames):
        import numpy as np
        sc, tp, fi = [], [], []
        for f, (scores, flags) in enumerate(frames):
            sc += list(scores)
            tp += [float(x) for x in flags]
            fi += [f] * len(scores)
        order = np.argsort(-np.asarray(sc, float), kind="stable")
        self.score = np.asarray(sc, float)[order]
        self.tp = np.asarray(tp, float)[order]
        self.frame = np.asarray(fi, int)[order]
        # the last position of each run of equal scores: a threshold keeps a whole run or none of it
        if len(self.score):
            self.ends = np.flatnonzero(np.r_[self.score[1:] != self.score[:-1], True])
        else:
            self.ends = np.zeros(0, int)

    def at(self, w, n_gt):
        """Weights per frame, the weighted truth count -> (threshold, precision, recall) arrays at every run end."""
        import numpy as np
        ww = w[self.frame]
        ctp = np.cumsum(ww * self.tp)[self.ends]
        cn = np.cumsum(ww)[self.ends]
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(cn > 0, ctp / cn, 0.0)
        rec = ctp / n_gt if n_gt else np.zeros_like(ctp)
        return self.score[self.ends], prec, rec


def recall_at_precision(thr, prec, rec, p_min, valid=None):
    """Largest recall among thresholds with precision >= p_min -> (recall, threshold), (0.0, None) when none. Ties in
    recall go to the higher threshold."""
    import numpy as np
    ok = prec >= p_min - 1e-12
    if valid is not None:
        ok &= valid
    if not ok.any():
        return 0.0, None
    idx = np.flatnonzero(ok)
    best = idx[np.argmax(rec[idx])]
    return float(rec[best]), float(thr[best])


def qualifies(diff: float, lo: float) -> bool:
    """The fixed rule: the point difference above zero and the interval wholly above zero."""
    return diff > 0 and lo > 0


def percentile_interval(xs):
    import numpy as np
    xs = np.asarray(xs, float)
    return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))


# ── box truth and the guard ──────────────────────────────────────────────────────────────────────────────────────────
def clip_games() -> tuple:
    """fetch_pbp.py's CLIP_GAME and GAMES, read with ast (that module imports requests and bs4 at the top)."""
    tree = ast.parse((REPO / "fetch_pbp.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in (
                "CLIP_GAME", "GAMES"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out["CLIP_GAME"], out["GAMES"]


def items_for(rec: dict, clip_game: dict, games: dict) -> tuple:
    """Every id, tag and path a box-truth file names -> (items, homes) for tools.p8_common.guard."""
    frame = rec["frame"]
    clip = frame.split("__")[0]
    items = [frame, clip, rec.get("image_path", "")]
    homes = []
    code = clip_game.get(clip)
    if code:
        g = games[code]
        y, m, d = g["date"].split("-")
        items += [code, f"{m}.{d}.{y}.{g['away']}.at.{g['home']}", f"{g['home'].lower()}_{g['away'].lower()}_{y}"]
        homes.append(g["home"])
    return items, homes, code


def check_record(rec: dict, clip_game: dict, games: dict) -> str | None:
    """None when the file passes the guard, else the reason it is dropped (a D85 test item, a spent held-out item, or no
    known game)."""
    from tools.p8_common import guard
    items, homes, code = items_for(rec, clip_game, games)
    if code is None:
        return "no known game for its clip"
    try:
        guard(items, WHO, homes=homes)
    except SystemExit as e:
        return str(e)
    return None


def cmd_frames(args) -> dict:
    clip_game, games = clip_games()
    files = sorted(truth_dir().glob("*.json"))
    kept, dropped = [], []
    for f in files:
        rec = json.loads(f.read_text())
        why = check_record(rec, clip_game, games)
        if why:
            dropped.append({"frame": rec["frame"], "why": why})
            continue
        if not Path(rec["image_path"]).exists():
            dropped.append({"frame": rec["frame"], "why": "image missing"})
            continue
        clip = rec["frame"].split("__")[0]
        kept.append({"frame": rec["frame"], "clip": clip, "game": clip_game[clip], "image_path": rec["image_path"],
                     "boxes": rec["boxes"]})
    out = {"truth_files": len(files), "kept": len(kept), "dropped": dropped,
           "truth_boxes": sum(len(k["boxes"]) for k in kept),
           "games": sorted({k["game"] for k in kept}), "clips": sorted({k["clip"] for k in kept}), "frames": kept}
    ws = workspace()
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "frames.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "frames"}, indent=1))
    return out


# ── model runners (teacher venv) ─────────────────────────────────────────────────────────────────────────────────────
def _yolox_preproc(img, size):
    """MixSort yolox/data/data_augment.preproc with the SportsMOT exp's mean and std (copied: that package's data
    module needs pycocotools)."""
    import cv2
    import numpy as np
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    padded = np.ones((size[0], size[1], 3)) * 114.0
    r = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    padded[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized.astype(np.float32)
    padded = padded[:, :, ::-1] / 255.0
    padded = (padded - mean) / std
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def load_runner(name: str, ws: Path, device: str):
    """name -> (detect(bgr) -> [[x1, y1, x2, y2, score, class name]], versions)."""
    import torch
    versions = {"torch": torch.__version__}
    if name == OURS:
        import ultralytics
        from ultralytics import YOLO
        versions["ultralytics"] = ultralytics.__version__
        model = YOLO(str(detector_weights()))

        def detect(img):
            r = model.predict(img, conf=DET_CONF, iou=DET_NMS_IOU, device=device, verbose=False)[0]
            return [[*map(float, b.xyxy[0]), float(b.conf), r.names[int(b.cls)]] for b in r.boxes]
        return detect, versions
    w = ws / "weights" / TEACHERS[name]["weights"]
    if name == "yolo11x":
        import ultralytics
        from ultralytics import YOLO
        versions["ultralytics"] = ultralytics.__version__
        model = YOLO(str(w))

        def detect(img):
            r = model.predict(img, conf=FLOOR, classes=[0], device=device, verbose=False)[0]
            return [[*map(float, b.xyxy[0]), float(b.conf), "person"] for b in r.boxes]
        return detect, versions
    if name == "rfdetr_large":
        import rfdetr
        from importlib.metadata import version
        versions["rfdetr"] = version("rfdetr")
        model = rfdetr.RFDETRLarge(pretrain_weights=str(w), device=device)

        def detect(img):
            d = model.predict(img[:, :, ::-1].copy(), threshold=FLOOR)
            return [[*map(float, xy), float(c), "person"] for xy, c, k in zip(d.xyxy, d.confidence, d.class_id)
                    if int(k) == 1]
        return detect, versions
    if name == "yolox_x_sportsmot":
        src = ws / "src" / "MixSort"
        sys.path.insert(0, str(src))
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
        from yolox.utils import postprocess
        backbone = YOLOPAFPN(1.33, 1.25, in_channels=[256, 512, 1024])
        head = YOLOXHead(1, 1.25, in_channels=[256, 512, 1024])
        model = YOLOX(backbone, head)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        ck = torch.load(str(w), map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        model.eval().to(device)
        size = (800, 1440)

        def detect(img):
            x, r = _yolox_preproc(img, size)
            with torch.no_grad():
                out = postprocess(model(torch.from_numpy(x).unsqueeze(0).to(device)), 1, FLOOR, 0.7)[0]
            if out is None:
                return []
            out = out.cpu().numpy()
            return [[*(float(v) / r for v in o[:4]), float(o[4] * o[5]), "player"] for o in out]
        return detect, versions
    raise SystemExit(f"{WHO}: unknown model {name}")


def cmd_run(args) -> dict:
    import cv2
    ws = workspace()
    fl = json.loads((ws / "frames.json").read_text())
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock (python3 -m export.heavy --name ... -- ...)")
    t0 = time.time()
    detect, versions = load_runner(args.model, ws, args.device)
    t_load = time.time() - t0
    preds = {}
    t1 = time.time()
    for fr in fl["frames"]:
        img = cv2.imread(fr["image_path"])
        if img is None:
            raise SystemExit(f"{WHO}: cannot read {fr['frame']}")
        preds[fr["frame"]] = [[round(v, 2) for v in p[:4]] + [round(p[4], 5), p[5]] for p in detect(img)]
    t_inf = time.time() - t1
    wpath = detector_weights() if args.model == OURS else ws / "weights" / TEACHERS[args.model]["weights"]
    out = {"model": args.model, "device": args.device, "versions": versions, "weights": wpath.name,
           "weights_sha256": sha256(wpath), "frames_n": len(preds), "seconds_load": round(t_load, 1),
           "seconds_inference": round(t_inf, 1), "seconds_per_frame": round(t_inf / max(1, len(preds)), 3),
           "floor": DET_CONF if args.model == OURS else FLOOR, "preds": preds}
    (ws / "preds").mkdir(exist_ok=True)
    (ws / "preds" / f"{args.model}.json").write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "preds"}, indent=1))
    return out


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def frame_flags(preds, gts):
    """-> (scores, tp flags) in descending score."""
    ps = by_score(preds)
    flags, _ = greedy(ps, gts)
    return [p[4] for p in ps], flags


def detector_counts(det_by_frame, gts_by_frame):
    """Per frame (tp, n_pred, n_gt) of a fixed detector."""
    out = []
    for d, g in zip(det_by_frame, gts_by_frame):
        flags, _ = greedy(by_score(d), g)
        out.append((sum(flags), len(d), len(g)))
    return out


def band(teacher, det, gts, thr):
    """One frame's teacher-only band at threshold thr -> (boxes, flags against the truth our detector did not match)."""
    t = [p for p in by_score(teacher) if p[4] >= thr]
    only = [p for p in t if max((iou(p, d) for d in det), default=0.0) < IOU]
    _, det_hit = greedy(by_score(det), gts)
    flags, _ = greedy(only, gts, skip=det_hit)
    return only, flags


def pair_band(only_a, only_b, det, gts):
    """Boxes two teacher-only bands agree on (greedy at IoU >= 0.5, the mean box, score the smaller of the pair) ->
    (boxes, flags against the truth our detector did not match)."""
    used, agreed = set(), []
    for a in only_a:
        best, bi = 0.0, None
        for i, b in enumerate(only_b):
            if i in used:
                continue
            v = iou(a, b)
            if v > best:
                best, bi = v, i
        if bi is not None and best >= IOU:
            used.add(bi)
            b = only_b[bi]
            agreed.append([(a[k] + b[k]) / 2 for k in range(4)] + [min(a[4], b[4])])
    _, det_hit = greedy(by_score(det), gts)
    flags, _ = greedy(by_score(agreed), gts, skip=det_hit)
    return agreed, flags


def resamples(n_frames: int, clips: list, b: int = B, seed: int = SEED):
    """Frame-bootstrap weights (b x n) and clip-bootstrap weights (b x n), from one seeded generator."""
    import numpy as np
    rng = np.random.default_rng(seed)
    fw = np.stack([np.bincount(rng.integers(0, n_frames, n_frames), minlength=n_frames) for _ in range(b)])
    names = sorted(set(clips))
    idx = {c: [i for i, x in enumerate(clips) if x == c] for c in names}
    cw = np.zeros((b, n_frames))
    for r in range(b):
        for c in rng.choice(names, size=len(names), replace=True):
            cw[r, idx[c]] += 1
    return fw.astype(float), cw


def qualify_one(curve: Curve, det_counts, n_gt_f, weights):
    """Per replicate: our precision and recall, the teacher's matched-precision threshold and recall, and their
    difference -> dict of arrays."""
    import numpy as np
    tp = np.array([c[0] for c in det_counts], float)
    npred = np.array([c[1] for c in det_counts], float)
    out = {"p_det": [], "r_det": [], "r_t": [], "thr": [], "diff": []}
    for w in weights:
        ngt = float(w @ n_gt_f)
        p_d = float(w @ tp) / float(w @ npred) if float(w @ npred) else 0.0
        r_d = float(w @ tp) / ngt if ngt else 0.0
        thr, prec, rec = curve.at(w, ngt)
        r_t, t = recall_at_precision(thr, prec, rec, p_d)
        out["p_det"].append(p_d)
        out["r_det"].append(r_d)
        out["r_t"].append(r_t)
        out["thr"].append(np.nan if t is None else t)
        out["diff"].append(r_t - r_d)
    return {k: np.asarray(v, float) for k, v in out.items()}


def ratio_interval(num, den, weights):
    import numpy as np
    num, den = np.asarray(num, float), np.asarray(den, float)
    vals = [(w @ num) / (w @ den) for w in weights if (w @ den) > 0]
    return percentile_interval(vals) if vals else (None, None)


def cmd_score(args) -> dict:
    import numpy as np
    ws = workspace()
    fl = json.loads((ws / "frames.json").read_text())
    clip_game, games = clip_games()
    for fr in fl["frames"]:                      # the guard again, on what is about to be scored
        why = check_record(fr, clip_game, games)
        if why:
            raise SystemExit(f"{WHO}: {fr['frame']} fails the guard at scoring time: {why}")
    frames = fl["frames"]
    names = [fr["frame"] for fr in frames]
    gts = [fr["boxes"] for fr in frames]
    clips = [fr["clip"] for fr in frames]
    n_gt_f = np.array([len(g) for g in gts], float)
    n_gt = float(n_gt_f.sum())
    ones = np.ones(len(frames))
    fw, cw = resamples(len(frames), clips)

    def load(name):
        p = ws / "preds" / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    ours = load(OURS)
    if ours is None:
        raise SystemExit(f"{WHO}: run our detector first (run --model ours)")
    det_people = [[p for p in ours["preds"][n] if p[5] in DET_PEOPLE] for n in names]
    det_player = [[p for p in ours["preds"][n] if p[5] == "player"] for n in names]
    comparators = {"players": det_player, "people_report_only": det_people}
    result = {"frames": len(frames), "truth_boxes": int(n_gt), "clips": sorted(set(clips)),
              "games": fl["games"], "dropped": fl["dropped"], "iou": IOU, "floor": FLOOR, "bootstrap": B, "seed": SEED,
              "ours": {"device": ours["device"], "versions": ours["versions"], "weights_sha256": ours["weights_sha256"],
                       "conf": DET_CONF, "nms_iou": DET_NMS_IOU, "seconds_inference": ours["seconds_inference"],
                       "seconds_load": ours["seconds_load"]},
              "comparators": {}, "teachers": {}}
    counts = {}
    for cname, det in comparators.items():
        c = detector_counts(det, gts)
        counts[cname] = c
        tp, npred = sum(x[0] for x in c), sum(x[1] for x in c)
        m = prf(tp, npred, int(n_gt))
        m["precision_interval"] = ratio_interval([x[0] for x in c], [x[1] for x in c], fw)
        m["recall_interval"] = ratio_interval([x[0] for x in c], n_gt_f, fw)
        result["comparators"][cname] = m

    bands = {}
    for name, spec in TEACHERS.items():
        tr = load(name)
        if tr is None:
            result["teachers"][name] = {"status": "not run"}
            continue
        tpreds = [tr["preds"][n] for n in names]
        curve = Curve([frame_flags(p, g) for p, g in zip(tpreds, gts)])
        entry = {"what": spec["what"], "license": spec["license"], "device": tr["device"],
                 "versions": tr["versions"], "weights_sha256": tr["weights_sha256"],
                 "seconds_load": tr["seconds_load"], "seconds_inference": tr["seconds_inference"],
                 "seconds_per_frame": tr["seconds_per_frame"]}
        # at the teacher's own default threshold
        thr, prec, rec = curve.at(ones, n_gt)
        dflt = [[p for p in fp if p[4] >= spec["default_conf"]] for fp in tpreds]
        c = detector_counts(dflt, gts)
        entry["at_default_conf"] = {"conf": spec["default_conf"],
                                    **prf(sum(x[0] for x in c), sum(x[1] for x in c), int(n_gt))}
        fl_c = detector_counts(tpreds, gts)
        entry["at_floor"] = {"conf": FLOOR, **prf(sum(x[0] for x in fl_c), sum(x[1] for x in fl_c), int(n_gt))}
        for cname in comparators:
            pt = qualify_one(curve, counts[cname], n_gt_f, ones[None, :])
            fb = qualify_one(curve, counts[cname], n_gt_f, fw)
            cb = qualify_one(curve, counts[cname], n_gt_f, cw)
            diff = float(pt["diff"][0])
            lo, hi = percentile_interval(fb["diff"])
            row = {"detector_precision": float(pt["p_det"][0]), "detector_recall": float(pt["r_det"][0]),
                   "teacher_recall_at_matched_precision": float(pt["r_t"][0]),
                   "threshold": None if np.isnan(pt["thr"][0]) else float(pt["thr"][0]),
                   "teacher_recall_interval": percentile_interval(fb["r_t"]),
                   "diff": diff, "diff_interval_frames": (lo, hi),
                   "diff_interval_clips_report_only": percentile_interval(cb["diff"]),
                   "replicates_without_matched_precision": int(np.isnan(fb["thr"]).sum()),
                   "qualifies": qualifies(diff, lo)}
            if row["threshold"] is not None:
                t = row["threshold"]
                tc = detector_counts([[p for p in fp if p[4] >= t] for fp in tpreds], gts)
                row["at_threshold"] = prf(sum(x[0] for x in tc), sum(x[1] for x in tc), int(n_gt))
            entry[cname] = row
        entry["qualifies"] = entry["players"]["qualifies"]           # the primary comparator decides
        # report-only: the teacher without the boxes that sit on one of our referee boxes
        det_ref = [[p for p in ours["preds"][n] if p[5] == "referee"] for n in names]
        noref = [[p for p in fp if max((iou(p, r) for r in rf), default=0.0) < IOU] for fp, rf in zip(tpreds, det_ref)]
        curve_nr = Curve([frame_flags(p, g) for p, g in zip(noref, gts)])
        pt = qualify_one(curve_nr, counts["players"], n_gt_f, ones[None, :])
        fb = qualify_one(curve_nr, counts["players"], n_gt_f, fw)
        entry["minus_our_referees_report_only"] = {
            "teacher_recall_at_matched_precision": float(pt["r_t"][0]),
            "threshold": None if np.isnan(pt["thr"][0]) else float(pt["thr"][0]),
            "diff": float(pt["diff"][0]), "diff_interval_frames": percentile_interval(fb["diff"])}
        # teacher-only band at the matched-precision threshold of the primary comparator, against our player boxes
        t = entry["players"]["threshold"]
        if t is not None:
            per = [band(tp_, d, g, t) for tp_, d, g in zip(tpreds, det_player, gts)]
            n_b = [len(o) for o, _ in per]
            k_b = [sum(f) for _, f in per]
            tp_det = sum(x[0] for x in counts["players"])
            entry["teacher_only_band"] = {
                "threshold": t, "boxes": int(sum(n_b)), "true": int(sum(k_b)),
                "precision": (sum(k_b) / sum(n_b)) if sum(n_b) else None,
                "precision_interval_frames": ratio_interval(k_b, n_b, fw) if sum(n_b) else (None, None),
                "union_recall": (tp_det + sum(k_b)) / n_gt,
                "missed_by_ours": int(n_gt - tp_det)}
            bands[name] = [o for o, _ in per]
        result["teachers"][name] = entry

    pairs = {}
    names_t = sorted(bands)
    for i, a in enumerate(names_t):
        for b_ in names_t[i + 1:]:
            per = [pair_band(oa, ob, d, g) for oa, ob, d, g in zip(bands[a], bands[b_], det_player, gts)]
            n_b = [len(o) for o, _ in per]
            k_b = [sum(f) for _, f in per]
            pairs[f"{a}+{b_}"] = {
                "boxes": int(sum(n_b)), "true": int(sum(k_b)),
                "precision": (sum(k_b) / sum(n_b)) if sum(n_b) else None,
                "precision_interval_frames": ratio_interval(k_b, n_b, fw) if sum(n_b) else (None, None),
                "both_qualified": bool(result["teachers"][a]["qualifies"] and result["teachers"][b_]["qualifies"])}
    result["two_teacher_band"] = pairs
    result["qualified"] = [n for n, e in result["teachers"].items() if e.get("qualifies")]
    (ws / "results.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_teachers", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frames", help="guard every box-truth file and write the frame list")
    r = sub.add_parser("run", help="one model on the kept frames (teacher venv, inside the heavy-job lock)")
    r.add_argument("--model", required=True, choices=[OURS, *TEACHERS])
    r.add_argument("--device", default="cpu")
    sub.add_parser("score", help="every number of the report, from the stored predictions")
    args = ap.parse_args(argv)
    {"frames": cmd_frames, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
