"""tools/p8_recallcause.py: where the build loses the boxes of D205's players with no box of their own
(docs/research/P8_recallcause.md, version 2.1). Report only: nothing is fitted, tuned or adopted.

    <frozen CV python> -m tools.p8_recallcause run      inside the heavy-job lock ("p8 recallcause"; V3_P8_ROOT =
                                                        the reference root) -> reports/research/p8_recallcause.json,
                                                        raw candidates in the reference root's recallcause/
    python3 -m tools.p8_recallcause selftest            (nothing; the pure parts are covered by the tests)

Steps: (1) D205's rows of gsw_cha_2016 and gsw_bkn_2015 recomputed in order through tools.p8_one2one's functions and
checked equal to its stored rows (class and both sensitivities) per window before its raw pass, with D205's own
checks per window (the inputs hash as D202 recorded them, the scorecard cache complete, the scorer's counts equal to
D202's); D205's records pinned by SHA-256 and run id; (2) the two build-environment gate windows, every frame equal to a3trace's and a3run's stored predict boxes;
(3) per window, a3run's read order, r23's raw prediction tensor through the predictor's own preprocess and inference,
three NMS calls and one at conf 0.25, each on its own copy of the tensor, an alignment check of R_post at f against
the sidecar at f - stride, f and f + stride; (4) the classes of section 5 as amended in versions 2 and 2.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

P8_NO_GAMES = False
WHO = "tools.p8_recallcause"
DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
CHA, BKN = "gsw_cha_2016", "gsw_bkn_2015"
POOL = (CHA, BKN)
UNIVERSE = ("P-i", "D-i")
R23_SHA = "4c886d5f1e603f9f0aa9748f5b4ed0b4dcc406053935ab5ca397c53c8151795d"
VERSIONS = {"opencv": "4.13.0", "torch": "2.2.2", "ultralytics": "8.4.48"}
CONF, IOU, MAX_DET, LOW_CONF = 0.40, 0.45, 300, 0.25
PRE_IOU, PRE_MAX_DET = 1.0, 30000
MISS_FT, NEAR_FT = 2.5, 5.0
ALIGN_FRAMES, ALIGN_BAR, MATCH_IOU = 5, 0.9, 0.5
LATE_S = 1.0
CAP_S = 75 * 60
BOOT_N, BOOT_SEED = 2000, 0
GATE_WINDOWS = {"gsw_cha_2016_s05_w00": ("a3trace", "gsw_cha_2016_s05_w00.json"),
                "gsw_bkn_2015_s04_w00": ("a3run", "gsw_bkn_2015_s04_w00.json")}
CLASSES = ("C1a_own", "C1a_other", "C1b", "C1c", "C2", "C3_late", "C3_not_confirmed", "C3_undetermined", "C4a", "C4b",
           "other")
GROUPS = {"C1": ("C1a_own", "C1a_other", "C1b", "C1c"), "C2": ("C2",),
          "C3": ("C3_late", "C3_not_confirmed", "C3_undetermined"), "C4": ("C4a", "C4b"), "other": ("other",)}
REFEREE_CLASS = 1                          # models/player_detector.pt: {0 player, 1 referee, 2 ball, 3 rim, 4 number}
OTHER_NAMES = {2: "ball", 3: "rim", 4: "number"}
RUN_ID = "20261127T171848Z"
D205_SHA = {CHA: "12de912598adf5dd6902ab63a4fafe4f72914aca280c228f3008128ba5897005",
            BKN: "501f61d5b3dee7554b56eb6fdf4ff53b11f6287cb872ffc7d8c52bf996b27747"}


def out_path() -> Path:
    from tools.p8_common import REPO
    return REPO / "reports" / "research" / "p8_recallcause.json"


def raw_dir(ref: Path) -> Path:
    return ref / "recallcause"


def read_json(p: Path):
    with open(p, "r") as fh:
        return json.load(fh)


def sha256(p: Path) -> str | None:
    if not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_boxes(a: list, b: list, thr: float = MATCH_IOU) -> dict:
    """One-to-one match of boxes a to boxes b by IoU (Hungarian on -IoU) -> {index in a: index in b} at IoU >= thr."""
    if not a or not b:
        return {}
    from scipy.optimize import linear_sum_assignment
    M = np.array([[iou(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(-M)
    return {int(i): int(j) for i, j in zip(ri, ci) if M[i, j] >= thr}


def same_boxes(mine: list, stored: list) -> bool:
    """Equal to a stored predict list ([x1, y1, x2, y2, score], coordinates to 0.01, score to 0.00001): same count and
    a one-to-one match where every rounded value is equal."""
    if len(mine) != len(stored):
        return False
    if not mine:
        return True
    r = [[round(float(v), 2) for v in m[:4]] + [round(float(m[4]), 5)] for m in mine]
    s = [[float(v) for v in x[:5]] for x in stored]
    from scipy.optimize import linear_sum_assignment
    D = np.array([[max(abs(p - q) for p, q in zip(x, y)) for y in s] for x in r])
    ri, ci = linear_sum_assignment(D)
    return bool(D[ri, ci].max() <= 1e-6)


def feet_of(boxes: list, H) -> np.ndarray:
    """Bottom centres of boxes through a homography (the corpus convention of D189)."""
    import cv2
    if not boxes:
        return np.zeros((0, 2))
    px = np.array([[(b[0] + b[2]) / 2.0, b[3]] for b in boxes], np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(px, np.asarray(H, np.float64).reshape(3, 3)).reshape(-1, 2)


def nearest_player(foot, sv) -> int:
    return int(np.argmin(np.hypot(*(np.asarray(sv, float) - np.asarray(foot, float)).T)))


def suppressors(pre: list, post: list, pre_ids: list, post_ids: list) -> dict:
    """{index in pre of a suppressed candidate: index in post of its suppressor} (the highest-score kept box at IoU
    above IOU with it, as NMS suppresses); a candidate with no such box maps to None."""
    kept = set(post_ids)
    order = sorted(range(len(post)), key=lambda k: -post[k][4])
    out = {}
    for i, a in enumerate(pre):
        if pre_ids[i] in kept:
            continue
        out[i] = next((k for k in order if iou(a, post[k]) > IOU), None)        # NMS suppresses at IoU > iou_thres
    return out


def is_c2(j: int, sv, pre_feet, post_feet, supp: dict) -> tuple:
    """Section 8.3: a suppressed candidate within 2.5 ft of j, nearer j than any other player, whose suppressor's foot
    is nearer another player -> (True, candidate index) or (False, None)."""
    for c, k in supp.items():
        if k is None:
            continue
        d = float(np.hypot(*(np.asarray(pre_feet[c]) - sv[j])))
        if d > MISS_FT or nearest_player(pre_feet[c], sv) != j:
            continue
        if nearest_player(post_feet[k], sv) != j:
            return True, c
    return False, None


def covers(box, pt, ext: float = 0.1) -> bool:
    """j's image position inside the box, its bottom extended by a tenth of its height."""
    x1, y1, x2, y2 = box[:4]
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2 + ext * (y2 - y1)


def classify(j, sv, px_j, own_post, pre, pre_feet, ref_feet, supp_c2, side_match, side_foot_ft, late, pre_owner=None):
    """The class of one miss (version 2.1's order). own_post: j's R_post box index or None; supp_c2: is_c2's result;
    pre_owner: per R_pre box, the player its kept box (or its suppressor) is assigned to, or None; ref_feet: referee
    class feet; side_match: the sidecar box index matched to j's R_post box or None; side_foot_ft: that sidecar box's
    stabilised foot in court ft; late: True, False or None (undetermined) for C3."""
    if own_post is None:
        if supp_c2:
            return "C2"
        owners = pre_owner if pre_owner is not None else [None] * len(pre)
        cov = [owners[i] for i, (b, f) in enumerate(zip(pre, pre_feet))
               if covers(b, px_j) and float(np.hypot(*(np.asarray(f) - sv[j]))) > NEAR_FT]
        if any(o is None or o == j for o in cov):
            return "C1a_own"                       # j's own box, its bottom cut short (the foot model)
        if cov:
            return "C1a_other"                     # covered only by another player's detection
        if any(float(np.hypot(*(np.asarray(f) - sv[j]))) <= NEAR_FT for f in ref_feet):
            return "C1b"
        return "C1c"
    if side_match is None:
        return {True: "C3_late", False: "C3_not_confirmed", None: "C3_undetermined"}[late]
    if side_foot_ft is not None and float(np.hypot(*(np.asarray(side_foot_ft) - sv[j]))) <= NEAR_FT:
        return "C4a"
    if side_foot_ft is not None:
        return "C4b"
    return "other"


def align_shares(post_f: list, side_prev: list, side_at: list, side_next: list) -> tuple:
    """R_post at frame f against the sidecar at f - stride, f and f + stride: the share of each sidecar frame's boxes
    that R_post(f) matches one to one at IoU 0.5 (aligned decoding matches best at f)."""
    out = []
    for sb in (side_prev, side_at, side_next):
        out.append(len(match_boxes(sb, [b[:4] for b in post_f])) / len(sb) if sb else 1.0)
    return tuple(out)


def alignment_ok(shares: list) -> bool:
    """shares: [(before, at, after)] per alignment frame -> at f higher than both neighbours and at least the bar."""
    if not shares:
        return False
    a = np.array(shares, float)
    return bool(a[:, 1].mean() > a[:, 0].mean() and a[:, 1].mean() > a[:, 2].mean() and a[:, 1].mean() >= ALIGN_BAR)


def share_table(rows: list, n_boot: int = BOOT_N, seed: int = BOOT_SEED, names: tuple = CLASSES) -> dict:
    """rows: (window, class) of one unit -> per class (or group, with names) count, share and a window-cluster
    bootstrap of the share."""
    from tools.p8_resetfix import resample_picks
    wins = sorted({w for w, _ in rows})
    M = np.zeros((len(wins), len(names)))
    for w, c in rows:
        M[wins.index(w), names.index(c)] += 1
    tot = M.sum()
    picks = resample_picks(len(wins), n_boot, seed) if len(wins) >= 2 else None
    out = {}
    for k, c in enumerate(names):
        n = M[:, k].sum()
        iv = None
        if picks is not None and tot:
            S, T = M[picks, k].sum(1), M[picks].sum((1, 2))
            v = S[T > 0] / T[T > 0]
            iv = [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        out[c] = {"count": int(n), "share": (n / tot) if tot else None, "interval": iv}
    return {"misses": int(tot), "windows": len(wins), "classes": out}


def groups_of(rows: list, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Version 2.1: the groups C1 to C4 and other, each with its count, share and window-bootstrap interval, and the
    subclasses inside."""
    group_of = {c: g for g, cs in GROUPS.items() for c in cs}
    t = share_table(rows, n_boot, seed)
    gt = share_table([(w, group_of[c]) for w, c in rows], n_boot, seed, tuple(GROUPS))
    return {g: {**gt["classes"][g], "subclasses": {c: t["classes"][c] for c in GROUPS[g]}} for g in GROUPS}


def main_loss(groups: dict) -> dict:
    """Version 2.1 (8.7): read on the groups C1 to C4; the largest group is "the main loss" only when its bootstrap
    lower bound is at least 0.5, else "largest, not main"."""
    top = max(("C1", "C2", "C3", "C4"), key=lambda g: (groups[g]["count"], g))
    lo = (groups[top]["interval"] or [None])[0]
    return {"group": top, "share": groups[top]["share"], "lower": lo,
            "reading": "main loss" if lo is not None and lo >= 0.5 else "largest, not main"}


# ── the run (frozen CV python, inside the lock) ────────────────────────────────────────────────────────────────────
def nms_once(raw_t, conf, iou_, classes, max_det):
    """ultralytics' non_max_suppression on a fresh copy of the raw tensor: the function rewrites its input in place
    (a transpose view and an xywh2xyxy assignment), so every call gets its own clone (code check B1)."""
    from ultralytics.utils import nms
    return nms.non_max_suppression(raw_t.clone(), conf, iou_, classes, False, max_det=max_det, return_idxs=True)


class Raw:
    """r23's raw prediction tensor through the predictor's own preprocess and inference, and the NMS calls."""

    def __init__(self, weights: Path):
        import sys
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.pred = None
        self.backend = None
        self._sys = sys

    def setup(self, img):
        import config
        self.model.predict(img, classes=[config.PERSON_CLASS], conf=CONF, iou=IOU, device="cpu", verbose=False)
        self.pred = self.model.predictor
        assert not getattr(self.pred.model, "end2end", False), "end-to-end head: no NMS step"

    def run(self, img) -> dict:
        import config
        from ultralytics.utils import nms, ops
        if self.pred is None:
            self.setup(img)
        im = self.pred.preprocess([img])
        raw = self.pred.inference(im)
        self.backend = "torchvision" if "torchvision" in self._sys.modules else "TorchNMS"

        raw_t = raw[0] if isinstance(raw, (list, tuple)) else raw

        def call(conf, iou_, max_det, classes):
            out, idx = nms_once(raw_t, conf, iou_, classes, max_det)
            d = out[0]
            if len(d):
                d[:, :4] = ops.scale_boxes(im.shape[2:], d[:, :4], img.shape)
            boxes = [[float(v) for v in row[:4]] + [float(row[4]), int(row[5])] for row in d.cpu().numpy()]
            ids = [int(x) for x in idx[0].reshape(-1).cpu().numpy()] if len(idx) else []
            return boxes, ids
        post, post_ids = call(CONF, IOU, MAX_DET, [config.PERSON_CLASS])
        pre, pre_ids = call(CONF, PRE_IOU, PRE_MAX_DET, [config.PERSON_CLASS])
        other, _ = call(CONF, IOU, MAX_DET, None)                  # every class, filtered below
        low, _ = call(LOW_CONF, IOU, MAX_DET, [config.PERSON_CLASS])
        n_cand = int(raw_t.shape[-1])
        return {"post": post, "post_ids": post_ids, "pre": pre, "pre_ids": pre_ids,
                "other": [b for b in other if b[5] != config.PERSON_CLASS], "low": [b for b in low if b[4] < CONF],
                "caps": {"post": len(post) >= MAX_DET, "pre": len(pre) >= PRE_MAX_DET}, "anchors": n_cand}


def read_window(clip: Path, frames: list, stride: int, need: set, raw: Raw) -> dict:
    """tools/p8_a3run.py's read order (the pre-gate pass, then frame 0 and every stride frame to the first processed
    frame, then the processed frames with gaps filled); predict only on the frames in need -> {frame: raw result}."""
    import cv2
    import config
    from videoseq import SEEK_MIN_GAP, SeqReader
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = SeqReader(cap)
    pg_stride, _ = config.pregate_params(fps)
    for i in range(0, total, pg_stride):
        ok, _ = reader.read(i)
        if not ok:
            break
    out, prev = {}, None
    if frames and frames[0] > 0:
        ok, _ = reader.read(0)
        if not ok:
            raise SystemExit(f"{WHO}: frame 0 unreadable")
        for i in range(stride, frames[0], stride):
            ok, img = reader.read(i)
            if not ok:
                raise SystemExit(f"{WHO}: start frame {i} unreadable")
            if i in need:
                out[i] = raw.run(img)
    for f in frames:
        if prev is not None and f - reader.next > SEEK_MIN_GAP:
            for i in range(prev + stride, f, stride):
                ok, img = reader.read(i)
                if not ok:
                    raise SystemExit(f"{WHO}: gap frame {i} unreadable")
                if i in need:
                    out[i] = raw.run(img)
        prev = f
        ok, img = reader.read(f)
        if not ok:
            raise SystemExit(f"{WHO}: frame {f} unreadable")
        if f in need:
            out[f] = raw.run(img)
    cap.release()
    return out


def d205_rows(game: str, P_t, w, comp, ctx, d202_win: dict, d202_hashes: dict) -> list:
    """D205's rows of one window in its order, with each row's frame, player, SportVU pid and frame context (its
    collect loop, with its checks: the inputs hash as D202 recorded them, the cache is complete, the counts equal)."""
    import cv2
    from sportvu import bench as B
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.sync import apply_mirror
    from tools import p8_one2one as O
    from tools import p8_scorecard as S
    from tools.p8_e1_score import window_stats
    from tools.p8_placecause import cache_gaps
    s = w["window"]
    for key, p in (("sidecar", comp / f"{s}_frames.json"), ("timemap", P_t["label_build"] / f"{s}_timemap.json"),
                   ("labels", P_t["labels"] / game / f"{s}.jsonl"), ("identity", comp / f"{s}_identity.json")):
        want = d202_hashes.get(key)
        if (want is None and key != "identity") or (want is not None and sha256(p) != want):
            raise SystemExit(f"{WHO}: {s} {key} does not hash as D202 recorded it")
    why = cache_gaps(P_t, game, w, comp, ctx["mom"], ctx["mirror"])
    if why:
        raise SystemExit(f"{WHO}: {s}: the scorecard cache lacks {why}")
    full = S.score_window(P_t, game, w, comp, "camera", ctx["mom"], ctx["mirror"], ctx["refs"], ctx["events"])
    if "skipped" in full[0]:
        raise SystemExit(f"{WHO}: {s} skipped")
    st = window_stats(full[0])
    if any(st[k] != d202_win.get(k) for k in ("in_frame", "missed", "nr_boxes", "nr_ghosts")):
        raise SystemExit(f"{WHO}: {s}: the scorer's counts differ from D202's record")
    scored = {int(o["f"]) for o in full[1]}
    side = read_json(comp / f"{s}_frames.json")
    srow = {int(x["frame"]): x for x in side["frames"]}
    sidecar_cls = any("cls" in b for x in srow.values() for b in x["boxes"])
    cache = read_json(P_t["cache"] / "scorecard" / f"{s}.json")
    refd = {int(k): v for k, v in cache.get("referees", {}).items()}
    cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    truth, _ = S.truth_rows(P_t, game, w, "camera", ctx["mom"].index, ctx["mirror"])
    out = []
    for row in truth:
        f = int(row["frame"])
        x = srow.get(f)
        if x is None or f not in scored:
            continue
        sv = apply_mirror(ctx["mom"].index.q[row["q"]]["xy"][row["moment"]], ctx["fx"], ctx["fy"])
        pids = [int(v) for v in ctx["mom"].index.q[row["q"]]["pid"][row["moment"]]]
        Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
        px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
        m = FRAME_MARGIN_PX
        inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
        feet = np.array([b["foot_stab"] for b in x["boxes"]], np.float32).reshape(-1, 1, 2)
        ft = cv2.perspectiveTransform(feet, Ht).reshape(-1, 2).astype(np.float64) if len(feet) else np.zeros((0, 2))
        is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                           any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, [])) for b in x["boxes"]],
                          bool)
        svf = np.asarray(sv, np.float64)
        for j, rec in O.frame_classes(ft, svf, inside, is_ref).items():
            out.append({"frame": f, "player": int(j), "pid": pids[int(j)], "pids": pids, "rec": rec, "sv": svf, "Ht": Ht,
                        "px": px, "side_feet": ft, "side_boxes": [b["bbox"] for b in x["boxes"]]})
    return out


def cmd_run(args) -> dict:
    import subprocess
    from sportvu import bench as B
    from sportvu.sync import MIRRORS
    from tools import p8_frozen as F
    from tools import p8_pin
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_common import REPO, atomic_write, guard, main_checkout, p8_root
    from tools import p8_one2one as O
    from tools.p8_placement import rnd
    from tools.research_testset import refuse_cv_test
    from export.schemas import find_private
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if p8_root().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    st = subprocess.run(["git", "status", "--porcelain", "--", "tools", "sportvu"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.strip()
    if st:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    pinned, now = read_json(p8_pin.PIN_FILE), p8_pin.current()
    if [k for k in now if now[k] != pinned.get(k)]:
        raise SystemExit(f"{WHO}: the scorecard does not match its pin")
    weights = (main_checkout() / "models" / "player_detector.pt").resolve()
    if sha256(weights) != R23_SHA:
        raise SystemExit(f"{WHO}: the detector is not r23")
    one2one = {g: r["ref"] / "one2one" / f"{g}.json" for g in POOL}
    pins = {g: sha256(p) for g, p in one2one.items()}
    if pins != D205_SHA:
        raise SystemExit(f"{WHO}: D205's records are not the pinned ones")
    d205 = {g: read_json(p) for g, p in one2one.items()}
    if any(d205[g].get("run_id") != RUN_ID for g in POOL):
        raise SystemExit(f"{WHO}: D205's records are not from run {RUN_ID}")
    d202 = {g: read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in POOL}
    if any(d202[g].get("run_id") != RUN_ID for g in POOL):
        raise SystemExit(f"{WHO}: D202's records are not from run {RUN_ID}")
    watch = lambda: {k: RB.manifest(r[k], ("labels", "label_build", "truth_accept", "cache")) for k in ("ref", "ms")} | \
        {"builds_r23": RB.manifest(r["ref"], ("builds/r23",))}
    before = watch()
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    comp = P["builds"] / "r23"
    raw = Raw(weights)
    import cv2
    import torch
    import ultralytics
    versions = {"opencv": cv2.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    if versions != VERSIONS:
        raise SystemExit(f"{WHO}: versions {versions} are not {VERSIONS}")
    # (2) the build-environment gate windows
    gate = {}
    for s, (kind, name) in GATE_WINDOWS.items():
        rec = read_json(r["ws"] / "placement" / kind / name)
        stored = {int(x["frame"]): x["pred"] for x in rec["per_frame"]}
        side = read_json(comp / f"{s}_frames.json")          # the composite's processed frames (equal to the old build's)
        frames = sorted(int(x["frame"]) for x in side["frames"])
        res = read_window(P["label_build"] / f"{s}.mp4", frames, int(side["stride"]), set(stored), raw)
        bad = [f for f in sorted(stored) if not same_boxes([b[:5] for b in res.get(f, {}).get("post", [])], stored[f])]
        gate[s] = {"frames": len(stored), "differing": len(bad), "boxes": sum(len(v["post"]) for v in res.values())}
        if bad:
            raise SystemExit(f"{WHO}: the build-environment gate fails on {s} ({len(bad)} frames, first {bad[:3]})")
    if gate["gsw_cha_2016_s05_w00"]["boxes"] != 8766:
        raise SystemExit(f"{WHO}: the cha gate window holds {gate['gsw_cha_2016_s05_w00']['boxes']} boxes, not 8,766")
    # (1) and (3) per game and window
    misses, dropped, align, caps, d205_gate = [], {}, {}, {"post": 0, "pre": 0}, {}
    raw_d = raw_dir(r["ref"])
    raw_d.mkdir(exist_ok=True)
    for g in POOL:
        gd = F.game_of(g)
        tags = [g, gd["sportvu"], gd["video_id"]]
        refuse_cv_test(tags, WHO)
        guard(tags, WHO)
        mirror = F.mirror_of(P, g)
        mom = S.Moments(read_json(P["sportvu"] / f"{gd['sportvu']}_moments.json")["moments"])
        ctx = {"mom": mom, "mirror": mirror, "refs": refs, "events": S.pbp_events(gd["sportvu"]),
               "fx": MIRRORS[mirror][0], "fy": MIRRORS[mirror][1]}
        P_t = {**P, "labels": RB.truths(r)[g]["current"] / "labels"}
        wins = sorted((w for w in F.label_windows(P, g) if (comp / f"{w['window']}_frames.json").exists()),
                      key=lambda w: w["window"])
        guard([w["window"] for w in wins], WHO)
        stored_rows = d205[g]["rows"]
        if sorted(d202[g]["windows"]) != [w["window"] for w in wins]:
            raise SystemExit(f"{WHO}: {g}: the windows differ from D202's record")
        mine_all = []
        for wi, w in enumerate(wins):
            if time.time() - t0 > CAP_S:
                raise SystemExit(f"{WHO}: the 75-minute cap was reached; nothing is reported")
            s = w["window"]
            rows = d205_rows(g, P_t, w, comp, ctx, d202[g]["windows"][s], (d202[g].get("hashes") or {}).get(s, {}))
            mine = [[wi, {k: x["rec"][k] for k in ("class", "single", "inframe")}] for x in rows]
            want_w = [[a, {k: b[k] for k in ("class", "single", "inframe")}] for a, b in stored_rows if a == wi]
            if mine != want_w:                     # code check M6: stop at the first window that differs
                raise SystemExit(f"{WHO}: the D205 row gate fails on {s}; nothing is reported")
            mine_all += mine
            uni = [x for x in rows if x["rec"]["class"] in UNIVERSE]
            if not uni:
                continue
            side = read_json(comp / f"{s}_frames.json")
            stride = int(side["stride"])
            frames = sorted(int(x["frame"]) for x in side["frames"])
            fset = set(frames)
            uframes = sorted({x["frame"] for x in uni})
            al = [f for f in uframes if f - stride in fset and f + stride in fset][:ALIGN_FRAMES]
            res = read_window(P["label_build"] / f"{s}.mp4", frames, stride, set(uframes), raw)
            srow = {int(x["frame"]): x for x in side["frames"]}
            shares = [align_shares(res[f]["post"], [b["bbox"] for b in srow[f - stride]["boxes"]],
                                   [b["bbox"] for b in srow[f]["boxes"]], [b["bbox"] for b in srow[f + stride]["boxes"]])
                      for f in al]
            ok = alignment_ok(shares)
            align[s] = {"frames": len(al), "shares": shares, "aligned": ok}
            caps["post"] += sum(v["caps"]["post"] for v in res.values())
            caps["pre"] += sum(v["caps"]["pre"] for v in res.values())
            atomic_write(raw_d / f"{s}.json", json.dumps({str(f): {k: v[k] for k in ("post", "pre", "other", "low")}
                                                         for f, v in res.items()}))
            if not ok:
                dropped[s] = len(uni)
                continue
            for x in uni:
                f, j, sv, Ht = x["frame"], x["player"], x["sv"], x["Ht"]
                rr = res[f]
                post, pre = rr["post"], rr["pre"]
                post_feet, pre_feet = feet_of(post, Ht), feet_of(pre, Ht)
                ref_boxes = [b for b in rr["other"] if b[5] == REFEREE_CLASS]
                ref_feet = feet_of(ref_boxes, Ht)
                near_other = {OTHER_NAMES[c]: sum(1 for b, fo in zip(rr["other"], feet_of(rr["other"], Ht))
                                                  if b[5] == c and float(np.hypot(*(fo - sv[j]))) <= NEAR_FT)
                              for c in OTHER_NAMES}
                main, claimed = O.two_stage(post_feet, sv, range(len(sv)), range(len(post)), O._assign_all)
                own = main.get(j, (None,))[0]
                supp = suppressors(pre, post, rr["pre_ids"], rr["post_ids"])
                pos = {a: k for k, a in enumerate(rr["post_ids"])}
                pre_owner = []
                for i, a in enumerate(rr["pre_ids"]):
                    k = pos.get(a, supp.get(i))
                    pre_owner.append(claimed.get(k) if k is not None else None)
                c2, cidx = is_c2(j, sv, pre_feet, post_feet, supp) if own is None else (False, None)
                side_m, side_ft, late = None, None, None
                if own is not None:
                    mm = match_boxes([post[own][:4]], x["side_boxes"])
                    side_m = mm.get(0)
                    if side_m is not None:
                        side_ft = x["side_feet"][side_m]
                    else:
                        later = [y for y in rows if y["pid"] == x["pid"] and f < y["frame"] <= f + LATE_S * float(side["fps"])]
                        if later:                  # matched on the SportVU pid (code check M3)
                            late = any(len(y["side_feet"]) and float(np.hypot(*(y["side_feet"] - y["sv"][y["player"]]).T)
                                                                     .min()) <= MISS_FT for y in later)
                cls = classify(j, sv, x["px"][j], own, pre, pre_feet, ref_feet, c2, side_m, side_ft, late, pre_owner)
                lowshare = cls == "C1c" and any(
                    float(np.hypot(*(np.asarray(fl) - sv[j]))) <= MISS_FT and nearest_player(fl, sv) == j
                    for fl in feet_of(rr["low"], Ht))
                misses.append({"game": g, "window": s, "frame": f, "player": j, "pid": x["pid"],
                               "d205": x["rec"]["class"], "class": cls, "low_conf_box": bool(lowshare),
                               "other_classes_within_5ft": near_other,
                               "c2_score": (pre[cidx][4] if c2 else None),
                               "c2_iou": (iou(pre[cidx], post[supp[cidx]]) if c2 else None)})
        want = [[a, {k: b[k] for k in ("class", "single", "inframe")}] for a, b in stored_rows]
        if mine_all != want:
            raise SystemExit(f"{WHO}: the D205 row gate fails on {g}; nothing is reported")
        d205_gate[g] = {"rows": len(mine_all), "sha256": pins[g], "run_id": d205[g].get("run_id")}
    if watch() != before:
        raise SystemExit(f"{WHO}: a watched folder changed; nothing is reported")
    units = {}
    for name, sel in (("P-i", lambda m: m["d205"] == "P-i"), ("D-i", lambda m: m["d205"] == "D-i"),
                      ("both", lambda m: True)):
        for scope in POOL + ("pooled",):
            mm = [m for m in misses if sel(m) and (scope == "pooled" or m["game"] == scope)]
            t = share_table([(m["window"], m["class"]) for m in mm])
            t["groups"] = groups_of([(m["window"], m["class"]) for m in mm])
            t["other_is_a_code_fault"] = t["classes"]["other"]["count"] > 0     # expected 0 (version 2.1)
            t["other_classes_within_5ft"] = {n: sum(m["other_classes_within_5ft"][n] for m in mm) for n in
                                             OTHER_NAMES.values()}
            t["low_conf_share_of_C1c"] = (sum(m["low_conf_box"] for m in mm if m["class"] == "C1c") /
                                          max(1, sum(m["class"] == "C1c" for m in mm)))
            if scope == "pooled":
                t["reading"] = main_loss(t["groups"])
            units[f"{name}__{scope}"] = t
    c2 = [m for m in misses if m["class"] == "C2"]
    out = {"what": "Where the build loses the boxes of D205's no-own-box misses (docs/research/P8_recallcause.md); "
                   "report only", "head": head, "versions": versions, "nms_backend": raw.backend,
           "d205_gate": d205_gate, "build_environment_gate": gate, "alignment": align, "windows_dropped": dropped,
           "cap_hits": caps, "units": units,
           "c2": {"count": len(c2), "score_median": float(np.median([m["c2_score"] for m in c2])) if c2 else None,
                  "iou_median": float(np.median([m["c2_iou"] for m in c2])) if c2 else None},
           "seconds": round(time.time() - t0, 1),
           "note": "each rate is an upper bound on how far a lever could lower missed; the ghost cost is not measured"}
    text = json.dumps(rnd(out), indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(out_path(), text)
    print(json.dumps({k: out["units"][f"{k}__pooled"]["reading"] for k in ("P-i", "D-i", "both")}, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_recallcause")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    a = ap.parse_args(argv)
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
