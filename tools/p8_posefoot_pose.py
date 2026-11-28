"""tools/p8_posefoot_pose.py: the pose stages of docs/research/P8_posefoot.md (versions 1.1 to 1.3 and section 15).
Report only; nothing is fitted or tuned.

    <frozen CV python> -m tools.p8_posefoot_pose pose GAME [--smoke]   inside the lock ("p8 posefoot pose ..."),
                                                                         V3_P8_ROOT = work_ref, HF_HUB_OFFLINE=1,
                                                                         PYTORCH_ENABLE_MPS_FALLBACK=1
    python3 -m tools.p8_posefoot_pose report [--smoke]                 -> reports/research/p8_posefoot_pose[_smoke].json

Refuses to start unless reports/research/p8_posefoot_align.json exists with stop false, and reads only its aligned
windows, through tools.p8_posefoot.build_read (the build's own read path). On each frame holding a selected box (every
P-ii box; up to 2 hit boxes per scored frame, numpy default_rng(0) over the game's windows and frames in order), the
pinned ViTPose-B (tools.p8_posefoot MODEL_REPO at MODEL_REV, CPU, offline) runs on the selected boxes only. The frame
goes BGR to RGB first. Per selected box: the ankle point (both ankles at score 0.3 or more: their midpoint; one: that
ankle); its lift pixel (15.1 (1): the floor point beneath the ankle lifted ANKLE_FT, through the label row's camera for
T and pose_from_h(B) for B, projected back through the same camera), then mapped through T or B; the pose-raw foot (the
lift pixel), the hybrid foot_stab + (lift pixel - foot_raw), the same-frame null from the nearest-height sampled hit box
with both ankles (15.1 (6)), the 0.35 ft and unlifted sensitivities, scorer-ghost flags.
hou_cle_2016 is refused before any file is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard
from tools.research_testset import refuse_cv_test
from tools import p8_posefoot as PF

WHO = "tools.p8_posefoot_pose"
ANKLES = (15, 16)                         # COCO left and right ankle
ANKLE_SCORE = 0.3
ANKLE_FT, ANKLE_FT_ALT = 0.25, 0.35       # 11.7 (4): 0.25 ruled, 0.35 and 0 reported
HITS_PER_FRAME = 2                        # 11.7 (6)
MECH_SHARE, MECH_BELOW = 0.95, 0.2        # stage 1's mechanical check
SMOKE_WINDOWS = {PF.CHA: "gsw_cha_2016_s00_w00", PF.BKN: "gsw_bkn_2015_s00_w00"}
POSE_CAP_S = 60 * 60                      # 15.1 (3): per game
READ_FAIL_MAX = 3                         # 15.1 (5): more than 3 unreproduced reads in a game stop the run
RECORD = REPO / "reports" / "research" / "p8_posefoot_pose.json"


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """The decoder's BGR frame as RGB (11.7 (5))."""
    return np.ascontiguousarray(img[:, :, ::-1])


def to_coco(bbox) -> list:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [x1, y1, x2 - x1, y2 - y1]


def ankle_point(kps, scores, thr: float = ANKLE_SCORE) -> tuple:
    """(pixel or None, number of confident ankles)."""
    ok = [i for i in ANKLES if float(scores[i]) >= thr]
    if not ok:
        return None, 0
    pts = np.asarray([kps[i] for i in ok], float)
    return [float(v) for v in pts.mean(0)], len(ok)


def lift_ground(px, C, rvec, f: float, W: int, H: int, h_ft: float) -> np.ndarray | None:
    """The floor point beneath the point h_ft above the floor on the pixel's ray (sportvu.camera's model: world
    (x, y, -height), cam = R (X - C), uv = K cam). h_ft 0 is the ground-plane mapping."""
    import cv2
    from sportvu.camera import K_of
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    d = R.T @ (np.linalg.inv(K_of(f, W, H)) @ np.array([float(px[0]), float(px[1]), 1.0]))
    C = np.asarray(C, np.float64).reshape(3)
    if abs(d[2]) < 1e-12:
        return None
    s = (-h_ft - C[2]) / d[2]
    if s <= 0:
        return None
    X = C + s * d
    return np.array([X[0], X[1]])


def lift_pixel(px, cam: tuple, W: int, H: int, h_ft: float) -> list | None:
    """15.1 (1): the camera gives the lift shift only. The floor point beneath the point h_ft up the pixel's ray is
    projected back through the same camera to a pixel; that pixel is then mapped through T or B like any foot. At
    h_ft 0 it is the pixel itself."""
    from tools.p8_scorecard import project3d
    C, rvec, f = cam
    g = lift_ground(px, C, rvec, f, W, H, h_ft)
    if g is None:
        return None
    uv, depth = project3d(C, rvec, f, W, H, [[float(g[0]), float(g[1]), 0.0]])
    if not (depth > 0).all():
        return None
    return [float(uv[0][0]), float(uv[0][1])]


def to_pixel(H_px2ft, q) -> np.ndarray:
    """A court point back to the image through H's inverse."""
    return PF.map_px(np.linalg.inv(np.asarray(H_px2ft, np.float64).reshape(3, 3)), [q])[0]


def hybrid_pixel(stab, raw, ground_px) -> list:
    """foot_stab plus the pose ground point's pixel offset from foot_raw (11.4)."""
    return [float(stab[0] + ground_px[0] - raw[0]), float(stab[1] + ground_px[1] - raw[1])]


def inside_or_below(px, bbox, below: float = MECH_BELOW) -> bool:
    """Stage 1's mechanical check: the ankle lies inside the box or just below it (within 0.2 box heights)."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    h = y2 - y1
    return bool(x1 <= px[0] <= x2 and y1 <= px[1] <= y2 + below * h)


def partner_order(heights: list, b: int, ok: list) -> list:
    """Non-referee boxes other than b (ok), by nearness of height to b's, ties by index."""
    return sorted((k for k in range(len(heights)) if k != b and ok[k]), key=lambda k: (abs(heights[k] - heights[b]), k))


def sample_hits(n_by_frame: list, rng) -> list:
    """For each scored frame in order, the indices of up to HITS_PER_FRAME of its hit boxes (without replacement)."""
    out = []
    for n in n_by_frame:
        out.append(sorted(int(i) for i in rng.choice(n, min(n, HITS_PER_FRAME), replace=False)) if n else [])
    return out


def fill_no_change(row: dict) -> dict:
    """15.1 (2): for est, a row with no hybrid foot reads as no change (its hybrid equals foot_stab)."""
    out = {**row, "dT": dict(row["dT"]), "gT": dict(row["gT"])}
    if out["dT"].get("hyb") is None:
        out["dT"]["hyb"] = out["dT"]["stab"]
    if out["gT"].get("hyb") is None:
        out["gT"]["hyb"] = out["gT"]["stab"]
    return out


def require_aligned(record_path: Path = PF.RECORD_ALIGN) -> dict:
    """The pose stages start only after stage A: its record exists and reads stop false. -> {game: aligned windows}."""
    if not Path(record_path).exists():
        raise SystemExit(f"{WHO}: stage A's record {Path(record_path).name} is missing; the pose stages do not start")
    rec = json.loads(Path(record_path).read_text())
    if rec.get("stop") is not False or rec.get("smoke"):
        raise SystemExit(f"{WHO}: stage A's record does not read stop false on the full pass")
    out = {}
    for g in PF.POOL:
        na = set(rec["per_game"][g]["not_aligned"])
        out[g] = sorted(w for w in rec["per_game"][g]["M_and_L0"] if w not in na)
    return out


# ── data (frozen CV python, under the lock) ────────────────────────────────────────────────────────────────────────
def load_pose():
    """The pinned model, offline, on CPU."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit(f"{WHO}: HF_HUB_OFFLINE=1 is required after the one download")
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    proc = AutoProcessor.from_pretrained(PF.MODEL_REPO, revision=PF.MODEL_REV)
    model = VitPoseForPoseEstimation.from_pretrained(PF.MODEL_REPO, revision=PF.MODEL_REV).to("cpu").eval()
    return proc, model


def run_pose(proc, model, img_bgr, bboxes: list) -> list:
    """Keypoints and scores for each box of one frame: [(kps (17, 2), scores (17,))]."""
    import torch
    if not bboxes:
        return []
    from PIL import Image
    rgb = Image.fromarray(bgr_to_rgb(img_bgr))
    boxes = [[to_coco(b) for b in bboxes]]
    inputs = proc(rgb, boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_pose_estimation(out, boxes=boxes)[0]
    return [(r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()) for r in res]


def collect_pose(game: str, smoke: bool = False) -> dict:
    PF.refuse_game(game)
    aligned = require_aligned()[game]
    bad_env = {k: os.environ.get(k) for k, v in PF.ALIGN_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        raise SystemExit(f"{WHO}: set {PF.ALIGN_ENV} before the run (got {bad_env})")
    import cv2
    import config
    from sportvu import bench as B
    from sportvu.camera import pose_from_h
    from sportvu.eval import FRAME_MARGIN_PX
    from sportvu.rebuild import build_stride
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools import p8_one2one as O
    from tools import p8_piibox as PB
    from tools import p8_rebaseline as RB
    from tools import p8_scorecard as S
    from tools.p8_common import main_checkout
    from gate.backbones import get_backbone
    from gate.trained_head import TrainedHeadGate
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    r = RB.roots()
    if Path(os.environ.get("V3_P8_ROOT", "")).expanduser().resolve() != r["ref"].resolve():
        raise SystemExit(f"{WHO}: V3_P8_ROOT must be the reference root")
    head = PB.git_head()
    if not head["tools_and_sportvu_clean"]:
        raise SystemExit(f"{WHO}: tools/ or sportvu/ has uncommitted changes")
    g = F.game_of(game)
    tags = [game, g["sportvu"], g["video_id"]]
    refuse_cv_test(tags, WHO)
    guard(tags, WHO)
    model_rec = RB.read_json(PF.RECORD_MODEL)
    if model_rec.get("revision") != PF.MODEL_REV:
        raise SystemExit(f"{WHO}: the model record is not the pinned revision")
    score = RB.read_json(r["ref"] / "rebaseline" / f"score__{game}__current.json")
    d207 = RB.read_json(r["ref"] / "piibox" / PF.D207_RUN_FILES[game])
    want = {}
    for x in d207["rows"]:
        want.setdefault(x["wi"], []).append((x["kind"], x["class"], x["h"]))
    root = RB.truths(r)[game]["current"]
    watched = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    t0 = time.time()
    P = F.install_paths()
    S.load_fixed_constants()
    refs = B.References()
    mirror = F.mirror_of(P, game)
    fx, fy = MIRRORS[mirror]
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    comp = P["builds"] / "r23"
    P_t = {**P, "labels": root / "labels"}
    wins = sorted((w for w in F.label_windows(P, game) if (comp / f"{w['window']}_frames.json").exists()),
                  key=lambda w: w["window"])
    stems = [w["window"] for w in wins]
    refuse_cv_test(stems, WHO)
    guard(stems, WHO)
    if sorted(score["windows"]) != stems or d207.get("windows") != len(stems):
        raise SystemExit(f"{WHO}: {game}: the windows differ from D202's or D207's records")
    proc, model = load_pose()
    mc = main_checkout() / "models"
    thr = json.loads((mc / config.THRESHOLDS_PATH.name).read_text())["v2"]
    gates = {}

    def gate_for(dev: str):
        if dev not in gates:
            gates[dev] = TrainedHeadGate.load(mc / config.HEAD_V2_PATH.name, backbone=get_backbone("clip", dev),
                                              threshold=thr)
        return gates[dev]
    rng = np.random.default_rng(0)
    rows, rates, sel_seconds = [], [], 0.0
    counters = {"no_partner_pose": 0, "no_B_pose": 0, "frames": 0, "crops": 0, "crops_all_aligned": 0,
                "read_not_reproduced": 0, "read_not_reproduced_windows": []}
    for wi, w in enumerate(wins):
        s = w["window"]
        if time.time() - t0 > POSE_CAP_S:                 # 15.1 (3): 60 minutes per game, checked before each window
            raise SystemExit(f"{WHO}: the 60-minute cap was reached; nothing is reported")
        tsel = time.time()
        full = S.score_window(P_t, game, w, comp, "camera", mom, mirror, refs, events)
        if "skipped" in full[0]:
            raise SystemExit(f"{WHO}: {s} skipped: {full[0]['skipped']}")
        scored = {int(o["f"]) for o in full[1]}
        side = json.loads((comp / f"{s}_frames.json").read_text())
        if RB.sha256(comp / f"{s}_frames.json") != score["hashes"][s]["sidecar"]:
            raise SystemExit(f"{WHO}: {s} sidecar differs from D202's record")
        srow = {int(x["frame"]): x for x in side["frames"]}
        sidecar_cls = any("cls" in b for x in srow.values() for b in x["boxes"])
        ident = json.loads((comp / f"{s}_identity.json").read_text()) if (comp / f"{s}_identity.json").exists() else {}
        idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
        cache = json.loads((P_t["cache"] / "scorecard" / f"{s}.json").read_text())
        refd = {int(k): v for k, v in cache.get("referees", {}).items()}
        tm = json.loads((P_t["label_build"] / f"{s}_timemap.json").read_text())
        fps = float(tm["fps"])
        cap = cv2.VideoCapture(str(P_t["label_build"] / f"{s}.mp4")); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
        truth, _ = S.truth_rows(P_t, game, w, "camera", mom.index, mirror)
        frames, sv_of, items, ctx = [], {}, [], {}
        for row in truth:
            f = int(row["frame"])
            x = srow.get(f)
            if x is None or f not in scored:
                continue
            d = mom.index.q[row["q"]]
            sv = np.asarray(apply_mirror(d["xy"][row["moment"]], fx, fy), np.float64)
            pids = [int(v) for v in d["pid"][row["moment"]]]
            Ht = np.asarray(row["H_truth"], np.float64).reshape(3, 3)
            px = cv2.perspectiveTransform(np.asarray(sv, np.float32).reshape(-1, 1, 2), np.linalg.inv(Ht)).reshape(-1, 2)
            m = FRAME_MARGIN_PX
            inside = (px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)
            ft = PF.map_px(Ht, [b["foot_stab"] for b in x["boxes"]]) if x["boxes"] else np.zeros((0, 2))
            is_ref = np.array([(b.get("cls") in ("referee", B.REFEREE_CLASS)) if sidecar_cls else
                               any(B._iou(b["bbox"], dd[:4]) >= B.REF_IOU for dd in refd.get(f, []))
                               for b in x["boxes"]], bool)
            cls = O.frame_classes(ft, sv, inside, is_ref)
            nonref = [int(b) for b in np.where(~is_ref)[0]]
            main, _ = O.two_stage(ft, sv, range(len(sv)), nonref, O._assign_all)
            cids = [idmap.get(int(b["tid"]), int(b["tid"])) for b in x["boxes"]]
            frames.append({"f": f, "cid": cids, "A": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "A"},
                           "B": {bk: pids[j] for j, (bk, dd, stg) in main.items() if stg == "B"},
                           "ref": list(is_ref), "ft": ft})
            sv_of[f] = {pids[j]: sv[j] for j in range(len(sv))}
            picked, _ = PB.select_items(cls, main)
            Hb = np.asarray(x["H"], np.float64).reshape(3, 3) if B.accepted(x) and x.get("H") is not None else None
            poseB = pose_from_h(Hb, W, H) if Hb is not None else None
            ctx[f] = {"x": x, "Ht": Ht, "Hb": Hb, "poseB": poseB, "sv": sv, "is_ref": is_ref, "row": row}
            for j, kind, bk in picked:
                items.append({"wi": wi, "kind": kind, "f": f, "pid": pids[j], "cid": cids[bk], "j": j, "bk": bk,
                              "h": float(x["boxes"][bk]["bbox"][3] - x["boxes"][bk]["bbox"][1])})
        tracks = PB.track_table(frames)
        got = []
        for it in items:
            ev = PB.evidence(tracks[it["cid"]], it["pid"], it["f"], sv_of, fps)
            it["class"] = PB.classify(ev["n_own"], ev["n_other"], ev["n_ownB"], ev["referee_flagged"], ev["present"])
            got.append((it["kind"], it["class"], it["h"]))
        if got != want.get(wi, []):
            raise SystemExit(f"{WHO}: D207 gate failed on {s}")
        # the hit sample draws for every window, aligned or not, so the sample does not depend on alignment
        fr_sorted = sorted({it["f"] for it in items})
        hits_by_f = {f: [k for k, it in enumerate(items) if it["f"] == f and it["kind"] == "hit"] for f in fr_sorted}
        draws = sample_hits([len(hits_by_f[f]) for f in fr_sorted], rng)
        keep_hits = {hits_by_f[f][i] for f, ii in zip(fr_sorted, draws) for i in ii}
        sel = [it for k, it in enumerate(items) if it["kind"] == "P-ii" or k in keep_hits]
        sel_seconds += time.time() - tsel
        if s not in aligned:
            continue
        counters["crops_all_aligned"] += len(sel)            # one crop per selected box (section 15.1 (4))
        if smoke and s != SMOKE_WINDOWS[game]:
            continue
        need = {}
        for it in sel:
            need.setdefault(it["f"], set()).add(it["bk"])
        max_frames = (w["f_end"] - w["f_start"]) // build_stride(w["fps"]) + 2
        dev_rec = RB.read_json(comp / f"{s}_device.json") if (comp / f"{s}_device.json").exists() else {}
        gdev = (dev_rec.get("resolved") or {}).get("gate") or "cpu"
        tw = time.time()
        frames_pose, pose_s = {}, [0.0]
        order = sorted(need)                              # main_reads calls predict once per frame, in frame order
        it_order = iter(order)

        def predict_seq(img):
            f = next(it_order)
            ks = sorted(need[f])
            tp = time.time()
            res = run_pose(proc, model, img, [ctx[f]["x"]["boxes"][k]["bbox"] for k in ks])
            pose_s[0] += time.time() - tp
            frames_pose[f] = dict(zip(ks, res))
            return True

        br = PF.build_read(P["label_build"] / f"{s}.mp4", side, max_frames, gate_for(gdev).is_court_visible,
                           set(order), predict_seq)
        if not br["validation"]["passes"] or set(order) - set(frames_pose):
            counters["read_not_reproduced"] += 1          # 15.1 (5): skipped and counted
            counters["read_not_reproduced_windows"].append(s)
            if counters["read_not_reproduced"] > READ_FAIL_MAX:
                raise SystemExit(f"{WHO}: more than {READ_FAIL_MAX} windows of {game} did not reproduce the read")
            continue
        rates.append({"window": s, "seconds": round(time.time() - tw, 1), "pose_seconds": round(pose_s[0], 2),
                      "frames": len(order), "crops": sum(len(v) for v in frames_pose.values())})
        counters["frames"] += len(order)
        counters["crops"] += rates[-1]["crops"]
        anks = {}
        for f_, kd in frames_pose.items():
            for k_, (kk, ss) in kd.items():
                anks[(f_, k_)] = ankle_point(kk, ss)
        hits_on = {}
        for it in sel:
            if it["kind"] == "hit":
                hits_on.setdefault(it["f"], []).append(it["bk"])
        for it in sel:
            c = ctx[it["f"]]
            x, Ht, Hb, poseB, sv, row = c["x"], c["Ht"], c["Hb"], c["poseB"], c["sv"], c["row"]
            bx = x["boxes"][it["bk"]]
            ank, n_ank = anks[(it["f"], it["bk"])]
            camT = (row["camera_C"], row["camera_rvec"], float(row["camera_f"]))
            camB = (poseB[2], poseB[0], poseB[1]) if poseB is not None else None
            rec = {"wi": wi, "kind": it["kind"], "class": it["class"], "h": it["h"], "speed": None, "ankles": n_ank,
                   "mech": None if ank is None else inside_or_below(ank, bx["bbox"]),
                   "dT": {k: None for k in ("poseraw", "hyb", "poseraw_035", "hyb_035", "poseraw_0", "hyb_0", "null")},
                   "dB": {k: None for k in ("poseraw", "hyb", "null")}, "gT": {"hyb": None}}
            j = it["j"]

            def dist(Hm, p):
                return float(np.hypot(*(PF.map_px(Hm, [p])[0] - sv[j])))

            def ghost(Hm, p):
                q = PF.map_px(Hm, [p])[0]
                return bool(np.sqrt(((sv - q) ** 2).sum(1)).min() > PF.GHOST_FT)

            for name, p in (("stab", bx["foot_stab"]), ("raw", bx["foot_raw"])):
                rec["dT"][name], rec["gT"][name] = dist(Ht, p), ghost(Ht, p)
                rec["dB"][name] = dist(Hb, p) if Hb is not None else None
            if ank is not None:
                for hname, hft in (("", ANKLE_FT), ("_035", ANKLE_FT_ALT), ("_0", 0.0)):
                    pl = lift_pixel(ank, camT, W, H, hft)          # 15.1 (1): the camera gives the lift only
                    if pl is None:
                        continue
                    hyb = hybrid_pixel(bx["foot_stab"], bx["foot_raw"], pl)
                    rec["dT"]["poseraw" + hname] = dist(Ht, pl)
                    rec["dT"]["hyb" + hname] = dist(Ht, hyb)
                    if hname == "":
                        rec["gT"]["hyb"] = ghost(Ht, hyb)
                if camB is not None:
                    plB = lift_pixel(ank, camB, W, H, ANKLE_FT)
                    if plB is not None:
                        rec["dB"]["poseraw"] = dist(Hb, plB)
                        rec["dB"]["hyb"] = dist(Hb, hybrid_pixel(bx["foot_stab"], bx["foot_raw"], plB))
            counters["no_B_pose"] += camB is None
            # 15.1 (6): the null partner is the nearest-height sampled hit box on the frame with both ankles
            hts = [float(b["bbox"][3] - b["bbox"][1]) for b in x["boxes"]]
            okp = [k in hits_on.get(it["f"], []) and anks.get((it["f"], k), (None, 0))[1] == 2 and
                   PF.box_iou(x["boxes"][k]["bbox"], bx["bbox"]) < PF.PARTNER_IOU for k in range(len(x["boxes"]))]
            order_k = partner_order(hts, it["bk"], okp)
            if order_k:
                k = order_k[0]
                bk_, a_k = x["boxes"][k], anks[(it["f"], k)][0]
                h_k = hts[k]
                pk = lift_pixel(a_k, camT, W, H, ANKLE_FT)
                if pk is not None:
                    off = np.asarray(pk) - np.asarray(bk_["foot_raw"], float)
                    rec["dT"]["null"] = dist(Ht, PF.null_foot(bx["foot_stab"], off, it["h"], h_k))
                if camB is not None:
                    pkB = lift_pixel(a_k, camB, W, H, ANKLE_FT)
                    if pkB is not None:
                        offB = np.asarray(pkB) - np.asarray(bk_["foot_raw"], float)
                        rec["dB"]["null"] = dist(Hb, PF.null_foot(bx["foot_stab"], offB, it["h"], h_k))
            else:
                counters["no_partner_pose"] += 1
            rows.append(rec)
    after = {k: RB.manifest(r[k], ("labels", "builds", "cache")) for k in ("ref", "ms")}
    import transformers
    out = {"game": game, "design": "docs/research/P8_posefoot.md section 15", "smoke": smoke, "git": head,
           "model": {"repo": PF.MODEL_REPO, "revision": PF.MODEL_REV, "transformers": transformers.__version__},
           "windows": len(stems), "aligned_windows": len(aligned), "rates": rates, "counters": counters,
           "in_frame_by_window": [score["windows"][s_]["in_frame"] for s_ in stems],
           "nr_boxes_by_window": [score["windows"][s_]["nr_boxes"] for s_ in stems],
           "selection_seconds": round(sel_seconds, 1),
           "manifest_unchanged": after == watched, "seconds": round(time.time() - t0, 1), "rows": rows}
    dd = r["ref"] / "posefoot" / ("pose_smoke" if smoke else "pose")
    dd.mkdir(parents=True, exist_ok=True)
    atomic_write(dd / f"{game}.json", json.dumps(out))
    print(json.dumps({k: out[k] for k in ("game", "smoke", "counters", "rates", "manifest_unchanged", "seconds")}))
    if after != watched:
        raise SystemExit(f"{WHO}: protected files changed")
    return out


# ── report (system python) ─────────────────────────────────────────────────────────────────────────────────────────
def report(smoke: bool = False) -> dict:
    from tools import p8_rebaseline as RB
    from tools.p8_placecause import clean
    from export.schemas import find_private
    r = RB.roots()
    sub = "pose_smoke" if smoke else "pose"
    per = {g: RB.read_json(r["ref"] / "posefoot" / sub / f"{g}.json") for g in PF.POOL}
    o2o = {g: RB.read_json(r["ref"] / "one2one" / f"{g}.json") for g in PF.POOL}
    hit_all = {g: sum(1 for _, rec in o2o[g]["rows"] if rec["class"] == "hit") for g in PF.POOL}
    both = {g: {**per[g], "rows": [x for x in per[g]["rows"] if x["ankles"] == 2]} for g in PF.POOL}
    filled = {g: {**per[g], "rows": [fill_no_change(x) for x in per[g]["rows"]]} for g in PF.POOL}
    one = {g: {**per[g], "rows": [x for x in per[g]["rows"] if x["ankles"] == 1]} for g in PF.POOL}
    out = {"what": "docs/research/P8_posefoot.md section 15, the pose " + ("smoke" if smoke else "falsifier") +
                   "; report only", "smoke": smoke, "git": {g: per[g]["git"] for g in PF.POOL},
           "model": per[PF.CHA]["model"], "counters": {g: per[g]["counters"] for g in PF.POOL},
           "rates": {g: per[g]["rates"] for g in PF.POOL}, "seconds": {g: per[g]["seconds"] for g in PF.POOL},
           "coverage": {}, "mechanical": {}, "hybrid": {}, "pose_raw_rule4_against_foot_raw": {}, "one_ankle": {}}
    for g in PF.POOL:
        rows = per[g]["rows"]
        for kind in ("P-ii", "hit"):
            rk = [x for x in rows if x["kind"] == kind]
            out["coverage"][f"{g}|{kind}"] = {a: (sum(x["ankles"] == a for x in rk) / len(rk) if rk else None)
                                              for a in (2, 1, 0)}
        m = [x["mech"] for x in rows if x["mech"] is not None]
        out["mechanical"][g] = {"share": float(np.mean(m)) if m else None, "n": len(m),
                                "passes": bool(m and np.mean(m) >= MECH_SHARE)}
    for g in list(PF.POOL) + ["pooled"]:
        gs = list(PF.POOL) if g == "pooled" else [g]
        out["hybrid"][g] = PF.unit_rows(both, gs, "hyb", hit_all)
        allr = PF.unit_rows(filled, gs, "hyb", hit_all)          # 15.1 (2): est over every selected row
        for k in ("est", "ghost_change_per_box", "s_hit"):
            out["hybrid"][g][k] = allr[k]
        out["hybrid"][g]["est_rows"] = "every selected row; a missing pose foot reads as no change"
        out["one_ankle"][g] = PF.unit_rows(one, gs, "hyb", hit_all)
        hit = [x for gg in gs for x in both[gg]["rows"] if x["kind"] == "hit" and x["dT"].get("poseraw") is not None]
        dv = [x["dT"]["poseraw"] - x["dT"]["raw"] for x in hit]
        out["pose_raw_rule4_against_foot_raw"][g] = {
            "n": len(hit), "median_d_poseraw_minus_d_raw_T": float(np.median(dv)) if dv else None,
            "share_lost": float(np.mean([x["dT"]["poseraw"] > PF.MISS_FT for x in hit])) if hit else None}
    out["report_only_added_2026_11_28"] = report_only_rows(both, per)
    out["not_produced"] = {"coverage_by_third": "registered (section 6) but not produced: the pose rows store no third",
                           "offset_vector_along_and_across_the_ray": "registered (section 6) but not produced: the pose "
                                                                     "rows store distances, not the offset vector",
                           "R_by_speed_and_no_speed_rows": "not computed for the pose stage: its rows store no speed "
                                                           "(the fields in hybrid and one_ankle are empty by design)"}
    out["rules_hybrid"] = {g: PF.rule_set(out["hybrid"][g]["P-ii own"], out["hybrid"][g]["hits"], out["hybrid"][g]["est"],
                                          "hybrid") for g in PF.POOL}
    if smoke:
        out["note"] = "smoke: two windows; the rules are computed for the mechanics only and are not read"
        out["projection"] = project_stage2(per, {g: RB.read_json(r["ref"] / "posefoot" / "alignment" / f"{g}.json")
                                                 for g in PF.POOL})
    out = clean(out)
    text = json.dumps(out, indent=1)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(RECORD.with_name("p8_posefoot_pose_smoke.json") if smoke else RECORD, text)
    print(json.dumps({k: out[k] for k in ("coverage", "mechanical", "rules_hybrid")}, indent=1))
    return out


def report_only_rows(both: dict, per: dict) -> dict:
    """The registered report-only rows the close review found missing (2026-11-28): the 0.35 ft and 0 ft lifts for the
    hybrid and pose-raw, rows per D207 class, the 3.0 ft ghost view, and coverage by box-height bin. Point estimates,
    with window intervals for the shares."""
    from tools.p8_piibox import median_iv, per_window, picks_for, ratio_iv, values_by_window
    out = {}
    sets = {**{g: [g] for g in PF.POOL}, "pooled": list(PF.POOL)}
    for name, gs in sets.items():
        n_w = [both[g]["windows"] for g in gs]
        picks = picks_for(n_w)
        rows_g = [both[g]["rows"] for g in gs]
        own = [[x for x in rs if x["kind"] == "P-ii" and x["class"] in PF.OWN] for rs in rows_g]
        hit = [[x for x in rs if x["kind"] == "hit"] for rs in rows_g]

        def share(rs_g, fn, need):
            return ratio_iv([per_window([x for x in rs if need(x)], nw, fn) for rs, nw in zip(rs_g, n_w)],
                            [per_window([x for x in rs if need(x)], nw, lambda x: 1) for rs, nw in zip(rs_g, n_w)],
                            picks)

        def med(rs_g, key, ref):
            return median_iv([values_by_window([{"wi": x["wi"], "v": x["dT"][key] - x["dT"][ref]} for x in rs
                                                if x["dT"].get(key) is not None], nw, "v")
                              for rs, nw in zip(rs_g, n_w)], picks)
        lifts = {}
        for key, ref in (("hyb", "stab"), ("hyb_035", "stab"), ("hyb_0", "stab"),
                         ("poseraw", "raw"), ("poseraw_035", "raw"), ("poseraw_0", "raw")):
            has = lambda x, key=key: x["dT"].get(key) is not None
            lifts[key] = {"against": ref,
                          "own_P-ii_R_within_2.5ft": share(own, lambda x, key=key: x["dT"][key] <= PF.MISS_FT, has),
                          "own_P-ii_median_change_T": med(own, key, ref),
                          "hits_share_lost": share(hit, lambda x, key=key: x["dT"][key] > PF.MISS_FT, has),
                          "hits_median_change_T": med(hit, key, ref)}
        classes = {}
        for c in PB_CLASSES:
            rc = [[x for x in rs if x["kind"] == "P-ii" and x["class"] == c] for rs in rows_g]
            if not any(rc):
                continue
            paired = lambda x: x["dT"].get("hyb") is not None and x["dT"].get("null") is not None
            classes[c] = {"n": sum(len(x) for x in rc),
                          "R_hyb": share(rc, lambda x: x["dT"]["hyb"] <= PF.MISS_FT, paired),
                          "R_null": share(rc, lambda x: x["dT"]["null"] <= PF.MISS_FT, paired),
                          "median_change_T": med(rc, "hyb", "stab")}
        ghosts = {}
        for kind, rs_g in (("P-ii", [[x for x in rs if x["kind"] == "P-ii"] for rs in rows_g]), ("hit", hit)):
            hasg = lambda x: x["gT"].get("hyb") is not None
            ghosts[kind] = {"scorer_ghost_share_stab": share(rs_g, lambda x: bool(x["gT"]["stab"]), hasg),
                            "scorer_ghost_share_hyb": share(rs_g, lambda x: bool(x["gT"]["hyb"]), hasg)}
        out[name] = {"lift_sensitivity": lifts, "per_D207_class": classes, "ghost_view_3ft": ghosts}
    cov = {}
    for g in PF.POOL:
        for kind in ("P-ii", "hit"):
            rk = [x for x in per[g]["rows"] if x["kind"] == kind]
            for hb in ("<150", "150-200", "200-250", ">=250"):
                rr = [x for x in rk if hbin_px(x["h"]) == hb]
                cov[f"{g}|{kind}|{hb}"] = {"n": len(rr), **{str(a): (sum(x["ankles"] == a for x in rr) / len(rr)
                                                                     if rr else None) for a in (2, 1, 0)}}
    out["coverage_by_height_bin"] = cov
    return out


def hbin_px(h: float) -> str:
    """D186's box-height bins (tools.p8_placement.hbin's edges)."""
    from tools.p8_placement import hbin
    return hbin(h)


PB_CLASSES = ("own", "own, with crossings", "another's, with crossings", "switch", "another's",
              "own, misplaced, with crossings", "own, misplaced", "ghost, referee-flagged", "ghost, short",
              "ghost, persistent")


def project_stage2(per: dict, align_games: dict) -> dict:
    """15.1 (4): stage 2's lock time per game = stage A's measured decode and gate seconds over the aligned windows +
    the selection seconds measured here over every window + the exact crops over the aligned windows x the smoke's pose
    seconds per crop."""
    crops_s = sum(x["crops"] for g in PF.POOL for x in per[g]["rates"])
    pose_s = sum(x["pose_seconds"] for g in PF.POOL for x in per[g]["rates"])
    spc = pose_s / crops_s if crops_s else None
    out = {"pose_seconds_per_crop": spc, "per_game": {}}
    for g in PF.POOL:
        dec = sum((x.get("seconds") or 0.0) for x in align_games[g]["windows"].values() if x["verdict"] == "aligned")
        crops = per[g]["counters"]["crops_all_aligned"]
        out["per_game"][g] = {"crops_all_aligned": crops, "stage_A_decode_and_gate_s": dec,
                              "selection_s": per[g]["selection_seconds"],
                              "pose_s": crops * spc if spc is not None else None}
        parts = [v for v in (dec, per[g]["selection_seconds"], out["per_game"][g]["pose_s"]) if v is not None]
        out["per_game"][g]["total_s"] = sum(parts)
    out["total_min"] = sum(v["total_s"] for v in out["per_game"].values()) / 60.0
    out["over_120_min"] = out["total_min"] > 120.0
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("pose")
    c.add_argument("game")
    c.add_argument("--smoke", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pose":
        PF.refuse_game(a.game)
        collect_pose(a.game, a.smoke)
    else:
        report(a.smoke)


if __name__ == "__main__":
    sys.exit(main())
