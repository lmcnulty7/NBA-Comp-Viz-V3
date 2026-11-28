"""tools/p8_c1reach_s2.py: stage 2 of the C1 reach audit, S2-C, CrowdDet (docs/research/P8_c1reach.md section 13).
Report only: nothing is fitted, tuned or trained. It reuses stage 1's 775 stored frames, inputs and teacher boxes.

    B2  <the CrowdDet venv python> -m tools.p8_c1reach_s2 probe     inside the lock: 5 frames, timed, not scored
    B2  <the CrowdDet venv python> -m tools.p8_c1reach_s2 detect    inside the lock: every stored frame
    C2  python3 -m tools.p8_c1reach_s2 report [--probe]             system python, no lock

The venv steps import nothing from the repository but this file (the repository's top-level config.py would shadow
CrowdDet's config module), read V3_P8_ROOT for the reference root and V3_CROWDDET_HOME for the CrowdDet folder (src/, a
clone at CROWDDET_COMMIT, and weights/), and run CrowdDet's own tools/inference.py get_data and post_process
unchanged, on the CPU. The report step runs every guard.
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
WHO = "tools.p8_c1reach_s2"
CROWDDET_COMMIT = "e3620a6bb300256fb8b460948138839ca5dc7c5f"     # xg-chu/CrowdDet master, read 2026-11-28
MODEL_DIR = "rcnn_emd_refine"
WEIGHTS_NAME = "rcnn_emd_refine.pth"                             # "RCNN EMD with RM (trained with PyTorch)"
WEIGHTS_SHA = None                                               # set by a dated amendment after the download
WEIGHTS_SIZE = None                                              # bytes, set with the SHA
HEADLINE = ("C1a_other", "C1c")                                  # 13.8: the population of H and the prior
APART = ("C1b", "C1a_own")
UNION_IOU = 0.5                                                  # 13.8: a CrowdDet box joins unless a teacher box overlaps it
REF_MASK_IOU = 0.5                                               # 13.8: the referee-mask sensitivity
GHOST_FACTOR = 2.0
SCORE, SENSITIVITIES = 0.5, (0.3, 0.7)
THREADS = 4
PROBE_N = 5
CAP_PROBE_S, CAP_DETECT_S = 10 * 60, 60 * 60
LARGER_GAP = 0.0069                                              # 10.12: the union-baseline gap with the residual
MARGIN = 0.0021                                                  # the bar's distance below the kill line (10.4)


# ── shared, stdlib and numpy only ──────────────────────────────────────────────────────────────────────────────────
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, p)


def ref_root() -> Path:
    return Path(os.environ["V3_P8_ROOT"]).expanduser().resolve()


def crowddet_home() -> Path:
    return Path(os.environ.get("V3_CROWDDET_HOME") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data"
                / "crowddet").expanduser().resolve()


# ── the CrowdDet venv ──────────────────────────────────────────────────────────────────────────────────────────────
def check_source(home: Path) -> dict:
    """The clone is at CROWDDET_COMMIT with no local change (13.3), and the weights hash as registered."""
    import subprocess
    src = home / "src"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=src, capture_output=True, text=True, check=True).stdout
    if head != CROWDDET_COMMIT:
        raise SystemExit(f"{WHO}: the CrowdDet clone is at {head}, not {CROWDDET_COMMIT}")
    if dirty.strip():
        raise SystemExit(f"{WHO}: the CrowdDet clone has local changes: {dirty.strip()[:200]}")
    w = home / "weights" / WEIGHTS_NAME
    if WEIGHTS_SHA is None:
        raise SystemExit(f"{WHO}: WEIGHTS_SHA is not registered yet (a dated amendment after the download)")
    if sha256_file(w) != WEIGHTS_SHA or w.stat().st_size != WEIGHTS_SIZE:
        raise SystemExit(f"{WHO}: {WEIGHTS_NAME} does not hash or size as registered")
    return {"commit": head, "weights_sha256": WEIGHTS_SHA, "weights_size": WEIGHTS_SIZE,
            "weights_path": f"weights/{WEIGHTS_NAME}"}


def load_detector(home: Path):
    """CrowdDet's network and its inference helpers, unchanged: the model folder's config and Network, and
    tools/inference.py's get_data and post_process. -> (detect(png_path) -> [[x1, y1, x2, y2, score, "person"]],
    versions)."""
    import importlib.util
    import sys
    import cv2
    import torch
    import torchvision
    if "config" in sys.modules:
        raise SystemExit(f"{WHO}: a config module is already imported; CrowdDet's would be shadowed")
    src = home / "src"
    sys.path.insert(0, str(src / "lib"))
    sys.path.insert(0, str(src / "model" / MODEL_DIR))
    from config import config
    from network import Network
    spec = importlib.util.spec_from_file_location("crowddet_inference", str(src / "tools" / "inference.py"))
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)
    import network as _network
    nms_mod = sys.modules.get("utils.nms_utils")
    files = {"config": getattr(sys.modules.get("config"), "__file__", None), "network": _network.__file__,
             "nms_utils": getattr(nms_mod, "__file__", None)}
    root = str(src.resolve())
    bad = {k: v for k, v in files.items() if not v or not str(Path(v).resolve()).startswith(root + os.sep)}
    if bad:
        raise SystemExit(f"{WHO}: CrowdDet modules not from the pinned clone: {bad}")
    torch.set_num_threads(THREADS)
    net = Network()
    net.eval()
    try:                                             # 13.8: weights_only first; never a fallback without the lead
        ck = torch.load(str(home / "weights" / WEIGHTS_NAME), map_location=torch.device("cpu"), weights_only=True)
    except Exception as e:
        raise SystemExit(f"{WHO}: torch.load with weights_only=True failed ({type(e).__name__}: {e}); stopped, "
                         "no fallback to weights_only=False without the lead")
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise SystemExit(f"{WHO}: the checkpoint has no 'state_dict' key (keys {list(ck)[:5] if isinstance(ck, dict) else type(ck)})")
    net.load_state_dict(ck["state_dict"], strict=True)
    load = {"method": "torch.load(weights_only=True, map_location=cpu)", "key": "state_dict", "strict": True,
            "checkpoint_keys": sorted(ck)}

    def detect(png_path: Path) -> list:
        _, resized, im_info = inf.get_data(str(png_path), config.eval_image_short_size, config.eval_image_max_size)
        with torch.no_grad():
            pred = net(resized, im_info).numpy()
        boxes = inf.post_process(pred, config, im_info[0, 2])
        return [[round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2),
                 round(float(b[4]), 5), "person"] for b in boxes]
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__,
                "numpy": np.__version__, "python": sys.version.split()[0],
                "post_process": {"test_nms_method": config.test_nms_method, "test_nms": config.test_nms,
                                 "pred_cls_threshold": config.pred_cls_threshold,
                                 "visulize_threshold": config.visulize_threshold}, "load": load,
                "module_files": {k: str(Path(v).resolve().relative_to(src.resolve())) for k, v in files.items()}}
    return detect, versions


def boxes_digest(boxes: dict) -> str:
    """13.8 (iv): the SHA-256 of a boxes map as canonical JSON."""
    return hashlib.sha256(json.dumps(boxes, sort_keys=True).encode()).hexdigest()


def _venv_common(probe: bool) -> tuple:
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    out_d = ref_root() / "c1reach"
    inp = json.loads((out_d / "inputs.json").read_text())
    frames = inp["frames"][:PROBE_N] if probe else inp["frames"]
    bad = [f"{x['window']}/{x['frame']}" for x in frames if sha256_file(out_d / x["png"]) != x["sha256"]]
    if bad:
        raise SystemExit(f"{WHO}: {len(bad)} stage 1 frames do not hash as written, first {bad[:3]}")
    home = crowddet_home()
    src = check_source(home)
    return out_d, inp, frames, home, src


def cmd_run(args, probe: bool) -> dict:
    out_d, inp, frames, home, src = _venv_common(probe)
    t0 = time.time()
    detect, versions = load_detector(home)
    t_load = time.time() - t0
    cap = CAP_PROBE_S if probe else CAP_DETECT_S
    probe_rec = None
    if not probe:                                    # 13.8 (iv): the probe's boxes, checked, to reproduce
        probe_rec = json.loads((out_d / "crowddet_probe.json").read_text())
        if boxes_digest(probe_rec["boxes"]) != probe_rec["boxes_sha256"]:
            raise SystemExit(f"{WHO}: the probe's stored boxes do not hash as recorded")
    boxes, secs = {}, []
    for x in frames:
        if time.time() - t0 > cap:
            raise SystemExit(f"{WHO}: past the {cap // 60}-minute cap; nothing is recorded")
        t1 = time.time()
        key = f"{x['window']}/{x['frame']}"
        boxes[key] = detect(out_d / x["png"])
        secs.append(time.time() - t1)
        if probe_rec is not None and key in probe_rec["boxes"] and boxes[key] != probe_rec["boxes"][key]:
            raise SystemExit(f"{WHO}: {key} does not reproduce the probe's boxes; stopped")
    per = float(np.mean(secs)) if secs else None
    out = {"what": f"P8 C1 reach stage 2 (S2-C, CrowdDet), {'the probe' if probe else 'every stored frame'}",
           "inputs_head": inp["head"], "source": src, "versions": versions, "threads": THREADS, "device": "cpu",
           "frames": len(frames), "seconds_load": round(t_load, 1), "seconds_per_frame": round(per, 3) if per else None,
           "projected_full_s": round(per * len(inp["frames"]) + t_load, 0) if per else None,
           "boxes_per_frame": [len(v) for v in boxes.values()], "boxes": boxes, "boxes_sha256": boxes_digest(boxes),
           "probe_reproduced": None if probe else sorted(k for k in boxes if k in probe_rec["boxes"]),
           "seconds": round(time.time() - t0, 1)}
    write_atomic(out_d / ("crowddet_probe.json" if probe else "crowddet.json"), json.dumps(out))
    print(json.dumps({k: out[k] for k in ("frames", "seconds_per_frame", "projected_full_s", "boxes_per_frame")
                      if k != "boxes_per_frame" or probe}))
    return out


# ── pure (the report) ──────────────────────────────────────────────────────────────────────────────────────────────
def bars(gap: float) -> dict:
    """10.6 under a gap: the kill line at -gap, the bar MARGIN below it (the registered gap 0.0021 gives -0.0021 and
    -0.0042; the larger gap 0.0069 gives -0.0069 and -0.0090)."""
    return {"kill": -gap, "bar": -gap - MARGIN}


def outcome(inc_point: float, inc_iv, inc_f_iv, pooled_point: float, ghost_ok: bool, gap: float) -> str:
    """10.6's zones for gsw_cha_2016's increment under one gap -> the action's label."""
    b = bars(gap)
    if inc_point > b["kill"]:
        return "kill: the CrowdHuman-detector line is closed for these sources"
    if inc_point > b["bar"]:
        return "short of the bar: no arm; the detector line stays paused"
    if inc_iv is None or inc_iv[1] >= b["kill"]:
        if inc_f_iv is not None and inc_f_iv[1] >= b["kill"]:
            return "not resolvable with these truths: no arm; paused"
        return "not shown: no arm; paused"
    if not pooled_point < 0:
        return "not shown (pooled): no arm; paused"
    if not ghost_ok:
        return "ghost failure: no arm on its raw boxes; a filtered-label design may be registered; paused"
    return "a training-arm design is justified (a union label set; E1b 2b's full bound first)"


def box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def union_boxes(teacher: list, crowd: list) -> list:
    """13.8: the teacher's boxes plus each CrowdDet box that no teacher box overlaps at IoU UNION_IOU or more."""
    return list(teacher) + [c for c in crowd if not any(box_iou(c, t) >= UNION_IOU for t in teacher)]


def referee_mask(crowd: list, referees: list) -> list:
    """13.8's sensitivity: CrowdDet boxes overlapping one of r23's class-1 (referee) boxes at IoU REF_MASK_IOU or more
    are dropped."""
    return [c for c in crowd if not any(box_iou(c, r) >= REF_MASK_IOU for r in referees)]


def increment_marks(frame: dict, teacher_boxes: list, crowd_boxes: list, score: float, referees=None) -> dict:
    """One frame (13.8): the union of the held teacher's boxes (stage 1's rule: at 0.1471) and CrowdDet's at score,
    after the merge rule; with referees given, CrowdDet's boxes on r23's referee boxes are masked first. Per C1 miss:
    reached by the floor, the held teacher, CrowdDet alone and the union; the increment is the union and neither the
    floor nor the teacher. Also the positive controls and the cost rows (court-filtered feet)."""
    from tools import p8_c1reach as C
    from tools.p8_recallcause import feet_of
    sv = np.asarray(frame["sv"], np.float64)
    Ht = np.asarray(frame["Ht"], np.float64)
    inside = np.asarray(frame["inside"], bool)
    base = C.frame_stats(frame, teacher_boxes)

    def kept_feet(bs):
        f = feet_of(bs, Ht)
        return f[C.court_keep(f)] if len(f) else f
    tb = [b for b in teacher_boxes if b[4] >= C.TEACHER_CONF]
    cb = [b for b in crowd_boxes if b[4] >= score]
    if referees is not None:
        cb = referee_mask(cb, referees)
    ub = union_boxes(tb, cb)
    cf, uf, tf = kept_feet(cb), kept_feet(ub), kept_feet(tb)
    c_m, u_m = C.matched(cf, sv), C.matched(uf, sv)
    p_m = C.matched(feet_of(frame["r23_post"], Ht), sv)
    hits = [j for j in p_m if inside[j]]
    misses = [{**m, "crowd": m["player"] in c_m, "union": m["player"] in u_m} for m in base["misses"]]
    for m in misses:
        m["increment"] = m["union"] and not m["teacher"] and not m["floor"]
    raw_c = feet_of(cb, Ht)
    return {"misses": misses, "hits": len(hits), "hits_reached_crowd": sum(1 for j in hits if j in c_m),
            "hits_reached_union": sum(1 for j in hits if j in u_m),
            "crowd_boxes_before_court": len(raw_c), "crowd_boxes_after_court": len(cf),
            "cost": {"crowd": C.cost_rows(cf, sv, inside), "union": C.cost_rows(uf, sv, inside),
                     "teacher": C.cost_rows(tf, sv, inside)}}


def as_reach(misses: list, subclasses=None) -> list:
    """The increment in stage 1's form for p8_c1reach.scaled and ceilings ("teacher" = the increment, "floor" = False,
    so their "beyond" is the increment), optionally on the given subclasses only."""
    return [{"unit": m["unit"], "subclass": m["subclass"], "teacher": m["increment"], "floor": False} for m in misses
            if subclasses is None or m["subclass"] in subclasses]


def reading(inc_cha: dict, pooled_point: float, ghost_ok: bool, gap: float, larger_gap: float = LARGER_GAP) -> dict:
    """13.8: the action follows the registered gap; the larger gap is a sensitivity. A pass under the registered gap
    but not under the larger one reads "passes, not robust to the transfer residual"."""
    a = outcome(inc_cha["point"], inc_cha["interval"], inc_cha["f_only_interval"], pooled_point, ghost_ok, gap)
    b = outcome(inc_cha["point"], inc_cha["interval"], inc_cha["f_only_interval"], pooled_point, ghost_ok, larger_gap)
    action = a
    if a.startswith("a training-arm design") and not b.startswith("a training-arm design"):
        action = ("passes, not robust to the transfer residual: a training-arm design is justified only with that "
                  "limit carried")
    return {"action": action, f"under gap {gap}": a, f"sensitivity, gap {larger_gap}": b}


def referees_for(inp: dict, ref: Path) -> dict:
    """13.8: r23's class-1 (referee) boxes at 0.40 per sampled frame: from step A's inputs when they carry them
    ("r23_ref"), else from the same stored candidate files step A read, whose R_post must equal step A's r23_post."""
    from tools import p8_recallcause as RC
    from tools import p8_recallsupp as RS
    if all("r23_ref" in x for x in inp["frames"]):
        return {f"{x['window']}/{x['frame']}": x["r23_ref"] for x in inp["frames"]}
    g0 = RC.read_json(RS.out_path("g0"))
    cert = set(g0["certified"])
    out, cache = {}, {}
    for x in inp["frames"]:
        s = x["window"]
        if s not in cache:
            src = RC.raw_dir(ref) if s in cert else ref / "recallsupp"
            cache[s] = RC.read_json(src / f"{s}.json")
        v = cache[s][str(x["frame"])]
        if v["post"] != x["r23_post"]:
            raise SystemExit(f"{WHO}: {s}/{x['frame']}: the stored candidates are not the ones step A read")
        out[f"{s}/{x['frame']}"] = [b for b in v["other"] if b[5] == RC.REFEREE_CLASS]
    return out


def increment_table(inp, tea, crowd, score, referees, e1, e1_keys, f_point, n_in, boot: bool) -> dict:
    """One configuration (score, with or without the referee mask): the headline increment on HEADLINE, the APART
    subclasses, the positive controls, the cost rows and the ghost bar on the union."""
    from tools import p8_c1reach as C
    ws_h, ws_a, ws_H, hits, hu, hc, frames = {}, {}, {}, 0, 0, 0, 0
    union_loss = {c: 0 for c in HEADLINE + APART}
    cost = {k: {"boxes": 0, "ghosts": 0, "near_no_player": 0, "duplicates": 0} for k in ("crowd", "union", "teacher")}
    before_court = after_court = 0
    for x in inp["frames"]:
        key = f"{x['window']}/{x['frame']}"
        st = increment_marks(x, tea["boxes"][key], crowd["boxes"][key], score,
                             None if referees is None else referees[key])
        ws_h.setdefault(x["window"], {"misses": []})["misses"] += as_reach(st["misses"], HEADLINE)
        ws_a.setdefault(x["window"], {"misses": []})["misses"] += as_reach(st["misses"], APART)
        ws_H.setdefault(x["window"], {"misses": []})["misses"] += [   # 13.9: H with the headline's own k
            {"unit": m["unit"], "subclass": m["subclass"], "teacher": not m["teacher"] and not m["floor"], "floor": False}
            for m in st["misses"] if m["subclass"] in HEADLINE]
        for m in st["misses"]:                       # 13.9: what the union loses of the teacher's reach
            if m["teacher"] and not m["union"]:
                union_loss[m["subclass"]] += 1
        hits += st["hits"]; hu += st["hits_reached_union"]; hc += st["hits_reached_crowd"]; frames += 1
        before_court += st["crowd_boxes_before_court"]; after_court += st["crowd_boxes_after_court"]
        for k, v in st["cost"].items():
            for kk, vv in v.items():
                cost[k][kk] += vv

    def ceil(ws, subclasses):
        pw = {s: {**inp["per_window"][s], "n": sum(inp["per_window"][s]["by"].get(f"{u}|{c}", 0)
                                                   for u in ("P-i", "D-i") for c in subclasses)}
              for s in ws}
        agg = C.aggregate(ws, pw)
        point = C.ceilings(agg, f_point, n_in)
        out = {g: {"point": round(point[g]["C"], 5), "all_plc": round(point[g]["C_all_plc"], 5),
                   "all_pres": round(point[g]["C_all_pres"], 5)} for g in point}
        if boot:
            rng = np.random.default_rng(C.BOOT_SEED)
            ws_by_g = {g: [s for s in sorted(ws) if s.startswith(g + "_")] for g in C.ALLOWED}
            reps, reps_f = [], []
            for _ in range(C.BOOT_N):
                f_b = {g: C.e1_fractions(e1, [e1_keys[g][i] for i in rng.integers(0, len(e1_keys[g]), len(e1_keys[g]))])
                       for g in C.ALLOWED}
                w_b = {}
                for g in C.ALLOWED:
                    for k, i in enumerate(rng.integers(0, len(ws_by_g[g]), len(ws_by_g[g])) if ws_by_g[g] else []):
                        w_b[f"{ws_by_g[g][i]}#{k}"] = ws[ws_by_g[g][i]]
                reps.append(C.ceilings(C.aggregate(w_b, {k: pw[k.split("#")[0]] for k in w_b}), f_b, n_in))
                reps_f.append(C.ceilings(agg, f_b, n_in))
            for g in point:
                for nm, rs in (("interval", reps), ("f_only_interval", reps_f)):
                    v = [r[g]["C"] for r in rs]
                    out[g][nm] = [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
        return out
    ghost_pf = {k: cost[k]["ghosts"] / frames for k in cost} if frames else {}
    ghost_ok = frames > 0 and ghost_pf["union"] <= GHOST_FACTOR * ghost_pf["teacher"]
    def lost(ws, subclasses):
        """Misses of the population in windows whose sampled frames hold none of it (scaled away, k = 0)."""
        tot = sum(inp["per_window"][s]["by"].get(f"{u}|{c}", 0) for s in inp["per_window"] for u in ("P-i", "D-i")
                  for c in subclasses)
        gone = sum(inp["per_window"][s]["by"].get(f"{u}|{c}", 0) for s in inp["per_window"]
                   for u in ("P-i", "D-i") for c in subclasses if not (ws.get(s) or {}).get("misses"))
        return {"lost": gone, "of": tot}
    return {"headline_increment": ceil(ws_h, HEADLINE), "apart_increment": ceil(ws_a, APART),
            "H_per_population": ceil(ws_H, HEADLINE), "union_losses_sampled": union_loss,
            "population_loss": {"headline": lost(ws_h, HEADLINE), "apart": lost(ws_a, APART)},
            "increment_sampled": {"headline": sum(m["teacher"] for w in ws_h.values() for m in w["misses"]),
                                  "apart": sum(m["teacher"] for w in ws_a.values() for m in w["misses"])},
            "positive_control": {"r23_hits": hits, "union_reached": hu, "crowd_reached": hc,
                                 "share_union": hu / hits if hits else None, "share_crowd": hc / hits if hits else None,
                                 "passes": bool(hits) and hc / hits >= C.POS_CTRL_MIN},
            "crowd_boxes": {"before_court": before_court, "after_court": after_court},
            "cost_rows": cost, "ghosts_per_frame": ghost_pf, "ghost_ok": ghost_ok}


def cmd_report(args) -> dict:
    from tools import p8_c1reach as C
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools.p8_common import REPO, atomic_write
    from tools.p8_e1b_bound import TKEYS
    from tools.research_testset import refuse_cv_test
    from export.schemas import find_private
    r = RB.roots()
    out_d = r["ref"] / "c1reach"
    inp = json.loads((out_d / "inputs.json").read_text())
    tea = json.loads((out_d / "teacher.json").read_text())
    crowd = json.loads((out_d / ("crowddet_probe.json" if args.probe else "crowddet.json")).read_text())
    if tea["inputs_head"] != inp["head"] or crowd["inputs_head"] != inp["head"]:
        raise SystemExit(f"{WHO}: the teacher's or CrowdDet's record does not belong to stage 1's inputs")
    if boxes_digest(crowd["boxes"]) != crowd["boxes_sha256"]:
        raise SystemExit(f"{WHO}: CrowdDet's boxes do not hash as recorded")
    refuse_cv_test(sorted(inp["per_window"]), WHO)
    C.allowed(sorted(inp["per_window"]))
    if args.probe:                                   # 13.8: the probe's checks, plus timing; not scored
        sub = {**inp, "frames": [x for x in inp["frames"] if f"{x['window']}/{x['frame']}" in crowd["boxes"]]}
        hits = hr = bc = ac = gh = 0
        for x in sub["frames"]:
            key = f"{x['window']}/{x['frame']}"
            st = increment_marks(x, tea["boxes"][key], crowd["boxes"][key], SCORE)
            hits += st["hits"]; hr += st["hits_reached_crowd"]
            bc += st["crowd_boxes_before_court"]; ac += st["crowd_boxes_after_court"]; gh += st["cost"]["crowd"]["ghosts"]
        n = len(sub["frames"])
        out = {"what": "P8 C1 reach stage 2 probe (S2-C, CrowdDet); checks and timing, reach not scored",
               "source": crowd["source"], "versions": crowd["versions"], "frames": n,
               "seconds_load": crowd["seconds_load"], "seconds_per_frame": crowd["seconds_per_frame"],
               "projected_full_s": crowd["projected_full_s"], "boxes_sha256": crowd["boxes_sha256"],
               "checks": {"positive_control": {"r23_hits": hits, "crowd_reached": hr,
                                               "share": hr / hits if hits else None,
                                               "passes": bool(hits) and hr / hits >= C.POS_CTRL_MIN},
                          "boxes_per_frame_before_court": bc / n if n else None,
                          "boxes_per_frame_after_court": ac / n if n else None,
                          "in_court_ghosts_per_frame": gh / n if n else None,
                          "timing": {"projected_minutes": round(crowd["projected_full_s"] / 60, 1),
                                     "passes": crowd["projected_full_s"] <= CAP_DETECT_S}}}
        out["checks"]["every_frame_has_boxes"] = all(v > 0 for v in crowd["boxes_per_frame"])     # 13.4
        out["passes"] = (out["checks"]["positive_control"]["passes"] and out["checks"]["timing"]["passes"]
                         and out["checks"]["every_frame_has_boxes"])
        name = "p8_c1reach_stage2_probe.json"
    else:
        if crowd["frames"] != len(inp["frames"]) or not crowd.get("probe_reproduced"):
            raise SystemExit(f"{WHO}: CrowdDet's full run is incomplete or did not reproduce the probe")
        d202 = {g: RC.read_json(r["ref"] / "rebaseline" / f"score__{g}__current.json") for g in C.ALLOWED}
        n_in = {g: sum(w["in_frame"] for w in d202[g]["windows"].values()) for g in C.ALLOWED}
        e1 = C.e1_per_window(r["ws"])
        rec = json.loads((REPO / "reports" / "research" / "p8_e1b_transfer.json").read_text())
        for g in C.ALLOWED:
            if {t: sum(e1[k][t] for k in e1 if k.startswith(g + "/")) for t in TKEYS} != rec["counts"][g]:
                raise SystemExit(f"{WHO}: E1's per-window counts do not reproduce E1b's record on {g}")
        e1_keys = {g: sorted(k for k in e1 if k.startswith(g + "/")) for g in C.ALLOWED}
        f_point = {g: C.e1_fractions(e1, e1_keys[g]) for g in C.ALLOWED}
        ub = C.union_baseline(rec["rates"]["gsw_cha_2016"]["C"])
        refs = referees_for(inp, r["ref"])
        main = increment_table(inp, tea, crowd, SCORE, None, e1, e1_keys, f_point, n_in, True)
        cha = main["headline_increment"]["gsw_cha_2016"]
        main["reading"] = (reading(cha, main["headline_increment"]["pooled"]["point"], main["ghost_ok"],
                                   round(ub["gap"], 4)) if main["positive_control"]["passes"]
                           else {"action": "the positive control fails; nothing is read"})
        sens = {"referee_mask_0.5": increment_table(inp, tea, crowd, SCORE, refs, e1, e1_keys, f_point, n_in, True)}
        for sc in SENSITIVITIES:                     # reported only; they cannot rescue a failed 0.5 (13.8)
            sens[f"score_{sc}"] = increment_table(inp, tea, crowd, sc, None, e1, e1_keys, f_point, n_in, False)
        out = {"what": "P8 C1 reach stage 2 (S2-C, CrowdDet; docs/research/P8_c1reach.md sections 13 and 13.8); "
                       "report only", "inputs_head": inp["head"], "source": crowd["source"],
               "versions": crowd["versions"], "frames": crowd["frames"], "seconds": crowd["seconds"],
               "score": SCORE, "main": main, "sensitivities": sens,
               "union_baseline": {k: round(v, 5) for k, v in ub.items()},
               "bars": {"registered": bars(round(ub["gap"], 4)), "sensitivity": bars(LARGER_GAP)},
               "fractions": f_point, "in_frame": n_in}
        name = "p8_c1reach_stage2.json"
    text = json.dumps(out, indent=1, default=float)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(REPO / "reports" / "research" / name, text)
    print(json.dumps(out.get("checks") or out["main"]["reading"], indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("detect")
    rp = sub.add_parser("report")
    rp.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        cmd_run(args, True)
    elif args.cmd == "detect":
        cmd_run(args, False)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
