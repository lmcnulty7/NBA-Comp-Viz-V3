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
    if sha256_file(w) != WEIGHTS_SHA:
        raise SystemExit(f"{WHO}: {WEIGHTS_NAME} does not hash as registered")
    return {"commit": head, "weights_sha256": WEIGHTS_SHA}


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
    torch.set_num_threads(THREADS)
    net = Network()
    net.eval()
    ck = torch.load(str(home / "weights" / WEIGHTS_NAME), map_location=torch.device("cpu"))
    net.load_state_dict(ck["state_dict"])

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
                                 "visulize_threshold": config.visulize_threshold}}
    return detect, versions


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
    boxes, secs = {}, []
    for x in frames:
        if time.time() - t0 > cap:
            raise SystemExit(f"{WHO}: past the {cap // 60}-minute cap; nothing is recorded")
        t1 = time.time()
        boxes[f"{x['window']}/{x['frame']}"] = detect(out_d / x["png"])
        secs.append(time.time() - t1)
    per = float(np.mean(secs)) if secs else None
    out = {"what": f"P8 C1 reach stage 2 (S2-C, CrowdDet), {'the probe' if probe else 'every stored frame'}",
           "inputs_head": inp["head"], "source": src, "versions": versions, "threads": THREADS, "device": "cpu",
           "frames": len(frames), "seconds_load": round(t_load, 1), "seconds_per_frame": round(per, 3) if per else None,
           "projected_full_s": round(per * len(inp["frames"]) + t_load, 0) if per else None,
           "boxes_per_frame": [len(v) for v in boxes.values()], "boxes": boxes, "seconds": round(time.time() - t0, 1)}
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


def increment_marks(frame: dict, teacher_boxes: list, crowd_boxes: list, score: float) -> dict:
    """One frame: per C1 miss, reached by the floor, the held teacher (stage 1's rule) and CrowdDet at score; the
    increment is CrowdDet and neither of the others. Also the positive control and the cost rows."""
    from tools import p8_c1reach as C
    from tools.p8_recallcause import feet_of
    sv = np.asarray(frame["sv"], np.float64)
    Ht = np.asarray(frame["Ht"], np.float64)
    inside = np.asarray(frame["inside"], bool)
    base = C.frame_stats(frame, teacher_boxes)
    cb = [b for b in crowd_boxes if b[4] >= score]
    cf = feet_of(cb, Ht)
    cf = cf[C.court_keep(cf)] if len(cf) else cf
    c_m = C.matched(cf, sv)
    p_m = C.matched(feet_of(frame["r23_post"], Ht), sv)
    hits = [j for j in p_m if inside[j]]
    misses = [{**m, "crowd": m["player"] in c_m} for m in base["misses"]]
    for m in misses:
        m["increment"] = m["crowd"] and not m["teacher"] and not m["floor"]
    return {"misses": misses, "hits": len(hits), "hits_reached": sum(1 for j in hits if j in c_m),
            "cost": {"crowd": C.cost_rows(cf, sv, inside), "teacher": base["cost"]["teacher"]}}


def as_reach(misses: list) -> list:
    """The increment in stage 1's form for p8_c1reach.scaled and ceilings: "teacher" = the increment, "floor" = False,
    so their "beyond" is the increment."""
    return [{"unit": m["unit"], "subclass": m["subclass"], "teacher": m["increment"], "floor": False} for m in misses]


def cmd_report(args) -> dict:
    from tools import p8_c1reach as C
    from tools import p8_rebaseline as RB
    from tools import p8_recallcause as RC
    from tools.p8_common import REPO, atomic_write
    from tools.p8_e1b_bound import TKEYS
    from export.schemas import find_private
    r = RB.roots()
    out_d = r["ref"] / "c1reach"
    inp = json.loads((out_d / "inputs.json").read_text())
    tea = json.loads((out_d / "teacher.json").read_text())
    crowd = json.loads((out_d / ("crowddet_probe.json" if args.probe else "crowddet.json")).read_text())
    if tea["inputs_head"] != inp["head"] or crowd["inputs_head"] != inp["head"]:
        raise SystemExit(f"{WHO}: the teacher's or CrowdDet's record does not belong to stage 1's inputs")
    from tools.research_testset import refuse_cv_test
    refuse_cv_test(sorted(inp["per_window"]), WHO)
    C.allowed(sorted(inp["per_window"]))
    if args.probe:
        out = {"what": "P8 C1 reach stage 2 probe (S2-C, CrowdDet); timing only, not scored", "source": crowd["source"],
               "versions": crowd["versions"], "frames": crowd["frames"], "seconds_load": crowd["seconds_load"],
               "seconds_per_frame": crowd["seconds_per_frame"], "projected_full_s": crowd["projected_full_s"],
               "boxes_per_frame": crowd["boxes_per_frame"],
               "gate": {"projected_minutes": round(crowd["projected_full_s"] / 60, 1),
                        "passes": crowd["projected_full_s"] <= CAP_DETECT_S}}
        name = "p8_c1reach_stage2_probe.json"
    else:
        if crowd["frames"] != len(inp["frames"]):
            raise SystemExit(f"{WHO}: CrowdDet ran on {crowd['frames']} frames, not {len(inp['frames'])}")
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
        by_score = {}
        for score in (SCORE,) + SENSITIVITIES:
            ws, hits, hits_r = {}, 0, 0
            cost = {k: {"boxes": 0, "ghosts": 0, "near_no_player": 0, "duplicates": 0} for k in ("crowd", "teacher")}
            for x in inp["frames"]:
                key = f"{x['window']}/{x['frame']}"
                st = increment_marks(x, tea["boxes"][key], crowd["boxes"][key], score)
                ws.setdefault(x["window"], {"misses": []})["misses"] += as_reach(st["misses"])
                hits += st["hits"]; hits_r += st["hits_reached"]
                for k, v in st["cost"].items():
                    for kk, vv in v.items():
                        cost[k][kk] += vv
            agg = C.aggregate(ws, inp["per_window"])
            point = C.ceilings(agg, f_point, n_in)
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
                pw_b = {k: inp["per_window"][k.split("#")[0]] for k in w_b}
                reps.append(C.ceilings(C.aggregate(w_b, pw_b), f_b, n_in))
                reps_f.append(C.ceilings(agg, f_b, n_in))
            iv = lambda rs, g: [round(float(np.percentile([x[g]["C"] for x in rs], 2.5)), 5),
                                round(float(np.percentile([x[g]["C"] for x in rs], 97.5)), 5)]
            inc = {g: {"point": round(point[g]["C"], 5), "interval": iv(reps, g), "f_only_interval": iv(reps_f, g)}
                   for g in point}
            for g in point:                              # the extremes (10.7)
                inc[g]["all_plc"] = round(point[g]["C_all_plc"], 5)
                inc[g]["all_pres"] = round(point[g]["C_all_pres"], 5)
            pos = hits_r / hits if hits else None
            ghost_ok = cost["crowd"]["ghosts"] <= 2 * cost["teacher"]["ghosts"]
            n_inc = sum(1 for w in ws.values() for m in w["misses"] if m["teacher"])
            ent = {"increment_sampled": n_inc, "increment_ceiling": inc, "cost_rows": cost, "ghost_ok": ghost_ok,
                   "positive_control": {"r23_hits": hits, "crowd_reached": hits_r, "share": pos,
                                        "passes": pos is not None and pos >= C.POS_CTRL_MIN}}
            if score == SCORE:
                cha = inc["gsw_cha_2016"]
                ent["reading"] = {f"gap {gap}": outcome(cha["point"], cha["interval"], cha["f_only_interval"],
                                                       inc["pooled"]["point"], ghost_ok, gap) if ent["positive_control"]["passes"]
                                  else "positive control fails; nothing is read"
                                  for gap in (round(ub["gap"], 4), LARGER_GAP)}
            by_score[str(score)] = ent
        out = {"what": "P8 C1 reach stage 2 (S2-C, CrowdDet; docs/research/P8_c1reach.md section 13); report only",
               "inputs_head": inp["head"], "source": crowd["source"], "versions": crowd["versions"],
               "frames": crowd["frames"], "seconds": crowd["seconds"], "score": SCORE, "by_score": by_score,
               "union_baseline": {k: round(v, 5) for k, v in ub.items()}, "bars": {str(g): bars(g) for g in
                                                                                    (round(ub["gap"], 4), LARGER_GAP)},
               "fractions": f_point, "in_frame": n_in}
        name = "p8_c1reach_stage2.json"
    text = json.dumps(out, indent=1, default=float)
    if find_private(text):
        raise SystemExit(f"{WHO}: the record holds private data")
    atomic_write(REPO / "reports" / "research" / name, text)
    print(json.dumps({k: v for k, v in out.items() if k in ("gate", "seconds_per_frame")} or
                     {s: e.get("reading") for s, e in out["by_score"].items()}, indent=1))
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
