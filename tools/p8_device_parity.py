"""tools/p8_device_parity.py: M7, device parity of the raw detections, CPU against MPS, on the E1 corpus JPEGs
(docs/research/P8_E1.md, "M7", definitions committed before any number). Report only: no build, no registered file.

    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_device_parity run --game G --device mps [--frames N]
    python3 -m export.heavy --name ... -- <frozen CV python> -m tools.p8_device_parity run --game G --device cpu --frames 3
    PYTORCH_ENABLE_MPS_FALLBACK=0 ... -m tools.p8_device_parity run --game G --device mps --frames 3 --tag probe
    python3 -m tools.p8_device_parity score [--games G,...]                     -> parity/p8_device_parity.json

run: r23 (player class) and the E1 fold that left G out, on G's corpus JPEGs in plan order, once at confidence 0.10,
NMS 0.45, image size 640, half off, on one device. Records versions, the loaded model's fp16 flag, every warning
raised during inference and the MPS fallback setting. score: M4's CPU outputs (if a fresh CPU pilot reproduces them
exactly) against the MPS outputs, at 0.40.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, guard, main_checkout, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_device_parity"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
MODELS = ("r23", "e1")
FLOOR, CUT, NMS_IOU, IMGSZ = 0.10, 0.40, 0.45, 640
IOU = 0.5
CONF_BANDS = (0.50, 0.70, 0.90)              # 0.40-0.50, 0.50-0.70, 0.70-0.90, >=0.90
HEIGHT_BANDS = (150.0, 200.0, 250.0)
LOSS_MAX, MISSED_MAX = 0.01, 0.005           # the parity rule, fixed before any number


def check_games(games) -> list:
    games = list(games)
    bad = [g for g in games if g not in GAMES]
    if bad:
        raise SystemExit(f"{WHO}: {bad[0]} is not a development pool game")
    refuse_cv_test(games, WHO)
    guard(games, WHO)
    return games


def out_dir() -> Path:
    d = p8_root() / "parity"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── pure ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def band(v: float, edges, fmt="%g") -> str:
    for k, e in enumerate(edges):
        if v < e:
            return ("<" + fmt % e) if k == 0 else (fmt % edges[k - 1] + "-" + fmt % e)
    return ">=" + fmt % edges[-1]


def conf_band(c: float) -> str:
    return band(c, CONF_BANDS, "%.2f").replace("<0.50", "0.40-0.50")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if inter > 0 and ua > 0 else 0.0


def match(cpu: list, mps: list, thr: float = IOU) -> list:
    """One-to-one Hungarian match on IoU -> [(cpu index, mps index)] at IoU >= thr."""
    from scipy.optimize import linear_sum_assignment
    if not cpu or not mps:
        return []
    M = np.array([[iou(a, b) for b in mps] for a in cpu])
    ri, ci = linear_sum_assignment(-M)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if M[i, j] >= thr]


def frame_parity(cpu: list, mps: list) -> dict:
    """One frame's boxes [x1, y1, x2, y2, conf] at the floor -> counts at the cut, lost CPU boxes by band, and the
    confidence differences of boxes matched at the floor."""
    c40 = [b for b in cpu if b[4] >= CUT - 1e-9]
    m40 = [b for b in mps if b[4] >= CUT - 1e-9]
    pairs = match(c40, m40)
    hit = {i for i, _ in pairs}
    lost = [c40[i] for i in range(len(c40)) if i not in hit]
    out = {"cpu": len(c40), "mps": len(m40), "matched": len(pairs), "cpu_only": len(lost),
           "mps_only": len(m40) - len(pairs), "by_conf": {}, "by_height": {}, "lost_by_conf": {}, "lost_by_height": {}}
    for b in c40:
        cb, hb = conf_band(b[4]), band(b[3] - b[1], HEIGHT_BANDS)
        out["by_conf"][cb] = out["by_conf"].get(cb, 0) + 1
        out["by_height"][hb] = out["by_height"].get(hb, 0) + 1
    for b in lost:
        cb, hb = conf_band(b[4]), band(b[3] - b[1], HEIGHT_BANDS)
        out["lost_by_conf"][cb] = out["lost_by_conf"].get(cb, 0) + 1
        out["lost_by_height"][hb] = out["lost_by_height"].get(hb, 0) + 1
    out["conf_diffs"] = [abs(cpu[i][4] - mps[j][4]) for i, j in match(cpu, mps)]
    return out


def merge(acc: dict, f: dict) -> None:
    for k, v in f.items():
        if isinstance(v, dict):
            d = acc.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv
        elif isinstance(v, list):
            acc.setdefault(k, []).extend(v)
        else:
            acc[k] = acc.get(k, 0) + v


def summarise(acc: dict, frames: int) -> dict:
    d = np.array(acc.get("conf_diffs", []), float)
    loss = acc["cpu_only"] / acc["cpu"] if acc["cpu"] else None
    return {"frames": frames, "cpu_boxes": acc["cpu"], "mps_boxes": acc["mps"],
            "cpu_boxes_per_frame": round(acc["cpu"] / frames, 4), "mps_boxes_per_frame": round(acc["mps"] / frames, 4),
            "matched": acc["matched"], "mps_box_loss": None if loss is None else round(loss, 5),
            "mps_only_share": round(acc["mps_only"] / acc["mps"], 5) if acc["mps"] else None,
            "loss_by_conf": {k: {"cpu": v, "lost": acc["lost_by_conf"].get(k, 0),
                                 "share": round(acc["lost_by_conf"].get(k, 0) / v, 5)} for k, v in sorted(acc["by_conf"].items())},
            "loss_by_height": {k: {"cpu": v, "lost": acc["lost_by_height"].get(k, 0),
                                   "share": round(acc["lost_by_height"].get(k, 0) / v, 5)} for k, v in sorted(acc["by_height"].items())},
            "conf_abs_diff_max": float(d.max()) if len(d) else None,
            "conf_abs_diff_p99": float(np.percentile(d, 99)) if len(d) else None,
            "conf_abs_diff_median": float(np.median(d)) if len(d) else None, "conf_pairs": int(len(d))}


def passes(loss, missed_diff) -> bool:
    return loss is not None and missed_diff is not None and loss < LOSS_MAX and abs(missed_diff) < MISSED_MAX


def same_preds(a: dict, b: dict) -> bool:
    """Two runs' preds agree on every frame present in both, box for box (M4's rounding)."""
    return all(a[k] == b[k] for k in a.keys() & b.keys()) and bool(a.keys() & b.keys())


# ── run (frozen CV python, inside the lock) ─────────────────────────────────────────────────────────────────────────
def _param_state(pm) -> dict:
    inner = getattr(pm, "model", pm)
    try:
        p0 = next(inner.parameters())
        return {"fp16": bool(getattr(pm, "fp16", False)), "param_dtype": str(p0.dtype), "device": str(p0.device)}
    except (StopIteration, AttributeError):
        return {"fp16": bool(getattr(pm, "fp16", False)), "param_dtype": None, "device": None}


def cmd_run(args) -> dict:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # as config.py sets it, before torch loads
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from tools.p8_e1_m4 import weights_of
    from tools.p8_e1_score import _sha, check_provenance
    from tools.p8_e1data import corpus_root
    game = check_games([args.game])[0]
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(f"{WHO}: MPS is not available")
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    prov = check_provenance(game, game, 0, weights_of(game))       # the fold that left this game out
    r23_w = main_checkout() / "models" / "player_detector.pt"
    if _sha(r23_w) != meta["detector"]["weights_sha256"]:
        raise SystemExit(f"{WHO}: the production detector is not the one the corpus used")
    paths = {"r23": r23_w, "e1": weights_of(game)}
    if args.tag == "fix":       # report only, added after the numbers: ultralytics' own macOS 14 clip_boxes branch
        import ultralytics.utils.ops as ops
        ops.NOT_MACOS14 = False
    models = {m: YOLO(str(p)) for m, p in paths.items()}
    frames = [fr for ch in plan["chunks"] for w in ch for fr in w["frames"] if fr["game"] == game]
    if args.frames:
        frames = frames[:args.frames]
    preds, secs, caught, fp16, probe_error = {}, {m: 0.0 for m in models}, [], {}, None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for fr in frames:
            stem = f"{fr['window']}_f{int(fr['frame']):06d}"
            img = cv2.imread(str(root / "images" / game / f"{stem}.jpg"))
            if img is None:
                raise SystemExit(f"{WHO}: {stem}.jpg unreadable")
            rec = {}
            for m, model in models.items():
                t0 = time.time()
                try:
                    r = model.predict(img, conf=FLOOR, iou=NMS_IOU, imgsz=IMGSZ, device=args.device, verbose=False)[0]
                except NotImplementedError as e:          # only possible with the fallback off (the probe)
                    if args.tag != "probe":
                        raise
                    probe_error = {"model": m, "error": str(e).split(". If you want")[0]}
                    break
                secs[m] += time.time() - t0
                rec[m] = [[round(float(v), 2) for v in b.xyxy[0]] + [round(float(b.conf), 5)]
                          for b in r.boxes if r.names[int(b.cls)] == "player"]
                if m not in fp16:
                    fp16[m] = _param_state(model.predictor.model)
            if probe_error:
                break
            preds[stem] = rec
        caught = sorted({f"{w.category.__name__}: {str(w.message)[:300]}" for w in wlist})
    out = {"game": game, "device": args.device, "tag": args.tag, "frames": len(preds), "conf_floor": FLOOR,
           "nms_iou": NMS_IOU, "imgsz": IMGSZ, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
           "weights_sha256": {"e1": prov["sha256"], "r23": _sha(r23_w)}, "model_state": fp16, "warnings": caught,
           "probe_error": probe_error,
           "seconds_inference": {m: round(s, 2) for m, s in secs.items()}, "preds": preds}
    name = f"parity__{game}__{args.device}{'__' + args.tag if args.tag else ''}.json"
    atomic_write(out_dir() / name, json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "preds"}, indent=1))
    return out


# ── score (system python) ───────────────────────────────────────────────────────────────────────────────────────────
def cpu_preds(game: str) -> tuple:
    """M4's CPU outputs, used only if this tool's fresh CPU pilot reproduces them exactly -> (preds, source)."""
    m4 = json.loads((p8_root() / "e1" / f"m4_preds__{game}.json").read_text())
    pilot_p = out_dir() / f"parity__{game}__cpu__pilot.json"
    full_p = out_dir() / f"parity__{game}__cpu.json"
    if full_p.exists():
        return json.loads(full_p.read_text())["preds"], "fresh CPU run"
    if not pilot_p.exists():
        raise SystemExit(f"{WHO}: no CPU pilot for {game}")
    pilot = json.loads(pilot_p.read_text())["preds"]
    if not same_preds(pilot, m4["preds"]):
        raise SystemExit(f"{WHO}: the CPU pilot does not reproduce M4 on {game}; run CPU in full")
    return m4["preds"], "M4 (reproduced by a fresh CPU pilot)"


def cmd_score(args) -> dict:
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    from tools.p8_e1_postmortem import ORACLE_KEYS, oracle_boot, oracle_counts, oracle_rates, scorer_inside
    from tools.p8_e1data import corpus_root
    suffix = f"__{args.mps_tag}" if args.mps_tag else ""
    games = check_games(args.games.split(",") if args.games else [g for g in GAMES
                                                                   if (out_dir() / f"parity__{g}__mps{suffix}.json").exists()])
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    corpus = {}
    for k in range(len(plan["chunks"])):
        corpus.update(json.loads((root / "preds" / f"chunk_{k:02d}.json").read_text())["preds"])
    per_game, pool_acc, pool_w, pool_frames, runs = {}, {m: {} for m in MODELS}, {}, 0, {}
    for game in games:
        mps_run = json.loads((out_dir() / f"parity__{game}__mps{'__' + args.mps_tag if args.mps_tag else ''}.json").read_text())
        runs[game] = {k: v for k, v in mps_run.items() if k != "preds"}
        cpu, source = cpu_preds(game)
        runs[game]["cpu_source"] = source
        probe_p = out_dir() / f"parity__{game}__mps__probe.json"
        if probe_p.exists():
            pr = json.loads(probe_p.read_text())
            runs[game]["probe"] = {"PYTORCH_ENABLE_MPS_FALLBACK": pr["PYTORCH_ENABLE_MPS_FALLBACK"],
                                   "frames_run": pr["frames"], "error": pr["probe_error"]}
        g = plan["games"][game]
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        acc, per_w, n = {m: {} for m in MODELS}, {}, 0
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != game:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    if stem not in mps_run["preds"]:
                        raise SystemExit(f"{WHO}: {stem} has no MPS output")
                    n += 1
                    rec = corpus[f"{game}/{stem}"]
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    inside = scorer_inside(sv, fr["H_truth"], *rec["size"])
                    for m in MODELS:
                        merge(acc[m], frame_parity(cpu[stem][m], mps_run["preds"][stem][m]))
                        for dev, src in (("cpu", cpu[stem][m]), ("mps", mps_run["preds"][stem][m])):
                            kept = [b for b in src if b[4] >= CUT - 1e-9]
                            c = oracle_counts(T.feet_ft(kept, fr["H_truth"]), sv, inside)
                            d = per_w.setdefault(fr["window"], {}).setdefault(f"{m}_{dev}", {k: 0 for k in ORACLE_KEYS})
                            for k, v in c.items():
                                d[k] += v
        per_game[game] = {"frames": n, **{m: _model_block(acc[m], n, per_w, m, oracle_boot, oracle_rates, ORACLE_KEYS)
                                          for m in MODELS}}
        for m in MODELS:
            merge(pool_acc[m], acc[m])
        pool_w.update({f"{game}/{k}": v for k, v in per_w.items()})
        pool_frames += n
    pooled = {"games": games, "frames": pool_frames,
              **{m: _model_block(pool_acc[m], pool_frames, pool_w, m, oracle_boot, oracle_rates, ORACLE_KEYS)
                 for m in MODELS}}
    what = ("M7 device parity of raw detections, CPU against MPS, corpus JPEGs (docs/research/P8_E1.md); report only"
            if not args.mps_tag else
            "M7 diagnostic run (report only, added after the numbers): MPS with ultralytics' out-of-place clip branch "
            "forced in the measuring process (ops.NOT_MACOS14 = False), against CPU; not the pipeline's setting")
    out = {"what": what, "rule": {"mps_box_loss_under": LOSS_MAX, "missed_diff_under": MISSED_MAX, "cut": CUT},
           "runs": runs, "games": per_game, "pooled": pooled}
    out["mps_tag"] = args.mps_tag or None
    atomic_write(out_dir() / f"p8_device_parity{suffix}.json", json.dumps(out, indent=1, default=float))
    print(json.dumps({m: {k: pooled[m][k] for k in ("mps_box_loss", "mps_only_share", "cpu_boxes_per_frame",
                                                     "mps_boxes_per_frame", "conf_abs_diff_max", "conf_abs_diff_p99",
                                                     "missed", "passes")} for m in MODELS}, indent=1, default=float))
    return out


def _model_block(acc, frames, per_w, m, oracle_boot, oracle_rates, keys) -> dict:
    s = summarise(acc, frames)
    tot = {dev: {k: sum(w[f"{m}_{dev}"][k] for w in per_w.values()) for k in keys} for dev in ("cpu", "mps")}
    rates = {dev: oracle_rates(tot[dev]) for dev in ("cpu", "mps")}
    flat = {w: {"cpu": v[f"{m}_cpu"], "mps": v[f"{m}_mps"]} for w, v in per_w.items()}
    diff = oracle_boot(flat, "cpu", "mps")
    s["missed"] = {"cpu": rates["cpu"]["missed_rate"], "mps": rates["mps"]["missed_rate"], **diff["missed_rate"]}
    s["ghosts"] = {"cpu": rates["cpu"]["ghost_rate"], "mps": rates["mps"]["ghost_rate"], **diff["ghost_rate"]}
    s["passes"] = passes(s["mps_box_loss"], diff["missed_rate"]["diff"])
    return s


# ── report-only diagnostics (added after the numbers) ──────────────────────────────────────────────────────────────
def row_aligned(cpu: dict, mps: dict, m: str, first_rows: int) -> dict:
    """CPU and MPS outputs compared row by row (output order after the class filter) on frames whose rows carry the
    same confidences -> counts of x and y changes, the rows the x changes sit in, zero-width boxes at x = 720 and y2
    above 720 on each device."""
    st = {"frames_aligned": 0, "frames_unaligned": 0, "rows": 0, "x_changed": 0, "y_changed": 0,
          "mps_zero_width": 0, "mps_zero_width_at_720": 0, "mps_y2_over_720": 0, "cpu_y2_over_720": 0}
    rows = {}
    for k in mps:
        a, b = cpu[k][m], mps[k][m]
        if len(a) != len(b) or any(abs(x[4] - y[4]) > 2e-5 for x, y in zip(a, b)):
            st["frames_unaligned"] += 1
            continue
        st["frames_aligned"] += 1
        for i, (x, y) in enumerate(zip(a, b)):
            st["rows"] += 1
            if abs(x[0] - y[0]) > 0.01 or abs(x[2] - y[2]) > 0.01:
                st["x_changed"] += 1
                rows[i] = rows.get(i, 0) + 1
            if abs(x[1] - y[1]) > 0.01 or abs(x[3] - y[3]) > 0.01:
                st["y_changed"] += 1
            if y[2] - y[0] < 1:
                st["mps_zero_width"] += 1
                st["mps_zero_width_at_720"] += abs(y[0] - 720) < 0.01
            st["mps_y2_over_720"] += y[3] > 720.0
            st["cpu_y2_over_720"] += x[3] > 720.0
    st["x_changed_by_row"] = {str(i): v for i, v in sorted(rows.items())}
    st[f"x_changed_in_rows_0_to_{first_rows - 1}"] = sum(v for i, v in rows.items() if i < first_rows)
    return st


def fix_differences(cpu: dict, fix: dict, m: str, tol: float = 0.05) -> list:
    """Boxes of the forced-clip run that differ from CPU by more than tol px. Boxes are paired one to one within a
    frame (Hungarian on the largest coordinate difference, plus a large cost when confidences differ), so two boxes
    of equal confidence that swap output rows are not counted."""
    from scipy.optimize import linear_sum_assignment
    out = []
    for k in fix:
        a, b = cpu[k][m], fix[k][m]
        if len(a) != len(b):
            out.append({"frame": k, "note": "box count differs"})
            continue
        if not a:
            continue
        C = np.array([[max(abs(p - q) for p, q in zip(x[:4], y[:4])) + (1e6 if abs(x[4] - y[4]) > 2e-5 else 0.0)
                       for y in b] for x in a])
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > tol:
                out.append({"frame": k, "conf": b[j][4], "diff_px": [round(q - p, 2) for p, q in zip(a[i][:4], b[j][:4])]})
    return out


def collapse_counts(side: dict) -> dict:
    c = {"boxes": 0, "zero_width_at_720": 0, "y2_over_720": 0}
    for r in side["frames"]:
        for b in r["boxes"]:
            x1, y1, x2, y2 = b["bbox"]
            c["boxes"] += 1
            c["zero_width_at_720"] += (x2 - x1) < 1 and abs(x1 - 720) < 1
            c["y2_over_720"] += y2 > 720
    return c


def cmd_diag(args) -> dict:
    """Report-only block of the M7 record: row-aligned stats per game, the forced-clip run's differing rows, the
    s04_w00 numbers of the M5 and M6 paragraph, and the sidecar collapse counts of gsw_bkn_2015's builds."""
    d = out_dir()
    e1d = p8_root() / "e1"
    block = {"note": "report only, added after the numbers (ml-review of M7)", "row_aligned": {}}
    for g in check_games(GAMES):
        cpu = json.loads((e1d / f"m4_preds__{g}.json").read_text())["preds"]
        mps = json.loads((d / f"parity__{g}__mps.json").read_text())["preds"]
        block["row_aligned"][g] = {"r23": row_aligned(cpu, mps, "r23", 4), "e1": row_aligned(cpu, mps, "e1", 2)}
    cpu = json.loads((e1d / "m4_preds__gsw_bkn_2015.json").read_text())["preds"]
    fix = json.loads((d / "parity__gsw_bkn_2015__mps__fix.json").read_text())["preds"]
    block["forced_clip_rows_differing_over_0.05px"] = {m: {"rows": sum(len(fix[k][m]) for k in fix),
                                                          "differing": fix_differences(cpu, fix, m)} for m in MODELS}
    reg = json.loads((e1d / "gsw_bkn_2015__seed0__gsw_bkn_2015.json").read_text())["windows"]
    cpuw = json.loads((e1d / "m6__e1__gsw_bkn_2015.json").read_text())["windows"]
    w = "gsw_bkn_2015_s04_w00"
    net = lambda src, skip=(): sum(v["new"]["missed"] - v["r23"]["missed"] for k, v in src.items() if k not in skip)
    block["s04_w00"] = {"net_extra_misses_mps_game": net(reg), "net_extra_misses_mps_window": net(reg) - net(reg, {w}),
                        **{f"{dev}_{s}": {k: src[w][s][k] for k in ("missed", "in_frame", "nr_boxes")}
                           for dev, src in (("mps", reg), ("cpu", cpuw)) for s in ("r23", "new")}}
    builds = p8_root() / "builds"
    block["sidecar_collapse_gsw_bkn_2015"] = {}
    for b in ("r23", "e1_gsw_bkn_2015_seed0", "e1m6_r23_detcpu", "e1m6_e1_gsw_bkn_2015_detcpu"):
        acc = {}
        for p in sorted((builds / b).glob("gsw_bkn_2015_*_frames.json")):
            merge(acc, collapse_counts(json.loads(p.read_text())))
        block["sidecar_collapse_gsw_bkn_2015"][b] = acc
    rec_p = d / "p8_device_parity.json"
    rec = json.loads(rec_p.read_text())
    rec["report_only"] = block
    atomic_write(rec_p, json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in block.items() if k != "forced_clip_rows_differing_over_0.05px"}, indent=1))
    return block


def game_of_stem(stem: str) -> str:
    """gsw_sac_2015_s03_w01 -> gsw_sac_2015 (the tag before the _s<section>_w<window> suffix)."""
    import re
    m = re.match(r"^(.*)_s\d+_w\d+$", stem)
    return m.group(1) if m else stem


def cmd_sweep(args) -> dict:
    """M7b: the collapse signature in every sidecar folder of the P8 workspace builds (report only)."""
    import cv2
    builds = p8_root() / "builds"
    out = {"what": "M7b collapse signature per build folder and game (docs/research/P8_E1.md); report only",
           "folders": {}, "skipped": {}, "image_size": {}}
    ok_games, bad_games = set(), set()
    for folder in sorted(p for p in builds.iterdir() if p.is_dir()):
        per = {}
        for p in sorted(folder.glob("*_frames.json")):
            stem = p.name[:-len("_frames.json")]
            game = game_of_stem(stem)
            if game not in ok_games | bad_games:
                try:
                    guard([game], WHO)
                    ok_games.add(game)
                except SystemExit:
                    bad_games.add(game)
            if game in bad_games:
                out["skipped"].setdefault(folder.name, set()).add(game)
                continue
            acc = per.setdefault(game, {})
            merge(acc, collapse_counts(json.loads(p.read_text())))
            acc["windows"] = acc.get("windows", 0) + 1
            if game not in out["image_size"]:
                cap = cv2.VideoCapture(str(p8_root() / "label_build" / f"{stem}.mp4"))
                wh = (int(cap.get(3)), int(cap.get(4)))
                cap.release()
                out["image_size"][game] = list(wh) if wh != (0, 0) else "not readable"
        if per:
            out["folders"][folder.name] = per
    out["skipped"] = {k: sorted(v) for k, v in out["skipped"].items()}
    atomic_write(out_dir() / "p8_collapse_sweep.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_device_parity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--game", required=True, choices=GAMES)
    r.add_argument("--device", required=True, choices=("cpu", "mps"))
    r.add_argument("--frames", type=int, default=None)
    r.add_argument("--tag", default="", choices=("", "pilot", "probe", "fix"))
    s = sub.add_parser("score")
    s.add_argument("--games", default=None)
    s.add_argument("--mps-tag", dest="mps_tag", default="", choices=("", "fix"))
    sub.add_parser("diag")
    sub.add_parser("sweep")
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score, "diag": cmd_diag, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
