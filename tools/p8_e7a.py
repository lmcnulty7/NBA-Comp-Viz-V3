"""tools/p8_e7a.py: E7a, a report-only first measurement of the ball row with WASB's released basketball model (PLAN.md
D155 (2), D160; docs/research/P8_E7a.md, which registers every rule below before any run). Nothing trains or tunes.

    python3 -m tools.p8_e7a plan                       system python: guard, sample frames, SportVU ball truth
    python3 -m export.heavy --name "p8 e7a <game>" -- $V3_CV_PYTHON -m tools.p8_e7a run --game <tag>
    python3 -m tools.p8_e7a score                      system python: counts, F1, bootstraps -> reports/research/p8_e7a.json

Frames: the accepted camera-truth frames of each game (P8 workspace labels/<game>/*.jsonl), in time order within each
window clip, keeping a frame when it is at least SPACING_FRAMES clip frames after the last kept one.

Truth: the SportVU ball of the frame's moment (row q and moment, the scorecard's Moments order), mirrored like the
players, projected through the row's camera truth (camera_C, camera_rvec, camera_f; tools.p8_scorecard.project3d).

Model, as released (github.com/nttcom/WASB-SBDT at 923462ca, configs model/wasb.yaml, detector/tracknetv2.yaml,
tracker/online.yaml, dataloader/default.yaml): HRNet, 3 frames in and 3 out at 512x288, the frame warped by the
repository's affine (centre, scale max(h, w)), ImageNet normalisation, sigmoid heatmaps, connected-component blobs above
0.5 weighted by the heatmap, step 3, the online tracker (max displacement 300 px). The only departures: CPU instead of
CUDA, and the clip is a segment of SEGMENT_TRIPLETS consecutive triplets ending at the sampled frame, the tracker
refreshed per segment (the repository runs whole annotated clips); a sampled frame 0 or 1 of a clip, which no triplet
can end at, takes the clip's first triplet (frames 0 to 2) alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e7a"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = ("hou_cle_2016",)
SPACING_FRAMES = 120              # 2.0 s at the clips' 59.94 fps
SEGMENT_TRIPLETS = 4              # 12 consecutive frames ending at the sampled frame
Z_RANGE_FT = (-1.0, 30.0)         # a SportVU ball height outside this range is not a usable truth frame
Z_BANDS_FT = (4.0, 10.0)          # low below 4 ft, mid 4 to 10 ft, high from 10 ft (reported apart)
TOL_PX = 32.07                    # 2 x 16.033 px: the scorecard's ball tolerance rule on the 3 games (P8_E7a.md)
REPORT_TOLS_PX = (2.67, 10.0, 20.0)
BOOT_N, BOOT_SEED = 2000, 7
WASB_COMMIT = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
WEIGHTS_SHA256 = "8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948"


def work() -> Path:
    d = p8_root().parent / "e7a"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir() -> Path:
    return p8_root().parent / "models" / "wasb"


# ── sampling and truth (pure parts tested) ──────────────────────────────────────────────────────────────────────────
def sample_by_time(rows: list, spacing: int = SPACING_FRAMES) -> list:
    """Rows of one window -> the kept rows: in clip-frame order, each at least `spacing` frames after the last kept."""
    out, last = [], None
    for r in sorted(rows, key=lambda r: r["frame"]):
        if last is None or r["frame"] - last >= spacing:
            out.append(r)
            last = r["frame"]
    return out


def z_band(z: float) -> str:
    return "low" if z < Z_BANDS_FT[0] else "mid" if z < Z_BANDS_FT[1] else "high"


def count(frames: list, tol: float = TOL_PX) -> dict:
    """The scorecard's ball counting (tools/p8_scorecard.py score_window) over truth frames: TP when the truth ball is
    in the image and the prediction is within tol px; else FP when there is a prediction, FN when the ball is in the
    image."""
    tp = fp = fn = 0
    for f in frames:
        pr = f.get("pred")
        inb = f["in_image"]
        hit = inb and pr is not None and ((pr[0] - f["uv"][0]) ** 2 + (pr[1] - f["uv"][1]) ** 2) ** 0.5 <= tol
        if hit:
            tp += 1
        else:
            fp += pr is not None
            fn += inb
    return {"tp": tp, "fp": fp, "fn": fn}


def prf(c: dict) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1}


def boot(groups: list, stat, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Percentile 95% interval of stat over resamples (with replacement) of the groups; None when undefined."""
    import random
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stat([groups[rng.randrange(len(groups))] for _ in groups])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


# ── plan (system python) ────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan(args) -> dict:
    import numpy as np
    from sportvu.sync import MIRRORS, apply_mirror
    from tools import p8_frozen as F
    from tools.p8_common import guard
    from tools.p8_scorecard import Moments, holder, project3d
    tags = list(GAMES + APART)
    refuse_cv_test(tags, WHO)
    P = F.layout(p8_root())
    plan = {"games": {}, "rules": {"spacing_frames": SPACING_FRAMES, "segment_triplets": SEGMENT_TRIPLETS,
                                   "z_range_ft": Z_RANGE_FT, "tol_px": TOL_PX}}
    for tag in tags:
        g = F.game_of(tag)
        res = json.loads((P["labels"] / f"{tag}_result.json").read_text())
        guard([tag, g["sportvu"], g["video_id"]], WHO, homes=[res["home"]])
        refuse_cv_test([g["sportvu"], g["video_id"]], WHO)
        mirror = F.mirror_of(P, tag)
        fx, fy = MIRRORS[mirror]
        mom = Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
        frames, n_rows = [], 0
        for f in sorted(glob.glob(str(P["labels"] / tag / "*.jsonl"))):
            rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
            n_rows += len(rows)
            for r in sample_by_time(rows):
                rec = {"game": tag, "window": r["window"], "frame": int(r["frame"]), "q": r["q"], "moment": r["moment"]}
                ball = mom.ball[r["q"]][r["moment"]]
                if not np.isfinite(ball).all():
                    rec["truth"] = "ball not tracked"
                elif not (Z_RANGE_FT[0] <= float(ball[2]) <= Z_RANGE_FT[1]):
                    rec["truth"] = "ball height out of range"
                else:
                    bxy = apply_mirror(ball[None, :2].copy(), fx, fy)[0]
                    uv, depth = project3d(r["camera_C"], r["camera_rvec"], r["camera_f"], 1280, 720,
                                          [[bxy[0], bxy[1], float(ball[2])]])
                    sv = apply_mirror(mom.index.q[r["q"]]["xy"][r["moment"]], fx, fy)
                    rec.update(truth="ok", ball_ft=[round(float(bxy[0]), 3), round(float(bxy[1]), 3), round(float(ball[2]), 3)],
                               uv=[round(float(uv[0][0]), 2), round(float(uv[0][1]), 2)], depth=round(float(depth[0]), 3),
                               in_image=bool(depth[0] > 0 and 0 <= uv[0][0] < 1280 and 0 <= uv[0][1] < 720),
                               band=z_band(float(ball[2])),
                               held=holder(np.array([bxy[0], bxy[1], float(ball[2])]), sv) is not None)
                frames.append(rec)
        plan["games"][tag] = {"sportvu": g["sportvu"], "mirror": mirror, "accepted_rows": n_rows, "sampled": len(frames),
                              "frames": frames}
    atomic_write(work() / "plan.json", json.dumps(plan))
    print(json.dumps({t: {k: v for k, v in g.items() if k != "frames"} for t, g in plan["games"].items()}, indent=1))
    return plan


# ── run (CV python with torch, inside the lock) ─────────────────────────────────────────────────────────────────────
class _A(dict):
    def __getattr__(self, k):
        return self[k]


def _attr(d):
    return _A({k: (_attr(v) if isinstance(v, dict) else v) for k, v in d.items()})


def load_wasb():
    """The released model and post-processing, on CPU. Weights loaded with weights_only=True."""
    import hashlib
    import importlib.util
    import torch
    import yaml
    md = model_dir()
    w = md / "wasb_basketball_best.pth.tar"
    if hashlib.sha256(w.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise SystemExit(f"{WHO}: weights sha256 differs from the registered one")
    src = md / "code" / "src"
    sys.path.insert(0, str(src))
    from models import build_model
    from trackers.online import OnlineTracker
    from utils.image import get_affine_transform
    spec = importlib.util.spec_from_file_location("wasb_postprocessor", src / "detectors" / "postprocessor.py")
    pp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp_mod)
    y = lambda p: yaml.safe_load((src / "configs" / p).read_text())
    cfg = {"model": y("model/wasb.yaml"), "detector": y("detector/tracknetv2.yaml"), "tracker": y("tracker/online.yaml"),
           "dataloader": y("dataloader/default.yaml")}
    model = build_model({"model": _attr(cfg["model"])})
    ck = torch.load(w, map_location="cpu", weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()})
    model.eval()
    post = pp_mod.TracknetV2Postprocessor(cfg)
    return model, post, (lambda: OnlineTracker(cfg)), get_affine_transform, cfg


def run_segment(model, post, make_tracker, get_affine_transform, cfg, frames_bgr: list) -> list:
    """Frames (BGR, consecutive, length a multiple of 3) -> the tracker's output per frame, as released."""
    import cv2
    import numpy as np
    import torch
    m = cfg["model"]
    in_wh, out_wh = (m["inp_width"], m["inp_height"]), (m["out_width"], m["out_height"])
    h, w = frames_bgr[0].shape[:2]
    c, s = np.array([w / 2.0, h / 2.0], np.float32), max(h, w) * 1.0
    t_in = get_affine_transform(c, s, 0, list(in_wh))
    t_out_inv = get_affine_transform(c, s, 0, list(out_wh), inv=1)
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    def prep(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.warpAffine(rgb, t_in, in_wh, flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return torch.from_numpy(((x - mean) / std).transpose(2, 0, 1))
    trips = [torch.cat([prep(f) for f in frames_bgr[i:i + 3]], 0) for i in range(0, len(frames_bgr), 3)]
    with torch.no_grad():
        preds = model(torch.stack(trips))
    aff = {sc: torch.from_numpy(np.stack([t_out_inv] * len(trips)).astype(np.float32)) for sc in preds}
    res = post.run(preds, aff)
    tracker = make_tracker()
    out = []
    for b in range(len(trips)):
        for e in range(3):
            dets = [{"xy": xy, "score": sc} for xy, sc in zip(res[b][e][0]["xys"], res[b][e][0]["scores"])]
            r = tracker.update(dets)
            out.append({"x": float(r["x"]), "y": float(r["y"]), "visi": bool(r["visi"]), "score": float(r["score"])})
    return out


def cmd_run(args) -> dict:
    import cv2
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    refuse_cv_test([args.game], WHO)
    plan = json.loads((work() / "plan.json").read_text())
    g = plan["games"][args.game]
    out_p = work() / f"preds_{args.game}.json"
    if out_p.exists() and not args.force:
        print(f"{WHO}: {args.game} already run")
        return {}
    t0 = time.time()
    model, post, make_tracker, gat, cfg = load_wasb()
    lb = p8_root() / "label_build"
    need = 3 * SEGMENT_TRIPLETS
    by_w = {}
    for f in g["frames"]:
        by_w.setdefault(f["window"], []).append(f)
    preds, secs_model, n_dec = {}, 0.0, 0
    for win, fs in sorted(by_w.items()):
        refuse_cv_test([win], WHO)
        targets = sorted({f["frame"] for f in fs})
        cap = cv2.VideoCapture(str(lb / f"{win}.mp4"))
        buf, pos, ti = [], 0, 0
        while ti < len(targets):
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"{WHO}: {win} ends at frame {pos}, before {targets[ti]}")
            n_dec += 1
            if img.shape[1] != 1280 or img.shape[0] != 720:
                raise SystemExit(f"{WHO}: {win} is {img.shape[1]}x{img.shape[0]}, not 1280x720")
            buf.append(img)
            buf = buf[-need:]
            while ti < len(targets) and pos >= 2 and targets[ti] <= pos:
                if targets[ti] == pos:
                    seg, k = buf[len(buf) % 3:], -1        # whole triplets ending at the sampled frame
                else:                                      # frames 0 and 1: the clip's first triplet holds them
                    seg, k = buf[:3], targets[ti]
                t1 = time.time()
                r = run_segment(model, post, make_tracker, gat, cfg, seg)
                secs_model += time.time() - t1
                preds[f"{win}/{targets[ti]}"] = dict(r[k], segment_frames=len(seg))
                ti += 1
            pos += 1
        cap.release()
    out = {"game": args.game, "frames": len(preds), "decoded": n_dec, "seconds_model": round(secs_model, 1),
           "seconds_total": round(time.time() - t0, 1), "wasb_commit": WASB_COMMIT, "weights_sha256": WEIGHTS_SHA256,
           "device": "cpu", "preds": preds}
    atomic_write(out_p, json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "preds"}, indent=1))
    return out


# ── score (system python) ───────────────────────────────────────────────────────────────────────────────────────────
def cmd_score(args) -> dict:
    plan = json.loads((work() / "plan.json").read_text())
    res = {"registered_in": "docs/research/P8_E7a.md", "tol_px": TOL_PX, "report_tols_px": REPORT_TOLS_PX,
           "games": {}, "date": dt.date.today().isoformat()}
    per_game_frames = {}
    for tag in GAMES + APART:
        pp = work() / f"preds_{tag}.json"
        if not pp.exists():
            res["games"][tag] = {"status": "not run"}
            continue
        run = json.loads(pp.read_text())
        frames = []
        for f in plan["games"][tag]["frames"]:
            p = run["preds"].get(f"{f['window']}/{f['frame']}")
            if f["truth"] != "ok":
                continue
            frames.append(dict(f, pred=[p["x"], p["y"]] if p and p["visi"] else None))
        per_game_frames[tag] = frames
        sampled = plan["games"][tag]["sampled"]
        c = count(frames)
        g = {"sampled": sampled, "truth_frames": len(frames),
             "excluded": {k: sum(1 for f in plan["games"][tag]["frames"] if f["truth"] == k)
                          for k in ("ball not tracked", "ball height out of range")},
             "in_image": sum(f["in_image"] for f in frames), "in_image_share": round(sum(f["in_image"] for f in frames) / max(1, len(frames)), 4),
             "counts": c, **{k: (round(v, 4) if v is not None else None) for k, v in prf(c).items()},
             "frame_bootstrap": {k: boot(frames, lambda fr, k=k: prf(count(fr))[k]) for k in ("f1", "precision", "recall")},
             "by_band": {b: {**count([f for f in frames if f.get("band") == b]),
                             "recall": prf(count([f for f in frames if f.get("band") == b]))["recall"]}
                         for b in ("low", "mid", "high")},
             "held": {h: {**count([f for f in frames if f.get("held") == h]),
                          "recall": prf(count([f for f in frames if f.get("held") == h]))["recall"]} for h in (True, False)},
             "report_tols": {str(t): {k: (round(v, 4) if v is not None else None) for k, v in prf(count(frames, t)).items()}
                             for t in REPORT_TOLS_PX},
             "seconds_model": run["seconds_model"], "seconds_total": run["seconds_total"]}
        res["games"][tag] = g
    dev = [per_game_frames[t] for t in GAMES if t in per_game_frames]
    if len(dev) == len(GAMES):
        pooled = [f for fr in dev for f in fr]
        c = count(pooled)
        res["pooled_3"] = {"counts": c, **{k: (round(v, 4) if v is not None else None) for k, v in prf(c).items()},
                           "in_image_share": round(sum(f["in_image"] for f in pooled) / max(1, len(pooled)), 4),
                           "game_bootstrap": {k: boot(dev, lambda gs, k=k: prf(count([f for g_ in gs for f in g_]))[k])
                                              for k in ("f1", "precision", "recall")},
                           "report_tols": {str(t): {k: (round(v, 4) if v is not None else None) for k, v in prf(count(pooled, t)).items()}
                                           for t in REPORT_TOLS_PX}}
    res["mac_hours_model_total"] = round(sum(g.get("seconds_total", 0) for g in res["games"].values()) / 3600, 3)
    atomic_write(work() / "result.json", json.dumps(res, indent=1))
    atomic_write(REPO / "reports" / "research" / "p8_e7a.json", json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--game", required=True, choices=GAMES + APART)
    r.add_argument("--force", action="store_true")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
