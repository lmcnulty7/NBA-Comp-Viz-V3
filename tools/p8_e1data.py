"""tools/p8_e1data.py: the E1 training corpus, built on the Mac (PLAN.md D155 (1); D131 sources 1 and 2; D150 (4)).
Nothing trains here.

    python3 -m tools.p8_e1data plan                                   guard, list every accepted camera-truth frame
    python3 -m export.heavy --name ... -- <teacher venv python> -m tools.p8_e1data run --chunk K
    python3 -m tools.p8_e1data build                                  labels, manifest, meta, counts

Corpus (outside the repository, in the P8 workspace, default <home>/Developer/nba-v3-worktrees/_p8-data/e1_corpus or
V3_P8_E1): images/<game>/<window>_f<frame>.jpg, labels/<game>/<same stem>.txt (YOLO format, class 0 player),
preds/chunk_<K>.json (raw detections of both models), manifest.json (every frame, its boxes by source), meta.json
(games, teacher, weights, thresholds, code commit, dataset versions; D150 condition 4). No data.yaml: the splits are
per fold (tools/p8_e1_colab.py).

Frames: every accepted camera-truth frame of the 3 R1.3 games with Mac truth (gsw_bkn_2015, gsw_cha_2016,
gsw_sac_2015), in time order; chosen by the camera-truth acceptance rule only, never by a model. gsw_ind_2016 (D98)
and hou_cle_2016 (the new-arena check) stay out. Each window clip is decoded from frame 0 without seeking (tools.p8_e3
found a seek landing short in these clips).

Rules (tools/p8_teachers.py and docs/research/P8_step0_trackB.md section 3, unchanged):
- Our detector: player boxes at conf 0.40 (config.PLAYER_CONF), NMS 0.45, CPU. Teacher: YOLOX-X SportsMOT at 0.1471
  (its matched-precision threshold), CPU.
- Source 1: our player box and a teacher box matched one to one, greedily in our confidence order, at IoU >= 0.5;
  the label is the mean box (tools.p8_teachers.agree_boxes).
- Source 2: a teacher box whose foot (bottom centre, through the frame's camera truth) a Hungarian match puts within
  2.5 ft (CONFIRM_FT) of a SportVU player in the image that our detector's feet do not reach (the misses rule of
  tools.p8_teachers misses-score); the label is the teacher box.
- A source 2 box with IoU >= 0.5 to a source 1 box is counted in both and written once (the source 1 box). Every other
  box is dropped (D131: disagreements are dropped).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e1data"
GAMES = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
OUT_OF_CORPUS = ("gsw_ind_2016", "hou_cle_2016")
CHUNK_FRAMES = 600
JPEG_QUALITY = 95


def corpus_root() -> Path:
    root = Path(os.environ.get("V3_P8_E1") or Path.home() / "Developer" / "nba-v3-worktrees" / "_p8-data" / "e1_corpus")
    return check_outside(root)


def chunks_of(windows: list, per: int = CHUNK_FRAMES) -> list:
    """[(window, [frames])] in time order -> chunks of whole windows with about `per` frames each."""
    out, cur, n = [], [], 0
    for w, frs in windows:
        cur.append((w, frs))
        n += len(frs)
        if n >= per:
            out.append(cur)
            cur, n = [], 0
    if cur:
        out.append(cur)
    return out


def merge_sources(s1: list, s2: list, iou_fn, thr: float = 0.5) -> tuple:
    """Source 1 boxes, then each source 2 box unless it overlaps a source 1 box at IoU >= thr -> (labels, number of
    source 2 boxes also in source 1)."""
    labels, both = [list(b[:4]) for b in s1], 0
    for b in s2:
        if any(iou_fn(b, a) >= thr for a in s1):
            both += 1
        else:
            labels.append(list(b[:4]))
    return labels, both


def yolo_line(box, w: int, h: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    return "0 %.6f %.6f %.6f %.6f" % ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


# ── plan (system python) ────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan(args) -> dict:
    from tools.p8_common import guard, r13_games
    refuse_cv_test(list(GAMES), WHO)
    games = r13_games()
    wr = p8_root()
    plan = {"games": {}, "chunks": []}
    windows = []
    for tag in GAMES:
        res = json.loads((wr / "labels" / f"{tag}_result.json").read_text())
        guard([tag, res["sportvu"], games[tag]["video_id"]], WHO, homes=[res["home"]])
        rows = []
        for f in sorted((wr / "labels" / tag).glob("*.jsonl")):
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: (r["section"], r["section_frame"]))
        by_w = {}
        for r in rows:
            by_w.setdefault(r["window"], []).append({k: r[k] for k in ("game", "window", "frame", "q", "moment", "H_truth")})
        plan["games"][tag] = {"sportvu": res["sportvu"], "mirror": res["mirror"], "video_id": games[tag]["video_id"],
                              "accepted_frames": len(rows), "windows": len(by_w)}
        windows += list(by_w.items())
    plan["chunks"] = [[{"window": w, "frames": frs} for w, frs in c] for c in chunks_of(windows)]
    root = corpus_root()
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "plan.json", json.dumps(plan))
    print(json.dumps({"games": plan["games"], "chunks": len(plan["chunks"]),
                      "frames": sum(len(w["frames"]) for c in plan["chunks"] for w in c)}, indent=1))
    return plan


# ── run (teacher venv, inside the lock) ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> dict:
    import cv2
    from tools import p8_teachers as T
    if os.environ.get("V3_HEAVY_HELD") is None:
        raise SystemExit(f"{WHO}: run inside the heavy-job lock")
    root = corpus_root()
    plan = json.loads((root / "plan.json").read_text())
    chunk = plan["chunks"][args.chunk]
    refuse_cv_test(sorted({w["window"] for w in chunk}), WHO)
    out_p = root / "preds" / f"chunk_{args.chunk:02d}.json"
    if out_p.exists() and not args.force:
        print(f"{WHO}: chunk {args.chunk} already done")
        return {}
    t0 = time.time()
    ws = T.workspace()
    models = {m: T.load_runner(m, ws, "cpu") for m in (T.OURS, T.MISS_TEACHER)}
    t_load = time.time() - t0
    preds, secs = {}, {m: 0.0 for m in models}
    lb = p8_root() / "label_build"
    for w in chunk:
        game = w["frames"][0]["game"]
        (root / "images" / game).mkdir(parents=True, exist_ok=True)
        want = sorted(int(f["frame"]) for f in w["frames"])
        cap = cv2.VideoCapture(str(lb / f"{w['window']}.mp4"))
        pos = 0
        for f in want:
            while pos < f:
                if not cap.grab():
                    raise SystemExit(f"{WHO}: {w['window']} ends at frame {pos}, before {f}")
                pos += 1
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"{WHO}: {w['window']} frame {f} unreadable")
            pos += 1
            stem = f"{w['window']}_f{f:06d}"
            cv2.imwrite(str(root / "images" / game / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            rec = {"size": [int(img.shape[1]), int(img.shape[0])]}
            for m, (detect, _) in models.items():
                t1 = time.time()
                rec[m] = [[round(v, 2) for v in p[:4]] + [round(p[4], 5), p[5]] for p in detect(img)]
                secs[m] += time.time() - t1
            preds[f"{game}/{stem}"] = rec
        cap.release()
    out = {"chunk": args.chunk, "frames": len(preds), "device": "cpu", "seconds_load": round(t_load, 1),
           "seconds_inference": {m: round(s, 1) for m, s in secs.items()},
           "versions": {m: v for m, (_, v) in models.items()}, "preds": preds}
    (root / "preds").mkdir(exist_ok=True)
    atomic_write(out_p, json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "preds"}, indent=1))
    return out


# ── build (system python) ───────────────────────────────────────────────────────────────────────────────────────────
def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_build(args) -> dict:
    import numpy as np
    from sportvu.sync import MIRRORS, SportVUIndex, apply_mirror
    from tools import p8_teachers as T
    refuse_cv_test(list(GAMES), WHO)
    root, wr = corpus_root(), p8_root()
    plan = json.loads((root / "plan.json").read_text())
    preds, secs, versions = {}, {}, {}
    for k in range(len(plan["chunks"])):
        p = root / "preds" / f"chunk_{k:02d}.json"
        if not p.exists():
            raise SystemExit(f"{WHO}: chunk {k} has not run")
        c = json.loads(p.read_text())
        preds.update(c["preds"])
        for m, s in c["seconds_inference"].items():
            secs[m] = secs.get(m, 0.0) + s
        versions = c["versions"]
    counts, manifest = {}, {}
    for tag, g in plan["games"].items():
        index = SportVUIndex(json.loads((wr / "sportvu" / f"{g['sportvu']}_moments.json").read_text())["moments"])
        fx, fy = MIRRORS[g["mirror"]]
        c = {"frames": 0, "ours_boxes": 0, "teacher_boxes": 0, "players_in_frame": 0, "ours_missed": 0,
             "source1": 0, "source2": 0, "source2_also_source1": 0, "labels": 0}
        (root / "labels" / tag).mkdir(parents=True, exist_ok=True)
        for ch in plan["chunks"]:
            for w in ch:
                for fr in w["frames"]:
                    if fr["game"] != tag:
                        continue
                    stem = f"{fr['window']}_f{int(fr['frame']):06d}"
                    rec = preds[f"{tag}/{stem}"]
                    wpx, hpx = rec["size"]
                    d = [p for p in rec[T.OURS] if p[5] == "player"]
                    t = [p for p in rec[T.MISS_TEACHER] if p[4] >= T.MISS_TEACHER_CONF]
                    s1 = T.agree_boxes(d, t)
                    sv = apply_mirror(index.q[fr["q"]]["xy"][fr["moment"]], fx, fy)
                    tg = T.targets_in_frame(sv, fr["H_truth"], wpx, hpx)
                    hit_d = T.found_targets(T.feet_ft(d, fr["H_truth"]), tg)
                    s2 = _teacher_at_missed(t, tg, hit_d, fr["H_truth"], T)
                    labels, both = merge_sources(s1, s2, T.iou)
                    (root / "labels" / tag / f"{stem}.txt").write_text("".join(yolo_line(b, wpx, hpx) + "\n" for b in labels))
                    manifest[f"{tag}/{stem}"] = {"source1": len(s1), "source2": len(s2), "labels": len(labels)}
                    c["frames"] += 1
                    c["ours_boxes"] += len(d)
                    c["teacher_boxes"] += len(t)
                    c["players_in_frame"] += len(tg)
                    c["ours_missed"] += len(tg) - len(hit_d)
                    c["source1"] += len(s1)
                    c["source2"] += len(s2)
                    c["source2_also_source1"] += both
                    c["labels"] += len(labels)
        c["source1_per_frame"] = round(c["source1"] / c["frames"], 4) if c["frames"] else None
        c["source2_per_frame"] = round(c["source2"] / c["frames"], 4) if c["frames"] else None
        counts[tag] = c
    ws = T.workspace()
    meta = {"rule": "PLAN.md D155 (1), D131 sources 1 and 2, D150 condition 4", "games": list(GAMES),
            "out_of_corpus": {"gsw_ind_2016": "D98", "hou_cle_2016": "new-arena check (D155)"},
            "frames": "every accepted Mac camera-truth frame, time order (labels/<game>/*.jsonl, P8 workspace)",
            "teacher": {"name": T.MISS_TEACHER, "what": T.TEACHERS[T.MISS_TEACHER]["what"],
                        "license": T.TEACHERS[T.MISS_TEACHER]["license"],
                        "weights": T.TEACHERS[T.MISS_TEACHER]["weights"],
                        "weights_sha256": T.sha256(ws / "weights" / T.TEACHERS[T.MISS_TEACHER]["weights"])},
            "detector": {"weights": "models/player_detector.pt (main checkout)", "weights_sha256": T.sha256(T.detector_weights()),
                         "conf": T.DET_CONF, "nms_iou": T.DET_NMS_IOU, "classes": ["player"]},
            "thresholds": {"teacher_conf": T.MISS_TEACHER_CONF, "agreement_iou": T.IOU, "confirm_ft": T.CONFIRM_FT},
            "device": "cpu", "library_versions": versions, "code_commit": git_head(),
            "datasets": {tag: {"sportvu": g["sportvu"], "video_id": g["video_id"], "mirror": g["mirror"],
                               "camera_truth": f"labels/{tag}_result.json (Mac, P8 workspace)",
                               "camera_truth_sha256": T.sha256(wr / "labels" / f"{tag}_result.json")}
                         for tag, g in plan["games"].items()},
            "label_classes": {"0": "player"},
            "cpu_seconds": {m: round(s, 1) for m, s in secs.items()}}
    atomic_write(root / "meta.json", json.dumps(meta, indent=1))
    atomic_write(root / "manifest.json", json.dumps({"counts": counts, "frames": manifest}))
    out = {"counts": counts, "cpu_hours": {m: round(s / 3600, 3) for m, s in secs.items()},
           "disk_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}
    print(json.dumps(out, indent=1))
    return out


def _teacher_at_missed(t: list, targets, hit_d: set, H_truth, T) -> list:
    """Teacher boxes whose feet a Hungarian match puts within CONFIRM_FT of an in-frame SportVU player our detector
    missed (tools.p8_teachers.found_targets, the misses rule), one box per player."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not t or len(targets) == 0:
        return []
    feet = T.feet_ft(t, H_truth)
    D = np.sqrt(((np.asarray(feet)[:, None, :] - np.asarray(targets)[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(D)
    return [t[i] for i, j in zip(ri, ci) if D[i, j] <= T.CONFIRM_FT and int(j) not in hit_d]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tools.p8_e1data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--chunk", type=int, required=True)
    r.add_argument("--force", action="store_true")
    sub.add_parser("build")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
