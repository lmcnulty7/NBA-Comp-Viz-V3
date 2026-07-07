#!/usr/bin/env python
"""
evaluate_teams.py — human-label eval for Component C1 (team classification).

Turns the label-free checks (contact sheet, 5v5 balance, silhouette) into a real
accuracy number. Three modes, run in order:

  1) --export   Run tracking + crop collection on a clip window (no gate/court
                needed), fit the team clusterer, and sample ~N crops stratified
                across predicted team A / team B / abstained. Writes:
                  data/teams_eval/<clip>/crop_0000.png …   numbered crops
                  data/teams_eval/<clip>/index.csv         crop_id → track_id, prediction
                  data/teams_eval/<clip>/sheet.png         indexed contact sheet
                  data/teams_eval/<clip>/meta.json         cluster → jersey color (light/dark)
  2) --label    Keypress labeler, one crop at a time (predictions are NOT shown —
                same truth-vs-predicted separation the gate labeling enforced):
                  w = white/light kit   n = navy/dark kit   a = no team / can't tell
                  u = undo   s = skip   q/Esc = quit (resume anytime)
                Your labels go to truth.csv, separate from index.csv.
  3) --report   Join truth ↔ predictions: cluster→color mapping (majority),
                full confusion matrix (incl. abstentions), per-team precision/
                recall, crop- AND track-level accuracy, abstention analysis.
                → reports/teams_eval_<clip>.{json,txt}

Metric caveat (baked into the report): predictions are TRACK-level (pooled
color) while you label each CROP's visible jersey — occlusion crops (an
opponent's body filling the box) count against the metric even when the track
assignment is right. So crop-level accuracy is a LOWER BOUND; the track-level
number (majority of your crop labels per track) is the fairer read.

Referee note: with the basketball-trained detector, referees are class 1 and the
tracker only tracks class 0 (player), so refs never reach team classification at
all. teams.py itself has no ref logic — a non-player that leaks through
detection either abstains (its pooled color sits between the kits) or lands in
the nearer cluster. Label those crops 'a' and the report counts them.

Examples
────────
  python evaluate_teams.py --export --start 11520 --max-frames 200
  python evaluate_teams.py --label
  python evaluate_teams.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("evaluate_teams")

PRED_NAMES = {0: "A", 1: "B", None: "abstain"}
HUMAN_KEYS = {"w": "white", "n": "navy", "a": "none"}


def eval_dir(source: Path) -> Path:
    return config.TEAMS_EVAL_DIR / source.stem


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── 1) export ─────────────────────────────────────────────────────────────────
def do_export(args) -> None:
    from detect import PlayerTracker
    from detect.reid import TorsoCropCollector
    from detect.teams import TeamClassifier

    config.set_seed()
    rng = np.random.default_rng(config.SEED)
    out = eval_dir(args.source)
    out.mkdir(parents=True, exist_ok=True)

    tracker = PlayerTracker(device=config.get_device())
    collector = TorsoCropCollector()
    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx, n = args.start, 0
    log.info("Collecting crops from %s (start %d, %d frames, stride %d) …",
             args.source.name, args.start, args.max_frames, args.stride)
    while idx < total and n < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        tracks = tracker.update(frame, idx, idx / fps)
        collector.add(frame, tracks)
        idx += args.stride
        n += 1
    cap.release()

    clf = TeamClassifier()
    if not clf.fit(collector.crops):
        raise SystemExit("Not enough crops to cluster — widen the window.")
    team_of = {tid: t for tid, (t, _) in clf.assign(collector.crops).items()}
    cols = clf.team_colors_bgr(collector.crops, team_of)

    # which cluster is the light kit? (helps the labeler map w/n back to A/B)
    def lightness(bgr):  # perceived luma
        b, g, r = bgr
        return 0.114 * b + 0.587 * g + 0.299 * r
    light = max(cols, key=lambda t: lightness(cols[t])) if len(cols) == 2 else None
    meta = {"source": str(args.source), "start": args.start, "stride": args.stride,
            "max_frames": args.max_frames, "silhouette": clf.silhouette,
            "cluster_colors_bgr": {PRED_NAMES[t]: list(c) for t, c in cols.items()},
            "light_cluster": PRED_NAMES.get(light)}

    # stratified sample: 40% A, 40% B, 20% abstained (adjusted to availability),
    # capped per track so one long track can't dominate its group
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)   # name -> [(tid, crop_i)]
    for tid, lst in collector.crops.items():
        name = PRED_NAMES[team_of.get(tid)]
        groups[name].extend((tid, i) for i in range(len(lst)))
    want = {"A": int(args.n * 0.4), "B": int(args.n * 0.4), "abstain": args.n - 2 * int(args.n * 0.4)}
    sampled: list[tuple[str, int, int]] = []   # (pred_name, tid, crop_i)
    for name, pairs in groups.items():
        k = min(want.get(name, 0), len(pairs))
        n_tracks = len({t for t, _ in pairs})
        cap_per_track = max(3, int(np.ceil(k / max(n_tracks, 1))))
        order = rng.permutation(len(pairs))
        taken: Counter = Counter()
        for j in order:
            tid, i = pairs[j]
            if taken[tid] >= cap_per_track or sum(taken.values()) >= k:
                continue
            taken[tid] += 1
            sampled.append((name, tid, i))
    rng.shuffle(sampled)   # labeling order must not leak the grouping

    rows, tiles = [], defaultdict(list)
    for cid, (name, tid, i) in enumerate(sampled):
        crop = collector.crops[tid][i]
        fname = f"crop_{cid:04d}.png"
        cv2.imwrite(str(out / fname), crop[:, :, ::-1])   # RGB → BGR
        rows.append({"crop_id": cid, "file": fname, "track_id": tid,
                     "pred": name, "n_track_crops": len(collector.crops[tid])})
        tile = cv2.resize(crop[:, :, ::-1], (64, 96))
        cv2.putText(tile, str(cid), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles[name].append(tile)
    write_csv_rows(out / "index.csv", rows, ["crop_id", "file", "track_id", "pred", "n_track_crops"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # indexed contact sheet (review-at-a-glance alternative to the labeler)
    sheet_rows, cols_per_row = [], 14
    for name in ("A", "B", "abstain"):
        if not tiles[name]:
            continue
        header = np.zeros((22, 64 * cols_per_row, 3), np.uint8)
        cv2.putText(header, f"predicted {name} ({len(tiles[name])} crops)", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        sheet_rows.append(header)
        ts = tiles[name][:]
        while len(ts) % cols_per_row:
            ts.append(np.zeros((96, 64, 3), np.uint8))
        for i in range(0, len(ts), cols_per_row):
            sheet_rows.append(np.hstack(ts[i:i + cols_per_row]))
    cv2.imwrite(str(out / "sheet.png"), np.vstack(sheet_rows))

    by_group = Counter(r["pred"] for r in rows)
    log.info("Exported %d crops → %s  (%s; silhouette %.3f, light kit = cluster %s)",
             len(rows), out, dict(by_group), clf.silhouette, meta["light_cluster"])
    log.info("Next: python evaluate_teams.py --label")


# ── 2) label ──────────────────────────────────────────────────────────────────
def do_label(args) -> None:
    out = eval_dir(args.source)
    index = read_csv_rows(out / "index.csv")
    if not index:
        raise SystemExit(f"No export at {out} — run --export first.")
    truth_path = out / "truth.csv"
    truth = {r["crop_id"]: r["label"] for r in read_csv_rows(truth_path)}
    pending = [r for r in index if r["crop_id"] not in truth]
    log.info("%d crops, %d already labeled, %d to go", len(index), len(truth), len(pending))

    win = "team crop labeler"
    cv2.namedWindow(win)
    history: list[str] = []
    i = 0
    while 0 <= i < len(pending):
        r = pending[i]
        img = cv2.imread(str(out / r["file"]))
        scale = 420 / img.shape[0]
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        canvas = cv2.copyMakeBorder(big, 56, 0, 40, 40, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        done = len(truth)
        cv2.putText(canvas, f"{done}/{len(index)} labeled   crop {r['crop_id']}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 255, 90), 1)
        cv2.putText(canvas, "w=white/light  n=navy/dark  a=no team/can't tell  u=undo  s=skip  q=quit",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and history:
            truth.pop(history.pop(), None)
            i = max(0, i - 1)
        elif key == ord("s"):
            i += 1
        elif chr(key) in HUMAN_KEYS:
            truth[r["crop_id"]] = HUMAN_KEYS[chr(key)]
            history.append(r["crop_id"])
            i += 1
        # persist on every change so quitting is always safe
        write_csv_rows(truth_path, [{"crop_id": k, "label": v} for k, v in truth.items()],
                       ["crop_id", "label"])
    cv2.destroyAllWindows()
    log.info("%d/%d labeled → %s", len(truth), len(index), truth_path)
    if len(truth) == len(index):
        log.info("All done. Next: python evaluate_teams.py --report")


# ── 3) report ─────────────────────────────────────────────────────────────────
def do_report(args) -> None:
    out = eval_dir(args.source)
    index = {r["crop_id"]: r for r in read_csv_rows(out / "index.csv")}
    truth = {r["crop_id"]: r["label"] for r in read_csv_rows(out / "truth.csv")}
    joined = [(index[c]["pred"], truth[c], index[c]["track_id"]) for c in truth if c in index]
    if len(joined) < 20:
        raise SystemExit(f"Only {len(joined)} labeled crops — label more first (--label).")

    # cluster → color mapping by majority of your labels (clusters are arbitrary ids)
    maj = {}
    for cl in ("A", "B"):
        votes = Counter(h for p, h, _ in joined if p == cl and h in ("white", "navy"))
        maj[cl] = votes.most_common(1)[0][0] if votes else "?"
    if maj["A"] == maj["B"]:
        log.warning("Both clusters map to '%s' by majority — clustering likely failed on this clip.", maj["A"])

    # confusion: predicted (A, B, abstain) × human (white, navy, none)
    conf = {p: Counter() for p in ("A", "B", "abstain")}
    for p, h, _ in joined:
        conf[p][h] += 1

    # crop-level per-team metrics on identifiable crops (human ∈ {white, navy})
    def prf(cl):
        col = maj[cl]
        tp = conf[cl][col]
        fp = sum(conf[cl][h] for h in ("white", "navy") if h != col)
        fn = sum(conf[o][col] for o in ("A", "B") if o != cl)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"maps_to": col, "precision": round(prec, 3), "recall_vs_other_cluster": round(rec, 3),
                "n_pred": sum(conf[cl].values())}

    ident = [(p, h) for p, h, _ in joined if p in ("A", "B") and h in ("white", "navy")]
    crop_acc = sum(1 for p, h in ident if maj[p] == h) / max(len(ident), 1)

    # track-level: majority of your crop labels per track vs the track's prediction
    by_track = defaultdict(list)
    for p, h, tid in joined:
        by_track[tid].append((p, h))
    track_rows, track_correct = [], 0
    for tid, lst in by_track.items():
        votes = Counter(h for _, h in lst if h in ("white", "navy"))
        if not votes or len(lst) < 3:
            continue   # too few labels to call the track
        human_team = votes.most_common(1)[0][0]
        pred = lst[0][0]
        ok = pred in ("A", "B") and maj[pred] == human_team
        track_correct += ok
        track_rows.append({"track_id": tid, "pred": pred, "human_majority": human_team,
                           "n_labeled": len(lst), "correct": bool(ok)})
    track_acc = track_correct / max(len(track_rows), 1)

    # abstention analysis: what did the clusterer refuse to call?
    abst = conf["abstain"]
    n_abst = sum(abst.values())

    report = {
        "n_labeled": len(joined),
        "cluster_mapping": maj,
        "confusion_pred_x_human": {p: dict(conf[p]) for p in conf},
        "crop_level": {"accuracy_identifiable": round(crop_acc, 3), "n_identifiable": len(ident),
                       "per_cluster": {cl: prf(cl) for cl in ("A", "B")},
                       "note": "lower bound — occlusion crops count against the track's team"},
        "track_level": {"accuracy": round(track_acc, 3), "n_tracks": len(track_rows),
                        "tracks": track_rows},
        "abstained": {"n": n_abst, "human_says": dict(abst),
                      "identifiable_but_abstained_pct":
                          round(100 * (abst["white"] + abst["navy"]) / max(n_abst, 1), 1)},
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    jpath = config.REPORTS_DIR / f"teams_eval_{args.source.stem}.json"
    jpath.write_text(json.dumps(report, indent=2))

    lines = [
        f"TEAM CLASSIFICATION EVAL — {args.source.stem} ({len(joined)} labeled crops)",
        f"cluster mapping (by your labels): A → {maj['A']}, B → {maj['B']}",
        "",
        "confusion (rows = predicted, cols = your label):",
        f"{'':>10}{'white':>8}{'navy':>8}{'none':>8}",
    ]
    for p in ("A", "B", "abstain"):
        lines.append(f"{p:>10}{conf[p]['white']:>8}{conf[p]['navy']:>8}{conf[p]['none']:>8}")
    lines += [
        "",
        f"crop-level accuracy (identifiable crops only): {crop_acc:.1%} on {len(ident)}"
        "   [lower bound — occlusion crops penalized]",
        f"track-level accuracy (majority vote, >=3 labels): {track_acc:.1%} on {len(track_rows)} tracks",
        f"abstained crops you could identify: {report['abstained']['identifiable_but_abstained_pct']}%"
        f" of {n_abst} (missed coverage, not errors)",
    ]
    txt = "\n".join(lines)
    (config.REPORTS_DIR / f"teams_eval_{args.source.stem}.txt").write_text(txt + "\n")
    print(txt)
    log.info("report → %s", jpath)


def main() -> None:
    ap = argparse.ArgumentParser(description="Human-label eval for team classification.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--label", action="store_true")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--source", type=Path, default=config.VIDEO_DIR / "curry_q1_clip.mp4")
    ap.add_argument("--start", type=int, default=11520)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--n", type=int, default=200, help="crops to sample (--export)")
    args = ap.parse_args()
    if args.export:
        do_export(args)
    elif args.label:
        do_label(args)
    else:
        do_report(args)


if __name__ == "__main__":
    main()
