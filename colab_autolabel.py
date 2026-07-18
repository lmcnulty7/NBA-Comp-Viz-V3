#!/usr/bin/env python
"""
colab_autolabel.py — Foundation Refresh pass 1: external teacher over the corpus.

Runs on Colab GPU (bootstrap: colab_autolabel.ipynb). See LABEL_SCHEMA.md for
the schema and the anti-inbreeding rules this implements.

Per corpus frame:
  teacher   Grounding DINO (open-vocab) at a LOW threshold — deliberately
            over-proposes so the in-house detector's false-negative class
            (scrum-occluded players) actually surfaces for adjudication.
            Classes: player / referee / rim / backboard / scorebug / ball.
  pose      YOLO person-pose keypoints (ankles → future foot-point fix),
            associated to teacher player boxes by IoU.
  pipeline  the CURRENT in-house detector, run alongside for the
            agree / teacher-only / pipeline-only diff. Comparison data,
            NEVER labels.

Outputs (Drive, nba_harvest/autolabels/):
  proposals/<tag>.jsonl     one record per frame (teacher + pose + pipeline)
  comparison_report.json    per-class agree/only counts per game
  qualification.json        (--qualify) teacher vs pipeline vs HUMAN GT on
                            data/tracking/box_truth — the gate that decides
                            whether the teacher deserves trust at all.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("autolabel")

# text prompts → schema classes (Grounding DINO is open-vocabulary)
PROMPTS = {
    "basketball player": "player",
    "referee": "referee",
    "basketball hoop rim": "rim",
    "basketball backboard": "backboard",
    "score bug graphic overlay": "scorebug",
    "basketball": "ball",
}
TEACHER_THRESHOLD = 0.18      # LOW by design: over-propose, adjudicate later
PIPELINE_THRESHOLD = 0.40     # the pipeline's own operating point
AGREE_IOU = 0.5


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_sets(teacher: list, pipeline: list, thr: float = AGREE_IOU) -> dict:
    """Greedy one-to-one matching by IoU (same class assumed pre-filtered).
    Returns counts: agree / teacher_only / pipeline_only."""
    used = set()
    agree = 0
    for t in sorted(teacher, key=lambda x: -x.get("score", 0)):
        best, best_i = 0.0, None
        for i, p in enumerate(pipeline):
            if i in used:
                continue
            v = iou(t["box"], p["box"])
            if v > best:
                best, best_i = v, i
        if best >= thr:
            used.add(best_i)
            agree += 1
    return {"agree": agree, "teacher_only": len(teacher) - agree,
            "pipeline_only": len(pipeline) - len(used)}


def pr_vs_gt(preds: list[list], gts: list[list], thr: float = 0.5) -> dict:
    """Precision/recall of box lists vs GT boxes (greedy, same as the repo's
    detection eval convention)."""
    used = set()
    tp = 0
    for p in preds:
        best, best_i = 0.0, None
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_i = v, i
        if best >= thr:
            used.add(best_i)
            tp += 1
    prec = tp / len(preds) if preds else 0.0
    rec = tp / len(gts) if gts else 0.0
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


# ── heavy model runners (Colab; imports deferred so tests stay light) ─────────
def load_teacher(device):
    import torch  # noqa: F401
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device)
    return proc, model


def map_prompt_label(label: str) -> str:
    """Grounding DINO returns matched prompt text (possibly a fragment) —
    map it back to a schema class by containment either way."""
    s = str(label).lower().strip()
    for prompt, cls in PROMPTS.items():
        if s == prompt or s in prompt or prompt in s:
            return cls
    # fragments like "basketball" alone: prefer the most specific word match
    for prompt, cls in PROMPTS.items():
        if any(w in prompt.split() for w in s.split()):
            return cls
    return s


def teacher_detect(proc, model, device, image_bgr) -> list[dict]:
    import torch
    from PIL import Image
    img = Image.fromarray(image_bgr[:, :, ::-1])
    text = ". ".join(PROMPTS) + "."
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # transformers renamed box_threshold→threshold across versions — accept both
    try:
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=TEACHER_THRESHOLD,
            text_threshold=0.15, target_sizes=[img.size[::-1]])[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=TEACHER_THRESHOLD,
            text_threshold=0.15, target_sizes=[img.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = []
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        dets.append({"cls": map_prompt_label(label),
                     "box": [round(float(v), 1) for v in box],
                     "score": round(float(score), 3)})
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description="External-teacher auto-label pass (Colab).")
    ap.add_argument("--qualify", action="store_true",
                    help="teacher + pipeline vs HUMAN box GT only; no corpus pass")
    ap.add_argument("--limit", type=int, default=0, help="frames per game (0 = all)")
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    on_colab = os.path.isdir("/content")
    root = Path("/content/drive/MyDrive/nba_harvest") if on_colab else (
        Path.home() / "Library/CloudStorage/GoogleDrive-lucienmmcnulty@gmail.com"
        / "My Drive/nba_harvest")
    corpus = root / "label_corpus"
    out_dir = root / "autolabels"
    (out_dir / "proposals").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not os.environ.get("AUTOLABEL_CPU_OK"):
        raise SystemExit("NO GPU: this pass is ~9 h on CPU vs ~1.5 h on T4. "
                         "Runtime → Change runtime type → T4 GPU, then rerun. "
                         "(Set AUTOLABEL_CPU_OK=1 to force CPU.)")
    log.info("device=%s corpus=%s", device, corpus)

    proc, teacher = load_teacher(device)
    pipeline = YOLO("models/player_detector.pt")
    pose = YOLO("yolov8x-pose.pt" if device == "cuda" else "yolov8n-pose.pt")

    def pipeline_detect(img):
        r = pipeline.predict(img, conf=PIPELINE_THRESHOLD, verbose=False, device=device)[0]
        names = r.names
        return [{"cls": names[int(b.cls)], "box": [round(float(v), 1) for v in b.xyxy[0]],
                 "score": round(float(b.conf), 3)} for b in r.boxes]

    def pose_detect(img):
        r = pose.predict(img, conf=0.3, verbose=False, device=device)[0]
        if r.keypoints is None:
            return []
        out = []
        for kp, box in zip(r.keypoints.data, r.boxes.xyxy):
            out.append({"box": [round(float(v), 1) for v in box],
                        "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                               for x, y, c in kp]})
        return out

    if args.qualify:
        import config
        gt_files = sorted(config.TRACK_BOX_TRUTH.glob("*.json"))

        # GT images are gitignored Mac-local files; on Colab they only exist
        # inside the labels backup zip on Drive — index it by basename
        zip_index, zf = {}, None
        backups = sorted((root / "labels_backup").glob("labels_*.zip"))
        if backups:
            import zipfile
            zf = zipfile.ZipFile(backups[-1])
            zip_index = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}

        def load_gt_image(rec):
            import numpy as np
            if os.path.exists(rec["image_path"]):
                return cv2.imread(rec["image_path"])
            name = Path(rec["image_path"]).name
            if name in zip_index:
                buf = np.frombuffer(zf.read(zip_index[name]), np.uint8)
                return cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return None

        agg_t, agg_p, agg_u = defaultdict(int), defaultdict(int), defaultdict(int)
        n_eval = 0
        preds_out = open(out_dir / "qualification_preds.jsonl", "w")
        for gf in gt_files:
            rec = json.loads(gf.read_text())
            img = load_gt_image(rec)
            if img is None:
                continue
            n_eval += 1
            gts = rec["boxes"]
            t_boxes = [d["box"] for d in teacher_detect(proc, teacher, device, img)
                       if d["cls"] == "player"]
            p_boxes = [d["box"] for d in pipeline_detect(img) if d["cls"] == "player"]
            # union: pipeline boxes (higher precision) + teacher boxes that
            # don't duplicate one — the candidate-set recall adjudication
            # would work from. This is the number that decides whether the
            # teacher adds ANY player the pipeline can't see.
            u_boxes = p_boxes + [t for t in t_boxes
                                 if max((iou(t, p) for p in p_boxes), default=0) < 0.5]
            for agg, boxes in ((agg_t, t_boxes), (agg_p, p_boxes), (agg_u, u_boxes)):
                m = pr_vs_gt(boxes, gts)
                for k in ("tp", "fp", "fn"):
                    agg[k] += m[k]
            preds_out.write(json.dumps({"frame": rec["frame"], "gt": gts,
                                        "teacher": t_boxes, "pipeline": p_boxes}) + "\n")
        preds_out.close()
        if n_eval == 0:
            raise SystemExit(
                "qualification evaluated ZERO frames — GT images not found "
                "locally or in the Drive labels_backup zip. Refusing to print "
                "a 0/0/0 table that looks like a result (run-14 lesson).")
        def finish(a):
            p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0
            r = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0
            return {**a, "precision": round(p, 3), "recall": round(r, 3),
                    "f1": round(2 * p * r / (p + r), 3) if p + r else 0}
        result = {"frames": n_eval, "iou": 0.5,
                  "teacher_grounding_dino": finish(agg_t),
                  "pipeline_current": finish(agg_p),
                  "union_candidates": finish(agg_u),
                  "note": "player class vs human box_truth. Teacher labels are "
                          "trusted only if teacher wins (LABEL_SCHEMA rule 1). "
                          "union_candidates = pipeline + non-duplicate teacher "
                          "boxes: the recall ceiling adjudication works from — "
                          "if union recall ≈ pipeline recall, the teacher adds "
                          "no players and a stronger teacher is needed."}
        (out_dir / "qualification.json").write_text(json.dumps(result, indent=1))
        log.info("QUALIFICATION: %s", json.dumps(result, indent=1))
        return

    manifest = [json.loads(l) for l in (corpus / "manifest.jsonl").read_text().splitlines()]
    by_tag = defaultdict(list)
    for m in manifest:
        by_tag[m["tag"]].append(m)
    report = {}
    t0 = time.time()
    for tag, recs in sorted(by_tag.items()):
        if args.limit:
            recs = recs[:args.limit]
        outp = out_dir / "proposals" / f"{tag}.jsonl"
        done = {json.loads(l)["frame"] for l in outp.read_text().splitlines()} \
            if outp.exists() else set()
        counts = defaultdict(lambda: defaultdict(int))
        with open(outp, "a") as fh:
            for i, m in enumerate(recs):
                if m["frame"] in done:
                    continue
                img = cv2.imread(str(corpus / tag / f"f{m['frame']:07d}.jpg"))
                if img is None:
                    continue
                t_dets = teacher_detect(proc, teacher, device, img)
                p_dets = pipeline_detect(img)
                poses = pose_detect(img)
                for cls in {d["cls"] for d in t_dets} | {d["cls"] for d in p_dets}:
                    mm = match_sets([d for d in t_dets if d["cls"] == cls],
                                    [d for d in p_dets if d["cls"] == cls])
                    for k, v in mm.items():
                        counts[cls][k] += v
                fh.write(json.dumps({"tag": tag, "frame": m["frame"],
                                     "teacher": t_dets, "pipeline": p_dets,
                                     "pose": poses}) + "\n")
                if (i + 1) % 50 == 0:
                    log.info("  %s %d/%d (t+%.0fs)", tag, i + 1, len(recs), time.time() - t0)
        report[tag] = {c: dict(v) for c, v in counts.items()}
        log.info("%s done: %s", tag, report[tag].get("player"))
    (out_dir / "comparison_report.json").write_text(json.dumps(report, indent=1))
    log.info("proposals + comparison_report → %s (%.0f min)", out_dir, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
