#!/usr/bin/env python
"""
adjudicate_labels.py — Foundation Refresh pass 2: Claude judges the proposals.

Runs LOCALLY (reads corpus + proposals via the Drive mount; needs
ANTHROPIC_API_KEY). The VLM's role is JUDGMENT, not geometry (LABEL_SCHEMA.md):

  per frame   one vision call: the frame rendered with numbered candidate
              boxes → per-index verdicts {keep, cls, tight, occlusion,
              team_kit, on_court, number} + frame shot_type. Teacher–pipeline
              AGREEMENTS are pre-accepted and only sampled for audit;
              adjudication focuses on the disagreement band (cost triage).
  court       on wide_broadcast frames: a second call proposing named court
              landmarks in pixel coords → stored for the RANSAC-fit +
              template-projection step (geometry sets precision, not the VLM).

Outputs (Drive, nba_harvest/autolabels/):
  adjudicated/<tag>.jsonl    verdicts per frame
  court_landmarks/<tag>.jsonl
  adjudication_costs.json    running token/cost tally

Pilot first: `--limit 5 --tags gsw_sac_klay37` (a few cents) — check verdict
quality before spending the ~$20-40 full pass. Default model: Haiku for box
verdicts, Sonnet for court landmarks.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("adjudicate")

ROOT = (Path.home() / "Library/CloudStorage"
        / "GoogleDrive-lucienmmcnulty@gmail.com/My Drive/nba_harvest")
BOX_MODEL = "claude-haiku-4-5-20251001"
COURT_MODEL = "claude-sonnet-5"
AGREE_IOU = 0.5

BOX_SYSTEM = """You are auditing auto-generated labels on an NBA broadcast frame.
The image shows numbered candidate boxes. For EACH index return a JSON object:
 keep: true/false (is there really an object of this kind here)
 cls: player|referee|rim|backboard|scorebug|ball (correct class, may differ from proposal)
 tight: true/false (box edges within ~5% of the object's true extent)
 occlusion: visible|partial|heavy (players only)
 team_kit: light|dark|other (players only — jersey lightness, not team identity)
 on_court: true/false (players only — standing on the playing floor vs bench/sideline)
 number: jersey number 0-99 if clearly readable, else null
Also return shot_type for the WHOLE frame: wide_broadcast|closeup|replay|graphic|split_screen.
Reply with ONLY JSON: {"shot_type": ..., "boxes": {"<idx>": {...}, ...}}"""

COURT_SYSTEM = """This is a wide NBA broadcast frame. Identify visible court landmarks and
return pixel coordinates as JSON: {"landmarks": [{"name": ..., "x": ..., "y": ...}]}.
Use only these names: baseline_sideline_near_left, baseline_sideline_far_left,
baseline_sideline_near_right, baseline_sideline_far_right, paint_corner_near_left,
paint_corner_far_left, paint_corner_near_right, paint_corner_far_right,
free_throw_center_left, free_throw_center_right, center_circle_top,
center_circle_bottom, halfcourt_sideline_near, halfcourt_sideline_far,
corner3_baseline_near_left, corner3_baseline_far_left, corner3_baseline_near_right,
corner3_baseline_far_right. Only include landmarks whose exact line intersection is
clearly visible; precision matters more than count. ONLY JSON."""


def b64_image(img, max_w=1092):
    import cv2
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.standard_b64encode(enc.tobytes()).decode()


def render_candidates(img, cands):
    import cv2
    out = img.copy()
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = [int(v) for v in c["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, str(i), (x1 + 2, max(14, y1 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return out


def call_claude(client, model, system, img_b64, user_text, costs) -> dict | None:
    msg = client.messages.create(
        model=model, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": img_b64}},
            {"type": "text", "text": user_text}]}])
    costs[model]["in"] += msg.usage.input_tokens
    costs[model]["out"] += msg.usage.output_tokens
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("unparseable verdict (skipped): %.120s", text)
        return None


def disagreement_band(rec) -> tuple[list, list]:
    """(pre_accepted, needs_adjudication): teacher∩pipeline agreements are
    pre-accepted; teacher-only + pipeline-only + all non-overlap classes go to
    Claude. Disagreements are ALWAYS adjudicated, never dropped (schema rule 2)."""
    from colab_autolabel import iou
    pre, adj = [], []
    pipe = rec.get("pipeline", [])
    for t in rec.get("teacher", []):
        best = max((iou(t["box"], p["box"]) for p in pipe if p["cls"] == t["cls"]),
                   default=0.0)
        (pre if best >= AGREE_IOU and t["cls"] in ("player", "referee") else adj).append(t)
    matched = [p for p in pipe if any(iou(p["box"], t["box"]) >= AGREE_IOU
                                      and t["cls"] == p["cls"] for t in rec.get("teacher", []))]
    adj += [dict(p, src="pipeline_only") for p in pipe if p not in matched]
    return pre, adj


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude adjudication of teacher proposals.")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="frames per tag (0 = all)")
    ap.add_argument("--court-every", type=int, default=4,
                    help="court-landmark call on every Nth wide frame")
    args = ap.parse_args()

    import cv2
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY (console.anthropic.com → API keys)")
    client = anthropic.Anthropic()

    prop_dir = ROOT / "autolabels" / "proposals"
    adj_dir = ROOT / "autolabels" / "adjudicated"
    court_dir = ROOT / "autolabels" / "court_landmarks"
    adj_dir.mkdir(parents=True, exist_ok=True)
    court_dir.mkdir(parents=True, exist_ok=True)
    costs = defaultdict(lambda: {"in": 0, "out": 0})

    tags = args.tags or sorted(p.stem for p in prop_dir.glob("*.jsonl"))
    t0 = time.time()
    for tag in tags:
        recs = [json.loads(l) for l in (prop_dir / f"{tag}.jsonl").read_text().splitlines()]
        if args.limit:
            recs = recs[:args.limit]
        outp = adj_dir / f"{tag}.jsonl"
        done = {json.loads(l)["frame"] for l in outp.read_text().splitlines()} \
            if outp.exists() else set()
        with open(outp, "a") as fh, open(court_dir / f"{tag}.jsonl", "a") as ch:
            for k, rec in enumerate(recs):
                if rec["frame"] in done:
                    continue
                img = cv2.imread(str(ROOT / "label_corpus" / tag / f"f{rec['frame']:07d}.jpg"))
                if img is None:
                    continue
                pre, adj = disagreement_band(rec)
                verdict = call_claude(
                    client, BOX_MODEL, BOX_SYSTEM,
                    b64_image(render_candidates(img, adj)),
                    f"{len(adj)} candidate boxes (indices 0..{len(adj) - 1}).", costs)
                if verdict is None:
                    continue
                out = {"tag": tag, "frame": rec["frame"],
                       "pre_accepted": pre, "adjudicated": adj,
                       "verdicts": verdict.get("boxes", {}),
                       "shot_type": verdict.get("shot_type")}
                fh.write(json.dumps(out) + "\n")
                if verdict.get("shot_type") == "wide_broadcast" and k % args.court_every == 0:
                    lm = call_claude(client, COURT_MODEL, COURT_SYSTEM,
                                     b64_image(img), "Locate the court landmarks.", costs)
                    if lm:
                        ch.write(json.dumps({"tag": tag, "frame": rec["frame"],
                                             **lm}) + "\n")
                if (k + 1) % 25 == 0:
                    log.info("  %s %d/%d (t+%.0fs)", tag, k + 1, len(recs), time.time() - t0)
        log.info("%s adjudicated", tag)
        (ROOT / "autolabels" / "adjudication_costs.json").write_text(
            json.dumps({m: c for m, c in costs.items()}, indent=1))
    log.info("token usage: %s", dict(costs))


if __name__ == "__main__":
    main()
