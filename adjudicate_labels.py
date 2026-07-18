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
 keep: true/false. keep=false for: spectators/fans, coaches and staff in street
   clothes, photographers, broadcasters, ball kids — these are NOT players or
   referees. Also keep=false for DUPLICATES: when several boxes cover the same
   object, keep=true ONLY for the tightest box, false for every other copy.
 cls: player|referee|rim|backboard|scorebug|ball
   player = anyone in a team uniform or team warmups, INCLUDING bench players.
   referee = game officials only. Street clothes = keep=false, never player.
 tight: true/false (box edges within ~5% of the object's true extent)
 occlusion: visible|partial|heavy (players only)
 team_kit: light|dark|other (players only — uniform lightness, not team identity)
 on_court: players only. true ONLY if the player is ON the playing floor as part
   of live play or a free-throw formation. Bench players, players at the scorer's
   table, and anyone outside the sidelines/baseline: false.
 number: jersey number 0-99 if clearly readable, else null
Also return shot_type for the WHOLE frame: wide_broadcast|closeup|replay|graphic|split_screen.
Reply with ONLY JSON: {"shot_type": ..., "boxes": {"<idx>": {...}, ...}}"""

# court path: geometry proposes, the VLM only NAMES. Pilot 1 showed Claude
# placing coordinates tens of px off and hallucinating out-of-frame features —
# so precise candidate points come from classical Hough line intersections
# (no learned model → no inbreeding) and Claude's job shrinks to semantics.
COURT_NAMES = [
    "baseline_sideline_near_left", "baseline_sideline_far_left",
    "baseline_sideline_near_right", "baseline_sideline_far_right",
    "paint_corner_near_left", "paint_corner_far_left",
    "paint_corner_near_right", "paint_corner_far_right",
    "paint_baseline_near_left", "paint_baseline_far_left",
    "paint_baseline_near_right", "paint_baseline_far_right",
    "free_throw_line_left_end", "free_throw_line_right_end",
    "halfcourt_sideline_near", "halfcourt_sideline_far",
    "corner3_baseline_near_left", "corner3_baseline_far_left",
    "corner3_baseline_near_right", "corner3_baseline_far_right",
]
COURT_SYSTEM = f"""The image is an NBA broadcast frame with numbered red markers placed at
DETECTED line intersections. For each marker index, say which court landmark the marker
sits exactly on, or "none" if it is not precisely at a named court-line intersection
(floor logos, paint texture edges, shadows, players, ads = "none").
Names (the only allowed values besides "none"): {", ".join(COURT_NAMES)}.
Be conservative: a wrong name poisons a homography; "none" is always safe.
Reply with ONLY JSON: {{"points": {{"<idx>": "<name-or-none>", ...}}}}"""


def line_intersections(segs, w, h, min_angle_deg=15.0, extend=0.35):
    """Pairwise intersections of line segments (as x1,y1,x2,y2), kept when the
    crossing point lies within each segment's span extended by `extend` of its
    length (court lines are partially occluded) and the lines cross at a real
    angle (near-parallel intersections are numerically garbage)."""
    import math
    pts = []
    for i in range(len(segs)):
        x1, y1, x2, y2 = segs[i]
        for j in range(i + 1, len(segs)):
            x3, y3, x4, y4 = segs[j]
            d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
            if abs(d) < 1e-9:
                continue
            a1 = math.atan2(y2 - y1, x2 - x1)
            a2 = math.atan2(y4 - y3, x4 - x3)
            ang = abs(a1 - a2) % math.pi
            if min(ang, math.pi - ang) < math.radians(min_angle_deg):
                continue
            t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
            u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
            if not (-extend <= t <= 1 + extend and -extend <= u <= 1 + extend):
                continue
            px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            if 0 <= px < w and 0 <= py < h:
                pts.append((px, py))
    return pts


def cluster_points(pts, radius=12.0):
    """Greedy merge; returns cluster centers sorted by support (biggest first) —
    real court intersections attract many segment pairs, noise attracts few."""
    clusters: list[list] = []
    for x, y in pts:
        for c in clusters:
            cx, cy = c[0] / c[2], c[1] / c[2]
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                c[0] += x; c[1] += y; c[2] += 1
                break
        else:
            clusters.append([x, y, 1])
    clusters.sort(key=lambda c: -c[2])
    return [(round(c[0] / c[2], 1), round(c[1] / c[2], 1)) for c in clusters]


def court_candidates(img, max_pts=20):
    """Classical intersection proposals: Canny + probabilistic Hough on the
    lower 2/3 of the frame (court region), long segments only."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    y0 = int(h * 0.30)
    gray = cv2.cvtColor(img[y0:], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90,
                            minLineLength=int(w * 0.12), maxLineGap=14)
    if lines is None:
        return []
    pts = line_intersections([l[0] for l in lines], w, h - y0)
    return [(x, y + y0) for x, y in cluster_points(pts)[:max_pts]]


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
                    cands = court_candidates(img)
                    if len(cands) >= 4:
                        marked = img.copy()
                        for ci, (cx, cy) in enumerate(cands):
                            cv2.drawMarker(marked, (int(cx), int(cy)), (0, 0, 255),
                                           cv2.MARKER_CROSS, 22, 2)
                            cv2.putText(marked, str(ci), (int(cx) + 6, int(cy) - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        lm = call_claude(client, COURT_MODEL, COURT_SYSTEM,
                                         b64_image(marked),
                                         f"{len(cands)} markers (indices 0..{len(cands) - 1}).",
                                         costs)
                        named = [{"x": cands[int(i)][0], "y": cands[int(i)][1], "name": n}
                                 for i, n in (lm or {}).get("points", {}).items()
                                 if n != "none" and str(i).isdigit() and int(i) < len(cands)]
                        if named:
                            ch.write(json.dumps({"tag": tag, "frame": rec["frame"],
                                                 "landmarks": named}) + "\n")
                if (k + 1) % 25 == 0:
                    log.info("  %s %d/%d (t+%.0fs)", tag, k + 1, len(recs), time.time() - t0)
        log.info("%s adjudicated", tag)
        (ROOT / "autolabels" / "adjudication_costs.json").write_text(
            json.dumps({m: c for m, c in costs.items()}, indent=1))
    log.info("token usage: %s", dict(costs))


if __name__ == "__main__":
    main()
