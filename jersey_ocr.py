#!/usr/bin/env python
"""
jersey_ocr.py — fragment → player-name mapping via jersey numbers (prototype).

THE gate for every player-level deliverable (named defender profiles,
player-level credit buckets, real comparisons). The design leans on three
things this pipeline already proved out:

  • torso crops: the re-ID collector's inner-upper-bbox crops contain the
    jersey number (chest or upper back) when it's visible at all;
  • pooled voting with abstention (the teams-classifier playbook): most crops
    are unreadable — blur, profile views, occlusion — but a fragment has up to
    dozens of crops and only needs a few decisive reads. No consensus ⇒ no name.
  • external ground truth for free: basketball-reference season rosters give
    the LEGAL number → player mapping per team, so OCR only reads digits, never
    identifies people. Illegal numbers are discarded outright, which kills most
    misreads before voting (a random misread rarely lands on a legal number).

OCR lessons inherited from the clock reader: NO digits allowlist (it mangles
wordmarks like "GOLDEN STATE" into digits) — read free-form and keep only
tokens that are already pure 1–2 digit strings, weighted by easyocr confidence.

Validation layers (all reported):
  roster legality (pre-vote) · decisive-majority abstention · simultaneity
  conflicts (two temporally-overlapping fragments named the same player ⇒ the
  weaker is demoted to abstain) · review montage for the human eyeball pass.

Modes
  --probe   Self-contained feasibility run on a clip window: track + collect
            crops (every frame — OCR wants lucky sharp frames), team-classify,
            OCR + vote, print the naming table + read-rate stats, write
            reports/viz/jersey_probe_<clip>.png.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from fetch_pbp import GAMES, game_for_clip, light_team_name, video_path
from videoseq import SeqReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("jersey_ocr")

ROSTER_DIR = config.PROJECT_ROOT / "data" / "rosters"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# voting thresholds (config-worthy once past prototype; printed with results)
READ_MIN_CONF = 0.30    # easyocr confidence floor for a read to vote
NAME_MIN_WEIGHT = 0.8   # summed confidence on the winning number
NAME_MIN_SHARE = 0.65   # winning number's share of the fragment's vote weight
DIGITS_RE = re.compile(r"^\d{1,2}$")



def ocr_crop_region(frame, bbox):
    """OCR-specific crop: FULL bbox width, upper 65% — back numbers span wider
    than the re-ID torso crop keeps, and easyocr's detector finds the digits
    within a larger context on its own."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    y2c = y1 + int(0.65 * (y2 - y1))
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2c = max(0, y1), min(H, y2c)
    if x2 - x1 < 20 or y2c - y1 < 20:
        return None
    return frame[y1:y2c, x1:x2].copy()


def sharpness(img) -> float:
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def enhance(img):
    """5x upscale + CLAHE on luma — digits at 720p are 10-20 px tall."""
    big = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(127, 127, 127))


def season_for_date(date: str) -> int:
    """NBA season label = end year (2015-16 → 2016)."""
    y, m = int(date[:4]), int(date[5:7])
    return y + 1 if m >= 8 else y


def fetch_roster(team: str, season: int) -> dict:
    """{number(str): [player names]} from basketball-reference, cached."""
    ROSTER_DIR.mkdir(parents=True, exist_ok=True)
    cache = ROSTER_DIR / f"{team}_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    table = BeautifulSoup(r.text, "html.parser").find("table", id="roster")
    roster: dict[str, list[str]] = defaultdict(list)
    for tr in table.find_all("tr"):
        num_td = tr.find(attrs={"data-stat": "number"})
        ply_td = tr.find(attrs={"data-stat": "player"})
        if not num_td or not ply_td or not ply_td.get_text(strip=True):
            continue
        name = ply_td.get_text(" ", strip=True)
        for n in re.split(r"[,\s]+", num_td.get_text(strip=True)):
            if n.isdigit():
                roster[str(int(n))].append(name)
    cache.write_text(json.dumps(roster, indent=1))
    log.info("roster %s %d: %d numbers → %s", team, season, len(roster), cache.name)
    return dict(roster)


def ocr_crops(crops: dict, reader) -> dict:
    """{tid: [(number_str, conf), ...]} — free-form OCR, digit-token filter."""
    reads = defaultdict(list)
    n_crops = sum(len(v) for v in crops.values())
    done = 0
    for tid, lst in crops.items():
        for c in lst:
            big = cv2.resize(c[:, :, ::-1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            for _, txt, conf in reader.readtext(big, detail=1):
                t = txt.strip().replace(" ", "")
                if DIGITS_RE.match(t) and conf >= READ_MIN_CONF:
                    reads[tid].append((str(int(t)), float(conf)))
            done += 1
            if done % 200 == 0:
                log.info("  OCR %d/%d crops …", done, n_crops)
    return dict(reads)


def vote(reads: list, legal: set) -> dict:
    """Weighted vote over roster-legal reads → name decision or abstention."""
    w = defaultdict(float)
    for num, conf in reads:
        if num in legal:
            w[num] += conf
    if not w:
        return {"number": None, "reason": "no_legal_reads", "n_reads": len(reads)}
    top, top_w = max(w.items(), key=lambda kv: kv[1])
    share = top_w / sum(w.values())
    if top_w < NAME_MIN_WEIGHT:
        return {"number": None, "reason": f"weak_evidence({top}:{top_w:.2f})", "n_reads": len(reads)}
    if share < NAME_MIN_SHARE:
        return {"number": None, "reason": f"contested({dict((k, round(v,2)) for k,v in w.items())})",
                "n_reads": len(reads)}
    return {"number": top, "weight": round(top_w, 2), "share": round(share, 2),
            "n_reads": len(reads)}


def probe(args) -> None:
    from detect import PlayerTracker
    from detect.reid import TorsoCropCollector
    from detect.teams import TeamClassifier
    import easyocr

    config.set_seed()
    code = game_for_clip(args.clip)
    game = GAMES[code]
    season = season_for_date(game["date"])
    rosters = {side: fetch_roster(game[side], season) for side in ("home", "away")}

    # ── collect crops at full cadence (OCR wants lucky sharp frames) ──────────
    tracker = PlayerTracker(device=config.get_device())
    collector = TorsoCropCollector(every=1, max_per_track=args.max_crops)
    ocr_pool = defaultdict(list)          # tid -> [(sharpness, bbox-crop BGR)]
    src = Path(args.video) if args.video else video_path(args.clip)
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    reader = SeqReader(cap)
    first_seen, last_seen = {}, {}
    idx, n = args.start, 0
    log.info("collecting crops: %s (%s) frames %d.. (%d live frames)",
             args.clip, src.name, args.start, args.max_frames)
    while n < args.max_frames:
        ret, frame = reader.read(idx)
        if not ret:
            break
        tracks = tracker.update(frame, idx, idx / fps)
        collector.add(frame, tracks)
        for t in tracks:
            first_seen.setdefault(t.track_id, idx)
            last_seen[t.track_id] = idx
            c = ocr_crop_region(frame, t.bbox)
            if c is not None:
                ocr_pool[t.track_id].append((sharpness(c), c))
        idx += args.stride
        n += 1
    cap.release()
    log.info("%d tracks, %d crops", len(collector.crops), sum(len(v) for v in collector.crops.values()))

    # ── team → real roster side per fragment ─────────────────────────────────
    clf = TeamClassifier()
    team_of = {}
    if clf.fit(collector.crops):
        team_of = {tid: t for tid, (t, _) in clf.assign(collector.crops).items()}
    cols = clf.team_colors_bgr(collector.crops, team_of)
    light_cluster = (max(cols, key=lambda t: sum(w * c for w, c in zip((.114, .587, .299), cols[t])))
                     if len(cols) == 2 else None)
    l_name = light_team_name(game)
    d_name = ({game["home"], game["away"]} - {l_name}).pop()

    def real_team(tid):
        t = team_of.get(tid)
        if t is None or light_cluster is None:
            return None
        return l_name if t == light_cluster else d_name

    # ── OCR + vote ────────────────────────────────────────────────────────────
    log.info("OCR over sharpest %d bbox-crops per fragment (CPU easyocr) …", args.top_k)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    sharp = {tid: [c for _, c in sorted(pool, key=lambda x: -x[0])[:args.top_k]]
             for tid, pool in ocr_pool.items()}
    reads = defaultdict(list)
    n_total = sum(len(v) for v in sharp.values())
    done = 0
    for tid, lst in sharp.items():
        for c in lst:
            for _, txt, conf in reader.readtext(enhance(c), detail=1, mag_ratio=1.5):
                t = txt.strip().replace(" ", "")
                if DIGITS_RE.match(t) and conf >= READ_MIN_CONF:
                    reads[tid].append((str(int(t)), float(conf)))
            done += 1
            if done % 100 == 0:
                log.info("  OCR %d/%d …", done, n_total)
    reads = dict(reads)

    named, abstained = {}, {}
    for tid, lst in collector.crops.items():
        team = real_team(tid)
        r = reads.get(tid, [])
        if team is None:
            abstained[tid] = {"reason": "team_unknown", "n_reads": len(r)}
            continue
        v_own = vote(r, set(rosters["home" if team == game["home"] else "away"].keys()))
        v_opp = vote(r, set(rosters["away" if team == game["home"] else "home"].keys()))
        if v_own["number"] is None and v_opp["number"] is None:
            abstained[tid] = v_own
            continue
        if v_opp["number"] is not None and (
                v_own["number"] is None or v_opp.get("weight", 0) > v_own.get("weight", 0)):
            # the OPPONENT roster explains the reads better ⇒ the color-team
            # call is suspect — never name across it silently
            abstained[tid] = {"reason": f"team_conflict(color={team}, number says "
                                        f"#{v_opp['number']} w={v_opp.get('weight')})",
                              "n_reads": len(r)}
            continue
        roster = rosters["home" if team == game["home"] else "away"]
        players = roster[v_own["number"]]
        named[tid] = {**v_own, "team": team,
                      "player": players[0] if len(players) == 1 else " | ".join(players),
                      "ambiguous_roster": len(players) > 1}

    # ── simultaneity conflicts: overlapping fragments can't share a player ────
    conflicts = 0
    by_player = defaultdict(list)
    for tid, rec in named.items():
        by_player[(rec["team"], rec["player"])].append(tid)
    for key, tids in by_player.items():
        tids.sort(key=lambda t: -named[t]["weight"])
        for a in range(len(tids)):
            for b in range(a + 1, len(tids)):
                t1, t2 = tids[a], tids[b]
                if t2 in named and first_seen[t2] <= last_seen[t1] and first_seen[t1] <= last_seen[t2]:
                    abstained[t2] = {"reason": f"simultaneity_conflict_with_{t1}",
                                     **{k: named[t2][k] for k in ("number", "weight", "share")}}
                    named.pop(t2)
                    conflicts += 1

    # ── report ────────────────────────────────────────────────────────────────
    n_reads_total = sum(len(v) for v in reads.values())
    n_crops = sum(len(v) for v in sharp.values())
    log.info("── JERSEY OCR PROBE — %s (%s %s vs %s, season %d) ──",
             args.clip, code, game["away"], game["home"], season)
    log.info(" crops OCR'd: %d | digit reads: %d (%.1f%% of crops) | fragments: %d",
             n_crops, n_reads_total, 100 * n_reads_total / max(n_crops, 1), len(collector.crops))
    log.info(" NAMED %d | abstained %d | simultaneity demotions %d", len(named), len(abstained), conflicts)
    for tid, r in sorted(named.items(), key=lambda kv: -kv[1]["weight"]):
        log.info("  frag %-5d → #%-2s %-22s (%s)  weight %.2f share %.0f%% reads %d%s",
                 tid, r["number"], r["player"], r["team"], r["weight"], 100 * r["share"],
                 r["n_reads"], "  [ROSTER-AMBIGUOUS]" if r["ambiguous_roster"] else "")
    reasons = defaultdict(int)
    for v in abstained.values():
        reasons[v["reason"].split("(")[0]] += 1
    log.info(" abstention reasons: %s", dict(reasons))

    # review montage: up to 5 crops per named fragment
    rows = []
    for tid, r in sorted(named.items(), key=lambda kv: -kv[1]["weight"])[:16]:
        tiles = [cv2.resize(c[:, :, ::-1], (72, 96)) for c in collector.crops[tid][:5]]
        while len(tiles) < 5:
            tiles.append(np.zeros((96, 72, 3), np.uint8))
        strip = np.hstack(tiles)
        label = np.zeros((22, strip.shape[1], 3), np.uint8)
        cv2.putText(label, f"frag {tid} -> #{r['number']} {r['player']} ({r['team']}) "
                           f"w={r['weight']} share={int(100*r['share'])}%",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows += [label, strip]
    if rows:
        out = config.VIZ_DIR / f"jersey_probe_{args.label or args.clip}.png"
        config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.vstack(rows))
        log.info(" review montage → %s", out)

    (config.PROJECT_ROOT / "data" / "rosters" / f"probe_{args.label or args.clip}.json").write_text(json.dumps(
        {"named": {str(k): v for k, v in named.items()},
         "abstained": {str(k): v for k, v in abstained.items()},
         "thresholds": {"read_min_conf": READ_MIN_CONF, "name_min_weight": NAME_MIN_WEIGHT,
                        "name_min_share": NAME_MIN_SHARE}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Jersey-number OCR → player names (prototype).")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--clip", default="curry_q1_clip")
    ap.add_argument("--start", type=int, default=11520)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-crops", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=15, help="sharpest bbox-crops OCR'd per fragment")
    ap.add_argument("--video", default=None,
                    help="explicit video file (e.g. a 1080p re-fetch of the clip's game); "
                         "clip still resolves rosters/teams")
    ap.add_argument("--label", default=None,
                    help="output artifact suffix (default: clip name) — keeps A/B runs "
                         "on the same clip from overwriting each other")
    args = ap.parse_args()
    if args.probe:
        probe(args)


if __name__ == "__main__":
    main()
