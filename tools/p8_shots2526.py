"""tools/p8_shots2526.py: the Level 4 shot-position check on NBA 2025-26 broadcasts (docs/research/P8_shots2526.md,
pre-registered in 58f54663; PLAN.md D144, D145).

    python3 -m tools.p8_shots2526 truth GAME                       shots of the mirror, in the court frame (light)
    V3_CV_PYTHON -m tools.p8_shots2526 place GAME [--budget-s S]   scorebug anchors, search, bisection (easyocr)
    V3_CV_PYTHON -m tools.p8_shots2526 build GAME [--budget-s S]   r23 windows around the placed shots (frozen build)
    python3 -m tools.p8_shots2526 score GAME                       nearest-track oracle rows, orientation rule
    python3 -m tools.p8_shots2526 reference                        R0 on gsw_bkn_2015 (2015-16, Phase A r23 build)
    python3 -m tools.p8_shots2526 report                           pooled rows, gate, reports/research/p8_shots2526.json

place and build run under the heavy-job lock with the CV Python of D87 and stop cleanly at --budget-s (each job at
most 30 minutes); a rerun continues from the cached reads and the built windows. The OCR reader is
tools.p8_identity's (same bands, same upscale, same confidence floor, same parse_tokens), held in one long-lived
easyocr process per job (D145); every read's seconds are recorded. Frames are extracted with ffmpeg's input seek,
the same seek the frozen clip cut uses, so a placed video time and a clip frame refer to the same frame.

The shooter is not named by r23 (no identity, no ball): the compared position is the r23 track nearest the logged
location, "nearest track, oracle; can show failure only" (D145). Every game id and video id passes refuse_cv_test with
its home team (two games are played in D85 test arenas and stop the tool) and the held-out guard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.p8_common import REPO, atomic_write, guard, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_shots2526"
CHECK_SET = ("0022501048", "0022501154", "0022501172", "0042500102", "0042500103", "0042500112", "0042500113",
             "0042500123", "0042500142", "0042500173")
KEPT_OUT = ("0022501165", "0042500171")          # D85 test arenas (TOR, LAL), D145
RETIRED = ("0042500112",)                         # D151 (2): left the scored set for good
PILOT = "0022501048"                             # D151 (3): the longest remaining video
FULL_MIN_S = 2880.0                              # D151 (1): a full broadcast lasts at least the regulation clock
# the scorebug crop (P8_shots2526.md section 11.3), fixed before any read of the new pilot
CROP_PCT = (10, 90)
CROP_MARGIN_PX = 16
CROP_MIN_FRAMES = 5
# placement (P8_shots2526.md section 3)
ANCHOR_STEP_S = 20.0
SEARCH_BEFORE_S = 5.0
SEARCH_MAX_READS = 60
BISECT_READS = 2
# implementation details fixed with the code, before any read (P8_shots2526.md section 9)
BRACKET_MAX_S = 2.0          # the two bracketing 1 Hz reads are at most 2 s apart (one unreadable read between)
JUMP_SLACK_S = 1.0           # across the bracket the clock may not fall by more than the video time + 1 s
MERGE_GAP_S = 10.0           # windows closer than this are built as one (each build reloads its models)
MAX_WINDOW_S = 120.0
# positions (sections 3 and 4)
WIN_BEFORE_S, WIN_AFTER_S = 3.0, 0.5
CLIP_AFTER_S = 1.0
MIRROR_MARGIN_FT = 2.0
SPOT_FT = (3.0, 6.0)
BASKETS = {"left": (5.25, 25.0), "right": (88.75, 25.0)}
# section 6 and the gate (section 5)
BOOT_N, BOOT_SEED = 1000, 0
TARGETS = {"shooter_p50_ft": 3.0, "shooter_p90_ft": 6.0}
GATE_MARGIN_FT = {"p50": 1.0, "p90": 2.0}
LABEL = "nearest track, oracle; can show failure only"
SUMMARY = REPO / "reports" / "research" / "p8_shots2526.json"


# ── pure helpers (unit tested) ──────────────────────────────────────────────────────────────────────────────────────
def court_xy(loc_x: float, loc_y: float, side: str) -> tuple:
    """LOC_X, LOC_Y (tenths of a foot from the attacked basket) and the attacked side -> (X, Y) on a 94 x 50 ft court."""
    if side == "left":
        return 5.25 + loc_y / 10.0, 25.0 - loc_x / 10.0
    if side == "right":
        return 88.75 - loc_y / 10.0, 25.0 + loc_x / 10.0
    raise ValueError(f"side {side!r}")


def frame_moments(parsed: dict, events: list) -> set:
    """Every (period, clock) of a parsed frame for which an ordered pair of its integers equals the play-by-play score
    (tools.p8_scorebug.matches), with tools.p8_identity.frame_matches's rules (no clock: none; no period: 1 to 4)."""
    import itertools
    from tools.p8_scorebug import matches
    if not parsed["clocks"] or len(parsed["ints"]) < 2:
        return set()
    periods = parsed["periods"] or [1, 2, 3, 4]
    pairs = set(itertools.permutations(parsed["ints"], 2))
    return {(p, c) for p in periods for c in parsed["clocks"] if any(matches(events, p, c, a, h) for a, h in pairs)}


def anchors_of(reads: dict, events: list) -> list:
    """[(video s, period, clock)] for the anchor reads that read exactly one moment."""
    out = []
    for t, parsed in sorted(reads.items()):
        m = frame_moments(parsed, events)
        if len(m) == 1:
            p, c = next(iter(m))
            out.append((float(t), int(p), float(c)))
    return out


def crop_from(frames: list, events: list, width: int = 1280, height: int = 720):
    """Section 11.3's fixed rule. frames: [(parsed, boxes)] of the anchor reads, boxes [x0, y0, x1, y1, text] in
    frame pixels. From frames that read exactly one moment, the boxes of texts carrying its clock, one of its matched
    scores or a period; the crop is the CROP_PCT percentiles of their edges widened by CROP_MARGIN_PX and clipped;
    None with fewer than CROP_MIN_FRAMES such frames."""
    import itertools
    from tools.p8_identity import parse_tokens
    from tools.p8_scorebug import matches
    kept, n = [], 0
    for parsed, boxes in frames:
        m = frame_moments(parsed, events)
        if len(m) != 1:
            continue
        p, c = next(iter(m))
        scores = {v for a, h in itertools.permutations(parsed["ints"], 2) if matches(events, p, c, a, h) for v in (a, h)}
        mine = []
        for b in boxes:
            q = parse_tokens([b[4]])
            if c in q["clocks"] or q["periods"] or scores & set(q["ints"]):
                mine.append(b[:4])
        if mine:
            kept += mine
            n += 1
    if n < CROP_MIN_FRAMES:
        return None
    a = np.asarray(kept, float)
    lo, hi = CROP_PCT
    x0, y0 = np.percentile(a[:, 0], lo) - CROP_MARGIN_PX, np.percentile(a[:, 1], lo) - CROP_MARGIN_PX
    x1, y1 = np.percentile(a[:, 2], hi) + CROP_MARGIN_PX, np.percentile(a[:, 3], hi) + CROP_MARGIN_PX
    box = [max(0, int(x0)), max(0, int(y0)), min(int(width), int(math.ceil(x1))), min(int(height), int(math.ceil(y1)))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def estimate(anchors: list, period: int, clock: float):
    """t* from the anchor of the same period nearest before the shot in game time (clock at or above it), assuming
    the clock ran from there; None when there is none."""
    cand = [(c - clock, t, c) for t, p, c in anchors if p == period and c >= clock]
    if not cand:
        return None
    _, t, c = min(cand)
    return t + (c - clock)


def place_shot(period: int, clock: float, anchors: list, read, duration: float) -> dict:
    """Section 3's search for one shot. read(t) -> set of (period, clock) moments the frame at video second t reads.
    -> {status, t (placed video s), bracket_s, reads}."""
    t_star = estimate(anchors, period, clock)
    if t_star is None:
        return {"status": "no anchor", "reads": 0}
    start = math.floor(t_star - SEARCH_BEFORE_S)
    n = 0
    prev = None                       # (t, clock) of the last read of this period showing a larger clock
    for k in range(SEARCH_MAX_READS):
        t = float(start + k)
        if t < 0:
            continue
        if t > duration:
            break
        n += 1
        ms = [c for p, c in read(t) if p == period]
        hi, lo = [c for c in ms if c > clock], [c for c in ms if c <= clock]
        if hi and not lo:
            prev = (t, max(hi))
            continue
        if lo and not hi:
            if prev is None:
                return {"status": "passed before a bracket", "reads": n}
            if t - prev[0] > BRACKET_MAX_S:
                return {"status": "gap at the moment", "reads": n}
            if prev[1] - max(lo) > (t - prev[0]) + JUMP_SLACK_S:
                return {"status": "clock jumped", "reads": n}
            a, b = prev[0], t
            for _ in range(BISECT_READS):
                m = (a + b) / 2.0
                n += 1
                ms = [c for p, c in read(m) if p == period]
                hi2, lo2 = [c for c in ms if c > clock], [c for c in ms if c <= clock]
                if lo2 and not hi2:
                    b = m
                elif hi2 and not lo2:
                    a = m
                else:
                    break
            return {"status": "placed", "t": round((a + b) / 2.0, 4), "bracket_s": round(b - a, 4), "reads": n}
    return {"status": "no bracket in the search", "reads": n}


def windows_for(placed: list, fps: float, section: str) -> list:
    """Frozen plan_windows-form windows around placed video seconds: [t - 3.0 - PAD_S, t + 1.0], merged when they
    overlap or lie closer than MERGE_GAP_S, capped at MAX_WINDOW_S."""
    from sportvu.rebuild import PAD_S
    iv = sorted((t - WIN_BEFORE_S, t + CLIP_AFTER_S) for t in placed)
    merged = []
    for a, b in iv:
        if merged and a - merged[-1][1] < MERGE_GAP_S and b - merged[-1][0] <= MAX_WINDOW_S:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for k, (a, b) in enumerate(merged):
        f0 = max(0, int(round(a * fps)))
        out.append({"window": "%s_w%03d" % (section, k), "section": section, "fps": fps,
                    "f_start": max(0, int(round((a - PAD_S) * fps))), "f0": f0, "f_end": int(round(b * fps)),
                    "period": None})
    return out


def oracle(points_by_frame: dict, target: tuple, frames: list) -> tuple:
    """(min distance ft, frame) of the nearest track to target over the given built frames; (None, None) if no point."""
    best, bf = None, None
    for f in frames:
        pts = points_by_frame.get(f)
        if not pts:
            continue
        dd = np.hypot(np.asarray(pts, float)[:, 0] - target[0], np.asarray(pts, float)[:, 1] - target[1])
        dd = dd[np.isfinite(dd)]
        if not len(dd):
            continue
        d = float(dd.min())
        if best is None or d < best:
            best, bf = d, f
    return best, bf


def choose_mirror(medians: dict, margin: float = MIRROR_MARGIN_FT) -> tuple:
    """(mirror, resolved): the mirror of smallest median; resolved when it beats the runner-up by at least margin."""
    ok = sorted((v, k) for k, v in medians.items() if v is not None)
    if not ok:
        return None, False
    if len(ok) == 1:
        return ok[0][1], False
    return ok[0][1], (ok[1][0] - ok[0][0]) >= margin


def pct(a, q):
    return round(float(np.percentile(a, q)), 3) if len(a) else None


def boot(groups: dict, stat, n: int = BOOT_N, seed: int = BOOT_SEED):
    from tools.p8_scorecard import boot_windows
    return boot_windows(groups, stat, n, seed)


# ── paths and guards ────────────────────────────────────────────────────────────────────────────────────────────────
def root() -> Path:
    return p8_root().parent / "level4" / "nba2526"


def work(gid: str) -> Path:
    d = root() / "shots2526" / gid
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_record(gid: str) -> dict:
    rec = json.loads((root() / "fetch.json").read_text())
    hits = [(vid, r) for vid, r in rec.items() if r.get("game_id") == gid and r.get("verified")]
    if len(hits) != 1:
        raise SystemExit(f"{WHO}: {gid} has no single verified download")
    return {"video_id": hits[0][0], **hits[0][1]}


def scored_set(fetch: dict) -> list:
    """D151 (1) and (2): the check-set games whose probed duration is at least FULL_MIN_S, less the retired pilot."""
    dur = {r["game_id"]: r.get("probe_duration_s") for r in fetch.values() if r.get("verified")}
    return [g for g in CHECK_SET if g not in RETIRED and (dur.get(g) or 0) >= FULL_MIN_S]


def checked(gid: str) -> dict:
    """The game's record after every guard: the scored set only (D145, D151), refuse_cv_test with its home team,
    held-out guard."""
    if gid in KEPT_OUT or gid not in CHECK_SET:
        raise SystemExit(f"{WHO}: {gid} is not in the check set of 10 (D145)")
    if gid in RETIRED or gid not in scored_set(json.loads((root() / "fetch.json").read_text())):
        raise SystemExit(f"{WHO}: {gid} is not in the scored set (D151: full broadcasts only, 0042500112 retired)")
    refuse_cv_test([gid], WHO)
    f = fetch_record(gid)
    refuse_cv_test([gid, f["video_id"], f["file"]], WHO)
    guard([gid, f["video_id"]], WHO)
    t = truth(gid, write=False)
    refuse_cv_test([gid], WHO, homes=[t["home"]])          # a game played in a D85 test arena stops here
    return {**f, "home": t["home"], "away": t["away"], "video": root() / "video" / f["file"]}


# ── truth ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def truth(gid: str, write: bool = True) -> dict:
    """The game's field-goal attempts: event, period, clock (s, tenths kept), side, made, truth (X, Y); cached."""
    p = work(gid) / "truth.json"
    if p.exists():
        return json.loads(p.read_text())
    import pandas as pd
    from tools import p8_nba2526_search as S
    sh = "shotdetail_po_2025/shotdetail_po_2025.csv" if gid.startswith("004") else "shotdetail_2025/shotdetail_2025.csv"
    cd = "cdnnba_po_2025/cdnnba_po_2025.csv" if gid.startswith("004") else "cdnnba_2025/cdnnba_2025.csv"
    s = pd.read_csv(S.shots_root() / sh, usecols=["GAME_ID", "GAME_EVENT_ID", "PERIOD", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "HTM", "VTM"])
    s = s[s.GAME_ID == int(gid)]
    c = pd.read_csv(S.shots_root() / cd, usecols=["gameId", "actionNumber", "clock", "period", "side", "isFieldGoal"], low_memory=False)
    c = c[(c.gameId == int(gid)) & (c.isFieldGoal == 1)]
    m = s.merge(c, left_on=["GAME_ID", "GAME_EVENT_ID"], right_on=["gameId", "actionNumber"], how="left")
    from tools.p8_nba_identity import clock_s
    shots = []
    for r in m.itertuples(index=False):
        X, Y = court_xy(float(r.LOC_X), float(r.LOC_Y), r.side) if isinstance(r.side, str) else (None, None)
        shots.append({"event": int(r.GAME_EVENT_ID), "period": int(r.PERIOD), "clock": clock_s(r.clock),
                      "side": r.side if isinstance(r.side, str) else None, "made": int(r.SHOT_MADE_FLAG),
                      "X": X, "Y": Y})
    out = {"game": gid, "home": str(s.HTM.iloc[0]), "away": str(s.VTM.iloc[0]), "attempts": len(shots), "shots": shots,
           "source": "github.com/shufinskiy/nba_data shotdetail and cdnnba (NBA Statistics, attribution NBA.com)"}
    if write:
        atomic_write(p, json.dumps(out))
    return out


# ── OCR worker (V3_CV_PYTHON) ───────────────────────────────────────────────────────────────────────────────────────
class Reader:
    """tools.p8_identity's reader on frames extracted by ffmpeg's input seek, in one long-lived easyocr process;
    reads cached by video second (2 decimals) in reads.json, with their seconds."""

    def __init__(self, gid: str, video: Path, events: list):
        self.gid, self.video, self.events = gid, Path(video), events
        self.path = work(gid) / "reads.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.tmp = work(gid) / "tmp"
        self.tmp.mkdir(exist_ok=True)
        self._ocr = None
        self.new = 0

    def tokens(self, t: float, crop=None) -> dict:
        """Band read (crop None): the registered bands, with easyocr's boxes kept in frame pixels. Crop read: the
        same reader on the crop box [x0, y0, x1, y1] only (section 11.3); cached under 'c' + the second."""
        from tools.p8_identity import BANDS, parse_tokens
        key = ("c%.2f" if crop else "%.2f") % t
        if key not in self.cache:
            import cv2
            t0 = time.time()
            jpg = self.tmp / ("t%09.2f.jpg" % t)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", "%.4f" % t, "-i", str(self.video),
                            "-frames:v", "1", str(jpg)], check=True)
            t1 = time.time()
            if self._ocr is None:
                import easyocr
                self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            t2 = time.time()
            img = cv2.imread(str(jpg))
            texts, boxes = [], []
            if img is not None:
                h = img.shape[0]
                if crop:
                    parts = [(int(crop[0]), int(crop[1]), img[int(crop[1]):int(crop[3]), int(crop[0]):int(crop[2])])]
                else:
                    parts = [(0, int(a * h), img[int(a * h):int(b * h)]) for _, a, b in BANDS]
                for ox, oy, part in parts:
                    big = cv2.resize(part, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    for bb, x, conf in self._ocr.readtext(big, detail=1):
                        if conf >= 0.2:
                            texts.append(x)
                            xs, ys = [q[0] / 2 + ox for q in bb], [q[1] / 2 + oy for q in bb]
                            boxes.append([round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1), x])
            jpg.unlink(missing_ok=True)
            self.cache[key] = {"texts": texts, "boxes": boxes, "extract_s": round(t1 - t0, 3),
                               "ocr_s": round(time.time() - t2, 3), "model_load_s": round(t2 - t1, 3) if t2 - t1 > 0.5 else 0.0,
                               "kind": "crop" if crop else "band"}
            self.new += 1
            if self.new % 20 == 0:
                self.save()
        return parse_tokens(self.cache[key]["texts"])

    def moments(self, t: float, crop=None) -> set:
        return frame_moments(self.tokens(t, crop), self.events)

    def save(self):
        atomic_write(self.path, json.dumps(self.cache))


def cmd_place(gid: str, budget_s: float) -> dict:
    from tools.p8_nba_identity import load_events
    g = checked(gid)
    T = truth(gid)
    events = load_events()[gid]
    R = Reader(gid, g["video"], events)
    t0 = time.time()
    dur = float(g["duration_s"])
    out = {"game": gid, "complete": False}
    anchor_ts = [float(k * ANCHOR_STEP_S) for k in range(int(dur // ANCHOR_STEP_S) + 1)]
    for t in anchor_ts:
        if time.time() - t0 > budget_s:
            R.save(); out["stopped"] = "budget during anchors"; return _place_out(gid, out, R)
        R.tokens(t)
    anchors = anchors_of({t: R.tokens(t) for t in anchor_ts}, events)
    crop = crop_from([(R.tokens(t), R.cache["%.2f" % t].get("boxes", [])) for t in anchor_ts], events,
                     g.get("width", 1280), g.get("height", 720))
    out["crop"] = crop
    if crop is None:
        R.save(); out["stopped"] = "no crop: fewer than %d single-moment anchor frames (section 11.3)" % CROP_MIN_FRAMES
        _place_out(gid, out, R)
        raise SystemExit(f"{WHO}: {gid}: no scorebug crop; the search does not run (tell the lead)")
    res = {}
    for s in T["shots"]:
        if time.time() - t0 > budget_s:
            R.save(); out["stopped"] = "budget during the search"; out["placements"] = res; return _place_out(gid, out, R)
        if s["clock"] is None:
            res[str(s["event"])] = {"status": "no clock in the mirror", "reads": 0}
            continue
        res[str(s["event"])] = place_shot(s["period"], s["clock"], anchors, lambda t: R.moments(t, crop), dur)
    R.save()
    out.update({"complete": True, "anchors": len(anchors), "anchor_reads": len(anchor_ts), "placements": res})
    return _place_out(gid, out, R)


def _place_out(gid, out, R):
    from tools.p8_identity import parse_tokens
    secs = [v for v in R.cache.values()]
    out["reads_cached"] = len(secs)
    ocr = [v["ocr_s"] for v in secs]
    ext = [v["extract_s"] for v in secs]
    out["per_read_s"] = {"ocr_p50": pct(ocr, 50), "ocr_p90": pct(ocr, 90), "extract_p50": pct(ext, 50),
                         "extract_p90": pct(ext, 90), "model_loads": sum(1 for v in secs if v.get("model_load_s"))}
    for kind in ("band", "crop"):                         # section 11.3: time and read rate per reader input
        ks = [k for k in R.cache if k.startswith("c") == (kind == "crop")]
        o = [R.cache[k]["ocr_s"] for k in ks]
        ps = [parse_tokens(R.cache[k]["texts"]) for k in ks]
        out[f"{kind}_reads"] = {"n": len(ks), "ocr_p50": pct(o, 50), "ocr_p90": pct(o, 90),
                                "share_with_clock": round(sum(1 for p in ps if p["clocks"]) / len(ks), 3) if ks else None,
                                "share_one_moment": round(sum(1 for p in ps if len(frame_moments(p, R.events)) == 1) / len(ks), 3) if ks else None}
    st = {}
    for v in out.get("placements", {}).values():
        st[v["status"]] = st.get(v["status"], 0) + 1
    out["status_counts"] = st
    atomic_write(work(gid) / "place.json", json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "placements"}, indent=1), flush=True)
    return out


# ── build (V3_CV_PYTHON) ────────────────────────────────────────────────────────────────────────────────────────────
def section_of(gid: str) -> str:
    return f"nba2526_{gid}_s00"


def cmd_build(gid: str, budget_s: float) -> dict:
    from tools import p8_frozen as F
    from sportvu.autolabel import gate_args
    g = checked(gid)
    P = F.install_paths(p8_root())
    sec = section_of(gid)
    refuse_cv_test([sec], WHO)
    link = P["video"] / f"{sec}.mp4"
    if not link.is_symlink():
        os.symlink(g["video"], link)
    pl = json.loads((work(gid) / "place.json").read_text())
    if not pl.get("complete"):
        raise SystemExit(f"{WHO}: place {gid} is not complete")
    fps = 30000 / 1001 if "30000/1001" in g.get("ffprobe", "") else float(g["fps"])
    placed = [v["t"] for v in pl["placements"].values() if v["status"] == "placed"]
    wins = windows_for(placed, fps, sec)
    out_dir = work(gid) / "build"
    out_dir.mkdir(exist_ok=True)
    atomic_write(out_dir / "windows.json", json.dumps(wins, indent=1))
    os.environ["COURT_GRID_WEIGHTS"] = str(F.court_weights("r23"))
    t0 = time.time()
    rec = {"windows": len(wins), "built": 0, "gate_empty": 0, "failed": [], "seconds": {}}
    for w in wins:
        if time.time() - t0 > budget_s:
            rec["stopped"] = "budget"; break
        stem = w["window"]
        if (out_dir / f"{stem}_frames.json").exists() or F.gate_empty_path(out_dir, stem).exists():
            rec["gate_empty" if F.gate_empty_path(out_dir, stem).exists() else "built"] += 1
            continue
        t1 = time.time()
        snip = F.safe_cut(P, w, out_dir)
        r = F.safe_build(P, w, snip, out_dir, extra_args=gate_args())
        rec["seconds"][stem] = {"wall_s": round(time.time() - t1, 1), "video_s": round((w["f_end"] - w["f_start"]) / fps, 2)}
        if not r["ok"]:
            rec["failed"].append(stem)
        elif r.get("empty"):
            rec["gate_empty"] += 1
        else:
            rec["built"] += 1
    prev = json.loads((out_dir / "build_log.json").read_text()) if (out_dir / "build_log.json").exists() else {"seconds": {}}
    rec["seconds"] = {**prev.get("seconds", {}), **rec["seconds"]}
    rec["complete"] = "stopped" not in rec and not rec["failed"]
    atomic_write(out_dir / "build_log.json", json.dumps(rec, indent=1))
    ws = list(rec["seconds"].values())
    print(json.dumps({**{k: v for k, v in rec.items() if k != "seconds"}, "timed_windows": len(ws),
                      "wall_s": round(sum(x["wall_s"] for x in ws), 1), "video_s": round(sum(x["video_s"] for x in ws), 1)},
                     indent=1), flush=True)
    return rec


# ── score ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def built_points(out_dir: Path, w: dict) -> tuple:
    """({frame: [(x, y)]} of accepted frames, accepted frame list) of one built window, frames in window frames."""
    from sportvu.bench import accepted
    side_p = out_dir / f"{w['window']}_frames.json"
    tr_p = out_dir / f"{w['window']}_trajectories.json"
    if not side_p.exists() or not tr_p.exists():
        return {}, []
    acc = sorted(r["frame"] for r in json.loads(side_p.read_text())["frames"] if accepted(r))
    keep = set(acc)
    pts = {}
    for tr in json.loads(tr_p.read_text()).values():
        for q in tr.get("cleaned") or []:
            if q[0] in keep and q[1] is not None and q[2] is not None and math.isfinite(q[1]) and math.isfinite(q[2]):
                pts.setdefault(int(q[0]), []).append((float(q[1]), float(q[2])))
    return pts, acc


def shot_obs(t: float, target: tuple, wins: list, fps: float, out_dir: Path, cache: dict) -> dict:
    """At the placed frame and windowed, under every mirror: nearest-track distance, best frame offset (s)."""
    from sportvu.sync import MIRRORS, apply_mirror
    f_abs = t * fps
    w = next((w for w in wins if w["f_start"] <= f_abs <= w["f_end"]), None)
    if w is None:
        return {"status": "no window"}
    if w["window"] not in cache:
        cache[w["window"]] = built_points(out_dir, w)
    pts, acc = cache[w["window"]]
    if not acc:
        return {"status": "window not built or no accepted frame"}
    loc = f_abs - w["f_start"]
    near = min(acc, key=lambda f: abs(f - loc))
    at_ok = abs(near - loc) <= 0.06 * fps                # the built frame of the placed moment (10 Hz)
    span = [f for f in acc if loc - WIN_BEFORE_S * fps <= f <= loc + WIN_AFTER_S * fps]
    out = {"status": "ok", "window": w["window"], "accepted_at_placed": bool(at_ok), "mirrors": {}}
    for name, (fx, fy) in MIRRORS.items():
        mp = {f: [tuple(v) for v in apply_mirror(np.asarray(p, float), fx, fy)] for f, p in pts.items() if f in set(span) | {near}}
        d_at = oracle(mp, target, [near])[0] if at_ok else None
        d_w, bf = oracle(mp, target, span)
        out["mirrors"][name] = {"at": None if d_at is None else round(d_at, 3), "win": None if d_w is None else round(d_w, 3),
                                "best_offset_s": None if bf is None else round((bf - loc) / fps, 3)}
    return out


def cmd_score(gid: str) -> dict:
    g = checked(gid)
    T = truth(gid)
    pl = json.loads((work(gid) / "place.json").read_text())
    out_dir = work(gid) / "build"
    wins = json.loads((out_dir / "windows.json").read_text())
    fps = wins[0]["fps"] if wins else 30.0
    cache, obs = {}, []
    for s in T["shots"]:
        p = pl["placements"].get(str(s["event"]), {"status": "not searched"})
        o = {"event": s["event"], "period": s["period"], "placement": p["status"]}
        if p["status"] == "placed" and s["X"] is not None:
            o.update(shot_obs(p["t"], (s["X"], s["Y"]), wins, fps, out_dir, cache))
        obs.append(o)
    # orientation on period 1 (section 4.3)
    med = {}
    from sportvu.sync import MIRRORS
    for name in MIRRORS:
        v = [o["mirrors"][name]["win"] for o in obs if o["period"] == 1 and o.get("status") == "ok" and o["mirrors"][name]["win"] is not None]
        med[name] = round(float(np.median(v)), 3) if v else None
    mirror, resolved = choose_mirror(med)
    rows = rows_of([o for o in obs if o["period"] != 1], mirror)
    rec = {"game": gid, "video_id": g["video_id"], "label": LABEL, "attempts": T["attempts"],
           "period1_medians_by_mirror": med, "mirror": mirror, "orientation_resolved": resolved,
           "rows": rows, "obs": obs}
    atomic_write(work(gid) / "score.json", json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "obs"}, indent=1), flush=True)
    return rec


def rows_of(obs: list, mirror) -> dict:
    """Rows of section 5 over the shots outside period 1, under the chosen mirror."""
    n = len(obs)
    placed = [o for o in obs if o["placement"] == "placed"]
    ok = [o for o in placed if o.get("status") == "ok"]
    at = [o["mirrors"][mirror]["at"] for o in ok if mirror and o["mirrors"][mirror]["at"] is not None]
    win = [o["mirrors"][mirror]["win"] for o in ok if mirror and o["mirrors"][mirror]["win"] is not None]
    offs = [o["mirrors"][mirror]["best_offset_s"] for o in ok if mirror and o["mirrors"][mirror]["best_offset_s"] is not None]
    st = {}
    for o in obs:
        k = o["placement"] if o["placement"] != "placed" else ("placed: " + o.get("status", "?"))
        st[k] = st.get(k, 0) + 1
    r = {"attempts_outside_period_1": n, "placed": len(placed), "placed_share": round(len(placed) / n, 3) if n else None,
         "court_share": round(len(at) / len(placed), 3) if placed else None, "status_counts": st,
         "at_placed": {"n": len(at), "p50_ft": pct(at, 50), "p90_ft": pct(at, 90),
                       **{f"within_{x:g}ft": round(float(np.mean(np.array(at) <= x)), 3) if at else None for x in SPOT_FT}},
         "windowed": {"n": len(win), "p50_ft": pct(win, 50), "p90_ft": pct(win, 90),
                      **{f"within_{x:g}ft": round(float(np.mean(np.array(win) <= x)), 3) if win else None for x in SPOT_FT},
                      "best_offset_s_quartiles": [pct(offs, 25), pct(offs, 50), pct(offs, 75)]}}
    for key, vals in (("at_placed", "at"), ("windowed", "win")):
        groups = {}
        for o in ok:
            v = o["mirrors"][mirror][vals] if mirror else None
            if v is not None:
                groups.setdefault(o["window"], []).append(v)
        r[key]["p50_window_ci"] = boot(groups, lambda a: pct(a, 50))
        r[key]["p90_window_ci"] = boot(groups, lambda a: pct(a, 90))
        r[key]["pass_targets"] = None if not r[key]["n"] else bool(r[key]["p50_ft"] <= TARGETS["shooter_p50_ft"] and r[key]["p90_ft"] <= TARGETS["shooter_p90_ft"])
    return r


# ── R0: the 2015-16 reference on gsw_bkn_2015 ───────────────────────────────────────────────────────────────────────
def cmd_reference() -> dict:
    import pandas as pd
    from tools import p8_frozen as F
    from sportvu.sync import MIRRORS, apply_mirror
    tag = "gsw_bkn_2015"
    refuse_cv_test([tag], WHO)
    gm = F.game_of(tag)
    sv = gm["sportvu"]
    guard([tag, sv, gm["video_id"]], WHO)
    P = F.install_paths(p8_root())
    fx, fy = MIRRORS[F.mirror_of(P, tag)]
    D = Path(os.environ["DATA_ROOT"])
    shots = pd.read_parquet(D / "analytics" / "shot_frames" / "2015-16" / f"{sv}.parquet", columns=["event", "release_frame", "shooter_id"])
    fr = pd.read_parquet(D / "sportvu_contract" / f"{sv}.parquet", columns=["frame", "player_id", "x", "y"])
    want = {(int(f), int(s)) for f, s in shots[["release_frame", "shooter_id"]].dropna().itertuples(index=False) if f >= 0}
    fr = fr[[(int(a), int(b)) in want for a, b in fr[["frame", "player_id"]].fillna(-1).itertuples(index=False)]]
    pos = {(int(a), int(b)): (float(x), float(y)) for a, b, x, y in fr.itertuples(index=False)}
    del fr
    rel = {int(e): pos.get((int(f), int(s))) for e, f, s in shots.itertuples(index=False) if f >= 0 and pd.notna(s)}
    ev = {}
    for p in sorted((P["builds"] / "e2").glob(f"{tag}_s*_events.json")):
        for e in json.loads(p.read_text())["events"]:
            if e["kind"] == "shot":
                ev[e["event"]] = (p.name.replace("_events.json", ""), e["frame"])
    obs, groups = [], {}
    cache = {}
    for e, (stem, f) in sorted(ev.items()):
        tgt = rel.get(e)
        if tgt is None:
            continue
        tgt = tuple(apply_mirror(np.asarray([tgt], float), fx, fy)[0])
        if stem not in cache:
            side = json.loads((P["builds"] / "r23" / f"{stem}_frames.json").read_text())
            fps = float(side.get("fps") or 59.94)
            cache[stem] = (built_points(P["builds"] / "r23", {"window": stem}), fps)
        (pts, acc), fps = cache[stem]
        if not acc:
            continue
        near = min(acc, key=lambda q: abs(q - f))
        at = oracle(pts, tgt, [near])[0] if abs(near - f) <= 0.06 * fps else None
        span = [q for q in acc if f - WIN_BEFORE_S * fps <= q <= f + WIN_AFTER_S * fps]
        win, _ = oracle(pts, tgt, span)
        obs.append({"event": e, "window": stem, "at": at, "win": win})
        if win is not None:
            groups.setdefault(stem, []).append(win)
    at = [o["at"] for o in obs if o["at"] is not None]
    win = [o["win"] for o in obs if o["win"] is not None]
    rec = {"reference": "R0", "game": tag, "label": LABEL + "; in sample for r23", "shots_with_release_and_placement": len(obs),
           "at_placed": {"n": len(at), "p50_ft": pct(at, 50), "p90_ft": pct(at, 90)},
           "windowed": {"n": len(win), "p50_ft": pct(win, 50), "p90_ft": pct(win, 90),
                        "p50_window_ci": boot(groups, lambda a: pct(a, 50)), "p90_window_ci": boot(groups, lambda a: pct(a, 90))}}
    atomic_write(root() / "shots2526" / "reference_R0.json", json.dumps({**rec, "obs": obs}, indent=1))
    print(json.dumps(rec, indent=1), flush=True)
    return rec


# ── report ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_report() -> dict:
    from export.schemas import find_private
    base = root() / "shots2526"
    R0 = json.loads((base / "reference_R0.json").read_text()) if (base / "reference_R0.json").exists() else None
    games = {}
    scored = scored_set(json.loads((root() / "fetch.json").read_text()))
    for gid in scored:
        p = base / gid / "score.json"
        if p.exists():
            s = json.loads(p.read_text())
            games[gid] = {k: v for k, v in s.items() if k != "obs"}
            games[gid]["_win"] = [o["mirrors"][s["mirror"]]["win"] for o in s["obs"] if o["period"] != 1 and o.get("status") == "ok"
                                  and s["mirror"] and o["mirrors"][s["mirror"]]["win"] is not None]
            pl = json.loads((base / gid / "place.json").read_text())
            games[gid]["per_read_s"] = pl.get("per_read_s")
            bl = base / gid / "build" / "build_log.json"
            if bl.exists():
                b = json.loads(bl.read_text())
                ws = list(b["seconds"].values())
                games[gid]["build"] = {"windows": b["windows"], "built": b["built"], "gate_empty": b["gate_empty"],
                                       "failed": len(b["failed"]), "wall_s": round(sum(x["wall_s"] for x in ws), 1),
                                       "video_s": round(sum(x["video_s"] for x in ws), 1)}
    resolved = {g: v for g, v in games.items() if v["orientation_resolved"]}
    pooled = [x for v in resolved.values() for x in v["_win"]]
    gate = None
    if R0 and pooled:
        p50, p90 = pct(pooled, 50), pct(pooled, 90)
        gate = {"pooled_windowed_p50_ft": p50, "pooled_windowed_p90_ft": p90,
                "limit_p50_ft": round(R0["windowed"]["p50_ft"] + GATE_MARGIN_FT["p50"], 3),
                "limit_p90_ft": round(R0["windowed"]["p90_ft"] + GATE_MARGIN_FT["p90"], 3)}
        gate["within_margin"] = bool(p50 <= gate["limit_p50_ft"] and p90 <= gate["limit_p90_ft"])
        gate["game_ci_p50"] = boot({g: v["_win"] for g, v in resolved.items()}, lambda a: pct(a, 50)) if len(resolved) >= 2 else None
        gate["game_ci_p90"] = boot({g: v["_win"] for g, v in resolved.items()}, lambda a: pct(a, 90)) if len(resolved) >= 2 else None
    res = {"check": "P8 Level 4 shot position, NBA 2025-26", "pre_registration": "docs/research/P8_shots2526.md (58f54663)",
           "date": dt.date.today().isoformat(), "label": LABEL, "check_set": list(CHECK_SET), "kept_out_d85_arenas": list(KEPT_OUT),
           "scored_set_d151": scored, "retired": list(RETIRED), "full_min_s": FULL_MIN_S,
           "games_scored": sorted(games), "orientation_unresolved": sorted(set(games) - set(resolved)),
           "R0": R0 and {k: v for k, v in R0.items() if k != "obs"}, "gate": gate, "targets": TARGETS,
           "games": {g: {k: v for k, v in v.items() if k != "_win"} for g, v in games.items()},
           "truth_source": "github.com/shufinskiy/nba_data (NBA Statistics; attribution NBA.com; private, non-commercial)"}
    text = json.dumps(res, indent=1, default=str)
    if find_private(text):
        raise SystemExit(f"{WHO}: the summary holds a private item: {find_private(text)[:3]}")
    atomic_write(SUMMARY, text + "\n")
    print(text, flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("truth", "place", "build", "score"):
        s = sub.add_parser(c)
        s.add_argument("game")
        if c in ("place", "build"):
            s.add_argument("--budget-s", type=float, default=25 * 60)
    sub.add_parser("reference")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "truth":
        if a.game not in CHECK_SET:
            raise SystemExit(f"{WHO}: {a.game} is not in the check set of 10 (D145)")
        t = truth(a.game)
        checked(a.game)
        print(json.dumps({k: v for k, v in t.items() if k != "shots"}, indent=1))
    elif a.cmd == "place":
        cmd_place(a.game, a.budget_s)
    elif a.cmd == "build":
        cmd_build(a.game, a.budget_s)
    elif a.cmd == "score":
        cmd_score(a.game)
    elif a.cmd == "reference":
        cmd_reference()
    else:
        cmd_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
