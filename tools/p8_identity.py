"""tools/p8_identity.py: the automatic game identity check (P8 step 0, second pass; PLAN.md D130, D135).

A found video counts as a SportVU game only when the scorebug, read automatically at 3 moments, matches the
play-by-play of exactly one candidate game at all 3. Nothing is read by a person: an unreadable or ambiguous video is
rejected.

    python3 -m tools.p8_identity check SOURCE VIDEO_ID --duration S --candidates GAME [GAME ...]
    python3 -m tools.p8_identity calibrate        # the 7 known development videos, true and wrong candidate lists
    python3 -m tools.p8_identity ocr DIR          # worker, run under V3_CV_PYTHON (easyocr): frames -> tokens.json

Per moment (D135): one 8-second section at 25, 50 and 75 percent of the video, downloaded with yt-dlp
--download-sections (YouTube with the proof-of-origin token arguments). A moment with no readable scorebug gets one
retry 90 s later. Frames at 1 per second; the bottom quarter and the top sixth of each frame are read by easyocr. A
frame matches a game when a period, a game clock and two scores read in it equal the play-by-play score at that clock
(tools.p8_scorebug.matches, within its 2 s slack), in either team order; a moment matches a game when at least
MIN_FRAMES_PER_MOMENT of its frames do. Verdict: confirmed when exactly one candidate matches at every moment,
ambiguous when more than one does, rejected otherwise.

Every video id and candidate game passes tools.p8_common.guard (the D85 test set and the spent held-out games) before
anything is downloaded. Sections are kept in the P8 workspace, outside the repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.p8_common import atomic_write, guard, p8_root
from tools.p8_scorebug import matches, pbp

WHO = "tools.p8_identity"
MOMENTS = (0.25, 0.50, 0.75)
SECTION_S = 8
RETRY_SHIFT_S = 90
MIN_FRAMES_PER_MOMENT = 2
MAX_SCORE = 170
BANDS = (("bottom", 0.75, 1.0), ("top", 0.0, 1 / 6))

CLOCK_RE = re.compile(r"^(\d{1,2})[:.](\d{2})$")
SUBMIN_RE = re.compile(r"^:?(\d{1,2})\.(\d)$")
PERIOD_RE = re.compile(r"^(1ST|2ND|3RD|4TH)(?:\s*(?:QTR|Q|QUARTER))?$|^Q([1-4])$|^(OT|1OT|2OT|3OT)$", re.I)
INT_RE = re.compile(r"^\d{1,3}$")


# ── parsing and matching (pure, tested without OCR) ─────────────────────────────
def parse_tokens(texts: list[str]) -> dict:
    """OCR texts of one frame's bands -> {'periods', 'clocks' (seconds), 'ints'}. A token is split on spaces too,
    since easyocr often joins a scorebug's fields."""
    toks = []
    for t in texts:
        t = t.strip().replace("O", "0") if re.fullmatch(r"[0-9O:.]+", t.strip()) else t.strip()
        parts = t.split()
        toks += parts if len(parts) > 1 else [t]
    periods, clocks, ints = set(), set(), []
    for t in toks:
        m = CLOCK_RE.match(t)
        if m and int(m.group(1)) <= 12 and int(m.group(2)) < 60:
            clocks.add(int(m.group(1)) * 60 + int(m.group(2)))
            continue
        m = SUBMIN_RE.match(t)
        if m and int(m.group(1)) < 60:
            clocks.add(int(m.group(1)) + int(m.group(2)) / 10)
            continue
        m = PERIOD_RE.match(t)
        if m:
            if m.group(3):
                periods.add(5 if m.group(3).upper() in ("OT", "1OT") else 4 + int(m.group(3)[0]))
            else:
                periods.add(int((m.group(1) or m.group(2))[0]))
            continue
        if INT_RE.match(t) and int(t) <= MAX_SCORE:
            ints.append(int(t))
    return {"periods": sorted(periods), "clocks": sorted(clocks), "ints": ints}


def frame_matches(parsed: dict, events: list) -> bool:
    """True when some period, clock and ordered pair of the frame's integers equal the play-by-play score at that
    clock. A frame with no clock never matches; with no period read, the 4 regulation periods are tried."""
    if not parsed["clocks"] or len(parsed["ints"]) < 2:
        return False
    periods = parsed["periods"] or [1, 2, 3, 4]
    pairs = set(itertools.permutations(parsed["ints"], 2))
    return any(matches(events, p, c, a, h) for p in periods for c in parsed["clocks"] for a, h in pairs)


def moment_games(frames: list[dict], events_by_game: dict) -> list[str]:
    """The candidate games that at least MIN_FRAMES_PER_MOMENT frames of one moment match."""
    return [g for g, ev in events_by_game.items()
            if sum(frame_matches(f, ev) for f in frames) >= MIN_FRAMES_PER_MOMENT]


def verdict(moment_hits: list[list[str]], candidates: list[str]) -> dict:
    """confirmed (exactly one candidate matched at every moment, and there are at least 3 moments), ambiguous (more
    than one did) or rejected."""
    full = [g for g in candidates if len(moment_hits) >= len(MOMENTS) and all(g in h for h in moment_hits)]
    if len(full) == 1:
        return {"status": "confirmed", "game": full[0]}
    if len(full) > 1:
        return {"status": "ambiguous", "games": full}
    return {"status": "rejected", "game": None}


# ── download and OCR ──────────────────────────────────────────────────────────
def video_url(source: str, vid: str) -> str:
    return ("https://www.youtube.com/watch?v=%s" % vid) if source == "youtube" else ("https://www.bilibili.com/video/%s" % vid)


def work_dir(source: str, vid: str) -> Path:
    d = p8_root() / "identity" / ("%s_%s" % (source, vid))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_section(source: str, vid: str, start: int) -> Path | None:
    """One SECTION_S-second section from start (seconds) -> the mp4 path, or None when yt-dlp fails."""
    from harvest_driver import YTDLP_FMT, ytdlp_pot_args
    d = work_dir(source, vid)
    out = d / ("s%06d.mp4" % start)
    if out.exists():
        return out
    fmt = YTDLP_FMT if source == "youtube" else "bv*[height<=720]+ba/b[height<=720]"
    extra = ytdlp_pot_args() if source == "youtube" else []
    from sportvu.manifest import YTDLP
    cmd = [YTDLP, "--no-update", "-q", "--no-warnings", "--js-runtimes", "node", *extra,
           "-f", fmt, "--download-sections", "*%d-%d" % (start, start + SECTION_S), "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", str(d / ("s%06d.part.%%(ext)s" % start)), video_url(source, vid)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    part = d / ("s%06d.part.mp4" % start)
    if r.returncode != 0 or not part.exists():
        return None
    os.replace(part, out)
    return out


def frames_of(section: Path) -> list[Path]:
    """1 frame per second of a section, as jpg files next to it."""
    stem = section.with_suffix("")
    existing = sorted(section.parent.glob(stem.name + "_f*.jpg"))
    if existing:
        return existing
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(section), "-vf", "fps=1",
                    str(section.parent / (stem.name + "_f%02d.jpg"))], check=True)
    return sorted(section.parent.glob(stem.name + "_f*.jpg"))


def ocr_dir(d: Path) -> dict:
    """Worker (V3_CV_PYTHON): easyocr on the bands of every jpg in d -> {jpg name: [texts]}, written to tokens.json.
    A jpg already read in an earlier call keeps its tokens (same file, same reader), so a retry reads only new frames."""
    import cv2
    tok = Path(d) / "tokens.json"
    out = json.loads(tok.read_text()) if tok.exists() else {}
    todo = [j for j in sorted(Path(d).glob("*_f*.jpg")) if j.name not in out]
    if not todo:
        return out
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    for jpg in todo:
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        h = img.shape[0]
        texts = []
        for _, a, b in BANDS:
            band = img[int(a * h):int(b * h)]
            band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            texts += [t for _, t, conf in reader.readtext(band, detail=1) if conf >= 0.2]
        out[jpg.name] = texts
    atomic_write(Path(d) / "tokens.json", json.dumps(out, indent=1))
    return out


def run_ocr(d: Path) -> dict:
    """Run the OCR worker under V3_CV_PYTHON inside the heavy-job lock -> tokens by jpg name."""
    py = os.environ.get("V3_CV_PYTHON")
    if not py:
        raise SystemExit(f"{WHO}: set V3_CV_PYTHON to the Python that has easyocr (the frozen CV code's)")
    subprocess.run([sys.executable, "-m", "export.heavy", "--name", "p8 identity ocr %s" % d.name, "--",
                    py, "-m", "tools.p8_identity", "ocr", str(d)], check=True, cwd=Path(__file__).resolve().parents[1])
    return json.loads((d / "tokens.json").read_text())


def check(source: str, vid: str, duration: float, candidates: list[str], record: bool = True) -> dict:
    """The identity check of one video against its candidate games -> the record (also written as identity.json)."""
    from tools.research_testset import refuse_cv_test
    refuse_cv_test([vid, *candidates], WHO)
    guard([vid, *candidates], WHO)
    events = {g: pbp(g) for g in candidates}
    d = work_dir(source, vid)
    sections = []
    for frac in MOMENTS:
        start = int(frac * duration)
        tries = []
        for s in (start, start + RETRY_SHIFT_S):
            sec = download_section(source, vid, s)
            tries.append(s)
            if sec is None:
                continue
            sections.append({"moment": frac, "start": s, "path": sec, "frames": frames_of(sec)})
            break
    tokens = run_ocr(d) if sections else {}
    moments = []
    for frac in MOMENTS:
        secs = [s for s in sections if s["moment"] == frac]
        frames = [parse_tokens(tokens.get(f.name, [])) for s in secs for f in s["frames"]]
        hits = moment_games(frames, events)
        if not hits and secs:                                 # one retry 90 s later when nothing matched (D135)
            s2 = download_section(source, vid, secs[0]["start"] + RETRY_SHIFT_S)
            if s2 is not None:
                fr = frames_of(s2)
                tok2 = run_ocr(d)
                frames = [parse_tokens(tok2.get(f.name, [])) for f in fr]
                hits = moment_games(frames, events)
        moments.append({"moment": frac, "games": hits, "frames": len(frames)})
    v = verdict([m["games"] for m in moments], candidates)
    rec = {"source": source, "video_id": vid, "duration": duration, "candidates": candidates, "moments": moments,
           **v, "rule": "PLAN.md D130 and D135; tools.p8_identity", "date": dt.date.today().isoformat()}
    if record:
        atomic_write(d / "identity.json", json.dumps(rec, indent=1))
    return rec


def duration_of(source: str, vid: str) -> float:
    from sportvu.manifest import YTDLP
    r = subprocess.run([YTDLP, "--no-update", "--no-warnings", "--js-runtimes", "node",
                        "--skip-download", "--print", "%(duration)s", video_url(source, vid)],
                       capture_output=True, text=True, timeout=300)
    return float(r.stdout.strip().splitlines()[-1])


def calibrate() -> dict:
    """The 7 known development videos (tools.p8_common.dev_games): each against its own matchup's games must be
    confirmed as its true game, and each against another video's candidate list must not be confirmed. The pass bar
    (7 of 7 confirmed, 0 false confirmations) is fixed in docs/research/P8_step0.md before the search's candidates
    are checked."""
    from tools.p8_common import dev_games
    from tools.p8_search import contract_games, game_class
    games = contract_games()
    dev = dev_games()
    rows = []
    for tag, g in sorted(dev.items()):
        true = g["sportvu"]
        pair = {games[true]["away"], games[true]["home"]}
        cands = sorted(n for n, x in games.items() if {x["away"], x["home"]} == pair and game_class(n, x) == "training")
        rows.append({"tag": tag, "video_id": g["video_id"], "true": true, "candidates": cands})
    out = []
    for i, r in enumerate(rows):
        dur = duration_of("youtube", r["video_id"])
        own = check("youtube", r["video_id"], dur, r["candidates"], record=False)
        wrong_list = rows[(i + 1) % len(rows)]["candidates"]
        wrong = check("youtube", r["video_id"], dur, wrong_list, record=False)
        out.append({"tag": r["tag"], "video_id": r["video_id"], "true": r["true"], "own_candidates": r["candidates"],
                    "own": {k: own.get(k) for k in ("status", "game", "games", "moments")},
                    "wrong_candidates": wrong_list, "wrong": {k: wrong.get(k) for k in ("status", "game", "games")}})
    confirmed = sum(o["own"]["status"] == "confirmed" and o["own"]["game"] == o["true"] for o in out)
    false_conf = sum(o["own"]["status"] == "confirmed" and o["own"]["game"] != o["true"] for o in out) + \
        sum(o["wrong"]["status"] == "confirmed" for o in out)
    res = {"written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "videos": len(out),
           "confirmed_true": confirmed, "false_confirmations": false_conf,
           "passes": confirmed == len(out) and false_conf == 0, "rows": out}
    atomic_write(p8_root() / "identity" / "calibration.json", json.dumps(res, indent=1))
    return res


def batch(limit: int | None = None) -> dict:
    """Every metadata candidate of the search (tools.p8_search parts, training class, verdict candidate), once per
    video, checked against every training-class game of its two teams. A video that a guard refuses is recorded as
    guarded and never downloaded. Results: <P8 workspace>/identity/<source>_<id>/identity.json and batch.json."""
    from tools.p8_search import contract_games, game_class, parts_dir
    games = contract_games()
    videos = {}
    for p in sorted(parts_dir().glob("*.json")):
        part = json.loads(p.read_text())
        if part["class"] != "training":
            continue
        for s in part["sources"].values():
            for c in s.get("probed", []):
                if c.get("verdict") == "candidate":
                    v = videos.setdefault((c["source"], c["id"]), {"duration": c.get("duration"), "found_for": [],
                                                                   "length": c.get("length"), "height": c.get("height")})
                    v["found_for"].append(part["game"])
    out, done = [], 0
    for (source, vid), v in sorted(videos.items()):
        rec_path = p8_root() / "identity" / ("%s_%s" % (source, vid)) / "identity.json"
        g0 = games[v["found_for"][0]]
        pair = {g0["away"], g0["home"]}
        cands = sorted(n for n, x in games.items() if {x["away"], x["home"]} == pair and game_class(n, x) == "training")
        if rec_path.exists():
            out.append({**json.loads(rec_path.read_text()), "found_for": v["found_for"], "length": v["length"]})
            continue
        try:
            rec = check(source, vid, float(v["duration"]), cands)
        except SystemExit as e:
            rec = {"source": source, "video_id": vid, "status": "guarded", "note": str(e)}
        out.append({**rec, "found_for": v["found_for"], "length": v["length"]})
        done += 1
        print(f"[identity] {done} {source} {vid} {rec['status']} {rec.get('game')}", flush=True)
        if limit and done >= limit:
            break
    res = {"written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "videos": len(videos),
           "checked": len(out), "rows": [{k: r.get(k) for k in ("source", "video_id", "status", "game", "games",
                                                                 "found_for", "length", "candidates")} for r in out]}
    atomic_write(p8_root() / "identity" / "batch.json", json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("source", choices=("youtube", "bilibili"))
    c.add_argument("video_id")
    c.add_argument("--duration", type=float, required=True)
    c.add_argument("--candidates", nargs="+", required=True)
    sub.add_parser("calibrate")
    b = sub.add_parser("batch")
    b.add_argument("--limit", type=int)
    o = sub.add_parser("ocr")
    o.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        print(json.dumps(check(a.source, a.video_id, a.duration, a.candidates), indent=1))
    elif a.cmd == "batch":
        r = batch(a.limit)
        print(json.dumps({k: r[k] for k in ("videos", "checked")}, indent=1))
    elif a.cmd == "calibrate":
        r = calibrate()
        print(json.dumps({k: r[k] for k in ("videos", "confirmed_true", "false_confirmations", "passes")}, indent=1))
    else:
        ocr_dir(Path(a.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
