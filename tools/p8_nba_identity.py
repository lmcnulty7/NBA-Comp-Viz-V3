"""tools/p8_nba_identity.py: the automatic identity check for 2025-26 NBA footage (P8 Level 4; PLAN.md D141;
docs/research/P8_level4.md section 11).

The rule is tools.p8_identity's (D130 as fixed in D138), imported and unchanged, with the public play-by-play mirror
in place of the SportVU play-by-play: three 8-second sections at 25, 50 and 75 percent of the video, one retry 90 s
later per moment, frames at 1 per second, easyocr on the bottom quarter and the top sixth. A frame matches a game
when a period, a game clock and both scores read in it equal the play-by-play score at that clock (2 s slack, either
team order); a moment matches when at least 2 of its frames do; confirmed only when all 3 moments match exactly one
candidate. Candidates for a video: every 2025-26 game (regular season and playoffs) between its two teams.

Play-by-play: `cdnnba_2025` and `cdnnba_po_2025` of github.com/shufinskiy/nba_data (period, clock, scoreHome,
scoreAway on every action). `events` turns them, under the heavy-job lock, into a compact file of score changes in
the P8 workspace; the check reads only that file.

    python3 -m export.heavy --name "nba pbp events" -- python3 -m tools.p8_nba_identity events
    python3 -m tools.p8_nba_identity calibrate                      # D141's bar on 5 league extended highlights
    python3 -m tools.p8_nba_identity check VIDEO_ID --game GAME_ID  # one video; candidates: every game of the pair
    python3 -m tools.p8_nba_identity candidates                     # the probed full-game candidates

Every video id passes refuse_cv_test and the held-out guard before anything is downloaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from tools import p8_identity as I
from tools import p8_nba2526_search as S
from tools.p8_common import atomic_write, guard

WHO = "tools.p8_nba_identity"
CLOCK_ISO_RE = re.compile(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")

# D141's calibration, fixed before it runs: 5 of the league channel's "EXTENDED: ... FULL GAME HIGHLIGHTS" uploads of
# regular-season games, drawn with random.Random(141).sample from the 162 matched ones sorted by video id (every one
# has at least 2 games against the same opponent, so uniqueness is tested). Identity is the league's own title.
# Bar: 5 of 5 confirmed as the titled game against every 2025-26 game of the same two teams, 0 confirmed as another.
CALIBRATION = (("qXCV4eissO4", "0022500332"), ("1krUPLClXYA", "0022500714"), ("tXKAuBt2vjw", "0022500300"),
               ("nb402apDzh8", "0022500881"), ("GDpD5lL9c20", "0022500227"))


def clock_s(iso: str) -> float | None:
    m = CLOCK_ISO_RE.match(str(iso or ""))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def score_changes(rows) -> list:
    """[(period, clock_s, scoreHome, scoreAway)] in play order -> [(period, clock_s, "away-home")] at each change,
    as tools.p8_scorebug.pbp returns events (sorted period up, clock down; the last score carries forward)."""
    out, last = [], None
    for p, c, h, a in rows:
        if c is None or h != h or a != a:              # NaN check without numpy
            continue
        sc = "%d-%d" % (int(a), int(h))
        if sc != last:
            out.append((int(p), float(c), sc))
            last = sc
    return sorted(out, key=lambda e: (e[0], -e[1]))


def events_path() -> Path:
    return S.out_dir() / "pbp_events.json"


def build_events() -> dict:
    """Under the heavy-job lock: both mirror files -> {game_id (10 digits): events}."""
    import pandas as pd
    out = {}
    for f in ("cdnnba_2025/cdnnba_2025.csv", "cdnnba_po_2025/cdnnba_po_2025.csv"):
        d = pd.read_csv(S.shots_root() / f, usecols=["gameId", "period", "clock", "scoreHome", "scoreAway", "orderNumber"])
        d = d.sort_values(["gameId", "orderNumber"], kind="stable")
        for gid, g in d.groupby("gameId", sort=False):
            rows = [(p, clock_s(c), h, a) for p, c, h, a in g[["period", "clock", "scoreHome", "scoreAway"]].itertuples(index=False)]
            out["%010d" % int(gid)] = score_changes(rows)
        del d
    atomic_write(events_path(), json.dumps(out))
    return out


def load_events() -> dict:
    return {k: [tuple(e) for e in v] for k, v in json.loads(events_path().read_text()).items()}


def pair_games(game_id: str) -> list[str]:
    gs = S.games()
    by = {g["game_id"]: g for g in gs}
    g = by[game_id]
    return sorted(x["game_id"] for x in S.pair_index(gs)[frozenset((g["home"], g["away"]))])


def duration_with_retry(source: str, vid: str, tries: int = 3) -> float:
    """tools.p8_identity.duration_of, retried: yt-dlp sometimes prints nothing (seen twice on 2026-11-21)."""
    import time
    for k in range(tries):
        try:
            return I.duration_of(source, vid)
        except (IndexError, ValueError):
            if k == tries - 1:
                raise
            time.sleep(30)


def check(vid: str, game_id: str, duration: float | None = None, label: str = "", record: bool = True) -> dict:
    """One video against every 2025-26 game of the two teams of game_id (the title's game) -> the record."""
    from tools.research_testset import refuse_cv_test
    refuse_cv_test([vid], WHO)
    guard([vid], WHO)
    allev = load_events()
    cands = pair_games(game_id)
    events = {g: allev[g] for g in cands if g in allev}
    missing = [g for g in cands if g not in allev]
    duration = duration or duration_with_retry("youtube", vid)
    d = I.work_dir("youtube", vid)
    moments = []
    for frac in I.MOMENTS:
        start = int(frac * duration)
        tries = []
        for s in (start, start + I.RETRY_SHIFT_S):              # one retry 90 s later when nothing matched (D135)
            if s + I.SECTION_S > duration:
                tries.append({"start": s, "downloaded": False, "beyond_end": True})
                continue
            sec = I.download_section("youtube", vid, s)
            if sec is None:
                tries.append({"start": s, "downloaded": False})
                continue
            names = [f.name for f in I.frames_of(sec)]
            tok = I.run_ocr(d)
            frames = [I.parse_tokens(tok.get(n, [])) for n in names]
            hits = I.moment_games(frames, events)
            tries.append({"start": s, "downloaded": True, "games": hits, "frames": len(frames),
                          "frames_with_clock": sum(1 for f in frames if f["clocks"])})
            if hits:
                break
        last = next((t for t in reversed(tries) if t.get("downloaded")), None)
        moments.append({"moment": frac, "tries": tries, "games": last["games"] if last else []})
    v = I.verdict([m["games"] for m in moments], cands)
    rec = {"video_id": vid, "label": label, "title_game": game_id, "duration": duration, "candidates": cands,
           "candidates_without_pbp": missing, "moments": moments, **v,
           "rule": "PLAN.md D130, D138 and D141 (tools.p8_identity, imported)", "date": dt.date.today().isoformat()}
    if record:
        atomic_write(d / "nba_identity.json", json.dumps(rec, indent=1))
    return rec


def calibrate() -> dict:
    rows = []
    for vid, game in CALIBRATION:
        r = check(vid, game, label="calibration (league extended highlights)")
        rows.append({"video_id": vid, "truth": game, "status": r["status"], "game": r.get("game"),
                     "games": r.get("games"), "moments": [m["games"] for m in r["moments"]]})
    ok = sum(1 for r in rows if r["status"] == "confirmed" and r["game"] == r["truth"])
    false = sum(1 for r in rows if r["status"] == "confirmed" and r["game"] != r["truth"])
    out = {"rows": rows, "confirmed": ok, "of": len(CALIBRATION), "false_confirmations": false,
           "passes": ok == len(CALIBRATION) and false == 0,
           "bar": "5 of 5 confirmed as the titled game, 0 confirmed as another game (D141)",
           "date": dt.date.today().isoformat()}
    atomic_write(S.out_dir() / "nba_identity_calibration.json", json.dumps(out, indent=1))
    return out


def full_candidates() -> list[dict]:
    """The probed full-game candidates of tools.p8_nba2526_search at 720p or more (YouTube)."""
    probes = S.load("probes.json", {})
    out = []
    for c in S.candidates():
        p = probes.get("%s:%s" % (c["source"], c["id"]), {})
        if c["cls"] == "full" and c["source"] == "youtube" and (p.get("max_height") or 0) >= S.MIN_HEIGHT:
            out.append({"id": c["id"], "game_id": c["game_id"], "duration": p.get("duration"), "channel": p.get("channel")})
    return out


def run_candidates() -> dict:
    out = {}
    for c in full_candidates():
        r = check(c["id"], c["game_id"], c["duration"], label="full-game candidate (%s)" % c["channel"])
        out[c["id"]] = {"title_game": c["game_id"], "status": r["status"], "game": r.get("game"),
                        "games": r.get("games"), "moments": [m["games"] for m in r["moments"]],
                        "candidates_without_pbp": r["candidates_without_pbp"]}
        atomic_write(S.out_dir() / "nba_identity_candidates.json", json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("events")
    sub.add_parser("calibrate")
    sub.add_parser("candidates")
    c = sub.add_parser("check")
    c.add_argument("video_id")
    c.add_argument("--game", required=True)
    c.add_argument("--duration", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "events":
        ev = build_events()
        print("%d games, %d score changes" % (len(ev), sum(len(v) for v in ev.values())))
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate(), indent=1))
    elif a.cmd == "candidates":
        print(json.dumps(run_candidates(), indent=1))
    else:
        r = check(a.video_id, a.game, a.duration)
        print(json.dumps({k: r[k] for k in ("video_id", "status", "moments")} | {"game": r.get("game")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
