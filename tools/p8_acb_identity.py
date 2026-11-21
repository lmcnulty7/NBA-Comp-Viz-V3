"""tools/p8_acb_identity.py: the automatic identity check for ACB footage against the 10 SkillCorner games (P8 Level
4; PLAN.md D130 and D135 adapted by the lead; docs/research/P8_level4.md).

The rule is tools.p8_identity's, unchanged, with SkillCorner's game state in place of the SportVU play-by-play:
three 8-second sections at 25, 50 and 75 percent of the video (one retry 90 s later per moment), frames at 1 per
second, easyocr on the bottom quarter and the top sixth. A frame matches a game when a period, a game clock and both
scores read in it equal SkillCorner's score at that clock (tools.p8_scorebug.matches, 2 s slack, either team order);
a moment matches when at least 2 of its frames do. Confirmed only when all 3 moments match one SkillCorner game and
no other of the 10; anything else is rejected or ambiguous, never sent to a person.

SkillCorner's score at a clock is built from its own `shots` (made field goals, 2 or 3 points) and `free_throws`
(made, 1 point), each credited to its `offTeamId`; KNOWN_ISSUES.md says these reproduce the official final score in
all 10 published games, and `final_scores` checks that before any video is read.

    python3 -m tools.p8_acb_identity scores                       # the final-score check of the 10 games
    python3 -m tools.p8_acb_identity check SOURCE VIDEO_ID [--duration S] [--label TEXT]

Download, frames and OCR are tools.p8_identity's functions, imported: sections go to the P8 workspace, outside the
repository, and the OCR runs under the heavy-job lock with V3_CV_PYTHON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from tools import p8_identity as I
from tools.p8_common import atomic_write, guard, p8_root

WHO = "tools.p8_acb_identity"
# The ACB (Liga Endesa) scorebug prints the game clock and the shot clock together, read by easyocr as "09.40/24",
# "09.40124" or "08.4524": minutes and seconds, then the shot clock. tools.p8_identity's clock forms do not read it.
# This form was written after the first run on 00k9_ueP2bY returned no clock at all; it is therefore not evidence by
# itself, and every record keeps both readings (the unchanged parser and this one).
ACB_CLOCK_RE = re.compile(r"^(\d{1,2})[.:*](\d{2})(?:\D?\d{1,3})?$")
FIBA_QUARTER_MIN = 10                 # FIBA Official Basketball Rules 2026 v1.1, Article 8.1


def skillcorner_root() -> Path:
    """The cloned SkillCorner repository (data/matches.json, data/matches/<id>/...), outside this repository."""
    return Path(os.environ.get("V3_SKILLCORNER_ROOT") or p8_root().parent / "skillcorner" / "repo")


def matches_list(root: Path | None = None) -> list[dict]:
    return json.loads(((root or skillcorner_root()) / "data" / "matches.json").read_text())


def events_from(dynamic: dict, home_id: int, away_id: int) -> list:
    """SkillCorner dynamic events -> [(period, clock_s, "away-home")] after each scoring event, sorted period up and
    clock down, as tools.p8_scorebug.pbp returns them. A made shot scores 3 when `three` is true, else 2, at its
    `endGameClock` (or `startGameClock` when that is missing); a made free throw scores 1 at its `gameClock`. Points
    go to `offTeamId`. Events at the same period and clock keep the order shots first, then free throws."""
    pts = []
    for s in dynamic.get("shots", []):
        if s.get("outcome"):
            clock = s.get("endGameClock") if s.get("endGameClock") is not None else s.get("startGameClock")
            pts.append((int(s["period"]), float(clock), 0, s["offTeamId"], 3 if s.get("three") else 2))
    for f in dynamic.get("free_throws", []):
        if f.get("outcome"):
            pts.append((int(f["period"]), float(f["gameClock"]), 1, f["offTeamId"], 1))
    pts.sort(key=lambda t: (t[0], -t[1], t[2]))
    away = home = 0
    out = []
    for period, clock, _, team, n in pts:
        if team == home_id:
            home += n
        elif team == away_id:
            away += n
        else:
            raise ValueError(f"{WHO}: offTeamId {team} is neither team")
        out.append((period, clock, "%d-%d" % (away, home)))
    return out


def game_events(game_id: int, root: Path | None = None) -> list:
    root = root or skillcorner_root()
    m = {x["id"]: x for x in matches_list(root)}[int(game_id)]
    dyn = json.loads((root / "data" / "matches" / str(game_id) / f"{game_id}_dynamic_events.json").read_text())
    return events_from(dyn, m["home_team"]["id"], m["away_team"]["id"])


def final_scores(root: Path | None = None) -> list[dict]:
    """Each game's last event score against matches.json's final score."""
    rows = []
    for m in matches_list(root):
        ev = game_events(m["id"], root)
        a, h = (int(x) for x in ev[-1][2].split("-"))
        rows.append({"game": m["id"], "built": "%d-%d" % (h, a), "official": "%d-%d" % (m["home_score"], m["away_score"]),
                     "equal": (h, a) == (m["home_score"], m["away_score"]), "events": len(ev)})
    return rows


def parse_acb(texts: list[str]) -> dict:
    """tools.p8_identity.parse_tokens plus the ACB combined clock form (minutes at most FIBA_QUARTER_MIN)."""
    p = I.parse_tokens(texts)
    extra = set()
    for t in texts:
        for part in t.split():
            m = ACB_CLOCK_RE.match(part.strip())
            if m and int(m.group(1)) <= FIBA_QUARTER_MIN and int(m.group(2)) < 60:
                extra.add(int(m.group(1)) * 60 + int(m.group(2)))
    p["clocks"] = sorted(set(p["clocks"]) | extra)
    return p


def readings(tokens: dict, frame_names: list[str], events: dict) -> dict:
    """Both readings of one moment's frames: the unchanged tools.p8_identity parser and parse_acb."""
    out = {}
    for name, fn in (("unchanged", I.parse_tokens), ("acb_clock", parse_acb)):
        frames = [fn(tokens.get(f, [])) for f in frame_names]
        out[name] = {"games": I.moment_games(frames, events), "frames": len(frames),
                     "frames_with_clock": sum(1 for f in frames if f["clocks"]),
                     "frames_with_two_ints": sum(1 for f in frames if len(f["ints"]) >= 2)}
    return out


def moment_result(tries: list[dict]) -> dict:
    """Per reading, the games of the first downloaded section where that reading matched, else of the last
    downloaded section (D135: a section, then one retry 90 s later when nothing matched)."""
    done = [t for t in tries if t.get("downloaded")]
    out = {}
    for k in ("unchanged", "acb_clock"):
        hit = next((t[k]["games"] for t in done if t[k]["games"]), None)
        out[k] = hit if hit is not None else (done[-1][k]["games"] if done else [])
    return out


def check(source: str, vid: str, duration: float | None = None, label: str = "", record: bool = True) -> dict:
    """D130 and D135 as adapted for ACB: one video against all 10 SkillCorner games -> the record (also written as
    acb_identity.json in the video's identity folder)."""
    from tools.research_testset import refuse_cv_test
    refuse_cv_test([vid], WHO)
    guard([vid], WHO)
    bad = [r for r in final_scores() if not r["equal"]]
    if bad:
        raise SystemExit(f"{WHO}: SkillCorner score rebuild differs from the official score: {bad}")
    games = [str(m["id"]) for m in matches_list()]
    events = {g: game_events(int(g)) for g in games}
    duration = duration or I.duration_of(source, vid)
    d = I.work_dir(source, vid)
    moments = []
    for frac in I.MOMENTS:
        start = int(frac * duration)
        tries = []
        for s in (start, start + I.RETRY_SHIFT_S):              # one retry 90 s later when nothing matched (D135)
            sec = I.download_section(source, vid, s)
            if sec is None:
                tries.append({"start": s, "downloaded": False})
                continue
            names = [f.name for f in I.frames_of(sec)]
            r = readings(I.run_ocr(d), names, events)
            tries.append({"start": s, "downloaded": True, **r})
            if r["unchanged"]["games"] and r["acb_clock"]["games"]:
                break
        moments.append({"moment": frac, "tries": tries, "games": moment_result(tries)})
    verdicts = {k: I.verdict([m["games"][k] for m in moments], games) for k in ("unchanged", "acb_clock")}
    rec = {"source": source, "video_id": vid, "label": label, "duration": duration, "candidates": games,
           "moments": moments, "verdicts": verdicts, "rule": "PLAN.md D130 and D135 adapted to SkillCorner (tools.p8_acb_identity)",
           "date": dt.date.today().isoformat()}
    if record:
        atomic_write(d / "acb_identity.json", json.dumps(rec, indent=1))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scores")
    c = sub.add_parser("check")
    c.add_argument("source", choices=("youtube", "bilibili"))
    c.add_argument("video_id")
    c.add_argument("--duration", type=float)
    c.add_argument("--label", default="")
    a = ap.parse_args(argv)
    if a.cmd == "scores":
        for r in final_scores():
            print(json.dumps(r))
        return 0
    rec = check(a.source, a.video_id, a.duration, a.label)
    print(json.dumps({k: rec[k] for k in ("video_id", "label", "verdicts", "moments")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
