"""tests/fixtures/bref/make_synthetic.py: shot chart pages in basketball-reference's markup, built from real play-by-play.

www.basketball-reference.com was not reachable from the cloud session that wrote collect/bref_shots.py (the
environment's network policy denied the host), so no real page could be saved there. These pages carry the REAL
field-goal attempts of data/pbp/<code>.json (fetch_pbp.py: player, period, clock, make or miss, value, distance)
placed at a deterministic angle on the 500 x 472 px chart, in the marker layout the collector parses. They test the
parser, the court mapping and the run loop, not the pixel frame of the live site: replace them with real pages
(on the Mac: python -m collect.bref_shots --games 12.16.2015.PHX.at.GSW 02.27.2016.GSW.at.OKC --save-pages
tests/fixtures/bref) and tests/test_bref_shots.py runs its generic checks on whatever pages are present.

  python tests/fixtures/bref/make_synthetic.py          # rewrites <code>.synthetic.html for the codes below
"""
from __future__ import annotations
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from collect.bref_shots import HOOP_PX, PX_PER_FT, CHART_PX   # noqa: E402
from sportvu.colab_contract import BR_ABBR                     # noqa: E402

CODES = ("201512160GSW", "201602270OKC")        # PHX at GSW (SportVU, held-out game); GSW at OKC (the OT game)
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
SHOT_RE = re.compile(r"^(?P<player>.+?) (?P<verb>makes|misses) (?P<pts>[23])-pt .*?(?:from (?P<dist>\d+) ft|(?P<rim>at rim))")


def period_label(p: int) -> str:
    return "%s Qtr" % ORDINAL[p] if p <= 4 else "%s OT" % ORDINAL.get(p - 4, "%dth" % (p - 4))


def slug_for(name: str) -> str:
    """'S. Curry' -> 'currys01' (a stable, bref-looking slug for the fixture)."""
    first, last = name.split(" ", 1) if " " in name else ("x", name)
    return re.sub(r"[^a-z]", "", last.lower())[:5] + re.sub(r"[^a-z]", "", first.lower())[:2] + "01"


def place(dist_ft: float, k: int) -> tuple[int, int]:
    """A marker dist_ft from the hoop at a deterministic angle, kept inside the chart."""
    for j in range(36):
        theta = math.radians(-80 + ((k * 37 + j * 10) % 161))     # about the downward normal, within the half court
        left = HOOP_PX[0] + dist_ft * PX_PER_FT * math.sin(theta)
        top = HOOP_PX[1] + dist_ft * PX_PER_FT * math.cos(theta)
        if 2 <= left <= CHART_PX[0] - 2 and 2 <= top <= CHART_PX[1] - 2:
            return int(round(left)), int(round(top))
    return int(round(HOOP_PX[0])), int(round(HOOP_PX[1] + dist_ft * PX_PER_FT))


def build(code: str) -> str:
    doc = json.loads((ROOT / "data" / "pbp" / ("%s.json" % code)).read_text())
    meta = doc["meta"]
    br = {"home": BR_ABBR.get(meta["home"], meta["home"]), "away": BR_ABBR.get(meta["away"], meta["away"])}
    areas = {br["away"]: [], br["home"]: []}
    k = 0
    for e in doc["events"]:
        if e["kind"] not in ("shot_made", "shot_missed") or e["team"] not in br:
            continue
        m = SHOT_RE.match(e["desc"])
        if not m:
            raise ValueError("unparsed shot description: %r" % e["desc"])
        dist = float(m.group("dist")) if m.group("dist") else 0.0
        left, top = place(dist, k); k += 1
        mm, ss = divmod(e["clock_s"], 60)
        head = "%s, %d:%04.1f remaining" % (period_label(e["period"]), int(mm), ss)
        body = "%s %s %s-pointer from %d ft" % (m.group("player"), "made" if e["kind"] == "shot_made" else "missed", m.group("pts"), int(dist))
        team = br[e["team"]]
        tail = ""
        if e.get("score"):
            a, h = e["score"].split("-")
            mine, theirs = (int(h), int(a)) if e["team"] == "home" else (int(a), int(h))
            tail = "%s %s %s-%s" % (team, "now leads" if mine > theirs else "trails" if mine < theirs else "tied", mine, theirs)
        tip = "<br>".join(p for p in (head, body, tail) if p)
        cls = "tooltip %s p-%s q-%d" % ("make" if e["kind"] == "shot_made" else "miss", slug_for(m.group("player")), e["period"])
        areas[team].append('<div style="top:%dpx;left:%dpx;" tip="%s" class="%s">%s</div>' % (top, left, tip, cls, "&#9679;" if e["kind"] == "shot_made" else "&#10005;"))
    parts = ['<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>%s at %s Shot Chart, %s | SYNTHETIC FIXTURE</title></head><body>'
             % (meta["away"], meta["home"], meta["date"]),
             '<!-- synthetic page: real attempts of data/pbp/%s.json at generated positions (tests/fixtures/bref/make_synthetic.py) -->' % code,
             '<div id="content">']
    for team, divs in areas.items():
        parts.append('<div id="shooting-%s"><h2>%s shooting</h2><div class="shot-area" id="shots-%s">' % (team, team, team))
        parts.extend(divs)
        parts.append('</div></div>')
    parts.append('<div id="all_shooting-%s"><!--\n<table class="stats_table" id="shooting-%s"><tr><td>wrapped in a comment, as basketball-reference does</td></tr></table>\n--></div>' % (br["home"], br["home"]))
    parts.append('</div></body></html>\n')
    return "\n".join(parts)


if __name__ == "__main__":
    for code in CODES:
        p = HERE / ("%s.synthetic.html" % code)
        p.write_text(build(code))
        print(p, p.stat().st_size, "bytes")
