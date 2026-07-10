#!/usr/bin/env python
"""
fetch_pbp.py — Tier 2 step 2: fetch + parse play-by-play for the clips' games.

Source: Basketball-Reference static PBP pages (stats.nba.com times out for
non-browser clients). Game identity was established from scorebug fingerprints
— every (period, clock, score) anchor read off the broadcast bug matched the
candidate game's PBP exactly (4/4), which is as deterministic as game ID gets:
  201602270OKC  GSW @ OKC 2016-02-27 (ESPN Saturday Primetime, the OT game)
  201206070BOS  MIA @ BOS 2012-06-07 (ECF Game 6)

Output: data/pbp/<code>.json — one event per row:
  {period, clock_s (game clock, seconds remaining), team ("away"/"home"/None),
   desc, points (scored on this event), score ("away-home" running),
   kind (shot_made/shot_missed/ft_made/ft_missed/turnover/rebound/foul/other)}

CLIP_GAME maps each clip to its game + real team names + which side wears the
LIGHT kit (pre-2017 NBA: home wears white/light — how team A/B ids from the
color clusterer get real names downstream).
"""
from __future__ import annotations

import json
import logging
import re

import requests
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_pbp")

PBP_DIR = config.PROJECT_ROOT / "data" / "pbp"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

GAMES = {
    "201602270OKC": {"away": "GSW", "home": "OKC", "date": "2016-02-27"},
    "201206070BOS": {"away": "MIA", "home": "BOS", "date": "2012-06-07"},
    # Phase B (from data/harvest/games.json; fingerprint-verified on fetch)
    "201606190GSW": {"away": "CLE", "home": "GSW", "date": "2016-06-19"},
    "201605160GSW": {"away": "OKC", "home": "GSW", "date": "2016-05-16"},
    "201605280OKC": {"away": "GSW", "home": "OKC", "date": "2016-05-28"},
    "201612050GSW": {"away": "IND", "home": "GSW", "date": "2016-12-05"},
    "201601250GSW": {"away": "SAS", "home": "GSW", "date": "2016-01-25"},
    "201604010GSW": {"away": "BOS", "home": "GSW", "date": "2016-04-01"},
    "201706120GSW": {"away": "CLE", "home": "GSW", "date": "2017-06-12"},
    "201501230GSW": {"away": "SAC", "home": "GSW", "date": "2015-01-23"},
    "201302270NYK": {"away": "GSW", "home": "NYK", "date": "2013-02-27"},
    "201701100GSW": {"away": "MIA", "home": "GSW", "date": "2017-01-10"},
    "201605110GSW": {"away": "POR", "home": "GSW", "date": "2016-05-11"},
    "201505210GSW": {"away": "HOU", "home": "GSW", "date": "2015-05-21"},
    "201612250CLE": {"away": "GSW", "home": "CLE", "date": "2016-12-25",
                     # 2016 Christmas special uniforms: home CLE is DARK —
                     # the pre-2017 light=home rule inverts for this game
                     "light_team_name": "GSW"},
}
CLIP_GAME = {
    "curry_q1_clip": "201602270OKC",
    "curry_classic_clip": "201602270OKC",
    "clip_10m00_18m00": "201206070BOS",
    "clip_26m00_34m00": "201206070BOS",
    "clip_40m00_48m00": "201206070BOS",
    "clip_55m00_63m00": "201206070BOS",
    "clip_70m00_78m00": "201206070BOS",
}
LIGHT_IS_HOME = True   # pre-2017 NBA default: home wears the light kit.
                       # Per-game override: GAMES[code]["light_team_name"] (e.g. special
                       # Christmas uniforms where the home kit is dark).


def game_for_clip(stem: str) -> str:
    """Game code for a clip stem: static registry first, then the harvest games
    registry (Phase B sections are named <tag>_sNN)."""
    if stem in CLIP_GAME:
        return CLIP_GAME[stem]
    import re
    tag = re.sub(r"_s\d+$", "", stem)
    reg = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    if tag in reg and "game_code" in reg[tag]:
        return reg[tag]["game_code"]
    raise KeyError(f"no game registered for {stem!r} (tag {tag!r})")


def light_team_name(game: dict) -> str:
    """Which REAL team wears the light kit in this game."""
    return game.get("light_team_name") or (game["home"] if LIGHT_IS_HOME else game["away"])


def video_path(stem: str):
    """Resolve a clip stem to its video file (harvest downloads first)."""
    h = config.PROJECT_ROOT / "data" / "harvest" / "video" / f"{stem}.mp4"
    return h if h.exists() else config.VIDEO_DIR / f"{stem}.mp4"


def classify(desc: str) -> str:
    d = desc.lower()
    if "free throw" in d:
        return "ft_missed" if "misses" in d else "ft_made"
    if "makes" in d:
        return "shot_made"
    if "misses" in d:
        return "shot_missed"
    if "turnover" in d:
        return "turnover"
    if "rebound" in d:
        return "rebound"
    if "foul" in d:
        return "foul"
    return "other"


def fetch_game(code: str) -> list[dict]:
    url = f"https://www.basketball-reference.com/boxscores/pbp/{code}.html"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    table = BeautifulSoup(r.text, "html.parser").find("table", id="pbp")
    events, period = [], 0
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) == 1 and re.match(r"(1st|2nd|3rd|4th|OT)", cells[0]):
            period += 1
            continue
        if len(cells) != 6 or not re.match(r"\d+:\d+", cells[0]):
            continue
        t, away_d, away_p, score, home_p, home_d = cells
        m, s = t.split(":")
        clock_s = int(m) * 60 + float(s)
        team, desc, pts_txt = (("away", away_d, away_p) if away_d.strip()
                               else ("home", home_d, home_p) if home_d.strip()
                               else (None, "", ""))
        pm = re.search(r"\+(\d)", pts_txt)
        events.append({"period": period, "clock_s": round(clock_s, 1), "team": team,
                       "desc": desc, "points": int(pm.group(1)) if pm else 0,
                       "score": score if "-" in score else None,
                       "kind": classify(desc)})
    return events


def main() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    for code, meta in GAMES.items():
        out = PBP_DIR / f"{code}.json"
        if out.exists():
            log.info("%s already fetched", code)
            continue
        events = fetch_game(code)
        out.write_text(json.dumps({"meta": meta, "events": events}, indent=1))
        n_periods = max(e["period"] for e in events)
        log.info("%s: %d events, %d periods → %s", code, len(events), n_periods, out)


if __name__ == "__main__":
    main()
