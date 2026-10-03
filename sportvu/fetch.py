"""
sportvu/fetch.py: one SportVU game log -> flat moments table (FIX_PLAN B1).

Source: github.com/linouk23/NBA-Player-Movements, data/2016.NBA.Raw.SportVU.Game.Logs/<MM.DD.YYYY.AWAY.at.HOME>.7z
(2015-16 season, 2015-10-27 .. 2016-01-23). Each archive holds one JSON:
  {"gameid", "gamedate", "events": [{"eventId", "visitor": {teamid, players[...]}, "home": {...},
   "moments": [[quarter, timestamp_ms, game_clock_s, shot_clock_s, None,
                [[team_id, player_id, x_ft, y_ft, z_ft] x 11 (ball first, team_id -1)]], ...]}]}
25 Hz, court coordinates x 0..94 ft, y 0..50 ft (corner origin, same frame as court/court33).
Events overlap in time (the same moments appear in consecutive events), so moments are deduped on
(quarter, game_clock_s, first player x). Nothing from here is committed: data/sportvu/ is gitignored.

  python -m sportvu.fetch 12.16.2015.PHX.at.GSW        -> data/sportvu/<game>.json (raw) + <game>_moments.json + report
"""
from __future__ import annotations
import io, json, sys, urllib.request
from pathlib import Path
import numpy as np
import config

RAW_BASE = "https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/data/2016.NBA.Raw.SportVU.Game.Logs/"
SV_DIR = config.PROJECT_ROOT / "data" / "sportvu"
HZ = 25.0
GAP_S = 2.0


def download(game: str) -> Path:
    """Fetch <game>.7z and extract its JSON to data/sportvu/<game>.json (cached)."""
    import py7zr
    SV_DIR.mkdir(parents=True, exist_ok=True)
    out = SV_DIR / (game + ".json")
    if out.exists():
        return out
    url = RAW_BASE + game + ".7z"
    blob = urllib.request.urlopen(url, timeout=120).read()
    tmp = SV_DIR / ("_extract_" + game)
    with py7zr.SevenZipFile(io.BytesIO(blob), "r") as z:
        names = [n for n in z.getnames() if n.lower().endswith(".json")]
        z.extract(path=tmp, targets=names)
    src = next(tmp.rglob("*.json"))
    out.write_bytes(src.read_bytes())
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return out


def parse(raw_path: Path) -> dict:
    """Flat, deduped moments plus the roster. Returns {"game", "date", "teams", "players", "moments"}."""
    g = json.loads(raw_path.read_text())
    teams, players = {}, {}
    for ev in g["events"]:
        for side in ("visitor", "home"):
            t = ev[side]
            teams[int(t["teamid"])] = {"abbr": t["abbreviation"], "name": t["name"], "side": side}
            for p in t["players"]:
                players[int(p["playerid"])] = {"name": "%s %s" % (p["firstname"], p["lastname"]),
                                               "jersey": p.get("jersey"), "team": int(t["teamid"])}
    seen, moments = set(), []
    for ev in g["events"]:
        for m in ev["moments"]:
            q, ts, clock, shot, _, ents = m[0], m[1], m[2], m[3], m[4], m[5]
            if clock is None or not ents:
                continue
            key = (q, round(clock, 2), round(ents[-1][2], 2) if len(ents) else 0)
            if key in seen:
                continue
            seen.add(key)
            ball = [e for e in ents if e[0] == -1]
            pl = [e for e in ents if e[0] != -1]
            moments.append({"q": int(q), "clock": float(clock), "shot": None if shot is None else float(shot),
                            "ts": int(ts),
                            "ball": [round(ball[0][2], 2), round(ball[0][3], 2), round(ball[0][4], 2)] if ball else None,
                            "players": [[int(e[0]), int(e[1]), round(e[2], 2), round(e[3], 2)] for e in pl]})
    moments.sort(key=lambda m: (m["q"], -m["clock"], m["ts"]))
    return {"game": raw_path.stem, "gameid": g.get("gameid"), "date": g.get("gamedate"),
            "teams": teams, "players": players, "moments": moments}


def coverage_report(parsed: dict) -> dict:
    """Per quarter: moments, seconds of clock covered, list of gaps > GAP_S (clock seconds)."""
    rep = {"game": parsed["game"], "date": parsed["date"], "teams": {str(k): v["abbr"] for k, v in parsed["teams"].items()},
           "n_players": len(parsed["players"]), "moments": len(parsed["moments"]), "quarters": {}}
    for q in sorted({m["q"] for m in parsed["moments"]}):
        cl = np.array([m["clock"] for m in parsed["moments"] if m["q"] == q])
        cl_desc = -np.sort(-cl)
        d = cl_desc[:-1] - cl_desc[1:]
        gaps = [[round(float(cl_desc[i]), 1), round(float(cl_desc[i + 1]), 1)] for i in np.where(d > GAP_S)[0]]
        n10 = sum(1 for m in parsed["moments"] if m["q"] == q and len(m["players"]) == 10)
        rep["quarters"][str(q)] = {"moments": int(len(cl)), "clock_span": [round(float(cl.max()), 1), round(float(cl.min()), 1)],
                                   "covered_s": round(float(len(cl) / HZ), 1), "gaps_over_%gs" % GAP_S: gaps,
                                   "moments_with_10_players": n10}
    return rep


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "12.16.2015.PHX.at.GSW"
    raw = download(game)
    parsed = parse(raw)
    (SV_DIR / (game + "_moments.json")).write_text(json.dumps(parsed))
    rep = coverage_report(parsed)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / ("sportvu_fetch_%s.json" % game)).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
