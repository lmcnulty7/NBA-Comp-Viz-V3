"""tools/p8_level4_search.py: metadata-only broadcast search for the 10 SkillCorner ACB 2025-26 games (P8 Level 4,
PLAN.md D137; docs/research/P8_level4.md).

For each game of the SkillCorner open basketball release (data/matches.json of github.com/SkillCorner/opendata-basketball)
it runs four YouTube query forms (Spanish and English) and one Bilibili form, keeps results whose title names both
teams, and probes each kept result with `yt-dlp --skip-download -j` for duration, channel, upload date and the
largest format height. No video is downloaded. The search reads no tracking frames, so it runs outside the
heavy-job lock.

Length classes, fixed before the search (as D134): full at least 3600 s, condensed 480 s to 3600 s, highlights under
480 s.

    python3 -m tools.p8_level4_search --matches <matches.json> --out <results.json> [--forms base|extra|all]
        [--max-probes N] [--long-only] [--games id,id]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unicodedata

from sportvu.manifest import YT_BASE as BASE  # the frozen yt-dlp command line, imported, not edited
THROTTLE_S = 1.0
MAX_PROBES = 8
FULL_MIN_S = 3600
CONDENSED_MIN_S = 480

# Common short names used in Spanish titles, and the words that identify a team in a title (accent folded).
SHORT = {
    1330: ("Baxi Manresa", ("manresa",)),
    1319: ("Coviran Granada", ("granada",)),
    1331: ("Casademont Zaragoza", ("zaragoza",)),
    1322: ("Rio Breogan", ("breogan",)),
    1334: ("UCAM Murcia", ("murcia",)),
    1336: ("Joventut Badalona", ("joventut", "badalona", "penya")),
    1315: ("Basquet Girona", ("girona",)),
    1316: ("San Pablo Burgos", ("burgos",)),
    1328: ("Dreamland Gran Canaria", ("gran canaria", "granca")),
    1326: ("Baskonia", ("baskonia",)),
    1332: ("MoraBanc Andorra", ("andorra",)),
    1335: ("Bilbao Basket", ("bilbao",)),
    1339: ("Unicaja Malaga", ("unicaja", "malaga")),
    1325: ("Valencia Basket", ("valencia",)),
    1338: ("Real Madrid", ("real madrid",)),
    1333: ("La Laguna Tenerife", ("tenerife", "laguna")),
    1340: ("Barca", ("barca", "barcelona")),
}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()


def run(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def team(t: dict) -> tuple[str, tuple[str, ...]]:
    return SHORT.get(t["id"], (t["name"], (fold(t["name"]).split()[-1],)))


MONTHS_ES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
             "noviembre", "diciembre")
MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December")


def queries(m: dict, forms: str = "base") -> list[tuple[str, str]]:
    """base: the first pass. extra: the second pass, written after the first pass returned 0 results for both dated
    forms on every game (cause not established); the extra forms write the date in words, with no digits joined by
    a hyphen or a slash."""
    hn, an = m["home_team"]["name"], m["away_team"]["name"]
    hs, as_ = team(m["home_team"])[0], team(m["away_team"])[0]
    y, mo, d = m["date_time"][:10].split("-")
    score = "%d-%d" % (m["home_score"], m["away_score"])
    if forms == "all":
        return queries(m, "base") + queries(m, "extra")
    if forms == "extra":
        return [
            ("youtube", "%s %s partido completo %d %s %s" % (hs, as_, int(d), MONTHS_ES[int(mo) - 1], y)),
            ("youtube", "%s vs %s full game %s %d %s" % (hs, as_, MONTHS_EN[int(mo) - 1], int(d), y)),
            ("youtube", "%s %s Liga Endesa 2025 26 completo" % (hs, as_)),
        ]
    return [
        ("youtube", "%s vs %s partido completo" % (hn, an)),
        ("youtube", "%s - %s (%s) Liga Endesa" % (hs, as_, score)),
        ("youtube", "%s vs %s %s-%s-%s full game" % (hs, as_, y, mo, d)),
        ("youtube", "Liga Endesa %s %s %s/%s/%s" % (hs, as_, d, mo, y)),
        ("bilibili", "%s %s ACB" % (hs, as_)),
    ]


def search(source: str, q: str) -> tuple[list[dict], str | None]:
    if source == "youtube":
        cmd = ["--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", "ytsearch10:" + q]
    else:
        cmd = ["--skip-download", "--print", "%(id)s\t%(duration)s\t%(uploader)s\t%(title)s", "bilisearch5:" + q]
    code, out, err = run(BASE + cmd)
    res = []
    for line in out.splitlines():
        p = line.split("\t", 3)
        if len(p) == 4:
            res.append({"id": p[0], "duration": None if p[1] in ("NA", "None") else float(p[1]),
                        "channel": p[2], "title": p[3], "source": source})
    if code != 0 and not res:
        return [], (err.strip().splitlines() or ["exit %d" % code])[-1][:200]
    return res, None


def probe(source: str, vid: str) -> dict:
    url = ("https://www.youtube.com/watch?v=%s" % vid) if source == "youtube" else ("https://www.bilibili.com/video/%s" % vid)
    code, out, err = run(BASE + ["--no-playlist", "--skip-download", "-j", url])
    if code != 0:
        return {"error": (err.strip().splitlines() or ["exit %d" % code])[-1][:200]}
    j = json.loads(out)
    heights = [f.get("height") for f in j.get("formats") or [] if f.get("height") and f.get("vcodec") not in (None, "none")]
    return {"title": j.get("title"), "channel": j.get("channel") or j.get("uploader"), "upload_date": j.get("upload_date"),
            "duration": j.get("duration"), "max_height": max(heights) if heights else None,
            "live_status": j.get("live_status"), "availability": j.get("availability")}


def length_class(d) -> str | None:
    if d is None:
        return None
    return "full" if d >= FULL_MIN_S else "condensed" if d >= CONDENSED_MIN_S else "highlights"


def names_both(title: str, m: dict) -> bool:
    t = fold(title)
    return all(any(k in t for k in team(m[side])[1]) for side in ("home_team", "away_team"))


def check(m: dict, forms: str = "base", max_probes: int = MAX_PROBES, long_only: bool = False) -> dict:
    out = {"game_id": m["id"], "date": m["date_time"][:10], "home": m["home_team"]["name"],
           "away": m["away_team"]["name"], "score": "%d-%d" % (m["home_score"], m["away_score"]),
           "queries": [], "candidates": []}
    seen = {}
    for source, q in queries(m, forms):
        res, err = search(source, q)
        out["queries"].append({"source": source, "query": q, "n_results": len(res), "error": err})
        for r in res:
            if names_both(r["title"], m) and (r["source"], r["id"]) not in seen:
                seen[(r["source"], r["id"])] = r
        time.sleep(THROTTLE_S)
    out["named_both"] = [dict(r) for r in seen.values()]
    todo = [k for k, r in seen.items() if not long_only or r.get("duration") is None or r["duration"] >= CONDENSED_MIN_S]
    for (source, vid) in todo[:max_probes]:
        r = seen[(source, vid)]
        p = probe(source, vid)
        c = {"source": source, "id": vid, "search_title": r["title"], **p}
        c["length_class"] = length_class(c.get("duration") or r.get("duration"))
        out["candidates"].append(c)
        time.sleep(THROTTLE_S)
    out["n_named_both"] = len(seen)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", choices=("base", "extra", "all"), default="base")
    ap.add_argument("--max-probes", type=int, default=MAX_PROBES)
    ap.add_argument("--long-only", action="store_true", help="probe only results listed at 480 s or more, or unlisted")
    ap.add_argument("--games", default="", help="comma-separated game ids (default: all)")
    a = ap.parse_args(argv)
    matches = json.load(open(a.matches))
    if a.games:
        keep = {int(x) for x in a.games.split(",")}
        matches = [m for m in matches if m["id"] in keep]
    results = []
    for m in matches:
        r = check(m, a.forms, a.max_probes, a.long_only)
        results.append(r)
        print("%s %s %s vs %s: %d named both, classes %s" % (
            r["game_id"], r["date"], r["home"], r["away"], r["n_named_both"],
            [c["length_class"] for c in r["candidates"]]), flush=True)
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
