"""tools/p8_nba2526_search.py: metadata-only search for public broadcasts of 2025-26 NBA games (P8 Level 4; the
lead's decision (e) after docs/research/P8_level4.md). No video is downloaded.

Games are the 1,230 regular-season and 85 playoff games of the public shot-chart mirror (shotdetail_2025 and
shotdetail_po_2025 of github.com/shufinskiy/nba_data). Videos come from three places:

  channels   the full upload list (yt-dlp --flat-playlist) of each channel given, for example the channels that the
             discovery searches show posting full games, and the league's own channel
  discover   YouTube searches for a seeded random sample of games, "{away} vs {home} full game {Month d, yyyy}",
             10 results each, to find channels and to estimate coverage beyond the listed channels
  bilibili   a few Bilibili searches (Chinese forms), 10 results each, each extracted (no flat listing there)

Matching and classes, fixed before any listing is read:
  - a title names a game when it names exactly two teams (nicknames, word bounded) that played each other, and either
    a date written in the title equals the game date (one day of slack for time zones), or the title has no date and
    the upload date is 0 to 3 days after exactly one game of that pair
  - full: at least 2,400 s (the frozen minimum) with none of the rejection words
  - extended: the league channel's "EXTENDED: ... FULL GAME HIGHLIGHTS" uploads, at least 480 s (condensed, D134)
  - league_highlights: the league channel's other "FULL GAME HIGHLIGHTS" uploads (sampled possessions only)
  - quarter: one full quarter uploaded on its own ("FULL ... 1ST-QTR"), at least 300 s; added after a channel
    listing showed games posted a quarter at a time; a game counts as complete in quarters when all 4 are found
  - rejection words: highlight (except in extended), recap, classic, 2k, simulation, reaction, watch party,
    live score, preview, prediction, breakdown, podcast, mix, top plays
  - usable: full or extended with a probed max height of at least 720 (probe: yt-dlp --skip-download -j)

    python3 -m tools.p8_nba2526_search channels URL [URL ...] [--end N]
    python3 -m tools.p8_nba2526_search discover --n 60 --seed 2526
    python3 -m tools.p8_nba2526_search bilibili
    python3 -m tools.p8_nba2526_search probe [--extended-sample 20]
    python3 -m tools.p8_nba2526_search summary

Output under <P8 workspace>/../level4/nba2526/ (outside the repository): listings, matches, probes, summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from sportvu.manifest import YT_BASE
from tools.p8_common import atomic_write, p8_root

WHO = "tools.p8_nba2526_search"
THROTTLE_S = 1.0
FULL_MIN_S = 2400
EXTENDED_MIN_S = 480
DATE_SLACK_DAYS = 1
UPLOAD_WINDOW_DAYS = 3
MIN_HEIGHT = 720
LEAGUE_CHANNELS = ("NBA", "@NBA")              # a flat channel listing names the channel by its handle

NICK = {
    "ATL": ("hawks",), "BKN": ("nets",), "BOS": ("celtics",), "CHA": ("hornets",), "CHI": ("bulls",),
    "CLE": ("cavaliers", "cavs"), "DAL": ("mavericks", "mavs"), "DEN": ("nuggets",), "DET": ("pistons",),
    "GSW": ("warriors",), "HOU": ("rockets",), "IND": ("pacers",), "LAC": ("clippers",), "LAL": ("lakers",),
    "MEM": ("grizzlies",), "MIA": ("heat",), "MIL": ("bucks",), "MIN": ("timberwolves", "wolves"),
    "NOP": ("pelicans",), "NYK": ("knicks",), "OKC": ("thunder",), "ORL": ("magic",), "PHI": ("76ers", "sixers"),
    "PHX": ("suns",), "POR": ("trail blazers", "blazers"), "SAC": ("kings",), "SAS": ("spurs",),
    "TOR": ("raptors",), "UTA": ("jazz",), "WAS": ("wizards",),
}
ZH = {"ATL": "老鹰", "BOS": "凯尔特人", "BKN": "篮网", "CHA": "黄蜂", "CHI": "公牛", "CLE": "骑士", "DAL": "独行侠",
      "DEN": "掘金", "DET": "活塞", "GSW": "勇士", "HOU": "火箭", "IND": "步行者", "LAC": "快船", "LAL": "湖人",
      "MEM": "灰熊", "MIA": "热火", "MIL": "雄鹿", "MIN": "森林狼", "NOP": "鹈鹕", "NYK": "尼克斯", "OKC": "雷霆",
      "ORL": "魔术", "PHI": "76人", "PHX": "太阳", "POR": "开拓者", "SAC": "国王", "SAS": "马刺", "TOR": "猛龙",
      "UTA": "爵士", "WAS": "奇才"}
REJECT_RE = re.compile(r"(highlight|recap|classic|\b2k|simulation|reaction|watch ?party|live score|preview|"
                       r"prediction|breakdown|podcast|\bmix\b|top plays|\bnba live\b|play.?by.?play|scoreboard|"
                       r"集锦|模拟)", re.I)
SIM_RE = re.compile(r"(\b2k\d*\b|nba2k|simulation|\bsim\b)", re.I)   # a channel with one such title is a simulation channel
EXTENDED_RE = re.compile(r"^\s*extended\b.*full game( \d)? highlights", re.I)
QUARTER_RE = re.compile(r"\b(1st|2nd|3rd|4th)[- ]?(qtr|quarter)\b", re.I)
QUARTER_MIN_S = 300
LEAGUE_HL_RE = re.compile(r"full game( \d)? highlights", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                            "nov", "dec"))}
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()


def out_dir() -> Path:
    d = p8_root().parent / "level4" / "nba2526"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shots_root() -> Path:
    return Path(os.environ.get("V3_NBA_SHOTS_ROOT") or p8_root().parent / "nba_shots")


# ── games ─────────────────────────────────────────────────────────────────────
def games(root: Path | None = None) -> list[dict]:
    """[{game_id, date (iso), home, away, kind}] from the mirror's shotdetail files."""
    import pandas as pd
    root = root or shots_root()
    out = []
    for kind, f in (("regular", "shotdetail_2025/shotdetail_2025.csv"), ("playoff", "shotdetail_po_2025/shotdetail_po_2025.csv")):
        d = pd.read_csv(root / f, usecols=["GAME_ID", "GAME_DATE", "HTM", "VTM"]).drop_duplicates("GAME_ID")
        for gid, date, h, v in d.itertuples(index=False):
            s = str(date)
            out.append({"game_id": "%010d" % int(gid), "date": "%s-%s-%s" % (s[:4], s[4:6], s[6:8]),
                        "home": h, "away": v, "kind": kind})
    return sorted(out, key=lambda g: (g["date"], g["game_id"]))


# ── title parsing (pure, tested) ──────────────────────────────────────────────
def teams_in(title: str) -> set[str]:
    t = fold(title)
    found = set()
    for abbr, names in NICK.items():
        if any(re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(n), t) for n in names):
            found.add(abbr)
    for abbr, zh in ZH.items():
        if zh in (title or ""):
            found.add(abbr)
    return found


def _ok(y: int, m: int, d: int):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def dates_in(title: str, upload_year: int | None = None) -> list[dt.date]:
    """Dates written in a title: 'October 24, 2025', 'Oct. 30, 2025', '24 Oct 2025', '10.24.2025', '10/24/25',
    '(1.31.26)', '2025-10-24', '2025年10月24日', and month-day without a year (the year from the season: October to
    December is 2025, else 2026)."""
    t = fold(title)
    out = []
    for m in re.finditer(r"\b(20\d\d)[-./](\d{1,2})[-./](\d{1,2})\b", t):
        out.append(_ok(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"(20\d\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", title or ""):
        out.append(_ok(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"(?<![\d.])(\d{1,2})[./](\d{1,2})[./](\d{2}|\d{4})(?![\d.])", t):
        y = int(m.group(3))
        out.append(_ok(y + 2000 if y < 100 else y, int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?!\d)(?:st|nd|rd|th)?"
                         r"(?:,?\s+(20\d\d))?", t):
        mo, d = MONTHS[m.group(1)], int(m.group(2))
        y = int(m.group(3)) if m.group(3) else (2025 if mo >= 10 else 2026)
        out.append(_ok(y, mo, d))
    for m in re.finditer(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(20\d\d)", t):
        out.append(_ok(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))))
    return sorted({d for d in out if d})


def video_class(title: str, channel: str | None, duration) -> str | None:
    if duration is None:
        return None
    if EXTENDED_RE.search(title or "") and (channel or "") in LEAGUE_CHANNELS and duration >= EXTENDED_MIN_S:
        return "extended"
    if LEAGUE_HL_RE.search(title or "") and (channel or "") in LEAGUE_CHANNELS and not EXTENDED_RE.search(title or ""):
        return "league_highlights"
    if REJECT_RE.search(title or ""):
        return None
    if QUARTER_RE.search(title or "") and re.search(r"\bfull\b", title or "", re.I):
        return "quarter" if duration >= QUARTER_MIN_S else None
    return "full" if duration >= FULL_MIN_S else None


def match_game(title: str, upload_date: str | None, by_pair: dict) -> str | None:
    """The one game a title names, or None (see the module docstring)."""
    ts = teams_in(title)
    if len(ts) != 2:
        return None
    cands = by_pair.get(frozenset(ts), [])
    if not cands:
        return None
    ds = dates_in(title)
    if ds:
        hit = {g["game_id"] for g in cands for d in ds
               if abs((dt.date.fromisoformat(g["date"]) - d).days) <= DATE_SLACK_DAYS}
        return hit.pop() if len(hit) == 1 else None
    if upload_date and re.fullmatch(r"\d{8}", upload_date):
        u = dt.datetime.strptime(upload_date, "%Y%m%d").date()
        hit = [g["game_id"] for g in cands if 0 <= (u - dt.date.fromisoformat(g["date"])).days <= UPLOAD_WINDOW_DAYS]
        return hit[0] if len(hit) == 1 else None
    return None


def pair_index(gs: list[dict]) -> dict:
    idx = {}
    for g in gs:
        idx.setdefault(frozenset((g["home"], g["away"])), []).append(g)
    return idx


# ── yt-dlp ────────────────────────────────────────────────────────────────────
def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def rows(out: str, source: str, via: str) -> list[dict]:
    res = []
    for line in out.splitlines():
        p = line.split("\t", 4)
        if len(p) == 5:
            res.append({"id": p[0], "duration": None if p[1] in ("NA", "None") else float(p[1]), "channel": p[2],
                        "upload_date": None if p[3] in ("NA", "None") else p[3], "title": p[4], "source": source,
                        "via": via})
    return res


FMT = "%(id)s\t%(duration)s\t%(channel)s\t%(upload_date)s\t%(title)s"


def list_channel(url: str, end: int | None = None) -> tuple[list[dict], str | None]:
    code, out, err = run(YT_BASE + ["--flat-playlist", "--print", FMT] + (["--playlist-end", str(end)] if end else []) + [url],
                         timeout=7200)
    res = rows(out, "youtube", url)
    if not res and code != 0:
        return [], (err.strip().splitlines() or ["exit %d" % code])[-1][:200]
    name = url.rstrip("/").split("/")[-2] if url.rstrip("/").endswith("videos") else url
    for r in res:                                   # a channel tab's flat listing leaves the channel name empty
        if r["channel"] in ("NA", "None", ""):
            r["channel"] = None
        r["listed_channel"] = name
    return res, None


def search(q: str, source: str, n: int = 10) -> tuple[list[dict], str | None]:
    if source == "youtube":
        cmd = ["--flat-playlist", "--print", FMT, "ytsearch%d:%s" % (n, q)]
    else:
        cmd = ["--skip-download", "--print", FMT.replace("%(channel)s", "%(uploader)s"), "bilisearch%d:%s" % (n, q)]
    code, out, err = run(YT_BASE + cmd, timeout=900)
    res = rows(out, source, q)
    if not res and code != 0:
        return [], (err.strip().splitlines() or ["exit %d" % code])[-1][:200]
    return res, None


def probe(source: str, vid: str) -> dict:
    from tools.research_testset import refuse_cv_test
    refuse_cv_test([vid], WHO)
    url = ("https://www.youtube.com/watch?v=%s" % vid) if source == "youtube" else ("https://www.bilibili.com/video/%s" % vid)
    code, out, err = run(YT_BASE + ["--no-playlist", "--skip-download", "-j", url], timeout=300)
    if code != 0:
        return {"error": (err.strip().splitlines() or ["exit %d" % code])[-1][:200]}
    j = json.loads(out)
    hs = [f.get("height") for f in j.get("formats") or [] if f.get("height") and f.get("vcodec") not in (None, "none")]
    return {"title": j.get("title"), "channel": j.get("channel") or j.get("uploader"), "upload_date": j.get("upload_date"),
            "duration": j.get("duration"), "max_height": max(hs) if hs else None,
            "avc1_720": any(f.get("height") == 720 and str(f.get("vcodec", "")).startswith("avc1") for f in j.get("formats") or [])}


# ── steps ─────────────────────────────────────────────────────────────────────
def save(name: str, obj) -> Path:
    p = out_dir() / name
    atomic_write(p, json.dumps(obj, indent=1, ensure_ascii=False))
    return p


def load(name: str, default=None):
    p = out_dir() / name
    return json.loads(p.read_text()) if p.exists() else default


def cmd_channels(urls: list[str], end: int | None = None) -> None:
    listings = load("listings.json", {})
    for u in urls:
        res, err = list_channel(u, end)
        listings[u] = {"n": len(res), "error": err, "videos": res, "date": dt.date.today().isoformat()}
        print("%s: %d videos%s" % (u, len(res), (" (error %s)" % err) if err else ""), flush=True)
        save("listings.json", listings)
        time.sleep(THROTTLE_S)


def cmd_discover(n: int, seed: int) -> None:
    gs = games()
    sample = random.Random(seed).sample(gs, n)
    found = load("discover.json", {"seed": seed, "games": {}})
    for g in sample:
        if g["game_id"] in found["games"]:
            continue
        d = dt.date.fromisoformat(g["date"])
        q = "%s vs %s full game %s %d, %d" % (NICK[g["away"]][0].title(), NICK[g["home"]][0].title(),
                                               MONTH_NAMES[d.month - 1], d.day, d.year)
        res, err = search(q, "youtube")
        found["games"][g["game_id"]] = {"query": q, "error": err, "results": res}
        save("discover.json", found)
        time.sleep(THROTTLE_S)
    print("discover: %d games searched" % len(found["games"]))


def cmd_bilibili() -> None:
    qs = ["NBA 全场回放 2025-26", "NBA常规赛 全场录像 2026", "NBA季后赛 全场回放 2026", "湖人 勇士 全场回放 2025",
          "雷霆 掘金 全场录像 2026"]
    found = load("bilibili.json", {})
    for q in qs:
        if q in found:
            continue
        res, err = search(q, "bilibili", 10)
        found[q] = {"error": err, "results": res}
        print("%s: %d results%s" % (q, len(res), (" (error %s)" % err) if err else ""), flush=True)
        save("bilibili.json", found)
        time.sleep(THROTTLE_S)


def candidates() -> list[dict]:
    """Every listed or searched video that names one game and has a class, deduplicated by (source, id)."""
    gs = games()
    idx = pair_index(gs)
    vids = []
    for u, L in (load("listings.json", {}) or {}).items():
        vids += L["videos"]
    for g in (load("discover.json", {"games": {}}) or {})["games"].values():
        vids += g["results"]
    for q in (load("bilibili.json", {}) or {}).values():
        vids += q["results"]
    sim_channels = {v.get("channel") or v.get("listed_channel") for v in vids if SIM_RE.search(v.get("title") or "")}
    sim_channels.discard(None)
    out, seen = [], set()
    for v in vids:
        k = (v["source"], v["id"])
        if k in seen:
            continue
        seen.add(k)
        ch = v.get("channel") or v.get("listed_channel")
        if ch in sim_channels and ch not in LEAGUE_CHANNELS:
            continue
        cls = video_class(v["title"], ch, v.get("duration"))
        if not cls:
            continue
        gid = match_game(v["title"], v.get("upload_date"), idx)
        if gid:
            out.append(dict(v, channel=ch, cls=cls, game_id=gid))
    return out


def cmd_probe(extended_sample: int) -> None:
    probes = load("probes.json", {})
    cs = candidates()
    full = [c for c in cs if c["cls"] == "full"]
    ext = [c for c in cs if c["cls"] == "extended"]
    todo = full + random.Random(2526).sample(ext, min(extended_sample, len(ext)))
    for c in todo:
        key = "%s:%s" % (c["source"], c["id"])
        if key in probes:
            continue
        probes[key] = probe(c["source"], c["id"])
        save("probes.json", probes)
        time.sleep(THROTTLE_S)
    print("probed %d (full %d, extended sample %d)" % (len(probes), len(full), min(extended_sample, len(ext))))


def summary() -> dict:
    gs = games()
    by_id = {g["game_id"]: g for g in gs}
    probes = load("probes.json", {})
    cs = candidates()
    per_game = {}
    for c in cs:
        p = probes.get("%s:%s" % (c["source"], c["id"]), {})
        usable = None if not p or p.get("error") else (p.get("max_height") or 0) >= MIN_HEIGHT
        per_game.setdefault(c["game_id"], []).append({"source": c["source"], "id": c["id"], "cls": c["cls"],
                                                       "channel": c["channel"], "duration": c.get("duration"),
                                                       "max_height": p.get("max_height"), "usable": usable})
    teams = sorted(NICK)
    def count(pred, kind=None):
        per_team = {t: 0 for t in teams}
        n = 0
        for gid, vs in per_game.items():
            g = by_id[gid]
            if kind and g["kind"] != kind:
                continue
            if any(pred(v) for v in vs):
                n += 1
                per_team[g["home"]] += 1
                per_team[g["away"]] += 1
        return n, per_team
    out = {"games": {"regular": sum(g["kind"] == "regular" for g in gs), "playoff": sum(g["kind"] == "playoff" for g in gs)},
           "listings": {u: L["n"] for u, L in (load("listings.json", {}) or {}).items()},
           "discover_searched": len((load("discover.json", {"games": {}}) or {})["games"]),
           "bilibili_queries": len(load("bilibili.json", {}) or {}),
           "candidates": len(cs), "per_game": per_game}
    for name, pred in (("full_any", lambda v: v["cls"] == "full"),
                       ("full_usable", lambda v: v["cls"] == "full" and v["usable"]),
                       ("extended_any", lambda v: v["cls"] == "extended"),
                       ("league_highlights_any", lambda v: v["cls"] == "league_highlights"),
                       ("quarter_any", lambda v: v["cls"] == "quarter")):
        for kind in ("regular", "playoff"):
            n, pt = count(pred, kind)
            out["%s_%s" % (name, kind)] = {"games": n, "per_team": pt}
    by_source = {}
    for gid, vs in per_game.items():
        for v in vs:
            key = "%s|%s|%s" % (v["source"], v["channel"], v["cls"])
            by_source.setdefault(key, set()).add(gid)
    quarters = {}
    for c in cs:
        if c["cls"] == "quarter":
            m = QUARTER_RE.search(c["title"])
            quarters.setdefault(c["game_id"], set()).add(m.group(1).lower())
    out["games_complete_in_quarters"] = sorted(g for g, q in quarters.items() if len(q) == 4)
    out["games_by_source"] = {k: len(v) for k, v in sorted(by_source.items(), key=lambda kv: -len(kv[1]))}
    save("summary.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("channels")
    c.add_argument("urls", nargs="+")
    c.add_argument("--end", type=int, help="list only the newest N uploads of each channel")
    d = sub.add_parser("discover")
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--seed", type=int, default=2526)
    sub.add_parser("bilibili")
    p = sub.add_parser("probe")
    p.add_argument("--extended-sample", type=int, default=20)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "channels":
        cmd_channels(a.urls, a.end)
    elif a.cmd == "discover":
        cmd_discover(a.n, a.seed)
    elif a.cmd == "bilibili":
        cmd_bilibili()
    elif a.cmd == "probe":
        cmd_probe(a.extended_sample)
    else:
        s = summary()
        print(json.dumps({k: v for k, v in s.items() if k != "per_game"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
