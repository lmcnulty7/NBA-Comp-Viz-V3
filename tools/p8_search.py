"""tools/p8_search.py: the broadcast search over every SportVU game (P8 step 0, second pass; PLAN.md D128 to D135).

The frozen R1.1 search (sportvu/manifest.py) looked at the 84 home games of GSW, CLE, OKC and NYK. This tool runs the
same rules over every game of the SportVU contract manifest, adds English city-name and Chinese query forms, searches
YouTube and then Bilibili, and accepts condensed games as their own class (D134). It lists and probes only: no video
is downloaded here. A candidate becomes a game only through the automatic identity check (tools.p8_identity, D130).

Game classes, fixed before any search (D129):
  training   searchable, may become a development game
  d85_test   a D85 broadcast CV test game or any game in a test arena: searched by metadata only, counted, never
             downloaded or identity-checked here
  spent      a spent held-out game (sportvu/splits.json): searched by metadata only, counted, never touched
  broken     a broken SportVU archive: not searched

    python3 -m tools.p8_search run [--limit N] [--source youtube|bilibili] [--retry-errors]
    python3 -m tools.p8_search summary            # -> workspace search/summary.json and a text table

Per-game results are cached under <P8 workspace>/search/parts/ so a rerun resumes. The search is network-bound and
reads no tracking frames, so it runs outside the heavy-job lock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from sportvu import manifest as M
from tools.p8_common import atomic_write, p8_root

WHO = "tools.p8_search"
THROTTLE_S = 1.0
MAX_PROBES = 3
CONDENSED_MIN_S = 480                 # D134
CONDENSED_RE = re.compile(r"(condensed|压缩版|浓缩版)", re.I)
SOURCES = ("youtube", "bilibili")

# 2015-16 teams in the Chinese forms used by Chinese-language uploads (nicknames as broadcast in mainland China)
ZH = {
    "ATL": "老鹰", "BOS": "凯尔特人", "BKN": "篮网", "CHA": "黄蜂", "CHI": "公牛", "CLE": "骑士", "DAL": "独行侠",
    "DEN": "掘金", "DET": "活塞", "GSW": "勇士", "HOU": "火箭", "IND": "步行者", "LAC": "快船", "LAL": "湖人",
    "MEM": "灰熊", "MIA": "热火", "MIL": "雄鹿", "MIN": "森林狼", "NOP": "鹈鹕", "NYK": "尼克斯", "OKC": "雷霆",
    "ORL": "魔术", "PHI": "76人", "PHX": "太阳", "POR": "开拓者", "SAC": "国王", "SAS": "马刺", "TOR": "猛龙",
    "UTA": "爵士", "WAS": "奇才",
}
ZH_ALIASES = {"DAL": ("小牛",)}       # the Mavericks were 小牛 in Chinese media in 2015-16
ZH_VIDEO_GAME_RE = re.compile(r"(2K|模拟|游戏|电竞|实况)", re.I)
ZH_POSTSEASON_RE = re.compile(r"(季后赛|总决赛|西决|东决|分区决赛|附加赛)")


def contract_games() -> dict:
    """Every game of the SportVU contract manifest (626), keyed by its archive name."""
    m = json.loads((Path(os.environ["DATA_ROOT"]) / "sportvu_contract" / "manifest.json").read_text())
    return m["games"]


def game_class(name: str, g: dict) -> str:
    """training | d85_test | spent | broken, by the existing guards' own rules (never raising)."""
    from analytics.eval import is_heldout_game
    from tools.research_testset import cv_test_hits
    if g.get("status") != "ok":
        return "broken"
    if is_heldout_game(name) or g.get("heldout"):
        return "spent"
    if cv_test_hits([name]):
        return "d85_test"
    return "training"


# ── queries ───────────────────────────────────────────────────────────────────
def en_forms(g: dict) -> list[str]:
    """The frozen R1.1 nickname forms plus two city-name forms (one in the title style of the channel that holds the
    known 2015-16 full games, docs/research/P8_FEASIBILITY.md section 4)."""
    d = dt.date.fromisoformat(g["date"])
    (ac, an, _), (hc, hn, _) = M.TEAMS[g["away"]], M.TEAMS[g["home"]]
    return M.query_forms(g) + ["%s %s vs %s %s %s %d %d" % (ac, an, hc, hn, d.strftime("%B"), d.day, d.year),
                               "Nba HD Full Game %s at %s 2015 / 2016" % (an, hn)]


def zh_forms(g: dict) -> list[str]:
    d = dt.date.fromisoformat(g["date"])
    a, h = ZH[g["away"]], ZH[g["home"]]
    return ["%s vs %s %d年%d月%d日 常规赛 全场" % (a, h, d.year, d.month, d.day),
            "%d.%02d.%02d %s %s 全场录像" % (d.year, d.month, d.day, a, h)]


def queries(g: dict, source: str) -> list[str]:
    """YouTube: the five English forms and both Chinese forms. Bilibili: the first Chinese form only (D136): its
    search returns no metadata in a flat listing, so every result is extracted, about 12 s per query."""
    return en_forms(g) + zh_forms(g) if source == "youtube" else zh_forms(g)[:1]


# ── text rules ────────────────────────────────────────────────────────────────
def mentions(text: str, abbr: str) -> bool:
    t = text or ""
    return M.mentions(t, abbr) or any(z in t for z in (ZH[abbr],) + ZH_ALIASES.get(abbr, ()))


def zh_dates(date: str) -> list[str]:
    d = dt.date.fromisoformat(date)
    return [r"%d年0?%d月0?%d日" % (d.year, d.month, d.day), r"\b%d\.0?%d\.0?%d\b" % (d.year, d.month, d.day),
            r"\b%d%02d%02d\b" % (d.year, d.month, d.day), r"(?<!\d)0?%d月0?%d日" % (d.month, d.day)]


def evidence(date: str, title: str, description: str = "", upload_date: str | None = None) -> str | None:
    """The frozen evidence rules (exact date, upload window, season string) plus Chinese date forms and the Chinese
    season form (2015-16赛季)."""
    text = "%s\n%s" % (title or "", description or "")
    for p in zh_dates(date):
        m = re.search(p, text)
        if m:
            return "date: %s" % m.group(0)
    e = M.season_evidence(date, title, description, upload_date)
    if e:
        return e
    m = re.search(r"2015\s*[-/]\s*(20)?16\s*赛季", text)
    return "season: %s" % m.group(0) if m else None


def rejection(title: str, channel: str | None, date: str) -> str | None:
    """The frozen rejection rules (video-game simulations, another game: postseason, year, date) plus their Chinese
    forms."""
    why = M.candidate_rejection(title, channel, date)
    if why:
        return why
    if ZH_VIDEO_GAME_RE.search(title or "") or ZH_VIDEO_GAME_RE.search(channel or ""):
        return "video-game simulation"
    if ZH_POSTSEASON_RE.search(title or ""):
        return "postseason game"
    return None


def length_class(duration, title: str) -> str | None:
    """full (at least the frozen MIN_BROADCAST_S), condensed (D134), or None."""
    d = duration or 0
    if d >= M.MIN_BROADCAST_S:
        return "full"
    if d >= CONDENSED_MIN_S and CONDENSED_RE.search(title or ""):
        return "condensed"
    return None


# ── yt-dlp ────────────────────────────────────────────────────────────────────
def video_url(source: str, vid: str) -> str:
    return ("https://www.youtube.com/watch?v=%s" % vid) if source == "youtube" else ("https://www.bilibili.com/video/%s" % vid)


def search(query: str, source: str) -> tuple[list[dict], str | None]:
    """YouTube: a flat listing of 10 results. Bilibili: 5 results, each extracted (its flat listing has no title or
    duration); no video is downloaded either way."""
    if source == "youtube":
        cmd = ["--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", "ytsearch10:" + query]
    else:
        cmd = ["--skip-download", "--print", "%(id)s\t%(duration)s\t%(uploader)s\t%(title)s", "bilisearch5:" + query]
    code, out, err = M._run(M.YT_BASE + cmd, timeout=300)
    if code != 0:
        return [], (err.strip().splitlines() or ["yt-dlp exit %d" % code])[-1][:200]
    res = []
    for line in out.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            try:
                dur = float(parts[1])
            except ValueError:
                dur = None
            res.append({"id": parts[0], "duration": dur, "channel": parts[2], "title": parts[3], "source": source})
    return res, None


def probe(source: str, vid: str) -> dict:
    """Metadata and the format that would be downloaded (YouTube: harvest_driver.YTDLP_FMT; Bilibili: the best video
    at most 720p), without downloading."""
    from harvest_driver import YTDLP_FMT
    fmt = YTDLP_FMT if source == "youtube" else "bv*[height<=720]+ba/b[height<=720]"
    code, out, err = M._run(M.YT_BASE + ["--no-playlist", "--skip-download", "-f", fmt, "-j", video_url(source, vid)])
    if code != 0:
        msg = (err.strip().splitlines() or ["yt-dlp exit %d" % code])[-1][:200]
        return {"error": msg, "no_format": "requested format is not available" in msg.lower()}
    j = json.loads(out)
    return {"title": j.get("title"), "channel": j.get("channel") or j.get("uploader"), "upload_date": j.get("upload_date"),
            "duration": j.get("duration"), "description": (j.get("description") or "")[:2000],
            "format_id": j.get("format_id"), "vcodec": j.get("vcodec"), "height": j.get("height"), "fps": j.get("fps")}


def verdict(c: dict, g: dict) -> str:
    """Metadata verdict of one probed candidate: probe_error | no_format | no_broadcast | low_res | candidate. Only an
    identity check makes a candidate a game."""
    if c.get("error"):
        return "no_format" if c.get("no_format") else "probe_error"
    if rejection(c.get("title") or c.get("search_title"), c.get("channel"), g["date"]):
        return "no_broadcast"
    text = "%s\n%s" % (c.get("title") or "", c.get("description") or "")
    if not (mentions(text, g["away"]) and mentions(text, g["home"])):
        return "no_broadcast"
    if not length_class(c.get("duration"), c.get("title") or ""):
        return "no_broadcast"
    if (c.get("height") or 0) < 720:
        return "low_res"
    return "candidate"


def check_game(name: str, g: dict, cls: str, sources=SOURCES) -> dict:
    """Search every source for one game, filter by the text rules, probe the best MAX_PROBES per source. Spent and
    D85 test games are searched and filtered but never probed (no request reaches their videos)."""
    g = {**g, "game": name}
    out = {"game": name, "class": cls, "date": g["date"], "away": g["away"], "home": g["home"], "sources": {}}
    for source in sources:
        qs, seen, errors, rejected = queries(g, source), {}, [], {}
        for q in qs:
            res, err = search(q, source)
            if err:
                errors.append(err)
            for r in res:
                seen.setdefault(r["id"], r)
            time.sleep(THROTTLE_S)
        cands = []
        for r in seen.values():
            if not length_class(r["duration"], r["title"]):
                continue
            if not (mentions(r["title"], g["away"]) or mentions(r["title"], g["home"])):
                continue
            why = rejection(r["title"], r.get("channel"), g["date"])
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            r = dict(r, length=length_class(r["duration"], r["title"]), title_evidence=evidence(g["date"], r["title"]))
            cands.append(r)
        rank = {"date": 0, "upload": 1, "season": 2, None: 3}
        cands.sort(key=lambda r: (rank[M.evidence_kind(r["title_evidence"])], r["length"] != "full", -(r["duration"] or 0)))
        probed = []
        if cls == "training":
            from tools.research_testset import refuse_cv_test
            from tools.p8_common import guard
            for r in cands[:MAX_PROBES]:
                try:                                          # a D85 test or spent held-out video found under another game
                    refuse_cv_test([r["id"]], WHO)
                    guard([r["id"]], WHO)
                except SystemExit:
                    continue
                p = probe(source, r["id"])
                time.sleep(THROTTLE_S)
                c = {"id": r["id"], "source": source, "search_title": r["title"], "length": r["length"], **p}
                if not p.get("error"):
                    c["evidence"] = evidence(g["date"], p["title"], p["description"], p["upload_date"])
                    c["length"] = length_class(p.get("duration"), p.get("title") or "") or r["length"]
                c["verdict"] = verdict(c, g)
                c.pop("description", None)
                probed.append(c)
        out["sources"][source] = {"queries": qs, "results": len(seen), "candidates": len(cands),
                                  "candidate_ids": [{"id": r["id"], "length": r["length"], "title": r["title"],
                                                     "evidence": r["title_evidence"]} for r in cands[:MAX_PROBES]],
                                  "rejected_before_probe": rejected, "probed": probed, "errors": errors}
    return out


# ── run and summary ───────────────────────────────────────────────────────────
def parts_dir() -> Path:
    d = p8_root() / "search" / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(limit: int | None, sources, retry_errors: bool, reverse: bool = False) -> None:
    games = contract_games()
    order = sorted(games, key=lambda n: (games[n]["date"], n), reverse=reverse)   # reverse: a second runner from the end
    done, consecutive = 0, 0
    for name in order:
        g = games[name]
        cls = game_class(name, g)
        path = parts_dir() / ("%s.json" % name)
        if cls == "broken":
            if not path.exists():
                atomic_write(path, json.dumps({"game": name, "class": cls, "date": g["date"], "away": g["away"],
                                               "home": g["home"], "sources": {}}, ensure_ascii=False, indent=1))
            continue
        if path.exists():
            prev = json.loads(path.read_text())
            errs = any(s.get("errors") for s in prev["sources"].values())
            if not (retry_errors and errs) and all(s in prev["sources"] for s in sources):
                continue
        res = check_game(name, g, cls, sources)
        atomic_write(path, json.dumps(res, ensure_ascii=False, indent=1))
        errs = sum(len(s["errors"]) for s in res["sources"].values())
        consecutive = consecutive + 1 if errs and not any(s["results"] for s in res["sources"].values()) else 0
        if consecutive >= M.MAX_CONSECUTIVE_ERRORS:
            raise SystemExit(f"{WHO}: {consecutive} games in a row returned only errors; stopped (last: {name})")
        done += 1
        print(f"[search] {done} {name} {cls} " + " ".join(f"{s}:{r['candidates']}" for s, r in res["sources"].items()),
              flush=True)
        if limit and done >= limit:
            break


def summary() -> dict:
    parts = [json.loads(p.read_text()) for p in sorted(parts_dir().glob("*.json"))]
    old84 = {g["game"] for g in M.list_games([{"game": p["game"], "date": p["date"], "away": p["away"],
                                                  "home": p["home"]} for p in parts])}
    rows = []
    for p in parts:
        cands = [c for s in p["sources"].values() for c in s.get("probed", []) if c.get("verdict") == "candidate"]
        unprobed = sum(s.get("candidates", 0) for s in p["sources"].values()) if p["class"] != "training" else 0
        rows.append({"game": p["game"], "class": p["class"], "home": p["home"], "away": p["away"], "in_old_84": p["game"] in old84,
                     "candidates": [{"id": c["id"], "source": c["source"], "length": c["length"], "height": c.get("height"),
                                     "evidence": c.get("evidence")} for c in cands],
                     "metadata_hits_not_probed": unprobed})
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], {"games": 0, "with_candidate": 0, "with_metadata_hit": 0})
        b = by_class[r["class"]]
        b["games"] += 1
        b["with_candidate"] += bool(r["candidates"])
        b["with_metadata_hit"] += bool(r["candidates"] or r["metadata_hits_not_probed"])
    return {"written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "games_searched": len(rows),
            "by_class": by_class, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int)
    r.add_argument("--source", choices=SOURCES, action="append")
    r.add_argument("--retry-errors", action="store_true")
    r.add_argument("--reverse", action="store_true", help="latest games first, for a second runner")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.limit, tuple(a.source or SOURCES), a.retry_errors, a.reverse)
    else:
        s = summary()
        atomic_write(p8_root() / "search" / "summary.json", json.dumps(s, ensure_ascii=False, indent=1))
        print(json.dumps(s["by_class"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
