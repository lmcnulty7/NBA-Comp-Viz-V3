"""
sportvu/manifest.py: availability manifest for the Stage 1 SportVU games (ROADMAP R1.1).

For every SportVU home game of GSW (Oracle), CLE, OKC and NYK in the public log window
(2015-10-27 .. 2016-01-23), two independent checks, then one verdict and a priority:

  SportVU   the .7z archive (cached under data/sportvu/archives/, gitignored) is parsed with
            sportvu.fetch.parse in a temp dir (the 100 MB raw JSON is not kept). An archive under
            MIN_ARCHIVE_BYTES or one that fails to download or parse is sportvu_broken. Per quarter:
            interior clock gaps > GAP_FLAG_S, a start or end edge gap > GAP_FLAG_S (720 s periods,
            300 s in overtime) and a missing quarter 1..4 are flagged (sportvu_gaps).
  Broadcast YouTube search with QUERY_FORMS (yt-dlp flat search, no download); results shorter than
            MIN_BROADCAST_S, ids registered in data/harvest/games.json to another game, titles naming
            neither team, video-game simulations (NBA 2K, NBA Live, console gameplay; title or channel)
            and titles that name another game (a different date, a year outside the season, or a
            playoff game; the SportVU window is regular season only) are dropped before probing. The best MAX_PROBES candidates are probed with
            harvest_driver.YTDLP_FMT (metadata and format selection only). A candidate needs both
            teams in its title or description. Evidence that it is THIS game: the game's exact date in
            the title or description (date), or an upload within SEASON_WINDOW_DAYS after the game
            when no other game between the same two teams falls in that window (upload; checked
            against the full SportVU schedule). A season string alone ("2015-2016") fits every game of
            the matchup and only makes a candidate ambiguous. A video whose text names one game's
            date counts for that game only.

Verdicts (precedence top to bottom): held_out (gsw_phx_2016, registry id, not searched),
sportvu_broken, no_broadcast, probe_error (a yt-dlp failure, never read as no_avc1), no_avc1,
low_res (avc1 only below 720p), short_broadcast (MIN_BROADCAST_S..FULL_S), ambiguous (full length,
avc1 720p, no season evidence), sportvu_gaps (broadcast ok, SportVU gaps flagged), ok_unverified
(full length, avc1 720p, season evidence, no SportVU flag). Identity is confirmed only by the
scorebug at R1.2. Priority: 12.25.2015 CLE at GSW first, then Oracle by date, then CLE, OKC, NYK by
date; usable verdicts (ok_unverified, sportvu_gaps) first within each group.

  python -m sportvu.manifest                  -> reports/sportvu_manifest.{json,txt}
  python -m sportvu.manifest --retry-errors   # re-probe games whose verdict was probe_error
Per-game results are cached in data/sportvu/manifest_parts/ so a rerun resumes.
"""
from __future__ import annotations
import argparse, datetime, io, json, re, shutil, subprocess, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import config
from sportvu.fetch import RAW_BASE, SV_DIR, parse

LIST_URL = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"
NAME_RE = re.compile(r"^(\d\d)\.(\d\d)\.(\d{4})\.([A-Z]{3})\.at\.([A-Z]{3})\.7z$")
ARENAS = ("GSW", "CLE", "OKC", "NYK")
FIRST = "12.25.2015.CLE.at.GSW"
HELD_OUT = {"12.16.2015.PHX.at.GSW": ("gsw_phx_2016", "f8lAcHg6kk0")}
MIN_ARCHIVE_BYTES = 1_000_000
GAP_FLAG_S = 10.0
FULL_S, MIN_BROADCAST_S = 4500, 2400
SEASON_WINDOW_DAYS = 30
MAX_PROBES = 3
MAX_CONSECUTIVE_ERRORS = 5
YTDLP = "/opt/anaconda3/bin/yt-dlp"
YT_BASE = [YTDLP, "--no-update", "--js-runtimes", "node", "--no-warnings"]
OVERRIDES = config.PROJECT_ROOT / "sportvu" / "manifest_overrides.json"   # human decisions, tracked
PARTS = SV_DIR / "manifest_parts"
ARCHIVES = SV_DIR / "archives"
USABLE = ("ok_unverified", "sportvu_gaps")
PRECEDENCE = ("held_out", "sportvu_broken", "no_broadcast", "probe_error", "no_avc1", "low_res",
              "short_broadcast", "ambiguous", "sportvu_gaps", "ok_unverified")

# 2015-16 teams: abbreviation -> (city, nickname, extra aliases matched as whole words)
TEAMS = {
    "ATL": ("Atlanta", "Hawks", ()), "BOS": ("Boston", "Celtics", ()), "BKN": ("Brooklyn", "Nets", ()),
    "CHA": ("Charlotte", "Hornets", ()), "CHI": ("Chicago", "Bulls", ()), "CLE": ("Cleveland", "Cavaliers", ("Cavs",)),
    "DAL": ("Dallas", "Mavericks", ("Mavs",)), "DEN": ("Denver", "Nuggets", ()), "DET": ("Detroit", "Pistons", ()),
    "GSW": ("Golden State", "Warriors", ("Dubs",)), "HOU": ("Houston", "Rockets", ()), "IND": ("Indiana", "Pacers", ()),
    "LAC": ("LA", "Clippers", ()), "LAL": ("Los Angeles", "Lakers", ()), "MEM": ("Memphis", "Grizzlies", ()),
    "MIA": ("Miami", "Heat", ()), "MIL": ("Milwaukee", "Bucks", ()), "MIN": ("Minnesota", "Timberwolves", ("Wolves",)),
    "NOP": ("New Orleans", "Pelicans", ()), "NYK": ("New York", "Knicks", ()), "OKC": ("Oklahoma City", "Thunder", ()),
    "ORL": ("Orlando", "Magic", ()), "PHI": ("Philadelphia", "76ers", ("Sixers",)), "PHX": ("Phoenix", "Suns", ()),
    "POR": ("Portland", "Trail Blazers", ("Blazers",)), "SAC": ("Sacramento", "Kings", ()), "SAS": ("San Antonio", "Spurs", ()),
    "TOR": ("Toronto", "Raptors", ()), "UTA": ("Utah", "Jazz", ()), "WAS": ("Washington", "Wizards", ()),
}
CITY_IS_UNIQUE = {"LAC": False, "LAL": False}       # "LA"/"Los Angeles" names two teams


def parse_name(name: str):
    """'12.25.2015.CLE.at.GSW.7z' -> {'game', 'date', 'away', 'home'}; None if not a game archive."""
    m = NAME_RE.match(name)
    if not m:
        return None
    mm, dd, yyyy, away, home = m.groups()
    return {"game": name[:-3], "date": "%s-%s-%s" % (yyyy, mm, dd), "away": away, "home": home}


def list_all() -> list[dict]:
    """Every game archive in the SportVU log listing (one API call)."""
    req = urllib.request.Request(LIST_URL, headers={"Accept": "application/vnd.github+json", "User-Agent": "nba-comp-viz"})
    entries = json.loads(urllib.request.urlopen(req, timeout=60).read())
    games = []
    for e in entries:
        g = parse_name(e["name"])
        if g:
            g["archive_bytes"] = int(e["size"])
            games.append(g)
    return games


def list_games(all_games: list[dict] | None = None) -> list[dict]:
    """The SportVU home games of ARENAS."""
    games = [g for g in (all_games if all_games is not None else list_all()) if g["home"] in ARENAS]
    return sorted(games, key=lambda g: (g["home"], g["date"]))


# ── SportVU check ─────────────────────────────────────────────────────────────
def gap_flags(moments: list[dict]) -> dict:
    """Clock-coverage flags over parsed moments: per quarter interior gaps and edge gaps above
    GAP_FLAG_S, and missing regulation quarters. Returns {'quarters', 'flags', 'gap_s_total'}."""
    out, flags, total = {}, [], 0.0
    qs = sorted({m["q"] for m in moments})
    for q in (1, 2, 3, 4):
        if q not in qs:
            flags.append("Q%d missing" % q)
    for q in qs:
        period = 720.0 if q <= 4 else 300.0
        cl = np.sort(np.array([m["clock"] for m in moments if m["q"] == q], float))[::-1]
        d = cl[:-1] - cl[1:]
        interior = [[round(float(cl[i]), 1), round(float(cl[i + 1]), 1)] for i in np.where(d > GAP_FLAG_S)[0]]
        start, end = period - float(cl[0]), float(cl[-1])
        qf = ["Q%d gap %.0f..%.0f s" % (q, a, b) for a, b in interior]
        if start > GAP_FLAG_S:
            qf.append("Q%d starts at %.0f s" % (q, cl[0]))
        if end > GAP_FLAG_S:
            qf.append("Q%d ends at %.0f s" % (q, cl[-1]))
        total += sum(a - b for a, b in interior) + (start if start > GAP_FLAG_S else 0.0) + (end if end > GAP_FLAG_S else 0.0)
        n10 = sum(1 for m in moments if m["q"] == q and len(m["players"]) == 10)
        out[str(q)] = {"clock_span": [round(float(cl[0]), 1), round(float(cl[-1]), 1)], "moments": int(len(cl)),
                       "share_10_players": round(n10 / max(len(cl), 1), 3), "flags": qf}
        flags += qf
    return {"quarters": out, "flags": flags, "gap_s_total": round(total, 1)}


def sportvu_check(g: dict) -> dict:
    if g["archive_bytes"] < MIN_ARCHIVE_BYTES:
        return {"status": "sportvu_broken", "error": "archive is %d bytes" % g["archive_bytes"]}
    try:
        import py7zr
        ARCHIVES.mkdir(parents=True, exist_ok=True)
        arc = ARCHIVES / (g["game"] + ".7z")
        if not arc.exists() or arc.stat().st_size != g["archive_bytes"]:
            arc.write_bytes(urllib.request.urlopen(RAW_BASE + g["game"] + ".7z", timeout=180).read())
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(arc, "r") as z:
                names = [n for n in z.getnames() if n.lower().endswith(".json")]
                z.extract(path=tmp, targets=names)
            raw = next(Path(tmp).rglob("*.json"))
            parsed = parse(raw)
        if not parsed["moments"]:
            return {"status": "sportvu_broken", "error": "no moments after parsing"}
        gf = gap_flags(parsed["moments"])
        return {"status": "sportvu_gaps" if gf["flags"] else "ok", "moments": len(parsed["moments"]),
                "players": len(parsed["players"]), **gf}
    except Exception as e:                                   # never abort the pass on one archive
        return {"status": "sportvu_broken", "error": "%s: %s" % (type(e).__name__, str(e)[:200])}


# ── broadcast probe ───────────────────────────────────────────────────────────
def _aliases(abbr: str) -> list[str]:
    city, nick, extra = TEAMS[abbr]
    names = [nick, *extra, abbr]
    if CITY_IS_UNIQUE.get(abbr, True):
        names.append(city)
    return names


def mentions(text: str, abbr: str) -> bool:
    t = text or ""
    return any(re.search(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(a), t, re.I) for a in _aliases(abbr))


def date_patterns(date: str) -> list[str]:
    d = datetime.date.fromisoformat(date)
    mon, mo3 = d.strftime("%B"), d.strftime("%b")
    y, m, dd = d.year, d.month, d.day
    return [r"\b%s\.? %d,? %d\b" % (mon, dd, y), r"\b%s\.? %d,? %d\b" % (mo3, dd, y), r"\b%d/%d/(%d|%02d)\b" % (m, dd, y, y % 100),
            r"\b%02d/%02d/(%d|%02d)\b" % (m, dd, y, y % 100), r"\b%d-%02d-%02d\b" % (y, m, dd), r"\b%02d\.%02d\.%d\b" % (m, dd, y)]


def season_pattern(date: str) -> str:
    d = datetime.date.fromisoformat(date)
    s0 = d.year if d.month >= 9 else d.year - 1
    return r"\b%d ?[-/] ?(%02d|%d)\b" % (s0, (s0 + 1) % 100, s0 + 1)


def season_evidence(date: str, title: str, description: str, upload_date: str | None) -> str | None:
    """Why this video belongs to the game ('date: ...', 'uploaded N days after the game') or only to
    its season ('season: ...'), or None. The strongest kind found wins."""
    text = "%s\n%s" % (title or "", description or "")
    for p in date_patterns(date):
        m = re.search(p, text, re.I)
        if m:
            return "date: %s" % m.group(0)
    if upload_date:
        up = datetime.date(int(upload_date[:4]), int(upload_date[4:6]), int(upload_date[6:8]))
        delta = (up - datetime.date.fromisoformat(date)).days
        if 0 <= delta <= SEASON_WINDOW_DAYS:
            return "uploaded %d days after the game" % delta
    m = re.search(season_pattern(date), text, re.I)
    return "season: %s" % m.group(0) if m else None


def evidence_kind(e: str | None) -> str | None:
    """'date' | 'upload' | 'season' | None (also reads the earlier 'text: ...' form)."""
    if not e:
        return None
    if e.startswith("human:"):
        return "human"
    if e.startswith("uploaded"):
        return "upload"
    if e.startswith("season:"):
        return "season"
    if e.startswith("text:"):
        return "season" if re.fullmatch(r"\d{4} ?[-/] ?\d{2,4}", e[5:].strip()) else "date"
    return "date"


def query_forms(g: dict) -> list[str]:
    d = datetime.date.fromisoformat(g["date"])
    a, h = TEAMS[g["away"]][1], TEAMS[g["home"]][1]
    return ["%s vs %s %s %d, %d full game" % (a, h, d.strftime("%B"), d.day, d.year),
            "%s vs %s %d/%d/%02d" % (h, a, d.month, d.day, d.year % 100),
            "%s %s %s %d %d full game" % (a, h, d.strftime("%b"), d.day, d.year)]


VIDEO_GAME_RE = re.compile(r"(nba\s?2k|\b2k1\d\b|nba\s?live|nbalive|cpu\s*vs\.?\s*cpu|\bps[345]\b|xbox|gameplay|simulation|\bsim\b|gaming)", re.I)
POSTSEASON_RE = re.compile(r"(play-?offs?|play-?in|conference (semi)?finals|finals,? game|\bgame [1-7]\b)", re.I)
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def is_video_game(title: str, channel: str | None = None) -> bool:
    return bool(VIDEO_GAME_RE.search(title or "") or VIDEO_GAME_RE.search(channel or ""))


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def dates_in(text: str) -> list[set]:
    """Explicit dates in a title, each as the set of dates it could mean (m/d and d/m readings)."""
    out = []
    for a, b, y in re.findall(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})\b", text or ""):
        cands = set()
        for m, d in ((int(a), int(b)), (int(b), int(a))):
            try:
                cands.add(datetime.date(_year(y), m, d))
            except ValueError:
                pass
        if cands:
            out.append(cands)
    for mon, d, y in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b", text or "", re.I):
        try:
            out.append({datetime.date(int(y), MONTHS[mon.lower()], int(d))})
        except ValueError:
            pass
    return out


def names_other_game(title: str, date: str) -> str | None:
    """Why a title belongs to another game: a postseason game, a year outside the game's season, or
    explicit dates none of which is the game's date. None if nothing contradicts."""
    if POSTSEASON_RE.search(title or ""):
        return "postseason game"
    d = datetime.date.fromisoformat(date)
    s0 = d.year if d.month >= 9 else d.year - 1
    years = {int(y) for y in re.findall(r"\b(19\d\d|20[0-3]\d)\b", title or "")}
    if years - {s0, s0 + 1}:
        return "year %s" % ",".join(str(y) for y in sorted(years - {s0, s0 + 1}))
    ds = dates_in(title)
    if ds and not any(d in c for c in ds):
        return "another date"
    return None


def candidate_rejection(title: str, channel: str | None, date: str) -> str | None:
    if is_video_game(title, channel):
        return "video-game simulation"
    return names_other_game(title, date)


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout after %d s" % timeout


def search(query: str) -> tuple[list[dict], str | None]:
    code, out, err = _run(YT_BASE + ["--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel)s\t%(title)s",
                                     "ytsearch10:" + query])
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
            res.append({"id": parts[0], "duration": dur, "channel": parts[2], "title": parts[3]})
    return res, None


def probe(video_id: str, fmt: str) -> dict:
    """Metadata plus the format harvest_driver would download, without downloading."""
    code, out, err = _run(YT_BASE + ["--no-playlist", "--skip-download", "-f", fmt, "-j",
                                     "https://www.youtube.com/watch?v=" + video_id])
    if code != 0:
        msg = (err.strip().splitlines() or ["yt-dlp exit %d" % code])[-1][:200]
        return {"error": msg, "no_format": "requested format is not available" in msg.lower()}
    j = json.loads(out)
    return {"title": j.get("title"), "channel": j.get("channel"), "upload_date": j.get("upload_date"),
            "duration": j.get("duration"), "description": (j.get("description") or "")[:2000],
            "format_id": j.get("format_id"), "vcodec": j.get("vcodec"), "height": j.get("height"), "fps": j.get("fps")}


def classify(c: dict, g: dict) -> str:
    """Verdict for one probed candidate."""
    if c.get("error"):
        return "no_avc1" if c.get("no_format") else "probe_error"
    if candidate_rejection(c.get("title") or c.get("search_title"), c.get("channel"), g["date"]):
        return "no_broadcast"
    if not (mentions(c["title"], g["away"]) or mentions(c["description"], g["away"])) or \
       not (mentions(c["title"], g["home"]) or mentions(c["description"], g["home"])):
        return "no_broadcast"
    if not str(c.get("vcodec") or "").startswith("avc1"):
        return "no_avc1"
    dur = c.get("duration") or 0
    if dur < FULL_S:
        return "short_broadcast"
    if (c.get("height") or 0) < 720:
        return "low_res"
    return "ok_unverified" if c.get("season_evidence") else "ambiguous"


def broadcast_check(g: dict, registered: dict, fmt: str) -> dict:
    queries, seen, errors = query_forms(g), {}, []
    for q in queries:
        res, err = search(q)
        if err:
            errors.append(err)
        for r in res:
            seen.setdefault(r["id"], r)
        time.sleep(1.0)
    cands, rejected = [], {}
    for r in seen.values():
        other = registered.get(r["id"])
        if other and other != g["date"]:
            continue                                         # registered to another game
        if (r["duration"] or 0) < MIN_BROADCAST_S:
            continue
        if not (mentions(r["title"], g["away"]) or mentions(r["title"], g["home"])):
            continue
        why = candidate_rejection(r["title"], r.get("channel"), g["date"])
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        r = dict(r)
        r["title_evidence"] = season_evidence(g["date"], r["title"], "", None)
        cands.append(r)
    cands.sort(key=lambda r: (r["title_evidence"] is None, -(r["duration"] or 0)))
    probed = []
    for r in cands[:MAX_PROBES]:
        p = probe(r["id"], fmt)
        time.sleep(1.0)
        c = {"id": r["id"], "search_title": r["title"], **p}
        if not p.get("error"):
            c["season_evidence"] = season_evidence(g["date"], p["title"], p["description"], p["upload_date"])
        c["verdict"] = classify(c, g)
        c.pop("description", None)
        probed.append(c)
    if probed:
        best = max(probed, key=lambda c: PRECEDENCE.index(c["verdict"]))   # later in PRECEDENCE = better
        verdict = best["verdict"]
    else:
        best, verdict = None, ("probe_error" if errors and not seen else "no_broadcast")
    return {"verdict": verdict, "best": best, "probed": probed, "queries": queries, "search_results": len(seen),
            "candidates_over_min": len(cands), "rejected_before_probe": rejected, "search_errors": errors}


# ── final verdicts across games ───────────────────────────────────────────────
def upload_conflict(g: dict, upload_date: str, schedule: list[dict]) -> str | None:
    """Another game between the same two teams in the SEASON_WINDOW_DAYS before the upload, if any."""
    up = datetime.date(int(upload_date[:4]), int(upload_date[4:6]), int(upload_date[6:8]))
    for o in schedule:
        if o["game"] != g["game"] and {o["away"], o["home"]} == {g["away"], g["home"]}:
            if 0 <= (up - datetime.date.fromisoformat(o["date"])).days <= SEASON_WINDOW_DAYS:
                return o["game"]
    return None


def final_candidate(c: dict, g: dict, schedule: list[dict]) -> tuple[str, str | None]:
    """A probed candidate's verdict once its evidence is weighed: only an exact date, or an upload
    window no other game of the matchup shares, keeps ok_unverified."""
    v = c["verdict"]
    why = candidate_rejection(c.get("title") or c.get("search_title"), c.get("channel"), g["date"]) if not c.get("error") else None
    if why:
        return "no_broadcast", why
    if v != "ok_unverified":
        return v, None
    k = evidence_kind(c.get("season_evidence"))
    if k == "date":
        return v, None
    if k == "upload":
        other = upload_conflict(g, c["upload_date"], schedule)
        return ("ambiguous", "upload window also fits " + other) if other else (v, None)
    return "ambiguous", "season string only"


def finalize(games: list[dict], parts: dict, schedule: list[dict], overrides: dict | None = None) -> dict:
    """Final broadcast verdict and best candidate per game; a video picked for several games stays
    only with the game whose exact date it names."""
    out = {}
    for g in games:
        bc = parts[g["game"]].get("broadcast") or {}
        if bc.get("verdict") == "held_out":
            out[g["game"]] = {"verdict": "held_out", "best": bc.get("best"), "note": bc.get("note")}
            continue
        probed = bc.get("probed") or []
        for c in probed:
            c["final"], c["note"] = final_candidate(c, g, schedule)
        if probed:
            best = max(probed, key=lambda c: PRECEDENCE.index(c["final"]))     # later in PRECEDENCE = better
            out[g["game"]] = {"verdict": best["final"], "best": best, "note": best["note"]}
        else:
            out[g["game"]] = {"verdict": bc.get("verdict", "probe_error"), "best": None, "note": bc.get("note")}
    dated_for = {}                                         # a video whose text names one game's date is that game
    for name, f in out.items():
        b = f.get("best") or {}
        if b.get("id") and evidence_kind(b.get("season_evidence")) == "date":
            dated_for.setdefault(b["id"], []).append(name)
    for name, f in out.items():
        b = f.get("best") or {}
        owners = [n for n in dated_for.get(b.get("id"), []) if n != name]
        if owners and evidence_kind(b.get("season_evidence")) != "date":
            f["verdict"], f["note"] = "no_broadcast", "video is dated for " + ", ".join(owners)
    by_id = {}
    for name, f in out.items():
        b = f.get("best") or {}
        if f["verdict"] == "ok_unverified" and b.get("id"):
            by_id.setdefault(b["id"], []).append(name)
    for names in by_id.values():
        if len(names) > 1:
            dated = [n for n in names if evidence_kind(out[n]["best"].get("season_evidence")) == "date"]
            for n in names:
                if not (len(dated) == 1 and n == dated[0]):
                    out[n]["verdict"], out[n]["note"] = "ambiguous", "same video picked for " + ", ".join(x for x in names if x != n)
    for name, o in (overrides or {}).items():            # a human decision outranks the search; the probe still judges format
        if name not in out or not isinstance(o, dict):
            continue
        p = dict(o.get("probe") or {})
        p.update({"id": o["video_id"], "season_evidence": "human: %s, %s (%s)" % (o.get("by"), o.get("date"), o.get("note", ""))})
        if p.get("error"):
            v = "no_avc1" if p.get("no_format") else "probe_error"
        elif not str(p.get("vcodec") or "").startswith("avc1"):
            v = "no_avc1"
        elif (p.get("height") or 0) < 720:
            v = "low_res"
        else:
            v = "ok_unverified"
        out[name] = {"verdict": v, "best": p, "note": "human decision"}
    return out


# ── verdict, priority, report ─────────────────────────────────────────────────
def overall(sv: dict | None, bc: dict | None, held_out: bool) -> str:
    if held_out:
        return "held_out"
    if sv and sv["status"] == "sportvu_broken":
        return "sportvu_broken"
    b = bc["verdict"] if bc else "probe_error"
    if b != "ok_unverified":
        return b
    return "sportvu_gaps" if sv and sv["status"] == "sportvu_gaps" else "ok_unverified"


def priority(rows: list[dict]) -> list[dict]:
    group = {"GSW": 1, "CLE": 2, "OKC": 3, "NYK": 4}
    def key(r):
        return (0 if r["game"] == FIRST else group[r["home"]], 0 if r["verdict"] in USABLE else 1, r["date"])
    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def _clean(s):
    return None if s is None else str(s).replace("—", " - ").replace("–", "-")


def registered_ids() -> dict:
    g = json.loads((config.PROJECT_ROOT / "data" / "harvest" / "games.json").read_text())
    return {v["video_id"]: v.get("date") for v in g.values() if isinstance(v, dict) and v.get("video_id")}


def held_out_sportvu(game: str) -> dict:
    p = SV_DIR / (game + "_moments.json")
    if not p.exists():
        return {"status": "ok", "note": "held-out game; moments not checked"}
    mom = json.loads(p.read_text())["moments"]
    gf = gap_flags(mom)
    return {"status": "sportvu_gaps" if gf["flags"] else "ok", "moments": len(mom), **gf}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retry-errors", action="store_true", help="re-probe games whose verdict was probe_error")
    ap.add_argument("--reprobe-rejected", action="store_true",
                    help="re-probe games whose cached probes include a candidate the current filters reject")
    ap.add_argument("--reprobe-unfiltered", action="store_true",
                    help="re-probe games whose cached search predates the pre-probe filters (no rejection counts)")
    args = ap.parse_args()
    from harvest_driver import YTDLP_FMT
    PARTS.mkdir(parents=True, exist_ok=True)
    all_games = list_all()
    games = list_games(all_games)
    print("games: %d (%s)" % (len(games), ", ".join("%s %d" % (a, sum(g["home"] == a for g in games)) for a in ARENAS)), flush=True)
    parts = {}
    for g in games:
        p = PARTS / (g["game"] + ".json")
        parts[g["game"]] = json.loads(p.read_text()) if p.exists() else {**g}

    def save(name):
        (PARTS / (name + ".json")).write_text(json.dumps(parts[name], indent=1, default=str))

    # SportVU (parallel downloads; archives cached)
    todo = [g for g in games if "sportvu" not in parts[g["game"]] or
            (args.retry_errors and parts[g["game"]]["sportvu"]["status"] == "sportvu_broken" and "bytes" not in parts[g["game"]]["sportvu"].get("error", ""))]
    def sv_job(g):
        res = held_out_sportvu(g["game"]) if g["game"] in HELD_OUT else sportvu_check(g)
        parts[g["game"]]["sportvu"] = res; save(g["game"])
        return g["game"], res["status"]
    with ThreadPoolExecutor(4) as ex:
        for i, (name, st) in enumerate(ex.map(sv_job, todo), 1):
            print("  sportvu %d/%d %s: %s" % (i, len(todo), name, st), flush=True)
    # broadcasts (sequential, throttled; stop probing after repeated yt-dlp failures; rejection counts per rule kept)
    reg = registered_ids()
    consecutive = 0
    for g in games:
        part = parts[g["game"]]
        if g["game"] in HELD_OUT:
            part["broadcast"] = {"verdict": "held_out", "best": {"id": HELD_OUT[g["game"]][1]}, "note": "registry id, not searched"}
            save(g["game"]); continue
        stale = args.reprobe_rejected and any(
            not c.get("error") and candidate_rejection(c.get("title") or c.get("search_title"), c.get("channel"), g["date"])
            for c in (part.get("broadcast") or {}).get("probed") or [])
        stale = stale or (args.reprobe_unfiltered and "rejected_before_probe" not in (part.get("broadcast") or {"rejected_before_probe": 0}))
        if "broadcast" in part and not stale and not (args.retry_errors and part["broadcast"]["verdict"] == "probe_error"):
            continue
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            part["broadcast"] = {"verdict": "probe_error", "best": None, "probed": [], "note": "skipped after %d consecutive yt-dlp failures" % consecutive}
            save(g["game"]); continue
        bc = broadcast_check(g, reg, YTDLP_FMT)
        consecutive = consecutive + 1 if bc["verdict"] == "probe_error" else 0
        part["broadcast"] = bc; save(g["game"])
        b = bc.get("best") or {}
        print("  broadcast %s: %s %s %s" % (g["game"], bc["verdict"], b.get("id", ""), _clean(b.get("title")) or ""), flush=True)

    overrides = {k: v for k, v in json.loads(OVERRIDES.read_text()).items() if not k.startswith("_")} if OVERRIDES.exists() else {}
    for name, o in overrides.items():                    # probe each human-chosen video once (cached in its part)
        if name in parts and (parts[name].get("override_probe") or {}).get("id") != o["video_id"]:
            parts[name]["override_probe"] = {"id": o["video_id"], **probe(o["video_id"], YTDLP_FMT)}
            parts[name]["override_probe"].pop("description", None)
            save(name)
        if name in parts:
            o["probe"] = parts[name]["override_probe"]
    final = finalize(games, parts, all_games, overrides)
    rows = []
    for g in games:
        part = parts[g["game"]]
        sv, bc = part.get("sportvu"), final[g["game"]]
        queries = (part.get("broadcast") or {}).get("queries")
        b = bc.get("best") or {}
        rows.append({"game": g["game"], "date": g["date"], "away": g["away"], "home": g["home"],
                     "verdict": overall(sv, bc, g["game"] in HELD_OUT),
                     "sportvu": {k: sv.get(k) for k in ("status", "moments", "flags", "gap_s_total", "error") if sv and k in sv},
                     "broadcast": {"verdict": bc.get("verdict"), "note": bc.get("note"), "video_id": b.get("id"),
                                   "url": "https://www.youtube.com/watch?v=" + b["id"] if b.get("id") else None,
                                   "title": _clean(b.get("title") or b.get("search_title")), "channel": _clean(b.get("channel")),
                                   "upload_date": b.get("upload_date"), "duration_s": b.get("duration"),
                                   "format_id": b.get("format_id"), "vcodec": b.get("vcodec"), "height": b.get("height"), "fps": b.get("fps"),
                                   "evidence": b.get("season_evidence"), "evidence_kind": evidence_kind(b.get("season_evidence")),
                                   "error": b.get("error"), "queries": queries,
                                   "rejected_before_probe": (part.get("broadcast") or {}).get("rejected_before_probe"),
                                   "candidates_probed": len((part.get("broadcast") or {}).get("probed") or [])}})
    rows = priority(rows)
    counts = {a: {} for a in ARENAS}
    rejections = {a: {} for a in ARENAS}
    for r in rows:
        counts[r["home"]][r["verdict"]] = counts[r["home"]].get(r["verdict"], 0) + 1
        for why, n in (r["broadcast"].get("rejected_before_probe") or {}).items():
            rejections[r["home"]][why] = rejections[r["home"]].get(why, 0) + n
    rep = {"item": "ROADMAP R1.1", "date": datetime.date.today().isoformat(), "source": LIST_URL,
           "rules": {"min_archive_bytes": MIN_ARCHIVE_BYTES, "gap_flag_s": GAP_FLAG_S, "full_broadcast_s": FULL_S,
                     "min_broadcast_s": MIN_BROADCAST_S, "season_window_days": SEASON_WINDOW_DAYS, "max_probes": MAX_PROBES,
                     "format": YTDLP_FMT, "precedence": PRECEDENCE, "usable": USABLE},
           "counts": counts, "candidate_rejections_by_arena": rejections, "usable_by_arena": {a: sum(c.get(v, 0) for v in USABLE) for a, c in counts.items()},
           "games": rows,
           "caveats": ["a broadcast verdict is a search result, not an identity check: the scorebug confirms the game at R1.2",
                       "YouTube availability changes over time and search results vary; probe_error means the probe failed, not that no video exists",
                       "SportVU flags count clock gaps above %g s; short stoppage-free gaps below that are not listed" % GAP_FLAG_S]}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "sportvu_manifest.json").write_text(json.dumps(rep, indent=1, default=str))
    L = ["SPORTVU AVAILABILITY MANIFEST (ROADMAP R1.1), %d games, %s" % (len(rows), rep["date"]),
         "verdicts per arena (usable = ok_unverified or sportvu_gaps):"]
    for a in ARENAS:
        L.append("  %s  usable %2d of %2d | %s" % (a, rep["usable_by_arena"][a], sum(counts[a].values()),
                                                 ", ".join("%s %d" % kv for kv in sorted(counts[a].items(), key=lambda kv: PRECEDENCE.index(kv[0])))))
    L += ["candidate videos rejected before probing, per arena and rule:"]
    for a in ARENAS:
        L.append("  %s  %s" % (a, ", ".join("%s %d" % kv for kv in sorted(rejections[a].items(), key=lambda kv: -kv[1])) or "none"))
    L += ["", "priority list:"]
    for r in rows:
        b = r["broadcast"]
        vid = b.get("video_id") if r["verdict"] not in ("no_broadcast",) else None
        L.append("  %2d %-24s %-15s %s %s %s%s" % (r["rank"], r["game"], r["verdict"], vid or "-",
                                                    "%ds" % b["duration_s"] if vid and b.get("duration_s") else "",
                                                    ("%sp %s" % (b["height"], b.get("evidence_kind") or "")).strip() if vid and b.get("height") else "",
                                                    ("  | " + "; ".join(r["sportvu"].get("flags") or [])[:120]) if r["sportvu"].get("flags") else ""))
    L += ["", "caveats:"] + ["  - " + c for c in rep["caveats"]]
    txt = "\n".join(L)
    (config.REPORTS_DIR / "sportvu_manifest.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
