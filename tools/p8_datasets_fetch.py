"""tools/p8_datasets_fetch.py: download outside labelled basketball datasets that need no account (P8; the lead's
task after docs/research/P8_datasets.md). Download only: nothing here trains, runs a model or opens the files
beyond listing, hashing and reading their license text.

Each set goes to its own folder <P8 data>/datasets/<name>/, outside the repository; contents are untrusted data and
nothing inside them is run. Sizes are read before each download and anything over MAX_BYTES (30 GB) is skipped and
reported. A gated or private Hugging Face repo stops the step. The fetch record (source URL, revision, files with
bytes and sha256, license text as found, date) is <P8 data>/datasets/fetch_record.json, and the same without any
path in reports/research/p8_datasets_fetch.json.

SHOT (muyu111/basketball, branch shotdatasets) is screened before any frame or video is fetched: its clip and game
names are listed from annotation and metadata files, and a clip is flagged (and not downloaded) when its name could
be a 2015-16 game in Miami, Toronto or Los Angeles, any game at CLE, or PHX at GSW (docs/research/P8_datasets.md,
rule 3). Clip names carry team codes, not always dates or a home side, so the screen flags on team codes alone where
the home side or season cannot be read.

    python3 -m tools.p8_datasets_fetch hf REPO [--rev REV] [--name NAME] [--only GLOB ...]
    python3 -m tools.p8_datasets_fetch url NAME URL [--file FILENAME]
    python3 -m tools.p8_datasets_fetch shot-screen                # screen SHOT's clip names, no request
    python3 -m tools.p8_datasets_fetch home-screen                # E-BARD, WASB, PL-NBA: drop D85 home games
    python3 -m tools.p8_datasets_fetch annotate                   # screen character, non-NBA names, empty files, gaps
    python3 -m tools.p8_datasets_fetch record                     # rewrite the repository copy of the record
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import re
import sys
import subprocess
import urllib.parse
from pathlib import Path

from tools.p8_common import REPO, atomic_write, check_outside, p8_root

WHO = "tools.p8_datasets_fetch"
MAX_BYTES = 30 * 1024 ** 3
WORKERS = 8                                # parallel downloads of one set
MANIFEST_OVER = 50                         # sets with more files keep their file list in a workspace manifest
UA = "generic-research-client/1.0"
# The SHOT screen (docs/research/P8_datasets.md rule 3): team codes as clip names write them.
SCREEN_HOME_2015_16 = {"MIA", "TOR", "LAC", "LAL"}
SCREEN_ANY_CLE = {"CLE"}
SCREEN_PAIR = ({"PHX", "PHO"}, {"GSW", "GS"})
TEAM_ALIASES = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "NJ": "BKN", "PHO": "PHX", "UTAH": "UTA",
                "WSH": "WAS", "BRK": "BKN", "LA": "LAC"}


def datasets_root() -> Path:
    d = p8_root().parent / "datasets"
    check_outside(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def curl(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    """curl with a generic user agent (system certificates, which `python3 -I` cannot see through certifi)."""
    return subprocess.run(["curl", "-sS", "-L", "--fail", "-A", UA, *args], capture_output=True, timeout=timeout)


def get_json(url: str, tries: int = 6):
    """-> (json, headers as a dict of lower-cased names). Retries with a growing wait (HTTP 429 from the hub)."""
    import time
    for k in range(tries):
        r = curl(["-D", "-", url], 120)
        if r.returncode == 0:
            break
        if k == tries - 1:
            raise SystemExit(f"{WHO}: {url}: {r.stderr.decode(errors='replace').strip()[-200:]}")
        time.sleep(30 * 2 ** k)
    raw = r.stdout.decode(errors="replace")
    blocks = raw.split("\r\n\r\n")
    heads, body = [], raw
    while blocks and blocks[0].startswith("HTTP/"):
        heads.append(blocks.pop(0))
        body = "\r\n\r\n".join(blocks)
    headers = {}
    for line in (heads[-1] if heads else "").split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return json.loads(body), headers


def stream(url: str, dest: Path, timeout: int = 7200) -> dict:
    """Download url to dest (atomic) -> {bytes, sha256}."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    import time
    for k in range(5):
        r = curl(["-o", str(part), url], timeout)
        if r.returncode == 0 and part.exists():
            break
        time.sleep(30 * 2 ** k)
    if r.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        raise SystemExit(f"{WHO}: {url}: {r.stderr.decode(errors='replace').strip()[-200:]}")
    part.replace(dest)
    return {"bytes": dest.stat().st_size, "sha256": sha256_of(dest)}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── records ───────────────────────────────────────────────────────────────────
def record_path() -> Path:
    return datasets_root() / "fetch_record.json"


def load_record() -> dict:
    p = record_path()
    return json.loads(p.read_text()) if p.exists() else {}


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def save_record(rec: dict) -> None:
    """The workspace record in full. The repository copy has email addresses removed (CLAUDE.md rule 9), long
    dashes replaced by hyphens (rule 12) and, for a set of more than MANIFEST_OVER files, the file list replaced by
    its count, bytes and the sha256 of a manifest file kept in the workspace (<name>.manifest.json: path, bytes,
    sha256 of every file)."""
    atomic_write(record_path(), json.dumps(rec, indent=1, ensure_ascii=False))
    short = {}
    for name, e in rec.items():
        e = dict(e)
        files = e.get("files") or {}
        if len(files) > MANIFEST_OVER:
            man = json.dumps({k: [v["bytes"], v["sha256"]] for k, v in sorted(files.items())}, separators=(",", ":"))
            mp = datasets_root() / ("%s.manifest.json" % name)
            atomic_write(mp, man)
            e["files"] = {"count": len(files), "bytes": sum(v["bytes"] for v in files.values()),
                          "manifest_sha256": hashlib.sha256(man.encode()).hexdigest(),
                          "lfs_sha256_mismatches": sum(1 for v in files.values() if v.get("lfs_sha256_matches") is False),
                          "listed_size_mismatches": sum(1 for v in files.values()
                                                        if v.get("listed_size") is not None and v["bytes"] != v["listed_size"])}
        short[name] = e
    text = json.dumps(short, indent=1, ensure_ascii=False)
    text = text.replace("\u2014", "-").replace("\u2013", "-")      # CLAUDE.md rule 12, in quoted text too
    atomic_write(REPO / "reports" / "research" / "p8_datasets_fetch.json", EMAIL_RE.sub("[email removed]", text) + "\n")


# ── Hugging Face ──────────────────────────────────────────────────────────────
def hf_info(repo: str) -> dict:
    d, _ = get_json("https://huggingface.co/api/datasets/%s" % repo)
    return d


def hf_tree(repo: str, rev: str = "main") -> list[dict]:
    """Every file of a dataset repo at rev (paginated) -> [{path, size, lfs_sha256}]."""
    url = "https://huggingface.co/api/datasets/%s/tree/%s?recursive=true&expand=false" % (repo, urllib.parse.quote(rev, safe=""))
    out = []
    while url:
        d, headers = get_json(url)
        out += [{"path": f["path"], "size": f.get("size"), "lfs_sha256": (f.get("lfs") or {}).get("oid")}
                for f in d if f.get("type") == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link") or "")
        url = m.group(1) if m else None
    return out


def hf_url(repo: str, rev: str, path: str) -> str:
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (repo, urllib.parse.quote(rev, safe=""),
                                                                  urllib.parse.quote(path))


def license_text(folder: Path, files: list[str]) -> dict:
    """License text as found: LICENSE files whole, and the license line of a README's metadata block."""
    out = {}
    for f in files:
        p = folder / f
        if not p.exists() or p.stat().st_size > 200_000:
            continue
        low = f.lower()
        if "license" in low or "licence" in low:
            out[f] = p.read_text(errors="replace")
        elif low.endswith("readme.md"):
            lines = [l for l in p.read_text(errors="replace").splitlines() if "licen" in l.lower()]
            if lines:
                out[f] = "\n".join(lines)
    return out


def fetch_hf(repo: str, rev: str = "main", name: str | None = None, only: list[str] | None = None,
             clear_frames: bool = False) -> dict:
    info = hf_info(repo)
    if info.get("gated") or info.get("private"):
        raise SystemExit(f"{WHO}: {repo} is gated or private; stopped (no account, no request)")
    name = name or repo.split("/")[-1]
    folder = datasets_root() / name
    cache = datasets_root() / ("%s.tree.%s.json" % (name, info.get("sha") if rev == "main" else rev))
    if cache.exists():
        tree = json.loads(cache.read_text())
    else:
        tree = hf_tree(repo, rev)
        atomic_write(cache, json.dumps(tree))
    todo = [f for f in tree if not only or any(fnmatch.fnmatch(f["path"], g) for g in only)]
    if clear_frames:                               # SHOT: frames and keyframes of clips the screen cleared, only
        sc = json.loads((datasets_root() / ("%s.screen.json" % name)).read_text())
        clear = {c for c, why in sc.items() if why is None}
        todo = [f for f in todo if f["path"].count("/") >= 3 and f["path"].split("/")[2] in clear
                and ("/frames/" in f["path"] or "/keyframes/" in f["path"])]
    total = sum(f["size"] or 0 for f in todo)
    rec = load_record()
    entry = rec.get(name) or {"source": "https://huggingface.co/datasets/%s" % repo, "revision": rev, "files": {}}
    entry.update(repo_sha=info.get("sha"), license_tag=[t for t in info.get("tags", []) if t.startswith("license")],
                 date=dt.date.today().isoformat(), listed_files=len(tree), listed_bytes=sum(f["size"] or 0 for f in tree))
    if total > MAX_BYTES:
        entry["skipped"] = "selected files total %d bytes, over the 30 GB limit" % total
        rec[name] = entry
        save_record(rec)
        print(entry["skipped"])
        return entry
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [f for f in todo if not (f["path"] in entry["files"] and (folder / f["path"]).exists())]

    def one(f):
        r = stream(hf_url(repo, rev, f["path"]), folder / f["path"])
        r["lfs_sha256_matches"] = (r["sha256"] == f["lfs_sha256"]) if f["lfs_sha256"] else None
        r["listed_size"] = f["size"]
        return f["path"], r

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, f) for f in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            path, r = fut.result()
            entry["files"][path] = r
            if k % 500 == 0 or k == len(futs) or r["bytes"] > 100_000_000:
                rec[name] = entry
                save_record(rec)
                print("%s %d of %d files (%s %d bytes)" % (name, k, len(futs), path, r["bytes"]), flush=True)
    entry["license_text"] = license_text(folder, list(entry["files"]))
    rec[name] = entry
    save_record(rec)
    return entry


# ── plain URLs (GitHub, Google Drive) ─────────────────────────────────────────
def head_size(url: str) -> int | None:
    r = curl(["-I", url], 60)
    if r.returncode != 0:
        return None
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", r.stdout.decode(errors="replace"))
    return int(sizes[-1]) if sizes else None


def fetch_url(name: str, url: str, filename: str, source: str, note: str = "") -> dict:
    folder = datasets_root() / name
    size = head_size(url)
    rec = load_record()
    entry = rec.get(name) or {"source": source, "files": {}}
    entry.update(date=dt.date.today().isoformat(), download_url=url, head_bytes=size, note=note or entry.get("note"))
    if size and size > MAX_BYTES:
        entry["skipped"] = "%d bytes, over the 30 GB limit" % size
    else:
        r = stream(url, folder / filename, timeout=3600)
        entry["files"][filename] = r
        print("%s %s %d bytes sha256 %s" % (name, filename, r["bytes"], r["sha256"][:16]), flush=True)
    rec[name] = entry
    save_record(rec)
    return entry


# ── SHOT screen (pure, tested) ────────────────────────────────────────────────
def teams_in_name(name: str) -> list[str]:
    """Team codes in a clip name such as 'CLEvsGS', 'ATL_vs_NJ' or 'LAL-BOS', in order, aliases resolved."""
    m = re.search(r"([A-Za-z]{2,4})\s*[-_ ]?\s*(?:vs\.?|v|@|at)\s*[-_ ]?\s*([A-Za-z]{2,4})", name, re.I)
    if not m:
        return []
    return [TEAM_ALIASES.get(t.upper(), t.upper()) for t in (m.group(1), m.group(2))]


def screen(name: str) -> str | None:
    """The reason a SHOT clip is flagged, or None when it is clear. Clip names give no home side or season for
    certain, so a clip with MIA, TOR, LAC or LAL in it is flagged unless a year outside 2015-16 is written in it."""
    teams = teams_in_name(name)
    if not teams:
        return "no team codes readable"
    s = set(teams)
    if s & SCREEN_ANY_CLE:
        return "a game with CLE"
    if s & {"PHX"} and s & {"GSW"}:
        return "PHX and GSW"
    years = {int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", name)}
    if s & SCREEN_HOME_2015_16 and not (years and all(y not in (2015, 2016) for y in years)):
        return "MIA, TOR or a Los Angeles team, season not excluded"
    return None


def shot_screen(name: str = "SHOT", rev: str = "shotdatasets") -> dict:
    """Screen every SHOT clip (sample folder) by name -> {clip: reason or None}; saved beside the record and
    summarised in it. Read from the cached file list, so no request is made."""
    from tools.research_testset import refuse_cv_test
    tree = json.loads((datasets_root() / ("%s.tree.%s.json" % (name, rev))).read_text())
    clips = sorted({f["path"].split("/")[2] for f in tree if f["path"].count("/") >= 3})
    out = {c: screen(c.split("-")[0]) for c in clips}
    clear = [c for c, why in out.items() if why is None]
    refuse_cv_test(clear, WHO)                         # the broadcast CV test set, by substring, on every clear clip
    atomic_write(datasets_root() / ("%s.screen.json" % name), json.dumps(out, indent=1))
    rec = load_record()
    games = sorted({c.split("-")[0] for c in clips})
    reasons = {}
    for why in out.values():
        reasons[str(why)] = reasons.get(str(why), 0) + 1
    rec[name]["screen"] = {"clips": len(clips), "game_names": len(games), "clear": len(clear),
                           "flagged": len(clips) - len(clear), "by_reason": reasons,
                           "flagged_game_names": sorted({c.split("-")[0] for c, w in out.items() if w}),
                           "rule": "docs/research/P8_datasets.md rule 3; tools.p8_datasets_fetch.screen"}
    save_record(rec)
    return rec[name]["screen"]


NON_NBA_SHOT = ("BKvsBJ", "BYvsFJ", "FJvsJS", "JSvsSD", "SCvsBK", "SXvsBK")   # Chinese-league game names in SHOT


def annotate() -> dict:
    """The lead's notes (D37 additions) written into the record from the files themselves: the SHOT screen's
    character, its non-NBA game names, its empty files (listed and downloaded), and the gaps found in WASB and PL-NBA."""
    import zipfile
    root = datasets_root()
    rec = load_record()
    sh = rec["SHOT"]
    tree = json.loads((root / "SHOT.tree.shotdatasets.json").read_text())
    def empty(files, ext):
        sel = [f for f in files if f[0].lower().endswith(ext)]
        return {"empty": sum(1 for f in sel if f[1] == 0), "of": len(sel)}
    listed = [(f["path"], f["size"]) for f in tree]
    got = [(k, v["bytes"]) for k, v in sh["files"].items()]
    sh["screen"]["character"] = "conservative: team code only, no season or home side in clip names"
    flagged_names = sh["screen"]["flagged_game_names"]
    sh["screen"]["flagged_names_predating_brooklyn"] = {
        "count": sum(1 for n in flagged_names if "NJ" in teams_in_name(n) or n.startswith("NJvs") or n.endswith("vsNJ")),
        "of": len(flagged_names),
        "note": "names with NJ (the New Jersey Nets) predate the move to Brooklyn; still flagged, as the screen reads "
                "no season"}
    lic = (root / "SHOT" / "LICENSE").read_text(errors="replace")
    sh["license_scope"] = " ".join(lic.split("The unmodified Creative Commons license text follows.")[0].split())
    sh["non_nba_game_names"] = {"names": list(NON_NBA_SHOT), "tag": "non_nba",
                                "clips": sorted(c for c in json.loads((root / "SHOT.screen.json").read_text())
                                                if c.split("-")[0] in NON_NBA_SHOT),
                                "note": "kept; whether they train is decided per experiment"}
    sh["empty_files"] = {"listed_in_repo": {"xml": empty(listed, ".xml"), "jpg": empty(listed, ".jpg")},
                         "downloaded": {"xml": empty(got, ".xml"), "jpg": empty(got, ".jpg"),
                                        "txt": empty(got, ".txt"), "json": empty(got, ".json")}}
    rec["WASB-basketball"]["gaps"] = ["The repository's license file (LICENSE.md, MIT, 'Copyright (c) 2023 NTT "
                                      "Communications Corporation') covers 'this software and associated documentation "
                                      "files'; it does not mention the annotation zip, so the labels' license is not "
                                      "stated. Not worked around."]
    folders = sorted({n.split("/")[1] for n in zipfile.ZipFile(root / "PL-NBA" / "PL-NBA-JSON.zip").namelist()
                      if n.count("/") >= 2})
    rec["PL-NBA"]["gaps"] = ["%d game folders in PL-NBA-JSON.zip against the README's 'collected from 40 official NBA "
                             "games'. Not worked around." % len(folders)]
    save_record(rec)
    return {"shot_empty": sh["empty_files"], "non_nba_clips": len(sh["non_nba_game_names"]["clips"]),
            "pl_nba_folders": len(folders)}


# ── home-team screen of E-BARD, WASB and PL-NBA (pure parts tested) ─────────────
D85_HOMES = {"MIA", "TOR", "LAC", "LAL"}
SCHEDULE_FILES = ("shotdetail_2018", "shotdetail_po_2018", "shotdetail_2022", "shotdetail_po_2022", "shotdetail_2023",
                  "shotdetail_po_2023", "shotdetail_2024", "shotdetail_po_2024")
SCHEDULE_SOURCE = ("github.com/shufinskiy/nba_data datasets/<file>.tar.xz (stats.nba.com shot chart rows: GAME_ID, "
                   "GAME_DATE, HTM, VTM); repository Apache-2.0, rows are NBA Statistics under the nba.com Terms of Use "
                   "(docs/DATA_SOURCES.md, NBA shot chart mirror row)")
CODE_ALIASES = {"WSH": "WAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "PHO": "PHX", "UTAH": "UTA",
                "BRK": "BKN"}


D149_SEASONS = {"2024-25", "2025-26"}
D149_NOTE = ("D149: D79 binds the play-by-play lanes only; frames of this game may train a detector, and no output "
             "on this game flows into a play-by-play lane")


def season_of(game_id: str) -> str:
    """The season of a 10-digit NBA game id: digits 4 and 5 are the season's first year ('0022400861' -> 2024-25)."""
    y = 2000 + int(game_id[3:5])
    return "%d-%02d" % (y, (y + 1) % 100)


def frame_map(zips) -> dict:
    """E-BARD: whether every image names its game (the game id is in each image file name)."""
    import zipfile
    out = {}
    for z in zips:
        imgs = [n for n in zipfile.ZipFile(z).namelist() if re.search(r"\.(jpe?g|png)$", n, re.I)]
        out[z.parent.name] = {"images": len(imgs), "images_without_game_id": sum(1 for n in imgs
                                                                                  if not re.search(r"00\d{8}", n))}
    return out


def schedule(root: Path) -> dict:
    """{10-digit game id: (iso date, home, away)} from the mirror's shot chart files (stdlib csv)."""
    import csv
    out = {}
    for f in SCHEDULE_FILES:
        with open(root / f / ("%s.csv" % f), newline="") as fh:
            for row in csv.DictReader(fh):
                gid = "%010d" % int(row["GAME_ID"])
                if gid not in out:
                    d = row["GAME_DATE"]
                    out[gid] = ("%s-%s-%s" % (d[:4], d[4:6], d[6:8]), row["HTM"], row["VTM"])
    return out


def resolve_folder(folder: str, sched: dict) -> tuple[str | None, str]:
    """A PL-NBA folder 'YYYYMMDD_T1_T2_label' -> (game id or None, how). A game resolves when the date (one day of
    slack) and the recognised team codes pick exactly one game; an unrecognised code is not guessed."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})_([A-Z]{2,4})_([A-Z]{2,4})", folder)
    if not m:
        return None, "folder name not parsed"
    day = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    known = {t for t in (CODE_ALIASES.get(c, c) for c in (m.group(4), m.group(5)))
             if t in {x for v in sched.values() for x in v[1:]}}
    if not known:
        return None, "no team code recognised"
    hits = [g for g, (d, h, a) in sched.items() if abs((dt.date.fromisoformat(d) - day).days) <= 1 and known <= {h, a}]
    if len(hits) != 1:
        return None, "%d games match" % len(hits)
    return hits[0], "date and %s" % ("both teams" if len(known) == 2 else "one team (%s)" % sorted(known)[0])


def home_screen() -> dict:
    """D150-style home screen of E-BARD, WASB and PL-NBA: drop MIA, TOR, LAC and LAL home games and unresolved
    games; recorded per set."""
    import zipfile
    root = datasets_root()
    sched = schedule(p8_root().parent / "nba_shots")
    rec = load_record()
    sets = {"E-BARD": [root / n / "all.zip" for n in ("E-BARD-detection", "E-BARD-TeamAttribution",
                                                      "E-BARD-JerseyNumberRecognition", "E-BARD-ObjectClassification")],
            "WASB-basketball": [root / "WASB-basketball" / "basketball_annos.zip"],
            "PL-NBA": [root / "PL-NBA" / "PL-NBA-JSON.zip"]}
    out = {}
    for name, zips in sets.items():
        names = [n for z in zips for n in zipfile.ZipFile(z).namelist()]
        rows = {}
        if name == "E-BARD":
            for gid in sorted({m for n in names for m in re.findall(r"00\d{8}", n)}):
                rows[gid] = (gid, "game id")
        elif name == "WASB-basketball":
            for gid in sorted({n.split("/")[1] for n in names if n.count("/") >= 2}):
                rows[gid] = ("%010d" % int(gid), "game id")
        else:
            for f in sorted({n.split("/")[1] for n in names if n.count("/") >= 2}):
                rows[f] = resolve_folder(f, sched)
        kept, dropped, unresolved = [], [], []
        for key, (gid, how) in rows.items():
            g = sched.get(gid) if gid else None
            if not g:
                u = {"id": key, "why": how if not gid else "not in the schedule source",
                     "reason": "dropped: home team not resolved (D150 screen cannot be applied)"}
                m = re.match(r"(\d{4})(\d{2})\d{2}_", key)
                if m:                                       # PL-NBA folder: the season from its date
                    y = int(m.group(1)) - (1 if int(m.group(2)) < 8 else 0)
                    u["season_by_folder_date"] = "%d-%02d" % (y, (y + 1) % 100)
                    if u["season_by_folder_date"] in D149_SEASONS:
                        u["d149"] = D149_NOTE + " (dropped here in any case)"
                unresolved.append(u)
                continue
            row = {"id": key, "game_id": gid, "season": season_of(gid), "date": g[0], "home": g[1], "away": g[2],
                   "resolved_by": how}
            if row["season"] in D149_SEASONS:
                row["d149"] = D149_NOTE
            if g[1] in D85_HOMES:
                dropped.append(dict(row, reason="dropped: home game of %s, a D85 arena (D150)" % g[1]))
            else:
                kept.append(dict(row, reason="kept: home %s is not MIA, TOR, LAC or LAL; not a 2015-16 game" % g[1]))
        out[name] = {"games": len(rows), "kept": kept, "dropped_d85_home": dropped, "dropped_unresolved": unresolved,
                     "frame_to_game": frame_map(zips) if name == "E-BARD" else None,
                     "schedule_source": SCHEDULE_SOURCE,
                     "schedule_files": {f: {"bytes": (p8_root().parent / "nba_shots" / (f + ".tar.xz")).stat().st_size,
                                            "sha256": sha256_of(p8_root().parent / "nba_shots" / (f + ".tar.xz"))}
                                        for f in SCHEDULE_FILES},
                     "note": "Home teams MIA, TOR, LAC and LAL dropped (the D85 arenas); unresolved games dropped and "
                             "counted. The spent held-out games are 2015-16 only (sportvu/splits.json), so none can "
                             "appear in these sets."}
        for k in ([n for n in rec if n.startswith("E-BARD")] if name == "E-BARD" else [name]):
            rec[k]["home_screen"] = out[name]
    save_record(rec)
    return {k: {"games": v["games"], "kept": len(v["kept"]), "dropped_d85_home": len(v["dropped_d85_home"]),
                "dropped_unresolved": len(v["dropped_unresolved"])} for k, v in out.items()}


def main(argv=None) -> int:
    from tools.research_testset import refuse_cv_test
    ap = argparse.ArgumentParser(prog=WHO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf")
    h.add_argument("repo")
    h.add_argument("--rev", default="main")
    h.add_argument("--name")
    h.add_argument("--only", nargs="*")
    h.add_argument("--clear-frames", action="store_true", help="SHOT: frames of screen-cleared clips only")
    u = sub.add_parser("url")
    u.add_argument("name")
    u.add_argument("url")
    u.add_argument("--file", required=True)
    u.add_argument("--source", required=True)
    u.add_argument("--note", default="")
    sub.add_parser("record")
    sub.add_parser("shot-screen")
    sub.add_parser("home-screen")
    sub.add_parser("annotate")
    a = ap.parse_args(argv)
    if a.cmd == "hf":
        refuse_cv_test([a.repo], WHO)
        fetch_hf(a.repo, a.rev, a.name, a.only, a.clear_frames)
    elif a.cmd == "url":
        refuse_cv_test([a.url], WHO)
        fetch_url(a.name, a.url, a.file, a.source, a.note)
    elif a.cmd == "annotate":
        print(json.dumps(annotate(), indent=1))
    elif a.cmd == "home-screen":
        print(json.dumps(home_screen(), indent=1))
    elif a.cmd == "shot-screen":
        print(json.dumps(shot_screen(), indent=1))
    else:
        rec = load_record()
        for name, e in rec.items():                    # license text as found, for sets fetched by URL too
            if not e.get("license_text"):
                e["license_text"] = license_text(datasets_root() / name, list(e.get("files") or {}))
        save_record(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
