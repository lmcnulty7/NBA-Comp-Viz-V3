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
    a = ap.parse_args(argv)
    if a.cmd == "hf":
        refuse_cv_test([a.repo], WHO)
        fetch_hf(a.repo, a.rev, a.name, a.only, a.clear_frames)
    elif a.cmd == "url":
        refuse_cv_test([a.url], WHO)
        fetch_url(a.name, a.url, a.file, a.source, a.note)
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
