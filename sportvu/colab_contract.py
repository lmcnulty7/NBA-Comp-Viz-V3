#!/usr/bin/env python
"""sportvu/colab_contract.py: the ENTIRE Colab run that builds the SportVU 2015-16 analytics dataset on Drive.

Same pattern as colab_run.py and colab_label.py: the notebook (colab_contract.ipynb) is a one-cell
bootstrap that mounts Drive, clones the repo, hard-resets to the branch and runs this file, so a stale
notebook copy cannot run stale code. The notebook then calls drive.flush_and_unmount() and prints the
last line of /content/run_status.txt: RUN OK or RUN FAILED.

Output: My Drive/nba_harvest/sportvu_contract/ (the frames of docs/DATA_CONTRACT.md)
  <game>.parquet            one row per tracked player per frame, ball x/y/z on every row
  <game>_pbp.parquet        basketball-reference play-by-play joined to frames (when the fetch worked)
  heldout/<game>*.parquet   the same files for the held-out games of sportvu/splits.json (never train on them)
  manifest.json             per game: moments, gaps > 2 s, pbp joined, status, split; rewritten after every game
  README.md                 source, coverage, layout, the held-out rule, not for redistribution

Steps, each timed and recorded in the run report:
  env       Colab only; Drive must be mounted AND be the project account (nba_harvest/sportvu/ACCOUNT.txt).
            No GPU needed.
  deps      py7zr (pyarrow, pandas, requests and bs4 ship with Colab).
  list      every game archive in github.com/linouk23/NBA-Player-Movements
            data/2016.NBA.Raw.SportVU.Game.Logs (one GitHub API call), ordered: the training games of
            sportvu/splits.json first, then every other GSW game by date, then the rest by date.
  games     per game, skipped when its parquet is already on Drive (a disconnect costs one game):
            download the .7z (the next one downloads in the background), extract with py7zr, dedupe
            moments with sportvu.fetch.parse, convert to contract frames, write the parquet to VM disk,
            copy it to Drive and verify it reads back, fetch and join basketball-reference play-by-play
            (failures are logged in the manifest, the run continues), rewrite manifest.json, delete the
            raw files from /content.
  finish    README.md, run report. RUN OK when every listed game has frames on Drive or is recorded as a
            broken source; RUN FAILED otherwise (rerun the notebook: it resumes).

Local use (no Drive, no account check), e.g. the two-game check behind tests/test_colab_contract.py:
  python -m sportvu.colab_contract --archives DIR_OF_7Z --out OUT_DIR [--games G ...] [--no-pbp]
"""
from __future__ import annotations
import argparse, glob, io, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from sportvu import splits
from sportvu.fetch import RAW_BASE, GAP_S, HZ, parse, coverage_report
from sportvu.manifest import LIST_URL, parse_name

CONTRACT_VERSION = 1
SOURCE_REPO = "github.com/linouk23/NBA-Player-Movements"
SOURCE_DIR = "data/2016.NBA.Raw.SportVU.Game.Logs"
COVERAGE = ("2015-10-27", "2016-01-23")
ACCOUNT = "lucienmmcnulty@gmail.com"
PERSIST = "/content/drive/MyDrive/nba_harvest"
OUT_DIR = PERSIST + "/sportvu_contract"
RAW_DIR = "/content/sportvu_raw"
STATUS_FILE = "/content/run_status.txt"
HELDOUT_SUBDIR = "heldout"
BR_ABBR = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}     # SportVU id -> basketball-reference code
PBP_TOL_S = 2.0        # a play-by-play event joins the nearest frame of its period within this clock distance
PBP_DELAY_S = 3.5      # basketball-reference rate limit: never two requests closer than this
PBP_RETRY_WAIT_S = 65  # one wait after an HTTP 429, then the game is logged as not joined
DOWNLOAD_TRIES = 3
PREFETCH = 2           # archives downloaded ahead of the game being converted

T0 = time.time()
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
_last_pbp_request = [0.0]


def banner(name: str) -> None:
    print("\n" + "=" * 70 + "\n" + name + "  (t+%.0f s)\n" % (time.time() - T0) + "=" * 70, flush=True)


def record(step: str, state: str, **kw) -> None:
    REPORT["steps"][step] = {"state": state, "t_s": round(time.time() - T0), **kw}
    print("[%s] %s %s" % (step, state, kw if kw else ""), flush=True)


# ── splits: which games are held out, which come first ─────────────────────────────────────────────
def split_of(game: dict, sp: dict) -> str | None:
    """'train', 'heldout_game', 'heldout_arena', 'heldout_era' or None for a SportVU game id. A game whose
    home arena is in heldout_arena.arenas is held out even when splits.json does not list it by name
    (sportvu/splits.py: every game at CLE is held out, synced or not)."""
    gid = game["game"]
    for s in ("train",) + splits.HELDOUT_SPLITS:
        if any(g.get("sportvu") == gid for g in sp[s]["games"].values()):
            return s
    if game["home"] in sp["heldout_arena"].get("arenas", []):
        return "heldout_arena"
    return None


def is_heldout(split: str | None) -> bool:
    return split in splits.HELDOUT_SPLITS


def order_games(games: list[dict], sp: dict) -> list[dict]:
    """Training games of splits.json first (their order there), then the other GSW games by date, then the
    rest by date."""
    train_ids = [g["sportvu"] for g in sp["train"]["games"].values() if g.get("sportvu")]
    rank = {gid: i for i, gid in enumerate(train_ids)}
    def key(g):
        if g["game"] in rank:
            return (0, rank[g["game"]], g["date"], g["game"])
        if "GSW" in (g["home"], g["away"]):
            return (1, 0, g["date"], g["game"])
        return (2, 0, g["date"], g["game"])
    return sorted(games, key=key)


# ── listing and download ───────────────────────────────────────────────────────────────────────────
def list_github() -> list[dict]:
    """Every game archive in the source listing (one GitHub API call, same parser as sportvu.manifest)."""
    req = urllib.request.Request(LIST_URL, headers={"Accept": "application/vnd.github+json", "User-Agent": "nba-comp-viz"})
    entries = json.loads(urllib.request.urlopen(req, timeout=60).read())
    out = []
    for e in entries:
        g = parse_name(e["name"])
        if g:
            g["archive_bytes"] = int(e["size"])
            out.append(g)
    return out


def list_local(archives: str | Path) -> list[dict]:
    """The game archives in a local directory (the local test path)."""
    out = []
    for p in sorted(Path(archives).glob("*.7z")):
        g = parse_name(p.name)
        if g:
            g["archive_bytes"] = p.stat().st_size
            g["local_path"] = str(p)
            out.append(g)
    return out


def download(game: dict, raw_dir: Path) -> Path:
    """<game>.7z into raw_dir: a local copy when the listing carries local_path, else the raw GitHub URL.
    Written to a temp name and renamed, so a half download never looks complete."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / (game["game"] + ".7z")
    if dst.exists() and dst.stat().st_size == game["archive_bytes"]:
        return dst
    if game.get("local_path"):
        shutil.copyfile(game["local_path"], dst)
        return dst
    url = RAW_BASE + game["game"] + ".7z"
    err = None
    for attempt in range(DOWNLOAD_TRIES):
        try:
            blob = urllib.request.urlopen(url, timeout=300).read()
            tmp = dst.with_suffix(".7z.part")
            tmp.write_bytes(blob)
            os.replace(tmp, dst)
            return dst
        except Exception as e:                                  # noqa: BLE001 (retry, then report)
            err = e
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("download failed after %d tries: %s" % (DOWNLOAD_TRIES, err))


def extract(archive: Path, raw_dir: Path) -> Path:
    """The one JSON inside <game>.7z -> raw_dir/<game>.json (py7zr)."""
    import py7zr
    tmp = raw_dir / ("_extract_" + archive.stem)
    shutil.rmtree(tmp, ignore_errors=True)
    with py7zr.SevenZipFile(archive, "r") as z:
        names = [n for n in z.getnames() if n.lower().endswith(".json")]
        if not names:
            raise ValueError("no JSON inside %s" % archive.name)
        z.extract(path=tmp, targets=names)
    src = next(tmp.rglob("*.json"))
    out = raw_dir / (archive.stem + ".json")
    os.replace(src, out)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def cleanup_raw(game: str, raw_dir: Path) -> None:
    """Delete the archive, the JSON and any extraction leftovers of one game from VM disk."""
    for p in list(raw_dir.glob(game + ".*")) + list(raw_dir.glob("_extract_" + game + "*")):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ── the converter: deduped moments -> contract frames ───────────────────────────────────────────────
def frames_schema():
    import pyarrow as pa
    return pa.schema([
        ("game", pa.string()), ("gameid", pa.string()), ("date", pa.string()),
        ("period", pa.int8()), ("frame", pa.int32()), ("ts_ms", pa.int64()),
        ("game_clock_s", pa.float32()), ("shot_clock_s", pa.float32()), ("n_players", pa.int8()),
        ("ball_x", pa.float32()), ("ball_y", pa.float32()), ("ball_z", pa.float32()),
        ("team_id", pa.int32()), ("team", pa.string()), ("home", pa.bool_()),
        ("player_id", pa.int32()), ("player", pa.string()), ("jersey", pa.string()),
        ("x", pa.float32()), ("y", pa.float32()),
    ])


def frames_table(parsed: dict):
    """docs/DATA_CONTRACT.md frames from sportvu.fetch.parse output: one row per tracked player per
    frame, the frame-level columns (clock, ball) repeated on each of its rows. Frames are numbered 0..N-1
    in the parse order (period, game clock descending, timestamp)."""
    import pyarrow as pa
    teams, players = parsed["teams"], parsed["players"]
    cols = {k: [] for k in ("period", "frame", "ts_ms", "game_clock_s", "shot_clock_s", "n_players",
                            "ball_x", "ball_y", "ball_z", "team_id", "player_id", "x", "y")}
    for i, m in enumerate(parsed["moments"]):
        ball = m["ball"] or (None, None, None)
        n = len(m["players"])
        for team_id, player_id, x, y in m["players"]:
            cols["period"].append(m["q"]); cols["frame"].append(i); cols["ts_ms"].append(m["ts"])
            cols["game_clock_s"].append(m["clock"]); cols["shot_clock_s"].append(m["shot"]); cols["n_players"].append(n)
            cols["ball_x"].append(ball[0]); cols["ball_y"].append(ball[1]); cols["ball_z"].append(ball[2])
            cols["team_id"].append(team_id); cols["player_id"].append(player_id); cols["x"].append(x); cols["y"].append(y)
    n_rows = len(cols["frame"])
    team_ids = np.array(cols["team_id"], dtype=np.int64) if n_rows else np.zeros(0, np.int64)
    player_ids = np.array(cols["player_id"], dtype=np.int64) if n_rows else np.zeros(0, np.int64)
    abbr = {t: v["abbr"] for t, v in teams.items()}
    home = {t: v["side"] == "home" for t, v in teams.items()}
    name = {p: v["name"] for p, v in players.items()}
    jersey = {p: v.get("jersey") for p, v in players.items()}
    sch = frames_schema()
    arrays = {
        "game": pa.array([parsed["game"]] * n_rows, pa.string()),
        "gameid": pa.array([str(parsed.get("gameid"))] * n_rows, pa.string()),
        "date": pa.array([parsed.get("date")] * n_rows, pa.string()),
        "team": pa.array([abbr.get(int(t)) for t in team_ids], pa.string()),
        "home": pa.array([home.get(int(t), False) for t in team_ids], pa.bool_()),
        "player": pa.array([name.get(int(p)) for p in player_ids], pa.string()),
        "jersey": pa.array([jersey.get(int(p)) for p in player_ids], pa.string()),
    }
    for k, v in cols.items():
        arrays[k] = pa.array(v, sch.field(k).type)
    return pa.table({f.name: arrays[f.name] for f in sch}, schema=sch)


def gap_summary(parsed: dict) -> dict:
    """Clock gaps over GAP_S per quarter, as sportvu.fetch.coverage_report counts them."""
    rep = coverage_report(parsed) if parsed["moments"] else {"quarters": {}}
    by_q = {q: v["gaps_over_%gs" % GAP_S] for q, v in rep["quarters"].items()}
    gaps = [g for v in by_q.values() for g in v]
    return {"count": len(gaps), "seconds": round(sum(a - b for a, b in gaps), 1), "by_quarter": by_q}


# ── play-by-play ───────────────────────────────────────────────────────────────────────────────────
def br_code(game: dict) -> str:
    """basketball-reference box score code: YYYYMMDD0HOME with its own abbreviations."""
    return game["date"].replace("-", "") + "0" + BR_ABBR.get(game["home"], game["home"])


def fetch_pbp_events(code: str) -> list[dict]:
    """fetch_pbp.fetch_game under the rate limit, one retry after an HTTP 429."""
    import fetch_pbp
    for attempt in (0, 1):
        wait = PBP_DELAY_S - (time.time() - _last_pbp_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_pbp_request[0] = time.time()
        try:
            return fetch_pbp.fetch_game(code)
        except Exception as e:                                  # noqa: BLE001 (logged by the caller)
            if attempt == 0 and "429" in str(e):
                time.sleep(PBP_RETRY_WAIT_S)
                continue
            raise


def pbp_schema():
    import pyarrow as pa
    return pa.schema([
        ("game", pa.string()), ("event", pa.int32()), ("period", pa.int8()), ("clock_s", pa.float32()),
        ("team", pa.string()), ("desc", pa.string()), ("points", pa.int8()), ("score", pa.string()),
        ("kind", pa.string()), ("frame", pa.int32()), ("frame_clock_diff_s", pa.float32()),
    ])


def join_pbp(game: str, events: list[dict], frame_period, frame_clock):
    """Each play-by-play event -> the frame of its period whose game clock is nearest (within PBP_TOL_S),
    else frame -1. frame_period and frame_clock are per-frame arrays indexed by frame number."""
    import pyarrow as pa
    frame_period = np.asarray(frame_period); frame_clock = np.asarray(frame_clock, float)
    by_period = {}
    for p in np.unique(frame_period):
        idx = np.where(frame_period == p)[0]
        order = np.argsort(frame_clock[idx], kind="stable")
        by_period[int(p)] = (frame_clock[idx][order], idx[order])
    rows = {k: [] for k in ("event", "period", "clock_s", "team", "desc", "points", "score", "kind", "frame", "frame_clock_diff_s")}
    for i, e in enumerate(events):
        frame, diff = -1, None
        if e["period"] in by_period:
            clocks, idx = by_period[e["period"]]
            j = int(np.searchsorted(clocks, e["clock_s"]))
            cands = [k for k in (j - 1, j) if 0 <= k < len(clocks)]
            k = min(cands, key=lambda k: abs(clocks[k] - e["clock_s"]))
            d = float(abs(clocks[k] - e["clock_s"]))
            if d <= PBP_TOL_S:
                frame, diff = int(idx[k]), round(d, 3)
        rows["event"].append(i); rows["period"].append(e["period"]); rows["clock_s"].append(e["clock_s"])
        rows["team"].append(e.get("team")); rows["desc"].append(e.get("desc")); rows["points"].append(e.get("points", 0))
        rows["score"].append(e.get("score")); rows["kind"].append(e.get("kind")); rows["frame"].append(frame)
        rows["frame_clock_diff_s"].append(diff)
    sch = pbp_schema()
    arrays = {"game": pa.array([game] * len(events), pa.string())}
    for k, v in rows.items():
        arrays[k] = pa.array(v, sch.field(k).type)
    return pa.table({f.name: arrays[f.name] for f in sch}, schema=sch)


def frame_clocks(table):
    """(frame_period, frame_clock) per frame from a frames table (one entry per frame, not per row)."""
    fr = table.column("frame").to_numpy()
    first = np.unique(fr, return_index=True)[1]
    return table.column("period").to_numpy()[first], table.column("game_clock_s").to_numpy()[first]


# ── parquet on Drive ───────────────────────────────────────────────────────────────────────────────
def parquet_ok(path: Path) -> int:
    """Row count when the file exists and its footer reads, else 0 (a truncated Drive copy reads as absent)."""
    import pyarrow.parquet as pq
    try:
        return pq.read_metadata(path).num_rows if path.exists() else 0
    except Exception:                                           # noqa: BLE001 (unreadable means rewrite)
        return 0


def write_parquet(table, dest: Path, scratch: Path) -> int:
    """Write on VM disk, copy to dest.part on Drive, rename, then read the footer back. Returns bytes."""
    import pyarrow.parquet as pq
    scratch.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = scratch / dest.name
    pq.write_table(table, local, compression="zstd")
    part = dest.with_name(dest.name + ".part")
    shutil.copyfile(local, part)
    os.replace(part, dest)
    if parquet_ok(dest) != table.num_rows:
        raise IOError("%s did not read back with %d rows" % (dest, table.num_rows))
    local.unlink(missing_ok=True)
    return dest.stat().st_size


def game_paths(game: dict, out_dir: Path, sp: dict) -> tuple[Path, Path, str | None]:
    """(frames parquet, pbp parquet, split): held-out games live under heldout/."""
    split = split_of(game, sp)
    d = out_dir / HELDOUT_SUBDIR if is_heldout(split) else out_dir
    return d / (game["game"] + ".parquet"), d / (game["game"] + "_pbp.parquet"), split


# ── manifest and README ────────────────────────────────────────────────────────────────────────────
def load_manifest(out_dir: Path) -> dict:
    p = out_dir / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                       # noqa: BLE001 (a torn write; rebuild)
            pass
    return {"contract_version": CONTRACT_VERSION, "contract": "docs/DATA_CONTRACT.md",
            "source": "%s %s" % (SOURCE_REPO, SOURCE_DIR), "coverage": list(COVERAGE),
            "heldout_rule": "games listed in heldout (and every file under heldout/) are the held-out games of "
                            "sportvu/splits.json: no training job may read them (sportvu.splits.refuse_heldout)",
            "heldout": [], "games": {}}


def write_manifest(out_dir: Path, man: dict) -> None:
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man["heldout"] = sorted(g for g, e in man["games"].items() if e.get("heldout"))
    man["counts"] = {"games": len(man["games"]),
                     "ok": sum(1 for e in man["games"].values() if e["status"] == "ok"),
                     "failed": sum(1 for e in man["games"].values() if e["status"] == "failed"),
                     "source_broken": sum(1 for e in man["games"].values() if e["status"] == "source_broken"),
                     "pbp_joined": sum(1 for e in man["games"].values() if e.get("pbp_joined")),
                     "heldout": len(man["heldout"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "manifest.json.part"
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    os.replace(tmp, out_dir / "manifest.json")


README = """# SportVU 2015-16 analytics dataset (contract v{version})

Built by `sportvu/colab_contract.py` (notebook `colab_contract.ipynb`) of lmcnulty7/NBA-Comp-Viz-V3.
Schema: `docs/DATA_CONTRACT.md` in that repo.

## Source
SportVU player tracking (STATS LLC for the NBA), the raw game logs republished at
{repo}, `{dir}`, one .7z per game. Every archive in that listing is converted.

## Coverage
Regular season games from {c0} to {c1} (the public log window); 25 Hz; court coordinates in feet,
x 0..94, y 0..50, corner origin; ball x, y, z on every row. Moments that the source repeats across
consecutive events are deduped as `sportvu/fetch.py` does. Some archives are truncated at the source
(status `source_broken` in `manifest.json`); some games have clock gaps (`gaps_over_2s`).

## Layout
- `<game>.parquet`: frames, one row per tracked player per frame (`MM.DD.YYYY.AWAY.at.HOME`).
- `<game>_pbp.parquet`: basketball-reference play-by-play joined to frames, only where the fetch worked
  (`pbp_joined` in the manifest).
- `heldout/`: the same files for the held-out games of `sportvu/splits.json` (the scorecard game and
  every game at the held-out arena). No training job may read anything under `heldout/`; the manifest
  lists them under `heldout` and `sportvu.splits.refuse_heldout` rejects their ids.
- `manifest.json`: per game: date, teams, split, moments, frames rows, gaps over 2 s, pbp joined, status.
  Rewritten after every game, so a partial run is readable.

## Not for redistribution
The tracking data belongs to the NBA and STATS LLC and was published for research. This folder is a
private working copy for this project: do not share, re-upload or redistribute it.
"""


def write_readme(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(README.format(version=CONTRACT_VERSION, repo=SOURCE_REPO, dir=SOURCE_DIR,
                                                     c0=COVERAGE[0], c1=COVERAGE[1]))


# ── one game ───────────────────────────────────────────────────────────────────────────────────────
def convert_game(game: dict, archive: Path, raw_dir: Path, out_dir: Path, scratch: Path, sp: dict,
                 pbp_fetch=fetch_pbp_events, do_pbp: bool = True) -> dict:
    """Archive -> frames parquet (+ pbp parquet) on out_dir. Returns the manifest entry; never raises for
    a play-by-play failure. Raw files are deleted by the caller."""
    t = time.time()
    frames_path, pbp_path, split = game_paths(game, out_dir, sp)
    entry = {"game": game["game"], "date": game["date"], "away": game["away"], "home": game["home"],
             "split": split, "heldout": is_heldout(split), "source_bytes": game.get("archive_bytes"),
             "status": "failed", "error": None, "moments": 0, "frames_rows": 0,
             "gaps_over_2s": {"count": 0, "seconds": 0.0, "by_quarter": {}},
             "pbp_joined": False, "pbp": None, "parquet": None, "parquet_bytes": 0, "pbp_parquet": None}
    try:
        raw_json = extract(archive, raw_dir)
        parsed = parse(raw_json)
        parsed["game"] = game["game"]
    except Exception as e:                                      # noqa: BLE001 (recorded per game)
        entry.update(status="source_broken", error="extract/parse: %r" % e)
        return entry
    if not parsed["moments"]:
        entry.update(status="source_broken", error="no moments after dedupe")
        return entry
    table = frames_table(parsed)
    entry.update(moments=len(parsed["moments"]), frames_rows=table.num_rows, gaps_over_2s=gap_summary(parsed))
    entry["parquet_bytes"] = write_parquet(table, frames_path, scratch)
    entry["parquet"] = str(frames_path.relative_to(out_dir))
    entry["status"] = "ok"
    if do_pbp:
        entry.update(join_game_pbp(game, table, pbp_path, out_dir, scratch, pbp_fetch))
    entry["seconds"] = round(time.time() - t, 1)
    entry["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return entry


def join_game_pbp(game: dict, table, pbp_path: Path, out_dir: Path, scratch: Path, pbp_fetch=fetch_pbp_events) -> dict:
    """Fetch, join and write the play-by-play of one game. Returns the manifest fields; a failure is a
    logged field, not an exception."""
    code = br_code(game)
    try:
        events = pbp_fetch(code)
        if not events:
            raise ValueError("no events parsed")
        fp, fc = frame_clocks(table)
        joined = join_pbp(game["game"], events, fp, fc)
        matched = int(np.sum(joined.column("frame").to_numpy() >= 0))
        write_parquet(joined, pbp_path, scratch)
        return {"pbp_joined": True, "pbp_parquet": str(pbp_path.relative_to(out_dir)),
                "pbp": {"code": code, "events": len(events), "matched": matched, "error": None}}
    except Exception as e:                                      # noqa: BLE001 (log, do not stop)
        print("  pbp FAILED %s (%s): %r" % (game["game"], code, e), flush=True)
        return {"pbp_joined": False, "pbp_parquet": None,
                "pbp": {"code": code, "events": 0, "matched": 0, "error": repr(e)[:300]}}


# ── the run ────────────────────────────────────────────────────────────────────────────────────────
def run(games: list[dict], out_dir: Path, raw_dir: Path, sp: dict | None = None, pbp_fetch=fetch_pbp_events,
        do_pbp: bool = True, retry_pbp: bool = False, downloader=download) -> dict:
    """Convert every listed game that is not already on out_dir. Returns the manifest."""
    sp = sp or splits.load()
    out_dir, raw_dir = Path(out_dir), Path(raw_dir)
    scratch = raw_dir / "_parquet"
    man = load_manifest(out_dir)
    write_readme(out_dir)
    games = order_games(games, sp)
    todo = []
    for g in games:
        frames_path, pbp_path, split = game_paths(g, out_dir, sp)
        old = man["games"].get(g["game"])
        if parquet_ok(frames_path):
            if old is None or old.get("status") != "ok":
                old = {"game": g["game"], "date": g["date"], "away": g["away"], "home": g["home"], "split": split,
                       "heldout": is_heldout(split), "status": "ok", "error": None, "moments": None,
                       "frames_rows": parquet_ok(frames_path), "gaps_over_2s": None, "pbp_joined": parquet_ok(pbp_path) > 0,
                       "pbp": None, "parquet": str(frames_path.relative_to(out_dir)), "parquet_bytes": frames_path.stat().st_size,
                       "pbp_parquet": str(pbp_path.relative_to(out_dir)) if parquet_ok(pbp_path) else None,
                       "note": "frames found on Drive without a manifest entry"}
                man["games"][g["game"]] = old
            if retry_pbp and do_pbp and not old.get("pbp_joined"):
                import pyarrow.parquet as pq
                table = pq.read_table(frames_path, columns=["frame", "period", "game_clock_s"])
                old.update(join_game_pbp(g, table, pbp_path, out_dir, scratch, pbp_fetch))
                write_manifest(out_dir, man)
            continue
        if old is not None and old.get("status") == "source_broken" and old.get("source_bytes") == g.get("archive_bytes"):
            continue                                            # same broken archive as last time
        todo.append(g)
    print("%d games listed, %d already on %s, %d to convert" % (len(games), len(games) - len(todo), out_dir, len(todo)), flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        futures = {}
        for i, g in enumerate(todo):
            for nxt in todo[i:i + 1 + PREFETCH]:
                if nxt["game"] not in futures:
                    futures[nxt["game"]] = ex.submit(downloader, nxt, raw_dir)
            label = "%s [%s]" % (g["game"], split_of(g, sp) or "rest")
            try:
                archive = futures.pop(g["game"]).result()
                entry = convert_game(g, archive, raw_dir, out_dir, scratch, sp, pbp_fetch, do_pbp)
            except Exception as e:                              # noqa: BLE001 (one game must not end the run)
                entry = {"game": g["game"], "date": g["date"], "away": g["away"], "home": g["home"],
                         "split": split_of(g, sp), "heldout": is_heldout(split_of(g, sp)), "status": "failed",
                         "error": repr(e)[:300], "moments": 0, "frames_rows": 0, "pbp_joined": False}
            finally:
                cleanup_raw(g["game"], raw_dir)
            man["games"][g["game"]] = entry
            write_manifest(out_dir, man)
            print("%4d/%d %-40s %-13s moments %6s gaps>2s %3s pbp %-5s (t+%.0f s)" % (
                i + 1, len(todo), label, entry["status"], entry.get("moments"),
                (entry.get("gaps_over_2s") or {}).get("count", "?"), entry.get("pbp_joined"), time.time() - T0), flush=True)
    write_manifest(out_dir, man)
    return man


def verdict(man: dict, listed: list[dict]) -> str:
    """RUN OK when every listed game is on Drive or recorded as a broken source; RUN FAILED otherwise."""
    missing = [g["game"] for g in listed if man["games"].get(g["game"], {}).get("status") not in ("ok", "source_broken")]
    c = man.get("counts", {})
    line = "%d of %d games on Drive, %d broken at the source, %d with play-by-play, %d held out" % (
        c.get("ok", 0), len(listed), c.get("source_broken", 0), c.get("pbp_joined", 0), c.get("heldout", 0))
    if missing:
        return "RUN FAILED: %d games not converted (%s ...); %s. Rerun: it resumes." % (len(missing), ", ".join(missing[:3]), line)
    return "RUN OK: " + line


# ── Colab steps ────────────────────────────────────────────────────────────────────────────────────
def step_env() -> None:
    banner("env: Drive account")
    if not os.path.isdir("/content"):
        sys.exit("colab_contract.py without --out is Colab-only; locally pass --archives DIR --out DIR.")
    acct = os.path.join(PERSIST, "sportvu", "ACCOUNT.txt")
    if not os.path.exists(acct) or open(acct).read().strip() != ACCOUNT:
        sys.exit("WRONG OR UNMOUNTED DRIVE: %s must read %s. Remount Drive with the %s account "
                 "(Runtime > Disconnect, then run again and pick that account)." % (acct, ACCOUNT, ACCOUNT))
    REPORT["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    record("env", "ok", account=ACCOUNT, commit=REPORT["commit"], out=OUT_DIR)


def step_deps() -> None:
    banner("deps")
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "py7zr", "pyarrow"], capture_output=True, text=True)
    record("deps", "ok" if r.returncode == 0 else "FAILED", tail=(r.stderr or "")[-300:])
    if r.returncode != 0:
        sys.exit("pip install failed: %s" % (r.stderr or "")[-300:])


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archives", help="local directory of <game>.7z files instead of the GitHub listing")
    ap.add_argument("--out", help="output directory (default: Drive, with the account check)")
    ap.add_argument("--raw", help="scratch directory for raw files (default %s)" % RAW_DIR)
    ap.add_argument("--games", nargs="*", help="only these game ids")
    ap.add_argument("--limit", type=int, help="stop after this many listed games (after ordering)")
    ap.add_argument("--no-pbp", action="store_true", help="skip basketball-reference")
    ap.add_argument("--retry-pbp", action="store_true", help="re-fetch play-by-play for games on Drive without it")
    a = ap.parse_args(argv)
    colab = a.out is None
    out_dir = Path(a.out or OUT_DIR)
    raw_dir = Path(a.raw or (RAW_DIR if colab else tempfile.mkdtemp(prefix="sportvu_raw_")))
    if colab:
        step_env()
        step_deps()
    banner("list: %s" % (a.archives or SOURCE_REPO))
    games = list_local(a.archives) if a.archives else list_github()
    if a.games:
        games = [g for g in games if g["game"] in set(a.games)]
    sp = splits.load()
    games = order_games(games, sp)
    if a.limit:
        games = games[:a.limit]
    record("list", "ok", games=len(games), train=sum(split_of(g, sp) == "train" for g in games),
           heldout=sum(is_heldout(split_of(g, sp)) for g in games))
    banner("games -> %s" % out_dir)
    man = run(games, out_dir, raw_dir, sp, do_pbp=not a.no_pbp, retry_pbp=a.retry_pbp)
    banner("finish")
    final = verdict(man, games)
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT["final"] = final
    REPORT["counts"] = man.get("counts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_report.json").write_text(json.dumps(REPORT, indent=1))
    print("\n" + final, flush=True)
    return final


if __name__ == "__main__":
    # The notebook prints this file after flushing Drive, so its last line says whether the run worked.
    status_file = STATUS_FILE if os.path.isdir("/content") else None
    try:
        final = main()
        if status_file:
            open(status_file, "w").write(final)
    except SystemExit as e:
        if status_file:
            open(status_file, "w").write("RUN FAILED: %s" % e)
        raise
    except Exception as e:
        if status_file:
            open(status_file, "w").write("RUN FAILED: %r" % e)
        raise
