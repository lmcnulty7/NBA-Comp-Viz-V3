"""tools/research_ingest.py: the misnamed SportVU archives of the public release, into the contract layout (D71, D77).

    python3 -m export.heavy --name "research ingest" -- \\
        python3 -m tools.research_ingest [--source KEY] [--root DIR] [--no-listing]

The public release (linouk23/NBA-Player-Movements) stores some archives under names that start with the folder name
(docs/research/DATA.md, question 1). The frozen name regex is anchored at the start, so the frozen pipeline never read
them. This tool:

1. Downloads each archive pinned by tools.research_testset.ingest_pins(source) (game id -> git blob sha1) with curl from
   raw.githubusercontent.com at the pinned commit, under the prefixed name, and checks its git blob sha1 against the
   pin (computed here and by git hash-object). With the listing check on, the GitHub contents API listing at the same
   commit must hold exactly these prefixed names with the same blob shas. Any mismatch stops the run.
2. Stores it as raw/<MM.DD.YYYY.AWAY.at.HOME>.7z under --root, the normal name the frozen regex accepts.
3. Converts it with the frozen code unchanged (sportvu.colab_contract.convert_game, with do_pbp=False) into
   sportvu_contract/<game>.parquet under --root, with manifest.json in the frozen form (colab_contract.write_manifest).
   One game at a time; each game's extracted JSON is deleted before the next.
4. Writes ingest.json under --root: per game the archive sha256 and blob sha1, the status, moments, frames rows and
   clock gaps, and the later steps that need basketball-reference pages (see BLOCKED).

    python3 -m export.heavy --name "research ingest bref" -- python3 -m tools.research_ingest bref [--root DIR]

Stage 2 (PLAN.md D78, bref_stage): the shot chart and play-by-play pages of the converted games through
collect.bref_shots unchanged (15 requests a minute, pages cached under collect/raw/ so none is fetched twice), the
play-by-play joined to the frames by sportvu.colab_contract.join_game_pbp, then analytics.run_phase0 (rosters,
shot_frames, possessions, gaps; rebound_frames is left out) with every output under --root, and a check that
analytics.xfg.validate.load_data_root reads the folder. Per-game shot counts go into ingest.json.

SEALED: the frozen tools this stage runs also write outcome aggregates of the test games under --root:
collect/bref_shots/manifest.json (a made count per game), analytics/phase0_manifest.json (possession end kinds,
points, final score) and the reports/ folder under --root. They are never opened, printed or read by any research
tool; only score_once's scorer reads a test outcome (PLAN.md D80, D81). SEALED_FILES lists them.

Nothing is fitted or scored here; no metric is computed. The output root may not be under DATA_ROOT or reports/.
No held-out game of sportvu/splits.json is accepted.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "linouk23/NBA-Player-Movements"
COMMIT = "f7d4cf2c7e607029abfd5715a0d4073bedae328e"
SOURCE_DIR = "data/2016.NBA.Raw.SportVU.Game.Logs"
PREFIX = "2016.NBA.Raw.SportVU.Game.Logs"     # the stray folder name the upload put in front of the file name
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{SOURCE_DIR}/"
LIST_URL = f"https://api.github.com/repos/{REPO}/contents/{SOURCE_DIR}?ref={COMMIT}"
SEALED_FILES = ("collect/bref_shots/manifest.json", "analytics/phase0_manifest.json", "reports/")
DEFAULT_ROOT = Path.home() / "Developer" / "nba-v3-worktrees" / "_research-data" / "sportvu_untouched"
WHO = "tools.research_ingest"

# The later steps that need basketball-reference pages, which this tool does not fetch (Sports Reference terms are
# Lucien's decision). Two pages a game: the play-by-play page and the shot chart page.
BLOCKED = {
    "pages_per_game": {"play_by_play": "https://www.basketball-reference.com/boxscores/pbp/<code>.html",
                       "shot_chart": "https://www.basketball-reference.com/boxscores/shot-chart/<code>.html"},
    "steps": {
        "play_by_play": ("sportvu.colab_contract.join_game_pbp (fetch_pbp.fetch_game) -> <game>_pbp.parquet and "
                         "pbp_joined in the manifest; needs the play-by-play page"),
        "possessions": "analytics.possessions -> analytics/possessions/<season>/<game>; reads <game>_pbp.parquet",
        "shot_frames": "analytics.shot_frames (xfg, decisions, matchups, gravity, epv, rebounds); reads the play-by-play",
        "rebound_frames": "analytics.rebound_frames (rebounds, decisions, epv); reads the play-by-play",
        "shots_table": ("collect.bref_shots -> collect/bref_shots/shotchart rows (xfg location part, epv, gravity, "
                        "embeddings); needs the shot chart page"),
        "rosters": "analytics.rosters (bref slugs for the xfg shooter join); reads the shot chart page and the play-by-play",
    },
    "lanes_needing_them": {
        "xfg": ["shots_table", "shot_frames", "rosters", "play_by_play"],
        "possessions based (ghosting, gravity, epv, actions, playcalling, coverage, matchups, embeddings)": [
            "play_by_play", "possessions"],
        "rebounds": ["play_by_play", "rebound_frames", "shot_frames"],
        "decisions": ["play_by_play", "shot_frames", "rebound_frames"],
        "load": [],
    },
}


# ── names and hashes (pure, unit tested) ────────────────────────────────────────────────────────────────────────────
def prefixed_name(game: str) -> str:
    """The file name the upload used: the folder name glued in front of <game>.7z."""
    return f"{PREFIX}{game}.7z"


def normal_name(name: str) -> str | None:
    """<game> for a listing file name with or without the stray folder prefix, when the frozen name regex accepts the
    result; None otherwise."""
    from sportvu.manifest import parse_name
    base = name[len(PREFIX):] if name.startswith(PREFIX) else name
    g = parse_name(base)
    return g["game"] if g else None


def git_blob_sha1(data: bytes) -> str:
    """The git blob id of a file's bytes (what git hash-object prints)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_blob(game: str, data: bytes, expected: str) -> str:
    """The blob sha1 of the bytes; SystemExit when it differs from the pin."""
    got = git_blob_sha1(data)
    if got != expected:
        raise SystemExit(f"{WHO}: {game}: archive blob sha1 {got} does not match the pinned {expected}")
    return got


def checked_pins(pins: dict) -> dict:
    """The {game: blob sha1} map of the door (tools.research_testset.ingest_pins), checked: at least one game, every id
    a normal game name, every pin a 40-digit hex blob id (a source without pins comes back with None and is refused)."""
    pins = dict(pins)
    if not pins:
        raise SystemExit(f"{WHO}: no pinned archive")
    bad = [g for g in pins if normal_name(str(g) + ".7z") != g]
    if bad:
        raise SystemExit(f"{WHO}: not a game name: {bad[0]}")
    hexdigits = set("0123456789abcdef")
    unpinned = [g for g, v in pins.items() if not (isinstance(v, str) and len(v) == 40 and set(v) <= hexdigits)]
    if unpinned:
        raise SystemExit(f"{WHO}: {len(unpinned)} game(s) without an archive blob pin, for example {unpinned[0]}")
    return pins


def check_listing(entries: list, pins: dict) -> int:
    """The prefixed entries of the GitHub listing must be exactly the pinned games, each with the pinned blob sha."""
    listed = {normal_name(e["name"]): e["sha"] for e in entries if e["name"].startswith(PREFIX)}
    if set(listed) != set(pins):
        raise SystemExit(f"{WHO}: the prefixed names in the listing ({len(listed)}) are not the pinned games ({len(pins)})")
    for g, sha in pins.items():
        if listed[g] != sha:
            raise SystemExit(f"{WHO}: {g}: listing blob {listed[g]} differs from the pin {sha}")
    return len(listed)


# ── network (curl: python urllib has SSL problems on this machine) ──────────────────────────────────────────────────
def curl(url: str, dest: Path) -> None:
    subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", "600", "-o", str(dest), url], check=True)


def fetch_listing(work: Path) -> list:
    """The listing at the pinned commit: curl, unauthenticated; when that is refused (the anonymous API limit is 60
    requests an hour and other jobs share it), the same read-only request through the gh CLI."""
    out = work / "listing.json"
    try:
        curl(LIST_URL, out)
        text = out.read_text()
        out.unlink()
    except subprocess.CalledProcessError:
        if not shutil.which("gh"):
            raise
        text = subprocess.run(["gh", "api", LIST_URL.split("api.github.com/", 1)[1]], check=True,
                              capture_output=True, text=True).stdout
    return json.loads(text)


def digests(data: bytes) -> dict:
    """The record of one archive's bytes: size, git blob sha1 and sha256, from a single pass over the bytes."""
    return {"archive_bytes": len(data), "archive_git_blob_sha1": git_blob_sha1(data),
            "archive_sha256": hashlib.sha256(data).hexdigest()}


def download(game: str, expected: str, raw_dir: Path) -> tuple[Path, dict]:
    """(raw/<game>.7z, digests), verified against the pin. Each archive is read and hashed once: an existing file is
    kept only when its blob sha1 matches the pin, a new one is checked before it takes the final name."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"{game}.7z"
    if dst.exists():
        d = digests(dst.read_bytes())
        if d["archive_git_blob_sha1"] == expected:
            return dst, d
    part = raw_dir / f"{game}.7z.part"
    curl(RAW_URL + prefixed_name(game), part)
    d = digests(part.read_bytes())
    if d["archive_git_blob_sha1"] != expected:
        part.unlink(missing_ok=True)
        raise SystemExit(f"{WHO}: {game}: archive blob sha1 {d['archive_git_blob_sha1']} does not match the pinned {expected}")
    os.replace(part, dst)
    return dst, d


def git_hash_object(path: Path) -> str:
    return subprocess.run(["git", "hash-object", "--no-filters", str(path)], check=True, capture_output=True,
                          text=True).stdout.strip()


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def guard_root(root: Path) -> Path:
    from tools.frozen_common import assert_safe_output
    return assert_safe_output(root)


def ingest(pins: dict, root: Path, listing: bool = True, force: bool = False) -> dict:
    from sportvu import colab_contract as cc
    from sportvu import splits
    from sportvu.manifest import parse_name
    from tools.frozen_common import training_games_only

    root = guard_root(root)
    games = sorted(pins)
    training_games_only(games, WHO)                     # no held-out game of sportvu/splits.json
    raw_dir, out_dir, work = root / "raw", root / "sportvu_contract", root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    record = {"tool": WHO, "decision": "PLAN.md D71, D77", "source": f"github.com/{REPO} {SOURCE_DIR}",
              "commit": COMMIT, "raw_url": RAW_URL + prefixed_name("<game>"), "listing_checked": False,
              "conversion": "sportvu.colab_contract.convert_game, do_pbp=False, frozen code unchanged",
              "games": {}, "blocked_on_basketball_reference": dict(BLOCKED, pages_total=2 * len(games))}
    if listing:
        record["listing_checked"] = check_listing(fetch_listing(work), pins) == len(pins)
        print(f"listing at {COMMIT[:12]}: the {len(pins)} prefixed names match the pins", flush=True)

    sp = splits.load()
    man = cc.load_manifest(out_dir)
    for i, game in enumerate(games, 1):
        t = time.time()
        archive, dig = download(game, pins[game], raw_dir)
        blob, size = dig["archive_git_blob_sha1"], dig["archive_bytes"]
        if git_hash_object(archive) != blob:                # the one independent cross-check
            raise SystemExit(f"{WHO}: {game}: git hash-object disagrees with the computed blob sha1")
        g = parse_name(f"{game}.7z")
        g["archive_bytes"] = size
        frames_path, _, split = cc.game_paths(g, out_dir, sp)
        if cc.is_heldout(split):
            raise SystemExit(f"{WHO}: {game} is a held-out game")
        old = man["games"].get(game)
        if not force and reusable(old, cc.parquet_ok(frames_path), blob):
            entry = old
        else:
            copy = work / archive.name                     # convert from a copy: cleanup_raw deletes what it reads
            shutil.copyfile(archive, copy)
            try:
                entry = cc.convert_game(g, copy, work, out_dir, work / "_parquet", sp, do_pbp=False)
            finally:
                cc.cleanup_raw(game, work)
            entry["archive_git_blob_sha1"] = blob        # the archive these frames came from (reuse check)
            man["games"][game] = entry
            cc.write_manifest(out_dir, man)
        record["games"][game] = {
            "source_name": prefixed_name(game), "archive": f"raw/{game}.7z", **dig, "status": entry["status"],
            "error": entry.get("error"), "moments": entry.get("moments"), "frames_rows": entry.get("frames_rows"),
            "gaps_over_2s": entry.get("gaps_over_2s"), "parquet": entry.get("parquet"),
            "parquet_bytes": entry.get("parquet_bytes"), "split": entry.get("split"), "pbp_joined": entry.get("pbp_joined"),
        }
        print("%2d/%d %-24s %-13s moments %6s rows %8s gaps>2s %s (%.0f s)" % (
            i, len(games), game, entry["status"], entry.get("moments"), entry.get("frames_rows"),
            (entry.get("gaps_over_2s") or {}).get("count"), time.time() - t), flush=True)
    cc.write_manifest(out_dir, man)
    shutil.rmtree(work, ignore_errors=True)
    statuses = [e["status"] for e in record["games"].values()]
    record["counts"] = {"games": len(statuses), "ok": statuses.count("ok"),
                        "source_broken": statuses.count("source_broken"), "failed": statuses.count("failed"),
                        "moments": sum(e["moments"] or 0 for e in record["games"].values()),
                        "frames_rows": sum(e["frames_rows"] or 0 for e in record["games"].values())}
    return update_record(root, record)


def reusable(old: dict | None, rows_on_disk: int, blob: str) -> bool:
    """A converted game is reused only when its manifest entry is ok, the parquet reads back with the recorded row
    count, and the entry names the same archive blob sha1 (an entry without one is converted again)."""
    return bool(old and old.get("status") == "ok" and rows_on_disk and rows_on_disk == old.get("frames_rows")
                and old.get("archive_git_blob_sha1") == blob)


def read_record(root: Path) -> dict:
    p = Path(root) / "ingest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def update_record(root: Path, updates: dict) -> dict:
    """ingest.json with the given top-level keys replaced and every other key (for example bref_stage) kept."""
    record = read_record(root)
    record.update(updates)
    record["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = Path(root) / "ingest.json.part"
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, Path(root) / "ingest.json")
    return record


# ── stage 2 (PLAN.md D78): basketball-reference pages, play-by-play join, the Phase 0 tables ──────────────────────────
BREF_RPM = 15.0                 # requests a minute, under the site's limit of 20 (D78)
PHASE0_STEPS = ("rosters", "shot_frames", "possessions", "gaps")      # rebound_frames left out: xfg does not need it


def cached_pbp_events(collect_dir: Path, code: str) -> list | None:
    """The play-by-play events collect.bref_shots cached for a code (fetch_pbp.fetch_game output), or None."""
    from collect import bref_shots, common
    rec = common.RawCache(Path(collect_dir) / "raw").get(bref_shots.RAW_PBP_ENDPOINT, {"code": code})
    return rec["body"] if rec else None


def pbp_source(entry: dict, pbp_path: Path, collect_dir: Path, project_root: Path) -> tuple[str, list | None]:
    """(source, events) of a game's play-by-play, in the order collect.bref_shots.pbp_shot_count consults them: the
    joined <game>_pbp.parquet (events None: already joined), data/pbp/<code>.json, then the page the collector cached.
    SystemExit when none holds it (the collector answered from somewhere this tool cannot see, or fetched nothing)."""
    from sportvu import colab_contract as cc
    code = cc.br_code(entry)
    if entry.get("pbp_joined") and cc.parquet_ok(pbp_path):
        return "pbp_parquet", None
    local = Path(project_root) / "data" / "pbp" / f"{code}.json"
    if local.exists():
        return "data_pbp_json", json.loads(local.read_text())["events"]
    events = cached_pbp_events(collect_dir, code)
    if events is not None:
        return "bref_pbp_cache", events
    raise SystemExit(f"{WHO}: {entry['game']}: no play-by-play for {code} in the joined parquet, data/pbp or the "
                     "collector's cache")


def pbp_field_goal_attempts(events: list | None, pbp_path: Path) -> int:
    """Shot events of either kind (a row count: no outcome is split out), from the events or the joined parquet."""
    if events is None:
        import pyarrow.parquet as pq
        kinds = pq.read_table(pbp_path, columns=["kind"]).column("kind").to_pylist()
    else:
        kinds = [ev.get("kind") for ev in events]
    return sum(k in ("shot_made", "shot_missed") for k in kinds)


def shot_counts(shots, shot_frames, features, pbp_shots: dict) -> dict:
    """Row counts per game, nothing about outcomes: shot chart rows, play-by-play field-goal attempts (shot events of
    either kind), shot_frames rows and high-confidence rows, assembled xFG rows and those with contest features.
    No made, missed or outcome column is read here (D71: the test set's outcomes are read only by score_once)."""
    def by_game(df):
        return df["sportvu_game"].astype(str)
    out = {}
    for g in sorted(set(by_game(shots))):
        f = shot_frames[by_game(shot_frames) == g]
        a = features[by_game(features) == g]
        out[g] = {"shot_chart_rows": int((by_game(shots) == g).sum()), "pbp_field_goal_attempts": pbp_shots.get(g),
                  "shot_frames_rows": int(len(f)), "shot_frames_high": int((f["confidence_flag"] == "high").sum()),
                  "assembled_rows": int(len(a)), "has_contest": int(a["has_contest"].astype(int).sum())}
    return out


def xfg_check(root: Path, pbp_shots: dict) -> tuple[dict, dict]:
    """The check the xfg lane needs, with the frozen functions: load_data_root, clean, assemble. Nothing fitted, no
    metric; the label column is dropped unread. -> (totals, per-game row counts)."""
    from analytics.xfg.train import assemble
    from analytics.xfg.validate import clean, load_data_root
    data = load_data_root(root, "2015-16")
    loaded = {"shots": int(len(data["shots"])), "shot_frames": int(len(data["shot_frames"])),
              "rosters": int(len(data["rosters"])), "possessions": int(len(data["possessions"])),
              "events": int(len(data["events"])), "heldout_shots": int(data["shots"]["heldout"].sum())}
    cleaned, _summaries = clean(data)
    rows, held_rows = assemble(cleaned)
    if len(held_rows):
        raise SystemExit(f"{WHO}: assemble set aside {len(held_rows)} held-out rows; none was expected")
    feats = rows.drop(columns=[c for c in ("made",) if c in rows.columns])
    counts = shot_counts(cleaned["shots"], cleaned["shot_frames"], feats, pbp_shots)
    loaded.update(assembled_rows=int(len(feats)), has_contest=int(feats["has_contest"].astype(int).sum()))
    return loaded, counts


def bref_stage(root: Path, collect=None, phase0=None, check=xfg_check, project_root=None) -> dict:
    """Fetch the shot chart and play-by-play pages with collect.bref_shots (unchanged; its throttle at BREF_RPM, its
    raw cache under collect/raw/), join the play-by-play to the frames with sportvu.colab_contract.join_game_pbp, run
    analytics.run_phase0 with explicit outputs under root, and check that the xfg lane loads and assembles the folder.
    The outcome goes into ingest.json (merged); a failed collector or Phase 0 run is recorded there and stops the
    stage. collect, phase0 and check default to the frozen functions; tests pass stand-ins."""
    import pyarrow.parquet as pq
    from sportvu import colab_contract as cc
    from sportvu import splits
    from tools.frozen_common import REPO_ROOT, training_games_only
    if collect is None:
        from collect.bref_shots import main as collect
    if phase0 is None:
        from analytics.run_phase0 import run as phase0
    project_root = Path(project_root or REPO_ROOT)

    root = guard_root(root)
    if not read_record(root):
        raise SystemExit(f"{WHO}: no ingest.json under the root; run the download stage first")
    out_dir, collect_dir, work = root / "sportvu_contract", root / "collect", root / "_work"
    man = cc.load_manifest(out_dir)
    games = sorted(g for g, e in man["games"].items() if e["status"] == "ok")
    if not games:
        raise SystemExit(f"{WHO}: no converted game under {out_dir.name}; run the download stage first")
    training_games_only(games, WHO)
    stage = {"decision": "PLAN.md D78", "rpm": BREF_RPM, "pages_total": 2 * len(games),
             "pbp_join": "sportvu.colab_contract.join_game_pbp on the play-by-play events (fetch_pbp.fetch_game output)",
             "phase0": "analytics.run_phase0.run(out_root=this folder, steps=%s)" % ",".join(PHASE0_STEPS)}

    def stop(verdict: str, why: str):
        stage["verdict"] = verdict
        update_record(root, {"bref_stage": stage})
        raise SystemExit(f"{WHO}: {why}")

    final = collect(["--out", str(collect_dir), "--contract-dir", str(out_dir), "--fetch-pbp", "--rpm", str(BREF_RPM)])
    stage["collector"] = final
    print(final, flush=True)
    if not str(final).startswith("RUN OK"):
        stop("COLLECTOR FAILED", f"collect.bref_shots did not finish: {final}")

    sp = splits.load()
    pbp_shots, sources = {}, {}
    for game in games:
        e = man["games"][game]
        _, pbp_path, _ = cc.game_paths(e, out_dir, sp)
        src, events = pbp_source(e, pbp_path, collect_dir, project_root)
        sources[game] = src
        if events is not None:
            frames_path, _, _ = cc.game_paths(e, out_dir, sp)
            table = pq.read_table(frames_path, columns=["frame", "period", "game_clock_s"])
            res = cc.join_game_pbp(e, table, pbp_path, out_dir, work / "_parquet", pbp_fetch=lambda _c, ev=events: ev)
            del table
            if not res.get("pbp_joined"):
                cc.write_manifest(out_dir, man)
                stop("PBP JOIN FAILED", f"{game}: {res['pbp']['error']}")
            res["pbp"]["source"] = "basketball-reference.com"
            e.update(res)
            cc.write_manifest(out_dir, man)
            print(f"{game}: play-by-play {res['pbp']['events']} events, {res['pbp']['matched']} matched to a frame",
                  flush=True)
        pbp_shots[game] = pbp_field_goal_attempts(events, pbp_path)
    shutil.rmtree(work, ignore_errors=True)
    stage["pbp_sources"] = sources

    rc = phase0(out_dir, out_root=root, steps=PHASE0_STEPS, bref_dir=collect_dir)
    stage["phase0_verdict"] = "RUN OK" if rc == 0 else "RUN FAILED"
    if rc != 0:
        stop("PHASE0 FAILED", f"analytics.run_phase0 returned {rc}; see reports/phase0_summary.md under the root")
    loaded, counts = check(root, pbp_shots)
    stage.update(verdict="OK", load_data_root=loaded, per_game=counts, manifest_counts=cc.load_manifest(out_dir)["counts"])
    update_record(root, {"bref_stage": stage,
                         "blocked_on_basketball_reference": {"status": "fetched under D78", "pages_total": 2 * len(games)}})
    return stage


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["bref"]:
        ap = argparse.ArgumentParser(prog=f"{WHO} bref")
        ap.add_argument("--root", default=str(DEFAULT_ROOT))
        a = ap.parse_args(argv[1:])
        from export.heavy import heavy_job
        with heavy_job("research ingest bref"):
            st = bref_stage(Path(a.root))
        for g, c in st["per_game"].items():
            print(g, c, flush=True)
        print(st["verdict"], st["load_data_root"], flush=True)
        return 0 if st["verdict"] == "OK" else 1
    ap = argparse.ArgumentParser(prog=WHO, description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=None, help="test set source whose archives to ingest (default: the tracking source)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="output root (never under DATA_ROOT or reports/)")
    ap.add_argument("--no-listing", action="store_true", help="skip the GitHub listing cross-check")
    ap.add_argument("--force", action="store_true", help="convert again even when the parquet is already there")
    a = ap.parse_args(argv)
    from export.heavy import heavy_job
    from tools.research_testset import TRACKING, ingest_pins
    pins = checked_pins(ingest_pins(a.source or TRACKING))
    with heavy_job("research ingest"):
        rec = ingest(pins, Path(a.root), listing=not a.no_listing, force=a.force)
    c = rec["counts"]
    print(f"{c['ok']} ok, {c['source_broken']} source_broken, {c['failed']} failed of {c['games']}; "
          f"{c['frames_rows']} frames rows; ingest.json written", flush=True)
    return 0 if c["ok"] + c["source_broken"] == c["games"] else 1


if __name__ == "__main__":
    sys.exit(main())
