"""tools/research_pbp2526.py: the 2025-26 NBA season from basketball-reference.com, the D71 test set of the
play-by-play lanes (PLAN.md D71, D78, D79, D83, D84).

    python3 -m tools.research_pbp2526 fetch [--root DIR] [--rpm 15] [--limit N] [--retry-failed]
    python3 -m tools.research_pbp2526 status [--root DIR]

A thin driver over the V3 collectors, unchanged, so the season is fetched and converted exactly as the frozen seasons
2020-21 to 2024-25 were:

1. collect.bref_schedule   the month pages of the season (leagues/NBA_2026_games-<month>.html)
2. collect.bref_seasons    one play-by-play page a game (boxscores/pbp/<code>.html), parsed by fetch_pbp.parse_events,
                           copied to collect/stints/playbyplay/2025-26.parquet (the table winprob and lineups read)
3. collect.bref_boxscores  one box score page a game (boxscores/<code>.html), the starters copied to
                           collect/stints/starters/2025-26.parquet (the table lineups reads)

The games are what collect.bref_schedule.games_for_season gives with the collectors' default season types, as for the
frozen seasons: games already played, Play-In games left out. Every page is cached under <root>/collect/raw/ and a
unit with status ok is never redone, so a rerun fetches nothing twice and a stopped run resumes where it stopped.
The rate is at most RPM_CAP requests a minute (below V3's 17 and the site's 20); the first HTTP 429 or 403 stops the
run and nothing later is attempted. Network only: no model code, so it runs outside the heavy-job lock (D84).

The root sits outside the repository, outside iCloud and outside DATA_ROOT. fetch.json under the root records the
pages fetched, the units per step, the failures and the cut-off (the last game date fetched). The season is then fixed
in the test set by the door: python3 -m tools.research_testset fix-pbp --root <root>.

SEALED: the frozen collectors also store outcome fields of these games under the root (the schedule's final scores,
the per-game event kinds in collect/bref_seasons/manifest.json, the box score stats). Nothing here prints or computes
any of them; this tool reports page, unit, row and game counts only. Only score_once's scorer reads a test outcome.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

SEASON = "2025-26"
RPM_CAP = 15.0
DEFAULT_ROOT = Path.home() / "Developer" / "nba-v3-worktrees" / "_research-data" / "pbp_2025_26"
REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS = ("bref_schedule", "bref_seasons", "bref_boxscores")
RAW_ENDPOINTS = ("bref_schedule", "bref_pbp_page", "bref_boxscore")
FORBIDDEN_HOME_DIRS = ("Documents", "Desktop", "Library/Mobile Documents")      # iCloud; the cloud drive is DATA_ROOT


def guard_root(root) -> Path:
    """The fetch root, refused when it is inside the repository, iCloud, DATA_ROOT or reports/."""
    from tools.frozen_common import assert_safe_output
    target = assert_safe_output(root)
    home = Path.home().resolve()
    for bad in [REPO_ROOT.resolve()] + [home / d for d in FORBIDDEN_HOME_DIRS]:
        if target == bad or bad in target.parents:
            raise SystemExit(f"refusing the root {target}: it is under {bad.name} (the repository or iCloud)")
    return target


def collect_out(root: Path) -> Path:
    return Path(root) / "collect"


def step_args(step: str, out: Path, rpm: float, limit=None, retry_failed=False) -> list:
    """The command line of one collector, every output under out (never the collectors' default --out)."""
    a = ["--out", str(out), "--rpm", str(min(rpm, RPM_CAP))]
    if step == "bref_seasons":
        a += ["--pbp-seasons", SEASON, "--shots-seasons"]          # no shot charts: an empty list
    else:
        a += ["--seasons", SEASON]
    if limit and step != "bref_schedule":
        a += ["--limit", str(int(limit))]
    if retry_failed:
        a += ["--retry-failed"]
    return a


def stopped(fam_dir: Path) -> str | None:
    """The error of a unit that stopped the run (HTTP 429 or 403), else None."""
    p = fam_dir / "manifest.json"
    if not p.exists():
        return None
    for key, u in json.loads(p.read_text()).get("units", {}).items():
        err = u.get("error") or ""
        if u.get("status") == "failed" and ("HTTP 429" in err or "HTTP 403" in err):
            return f"{key}: {err}"
    return None


def unit_counts(fam_dir: Path) -> dict:
    """{status: units} of the season in one collector's manifest, and the failed unit keys with their errors."""
    p = fam_dir / "manifest.json"
    out = {"ok": 0, "empty": 0, "failed": 0, "failures": {}}
    if not p.exists():
        return out
    for key, u in json.loads(p.read_text()).get("units", {}).items():
        if f"|{SEASON}|" not in key:
            continue
        st = u.get("status")
        out[st] = out.get(st, 0) + 1
        if st == "failed":
            out["failures"][key] = (u.get("error") or "")[:200]
    return out


def pages_on_disk(out: Path) -> dict:
    """Pages in the raw cache per endpoint (each was fetched once; a cached page is never requested again)."""
    raw = out / "raw"
    return {ep: len(list((raw / ep).glob("*.json.gz"))) if (raw / ep).exists() else 0 for ep in RAW_ENDPOINTS}


def season_games(out: Path) -> dict:
    """Game and date counts of the season's converted tables (no outcome column is read)."""
    import pyarrow.parquet as pq
    res = {}
    for name in ("playbyplay", "starters"):
        p = out / "stints" / name / f"{SEASON}.parquet"
        if p.exists():
            t = pq.read_table(p, columns=["game_id", "game_date"]).to_pandas()
            res[name] = {"rows": int(len(t)), "games": int(t["game_id"].nunique()),
                         "first_date": str(t["game_date"].min()), "last_date": str(t["game_date"].max())}
    return res


def scheduled_games(out: Path) -> dict:
    """The season's games the collectors would fetch (played, default season types), as a count; no score is read."""
    from collect import bref_schedule as sched
    from collect import common
    fam = common.Family(out, sched.FAMILY)
    games = sched.games_for_season(fam, SEASON)
    return {"games": len(games), "last_date": max((g["date"] for g in games), default=None)}


def record(root: Path, started: float | None, pages_before: dict | None, note: str = "") -> dict:
    out = collect_out(root)
    pages = pages_on_disk(out)
    rec = {"tool": "tools.research_pbp2526", "decision": "PLAN.md D71, D78, D79, D83, D84", "season": SEASON,
           "source": "basketball-reference.com", "rpm_cap": RPM_CAP,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "pages_in_cache": pages, "units": {s: unit_counts(out / s) for s in STEPS},
           "schedule": scheduled_games(out) if (out / "bref_schedule").exists() else {},
           "converted": season_games(out),
           "sealed": "outcome fields stored by the frozen collectors under this root are never printed or read here"}
    if pages_before is not None:
        rec["pages_fetched_this_run"] = {k: pages[k] - pages_before.get(k, 0) for k in pages}
    if started is not None:
        rec["seconds_this_run"] = round(time.time() - started, 1)
    if note:
        rec["note"] = note
    conv = rec["converted"].get("playbyplay")
    rec["cut_off_date"] = conv["last_date"] if conv else None
    (root / "fetch.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def fetch(root: Path, rpm: float, limit=None, retry_failed=False) -> dict:
    from collect import bref_boxscores, bref_schedule, bref_seasons
    mains = {"bref_schedule": bref_schedule.main, "bref_seasons": bref_seasons.main, "bref_boxscores": bref_boxscores.main}
    out = collect_out(root)
    out.mkdir(parents=True, exist_ok=True)
    rpm = min(float(rpm), RPM_CAP) if rpm and rpm > 0 else RPM_CAP
    t0, before = time.time(), pages_on_disk(out)
    note = ""
    for step in STEPS:
        print(f"== {step} {SEASON} at {rpm:.0f} requests a minute", flush=True)
        final = mains[step](step_args(step, out, rpm, limit, retry_failed))
        print(f"== {step}: {final}", flush=True)
        why = stopped(out / step)
        if why:
            note = f"stopped by the site at {step}: {why}"
            print("!! " + note + "; no later step runs. Back off and rerun later; nothing is fetched twice.", flush=True)
            break
        time.sleep(60.0 / rpm)                 # the next collector starts its own throttle: keep the gap
    return record(root, t0, before, note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.research_pbp2526", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("fetch", "status"))
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="fetch root (outside the repository, iCloud and DATA_ROOT)")
    ap.add_argument("--rpm", type=float, default=RPM_CAP, help=f"requests a minute (at most {RPM_CAP:.0f})")
    ap.add_argument("--limit", type=int, help="at most this many games per per-game step (a trial run)")
    ap.add_argument("--retry-failed", action="store_true", help="redo units whose last attempt failed (not after a 429)")
    a = ap.parse_args(argv)
    root = guard_root(a.root)
    root.mkdir(parents=True, exist_ok=True)
    rec = fetch(root, a.rpm, a.limit, a.retry_failed) if a.command == "fetch" else record(root, None, None)
    shown = {k: rec[k] for k in ("pages_in_cache", "pages_fetched_this_run", "seconds_this_run", "schedule",
                                 "converted", "cut_off_date", "note") if k in rec}
    shown["units"] = {s: {k: v for k, v in u.items() if k != "failures"} | {"failed_keys": sorted(u["failures"])[:20]}
                      for s, u in rec["units"].items()}
    print(json.dumps(shown, indent=2), flush=True)
    return 1 if rec.get("note") else 0


if __name__ == "__main__":
    sys.exit(main())
