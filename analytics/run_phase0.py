"""analytics/run_phase0.py: run the Phase 0 localisers over every game of a Step 1 manifest.

For every game of DATA_ROOT/manifest.json (sportvu/colab_contract.py) with status ok and pbp_joined true, the
driver runs the four per-game modules through their module functions (never subprocesses):

    rosters         analytics.rosters.build + write_table      -> OUT_ROOT/analytics/rosters/<season>/<game>.parquet
    shot_frames     analytics.shot_frames.run                  -> OUT_ROOT/analytics/shot_frames/...
    rebound_frames  analytics.rebound_frames.run               -> OUT_ROOT/analytics/rebound_frames/...
    possessions     analytics.possessions.run                  -> OUT_ROOT/analytics/possessions/...
    gaps            analytics.gaps.write_report over the manifest -> OUT_ROOT/reports/sportvu_gaps.md (once, not per game)

Held-out games (sportvu/splits.json, through each module's own routing) land under <table>/heldout/<season>/.
Every localiser is deterministic and nothing here fits or tunes anything, so a held-out game is only ever
read and written, never learned from. Games without play-by-play are listed as skipped with the reason.

OUT_ROOT/analytics/phase0_manifest.json records, per game and step, the status, output path, rows, the
module's validate() numbers, schema.validate problems, seconds and error, so a rerun resumes: a step whose
output exists and reads back is reused, a failed step is left alone unless --retry-failed. At the end the
driver writes OUT_ROOT/reports/phase0_summary.md (games processed, per-step totals, the points-sum check
against the final score, the gaps summary, a per-game table) and prints RUN OK or RUN FAILED last.

    python -m analytics.run_phase0 --data DATA_ROOT [--out OUT_ROOT] [--games G ...] [--limit N]
        [--steps rosters,shot_frames,rebound_frames,possessions,gaps] [--retry-failed] [--dry-run] [--workers 1]
        [--fixtures]

--workers N > 1 runs games in a multiprocessing pool. The default is 1: Drive FUSE on a Mac serialises and
sometimes corrupts concurrent writers, so keep 1 when OUT_ROOT is on Drive. --fixtures writes a synthetic
DATA_ROOT (tests/_fx_phase0.py) when --data is not given and copies the summary to reports/phase0_driver.md.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import schema  # noqa: E402

GAME_STEPS = ("rosters", "shot_frames", "rebound_frames", "possessions")
ALL_STEPS = GAME_STEPS + ("gaps",)
MANIFEST_NAME = "phase0_manifest.json"
SUMMARY_NAME = "phase0_summary.md"
FIXTURE_REPORT = ROOT / "reports" / "phase0_driver.md"
MODULE_REPORTS = {"shot_frames": "localisation_shots.md", "rebound_frames": "localisation_rebounds.md"}
VERSION = 1


# ── manifest and game selection ───────────────────────────────────────────────────────────────────
def load_step1_manifest(data_root: Path) -> dict:
    p = Path(data_root) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("no Step 1 manifest at %s" % p)
    return json.loads(p.read_text())


def season_of_game(game: str) -> str:
    """'2015-16' from 'MM.DD.YYYY.AWAY.at.HOME' (seasons turn over in July, as the localisers do)."""
    mm, _dd, yyyy = game.split(".")[:3]
    y = int(yyyy) if int(mm) >= 7 else int(yyyy) - 1
    return "%d-%02d" % (y, (y + 1) % 100)


def select_games(manifest: dict, games: list[str] | None = None, limit: int | None = None):
    """(eligible [(game, entry)], skipped {game: reason}) over the manifest, in manifest order (the Step 1 order:
    training games first, then by date). Eligible: status ok and pbp_joined true."""
    entries = manifest.get("games", {})
    skipped: dict[str, str] = {}
    wanted = list(games) if games else list(entries)
    for g in wanted:
        if g not in entries:
            skipped[g] = "not in the manifest"
    eligible = []
    for g in wanted:
        e = entries.get(g)
        if e is None:
            continue
        if e.get("status") != "ok":
            skipped[g] = "status %s%s" % (e.get("status"), (": " + str(e.get("error"))[:120]) if e.get("error") else "")
        elif not e.get("pbp_joined"):
            err = (e.get("pbp") or {}).get("error")
            skipped[g] = "no play-by-play (pbp_joined false%s)" % ((": " + str(err)[:120]) if err else "")
        else:
            eligible.append((g, e))
    if limit is not None:
        eligible = eligible[:limit]
    return eligible, skipped


def input_paths(entry: dict, data_root: Path, sp: dict | None = None) -> tuple[Path, Path]:
    """(frames parquet, pbp parquet) of a manifest entry: the split routing of sportvu.colab_contract.game_paths,
    falling back to the other directory (or the manifest's own relative paths) when the file is not there."""
    from sportvu import splits
    from sportvu.colab_contract import game_paths
    sp = sp or splits.load()
    data_root = Path(data_root)
    fp, pp, _split = game_paths({"game": entry["game"], "home": entry["home"]}, data_root, sp)
    candidates = [(fp, pp)]
    other = data_root / "heldout" if fp.parent == data_root else data_root
    candidates.append((other / fp.name, other / pp.name))
    if entry.get("parquet"):
        candidates.append((data_root / entry["parquet"],
                           data_root / entry["pbp_parquet"] if entry.get("pbp_parquet") else other / pp.name))
    for f, p in candidates:
        if f.exists():
            return f, p
    return fp, pp


def expected_output(step: str, game: str, out_root: Path) -> Path:
    """Where the module will write: schema.table_path with the held-out flag of sportvu/splits.json."""
    from sportvu import splits
    from sportvu.colab_contract import is_heldout, split_of
    held = is_heldout(split_of({"game": game, "home": game.rsplit(".", 1)[-1]}, splits.load()))
    return schema.table_path(Path(out_root) / "analytics", step, season_of_game(game), game, heldout=held)


def reads_back(path: Path) -> int | None:
    """Row count when the parquet exists and its footer reads, else None."""
    try:
        return int(pq.read_metadata(str(path)).num_rows)
    except Exception:  # noqa: BLE001 (absent or torn)
        return None


def load_phase0_manifest(out_root: Path) -> dict:
    p = Path(out_root) / "analytics" / MANIFEST_NAME
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001 (a torn write; rebuild)
            pass
    return {"version": VERSION, "games": {}, "skipped": {}, "gaps": None}


def write_phase0_manifest(out_root: Path, man: dict) -> Path:
    p = Path(out_root) / "analytics" / MANIFEST_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    man["version"] = VERSION
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    games = man.get("games", {})
    man["counts"] = {"games": len(games),
                     "ok": sum(1 for e in games.values() if e.get("status") == "ok"),
                     "failed": sum(1 for e in games.values() if e.get("status") == "failed"),
                     "skipped": len(man.get("skipped", {}))}
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    os.replace(tmp, p)
    return p


# ── the steps ─────────────────────────────────────────────────────────────────────────────────────
def final_score(events: pd.DataFrame) -> dict | None:
    """{'away', 'home', 'total'} from the last non-null 'away-home' score of the events, or None."""
    if "score" not in events.columns:
        return None
    s = events["score"].dropna()
    if not len(s):
        return None
    try:
        away, home = str(s.iloc[-1]).split("-")
        away, home = int(away), int(home)
    except ValueError:
        return None
    return {"away": away, "home": home, "total": away + home}


def _run_rosters(game: str, frames: Path, pbp: Path, out_dir: Path) -> tuple[dict, Path]:
    from analytics import rosters
    events = pq.read_table(pbp, columns=["desc"]).to_pandas()
    pbp_json = {"events": [{"desc": d} for d in events["desc"].astype(object).tolist()]}
    table = rosters.build(game, frames, pbp_json=pbp_json)
    names = rosters.pbp_names(pbp_json)
    dest = rosters.write_table(table, rosters.out_path_for(game, out_dir))
    s = rosters.summarize(game, table, [], names)
    v = {"n_players": s["roster_size"], "pbp_total": s["pbp_total"], "pbp_matched": s["pbp_matched"],
         "methods": s["methods"], "unmatched": s["unmatched"][:10]}
    return v, dest


def _run_shot_frames(game: str, frames: Path, pbp: Path, out_dir: Path, report: Path) -> dict:
    from analytics import shot_frames
    with contextlib.redirect_stdout(io.StringIO()):
        return shot_frames.run(game, frames, pbp, out_dir, report)


def _run_rebound_frames(game: str, frames: Path, pbp: Path, out_dir: Path, report: Path) -> dict:
    from analytics import rebound_frames
    with contextlib.redirect_stdout(io.StringIO()):
        r = rebound_frames.run(game, frames, pbp, out_dir, report)
    return {k: v for k, v in r.items() if k not in ("out", "report", "split")}


def _run_possessions(game: str, frames: Path, pbp: Path, out_dir: Path) -> dict:
    from analytics import possessions
    with contextlib.redirect_stdout(io.StringIO()):
        s = possessions.run(game, frames, pbp, out_dir)
    s.pop("path", None)
    return s


def _extra_numbers(step: str, df: pd.DataFrame) -> dict:
    """Counts the summary adds over games (shares do not add; counts do)."""
    if step in ("shot_frames", "rebound_frames"):
        ident = df["identity_match"]
        known = ident.dropna()
        return {"n_identity_agree": int(known.astype(bool).sum()), "n_high": int((df["confidence_flag"] == "high").sum()),
                "n_low": int((df["confidence_flag"] == "low").sum()),
                "n_unmatched": int((df["confidence_flag"] == "unmatched").sum())}
    return {}


def run_step(step: str, game: str, frames: Path, pbp: Path, out_root: Path) -> dict:
    """One step of one game: run the module, read the output back, validate it against the contract."""
    out_root = Path(out_root)
    out_dir = out_root / "analytics" / step
    t0 = time.time()
    rec = {"status": "failed", "path": None, "rows": None, "validate": None, "schema_problems": None,
           "seconds": None, "error": None, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        expected = expected_output(step, game, out_root)
        if step == "rosters":
            v, dest = _run_rosters(game, frames, pbp, out_dir)
        elif step == "shot_frames":
            v = _run_shot_frames(game, frames, pbp, out_dir, out_root / "reports" / MODULE_REPORTS[step])
            dest = expected
        elif step == "rebound_frames":
            v = _run_rebound_frames(game, frames, pbp, out_dir, out_root / "reports" / MODULE_REPORTS[step])
            dest = expected
        elif step == "possessions":
            v = _run_possessions(game, frames, pbp, out_dir)
            dest = expected
        else:
            raise ValueError("unknown step %r" % step)
        if Path(dest) != expected:
            raise RuntimeError("module wrote %s, the driver expected %s" % (dest, expected))
        df = pq.read_table(str(dest)).to_pandas()
        problems = schema.validate(df, step)
        v = dict(v)
        v.update(_extra_numbers(step, df))
        rec.update(status="ok", path=str(Path(dest).relative_to(out_root)), rows=int(len(df)), validate=v,
                   schema_problems=problems[:5])
    except Exception as e:  # noqa: BLE001 (recorded per step; one game never ends the run)
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def process_game(args: tuple) -> dict:
    """(game, entry, data_root, out_root, steps to run, previous entry or None) -> the manifest entry of the game.
    Top level so a multiprocessing pool can pickle it."""
    game, entry, data_root, out_root, steps, previous = args
    data_root, out_root = Path(data_root), Path(out_root)
    frames, pbp = input_paths(entry, data_root)
    prev_steps = (previous or {}).get("steps", {})
    rec = {"game": game, "date": entry.get("date"), "away": entry.get("away"), "home": entry.get("home"),
           "split": entry.get("split"), "heldout": bool(entry.get("heldout")), "season": season_of_game(game),
           "frames": str(frames), "pbp": str(pbp), "steps": dict(prev_steps), "final_score": None,
           "points_check": None, "status": "ok", "seconds": 0.0}
    t0 = time.time()
    try:
        rec["final_score"] = final_score(pq.read_table(str(pbp), columns=["score"]).to_pandas())
    except Exception as e:  # noqa: BLE001
        rec["final_score"] = None
        rec["pbp_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    for step in steps:
        if not frames.exists():
            rec["steps"][step] = {"status": "failed", "error": "frames parquet missing: %s" % frames, "seconds": 0.0,
                                  "path": None, "rows": None, "validate": None, "schema_problems": None}
            continue
        if not pbp.exists():
            rec["steps"][step] = {"status": "failed", "error": "pbp parquet missing: %s" % pbp, "seconds": 0.0,
                                  "path": None, "rows": None, "validate": None, "schema_problems": None}
            continue
        rec["steps"][step] = run_step(step, game, frames, pbp, out_root)
    fs, poss = rec["final_score"], rec["steps"].get("possessions")
    if fs and poss and poss.get("status") == "ok":
        pts = int((poss.get("validate") or {}).get("points_total", 0))
        rec["points_check"] = {"possession_points": pts, "final_total": fs["total"], "ok": pts == fs["total"]}
    rec["status"] = "failed" if any(s.get("status") == "failed" for s in rec["steps"].values()) else "ok"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def pending_steps(game: str, previous: dict | None, steps, out_root: Path, retry_failed: bool) -> tuple[list, dict]:
    """(steps to run, reused step records). A step is reused when its last record is ok and the output reads
    back, or when it failed and --retry-failed is not set."""
    prev = (previous or {}).get("steps", {})
    todo, reused = [], {}
    for step in steps:
        p = prev.get(step)
        if p and p.get("status") == "ok" and p.get("path") and reads_back(Path(out_root) / p["path"]) is not None:
            reused[step] = {**p, "reused": True}
        elif p and p.get("status") == "failed" and not retry_failed:
            reused[step] = {**p, "reused": True}
        else:
            todo.append(step)
    return todo, reused


# ── the summary ───────────────────────────────────────────────────────────────────────────────────
def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "%.4f" % x
    return str(x)


def _ratio(num: int, den: int):
    return round(num / den, 4) if den else None


def totals(man: dict) -> dict:
    """Per-step totals over the games of the phase0 manifest (sums of counts, ratios from the sums)."""
    games = man.get("games", {})
    t = {s: {"games_ok": 0, "games_failed": 0, "rows": 0, "seconds": 0.0} for s in GAME_STEPS}
    for e in games.values():
        for step, r in e.get("steps", {}).items():
            if step not in t:
                continue
            if r.get("status") == "ok":
                t[step]["games_ok"] += 1
                t[step]["rows"] += int(r.get("rows") or 0)
            elif r.get("status") == "failed":
                t[step]["games_failed"] += 1
            t[step]["seconds"] += float(r.get("seconds") or 0.0)
    acc = {}
    for step, keys in (("shot_frames", ("n_shots", "n_release_found", "n_identity_known", "n_identity_agree", "n_high")),
                       ("rebound_frames", ("n_misses", "n_rebound_found", "n_identity_known", "n_identity_agree", "n_high")),
                       ("rosters", ("n_players", "pbp_total", "pbp_matched")),
                       ("possessions", ("n", "points_total", "n_gap", "n_dropped"))):
        acc[step] = {k: 0 for k in keys}
        for e in games.values():
            r = e.get("steps", {}).get(step)
            if r and r.get("status") == "ok" and r.get("validate"):
                for k in keys:
                    acc[step][k] += int(r["validate"].get(k) or 0)
    sf, rb, ro, po = acc["shot_frames"], acc["rebound_frames"], acc["rosters"], acc["possessions"]
    t["shot_frames"].update(sf, share_release_found=_ratio(sf["n_release_found"], sf["n_shots"]),
                            identity_agreement=_ratio(sf["n_identity_agree"], sf["n_identity_known"]),
                            share_high=_ratio(sf["n_high"], sf["n_shots"]))
    t["rebound_frames"].update(rb, share_rebound_found=_ratio(rb["n_rebound_found"], rb["n_misses"]),
                               identity_agreement=_ratio(rb["n_identity_agree"], rb["n_identity_known"]),
                               share_high=_ratio(rb["n_high"], rb["n_misses"]))
    t["rosters"].update(ro, share_pbp_matched=_ratio(ro["pbp_matched"], ro["pbp_total"]))
    methods = {}
    for e in games.values():
        r = e.get("steps", {}).get("rosters")
        for m, n in (((r or {}).get("validate") or {}).get("methods") or {}).items():
            methods[m] = methods.get(m, 0) + int(n)
    t["rosters"]["methods"] = methods
    by_kind: dict = {}
    for e in games.values():
        r = e.get("steps", {}).get("possessions")
        for k, n in (((r or {}).get("validate") or {}).get("by_end_kind") or {}).items():
            by_kind[k] = by_kind.get(k, 0) + int(n)
    checks = [e["points_check"] for e in games.values() if e.get("points_check")]
    t["possessions"].update(po, by_end_kind=by_kind, points_checked=len(checks),
                            points_match=sum(1 for c in checks if c["ok"]),
                            points_mismatch=[g for g, e in games.items() if e.get("points_check") and not e["points_check"]["ok"]])
    t["games"] = {"processed": len(games), "ok": sum(1 for e in games.values() if e.get("status") == "ok"),
                  "failed": sum(1 for e in games.values() if e.get("status") == "failed"),
                  "heldout": sum(1 for e in games.values() if e.get("heldout")),
                  "skipped": len(man.get("skipped", {}))}
    return t


def write_summary(man: dict, step1: dict, out_path: Path, data_root, out_root, verdict: str,
                  fixtures: bool = False) -> dict:
    """OUT_ROOT/reports/phase0_summary.md; returns totals()."""
    from analytics import gaps as gaps_mod
    t = totals(man)
    games = man.get("games", {})
    gs = man.get("gaps") or gaps_mod.summary(step1)
    L = ["# Phase 0 driver summary", ""]
    if fixtures:
        L += ["Fixture output: synthetic games written by tests/_fx_phase0.py (or analytics/fixtures.py), not real "
              "tracking data. The numbers only show the driver works end to end.", ""]
    L += ["Data root: `%s`. Output root: `%s`. Written %s by `python -m analytics.run_phase0`." % (
        data_root, out_root, man.get("updated", "")),
          "Held-out games (sportvu/splits.json) are written under `heldout/` by the modules' own routing and are never "
          "fit or tuned on; every localiser is deterministic with its thresholds fixed in the contract.", ""]
    g = t["games"]
    L += ["## Games", "", "| metric | value |", "|---|---|",
          "| games in the Step 1 manifest | %d |" % len(step1.get("games", {})),
          "| games processed (status ok, pbp joined) | %d |" % g["processed"],
          "| games ok (every step) | %d |" % g["ok"], "| games with a failed step | %d |" % g["failed"],
          "| held-out games among the processed | %d |" % g["heldout"], "| games skipped | %d |" % g["skipped"], ""]
    sf, rb, ro, po = t["shot_frames"], t["rebound_frames"], t["rosters"], t["possessions"]
    L += ["## Per-step totals", "", "| step | games ok | games failed | rows | seconds |", "|---|---|---|---|---|"]
    for s in GAME_STEPS:
        L.append("| %s | %d | %d | %d | %.1f |" % (s, t[s]["games_ok"], t[s]["games_failed"], t[s]["rows"], t[s]["seconds"]))
    L += ["", "### rosters", "", "| metric | value |", "|---|---|",
          "| players | %d |" % ro["n_players"], "| play-by-play names matched / total | %d / %d |" % (ro["pbp_matched"], ro["pbp_total"]),
          "| share_pbp_matched | %s |" % _fmt(ro["share_pbp_matched"]),
          "| match_method counts | %s |" % (", ".join("%s %d" % kv for kv in sorted(ro["methods"].items())) or "n/a"), ""]
    L += ["### shot_frames", "", "| metric | value |", "|---|---|",
          "| n_shots | %d |" % sf["n_shots"], "| n_release_found | %d |" % sf["n_release_found"],
          "| share_release_found | %s |" % _fmt(sf["share_release_found"]),
          "| n_identity_known | %d |" % sf["n_identity_known"], "| n_identity_agree | %d |" % sf["n_identity_agree"],
          "| identity_agreement (over all shots with both ids) | %s |" % _fmt(sf["identity_agreement"]),
          "| n_high | %d |" % sf["n_high"], "| share_high | %s |" % _fmt(sf["share_high"]), ""]
    L += ["### rebound_frames", "", "| metric | value |", "|---|---|",
          "| n_misses | %d |" % rb["n_misses"], "| n_rebound_found | %d |" % rb["n_rebound_found"],
          "| share_rebound_found | %s |" % _fmt(rb["share_rebound_found"]),
          "| n_identity_known | %d |" % rb["n_identity_known"], "| n_identity_agree | %d |" % rb["n_identity_agree"],
          "| identity_agreement (over all misses with both ids) | %s |" % _fmt(rb["identity_agreement"]),
          "| n_high | %d |" % rb["n_high"], "| share_high | %s |" % _fmt(rb["share_high"]), ""]
    L += ["### possessions", "", "| metric | value |", "|---|---|",
          "| possessions | %d |" % po["n"], "| points_total | %d |" % po["points_total"],
          "| possessions with a clock gap | %d |" % po["n_gap"], "| dropped (shorter than one frame) | %d |" % po["n_dropped"],
          "| by end_kind | %s |" % (", ".join("%s %d" % kv for kv in sorted(po["by_end_kind"].items())) or "n/a"),
          "| points-sum check: games with a final score in the events | %d |" % po["points_checked"],
          "| points-sum check: possession points equal the final score | %d |" % po["points_match"],
          "| points-sum mismatches | %s |" % (", ".join(po["points_mismatch"]) or "none"), ""]
    L += ["## Clock gaps (analytics.gaps over the Step 1 manifest)", "", "| metric | value |", "|---|---|",
          "| games in manifest | %d |" % gs["games"], "| games with status ok | %d |" % gs["games_ok"],
          "| games with any gap over 2 s | %d |" % gs["games_with_gap_over_2s"],
          "| games with any gap over %s s | %d |" % (gaps_mod._fmt(gs["threshold_s"]), gs["games_with_gap_over_threshold"]),
          "| gaps over 2 s | %d |" % gs["gaps_over_2s"], "| gaps over %s s | %d |" % (gaps_mod._fmt(gs["threshold_s"]), gs["gaps_over_threshold"]),
          "| total gap seconds | %s |" % gaps_mod._fmt(gs["total_gap_seconds"]), ""]
    if gs.get("by_split"):
        L += ["| split | games | ok | with gap over 2 s | with gap over threshold | gap seconds |", "|---|---|---|---|---|---|"]
        for split, b in gs["by_split"].items():
            L.append("| %s | %d | %d | %d | %d | %s |" % (split, b["games"], b["games_ok"], b["games_with_gap_over_2s"],
                                                       b["games_with_gap_over_threshold"], gaps_mod._fmt(b["total_gap_seconds"])))
        L.append("")
    skipped = man.get("skipped", {})
    L += ["## Skipped games", ""]
    if skipped:
        L += ["| game | reason |", "|---|---|"] + ["| %s | %s |" % (g, r) for g, r in sorted(skipped.items())]
    else:
        L.append("None.")
    L += ["", "## Per game", "",
          "| game | split | rosters | shot_frames | rebound_frames | possessions | n_shots | share_release_found "
          "| n_misses | share_rebound_found | possessions | points / final | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for gname in sorted(games):
        e = games[gname]
        st = e.get("steps", {})
        def status(step):
            r = st.get(step)
            return (r.get("status") or "n/a") if r else "not run"
        sv = (st.get("shot_frames") or {}).get("validate") or {}
        rv = (st.get("rebound_frames") or {}).get("validate") or {}
        pv = (st.get("possessions") or {}).get("validate") or {}
        pc = e.get("points_check")
        pts = ("%d / %d%s" % (pc["possession_points"], pc["final_total"], "" if pc["ok"] else " MISMATCH")) if pc else "n/a"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f |" % (
            gname, "held-out" if e.get("heldout") else (e.get("split") or "train"), status("rosters"), status("shot_frames"),
            status("rebound_frames"), status("possessions"), _fmt(sv.get("n_shots")), _fmt(sv.get("share_release_found")),
            _fmt(rv.get("n_misses")), _fmt(rv.get("share_rebound_found")), _fmt(pv.get("n")), pts,
            sum(float(r.get("seconds") or 0) for r in st.values())))
    failed = [(g, s, r.get("error")) for g, e in games.items() for s, r in e.get("steps", {}).items() if r.get("status") == "failed"]
    if failed:
        L += ["", "## Failed steps", "", "| game | step | error |", "|---|---|---|"]
        L += ["| %s | %s | %s |" % (g, s, (err or "").replace("|", "/")) for g, s, err in sorted(failed)]
    short = verdict.split(" (")[0]
    L += ["", verdict, ""] if verdict == short else ["", verdict, "", short, ""]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return t


# ── the run ───────────────────────────────────────────────────────────────────────────────────────
def run(data_root, out_root=None, games=None, limit=None, steps=ALL_STEPS, retry_failed=False, dry_run=False,
        workers=1, fixtures=False, log=print) -> int:
    """The driver. Returns 0 for RUN OK, 1 for RUN FAILED."""
    data_root = Path(data_root)
    out_root = Path(out_root) if out_root else data_root
    steps = tuple(steps)
    game_steps = [s for s in GAME_STEPS if s in steps]
    step1 = load_step1_manifest(data_root)
    eligible, skipped = select_games(step1, games, limit)
    man = load_phase0_manifest(out_root)
    man["data_root"], man["out_root"], man["steps"] = str(data_root), str(out_root), list(steps)
    log("phase0 driver: %d games in the manifest, %d eligible (status ok, pbp joined), %d skipped; steps %s; out %s" % (
        len(step1.get("games", {})), len(eligible), len(skipped), ",".join(steps), out_root))
    for g, reason in sorted(skipped.items()):
        log("  skip %-32s %s" % (g, reason))
    plan = []
    for g, e in eligible:
        todo, reused = pending_steps(g, man["games"].get(g), game_steps, out_root, retry_failed)
        plan.append((g, e, todo, reused))
    if dry_run:
        for g, e, todo, reused in plan:
            fp, pp = input_paths(e, data_root)
            log("  %-32s %-14s frames %s pbp %s | run %s | reuse %s" % (
                g, "held-out" if e.get("heldout") else (e.get("split") or "train"), "ok" if fp.exists() else "MISSING",
                "ok" if pp.exists() else "MISSING", ",".join(todo) or "-",
                ",".join("%s(%s)" % (s, r.get("status")) for s, r in reused.items()) or "-"))
        log("dry run: %d games would run, nothing written" % sum(1 for p in plan if p[2]))
        log("RUN OK")
        return 0
    if workers > 1:
        log("workers %d: games run in a multiprocessing pool; keep 1 when OUT_ROOT is on Google Drive (FUSE on a Mac "
            "misbehaves with many writers) and note the module reports may interleave" % workers)
    man.setdefault("skipped", {}).update(skipped)
    for g, _e in eligible:
        man["skipped"].pop(g, None)
    for g in list(man["games"]):
        if g in man["skipped"]:
            del man["games"][g]
    jobs, n_reused = [], 0
    for g, e, todo, reused in plan:
        prev = man["games"].get(g) or {}
        base = {"game": g, "date": e.get("date"), "away": e.get("away"), "home": e.get("home"), "split": e.get("split"),
                "heldout": bool(e.get("heldout")), "season": season_of_game(g), "steps": {},
                "final_score": prev.get("final_score"), "points_check": prev.get("points_check"), "status": "ok",
                "seconds": prev.get("seconds", 0.0), "frames": prev.get("frames"), "pbp": prev.get("pbp")}
        base["steps"] = reused
        man["games"][g] = base
        if todo:
            jobs.append((g, e, str(data_root), str(out_root), todo, {"steps": reused}))
        else:
            n_reused += 1
            base["status"] = "failed" if any(r.get("status") == "failed" for r in reused.values()) else "ok"
            if base["points_check"] is None and base.get("final_score") and reused.get("possessions", {}).get("status") == "ok":
                pts = int((reused["possessions"].get("validate") or {}).get("points_total", 0))
                base["points_check"] = {"possession_points": pts, "final_total": base["final_score"]["total"],
                                        "ok": pts == base["final_score"]["total"]}
    write_phase0_manifest(out_root, man)
    log("%d games to run, %d fully reused from %s" % (len(jobs), n_reused, MANIFEST_NAME))
    t0 = time.time()

    def finish(i: int, rec: dict) -> None:
        reused_steps = (man["games"].get(rec["game"]) or {}).get("steps", {})
        merged = dict(reused_steps)
        merged.update(rec["steps"])
        rec["steps"] = merged
        rec["status"] = "failed" if any(r.get("status") == "failed" for r in merged.values()) else "ok"
        man["games"][rec["game"]] = rec
        write_phase0_manifest(out_root, man)
        parts = ["%s %s" % (s, merged[s].get("status")) for s in game_steps if s in merged]
        log("%4d/%d %-32s %-9s %s (t+%.0f s)" % (i + 1, len(jobs), rec["game"], "held-out" if rec["heldout"] else "train",
                                                " ".join(parts), time.time() - t0))

    if jobs:
        if workers > 1:
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")           # the Mac default; uniform and fork-safe everywhere
            with ctx.Pool(workers) as pool:
                for i, rec in enumerate(pool.imap(process_game, jobs)):
                    finish(i, rec)
        else:
            for i, job in enumerate(jobs):
                finish(i, process_game(job))
    if "gaps" in steps:
        from analytics import gaps as gaps_mod
        try:
            man["gaps"] = gaps_mod.write_report(data_root / "manifest.json", out_root / "reports" / "sportvu_gaps.md")
            log("gaps: %d games, %d with a gap over %s s -> %s" % (
                man["gaps"]["games"], man["gaps"]["games_with_gap_over_threshold"], gaps_mod._fmt(man["gaps"]["threshold_s"]),
                out_root / "reports" / "sportvu_gaps.md"))
        except Exception as e:  # noqa: BLE001
            man["gaps"] = None
            man["gaps_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            log("gaps FAILED: %s" % man["gaps_error"])
    failed = [g for g, e in man["games"].items() if e.get("status") == "failed"] + (["gaps"] if man.get("gaps_error") else [])
    verdict = "RUN OK" if not failed else "RUN FAILED (%d game(s) with a failed step: %s)" % (
        len(failed), ", ".join(failed[:8]) + (" ..." if len(failed) > 8 else ""))
    write_phase0_manifest(out_root, man)
    summary_path = out_root / "reports" / SUMMARY_NAME
    t = write_summary(man, step1, summary_path, data_root, out_root, verdict, fixtures=fixtures)
    log("summary -> %s (%d games processed, %d skipped)" % (summary_path, t["games"]["processed"], t["games"]["skipped"]))
    if fixtures:
        FIXTURE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_REPORT.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        log("fixture report -> %s" % FIXTURE_REPORT)
    log(verdict.split(" (")[0])
    return 0 if not failed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m analytics.run_phase0", description=__doc__.split("\n")[0])
    ap.add_argument("--data", help="DATA_ROOT: the Step 1 sportvu_contract folder (manifest.json, <game>.parquet, heldout/)")
    ap.add_argument("--out", default=None, help="OUT_ROOT (default DATA_ROOT): analytics/<table>/... and reports/ go here")
    ap.add_argument("--games", nargs="*", default=None, help="only these SportVU game ids")
    ap.add_argument("--limit", type=int, default=None, help="at most N eligible games, in manifest order")
    ap.add_argument("--steps", default=",".join(ALL_STEPS), help="comma list of %s" % ",".join(ALL_STEPS))
    ap.add_argument("--retry-failed", action="store_true", help="rerun steps the phase0 manifest marks failed")
    ap.add_argument("--dry-run", action="store_true", help="list what would run; write nothing")
    ap.add_argument("--workers", type=int, default=1, help="games in parallel (default 1; keep 1 on Drive FUSE)")
    ap.add_argument("--fixtures", action="store_true",
                    help="use a synthetic DATA_ROOT (tests/_fx_phase0.py) when --data is absent and copy the summary to reports/phase0_driver.md")
    a = ap.parse_args(argv)
    steps = tuple(s.strip() for s in a.steps.split(",") if s.strip())
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        ap.error("unknown step(s) %s; known: %s" % (bad, ",".join(ALL_STEPS)))
    if a.workers < 1:
        ap.error("--workers must be at least 1")
    data = a.data
    if a.fixtures and not data:
        sys.path.insert(0, str(ROOT / "tests"))
        from _fx_phase0 import write_fixture_root
        data = tempfile.mkdtemp(prefix="phase0_fixtures_")       # left in place so the outputs can be inspected
        write_fixture_root(data)
        print("fixtures: synthetic DATA_ROOT written to %s" % data)
    if not data:
        ap.error("--data is required (or --fixtures)")
    return run(data, a.out, a.games, a.limit, steps, a.retry_failed, a.dry_run, a.workers, a.fixtures)


if __name__ == "__main__":
    sys.exit(main())
