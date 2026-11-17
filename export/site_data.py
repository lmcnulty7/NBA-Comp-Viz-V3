"""export/site_data.py: one command that builds the site's data, end to end (PLAN.md M1).

    python3 -m export.site_data --out web/data

It reads the frozen models' stored outputs under the data root and the reports of this repository, and writes the
files of the contract (export/schemas.py): games/<slug>/<possession>.json and games/<slug>/possessions.json for the
held-out games, players.json, scorecard.json, results.json, methods.json (export/research_methods.py, from
reports/research/methods.json and the result files beside it) and, last, manifest.json. Then it validates the whole
tree with export.schemas.validate_tree and exits non-zero on any problem. Nothing is fitted, tuned or scored.

  --out        the data root to write (default web/data)
  --data       the folder that holds sportvu_contract/, analytics/ and collect/ (default: the DATA_ROOT variable)
  --artifacts  rebuilt frozen outputs (default: the V3_ARTIFACTS variable, else artifacts/)
  --only       a subset of the steps games, players, scorecard, results, methods; the manifest is always rewritten
  --games      a subset of the held-out games, by id or slug, for a quick run into a scratch folder
  --swaps      a json list of featured possessions picked by hand (PLAN.md D48, D52); each is marked as hand-picked
  --summary    also write what the run reports (counts, picks, sizes, problems) to this json file

demo/swap.json and demo/frames.json belong to export/demo_bridge.py. This command never writes or removes them: it
lists them in the manifest when they exist and lets them be absent. It never deletes anything: a file under the
output root that this run did not write and the contract does not expect is reported as a problem.

The manifest names the commit of the code that ran. A run on real data whose code or reports differ from that
commit (the manifest then ends in +uncommitted) is reported as a problem: commit first, then build.

The run holds the heavy-job lock and reads tracking frames one game at a time.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from export import genai
from export import heavy
from export import schemas as S
from export import site_common as C
from export import site_games
from export import site_players
from export import research_methods
from export import site_reports
from tools.frozen_common import artifacts_dir

SCORECARD_BUILD = "r23"        # the adopted tracking build (PLAN.md D5, D42); change this line to reverse D42
CANDIDATE_BUILD = "r23_cam"    # measured and not adopted: shown beside the adopted build with its paired differences
STEPS = ("games", "players", "scorecard", "results", "methods")
DEMO_FILES = S.STAGED_FILES    # written by export/demo_bridge.py: never written or removed here, may be absent
FIXED_FILES = ("players.json", "scorecard.json", "results.json", "methods.json") + DEMO_FILES

ExportProblem = C.ExportProblem


# ------------------------------------------------------------------------------------------------ steps


def _game_folders(out: Path) -> set:
    folder = out / "games"
    return {p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")} if folder.exists() else set()


def read_swaps(path) -> list:
    """The hand picks of a swaps file: a json list of {"cell": ..., "game_id": ..., "possession": ...}."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(doc, list):
            raise TypeError("the file holds no list")
        return [{"cell": str(item["cell"]), "game_id": str(item["game_id"]), "possession": int(item["possession"])}
                for item in doc]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ExportProblem("--swaps: the file is not a json list of objects with cell, game_id and possession "
                            f"({type(exc).__name__})") from None


def run_games(data_root: Path, artifacts: Path, out: Path, subset=None, log=print, swaps=()) -> dict:
    """Write the possession files and the index of every chosen held-out game. Returns what the run reports."""
    games = site_games.heldout_games(data_root)
    chosen = games
    if subset:
        by_name = {name: g for g in games for name in (g.game_id, g.slug)}
        unknown = [name for name in subset if name not in by_name]
        if unknown:
            raise ExportProblem(f"--games: {unknown} are not held-out games on the data root")
        chosen = [g for g in games if g.game_id in subset or g.slug in subset]
    extra = sorted(_game_folders(out) - {g.slug for g in chosen})
    if extra:
        raise ExportProblem(f"games/{extra[0]} (and {len(extra) - 1} more) under the output root are not written by "
                            "this run. Nothing is deleted here: use an empty output folder, or move them away.")
    ctx = site_games.load_shot_context(data_root, [g.game_id for g in games])
    demo = site_games.demo_possessions(artifacts)
    if demo:                                          # checked before anything is written
        listed = set()
        for game in games:
            path = Path(data_root) / "analytics" / "possessions" / "heldout" / C.SEASON / f"{game.game_id}.parquet"
            listed |= {(game.game_id, int(p)) for p in pd.read_parquet(path, columns=["possession"])["possession"]}
        if demo - listed:
            raise ExportProblem("artifacts/demo/possessions.json names possessions that do not exist: "
                                f"{sorted(demo - listed)[:3]}")

    results, totals = [], {}
    for game in chosen:
        started = time.time()
        result = site_games.build_game(data_root, game, ctx)
        written = set()
        for model in result.possessions:
            name = f"{model.possession}.json"
            C.write_model(out, f"games/{game.slug}/{name}", model)
            written.add(name)
        stale = sorted(p.name for p in (out / "games" / game.slug).iterdir()
                       if p.is_file() and not p.name.startswith(".") and p.name not in written | {"possessions.json"})
        if stale:
            raise ExportProblem(f"games/{game.slug}/{stale[0]} (and {len(stale) - 1} more) were not written by this "
                                "run. Nothing is deleted here: move them away.")
        result.possessions = []                       # the files are on disk: keep only the index rows
        results.append(result)
        for key, value in result.counts.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
        log(f"  {game.slug}: {result.counts['possessions']} possessions, {result.counts['predictions']} xFG rows, "
            f"{result.counts['eligible']} eligible to be featured, {time.time() - started:.1f} s")

    rows = {(r.game.game_id, row["possession"]): row for r in results for row in r.rows}
    featured = site_games.pick_featured([(game_id, number, row["cell"]) for (game_id, number), row in rows.items()
                                         if row["eligible"]])
    marks = site_games.featured_marks(featured["picks"], list(swaps), rows)
    by_hand = sorted(cell for cell, who in marks.values() if who == "hand")
    n_cells = len(featured["eligible"])
    note = (f"Featured possessions (PLAN.md D48): one drawn at random (seed {site_games.FEATURED_SEED}) from each of "
            f"the {n_cells} cells of result (made, missed) by xFG fifth, among possessions that end in a field-goal "
            f"attempt with a located release, an xFG, at least {site_games.FEATURED_MIN_EPV_STATES} EPV states, ghosts "
            f"on at least {site_games.FEATURED_MIN_GHOST_SHARE:.0%} of the frames, no clock gap, "
            f"{site_games.N_ON_COURT} tracked players and {site_games.FEATURED_MIN_S:g} to "
            f"{site_games.FEATURED_MAX_S:g} s of play. {len(marks)} of {n_cells} cells have a pick"
            + (f" in this run of {len(chosen)} of {len(games)} games." if len(chosen) < len(games) else ".")
            + (f" {len(by_hand)} of them were put there by hand in place of the rule's pick." if by_hand else ""))
    for result in results:
        index = site_games.index_of(result, ctx, marks, demo, note)
        C.write_model(out, f"games/{result.game.slug}/possessions.json", index)
    return {
        "games": len(chosen), "games_on_data_root": len(games), "partial": len(chosen) < len(games), "totals": totals,
        "xfg_bucket_edges": ctx.edges, "xfg_bucket_rows": ctx.n_bucket_rows, "xfg_bucket_games": ctx.n_bucket_games,
        "xfg_rows_in_games_shown": ctx.n_rows_shown, "xfg_games_not_shown": ctx.games_not_shown,
        "featured": dict(featured, by_hand=by_hand, n_cells=n_cells, n_filled=len(marks)),
        "demo_flagged": len(demo & set(rows)),
        "orientation": {r.game.slug: r.counts["orientation"] for r in results},
        "per_game": {r.game.slug: {k: v for k, v in r.counts.items() if k != "orientation"} for r in results},
    }


# ------------------------------------------------------------------------------------------------ manifest


def _sources_of(text: str) -> list:
    """Every source record inside a contract file, wherever it sits (the file's own list, a metric note, the
    shot-making baseline)."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if set(value) == set(S.Source.model_fields):
                found.append(S.Source(**value))
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json.loads(text))
    return C.merge_sources(found)


def build_manifest(out: Path, fixture_root: bool = False) -> S.Manifest:
    """The manifest of the tree as it is on disk: every file or file family with its model, count, size, hash and
    sources, and the size of each game. The possession files of a game are one family.

    `synthetic` is false: the contract reserves it for the invented values of export/sample.py, whose every source
    is that module. A tree this command builds names real tables as its sources and is checked by the rules for
    real data. When the data root was written by a test (`fixture_root`), the build id says so."""
    out = Path(out)

    def entry(rel: str) -> S.ManifestFile:
        data = (out / rel).read_bytes()
        return S.ManifestFile(path=rel, model=S.model_for(rel).__name__, count=1, bytes=len(data),
                              sha256=C.sha256_bytes(data), sources=_sources_of(data.decode("utf-8")))

    files = [entry(rel) for rel in FIXED_FILES if (out / rel).exists()]
    games = []
    for slug in sorted(_game_folders(out)):
        folder = out / "games" / slug
        index_rel = f"games/{slug}/possessions.json"
        if not (out / index_rel).exists():
            raise ExportProblem(f"games/{slug} has no possessions.json")
        index_entry = entry(index_rel)
        index = json.loads((out / index_rel).read_text(encoding="utf-8"))
        members = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
                         and S.model_for(f"games/{slug}/{p.name}") is S.Possession)
        if len(members) != len(index["possessions"]):
            raise ExportProblem(f"games/{slug}: {len(members)} possession files, {len(index['possessions'])} in the index")
        hashes, size, sources = [], 0, []
        for path in members:
            data = path.read_bytes()
            hashes.append(C.sha256_bytes(data))
            size += len(data)
            sources.append(_sources_of(data.decode("utf-8")))
        family = S.ManifestFile(path=f"games/{slug}/<possession>.json", model=S.Possession.__name__, count=len(members),
                                bytes=size, sha256=C.family_sha256(hashes), sources=C.merge_sources(*sources))
        files += [index_entry, family]
        games.append(S.ManifestGame(**index["game"], n_possessions=len(index["possessions"]),
                                    n_featured=sum(bool(p["featured"]) for p in index["possessions"]),
                                    bytes=index_entry.bytes + size))
    games.sort(key=lambda g: (g.date, g.slug))
    # Machine-written texts (export/genai.py) are listed when they exist, before the build id is hashed, so a
    # change to a text alone gives a new build id. This command never writes a text.
    files += genai.genai_entries(out)
    content = S.content_id(files)
    return S.Manifest(
        schema_version=S.SCHEMA_VERSION, build_id=f"{'test-fixture' if fixture_root else 'site'}-{content[:12]}",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repo_commit=C.repo_commit(),
        freeze_tag=C.FREEZE_TAG, synthetic=False, season=C.SEASON, n_games=len(games),
        n_possessions=sum(g.n_possessions for g in games), games=games, files=files)


# ------------------------------------------------------------------------------------------------ the run


def _safe_output(out: Path, data_root) -> None:
    """Refuse an output folder inside the data root or inside the repository's inputs."""
    target = out.expanduser().resolve()
    forbidden = [C.REPO_ROOT / name for name in ("reports", "analytics", "export", "tools", "tests", "sportvu")]
    if data_root is not None:
        forbidden.append(Path(data_root).expanduser().resolve())
    for folder in forbidden:
        if target == folder or folder in target.parents:
            raise ExportProblem(f"refusing to write the site data under {folder.name}")


def run(out, data_root=None, artifacts=None, only=None, games=None, log=print, allow_fixture_root=False, swaps=None,
        checks_copy=None):
    """Run the chosen steps, rewrite the manifest and validate the tree. Returns (problems, info). A data root
    written by a test is refused unless `allow_fixture_root` is true: only tests pass it. `checks_copy` is the
    tracked copy of the rebuild check records (default: reports/export/frozen_checks.json of this repository)."""
    out = Path(out)
    steps = [s for s in STEPS if not only or s in only]
    data_root = Path(data_root) if data_root else None
    artifacts = artifacts_dir(artifacts)
    problems, info = [], {"steps": steps}
    started = time.time()

    def step(name: str, body) -> None:
        if name not in steps:
            return
        t0 = time.time()
        try:
            info[name] = body()
        except (ExportProblem, research_methods.MethodsProblem) as exc:
            problems.append(f"{name}: {exc}")
            log(f"PROBLEM in {name}: {exc}")
        except ValueError as exc:                     # a value the contract refuses (pydantic) is a problem too
            problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:400]}")
            log(f"PROBLEM in {name}: {type(exc).__name__}")
        else:
            log(f"{name}: done in {time.time() - t0:.1f} s")

    try:
        _safe_output(out, data_root)
    except ExportProblem as exc:
        return [str(exc)], info
    if data_root is None and {"games", "players"} & set(steps):
        return ["no data root: pass --data or set DATA_ROOT"], info
    fixture_root = bool(data_root is not None and (data_root / C.FIXTURE_MARKER).exists())
    if fixture_root and not allow_fixture_root:
        return ["the data root was written by a test: this command builds site data from the real tables only"], info
    doc = None
    if {"players", "results"} & set(steps):
        try:
            doc = site_reports.read_results()
        except ExportProblem as exc:
            problems.append(f"reports: {exc}")
            log(f"PROBLEM in reports: {exc}")

    def players():
        if doc is None:
            raise ExportProblem("reports/results.md could not be read")
        file, players_info = site_players.build_players(data_root, artifacts, doc=doc, checks_copy=checks_copy)
        C.write_model(out, "players.json", file)
        return vars(players_info)

    def results():
        if doc is None:
            raise ExportProblem("reports/results.md could not be read")
        file = site_reports.build_results(doc=doc)
        C.write_model(out, "results.json", file)
        return {"lanes": len(file.lanes), "pending": [p.lane for p in file.pending],
                "classes": {lane.lane: lane.verdict_class for lane in file.lanes},
                "class_decided_by": site_reports.class_bases(doc)}

    def scorecard():
        file = site_reports.build_scorecard(SCORECARD_BUILD, CANDIDATE_BUILD)
        C.write_model(out, "scorecard.json", file)
        return {"builds": {b.build: [row.key for row in b.rows] for b in file.builds}, "paired": len(file.paired)}

    def methods():
        file = research_methods.build_methods()
        C.write_model(out, "methods.json", file)
        return {"lanes": len(file.lanes),
                "experiments": {e.id: e.status for lane in file.lanes for e in lane.experiments if e.status !=
                                "pre_registered"}}

    step("games", lambda: run_games(data_root, artifacts, out, games, log, read_swaps(swaps) if swaps else ()))
    step("players", players)
    step("scorecard", scorecard)
    step("results", results)
    step("methods", methods)

    try:
        # A text whose page changed is dropped before the manifest is written (no model call; export/genai.py).
        info["genai_dropped"] = genai.prune(out)
        manifest = build_manifest(out, fixture_root)
        C.write_model(out, "manifest.json", manifest)
        info["manifest"] = {
            "build_id": manifest.build_id, "repo_commit": manifest.repo_commit, "synthetic": manifest.synthetic,
            "n_games": manifest.n_games, "n_possessions": manifest.n_possessions,
            "n_featured": sum(g.n_featured for g in manifest.games), "entries": len(manifest.files),
            "files": sum(f.count for f in manifest.files) + 1, "bytes": sum(f.bytes for f in manifest.files),
            "game_bytes": {g.slug: g.bytes for g in manifest.games},
            "demo_files": [rel for rel in DEMO_FILES if (out / rel).exists()]}
        log(f"manifest: {manifest.n_games} games, {manifest.n_possessions} possessions, "
            f"{info['manifest']['bytes'] / 1e6:.1f} MB in {info['manifest']['files']} files")
        if manifest.repo_commit.endswith(C.UNCOMMITTED) and not fixture_root:
            # real data must trace to a commit that holds the code and the reports that built it
            problems.append(f"manifest: repo_commit is {manifest.repo_commit}: files under "
                            f"{', '.join(C.CODE_AND_INPUTS)} are changed or new against that commit. Commit them and "
                            "run again, so the manifest names the code that built the data.")
            log("PROBLEM in manifest: the code or the reports are not committed")
    except ExportProblem as exc:
        problems.append(f"manifest: {exc}")
        log(f"PROBLEM in manifest: {exc}")
    problems += S.validate_tree(out, allow_missing=DEMO_FILES)
    info["seconds"] = round(time.time() - started, 1)
    info["peak_memory_gb"] = round(heavy.peak_rss_gb(), 2)
    return problems, info


def _report(info: dict, problems: list) -> list:
    """What the command prints at its end: the counts a reader checks first."""
    lines = []
    games = info.get("games")
    if games:
        totals, featured = games["totals"], games["featured"]
        lines += [
            f"games: {games['games']} of {games['games_on_data_root']} held-out games, {totals['possessions']} possessions "
            f"({totals['one_frame']} single frame, {totals['gap']} with a clock gap, {totals['no_ghosts']} without ghosts, "
            f"{totals['no_epv']} without EPV)",
            f"shots: {totals['predictions']} predictions ({totals['predictions_unplaced']} in no possession); time from "
            f"release {totals['shots_release']}, play-by-play {totals['shots_event_time']}, none {totals['shots_no_time']}; "
            f"place from shot chart {totals['shots_chart_location']}, none {totals['shots_no_location']}; "
            f"{totals['shots_with_contest']} with contest features",
            f"featured: {featured['n_filled']} of {featured['n_cells']} cells filled from {totals['eligible']} eligible "
            f"possessions; empty: {featured['empty'] or 'none'}; by hand: {featured['by_hand'] or 'none'}",
            f"xFG fifths: edges {games['xfg_bucket_edges']} over {games['xfg_bucket_rows']} held-out predictions of "
            f"{games['xfg_bucket_games']} games ({games['xfg_rows_in_games_shown']} in the games with a review; not "
            f"shown: {games['xfg_games_not_shown'] or 'none'})",
        ]
    players = info.get("players")
    if players:
        lines.append(f"players: {players['n_players']} in training rosters, {players['n_shown']} shown, "
                     f"{players['n_hidden']} hidden; shown by metric {players['shown_by_metric']}; absence "
                     f"{players['absence']}; similar {players['similar']}")
        lines.append(f"shot-making: {players['attempts_left_out']} of {players['attempts_total']} out-of-fold attempts "
                     f"left out ({players['attempts_left_out_by_reason']}); {players['games_with_attempts']} games "
                     f"with attempts, {players['games_without_shot_frames']} of them without a shot_frames table")
    manifest = info.get("manifest")
    if manifest:
        lines.append(f"manifest: build {manifest['build_id']}, commit {manifest['repo_commit']}, {manifest['files']} files, "
                     f"{manifest['bytes'] / 1e6:.1f} MB; demo files listed: {manifest['demo_files'] or 'none'}")
        lines.append("size per game, MB: " + ", ".join(f"{slug} {size / 1e6:.2f}"
                                                        for slug, size in manifest["game_bytes"].items()))
    if "seconds" in info:
        lines.append(f"run: {info['seconds']} s, peak memory {info['peak_memory_gb']} GB")
    lines += [f"PROBLEM: {line}" for line in problems[:50]]
    lines.append(f"{len(problems)} problem(s)" if problems else "site data written and valid")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the site's data files and validate them against the contract.")
    parser.add_argument("--out", default="web/data", help="the data root to write (default web/data)")
    parser.add_argument("--data", default=None, help="the data root to read (default: the DATA_ROOT variable)")
    parser.add_argument("--artifacts", default=None, help="rebuilt frozen outputs (default: V3_ARTIFACTS, else artifacts/)")
    parser.add_argument("--only", nargs="+", choices=STEPS, help="run a subset of the steps")
    parser.add_argument("--games", nargs="+", help="a subset of the held-out games, by id or slug (quick runs)")
    parser.add_argument("--swaps", help="a json list of featured possessions picked by hand: cell, game_id, possession")
    parser.add_argument("--summary", help="also write what the run reports to this json file")
    args = parser.parse_args(argv)
    with heavy.heavy_job("export.site_data"):
        problems, info = run(args.out, args.data or os.environ.get("DATA_ROOT"), args.artifacts, args.only, args.games,
                             swaps=args.swaps)
    if args.summary:
        Path(args.summary).write_text(json.dumps({"problems": problems, **info}, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    for line in _report(info, problems):
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
