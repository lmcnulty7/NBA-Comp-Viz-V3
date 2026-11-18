"""tools/genai_data.py: the frozen splits and items of the genai research lane (PLAN.md D92, docs/research/genai.md).

    python3 -m tools.genai_data freeze [--site-root DIR] [--cache DIR]     snapshot, build every item, write the pins
    python3 -m tools.genai_data verify                                     recompute every pin of the committed file
    python3 -m tools.genai_data ledger                                     print the test-scoring ledger

`freeze` reads DATA_ROOT at scale (the small per-game tables of 180 training games), so it runs under the lock:
    python3 -m export.heavy --name genai-data -- /usr/local/bin/python3 -m tools.genai_data freeze

An item is the input export.genai builds for one text: a list of facts, each a field of the page and its value as the
page shows it. Two kinds:
- scouting: export.genai.player_input on players.json (descriptive training-game tables, PLAN.md rule 4), split by
  player. The data root is a snapshot of the site data root that holds the claude-haiku-5-5 texts (the comparison
  arm of genai-E1), copied with those texts and the cache records of their calls into the lane's data folder.
- possession: export.genai.possession_input on possessions of TRAINING games, split by game. The site's possession
  review shows held-out games only (export.site_games refuses any other game), so this module builds the same two
  contract models it would build (PossessionSummary and Possession, then export.public_data.strip_possession inside
  possession_input), from the same lane tables, for training games. Every model value is an out-of-fold row (fold
  >= 0): EPV states and xFG predictions of a training game. No tracking frame is read. Two differences from a site
  build, both outside what possession_input reads or limited to one value, are listed in TRAINING_BUILD_NOTES.

Splits are fixed by a salted sha256 of each key, before any text is written or scored: no outcome, no text and no
model output decides where an item goes. The committed file reports/research/genai-splits.json holds counts and
hashes only (no text, no fact, no name). The items themselves live outside the repository (V3_GENAI_DATA, default
a _genai-data folder beside the worktrees): they hold numbers of the data root and are never committed.

The test items are read in one place only: score_test_once(experiment, scorer), which writes one entry per
experiment to reports/research/genai-ledger.json before the scorer runs and refuses a second call. Everything else
gets train and dev items only (items(kind, split) refuses "test" outside that scorer).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_FILE = REPO_ROOT / "reports" / "research" / "genai-splits.json"
LEDGER_FILE = REPO_ROOT / "reports" / "research" / "genai-ledger.json"
DATA_ENV = "V3_GENAI_DATA"
DEFAULT_DATA = Path.home() / "Developer" / "nba-v3-worktrees" / "_genai-data"
DEFAULT_SITE_ROOT = Path.home() / "Developer" / "nba-v3-worktrees" / "_site-data-v4"
DEFAULT_P3_CACHE = Path.home() / "Developer" / "nba-v3-worktrees" / "_p3-genai-cache"

SALT = "genai-v1"
SPLITS = ("train", "dev", "test")
KINDS = ("scouting", "possession")
# Scouting: shares of the players, in rank order (test first). Possession: whole games, in rank order.
PLAYER_SHARES = {"test": 0.20, "dev": 0.10}
GAME_COUNTS = {"test": 40, "dev": 20, "train": 120}
POSSESSIONS_PER_GAME = {"test": 12, "dev": 12, "train": 25}
SITE_FILES = ("players.json", "results.json", "genai/scouting.json")
TRAINING_BUILD_NOTES = (
    "Shot time: a site build takes the release frame when it finds the shooter's tracked position at that frame, "
    "else the play-by-play event frame. This builder reads no tracking frame, so it takes the release frame whenever "
    "the attempt has a shooter and a release frame inside the possession, else the event frame.",
    "Shooter fallback: a site build names a shooter with no tracked id from the shot chart's player slug when exactly "
    "one rostered player has it. This builder does not read the shot chart, so such a shot has no shooter.",
    "attack_x is set to 0.0 and the frame arrays are empty: possession_input reads neither (it works on the public "
    "copy, which empties the frame arrays).",
)

_SCORING = None          # the experiment whose scorer is running inside score_test_once


class GenaiDataProblem(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ paths, hashes


def data_dir() -> Path:
    return Path(os.environ.get(DATA_ENV) or DEFAULT_DATA)


def rank_key(key: str) -> str:
    return hashlib.sha256(f"{SALT}:{key}".encode("utf-8")).hexdigest()


def list_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(rel: str) -> str:
    out = subprocess.run(["git", "hash-object", rel], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


# ------------------------------------------------------------------------------------------------ splits


def split_players(slugs) -> dict:
    """{split: sorted slugs}: players in salted-hash order, the first 20 percent test, the next 10 percent dev."""
    ordered = sorted(set(slugs), key=rank_key)
    n = len(ordered)
    n_test, n_dev = round(PLAYER_SHARES["test"] * n), round(PLAYER_SHARES["dev"] * n)
    return {"test": sorted(ordered[:n_test]), "dev": sorted(ordered[n_test:n_test + n_dev]),
            "train": sorted(ordered[n_test + n_dev:])}


def split_games(game_ids) -> dict:
    """{split: sorted game ids}: development games in salted-hash order, GAME_COUNTS of each, test first. The rest
    are left unused (reserve)."""
    ordered = sorted(set(game_ids), key=rank_key)
    out, at = {}, 0
    for split in ("test", "dev", "train"):
        out[split] = sorted(ordered[at:at + GAME_COUNTS[split]])
        at += GAME_COUNTS[split]
    out["reserve"] = sorted(ordered[at:])
    return out


def pick_possessions(game_id: str, numbers, k: int) -> list:
    """k possession numbers of a game drawn at random, seeded by the game's rank key: all when there are k or fewer."""
    numbers = sorted(int(n) for n in numbers)
    if len(numbers) <= k:
        return numbers
    return sorted(random.Random(rank_key(game_id)).sample(numbers, k))


def training_games(data_root) -> list:
    """Every 2015-16 game with a possessions table that the D71 door calls a development game (no test game, no
    spent held-out game)."""
    from tools.research_testset import dev_games
    folder = Path(data_root) / "analytics" / "possessions" / "2015-16"
    ids = sorted(p.stem for p in folder.glob("*.parquet"))
    if not ids:
        raise GenaiDataProblem("no training possessions tables under analytics/possessions/2015-16")
    return dev_games(ids, "genai_data")


# ------------------------------------------------------------------------------------------------ possessions


def _round_list(values, digits):
    from export import site_common as C
    return C.round_list(values, digits)


def build_training_game(data_root, game_id: str, xfg_rows, numbers) -> list:
    """[(PossessionSummary, Possession)] for the chosen possession numbers of one training game, built from the lane
    tables as export.site_games.build_game builds them (module docstring, TRAINING_BUILD_NOTES)."""
    import numpy as np
    import pandas as pd
    from analytics import schema as contract_tables
    from analytics.eval import is_heldout_game
    from analytics.possessions import game_meta
    from export import schemas as S
    from export import site_common as C
    from export import site_games as SG

    if is_heldout_game(game_id):
        raise GenaiDataProblem(f"{game_id} is a held-out game: the genai lane reads training games only")
    data_root = Path(data_root)
    analytics = data_root / "analytics"

    def lane(table):
        return contract_tables.table_path(analytics, table, C.SEASON, game_id, heldout=False)

    possessions = C.read_table(lane("possessions"), what=f"possessions of {game_id}")
    rosters = C.read_table(lane("rosters"), ["player_id", "player", "team_id", "team"], f"rosters of {game_id}")
    attempts = C.read_table(lane("shot_frames"), ["event", "period", "shooter_id", "release_frame", "made"],
                            f"shot_frames of {game_id}")
    events = C.read_table(data_root / "sportvu_contract" / f"{game_id}_pbp.parquet", ["event", "desc", "frame"],
                          f"play-by-play of {game_id}")
    epv = C.read_table(analytics / "epv" / C.SEASON / f"{game_id}.parquet",
                       ["possession", "frame", "epv_points", "fold", "heldout"], f"EPV states of {game_id}")
    if len(epv) and not ((epv["fold"].to_numpy() >= 0).all() and not epv["heldout"].astype(bool).any()):
        raise GenaiDataProblem(f"EPV states of {game_id} hold rows that are not out-of-fold rows (fold >= 0)")
    rows = xfg_rows[xfg_rows["sportvu_game"] == game_id]
    if len(rows) and not (rows["fold"].to_numpy() >= 0).all():
        raise GenaiDataProblem(f"xFG rows of {game_id} are not out-of-fold rows (fold >= 0)")
    if possessions["possession"].duplicated().any() or attempts["event"].duplicated().any():
        raise GenaiDataProblem(f"{game_id}: a possession or an attempt is listed twice")

    meta = game_meta(game_id)
    slug = "%s-%s-at-%s" % (meta["date"], meta["away"].lower(), meta["home"].lower())
    game = SG.Game(game_id=game_id, slug=slug, label="%s at %s" % (meta["away"], meta["home"]), date=meta["date"],
                   home=meta["home"], away=meta["away"], split="training")
    teams = SG._teams(rosters, game)
    people = {int(r.player_id): r.player for r in rosters.itertuples(index=False)}
    located = attempts.set_index("event")
    by_event = events.drop_duplicates("event").set_index("event")
    epv_by = {int(k): g.sort_values("frame", kind="stable") for k, g in epv.groupby("possession")}
    sources = [C.source(C.data_path("analytics/possessions", C.SEASON, f"{game_id}.parquet"),
                        "possession spans, offense, result and points", "training"),
               C.source(C.data_path("analytics/epv", C.SEASON, f"{game_id}.parquet"),
                        "EPV states of the frozen model, out-of-fold rows (fold >= 0); point estimates", "out_of_fold"),
               C.source(C.data_path("analytics/xfg", C.SEASON, "xfg_predictions.parquet"),
                        "xFG of the frozen model, out-of-fold rows (fold >= 0); point estimates", "out_of_fold")]

    wanted = set(int(n) for n in numbers)
    out = []
    for row in possessions.sort_values("possession", kind="stable").itertuples(index=False):
        number = int(row.possession)
        if number not in wanted:
            continue
        start, end = int(row.start_frame), int(row.end_frame)
        where = f"{game_id} possession {number}"
        if row.end_kind not in SG.RESULTS:
            raise GenaiDataProblem(f"{where}: unknown result {row.end_kind!r}")
        offense_id = int(row.offense_team_id)
        defense_id = next(t for t in teams if t != offense_id)
        n = (end - start) // C.STRIDE + 1
        grid = SG.Grid(start=start, end=end, n=n, rows=np.array([], dtype=np.int64),
                       slot=np.array([], dtype=np.int64), per_frame=np.zeros(n, dtype=np.int64), level={})
        curve = SG.epv_curve(epv_by.get(number), grid, where)
        t_grid = _round_list(np.arange(n) * C.STRIDE / C.FPS, 2)

        inside = rows[(rows["event"] >= row.start_event) & (rows["event"] <= row.end_event)]
        same_period = np.array([e in located.index and int(located.at[e, "period"]) == int(row.period)
                                for e in inside["event"]], dtype=bool)
        inside = inside.loc[same_period] if len(inside) else inside
        shot_models = []
        for shot in inside.sort_values("event", kind="stable").itertuples(index=False):
            attempt = located.loc[int(shot.event)]
            shooter_id = None if pd.isna(attempt["shooter_id"]) else int(attempt["shooter_id"])
            release = -1 if pd.isna(attempt["release_frame"]) else int(attempt["release_frame"])
            t, t_source = None, "none"
            if shooter_id is not None and release >= 0 and start <= release <= end:
                t, t_source = C.rnd((release - start) / C.FPS, 2), "release_frame"
            elif int(shot.event) in by_event.index:
                frame = by_event.at[int(shot.event), "frame"]
                if not pd.isna(frame) and start <= int(frame) <= end:
                    t, t_source = C.rnd((int(frame) - start) / C.FPS, 2), "play_by_play"
            desc = str(by_event.at[int(shot.event), "desc"]) if int(shot.event) in by_event.index else ""
            three = True if "3-pt" in desc else False if "2-pt" in desc else None
            name = people.get(shooter_id) if shooter_id is not None else None
            shot_models.append(S.Shot(event=int(shot.event), t=t, t_source=t_source, x=None, y=None,
                                      location_source="none", shooter_id=shooter_id,
                                      shooter=name if isinstance(name, str) else None, made=bool(shot.made),
                                      three=three, xfg=float(SG.published_xfg(shot.p_selected)), has_contest=None))
        last_attempts = attempts[(attempts["period"] == row.period) & (attempts["event"] >= row.start_event)
                                 & (attempts["event"] <= row.end_event)]
        predicted = {s.event: s for s in shot_models}
        last = predicted.get(int(last_attempts["event"].max())) if len(last_attempts) else None
        possession = S.Possession(
            schema_version=S.SCHEMA_VERSION, game_id=game_id, game_slug=slug, possession=number,
            period=int(row.period), hz=C.HZ, frame0=start, t=[], game_clock_s=[], shot_clock_s=[],
            offense=teams[offense_id], defense=teams[defense_id], attack_x=0.0, result=row.end_kind,
            points=int(row.points), gap=bool(row.gap), players=[], ball=S.BallTrack(x=[], y=[], z=[]), epv=curve,
            ghosts=None, shots=shot_models, notes=list(TRAINING_BUILD_NOTES), sources=sources)
        summary = S.PossessionSummary(
            possession=number, period=int(row.period), start_clock_s=C.rnd(row.start_clock_s, 2),
            end_clock_s=C.rnd(row.end_clock_s, 2), duration_s=t_grid[-1], offense=teams[offense_id],
            defense=teams[defense_id], result=row.end_kind, points=int(row.points), gap=bool(row.gap), n_frames=n,
            xfg=last.xfg if last is not None else None, xfg_bucket=None,
            epv_start=curve.value[0] if curve is not None else None,
            epv_end=curve.value[-1] if curve is not None else None, has_ghosts=False, featured=False, demo=False)
        out.append((summary, possession))
    missing = wanted - {s.possession for s, _ in out}
    if missing:
        raise GenaiDataProblem(f"{game_id}: possessions {sorted(missing)[:3]} are not in its table")
    return out


# ------------------------------------------------------------------------------------------------ items on disk


def item_path(kind: str, split: str) -> Path:
    return data_dir() / "items" / f"{kind}-{split}.jsonl"


def write_items(kind: str, split: str, items: list) -> None:
    path = item_path(kind, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def _read_items(kind: str, split: str) -> list:
    path = item_path(kind, split)
    if not path.exists():
        raise GenaiDataProblem(f"no {kind} {split} items at {path.name}: run python3 -m tools.genai_data freeze")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_pins(kind: str, split: str, items: list) -> None:
    from export.genai import input_sha256
    pins = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))[kind]["items"][split]
    if len(items) != pins["n"] or list_hash(i["key"] for i in items) != pins["keys_sha256"] \
            or list_hash(input_sha256(i) for i in items) != pins["inputs_sha256"]:
        raise GenaiDataProblem(f"{kind} {split} items do not match their pins in genai-splits.json")


def items(kind: str, split: str) -> list:
    """The frozen items of a kind and split, checked against the committed pins. Test items only inside the scorer
    of score_test_once."""
    if split not in SPLITS:
        raise GenaiDataProblem(f"unknown split {split!r}")
    if split == "test" and _SCORING is None:
        raise GenaiDataProblem("test items are read only inside the scorer given to score_test_once")
    found = _read_items(kind, split)
    _check_pins(kind, split, found)
    return found


def snapshot_file(rel: str) -> Path:
    """A file of the site-root snapshot (players.json, results.json, genai/scouting.json), checked against its pin."""
    path = data_dir() / "site" / rel
    pins = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["sources"]["site_files"]
    if sha256_file(path) != pins[rel]:
        raise GenaiDataProblem(f"the snapshot of {rel} does not match its pin")
    return path


def haiku_records() -> list:
    """The cache records of the claude-haiku-5-5 scouting calls (P3), as snapshotted, checked against their pin."""
    folder = data_dir() / "haiku-cache" / "scouting"
    files = sorted(folder.glob("*.json"))
    pins = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["sources"]["haiku_cache"]
    if list_hash(f"{p.name} {sha256_file(p)}" for p in files) != pins["records_sha256"]:
        raise GenaiDataProblem("the snapshot of the Haiku cache records does not match its pin")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


# ------------------------------------------------------------------------------------------------ the test door


def _ledger(path=None) -> list:
    path = Path(path or LEDGER_FILE)
    return json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []


def score_test_once(experiment: str, scorer, ledger_path=None):
    """Run `scorer()` with the test items open, once per experiment. The ledger entry is written before the scorer
    runs, so a crashed scoring run still uses up its score."""
    global _SCORING
    path = Path(ledger_path or LEDGER_FILE)
    entries = _ledger(path)
    if any(e["experiment"] == experiment for e in entries):
        raise GenaiDataProblem(f"{experiment} was already scored on the genai test items (genai-ledger.json)")
    if _SCORING is not None:
        raise GenaiDataProblem("score_test_once is already running")
    pins = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    entries.append({"experiment": experiment, "started": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "splits_sha256": sha256_file(SPLITS_FILE), "code_commit": git_head(),
                    "test_inputs_sha256": {k: pins[k]["items"]["test"]["inputs_sha256"] for k in KINDS}})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"_comment": "One entry per genai experiment scored on the genai test items "
                               "(tools/genai_data.py score_test_once). Entries are never edited or removed.",
                               "entries": entries}, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    _SCORING = experiment
    try:
        return scorer()
    finally:
        _SCORING = None


# ------------------------------------------------------------------------------------------------ freeze


def _snapshot(site_root: Path, cache: Path) -> dict:
    target = data_dir()
    for rel in SITE_FILES:
        dest = target / "site" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(site_root / rel, dest)
    folder = target / "haiku-cache" / "scouting"
    folder.mkdir(parents=True, exist_ok=True)
    for path in sorted((cache / "scouting").glob("*.json")):
        shutil.copy2(path, folder / path.name)
    files = sorted(folder.glob("*.json"))
    return {"site_files": {rel: sha256_file(target / "site" / rel) for rel in SITE_FILES},
            "haiku_cache": {"n_records": len(files),
                            "records_sha256": list_hash(f"{p.name} {sha256_file(p)}" for p in files)}}


def _pin(items: list) -> dict:
    from export.genai import input_sha256
    return {"n": len(items), "keys_sha256": list_hash(i["key"] for i in items),
            "inputs_sha256": list_hash(input_sha256(i) for i in items)}


def freeze(site_root=DEFAULT_SITE_ROOT, cache=DEFAULT_P3_CACHE, data_root=None, log=print) -> dict:
    """Snapshot the site root and the Haiku records, split, build every item and write the committed pins."""
    import pandas as pd
    from export import genai as G
    from export import schemas as S
    from export import site_common as C

    if SPLITS_FILE.exists():
        raise GenaiDataProblem("genai-splits.json exists: the splits are frozen and are never built again")
    data_root = Path(data_root or os.environ.get("DATA_ROOT") or "")
    if not (data_root / "analytics").exists():
        raise GenaiDataProblem("DATA_ROOT is not set to the data root")
    sources = _snapshot(Path(site_root), Path(cache))
    site = data_dir() / "site"
    players = S.PlayersFile.model_validate_json((site / "players.json").read_bytes())
    results = S.Results.model_validate_json((site / "results.json").read_bytes())

    record = {"_comment": "genai lane splits (tools/genai_data.py, docs/research/genai.md). Counts and hashes only: "
                          "the items hold numbers of the data root and live outside the repository.",
              "salt": SALT, "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "code_commit": git_head(),
              "checker": {"file": "export/genai.py", "git_blob_sha1": git_blob("export/genai.py")},
              "rule": {"scouting": f"players of players.json in sha256('{SALT}:' + slug) order: the first 20 percent "
                                   "test, the next 10 percent dev, the rest train",
                       "possession": f"development games (tools.research_testset.dev_games) with a possessions table, "
                                     f"in sha256('{SALT}:' + game id) order: {GAME_COUNTS['test']} test, "
                                     f"{GAME_COUNTS['dev']} dev, {GAME_COUNTS['train']} train, the rest unused; per "
                                     f"game {POSSESSIONS_PER_GAME} possessions drawn with random.Random(sha256 key)"},
              "training_build_notes": list(TRAINING_BUILD_NOTES), "sources": sources}

    by_split = split_players([p.slug for p in players.players])
    by_slug = {p.slug: p for p in players.players}
    record["scouting"] = {"items": {}}
    for split in SPLITS:
        built = [G.player_input(by_slug[s], players, results) for s in by_split[split]]
        write_items("scouting", split, built)
        record["scouting"]["items"][split] = _pin(built)
        log(f"scouting {split}: {len(built)} players")

    games = split_games(training_games(data_root))
    record["possession"] = {"games": {k: {"n": len(v), "ids_sha256": list_hash(v)} for k, v in games.items()},
                            "items": {}}
    xfg = C.read_table(data_root / "analytics" / "xfg" / C.SEASON / "xfg_predictions.parquet",
                       ["sportvu_game", "event", "p_selected", "made", "fold"], "xfg_predictions")
    chosen = set(games["test"] + games["dev"] + games["train"])
    xfg = xfg[xfg["sportvu_game"].astype(str).isin(chosen)].reset_index(drop=True)
    counts = {"out_of_fold_xfg_rows": int(len(xfg))}
    for split in SPLITS:
        built = []
        for game_id in games[split]:
            table = pd.read_parquet(data_root / "analytics" / "possessions" / C.SEASON / f"{game_id}.parquet",
                                    columns=["possession"])
            numbers = pick_possessions(game_id, table["possession"], POSSESSIONS_PER_GAME[split])
            for summary, possession in build_training_game(data_root, game_id, xfg, numbers):
                item = G.possession_input(summary, possession, results)
                item["game"] = game_id
                built.append(item)
        write_items("possession", split, built)
        record["possession"]["items"][split] = _pin(built)
        log(f"possession {split}: {len(built)} possessions of {len(games[split])} games")
    record["possession"]["counts"] = counts
    SPLITS_FILE.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def verify() -> list:
    """Problems with the committed pins against the items and snapshot on disk (empty list: all hold)."""
    global _SCORING
    problems = []
    record = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    for rel in SITE_FILES:
        try:
            snapshot_file(rel)
        except GenaiDataProblem as exc:
            problems.append(str(exc))
    try:
        haiku_records()
    except GenaiDataProblem as exc:
        problems.append(str(exc))
    for kind in KINDS:
        for split in SPLITS:
            try:
                _check_pins(kind, split, _read_items(kind, split))
            except GenaiDataProblem as exc:
                problems.append(str(exc))
    if record["salt"] != SALT:
        problems.append("the salt changed")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="genai lane splits (docs/research/genai.md)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--site-root", default=str(DEFAULT_SITE_ROOT))
    f.add_argument("--cache", default=str(DEFAULT_P3_CACHE))
    sub.add_parser("verify")
    sub.add_parser("ledger")
    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        from export.heavy import heavy_job
        with heavy_job("genai-data freeze"):
            record = freeze(args.site_root, args.cache)
        print(json.dumps({k: record[k] for k in ("scouting", "possession")}, indent=1))
        return 0
    if args.cmd == "verify":
        problems = verify()
        for line in problems:
            print("PROBLEM:", line)
        print("all pins hold" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(json.dumps(_ledger(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
