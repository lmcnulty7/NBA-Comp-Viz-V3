"""tools/research_testset.py: the D71 test set of the post-V3 experiments, and the only door to it.

    python3 -m tools.research_testset verify                   recompute every hash of the committed file
    python3 -m tools.research_testset ledger                   print the scoring ledger
    python3 -m tools.research_testset fix-pbp --root DIR       fix the 2025-26 season (once; tools.research_pbp2526)

The committed file reports/research/testset.json lists, per source, the test games and the sha256 of the sorted list.
It was fixed before any experiment result existed (PLAN.md D71, D77): the 10 2015-16 games the frozen models never
read, and 10 Liga ACB 2025-26 games as an out-of-distribution check. The 17 held-out games of sportvu/splits.json are
spent and are never in it. A third source (PLAN.md D85 (1)) is the broadcast CV test set of P8: 4 games in 3 whole
arenas, with their video ids, their SportVU archive blob pins and the home teams of those arenas, all hashed. No P8
development game may be one of its games, use one of its videos or be played in one of its arenas (refuse_cv_test).
The play-by-play lanes (winprob, lineups) score on the 2025-26 NBA season (source PBP, basketball-reference game codes
YYYYMMDD0HOME), fixed by fix_pbp from the files tools.research_pbp2526 converted, with the sha256 of each file, before
any play-by-play experiment ran (D78, D79, D83, D84). Their development seasons are 2020-21 to 2023-24
(PBP_DEV_SEASONS); 2024-25 was their held-out season and is spent.

Rules, enforced here and by tests/test_research_testset.py:
- Experiment code gets its development games from dev_games() and runs refuse_test() on every game id it fits,
  tunes or validates on. It never opens the committed file and never names a test game.
- Test games are read in exactly one place: score_once(experiment, source, scorer). It refuses an experiment that
  already has a ledger entry, so each experiment is scored on the test set once. The ledger
  (reports/research/scoring_ledger.json) is committed with the experiment's report.
- The converted files of a source with file pins (files_sha256) are hashed by score_once against the pins before
  its scorer runs, and are reached only through scoring_data_root(source), which works only inside that scorer.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_FILE = REPO_ROOT / "reports" / "research" / "testset.json"
LEDGER_FILE = REPO_ROOT / "reports" / "research" / "scoring_ledger.json"

TRACKING = "sportvu_2015_16_untouched"
OOD = "skillcorner_acb_2025_26"
BROADCAST_CV = "broadcast_cv_2015_16"
# The optional keyed lists of a source and the hash that seals each one (a sorted "key value" line per entry).
SEALED_MAPS = {"archive_git_blob_sha1": "pins_sha256", "video_ids": "video_pins_sha256",
               "arena_homes": "arena_homes_sha256", "files_sha256": "pins_sha256"}
# Keys a source must carry (a missing key is an edit of the test set, not an empty list).
REQUIRED_KEYS = {BROADCAST_CV: ("video_ids", "video_pins_sha256", "arena_homes", "arena_homes_sha256",
                                "archive_git_blob_sha1", "pins_sha256")}
PBP = "pbp_2025_26"
PBP_TEST_SEASON = "2025-26"
PBP_DEV_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24")      # D79
PBP_SPENT_SEASONS = ("2024-25",)                                    # scored once at the freeze (D79, rule 4)
PBP_FILES = ("collect/stints/playbyplay/2025-26.parquet", "collect/stints/starters/2025-26.parquet",
             "collect/bref_schedule/games/2025-26.parquet")
PBP_ROOT_ENV = "V3_RESEARCH_PBP"
DEFAULT_PBP_ROOT = Path.home() / "Developer" / "nba-v3-worktrees" / "_research-data" / PBP
REQUIRED_KEYS[PBP] = ("files_sha256", "pins_sha256")
BREF_CODE = re.compile(r"^(\d{4})(\d{2})(\d{2})0([A-Z]{3})$")
# First and last game date (any season type) of each season, as basketball-reference dates them. 2015-16 and 2020-21
# to 2024-25 equal the frozen schedule tables of DATA_ROOT (collect/bref_schedule/games), 2025-26 the table of
# tools.research_pbp2526; tests/test_research_testset.py checks every table it finds against this one. 2016-17 to
# 2019-20 have no schedule table; their dates are the league calendar (2019-20 ends with the bubble playoffs).
PBP_SEASON_DATES = {
    "2015-16": ("2015-10-27", "2016-06-19"), "2016-17": ("2016-10-25", "2017-06-12"),
    "2017-18": ("2017-10-17", "2018-06-08"), "2018-19": ("2018-10-16", "2019-06-13"),
    "2019-20": ("2019-10-22", "2020-10-11"), "2020-21": ("2020-12-22", "2021-07-20"),
    "2021-22": ("2021-10-19", "2022-06-16"), "2022-23": ("2022-10-18", "2023-06-12"),
    "2023-24": ("2023-10-24", "2024-06-17"), "2024-25": ("2024-10-22", "2025-06-22"),
    "2025-26": ("2025-10-21", "2026-06-13"),
}
_SCORING = None          # (experiment, source, verified data root or None) while score_once runs a scorer


def list_hash(games) -> str:
    return hashlib.sha256("\n".join(sorted(games)).encode()).hexdigest()


def is_spent(game_id) -> bool:
    """True for a game the frozen V3 split holds out (spent; never a test or development game here). The frozen
    check analytics.eval.is_heldout_game decides, so the arena rule (every game at CLE) is included."""
    from analytics.eval import is_heldout_game
    return bool(is_heldout_game(game_id))


def map_hash(mapping: dict) -> str:
    """The sha256 of a keyed list: one sorted "key value" line per entry, a list value giving one line per item."""
    return list_hash(f"{k} {v}" for k, vs in mapping.items() for v in (vs if isinstance(vs, list) else [vs]))


def _load(path=None) -> dict:
    data = json.loads(Path(path or TESTSET_FILE).read_text())
    for name, src in data["sources"].items():
        if list_hash(src["games"]) != src["sha256"]:
            raise SystemExit(f"{name}: the committed test games do not match their sha256; the test set was edited")
        missing = [k for k in REQUIRED_KEYS.get(name, ()) if not src.get(k)]
        if missing:
            raise SystemExit(f"{name}: the committed test set lacks {missing}; the test set was edited")
        for key, seal in SEALED_MAPS.items():
            if key in src and (not src.get(seal) or map_hash(src[key]) != src[seal]):
                raise SystemExit(f"{name}: {key} does not match {seal}; the test set was edited")
    return data


def _test_games(source: str, path=None) -> list:
    """The test games of one source. Private: only score_once and refuse_test may call it."""
    return list(_load(path)["sources"][source]["games"])


def ingest_pins(source: str = TRACKING, path=None) -> dict:
    """Game id -> archive git blob sha1 of a source, for tools.research_ingest only: it downloads and converts the
    test games and checks their bytes. Ingest fits, tunes and scores nothing; no experiment code may call this."""
    src = _load(path)["sources"][source]
    pins = dict(src.get("archive_git_blob_sha1") or {})
    if sorted(pins) != sorted(src["games"]):
        raise SystemExit(f"{source}: the archive pins do not name exactly the committed test games")
    if not src.get("pins_sha256") or list_hash(f"{g} {pins[g]}" for g in pins) != src["pins_sha256"]:
        raise SystemExit(f"{source}: the archive pins do not match their sha256; the pins were edited")
    return pins


def pbp_season(code) -> str:
    """'202510210LAL' -> '2025-26': the season of a basketball-reference game code, by the date ranges of
    PBP_SEASON_DATES (so a July 2021 Finals game is 2020-21 and an August 2020 bubble game is 2019-20). SystemExit for
    anything else, a code dated outside every range included, so a play-by-play guard never passes an id it cannot
    place."""
    m = BREF_CODE.match(str(code))
    if not m:
        raise SystemExit(f"{code!r} is not a basketball-reference game code (YYYYMMDD0HOME)")
    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    for season, (first, last) in PBP_SEASON_DATES.items():
        if first <= date <= last:
            return season
    raise SystemExit(f"{code!r} is dated outside every season of PBP_SEASON_DATES")


def refuse_test(game_ids, who: str, source: str = TRACKING, path=None) -> int:
    """SystemExit when any id is a test game of the source -> the number of ids checked. For PBP, every game of the
    test season and of the spent season 2024-25 is refused too, listed or not."""
    ids = {str(g) for g in game_ids}
    bad = sorted(ids & set(_test_games(source, path)))
    if source == PBP:
        bad = sorted(set(bad) | {g for g in ids if pbp_season(g) in (PBP_TEST_SEASON,) + PBP_SPENT_SEASONS})
    if bad:
        raise SystemExit(f"{who}: {len(bad)} D71 test game(s) reached development code, for example {bad[0]}")
    return len(ids)


def dev_games(all_games, who: str, source: str = TRACKING, path=None) -> list:
    """The development games: every given game that is neither a D71 test game nor a spent held-out game. For the
    broadcast CV source, a game played in one of its test arenas is left out too. For PBP, only games of the
    development seasons 2020-21 to 2023-24 (D79)."""
    test = set(_test_games(source, path))
    if source == PBP:
        out = sorted(str(g) for g in all_games if str(g) not in test and pbp_season(g) in PBP_DEV_SEASONS)
    else:
        homes = _test_homes(source, path)
        out = sorted(str(g) for g in all_games
                     if str(g) not in test and not is_spent(g) and _home_of(str(g)) not in homes)
    refuse_test(out, who, source, path)
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_pins(source: str = PBP, path=None) -> dict:
    """Relative path -> sha256 of the converted files of a source, checked against their committed pins_sha256."""
    src = _load(path)["sources"][source]
    pins = dict(src.get("files_sha256") or {})
    if not pins or not src.get("pins_sha256") or list_hash(f"{f} {pins[f]}" for f in pins) != src["pins_sha256"]:
        raise SystemExit(f"{source}: the file pins are missing or do not match their sha256; the pins were edited")
    return pins


def _checked_root(source: str, root=None, path=None) -> Path:
    """The data root of a file-pinned source after every pinned file under it matched its sha256, else SystemExit."""
    base = Path(root or os.environ.get(PBP_ROOT_ENV) or DEFAULT_PBP_ROOT)
    for rel, sha in file_pins(source, path).items():
        if not (base / rel).exists() or sha256_file(base / rel) != sha:
            raise SystemExit(f"{source}: {rel} under {base} is missing or differs from its pinned sha256")
    return base


def scoring_data_root(source: str = PBP) -> Path:
    """The folder holding a source's converted test files, whose bytes score_once checked against the pins before
    the scorer started. Refused unless score_once is running a scorer of this file-pinned source."""
    if _SCORING is None or _SCORING[1] != source or _SCORING[2] is None:
        raise SystemExit(f"the test data of {source} is read only inside the scorer that score_once runs")
    return _SCORING[2]


def _home_of(game_id: str):
    """The home team of a SportVU game name (MM.DD.YYYY.AWAY.at.HOME); None for any other id."""
    parts = game_id.split(".")
    return parts[5] if len(parts) == 6 and parts[4] == "at" else None


def _test_homes(source: str, path=None) -> set:
    """The home team codes of a source's test arenas (empty for a source without arenas). Private."""
    src = _load(path)["sources"][source]
    return {h for hs in (src.get("arena_homes") or {}).values() for h in hs}


def cv_test_hits(items, path=None, homes=()) -> list:
    """The items that reach the broadcast CV test set, by substring and without regard to case, like
    sportvu.splits.heldout_hits: an item (a game id, video id, tag, section stem, path or URL) is a hit when it
    contains a test game id, a test video id, ".at.<home>" of a test arena, or a tag that starts with "<home>_" of a
    test arena (home-first tags such as mia_nop_2015, at the start of the item or of any path segment). The optional
    homes are team codes checked as homes against the test arenas (a registry entry's own home)."""
    src = _load(path)["sources"][BROADCAST_CV]
    tokens = [t.lower() for t in list(src["games"]) + list(src["video_ids"].values())]
    test_homes = sorted({h.lower() for hs in src["arena_homes"].values() for h in hs})
    tokens += [f".at.{h}" for h in test_homes]
    tag_re = re.compile(r"(?:^|[/\\\s=])(?:%s)_" % "|".join(test_homes))
    hits = [str(i) for i in items if any(t in str(i).lower() for t in tokens) or tag_re.search(str(i).lower())]
    hits += [f"home {h}" for h in homes if str(h).lower() in test_homes]
    return hits


def refuse_cv_test(items, who: str, path=None, homes=()) -> int:
    """The P8 guard. SystemExit when any item reaches the broadcast CV test set (cv_test_hits: a test game, a test
    video, a game in a test arena, by substring, in any id, tag, stem, path or URL), or when one of the given home team
    codes plays in a test arena -> the number of items checked. Every P8 tool runs it on each id, tag and path it
    downloads, syncs, builds, labels, fits, tunes or scores on, outside score_once."""
    items = [str(i) for i in items]
    bad = cv_test_hits(items, path, homes)
    if bad:
        raise SystemExit(f"{who}: {len(bad)} broadcast CV test item(s) reached development code (a test game, a test "
                         f"video or a game in a test arena), for example {bad[0]}")
    return len(items)


def cv_test_videos(path=None) -> dict:
    """Game id -> video id of the broadcast CV test set, for tools.p8_fetch only: it downloads the test videos into
    their sealed folder and checks the container with ffprobe. It runs no model on them; no other P8 code may call
    this (tests/test_research_testset.py)."""
    src = _load(path)["sources"][BROADCAST_CV]
    vids = dict(src["video_ids"])
    if sorted(vids) != sorted(src["games"]):
        raise SystemExit(f"{BROADCAST_CV}: the video ids do not name exactly the committed test games")
    return vids


def _ledger(path=None) -> list:
    p = Path(path or LEDGER_FILE)
    return json.loads(p.read_text()) if p.exists() else []


def score_once(experiment: str, source: str, scorer, testset_path=None, ledger_path=None, data_root=None) -> dict:
    """The single final scoring call of an experiment on one source: scorer(test_games) -> a JSON-able summary.
    Refused when (experiment, source) already has a ledger entry, finished or not. A "started" entry is written
    before the scorer runs, so a run that crashes still uses up its one score; the entry is completed afterwards.
    For a source with file pins, every pinned file under data_root (else $V3_RESEARCH_PBP, else DEFAULT_PBP_ROOT) is
    hashed against its pin first; a missing or changed file refuses the call before the ledger entry is written."""
    p = Path(ledger_path or LEDGER_FILE)
    entries = _ledger(ledger_path)
    if any(e["experiment"] == experiment and e["source"] == source for e in entries):
        raise SystemExit(f"{experiment} was already scored on {source}; each experiment is scored once per source")
    data = _load(testset_path)
    games = list(data["sources"][source]["games"])
    root = _checked_root(source, data_root, testset_path) if data["sources"][source].get("files_sha256") else None
    entry = {"experiment": experiment, "source": source, "sha256": data["sources"][source]["sha256"],
             "games": len(games), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "status": "started"}
    entries.append(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(p, entries)
    global _SCORING
    _SCORING = (experiment, source, root)
    try:
        summary = scorer(games)
    finally:
        _SCORING = None
    entry.update(status="scored", scored_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 summary=summary)
    _write_atomic(p, entries)
    return summary


def _write_atomic(path: Path, entries: list) -> None:
    import os
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(part, path)


def fix_pbp(root, path=None, ledger_path=None, accept_missing: bool = False) -> dict:
    """Move the 2025-26 season from pending to sources, once, before any play-by-play result exists: the game list
    (every game with converted play-by-play), the cut-off date, and the sha256 of each converted file (PBP_FILES).
    Reads game ids and dates only, no outcome column. Refused when the source is already fixed, when the ledger holds
    any entry of it, or (unless accept_missing) when a game of the collected schedule has no play-by-play."""
    import pyarrow.parquet as pq
    p = Path(path or TESTSET_FILE)
    data = _load(p)
    if PBP in data["sources"]:
        raise SystemExit(f"{PBP} is already fixed; a fixed source is never rewritten")
    if PBP not in data.get("pending", {}):
        raise SystemExit(f"{PBP} is not pending in {p.name}")
    if any(e["source"] == PBP for e in _ledger(ledger_path)):
        raise SystemExit(f"{PBP} already has a ledger entry, so a result exists; it can no longer be fixed")
    root = Path(root)
    missing = [f for f in PBP_FILES if not (root / f).exists()]
    if missing:
        raise SystemExit(f"{PBP}: converted files missing under {root}: {missing}")
    pbp = pq.read_table(root / PBP_FILES[0], columns=["game_id", "game_date", "season"]).to_pandas()
    starters = pq.read_table(root / PBP_FILES[1], columns=["game_id"]).to_pandas()
    sched = pq.read_table(root / PBP_FILES[2], columns=["game_id", "date", "season_type"]).to_pandas()
    games = sorted(str(g) for g in pbp["game_id"].unique())
    if set(pbp["season"].astype(str)) != {PBP_TEST_SEASON} or any(pbp_season(g) != PBP_TEST_SEASON for g in games):
        raise SystemExit(f"{PBP}: the play-by-play table holds games outside {PBP_TEST_SEASON}")
    fetch = json.loads((root / "fetch.json").read_text()) if (root / "fetch.json").exists() else {}
    listed = (fetch.get("schedule") or {}).get("games")
    not_fetched = (listed - len(games)) if listed is not None else None
    if not_fetched and not accept_missing:
        raise SystemExit(f"{PBP}: {not_fetched} scheduled game(s) have no play-by-play; finish the fetch or pass "
                         "--accept-missing to record them as missing")
    pins = {f: sha256_file(root / f) for f in PBP_FILES}
    src = {
        "role": "the scoring set of the play-by-play lanes (winprob, lineups), forward in time",
        "origin": ("basketball-reference.com, 2025-26 season, fetched by tools.research_pbp2526 with the V3 collectors "
                   "unchanged (collect.bref_schedule, collect.bref_seasons, collect.bref_boxscores) at most 15 requests "
                   "a minute"),
        "selection": ("every game collect.bref_schedule.games_for_season lists with the collectors' default season "
                      "types (played games; Play-In left out; the playoffs and the NBA Cup final carry the Regular "
                      "Season tag, as in the frozen seasons 2020-21 to 2024-25) that has converted play-by-play"),
        "season": PBP_TEST_SEASON,
        "cut_off_date": str(pbp["game_date"].max()),
        "first_date": str(pbp["game_date"].min()),
        "fetched_on": (fetch.get("written") or "")[:10] or None,
        "games": games,
        "counts": {"playbyplay_rows": int(len(pbp)), "games": len(games),
                   "games_with_starters": len(set(starters["game_id"].astype(str)) & set(games)),
                   "scheduled_games": int(listed) if listed is not None else None,
                   "scheduled_without_playbyplay": int(not_fetched) if not_fetched is not None else None,
                   "schedule_rows_by_type": {str(k): int(v) for k, v in sched["season_type"].value_counts().items()}},
        "files_sha256": pins,
        "sha256": list_hash(games),
        "pins_sha256": list_hash(f"{f} {pins[f]}" for f in pins),
    }
    data["sources"][PBP] = src
    data["pending"].pop(PBP)
    if not data["pending"]:
        data.pop("pending")
    data.setdefault("fixed_later", {})[PBP] = {"fixed_on": dt.date.today().isoformat(),
                                               "decision": "PLAN.md D78, D79, D83, D84"}
    part = p.with_name(p.name + ".part")
    part.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(part, p)
    _load(p)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.research_testset")
    ap.add_argument("command", choices=("verify", "ledger", "fix-pbp"))
    ap.add_argument("--root", default=str(DEFAULT_PBP_ROOT), help="fix-pbp: the fetch root of tools.research_pbp2526")
    ap.add_argument("--accept-missing", action="store_true", help="fix-pbp: record scheduled games without play-by-play")
    a = ap.parse_args(argv)
    if a.command == "verify":
        data = _load()
        for name, src in data["sources"].items():
            extra = f"; {len(file_pins(name))} file pins verified" if src.get("files_sha256") else ""
            print(f"{name}: {len(src['games'])} games, sha256 {src['sha256']} verified{extra}")
    elif a.command == "fix-pbp":
        src = fix_pbp(a.root, accept_missing=a.accept_missing)
        print(json.dumps({k: src[k] for k in ("season", "first_date", "cut_off_date", "fetched_on", "counts",
                                              "files_sha256", "sha256", "pins_sha256")}, indent=2))
    else:
        print(json.dumps(_ledger(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
