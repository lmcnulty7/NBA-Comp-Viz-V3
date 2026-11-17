"""tools/research_testset.py: the D71 test set of the post-V3 experiments, and the only door to it.

    python3 -m tools.research_testset verify                   recompute every hash of the committed file
    python3 -m tools.research_testset ledger                   print the scoring ledger

The committed file reports/research/testset.json lists, per source, the test games and the sha256 of the sorted list.
It was fixed before any experiment result existed (PLAN.md D71, D77): the 10 2015-16 games the frozen models never
read, and 10 Liga ACB 2025-26 games as an out-of-distribution check. The 17 held-out games of sportvu/splits.json are
spent and are never in it. A third source (PLAN.md D85 (1)) is the broadcast CV test set of P8: 4 games in 3 whole
arenas, with their video ids, their SportVU archive blob pins and the home teams of those arenas, all hashed. No P8
development game may be one of its games, use one of its videos or be played in one of its arenas (refuse_cv_test).

Rules, enforced here and by tests/test_research_testset.py:
- Experiment code gets its development games from dev_games() and runs refuse_test() on every game id it fits,
  tunes or validates on. It never opens the committed file and never names a test game.
- Test games are read in exactly one place: score_once(experiment, source, scorer). It refuses an experiment that
  already has a ledger entry, so each experiment is scored on the test set once. The ledger
  (reports/research/scoring_ledger.json) is committed with the experiment's report.
"""

import argparse
import datetime as dt
import hashlib
import json
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
               "arena_homes": "arena_homes_sha256"}
# Keys a source must carry (a missing key is an edit of the test set, not an empty list).
REQUIRED_KEYS = {BROADCAST_CV: ("video_ids", "video_pins_sha256", "arena_homes", "arena_homes_sha256",
                                "archive_git_blob_sha1", "pins_sha256")}


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


def refuse_test(game_ids, who: str, source: str = TRACKING, path=None) -> int:
    """SystemExit when any id is a test game of the source -> the number of ids checked."""
    ids = {str(g) for g in game_ids}
    bad = sorted(ids & set(_test_games(source, path)))
    if bad:
        raise SystemExit(f"{who}: {len(bad)} D71 test game(s) reached development code, for example {bad[0]}")
    return len(ids)


def dev_games(all_games, who: str, source: str = TRACKING, path=None) -> list:
    """The development games: every given game that is neither a D71 test game nor a spent held-out game. For the
    broadcast CV source, a game played in one of its test arenas is left out too."""
    test = set(_test_games(source, path))
    homes = _test_homes(source, path)
    out = sorted(str(g) for g in all_games
                 if str(g) not in test and not is_spent(g) and _home_of(str(g)) not in homes)
    refuse_test(out, who, source, path)
    return out


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


def score_once(experiment: str, source: str, scorer, testset_path=None, ledger_path=None) -> dict:
    """The single final scoring call of an experiment on one source: scorer(test_games) -> a JSON-able summary.
    Refused when (experiment, source) already has a ledger entry, finished or not. A "started" entry is written
    before the scorer runs, so a run that crashes still uses up its one score; the entry is completed afterwards."""
    p = Path(ledger_path or LEDGER_FILE)
    entries = _ledger(ledger_path)
    if any(e["experiment"] == experiment and e["source"] == source for e in entries):
        raise SystemExit(f"{experiment} was already scored on {source}; each experiment is scored once per source")
    data = _load(testset_path)
    games = list(data["sources"][source]["games"])
    entry = {"experiment": experiment, "source": source, "sha256": data["sources"][source]["sha256"],
             "games": len(games), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "status": "started"}
    entries.append(entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(p, entries)
    summary = scorer(games)
    entry.update(status="scored", scored_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 summary=summary)
    _write_atomic(p, entries)
    return summary


def _write_atomic(path: Path, entries: list) -> None:
    import os
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(part, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.research_testset")
    ap.add_argument("command", choices=("verify", "ledger"))
    a = ap.parse_args(argv)
    if a.command == "verify":
        data = _load()
        for name, src in data["sources"].items():
            print(f"{name}: {len(src['games'])} games, sha256 {src['sha256']} verified")
    else:
        print(json.dumps(_ledger(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
