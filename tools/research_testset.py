"""tools/research_testset.py: the D71 test set of the post-V3 experiments, and the only door to it.

    python3 -m tools.research_testset verify                   recompute every hash of the committed file
    python3 -m tools.research_testset ledger                   print the scoring ledger

The committed file reports/research/testset.json lists, per source, the test games and the sha256 of the sorted list.
It was fixed before any experiment result existed (PLAN.md D71, D77): the 10 2015-16 games the frozen models never
read, and 10 Liga ACB 2025-26 games as an out-of-distribution check. The 17 held-out games of sportvu/splits.json are
spent and are never in it.

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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_FILE = REPO_ROOT / "reports" / "research" / "testset.json"
LEDGER_FILE = REPO_ROOT / "reports" / "research" / "scoring_ledger.json"

TRACKING = "sportvu_2015_16_untouched"
OOD = "skillcorner_acb_2025_26"


def list_hash(games) -> str:
    return hashlib.sha256("\n".join(sorted(games)).encode()).hexdigest()


def is_spent(game_id) -> bool:
    """True for a game the frozen V3 split holds out (spent; never a test or development game here). The frozen
    check analytics.eval.is_heldout_game decides, so the arena rule (every game at CLE) is included."""
    from analytics.eval import is_heldout_game
    return bool(is_heldout_game(game_id))


def _load(path=None) -> dict:
    data = json.loads(Path(path or TESTSET_FILE).read_text())
    for name, src in data["sources"].items():
        if list_hash(src["games"]) != src["sha256"]:
            raise SystemExit(f"{name}: the committed test games do not match their sha256; the test set was edited")
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
    if src.get("pins_sha256") and list_hash(f"{g} {pins[g]}" for g in pins) != src["pins_sha256"]:
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
    """The development games: every given game that is neither a D71 test game nor a spent held-out game."""
    test = set(_test_games(source, path))
    out = sorted(str(g) for g in all_games if str(g) not in test and not is_spent(g))
    refuse_test(out, who, source, path)
    return out


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
