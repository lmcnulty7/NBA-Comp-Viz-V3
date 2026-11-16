"""The D71 test set (PLAN.md D71, D77): fixed and hashed, and read only by one final scoring call per experiment."""

import ast
import json
import re
from pathlib import Path

import pytest

from tools import research_testset as R

REPO = Path(__file__).resolve().parents[1]
GAME_ID = re.compile(r"\d{2}\.\d{2}\.\d{4}\.[A-Z]{3}\.at\.[A-Z]{3}")
# Code that may run an experiment: every Python file under tools/ and export/. Only the door itself may read the file.
DOOR = REPO / "tools" / "research_testset.py"
PRIVATE_NAMES = ("_test_games", "TESTSET_FILE", "testset.json")


def experiment_files():
    return sorted(p for d in ("tools", "export") for p in (REPO / d).rglob("*.py") if p != DOOR)


def committed():
    return json.loads(R.TESTSET_FILE.read_text())


def test_the_committed_test_set_matches_its_hashes():
    data = committed()
    assert data["sources"], "the committed test set names at least one source"
    for name, src in data["sources"].items():
        assert R.list_hash(src["games"]) == src["sha256"], name
        assert len(src["games"]) == len(set(src["games"])), name


def test_no_spent_heldout_game_is_a_test_game():
    for name, src in committed()["sources"].items():
        assert not [g for g in src["games"] if R.is_spent(g)], name


def test_the_untouched_tracking_games_were_never_read_by_a_frozen_model():
    """None of the 10 games is in the frozen xFG table's game list or the contract manifest of DATA_ROOT; checked by the
    data survey (docs/research/DATA.md). Here: they are the 2015-12-05 games, none at CLE, and distinct from every
    spent held-out game."""
    games = committed()["sources"][R.TRACKING]["games"]
    assert len(games) == 10 and all(g.startswith("12.05.2015.") for g in games)
    assert not any(g.endswith(".at.CLE") for g in games)


def test_no_experiment_code_opens_the_test_set_or_names_a_test_game():
    test_ids = {g for src in committed()["sources"].values() for g in src["games"]}
    problems = []
    for path in experiment_files():
        text = path.read_text()
        for name in PRIVATE_NAMES:
            if name in text:
                problems.append(f"{path.relative_to(REPO)} uses {name}")
        for game in set(GAME_ID.findall(text)) & test_ids:
            problems.append(f"{path.relative_to(REPO)} names the test game {game}")
    assert problems == []


def test_each_experiment_module_has_at_most_one_final_scoring_call_and_it_is_not_in_a_loop():
    problems = []
    for path in experiment_files():
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", getattr(n.func, "id", None)) == "score_once"]
        if len(calls) > 1:
            problems.append(f"{path.relative_to(REPO)}: {len(calls)} score_once calls")
        loops = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        for loop in (n for n in ast.walk(tree) if isinstance(n, loops)):
            if any(inner in calls for inner in ast.walk(loop)):
                problems.append(f"{path.relative_to(REPO)}: score_once inside a loop")
        if calls and "dev_games" not in path.read_text() and "refuse_test" not in path.read_text():
            problems.append(f"{path.relative_to(REPO)} scores on the test set but never guards its development games")
    assert problems == []


@pytest.fixture
def tiny(tmp_path):
    games = ["11.01.2015.AAA.at.BBB", "11.02.2015.CCC.at.DDD"]
    path = tmp_path / "testset.json"
    path.write_text(json.dumps({"sources": {R.TRACKING: {"games": games, "sha256": R.list_hash(games)}}}))
    return path, games


def test_refuse_test_and_dev_games_keep_test_games_out(tiny):
    path, games = tiny
    with pytest.raises(SystemExit):
        R.refuse_test([games[0], "11.03.2015.EEE.at.FFF"], "t", path=path)
    dev = R.dev_games(games + ["11.03.2015.EEE.at.FFF", "11.04.2015.EEE.at.CLE"], "t", path=path)
    assert dev == ["11.03.2015.EEE.at.FFF"]          # the test games and the spent CLE game are both left out


def test_score_once_scores_each_experiment_once(tiny, tmp_path):
    path, games = tiny
    ledger = tmp_path / "ledger.json"
    seen = []
    out = R.score_once("x-E1", R.TRACKING, lambda g: seen.append(list(g)) or {"n": len(g)}, path, ledger)
    assert out == {"n": 2} and seen == [games]
    with pytest.raises(SystemExit):
        R.score_once("x-E1", R.TRACKING, lambda g: {"n": 0}, path, ledger)
    assert len(json.loads(ledger.read_text())) == 1


def test_an_edited_test_set_is_refused(tiny):
    path, games = tiny
    data = json.loads(path.read_text())
    data["sources"][R.TRACKING]["games"] = games[:1]
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        R.refuse_test(["11.09.2015.ZZZ.at.YYY"], "t", path=path)
