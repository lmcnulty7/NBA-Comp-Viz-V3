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
    pins = data["sources"][R.TRACKING]["archive_git_blob_sha1"]
    assert data["sources"][R.TRACKING]["pins_sha256"] == R.list_hash(f"{g} {pins[g]}" for g in pins)


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


# Names that reach the test games' data. Allowed only where they are defined or in the ingest tool.
TEST_DATA_NAMES = ("research_data_root", "test_features", "_research-data", "sportvu_untouched", "V3_RESEARCH_DATA",
                   "R._load", "import _load", "ingest_pins")
ALLOWED = {"tools/research_ingest.py": TEST_DATA_NAMES,
           "tools/research_xfg.py": ("research_data_root", "test_features", "_research-data", "sportvu_untouched",
                                     "V3_RESEARCH_DATA")}


def test_only_named_places_reach_the_test_games_data():
    problems = []
    for path in experiment_files():
        rel = str(path.relative_to(REPO))
        text = path.read_text()
        for name in TEST_DATA_NAMES:
            if name in text and name not in ALLOWED.get(rel, ()):
                problems.append(f"{rel} uses {name}")
    assert problems == []


def test_test_features_is_called_only_inside_the_scorer_given_to_score_once():
    tree = ast.parse((REPO / "tools" / "research_xfg.py").read_text())
    scorers = {a.id for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "score_once" for a in n.args if isinstance(a, ast.Name)}
    problems = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "test_features"]
        inner = {c for g in ast.walk(fn) if isinstance(g, ast.FunctionDef) and g is not fn for c in ast.walk(g)}
        for c in calls:
            if c not in inner and fn.name not in scorers:
                problems.append(f"test_features called in {fn.name}")
    ready = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "test_root_ready")
    body = [b for b in ready.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value.func.attr == "exists"
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name not in ("test_features", "test_root_ready")):
        if any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "research_data_root" for c in ast.walk(fn)):
            problems.append(f"research_data_root called in {fn.name}")
    assert scorers and problems == []


def test_score_once_is_never_aliased_or_reached_by_name():
    problems = []
    for path in experiment_files():
        text = path.read_text()
        if "score_once as" in text or '"score_once"' in text or "'score_once'" in text:
            problems.append(str(path.relative_to(REPO)))
    assert problems == []


def test_the_scoring_ledger_only_grows_across_its_whole_history():
    """Every committed version of the ledger is a prefix of the next and of the working tree, and an entry that was
    "scored" in any version is byte-identical in every later one."""
    import subprocess
    rel = "reports/research/scoring_ledger.json"
    shas = subprocess.run(["git", "log", "--reverse", "--format=%H", "--", rel], cwd=REPO, capture_output=True,
                          text=True).stdout.split()
    versions = []
    for sha in shas:
        shown = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=REPO, capture_output=True, text=True)
        if shown.returncode == 0:
            versions.append(json.loads(shown.stdout))
    if R.LEDGER_FILE.exists():
        versions.append(json.loads(R.LEDGER_FILE.read_text()))
    for older, newer in zip(versions, versions[1:]):
        assert len(newer) >= len(older), "a ledger entry was removed"
        for old, new in zip(older, newer):
            assert (old["experiment"], old["source"], old["started_at"]) == (new["experiment"], new["source"], new["started_at"])
            if old.get("status") == "scored":
                assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True), "a scored entry was edited"


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


def test_only_the_ingest_tool_calls_ingest_pins():
    users = [p.relative_to(REPO) for p in experiment_files() if "ingest_pins" in p.read_text()]
    assert [str(u) for u in users] in ([], ["tools/research_ingest.py"])


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
    entries = json.loads(ledger.read_text())
    assert len(entries) == 1 and entries[0]["status"] == "scored"


def test_a_crashed_scoring_run_still_uses_up_its_score(tiny, tmp_path):
    path, _games = tiny
    ledger = tmp_path / "ledger.json"

    def crash(_g):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        R.score_once("x-E2", R.TRACKING, crash, path, ledger)
    assert json.loads(ledger.read_text())[0]["status"] == "started"
    with pytest.raises(SystemExit):
        R.score_once("x-E2", R.TRACKING, lambda g: {}, path, ledger)


def test_an_edited_test_set_is_refused(tiny):
    path, games = tiny
    data = json.loads(path.read_text())
    data["sources"][R.TRACKING]["games"] = games[:1]
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        R.refuse_test(["11.09.2015.ZZZ.at.YYY"], "t", path=path)


def test_ingest_pins_must_name_exactly_the_test_games_and_match_their_hash(tmp_path):
    games = ["11.01.2015.AAA.at.BBB", "11.02.2015.CCC.at.DDD"]
    pins = {games[0]: "a" * 40, games[1]: "b" * 40}
    good = {"games": games, "sha256": R.list_hash(games), "archive_git_blob_sha1": pins,
            "pins_sha256": R.list_hash(f"{g} {pins[g]}" for g in pins)}
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"sources": {R.TRACKING: good}}))
    assert R.ingest_pins(path=path) == pins
    for broken in ({**good, "archive_git_blob_sha1": {games[0]: "a" * 40}},
                   {**good, "archive_git_blob_sha1": {**pins, games[1]: "c" * 40}}):
        path.write_text(json.dumps({"sources": {R.TRACKING: broken}}))
        with pytest.raises(SystemExit):
            R.ingest_pins(path=path)
