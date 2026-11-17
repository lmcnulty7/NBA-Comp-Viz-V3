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


# ── the broadcast CV test set of P8 (PLAN.md D85 (1)) ───────────────────────────────────────────────────────────────
# Names that reach the CV test videos. Allowed only in the download tool, which runs no model on them.
CV_TEST_NAMES = ("cv_test_videos", "p8_cv_test")
CV_ALLOWED = {"tools/p8_fetch.py": CV_TEST_NAMES}
# Registries the frozen CV code reads (sportvu/, harvest_driver, clock_reader): a CV test game or video id listed in
# one of them could be picked up by a frozen tool, so none may hold one.
FROZEN_CV_REGISTRIES = ("data/harvest/games.json", "sportvu/manifest_overrides.json", "sportvu/splits.json")


def cv_source():
    return committed()["sources"][R.BROADCAST_CV]


def test_the_cv_test_set_is_four_games_in_three_whole_arenas_with_sealed_ids_and_pins():
    src = cv_source()
    games = src["games"]
    assert len(games) == 4 and len(set(games)) == 4
    assert R.list_hash(games) == src["sha256"]
    for key, seal in R.SEALED_MAPS.items():
        assert R.map_hash(src[key]) == src[seal], key
    assert sorted(src["video_ids"]) == sorted(games) == sorted(src["archive_git_blob_sha1"])
    assert len(set(src["video_ids"].values())) == 4
    assert all(re.fullmatch(r"[0-9a-f]{40}", p) for p in src["archive_git_blob_sha1"].values())
    homes = {h for hs in src["arena_homes"].values() for h in hs}
    assert len(src["arena_homes"]) == 3 and {g.split(".")[-1] for g in games} <= homes
    for g in games:                                   # inside the public SportVU window, regular season, not spent
        mm, dd, yyyy = g.split(".")[:3]
        assert "2015-10-27" <= f"{yyyy}-{mm}-{dd}" <= "2016-01-23", g
        assert not R.is_spent(g) and not g.endswith(".at.CLE"), g


def test_the_cv_test_games_are_in_no_other_source():
    data = committed()["sources"]
    cv = set(data[R.BROADCAST_CV]["games"])
    for name, src in data.items():
        if name != R.BROADCAST_CV:
            assert not cv & set(src["games"]), name


def test_no_experiment_code_names_a_cv_test_video():
    vids = set(cv_source()["video_ids"].values())
    problems = [f"{p.relative_to(REPO)} names the CV test video {v}" for p in experiment_files()
                for v in vids if v in p.read_text()]
    assert problems == []


def test_no_frozen_cv_registry_holds_a_cv_test_game_or_video():
    src = cv_source()
    ids = set(src["games"]) | set(src["video_ids"].values())
    problems = [f"{rel} holds {i}" for rel in FROZEN_CV_REGISTRIES for i in ids if i in (REPO / rel).read_text()]
    assert problems == []


def test_only_the_download_tool_reaches_the_cv_test_videos():
    problems = []
    for path in experiment_files():
        rel = str(path.relative_to(REPO))
        text = path.read_text()
        for name in CV_TEST_NAMES:
            if name in text and name not in CV_ALLOWED.get(rel, ()):
                problems.append(f"{rel} uses {name}")
    assert problems == []


def test_every_p8_tool_runs_the_cv_guard():
    """A P8 tool that touches games (tools/p8_*.py) must run refuse_cv_test; a module that only holds constants or
    pure functions says so with P8_NO_GAMES = True."""
    problems = []
    for path in sorted((REPO / "tools").glob("p8_*.py")):
        text = path.read_text()
        if "refuse_cv_test(" not in text and "P8_NO_GAMES = True" not in text:
            problems.append(str(path.relative_to(REPO)))
    assert problems == []


@pytest.fixture
def tiny_cv(tmp_path):
    games = ["12.01.2015.AAA.at.MIA", "12.02.2015.BBB.at.LAL"]
    vids = {games[0]: "vidA0000000", games[1]: "vidB0000000"}
    homes = {"Miami": ["MIA"], "Staples Center": ["LAC", "LAL"]}
    src = {"games": games, "sha256": R.list_hash(games), "video_ids": vids, "video_pins_sha256": R.map_hash(vids),
           "arena_homes": homes, "arena_homes_sha256": R.map_hash(homes)}
    path = tmp_path / "testset.json"
    path.write_text(json.dumps({"sources": {R.BROADCAST_CV: src}}))
    return path, games, vids


def test_refuse_cv_test_refuses_test_games_test_videos_and_whole_test_arenas(tiny_cv):
    path, games, vids = tiny_cv
    ok = ["11.27.2015.CHI.at.IND", "01.12.2016.BOS.at.NYK", "someVideoId1"]
    assert R.refuse_cv_test(ok, "t", path=path) == 3
    for bad in ([games[0]], [vids[games[1]]], ["01.01.2016.DAL.at.MIA"], ["11.11.2015.SAS.at.LAC"],
                ok + ["12.30.2015.UTA.at.LAL"]):
        with pytest.raises(SystemExit):
            R.refuse_cv_test(bad, "t", path=path)


def test_dev_games_of_the_cv_source_leave_out_whole_test_arenas(tiny_cv):
    path, games, _ = tiny_cv
    pool = games + ["01.01.2016.DAL.at.MIA", "11.11.2015.SAS.at.LAC", "11.27.2015.CHI.at.IND", "12.23.2015.NYK.at.CLE"]
    assert R.dev_games(pool, "t", R.BROADCAST_CV, path) == ["11.27.2015.CHI.at.IND"]


def test_an_edited_cv_video_id_or_arena_is_refused(tiny_cv):
    path, games, vids = tiny_cv
    for key, value in (("video_ids", {**vids, games[0]: "otherVideo1"}), ("arena_homes", {"Miami": ["MIA"]})):
        data = json.loads(path.read_text())
        data["sources"][R.BROADCAST_CV][key] = value
        path.write_text(json.dumps(data))
        with pytest.raises(SystemExit):
            R.refuse_cv_test(["11.27.2015.CHI.at.IND"], "t", path=path)
        tiny = json.loads(path.read_text())
        tiny["sources"][R.BROADCAST_CV][key] = vids if key == "video_ids" else {"Miami": ["MIA"], "Staples Center": ["LAC", "LAL"]}
        path.write_text(json.dumps(tiny))


def test_cv_test_videos_must_name_exactly_the_test_games(tiny_cv):
    path, games, vids = tiny_cv
    assert R.cv_test_videos(path) == vids
    data = json.loads(path.read_text())
    one = {games[0]: vids[games[0]]}
    data["sources"][R.BROADCAST_CV].update(video_ids=one, video_pins_sha256=R.map_hash(one))
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        R.cv_test_videos(path)
