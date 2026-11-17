"""The D71 test set (PLAN.md D71, D77): fixed and hashed, and read only by one final scoring call per experiment."""

import ast
import json
import re
from pathlib import Path

import pytest

from tools import research_testset as R

REPO = Path(__file__).resolve().parents[1]
GAME_ID = re.compile(r"\d{2}\.\d{2}\.\d{4}\.[A-Z]{3}\.at\.[A-Z]{3}")
BREF_ID = re.compile(r"\b\d{8}0[A-Z]{3}\b")        # basketball-reference game codes (the play-by-play source)
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
        for game in (set(GAME_ID.findall(text)) | set(BREF_ID.findall(text))) & test_ids:
            problems.append(f"{path.relative_to(REPO)} names the test game {game}")
    assert problems == []


# Names that reach the test games' data. Allowed only where they are defined or in the ingest tool.
TEST_DATA_NAMES = ("research_data_root", "test_features", "_research-data", "sportvu_untouched", "V3_RESEARCH_DATA",
                   "R._load", "import _load", "ingest_pins", "pbp_2025_26", "V3_RESEARCH_PBP", "scoring_data_root",
                   "file_pins", "fix_pbp")
ALLOWED = {"tools/research_ingest.py": TEST_DATA_NAMES,
           "tools/research_xfg.py": ("research_data_root", "test_features", "_research-data", "sportvu_untouched",
                                     "V3_RESEARCH_DATA"),
           "tools/research_pbp2526.py": ("_research-data", "pbp_2025_26")}


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


# ── the play-by-play source (2025-26 season; PLAN.md D78, D79, D83, D84) ─────────────────────────────

def test_the_committed_pbp_source_is_fixed_once_with_its_file_pins():
    data = committed()
    assert not (R.PBP in data["sources"] and R.PBP in data.get("pending", {})), "a source is pending or fixed, not both"
    if R.PBP not in data["sources"]:
        pytest.skip("the 2025-26 season is not fixed yet")
    src = data["sources"][R.PBP]
    assert all(R.pbp_season(g) == R.PBP_TEST_SEASON for g in src["games"])
    assert src["cut_off_date"].replace("-", "") == max(g[:8] for g in src["games"])
    assert sorted(src["files_sha256"]) == sorted(R.PBP_FILES)
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in src["files_sha256"].values())
    assert R.file_pins(R.PBP) == src["files_sha256"]
    fixed_on = data["fixed_later"][R.PBP]["fixed_on"]
    entries = json.loads(R.LEDGER_FILE.read_text()) if R.LEDGER_FILE.exists() else []
    assert all(e["started_at"][:10] >= fixed_on for e in entries if e["source"] == R.PBP), "scored before it was fixed"


OUTCOME_COLUMNS = ('"score"', '"points"', '"home_score"', '"away_score"', '"pts"', '"plus_minus"', '"margin"', '"kinds"')


def test_the_pbp_fetch_and_fix_read_no_outcome_column():
    import inspect
    texts = {"fix_pbp": inspect.getsource(R.fix_pbp),
             "tools/research_pbp2526.py": (REPO / "tools" / "research_pbp2526.py").read_text()}
    problems = [f"{where} names {c}" for where, t in texts.items() for c in OUTCOME_COLUMNS if c in t]
    assert problems == []
    for where, t in texts.items():
        for call in re.findall(r"read_table\([^)]*\)", t):
            assert "columns=" in call, f"{where}: {call} reads every column"


def _pbp_root(tmp_path, games, extra_scheduled=0):
    import pandas as pd
    root = tmp_path / "root"
    for rel in R.PBP_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
    dates = [f"{g[:4]}-{g[4:6]}-{g[6:8]}" for g in games]
    pd.DataFrame({"game_id": [g for g in games for _ in range(3)], "game_date": [d for d in dates for _ in range(3)],
                  "season": "2025-26"}).to_parquet(root / R.PBP_FILES[0])
    pd.DataFrame({"game_id": [g for g in games for _ in range(10)]}).to_parquet(root / R.PBP_FILES[1])
    pd.DataFrame({"game_id": games, "date": dates, "season_type": "Regular Season"}).to_parquet(root / R.PBP_FILES[2])
    (root / "fetch.json").write_text(json.dumps({"written": "2026-11-17T00:00:00+00:00",
                                                 "schedule": {"games": len(games) + extra_scheduled}}))
    return root


@pytest.fixture
def pbp_case(tmp_path):
    games = ["202510210LAL", "202510210OKC", "202606130SAS"]
    track = ["11.01.2015.AAA.at.BBB"]
    path = tmp_path / "testset.json"
    path.write_text(json.dumps({"sources": {R.TRACKING: {"games": track, "sha256": R.list_hash(track)}},
                                "pending": {R.PBP: "the 2025-26 season"}}))
    return path, games, _pbp_root(tmp_path, games), tmp_path / "ledger.json"


def test_fix_pbp_moves_the_season_from_pending_to_a_hashed_source_once(pbp_case):
    path, games, root, ledger = pbp_case
    src = R.fix_pbp(root, path, ledger)
    data = json.loads(path.read_text())
    assert R.PBP in data["sources"] and "pending" not in data
    assert src["games"] == sorted(games) and src["sha256"] == R.list_hash(games)
    assert src["cut_off_date"] == "2026-06-13" and src["counts"]["games_with_starters"] == 3
    assert R.file_pins(R.PBP, path) == {f: R.sha256_file(root / f) for f in R.PBP_FILES}
    with pytest.raises(SystemExit):
        R.fix_pbp(root, path, ledger)                    # a fixed source is never rewritten


def test_fix_pbp_is_refused_once_a_result_exists_or_a_game_is_missing(pbp_case, tmp_path):
    path, games, root, ledger = pbp_case
    ledger.write_text(json.dumps([{"experiment": "winprob-E1", "source": R.PBP, "started_at": "x"}]))
    with pytest.raises(SystemExit):
        R.fix_pbp(root, path, ledger)
    ledger.unlink()
    short = _pbp_root(tmp_path / "short", games, extra_scheduled=1)
    with pytest.raises(SystemExit):
        R.fix_pbp(short, path, ledger)
    assert R.fix_pbp(short, path, ledger, accept_missing=True)["counts"]["scheduled_without_playbyplay"] == 1


def test_pbp_test_data_is_read_only_inside_score_once(pbp_case, monkeypatch):
    path, games, root, ledger = pbp_case
    R.fix_pbp(root, path, ledger)
    monkeypatch.setenv(R.PBP_ROOT_ENV, str(root))
    with pytest.raises(SystemExit):
        R.scoring_data_root(R.PBP, path=path)          # outside score_once: refused
    seen = {}

    def scorer(g):
        seen["root"] = R.scoring_data_root(R.PBP, path=path)
        return {"n": len(g)}

    assert R.score_once("winprob-E0", R.PBP, scorer, path, ledger) == {"n": 3} and seen["root"] == root
    with pytest.raises(SystemExit):
        R.scoring_data_root(R.PBP, path=path)          # and refused again once the scorer returned
    (root / R.PBP_FILES[0]).write_bytes(b"edited")

    def tampered(_g):
        return R.scoring_data_root(R.PBP, path=path)

    with pytest.raises(SystemExit):
        R.score_once("winprob-E9", R.PBP, tampered, path, ledger)


def test_pbp_dev_games_never_return_the_test_season_or_the_spent_season(pbp_case):
    path, games, root, ledger = pbp_case
    R.fix_pbp(root, path, ledger)
    dev, spent, unlisted = "202401150BOS", "202501150BOS", "202604150MIA"     # 2023-24, 2024-25, 2025-26 Play-In
    assert R.dev_games(games + [dev, spent, unlisted, "201601150BOS"], "t", R.PBP, path) == [dev]
    for bad in ([games[0]], [spent], [unlisted]):
        with pytest.raises(SystemExit):
            R.refuse_test(bad, "t", R.PBP, path)
    with pytest.raises(SystemExit):
        R.refuse_test(["12.01.2015.AAA.at.BBB"], "t", R.PBP, path)   # not a game code: cannot be placed, refused
    assert R.refuse_test([dev], "t", R.PBP, path) == 1


def test_the_pbp_fetch_root_stays_out_of_the_repository_and_icloud(tmp_path, monkeypatch):
    from tools import research_pbp2526 as P
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    for bad in (REPO / "scratch", Path.home() / "Documents" / "x", tmp_path / "data_root" / "x"):
        with pytest.raises(SystemExit):
            P.guard_root(bad)
    assert P.guard_root(tmp_path / "ok") == (tmp_path / "ok").resolve()
    args = P.step_args("bref_seasons", tmp_path / "out", 40)
    assert args[:4] == ["--out", str(tmp_path / "out"), "--rpm", str(P.RPM_CAP)] and args[-1] == "--shots-seasons"
    assert P.RPM_CAP < 17
