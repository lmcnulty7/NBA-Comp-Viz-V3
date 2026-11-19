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
                   "file_pins", "fix_pbp", "DEFAULT_PBP_ROOT", "PBP_FILES", "PBP_ROOT_ENV")
ALLOWED = {"tools/research_ingest.py": TEST_DATA_NAMES,
           "tools/research_xfg.py": ("research_data_root", "test_features", "_research-data", "sportvu_untouched",
                                     "V3_RESEARCH_DATA"),
           "tools/research_pbp2526.py": ("_research-data", "pbp_2025_26"),
           # decisions-E1 imports the two from research_xfg and calls them only inside its scorer
           # (tests/test_research_decisions.py checks that)
           "tools/research_decisions.py": ("research_data_root", "test_features"),
           # winprob-E1 reads the 2025-26 files only inside the scorer it passes to score_once
           # (tests/test_research_winprob.py checks that)
           "tools/research_winprob.py": ("scoring_data_root",),
           # winprob-E2 likewise (tests/test_research_winprob_e2.py checks that)
           "tools/research_winprob_e2.py": ("scoring_data_root",),
           # lineups-E1 calls it only inside the scorer it passes to score_once (the AST rule below checks that)
           "tools/research_lineups.py": ("scoring_data_root",),
           # embeddings-E1 imports it from research_xfg and calls it only inside its scorer
           # (tests/test_research_embeddings.py checks that)
           "tools/research_embeddings.py": ("research_data_root",),
           # matchups-E1 imports the two from research_xfg and calls them only inside its scorer
           # (tests/test_research_matchups.py checks that)
           "tools/research_matchups.py": ("research_data_root", "test_features"),
           # matchups-E2 imports it from research_xfg and calls it only inside its scorer
           # (tests/test_research_matchups_e2.py checks that)
           "tools/research_matchups_e2.py": ("research_data_root",),
           "tools/research_lineups_e2.py": ("scoring_data_root",),
           # gravity-E1 imports the two from research_xfg and calls them only inside its scorer
           # (tests/test_research_gravity.py checks that)
           "tools/research_gravity.py": ("research_data_root", "test_features"),
           # ghosting-E1 imports it from research_xfg and calls it only inside its scorer
           # (tests/test_research_ghosting.py checks that)
           "tools/research_ghosting.py": ("research_data_root",)}


def name_problems(rel: str, text: str) -> list:
    return [f"{rel} uses {name}" for name in TEST_DATA_NAMES if name in text and name not in ALLOWED.get(rel, ())]


def test_only_named_places_reach_the_test_games_data():
    problems = []
    for path in experiment_files():
        problems += name_problems(str(path.relative_to(REPO)), path.read_text())
    assert problems == []


def test_a_direct_read_of_the_pbp_test_files_is_flagged():
    """The bypass of the review (M1): reading the pinned files through the door's constants, not score_once."""
    for snippet in ("df = pd.read_parquet(R.DEFAULT_PBP_ROOT / R.PBP_FILES[0])\n",
                    "root = os.environ[R.PBP_ROOT_ENV]\n"):
        assert name_problems("tools/research_winprob.py", snippet), snippet


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


LEDGER_REL = "reports/research/scoring_ledger.json"


def _entry_key(entry: dict) -> tuple:
    return entry["experiment"], entry["source"], entry["started_at"]


def ledger_growth_problems(parent: list, child: list, where: str) -> list:
    """How a child version of the ledger fails to grow from one parent version (PLAN.md D105): every parent entry is
    in the child with the same experiment, source and start time; an entry the parent holds as scored is
    byte-identical in the child; and the parent's entries keep their order in the child (a subsequence, so entries
    appended on another branch may sit between them)."""
    problems, at = [], {}
    for i, entry in enumerate(child):
        at.setdefault(_entry_key(entry), i)
    last = -1
    for entry in parent:
        key = _entry_key(entry)
        if key not in at:
            problems.append(f"{where}: the entry {key} was removed")
            continue
        if entry.get("status") == "scored" and json.dumps(entry, sort_keys=True) != json.dumps(child[at[key]],
                                                                                               sort_keys=True):
            problems.append(f"{where}: the scored entry {key} was edited")
        if at[key] < last:
            problems.append(f"{where}: the entry {key} moved before an entry that came before it")
        last = max(last, at[key])
    return problems


def ledger_history_problems(repo, rel: str = LEDGER_REL, working_tree: bool = True) -> list:
    """Every commit that touches the ledger, compared with each of its own parents (a merge with both of them), and
    the working tree with HEAD. --full-history keeps the merges that bring an entry in from one side. A parent
    without the file holds an empty ledger."""
    import subprocess

    def version(rev: str) -> list:
        shown = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=repo, capture_output=True, text=True)
        return json.loads(shown.stdout) if shown.returncode == 0 else []

    lines = subprocess.run(["git", "log", "--full-history", "--format=%H %P", "--", rel], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    problems = []
    for line in lines:
        sha, *parents = line.split()
        child = version(sha)
        for parent in parents:
            problems += ledger_growth_problems(version(parent), child, f"{sha[:8]} against its parent {parent[:8]}")
    path = Path(repo) / rel
    if working_tree and path.exists():
        problems += ledger_growth_problems(version("HEAD"), json.loads(path.read_text()), "the working tree")
    return problems


def test_the_scoring_ledger_only_grows_across_its_whole_history():
    """PLAN.md D105: each ledger-touching commit grows from each of its own parents, and the working tree from HEAD.
    On a linear history this is the old prefix check; it also holds across merges of parallel branches."""
    assert ledger_history_problems(REPO) == []


def _git(repo, *args) -> str:
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c",
                           "commit.gpgsign=false", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout


def _ledger_entry(experiment: str, status: str = "scored", summary=None) -> dict:
    return {"experiment": experiment, "source": "s", "sha256": "0" * 64, "games": 1,
            "started_at": f"2026-11-17T00:00:0{len(experiment) % 10}+00:00", "status": status,
            **({"scored_at": "2026-11-17T01:00:00+00:00", "summary": summary or {"n": 1}} if status == "scored" else {})}


def _commit_ledger(repo, entries: list, message: str) -> None:
    path = Path(repo) / LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n")
    _git(repo, "add", LEDGER_REL)
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def two_branch_ledger(tmp_path):
    """A ledger [a] on main; branch x appends b, branch y appends c; y is merged into x with [a, b, c]."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    a, b, c = _ledger_entry("a-E1"), _ledger_entry("b-E10"), _ledger_entry("c-E100")
    _commit_ledger(repo, [a], "a")
    _git(repo, "checkout", "-q", "-b", "y")
    _commit_ledger(repo, [a, c], "c")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "checkout", "-q", "-b", "x")
    _commit_ledger(repo, [a, b], "b")
    return repo, (a, b, c)


def _merge(repo, entries: list) -> None:
    import subprocess
    subprocess.run(["git", "merge", "-q", "--no-ff", "--no-commit", "y"], cwd=repo, capture_output=True, text=True)
    _commit_ledger(repo, entries, "merge y")


def test_a_ledger_grown_on_two_branches_and_merged_passes(two_branch_ledger):
    repo, (a, b, c) = two_branch_ledger
    _merge(repo, [a, b, c])
    assert ledger_history_problems(repo) == []
    assert len(_git(repo, "log", "--full-history", "--format=%H", "--", LEDGER_REL).split()) == 4


def test_a_merge_that_drops_an_entry_fails(two_branch_ledger):
    repo, (a, b, c) = two_branch_ledger
    _merge(repo, [a, b])
    assert any("was removed" in p for p in ledger_history_problems(repo))


def test_a_merge_that_edits_a_scored_entry_fails(two_branch_ledger):
    repo, (a, b, c) = two_branch_ledger
    _merge(repo, [a, b, {**c, "summary": {"n": 2}}])
    assert any("was edited" in p for p in ledger_history_problems(repo))


def test_a_merge_that_reorders_entries_fails(two_branch_ledger):
    repo, (a, b, c) = two_branch_ledger
    _merge(repo, [b, a, c])
    assert any("moved before" in p for p in ledger_history_problems(repo))


def test_the_working_tree_is_checked_against_head(two_branch_ledger):
    repo, (a, b, c) = two_branch_ledger
    _merge(repo, [a, b, c])
    (Path(repo) / LEDGER_REL).write_text(json.dumps([a, c], indent=2) + "\n")
    assert any("the working tree" in p and "was removed" in p for p in ledger_history_problems(repo))
    started = _ledger_entry("d-E1", status="started")
    finished = {**started, "status": "scored", "scored_at": "x", "summary": {"n": 1}}
    assert ledger_growth_problems([a, started], [a, finished], "w") == []       # a started entry may be completed


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
# Strings that name the sealed test-video root: allowed only in the download tool, which runs no model on them.
CV_TEST_STRINGS = ("V3_P8_TEST_ROOT", "_p8-cvtest", "p8_cv_test")
# Door functions that reach the test videos: called only in the download tool, or inside the scorer function a module
# passes to score_once (the AST rule below, like test_features in research_xfg).
CV_DOORS = ("cv_test_videos", "cv_test_dir", "cv_test_video")
CV_DOWNLOAD_TOOL = "tools/p8_fetch.py"
# Registries the frozen CV code reads (sportvu/, harvest_driver, clock_reader): a CV test game or video id listed in
# one of them could be picked up by a frozen tool, so none may hold one.
FROZEN_CV_REGISTRIES = ("data/harvest/games.json", "sportvu/manifest_overrides.json", "sportvu/splits.json")
CV_SOURCE_SINCE = "1670e3d6"          # the commit that hashed the CV test set


def cv_source():
    return committed()["sources"][R.BROADCAST_CV]


def tracked_code():
    """Every git-tracked .py, .ipynb and .sh file outside tests/ and the door itself."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.py", "*.ipynb", "*.sh"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.split()
    return sorted(REPO / p for p in out if not p.startswith("tests/") and REPO / p != DOOR and (REPO / p).is_file())


def test_the_cv_test_set_is_four_games_in_three_whole_arenas_with_sealed_ids_and_pins():
    src = cv_source()
    games = src["games"]
    assert len(games) == 4 and len(set(games)) == 4
    assert R.list_hash(games) == src["sha256"]
    for key, seal in R.SEALED_MAPS.items():
        if key in R.REQUIRED_KEYS[R.BROADCAST_CV]:       # files_sha256 seals the pbp source's files, not this one
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


def test_the_cv_source_is_byte_identical_in_every_committed_version_since_it_was_hashed():
    import subprocess
    rel = "reports/research/testset.json"
    # the commit that hashed it and every commit descending from it (a merged branch that forked earlier never held
    # the source, so --ancestry-path leaves its commits out)
    shas = [CV_SOURCE_SINCE] + subprocess.run(["git", "log", "--ancestry-path", "--format=%H",
                                               f"{CV_SOURCE_SINCE}..HEAD", "--", rel], cwd=REPO,
                                              capture_output=True, text=True, check=True).stdout.split()
    now = json.dumps(cv_source(), sort_keys=True)
    assert shas, "the commit that hashed the CV test set is in this history"
    for sha in shas:
        shown = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True)
        old = json.loads(shown.stdout)["sources"][R.BROADCAST_CV]
        assert json.dumps(old, sort_keys=True) == now, f"the CV test source changed in {sha[:8]}"


def test_the_cv_test_games_are_in_no_other_source():
    data = committed()["sources"]
    cv = set(data[R.BROADCAST_CV]["games"])
    for name, src in data.items():
        if name != R.BROADCAST_CV:
            assert not cv & set(src["games"]), name


def test_no_tracked_code_names_a_cv_test_game_or_video():
    src = cv_source()
    ids = list(src["games"]) + list(src["video_ids"].values())
    problems = [f"{p.relative_to(REPO)} names {i}" for p in tracked_code() for i in ids
                if i.lower() in p.read_text(errors="ignore").lower()]
    assert problems == []


def test_no_frozen_cv_registry_holds_a_cv_test_game_or_video():
    src = cv_source()
    ids = set(src["games"]) | set(src["video_ids"].values())
    problems = [f"{rel} holds {i}" for rel in FROZEN_CV_REGISTRIES for i in ids if i in (REPO / rel).read_text()]
    assert problems == []


def test_only_the_download_tool_names_the_sealed_test_root():
    problems = [f"{p.relative_to(REPO)} uses {n}" for p in tracked_code() for n in CV_TEST_STRINGS
                if n in p.read_text(errors="ignore") and str(p.relative_to(REPO)) != CV_DOWNLOAD_TOOL]
    assert problems == []


def door_call_problems(tree, rel: str) -> list:
    """Calls of a CV door outside the download tool that are not inside a function passed to score_once."""
    if rel == CV_DOWNLOAD_TOOL:
        return []
    scorers = {a.id for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", getattr(n.func, "id", None)) == "score_once"
               for a in list(n.args) + [k.value for k in n.keywords] if isinstance(a, ast.Name)}
    inside = {id(c) for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
              and fn.name in scorers for c in ast.walk(fn)}
    problems = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in CV_DOORS \
                and id(n) not in inside:
            problems.append(f"{rel}:{n.lineno} calls {getattr(n.func, 'attr', getattr(n.func, 'id', None))}")
    return problems


def test_cv_doors_are_called_only_by_the_download_tool_or_inside_a_score_once_scorer():
    problems = []
    for p in tracked_code():
        if p.suffix != ".py":
            text = p.read_text(errors="ignore")
            problems += [f"{p.relative_to(REPO)} names {d}" for d in CV_DOORS if d in text]
            continue
        problems += door_call_problems(ast.parse(p.read_text()), str(p.relative_to(REPO)))
    assert problems == []


def test_the_door_call_rule_allows_a_scorer_and_refuses_anything_else():
    ok = ast.parse("def scorer(games):\n    return [cv_test_video(g) for g in games]\n"
                   "def run():\n    return score_once('x', 'src', scorer)\n")
    bad = ast.parse("def prep():\n    return cv_test_video('g')\n"
                    "def scorer(games):\n    return games\n"
                    "def run():\n    return score_once('x', 'src', scorer)\n")
    loose = ast.parse("x = cv_test_dir()\n")
    assert door_call_problems(ok, "tools/x.py") == []
    assert len(door_call_problems(bad, "tools/x.py")) == 1 and len(door_call_problems(loose, "tools/x.py")) == 1
    assert door_call_problems(loose, CV_DOWNLOAD_TOOL) == []


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
    pins = {games[0]: "a" * 40, games[1]: "b" * 40}
    src = {"games": games, "sha256": R.list_hash(games), "video_ids": vids, "video_pins_sha256": R.map_hash(vids),
           "arena_homes": homes, "arena_homes_sha256": R.map_hash(homes),
           "archive_git_blob_sha1": pins, "pins_sha256": R.map_hash(pins)}
    path = tmp_path / "testset.json"
    path.write_text(json.dumps({"sources": {R.BROADCAST_CV: src}}))
    return path, games, vids


def test_refuse_cv_test_refuses_test_games_test_videos_and_whole_test_arenas(tiny_cv):
    path, games, vids = tiny_cv
    ok = ["11.27.2015.CHI.at.IND", "01.12.2016.BOS.at.NYK", "someVideoId1", "gsw_mia_2017", "nyk_bos_2016",
          "/w/video/okc_sac_2015_s00.mp4", "replace_me"]
    assert R.refuse_cv_test(ok, "t", path=path) == len(ok)
    aliases = [[games[0]], [vids[games[1]]], ["01.01.2016.DAL.at.MIA"], ["11.11.2015.SAS.at.LAC"],
               ok + ["12.30.2015.UTA.at.LAL"],
               [f"https://www.youtube.com/watch?v={vids[games[0]]}"],          # a watch URL
               [f"/data/video/{games[1]}.mp4"],                                # an .mp4 path
               [f"{games[0]}_s00"], ["mia_nop_2015_s00"],                     # a section stem
               ["mia_nop_2015"], ["/x/lal_aaa_2015.mp4"], ["lac_bbb_2016"],    # home-first tags
               [games[0].lower()], [vids[games[0]].lower()], ["01.01.2016.dal.at.mia"]]   # lower case
    for bad in aliases:
        with pytest.raises(SystemExit):
            R.refuse_cv_test(bad, "t", path=path)
    with pytest.raises(SystemExit):
        R.refuse_cv_test(["por_okc_2016"], "t", path=path, homes=["LAL"])    # a registry entry's own home


def test_the_cv_source_must_carry_every_sealed_key(tiny_cv):
    path, _, _ = tiny_cv
    for key in R.REQUIRED_KEYS[R.BROADCAST_CV]:
        data = json.loads(path.read_text())
        saved = data["sources"][R.BROADCAST_CV].pop(key)
        p2 = path.with_name("t_" + key + ".json")
        p2.write_text(json.dumps(data))
        with pytest.raises(SystemExit):
            R.refuse_cv_test(["11.27.2015.CHI.at.IND"], "t", path=p2)
        assert saved


def test_dev_games_of_the_cv_source_leave_out_whole_test_arenas(tiny_cv):
    path, games, _ = tiny_cv
    pool = games + ["01.01.2016.DAL.at.MIA", "11.11.2015.SAS.at.LAC", "11.27.2015.CHI.at.IND", "12.23.2015.NYK.at.CLE"]
    assert R.dev_games(pool, "t", R.BROADCAST_CV, path) == ["11.27.2015.CHI.at.IND"]


def test_an_edited_cv_video_id_or_arena_is_refused(tiny_cv):
    path, games, vids = tiny_cv
    good = path.read_text()
    for key, value in (("video_ids", {**vids, games[0]: "otherVideo1"}), ("arena_homes", {"Miami": ["MIA"]})):
        data = json.loads(good)
        data["sources"][R.BROADCAST_CV][key] = value
        path.write_text(json.dumps(data))
        with pytest.raises(SystemExit):
            R.refuse_cv_test(["11.27.2015.CHI.at.IND"], "t", path=path)
    path.write_text(good)


def test_cv_test_videos_must_name_exactly_the_test_games(tiny_cv):
    path, games, vids = tiny_cv
    assert R.cv_test_videos(path) == vids
    data = json.loads(path.read_text())
    one = {games[0]: vids[games[0]]}
    data["sources"][R.BROADCAST_CV].update(video_ids=one, video_pins_sha256=R.map_hash(one))
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        R.cv_test_videos(path)


def test_the_test_video_root_is_outside_the_p8_workspace(monkeypatch, tmp_path):
    from tools import p8_fetch
    d = p8_fetch.cv_test_dir()
    from tools.p8_common import p8_root
    assert p8_root().resolve() not in d.parents and d.parent.resolve() not in p8_root().resolve().parents
    monkeypatch.setenv("V3_P8_ROOT", str(tmp_path / "w"))
    monkeypatch.setenv("V3_P8_TEST_ROOT", str(tmp_path / "w" / "t"))
    with pytest.raises(SystemExit):
        p8_fetch.cv_test_dir()


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
        R.scoring_data_root(R.PBP)                     # outside score_once: refused
    seen = {}

    def scorer(g):
        seen["root"] = R.scoring_data_root(R.PBP)
        return {"n": len(g)}

    assert R.score_once("winprob-E0", R.PBP, scorer, path, ledger) == {"n": 3} and seen["root"] == root
    with pytest.raises(SystemExit):
        R.scoring_data_root(R.PBP)                     # and refused again once the scorer returned


def test_score_once_hashes_the_pinned_files_before_the_scorer_runs(pbp_case):
    path, games, root, ledger = pbp_case
    R.fix_pbp(root, path, ledger)
    (root / R.PBP_FILES[0]).write_bytes(b"edited")
    ran = []
    with pytest.raises(SystemExit):
        R.score_once("winprob-E9", R.PBP, lambda g: ran.append(1) or {}, path, ledger, data_root=root)
    assert ran == [] and not ledger.exists()           # refused before the scorer and before the ledger entry
    with pytest.raises(SystemExit):
        R.score_once("winprob-E9", R.PBP, lambda g: ran.append(1) or {}, path, ledger, data_root=root / "missing")
    assert ran == []


def scoring_root_problems(tree, rel: str) -> list:
    """Calls of scoring_data_root that are not inside a function passed to score_once (as a name argument)."""
    scorers = {a.id for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", getattr(n.func, "id", None)) == "score_once"
               for a in list(n.args) + [k.value for k in n.keywords] if isinstance(a, ast.Name)}
    inside = {id(c) for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
              and fn.name in scorers for c in ast.walk(fn)}
    return [f"{rel}:{n.lineno} calls scoring_data_root outside a score_once scorer" for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "scoring_data_root"
            and id(n) not in inside]


def test_scoring_data_root_is_called_only_inside_a_function_passed_to_score_once():
    problems = []
    for path in tracked_code():
        if path.suffix == ".py":
            problems += scoring_root_problems(ast.parse(path.read_text()), str(path.relative_to(REPO)))
    assert problems == []
    good = ("def scorer(games):\n    root = R.scoring_data_root(R.PBP)\n    return {}\n"
            "R.score_once('w-E1', R.PBP, scorer)\n")
    bad = ("def load():\n    return R.scoring_data_root(R.PBP)\n"
           "def scorer(games):\n    return {}\nR.score_once('w-E1', R.PBP, scorer)\nroot = R.scoring_data_root(R.PBP)\n")
    assert scoring_root_problems(ast.parse(good), "good.py") == []
    assert len(scoring_root_problems(ast.parse(bad), "bad.py")) == 2


PBP_SOURCE_SINCE = "d86a9505"          # the commit that fixed the 2025-26 season


def test_the_pbp_source_is_byte_identical_in_every_commit_since_it_first_appeared():
    import subprocess
    rel = "reports/research/testset.json"
    # the commit that fixed it and every later commit descending from it that touched the file
    shas = [PBP_SOURCE_SINCE] + subprocess.run(["git", "log", "--ancestry-path", "--format=%H",
                                                f"{PBP_SOURCE_SINCE}..HEAD", "--", rel], cwd=REPO,
                                               capture_output=True, text=True, check=True).stdout.split()
    texts = [(sha[:8], subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=REPO, capture_output=True, text=True,
                                      check=True).stdout) for sha in shas]
    texts.append(("working tree", R.TESTSET_FILE.read_text()))
    first = json.dumps(json.loads(texts[0][1])["sources"][R.PBP], indent=2).replace("\n", "\n    ")
    for sha, text in texts:
        assert f'    "{R.PBP}": {first}' in text, f"the pbp source is not byte-identical in {sha}"


@pytest.mark.parametrize("code,season", [("202107200MIL", "2020-21"), ("202008150LAL", "2019-20"),
                                         ("202010110LAL", "2019-20"), ("201910220TOR", "2019-20"),
                                         ("202506220OKC", "2024-25"), ("202606130SAS", "2025-26"),
                                         ("202510210LAL", "2025-26")])
def test_pbp_season_maps_by_season_date_ranges(code, season):
    assert R.pbp_season(code) == season


def test_pbp_season_refuses_a_code_dated_between_seasons():
    for code in ("202108150MIL", "202011150LAL", "201601150BOS0"):
        with pytest.raises(SystemExit):
            R.pbp_season(code)


def test_the_season_date_table_matches_every_schedule_table_found():
    """PBP_SEASON_DATES against the frozen schedule tables of DATA_ROOT and the 2025-26 table of the fetch root:
    game ids and dates only."""
    import os
    import pyarrow.parquet as pq
    dirs = [Path(os.environ["DATA_ROOT"]) / "collect" / "bref_schedule" / "games"] if os.environ.get("DATA_ROOT") else []
    dirs.append(Path(os.environ.get(R.PBP_ROOT_ENV) or R.DEFAULT_PBP_ROOT) / "collect" / "bref_schedule" / "games")
    tables = [f for d in dirs if d.exists() for f in sorted(d.glob("*.parquet"))]
    if not tables:
        pytest.skip("no schedule table on this machine")
    for f in tables:
        t = pq.read_table(f, columns=["game_id", "date"]).to_pandas()
        assert (str(t["date"].min()), str(t["date"].max())) == R.PBP_SEASON_DATES[f.stem], f.stem
        assert {R.pbp_season(g) for g in t["game_id"]} == {f.stem}, f.stem


def test_pbp_dev_games_never_return_the_test_season_or_the_spent_season(pbp_case):
    path, games, root, ledger = pbp_case
    R.fix_pbp(root, path, ledger)
    dev, spent, unlisted = "202401150BOS", "202501150BOS", "202604150MIA"     # 2023-24, 2024-25, 2025-26 Play-In
    finals = "202107200MIL"                                                   # 2020-21, played in July 2021
    assert R.dev_games(games + [dev, spent, unlisted, finals, "201601150BOS", "202008150LAL"], "t", R.PBP,
                       path) == sorted([dev, finals])
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


# ── D89: 2024-25 results as winprob-E1's time-forward Elo state, through one door function ─────────────────

def test_only_the_elo_state_builder_names_the_2024_25_door():
    rel_ok = R.ELO_STATE_MODULE.replace(".", "/") + ".py"
    problems = [str(p.relative_to(REPO)) for p in tracked_code()
                if "elo_state_games_2024_25" in p.read_text(errors="ignore") and str(p.relative_to(REPO)) != rel_ok]
    assert problems == []


def _call_elo_door_as(module_name: str, data_root):
    scope = {"__name__": module_name, "R": R, "root": data_root}
    exec("out = R.elo_state_games_2024_25(root)", scope)
    return scope["out"]


def test_the_elo_door_refuses_any_other_caller_and_returns_ids_dates_teams_and_scores_only(tmp_path):
    import pandas as pd
    table = tmp_path / "collect" / "bref_schedule" / "games" / "2024-25.parquet"
    table.parent.mkdir(parents=True)
    pd.DataFrame({"game_id": ["202410220BOS", "202410220LAL", "202504150GSW"],
                  "date": ["2024-10-22", "2024-10-22", "2025-04-15"], "home": ["BOS", "LAL", "GSW"],
                  "away": ["NYK", "MIN", "MEM"], "home_score": pd.array([1, 2, None], dtype="Int32"),
                  "away_score": pd.array([3, 4, None], dtype="Int32"), "season_type": "Regular Season",
                  "attendance": 1.0, "notes": "x"}).to_parquet(table)
    for who in ("tools.research_winprob", "__main__", "tests.test_research_testset"):
        with pytest.raises(SystemExit):
            _call_elo_door_as(who, tmp_path)
    with pytest.raises(SystemExit):
        R.elo_state_games_2024_25(tmp_path)
    out = _call_elo_door_as(R.ELO_STATE_MODULE, tmp_path)
    assert list(out.columns) == list(R.ELO_STATE_COLUMNS)
    assert list(out["game_id"]) == ["202410220BOS", "202410220LAL"]          # played games only, by date and id
    bad = tmp_path / "bad"
    (bad / "collect" / "bref_schedule" / "games").mkdir(parents=True)
    pd.read_parquet(table).assign(game_id=["202410220BOS", "202310240BOS", "202504150GSW"]).to_parquet(
        bad / "collect" / "bref_schedule" / "games" / "2024-25.parquet")
    with pytest.raises(SystemExit):
        _call_elo_door_as(R.ELO_STATE_MODULE, bad)                               # a game outside 2024-25


def test_the_elo_door_function_computes_nothing():
    import inspect
    src = inspect.getsource(R.elo_state_games_2024_25)
    for op in (".mean(", ".sum(", ".std(", ".var(", ".diff(", ".corr(", "home_score -", "away_score -", "print("):
        assert op not in src, op


def test_2024_25_stays_out_of_development_after_d89(pbp_case):
    path, games, root, ledger = pbp_case
    R.fix_pbp(root, path, ledger)
    spent = ["202410220BOS", "202504150GSW", "202506220OKC"]
    assert R.dev_games(spent + ["202401150BOS"], "t", R.PBP, path) == ["202401150BOS"]
    for g in spent:
        with pytest.raises(SystemExit):
            R.refuse_test([g], "t", R.PBP, path)
