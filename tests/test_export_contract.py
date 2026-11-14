"""The site data contract (export/schemas.py), its synthetic sample, the heavy-job lock, and the rules every file
under export/ must keep: no fit call, no private string, no em or en dash."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from export import heavy
from export import sample
from export import schemas as S
from tools import frozen_common as F

REPO = Path(__file__).resolve().parents[1]
EXPORT_FILES = sorted((REPO / "export").rglob("*.py"))
TOOLS_FILES = sorted((REPO / "tools").rglob("*.py"))
OWNED_TEXT = EXPORT_FILES + TOOLS_FILES + [REPO / "CLAUDE.md", REPO / ".claude" / "agents" / "ml-reviewer.md",
                                           REPO / ".claude" / "agents" / "verifier.md", Path(__file__)]
# Calls and imports that fit, tune or calibrate a model. export/ may hold none of them (PLAN.md M1 DoD).
FIT_PATTERNS = (r"\.fit\(", r"\.fit_transform\(", r"\.partial_fit\(", r"\.fit_predict\(", r"^\s*(from|import)\s+sklearn",
                r"^\s*(from|import)\s+torch", r"\.backward\(", r"optimizer\.step\(", r"GridSearchCV", r"cross_val_",
                # a refit through frozen code: fit helpers, fold loops, held-out scoring, a lane's train module
                r"\bfit_(?!threads\b)\w+\(", r"out_of_fold\(", r"heldout_once\(", r"polyfit", r"curve_fit", r"minimize\(",
                r"^\s*from\s+analytics\.\w+\.train\s+import", r"^\s*import\s+analytics\.\w+\.train",
                r"^\s*from\s+analytics\.\w+\s+import\s+.*\btrain\b")
DASHES = (chr(0x2013), chr(0x2014))


@pytest.fixture(scope="module")
def sample_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample_root")
    sample.write_sample(root)
    return root


def test_schema_is_deterministic_and_names_every_file_model():
    first, second = S.schema_text(), S.schema_text()
    assert first == second
    doc = json.loads(first)
    assert doc["x-schema-version"] == S.SCHEMA_VERSION
    assert set(doc["properties"]) == set(S.MODELS)
    assert set(doc["x-files"].values()) == set(S.MODELS)
    for name in S.MODELS:
        assert doc["$defs"][name]["additionalProperties"] is False


def test_every_file_model_carries_the_schema_version():
    for name, model in S.MODELS.items():
        field = model.model_fields.get("schema_version")
        assert field is not None, name
        assert field.annotation == S.SchemaVersion, name


@pytest.mark.parametrize("path, expected", [
    ("manifest.json", "Manifest"),
    ("players.json", "PlayersFile"),
    ("scorecard.json", "Scorecard"),
    ("results.json", "Results"),
    ("demo/swap.json", "DemoSwap"),
    ("demo/frames.json", "DemoFrames"),
    ("games/2015-12-16-phx-at-gsw/possessions.json", "PossessionIndex"),
    ("games/2015-12-16-phx-at-gsw/42.json", "Possession"),
    ("games/2015-12-16-phx-at-gsw/42 2.json", None),
    ("games/12.16.2015.PHX.at.GSW/42.json", None),
    ("games/possessions.json", None),
    ("notes.json", None),
])
def test_model_for_maps_paths(path, expected):
    model = S.model_for(path)
    assert (model.__name__ if model else None) == expected


def test_estimate_has_an_interval_or_is_a_point_estimate():
    S.Estimate(value=1.0, point_estimate=True)
    S.Estimate(value=1.0, lo=0.5, hi=1.5, interval="95% bootstrap by game", point_estimate=False)
    for bad in (dict(value=1.0, lo=0.5, hi=1.5, interval="x", point_estimate=True),
                dict(value=1.0, point_estimate=False),
                dict(value=1.0, lo=0.5, point_estimate=False),
                dict(value=1.0, lo=0.5, hi=1.5, point_estimate=False),
                dict(value=1.0, lo=2.0, hi=1.5, interval="x", point_estimate=False)):
        with pytest.raises(ValueError):
            S.Estimate(**bad)


@pytest.mark.parametrize("text, private", [
    ("reports/xfg.md", False),
    ("$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet", False),
    ("12.16.2015.PHX.at.GSW", False),
    ("p50 1.92 ft at 5 Hz", False),
    ("/Users/someone/project", True),
    ("/home/someone", True),
    ("/private/tmp/x", True),
    ("~/Developer/x", True),
    ("someone@example.com", True),
    ("Library/CloudStorage/GoogleDrive-account/My Drive", True),
    # a path is private wherever it sits in a longer path
    ("/System/Volumes/Data/Users/someone/project", True),
    ("/data/home/someone/run", True),
    ("/mnt/c/Users/someone", True),
    ("../../Users/someone/x", True),
    ("C:\\Users\\someone", True),
    ("/Volumes/Ext/x", True),
    ("/content/drive/MyDrive/nba/x.parquet", True),
    ("/root/x", True),
    ("/opt/anaconda3/lib", True),
    ("\\/Users\\/someone", True),
    ("see file:///Users/someone/x", True),
    # a web address is not a local path
    ("https://example.org/home/page", False),
    ("https://github.com/owner/repo/blob/main/reports/xfg.md", False),
])
def test_find_private(text, private):
    assert bool(S.find_private(text)) is private


def test_sample_root_validates(sample_root):
    assert S.validate_tree(sample_root) == []
    manifest = S.Manifest.model_validate_json((sample_root / "manifest.json").read_text())
    assert manifest.synthetic is True
    listed = {entry.path for entry in manifest.files}
    on_disk = {p.relative_to(sample_root).as_posix() for p in sample_root.rglob("*.json")}
    assert listed == on_disk - {"manifest.json"}
    assert all(entry.sources for entry in manifest.files)


def test_sample_is_deterministic(sample_root, tmp_path):
    sample.write_sample(tmp_path)
    for path in sorted(sample_root.rglob("*.json")):
        rel = path.relative_to(sample_root)
        assert (tmp_path / rel).read_bytes() == path.read_bytes(), rel


def test_validate_tree_reports_what_is_wrong(sample_root, tmp_path):
    root = tmp_path / "broken"
    sample.write_sample(root)
    (root / "results.json").unlink()
    (root / "games" / "fixture-game-a" / "1 2.json").write_text("{}")
    index_path = root / "games" / "fixture-game-a" / "possessions.json"
    doc = json.loads(index_path.read_text())
    doc["unknown_field"] = 1
    index_path.write_text(json.dumps(doc))
    scorecard_path = root / "scorecard.json"
    doc = json.loads(scorecard_path.read_text())
    doc["decision"] = "written at /Users/someone/project"
    scorecard_path.write_text(json.dumps(doc))
    problems = "\n".join(S.validate_tree(root))
    assert "results.json: missing" in problems
    assert "1 2.json: not a file the contract knows" in problems
    assert "possessions.json: ValidationError" in problems
    assert "scorecard.json: private string '/Users/'" in problems
    with pytest.raises(ValueError):
        S.validate_tree(root, allow_missing=("results.json",))


def test_only_the_demo_files_may_be_missing(tmp_path):
    root = tmp_path / "staged"
    sample.write_sample(root)
    (root / "demo" / "swap.json").unlink()
    assert any("demo/swap.json: missing" in line for line in S.validate_tree(root))
    staged = S.validate_tree(root, allow_missing=S.STAGED_FILES)
    assert not any("demo/swap.json: missing" in line for line in staged)
    assert any("demo/swap.json is listed and not on disk" in line for line in staged)
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = [line for line in S.validate_tree(empty, allow_missing=S.STAGED_FILES) if line.endswith(": missing")]
    fixed = [pattern for pattern, _ in S.FILES if "<" not in pattern]
    assert len(missing) == len(fixed) - len(S.STAGED_FILES)


def test_possession_tracks_must_be_aligned(sample_root):
    path = sample_root / "games" / "fixture-game-a" / "1.json"
    doc = json.loads(path.read_text())
    doc["players"][0]["x"] = doc["players"][0]["x"][:-1]
    with pytest.raises(ValueError):
        S.Possession.model_validate(doc)


def test_ghost_label_is_fixed(sample_root):
    path = sample_root / "games" / "fixture-game-a" / "1.json"
    doc = json.loads(path.read_text())
    doc["ghosts"]["label"] = "best defense"
    with pytest.raises(ValueError):
        S.Possession.model_validate(doc)


def test_export_holds_no_fit_call():
    assert EXPORT_FILES, "export/ has no Python files"
    hits = []
    for path in EXPORT_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(re.search(pattern, line) for pattern in FIT_PATTERNS):
                hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == [], "export/ must not fit, tune or calibrate anything:\n" + "\n".join(hits)


def test_owned_files_hold_no_dash_and_no_private_string():
    problems = []
    for path in OWNED_TEXT:
        text = path.read_text(encoding="utf-8")
        if any(dash in text for dash in DASHES):
            problems.append(f"{path.relative_to(REPO)}: em or en dash")
        if path.suffix == ".py" and path.relative_to(REPO).parts[0] in ("export", "tools"):
            # The contract module lists the private patterns themselves; everything else in export/ must be clean.
            body = text if path.name != "schemas.py" else ""
            for found in S.find_private(body):
                problems.append(f"{path.relative_to(REPO)}: private string {found!r}")
    assert problems == []


def test_heavy_job_lock_excludes_a_second_job(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    quiet = lambda message: None
    probe = [sys.executable, "-m", "export.heavy", "--name", "probe", "--no-wait", "--", sys.executable, "-c", "pass"]
    with heavy.heavy_job("first", log=quiet):
        # An unrelated process: it does not inherit the holder's marker, so it must be refused.
        unrelated = {k: v for k, v in os.environ.items() if k != heavy.HELD_ENV}
        blocked = subprocess.run(probe, cwd=REPO, capture_output=True, text=True, env=unrelated)
        assert blocked.returncode != 0
        assert "another heavy job is running" in blocked.stderr
    free = subprocess.run(probe, cwd=REPO, capture_output=True, text=True)
    assert free.returncode == 0, free.stderr


def test_contract_refuses_nan_and_infinity():
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError):
            S.Estimate.model_validate_json('{"value": %s, "point_estimate": true}' % token)


def test_point_estimate_names_no_interval_method():
    with pytest.raises(ValueError):
        S.Estimate(value=1.0, interval="95% bootstrap", point_estimate=True)


def test_shown_needs_the_minimum_sample_and_a_value():
    est = S.Estimate(value=1.0, lo=0.0, hi=2.0, interval="95% bootstrap over the player's games", point_estimate=False)
    S.ShotMaking(shown=True, attempts=100, min_attempts=100, over_expected_per_100=est)
    S.ShotMaking(shown=False, attempts=40, min_attempts=100)
    for bad in (dict(shown=True, attempts=99, min_attempts=100, over_expected_per_100=est),
                dict(shown=True, attempts=200, min_attempts=100),
                dict(shown=False, attempts=40, min_attempts=100, over_expected_per_100=est)):
        with pytest.raises(ValueError):
            S.ShotMaking(**bad)
    with pytest.raises(ValueError):
        S.Similar(shown=True, available=False, possessions=900, min_possessions=300,
                  neighbours=[S.SimilarPlayer(rank=1, player_id=1, name="x", distance=0.1)])
    with pytest.raises(ValueError):
        S.AbsenceRisk(shown=True, games=20, min_games=10, absent=1, kind="observed_only", observed_rate=est,
                      model_probability=est)


def test_paired_difference_and_metric_value_keep_their_flags_honest():
    S.PairedDifference(game="g", metric="m", baseline="a", build="b", diff=0.1, lo=0.05, hi=0.2, excludes_zero=True)
    S.PairedDifference(game="g", metric="m", baseline="a", build="b", diff=0.1, lo=-0.05, hi=0.2, excludes_zero=False)
    for bad in (dict(diff=0.1, lo=-0.05, hi=0.2, excludes_zero=True), dict(diff=0.1, lo=0.3, hi=0.2, excludes_zero=True)):
        with pytest.raises(ValueError):
            S.PairedDifference(game="g", metric="m", baseline="a", build="b", **bad)
    S.MetricValue(text="x 1.0 (point estimate)", value=1.0, point_estimate=True)
    S.MetricValue(text="x 1.0 [0.9, 1.1]", value=1.0, lo=0.9, hi=1.1, point_estimate=False)
    for bad in (dict(value=1.0, lo=0.9, point_estimate=False), dict(value=1.0, point_estimate=False),
                dict(value=1.0, lo=0.9, hi=1.1, point_estimate=True)):
        with pytest.raises(ValueError):
            S.MetricValue(text="x", **bad)


def test_validate_tree_checks_paths_the_manifest_and_ignores_hidden_files(sample_root, tmp_path):
    root = tmp_path / "moved"
    sample.write_sample(root)
    (root / ".DS_Store").write_text("finder")
    assert S.validate_tree(root) == []
    (root / ".notes.json").write_text('{"path": "/Users/someone"}')
    (root / "games" / ".cache").mkdir()
    (root / "games" / ".cache" / "1.json").write_text("{}")
    hidden = "\n".join(S.validate_tree(root))
    assert ".notes.json: not a file the contract knows" in hidden
    assert "games/.cache/1.json: not a file the contract knows" in hidden
    (root / ".notes.json").unlink()
    (root / "games" / ".cache" / "1.json").unlink()
    (root / "games" / ".cache").rmdir()
    game = root / "games" / "fixture-game-a"
    one, two = (game / "1.json").read_text(), (game / "2.json").read_text()
    (game / "1.json").write_text(two)
    (game / "2.json").write_text(one)
    problems = "\n".join(S.validate_tree(root))
    assert "games/fixture-game-a/1.json: holds possession 2" in problems
    assert "does not match the files on disk" in problems
    (game / "9.json").write_text(one.replace('"possession":1,', '"possession":9,'))
    assert "9.json is on disk and not listed" in "\n".join(S.validate_tree(root))


def test_heavy_job_is_reentrant_down_the_process_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    quiet = lambda message: None
    inner = [sys.executable, "-c",
             "from export.heavy import heavy_job\nwith heavy_job('inner', wait=False, log=lambda m: None):\n    print('ran')"]
    with heavy.heavy_job("outer", log=quiet):
        with heavy.heavy_job("nested", wait=False, log=quiet):
            pass
        child = subprocess.run(inner, cwd=REPO, capture_output=True, text=True)
        assert child.returncode == 0 and "ran" in child.stdout, child.stderr
    wrapped = subprocess.run([sys.executable, "-m", "export.heavy", "--name", "wrapper", "--no-wait", "--"] + inner,
                             cwd=REPO, capture_output=True, text=True)
    assert wrapped.returncode == 0 and "ran" in wrapped.stdout, wrapped.stderr


def test_a_made_up_or_stale_holder_does_not_get_past_the_lock(tmp_path, monkeypatch):
    import os

    lock = tmp_path / "heavy.lock"
    monkeypatch.setenv("V3_HEAVY_LOCK", str(lock))
    monkeypatch.delenv(heavy.HELD_ENV, raising=False)
    quiet = lambda message: None
    probe = [sys.executable, "-m", "export.heavy", "--name", "probe", "--no-wait", "--", sys.executable, "-c", "pass"]
    with heavy.heavy_job("first", log=quiet):
        for forged in ("999999", "x", "1", str(os.getppid())):
            env = dict(os.environ, **{heavy.HELD_ENV: forged})
            blocked = subprocess.run(probe, cwd=REPO, capture_output=True, text=True, env=env)
            assert blocked.returncode != 0, f"{heavy.HELD_ENV}={forged} got past the lock"
    # Nobody holds the lock now: a stale value must not make a job skip taking it.
    monkeypatch.setenv(heavy.HELD_ENV, str(os.getpid()))
    assert heavy.held_by_this_tree() is False
    with heavy.heavy_job("after", log=quiet):
        assert f"pid {os.getpid()} after" in lock.read_text()


def test_training_games_guard_takes_a_single_id_and_refuses_nothing_to_check():
    assert F.training_games_only(["11.11.2015.NYK.at.CHA"], "test") == 1
    for bad in ("12.16.2015.PHX.at.GSW", ["11.11.2015.NYK.at.CHA", "01.18.2016.GSW.at.CLE"], [], ()):
        with pytest.raises(SystemExit):
            F.training_games_only(bad, "test")


def test_rebuild_checks_cannot_vouch_for_what_they_did_not_cover(tmp_path, monkeypatch):
    import numpy as np

    data_root = tmp_path / "data_root"
    data_root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    folder = tmp_path / "artifacts" / "xfg"
    with pytest.raises(SystemExit):
        F.assert_safe_output(data_root / "analytics" / "x")
    with pytest.raises(SystemExit):
        F.assert_safe_output(REPO / "reports" / "x.md")
    F.begin_rebuild(folder, "xfg")
    with pytest.raises(RuntimeError):
        F.require_passed(folder, "xfg")
    (folder / "model.bin").write_bytes(b"weights")
    good = dict(name="xfg", compared="predictions with stored held-out predictions", n=np.int64(10),
                n_expected=10, max_abs_diff=np.float32(1e-7), tolerance=1e-6, passed=np.bool_(True))
    F.write_check(folder, good, files=["model.bin"])
    assert F.require_passed(folder, "xfg")["passed"] is True
    with pytest.raises(RuntimeError):
        F.require_passed(folder, "epv")
    # every way a check can fall short of the fixed rule
    for change, files in ((dict(max_abs_diff=1e-3), ["model.bin"]),             # above the tolerance
                          (dict(max_abs_diff=float("nan")), ["model.bin"]),       # not a number
                          (dict(tolerance=1e-3, max_abs_diff=1e-4), ["model.bin"]),  # tolerance looser than the cap
                          (dict(n=9), ["model.bin"]),                             # not every stored prediction compared
                          (dict(n_expected=None), ["model.bin"]),                 # no expected count
                          (dict(n=0, n_expected=0), ["model.bin"]),               # nothing compared
                          (dict(name="something"), ["model.bin"]),                # unknown artifact
                          (dict(passed=False), ["model.bin"]),                    # the rebuild said no
                          ({}, [])):                                              # vouches for no file
        F.write_check(folder, dict(good, **change), files=files)
        record = F.read_check(folder)
        assert record["passed"] is False and record["reasons"], change
        with pytest.raises(RuntimeError):
            F.require_passed(folder, record["name"])
    # an output check (reported numbers) needs no expected row count, but keeps its own cap
    out = tmp_path / "artifacts" / "absence"
    out.mkdir()
    (out / "oof.parquet").write_bytes(b"rows")
    F.write_check(out, dict(name="absence", compared="out-of-fold log loss with reports/load.md", n=2,
                            max_abs_diff=2e-5, tolerance=5e-5), files=["oof.parquet"])
    assert F.require_passed(out, "absence")["passed"] is True
    # a file written after its check, or changed, is not covered by it
    (out / "oof.parquet").write_bytes(b"other rows")
    with pytest.raises(RuntimeError):
        F.require_passed(out, "absence")
    published = F.publish_checks(tmp_path / "artifacts", tmp_path / "frozen_checks.json")
    doc = json.loads(published.read_text())
    assert [c["name"] for c in doc["checks"]] == sorted(F.TOLERANCE_CAP)
    assert S.find_private(published.read_text()) == []
    monkeypatch.delenv("DATA_ROOT")
    with pytest.raises(SystemExit):
        F.assert_safe_output(folder)


def test_rebuild_code_never_calls_a_frozen_entry_point_that_scores_heldout():
    hits = []
    for path in TOOLS_FILES:
        if path.name == "frozen_common.py":
            continue  # it defines the list
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(call in line for call in F.FORBIDDEN_CALLS):
                hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == [], "a rebuild must not score held-out outcomes:\n" + "\n".join(hits)


# ----------------------------------------------------------------------------------------------- contract 1.1


def _load(root, rel):
    return json.loads((root / rel).read_text())


def _refused(model, doc):
    with pytest.raises(ValueError):
        model.model_validate(doc)


def test_every_file_names_a_source_and_a_source_says_which_rows_it_read(sample_root):
    for rel, model in (("players.json", S.PlayersFile), ("scorecard.json", S.Scorecard), ("results.json", S.Results),
                       ("demo/swap.json", S.DemoSwap), ("demo/frames.json", S.DemoFrames),
                       ("games/fixture-game-a/1.json", S.Possession),
                       ("games/fixture-game-a/possessions.json", S.PossessionIndex)):
        doc = _load(sample_root, rel)
        model.model_validate(doc)
        _refused(model, dict(doc, sources=[]))
    S.Source(path="reports/xfg.md", what="held-out log loss", rows="heldout")
    for bad in (dict(path="", what="x", rows="heldout"), dict(path="reports/xfg.md", what="", rows="heldout"),
                dict(path="/data/x.parquet", what="x", rows="heldout"), dict(path="reports/xfg.md", what="x"),
                dict(path="reports/xfg.md", what="x", rows="in_sample")):
        with pytest.raises(ValueError):
            S.Source(**bad)


def test_a_rebuilt_object_reaches_the_data_only_with_a_passed_check(sample_root):
    good = dict(name="xfg", compared="predictions with stored predictions", n=10, n_expected=10, max_abs_diff=1e-7,
                tolerance=1e-6, passed=True, unit="probability")
    S.RebuildCheck(**good)
    S.RebuildCheck(**dict(good, passed=False, max_abs_diff=0.3))
    for bad in (dict(max_abs_diff=0.3), dict(n=0, n_expected=0), dict(n=9), dict(n_expected=None)):
        with pytest.raises(ValueError):
            S.RebuildCheck(**dict(good, **bad))
    swap = _load(sample_root, "demo/swap.json")
    S.DemoSwap.model_validate(swap)
    _refused(S.DemoSwap, dict(swap, checks=[]))
    failed = [dict(c, passed=False) if c["name"] == "epv" else c for c in swap["checks"]]
    _refused(S.DemoSwap, dict(swap, checks=failed))
    # the honest form of a model that is not available: no items, no numbers, a reason
    unavailable = dict(swap, checks=failed, items=[i for i in swap["items"] if i["kind"] != "epv"],
                       summaries=[dict(s, available=False, n=0, mean_abs_diff=None, median_abs_diff=None,
                                       note="the EPV rebuild did not pass its check") if s["kind"] == "epv" else s
                                  for s in swap["summaries"]])
    S.DemoSwap.model_validate(unavailable)
    placeholder = dict(unavailable, summaries=[dict(s, mean_abs_diff=0.0) if s["kind"] == "epv" else s
                                               for s in unavailable["summaries"]])
    _refused(S.DemoSwap, placeholder)
    _refused(S.DemoSwap, dict(swap, summaries=swap["summaries"][:2]))
    players = _load(sample_root, "players.json")
    S.PlayersFile.model_validate(players)
    _refused(S.PlayersFile, dict(players, checks=[c for c in players["checks"] if c["name"] != "absence"]))
    _refused(S.PlayersFile, dict(players, checks=[c for c in players["checks"] if c["name"] != "embeddings"]))


def test_players_file_counts_rules_and_minimums_agree(sample_root):
    players = _load(sample_root, "players.json")
    _refused(S.PlayersFile, dict(players, n_shown=players["n_shown"] + 1))
    _refused(S.PlayersFile, dict(players, n_hidden=players["n_hidden"] + 1))
    _refused(S.PlayersFile, dict(players, rules=players["rules"][:4]))
    looser = [dict(r, min_n=1) if r["metric"] == "shot_making" else r for r in players["rules"]]
    _refused(S.PlayersFile, dict(players, rules=looser))
    hidden = dict(players["players"][0])
    for key in ("shot_making", "gravity", "rebounding", "absence_risk", "similar"):
        hidden[key] = {k: v for k, v in hidden[key].items() if k.startswith("min_") or k in ("available", "kind")}
        hidden[key]["shown"] = False
    hidden["shot_making"]["attempts"] = hidden["gravity"]["possessions"] = hidden["gravity"]["games"] = 0
    hidden["rebounding"]["chances"] = hidden["absence_risk"]["games"] = hidden["absence_risk"]["absent"] = 0
    hidden["similar"]["possessions"], hidden["similar"]["neighbours"] = 0, []
    _refused(S.PlayersFile, dict(players, players=[hidden] + players["players"][1:]))


def test_featured_scorecard_and_no_signal_rules(sample_root):
    index = _load(sample_root, "games/fixture-game-a/possessions.json")
    rows = index["possessions"]
    featured = next(r for r in rows if r["featured"])
    plain = next(r for r in rows if not r["featured"])
    S.PossessionSummary.model_validate(featured)
    _refused(S.PossessionSummary, dict(featured, featured_by=None))
    _refused(S.PossessionSummary, dict(featured, result="turnover"))
    _refused(S.PossessionSummary, dict(featured, xfg=None))
    _refused(S.PossessionSummary, dict(plain, featured_by="hand", featured_cell="x"))
    _refused(S.PossessionSummary, dict(featured, xfg=1.7))
    card = _load(sample_root, "scorecard.json")
    cell = card["builds"][0]["rows"][0]["cells"][0]
    S.ScorecardCell.model_validate(cell)
    _refused(S.ScorecardCell, dict(cell, status="met", passed=False))
    _refused(S.ScorecardCell, dict(cell, status="reported", passed=True))
    both_adopted = [dict(b, role="adopted") for b in card["builds"]]
    _refused(S.Scorecard, dict(card, builds=both_adopted))
    _refused(S.Scorecard, dict(card, builds=[dict(b, role="candidate") for b in card["builds"]]))
    lane = _load(sample_root, "results.json")["lanes"][0]
    S.LaneResult.model_validate(dict(lane, lane="decisions", verdict_class="failed"))
    for name in S.NO_SIGNAL_LANES:
        _refused(S.LaneResult, dict(lane, lane=name, verdict_class="works"))
    box = _load(sample_root, "demo/frames.json")["scenes"][0]["frames"][0]["boxes"][0]
    _refused(S.DemoBox, dict(box, status="ghost"))


def test_a_real_data_root_keeps_heldout_and_training_apart(tmp_path):
    root = tmp_path / "real"
    sample.write_sample(root)
    assert S.validate_tree(root) == []

    def rewrite(rel, change):
        doc = json.loads((root / rel).read_text())
        change(doc)
        (root / rel).write_text(json.dumps(doc))

    # marking the fixture as real makes every fixture value a problem
    rewrite("manifest.json", lambda d: d.update(synthetic=False))
    problems = "\n".join(S.validate_tree(root))
    assert "fixture values in a data root that is not marked synthetic" in problems
    assert "is not a held-out game" in problems
    assert "freeze_tag is 'fixture'" in problems
    assert "is fixture output and cannot be shown as a result" in problems
    # a synthetic root may name only the fixture generator
    rewrite("manifest.json", lambda d: d.update(synthetic=True))
    rewrite("players.json", lambda d: d["sources"].append(
        {"path": "$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet", "what": "x", "rows": "out_of_fold"}))
    assert any("a synthetic data root names only" in line for line in S.validate_tree(root))


def test_content_rules_on_real_looking_objects(sample_root):
    def real(obj_json, model, sources):
        doc = json.loads(obj_json)
        doc["sources"] = sources
        return model.model_validate(doc)

    held = {"path": "$DATA_ROOT/analytics/epv/heldout/2015-16/12.16.2015.PHX.at.GSW.parquet", "what": "x", "rows": "heldout"}
    oof = {"path": "$DATA_ROOT/analytics/xfg/2015-16/xfg_predictions.parquet", "what": "x", "rows": "out_of_fold"}
    text = (sample_root / "games" / "fixture-game-a" / "1.json").read_text()
    possession = real(text.replace("00.00.0000.AAA.at.BBB", "12.16.2015.PHX.at.GSW"), S.Possession, [held])
    assert S._content_problems("x.json", possession, synthetic=False) == []
    leaked = real(text.replace("00.00.0000.AAA.at.BBB", "12.16.2015.PHX.at.GSW"), S.Possession, [held, oof])
    assert any("held-out rows only" in line for line in S._content_problems("x.json", leaked, synthetic=False))
    training_game = real(text.replace("00.00.0000.AAA.at.BBB", "11.14.2015.BKN.at.GSW"), S.Possession, [held])
    assert any("not a held-out game" in line for line in S._content_problems("x.json", training_game, synthetic=False))
    players = real((sample_root / "players.json").read_text().replace(S.FIXTURE_SOURCE, "reports/xfg.md"),
                   S.PlayersFile, [oof])
    assert S._content_problems("players.json", players, synthetic=False) == []
    with_heldout = real((sample_root / "players.json").read_text().replace(S.FIXTURE_SOURCE, "reports/xfg.md"),
                        S.PlayersFile, [oof, held])
    assert any("never read held-out rows" in line for line in S._content_problems("players.json", with_heldout, False))
