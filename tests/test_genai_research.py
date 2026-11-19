"""The genai research lane (docs/research/genai.md): splits, the test door, the template writer, the evaluation."""

import ast
import json
import re
from pathlib import Path

import pytest

from export import genai as G
from export import sample
from export import schemas as S
from tools import genai_data as D
from tools import genai_eval as E
from tools import genai_templates as T

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def root(tmp_path):
    out = tmp_path / "root"
    sample.write_sample(out)
    return out


def _items(root, kind):
    return [build() for _key, build, *_ in G.items_of(root, kind)]


# ------------------------------------------------------------------------------------------------ splits


def test_player_split_is_fixed_by_the_salted_hash_and_disjoint():
    slugs = [f"player-{k}" for k in range(405)]
    a, b = D.split_players(slugs), D.split_players(list(reversed(slugs)))
    assert a == b
    assert {k: len(v) for k, v in a.items()} == {"test": 81, "dev": 40, "train": 284}
    assert not set(a["test"]) & set(a["dev"]) and not set(a["test"]) & set(a["train"])
    assert set(a["test"]) | set(a["dev"]) | set(a["train"]) == set(slugs)


def test_game_split_takes_whole_games_in_hash_order():
    games = [f"01.{k:02d}.2016.AAA.at.BBB{k}" for k in range(300)]
    s = D.split_games(games)
    assert [len(s[k]) for k in ("test", "dev", "train")] == [40, 20, 120]
    assert len(s["reserve"]) == 120
    ordered = sorted(games, key=D.rank_key)
    assert set(s["test"]) == set(ordered[:40])


def test_possession_draw_is_seeded_by_the_game():
    numbers = list(range(200))
    assert D.pick_possessions("g1", numbers, 12) == D.pick_possessions("g1", reversed(numbers), 12)
    assert D.pick_possessions("g1", numbers, 12) != D.pick_possessions("g2", numbers, 12)
    assert D.pick_possessions("g1", range(5), 12) == [0, 1, 2, 3, 4]


def test_the_training_builder_refuses_a_held_out_game():
    with pytest.raises(SystemExit, match="held-out"):
        D.build_training_game(Path("."), "12.16.2015.PHX.at.GSW", None, [0])


# ------------------------------------------------------------------------------------------------ the test door


def test_test_items_are_closed_outside_the_scorer():
    with pytest.raises(SystemExit, match="score_test_once"):
        D.items("scouting", "test")
    with pytest.raises(SystemExit, match="score_test_once"):
        D.items("possession", "test")


def test_score_test_once_scores_each_experiment_once_and_a_crash_uses_it_up(tmp_path, monkeypatch):
    if not D.SPLITS_FILE.exists():
        pytest.skip("splits not frozen in this checkout")
    ledger = tmp_path / "ledger.json"
    seen = []

    def scorer():
        seen.append(D._SCORING)
        return 1

    assert D.score_test_once("x-E1", scorer, ledger_path=ledger) == 1
    assert seen == ["x-E1"] and D._SCORING is None
    with pytest.raises(SystemExit, match="already scored"):
        D.score_test_once("x-E1", scorer, ledger_path=ledger)

    def crash():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        D.score_test_once("x-E2", crash, ledger_path=ledger)
    with pytest.raises(SystemExit, match="already scored"):
        D.score_test_once("x-E2", scorer, ledger_path=ledger)
    assert [e["experiment"] for e in json.loads(ledger.read_text())["entries"]] == ["x-E1", "x-E2"]


def _tool_files():
    return sorted((REPO / "tools").glob("genai_*.py"))


def test_only_the_evaluation_calls_score_test_once_and_at_most_once_per_experiment_name():
    callers = {}
    for path in _tool_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "score_test_once":
                name = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                callers.setdefault(path.name, []).append(name)
    assert set(callers) <= {"genai_eval.py", "genai_e2.py"}
    names = [n for calls in callers.values() for n in calls]
    assert None not in names and len(names) == len(set(names))


def test_no_genai_tool_reads_a_held_out_folder_or_the_review_texts():
    for path in _tool_files():
        text = path.read_text(encoding="utf-8")
        assert "heldout=True" not in text and not re.search(r'/ "heldout"|"heldout" /', text), path.name
        assert "genai/possession.json" not in text, path.name


def test_the_ledger_only_grows_and_names_each_experiment_once():
    if not D.LEDGER_FILE.exists():
        return
    names = [e["experiment"] for e in json.loads(D.LEDGER_FILE.read_text())["entries"]]
    assert len(names) == len(set(names))


def test_the_committed_splits_hold_counts_and_hashes_only():
    if not D.SPLITS_FILE.exists():
        pytest.skip("splits not frozen in this checkout")
    text = D.SPLITS_FILE.read_text(encoding="utf-8")
    assert not S.find_private(text)
    record = json.loads(text)
    for kind in D.KINDS:
        for split in D.SPLITS:
            pin = record[kind]["items"][split]
            assert set(pin) == {"n", "keys_sha256", "inputs_sha256"}
            assert re.fullmatch(r"[0-9a-f]{64}", pin["keys_sha256"]) and pin["n"] > 0
    assert record["scouting"]["items"]["test"]["n"] == 81
    # no player name or game id is written in it
    assert not re.search(r"\d{2}\.\d{2}\.\d{4}\.[A-Z]{3}\.at\.[A-Z]{3}", text)


# ------------------------------------------------------------------------------------------------ templates


def test_template_texts_pass_the_freeze_checker_on_the_synthetic_root(root):
    """genai-E1's claim: the template passes the checker pinned at the split freeze (loaded from git by its commit,
    as genai-E2 loads it), on every item of the synthetic root."""
    from tools.genai_e2 import checker
    check = checker("freeze")
    for kind in ("scouting", "possession"):
        for item in _items(root, kind):
            text = T.template_text(item)
            problems, _ = check(text, item)
            assert not problems, (item["key"], text, problems)


def test_template_fails_the_primary_checker_only_on_its_known_defects(root):
    """Under the merged checker (genai-E2's primary) the template E1 scored keeps its two known defects (the sign
    rule and the subject rule of a not-shown view, genai-E1.md); it is not changed, and nothing else fails."""
    from tools.genai_e2 import checker
    check = checker("primary")
    for kind in ("scouting", "possession"):
        for item in _items(root, kind):
            text = T.template_text(item)
            rules = {E.rule_of(p) for p in check(text, item)[0]}
            assert rules <= {"sign", "subject"}, (item["key"], rules)
            if kind == "possession":
                assert not rules, item["key"]


def test_template_is_deterministic(root):
    item = _items(root, "scouting")[0]
    assert T.template_text(item) == T.template_text(json.loads(json.dumps(item)))


# ------------------------------------------------------------------------------------------------ evaluation


def test_every_checker_problem_falls_under_a_named_rule(root):
    item = _items(root, "scouting")[0]
    facts = {f["field"]: f["shown"] for f in item["facts"]}
    texts = ["He made 99.9 shots and two more. This is the best player, and he passes well.",
             "The " + "very " * 170 + "end.",
             f"He played {facts['games']} games."]
    for text in texts:
        problems, _ = G.check_text(text, item)
        assert problems
        assert all(E.rule_of(p) != "other" for p in problems), [p for p in problems if E.rule_of(p) == "other"]


def test_frames_blank_names_and_numbers(root):
    item = _items(root, "scouting")[0]
    text = T.template_text(item)
    frames = E.frames_of(text, item)
    assert frames and not any(re.search(r"\d", f.replace("per 100", "")) for f in frames)
    assert not any("\u2212" in f for f in frames)
    assert not any(item["context"]["name"].lower() in f for f in frames)


def test_bootstrap_of_one_arm_against_itself_is_zero():
    rows = [{"unit": f"u{k % 7}", "passed": k % 3 != 0, "rules": [], "grade": 5.0, "words": 50,
             "frames": [f"frame {k % 4}"]} for k in range(40)]
    out = E.bootstrap({"a": rows, "b": rows}, ("pass_rate",), n_boot=200)
    assert out["differences"]["b minus a: pass_rate"] == [0.0, 0.0]
    lo, hi = out["arms"]["a"]["pass_rate"]
    assert lo <= E.arm_measures(rows)["pass_rate"] <= hi


def test_an_item_with_no_text_fails_as_no_text(root):
    item = _items(root, "scouting")[0]
    row = E.evaluate(None, item)
    assert row["passed"] is False and row["rules"] == ["no_text"]


def test_genai_code_and_dossier_are_private_clean():
    files = _tool_files() + [REPO / "docs" / "research" / "genai.md", Path(__file__)]
    for path in files:
        assert not S.find_private(path.read_text(encoding="utf-8")), path.name


def test_committed_genai_reports_are_private_clean_and_hold_no_text():
    for path in sorted((REPO / "reports" / "research").glob("genai-*")):
        text = path.read_text(encoding="utf-8")
        assert not S.find_private(text), path.name
        if path.suffix == ".json":
            assert '"text"' not in text and '"frames"' not in text, path.name


# ------------------------------------------------------------------------------------------------ genai-E2 (no model)


def _guard(neg, pos):
    from tools.genai_e2 import NumberGuard
    g = NumberGuard.__new__(NumberGuard)
    g.sets = [(set(neg), set(pos) | {"100"})]
    return g


def test_number_guard_allows_only_prefixes_of_the_items_values():
    g = _guard(neg={"4.4"}, pos={"0.143", "6.0", "1,874"})
    assert g.allowed_next("He made ", 0) == ({"0", "6", "1"}, False)
    assert g.allowed_next("He made 0.1", 0) == ({"4"}, True)
    chars, forced = g.allowed_next("He made 0.143", 0)
    assert not forced and "." in chars and not any(c.isdigit() for c in chars)
    assert g.allowed_next("a 0.143.", 0) == (set(), False)          # punctuation ends it; no glued digit
    assert g.allowed_next("range \u2212", 0) == ({"4"}, False)      # after a minus, only negative values
    assert g.allowed_next("range \u22124.", 0) == ({"4"}, True)
    assert g.allowed_next("per 1", 0)[0] == {",", "0"}               # 1,874 or 100


def test_readings_follow_the_pre_registration():
    from tools.genai_e2 import reading
    assert reading([0.05, 0.3], 0.15, "h1") == "passes"
    assert reading([-0.1, 0.08], 0.0, "h1") == "fails"
    assert reading([-0.02, 0.2], 0.09, "h1") == "inconclusive"
    assert reading([-0.04, 0.1], 0.02, "h3") == "non-inferior"
    assert reading([-0.3, -0.06], -0.2, "h3") == "inferior"
    assert reading([0.01, 0.2], 0.1, "h2") == "helps" and reading([-0.2, -0.01], -0.1, "h2") == "hurts"


def test_e2_checkers_are_pinned_by_blob_and_the_primary_is_the_merged_checker():
    from tools import genai_e2 as X
    import subprocess
    for name, (commit, blob) in X.CHECKERS.items():
        try:
            found = subprocess.run(["git", "rev-parse", f"{commit}:export/genai.py"], cwd=REPO, capture_output=True,
                                   text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError:
            pytest.skip("pinned commit not in this clone")
        assert found == blob, name
    assert X.CHECKERS["freeze"][1] == json.loads(D.SPLITS_FILE.read_text())["checker"]["git_blob_sha1"]


def test_e2_stretch_is_capped_at_three_hours():
    from tools import genai_e2 as X
    assert X.MAX_STRETCH_H <= 3.0
    assert X.main(["sft", "--max-hours", "3.5"]) == 1


def _dev_entry(epoch, scouting, possession):
    def m(rate):
        return {"pass_rate": rate}
    return {"epoch": epoch, "primary": {"scouting": m(scouting), "possession": m(possession)},
            "mean_pass": (scouting + possession) / 2}


def _fake_runs(tmp_path, monkeypatch, a1, a2r1=None):
    from tools import genai_e2 as X
    monkeypatch.setenv("V3_GENAI_MODELS", str(tmp_path))
    for tag, dev in (("a1", a1), ("a2-r1", a2r1)):
        if dev is None:
            continue
        folder = X.run_dir() / tag
        folder.mkdir(parents=True)
        (folder / "state.json").write_text(json.dumps({"epoch": len(dev), "dev": dev}))
    return X


def test_epochs_and_the_a3_arm_weigh_the_two_kinds_equally(tmp_path, monkeypatch):
    # epoch 2 wins on pooled items (possession heavy) but epoch 1 wins on the mean of the kinds
    X = _fake_runs(tmp_path, monkeypatch, [_dev_entry(1, 0.80, 0.70), _dev_entry(2, 0.40, 0.95)])
    assert X.best_epoch("a1") == 1
    assert X.dev_best_arm() == "a1-epoch1"


def test_rft_round_two_is_refused_below_the_pre_registered_gain(tmp_path, monkeypatch):
    X = _fake_runs(tmp_path, monkeypatch, [_dev_entry(1, 0.50, 0.50)], [_dev_entry(1, 0.52, 0.53)])
    allowed, gain = X.round2_allowed()
    assert not allowed and abs(gain - 0.025) < 1e-9
    with pytest.raises(SystemExit, match="round 2 refused"):
        X.stage_rft(deadline=0, round_=2)
    X = _fake_runs(tmp_path / "b", monkeypatch, [_dev_entry(1, 0.50, 0.50)], [_dev_entry(1, 0.54, 0.53)])
    assert X.round2_allowed()[0]
    assert X.dev_best_arm() == "a2-r1-epoch1"


def test_the_test_stage_refuses_without_committed_dev_choices(monkeypatch, tmp_path):
    from tools import genai_e2 as X
    monkeypatch.setattr(X, "DEV_JSON", REPO / "reports" / "research" / "genai-E2-0.5B-dev-missing.json")
    with pytest.raises(SystemExit, match="committed and unchanged"):
        X.dev_json_clean()


def test_test_stage_refuses_without_a_matching_rehearsal(tmp_path, monkeypatch):
    from tools import genai_e2 as X
    monkeypatch.setenv(D.DATA_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="rehearse"):
        X.rehearsal_matches("a" * 64, "b" * 40)
    (tmp_path / "e2").mkdir()
    (tmp_path / "e2" / "rehearse.json").write_text(json.dumps({"dev_json_sha256": "a" * 64, "head": "c" * 40}))
    with pytest.raises(SystemExit, match="another dev json or another commit"):
        X.rehearsal_matches("a" * 64, "b" * 40)
    X.rehearsal_matches("a" * 64, "c" * 40)


def test_stage_test_calls_score_test_once_once_with_the_experiment_name():
    tree = ast.parse((REPO / "tools" / "genai_e2.py").read_text(encoding="utf-8"))
    found = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                     and getattr(c.func, "attr", getattr(c.func, "id", None)) == "score_test_once"]
            if calls:
                found[fn.name] = [c.args[0].value for c in calls]
    assert found == {"stage_test": ["genai-E2-0.5B", "genai-E2-1.5B"],
                     "stage_test_resume": ["genai-E2-0.5B-resume", "genai-E2-1.5B-resume"]}
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))
             and any(isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "score_test_once" for c in ast.walk(n))]
    assert not loops


def test_the_resume_entry_refuses_without_a_crash_record(tmp_path, monkeypatch):
    from tools import genai_e2 as X
    monkeypatch.setenv(D.DATA_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="no crash record"):
        X.stage_test_resume()


def test_a_checker_that_raises_fails_the_text_as_other(monkeypatch):
    from tools import genai_e2 as X

    def boom(text, item):
        raise ValueError("x")

    monkeypatch.setitem(X._CHECKS, "primary", boom)
    item = {"names": [], "context": {"name": "A"}, "facts": []}
    row = X.evaluate("Some text.", item, "primary")
    assert row["passed"] is False and row["rules"] == ["other"]


def test_each_size_has_its_own_folders_dev_json_and_number_format(monkeypatch):
    from tools import genai_e2 as X
    try:
        X.set_size("1.5B")
        assert X.MODEL[0].endswith("1.5B-Instruct") and X.DTYPE == "bfloat16" and X.EXPERIMENT == "genai-E2-1.5B"
        assert X.out_dir().name == "e2-1.5B" and X.run_dir().name == "genai-E2-1.5B"
        assert X.DEV_JSON.name == "genai-E2-1.5B-dev.json"
    finally:
        X.set_size("0.5B")
    assert X.out_dir().name == "e2" and X.DTYPE == "float32" and X.DEV_JSON.name == "genai-E2-0.5B-dev.json"
