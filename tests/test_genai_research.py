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
    assert set(callers) <= {"genai_eval.py"}
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


def test_template_texts_pass_the_checker_on_the_synthetic_root(root):
    for kind in ("scouting", "possession"):
        for item in _items(root, kind):
            text = T.template_text(item)
            problems, _ = G.check_text(text, item)
            assert not problems, (item["key"], text, problems)


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
