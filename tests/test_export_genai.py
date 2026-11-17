"""export/genai.py: machine-written texts (PLAN.md P3, D76).

The number check is the heart of it: every number of a text must be a value of its page, as the page rounds it,
in the same unit. The adversarial cases (a made-up number, a value rounded another way, a percent written as a
share and the reverse, a dropped sign, a number in words) live in tests/fixtures/genai_cases.json, which the web
side's checker (web/lib/genai.ts) must pass too. No test here calls a model: the live path runs on a fake client.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from export import genai
from export import public_data
from export import sample
from export import schemas as S

REPO = Path(__file__).resolve().parent.parent
CASES = json.loads((REPO / "tests" / "fixtures" / "genai_cases.json").read_text(encoding="utf-8"))


@pytest.fixture()
def root(tmp_path):
    out = tmp_path / "data"
    sample.write_sample(out)
    return out


def _facts(rows):
    return [genai.Fact(**row) for row in rows]


# --------------------------------------------------------------------------------------------- page rounding


@pytest.mark.parametrize("case", CASES["format"], ids=lambda c: f"{c['fn']}-{c['value']}")
def test_shown_values_follow_the_rounding_of_the_page(case):
    fn = {"number": lambda c: genai.format_number(c["value"], c["digits"]),
          "signed": lambda c: genai.format_signed(c["value"], c["digits"]),
          "percent": lambda c: genai.format_percent(c["value"], c["digits"]),
          "count": lambda c: genai.format_count(c["value"]),
          "probability": lambda c: genai.format_probability(c["value"]),
          "clock": lambda c: genai.format_clock(c["value"]),
          "seconds": lambda c: genai.format_seconds(c["value"]),
          "stored": lambda c: genai.format_stored(c["value"]),
          "integer": lambda c: genai.format_integer(c["value"])}[case["fn"]]
    assert fn(case) == case["shown"]


# ------------------------------------------------------------------------------------------- the number check


@pytest.mark.parametrize("case", CASES["numbers"], ids=lambda c: c["name"])
def test_number_check_on_the_shared_cases(case):
    problems, matched = genai.check_numbers(case["text"], _facts(case["facts"]), case["names"])
    if case["ok"]:
        assert problems == []
        assert matched, "a passing case names the fact of each number"
    else:
        assert problems, f"{case['name']}: the check let it through"
        assert any(case["problem"] in line for line in problems), problems


def test_every_digit_of_a_text_is_read_as_a_number():
    text = "Values 1 and 22.5 and 3,000 and 7:38 and 4th and x9 and -2 and 50% and 0.5 percent."
    tokens = genai.numbers_in(text)
    digits = sum(ch.isdigit() for ch in text)
    assert sum(ch.isdigit() for t in tokens for ch in t.raw) == digits
    assert [t.canon for t in tokens] == ["1", "22.5", "3000", "7:38", "4", "9", "-2", "50%", "0.5%"]


def test_a_shown_value_matches_itself_in_every_unit_it_allows():
    facts = _facts([{"field": "a", "shown": "+3.9", "unit": "none", "kind": "fact", "group": ""}])
    assert genai.check_numbers("It was +3.9 and also 3.9 here today.", facts)[0] == []
    assert genai.check_numbers("It was -3.9 here today.", facts)[0] != []


# ----------------------------------------------------------------------------------------------- word rules


@pytest.mark.parametrize("text, found", [
    ("The ghost defender moved left of the man in the corner of the court today for a long time here.", "jargon"),
    ("The xFG of the shot was low and the man in the corner of the court had a good look at it.", "jargon"),
    ("This was the best defense on the floor that night and the team held on to win the game.", "forbidden"),
    ("The team should have passed the ball more in that spot and then it would have had a good look.", "forbidden"),
    ("The team moved the ball well — and then it shot from the corner of the court near the end here.", "dash"),
    ("Short.", "words, not"),
])
def test_word_rules_refuse_what_site_copy_refuses(text, found):
    assert any(found in line for line in genai.check_words(text))


def test_reading_grade_matches_the_site_rule():
    hard = ("Comprehensive institutional probabilistic considerations notwithstanding, organizational "
            "infrastructure deteriorated considerably throughout intercontinental collaborations yesterday.")
    assert genai.round1(genai.fk_grade(hard)) > genai.MAX_GRADE
    assert any("grade" in line for line in genai.check_words(" ".join([hard] * 3)))
    # The syllable rule is the one of web/scripts/lib/readability.mjs and tests/test_public_docs.py.
    checked = {"player": 2, "players": 2, "playing": 2, "frame": 1, "frames": 1, "table": 2, "boxes": 2,
               "moved": 1, "wanted": 2, "possession": 3, "value": 2, "average": 3, "video": 3}
    assert {w: genai.count_syllables(w) for w in checked} == checked


def test_a_model_that_did_not_work_is_named(root):
    players = S.PlayersFile.model_validate_json((root / "players.json").read_bytes())
    player = next(p for p in players.players if p.shot_making.shown)
    item = genai.player_input(player, players, None)
    assert item["context"]["model_status"]["shot_making"] == "no_result"
    facts = {f["field"]: f["shown"] for f in item["facts"]}
    text = (f"He made {facts['shot_making.over_expected_per_100.value']} shots per 100 above the rate, range "
            f"{facts['shot_making.over_expected_per_100.lo']} to {facts['shot_making.over_expected_per_100.hi']}.")
    assert any("status" in line for line in genai.check_status(text, item))
    assert genai.check_status(text + " That model has no result yet.", item) == []
    failed = S.Results(schema_version=S.SCHEMA_VERSION, freeze_tag="t", generated="t", pending=[], notes=[],
                       sources=sample.SRC_FIXTURE, lanes=[S.LaneResult(
                           lane="xfg", question="q", selected_model="m", status="s", verdict="failed",
                           verdict_source="s", verdict_class="failed", report="reports/xfg.md", rows=[],
                           out_of_fold=S.MetricValue(text="t", point_estimate=True))])
    item = genai.player_input(player, players, failed)
    assert any("(failed)" in line for line in genai.check_status(text, item))
    assert genai.check_status(text + " Its model failed.", item) == []


# ------------------------------------------------------------------------------------------------ the inputs


def test_inputs_hold_only_what_the_page_shows(root):
    players = S.PlayersFile.model_validate_json((root / "players.json").read_bytes())
    hidden = next(p for p in players.players if not p.shot_making.shown)
    fields = {f["field"] for f in genai.player_input(hidden, players, None)["facts"]}
    assert not any(field.startswith("shot_making.over_expected") for field in fields)
    assert {"shot_making.attempts", "shot_making.min_attempts"} <= fields
    shown = next(p for p in players.players if p.shot_making.shown)
    facts = {f["field"]: f for f in genai.player_input(shown, players, None)["facts"]}
    assert facts["shot_making.fg_pct"]["shown"] == genai.format_percent(shown.shot_making.fg_pct, 1)
    assert facts["gravity.gravity_ft.value"]["shown"] == genai.format_number(shown.gravity.gravity_ft.value, 2)


def test_a_possession_input_is_the_same_on_the_public_root(root, tmp_path):
    public = tmp_path / "public"
    public_data.build_public(root, public)
    for key, build in genai.choose(root, "possession", None):
        slug, number = key.split("/")
        index = S.PossessionIndex.model_validate_json((public / f"games/{slug}/possessions.json").read_bytes())
        row = next(r for r in index.possessions if str(r.possession) == number)
        stripped = S.Possession.model_validate_json((public / f"games/{slug}/{number}.json").read_bytes())
        results = S.Results.model_validate_json((public / "results.json").read_bytes())
        assert genai.possession_input(row, stripped, results) == build()
    # The public root carries the texts and they pass every check there too.
    assert S.validate_tree(public, allow_missing=S.STAGED_FILES) == []
    assert (public / "genai" / "possession.json").exists()


def test_a_possession_input_names_no_position(root):
    key, build = genai.choose(root, "possession", None)[0]
    text = json.dumps(build())
    assert '"x"' not in text and '"y"' not in text and "ghost" not in text


# ------------------------------------------------------------------------------------------------- data root


def test_sample_root_is_valid_with_texts_and_some_pages_have_none(root):
    assert S.validate_tree(root) == []
    for kind, rel in S.GENAI_FILES.items():
        texts = S.GeneratedTexts.model_validate_json((root / rel).read_bytes())
        assert texts.kind == kind and texts.label == "machine-written"
        assert 0 < len(texts.items) < len(genai.items_of(root, kind))
        assert all(item.model == genai.FIXTURE_MODEL for item in texts.items)
    manifest = S.Manifest.model_validate_json((root / "manifest.json").read_bytes())
    assert manifest.schema_version == "1.2.0"
    assert sorted(f.path for f in manifest.files if f.model == "GeneratedTexts") == sorted(S.GENAI_FILES.values())


def test_a_root_with_no_texts_is_complete(root):
    for rel in S.GENAI_FILES.values():
        (root / rel).unlink()
    (root / "genai").rmdir()
    manifest = S.Manifest.model_validate_json((root / "manifest.json").read_bytes())
    (root / "manifest.json").write_text(genai._dump(genai.with_genai_entry(manifest, root)), encoding="utf-8")
    assert S.validate_tree(root) == []


def test_older_data_files_read_under_a_1_2_0_manifest_and_texts_are_1_2_0_only(root):
    """Contract 1.2.0: a data file of 1.1.0 still validates beside the texts, while the manifest and a genai file
    carry 1.2.0 itself."""
    for rel in ("players.json", "results.json", "games/fixture-game-a/possessions.json"):
        _rewrite(root, rel, lambda doc: doc.update(schema_version="1.1.0"))
    assert S.validate_tree(root) == []
    _rewrite(root, "genai/scouting.json", lambda doc: doc.update(schema_version="1.1.0"))
    assert any(line.startswith("genai/scouting.json: ValidationError") for line in S.validate_tree(root))
    with pytest.raises(ValueError):
        S.Manifest.model_validate({**json.loads((root / "manifest.json").read_text()), "schema_version": "1.1.0"})


def _rewrite(root: Path, rel: str, change) -> None:
    """Change one file and write the manifest again, so only the contract's own rules can refuse it."""
    doc = json.loads((root / rel).read_text(encoding="utf-8"))
    change(doc)
    (root / rel).write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == rel:
            data = (root / rel).read_bytes()
            entry["bytes"], entry["sha256"] = len(data), hashlib.sha256(data).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")


def test_a_wrong_number_in_a_stored_text_fails_validation(root):
    def change(doc):
        item = doc["items"][0]
        number = item["numbers"][0]["shown"]
        item["text"] = item["text"].replace(number, number + "1", 1)
    _rewrite(root, "genai/scouting.json", change)
    problems = S.validate_tree(root)
    assert any("not a value of this page" in line for line in problems), problems


def test_a_stale_text_fails_validation_and_prune_drops_it(root):
    texts = S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes())
    slug = texts.items[0].key

    def change(doc):
        player = next(p for p in doc["players"] if p["slug"] == slug)
        player["games"] += 1
    _rewrite(root, "players.json", change)
    assert any(f"{slug}: stale" in line for line in S.validate_tree(root))
    assert genai.prune(root) == {"scouting": 1, "possession": 0}
    manifest = S.Manifest.model_validate_json((root / "manifest.json").read_bytes())
    (root / "manifest.json").write_text(genai._dump(genai.with_genai_entry(manifest, root)), encoding="utf-8")
    assert S.validate_tree(root) == []


def test_texts_sit_at_their_own_path_and_never_hold_fixture_words_on_real_data(root):
    texts = S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes())
    assert S._content_problems("genai/scouting.json", texts, synthetic=False)
    moved = texts.model_copy(update={"kind": "possession"})
    assert genai.root_problems(moved, {}) and moved.kind == "possession"
    with pytest.raises(ValueError):
        S.GeneratedTexts.model_validate({**texts.model_dump(), "items": texts.model_dump()["items"] * 2})


def test_fixture_texts_go_only_into_a_synthetic_root(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["synthetic"] = False
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(genai.GenaiProblem):
        genai.run(root, log=lambda line: None)


# ------------------------------------------------------------------------------------------------------ prompts


def test_prompt_versions_pin_their_files():
    versions = json.loads(genai.PROMPT_VERSIONS.read_text(encoding="utf-8"))
    for version, digest in versions["sha256"].items():
        text = (genai.PROMPT_DIR / f"{version}.md").read_bytes()
        assert hashlib.sha256(text).hexdigest() == digest, f"{version} was edited: make a new version instead"
    for kind in genai.KINDS:
        version, text, digest = genai.prompt(kind)
        assert version.startswith(kind) and digest == versions["sha256"][version]
    files = {p.stem for p in genai.PROMPT_DIR.glob("*.md")}
    assert files == set(versions["sha256"]), "every prompt file has a pinned hash and the reverse"


# ------------------------------------------------------------------------------------------- the live path


class FakeClient:
    """Stands in for anthropic.Anthropic: answers from a list of texts and records each call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.answers.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


def test_live_path_retries_once_caches_and_records_model_and_prompt(root, tmp_path):
    key, build = genai.choose(root, "possession", None)[0]
    item = build()
    good = genai.fixture_text(item)
    client = FakeClient(["The offense scored 99 points on a 0.123 chance.", good])
    record, cached = genai.text_for(item, live=True, client=client, cache=tmp_path / "cache")
    assert not cached and record["text"] == good and len(record["attempts"]) == 2
    assert record["model"] == genai.MODEL and record["prompt_version"] == genai.prompt("possession")[0]
    assert "not a value of this page" in " ".join(record["attempts"][0]["problems"])
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user" and "broke these rules" in second[-1]["content"]
    assert client.calls[0]["model"] == genai.MODEL and client.calls[0]["output_config"] == {"effort": "low"}
    # The same input, model and prompt again: the cache answers and no call is made.
    again, cached = genai.text_for(item, live=True, client=FakeClient([]), cache=tmp_path / "cache")
    assert cached and again["text"] == good


def test_live_path_drops_a_text_that_fails_twice(root, tmp_path):
    key, build = genai.choose(root, "scouting", None)[0]
    client = FakeClient(["He is the best shooter here and made 77.7% of shots, which proves a lot."] * 2)
    record, _ = genai.text_for(build(), live=True, client=client, cache=tmp_path / "cache")
    assert record["text"] is None and record["problems"] and len(client.calls) == genai.MAX_ATTEMPTS


def test_run_writes_live_texts_into_a_real_root_and_reports_tokens(root, tmp_path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for kind, rel in S.GENAI_FILES.items():
        (root / rel).unlink()
    answers = []
    for kind in genai.KINDS:
        for key, build in genai.choose(root, kind, 2):
            answers.append(genai.fixture_text(build()))
    client = FakeClient(answers)
    info = genai.run(root, n_players=2, n_possessions=2, live=True, client=client, cache=tmp_path / "cache",
                     log=lambda line: None)
    assert manifest["synthetic"] is True
    assert info["kinds"]["scouting"]["kept"] == 2 and info["kinds"]["possession"]["kept"] == 2
    assert info["kinds"]["scouting"]["input_tokens"] == 2000
    texts = S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes())
    assert {item.model for item in texts.items} == {genai.MODEL}
    lines = genai.report(info)
    assert any("projected" in line for line in lines)
    assert info["problems"] == []


def test_the_validate_command_checks_texts_with_the_same_contract_classes(root):
    """python3 -m export.schemas runs as __main__; the text checks must still meet the classes the files were read
    with, or every possession text fails to build its input."""
    import subprocess
    import sys

    run = subprocess.run([sys.executable, "-m", "export.schemas", "--validate", str(root)], cwd=REPO,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "data root is valid" in run.stdout


# ------------------------------------------------------------------------------------- review fixes (p3-fix)


def test_a_partial_run_keeps_earlier_texts_that_still_hold(root):
    """A sample run after a fuller one merges: earlier texts of items it did not choose stay when they still pass,
    and a stale earlier text goes."""
    genai.run(root, n_players=None, n_possessions=None, log=lambda line: None)
    full = S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes())
    assert len(full.items) == len(genai.items_of(root, "scouting"))
    info = genai.run(root, n_players=2, n_possessions=2, log=lambda line: None)
    after = S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes())
    assert [t.key for t in after.items] == [t.key for t in full.items]
    assert info["kinds"]["scouting"]["earlier_kept"] == len(full.items) - 2
    assert info["problems"] == []
    slug = full.items[-1].key

    def change(doc):
        next(p for p in doc["players"] if p["slug"] == slug)["games"] += 1
    _rewrite(root, "players.json", change)
    chosen = {key for key, _ in genai.choose(root, "scouting", 2)}
    assert slug not in chosen
    info = genai.run(root, n_players=2, n_possessions=2, log=lambda line: None)
    kept = {t.key for t in S.GeneratedTexts.model_validate_json((root / "genai/scouting.json").read_bytes()).items}
    assert slug not in kept and info["kinds"]["scouting"]["earlier_dropped"] == 1
    assert info["problems"] == []


def test_a_change_to_the_texts_alone_gives_a_new_build_id(root):
    from export import site_data

    first = site_data.build_manifest(root)
    assert any(f.model == "GeneratedTexts" for f in first.files)

    def change(doc):
        doc["notes"].append("Another note.")
    _rewrite(root, "genai/scouting.json", change)
    second = site_data.build_manifest(root)
    assert second.build_id != first.build_id
    # The genai writer makes a content-derived build id again too; a hand-set id is kept.
    manifest = first.model_copy(update={"build_id": f"site-{S.content_id(first.files)[:12]}"})
    assert genai.with_genai_entry(manifest, root).build_id == f"site-{S.content_id(second.files)[:12]}"
    assert genai.with_genai_entry(first.model_copy(update={"build_id": "fixture"}), root).build_id == "fixture"


def test_sources_cite_the_prompt_versions_the_texts_carry(root):
    key = genai.choose(root, "scouting", None)[0][0]
    texts = [S.GeneratedText(key=key, text="t", input_sha256="0" * 64, model=genai.MODEL, prompt_version=version,
                             numbers=[]) for version in ("scouting-v0",)]
    paths = [source.path for source in genai._sources(root, "scouting", texts, synthetic=False)]
    assert "export/prompts/scouting-v0.md" in paths
    assert f"export/prompts/{genai.prompt('scouting')[0]}.md" not in paths
