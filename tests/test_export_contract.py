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

REPO = Path(__file__).resolve().parents[1]
EXPORT_FILES = sorted((REPO / "export").glob("*.py"))
OWNED_TEXT = EXPORT_FILES + [REPO / "CLAUDE.md", REPO / ".claude" / "agents" / "ml-reviewer.md",
                             REPO / ".claude" / "agents" / "verifier.md", Path(__file__)]
# Calls and imports that fit, tune or calibrate a model. export/ may hold none of them (PLAN.md M1 DoD).
FIT_PATTERNS = (r"\.fit\(", r"\.fit_transform\(", r"\.partial_fit\(", r"\.fit_predict\(", r"^\s*(from|import)\s+sklearn",
                r"^\s*(from|import)\s+torch", r"\.backward\(", r"optimizer\.step\(", r"GridSearchCV", r"cross_val_")
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
        if path.suffix == ".py" and path.parent.name == "export":
            # The contract module lists the private patterns themselves; everything else in export/ must be clean.
            body = text if path.name != "schemas.py" else ""
            for found in S.find_private(body):
                problems.append(f"{path.relative_to(REPO)}: private string {found!r}")
    assert problems == []


def test_heavy_job_lock_excludes_a_second_job(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    quiet = lambda message: None
    probe = [sys.executable, "-m", "export.heavy", "--name", "probe", "--no-wait", "--", sys.executable, "-c", "pass"]
    with heavy.heavy_job("first", log=quiet):
        blocked = subprocess.run(probe, cwd=REPO, capture_output=True, text=True)
        assert blocked.returncode != 0
        assert "another heavy job is running" in blocked.stderr
    free = subprocess.run(probe, cwd=REPO, capture_output=True, text=True)
    assert free.returncode == 0, free.stderr
