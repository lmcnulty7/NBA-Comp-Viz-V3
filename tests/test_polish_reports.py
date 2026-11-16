"""The polish baseline (PLAN.md P4, docs/POLISH_LOOP.md): reports/polish/baseline-<variant>.json.

Each file is the scorer's output for the unchanged site: it passed its own gate after npm run check, it was
scored three times so it carries the noise the loop must beat, its weights sum to one, and it holds no private
string (no local path, no cloud mount name, no email address) and no long dash.
"""

import json
from pathlib import Path

import pytest

from export import schemas as S

REPO = Path(__file__).resolve().parents[1]
POLISH = REPO / "reports" / "polish"
VARIANTS = ("public", "private")
METRICS = {"a11yWeighted", "firstViewMs", "firstLoadJsKb", "pageWeightKb", "readingGrade"}


def _load(variant):
    path = POLISH / f"baseline-{variant}.json"
    assert path.exists(), f"missing {path.relative_to(REPO)}"
    return path.read_text(encoding="utf-8"), json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_holds_no_private_string_and_no_long_dash(variant):
    text, _ = _load(variant)
    assert S.find_private(text) == []
    assert chr(0x2013) not in text and chr(0x2014) not in text


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_passed_its_gate_on_a_clean_commit_after_the_check(variant):
    _, score = _load(variant)
    assert score["schema"] == "polish-score/1"
    assert score["variant"] == variant
    assert score["synthetic"] is False
    assert score["commit"] and score["dirty"] is False
    assert score["check"]["ran"] and score["check"]["passed"]
    assert score["gate"] == {"passed": True, "against": None, "reasons": []}
    assert score["pageFailures"] == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_carries_the_noise_and_its_cost_is_one(variant):
    _, score = _load(variant)
    assert set(score["values"]) == METRICS == set(score["weights"])
    assert abs(sum(score["weights"].values()) - 1) < 1e-9
    assert score["reference"]["values"] == score["values"]
    assert score["cost"] == 1
    noise = score["noise"]
    assert noise["runs"] == 3 and len(score["runs"]) == 3
    assert noise["costSpread"] == pytest.approx(max(noise["costs"]) - min(noise["costs"]), abs=1e-4)


def test_the_private_baseline_scores_the_talk_and_the_public_one_does_not():
    _, public = _load("public")
    _, private = _load("private")
    assert "talk/" not in {row["pattern"] for row in public["routes"]["sampled"]}
    assert "talk/" in {row["pattern"] for row in private["routes"]["sampled"]}
    assert public["dataBuild"].endswith("-public")
