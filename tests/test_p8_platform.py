"""docs/research/P8_PLATFORM.md: every number in its tables and findings is in reports/research/p8_platform.json."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "research" / "P8_PLATFORM.md"
DATA = REPO / "reports" / "research" / "p8_platform.json"
# Numbers in the doc that are not measurements: dates, decision ids, thresholds named in the text, row labels.
NOT_MEASURED = {"2026", "11", "18", "98", "99", "8", "6", "0.40", "0.45", "640", "32", "16", "1", "2", "3", "02",
                "04", "03", "0.2"}
# Planning figures quoted from the lead's messages and the runner's own estimates, not from the diagnosis files.
ESTIMATES = {"2.4", "12", "29"}


def flat_numbers(x) -> set:
    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            out |= flat_numbers(v) | flat_numbers(k)
    elif isinstance(x, list):
        for v in x:
            out |= flat_numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out |= {str(x), f"{x:g}", f"{x:.2f}"}
    elif isinstance(x, str):
        out |= set(re.findall(r"\d+(?:\.\d+)?", x))
    return out


def test_every_number_of_the_platform_note_is_in_its_json():
    data = json.loads(DATA.read_text())
    known = flat_numbers(data)
    text = DOC.read_text()
    found = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text))
    missing = sorted(n for n in found if n not in known and n not in NOT_MEASURED and n not in ESTIMATES)
    assert missing == []


def test_the_platform_json_holds_no_local_path_and_no_held_out_or_test_game():
    from export.schemas import find_private
    text = DATA.read_text()
    assert find_private(text) == []
    from tools.research_testset import cv_test_hits
    assert cv_test_hits([text]) == []
    assert "cle_" not in text and "gsw_phx_2016" not in text
