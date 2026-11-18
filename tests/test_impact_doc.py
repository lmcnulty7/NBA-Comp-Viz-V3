"""docs/IMPACT.md, the measured-impact summary for a hiring manager.

What these tests hold:
- every number of the document sits in a block (a paragraph or a list item) that names its sources after
  "Source:" or "Sources:" as repository paths in backticks, every named source is a committed file, and every
  number of the block is a number of one of those files (read the way export.research_methods reads numbers: a
  digit inside a name such as p50, E1 or gsw_phx_2016 is not a number, the true minus is a hyphen, a leading plus
  and thousands commas do not count);
- a heading holds no number;
- no long dash, no absolute path, no email address;
- the reading grade of the whole document, and of each block, is 9 or lower by the counter of
  tests/test_public_docs.py, with a number counted as one syllable;
- the document claims no visitors or users reached, since the site measures none.

Nothing here reads DATA_ROOT. Every test can fail.
"""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from export import research_methods as R

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "IMPACT.md"
SOURCE_RE = re.compile(r"Sources?:\s*((?:`[^`]+`(?:,\s*)?)+)")
CODE_RE = re.compile(r"`[^`\n]*`")


def _public_docs():
    spec = importlib.util.spec_from_file_location("public_docs_counter", REPO / "tests" / "test_public_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocks(text: str) -> list[str]:
    """Paragraphs and list items, each flattened to one line. Headings are blocks of their own."""
    out = []
    for chunk in re.split(r"\n\s*\n", text):
        for item in re.split(r"(?m)^- ", chunk):
            if item.strip():
                out.append(" ".join(item.split()))
    return out


def numbers_of_block(block: str) -> list[str]:
    """The numbers of a block's prose: code spans (paths, names) and link targets are left out."""
    prose = CODE_RE.sub(" ", re.sub(r"\]\([^)]*\)", "]", block))
    return R.numbers(prose)


def sources_of_block(block: str) -> list[str]:
    found = []
    for m in SOURCE_RE.finditer(block):
        found += re.findall(r"`([^`]+)`", m.group(1))
    return found


def trace_problems(text: str, repo: Path = REPO) -> list[str]:
    """Every number of `text` that its block's sources do not hold, every block with numbers and no source, and
    every source that is not a committed file."""
    tracked = set(subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True).stdout.split())
    problems = []
    for block in blocks(text):
        found = numbers_of_block(block)
        if block.startswith("#"):
            if found:
                problems.append(f"a heading holds a number: {block[:60]!r}")
            continue
        if not found:
            continue
        sources = sources_of_block(block)
        if not sources:
            problems.append(f"numbers {found} with no source: {block[:60]!r}")
            continue
        held = set()
        for path in sources:
            if path not in tracked:
                problems.append(f"{path} is not a committed file")
                continue
            held |= set(R.numbers((repo / path).read_text(encoding="utf-8")))
        for number in found:
            if number not in held:
                problems.append(f"{number} is in none of {sources}: {block[:60]!r}")
    return problems


def test_every_number_of_the_impact_doc_is_in_its_cited_file():
    assert trace_problems(DOC.read_text(encoding="utf-8")) == []


def test_the_trace_check_can_fail():
    good = "- RMSE difference -1.15 [-1.59, -0.77] points. Source: `reports/research/lineups-E1.md`\n"
    assert trace_problems(good) == []
    assert any("-1.16" in p for p in trace_problems(good.replace("-1.15", "-1.16")))
    assert any("no source" in p for p in trace_problems("- It passed 1300 games.\n"))
    assert any("not a committed file" in p for p in trace_problems("- 3 games. Source: `reports/nothing.md`\n"))
    assert any("heading" in p for p in trace_problems("## The 17 games\n"))
    # a number in a code span or a name is not a number
    assert trace_problems("- The scorer `p50` of `gsw_phx_2016` and xfg-E1, P7.\n") == []


def test_the_impact_doc_is_plain_and_clean():
    text = DOC.read_text(encoding="utf-8")
    docs = _public_docs()
    assert docs.hygiene_problems(text) == []
    prose = CODE_RE.sub("x", text)
    grade = docs.fk_grade(prose)
    assert grade <= 9, grade
    for block in blocks(text):
        if not block.startswith("#"):
            assert docs.fk_grade(CODE_RE.sub("x", block)) <= 9, block[:80]


@pytest.mark.parametrize("claim", [r"\bvisitors? (?:came|visited|used)", r"\busers? reached\b", r"\bpage ?views\b",
                                   r"\b\d[\d,]* (?:visitors|users|readers)\b"])
def test_the_impact_doc_claims_no_audience(claim):
    assert not re.search(claim, DOC.read_text(encoding="utf-8"), re.I)
