"""Public documents (PLAN.md M4 and M8, checklist C3): README.md, docs/SCRIPT.md, PROJECT_SUMMARY.md, PIPELINE.md.

What these tests hold:
- README.md and docs/SCRIPT.md carry no long dash, no absolute path and no email address. PROJECT_SUMMARY.md,
  PIPELINE.md and CLAIMS.md, which keep the long dashes of their older text, carry no absolute path and no email.
- Every number in the prose of README.md is registered in README_NUMBERS with the file and the line it was copied
  from, and the numbers in the README phrase equal the numbers in the source phrase, character for character.
- A prototype number in README.md, PROJECT_SUMMARY.md or PIPELINE.md carries the tag [prototype] on its own line.
  This is the grep check of M8. The labelling convention is stated at PROTOTYPE_TAG below. What this check can
  see: a number that is registered in PROTOTYPE_NUMBERS, in its registered spelling or in one of the respellings
  of PROTOTYPE_RESPELLINGS, in those three files. It cannot see a prototype number that nobody registered, a
  spelling nobody thought of, a second number on a line whose tag belongs to another number (only the tag count
  of MIN_TAGS guards that), or any other file (CLAIMS.md row A2 holds untagged prototype numbers and is not read
  here). New text in a public document still needs a read by a person or a reviewer.
- README.md says that the export, bridge, rebuild and web paths are not on the branch for as long as one of them
  is missing, and stops saying so once all of them exist.
- docs/SCRIPT.md: the copy and the labels hold no digit, no word from the jargon list and no forbidden claim;
  every token is listed, resolves to a field of the site data contract (export/schemas.py) and resolves to one
  value on the synthetic data of export/sample.py; each scene's reading grade is 9 or lower and equals the grade
  the file states, and no single line other than the headline is above 9, with a token read as one syllable or
  as four.

Nothing here reads DATA_ROOT or needs a package beyond pydantic. Every test can fail.
"""

import json
import re
import types
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from export import schemas as S

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
SCRIPT = REPO / "docs" / "SCRIPT.md"
PROJECT_SUMMARY = REPO / "PROJECT_SUMMARY.md"
PIPELINE = REPO / "PIPELINE.md"
CLAIMS = REPO / "CLAIMS.md"

HEADLINE = "From broadcast pixels to possession value, measured against ground truth at every step"  # PLAN.md D17


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------- dashes, paths, addresses

LONG_DASHES = {chr(0x2012): "figure dash", chr(0x2013): "en dash", chr(0x2014): "em dash",
               chr(0x2015): "horizontal bar"}
# A path that starts at the root of a machine. export.schemas.find_private covers home folders, cloud mounts and
# email addresses; this adds the system roots, so a document cannot name any location on a particular machine.
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w.$~/])/(?:Users|home|root|private|var|tmp|opt|usr|etc|Volumes|content|mnt|srv|bin|sbin|Library"
    r"|Applications|System)(?=/|\b)|\b[A-Za-z]:\\")


def hygiene_problems(text: str) -> list[str]:
    """Long dashes, absolute paths and email addresses in a document, one line of text per finding."""
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        for char, name in LONG_DASHES.items():
            if char in line:
                problems.append(f"line {number}: {name}")
        for found in S.find_private(line):
            problems.append(f"line {number}: private string {found!r}")
        for match in ABSOLUTE_PATH_RE.finditer(line):
            problems.append(f"line {number}: absolute path {match.group(0)!r}")
    return problems


@pytest.mark.parametrize("path", [README, SCRIPT], ids=["README.md", "docs/SCRIPT.md"])
def test_document_has_no_long_dash_no_absolute_path_and_no_email(path):
    assert hygiene_problems(read(path)) == []


@pytest.mark.parametrize("path", [PROJECT_SUMMARY, PIPELINE, CLAIMS], ids=lambda path: path.name)
def test_labelled_and_amended_documents_have_no_absolute_path_and_no_email(path):
    """These three keep the long dashes of their older text (only labels and one amendment were added), so the
    dash findings are left out here."""
    assert [problem for problem in hygiene_problems(read(path)) if not problem.endswith(" dash")] == []


@pytest.mark.parametrize("line, bad", [
    ("plain text with a hyphen-ated word and a range 1 to 2", False),
    ("run python3 -m export.sample --out web/data", False),
    ("see reports/scorecard/gsw_phx_2016__r23.txt and web/opt/x", False),
    ("an em dash " + chr(0x2014) + " here", True),
    ("an en dash " + chr(0x2013) + " here", True),
    ("open /Users/someone/project", True),
    ("use /usr/local/bin/python3", True),
    ("use /opt/anaconda3/bin/python", True),
    ("mail someone@example.com", True),
    ("upload to My Drive", True),
])
def test_hygiene_check_finds_what_it_should(line, bad):
    assert bool(hygiene_problems(line)) is bad


def test_this_file_holds_no_long_dash_and_no_private_string():
    text = read(Path(__file__))
    assert not any(char in text for char in LONG_DASHES)
    # the parametrized examples above are the only private-looking strings this file may hold
    allowed = {"/Users/", "/opt/anaconda", "someone@example.com", "My Drive"}
    assert set(S.find_private(text)) <= allowed


# -------------------------------------------------------------------------------------------- README: numbers

# One entry per phrase of README.md that holds a number: (phrase in README.md, source file, line in that file,
# phrase in that line). The numbers of the two phrases must be equal, in order, character for character, so a
# value cannot be rounded or retyped differently. Every number in the prose of README.md must sit inside one of
# these phrases. Numbers inside code spans and code blocks are names (files, builds, tags, commands), not values.
R23_GAME = "reports/scorecard/gsw_phx_2016__r23.txt"
R23_NYK = "reports/scorecard/cle_nyk_2015__r23.txt"
R23_CLE = "reports/scorecard/cle_gsw_2016__r23.txt"
V3_GAME = "reports/scorecard/gsw_phx_2016__v3.txt"
RESULTS = "reports/results.md"
README_NUMBERS = (
    # the short version
    ("the 2015-16 season", RESULTS, 11, "the 2015-16 SportVU season"),
    ("median position error is 1.92 ft", R23_GAME, 7, "p50 1.92 ft"),
    # the scorecard table: the rows that are judged on every game
    ("<= 2.0 / 5.0 ft", R23_GAME, 8, "target p50 <= 2.0 ft, p90 <= 5.0 ft"),
    ("1.92 / 4.98 ft", R23_GAME, 7, "p50 1.92 ft, p90 4.98 ft"),
    ("2.25 / 5.46 ft", R23_NYK, 7, "p50 2.25 ft, p90 5.46 ft"),
    ("2.22 / 5.5 ft", R23_CLE, 7, "p50 2.22 ft, p90 5.5 ft"),
    ("within +-0.5 ft", R23_GAME, 10, "within +-0.5 ft in the near and the far third"),
    ("-0.67 / +1.52 ft", R23_GAME, 9, "near -0.67 ft, far +1.52 ft"),
    ("-0.63 / +1.66 ft", R23_NYK, 9, "near -0.63 ft, far +1.66 ft"),
    ("-0.12 / +2.00 ft", R23_CLE, 9, "near -0.12 ft, far +2.00 ft"),
    (">= 50%", R23_GAME, 12, "target >= 50% of live wide seconds"),
    ("99.3%", R23_GAME, 11, "MET       99.3%"),
    ("92.3%", R23_NYK, 11, "MET       92.3%"),
    ("39.7%", R23_CLE, 11, "not met   39.7%"),
    ("<= 5%", R23_GAME, 14, "target <= 5%"),
    ("<= 5%", R23_GAME, 16, "target <= 5%, referees excluded by class"),
    ("25.1%", R23_GAME, 13, "not met   25.1%"),
    ("24.4%", R23_NYK, 13, "not met   24.4%"),
    ("26.2%", R23_CLE, 13, "not met   26.2%"),
    ("15.1%", R23_GAME, 15, "not met   15.1% non-referee"),
    ("14.8%", R23_NYK, 15, "not met   14.8% non-referee"),
    ("14.8%", R23_CLE, 15, "not met   14.8% non-referee"),
    (">= 95%", R23_GAME, 18, "target >= 95%"),
    ("81.9%", R23_GAME, 17, "not met   81.9%"),
    ("73.7%", R23_NYK, 17, "not met   73.7%"),
    ("78.2%", R23_CLE, 17, "not met   78.2%"),
    # the four facts under the table
    ("1447 of 9500 window frames", R23_GAME, 2, "1447 testable of 9500 window frames"),
    ("Its p90 is 4.98 ft", "reports/r23_verification.txt", 9, "phx p90 4.98 ft"),
    ("against 5.0 ft", R23_GAME, 8, "p90 <= 5.0 ft"),
    ("95% interval", "reports/r23_verification.txt", 9, "window-bootstrap 95% CIs"),
    ("[4.42, 6.02]", "reports/r23_verification.txt", 9, "CI [4.42, 6.02]"),
    ("trained on 294 of\n  its frames", "reports/sportvu_splits.txt", 15, "'gsw_phx_2016': 294"),
    # what the models found
    ("95 percent", RESULTS, 6, "Intervals are 95 percent percentile intervals of a bootstrap by game"),
    ("0.6232 [0.6132, 0.6327]", RESULTS, 25, "0.6232 [0.6132, 0.6327]"),
    ("against 0.6735", RESULTS, 64, "0.6735 held out"),
    ("1.2362 [1.1899, 1.2812]", RESULTS, 137, "1.2362 [1.1899, 1.2812] held out"),
    ("against 1.3319", RESULTS, 138, "against 1.3319"),
    ("-0.014 [-0.116, 0.100]", RESULTS, 32, "-0.014 [-0.116, 0.100]"),
    # the tracking measurement in detail: the rest of the scorecard
    ("held-out arena within 1.5x of the held-out game", R23_GAME, 22,
     "target held-out arena within 1.5x of the held-out game"),
    ("p50 1.17x, p90 1.10x", R23_NYK, 21, "p50 1.17x, p90 1.10x"),
    ("p50 1.16x, p90 1.10x", R23_CLE, 21, "p50 1.16x, p90 1.10x"),
    ("21.39 id switches / possession", R23_GAME, 19, "21.39 id switches / possession"),
    ("31.07", R23_NYK, 19, "31.07 id switches / possession"),
    ("14.29", R23_CLE, 19, "14.29 id switches / possession"),
    ("1447 of 9500", R23_GAME, 2, "1447 testable of 9500 window frames"),
    ("485 of 6393", R23_NYK, 2, "485 testable of 6393 window frames"),
    ("158 of 3380", R23_CLE, 2, "158 testable of 3380 window frames"),
    ("untestable 84.8%", R23_GAME, 2, "untestable 84.8%"),
    # the tracking measurement in detail: splits, builds, limits
    ("4 games", "reports/sportvu_splits.txt", 2, "| 4 games |"),
    ("p50 1.92 ft, p90 4.98 ft", R23_GAME, 7, "p50 1.92 ft, p90 4.98 ft"),
    ("p50 7.33 ft,\n  p90 15.35 ft", V3_GAME, 7, "p50 7.33 ft, p90 15.35 ft"),
    ("on 2026-10-04", "CLAIMS.md", 43, "adopted 2026-10-04 under the amended rule"),
    ("56 frames of 2023-24 Cleveland home games", "CLAIMS.md", 84, "holds 56 frames of 2023-24 Cleveland home games"),
    ("56 frames of 2023-24 Cleveland home games", "reports/r23_verification.txt", 7,
     "56 frames of 2023-24 Cleveland home games"),
    ("Cleveland as 9 clips / 88 images", "ROADMAP_LOG.md", 126, "Cleveland is 9 clips / 88 images"),
    ("about 1.5 ft median at best", "CLAIMS.md", 72, "about 1.5 ft median at best"),
    ("with 14 cameras", "CLAIMS.md", 72, "(14 cameras, about"),
    ("about\n  1 inch", "CLAIMS.md", 73, "1 inch)"),
    ("about 2.35 / 5.9 ft, not this model's 2.24 / 5.47", "ROADMAP_LOG.md", 129,
     "~2.35 / ~5.9 ft, not r23's 2.24 / 5.47"),
    # the models in detail: the split, then cells copied from the summary table of reports/results.md
    ("604 train and 17 held out", RESULTS, 11, "of which 604 train and 17 are held out"),
    ("hold out\nthe 2024-25 season", RESULTS, 159, "2024-25 held out"),
    ("works; beats the location-only baseline by 0.050 on held-out; held-out calibration worse (ECE 0.019)", RESULTS,
     25, "works; beats the location-only baseline by 0.050 on held-out; held-out calibration worse (ECE 0.019)"),
    ("| 1.7115 | works, far above the 2.39 floor; held-out worse by 0.11 |", RESULTS, 26,
     "| 1.7115 | works, far above the 2.39 floor; held-out worse by 0.11 |"),
    ("| 1.3195 vs constant 1.3251 | assignment is stable (0.82); no defender signal beats the constant out of fold |",
     RESULTS, 27,
     "| 1.3195 vs constant 1.3251 | assignment is stable (0.82); no defender signal beats the constant out of fold |"),
    ("| 29.91 (skill 0.434) | expectation holds on held-out; teammates of top-gravity players shoot slightly better "
     "(+0.004, training games) |", RESULTS, 28,
     "| 29.91 (skill 0.434) | expectation holds on held-out; teammates of top-gravity players shoot slightly better "
     "(+0.004, training games) |"),
    ("| 3.417 ft | works, best on held-out; a deviation sign test agrees on 13 of 17 held-out games |", RESULTS, 29,
     "| 3.417 ft | works, best on held-out; a deviation sign test agrees on 13 of 17 held-out games |"),
    ("| none by design | passes its 0.60 gate; no held-out score exists |", RESULTS, 30,
     "| none by design | passes its 0.60 gate; no held-out score exists |"),
    ("| 1.2362 [1.1899, 1.2812] | works; 5 percent better than the constant out of fold, 7 percent on held-out |",
     RESULTS, 31,
     "| 1.2362 [1.1899, 1.2812] | works; 5 percent better than the constant out of fold, 7 percent on held-out |"),
    ("| -0.014 [-0.116, 0.100] | **failed**: no validated signal in either sample |", RESULTS, 32,
     "| -0.014 [-0.116, 0.100] | **failed**: no validated signal in either sample |"),
    ("| 0.065 [-0.488, 0.482] | **inconclusive**: about 20 recurring lineups a season, every interval spans zero |",
     RESULTS, 33,
     "| 0.065 [-0.488, 0.482] | **inconclusive**: about 20 recurring lineups a season, every interval spans zero |"),
    ("| 0.1486; log loss 0.4879 [0.4700, 0.5077] | works; held-out slightly worse and slightly over-confident |",
     RESULTS, 34,
     "| 0.1486; log loss 0.4879 [0.4700, 0.5077] | works; held-out slightly worse and slightly over-confident |"),
    ("| 0.5164; 0.2776 | absence works; decline barely beats the mean on held-out |", RESULTS, 35,
     "| 0.5164; 0.2776 | absence works; decline barely beats the mean on held-out |"),
    # the broadcast pipeline
    ("fixed at 42", "config.py", 25, "SEED = 42"),
    ("standing caveat 1", "CLAIMS.md", 161, "1. **Gate operating point:**"),
    ("standing caveat 3", "CLAIMS.md", 178, "3. **Identity = track fragment**"),
    # licence
    ("standing caveat 4", "CLAIMS.md", 179, "4. **Data provenance:**"),
    ("amended 2026-11-14", "CLAIMS.md", 182, "Amended 2026-11-14"),
)

# A number in running text: a date, a season, or a plain number with its sign and its percent or ratio mark.
# A digit that is part of a name (p50, D17, r23, gsw_phx_2016) is not a number: the look-behind skips it.
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d{4}-\d{2}-\d{2}|\d{4}-\d{2}(?!\d)|[+-]{0,2}\d+(?:,\d{3})*(?:\.\d+)?)(?:%|x\b)?")


def numbers_in(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def prose_of(markdown: str) -> str:
    """A Markdown document with its code removed: fenced blocks, code spans and link targets become spaces, so
    what is left is the running text and the tables. Line breaks and positions are kept."""
    lines, fenced = [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            lines.append(" " * len(line))
            continue
        lines.append(" " * len(line) if fenced else line)
    text = "\n".join(lines)
    text = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"\]\([^)\n]*\)", lambda m: "]" + " " * (len(m.group(0)) - 1), text)
    return re.sub(r"(?m)^(\s*)\d+\.(?=\s)", lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), text)


def code_names_with_digits(markdown: str) -> list[str]:
    """The code spans of a document that hold a digit: the names the number check deliberately leaves out."""
    spans = re.findall(r"`([^`\n]*)`", "\n".join(
        line for line in markdown.split("\n") if not line.lstrip().startswith("```")))
    return sorted({span for span in spans if re.search(r"\d", span)})


def unregistered_numbers(markdown: str, registry) -> list[str]:
    """Numbers in the prose of `markdown` that no registered phrase covers, each with its line number."""
    prose = prose_of(markdown)
    covered = []
    for phrase in {entry[0] for entry in registry}:
        start = prose.find(phrase)
        while start != -1:
            covered.append((start, start + len(phrase)))
            start = prose.find(phrase, start + 1)
    loose = []
    for match in NUMBER_RE.finditer(prose):
        if not any(a <= match.start() and match.end() <= b for a, b in covered):
            loose.append(f"line {prose.count(chr(10), 0, match.start()) + 1}: {match.group(0)}")
    return loose


def test_every_registered_readme_number_equals_its_source():
    prose = prose_of(read(README))
    problems = []
    for phrase, source, line_number, source_phrase in README_NUMBERS:
        if phrase not in prose:
            problems.append(f"README.md no longer holds {phrase!r}")
        lines = read(REPO / source).split("\n")
        line = lines[line_number - 1] if line_number <= len(lines) else ""
        if source_phrase not in line:
            problems.append(f"{source}:{line_number} does not hold {source_phrase!r}")
        if not numbers_in(phrase) or numbers_in(phrase) != numbers_in(source_phrase):
            problems.append(f"{phrase!r} holds {numbers_in(phrase)}, {source}:{line_number} holds "
                            f"{numbers_in(source_phrase)}")
    assert problems == []


def test_every_number_in_the_readme_prose_is_registered():
    assert unregistered_numbers(read(README), README_NUMBERS) == []


def test_the_number_check_can_fail():
    registry = (("p50 1.92 ft", "x", 1, "p50 1.92 ft"),)
    assert unregistered_numbers("Position error p50 1.92 ft on build `r23`, see D17 and gsw_phx_2016.", registry) == []
    assert unregistered_numbers("Position error p50 1.93 ft.", registry) == ["line 1: 1.93"]
    assert unregistered_numbers("first line\nIt was 25.1% on 2026-10-04, 1.5x.", registry) == [
        "line 2: 25.1%", "line 2: 2026-10-04", "line 2: 1.5x"]
    assert unregistered_numbers("```\npython3 tool.py --n 200\n```\nand [a link](path/v2/file_3.md)", registry) == []
    assert numbers_in("near -0.67 ft, far +1.52 ft, within +-0.5 ft, 5.5 not 5.50") == [
        "-0.67", "+1.52", "+-0.5", "5.5", "5.50"]
    assert numbers_in("p50 and p90 of r23 in 2015-16, 11,292 players") == ["2015-16", "11,292"]


def markdown_table(text: str, first_header: str) -> list[list[str]]:
    """The rows of the Markdown table whose first header cell is `first_header`: the header row, then the body
    rows, each as a list of cells. The rule row under the header is left out."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not all(cell and set(cell) <= set("-:") for cell in cells):
            rows.append(cells)
    return rows


SCORECARD_GAMES = ["gsw_phx_2016", "cle_nyk_2015", "cle_gsw_2016"]
# The rows of the scorecard table at the top of README.md: (how the row label starts, key under metrics in the
# scorecard report, the value part of a cell). Shares are printed as the scorecard text prints them.
JUDGED_ROWS = (
    ("Position error", "position_error", lambda m: f"{m['p50_ft']} / {m['p90_ft']} ft"),
    ("Near-field bias", "near_field_bias", lambda m: f"{m['near_third_ft']:+.2f} / {m['far_third_ft']:+.2f} ft"),
    ("Coverage", "coverage", lambda m: f"{m['coverage'] * 100:.1f}%"),
    ("Missed players", "missed_players", lambda m: f"{m['rate'] * 100:.1f}%"),
    ("Ghost boxes", "ghost_boxes", lambda m: f"{m['rate'] * 100:.1f}%"),
    ("Team labels", "team_labels", lambda m: f"{m['accuracy'] * 100:.1f}%"),
)


def _scorecard(game: str) -> dict:
    return json.loads(read(REPO / "reports" / "scorecard" / f"{game}__r23.json"))


def test_readme_scorecard_tables_equal_the_scorecard_reports_cell_by_cell():
    """The status (met or not met) and the value of every cell, placed under the right game. The number registry
    above cannot see a value that sits in the wrong column or carries the wrong status; this can."""
    readme = read(README)
    problems = []
    top = markdown_table(readme, "Row and target")
    assert [re.search(r"`(\w+)`", cell).group(1) for cell in top[0][1:]] == SCORECARD_GAMES
    assert len(top) - 1 == len(JUDGED_ROWS)
    for (label, key, value), row in zip(JUDGED_ROWS, top[1:]):
        assert row[0].startswith(label), row[0]
        for game, cell in zip(SCORECARD_GAMES, row[1:]):
            metric = _scorecard(game)["metrics"][key]
            want = ("**met**" if metric["pass"] else "not met") + ": " + value(metric)
            if cell != want:
                problems.append(f"{label}, {game}: README.md says {cell!r}, the scorecard says {want!r}")
    rest = {row[0]: row[2:] for row in markdown_table(readme, "Row")[1:]}
    for index, game in enumerate(SCORECARD_GAMES):
        card = _scorecard(game)
        ratio = card["metrics"]["generalisation"]
        identity = card["metrics"]["identity"]["id_switches_per_possession"]
        want = {
            "Generalisation": "the reference" if ratio["value"] is None else
            ("**met**" if ratio["pass"] else "not met")
            + f": p50 {ratio['value']['p50_ratio']:.2f}x, p90 {ratio['value']['p90_ratio']:.2f}x",
            "Identity": f"{identity} id switches / possession" if index == 0 else f"{identity}",
            "Court line error": "not measurable" if card["metrics"]["court_line_error"]["pass"] is None else "?",
            "Frames with usable truth": f"{card['testable']} of {card['truth_frames']}",
        }
        for label, cell in want.items():
            if rest[label][index] != cell:
                problems.append(f"{label}, {game}: README.md says {rest[label][index]!r}, the scorecard says {cell!r}")
    assert problems == []


def test_readme_model_table_equals_the_results_summary_row_by_row():
    """Each lane's held-out cell and verdict cell are the cells of the same lane in reports/results.md."""
    ours = markdown_table(read(README), "Lane")
    theirs = {row[0]: row for row in markdown_table(read(REPO / RESULTS), "lane")[1:]}
    assert [row[0] for row in ours[1:]] == list(theirs), "the lanes of reports/results.md, in its order"
    problems = []
    for lane, _question, _metric, heldout, verdict in ours[1:]:
        if (heldout, verdict) != (theirs[lane][4], theirs[lane][5]):
            problems.append(f"{lane}: README.md says {(heldout, verdict)}, reports/results.md says "
                            f"{(theirs[lane][4], theirs[lane][5])}")
    assert problems == []


def test_readme_opens_with_the_headline_and_has_one_site_line():
    lines = read(README).split("\n")
    assert lines[0] == "# " + HEADLINE
    assert sum(line.startswith("**Site:**") for line in lines) == 1
    assert not any("PLACEHOLDER" in line for line in lines), "a placeholder would be the third line a reader sees"


# Paths the README describes that other teammates build (PLAN.md D39). While one is missing, README.md must say
# so in a section with this heading; once all exist, the section must go, so that the README never says that
# something is missing when it is there.
BUILT_ELSEWHERE = ("export/site_data.py", "export/demo_bridge.py", "tools/frozen_models.py", "web/package.json")
STATUS_HEADING = "## Status of this branch"


def readme_status_problem(text: str, missing: list[str]) -> str:
    """Why the status section of README.md is wrong for the paths that are missing, or an empty string."""
    has_section = any(line == STATUS_HEADING for line in text.split("\n"))
    if missing and not has_section:
        return f"README.md describes {missing} as present; add the section {STATUS_HEADING!r}"
    if not missing and has_section:
        return f"every path of BUILT_ELSEWHERE exists now: remove the section {STATUS_HEADING!r} from README.md"
    return ""


def test_readme_says_what_is_not_on_the_branch_yet():
    missing = [path for path in BUILT_ELSEWHERE if not (REPO / path).exists()]
    assert readme_status_problem(read(README), missing) == ""
    section = read(README).split(STATUS_HEADING + "\n", 1)
    if missing:
        body = section[1].split("\n## ", 1)[0]
        for name in ("export/site_data.py", "export/demo_bridge.py", "tools/frozen_models.py", "web/"):
            assert f"`{name}`" in body, f"the status section does not name {name}"


def test_the_status_check_can_fail():
    with_section = "# title\n\n" + STATUS_HEADING + "\n\ntext\n\n## Run it\n"
    assert readme_status_problem(with_section, ["web/package.json"]) == ""
    assert readme_status_problem("# title\n\n## Run it\n", []) == ""
    assert "add the section" in readme_status_problem("# title\n\n## Run it\n", ["web/package.json"])
    assert "remove the section" in readme_status_problem(with_section, [])


def test_readme_links_point_at_files_that_exist():
    targets = re.findall(r"\]\(([^)#\s]+)\)", read(README))
    assert targets, "README.md links to nothing"
    missing = [target for target in targets if not re.match(r"https?://", target) and not (REPO / target).exists()]
    assert missing == []


# ------------------------------------------------------------------------------------- prototype numbers (M8)

# The labelling convention. A prototype number is a number measured during the prototype stage: on the 7
# prototype clips the models were tuned on, on hand labels drawn from those clips, on a model's own validation
# split, or by a label-free check on a few extra videos. It says nothing about tracking accuracy on held-out
# footage. In README.md, PROJECT_SUMMARY.md and PIPELINE.md, every line that holds a prototype number also holds
# the tag below, placed after the number; in a table row, every cell that holds one holds the tag itself. One
# exception, stated in the note of PROJECT_SUMMARY.md: in its detector table every number of a row is a prototype
# number and the tag in the first cell covers the row. The same thing as a shell check, for one pattern:
#     grep -n '98\.7%' README.md PROJECT_SUMMARY.md PIPELINE.md | grep -v '\[prototype\]'      (must print nothing)
PROTOTYPE_TAG = "[prototype]"
PUBLIC_DOCS = (README, PROJECT_SUMMARY, PIPELINE)
# How many times each labelled document held the tag when it was labelled (2026-11-14), the note that explains
# the tag included. A tag may be added, never dropped: where two numbers share a line, the rule above is still
# satisfied after one of their tags is removed, and this count is what notices.
MIN_TAGS = {"PROJECT_SUMMARY.md": 48, "PIPELINE.md": 20}
# The prototype numbers of PROJECT_SUMMARY.md and PIPELINE.md, by stage. Bare values where the value is
# distinctive; a value with its wording where the bare value could also be a harvest number or a setting.
PROTOTYPE_NUMBERS = (
    # gate: test split of the prototype frames
    r"(?<![\d.])98\.7%", r"macro-F1 0\.986", r"1 FN / 1 FP", r"\(0\.886\)", r"\(0\.873\)", r"\(88\.6%\)",
    r"\(87\.3%\)",
    # detection: 50 hand-labelled prototype frames, and the detector's own validation split
    r"P \.89 / R \.87", r"P 0\.68 / R 0\.74 / F1 0\.71", r"\| 0\.68 \| 0\.74 \| 0\.71 \|",
    r"\*\*0\.89\*\* \| \*\*0\.87\*\* \| \*\*0\.88\*\*", r"0\.71 (?:\u2192|to) 0\.88", r"0\.68 \u2192 0\.89",
    r"mAP@50", r"~0?\.74\b", r"(?<![\d.])0\.695\b", r"\+2% recall",
    # court model: validation frames, a label-free check on other videos, the line model's validation split
    r"(?<![\d.])0\.30 ft", r"(?<![\d.])1\.7 px", r"100% homography success", r"100% success on 28",
    r"66\u201389% sane-H", r"Dice 0\.587", r"~3\u00d7 better refinement", r"p99 ~40 ft",
    # identity: label-free figures on a prototype clip
    r"impossible steps\W+15\.4", r"(?<![\d.])6\.57\b", r"churn (?:still|remains) ~5\u00d7",
    # teams: hand labels on one prototype clip
    r"(?<![\d.])87\.1%", r"(?<![\d.])83\.9%", r"~13% track misassignment",
    # possessions: hand labels on four prototype clips
    r"(?<![\d.])96\.5%", r"(?<![\d.])91\.2%", r"100% ?/ ?100%", r"\u00b11\u20132 s",
    # matchups: prototype clips after the ghost-track audit
    r"2\.5\u20134\.0 pairs/frame", r"7\u201362%", r"20\u201330% invented presence",
    # clock, alignment, join: the validated prototype clips
    r"100%(?:\*\*)?[ -]on[ -]readable", r"0 confident-wrong", r"(?<![\d/])9/9\b", r"100% overlap",
    r"~10% anchor failure", r"(?<![\d/])7/7\b", r"(?<![\d/])10/12\b", r"(?<![\d/])7/9\b", r"~58%",
    r"58% span-to-join",
    # the teacher comparison: the same 50 hand-labelled prototype frames
    r"P 0\.70 / R 0\.94", r"P 0\.956 / R 0\.967", r"27 vs\. 15 FN",
)


# Other spellings of the same numbers that a later edit could bring in (CLAIMS.md row A2 spells two of them this
# way). They are checked like the patterns above, and they need not match anything today.
PROTOTYPE_RESPELLINGS = (
    r"98\.7 ?(?:percent|per cent)", r"(?<![\d.])0\.987\b", r"P ?\.89 ?/ ?R ?\.87", r"(?<![\d.])0\.3 ft",
    r"87\.1 ?(?:percent|per cent)", r"96\.5 ?/ ?91\.2 ?%", r"96\.5 ?(?:percent|per cent)",
    r"91\.2 ?(?:percent|per cent)",
)


def unlabelled_prototype_numbers(text: str) -> list[str]:
    """Lines of a document that hold a prototype number and no tag. A table row is checked as a whole and then
    cell by cell, so a tag in one cell does not cover a number in another."""
    hits = []
    for number, line in enumerate(text.split("\n"), 1):
        parts = [line] + (line.split("|") if line.lstrip().startswith("|") else [])
        for part in parts:
            if PROTOTYPE_TAG in part:
                continue
            found = next((m for m in (re.search(pattern, part)
                                      for pattern in PROTOTYPE_NUMBERS + PROTOTYPE_RESPELLINGS) if m), None)
            if found:
                hits.append(f"line {number}: {found.group(0)}")
                break
    return hits


@pytest.mark.parametrize("path", PUBLIC_DOCS, ids=[p.name for p in PUBLIC_DOCS])
def test_no_unlabelled_prototype_number(path):
    text = read(path)
    assert unlabelled_prototype_numbers(text) == []
    assert text.count(PROTOTYPE_TAG) >= MIN_TAGS.get(path.name, 0), "a [prototype] tag was removed"


@pytest.mark.parametrize("line, flagged", [
    ("**Result: 98.7% test accuracy** (macro-F1 0.986)", True),
    ("**Result: 98.7% test accuracy** [prototype] (macro-F1 0.986)", False),
    ("| 3 | Homography | H-err **p50 1.7 px**, 279 val frames |", True),
    ("| 3 | Homography | H-err **p50 1.7 px**, 279 val frames [prototype] |", False),
    ("| 5 | Teams | **87.1% track**, 160 labels | ~13% track misassignment [prototype] |", True),
    ("| 5 | Teams | **87.1% track**, 160 labels [prototype] | ~13% track misassignment [prototype] |", False),
    ("| COCO person (baseline) [prototype] | 0.68 | 0.74 | 0.71 |", False),
    ("| COCO person (baseline) | 0.68 | 0.74 | 0.71 |", True),
    ("Teams: 87.1% track-level accuracy", True),
    ("homography 0.30 ft median", True),
    ("position error p50 1.92 ft, p90 4.98 ft on the held-out game", False),
    ("log loss 0.6232 [0.6132, 0.6327]; per-miss log loss 1.7115", False),
    ("agreement with play-by-play 87.8% (n=990)", False),
    ("gate 98.7 percent on the test split", True),
    ("gate accuracy 0.987", True),
    ("detection P.89/R.87; homography 0.3 ft median", True),
    ("possessions 96.5/91.2%", True),
    ("detection P.89/R.87 [prototype]", False),
    ("the truth fit has a 0.33 ft median residual on the feet", False),
])
def test_prototype_check_finds_what_it_should(line, flagged):
    assert bool(unlabelled_prototype_numbers(line)) is flagged


def test_every_prototype_pattern_is_alive_and_the_tag_is_explained():
    both = read(PROJECT_SUMMARY) + "\n" + read(PIPELINE)
    dead = [pattern for pattern in PROTOTYPE_NUMBERS if not re.search(pattern, both)]
    assert dead == [], "patterns that match nothing in PROJECT_SUMMARY.md or PIPELINE.md"
    for path in (PROJECT_SUMMARY, PIPELINE):
        text = read(path)
        assert "A number followed by `[prototype]` is a prototype number." in text, path.name
        assert "It is" in text and "not tracking accuracy" in " ".join(text.split()), path.name
    assert "`[prototype]`" in read(README)


# ----------------------------------------------------------------------------- docs/SCRIPT.md: reading grade

TOKEN_RE = re.compile(r"\{([a-z0-9_]+(?:\.[a-z0-9_]+)+)\}")
# Syllable counts the rule below gets wrong, each checked by hand.
SYLLABLE_OVERRIDES = {"video": 3, "area": 3, "idea": 3, "create": 2, "science": 2, "league": 1, "leagues": 1,
                      "nba": 3, "tv": 2}


def count_syllables(word: str) -> int:
    """Syllables of one word: groups of vowels, less a silent final e, es or ed. A word with no letter (a numeral)
    counts as one syllable."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    if w in SYLLABLE_OVERRIDES:
        return SYLLABLE_OVERRIDES[w]
    # y is a consonant at the start of a word and between vowels (player, beyond), a vowel elsewhere
    s = "".join("j" if c == "y" and (i == 0 or w[i - 1] in "aeiou") and w[i + 1:i + 2] in tuple("aeiou") else c
                for i, c in enumerate(w))
    n = len(re.findall(r"[aeiouy]+", s))
    if re.search(r"[^aeiou]ying$", s) or re.search(r"[aeiou]ing$", s):
        n += 1                                    # try-ing, be-ing, go-ing
    if re.search(r"[^aeiouy]ier$", s) and n > 1:
        n += 1                                    # eas-i-er
    if s.endswith("e") and n > 1:
        if s.endswith("le") and len(s) > 2 and s[-3] not in "aeiouy":
            pass                                  # ta-ble, sin-gle
        elif s[-2] not in "aeiouy":
            n -= 1                                # silent e after a consonant: frame, these, whole
    elif s.endswith("es") and n > 1:
        if not (re.search(r"(s|x|z|ch|sh|c|g)es$", s) or re.search(r"[^aeiouy]les$", s)):
            n -= 1                                # frames, games; not box-es, ta-bles
    elif s.endswith("ed") and n > 1:
        if not re.search(r"[td]ed$", s):
            n -= 1                                # moved, trained; not want-ed
    return max(1, n)


def words_of(text: str) -> list[str]:
    """The words of a piece of copy. A token is one word; a hyphenated word is one word per part."""
    words = []
    for raw in TOKEN_RE.sub(" TOKEN ", text).split():
        for part in raw.split("-"):
            core = re.sub(r"[^A-Za-z0-9']", "", part)
            if core:
                words.append(core)
    return words


def sentence_count(text: str) -> int:
    """Sentences of a piece of copy: runs of closing punctuation, plus a paragraph that ends without one."""
    total = 0
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        total += len(re.findall(r"[.!?]+(?=\s|$)", paragraph))
        if not re.search(r"[.!?]$", paragraph):
            total += 1
    return max(1, total)


def fk_grade(text: str, token_syllables: int = 1) -> float:
    """Flesch-Kincaid grade: 0.39 x (words per sentence) + 11.8 x (syllables per word) - 15.59."""
    words = words_of(text)
    syllables = sum(token_syllables if word == "TOKEN" else count_syllables(word) for word in words)
    return 0.39 * len(words) / sentence_count(text) + 11.8 * syllables / len(words) - 15.59


def test_syllable_counts_on_a_checked_vocabulary():
    checked = {"player": 2, "players": 2, "playing": 2, "trying": 2, "frame": 1, "frames": 1, "these": 1,
               "whole": 1, "table": 2, "tables": 2, "boxes": 2, "places": 2, "moved": 1, "trained": 1,
               "wanted": 2, "missed": 1, "checked": 1, "possession": 3, "value": 2, "measured": 2, "against": 2,
               "average": 3, "illustration": 4, "position": 3, "quality": 3, "broadcast": 2, "camera": 3,
               "sensor": 2, "people": 2, "single": 2, "league": 1, "true": 1, "video": 3, "arena": 3, "data": 2,
               "you": 1, "beyond": 2, "eye": 1, "percentage": 3, "defender": 3, "easier": 3, "being": 2,
               "typical": 3, "basketball": 3, "TV": 2, "NBA": 3, "1.92": 1}
    wrong = {word: (count_syllables(word), want) for word, want in checked.items() if count_syllables(word) != want}
    assert wrong == {}


def test_reading_grade_on_known_text():
    easy = "The cat sat on the mat. The dog ran to the man."
    hard = "Institutional considerations necessitate comprehensive organizational restructuring immediately."
    assert words_of(easy) == ["The", "cat", "sat", "on", "the", "mat", "The", "dog", "ran", "to", "the", "man"]
    assert sentence_count(easy) == 2
    assert fk_grade(easy) == pytest.approx(0.39 * 6 + 11.8 * 1 - 15.59)
    assert fk_grade(hard) > 20
    assert words_of("It is {swap.fill_share} percent.") == ["It", "is", "TOKEN", "percent"]
    assert fk_grade("It is {swap.fill_share} percent.", 4) > fk_grade("It is {swap.fill_share} percent.")
    assert sentence_count("A headline with no full stop\n\nA sentence. And another.") == 3


# ----------------------------------------------------------------------- docs/SCRIPT.md: scenes, copy, tokens

SCENE_RE = re.compile(r"^## Scene (\d+)\. (.+)$", re.M)
MAX_GRADE = 9.0
# Words that belong in the evidence layer and the presenter notes, never in the on-screen copy. The second item
# says whether the match is case-sensitive.
JARGON = (
    (r"homograph", False), (r"reprojection", False), (r"bootstrap", False), (r"Hungarian", False),
    (r"RANSAC", False), (r"YOLO", False), (r"BoT-SORT", False), (r"\bCLIP\b", True), (r"embedding", False),
    (r"confidence interval", False), (r"\bCI\b", True), (r"\bPBP\b", True), (r"\bxFG\b", False), (r"\bEPV\b", True),
    (r"ghost", False), (r"held[- ]out", False), (r"out[- ]of[- ]fold", False), (r"log loss", False),
    (r"\bMSE\b", True), (r"Brier", False), (r"calibrat", False), (r"percentile", False), (r"\bp50\b", False),
    (r"\bp90\b", False), (r"\bmedian\b", False), (r"ground truth", False),
)
# Claims the copy never makes (PLAN.md C10, D16; CLAUDE.md, Words).
FORBIDDEN = (r"\bbest\b", r"\boptimal", r"\bideal\b", r"\bshould\b", r"no model (?:saw|has seen|ever saw)",
             r"\bpasses\b", r"\bproves?\b")
TOKEN_FILES = {"scorecard": "scorecard.json", "swap": "demo/swap.json", "frames": "demo/frames.json",
               "manifest": "manifest.json", "players": "players.json", "results": "results.json"}


def block_of(body: str, label: str) -> str:
    """The text of one labelled block of a scene (Visual, Labels, ...), up to the next block label."""
    start = body.find(f"**{label}.**")
    if start == -1:
        return ""
    rest = body[start + len(label) + 5:]
    following = re.search(r"^\*\*[A-Z][A-Za-z ]+\.\*\*", rest, re.M)
    return rest[:following.start()] if following else rest


def script_scenes(text: str) -> list[dict]:
    """The scenes of docs/SCRIPT.md. The copy of a scene is its quoted lines: the scroll steps, a caption and
    the fallback lines. The labels are the double-quoted strings of the list under Labels: the words that sit on
    the picture."""
    starts = list(SCENE_RE.finditer(text))
    scenes = []
    for index, match in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1].start()
        else:
            following = re.search(r"^## ", text[match.end():], re.M)
            end = match.end() + following.start() if following else len(text)
        body = text[match.end():end]
        stated = re.search(r"^\*\*Reading grade\.\*\* (\d+\.\d)$", body, re.M)
        scenes.append({
            "number": int(match.group(1)), "title": match.group(2), "body": body,
            "copy": [line[1:].strip() for line in body.split("\n") if line.startswith(">")],
            "labels": [label for line in block_of(body, "Labels").split("\n") if line.startswith("- ")
                       for label in re.findall(r'"([^"]+)"', line.split(" (", 1)[0])],
            "visual": block_of(body, "Visual"),
            "stated_grade": float(stated.group(1)) if stated else None,
        })
    return scenes


def scene_stats(scene: dict) -> dict:
    """Words, sentences and grade of a scene's copy, and the highest grade of one line of it (the headline, which
    D17 fixes, left out), each with a token as one syllable and then as four."""
    text = "\n\n".join(scene["copy"])
    lines = [line for line in scene["copy"] if line != HEADLINE + "."]
    return {"words": len(words_of(text)), "sentences": sentence_count(text),
            "grade": round(fk_grade(text), 1), "upper": round(fk_grade(text, 4), 1),
            "hard": round(max(fk_grade(line) for line in lines), 1),
            "hard_upper": round(max(fk_grade(line, 4) for line in lines), 1)}


def token_table(text: str) -> dict:
    """The token table of docs/SCRIPT.md: token -> (data file, field path, shown as, copied from)."""
    section = text.split("\n## Tokens\n", 1)[1].split("\n## ", 1)[0]
    table = {}
    for line in section.split("\n"):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].startswith("`{"):
            token = cells[0].strip("`")[1:-1]
            assert token not in table, f"token listed twice: {token}"
            table[token] = (cells[1].strip("`"), cells[2].strip("`"), cells[3], cells[4])
    return table


def _unwrap(annotation):
    """The plain type inside Optional[X] and Annotated[X, ...]."""
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        elif origin in (typing.Union, types.UnionType):
            inner = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
            if len(inner) != 1:
                return annotation
            annotation = inner[0]
        else:
            return annotation


def _is_model(annotation) -> bool:
    return typing.get_origin(annotation) is None and isinstance(annotation, type) and issubclass(annotation, BaseModel)


def resolve_field(model, path: str):
    """Walk a field path such as builds[build=r23].rows[key=position_error].cells[game=x].values[rate] through
    the contract models. A selector in brackets picks a list item by the value of one of its fields, by position
    (a number, or k for the frame on screen), or a key of a dictionary. Returns the type at the end of the path."""
    current = model
    for segment in path.split("."):
        parsed = re.fullmatch(r"([a-z_0-9]+)(?:\[([^\]]+)\])?", segment)
        assert parsed, f"{path}: cannot read {segment!r}"
        name, selector = parsed.groups()
        assert _is_model(current), f"{path}: nothing to select {name} from"
        assert name in current.model_fields, f"{path}: {current.__name__} has no field {name}"
        annotation = _unwrap(current.model_fields[name].annotation)
        origin = typing.get_origin(annotation)
        if selector is None:
            current = annotation
        elif origin is list:
            (item,) = typing.get_args(annotation)
            if "=" in selector:
                key = selector.split("=", 1)[0]
                assert key in item.model_fields, f"{path}: {item.__name__} has no field {key}"
            else:
                assert selector.isdigit() or selector == "k", f"{path}: cannot select [{selector}] from a list"
            current = item
        elif origin is dict:
            current = typing.get_args(annotation)[1]
        else:
            raise AssertionError(f"{path}: {name} is neither a list nor a dictionary")
    return _unwrap(current)


def test_field_paths_resolve_against_the_contract():
    path = "builds[build=r23].rows[key=position_error].cells[game=gsw_phx_2016].values[p50_ft]"
    assert resolve_field(S.Scorecard, path) is float
    assert resolve_field(S.Scorecard, "builds[build=r23].games[game=gsw_phx_2016].testable") is int
    assert resolve_field(S.DemoSwap, "summaries[kind=xfg].mean_abs_diff") is float
    assert resolve_field(S.DemoFrames, "scenes[0].frames[k].error_ft") is float
    assert resolve_field(S.DemoSwap, "fill_share") is float                      # a share, declared with bounds
    assert resolve_field(S.PossessionIndex, "possessions[0].xfg") is float       # an optional share
    assert _unwrap(typing.Optional[typing.Annotated[float, "bounded"]]) is float
    assert _unwrap(int | None) is int
    for bad in ("builds[build=r23].rows[key=position_error].cells[game=x].values",   # a dictionary, not a value
                "builds[name=r23].rows", "builds.rows", "summaries[kind=xfg].mean_diff", "fill_share[0]"):
        with pytest.raises(AssertionError):
            end = resolve_field(S.Scorecard if bad.startswith("builds") else S.DemoSwap, bad)
            assert end in (int, float, str, bool), f"{bad} ends at {end}"


@pytest.fixture(scope="module")
def script():
    text = read(SCRIPT)
    return {"text": text, "scenes": script_scenes(text), "tokens": token_table(text)}


def test_script_has_the_scenes_of_the_plan_in_order(script):
    scenes = script["scenes"]
    assert [scene["number"] for scene in scenes] == list(range(7))
    for scene in scenes:
        for label in ("**Question.**", "**Visual.**", "**Labels.**", "**Copy.**", "**Numbers.**",
                      "**Evidence layer.**", "**Presenter notes.**", "**Reading grade.**"):
            assert label in scene["body"], (scene["number"], label)
        assert scene["copy"], scene["number"]
    assert scenes[0]["copy"][0] == HEADLINE + "."


def test_script_copy_reads_at_grade_nine_or_lower_and_states_its_grade(script):
    problems = []
    for scene in script["scenes"]:
        stats = scene_stats(scene)
        if stats["grade"] > MAX_GRADE or stats["upper"] > MAX_GRADE:
            problems.append(f"scene {scene['number']}: grade {stats['grade']} (upper {stats['upper']})")
        if stats["hard"] > MAX_GRADE or stats["hard_upper"] > MAX_GRADE:
            problems.append(f"scene {scene['number']}: one line reads at grade {stats['hard']} "
                            f"(upper {stats['hard_upper']})")
        if scene["stated_grade"] != stats["grade"]:
            problems.append(f"scene {scene['number']}: states {scene['stated_grade']}, the copy reads {stats['grade']}")
        row = (f"| {scene['number']}. {scene['title']} | {stats['words']} | {stats['sentences']} | "
               f"{stats['grade']} | {stats['upper']} | {stats['hard']} / {stats['hard_upper']} |")
        if row not in script["text"]:
            problems.append(f"the grade table lacks the row: {row}")
    assert problems == []


def test_the_hardest_line_of_a_scene_is_found():
    scene = {"copy": [HEADLINE + ".", "The cat sat. The dog ran.",
                      "Institutional considerations necessitate comprehensive organizational restructuring."]}
    stats = scene_stats(scene)
    assert stats["hard"] > 20 > stats["grade"]
    assert scene_stats({"copy": ["It is {swap.fill_share} percent."]})["hard_upper"] > 9 > scene_stats(
        {"copy": ["It is {swap.fill_share} percent."]})["hard"]


def test_script_copy_holds_no_digit_no_jargon_and_no_forbidden_claim(script):
    problems = []
    for scene in script["scenes"]:
        # every double-quoted string of a Visual block is a label, so no word reaches the picture unchecked
        for quoted in re.findall(r'"([^"]+)"', scene["visual"]):
            if quoted not in scene["labels"]:
                problems.append(f"scene {scene['number']}: the visual puts {quoted!r} on screen, not listed as a label")
        for line in scene["copy"] + scene["labels"]:
            plain = TOKEN_RE.sub(" ", line)
            if re.search(r"\d", plain):
                problems.append(f"scene {scene['number']}: a digit in {line!r}")
            for pattern, cased in JARGON:
                if pattern == "ground truth" and line == HEADLINE + ".":
                    continue                      # the headline is fixed by D17
                if re.search(pattern, plain, 0 if cased else re.I):
                    problems.append(f"scene {scene['number']}: jargon {pattern!r} in {line!r}")
            for pattern in FORBIDDEN:
                if re.search(pattern, plain, re.I):
                    problems.append(f"scene {scene['number']}: forbidden claim {pattern!r} in {line!r}")
    assert problems == []


def test_script_copy_keeps_the_honesty_constraints(script):
    """The constraints the copy must respect (PLAN.md D16, D17, D21, D27, D45, C6, C10). Each one is held by its
    tokens and by a loose pattern, so the wording can change and the statement cannot be dropped."""
    scenes = script["scenes"]
    copy = {scene["number"]: " ".join(scene["copy"]) for scene in scenes}
    everything = " ".join(copy.values()) + " " + " ".join(label for scene in scenes for label in scene["labels"])
    game = "{scorecard.adopted.heldout_game."

    def has(pattern: str, text: str) -> bool:
        return re.search(pattern, text, re.I) is not None

    def step_with(number: int, needle: str) -> str:
        (line,) = [line for line in scenes[number]["copy"] if needle in line and "Across all" not in line[:10]]
        return line

    # the demo is an illustration: the cold open asks a question and never calls itself a test
    assert has(r"opens with a question", copy[0]) and not has(r"\btest\b", copy[0])
    # the demo game: held out from the court and player models, with the frame filter named in the same step,
    # and the word training explained before it is used
    steps = [line for line in scenes[1]["copy"] if has(r"(never|not) trained on this game", line)]
    assert steps and all(has(r"filter", line) and has(r"training frames", line) for line in steps)
    assert has(r"learn from[^.]*\. That is called training", steps[0])
    # the public build says that it shows a still and how to see the clips
    assert has(r"still frame", copy[1]) and has(r"ask for access", copy[1])
    # the sensor data is the answer key and its source is named
    assert has(r"SportVU", copy[1]) and has(r"sensor data", copy[1])
    # the rows that miss their targets are stated as measured, each as its scorecard defines it
    for token in ("missed_players.rate}", "ghost_boxes.rate}", "team_labels.accuracy}"):
        assert game + token in copy[2]
    assert has(r"leaving out the referees,? " + re.escape(game + "ghost_boxes.rate}"), copy[2])
    assert has(r"\bat most " + re.escape(game + "team_labels.accuracy}"), copy[2])
    assert has(r"\b(none|neither|not one)\b[^.]*\btarget", copy[2])
    assert "not met" in scenes[2]["labels"]
    # position error: both values, the margin that crosses the target, and how little of the game could be checked
    assert game + "position_error.p50_ft}" in copy[4] and game + "position_error.p90_ft}" in copy[4]
    assert has(r"margin of error[^.]*past the target", copy[4]) and has(r"neither is safely met", copy[4])
    assert game + "testable}" in copy[4] and game + "truth_frames}" in copy[4]
    assert "met, not safely" in scenes[4]["labels"]
    # a scorecard value covers the scored stretches of the game, never the whole game
    assert has(r"scored stretches[^.]*only part of the game", copy[4])
    assert has(r"all the scored stretches of this game", step_with(4, game + "position_error.p50_ft}"))
    assert not has(r"whole[- ]game|game[- ]wide|across the whole game", everything)
    # the sensor position is a reference with an error of its own, not the truth
    assert not has(r"\btrue (spot|position|place)", everything)
    # the comparison: an illustration before its first number, n and the mean difference for each model, the
    # fill share, the dropped boxes, position error only
    assert has(r"\billustration\b[^.]*\bnot a test\b", copy[5])
    assert copy[5].index("illustration") < copy[5].index("{swap.")
    assert "Illustration" in scenes[5]["labels"]
    for kind in ("xfg", "epv", "ghosting"):
        assert "{swap." + kind + ".mean_abs_diff}" in copy[5] and "{swap." + kind + ".n}" in copy[5]
    assert "{swap.fill_share}" in copy[5] and has(r"position error (alone|only)", copy[5])
    assert has(r"boxes that matched no player were dropped", copy[5])
    # the models were rebuilt from the locked code and checked (D45, D49); the copy does not say otherwise
    assert has(r"built again", copy[5]) and has(r"only if it gave back its saved answers", copy[5])
    assert not has(r"nothing was retrained|never retrained", everything)
    # the question of the cold open gets its answer, and the model results name both of their sources
    assert has(r"can the broadcast stand in yet\? No\b", copy[5])
    assert has(r"sensor data or from play-by-play records, not from the broadcast", copy[5])
    # an average defense, never more than that
    assert has(r"average defender", copy[5]) and has(r"average defense", copy[6])


# The builds of the site's scorecard by role (PLAN.md D42), and the two sentences of D42 that say so.
SITE_BUILDS = {"adopted": "r23", "candidate": "r23_cam"}
D42_SAYS = ("use build r23 (adopted", "r23_cam is shown beside it as a measured candidate")


def heldout_game() -> str:
    (game,) = json.loads(read(REPO / "sportvu" / "splits.json"))["heldout_game"]["games"]
    return game


def resolve_value(data, path: str):
    """Walk a field path through the data of one site file and return the one value it selects. A selector must
    pick exactly one list item, and a dictionary key must exist, so a wrong name fails here even where the
    contract leaves the name open."""
    current = data
    for segment in path.split("."):
        parsed = re.fullmatch(r"([a-z_0-9]+)(?:\[([^\]]+)\])?", segment)
        assert parsed, f"{path}: cannot read {segment!r}"
        name, selector = parsed.groups()
        assert isinstance(current, dict) and name in current, f"{path}: the data has no {name}"
        current = current[name]
        if selector is None:
            continue
        if isinstance(current, list) and "=" in selector:
            key, value = selector.split("=", 1)
            found = [item for item in current if isinstance(item, dict) and str(item.get(key)) == value]
            assert len(found) == 1, f"{path}: [{selector}] selects {len(found)} items, not one"
            current = found[0]
        elif isinstance(current, list):
            index = 0 if selector == "k" else int(selector)
            assert index < len(current), f"{path}: no item [{selector}]"
            current = current[index]
        else:
            assert isinstance(current, dict) and selector in current, f"{path}: no key {selector}"
            current = current[selector]
    assert not isinstance(current, (bool, dict, list)) and current is not None, f"{path}: ends at {current!r}"
    return current


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    """The synthetic data root of export/sample.py, the data the web work is built against."""
    from export import sample
    root = tmp_path_factory.mktemp("synthetic_site_data")
    sample.write_sample(root)
    return root


def test_field_paths_resolve_on_data_and_wrong_names_fail(synthetic_root):
    card = json.loads(read(synthetic_root / "scorecard.json"))
    good = "builds[role=adopted].rows[key=position_error].cells[split=heldout_game].values[p50_ft]"
    assert isinstance(resolve_value(card, good), float)
    assert isinstance(resolve_value(card, "builds[role=adopted].games[split=heldout_game].testable"), int)
    for bad in (good.replace("p50_ft", "p50"),                      # a value name the data does not hold
                good.replace("position_error", "position"),         # a row key the data does not hold
                good.replace("role=adopted", "build=r23"),          # a build name the synthetic data lacks
                good.replace("split=heldout_game", "split=heldout_era"),
                "builds[role=adopted].rows", "builds[role=adopted].games[split=heldout_game].frames"):
        with pytest.raises(AssertionError):
            resolve_value(card, bad)
    frames = json.loads(read(synthetic_root / "demo" / "frames.json"))
    assert isinstance(resolve_value(frames, "scenes[0].frames[k].error_ft"), float)


def test_every_script_token_is_listed_and_resolves(script, synthetic_root):
    text, table = script["text"], script["tokens"]
    before_table = text.split("\n## Tokens\n", 1)[0]
    used = set(TOKEN_RE.findall(before_table))
    assert used - set(table) == set(), "tokens used and not listed"
    assert set(table) - used == set(), "tokens listed and never used"
    d42 = next(line for line in read(REPO / "PLAN.md").split("\n") if line.startswith("- D42 "))
    assert all(sentence in d42 for sentence in D42_SAYS), "PLAN.md D42 no longer names the builds this way"
    problems = []
    for token, (data_file, field, _shown, copied) in sorted(table.items()):
        parts = token.split(".")
        if TOKEN_FILES.get(parts[0]) != data_file:
            problems.append(f"{token}: names {parts[0]}, the table says {data_file}")
            continue
        model = S.model_for(data_file)
        try:
            end = resolve_field(model, field)
        except AssertionError as error:
            problems.append(f"{token}: {error}")
            continue
        if end not in (int, float, str, bool):
            problems.append(f"{token}: {field} ends at {end}, not at a value")
        # the token names what the field path selects, and ends with the field's own name
        selected = [value.split("=", 1)[1] for value in re.findall(r"\[([^\]]+=[^\]]+)\]", field)]
        last = re.findall(r"[a-z_0-9]+", field)[-1]
        if any(value not in parts for value in selected) or parts[-1] != last:
            problems.append(f"{token}: does not match its field {field}")
        # the same path selects exactly one value on the synthetic data, so a row key or a value name that the
        # contract leaves open is checked against what export/sample.py writes
        try:
            resolve_value(json.loads(read(synthetic_root / data_file)), field)
        except AssertionError as error:
            problems.append(f"{token}: on the synthetic data, {error}")
        # a value copied from a scorecard report must exist there, as a number, under the same names, in the
        # report of the build that has the token's role and of the held-out game
        report = re.fullmatch(r"`(reports/scorecard/(\w+?)__(\w+)\.json)`, `([\w.]+)`", copied)
        if parts[0] == "scorecard":
            if not report:
                problems.append(f"{token}: no scorecard report named in {copied!r}")
                continue
            path, game, build, dotted = report.groups()
            if build != SITE_BUILDS.get(parts[1]) or parts[2] != "heldout_game" or game != heldout_game():
                problems.append(f"{token}: copied from {path}")
            want = ".".join(["metrics"] + parts[3:]) if len(parts) == 5 else ".".join(parts[3:])
            if dotted != want:
                problems.append(f"{token}: copied from {dotted}, its names say {want}")
            value = json.loads(read(REPO / path))
            for key in dotted.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{token}: {path} holds no number at {dotted}")
    assert problems == []
