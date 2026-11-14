"""export/site_reports.py: results.json and scorecard.json, copied from tracked files (PLAN.md M1, C5, C7).

results.json comes from reports/results.md through analytics/results.py: each lane report is parsed with
parse_report, and the records must render to the generated block of reports/results.md byte for byte. The question,
the model name and the verdict sentence of each lane are the cells of the hand-written Summary table, copied word
for word (the verdict is a quote from the write-up, D53) with only the Markdown bold markers removed (plain_cell):
results.json carries plain text, and the site does not render Markdown. The metrics and the rows come from the generated tables.
The verdict class comes from the generated tables where they decide it (an interval of a correlation, or a floor
candidate to beat). Where they do not, the class is the one the write-up's sentence names with a keyword, and
results.json says which lanes were classed that way. A lane that neither decides is a problem, never a default.

scorecard.json comes from the tracked scorecard files of the adopted build and of the measured candidate, and from
the paired comparison file (D42, D43, D52). The display line of a cell is composed from the file's own values and
must equal the line the scorecard's text file states.

Only named files are read: reports/ is never globbed. Nothing is computed beyond that display line.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from analytics import results as R
from analytics.registry import EXTENSION_NAMES
from export import schemas as S
from export import site_common as C

ExportProblem = C.ExportProblem

RESULTS_MD = "reports/results.md"
SCORECARD_DIR = "reports/scorecard"
SCORECARD_GAMES = ("gsw_phx_2016", "cle_nyk_2015", "cle_gsw_2016")
PAIRED_METRICS = ("p50", "p90", "abs_near", "abs_far")
# The rows a court replay can speak for: it keeps the earlier build's boxes and track linking (PLAN.md D52).
COURT_ROWS = ("position_error", "near_field_bias", "coverage", "generalisation")
SEEDS_LOG = "ROADMAP_LOG.md"
SUMMARY_COLUMNS = ("lane", "question", "selected model", "verdict")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_INTERVAL_RE = re.compile(r"\[(\S+), (\S+)\]")
_GENERATED_RE = re.compile(r"^Generated (\S+) by `([^`]+)`", re.M)
_INTERVALS_SENTENCE_RE = re.compile(r"Intervals are [^.;]*(?:;[^.]*)?\.", re.S)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# Verdict class from the generated tables (see class_from_tables).
LOWER_IS_BETTER = ("loss", "mse", "error", "brier")
CORRELATIONS = ("spearman",)
FLOOR_CANDIDATES = ("constant", "mean", "prior", "uniform", "persistence", "majority")
# The words by which a verdict sentence of the write-up names its own class (see class_from_words).
WORKS_WORDS = ("works", "passes")


# ------------------------------------------------------------------------------------------------ results.json


@dataclass
class ResultsDoc:
    """reports/results.md as the export reads it."""

    generated: str
    command: str
    records: dict                                   # lane -> analytics.results.LaneResult
    rows: dict                                      # lane -> [(field, value)] of the generated lane table
    summary: dict                                   # lane -> {question, selected model, verdict} of the Summary table
    intervals_sentence: str
    lanes: list = field(default_factory=list)       # Summary table order


def _lane_reports() -> list:
    return list(EXTENSION_NAMES) + sorted(R.OPTIONAL)


def _lane_rows(block: str) -> dict:
    """The field and value rows of each generated lane table, keyed by lane."""
    out = {}
    for section in R.Doc(block).sections:
        if section.level != 3:
            continue
        tables = [t for t in section.tables() if t.header == ["field", "value"]]
        if not tables:
            continue
        rows = tables[0].rows
        if any(len(row) != 2 for row in rows):
            raise ExportProblem(f"{RESULTS_MD}: a row of the {section.title} table does not split into field and value")
        out[section.title] = [(row[0], row[1]) for row in rows]
    return out


def read_results(repo_root=C.REPO_ROOT) -> ResultsDoc:
    """Parse the lane reports and reports/results.md, and require that they agree: the records parsed from the lane
    reports must render to the generated block on disk. The paths given to the parser are repository-relative, so
    the result does not depend on the working directory."""
    repo_root = Path(repo_root)
    path = repo_root / RESULTS_MD
    if not path.exists():
        raise ExportProblem(f"{RESULTS_MD} is missing")
    text = path.read_text(encoding="utf-8")
    try:
        parts = R._split_existing(text)
    except SystemExit as exc:
        raise ExportProblem(f"{RESULTS_MD}: {exc}") from None
    if parts is None:
        raise ExportProblem(f"{RESULTS_MD} has no results markers")
    before, block, _after = parts
    found = _GENERATED_RE.search(block)
    if not found:
        raise ExportProblem(f"{RESULTS_MD}: the generated block does not say when and how it was generated")
    generated, command = found.group(1), found.group(2)

    records = []
    for lane in _lane_reports():
        report = repo_root / "reports" / f"{lane}.md"
        if not report.exists():
            if lane in R.OPTIONAL:
                continue
            raise ExportProblem(f"reports/{lane}.md is missing")
        records.append(R.parse_report(lane, report.read_text(encoding="utf-8"), f"reports/{lane}.md"))
    unparsed = [r.lane for r in records if r.unparsed]
    if unparsed:
        raise ExportProblem(f"lane reports with unparsed fields: {unparsed}")
    if R.render_block(records, generated, command) != block:
        raise ExportProblem(f"{RESULTS_MD} does not equal what the lane reports render to: it is stale or was edited "
                            "between the results markers (run python3 -m analytics.results from the repository root)")

    doc = R.Doc(before)
    try:
        table = doc.need("Summary", level=2).table(*SUMMARY_COLUMNS)
    except R.ParseError as exc:
        raise ExportProblem(f"{RESULTS_MD}: {exc}") from None
    summary, lanes = {}, []
    for row in table.rows:
        cells = {name: plain_cell(table.cell(row, name)) for name in SUMMARY_COLUMNS}
        lane = cells.pop("lane")
        if lane in summary:
            raise ExportProblem(f"{RESULTS_MD}: lane {lane} is twice in the Summary table")
        summary[lane] = cells
        lanes.append(lane)
    sentence = _INTERVALS_SENTENCE_RE.search(before)
    if not sentence:
        raise ExportProblem(f"{RESULTS_MD}: the sentence that says what the intervals are was not found")
    if C.FREEZE_TAG not in before:
        raise ExportProblem(f"{RESULTS_MD} does not name the tag {C.FREEZE_TAG}")
    return ResultsDoc(generated=generated, command=command, records={r.lane: r for r in records},
                      rows=_lane_rows(block), summary=summary, lanes=lanes,
                      intervals_sentence=" ".join(sentence.group(0).split()))


def plain_cell(text: str) -> str:
    """A cell of the Summary table as plain text: each **bold** run loses its two markers, nothing else changes
    ("**failed**: no validated signal" becomes "failed: no validated signal"). The words stay those of the
    write-up; only the Markdown that the site would print raw is removed."""
    return _BOLD_RE.sub(r"\1", text)


def _number(text):
    """The float a report string states, or None when it is not a plain number (n/a, 317 / 603)."""
    text = str(text).strip()
    return float(text) if _NUMBER_RE.fullmatch(text) else None


def interval_of(metric):
    """(lo, hi) of a report metric, or (None, None) when the report states no interval."""
    if metric is None or not metric.interval:
        return None, None
    found = _INTERVAL_RE.fullmatch(metric.interval)
    lo, hi = (_number(found.group(1)), _number(found.group(2))) if found else (None, None)
    if lo is None or hi is None:
        raise ExportProblem(f"interval {metric.interval!r} of {metric.name} is not two numbers")
    return lo, hi


def metric_value(metric) -> S.MetricValue:
    """A report metric as the contract carries it: the text exactly as the generated block shows it, and the
    numbers parsed from the same strings."""
    lo, hi = interval_of(metric)
    return S.MetricValue(text=metric.show(primary=True), name=metric.name, value=_number(metric.value), lo=lo, hi=hi,
                         point_estimate=lo is None)


def class_from_tables(record):
    """The verdict class of a lane, from the values of its generated table only (D53), or None when nothing in
    the table decides it (the metric is not a correlation and the table holds no floor candidate). The rule:

    A lane whose out-of-fold metric is a correlation (Spearman, null value zero) has a signal when an interval
    excludes zero. With a stated out-of-fold interval that contains zero it is `failed`: it was tested and found
    nothing. With no out-of-fold interval it is `inconclusive` when its held-out interval contains zero or is
    absent, and `works` otherwise.

    Any other lane is `no_signal` when its selected model is no better out of fold than a floor candidate of the
    same table (a constant, a mean, a prior, a uniform or majority reference, persistence), and `works` when it
    is better than every such floor. Lower is better for a loss, an MSE, an error or a Brier score, higher for
    everything else. With no floor candidate there is nothing to compare with: the answer is None, not `works`."""
    name = record.oof.name.lower()
    held = record.heldout.metric if record.heldout is not None and record.heldout.state == R.HO_SCORED else None
    if any(word in name for word in CORRELATIONS):
        lo, hi = interval_of(record.oof)
        if lo is not None:
            return "works" if (lo > 0 or hi < 0) else "failed"
        lo, hi = interval_of(held)
        return "works" if lo is not None and (lo > 0 or hi < 0) else "inconclusive"
    value = _number(record.oof.value)
    floors = []
    for candidate, metric in record.candidates:
        plain = re.sub(r"^\d+\s+", "", R.clean(candidate)).lower()
        number = _number(metric.value)
        if number is not None and plain.startswith(FLOOR_CANDIDATES):
            floors.append(number)
    if value is None or not floors:
        return None
    if any(word in name for word in LOWER_IS_BETTER):
        return "no_signal" if value >= min(floors) else "works"
    return "no_signal" if value <= max(floors) else "works"


def class_from_words(verdict: str):
    """The class the write-up's own verdict sentence names, by keyword, in this order: failed, inconclusive,
    no signal, then works or passes. None when the sentence holds none of these words: a sentence that names no
    class is never read as `works`."""
    text = verdict.lower()
    if "failed" in text:
        return "failed"
    if "inconclusive" in text:
        return "inconclusive"
    if re.search(r"\bno (?:\w+ )?signal\b", text):
        return "no_signal"
    if re.search(r"\b(?:%s)\b" % "|".join(WORKS_WORDS), text):
        return "works"
    return None


def verdict_class(lane: str, record, verdict: str):
    """(class, what decided it) of one lane. The tables decide when they can ("tables"), and the write-up's
    sentence must not name another class. When the tables cannot decide, the class is the one the sentence names
    ("write-up"). When neither decides, or the two disagree, nothing is written."""
    tables, words = class_from_tables(record), class_from_words(verdict)
    if tables is None and words is None:
        raise ExportProblem(f"lane {lane}: its generated table holds no interval of a correlation and no floor "
                            "candidate, and the verdict sentence of the write-up names no class (failed, "
                            f"inconclusive, no signal, {' or '.join(WORKS_WORDS)}). The class is not defaulted.")
    if tables is not None and words is not None and tables != words:
        raise ExportProblem(f"lane {lane}: the generated tables read as {tables}, the verdict sentence of the "
                            f"write-up as {words}. One of the two has to change before the site shows either.")
    if tables is None:
        return words, "write-up"
    return tables, "tables" if words is not None else "tables, write-up silent"


def class_bases(doc: ResultsDoc) -> dict:
    """lane -> what decided its verdict class, for the lanes of the Summary table that have a report."""
    return {lane: verdict_class(lane, doc.records[lane], doc.summary[lane]["verdict"])[1]
            for lane in doc.lanes if lane in doc.records and doc.records[lane].oof is not None}


def build_results(repo_root=C.REPO_ROOT, doc: ResultsDoc = None) -> S.Results:
    doc = doc or read_results(repo_root)
    lanes, decided_by = [], {}
    for lane in doc.lanes:
        record = doc.records.get(lane)
        if record is None:
            raise ExportProblem(f"{RESULTS_MD}: the Summary table names lane {lane}, which has no report")
        if record.status in R.FIXTURE_LIKE:
            raise ExportProblem(f"{RESULTS_MD}: the Summary table names lane {lane}, whose report is {record.status}")
        if record.oof is None or lane not in doc.rows:
            raise ExportProblem(f"{RESULTS_MD}: lane {lane} has no generated table")
        held = record.heldout
        cells = doc.summary[lane]
        lane_class, decided_by[lane] = verdict_class(lane, record, cells["verdict"])
        lanes.append(S.LaneResult(
            lane=lane, question=cells["question"], selected_model=cells["selected model"], status=record.status,
            out_of_fold=metric_value(record.oof),
            heldout=metric_value(held.metric) if held is not None and held.state == R.HO_SCORED else None,
            verdict=cells["verdict"], verdict_source=RESULTS_MD, verdict_class=lane_class, report=record.path,
            rows=[S.ReportRow(field=f, value=v) for f, v in doc.rows[lane]]))
    pending = [S.PendingLane(lane=r.lane, reason=f"{r.path} is {r.status}, so no number from it is shown.")
               for r in doc.records.values() if r.status in R.FIXTURE_LIKE]
    notes = [
        doc.intervals_sentence,
        "Question, model name and verdict are the cells of the hand-written Summary table of reports/results.md, "
        "copied word for word with only the Markdown bold markers removed: the verdict is a quote from the "
        "write-up. The out-of-fold and held-out metrics and "
        "the rows are copied from the tables that analytics/results.py generates from each lane report.",
        "The verdict class comes from the generated tables where they decide it. A lane scored by a correlation "
        "with a stated out-of-fold interval that contains zero is failed; with no out-of-fold interval and a "
        "held-out interval that contains zero it is inconclusive. Any other lane is no_signal when its selected "
        "model is no better out of fold than a floor candidate of its own table (a constant, a mean, a prior, a "
        "uniform or majority reference, persistence), and works when it beats every such floor. The build stops "
        "if this class and the class the write-up's sentence names disagree. No class is a default.",
        "Held-out games were scored once, by the models frozen at the tag named in freeze_tag.",
    ]
    by_words = [lane for lane, basis in decided_by.items() if basis == "write-up"]
    silent = [lane for lane, basis in decided_by.items() if basis == "tables, write-up silent"]
    if by_words:
        notes.insert(3, "Lanes whose table decides no class (no correlation interval and no floor candidate), "
                        f"classed by the word of the write-up's own sentence instead: {', '.join(by_words)}.")
    if silent:
        notes.insert(3 + bool(by_words), "Lanes classed by their table alone, because the write-up's sentence names "
                                         f"no class: {', '.join(silent)}.")
    sources = [C.source(RESULTS_MD, "Summary table (question, model, verdict) and the generated lane tables", "not_split")]
    sources += [C.source(doc.records[lane.lane].path, f"lane report of {lane.lane}, parsed by analytics/results.py",
                         "not_split") for lane in lanes]
    return S.Results(schema_version=S.SCHEMA_VERSION, freeze_tag=C.FREEZE_TAG, generated=doc.generated, lanes=lanes,
                     pending=pending, notes=notes, sources=sources)


# ---------------------------------------------------------------------------------------------- scorecard.json

# Row labels of the scorecard text files, by target key (sportvu/bench.py, render_txt).
ROW_LABELS = {
    "court_line_error": "court line error", "position_error": "position error", "near_field_bias": "near-field bias",
    "coverage": "coverage", "missed_players": "missed players", "ghost_boxes": "ghost boxes",
    "team_labels": "team labels", "identity": "identity", "generalisation": "generalisation",
}
# The verdict word the text file prints for each status of a row.
TXT_VERDICTS = {"met": ("MET",), "not met": ("not met",), "reported": ("reported",), "not measurable": ("n/a", "-")}


def _fmt(value, pattern: str = "%s") -> str:
    return "n/a" if value is None else pattern % value


def _pct(value) -> str:
    return "n/a" if value is None else "%.1f%%" % (100 * value)


def _status(block: dict):
    """(status, passed) of one scorecard row from its pass field, as the file says."""
    passed = block.get("pass")
    if passed is True:
        return "met", True
    if passed is False:
        return "not met", False
    reason = block.get("pass_withheld") or block.get("not_measurable") or ""
    if block.get("not_measurable") or str(reason).startswith("not measurable"):
        return "not measurable", None
    return "reported", None


def _numbers(block: dict, prefix: str = "") -> dict:
    """Every number of a scorecard block, nested keys joined with a dot. The pass field and text are left out."""
    out = {}
    for key, value in block.items():
        if key == "pass" or isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[prefix + key] = float(value)
        elif isinstance(value, dict):
            out.update(_numbers(value, f"{prefix}{key}."))
    return out


def _cell_text(key: str, m: dict) -> str:
    """The display line of one row, composed from the file's values in the scorecard's own wording."""
    if key == "court_line_error":
        if str(m.get("pass_withheld") or "").startswith("not measurable"):
            return ("not measurable (truth H fits the feet, not the court; diagnostic only: p50 %s px vs the truth "
                    "template)" % _fmt(m["p50_px"]))
        return "%s of frames <= 3 px; p50 %s px, p95 %s px (floor: truth fit residual p50 %s px, p90 %s px)" % (
            _pct(m["share_le_target"]), _fmt(m["p50_px"]), _fmt(m["p95_px"]),
            _fmt(m["truth_fit_residual_p50_px"]), _fmt(m["truth_fit_residual_p90_px"]))
    if key == "position_error":
        return "p50 %s ft, p90 %s ft" % (_fmt(m["p50_ft"]), _fmt(m["p90_ft"]))
    if key == "near_field_bias":
        return "near %s ft, far %s ft (toward camera)" % (_fmt(m["near_third_ft"], "%+.2f"),
                                                          _fmt(m["far_third_ft"], "%+.2f"))
    if key == "coverage":
        return "%s (%s of %s live wide s; %s s accepted on live non-wide)" % (
            _pct(m["coverage"]), m.get("accepted_live_wide_s"), m.get("live_wide_s"), m.get("accepted_live_not_wide_s"))
    if key == "missed_players":
        return _pct(m["rate"])
    if key == "ghost_boxes":
        return "%s non-referee (%s with referees; %d referee boxes)" % (
            _pct(m["rate"]), _pct(m["rate_all"]), m["referee_boxes"])
    if key == "team_labels":
        return _pct(m["accuracy"])
    if key == "identity":
        return "%s id switches / possession (%s before linking); jersey: %s" % (
            _fmt(m["id_switches_per_possession"]), _fmt(m["fragment_switches_per_possession"]),
            "not measurable" if m["jersey_read_rate"] is None else m["jersey_read_rate"])
    if key == "generalisation":
        value = m.get("value")
        if not value:
            return "not measurable"
        return "position error p50 %.2fx, p90 %.2fx of the held-out game (%s ft / %s ft there; %s)" % (
            value["p50_ratio"], value["p90_ratio"], value["heldout_game_p50_ft"], value["heldout_game_p90_ft"],
            value["vs"])
    raise ExportProblem(f"scorecard target {key!r} has no display rule")


def _cell_n(key: str, m: dict):
    """(n, unit) of one row, the count the scorecard's text file prints beside the target."""
    return {
        "court_line_error": (m.get("frames_defined"), "frames"),
        "position_error": (m.get("matched_players"), "matched players"),
        "near_field_bias": (None, None),             # two counts (near and far third): both are in values
        "coverage": (m.get("live_s"), "live seconds"),
        "missed_players": (m.get("in_frame_players"), "players in frame"),
        "ghost_boxes": (m.get("boxes"), "boxes"),
        "team_labels": (m.get("n"), "labelled players"),
        "identity": (m.get("possessions"), "possessions"),
        "generalisation": (None, None),
    }[key]


def _txt_lines(text: str) -> dict:
    """label -> (verdict word, line) of the rows of a scorecard text file."""
    out = {}
    for label in ROW_LABELS.values():
        found = re.search(r"^%s\s{2,}(MET|not met|n/a|reported|-)\s+(.*)$" % re.escape(label), text, re.M)
        if found:
            out[label] = (found.group(1), found.group(2).rstrip())
    return out


def _read_json(repo_root: Path, rel: str) -> dict:
    path = repo_root / rel
    if not path.exists():
        raise ExportProblem(f"{rel} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(repo_root: Path, rel: str) -> str:
    path = repo_root / rel
    if not path.exists():
        raise ExportProblem(f"{rel} is missing")
    return path.read_text(encoding="utf-8")


def _build_cards(repo_root: Path, build: str) -> dict:
    """game -> (card json, rows of its text file) for one build."""
    cards = {}
    for game in SCORECARD_GAMES:
        stem = f"{SCORECARD_DIR}/{game}__{build}"
        card = _read_json(repo_root, stem + ".json")
        if card.get("game") != game or card.get("build") != build:
            raise ExportProblem(f"{stem}.json names game {card.get('game')!r} and build {card.get('build')!r}")
        if card.get("split") not in ("heldout_game", "heldout_arena"):
            raise ExportProblem(f"{stem}.json is not a held-out scorecard (split {card.get('split')!r})")
        cards[game] = (card, _txt_lines(_read_text(repo_root, stem + ".txt")))
    return cards


def _caveats(cards: dict) -> list:
    """The caveats of a build's scorecards: one that every game states is given once, one that only some games
    state is given per game with the game named first."""
    order, by_game = [], {}
    for game, (card, _lines) in cards.items():
        by_game[game] = list(card.get("caveats") or [])
        order += [c for c in by_game[game] if c not in order]
    out = []
    for caveat in order:
        holders = [g for g in cards if caveat in by_game[g]]
        out += [caveat] if len(holders) == len(cards) else [f"{g}: {caveat}" for g in holders]
    return out


def _scorecard_build(repo_root: Path, build: str, role: str, note: str, keys=None, extra_caveats=()) -> S.ScorecardBuild:
    """One build's scorecard: one row per target of the file's targets block (or per target in `keys`)."""
    cards = _build_cards(repo_root, build)
    first = next(iter(cards.values()))[0]
    targets = first["targets"]
    missing = [key for key in (keys or ()) if key not in targets]
    if missing:
        raise ExportProblem(f"{SCORECARD_DIR}: build {build} has no target {missing}")
    rows = []
    for key, target in targets.items():
        if keys is not None and key not in keys:
            continue
        cells = []
        for game, (card, lines) in cards.items():
            if card["targets"] != targets:
                raise ExportProblem(f"{SCORECARD_DIR}/{game}__{build}.json states other targets than the first game")
            block = card["metrics"].get(key)
            if block is None:
                raise ExportProblem(f"{SCORECARD_DIR}/{game}__{build}.json has no metrics for target {key}")
            status, passed = _status(block)
            text = _cell_text(key, block)
            stated = lines.get(ROW_LABELS.get(key, key))
            if stated is None or stated[1] != text or stated[0] not in TXT_VERDICTS[status]:
                raise ExportProblem(f"{SCORECARD_DIR}/{game}__{build}: row {key} composed from the json ({status}: "
                                    f"{text!r}) is not the line of the text file ({stated!r})")
            values = _numbers(block.get("value") or {}) if key == "generalisation" else _numbers(block)
            n, unit = _cell_n(key, block)
            cells.append(S.ScorecardCell(game=game, split=card["split"], status=status, passed=passed, text=text,
                                         values=values, n=None if n is None else float(n), n_unit=unit))
        rows.append(S.ScorecardRow(key=key, target=target, cells=cells))
    games = [S.ScorecardGame(game=game, game_id=card["sportvu_game"], split=card["split"],
                             truth_frames=card["truth_frames"], testable=card["testable"],
                             untestable_share=card["untestable_share"]) for game, (card, _l) in cards.items()]
    return S.ScorecardBuild(build=build, role=role, note=note, games=games, rows=rows,
                            caveats=_caveats(cards) + list(extra_caveats))


def _paired(repo_root: Path, baseline: str, candidate: str):
    """Paired differences of the comparison file, and the method words its text file states."""
    stem = f"{SCORECARD_DIR}/compare__{baseline}__{candidate}"
    doc = _read_json(repo_root, stem + ".json")
    if doc.get("builds") != [baseline, candidate]:
        raise ExportProblem(f"{stem}.json compares {doc.get('builds')}, not {[baseline, candidate]}")
    out, method = [], {}
    for game in SCORECARD_GAMES:
        block = (doc.get("games", {}).get(game) or {}).get("paired", {}).get(candidate)
        if not block:
            raise ExportProblem(f"{stem}.json has no paired block for {game}")
        for metric in PAIRED_METRICS:
            lo, hi = block[metric]["ci95"]
            out.append(S.PairedDifference(game=game, metric=metric, baseline=baseline, build=candidate,
                                          diff=block[metric]["diff"], lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
        method[f"paired_n_{game}"] = f"{block['windows']} windows, {block['observations']} observations"
    text = _read_text(repo_root, stem + ".txt")
    stated = re.search(r"paired %s - %s \(\d+ windows, \d+ obs; ([^)]*)\)" % (re.escape(baseline), re.escape(candidate)),
                       text)
    if not stated:
        raise ExportProblem(f"{stem}.txt does not state how the paired differences were made")
    method["paired"] = (f"{baseline} minus {candidate} on the observations both builds score ({stated.group(1)}); "
                        f"the metrics are errors, so a positive difference means {candidate} is better")
    return out, method


def _seeds_caveat(repo_root: Path, adopted: str) -> str:
    """The caveat PLAN.md D52 asks for: the adopted court model is the best of three training seeds of its recipe.
    The statement is taken from the project log, which must still make it."""
    text = _read_text(repo_root, SEEDS_LOG)
    if not re.search(r"\b%s is the best of its three seeds\b" % re.escape(adopted), text):
        raise ExportProblem(f"{SEEDS_LOG} no longer says that {adopted} is the best of its three seeds (PLAN.md D52)")
    return (f"Build {adopted} is the best of the three training seeds of its recipe on the held-out arena, so the "
            f"recipe's expected error on a new arena is larger than this build's arena rows ({SEEDS_LOG}, third "
            "scaling pass; PLAN.md D52).")


def build_scorecard(adopted: str, candidate: str, repo_root=C.REPO_ROOT) -> S.Scorecard:
    """The tracking scorecard of the adopted build and of the measured candidate (PLAN.md D42, D43, D52). Every
    value is a point estimate copied from the scorecard files; the only intervals are those of the paired
    differences, which the comparison file states. The candidate is a court replay, so it is shown on the court
    rows only."""
    repo_root = Path(repo_root)
    baseline = f"{adopted}_replay"
    builds = [
        _scorecard_build(repo_root, adopted, "adopted",
                         f"Build {adopted}: the adopted build (PLAN.md D5, D42). The site and the demo use it.",
                         extra_caveats=[_seeds_caveat(repo_root, adopted)]),
        _scorecard_build(repo_root, candidate, "candidate",
                         f"Build {candidate}: {adopted} with the camera model added. Measured and not adopted "
                         f"(PLAN.md D42). It is a court replay: it keeps the earlier boxes and track linking, so it "
                         f"is shown on the court rows only. Its paired differences are against {baseline}, which "
                         f"equals {adopted} on those rows (PLAN.md D52).", keys=COURT_ROWS),
    ]
    paired, paired_method = _paired(repo_root, baseline, candidate)
    methods = []
    for build in (adopted, candidate):
        for game in SCORECARD_GAMES:
            methods.append(_read_json(repo_root, f"{SCORECARD_DIR}/{game}__{build}.json").get("method") or {})
    if any(m != methods[0] for m in methods):
        raise ExportProblem("the scorecard files do not share one method block")
    method = {key: str(value) for key, value in methods[0].items()}
    method.update(paired_method)
    decision = (f"PLAN.md D42: the site shows build {adopted}, the adopted build. Build {candidate} is shown beside "
                "it as a measured candidate with its paired differences. The two builds differ only on held-out "
                "footage, so showing the better one would be a selection on held-out data.")
    sources = []
    for build in (adopted, candidate):
        for game in SCORECARD_GAMES:
            sources.append(C.source(f"{SCORECARD_DIR}/{game}__{build}.json", f"scorecard of build {build} on {game}",
                                    "heldout"))
            sources.append(C.source(f"{SCORECARD_DIR}/{game}__{build}.txt",
                                    "the same scorecard as text: each display line is checked against it", "heldout"))
    sources.append(C.source(f"{SCORECARD_DIR}/compare__{baseline}__{candidate}.json", "paired differences", "heldout"))
    sources.append(C.source(f"{SCORECARD_DIR}/compare__{baseline}__{candidate}.txt", "method of the paired differences",
                            "heldout"))
    sources.append(C.source(SEEDS_LOG, f"build {adopted} is the best of the three training seeds of its recipe",
                            "heldout"))
    return S.Scorecard(schema_version=S.SCHEMA_VERSION, decision=decision, builds=builds, paired=paired,
                       method=method, sources=sources)
