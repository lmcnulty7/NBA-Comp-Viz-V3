"""export/research_methods.py: the data of the methods and literature page (PLAN.md D72).

    python3 -m export.research_methods --check          check the source and print what the page would hold
    python3 -m export.site_data --out <root> --only methods   write methods.json into a data root (and its manifest)

The source is reports/research/methods.json, committed and written by hand from the lane dossiers under
docs/research/ (the dossiers are where each fetched citation lives) and docs/research/BACKLOG.md. Before anything
is written, every text and every number of it is traced:

- a text copied from a dossier is in that dossier word for word, white space aside, and whole: a paper's citation,
  what the lane takes from it and how ours differs, and a benchmark row's published cell, paper, what differs and
  comparability verdict each equal a whole table cell; `ours_text` is a whole cell or its opening words; a note is
  one or more whole sentences of a paragraph; an experiment's title is its heading, and its hypothesis, adoption
  threshold and paired test equal the text of those bullets under that heading (after the bullet's own label). So
  a quote cannot drop a "not" or end before its qualifier;
- every number of ours (its value, its interval and any number in its label) is a number of the report it names,
  and value, lo and hi stand on one line of that report;
- every published value is a number of the row's published cell, and `where` (the table or section) is part of
  the published or the paper cell;
- every link is in the dossier, or is the doi.org or arxiv.org address of an identifier the dossier names;
- every experiment's backlog fields equal its row of BACKLOG.md, every row of BACKLOG.md is an experiment here,
  and every experiment a dossier pre-registers is listed under its lane;
- `written` is the date every dossier states.

Then the file is built against the contract (export.schemas.MethodsFile). The comparability verdict is sorted into
three kinds by the words it opens with. An experiment's results are read from the files its source entry names
under reports/research/: none yet means pre-registered and not run; a development result that meets its rule
waits for its single test score (PLAN.md D71, D77); when the test file appears, the next export picks it up. The
paired difference, its interval and the threshold are rounded once, here (PLAN.md D59). Nothing is fitted, tuned
or scored: the result files are read as they are.
"""

import argparse
import json
import re
from pathlib import Path

from export import schemas as S

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "reports/research/methods.json"
BACKLOG = "docs/research/BACKLOG.md"


class MethodsProblem(Exception):
    """The source does not trace to the dossiers and reports it names."""


# ------------------------------------------------------------------------------------------------ helpers


def _flat(text: str) -> str:
    """Text with every run of white space made one space, so a wrapped dossier line reads as one line."""
    return re.sub(r"\s+", " ", text).strip()


def _read(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.is_file():
        raise MethodsProblem(f"{rel}: no such file")
    return path.read_text(encoding="utf-8")


MINUS = "\u2212"


def number_token(token: str) -> str:
    """A number as one spelling: the true minus sign read as a hyphen, a leading plus dropped. The sign of a
    negative number is kept, so -0.5 never matches 0.5."""
    token = token.replace(MINUS, "-")
    return token[1:] if token.startswith("+") else token


def numbers(text: str) -> list[str]:
    """The numbers of a text (S.NUMBER_TEXT_RE), each in the one spelling of number_token. A true minus sign is
    read as a hyphen before the numbers are found, so it stays part of its number."""
    return [number_token(token) for token in S.NUMBER_TEXT_RE.findall(text.replace(MINUS, "-"))]


def _unsigned(value: str) -> str:
    return number_token(value)


def comparability_class(verdict: str) -> str:
    """The kind of a dossier's comparability verdict, by the words it opens with."""
    text = verdict.lower()
    if text.startswith("roughly comparable"):
        return "roughly"
    if text.startswith("not directly comparable"):
        return "not_directly"
    if text.startswith(("no comparable", "no published comparable", "nothing of ours", "cannot be compared",
                        "not a performance number")):
        return "none"
    raise MethodsProblem(f"comparability verdict {verdict[:60]!r} opens with words this export does not sort")


def link_traces(url: str, dossier: str) -> bool:
    """A link is in the dossier, or is the doi.org or arxiv.org address of an identifier the dossier names."""
    if url in dossier:
        return True
    doi = re.fullmatch(r"https://doi\.org/(10\..+)", url)
    if doi and re.search(rf"(?:doi:|DOI ){re.escape(doi.group(1))}(?![\w./-])", dossier):
        return True
    arxiv = re.fullmatch(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5})", url)
    return bool(arxiv and f"arXiv:{arxiv.group(1)}" in dossier)


def backlog_rows(text: str) -> dict:
    """The ranked table of BACKLOG.md, by experiment id: rank, id, lane, what, gain, effort, ratio, test data,
    status."""
    rows = {}
    for line in text.split("## The top three")[0].split("\n"):
        if not line.startswith("| ") or line.startswith("| Rank") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        rows[cells[1]] = cells
    return rows


def dossier_experiments(text: str) -> dict:
    """The experiments a dossier pre-registers: id to title, from its ### headings."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^### ([a-z]+-E\d+): (.*)$", text, re.M)}


def dossier_cells(text: str) -> set:
    """Every cell of every table of a dossier, white space made single."""
    cells = set()
    for line in text.split("\n"):
        if line.startswith("|"):
            cells.update(_flat(cell) for cell in line.strip().strip("|").split(" | "))
    return cells


def dossier_paragraphs(text: str) -> list:
    """The paragraphs of a dossier or a write-up, white space made single (table rows and headings included, one
    block each). Each item of a bullet list is a paragraph of its own, without its marker."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        for item in re.split(r"^- ", block, flags=re.M):
            if item.strip():
                paragraphs.append(_flat(item))
    return paragraphs


def verdict_paragraph(text: str) -> list:
    """The paragraph of a write-up that opens with its verdict in bold ("**Verdict.**", "**Verdict: passes ...**"), as
    a one-item list."""
    return [p for p in dossier_paragraphs(text) if re.match(r"\*\*Verdict[.:]", p)]


def is_sentences(note: str, paragraphs: list) -> bool:
    """True when `note` is one or more whole sentences of one paragraph: it starts the paragraph or follows the
    end of a sentence, and it ends a sentence."""
    note = _flat(note)
    if not note or note[-1] not in ".?!":
        return False
    for paragraph in paragraphs:
        start = paragraph.find(note)
        while start != -1:
            before = paragraph[:start].rstrip()
            after = paragraph[start + len(note):]
            lead = before.rstrip("*").rstrip()     # a bold lead that ends a sentence ("**Caveats written now.**")
            if (not before or before[-1] in ".?!" or (before.endswith("**") and lead[-1:] in (".", "?", "!"))) and (
                    not after or after[0] == " "):
                return True
            start = paragraph.find(note, start + 1)
    return False


# A bullet's own label, as the dossiers write it: "Hypothesis.", "Adoption threshold (fixed now).", "Paired test:".
BULLET_LABELS = {"hypothesis": r"Hypothes[ie]s", "threshold": r"Adoption threshold", "interval": r"Paired test"}


def experiment_bullets(text: str) -> dict:
    """For every pre-registered experiment of a dossier, the text of its labelled bullets after the label:
    {id: {"hypothesis": ..., "threshold": ..., "interval": ...}}. A bullet runs over its indented lines."""
    found = {}
    headings = list(re.finditer(r"^### ([a-z]+-E\d+): .*$", text, re.M))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = re.split(r"^## ", text[heading.end():end], maxsplit=1, flags=re.M)[0]
        bullets, current = [], None
        for line in block.split("\n"):
            if line.startswith("- "):
                current = [line[2:]]
                bullets.append(current)
            elif current is not None and line.startswith("  ") and line.strip():
                current.append(line.strip())
            else:
                current = None
        texts = {"_all": [_flat(" ".join(bullet)) for bullet in bullets]}
        for bullet in bullets:
            joined = _flat(" ".join(bullet))
            for key, label in BULLET_LABELS.items():
                m = re.match(rf"^{label}[^.:]*[.:]\s*", joined)
                if m and key not in texts:
                    texts[key] = joined[m.end():]
        found[heading.group(1)] = texts
    return found


def _path_value(doc: dict, path: str):
    """The value at a dotted path of a result file; a whole-number segment indexes a list (an interval's ends)."""
    value = doc
    for key in path.split("."):
        if isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise MethodsProblem(f"the result file has no {path}")
    return value


# ------------------------------------------------------------------------------------------------ the trace


def trace_problems(source: dict, repo: Path = REPO_ROOT) -> list[str]:
    """Every way the source fails to trace to its dossiers, reports and BACKLOG.md. Empty when it traces."""
    problems = []
    backlog = backlog_rows(_read(repo, source["backlog"]))
    reports: dict = {}

    def report_lines(rel: str) -> list[str]:
        if rel not in reports:
            reports[rel] = _read(repo, rel).split("\n")
        return reports[rel]

    listed = set()
    for lane in source["lanes"]:
        name = lane["lane"]
        dossier = _read(repo, lane["dossier"])
        cells = dossier_cells(dossier)
        paragraphs = dossier_paragraphs(dossier)
        bullets = experiment_bullets(dossier)
        _read(repo, lane["report"])

        def miss(where: str, text: str, what: str) -> None:
            problems.append(f"{name}: {where} is not {what} of {lane['dossier']} word for word: {text[:70]!r}")

        def cell(where: str, text) -> None:
            if text is not None and _flat(text) not in cells:
                miss(where, text, "a whole table cell")

        def opening(where: str, text) -> None:
            flat = _flat(text) if text is not None else None
            if flat is not None and not any(c == flat or c.startswith(flat + " ") or c.startswith(flat + ",")
                                            or c.startswith(flat + ";") or c.startswith(flat + " (") for c in cells):
                miss(where, text, "a table cell or the opening words of one")

        written = lane.get("written", source["written"])
        if written not in dossier:
            problems.append(f"{name}: {lane['dossier']} does not state the date {written}")
        for i, paper in enumerate(lane["foundations"]):
            cell(f"foundations[{i}].cite", paper["cite"])
            cell(f"foundations[{i}].take", paper["take"])
            cell(f"foundations[{i}].differs", paper.get("differs"))
            if not link_traces(paper["url"], dossier):
                problems.append(f"{name}: foundations[{i}].url {paper['url']} is not named in the dossier")
        for i, row in enumerate(lane["benchmarks"]):
            where = f"benchmarks[{i}]"
            for key in ("published", "paper", "differs", "comparability"):
                cell(f"{where}.{key}", row.get(key))
            opening(f"{where}.ours_text", row.get("ours_text"))
            try:
                comparability_class(row["comparability"])
            except MethodsProblem as exc:
                problems.append(f"{name}: {where}: {exc}")
            published = numbers(row["published"])
            for value in row["published_values"]:
                if number_token(value) not in published:
                    problems.append(f"{name}: {where}: published value {value} is not a number of the published cell")
            if row.get("where") and row["where"] not in row["published"] and row["where"] not in row["paper"]:
                problems.append(f"{name}: {where}.where {row['where']!r} is not part of the published or paper cell")
            if row.get("url") and not link_traces(row["url"], dossier):
                problems.append(f"{name}: {where}.url {row['url']} is not named in the dossier")
            for ours in row["ours"]:
                lines = report_lines(ours["source"])
                tokens = [set(numbers(line)) for line in lines]
                wanted = [_unsigned(t) for t in (ours["value"], ours.get("lo"), ours.get("hi")) if t is not None]
                if not any(set(wanted) <= line for line in tokens):
                    problems.append(f"{name}: {where}: {ours['label']} {wanted} do not stand on one line of "
                                    f"{ours['source']}")
                every = set().union(*tokens) if tokens else set()
                for token in numbers(ours["label"]):
                    if token not in every:
                        problems.append(f"{name}: {where}: the label {ours['label']!r} holds {token}, which "
                                        f"{ours['source']} does not")
        for i, note in enumerate(lane["notes"]):
            if not is_sentences(note, paragraphs):
                miss(f"notes[{i}]", note, "one or more whole sentences")
        registered = dossier_experiments(dossier)
        ids = [e["id"] for e in lane["experiments"]]
        if sorted(ids) != sorted(registered):
            problems.append(f"{name}: experiments {ids} are not the dossier's {sorted(registered)}")
        for exp in lane["experiments"]:
            eid = exp["id"]
            listed.add(eid)
            if registered.get(eid) != exp["title"]:
                problems.append(f"{eid}: the title is not the dossier's heading")
            for key in ("hypothesis", "threshold"):
                # a dossier that names its bullets otherwise ("Question.", "Hypotheses and readings (fixed now).")
                # says which bullet holds the text: `hypothesis_label`, `threshold_label`
                label = exp.get(f"{key}_label")
                if label:
                    want = next((b[m.end():] for b in bullets.get(eid, {}).get("_all", [])
                                 if (m := re.match(rf"^{re.escape(label)}[^.:]*[.:]\s*", b))), None)
                else:
                    want = bullets.get(eid, {}).get(key)
                if _flat(exp[key]) != want:
                    miss(f"{eid}.{key}", exp[key], f"the {label or key} bullet under {eid}")
            if exp.get("test_not_scored"):
                problems += _not_scored_problems(eid, exp, repo)
            if exp.get("writeup") and not exp.get("result_in_writeup") and not exp["backlog_status"].startswith("done"):
                problems.append(f"{eid}: an experiment with a write-up has run, so it needs a done line in the backlog")
            labels = [part.get("label") for part in exp.get("parts", [])]
            if len(set(labels)) != len(labels):
                problems.append(f"{eid}: every part has its own label")
            if exp.get("result_in_writeup"):
                if not exp.get("writeup") or not (repo / exp["writeup"]).is_file():
                    problems.append(f"{eid}: a result kept in its write-up needs the write-up")
                if exp.get("results") or exp.get("protocol"):
                    problems.append(f"{eid}: a result kept in its write-up has no results or protocol here")
                if not exp["backlog_status"].startswith("done"):
                    problems.append(f"{eid}: a result kept in its write-up needs a done line in the backlog")
            row = backlog.get(eid)
            if row is None:
                problems.append(f"{eid}: not a row of {source['backlog']}")
                continue
            want = {"rank": row[0], "lane": row[2], "what": row[3], "gain": row[4], "effort": row[5],
                    "test_data": row[7], "backlog_status": row[8]}
            have = {"rank": str(exp["rank"]), "lane": name, "what": exp["what"], "gain": str(exp["gain"]),
                    "effort": str(exp["effort"]), "test_data": exp["test_data"],
                    "backlog_status": exp["backlog_status"]}
            for key in want:
                if want[key] != have[key]:
                    problems.append(f"{eid}: {key} is {have[key]!r}, {source['backlog']} says {want[key]!r}")
            writeup = exp.get("writeup")
            if writeup is not None and (not writeup.startswith("reports/research/") or not (repo / writeup).is_file()):
                problems.append(f"{eid}: the write-up {writeup!r} is not a file under reports/research/")
            notes = exp.get("writeup_notes", [])
            if notes:
                for note in notes:
                    where, quote = _note_place(note)
                    # a note may quote a second write-up of the same experiment (genai-E2's 1.5B report)
                    named = note.get("file", writeup) if isinstance(note, dict) else writeup
                    if not named or not named.startswith("reports/research/") or not (repo / named).is_file():
                        problems.append(f"{eid}: a write-up note names {named!r}, not a write-up under "
                                        "reports/research/")
                        continue
                    text = (repo / named).read_text(encoding="utf-8")
                    note_paragraphs = verdict_paragraph(text) if where is None else dossier_paragraphs(
                        section_text(text, where))
                    if not is_sentences(quote, note_paragraphs):
                        place = "the verdict paragraph" if where is None else f"the section {where!r}"
                        problems.append(f"{eid}: a write-up note is not whole sentences of {place} of {named} word "
                                        f"for word: {quote[:70]!r}")
            protocol = exp.get("protocol")
            if protocol:
                if results_problem := _protocol_trace(eid, exp, protocol, repo):
                    problems += results_problem
            if exp.get("stages") is not None and protocol:
                problems.append(f"{eid}: stages and a protocol together: a protocol has its own stage")
            results = exp.get("results")
            if exp.get("stages") == "test_only" and results and results.get("development") is not None:
                problems.append(f"{eid}: a test-only experiment names no development result file")
            says_research = "adopted as a research result" in " ".join(
                verdict_paragraph((repo / writeup).read_text(encoding="utf-8")) if writeup and (repo / writeup)
                .is_file() else []).lower() or any("adopted as a research result" in _note_place(n)[1].lower()
                                                   for n in exp.get("writeup_notes", []))
            if says_research and exp.get("adopted_as") != "research_result":
                problems.append(f"{eid}: the write-up says \"adopted as a research result\", so adopted_as is "
                                "research_result")
            if exp.get("adopted_as") is not None and (exp.get("protocol") or exp.get("stages") == "test_only"):
                problems.append(f"{eid}: adopted_as belongs to a pass on both stages, not to a protocol or a test-only "
                                "experiment")
            if exp.get("adopted_as") is not None:
                quotes = [_note_place(n)[1] for n in exp.get("writeup_notes", [])]
                if exp["adopted_as"] != "research_result" or not any("adopted as a research result" in q
                                                                        for q in quotes):
                    problems.append(f"{eid}: adopted_as {exp['adopted_as']!r} needs a write-up note that says "
                                    "\"adopted as a research result\"")
            stage_specs = ([(None, results, exp.get("stages"), exp.get("test_not_scored"))] if results else []) + [
                (part.get("label"), part.get("results") or {}, part.get("stages"), part.get("test_not_scored"))
                for part in exp.get("parts", [])]
            if exp.get("parts") and (results or exp.get("protocol") or exp.get("stages")):
                problems.append(f"{eid}: an experiment decided part by part has no results, protocol or stages of its "
                                "own")
            for label, spec, stages, not_scored in stage_specs:
                where = eid if label is None else f"{eid} {label}"
                try:
                    check_results_spec(spec)
                except MethodsProblem as exc:
                    problems.append(f"{where}: {exc}")
                    continue
                text = (repo / writeup).read_text(encoding="utf-8") if writeup and (repo / writeup).is_file() else ""
                for kind in ("rule_quote", "reading_quote"):
                    quote = spec.get(kind)
                    if quote and not is_sentences(quote["text"], dossier_paragraphs(section_text(text,
                                                                                                 quote["section"]))):
                        problems.append(f"{where}: the {kind.replace('_', ' ')} is not whole sentences of the section "
                                        f"{quote['section']!r} of {writeup} word for word")
                label_of_interval = spec.get("interval_label")
                source_of_interval = spec.get("interval_from")
                if source_of_interval in ("dossier_sentence", "writeup_sentence"):
                    pool = paragraphs if source_of_interval == "dossier_sentence" else dossier_paragraphs(text)
                    if not is_sentences(spec.get("interval") or "", pool):
                        problems.append(f"{where}: the interval is not whole sentences of the "
                                        f"{source_of_interval.split('_')[0]} word for word")
                else:
                    if label_of_interval:
                        want = next((b[m.end():] for b in bullets.get(eid, {}).get("_all", [])
                                     if (m := re.match(rf"^{re.escape(label_of_interval)}[^.:]*[.:]\s*", b))), None)
                    else:
                        want = bullets.get(eid, {}).get("interval")
                    if _flat(spec.get("interval") or "") != want:
                        miss(f"{where}.results.interval", spec.get("interval", ""),
                             f"the {label_of_interval or 'paired test'} bullet under {eid}")
                for stage in ("development", "test"):
                    if spec.get(stage) is None and stage == "development" and stages == "test_only":
                        continue
                    if spec.get(stage) is None and stage == "test" and not_scored:
                        continue
                    if spec.get(stage) is None or not spec[stage].startswith("reports/research/"):
                        problems.append(f"{where}: the {stage} result is a file under reports/research/")
                    elif stage == "development" and stages == "test_only":
                        problems.append(f"{where}: a test-only part names no development result file")
    for eid in sorted(set(backlog) - listed):
        problems.append(f"{eid}: a row of {source['backlog']} that no lane lists")
    return problems


# ------------------------------------------------------------------------------------------------ the build


SCHEMA_COMPARE = {"below": lambda v, t: v < t, "at_most": lambda v, t: v <= t, "above": lambda v, t: v > t,
                  "at_least": lambda v, t: v >= t}
# How a rule's text states each comparison, right before its threshold.
COMPARISON_WORDS = {"below": (r"below",), "at_most": (r"at most", r"at or below", r"<="), "above": (r"above",),
                    "at_least": (r"at least", r"at or above", r">=")}
# The end of an interval a comparison reads: an interval wholly below (or at most) reads its upper end, an interval
# above (or at least) its lower end. A condition may read the point value instead, for any comparison.
BOUND_OF = {"below": "hi", "at_most": "hi", "above": "lo", "at_least": "lo"}
CONDITION_KEYS = ("label", "value", "lo", "hi", "bound", "comparison", "met", "digits")
# The only verdict flags a source may name as outside the adoption rule, by experiment: each was checked against its
# pre-registration and is not a part of the rule a condition could read. A new one is added here, by hand, with
# its reason in the source.
FLAGS_OUTSIDE_RULE_ALLOWED = {
    "xfg-E1": {"verdict.width_p95_under_0_01"},            # the site copy rule of the pre-registration
    "winprob-E1": {"{verdict}.interval_straddles_threshold",   # a reading printed beside the verdict
                   "{verdict}.regulation_only_reading.adopt", "{verdict}.regulation_only_reading.differs",
                   "{verdict}.ablation.drift_form_adds_beyond_mu_alone"},  # the regulation-only and ablation readings
    "winprob-E2": {"{verdict}.ablation.forward_beats_grouped"},          # the ablation reading
    "lineups-E2": {"scored.results.verdict.toward_their_players_supported"},  # the shrinkage ablation reading
}
GUARD_KEYS = ("label", "path", "buckets", "worse", "bound", "comparison", "threshold", "digits")
READING_KINDS = ("met", "not_met", "otherwise")


def check_results_spec(results: dict) -> None:
    """The source's description of a stage result, checked before any file is read: a known comparison for the
    main rule, the end its comparison reads (keys.upper, when named, is that end), a list of further conditions
    (empty when the rule has none), each condition complete, with a threshold (stated, or read from the file with its
    sign) and the end of its interval that its comparison reads (or its point value), each guard complete, and the
    way the file states its verdict (a flag, a reading word, or a quoted reading of the write-up)."""
    comparison = results.get("comparison", "at_most")
    if comparison not in SCHEMA_COMPARE:
        raise MethodsProblem(f"the main rule's comparison {comparison!r} is unknown")
    by_stage = results.get("stage_keys", {})
    if set(by_stage) - {"development", "test"} or any(set(k) - {"met", "main_met", "reading"} for k in by_stage.values()):
        raise MethodsProblem("stage_keys names a development or a test stage, and only its met, main_met or reading")
    for stage_name in ("development", "test"):
        if results.get(stage_name) is not None and stage_name in by_stage:
            check_results_spec({**results, "keys": {**results.get("keys", {}), **by_stage[stage_name]},
                                "stage_keys": {}})
    if by_stage:
        return
    keys = results.get("keys", {})
    end_key = keys.get(BOUND_OF[comparison])
    if results.get("flip"):
        if "upper" in keys:
            raise MethodsProblem("a flipped interval names no keys.upper: the end it reads follows from the flip")
    elif keys.get("upper", end_key) != end_key:
        raise MethodsProblem(f"the main rule ({comparison.replace('_', ' ')}) reads the {BOUND_OF[comparison]} end: "
                             f"keys.upper must be keys.{BOUND_OF[comparison]}")
    if results.get("development") is not None and results.get("development") == results.get("test"):
        raise MethodsProblem("the development and the test result are one file")
    if not isinstance(results.get("conditions"), list):
        raise MethodsProblem("the results name no list of further conditions (an empty list when the rule has none)")
    if not ("met" in keys or "reading" in keys or results.get("reading_quote")):
        raise MethodsProblem("the results name how the file states its verdict: keys.met, keys.reading or a "
                             "reading_quote of the write-up")
    if ("reading" in keys or results.get("reading_quote")) and not results.get("readings"):
        raise MethodsProblem("a reading word needs the readings it may take (met, not_met, or a condition)")
    for word, kind in results.get("readings", {}).items():
        if kind not in READING_KINDS and not (isinstance(kind, dict) and {"bound", "comparison", "threshold"} <= set(
                kind) and kind["comparison"] in SCHEMA_COMPARE):
            raise MethodsProblem(f"the reading {word!r} is not met, not_met, otherwise or a condition on an end")
    for condition in results["conditions"]:
        missing = [key for key in CONDITION_KEYS if key not in condition]
        if "threshold" not in condition and "threshold_key" not in condition:
            missing.append("threshold")
        if missing:
            raise MethodsProblem(f"the condition {condition.get('label')!r} lacks {missing}")
        if condition["comparison"] not in SCHEMA_COMPARE:
            raise MethodsProblem(f"the condition {condition['label']!r} has an unknown comparison "
                                 f"{condition['comparison']!r}")
        if condition["bound"] != "point" and condition["bound"] != BOUND_OF[condition["comparison"]]:
            raise MethodsProblem(f"the condition {condition['label']!r} reads the {condition['bound']} end, but "
                                 f"{condition['comparison']} reads the {BOUND_OF[condition['comparison']]} end")
        if "threshold_key" in condition and condition.get("sign") not in (1, -1):
            raise MethodsProblem(f"the condition {condition['label']!r} reads its threshold from the file without "
                                 "a sign (1 or -1)")
    for guard in _guards_of(results):
        missing = [key for key in GUARD_KEYS if key not in guard]
        if missing:
            raise MethodsProblem(f"the guard lacks {missing}")
        if guard["comparison"] not in SCHEMA_COMPARE or guard["bound"] != BOUND_OF[guard["comparison"]]:
            raise MethodsProblem(f"the guard reads the {guard['bound']} end with {guard['comparison']!r}")


def _guards_of(results: dict) -> list:
    return ([results["guard"]] if results.get("guard") else []) + list(results.get("guards", []))


def states(rule: str, comparison: str, threshold) -> bool:
    """True when a rule's text states `comparison` right before `threshold` ("below 0", "below zero",
    "at most +0.0005", "at most -0.002", ">= 0.5", "at or above -0.05"). The sign is part of the threshold. A
    threshold read from a file may appear rounded, to three decimals or more ("above 0.038" for 0.0380059...)."""
    rule = _flat(rule)
    for word in COMPARISON_WORDS[comparison]:
        for m in re.finditer(rf"(?<![\w]){word}\s*(zero|[+-]?\d+(?:\.\d+)?)(?!\d|\.\d)", rule):
            text = m.group(1)
            if text == "zero":
                if float(threshold) == 0:
                    return True
                continue
            written = float(text)
            decimals = len(text.split(".")[1]) if "." in text else 0
            if written == float(threshold) or (decimals >= 3 and abs(written - float(threshold)) <= 0.5 * 10 ** -decimals):
                return True
    return False


def section_text(text: str, heading: str) -> str:
    """The text of the section of a write-up under `## heading`, up to the next `## ` heading ('' when absent)."""
    if heading == "":                  # the text before the first section: the verdict and its narrow claim
        return re.split(r"^## ", text, maxsplit=1, flags=re.M)[0]
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def _note_place(note) -> tuple:
    """A write-up note is a quote of the verdict paragraph (a string) or of a named section ({section, text})."""
    if isinstance(note, dict):
        return note["section"], note["text"]
    return None, note


def number_text(value) -> str:
    """A threshold written as a number of a rule's text (0 as "0", 0.0005 as "0.0005", -0.002 as "-0.002")."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _flags(value, prefix: str) -> dict:
    """Every boolean of a verdict object, nested objects included, by its dotted path."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, bool):
                found[f"{prefix}.{key}"] = item
            elif isinstance(item, dict):
                found.update(_flags(item, f"{prefix}.{key}"))
    return found


def _stage(results: dict, stage: str, repo: Path, rule_text: str = "", eid: str = "", extra_mapped=frozenset(),
           writeup_text: str = ""):
    """The result of one stage, or None when its file does not exist yet. The file's verdict on the main rule, on
    each further condition and on each guard is recomputed from its numbers, and a file that disagrees is refused.

    Each comparison, with its threshold and its sign, must be the one the rule's text states: the file's own rule
    (keys["rule"]), else the source's quote of the write-up (`rule_quote`, traced word for word), else `rule_text`,
    the pre-registered adoption threshold. The file states its verdict by a flag (keys.met), by a reading word
    (keys.reading, which must agree with the recomputed rule as `readings` says), or not at all, when the write-up's
    quoted reading (`reading_quote`) carries the word instead. Every boolean of the file's verdict object, nested
    ones included, other than the overall flag must be the main rule's flag (keys["main_met"]), a condition's flag,
    a flag of another part of the same rule (`extra_mapped`), or a flag the source names as outside the rule
    (`flags_outside_rule`, on the code's list). A key path may name the stage's own place in its file through
    `paths` ({data}, {verdict}), so one description reads a development and a test file of different layouts. An
    interval the file states the other way round (`flip`) is turned: its difference and its ends change sign.

    Further shapes (PLAN.md D116): a stage may state its verdict its own way (`stage_keys`: a flag on development, a
    reading word on the test set); a reading may be `otherwise`, which holds only when no conditional reading does;
    a condition or guard whose bar is another statistic's end names that bar in the rule's words (`states`) and
    reads the number from the file; a flag the file states as the conjunction of parts of the rule
    (`conjunctions`) must equal them; a flag that only marks the deciding comparison must be true (`true_flags`);
    a reported value may belong to one stage only (`stage`) or read another result file (`file`, listed as
    also_read)."""
    check_results_spec(results)
    rel = results[stage]
    if rel is None:
        return None
    path = repo / rel
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    places_of = results.get("paths", {}).get(stage, {})
    docs = {rel: doc}

    def at(key_path: str, file: str = None):
        source = doc
        if file is not None:
            if not file.startswith("reports/research/") or not (repo / file).is_file():
                raise MethodsProblem(f"{rel}: a value reads {file!r}, not a file under reports/research/")
            source = docs.setdefault(file, json.loads((repo / file).read_text(encoding="utf-8")))
        try:
            return _path_value(source, key_path.format(**places_of))
        except (KeyError, IndexError) as exc:
            raise MethodsProblem(f"{rel}: the key path {key_path!r} names an unknown place {exc}") from None

    # a stage may state its verdict its own way (a flag on development, a reading word on the test set)
    keys = {**results["keys"], **results.get("stage_keys", {}).get(stage, {})}
    comparison = results.get("comparison", "at_most")
    compare = SCHEMA_COMPARE[comparison]
    sign = -1.0 if results.get("flip") else 1.0
    lo_raw, hi_raw = float(at(keys["lo"])), float(at(keys["hi"]))
    lo_v, hi_v = (-hi_raw, -lo_raw) if sign < 0 else (lo_raw, hi_raw)
    diff_v = sign * float(at(keys["diff"])) if "diff" in keys else None
    upper = lo_v if BOUND_OF[comparison] == "lo" else hi_v
    if "rule" in keys:
        rule = at(keys["rule"])
    elif results.get("rule_quote"):
        rule = results["rule_quote"]["text"]
    else:
        rule = rule_text
    threshold = float(at(keys["threshold"])) if "threshold" in keys else float(results["threshold"])
    if not states(rule, comparison, threshold):
        raise MethodsProblem(f"{rel}: the rule does not state {comparison.replace('_', ' ')} {threshold}: {rule[:90]!r}")
    if "rule" in keys and "threshold" not in keys and number_text(threshold) not in numbers(rule):
        raise MethodsProblem(f"{rel}: the rule {keys['rule']} does not name the threshold {threshold}")

    def bar_of(spec) -> float:
        if "threshold_key" in spec:
            return spec["sign"] * float(at(spec["threshold_key"]))
        value = float(spec["threshold"])
        if "rule" in keys and number_text(value) not in numbers(rule):
            raise MethodsProblem(f"{rel}: the rule {keys['rule']} does not name the threshold {value}")
        return value

    for spec in results["conditions"]:
        bar = bar_of(spec)
        if spec.get("states"):
            # a threshold that is another statistic's end is named in words ("above the upper end of the
            # position-only retriever's interval"): the words must be in the rule, and the number is read from the file
            if _flat(spec["states"]) not in _flat(rule) or "threshold_key" not in spec:
                raise MethodsProblem(f"{rel}: the rule does not say {spec['states']!r}, or the condition reads no "
                                     "threshold from the file")
        elif not states(rule, spec["comparison"], bar):
            raise MethodsProblem(f"{rel}: the rule does not state {spec['comparison'].replace('_', ' ')} {bar} for "
                                 f"{spec['label']!r}")
    for guard_spec in _guards_of(results):
        if guard_spec.get("states"):
            if _flat(guard_spec["states"]) not in _flat(rule):
                raise MethodsProblem(f"{rel}: the rule does not say {guard_spec['states']!r}")
        elif not states(rule, guard_spec["comparison"], guard_spec["threshold"]):
            raise MethodsProblem(f"{rel}: the rule does not state the guard's {guard_spec['comparison']} "
                                 f"{guard_spec['threshold']}")
    for flag in results.get("true_flags", []):
        # a flag that says which comparison decides ("decides": the Haiku cache arm is H3's deciding arm)
        if at(flag) is not True:
            raise MethodsProblem(f"{rel}: {flag} is not true, so this is not the comparison that decides")
    for a, b in results.get("equalities", []):
        if float(at(a)) != float(at(b)):
            raise MethodsProblem(f"{rel}: {a} is not {b}, though the file states them as one number")
    outside = {flag.format(**places_of) for flag in results.get("flags_outside_rule", {})}
    allowed = {flag.format(**places_of) for flag in FLAGS_OUTSIDE_RULE_ALLOWED.get(eid, set())}
    if outside - allowed:
        raise MethodsProblem(f"{rel}: {sorted(outside - allowed)} are not on the list of flags outside the rule for "
                             f"{eid}")
    rule_flags = {k.format(**places_of) for k in [keys.get("met"), keys.get("main_met")] if k} | {
        spec["met"].format(**places_of) for spec in results["conditions"] if spec["met"]} | {
        c["flag"].format(**places_of) for c in results.get("conjunctions", [])}
    if outside & rule_flags:
        raise MethodsProblem(f"{rel}: {sorted(outside & rule_flags)} are flags of the rule, not outside it")
    verdict_key = keys.get("met") or keys.get("reading")
    if verdict_key:
        verdict_path = verdict_key.format(**places_of).rsplit(".", 1)[0]
        verdict = _path_value(doc, verdict_path)
        if "threshold" in verdict:
            read = {keys.get("threshold", "").format(**places_of)} | {
                spec["threshold_key"].format(**places_of) for spec in results["conditions"] if "threshold_key" in spec}
            if f"{verdict_path}.threshold" not in read:
                raise MethodsProblem(f"{rel}: the file holds {verdict_path}.threshold, so the rule reads it "
                                     "(keys.threshold or a condition's threshold_key), not a threshold the source "
                                     "states")
            stated = [spec["threshold"] for spec in results["conditions"] if "threshold" in spec and spec["threshold"]]
            if stated or ("threshold" not in keys and float(results.get("threshold", 0)) != 0):
                raise MethodsProblem(f"{rel}: the file holds its threshold, so the source states none but zero bounds")
        if "upper_95" in verdict and float(verdict["upper_95"]) != hi_raw:
            raise MethodsProblem(f"{rel}: {verdict_path}.upper_95 is not the upper end of the paired interval")
        mapped = rule_flags | outside | {flag.format(**places_of) for flag in
                                         set(extra_mapped) | set(results.get("true_flags", []))}
        loose = sorted(flag for flag in _flags(verdict, verdict_path) if flag not in mapped)
        if loose:
            raise MethodsProblem(f"{rel}: the verdict flags {loose} belong to no rule of the source: name them as the "
                                 "main rule's flag, a condition's flag, or a flag outside the rule")
    for guard_spec in _guards_of(results):
        paired = ".".join(keys["diff"].split(".")[:-2]) if "diff" in keys else None
        if guard_spec.get("same_object", True) and (paired is None or not guard_spec["path"].startswith(paired + ".")):
            raise MethodsProblem(f"{rel}: the guard reads {guard_spec['path']}, not a table of the paired object "
                                 f"{paired} the rule's difference comes from")
    main = compare(upper, threshold)
    if "main_met" in keys and bool(at(keys["main_met"])) != main:
        raise MethodsProblem(f"{rel}: {keys['main_met']} disagrees with its bound {upper} against {threshold}")
    conditions = []
    for spec in results["conditions"]:
        lo, hi, value = float(at(spec["lo"])), float(at(spec["hi"])), float(at(spec["value"]))
        end = {"lo": lo, "hi": hi, "point": value}[spec["bound"]]
        bar = bar_of(spec)
        ok = SCHEMA_COMPARE[spec["comparison"]](end, bar)
        if spec["met"] and bool(at(spec["met"])) != ok:
            raise MethodsProblem(f"{rel}: {spec['met']} disagrees with its {spec['bound']} {end} against {bar}")
        places = spec["digits"]
        while SCHEMA_COMPARE[spec["comparison"]](round(end, places), round(bar, places)) != ok and places < 17:
            places += 1
        conditions.append(S.Condition(label=spec["label"], value=round(value, places), lo=round(lo, places),
                                      hi=round(hi, places), bound=spec["bound"], comparison=spec["comparison"],
                                      threshold=round(bar, places), met=ok))
    guards = []
    for guard_spec in _guards_of(results):
        table = at(guard_spec["path"])
        if sorted(table) != sorted(guard_spec["buckets"]):
            raise MethodsProblem(f"{rel}: the guard reads every bucket of {guard_spec['path']}: {sorted(table)}, "
                                 f"not {guard_spec['buckets']}")
        worse_compare = SCHEMA_COMPARE[guard_spec["comparison"]]
        cell_of = guard_spec.get("cell", "")
        names = {"point": guard_spec.get("cell_value", "point"), "lo": guard_spec.get("cell_lo", "lo"),
                 "hi": guard_spec.get("cell_hi", "hi")}
        buckets, worse_names = [], []
        for name in guard_spec["buckets"]:
            cell = _path_value(table[name], cell_of) if cell_of else table[name]
            point, lo, hi = (float(_path_value(cell, names[k])) for k in ("point", "lo", "hi"))
            end = lo if guard_spec["bound"] == "lo" else hi
            worse = worse_compare(end, float(guard_spec["threshold"]))
            if worse:
                worse_names.append(name)
            places = guard_spec["digits"]
            while worse_compare(round(end, places), float(guard_spec["threshold"])) != worse and places < 17:
                places += 1
            buckets.append(S.GuardBucket(label=name, value=round(point, places), lo=round(lo, places),
                                         hi=round(hi, places), worse=worse))
        stated = list(at(guard_spec["worse"]))
        if sorted(stated) != sorted(worse_names):
            raise MethodsProblem(f"{rel}: {guard_spec['worse']} lists {stated}, but its numbers give {worse_names}")
        guards.append(S.BucketGuard(label=guard_spec["label"], bound=guard_spec["bound"],
                                    comparison=guard_spec["comparison"], threshold=float(guard_spec["threshold"]),
                                    buckets=buckets, met=not worse_names))
    reading_ok = main and all(c.met for c in conditions) and all(g.met for g in guards)
    # a flag the file states as the conjunction of some of the rule's parts ("holds against raw") equals it
    part_of = {keys.get("main_met"): main} | {spec["met"]: c.met for spec, c in zip(results["conditions"], conditions)
                                              if spec["met"]} | {g_spec["worse"]: g.met for g_spec, g in
                                                                 zip(_guards_of(results), guards)}
    for conj in results.get("conjunctions", []):
        unknown = [f for f in conj["of"] if f not in part_of]
        if unknown:
            raise MethodsProblem(f"{rel}: {conj['flag']} is said to join {unknown}, which are no parts of the rule")
        if bool(at(conj["flag"])) != all(part_of[f] for f in conj["of"]):
            raise MethodsProblem(f"{rel}: {conj['flag']} disagrees with the parts it joins")
    reading_word = None
    if "met" in keys:
        met = bool(at(keys["met"]))
        if met != reading_ok:
            raise MethodsProblem(f"{rel}: {keys['met']} is {met}, but its rule read on its own numbers gives "
                                 f"{reading_ok}")
    else:
        met = reading_ok
    if "reading" in keys or results.get("reading_quote"):
        if "reading" in keys:
            reading_word = str(at(keys["reading"]))
        else:
            quote = results["reading_quote"]
            reading_word = quote["word"]
            if not is_sentences(quote["text"], dossier_paragraphs(section_text(writeup_text, quote["section"]))):
                raise MethodsProblem(f"{rel}: the reading quote is not whole sentences of the section "
                                     f"{quote['section']!r} of the write-up word for word")
            if reading_word not in quote["text"]:
                raise MethodsProblem(f"{rel}: the reading quote does not say {reading_word!r}")
        word = next((w for w in results["readings"] if reading_word == w or reading_word.startswith(w + " ")), None)
        if word is None:
            raise MethodsProblem(f"{rel}: the reading {reading_word!r} is none of {sorted(results['readings'])}")
        kind = results["readings"][word]
        conditional = [k for k in results["readings"].values() if isinstance(k, dict)]

        def holds(k) -> bool:
            return SCHEMA_COMPARE[k["comparison"]](lo_v if k["bound"] == "lo" else hi_v, float(k["threshold"]))

        if kind == "otherwise" and any(holds(k) for k in conditional):
            raise MethodsProblem(f"{rel}: the reading {reading_word!r} is for the case no other reading fits, but "
                                 "one does")
        if kind == "met" and not met or kind != "met" and met:
            raise MethodsProblem(f"{rel}: the reading {reading_word!r} disagrees with the rule read on the numbers "
                                 f"({'met' if met else 'not met'})")
        if isinstance(kind, dict):
            end = lo_v if kind["bound"] == "lo" else hi_v
            if not SCHEMA_COMPARE[kind["comparison"]](end, float(kind["threshold"])):
                raise MethodsProblem(f"{rel}: the reading {reading_word!r} needs the {kind['bound']} end "
                                     f"{kind['comparison']} {kind['threshold']}, and it is {end}")
    # Rounded once, here. The rule's two numbers keep as many decimals as it takes for the rounded comparison to
    # read as the file's own verdict, so rounding can never flip it (an upper end of 0.000502 against 0.0005
    # shows six decimals, not five).
    digits = results["digits"]
    while compare(round(upper, digits), round(threshold, digits)) != main and digits < 17:
        digits += 1

    values = []
    for spec in results["values"]:
        if spec.get("stage", stage) != stage:      # a value one stage's file reports and the other's does not
            continue
        file = spec.get("file")
        lo = round(float(at(spec["lo"], file)), spec["digits"]) if spec.get("lo") else None
        hi = round(float(at(spec["hi"], file)), spec["digits"]) if spec.get("hi") else None
        values.append(S.StageValue(label=spec["label"], value=round(float(at(spec["value"], file)), spec["digits"]),
                                   lo=lo, hi=hi, point_estimate=lo is None))
    rows = "testset" if stage == "test" else "out_of_fold"
    what = ("experiment result: fitted on every development game, scored once on the D71 test set" if stage == "test"
            else "experiment result: out of fold on the development games")
    sources = sorted({spec["file"] for spec in results["values"] if spec.get("file")} - {rel})
    return S.StageResult(
        stage=stage, source=S.Source(path=rel, what=what, rows=rows),
        also_read=[S.Source(path=f, what="numbers reported beside the rule, read by no verdict",
                            rows="out_of_fold" if stage == "development" else "testset") for f in sources],
        design=str(at(results["design"])) if results.get("design") else None, metric=results["metric"],
        rows=int(at(keys["rows"])), rows_unit=results["rows_unit"],
        games=int(at(keys["games"])) if "games" in keys else None,
        diff=round(diff_v, digits) if diff_v is not None else None, lo=round(lo_v, digits), hi=round(hi_v, digits),
        interval=results["interval"], threshold=round(threshold, digits), upper=round(upper, digits),
        comparison=comparison, conditions=conditions, guard=guards[0] if len(guards) == 1 and results.get("guard")
        else None, guards=guards if not (len(guards) == 1 and results.get("guard")) else [], met=met,
        reading=reading_word, values=values)


def _protocol_trace(eid: str, exp: dict, protocol: dict, repo: Path) -> list:
    """The trace of a protocol entry: its files are under reports/research/, its caveat is whole sentences of the
    write-up's verdict paragraph and its not-scored reasons whole sentences of the write-up."""
    problems = []
    for key in ("development", "intervals"):
        if not protocol[key].startswith("reports/research/"):
            problems.append(f"{eid}: the protocol's {key} file is not under reports/research/")
    writeup = exp.get("writeup")
    if not writeup or not (repo / writeup).is_file():
        problems.append(f"{eid}: a protocol result needs its write-up")
        return problems
    text = (repo / writeup).read_text(encoding="utf-8")
    if protocol.get("caveat") and not is_sentences(protocol["caveat"], verdict_paragraph(text)):
        problems.append(f"{eid}: the caveat is not whole sentences of the verdict paragraph of {writeup} word for "
                        f"word: {protocol['caveat'][:70]!r}")
    for sentence in protocol["not_scored"]:
        if not is_sentences(sentence, dossier_paragraphs(text)):
            problems.append(f"{eid}: a not-scored reason is not whole sentences of {writeup} word for word: "
                            f"{sentence[:70]!r}")
    return problems


# The only interval of a post hoc file the page may show: the bootstrap by player (PLAN.md D86). The game draws
# (game_basic) and the pre-registered game-bootstrap ranges are not valid confidence intervals.
INTERVAL_KEY = "player_bootstrap"
# The comparisons a protocol's statement or rule may use, as its text writes them.
COMPARISONS = {"below": lambda v, t: v < t, "above": lambda v, t: v > t, "at least": lambda v, t: v >= t,
               "at most": lambda v, t: v <= t}


def holds_by_threshold(value: float, threshold: str) -> bool:
    """Whether `value` meets a threshold written as words and a number ("below 0.2", "at least 0.5")."""
    m = re.fullmatch(r"(below|above|at least|at most) ([+-]?\d+(?:\.\d+)?)", threshold)
    if not m:
        raise MethodsProblem(f"the threshold {threshold!r} is not one of {sorted(COMPARISONS)} and a number")
    return COMPARISONS[m.group(1)](value, float(m.group(2)))


def _protocol(spec: dict, repo: Path, writeup=None):
    """The protocol result of an experiment, or None when its development file does not exist yet. The value of
    each statement and the adoption rule are read from the development file; each valid interval from the post hoc
    file, whose statistic must be the same number as the development file's (so the interval belongs to it), and
    only its bootstrap by player (INTERVAL_KEY). The pre-registered ranges of the development file are read only to
    recompute the adoption rule, never shown: they are not valid confidence intervals. Each statement's holds and
    the rule's adopted are recomputed from the values and thresholds, and a file that disagrees is refused. An
    experiment that names a write-up has run, so its development file must exist."""
    if spec.get("interval_key", INTERVAL_KEY) != INTERVAL_KEY:
        raise MethodsProblem(f"the page shows the {INTERVAL_KEY} interval only, not {spec['interval_key']!r}")
    path = repo / spec["development"]
    if not path.is_file():
        if writeup:
            raise MethodsProblem(f"{spec['development']}: no such file, but the experiment names its write-up "
                                 f"{writeup}, so it has run")
        return None
    dev = json.loads(path.read_text(encoding="utf-8"))
    posthoc_path = repo / spec["intervals"]
    if not posthoc_path.is_file():
        raise MethodsProblem(f"{spec['intervals']}: no such file, so the protocol has no valid interval")
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    digits = spec["digits"]
    method = _path_value(posthoc, spec["interval_method_key"])
    if spec["interval_method"] not in method:
        raise MethodsProblem(f"{spec['intervals']}: the interval method does not say {spec['interval_method']!r}")
    interval_source = S.Source(path=spec["intervals"], what="valid intervals computed post hoc (bootstrap by player); "
                               "no verdict reads them", rows="training")

    def interval(statistic: str, value: float):
        if not statistic:
            return None
        table = _path_value(posthoc, spec["statistics_key"])
        if statistic not in table:        # the names of the statistics hold dots, so they are looked up whole
            raise MethodsProblem(f"{spec['intervals']}: no statistic {statistic}")
        stat = table[statistic]
        if not isinstance(stat.get("value"), (int, float)) or abs(float(stat["value"]) - value) > 1e-9:
            raise MethodsProblem(f"{spec['intervals']}: {statistic} is {stat.get('value')}, not the development "
                                 f"value {value}, so its interval belongs to another number")
        boot = stat.get(INTERVAL_KEY)
        if not isinstance(boot, dict) or not all(isinstance(boot.get(k), (int, float)) for k in ("lo", "hi")):
            raise MethodsProblem(f"{spec['intervals']}: {statistic} has no {INTERVAL_KEY} interval with lo and hi")
        return S.PostHocInterval(lo=round(float(boot["lo"]), digits), hi=round(float(boot["hi"]), digits),
                                 method=spec["interval_method"], source=interval_source)

    hypotheses = []
    for item in spec["hypotheses"]:
        h = _path_value(dev, f"{spec['verdict_key']}.{item['key']}")
        raw = h.get("value")
        finite = isinstance(raw, (int, float)) and raw == raw and abs(raw) != float("inf")
        holds = bool(h["holds"]) if finite and h.get("holds") is not None else None
        if finite and holds is not None and holds != holds_by_threshold(float(raw), item["threshold"]):
            raise MethodsProblem(f"{spec['development']}: {item['key']} says holds {holds}, but {raw} "
                                 f"{item['threshold']} is {not holds}")
        if (holds is None) != (h.get("result") == "undefined") or (holds is not None and h["result"] !=
                                                                      ("holds" if holds else "fails")):
            raise MethodsProblem(f"{spec['development']}: {item['key']} reads {h.get('result')!r} with holds "
                                 f"{h.get('holds')!r} and value {raw!r}")
        found = interval(item.get("interval"), float(raw)) if finite else None
        hypotheses.append(S.Hypothesis(label=item["key"], statement=h["statement"], threshold=item["threshold"],
                                       value=round(float(raw), digits) if finite else None, holds=holds,
                                       result=h["result"], interval=found, point_estimate=found is None))
    adoption_doc = _path_value(dev, f"{spec['verdict_key']}.{spec['adoption']['key']}")
    rule_numbers = numbers(adoption_doc["rule"])
    met = []
    loose = sorted(k for k, v in adoption_doc.items() if isinstance(v, bool) and k != "adopted")
    if loose:
        raise MethodsProblem(f"{spec['development']}: the adoption flags {loose} belong to no condition of the source")
    if not isinstance(spec["adoption"].get("conditions"), list) or not spec["adoption"]["conditions"]:
        raise MethodsProblem(f"{spec['development']}: the source names no conditions of the adoption rule")
    for condition in spec["adoption"]["conditions"]:
        words, number = condition["threshold"].rsplit(" ", 1)
        comparison = words.replace(" ", "_")
        if number not in rule_numbers:
            raise MethodsProblem(f"{spec['development']}: the rule does not name the threshold {number}")
        if comparison not in SCHEMA_COMPARE or not states(adoption_doc["rule"], comparison, float(number)):
            raise MethodsProblem(f"{spec['development']}: the rule does not state {condition['threshold']!r}")
        if condition["value"] not in adoption_doc:
            raise MethodsProblem(f"{spec['development']}: the adoption entry has no {condition['value']}")
        met.append(holds_by_threshold(float(adoption_doc[condition["value"]]), condition["threshold"]))
    if all(met) != bool(adoption_doc["adopted"]):
        raise MethodsProblem(f"{spec['development']}: adopted is {adoption_doc['adopted']}, but its rule read on "
                             f"its own values gives {all(met)}")
    adoption = []
    for item in spec["adoption"]["values"]:
        if item["value"] not in adoption_doc:
            raise MethodsProblem(f"{spec['development']}: the adoption entry has no {item['value']}")
        raw = float(adoption_doc[item["value"]])
        found = interval(item.get("interval"), raw)
        adoption.append(S.AdoptionValue(label=item["label"], value=round(raw, digits), interval=found,
                                        point_estimate=found is None))
    return S.ProtocolResult(
        stage="development",
        source=S.Source(path=spec["development"], what="experiment result: the pre-registered statistics on the "
                        "development games, read on their point values", rows="training"),
        games=int(_path_value(dev, spec["games"])), rows=int(_path_value(dev, spec["rows"])),
        rows_unit=spec["rows_unit"], hypotheses=hypotheses, rule=adoption_doc["rule"],
        adoption=adoption, adopted=bool(adoption_doc["adopted"]),
        reading=_path_value(dev, f"{spec['verdict_key']}.protocol"), caveat=spec.get("caveat"),
        not_scored=spec["not_scored"])


def _protocol_status(protocol) -> str:
    if protocol is None:
        return "pre_registered"
    return "development_met_test_pending" if protocol.adopted else "not_adopted_development"


LEDGER = "reports/research/scoring_ledger.json"


NOT_SCORED = re.compile(r"\b(?:Set A|[Tt]he test (?:season|set|games))(?: was| were)? not scored\b")


def _not_scored_problems(eid: str, exp: dict, repo: Path) -> list:
    """A test left unscored on purpose: the source names no test file, the write-up's verdict paragraph says the
    test was not scored, and the scoring ledger holds no entry for the experiment."""
    problems = []
    results = exp.get("results") or {}
    if results.get("test") is not None or results.get("development") is None or exp.get("stages") or exp.get(
            "protocol") or exp.get("adopted_as"):
        problems.append(f"{eid}: a test left unscored has a development file, no test file, and no other shape")
    writeup = exp.get("writeup")
    text = (repo / writeup).read_text(encoding="utf-8") if writeup and (repo / writeup).is_file() else ""
    # the verdict paragraph names the test set and says it was not scored ("Set A was not scored", "the test
    # season was not scored", "Set A not scored"), not merely the words "not scored" about something else
    if not NOT_SCORED.search(" ".join(verdict_paragraph(text))):
        problems.append(f"{eid}: the verdict paragraph of {writeup} does not say the test was not scored")
    ledger_path = repo / LEDGER
    entries = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else []
    if any(isinstance(e, dict) and e.get("experiment") == eid for e in entries):
        problems.append(f"{eid}: {LEDGER} holds an entry for it, so its test was scored")
    return problems


def _rule_flags(results: dict) -> set:
    """The flags a stage description reads as parts of its rule (unformatted key paths)."""
    key_sets = [results.get("keys", {})] + list(results.get("stage_keys", {}).values())
    return {keys[name] for keys in key_sets for name in ("met", "main_met", "reading") if keys.get(name)} | {
        c["met"] for c in results.get("conditions", []) if c.get("met")}


def _parts(exp: dict, repo: Path, writeup_text: str) -> list:
    """The parts of an experiment decided part by part, each read through the stage reader. Each part's file
    flags may name the other parts' flags; a flag no part reads is still refused. A part whose test was not spent
    must have missed its development bar."""
    if exp.get("results") or exp.get("protocol") or exp.get("stages") or exp.get("test_not_scored"):
        raise MethodsProblem(f"{exp['id']}: an experiment decided part by part has no results, protocol, stages or "
                             "test_not_scored of its own")
    others = set().union(*(_rule_flags(p["results"]) for p in exp["parts"]))
    built = []
    for part in exp["parts"]:
        results, test_only = part["results"], part.get("stages") == "test_only"
        dev = None if test_only else _stage(results, "development", repo, exp["threshold"], exp["id"], others,
                                            writeup_text)
        test = (_stage(results, "test", repo, exp["threshold"], exp["id"], others, writeup_text)
                if (dev is not None or test_only) else None)
        if part.get("test_not_scored") and (results.get("test") is not None or dev is None or dev.met):
            raise MethodsProblem(f"{exp['id']} {part['label']}: a part left unscored has no test file and missed its "
                                 "development bar")
        if test_only:
            status = "pre_registered" if test is None else "passed_test" if test.met else "not_passed_test"
        else:
            status = ("pre_registered" if dev is None else "not_adopted_development" if not dev.met
                      else "development_met_test_pending" if test is None
                      else "confirmed" if test.met else "development_met_test_not_met")
        built.append(S.ExperimentPart(label=part["label"], stages="test_only" if test_only else
                                      "development_and_test", development=dev, test=test,
                                      test_not_scored=bool(part.get("test_not_scored")), status=status))
    if exp.get("writeup") and not any(p.development or p.test for p in built):
        raise MethodsProblem(f"{exp['id']}: no part has a result file, but the experiment names its write-up")
    return built


def _test_only_status(test) -> str:
    if test is None:
        return "pre_registered"
    return "passed_test" if test.met else "not_adopted_test"


def _status(dev, test, adopted_as=None) -> str:
    if dev is None:
        return "pre_registered"
    if not dev.met:
        return "not_adopted_development"
    if test is None:
        return "development_met_test_pending"
    if not test.met:
        return "not_adopted_test"
    return "adopted_research" if adopted_as == "research_result" else "adopted"


def load_source(repo: Path = REPO_ROOT) -> dict:
    try:
        return json.loads(_read(repo, SOURCE))
    except json.JSONDecodeError as exc:
        raise MethodsProblem(f"{SOURCE}: not valid JSON ({exc})") from None


def build_methods(repo: Path = REPO_ROOT, source: dict = None) -> S.MethodsFile:
    """The methods file, after the source traces. Raises MethodsProblem when it does not."""
    repo = Path(repo)
    source = load_source(repo) if source is None else source
    problems = trace_problems(source, repo)
    if problems:
        raise MethodsProblem(f"{len(problems)} trace problem(s); the first: {problems[0]}")
    lanes, sources = [], []
    dossier_what = "lane dossier: papers, benchmark rows and pre-registered experiments, copied word for word"
    for lane in source["lanes"]:
        benchmarks = []
        for row in lane["benchmarks"]:
            ours = [S.OurValue(label=v["label"], value=v["value"], lo=v.get("lo"), hi=v.get("hi"), kind=v["kind"],
                               source=v["source"], rows=v["rows"]) for v in row["ours"]]
            sources += [S.Source(path=v.source, what="numbers of ours beside the published ones", rows=v.rows)
                        for v in ours]
            benchmarks.append(S.Benchmark(
                ours=ours, ours_text=row.get("ours_text"), published=row["published"],
                published_values=row["published_values"], paper=row["paper"], where=row.get("where"),
                url=row.get("url"), differs=row["differs"], comparability=row["comparability"],
                comparability_class=comparability_class(row["comparability"])))
        experiments = []
        for exp in lane["experiments"]:
            dev = test = protocol = None
            writeup_text = _read(repo, exp["writeup"]) if exp.get("writeup") else ""
            parts = _parts(exp, repo, writeup_text) if exp.get("parts") else []
            for part in parts:
                for stage in (part.development, part.test):
                    if stage is not None:
                        sources += [stage.source] + list(stage.also_read)
            if exp.get("protocol"):
                protocol = _protocol(exp["protocol"], repo, exp.get("writeup"))
                if protocol is not None:
                    sources.append(protocol.source)
                    sources += [i.source for i in [h.interval for h in protocol.hypotheses]
                                + [a.interval for a in protocol.adoption] if i]
            test_only = exp.get("stages") == "test_only"
            if exp.get("results"):
                dev = None if test_only else _stage(exp["results"], "development", repo, exp["threshold"], exp["id"],
                                                    writeup_text=writeup_text)
                test = (_stage(exp["results"], "test", repo, exp["threshold"], exp["id"], writeup_text=writeup_text)
                        if (dev is not None or test_only) else None)
                if exp.get("test_not_scored") and dev is not None and dev.met:
                    raise MethodsProblem(f"{exp['id']}: the development result met its bar, so a test left unscored "
                                         "would hide a pass")
                if exp.get("writeup") and dev is None and test is None:
                    raise MethodsProblem(f"{exp['id']}: no result file, but the experiment names its write-up "
                                         f"{exp['writeup']}, so it has run")
                for stage in (dev, test):
                    if stage is not None:
                        sources += [stage.source] + list(stage.also_read)
            experiments.append(S.Experiment(
                id=exp["id"], title=exp["title"], what=exp["what"], hypothesis=exp["hypothesis"],
                threshold=exp["threshold"], rank=exp["rank"], gain=exp["gain"], effort=exp["effort"],
                test_data=exp["test_data"], backlog_status=exp["backlog_status"],
                status=("decided_by_part" if parts else "ran_result_in_writeup" if exp.get("result_in_writeup") else
                        _protocol_status(protocol) if protocol is not None else _test_only_status(test) if test_only
                        else _status(dev, test, exp.get("adopted_as"))),
                result_in_writeup=bool(exp.get("result_in_writeup")),
                test_not_scored=bool(exp.get("test_not_scored")), parts=parts,
                adopted_as=exp.get("adopted_as"),
                writeup=exp.get("writeup"), writeup_notes=[_note_place(n)[1] for n in exp.get("writeup_notes", [])],
                stages="test_only" if test_only else "development_and_test", development=dev, test=test,
                protocol=protocol))
            if exp.get("writeup"):
                sources.append(S.Source(path=exp["writeup"], what="the write-up of an experiment that has run",
                                        rows="not_split"))
        lanes.append(S.MethodsLane(
            lane=lane["lane"], dossier=lane["dossier"], report=lane["report"],
            foundations=[S.Paper(**paper) for paper in lane["foundations"]], benchmarks=benchmarks,
            notes=lane["notes"], experiments=experiments))
        sources.append(S.Source(path=lane["dossier"], what=dossier_what, rows="not_split"))
    sources += [S.Source(path=source["backlog"], what="every proposal ranked by expected gain over effort",
                         rows="not_split"),
                S.Source(path=SOURCE, what="the structured source of this file, traced to the files above",
                         rows="not_split")]
    seen, unique = set(), []
    for item in sources:
        key = (item.path, item.what, item.rows)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return S.MethodsFile(schema_version=S.SCHEMA_VERSION, written=source["written"], backlog=source["backlog"],
                         lanes=lanes, sources=unique)


def summary(methods: S.MethodsFile) -> list[str]:
    """What the page will hold, one line a lane, for the command line."""
    lines = []
    for lane in methods.lanes:
        kinds = [row.comparability_class for row in lane.benchmarks]
        statuses = ", ".join(f"{e.id} {e.status}" for e in lane.experiments)
        lines.append(f"{lane.lane}: {len(lane.foundations)} papers, {len(lane.benchmarks)} benchmark rows "
                     f"({kinds.count('roughly')} roughly comparable), {sum(len(r.ours) for r in lane.benchmarks)} "
                     f"numbers of ours; {statuses}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the source of the methods page against its dossiers.")
    parser.add_argument("--check", action="store_true", help="trace the source and print what the page holds")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("pass --check; python3 -m export.site_data --only methods writes the file into a data root")
    source = load_source()
    problems = trace_problems(source)
    for line in problems:
        print(f"PROBLEM: {line}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    for line in summary(build_methods(source=source)):
        print(line)
    print(f"{SOURCE} traces to its dossiers, reports and {BACKLOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
