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
    """The paragraph of a write-up that opens with its verdict ("**Verdict.**"), as a one-item list."""
    return [p for p in dossier_paragraphs(text) if p.startswith("**Verdict.**")]


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
            if (not before or before[-1] in ".?!") and (not after or after[0] == " "):
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
        texts = {}
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

        if source["written"] not in dossier:
            problems.append(f"{name}: {lane['dossier']} does not state the date {source['written']}")
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
                if _flat(exp[key]) != bullets.get(eid, {}).get(key):
                    miss(f"{eid}.{key}", exp[key], f"the {key} bullet under {eid}")
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
            protocol = exp.get("protocol")
            if protocol:
                if results_problem := _protocol_trace(eid, exp, protocol, repo):
                    problems += results_problem
            results = exp.get("results")
            if results:
                if _flat(results["interval"]) != bullets.get(eid, {}).get("interval"):
                    miss(f"{eid}.results.interval", results["interval"], f"the paired test bullet under {eid}")
                for stage in ("development", "test"):
                    if not results[stage].startswith("reports/research/"):
                        problems.append(f"{eid}: the {stage} result is a file under reports/research/")
    for eid in sorted(set(backlog) - listed):
        problems.append(f"{eid}: a row of {source['backlog']} that no lane lists")
    return problems


# ------------------------------------------------------------------------------------------------ the build


def _stage(results: dict, stage: str, repo: Path):
    """The result of one stage, or None when its file does not exist yet."""
    rel = results[stage]
    path = repo / rel
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    keys = results["keys"]
    met = bool(_path_value(doc, keys["met"]))
    upper, threshold = float(_path_value(doc, keys["upper"])), float(_path_value(doc, keys["threshold"]))
    if met != (upper <= threshold):
        raise MethodsProblem(f"{rel}: {keys['met']} is {met}, but its bound {upper} against its threshold "
                             f"{threshold} says {upper <= threshold}")
    # Rounded once, here. The rule's two numbers keep as many decimals as it takes for the rounded comparison to
    # read as the file's own verdict, so rounding can never flip it (an upper end of 0.000502 against 0.0005
    # shows six decimals, not five).
    digits = results["digits"]
    while (round(upper, digits) <= round(threshold, digits)) != met and digits < 17:
        digits += 1

    def num(key: str, places: int = digits) -> float:
        return round(float(_path_value(doc, keys[key] if key in keys else key)), places)

    values = []
    for spec in results["values"]:
        lo = round(float(_path_value(doc, spec["lo"])), spec["digits"]) if spec.get("lo") else None
        hi = round(float(_path_value(doc, spec["hi"])), spec["digits"]) if spec.get("hi") else None
        values.append(S.StageValue(label=spec["label"], value=round(float(_path_value(doc, spec["value"])),
                                                                    spec["digits"]),
                                   lo=lo, hi=hi, point_estimate=lo is None))
    rows = "testset" if stage == "test" else "out_of_fold"
    what = ("experiment result: fitted on every development game, scored once on the D71 test set" if stage == "test"
            else "experiment result: out of fold on the development games")
    return S.StageResult(
        stage=stage, source=S.Source(path=rel, what=what, rows=rows), metric=results["metric"],
        rows=int(_path_value(doc, keys["rows"])), rows_unit=results["rows_unit"],
        games=int(_path_value(doc, keys["games"])), diff=num("diff"), lo=num("lo"), hi=num("hi"),
        interval=results["interval"], threshold=num("threshold"), upper=num("upper"), met=met, values=values)


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


def _protocol(spec: dict, repo: Path):
    """The protocol result of an experiment, or None when its development file does not exist yet. The value of
    each statement and the adoption rule are read from the development file; each valid interval from the post hoc
    file, whose statistic must be the same number as the development file's (so the interval belongs to it). The
    pre-registered ranges of the development file are not read: they are not valid confidence intervals."""
    path = repo / spec["development"]
    if not path.is_file():
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
        boot = stat[spec["interval_key"]]
        return S.PostHocInterval(lo=round(float(boot["lo"]), digits), hi=round(float(boot["hi"]), digits),
                                 method=spec["interval_method"], source=interval_source)

    hypotheses = []
    for item in spec["hypotheses"]:
        h = _path_value(dev, f"{spec['verdict_key']}.{item['key']}")
        raw = h.get("value")
        finite = isinstance(raw, (int, float)) and raw == raw and abs(raw) != float("inf")
        holds = bool(h["holds"]) if finite and h.get("holds") is not None else None
        if (holds is None) != (h.get("result") == "undefined") or (holds is not None and h["result"] !=
                                                                      ("holds" if holds else "fails")):
            raise MethodsProblem(f"{spec['development']}: {item['key']} reads {h.get('result')!r} with holds "
                                 f"{h.get('holds')!r} and value {raw!r}")
        found = interval(item.get("interval"), float(raw)) if finite else None
        hypotheses.append(S.Hypothesis(label=item["key"], statement=h["statement"], threshold=item["threshold"],
                                       value=round(float(raw), digits) if finite else None, holds=holds,
                                       result=h["result"], interval=found, point_estimate=found is None))
    adoption_doc = _path_value(dev, f"{spec['verdict_key']}.{spec['adoption']['key']}")
    adoption = []
    for item in spec["adoption"]["values"]:
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


def _status(dev, test) -> str:
    if dev is None:
        return "pre_registered"
    if not dev.met:
        return "not_adopted_development"
    if test is None:
        return "development_met_test_pending"
    return "adopted" if test.met else "not_adopted_test"


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
            if exp.get("protocol"):
                protocol = _protocol(exp["protocol"], repo)
                if protocol is not None:
                    sources.append(protocol.source)
                    sources += [i.source for i in [h.interval for h in protocol.hypotheses]
                                + [a.interval for a in protocol.adoption] if i]
            if exp.get("results"):
                dev = _stage(exp["results"], "development", repo)
                test = _stage(exp["results"], "test", repo) if dev is not None else None
                sources += [stage.source for stage in (dev, test) if stage is not None]
            experiments.append(S.Experiment(
                id=exp["id"], title=exp["title"], what=exp["what"], hypothesis=exp["hypothesis"],
                threshold=exp["threshold"], rank=exp["rank"], gain=exp["gain"], effort=exp["effort"],
                test_data=exp["test_data"], backlog_status=exp["backlog_status"],
                status=_protocol_status(protocol) if protocol is not None else _status(dev, test),
                writeup=exp.get("writeup"), development=dev, test=test, protocol=protocol))
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
