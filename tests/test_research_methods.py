"""The methods and literature page's data (PLAN.md D72): export/research_methods.py and its contract.

What these tests hold:
- the committed source reports/research/methods.json traces: every copied text is in its dossier word for word,
  every number of ours is a number of the report it names, every published value is in its cell, every link is
  named by the dossier, and the experiments are exactly the rows of docs/research/BACKLOG.md;
- the trace check can fail, on each of those rules;
- results are read from the files the source names, and a test-set result is picked up when its file appears;
- the contract refuses a status the results do not support, and a methods file in a data root validates;
- README.md has a "Methods and literature" section that names every lane and experiment and links the backlog.

Nothing here reads DATA_ROOT, fits or scores anything.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from export import research_methods as R
from export import sample
from export import schemas as S
from export import site_data

REPO = Path(__file__).resolve().parents[1]
QUIET = lambda *_args, **_kwargs: None  # noqa: E731


@pytest.fixture(scope="module")
def source():
    return R.load_source(REPO)


def _lane(doc, name):
    return next(lane for lane in doc["lanes"] if lane["lane"] == name)


# ------------------------------------------------------------------------------------------ the committed source


def test_the_committed_source_traces_and_builds(source):
    assert R.trace_problems(source, REPO) == []
    methods = R.build_methods(REPO, source)
    backlog = R.backlog_rows((REPO / R.BACKLOG).read_text(encoding="utf-8"))
    ids = sorted(e.id for lane in methods.lanes for e in lane.experiments)
    assert ids == sorted(backlog)
    assert [lane.lane for lane in methods.lanes] == ["xfg", "rebounds", "matchups", "gravity", "ghosting",
                                                     "embeddings", "epv", "decisions", "lineups", "winprob",
                                                     "load", "tracking", "genai"]
    assert S.find_private(methods.model_dump_json()) == []
    # every lane cites its dossier and every source path is a file of the repository
    for item in methods.sources:
        assert (REPO / item.path).is_file(), item.path


RAN = ("xfg-E1", "decisions-E1", "lineups-E1", "winprob-E1", "genai-E1")   # the experiments with a result file today


def test_only_the_tests_that_ran_have_a_result_and_every_other_test_is_pre_registered(source):
    methods = R.build_methods(REPO, source)
    states = {e.id: e for lane in methods.lanes for e in lane.experiments}
    others = {eid: e.status for eid, e in states.items() if eid not in RAN}
    assert set(others.values()) == {"pre_registered"}
    assert all(e.development is None and e.test is None and e.protocol is None
               for eid, e in states.items() if eid not in RAN)
    pilot = states["xfg-E1"]
    final = REPO / "reports/research/xfg-E1-final.json"
    for stage, path in ((pilot.development, REPO / "reports/research/xfg-E1-dev.json"), (pilot.test, final)):
        if not path.exists():
            assert stage is None and pilot.status == "development_met_test_pending"
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert stage is not None and stage.met is bool(doc["verdict"]["passes_threshold"])
        assert stage.rows == doc["shots"] and stage.games == doc["games"]
        paired = doc["paired_cvap_minus_iso_shot_weighted"]
        assert (stage.diff, stage.lo, stage.hi) == tuple(round(paired[k], 5) for k in ("diff", "lo", "hi"))
        assert stage.threshold == doc["verdict"]["threshold"] and stage.upper == round(doc["verdict"]["upper_95"], 5)
        assert stage.met == (stage.upper <= stage.threshold)
    if final.exists():
        # the single test score (PLAN.md D80): met on the development games, not on the test set, so not adopted
        assert (pilot.development.met, pilot.test.met, pilot.status) == (True, False, "not_adopted_test")
        assert pilot.writeup == "reports/research/xfg-E1.md" and pilot.test.source.rows == "testset"


def test_every_number_of_ours_is_on_one_line_of_its_report(source):
    """The same rule the trace applies, written out once more so a reader sees it: value, lo and hi together."""
    for lane in source["lanes"]:
        for row in lane["benchmarks"]:
            for ours in row["ours"]:
                lines = (REPO / ours["source"]).read_text(encoding="utf-8").split("\n")
                wanted = {R._unsigned(t) for t in (ours["value"], ours.get("lo"), ours.get("hi")) if t}
                assert any(wanted <= set(R.numbers(line)) for line in lines), (lane["lane"], ours["label"])
                assert ours["source"].startswith("reports/")


# ------------------------------------------------------------------------------------------ the trace can fail


def _problems_after(source, change):
    doc = copy.deepcopy(source)
    change(doc)
    return R.trace_problems(doc, REPO)


@pytest.mark.parametrize("name, change, message", [
    ("a number of ours retyped", lambda d: _lane(d, "xfg")["benchmarks"][0]["ours"][0].update(value="0.6295"),
     "do not stand on one line"),
    ("an interval end from another line", lambda d: _lane(d, "xfg")["benchmarks"][0]["ours"][0].update(hi="0.2233"),
     "do not stand on one line"),
    ("a number of ours from the wrong report",
     lambda d: _lane(d, "xfg")["benchmarks"][0]["ours"][0].update(source="reports/epv.md"), "do not stand on one line"),
    ("a number in a label the report lacks",
     lambda d: _lane(d, "xfg")["benchmarks"][0]["ours"][0].update(label="Log loss over 7777 shots"), "holds 7777"),
    ("a published value not in its cell",
     lambda d: _lane(d, "xfg")["benchmarks"][0]["published_values"].append("0.6523"), "not a number of the published"),
    ("a retyped published cell",
     lambda d: _lane(d, "winprob")["benchmarks"][0].update(
         published=_lane(d, "winprob")["benchmarks"][0]["published"].replace("0.1663", "0.1662")), "word for word"),
    ("a softened comparability verdict",
     lambda d: _lane(d, "winprob")["benchmarks"][0].update(comparability="directly comparable"), "word for word"),
    ("a quote that drops its not", lambda d: _lane(d, "xfg")["benchmarks"][1].update(
        comparability=_lane(d, "xfg")["benchmarks"][1]["comparability"].replace("not directly", "directly")),
     "a whole table cell"),
    ("a quote cut before its qualifier", lambda d: _lane(d, "lineups")["benchmarks"][2].update(
        comparability="roughly comparable as a setting"), "a whole table cell"),
    ("a note cut before the end of its sentence", lambda d: _lane(d, "ghosting")["notes"].__setitem__(
        1, _lane(d, "ghosting")["notes"][1].split(", so")[0] + "."), "whole sentences"),
    ("a hypothesis cut short", lambda d: _lane(d, "xfg")["experiments"][1].update(
        hypothesis=_lane(d, "xfg")["experiments"][1]["hypothesis"].split(",")[0]), "the hypothesis bullet"),
    ("a number of ours in words where none is traced",
     lambda d: _lane(d, "tracking")["benchmarks"][5].update(ours_text="No ball, 48.7% filled"),
     "opening words"),
    ("a table that is not in the cell", lambda d: _lane(d, "xfg")["benchmarks"][0].update(where="Table 9"),
     "not part of the published or paper cell"),
    ("a link the dossier does not name",
     lambda d: _lane(d, "xfg")["benchmarks"][0].update(url="https://example.org/a-paper"), "not named in the dossier"),
    ("a paper link the dossier does not name",
     lambda d: _lane(d, "lineups")["foundations"][0].update(url="https://arxiv.org/abs/9999.99999"),
     "not named in the dossier"),
    ("an edited threshold", lambda d: _lane(d, "xfg")["experiments"][0].update(
        threshold=_lane(d, "xfg")["experiments"][0]["threshold"].replace("+0.0005", "+0.0010")), "word for word"),
    ("a gain that is not the backlog's", lambda d: _lane(d, "load")["experiments"][0].update(gain=5),
     "gain is '5'"),
    ("a rank that is not the backlog's", lambda d: _lane(d, "load")["experiments"][0].update(rank=1),
     "rank is '1'"),
    ("an experiment dropped", lambda d: _lane(d, "epv")["experiments"].pop(), "no lane lists"),
    ("a note that is not the dossier's", lambda d: _lane(d, "epv")["notes"].append("A note we made up."),
     "word for word"),
    ("a date the dossiers do not state", lambda d: d.update(written="2026-11-17"), "does not state the date"),
    ("a result outside reports/research",
     lambda d: _lane(d, "xfg")["experiments"][0]["results"].update(test="reports/xfg-E1-final.json"),
     "under reports/research/"),
])
def test_the_trace_check_can_fail(source, name, change, message):
    problems = _problems_after(source, change)
    assert any(message in line for line in problems), (name, problems[:3])


def test_comparability_is_sorted_by_its_opening_words():
    assert R.comparability_class("roughly comparable: same league") == "roughly"
    assert R.comparability_class("Not directly comparable: different target") == "not_directly"
    assert R.comparability_class("No published comparable number.") == "none"
    assert R.comparability_class("Nothing of ours yet") == "none"
    with pytest.raises(R.MethodsProblem):
        R.comparability_class("comparable enough")


def test_a_link_traces_to_the_dossier_or_to_an_identifier_it_names():
    dossier = "see doi:10.1214/14-AOAS799; arXiv:1405.0231 and https://example.org/x"
    assert R.link_traces("https://example.org/x", dossier)
    assert R.link_traces("https://doi.org/10.1214/14-AOAS799", dossier)
    assert R.link_traces("https://arxiv.org/abs/1405.0231", dossier)
    assert not R.link_traces("https://doi.org/10.1214/14-AOAS7", dossier)
    assert not R.link_traces("https://arxiv.org/abs/1405.0232", dossier)


# ------------------------------------------------------------------------------------------ results


def _repo_copy(tmp_path, source) -> Path:
    """A copy of the files the source reads, so a test can add a result file without touching the repository."""
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "docs" / "research", repo / "docs" / "research")
    names = {R.SOURCE}
    for lane in source["lanes"]:
        names.add(lane["report"])
        names.update(v["source"] for row in lane["benchmarks"] for v in row["ours"])
        for experiment in lane["experiments"]:
            names.update(n for n in [experiment.get("writeup")] + list((experiment.get("results") or {}).get(
                stage) for stage in ("development", "test")) + [(experiment.get("protocol") or {}).get(key) for key in
                                                                  ("development", "intervals")]
                if n and (REPO / n).exists())
    for name in names:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, repo / name)
    return repo


def _pilot(methods):
    return next(e for lane in methods.lanes for e in lane.experiments if e.id == "xfg-E1")


def test_a_test_set_result_is_picked_up_when_its_file_appears(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    final = repo / "reports" / "research" / "xfg-E1-final.json"
    if final.exists():
        final.unlink()
    assert _pilot(R.build_methods(repo, source)).status == "development_met_test_pending"
    doc = json.loads((repo / "reports/research/xfg-E1-dev.json").read_text(encoding="utf-8"))
    doc.update(stage="a test file written by this test", shots=1700, games=10)
    final.write_text(json.dumps(doc), encoding="utf-8")
    pilot = _pilot(R.build_methods(repo, source))
    assert pilot.status == "adopted" and pilot.test.rows == 1700 and pilot.test.source.rows == "testset"
    doc["paired_cvap_minus_iso_shot_weighted"].update(lo=-0.0002, diff=0.0004, hi=0.0011)
    doc["verdict"].update(upper_95=0.0011, passes_threshold=False)
    final.write_text(json.dumps(doc), encoding="utf-8")
    pilot = _pilot(R.build_methods(repo, source))
    assert pilot.status == "not_adopted_test" and not pilot.test.met and pilot.test.upper == 0.0011


def test_a_development_result_that_misses_its_bar_is_not_adopted(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["paired_cvap_minus_iso_shot_weighted"].update(lo=-0.0001, diff=0.0003, hi=0.0009)
    doc["verdict"].update(upper_95=0.0009, passes_threshold=False)
    path.write_text(json.dumps(doc), encoding="utf-8")
    pilot = _pilot(R.build_methods(repo, source))
    # a test score that exists is still shown, but adoption needs the bar met on the development games first
    assert pilot.status == "not_adopted_development" and not pilot.development.met
    assert (pilot.test is not None) == (repo / "reports/research/xfg-E1-final.json").exists()


@pytest.mark.parametrize("upper, met", [(0.000502, False), (0.000499, True), (0.0005, True)])
def test_rounding_never_flips_the_rule_near_its_threshold(source, tmp_path, upper, met):
    """A bound a hair from the threshold is shown with enough decimals that the rounded comparison reads as the
    file's own verdict, and met keeps its meaning (the file's verdict)."""
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["paired_cvap_minus_iso_shot_weighted"].update(lo=upper - 0.001, diff=upper - 0.0005, hi=upper)
    doc["verdict"].update(threshold=0.0005, upper_95=upper, passes_threshold=met)
    path.write_text(json.dumps(doc), encoding="utf-8")
    dev = _pilot(R.build_methods(repo, source)).development
    assert dev.met is met and (dev.upper <= dev.threshold) is met and dev.upper == dev.hi
    if upper == 0.000502:
        assert dev.upper == 0.000502


def test_a_result_file_whose_verdict_contradicts_its_numbers_is_a_problem(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["verdict"].update(upper_95=0.0009, passes_threshold=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="upper_95 is not the upper end"):
        R.build_methods(repo, source)
    doc["paired_cvap_minus_iso_shot_weighted"].update(diff=0.0004, hi=0.0009)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="passes_threshold is True"):
        R.build_methods(repo, source)


def test_numbers_are_read_in_one_spelling_with_the_sign_kept():
    minus = "\u2212"
    assert R.numbers(f"a {minus}0.5 b +0.0012 c 0.5 d -0.25") == ["-0.5", "0.0012", "0.5", "-0.25"]
    assert R.number_token("+0.0012") == "0.0012" and R.number_token(f"{minus}0.014") == "-0.014"
    # a report that writes the true minus: an unsigned value of ours does not match it, a signed one does
    tokens = set(R.numbers(f"| gap | {minus}0.014 [{minus}0.116, 0.100] |"))
    assert R.number_token("0.014") not in tokens and R.number_token("-0.014") in tokens
    assert {R.number_token(v) for v in ("-0.014", "-0.116", "+0.100")} <= tokens


def test_the_trace_matches_a_true_minus_in_a_report_only_with_its_sign(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    report = repo / "reports" / "decisions.md"
    report.write_text(report.read_text(encoding="utf-8").replace("-0.014", "\u22120.014"), encoding="utf-8")
    assert R.trace_problems(source, repo) == []
    doc = copy.deepcopy(source)
    for row in _lane(doc, "decisions")["benchmarks"]:
        for value in row["ours"]:
            if value["value"] == "-0.014":
                value["value"] = "0.014"
    assert any("do not stand on one line" in p for p in R.trace_problems(doc, repo))


def test_a_result_file_without_a_named_key_is_a_problem(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["paired_cvap_minus_iso_shot_weighted"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="paired_cvap_minus_iso_shot_weighted"):
        R.build_methods(repo, source)


def test_a_source_that_does_not_trace_is_not_built(source):
    doc = copy.deepcopy(source)
    _lane(doc, "xfg")["benchmarks"][0]["ours"][0]["value"] = "0.6000"
    with pytest.raises(R.MethodsProblem, match="trace problem"):
        R.build_methods(REPO, doc)


# ------------------------------------------------------------------------------------------ the contract


def test_the_contract_refuses_a_status_the_results_do_not_support(source):
    methods = R.build_methods(REPO, source)
    pilot = _pilot(methods).model_dump()
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**pilot, "status": "adopted"})
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**pilot, "test": None})
    with pytest.raises(ValidationError, match="scored after the development result"):
        S.Experiment(**{**pilot, "development": None})
    other = next(e for lane in methods.lanes for e in lane.experiments if e.status == "pre_registered").model_dump()
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**other, "status": "development_met_test_pending"})


def test_the_contract_refuses_a_number_of_ours_that_is_not_a_number_or_lacks_half_an_interval():
    good = dict(label="Log loss", value="0.6294", lo="0.6277", hi="0.6316", kind="estimate",
                source="reports/xfg.md", rows="out_of_fold")
    S.OurValue(**good)
    for bad in ({"value": "about 0.63"}, {"hi": None}, {"kind": "setting"}):
        with pytest.raises(ValidationError):
            S.OurValue(**{**good, **bad})
    with pytest.raises(ValidationError, match="holds no number"):
        S.Benchmark(ours=[], ours_text="ours is 0.5", published="none", published_values=[], paper="none",
                    differs="x", comparability="No published comparable number.", comparability_class="none")


def test_the_methods_file_is_a_contract_file_in_both_kinds_of_data_root(tmp_path, source):
    assert S.model_for("methods.json") is S.MethodsFile
    root = tmp_path / "sample"
    sample.write_sample(root)
    assert S.validate_tree(root) == []
    statuses = {e.status for lane in S.MethodsFile.model_validate_json(
        (root / "methods.json").read_text()).lanes for e in lane.experiments}
    assert statuses == {"pre_registered", "development_met_test_pending", "not_adopted_test", "not_adopted_development",
                        "passed_test"}
    # real data: the committed source, through the methods step of export.site_data, with its manifest
    out = tmp_path / "real"
    problems, info = site_data.run(out, None, None, only=["methods"], log=QUIET)
    assert info["methods"]["lanes"] == len(source["lanes"]) and info["methods"]["experiments"]["xfg-E1"]
    assert not [p for p in problems if "methods.json" in p and "missing" not in p]
    manifest = json.loads((out / "manifest.json").read_text())
    assert [f["path"] for f in manifest["files"]] == ["methods.json"]
    obj = S.MethodsFile.model_validate_json((out / "methods.json").read_text())
    assert S._content_problems("methods.json", obj, synthetic=False) == []


def test_the_real_data_rules_refuse_a_number_of_ours_from_outside_reports(source):
    methods = R.build_methods(REPO, source)
    doc = methods.model_dump()
    doc["lanes"][0]["benchmarks"][0]["ours"][0]["source"] = "docs/research/xfg.md"
    obj = S.MethodsFile.model_validate(doc)
    assert any("never from the test set" in p for p in S._content_problems("methods.json", obj, synthetic=False))


# ------------------------------------------------------------------------------------------ README


def test_the_readme_has_a_methods_section_naming_every_lane_and_experiment(source):
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "\n## Methods and literature\n" in readme
    section = readme.split("\n## Methods and literature\n", 1)[1].split("\n## ", 1)[0]
    assert "](docs/research/BACKLOG.md)" in section
    for lane in source["lanes"]:
        assert f"]({lane['dossier']})" in section, lane["lane"]
        for experiment in lane["experiments"]:
            assert f"`{experiment['id']}`" in section, experiment["id"]


# ------------------------------------------------------------------------------------------ a protocol result


DEC_DEV = "reports/research/decisions-E1-dev.json"
DEC_POSTHOC = "reports/research/decisions-E1-posthoc.json"
CAVEAT = ("Post hoc, not pre-registered: stepped down to one game, the target's reliability is about 0.04, which "
          "would meet H1's intent; part of the half-season reliability may be team offence shared by teammates (not "
          "measured).")


def _decisions(methods):
    return next(e for lane in methods.lanes for e in lane.experiments if e.id == "decisions-E1")


def test_the_decisions_protocol_is_read_from_its_files_with_valid_intervals_only(source):
    experiment = _decisions(R.build_methods(REPO, source))
    dev = json.loads((REPO / DEC_DEV).read_text(encoding="utf-8"))["verdict"]
    posthoc = json.loads((REPO / DEC_POSTHOC).read_text(encoding="utf-8"))["valid_intervals"]["statistics"]
    protocol = experiment.protocol
    assert experiment.status == "not_adopted_development" and experiment.development is None
    assert experiment.test is None and experiment.writeup == "reports/research/decisions-E1.md"
    assert [h.label for h in protocol.hypotheses] == ["H1", "H2", "H3"]
    for h, statistic in zip(protocol.hypotheses, ("target.split_half_r", "frozen.stability",
                                                  "frozen.r2_on_kind_shares")):
        assert h.statement == dev[h.label]["statement"] and h.result == dev[h.label]["result"]
        assert h.value == round(dev[h.label]["value"], 3) and h.holds is dev[h.label]["holds"]
        boot = posthoc[statistic]["player_bootstrap"]
        assert (h.interval.lo, h.interval.hi) == (round(boot["lo"], 3), round(boot["hi"], 3))
        assert h.interval.source.path == DEC_POSTHOC and not h.point_estimate
    assert [h.result for h in protocol.hypotheses] == ["fails", "holds", "fails"]
    assert protocol.adopted is False and protocol.rule == dev["adoption"]["rule"]
    assert [a.value for a in protocol.adoption] == [round(dev["adoption"]["stability"], 3),
                                                   round(dev["adoption"]["r2_on_kind_shares"], 3)]
    assert protocol.caveat == CAVEAT and len(protocol.not_scored) == 3


def _invalid_numbers():
    """Every end of a range that is not a valid confidence interval (PLAN.md D86): the pre-registered game
    bootstrap ranges of the development file (interval_95 of each statement, the rule's stability_lo) and every
    game_basic range of the post hoc file. Each as the file holds it and rounded as the page would round it."""
    dev = json.loads((REPO / DEC_DEV).read_text(encoding="utf-8"))["verdict"]
    posthoc = json.loads((REPO / DEC_POSTHOC).read_text(encoding="utf-8"))["valid_intervals"]["statistics"]
    raw = [end for key in ("H1", "H2", "H3") for end in dev[key]["interval_95"]] + [dev["adoption"]["stability_lo"]]
    raw += [stat["game_basic"][end] for stat in posthoc.values() if "game_basic" in stat for end in ("lo", "hi")]
    return {float(x) for x in raw} | {round(float(x), 3) for x in raw}


def _numbers_of(document) -> set:
    """Every number of a JSON document, as floats: numeric fields and the numbers inside its strings."""
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
        elif isinstance(value, str):
            found.update(float(t.rstrip("%").replace(",", "")) for t in R.numbers(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(document)
    return found


def _valid_numbers(protocol) -> set:
    """The numbers the protocol may legitimately show: its values and the ends of its valid intervals."""
    found = set()
    for item in list(protocol.hypotheses) + list(protocol.adoption):
        if item.value is not None:
            found.add(item.value)
        if item.interval is not None:
            found.update((item.interval.lo, item.interval.hi))
    return found


def test_no_range_that_is_not_a_valid_interval_reaches_the_written_file(source, tmp_path):
    """Compared as numbers, exactly: no end of a pre-registered game range or of a game_basic range is in the
    written methods.json, except where the same number is a value the page reads validly (0.961 is the adjusted
    stability and the upper end of the frozen stability's valid interval)."""
    protocol = _decisions(R.build_methods(REPO, source)).protocol
    leaks = _invalid_numbers() - _valid_numbers(protocol)
    assert 0.902 in leaks and 0.961 not in leaks
    assert _numbers_of(protocol.model_dump(mode="json")) & leaks == set()
    out = tmp_path / "out"
    site_data.run(out, None, None, only=["methods"], log=QUIET)
    written = json.loads((out / "methods.json").read_text(encoding="utf-8"))
    entry = next(e for lane in written["lanes"] for e in lane["experiments"] if e["id"] == "decisions-E1")
    assert _numbers_of(entry) & leaks == set()


def test_the_caveat_is_word_for_word_the_verdict_paragraph_of_the_write_up():
    text = (REPO / "reports/research/decisions-E1.md").read_text(encoding="utf-8")
    assert R.is_sentences(CAVEAT, R.verdict_paragraph(text))
    assert not R.is_sentences(CAVEAT.replace("about 0.04", "about 0.05"), R.verdict_paragraph(text))
    # a sentence of the write-up outside its verdict paragraph is not the verdict's caveat
    assert not R.is_sentences("A note on H1's unit.", R.verdict_paragraph(text))


@pytest.mark.parametrize("name, change, message", [
    ("an edited caveat", lambda p: p.update(caveat=CAVEAT.replace("(not measured)", "(measured)")),
     "verdict paragraph"),
    ("a not-scored reason cut short", lambda p: p["not_scored"].__setitem__(0, p["not_scored"][0].split(". ")[0]),
     "not-scored reason"),
    ("a protocol file outside reports/research",
     lambda p: p.update(intervals="reports/decisions-E1-posthoc.json"), "not under reports/research/"),
])
def test_the_protocol_trace_can_fail(source, name, change, message):
    problems = _problems_after(source, lambda d: change(next(
        e for e in _lane(d, "decisions")["experiments"] if e["id"] == "decisions-E1")["protocol"]))
    assert any(message in line for line in problems), (name, problems[:3])


def test_an_interval_that_belongs_to_another_statistic_is_refused(source):
    doc = copy.deepcopy(source)
    spec = next(e for e in _lane(doc, "decisions")["experiments"] if e["id"] == "decisions-E1")["protocol"]
    spec["hypotheses"][0]["interval"] = "frozen.split_half_r"
    with pytest.raises(R.MethodsProblem, match="belongs to another number"):
        R.build_methods(REPO, doc)
    spec["hypotheses"][0]["interval"] = "no.such_statistic"
    with pytest.raises(R.MethodsProblem, match="no statistic"):
        R.build_methods(REPO, doc)


def test_an_undefined_statistic_reads_undefined_with_no_interval(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / DEC_DEV
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["verdict"]["H3"].update(value=None, holds=None, result="undefined")
    path.write_text(json.dumps(doc), encoding="utf-8")
    h3 = _decisions(R.build_methods(repo, source)).protocol.hypotheses[2]
    assert (h3.value, h3.holds, h3.result, h3.interval, h3.point_estimate) == (None, None, "undefined", None, True)
    doc["verdict"]["H3"].update(value=0.342, holds=None, result="holds")
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="H3 reads"):
        R.build_methods(repo, source)


def test_the_contract_refuses_a_status_or_a_reading_the_protocol_files_do_not_support(source):
    experiment = _decisions(R.build_methods(REPO, source)).model_dump()
    for status in ("development_met_test_pending", "adopted", "not_adopted_test"):
        with pytest.raises(ValidationError, match="the results say"):
            S.Experiment(**{**experiment, "status": status})
    with pytest.raises(ValidationError, match="has run"):
        S.Experiment(**{**experiment, "status": "pre_registered"})
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**experiment, "status": "pre_registered", "writeup": None})
    adopted = {**experiment["protocol"], "adopted": True}
    with pytest.raises(ValidationError, match="has no status"):
        S.Experiment(**{**experiment, "protocol": adopted})
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**experiment, "protocol": {**adopted, "not_scored": []}})
    pilot = next(e for lane in R.build_methods(REPO, source).lanes for e in lane.experiments if e.id == "xfg-E1")
    with pytest.raises(ValidationError, match="replaces the stage results"):
        S.Experiment(**{**experiment, "development": pilot.development.model_dump()})
    h1 = experiment["protocol"]["hypotheses"][0]
    for bad, message in (({"result": "holds"}, "holds or fails"), ({"point_estimate": True}, "valid interval"),
                         ({"threshold": "below 0.3"}, "part of the statement"), ({"value": None}, "holds or fails")):
        with pytest.raises(ValidationError, match=message):
            S.Hypothesis(**{**h1, **bad})
    with pytest.raises(ValidationError, match="rows training"):
        S.ProtocolResult(**{**experiment["protocol"], "source": {**experiment["protocol"]["source"],
                                                                "rows": "heldout"}})


def test_only_the_bootstrap_by_player_may_be_shown(source):
    doc = copy.deepcopy(source)
    spec = next(e for e in _lane(doc, "decisions")["experiments"] if e["id"] == "decisions-E1")["protocol"]
    spec["interval_key"] = "game_basic"
    with pytest.raises(R.MethodsProblem, match="player_bootstrap interval only"):
        R.build_methods(REPO, doc)
    spec["interval_key"] = "player_bootstrap"
    assert _decisions(R.build_methods(REPO, doc)).protocol.adopted is False


def test_a_missing_interval_or_development_file_is_a_problem_not_a_crash(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / DEC_POSTHOC
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["valid_intervals"]["statistics"]["frozen.stability"]["player_bootstrap"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="no player_bootstrap interval"):
        R.build_methods(repo, source)
    (repo / DEC_DEV).unlink()
    with pytest.raises(R.MethodsProblem, match="names its write-up"):
        R.build_methods(repo, source)


@pytest.mark.parametrize("change, message", [
    (lambda v: v["H2"].update(holds=False, result="fails"), "says holds False"),
    (lambda v: v["H1"].update(holds=True, result="holds"), "says holds True"),
    (lambda v: v["adoption"].update(adopted=True, result="adopted"), "its rule read on its own values"),
    (lambda v: v["adoption"].pop("stability_lo"), "has no stability_lo"),
])
def test_a_reading_the_values_do_not_support_is_refused(source, tmp_path, change, message):
    repo = _repo_copy(tmp_path, source)
    path = repo / DEC_DEV
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc["verdict"])
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match=message):
        R.build_methods(repo, source)


def test_the_contract_keeps_a_value_inside_its_interval_and_a_write_up_off_a_test_not_run(source):
    experiment = _decisions(R.build_methods(REPO, source)).model_dump()
    h1 = experiment["protocol"]["hypotheses"][0]
    with pytest.raises(ValidationError, match="inside its valid interval"):
        S.Hypothesis(**{**h1, "value": 0.6, "holds": False, "result": "fails"})
    a = experiment["protocol"]["adoption"][0]
    with pytest.raises(ValidationError, match="inside its valid interval"):
        S.AdoptionValue(**{**a, "value": 0.5})
    other = next(e for lane in R.build_methods(REPO, source).lanes for e in lane.experiments
                 if e.status == "pre_registered").model_dump()
    with pytest.raises(ValidationError, match="a write-up belongs to an experiment that has run"):
        S.Experiment(**{**other, "writeup": "reports/research/decisions-E1.md"})


# ------------------------------------------------------------------------------------------ a test-only result


LIN_FINAL = "reports/research/lineups-E1-final.json"


def _lineups(methods):
    return next(e for lane in methods.lanes for e in lane.experiments if e.id == "lineups-E1")


def test_lineups_is_a_scored_test_with_two_conditions_read_from_its_final_file(source):
    experiment = _lineups(R.build_methods(REPO, source))
    final = json.loads((REPO / LIN_FINAL).read_text(encoding="utf-8"))
    paired = final["whole_game"]["paired"]["rapm_minus_plus_minus"]
    r2 = final["whole_game"]["r2_vs_floor"]["rapm"]
    assert (experiment.stages, experiment.status, experiment.development) == ("test_only", "passed_test", None)
    test = experiment.test
    assert (test.diff, test.lo, test.hi) == (round(paired["point"], 2), round(paired["ci95"][0], 2),
                                             round(paired["ci95"][1], 2))
    assert test.upper == test.hi and test.threshold == 0 and test.comparison == "below"
    assert test.met is final["verdict"]["passes"] is True
    assert test.games == final["games"] and test.rows == final["stints"] and test.source.rows == "testset"
    (condition,) = test.conditions
    assert (condition.value, condition.lo, condition.hi) == (round(r2["point"], 3), round(r2["ci95"][0], 3),
                                                             round(r2["ci95"][1], 3))
    assert (condition.bound, condition.comparison, condition.threshold, condition.met) == ("lo", "above", 0, True)
    text = (REPO / "reports/research/lineups-E1.md").read_text(encoding="utf-8")
    assert len(experiment.writeup_notes) == 5
    assert all(R.is_sentences(note, R.verdict_paragraph(text)) for note in experiment.writeup_notes[:2])
    assert "1298 of 1300 games" in experiment.writeup_notes[1]


@pytest.mark.parametrize("change, message", [
    (lambda d: d["verdict"].update(passes=False), "verdict.passes is False"),
    (lambda d: d["verdict"].update(diff_ci_below_zero=False), "diff_ci_below_zero disagrees"),
    (lambda d: d["verdict"].update(r2_above_zero_with_interval=False), "r2_above_zero_with_interval disagrees"),
    (lambda d: d["verdict"].update(rule=d["verdict"]["rule"].replace("below 0", "below one")
                                   .replace("above 0", "above one")), "the rule does not state below 0"),
])
def test_a_test_file_whose_verdict_contradicts_its_numbers_is_refused(source, tmp_path, change, message):
    repo = _repo_copy(tmp_path, source)
    path = repo / LIN_FINAL
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match=message):
        R.build_methods(repo, source)


def test_a_failed_condition_fails_the_test_and_a_missing_final_file_with_a_write_up_is_a_problem(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / LIN_FINAL
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["whole_game"]["r2_vs_floor"]["rapm"].update(point=0.02, ci95=[-0.01, 0.05])
    doc["verdict"].update(r2_above_zero_with_interval=False, passes=False)
    path.write_text(json.dumps(doc), encoding="utf-8")
    experiment = _lineups(R.build_methods(repo, source))
    assert experiment.status == "not_adopted_test" and not experiment.test.conditions[0].met
    path.unlink()
    with pytest.raises(R.MethodsProblem, match="names its write-up"):
        R.build_methods(repo, source)


def test_the_contract_refuses_a_test_only_reading_the_files_do_not_support(source):
    experiment = _lineups(R.build_methods(REPO, source)).model_dump()
    for status in ("adopted", "development_met_test_pending", "not_adopted_test"):
        with pytest.raises(ValidationError, match="the results say"):
            S.Experiment(**{**experiment, "status": status})
    pilot = next(e for lane in R.build_methods(REPO, source).lanes for e in lane.experiments if e.id == "xfg-E1")
    with pytest.raises(ValidationError, match="no development verdict"):
        S.Experiment(**{**experiment, "development": pilot.development.model_dump()})
    test = experiment["test"]
    with pytest.raises(ValidationError, match="every further condition"):
        S.StageResult(**{**test, "conditions": [{**test["conditions"][0], "lo": -0.01, "met": False}]})
    with pytest.raises(ValidationError, match="meets the threshold"):
        S.Condition(**{**test["conditions"][0], "met": False})
    # strictly below: an upper end equal to the threshold does not meet the rule
    with pytest.raises(ValidationError, match="every further condition"):
        S.StageResult(**{**test, "diff": -0.5, "hi": 0.0, "upper": 0.0})
    with pytest.raises(ValidationError, match="a write-up belongs"):
        S.Experiment(**{**experiment, "test": None, "status": "pre_registered"})


# ------------------------------------------------------------------------------------------ the second review


def _exp(doc, eid):
    return next(e for lane in doc["lanes"] for e in lane["experiments"] if e["id"] == eid)


def _lineups_results(doc):
    return _exp(doc, "lineups-E1")["results"]


@pytest.mark.parametrize("name, change, message", [
    ("main comparison at most where the file says strictly below",
     lambda d: _lineups_results(d).update(comparison="at_most"), "does not state at most 0"),
    ("a condition reading the upper end", lambda d: _lineups_results(d)["conditions"][0].update(bound="hi"),
     "reads the hi end"),
    ("a condition at least where the file says above",
     lambda d: _lineups_results(d)["conditions"][0].update(comparison="at_least"), "does not state at least 0"),
    ("no conditions while the file has a second flag", lambda d: _lineups_results(d).update(conditions=[]),
     "r2_above_zero_with_interval"),
    ("a condition without its met key", lambda d: _lineups_results(d)["conditions"][0].pop("met"), "lacks ['met']"),
    ("no main flag while the file has one", lambda d: _lineups_results(d)["keys"].pop("main_met"),
     "diff_ci_below_zero"),
    ("an xfg flag outside the rule left unnamed", lambda d: _exp(d, "xfg-E1")["results"].pop("flags_outside_rule"),
     "width_p95_under_0_01"),
    ("decisions adoption read as at most 0.5", lambda d: _exp(d, "decisions-E1")["protocol"]["adoption"][
        "conditions"][0].update(threshold="at most 0.5"), "does not state 'at most 0.5'"),
    ("an unknown main comparison", lambda d: _lineups_results(d).update(comparison="under"), "is not at_most or below"),
    ("a condition without its bound", lambda d: _lineups_results(d)["conditions"][0].pop("bound"), "lacks ['bound']"),
    ("a condition without its threshold", lambda d: _lineups_results(d)["conditions"][0].pop("threshold"),
     "lacks ['threshold']"),
    ("no conditions list", lambda d: _lineups_results(d).pop("conditions"), "no list of further conditions"),
    ("an unknown condition comparison", lambda d: _lineups_results(d)["conditions"][0].update(comparison="over"),
     "unknown comparison"),
])
def test_the_reviewers_probes_are_refused(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = R.trace_problems(doc, REPO)
    if not problems:
        with pytest.raises(R.MethodsProblem) as caught:
            R.build_methods(REPO, doc)
        problems = [str(caught.value)]
    assert any(message in line for line in problems), (name, problems[:3])


@pytest.mark.parametrize("name, change, message", [
    ("a test-only experiment naming a development file",
     lambda d: _lineups_results(d).update(development="reports/research/lineups-E1-dev.json"),
     "names no development result file"),
    ("stages beside a protocol", lambda d: _exp(d, "decisions-E1").update(stages="test_only"),
     "stages and a protocol together"),
    ("a section note from another section", lambda d: _exp(d, "lineups-E1")["writeup_notes"][2].update(
        section="Gate"), "the section 'Gate'"),
    ("a section note cut short", lambda d: _exp(d, "lineups-E1")["writeup_notes"][2].update(
        text="In both development checks it loses to home court alone. A pass on the first condition is therefore "
             "close to certain and says little;"), "whole sentences"),
])
def test_the_trace_refuses_a_shape_the_files_do_not_support(source, name, change, message):
    problems = _problems_after(source, change)
    assert any(message in line for line in problems), (name, problems[:3])


def test_lineups_quotes_the_commitments_and_the_dropped_games_word_for_word(source):
    notes = _lineups(R.build_methods(REPO, source)).writeup_notes
    assert len(notes) == 5
    assert any("the informative condition is R squared against the floor" in note for note in notes)
    assert any("is a deviation from the pre-registration's \"all test games\"" in note for note in notes)
    assert any("The frozen builder dropped 16 whole" in note and "1316 listed" in note for note in notes)
    final = json.loads((REPO / LIN_FINAL).read_text(encoding="utf-8"))
    assert (final["test_games_listed"], final["games_dropped_shifted_periods"], final["games"]) == (1316, 16, 1300)


def test_the_states_helper_reads_a_comparison_right_before_its_threshold():
    assert R.states("interval wholly below 0, and above 0 with its interval", "below", 0)
    assert R.states("an upper bound of at most +0.0005", "at_most", 0.0005)
    assert R.states("stability >= 0.5 with", "at_least", 0.5)
    assert not R.states("interval wholly below 0", "at_most", 0)
    assert not R.states("stability >= 0.5", "at_most", 0.5)
    assert not R.states("below 0.25", "below", 0.2)


# ------------------------------------------------------------------------------------------ a rule with its sign


WIN_DEV, WIN_FINAL = "reports/research/winprob-E1-dev.json", "reports/research/winprob-E1-final.json"
BUCKETS = ["48-36", "36-24", "24-12", "12-4", "4-0", "OT"]


def _winprob(methods):
    return next(e for lane in methods.lanes for e in lane.experiments if e.id == "winprob-E1")


def test_winprob_shows_both_stages_with_the_signed_rule_and_every_bucket(source):
    experiment = _winprob(R.build_methods(REPO, source))
    assert (experiment.status, experiment.adopted_as) == ("adopted_research", "research_result")
    files = {"development": (json.loads((REPO / WIN_DEV).read_text())["pooled_forward"], None),
             "test": (json.loads((REPO / WIN_FINAL).read_text())["scored"]["results"], None)}
    for stage in (experiment.development, experiment.test):
        data = files[stage.stage][0]
        paired = data["paired"]["e1_minus_baseline"]
        mean = paired["mean_bucket_brier"]
        assert (stage.diff, stage.lo, stage.hi) == tuple(round(mean[k], 4) for k in ("point", "lo", "hi"))
        assert (stage.upper, stage.threshold, stage.comparison, stage.met) == (stage.hi, 0, "below", True)
        (size,) = stage.conditions
        assert (size.bound, size.comparison, size.threshold, size.met) == ("point", "at_most",
                                                                           -data["verdict"]["threshold"], True)
        assert [b.label for b in stage.guard.buckets] == BUCKETS and stage.guard.met
        for bucket in stage.guard.buckets:
            cell = paired["bucket_brier"][bucket.label]
            assert (bucket.lo, bucket.worse) == (round(cell["lo"], 4), cell["lo"] > 0)
        slopes = [v for v in stage.values if v.label.startswith("Calibration slope of the drift arm")]
        assert len(slopes) == 6 and all(v.point_estimate and v.lo is None for v in slopes)
        assert stage.games == data["games"] and stage.rows == data["rows"] and stage.design
    text = (REPO / "reports/research/winprob-E1.md").read_text(encoding="utf-8")
    assert len(experiment.writeup_notes) == 5
    assert all(R.is_sentences(note, R.verdict_paragraph(text)) for note in experiment.writeup_notes[:3])
    assert "adopted as a research result" in experiment.writeup_notes[0]


def _win(doc):
    return _exp(doc, "winprob-E1")["results"]


@pytest.mark.parametrize("name, change, message", [
    ("the size read without its sign", lambda d: _win(d)["conditions"][0].update(sign=1), "does not state at_most 0.002"),
    ("a guard that leaves out overtime", lambda d: _win(d)["guard"].update(buckets=BUCKETS[:5]), "every bucket"),
    ("a guard on the upper end", lambda d: _win(d)["guard"].update(bound="hi"), "the guard reads the hi end"),
    ("the straddle flag left unnamed", lambda d: _win(d).pop("flags_outside_rule"), "interval_straddles_threshold"),
    ("the size condition without its flag", lambda d: _win(d)["conditions"][0].pop("met"), "lacks ['met']"),
    ("a rule quote that is not the write-up's", lambda d: _win(d)["rule_quote"].update(
        text=_win(d)["rule_quote"]["text"].replace("-0.002", "-0.001")), "rule quote"),
    ("a research adoption with no note", lambda d: _exp(d, "winprob-E1").update(writeup_notes=[]), "adopted_as"),
])
def test_the_signed_rule_and_its_guard_are_refused_when_the_source_misreads_them(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = R.trace_problems(doc, REPO)
    if not problems:
        with pytest.raises(R.MethodsProblem) as caught:
            R.build_methods(REPO, doc)
        problems = [str(caught.value)]
    assert any(message.replace("at_most", "at most") in line for line in problems), (name, problems[:3])


@pytest.mark.parametrize("change, message", [
    (lambda v: v.update(buckets_worse=["OT"]), "buckets_worse lists"),
    (lambda v: v.update(size_at_least_threshold=False), "size_at_least_threshold disagrees"),
    (lambda v: v.update(interval_excludes_zero=False), "interval_excludes_zero disagrees"),
    (lambda v: v.update(adopt=False), "adopt is False"),
])
def test_a_winprob_file_whose_verdict_contradicts_its_numbers_is_refused(source, tmp_path, change, message):
    repo = _repo_copy(tmp_path, source)
    path = repo / WIN_FINAL
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc["scored"]["results"]["verdict"])
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match=message):
        R.build_methods(repo, source)


def test_the_contract_keeps_research_adoption_and_the_guard_honest(source):
    experiment = _winprob(R.build_methods(REPO, source)).model_dump()
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**experiment, "status": "adopted"})
    with pytest.raises(ValidationError, match="the results say"):
        S.Experiment(**{**experiment, "adopted_as": None})
    failed = {**experiment["test"], "met": False, "guard": {**experiment["test"]["guard"], "met": True}}
    with pytest.raises(ValidationError):
        S.StageResult(**failed)
    guard = experiment["test"]["guard"]
    with pytest.raises(ValidationError, match="worse says"):
        S.BucketGuard(**{**guard, "buckets": [{**guard["buckets"][0], "worse": True}] + guard["buckets"][1:]})
    with pytest.raises(ValidationError, match="no bucket is worse"):
        S.BucketGuard(**{**guard, "met": False})


# ------------------------------------------------------------------------------------------ builder refusals


def _drop_threshold_key(d):
    condition = _win(d)["conditions"][0]
    condition.pop("threshold_key")
    condition.pop("sign")
    condition["threshold"] = -0.002


@pytest.mark.parametrize("name, change, message", [
    ("(a) a guard read from another paired object", lambda d: _win(d)["guard"].update(
        path="{data}.paired.ablation_minus_baseline.bucket_brier"), "not a table of the paired object"),
    ("(b) a rule flag named as outside the rule", lambda d: _win(d)["flags_outside_rule"].update(
        {"{verdict}.size_at_least_threshold": "a flag the size condition reads"}), "not on the list"),
    ("(b) an allowed flag that a condition reads", lambda d: _win(d)["conditions"][0].update(
        met="{verdict}.interval_straddles_threshold"), "flags of the rule"),
    ("(c) a stated threshold where the file holds its own", _drop_threshold_key, "so the rule reads it"),
    ("(d) the main rule read on the lower end", lambda d: _win(d)["keys"].update(
        upper=_win(d)["keys"]["lo"]), "keys.upper must be keys.hi"),
    ("(d) xfg's upper end taken from elsewhere", lambda d: _exp(d, "xfg-E1")["results"]["keys"].update(
        upper="verdict.upper_95"), "keys.upper must be keys.hi"),
    ("(e) one file for both stages", lambda d: _win(d).update(development=_win(d)["test"]), "are one file"),
    ("(f) a research adoption left unmarked", lambda d: _exp(d, "winprob-E1").pop("adopted_as"),
     "so adopted_as is research_result"),
    ("adopted_as on a test-only experiment", lambda d: _exp(d, "lineups-E1").update(adopted_as="research_result"),
     "not to a protocol or a test-only experiment"),
    ("adopted_as on a protocol", lambda d: _exp(d, "decisions-E1").update(adopted_as="research_result"),
     "not to a protocol or a test-only experiment"),
])
def test_the_builder_refuses_a_weakening_spec_edit(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = R.trace_problems(doc, REPO)
    if not problems:
        with pytest.raises(R.MethodsProblem) as caught:
            R.build_methods(REPO, doc)
        problems = [str(caught.value)]
    assert any(message in line for line in problems), (name, problems[:3])


def test_the_contract_reads_the_main_rule_on_the_upper_end_and_keeps_research_adoption_to_two_stages(source):
    methods = R.build_methods(REPO, source)
    stage = _winprob(methods).test.model_dump()
    with pytest.raises(ValidationError, match="reads the upper end"):
        S.StageResult(**{**stage, "upper": stage["lo"]})
    lineups = _lineups(methods).model_dump()
    with pytest.raises(ValidationError, match="not to a protocol or a test-only"):
        S.Experiment(**{**lineups, "adopted_as": "research_result"})
    decisions = _decisions(methods).model_dump()
    with pytest.raises(ValidationError, match="not to a protocol or a test-only"):
        S.Experiment(**{**decisions, "adopted_as": "research_result"})


def test_winprob_shows_the_control_slopes_and_quotes_the_caveats(source):
    experiment = _winprob(R.build_methods(REPO, source))
    for stage in (experiment.development, experiment.test):
        control = [v for v in stage.values if v.label.startswith("Calibration slope of the control arm")]
        assert len(control) == 6 and all(v.point_estimate for v in control)
    assert any("a little optimistic" in note for note in experiment.writeup_notes)
    assert any("dropped by the frozen builders 16" in note for note in experiment.writeup_notes)


# ------------------------------------------------------------------------------------------ the backlog stays in step


def test_every_backlog_row_is_on_the_methods_page_with_its_rank_and_status(source):
    """A change to docs/research/BACKLOG.md (a new row, a moved rank, a new status) that methods.json does not
    follow fails here, in the fast suite, before any export runs."""
    backlog = R.backlog_rows((REPO / R.BACKLOG).read_text(encoding="utf-8"))
    listed = {e["id"]: e for lane in source["lanes"] for e in lane["experiments"]}
    assert sorted(listed) == sorted(backlog), "every backlog row is an experiment of methods.json, and only those"
    for eid, row in backlog.items():
        e = listed[eid]
        assert (str(e["rank"]), e["what"], str(e["gain"]), str(e["effort"]), e["test_data"], e["backlog_status"]) == (
            row[0], row[3], row[4], row[5], row[7], row[8]), eid
    # a done line in the backlog is never shown as not run
    built = {e.id: e for lane in R.build_methods(REPO, source).lanes for e in lane.experiments}
    for eid, row in backlog.items():
        if row[8].startswith("done"):
            assert built[eid].status != "pre_registered", eid


def test_a_moved_backlog_rank_fails_the_trace(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / R.BACKLOG
    path.write_text(path.read_text(encoding="utf-8").replace("| 8 | xfg-E1 |", "| 9 | xfg-E1 |"), encoding="utf-8")
    assert any("xfg-E1: rank is" in p for p in R.trace_problems(source, repo))


def test_genai_e1_has_run_and_links_its_write_up_while_genai_e2_is_pre_registered(source):
    methods = R.build_methods(REPO, source)
    lane = next(lane for lane in methods.lanes if lane.lane == "genai")
    e1, e2 = lane.experiments
    assert (e1.status, e1.result_in_writeup, e1.writeup) == ("ran_result_in_writeup", True,
                                                             "reports/research/genai-E1.md")
    assert e1.development is None and e1.test is None and e1.protocol is None
    assert e2.status == "pre_registered" and lane.benchmarks == []
    other = e2.model_dump()
    with pytest.raises(ValidationError, match="kept in its write-up"):
        S.Experiment(**{**other, "result_in_writeup": True})
    with pytest.raises(ValidationError, match="kept in its write-up"):
        S.Experiment(**{**e1.model_dump(), "writeup": None})
    doc = copy.deepcopy(source)
    _exp(doc, "genai-E2").update(result_in_writeup=True, writeup="reports/research/genai-E1.md")
    assert any("needs a done line" in p for p in R.trace_problems(doc, REPO))
