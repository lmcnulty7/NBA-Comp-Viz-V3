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


RAN = ("xfg-E1", "decisions-E1", "lineups-E1", "winprob-E1", "genai-E1", "winprob-E2", "genai-E2",
       "embeddings-E1", "lineups-E2", "gravity-E1", "matchups-E1", "matchups-E2", "ghosting-E1", "epv-E1",
       "embeddings-E2", "decisions-E2")   # the experiments with a result file today


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

    def walk(value):          # every repository file the source names, at any depth (parts, values, notes)
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith(("reports/", "docs/")) and (REPO / value).is_file():
            names.add(value)

    walk(source)
    names.add(R.LEDGER)                 # the scoring ledger decides which test stages were scored on the D71 test set
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
    ("an unknown main comparison", lambda d: _lineups_results(d).update(comparison="under"), "is unknown"),
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
        path="{data}.paired.ablation_minus_baseline.bucket_brier"), "another arm"),
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


def test_both_genai_experiments_ran_and_are_read_part_by_part(source):
    methods = R.build_methods(REPO, source)
    lane = next(lane for lane in methods.lanes if lane.lane == "genai")
    e1, e2 = lane.experiments
    assert (e1.status, e1.writeup) == ("decided_by_part", "reports/research/genai-E1.md")
    assert (e2.status, e2.writeup) == ("decided_by_part", "reports/research/genai-E2.md")
    for e in (e1, e2):
        assert e.development is None and e.test is None and e.protocol is None and not e.result_in_writeup
    assert lane.benchmarks == []
    # the shape of a result kept in its write-up is still refused without the write-up
    kept = {**e1.model_dump(), "parts": [], "status": "ran_result_in_writeup", "result_in_writeup": True}
    S.Experiment(**kept)
    with pytest.raises(ValidationError, match="kept in its write-up"):
        S.Experiment(**{**kept, "writeup": None})
    pending = next(e for ln in methods.lanes for e in ln.experiments if e.status == "pre_registered")
    with pytest.raises(ValidationError, match="kept in its write-up"):
        S.Experiment(**{**pending.model_dump(), "result_in_writeup": True})
    # a result still needs a done line in the backlog
    doc = copy.deepcopy(source)
    _exp(doc, "genai-E2").update(backlog_status="proposed")
    assert any("needs a done line" in p for p in R.trace_problems(doc, REPO))


# ------------------------------------------------------------------------------------------ a test left unscored


WP2_DEV = "reports/research/winprob-E2-dev.json"


def _wp2(methods):
    return next(e for lane in methods.lanes for e in lane.experiments if e.id == "winprob-E2")


def test_winprob_e2_is_a_development_result_whose_test_was_not_scored(source):
    experiment = _wp2(R.build_methods(REPO, source))
    data = json.loads((REPO / WP2_DEV).read_text())["pooled_nested_forward"]
    verdict, paired = data["verdict"], data["paired"]["e2_minus_e1"]
    assert (experiment.status, experiment.test, experiment.test_not_scored) == ("not_adopted_development", None, True)
    dev = experiment.development
    assert (dev.diff, dev.lo, dev.hi) == tuple(round(paired["mean_bucket_ece"][k], 4) for k in ("point", "lo", "hi"))
    assert (dev.comparison, dev.threshold, dev.met) == ("below", 0, False) and verdict["adopt"] is False
    (guard,) = dev.conditions
    assert (guard.bound, guard.comparison, guard.threshold, guard.met) == ("hi", "at_most", verdict["brier_guard"],
                                                                          True)
    ledger = json.loads((REPO / R.LEDGER).read_text())
    assert not any(e.get("experiment") == "winprob-E2" for e in ledger)
    assert any("not read for winprob-E2" in note for note in experiment.writeup_notes)


EMB2_DEV = "reports/research/embeddings-E2-dev.json"


def test_embeddings_e2_is_a_development_result_whose_test_was_not_scored(source):
    experiment = next(e for lane in R.build_methods(REPO, source).lanes for e in lane.experiments
                      if e.id == "embeddings-E2")
    data = json.loads((REPO / EMB2_DEV).read_text())
    verdict, paired = data["verdict"], data["split_half"]["rates_with_player_bootstrap"]["nmf_minus_frozen"]
    assert (experiment.status, experiment.test, experiment.test_not_scored) == ("not_adopted_development", None, True)
    dev = experiment.development
    assert (dev.diff, dev.lo, dev.hi) == tuple(round(paired[k], 3) for k in ("value", "lo", "hi"))
    assert (dev.comparison, dev.threshold, dev.met) == ("above", 0, False) and verdict["adopt"] is False
    (size,) = dev.conditions
    assert (size.bound, size.comparison, size.threshold, size.met) == ("point", "at_least", verdict["threshold"], False)
    ledger = json.loads((REPO / R.LEDGER).read_text())
    assert not any(e.get("experiment") == "embeddings-E2" for e in ledger)
    assert any("Set A was not scored" in note for note in experiment.writeup_notes)


def test_a_test_left_unscored_is_refused_when_the_ledger_has_it(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    (repo / R.LEDGER).parent.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((REPO / R.LEDGER).read_text())
    ledger.append({"experiment": "winprob-E2", "status": "scored"})
    (repo / R.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert any("holds an entry for it" in p for p in R.trace_problems(source, repo))


def test_a_test_left_unscored_is_refused_when_the_development_result_passed(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / WP2_DEV
    doc = json.loads(path.read_text(encoding="utf-8"))
    cell = doc["pooled_nested_forward"]["paired"]["e2_minus_e1"]["mean_bucket_ece"]
    cell.update(point=-0.004, lo=-0.006, hi=-0.002)
    doc["pooled_nested_forward"]["verdict"].update(ece_falls=True, adopt=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="would hide a pass"):
        R.build_methods(repo, source)


@pytest.mark.parametrize("name, change, message", [
    ("a test file named beside it", lambda d: _exp(d, "winprob-E2")["results"].update(
        test="reports/research/winprob-E2-final.json"), "no test file"),
    ("no write-up saying so", lambda d: _exp(d, "winprob-E2").update(writeup="reports/research/winprob-E1.md"),
     "does not say the test was not scored"),
    ("a test-only experiment left unscored", lambda d: _exp(d, "lineups-E1").update(test_not_scored=True),
     "no other shape"),
])
def test_a_test_left_unscored_needs_its_shape(source, name, change, message):
    problems = _problems_after(source, change)
    assert any(message in line for line in problems), (name, problems[:3])


def test_the_contract_keeps_an_unscored_test_to_a_missed_development_bar(source):
    methods = R.build_methods(REPO, source)
    wp2 = _wp2(methods).model_dump()
    passed = _winprob(methods).development.model_dump()     # a development stage that met its bar
    with pytest.raises(ValidationError, match="test left unscored"):
        S.Experiment(**{**wp2, "development": passed, "status": "development_met_test_pending"})
    pilot = next(e for lane in methods.lanes for e in lane.experiments if e.id == "xfg-E1").model_dump()
    with pytest.raises(ValidationError, match="test left unscored"):
        S.Experiment(**{**pilot, "test_not_scored": True})




# ------------------------------------------------------------------------------------------ the D116 results


def _built(source):
    return {x.id: x for lane in R.build_methods(REPO, source).lanes for x in lane.experiments}


def _file(name):
    return json.loads((REPO / "reports" / "research" / name).read_text(encoding="utf-8"))


def _part(doc, eid, label_start):
    return next(p for p in _exp(doc, eid)["parts"] if p["label"].startswith(label_start))


def _build_problems(doc, repo=REPO):
    """The trace problems, or else the builder's refusal, of a changed source."""
    problems = R.trace_problems(doc, repo)
    if problems:
        return problems
    try:
        R.build_methods(repo, doc)
    except R.MethodsProblem as exc:
        return [str(exc)]
    return []


def test_embeddings_e1_is_a_test_set_pass_on_its_lower_end_against_two_bars(source):
    e = _built(source)["embeddings-E1"]
    final = _file("embeddings-E1-final.json")["scored"]
    rates, rule = final["rates_with_game_bootstrap"], final["rule"]
    assert (e.status, e.stages, e.development) == ("passed_test", "test_only", None)
    test = e.test
    assert (test.comparison, test.threshold, test.met) == ("above", rule["threshold"], True)
    assert (test.diff, test.lo, test.hi) == tuple(round(rates["pca"][k], 3) for k in ("value", "lo", "hi"))
    assert test.upper == test.lo == round(rule["lower_end"], 3)
    (position,) = test.conditions
    assert (position.bound, position.comparison, position.met) == ("lo", "above", True)
    assert position.threshold == round(rule["position_upper_end"], 3) == round(rates["position"]["hi"], 3)
    assert (test.rows, test.games) == (final["queries"], final["games"])
    assert rule["passes"] is True


def test_gravity_e1_reads_both_parts_of_its_rule_on_development_and_leaves_set_a_unscored(source):
    e = _built(source)["gravity-E1"]
    dev_file = _file("gravity-E1-dev.json")
    cell = dev_file["bootstrap"]["intervals"]["candidate_minus_baseline"]
    assert (e.status, e.test, e.test_not_scored) == ("not_adopted_development", None, True)
    dev = e.development
    assert (dev.diff, dev.lo, dev.hi) == tuple(round(cell[k], 4) for k in ("value", "lo", "hi"))
    assert (dev.comparison, dev.threshold) == ("below", 0)
    assert dev.hi < 0                      # the MSE part holds
    (effect,) = dev.conditions
    assert (effect.bound, effect.comparison, effect.threshold, effect.met) == ("point", "at_least", 0.2, False)
    assert dev.met is False and dev_file["decision"]["adopted"] is False
    assert [s.path for s in dev.also_read] == ["reports/research/gravity-E1-posthoc.json"]
    ledger = json.loads((REPO / R.LEDGER).read_text(encoding="utf-8"))
    assert not [x for x in ledger if x.get("experiment") == "gravity-E1"]


def test_matchups_e1_decides_each_part_on_development_and_reads_set_a_as_a_confirmation(source):
    e = _built(source)["matchups-E1"]
    dev_file, final = _file("matchups-E1-dev.json"), _file("matchups-E1-final.json")["scored"]["parts"]
    assert e.status == "decided_by_part" and e.development is None and e.test is None
    a, b = e.parts
    assert (a.status, b.status) == ("confirmed", "not_adopted_development")
    for part, key in ((a, "a"), (b, "b")):
        cell = dev_file["part_" + key]["defender_minus_no_defender"]
        assert (part.development.diff, part.development.lo, part.development.hi) == (
            round(cell["point"], 6), round(cell["ci95"][0], 6), round(cell["ci95"][1], 6))
        assert part.test.reading == final[key]["reading"]
    assert a.development.met and a.test.met and a.test.reading.startswith("confirms")
    assert not b.development.met and not b.test.met and b.test.reading.startswith("inconclusive")


def test_epv_e1_reads_both_parts_of_its_rule_on_development_and_leaves_set_a_unscored(source):
    """epv-E1 (PLAN.md D117): the paired MSE difference with its interval above 0 fails; the 0 to 4 s slope band (a
    point estimate, no interval in the file) holds. Set A is not scored and no Set A path exists."""
    e = _built(source)["epv-E1"]
    dev_file = _file("epv-E1-dev.json")
    cell = dev_file["overall"]["baseline_minus_e1"]
    assert (e.status, e.test, e.test_not_scored) == ("not_adopted_development", None, True)
    dev = e.development
    assert (dev.diff, dev.lo, dev.hi) == (round(cell["point"], 5), round(cell["ci95"][0], 5), round(cell["ci95"][1], 5))
    assert (dev.comparison, dev.threshold, dev.upper, dev.met) == ("above", 0, dev.lo, False)
    low, high = dev.conditions
    slope = round(dev_file["buckets"]["0-4"]["slope_e1"], 3)
    assert (low.value, low.lo, low.comparison, low.threshold, low.met) == (slope, None, "at_least", 0.9, True)
    assert (high.value, high.comparison, high.threshold, high.met) == (slope, "at_most", 1.1, True)
    assert dev_file["verdict"]["passes"] is False and dev_file["verdict"]["slope_inside"] is True
    assert not (REPO / "reports" / "research" / "epv-E1-final.json").exists()
    ledger = json.loads((REPO / R.LEDGER).read_text(encoding="utf-8"))
    assert not [x for x in ledger if x.get("experiment") == "epv-E1"]


def test_decisions_e2_reads_its_three_checks_on_development_and_leaves_set_a_unscored(source):
    """decisions-E2 (PLAN.md D122): the regret coefficient above 0 passes; the paired difference above 0 and the slope
    inside 0.8 to 1.2 fail. Set A is not scored; the first run's looser filter is shown as report only."""
    e = _built(source)["decisions-E2"]
    dev_file = _file("decisions-E2-dev.json")
    regret = dev_file["metrics"]["new"]["regret_coef"]
    assert (e.status, e.test, e.test_not_scored) == ("not_adopted_development", None, True)
    dev = e.development
    assert (dev.diff, dev.lo, dev.hi) == tuple(round(regret[k], 4) for k in ("point", "lo", "hi"))
    assert (dev.comparison, dev.threshold, dev.met) == ("above", 0, False) and dev.lo > 0
    paired, low, high = dev.conditions
    assert (paired.bound, paired.comparison, paired.met) == ("lo", "above", False)
    assert (low.bound, low.threshold, low.met, high.threshold, high.met) == ("point", 0.8, False, 1.2, True)
    checks = dev_file["decision"]["checks"]
    assert (checks["regret_coef_above_zero"], checks["paired_difference_above_zero"], checks["slope_in_band"]) == (
        True, False, False)
    assert any(v.label.startswith("Report only (first run") for v in dev.values)
    ledger = json.loads((REPO / R.LEDGER).read_text(encoding="utf-8"))
    assert not [x for x in ledger if x.get("experiment") == "decisions-E2"]


def _epv(doc):
    return _exp(doc, "epv-E1")["results"]


def _dec2(doc):
    return _exp(doc, "decisions-E2")["results"]


@pytest.mark.parametrize("name, change, message", [
    ("epv: the slope band dropped", lambda d: _epv(d).update(conditions=[], conjunctions=[]),
     "is read by no part"),
    ("epv: one end of the band dropped", lambda d: (_epv(d)["conditions"].pop(), _epv(d)["conjunctions"][0].update(
        of=["@0"])), "'at most 1.1' is read by no part"),
    ("epv: the band widened", lambda d: _epv(d)["conditions"][1].update(threshold=1.3), "names no threshold"),
    ("epv: the MSE bar read below", lambda d: _epv(d).update(comparison="below"), "does not state below 0"),
    ("epv: a rule quote not in the dossier", lambda d: _epv(d)["rule_quote"]["texts"].__setitem__(
        0, "Development decides: the 0 to 4 s slope inside 0.9 to 1.1."), "not whole sentences"),
    ("epv: Set A said scored", lambda d: _exp(d, "epv-E1").pop("test_not_scored"), ""),
    ("decisions-E2: the paired check dropped", lambda d: _dec2(d)["conditions"].pop(0), ""),
    ("decisions-E2: the slope band dropped", lambda d: (_dec2(d)["conditions"].__delitem__(slice(1, 3)),
                                                        _dec2(d).update(conjunctions=[])),
     "is read by no part"),
    ("decisions-E2: the slope read as passing", lambda d: _dec2(d)["conditions"][1].update(threshold=0.3),
     "names no threshold"),
    ("decisions-E2: the main flag read as the paired flag", lambda d: _dec2(d)["keys"].update(
        main_met="decision.checks.paired_difference_above_zero"), ""),
])
def test_epv_e1_and_decisions_e2_are_refused_when_the_source_misreads_them(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert problems, name
    if message:
        assert any(message in line for line in problems), (name, problems[:3])


@pytest.mark.parametrize("name, file, change, message", [
    ("epv said to pass", "epv-E1-dev.json", lambda d: d["verdict"].update(passes=True), "rule read on its own numbers"),
    ("epv slope flag against its numbers", "epv-E1-dev.json", lambda d: d["verdict"].update(slope_inside=False),
     "disagrees with the parts it joins"),
    ("epv slope moved out of the band", "epv-E1-dev.json", lambda d: d["buckets"]["0-4"].update(slope_e1=0.85),
     "states them as one number"),
    ("decisions-E2 said to adopt", "decisions-E2-dev.json", lambda d: d["decision"].update(adopt_on_development=True),
     "rule read on its own numbers"),
    ("decisions-E2 slope said in band", "decisions-E2-dev.json",
     lambda d: d["decision"]["checks"].update(slope_in_band=True), "disagrees with the parts it joins"),
    ("decisions-E2 paired flag against its numbers", "decisions-E2-dev.json",
     lambda d: d["decision"]["checks"].update(paired_difference_above_zero=True), "disagrees with its lo"),
])
def test_an_epv_e1_or_decisions_e2_file_that_contradicts_its_numbers_is_refused(source, tmp_path, name, file, change,
                                                                                  message):
    repo = _mutated_repo(tmp_path, source, file, change)
    with pytest.raises(R.MethodsProblem) as caught:
        R.build_methods(repo, source)
    assert message in str(caught.value), (name, str(caught.value))


def test_a_set_a_section_that_does_not_open_with_not_scored_is_refused(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports" / "research" / "decisions-E2.md"
    path.write_text(path.read_text(encoding="utf-8").replace("Not scored (PLAN.md D122)", "Scored later (PLAN.md D122)"),
                    encoding="utf-8")
    assert any("decisions-E2: the verdict paragraph" in p for p in R.trace_problems(source, repo))


def test_matchups_e2_is_a_development_fail_whose_set_a_was_not_scored(source):
    e = _built(source)["matchups-E2"]
    cell = _file("matchups-E2-dev.json")["hmm_minus_hungarian"]
    assert (e.status, e.test, e.test_not_scored) == ("not_adopted_development", None, True)
    dev = e.development
    assert (dev.diff, dev.lo, dev.hi) == (round(cell["point"], 6), round(cell["ci95"][0], 6), round(cell["ci95"][1], 6))
    assert (dev.comparison, dev.threshold, dev.upper, dev.met) == ("above", 0, dev.lo, False)


def test_lineups_e2_fails_on_the_test_season_against_rapm_synergy_with_both_baselines_read(source):
    e = _built(source)["lineups-E2"]
    results = _file("lineups-E2-final.json")["scored"]["results"]
    assert (e.status, e.stages) == ("not_adopted_test", "test_only")
    test = e.test
    paired = results["paired_lrapm_minus_rapm_synergy"]
    assert (test.lo, test.hi) == (round(paired["ci95"][0], 4), round(paired["ci95"][1], 4))
    assert test.hi > 0 and test.met is False and results["verdict"]["passes"] is False
    d_synergy, raw_diff, d_raw = test.conditions
    assert (d_synergy.met, raw_diff.met, d_raw.met) == (False, True, True)
    assert [g.met for g in test.guards] == [True, True]
    assert [len(g.buckets) for g in test.guards] == [3, 3]


def test_genai_e1_reads_both_hypotheses_on_the_test_players(source):
    e = _built(source)["genai-E1"]
    diffs = _file("genai-E1-final.json")["scouting"]["bootstrap_by_player"]["differences"]
    h1, h2 = e.parts
    assert (h1.status, h2.status) == ("passed_test", "passed_test")
    lo, hi = diffs["haiku minus template: pass_rate"]
    assert (h1.test.lo, h1.test.hi) == (round(-hi, 3), round(-lo, 3))       # template minus Haiku, turned
    assert (h1.test.comparison, h1.test.threshold, h1.test.reading) == ("at_least", -0.05, "holds")
    assert (h2.test.lo, h2.test.comparison, h2.test.threshold) == (round(diffs["haiku minus template: frame_variety"][0],
                                                                         3), "above", 0)


def test_genai_e2_reads_every_hypothesis_at_both_sizes_as_the_file_reads_it(source):
    e = _built(source)["genai-E2"]
    assert len(e.parts) == 14
    for part in e.parts:
        size, hypothesis = part.label.split()[:2]
        name = next(k for k in _file(f"genai-E2-{size}-final.json")["hypotheses"]
                    if k.startswith(hypothesis) and (k.split()[1].rstrip(":") in part.label)
                    and ("haiku_cache" in k or not k.startswith("H3")))
        cell = _file(f"genai-E2-{size}-final.json")["hypotheses"][name]
        assert part.test.reading == cell["reading"], part.label
        assert part.status == ("passed_test" if cell["reading"] in ("passes", "non-inferior") else "not_passed_test")
    statuses = {p.label: p.status for p in e.parts}
    assert statuses["1.5B H3 scouting: near the API model"] == "passed_test"
    assert statuses["0.5B H3 scouting: near the API model"] == "not_passed_test"


def _mutated_repo(tmp_path, source, name, change):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports" / "research" / name
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return repo


@pytest.mark.parametrize("name, change, message", [
    ("embeddings: the second bar not in the rule", lambda d: _exp(d, "embeddings-E1")["results"]["conditions"][0]
     .update(states="above the lower end of the raw retriever's interval"), "the rule does not say"),
    ("embeddings: a higher bar than the rule states", lambda d: (
        _exp(d, "embeddings-E1")["results"]["keys"].pop("threshold"),
        _exp(d, "embeddings-E1")["results"].update(threshold=0.25)), "does not state above 0.25"),
    ("embeddings: the second bar read from the wrong end", lambda d: _exp(d, "embeddings-E1")["results"]
     ["equalities"][1].__setitem__(1, "scored.rates_with_game_bootstrap.position.lo"), "states them as one number"),
    ("gravity: the effect bar retyped", lambda d: (
        _exp(d, "gravity-E1")["results"]["conditions"][0].pop("threshold_key"),
        _exp(d, "gravity-E1")["results"]["conditions"][0].update(threshold=0.01)), "does not state at least 0.01"),
    ("gravity: the effect part dropped", lambda d: _exp(d, "gravity-E1")["results"].update(conditions=[]),
     "is read by no part"),
    ("gravity: the main flag read as the effect flag", lambda d: _exp(d, "gravity-E1")["results"]["keys"].update(
        main_met="decision.effect_ok"), "belong to no rule"),
    ("gravity: Set A said scored", lambda d: _exp(d, "gravity-E1").pop("test_not_scored"), ""),
    ("matchups-E1: inconclusive read as a pass", lambda d: _part(d, "matchups-E1", "Part (b)")["results"]["readings"]
     .update(inconclusive="met"), "disagrees with the rule"),
    ("matchups-E1: a part left unscored with its test file", lambda d: _part(d, "matchups-E1", "Part (b)").update(
        test_not_scored=True), "a part left unscored"),
    ("matchups-E1: two parts with one label", lambda d: _part(d, "matchups-E1", "Part (b)").update(
        label=_part(d, "matchups-E1", "Part (a)")["label"]), "every part has its own label"),
    ("matchups-E2: a bar below zero", lambda d: _exp(d, "matchups-E2")["results"].update(comparison="below"),
     "does not state below 0"),
    ("lineups-E2: a bin guard dropped", lambda d: _exp(d, "lineups-E2")["results"]["guards"].pop(),
     "is said to join"),
    ("lineups-E2: the ablation flag unnamed", lambda d: _exp(d, "lineups-E2")["results"].pop("flags_outside_rule"),
     "belong to no rule"),
    ("lineups-E2: the d bar loosened", lambda d: _exp(d, "lineups-E2")["results"]["conditions"][0].update(
        threshold=-0.0001), "does not state at most -0.0001"),
    ("genai-E1: H1 read unturned", lambda d: _part(d, "genai-E1", "H1")["results"].update(flip=False),
     "disagrees with the rule"),
    ("genai-E1: a reading quote not in the write-up", lambda d: _part(d, "genai-E1", "H2")["results"]
     ["reading_quote"].update(text="H2 holds."), "not whole sentences"),
    ("genai-E2: the deciding arm unmarked", lambda d: _part(d, "genai-E2", "0.5B H3")["results"].pop("true_flags"),
     "belong to no rule"),
    ("genai-E2: a non-inferiority bar of zero", lambda d: _part(d, "genai-E2", "1.5B H3")["results"].update(
        threshold=0), "does not state at least 0.0"),
])
def test_the_d116_results_are_refused_when_the_source_misreads_them(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    if message:
        assert any(message in line for line in problems), (name, problems[:3])
    else:
        assert problems, name


@pytest.mark.parametrize("name, file, change, message", [
    ("matchups-E1 part (b) on Set A said to confirm", "matchups-E1-final.json",
     lambda d: d["scored"]["parts"]["b"].update(reading="confirms (interval wholly below zero)"),
     "disagrees with the rule"),
    ("matchups-E1 part (a) flag against its numbers", "matchups-E1-dev.json",
     lambda d: d["verdict"].update(part_a_passes=False), "rule read on its own numbers"),
    ("matchups-E2 said to pass", "matchups-E2-dev.json", lambda d: d["verdict"].update(passes=True),
     "rule read on its own numbers"),
    ("gravity-E1 effect flag against its numbers", "gravity-E1-dev.json",
     lambda d: d["decision"].update(effect_ok=True), "disagrees with its point"),
    ("lineups-E2 a bin listed as worse", "lineups-E2-final.json",
     lambda d: d["scored"]["results"]["verdict"]["against_rapm_synergy"].update(bins_worse_beyond_interval=["over_100"]),
     "but its numbers give"),
    ("lineups-E2 holds against rapm_synergy", "lineups-E2-final.json",
     lambda d: d["scored"]["results"]["verdict"]["against_rapm_synergy"].update(holds=True),
     "disagrees with the parts it joins"),
    ("embeddings-E1 a pass flag against its numbers", "embeddings-E1-final.json",
     lambda d: d["scored"]["rates_with_game_bootstrap"]["pca"].update(lo=0.02), "states them as one number"),
    ("genai-E2 inconclusive where it hurts", "genai-E2-1.5B-final.json",
     lambda d: d["hypotheses"]["H2 scouting: a2-r1-epoch1 minus a1-epoch1"].update(reading="inconclusive"),
     "no other reading fits"),
    ("genai-E2 H3 passes where it is inferior", "genai-E2-0.5B-final.json",
     lambda d: d["hypotheses"]["H3 scouting: a1-epoch1+repair minus haiku_cache"].update(reading="non-inferior"),
     "disagrees with the rule"),
    ("genai-E2 H1 passes under its size bar", "genai-E2-0.5B-final.json",
     lambda d: d["hypotheses"]["H1 possession: a1-epoch1 minus a0"].update(point=0.05, interval=[0.01, 0.2]),
     "disagrees with the rule"),
    ("genai-E2 the deciding flag moved", "genai-E2-0.5B-final.json",
     lambda d: d["hypotheses"]["H3 scouting: a1-epoch1+repair minus haiku_cache"].update(decides=False),
     "not the comparison that decides"),
])
def test_a_d116_result_file_that_contradicts_its_numbers_is_refused(source, tmp_path, name, file, change, message):
    repo = _mutated_repo(tmp_path, source, file, change)
    with pytest.raises(R.MethodsProblem) as caught:
        R.build_methods(repo, source)
    assert message in str(caught.value), (name, str(caught.value))


def test_the_contract_keeps_a_part_to_the_status_its_stages_give(source):
    e = _built(source)["matchups-E1"]
    a, b = (p.model_dump() for p in e.parts)
    with pytest.raises(ValidationError, match="the results say 'confirmed'"):
        S.ExperimentPart(**{**a, "status": "development_met_test_not_met"})
    with pytest.raises(ValidationError, match="the results say 'not_adopted_development'"):
        S.ExperimentPart(**{**b, "status": "confirmed"})
    with pytest.raises(ValidationError, match="a missed development bar"):
        S.ExperimentPart(**{**a, "test": None, "test_not_scored": True, "status": "development_met_test_pending"})
    with pytest.raises(ValidationError, match="no development result"):
        S.ExperimentPart(**{**a, "stages": "test_only", "status": "passed_test"})
    whole = e.model_dump()
    with pytest.raises(ValidationError, match="decided part by part"):
        S.Experiment(**{**whole, "development": a["development"]})
    with pytest.raises(ValidationError, match="decided part by part"):
        S.Experiment(**{**whole, "status": "passed_test"})
    with pytest.raises(ValidationError, match="its own label"):
        S.Experiment(**{**whole, "parts": [a, {**b, "label": a["label"]}]})
    genai = _built(source)["genai-E1"].parts[0].test.model_dump()
    assert genai["diff"] is None and genai["games"] is None
    with pytest.raises(ValidationError, match="the lower end for at least and above"):
        S.StageResult(**{**genai, "upper": genai["hi"]})


@pytest.mark.parametrize("eid, extra", [
    ("embeddings-E1", {}),                                     # test only
    ("decisions-E1", {}),                                      # a protocol
    ("lineups-E2", {}),                                        # test only, failed
    ("matchups-E1", {}),                                       # decided part by part
    ("genai-E2", {"parts": [], "status": "ran_result_in_writeup", "result_in_writeup": True}),   # kept in a write-up
])
def test_the_contract_refuses_a_test_left_unscored_on_any_other_shape(source, eid, extra):
    """A test left unscored belongs only to a development result that missed its bar; the check runs before any
    shape's early return (review finding on export/schemas.py)."""
    whole = {**_built(source)[eid].model_dump(), **extra}
    S.Experiment(**whole)                                      # valid as it is
    with pytest.raises(ValidationError, match="a test left unscored"):
        S.Experiment(**{**whole, "test_not_scored": True})


@pytest.mark.parametrize("sentence, ok", [
    ("The test season was not scored.", True),
    ("Set A was not scored (PLAN.md D109).", True),
    ("**Verdict: not adopted, on development; Set A not scored.**", True),
    ("The ablation was not scored on its own.", False),
    ("Not scored: nothing.", False),
])
def test_the_not_scored_sentence_names_the_test_set(sentence, ok):
    assert bool(R.NOT_SCORED.search(sentence)) is ok


# ------------------------------------------------------------------------------------------ ghosting-E1 (D113, D115, D119)


def _ghost(doc):
    return _exp(doc, "ghosting-E1")["results"]


def test_ghosting_e1_met_its_three_parts_on_development_and_set_a_read_inconsistent(source):
    e = _built(source)["ghosting-E1"]
    dev_file, scored = _file("ghosting-E1-dev.json"), _file("ghosting-E1-final.json")["scored"]
    # adopted on development, stopped by Set A's D113 reading: the status never says adopted
    assert e.status == "not_adopted_test" and e.adopted_as is None
    dev, test = e.development, e.test
    cell = dev_file["paired_ft"]["1.0"]["rollout_minus_mlp"]
    assert (dev.diff, dev.lo, dev.hi) == tuple(round(cell[k], 3) for k in ("diff", "lo", "hi"))
    assert (dev.comparison, dev.threshold, dev.met) == ("below", 0, True) and dev_file["decision"]["adopted"] is True
    size, guard, signs, p = dev.conditions
    assert (size.bound, size.comparison, size.threshold, size.met) == ("point", "at_most", -0.1, True)
    assert (guard.bound, guard.comparison, guard.threshold, guard.met) == ("hi", "below", 0.02, True)
    assert (signs.value, signs.threshold, signs.lo, p.lo) == (414, 190, None, None)
    assert p.value > 0 and p.met                     # a tiny p value is rounded to its first digit, never to zero
    # Set A: the D113 consistency check, the range must hold the development estimate
    one = scored["1.0"]["rollout_minus_mlp"]
    assert (test.lo, test.hi) == (round(one["lo"], 4), round(one["hi"], 4))
    assert (test.comparison, test.threshold) == ("contains", round(scored["development_estimate_1s"], 4))
    assert test.threshold < test.lo and test.upper == test.lo and test.met is False
    assert test.reading == scored["reading"] == "inconsistent"
    report_only = [v for v in test.values if v.label.startswith("Report only (D115)")]
    assert len(report_only) == 1 and report_only[0].value == round(scored["report_only_size_reading"]["upper_end"], 4)
    ledger = json.loads((REPO / R.LEDGER).read_text(encoding="utf-8"))
    assert [x["status"] for x in ledger if x.get("experiment") == "ghosting-E1"] == ["scored"]


def _ghost_test(doc):
    return _ghost(doc)["by_stage"]["test"]


@pytest.mark.parametrize("name, change, message", [
    ("the size part dropped", lambda d: _ghost(d)["conjunctions"][0].update(of=["@main"]), "another number of parts"),
    ("the size bar retyped", lambda d: _ghost(d)["conditions"][0].update(threshold=-0.01,
                                                                          states="1.0 s error lower by at least 0.01"),
     ""),
    ("the sign test read against a made-up bar", lambda d: _ghost(d)["conditions"][3].update(threshold=0.5),
     "names no threshold"),
    ("the guard bar loosened", lambda d: _ghost(d)["conditions"][1].update(threshold=0.2), "does not state below 0.2"),
    ("the Set A check reads another number", lambda d: _ghost_test(d)["keys"].update(threshold="scored.0.2.rows"),
     "the development estimate the range must hold"),
    ("the Set A check without the development file's number", lambda d: _ghost_test(d)["keys"].pop(
        "threshold_development"), "keys.threshold_development"),
    ("the Set A check's words not in the rule", lambda d: _ghost_test(d).update(main_states="lies near zero"),
     "the rule does not say"),
    ("inconsistent read as a pass", lambda d: _ghost_test(d)["readings"].update(inconsistent="met"),
     "disagrees with the rule"),
    ("the size reading left unnamed", lambda d: _ghost_test(d)["flags_outside_rule"].pop(
        "scored.report_only_size_reading.below"), "belong to no rule"),
    ("a stage override that names a file", lambda d: _ghost(d)["by_stage"]["test"].update(
        test="reports/research/ghosting-E1-dev.json"), "are one file"),
])
def test_ghosting_e1_is_refused_when_the_source_misreads_it(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert problems, name
    if message:
        assert any(message in line for line in problems), (name, problems[:3])


@pytest.mark.parametrize("name, file, change, message", [
    ("Set A said consistent", "ghosting-E1-final.json", lambda d: d["scored"].update(reading="consistent"),
     "no other reading fits"),
    ("Set A said confirms (D113 reads inconsistent first)", "ghosting-E1-final.json",
     lambda d: d["scored"].update(reading="confirms"), "disagrees with the rule"),
    ("the development estimate moved", "ghosting-E1-final.json",
     lambda d: d["scored"].update(development_estimate_1s=-1.2), "the development estimate the range must hold"),
    ("the sign test flag against its numbers", "ghosting-E1-dev.json",
     lambda d: d["sign_test"]["rollout"].update(p_value=0.2), "states them as one number"),
    ("the guard flag against its numbers", "ghosting-E1-dev.json",
     lambda d: d["decision"]["guard_02s"].update(ok=False), "disagrees with its hi"),
])
def test_a_ghosting_e1_file_that_contradicts_its_numbers_is_refused(source, tmp_path, name, file, change, message):
    repo = _mutated_repo(tmp_path, source, file, change)
    with pytest.raises(R.MethodsProblem) as caught:
        R.build_methods(repo, source)
    assert message in str(caught.value), (name, str(caught.value))


def test_the_contract_reads_a_contains_rule_and_a_point_only_condition(source):
    e = _built(source)["ghosting-E1"]
    test = e.test.model_dump()
    with pytest.raises(ValidationError, match="the end nearer its number"):
        S.StageResult(**{**test, "upper": test["hi"]})
    with pytest.raises(ValidationError, match="met says whether"):
        S.StageResult(**{**test, "met": True})
    signs = e.development.conditions[2].model_dump()
    with pytest.raises(ValidationError, match="both ends or neither"):
        S.Condition(**{**signs, "lo": 100})
    with pytest.raises(ValidationError, match="both ends or neither"):
        S.Condition(**{**signs, "bound": "lo"})


# ------------------------------------------------------------------------------------------ the ml-review probes (M1, M2)


def _genai_part(doc, eid, label):
    return next(p for p in _exp(doc, eid)["parts"] if p["label"] == label)


def _probe_h4_as_non_inferiority(d):
    r = _genai_part(d, "genai-E2", "0.5B H4 scouting: constrained decoding, pass rate")["results"]
    r.update(comparison="at_least", threshold=-0.05)
    r["readings"]["inconclusive"] = "met"


def _probe_h2_as_non_inferiority(d):
    r = _genai_part(d, "genai-E2", "0.5B H2 possession: rejection sampling")["results"]
    r.update(comparison="at_least", threshold=-0.05)
    r["readings"]["inconclusive"] = "met"


def _drop_part(eid, label):
    return lambda d: _exp(d, eid).update(parts=[p for p in _exp(d, eid)["parts"] if p["label"] != label])


def _hide_development(d):
    r = _part(d, "matchups-E1", "Part (b)")
    r.update(stages="test_only")
    r["results"].update(development=None)
    r["results"]["stage_keys"].pop("development")


@pytest.mark.parametrize("name, change, message", [
    ("(a) H4 read as non-inferior with inconclusive as a pass", _probe_h4_as_non_inferiority, "does not state at "
     "least -0.05"),
    ("(a) H2 possession read the same way", _probe_h2_as_non_inferiority, "does not state at least -0.05"),
    ("(a) inconclusive alone read as a pass", lambda d: _genai_part(d, "genai-E2", "0.5B H2 possession: rejection "
     "sampling")["results"]["readings"].update(inconclusive="met"), "does not state the lo above 0"),
    ("(a) 1.5B H3 bar moved to above 0", lambda d: _genai_part(d, "genai-E2", "1.5B H3 scouting: near the API model")
     ["results"].update(comparison="above", threshold=0), "does not state above 0.0"),
    ("(a) H4 given H3's rule", lambda d: _genai_part(d, "genai-E2", "0.5B H4 scouting: constrained decoding, pass rate")
     ["results"]["rule_quote"].update(texts=[
         "H3, near the API model (scouting only): the dev-best arm with the repair loop minus Haiku as delivered. "
         "Lower bound at or above -0.05: non-inferior; upper bound below -0.05: inferior; otherwise inconclusive."]),
     "holds its own sentence"),
    ("(a) a rule quote not in the dossier", lambda d: _genai_part(d, "genai-E2", "1.5B H1 scouting: training helps")
     ["results"]["rule_quote"]["texts"].__setitem__(0, "H1, training helps: passes when above 0."), "not whole "
     "sentences"),
    ("(b) 1.5B H2 scouting dropped", _drop_part("genai-E2", "1.5B H2 scouting: rejection sampling"), "no part reads"),
    ("(b) genai-E1 H2 dropped", _drop_part("genai-E1", "H2: the model writes with more variety"),
     "the dossier names"),
    ("(c) embeddings' position-only bar dropped", lambda d: _exp(d, "embeddings-E1")["results"].update(conditions=[]),
     "is read by no part"),
    ("(c) genai-E2 H1's +0.10 bar dropped", lambda d: _genai_part(d, "genai-E2", "1.5B H1 scouting: training helps")
     ["results"].update(conditions=[]), "is read by no part"),
    ("(d) matchups-E1 part (b) read test only", _hide_development, "record a development verdict"),
    ("M1: a genai test stage without its own test set", lambda d: _genai_part(
        d, "genai-E1", "H1: the checker does not separate a template from the model")["results"].pop("test_set"),
     "holds no entry for genai-E1"),
    ("M1: a genai test stage claiming D71", lambda d: _genai_part(
        d, "genai-E1", "H1: the checker does not separate a template from the model")["results"]["test_set"].update(
        what="experiment result: scored once on the D71 test set"), "names neither D71 nor test games"),
    ("M1: a genai heading saying test games", lambda d: _genai_part(
        d, "genai-E2", "0.5B H1 scouting: training helps")["results"]["test_set"].update(
        heading="Test games, scored once"), "names neither D71 nor test games"),
])
def test_the_ml_review_probes_are_refused(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert any(message in line for line in problems), (name, problems[:3])


def test_a_value_reading_against_its_numbers_is_refused(source, tmp_path):
    key = "H4 scouting: a3 minus a1-epoch1"
    repo = _mutated_repo(tmp_path, source, "genai-E2-0.5B-final.json",
                         lambda d: d["hypotheses"][key].update(number_value_reading="fewer wrong numbers"))
    with pytest.raises(R.MethodsProblem, match="its numbers give 'inconclusive'"):
        R.build_methods(repo, source)


def test_genai_stages_name_their_own_test_set_and_h4_shows_its_failure_rate_reading(source):
    built = _built(source)
    for eid in ("genai-E1", "genai-E2"):
        for part in built[eid].parts:
            assert "D71" not in part.test.source.what and "genai-splits.json" in part.test.source.what
            assert part.test.test_set and "test games" not in part.test.test_set.lower()
    # a test stage the scoring ledger holds says what was scored in the ledger's own words when it has them
    ledger = {e["experiment"]: e for e in json.loads((REPO / R.LEDGER).read_text(encoding="utf-8"))}
    for eid in ("embeddings-E1", "lineups-E1", "lineups-E2", "winprob-E1", "matchups-E1", "ghosting-E1", "xfg-E1"):
        stages = [built[eid].test] if built[eid].test else [p.test for p in built[eid].parts if p.test]
        stated = ledger[eid].get("summary", {}).get("stage")
        want = (f"experiment result: {stated}" if stated else
                "experiment result: fitted on every development game, scored once on the D71 test set")
        assert stages and all(s.source.what == want and s.test_set is None for s in stages), eid
    h4 = [p for p in built["genai-E2"].parts if " H4 " in p.label]
    assert len(h4) == 4 and all(p.test.values[0].reading == "inconclusive" for p in h4)
    h2 = next(p for p in built["genai-E1"].parts if p.label.startswith("H2"))
    assert any("ranges of plausible values" in note for note in h2.notes)
    with pytest.raises(ValidationError, match="heads a test stage"):
        S.StageResult(**{**built["ghosting-E1"].development.model_dump(), "test_set": "Training games"})


def test_a_tiny_condition_value_keeps_its_first_digit(source):
    p = _built(source)["ghosting-E1"].development.conditions[3]
    assert p.value != 0 and f"{p.value:.0e}" == "2e-20"


# ------------------------------------------------------------------------------------------ the re-review probes


def _drop_size(size):
    return lambda d: _exp(d, "genai-E2").update(parts=[p for p in _exp(d, "genai-E2")["parts"]
                                                       if not p["label"].startswith(size)])


def _h4_values(d, change):
    for part in _exp(d, "genai-E2")["parts"]:
        if " H4 " in part["label"]:
            change(part["results"])


@pytest.mark.parametrize("name, change, message", [
    ("1. every 1.5B part dropped", _drop_size("1.5B"), "no part reads ['reports/research/genai-E2-1.5B-final.json']"),
    ("4. the H4 failure-rate rule cut to otherwise", lambda d: _h4_values(d, lambda r: r["values"][0].update(
        reading_rule={"inconclusive": "otherwise"})), "the quoted rule reads"),
    ("4. the H4 failure-rate bar moved", lambda d: _h4_values(d, lambda r: r["values"][0]["reading_rule"][
        "fewer wrong numbers"].update(threshold=-0.01)), "does not state its hi end below -0.01"),
    ("4. the H4 value dropped", lambda d: _h4_values(d, lambda r: r.update(values=[])),
     "read on its number failure rate too"),
    ("4. the H4 failure-rate rule not from the dossier", lambda d: _h4_values(d, lambda r: r["values"][0][
        "reading_quote"]["texts"].__setitem__(0, 'It reads "fewer wrong numbers" when its whole interval is below 0, '
                                                 '"more wrong numbers" when the whole interval is above 0, and '
                                                 '"inconclusive" otherwise.')), "not whole sentences"),
])
def test_the_re_review_probes_are_refused(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert any(message in line for line in problems), (name, problems[:3])


def test_a_note_cannot_start_after_a_hyphen_inside_a_sentence():
    # ghosting-E1.md: "... 7.6843 - 7.6810). The check was there ..." is not an inline list item
    paragraph = "the range (computed: 7.6843 - 7.6810). The check was there to catch a broken refit."
    assert R.is_sentences("The check was there to catch a broken refit.", [paragraph])
    assert not R.is_sentences("7.6810). The check was there to catch a broken refit.", [paragraph])
    assert R.is_sentences("H2, the checker.", ["Rule one holds. - H2, the checker."])
    assert R.is_sentences("H2, the checker.", ["Rules, fixed now: - H2, the checker."])
    assert not R.is_sentences("H2, the checker.", ["a minus - H2, the checker."])


# ------------------------------------------------------------------------------------------ methods-results2 review probes


FIRST_RUN = "report_only_first_run_filter."


def _first_run(spec, keys=("value", "lo", "hi")):
    for k in keys:
        spec[k] = FIRST_RUN + spec[k]


def _first_run_extra(d):
    extra = copy.deepcopy(_dec2(d)["conditions"][0])
    extra.update(label="first run paired", met=None)
    _first_run(extra)
    _dec2(d)["conditions"].append(extra)


def _swap(spec, old, new, keys=("value", "lo", "hi")):
    for k in keys:
        if k in spec:
            spec[k] = spec[k].replace(old, new)


@pytest.mark.parametrize("name, change, message", [
    # 1. a report-only reading never decides
    ("the main rule on the first run", lambda d: _first_run(_dec2(d)["keys"], ("diff", "lo", "hi")),
     "a report-only reading"),
    ("the paired condition on the first run", lambda d: _first_run(_dec2(d)["conditions"][0]), "a report-only reading"),
    ("the slope on the first run", lambda d: [_first_run(c) for c in _dec2(d)["conditions"][1:]],
     "a report-only reading"),
    ("an extra condition on the first run", _first_run_extra, "a report-only reading"),
    ("the rows of the first run", lambda d: _dec2(d)["keys"].update(rows=FIRST_RUN + "rows"), "a report-only reading"),
    ("a report-only value without its label", lambda d: _dec2(d)["values"][3].update(
        label="Paired difference, new minus frozen"), "its label does not say so"),
    # 2. a rule part reads the statistic it names, and no extra part rides on a bar already read
    ("epv slope read on the baseline", lambda d: [c.update(value="buckets.0-4.slope_baseline")
                                                  for c in _epv(d)["conditions"]], "another arm"),
    ("epv slope read on the threes-only ablation", lambda d: [c.update(value="buckets.0-4.slope_three_only")
                                                              for c in _epv(d)["conditions"]], "another arm"),
    ("decisions-E2 main read on the frozen arm", lambda d: _swap(_dec2(d)["keys"], "metrics.new.", "metrics.frozen.",
                                                                  ("diff", "lo", "hi")), "another arm"),
    ("decisions-E2 slope read on the frozen arm", lambda d: [_swap(c, "metrics.new.", "metrics.frozen.")
                                                             for c in _dec2(d)["conditions"][1:]], "another arm"),
    ("decisions-E2 main read on the ablation", lambda d: _swap(_dec2(d)["keys"], "metrics.new.regret_coef",
                                                                "metrics.ablation_delta_epv_coef", ("diff", "lo", "hi")),
     "another arm"),
    ("an extra condition on the kind-share R squared", lambda d: _dec2(d)["conditions"].append(
        {"label": "x", "value": "secondary_report_only.new.r2_on_kind_shares", "bound": "point",
         "comparison": "above", "threshold": 0, "met": None, "digits": 4}), "a report-only reading"),
    ("an extra condition riding on the rule's own bar", lambda d: _dec2(d)["conditions"].append(
        {"label": "x", "value": "metrics.new.slope.point", "lo": "metrics.new.slope.lo", "hi": "metrics.new.slope.hi",
         "bound": "lo", "comparison": "above", "threshold": 0, "met": None, "digits": 4}),
     "the rule does not state for it"),
    ("an extra condition on epv's ablation", lambda d: _epv(d)["conditions"].append(
        {"label": "x", "value": "overall.baseline_minus_three_only.point", "lo": "overall.baseline_minus_three_only.ci95.0",
         "hi": "overall.baseline_minus_three_only.ci95.1", "bound": "lo", "comparison": "above", "threshold": 0,
         "met": None, "digits": 5}), "another arm"),
    ("epv slope on a value its verdict was not decided on", lambda d: [c.update(value="buckets.4-10.slope_e1")
                                                                        for c in _epv(d)["conditions"]],
     "was decided on"),
    # 4. a bar said of an interval is read on the interval's end, and a bar on the value on its point
    ("decisions-E2 paired difference read on its point", lambda d: _dec2(d)["conditions"][0].update(bound="point"),
     "the rule does not state for it"),
    ("decisions-E2 slope read on its lower end", lambda d: _dec2(d)["conditions"][1].update(bound="lo"),
     "the rule does not state for it"),
    # 5. a dossier rule quote comes from the experiment's own section
    ("epv's rule quoted from epv-E2's section", lambda d: _epv(d)["rule_quote"].update(
        texts=[_e2_threshold_sentence()]), "not whole sentences"),
    # the shapes around it
    ("an unknown experiment key", lambda d: _exp(d, "epv-E1").update(test_only=True), "are no keys of an experiment"),
    ("a result back in the write-up with its files on disk", lambda d: _exp(d, "epv-E1").update(
        results=None, result_in_writeup=True, test_not_scored=False), "read through them"),
    ("a repeat the rule does not state", lambda d: _exp(d, "decisions-E2")["results"].update(
        repeats={"times": 2, "states": "against raw and against rapm_synergy each"}), "so it is stated once"),
])
def test_the_methods_results2_probes_are_refused(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert any(message in line for line in problems), (name, problems[:3])


def _e2_threshold_sentence():
    bullets = R.experiment_bullets((REPO / "docs/research/epv.md").read_text(encoding="utf-8"))["epv-E2"]["_all"]
    line = next(b for b in bullets if b.startswith("Adoption threshold"))
    return line.split(". ")[0].rstrip(".") + "."


def test_the_first_run_is_a_labelled_value_and_not_a_note(source):
    e = _built(source)["decisions-E2"]
    assert not any("first e2-dev run" in note for note in e.writeup_notes)
    first = [v for v in e.development.values if "first run" in v.label]
    assert len(first) == 1 and first[0].label.startswith("Report only")


# ------------------------------------------------------------------------------------------ methods-harden probes


def _set_main(results, stat, ends=("ci95.0", "ci95.1"), value="point"):
    results["keys"].update(diff=f"{stat}.{value}", lo=f"{stat}.{ends[0]}", hi=f"{stat}.{ends[1]}")


_ABLATION_CONDITION = {"label": "abl", "value": "overall.baseline_minus_three_only.point",
                       "lo": "overall.baseline_minus_three_only.ci95.0", "hi": "overall.baseline_minus_three_only.ci95.1",
                       "bound": "lo", "comparison": "above", "met": None, "digits": 5}


@pytest.mark.parametrize("name, change, message", [
    # 1. inside a paired difference only the frozen or baseline comparator may appear
    ("epv main on baseline minus three_only (an ablation)", lambda d: _set_main(_epv(d), "overall.baseline_minus_three_only"),
     "another arm"),
    ("decisions-E2 slope on the ablation", lambda d: [c.update(value="metrics.ablation_delta_epv_coef.point",
                                                                lo="metrics.ablation_delta_epv_coef.lo",
                                                                hi="metrics.ablation_delta_epv_coef.hi")
                                                       for c in _dec2(d)["conditions"][1:]], "another arm"),
    # 2. each part reads the statistic held for it
    ("decisions-E2 main on the new arm's slope", lambda d: _set_main(_dec2(d), "metrics.new.slope", ("lo", "hi")),
     "not the statistics held for it"),
    ("decisions-E2 slope on the new arm's regret coefficient", lambda d: [
        c.update(value="metrics.new.regret_coef.point", lo="metrics.new.regret_coef.lo", hi="metrics.new.regret_coef.hi")
        for c in _dec2(d)["conditions"][1:]], "not the statistics held for it"),
    ("epv main on one time bucket", lambda d: _set_main(_epv(d), "buckets.0-4.baseline_minus_e1"),
     "not the statistics held for it"),
    ("epv slope on the overall slope", lambda d: [c.update(value="slope_all.e1") for c in _epv(d)["conditions"]],
     "was decided on"),
    ("epv slope on the old bucket path", lambda d: [c.update(value="buckets.0-4.slope_e1")
                                                    for c in _epv(d)["conditions"]], "not the statistics held for it"),
    ("decisions-E2 extra condition on the regret coefficient again", lambda d: _dec2(d)["conditions"].append(
        {"label": "x", "value": "metrics.new.regret_coef.point", "lo": "metrics.new.regret_coef.lo",
         "hi": "metrics.new.regret_coef.hi", "bound": "lo", "comparison": "above", "threshold": 0, "met": None,
         "digits": 4}), ""),
    # 3. a condition's own words state its bar after a comparison
    ("an epv condition on '95 percent'", lambda d: _epv(d)["conditions"].append(
        {"label": "x", "value": "overall.baseline_minus_e1.point", "bound": "point", "comparison": "above",
         "threshold": 95, "met": None, "digits": 5, "states": "95 percent"}), "names no threshold"),
    ("an epv condition on 'interval above 0'", lambda d: _epv(d)["conditions"].append(
        dict(_ABLATION_CONDITION, threshold=0, states="interval above 0")), ""),
    # rule quotes from another experiment's section
    ("decisions-E2 rule quote from decisions-E1", lambda d: _dec2(d)["rule_quote"].update(
        texts=["Ablation: kind-adjusted without the court-zone term."]), "not whole sentences"),
])
def test_the_methods_harden_probes_are_refused(source, name, change, message):
    doc = copy.deepcopy(source)
    change(doc)
    problems = _build_problems(doc)
    assert problems, name
    if message:
        assert any(message in line for line in problems), (name, problems[:3])


def test_every_rule_has_its_statistics_held_in_code_and_the_controls_build(source):
    from export.methods_statistics import RULE_STATISTICS
    R.COLLECT_STATISTICS = {}
    try:
        R.build_methods(REPO, source)
        seen = R.COLLECT_STATISTICS
    finally:
        R.COLLECT_STATISTICS = None
    assert seen == RULE_STATISTICS
    # controls: a report-only value labelled so, and the experiment's own adoption bullet as its rule, still build
    doc = copy.deepcopy(source)
    _dec2(doc)["values"].append({"label": "Paired difference, first run (report-only)",
                                 "value": FIRST_RUN + "metrics.paired_regret_coef_new_minus_frozen.point", "digits": 4})
    _dec2(doc)["rule_quote"].update(texts=[
        "Adoption threshold: the regret coefficient's 95 percent interval above 0, the paired difference (new minus "
        "frozen) in that coefficient with interval above 0, and the chosen-value calibration slope inside 0.8 to 1.2, "
        "all on development games."])
    assert _build_problems(doc) == []


def test_a_states_bar_follows_a_comparison():
    assert R._states_bar("1.0 s error lower by at least 0.10 ft", -0.1)
    assert R._states_bar("one-sided p < 0.05", 0.05)
    assert R._states_bar("inside 0.9 to 1.1", 0.9) and R._states_bar("inside 0.9 to 1.1", 1.1)
    assert not R._states_bar("95 percent", 95)
    assert not R._states_bar("its 95 percent interval above 0", 95)
