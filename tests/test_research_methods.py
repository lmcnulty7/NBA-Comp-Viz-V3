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
    assert ids == sorted(backlog) and len(ids) == 24
    assert [lane.lane for lane in methods.lanes] == ["xfg", "rebounds", "matchups", "gravity", "ghosting",
                                                     "embeddings", "epv", "decisions", "lineups", "winprob",
                                                     "load", "tracking"]
    assert S.find_private(methods.model_dump_json()) == []
    # every lane cites its dossier and every source path is a file of the repository
    for item in methods.sources:
        assert (REPO / item.path).is_file(), item.path


def test_only_the_pilot_has_a_result_and_every_other_test_is_pre_registered(source):
    methods = R.build_methods(REPO, source)
    states = {e.id: e for lane in methods.lanes for e in lane.experiments}
    others = {eid: e.status for eid, e in states.items() if eid != "xfg-E1"}
    assert set(others.values()) == {"pre_registered"}
    assert all(e.development is None and e.test is None for eid, e in states.items() if eid != "xfg-E1")
    pilot = states["xfg-E1"]
    dev = json.loads((REPO / "reports/research/xfg-E1-dev.json").read_text(encoding="utf-8"))
    final_exists = (REPO / "reports/research/xfg-E1-final.json").exists()
    assert pilot.development is not None and pilot.development.met is bool(dev["verdict"]["adopt"])
    assert pilot.development.rows == dev["shots"] and pilot.development.games == dev["games"]
    paired = dev["paired_cvap_minus_iso"]
    assert (pilot.development.diff, pilot.development.lo, pilot.development.hi) == tuple(
        round(paired[k], 5) for k in ("diff", "lo", "hi"))
    assert pilot.development.threshold == dev["verdict"]["threshold"]
    if not final_exists:
        assert pilot.test is None and pilot.status == "development_met_test_pending"


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
    names = {R.SOURCE, "reports/research/xfg-E1-dev.json"}
    for lane in source["lanes"]:
        names.add(lane["report"])
        names.update(v["source"] for row in lane["benchmarks"] for v in row["ours"])
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
    doc["paired_cvap_minus_iso"].update(lo=-0.0002, diff=0.0004, hi=0.0011)
    doc["verdict"].update(upper_95=0.0011, adopt=False)
    final.write_text(json.dumps(doc), encoding="utf-8")
    pilot = _pilot(R.build_methods(repo, source))
    assert pilot.status == "not_adopted_test" and not pilot.test.met and pilot.test.upper == 0.0011


def test_a_development_result_that_misses_its_bar_is_not_adopted(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["paired_cvap_minus_iso"].update(lo=-0.0001, diff=0.0003, hi=0.0009)
    doc["verdict"].update(upper_95=0.0009, adopt=False)
    path.write_text(json.dumps(doc), encoding="utf-8")
    pilot = _pilot(R.build_methods(repo, source))
    assert pilot.status == "not_adopted_development" and pilot.test is None


def test_a_result_file_without_a_named_key_is_a_problem(source, tmp_path):
    repo = _repo_copy(tmp_path, source)
    path = repo / "reports/research/xfg-E1-dev.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["paired_cvap_minus_iso"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(R.MethodsProblem, match="paired_cvap_minus_iso"):
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
    assert statuses == {"pre_registered", "development_met_test_pending", "not_adopted_test"}
    # real data: the committed source, through the methods step of export.site_data, with its manifest
    out = tmp_path / "real"
    problems, info = site_data.run(out, None, None, only=["methods"], log=QUIET)
    assert info["methods"]["lanes"] == 12 and info["methods"]["experiments"]["xfg-E1"]
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
