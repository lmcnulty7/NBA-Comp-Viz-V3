"""analytics/results.py: the reports/results.md generator.

Inputs are the committed lane reports (each written by its lane's train.py), copies of them edited the way a real run
would change them (fixture banner removed, the stage 5 banner or the --skip-heldout phrase added, held-out sections
dropped), and one report written by analytics.train_xfg's own writer on a tiny synthetic shots table. Nothing reads
DATA_ROOT. Every test here can fail: the committed-report test is the guard against format drift in a writer.
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from analytics import results as R
from analytics.registry import EXTENSION_NAMES

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
LANES = list(EXTENSION_NAMES) + (["xfg_baseline"] if (REPORTS / "xfg_baseline.md").exists() else [])
TAG = re.compile(r"\s*\(fixture output, not a result\)", re.I)
# the lanes whose writer states the selected model only inside the held-out section
SELECTION_IN_HELDOUT = {"matchups", "playcalling", "load"}


def report(lane: str) -> str:
    return (REPORTS / ("%s.md" % lane)).read_text()


def realify(text: str) -> str:
    """A fixture report as a real run would write it: no fixture tag on any heading, no fixture banner or status line."""
    text = TAG.sub("", text)
    keep = [ln for ln in text.split("\n") if "fixture output, not a result" not in ln.lower()]
    out = "\n".join(keep)
    assert "fixture output, not a result" not in out.lower()
    return out


def drop_heldout(text: str) -> str:
    """Remove every section whose heading starts with Held-out (any level) and the betting held-out season row."""
    lines, out, skip_level = text.split("\n"), [], None
    for ln in lines:
        m = re.match(r"^(#{1,6}) (.*)$", ln)
        if m:
            lvl = len(m.group(1))
            if skip_level is not None and lvl <= skip_level:
                skip_level = None
            if skip_level is None and R.clean(m.group(2)).startswith("Held-out"):
                skip_level = lvl
        if skip_level is None and not re.match(r"^\| \S+ \| heldout \|", ln):
            out.append(ln)
    return "\n".join(out)


def copied(m, text: str) -> bool:
    """The value and both interval ends appear in the report (some intervals are stated as lo and hi columns)."""
    ends = [x.strip() for x in m.interval.strip("[]").split(",")] if m.interval else []
    return m.value in text and all(e in text for e in ends)


def add_line_after_written(text: str, line: str) -> str:
    return re.sub(r"^(Written [^\n]*\n)", lambda m: m.group(1) + "\n" + line + "\n", text, count=1, flags=re.M)


# ── the committed reports ──────────────────────────────────────────────────────────────────────────
def test_lanes_follow_the_registry():
    assert list(R.LANES)[:15] == list(EXTENSION_NAMES) and set(R.LANES) - set(EXTENSION_NAMES) == {"xfg_baseline"}


@pytest.mark.parametrize("lane", LANES)
def test_every_committed_report_parses(lane):
    """Format drift guard: every reports/<ext>.md parses with no unparsed field, and every copied value is in the file."""
    text = report(lane)
    r = R.parse_report(lane, text, "reports/%s.md" % lane)
    assert not r.problems, r.problems
    assert r.selected and r.oof is not None and r.written_at and r.command
    assert r.status in R.FIXTURE_LIKE                       # every committed report is fixture output today
    for m in [r.oof] + r.oof_extra + [c for _, c in r.candidates]:
        assert copied(m, text), m
    if lane == "embeddings":
        assert r.heldout.state == R.HO_ABSENT               # the lane scores no held-out rows
    elif lane == "xfg_baseline":
        assert r.heldout.state == R.HO_NOT_RUN              # the smoke run never scores the held-out game
    else:
        # a lane that scores held-out rows: a renamed held-out section must not read as "absent"
        assert r.heldout.state in (R.HO_SCORED, R.HO_SKIPPED), r.heldout.state
    if r.heldout.state == R.HO_SCORED:
        for m in [r.heldout.metric] + r.heldout.extra:
            assert copied(m, text), m
    for label, sentence in r.gates:
        assert R.clean(sentence)


def test_committed_reports_render_with_fixture_banner(tmp_path):
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(REPORTS), "--out", str(out), "--strict"]) == 0
    text = out.read_text()
    block = text[text.index(R.BEGIN):text.index(R.END)]
    assert "**Fixture output, not a result.**" in block.split("### Sources")[0]
    for lane in LANES:
        assert "### %s (fixture output, not a result)" % lane in block
    assert "unparsed" not in block
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_committed_results_md_is_labelled_fixture_output():
    text = (REPORTS / "results.md").read_text()
    assert text.count(R.BEGIN) == 1 and text.count(R.END) == 1
    assert "**Fixture output, not a result.**" in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


# ── data status ────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lane", LANES)
def test_real_data_copy_is_labelled_real(lane):
    """The same report without its fixture banner is real data; its held-out numbers are copied unchanged."""
    fx = R.parse_report(lane, report(lane))
    r = R.parse_report(lane, realify(report(lane)))
    assert not r.problems, r.problems
    if lane == "xfg_baseline":
        assert r.status == R.SMOKE                          # its smoke status line survives: still not a result
        return
    assert r.status == (R.REAL_HELDOUT if fx.heldout.state == R.HO_SCORED else R.REAL)
    assert (r.selected, r.oof) == (fx.selected, fx.oof)
    if fx.heldout.state == R.HO_SCORED:
        assert r.heldout.metric == fx.heldout.metric


@pytest.mark.parametrize("lane", [x for x in LANES if x != "xfg_baseline"])
def test_skip_heldout_phrase_shows_not_scored(lane):
    text = add_line_after_written(drop_heldout(realify(report(lane))), "Held-out: %s." % R.SKIP_PHRASE)
    r = R.parse_report(lane, text)
    assert r.heldout.state == R.HO_SKIPPED and r.heldout.metric is None
    assert r.status == R.REAL_TRAINING_ONLY
    if lane in SELECTION_IN_HELDOUT:
        # the writer names the selected model only in the held-out block: without it the summary says unparsed, never guesses
        assert [f for f, _ in r.problems] == ["training"] and r.problem("training").startswith("unparsed")
    else:
        assert not r.problems, r.problems


def test_skip_heldout_phrase_on_a_fixture_report_keeps_the_fixture_label():
    text = add_line_after_written(drop_heldout(report("xfg")), "Held-out: %s." % R.SKIP_PHRASE)
    r = R.parse_report("xfg", text)
    assert r.status == R.FIXTURE and r.heldout.state == R.HO_SKIPPED and not r.problems


def test_stage5_banner_is_reflected_and_contradiction_is_unparsed():
    base = add_line_after_written(realify(report("rebounds")), "**%s.**" % R.STAGE5_PHRASE.capitalize())
    r = R.parse_report("rebounds", drop_heldout(base))
    assert r.status == R.REAL_TRAINING_ONLY and r.heldout.state == R.HO_NOT_SCORED_BANNER and not r.problems
    r = R.parse_report("rebounds", base)                    # banner says not scored, yet held-out numbers are there
    assert r.problem("heldout").startswith("unparsed") and r.status == R.REAL_TRAINING_ONLY
    r = R.parse_report("rebounds", add_line_after_written(realify(report("rebounds")), R.SKIP_PHRASE))
    assert r.problem("heldout") and "states held-out numbers" in r.problem("heldout")


# ── copying, never computing ───────────────────────────────────────────────────────────────────────
def test_heldout_numbers_are_copied_exactly():
    text = report("xfg")
    sel = R.parse_report("xfg", text).selected
    sub = "### %s (held-out, scored once)" % sel
    i = text.index(sub)
    j = text.index("| log loss |", i)
    k = text.index("\n", j)
    text = text[:j] + "| log loss | 0.1234567 | -0.000 | 9.99e-1 |" + text[k:]
    r = R.parse_report("xfg", text)
    assert not r.problems
    assert (r.heldout.metric.value, r.heldout.metric.interval) == ("0.1234567", "[-0.000, 9.99e-1]")


def test_heldout_without_training_numbers_is_unparsed():
    text = report("xfg")
    assert R.parse_report("xfg", text).heldout.state == R.HO_SCORED
    text = text.replace("| candidate | log loss | Brier | ECE (10 bins) | ECE under 0.02 |",
                        "| candidate | log-loss | Brier | ECE (10 bins) | ECE under 0.02 |", 1)
    r = R.parse_report("xfg", text)
    assert r.problem("training").startswith("unparsed") and r.problem("heldout").startswith("unparsed")
    assert "without training-fold numbers" in r.problem("heldout")
    block = R.render_block([r], "T", "cmd")
    overview = block.split("### Overview")[1].split("\n### ")[0]
    assert "0.6770" not in overview and "unparsed" in overview


def _shots(n_games, per_game, heldout_game):
    rng = np.random.default_rng(0)
    rows = []
    for g in ["01.%02d.2016.AAA.at.BBB" % (d + 1) for d in range(n_games)] + [heldout_game]:
        for _ in range(per_game):
            x, y = rng.uniform(-220, 220), rng.uniform(-5, 280)
            dist = float(np.hypot(x, y) / 10)
            rows.append({"game": g, "sportvu_game": g, "period": int(rng.integers(1, 5)), "clock_s": float(rng.uniform(0, 720)),
                         "made": bool(rng.uniform() < 1 / (1 + np.exp(-(1.2 - 0.07 * dist)))), "loc_x": round(x, 1),
                         "loc_y": round(y, 1), "shot_type": "3PT" if dist >= 22 else "2PT", "shot_distance_ft": float(round(dist)),
                         "court_x": 5.25 + y / 10, "court_y": 25 + x / 10, "split": "heldout_game" if g == heldout_game else None,
                         "heldout": g == heldout_game, "source": "bref", "release_frame": None, "closest_defender_dist_ft": None})
    return rows


def test_report_from_the_train_xfg_writer_copies_heldout_exactly(tmp_path):
    """The Phase 1 baseline writer itself, on a tiny synthetic table: a full (not smoke) run states held-out numbers."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from analytics import train_xfg as tx
    src = tmp_path / "shots.parquet"
    pd.DataFrame(_shots(4, 60, "12.16.2015.PHX.at.GSW")).to_parquet(src)
    out = tmp_path / "reports" / "xfg_baseline.md"
    ctx = tx.main(["--shots", str(src), "--report", str(out), "--folds", "4", "--seed", "0", "--bootstrap", "10"])
    r = R.parse_report("xfg_baseline", out.read_text())
    assert not r.problems, r.problems
    assert r.status == R.REAL_HELDOUT and r.heldout.state == R.HO_SCORED
    assert r.heldout.metric.value == "%.4f" % ctx["heldout_scores"]["logistic"]["log_loss"]
    s = ctx["scores"]["logistic"]["log_loss"]
    assert r.oof.name.startswith("log loss (out-of-fold")
    assert (r.oof.value, r.oof.interval) == ("%.4f" % s["point"], "[%.4f, %.4f]" % (s["lo"], s["hi"]))
    smoke = tmp_path / "smoke.md"
    tx.main(["--shots", str(src), "--report", str(smoke), "--bootstrap", "10", "--smoke"])
    r = R.parse_report("xfg_baseline", smoke.read_text())
    assert r.status == R.SMOKE and r.heldout.state == R.HO_NOT_RUN and not r.problems


# ── the output file: banner, markers, strict, check ────────────────────────────────────────────────
def _copy_reports(dst: Path, lanes=LANES, transform=None) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for lane in lanes:
        text = report(lane)
        (dst / ("%s.md" % lane)).write_text(transform(lane, text) if transform else text)
    return dst


def test_fixture_banner_propagates_and_real_inputs_carry_none(tmp_path):
    real = _copy_reports(tmp_path / "real", [x for x in LANES if x != "xfg_baseline"], lambda lane, t: realify(t))
    out = tmp_path / "real.md"
    assert R.main(["--reports", str(real), "--out", str(out), "--strict"]) == 0
    assert "fixture output, not a result" not in out.read_text().lower()
    assert R.REAL_HELDOUT in out.read_text()
    mixed = _copy_reports(tmp_path / "mixed", [x for x in LANES if x != "xfg_baseline"],
                          lambda lane, t: t if lane == "gravity" else realify(t))
    out = tmp_path / "mixed.md"
    assert R.main(["--reports", str(mixed), "--out", str(out), "--strict"]) == 0
    text = out.read_text()
    assert "**Fixture output, not a result.** 1 of 15 lane reports" in text
    assert "### gravity (fixture output, not a result)" in text and "### xfg\n" in text
    assert re.search(r"^\| gravity \| fixture output, not a result \|", text, re.M)
    assert re.search(r"^\| xfg \| real data, held-out scored once \|", text, re.M)


def test_hand_text_outside_the_markers_survives_regeneration(tmp_path):
    reports = _copy_reports(tmp_path / "reports")
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    text = out.read_text()
    hand_top, hand_bottom = "Summary written by hand.\n\n", "\n\n## xfg\n\nThe question, by hand.\n"
    text = text.replace(R.BEGIN, hand_top + R.BEGIN).replace(R.END, R.END + hand_bottom)
    text = text.replace("### Overview", "### Overview\n\nstale generated line")
    out.write_text(text)
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    new = out.read_text()
    assert hand_top + R.BEGIN in new and R.END + hand_bottom in new
    assert "stale generated line" not in new and new.count(R.BEGIN) == 1
    # a file without markers keeps its text; the block goes under the title
    out.write_text("# Results\n\nHand-written summary.\n")
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    new = out.read_text()
    assert new.startswith("# Results\n\n" + R.BEGIN) and new.rstrip().endswith("Hand-written summary.")
    out.write_text("# Results\n\n%s\nno end marker\n" % R.BEGIN)
    with pytest.raises(SystemExit):
        R.main(["--reports", str(reports), "--out", str(out)])
    assert out.read_text() == "# Results\n\n%s\nno end marker\n" % R.BEGIN      # never clobbered


def test_strict_fails_on_a_broken_report_and_a_missing_one(tmp_path, capsys):
    def broken(lane, text):
        return text.replace("## Candidate table (out-of-fold", "## Candidates, renamed") if lane == "xfg" else text
    reports = _copy_reports(tmp_path / "reports", transform=broken)
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    text = out.read_text()
    assert re.search(r"^\| xfg \| fixture output, not a result; unparsed \| unparsed \(no section starting", text, re.M)
    assert "unparsed xfg" in capsys.readouterr().err
    assert R.main(["--reports", str(reports), "--out", str(out), "--strict"]) == 1
    reports = _copy_reports(tmp_path / "ok")
    assert R.main(["--reports", str(reports), "--out", str(out), "--strict"]) == 0
    (reports / "load.md").unlink()
    assert R.main(["--reports", str(reports), "--out", str(out), "--strict"]) == 1
    assert re.search(r"^\| load \| report missing; unparsed \|", out.read_text(), re.M)


def test_check_reports_staleness_and_writes_nothing(tmp_path):
    reports = _copy_reports(tmp_path / "reports")
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(reports), "--out", str(out), "--check"]) == 1 and not out.exists()
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    assert R.main(["--reports", str(reports), "--out", str(out), "--check"]) == 0
    before = out.read_text()
    p = reports / "rebounds.md"
    p.write_text(p.read_text().replace("Selected: prior.", "Selected: logit."))
    assert R.main(["--reports", str(reports), "--out", str(out), "--check"]) == 1
    assert out.read_text() == before


def test_module_has_no_long_dashes():
    for p in (REPO / "analytics" / "results.py", Path(__file__), REPO / "docs" / "contract" / "results.md"):
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
