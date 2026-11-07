"""analytics/results.py: the reports/results.md generator.

Inputs are frozen copies of the lane reports in tests/fixtures/results/ (the fixture reports of 2026-11-07, each
written by its lane's train.py; frozen so that stage 5 regenerating reports/ does not move these tests), copies of them
edited the way a real run would change them (fixture banner removed, the stage 5 banner or the --skip-heldout phrase
added, held-out sections dropped, a second validation pair), and one report written by analytics.train_xfg's own
writer on a tiny synthetic shots table. One test also parses the live reports/<ext>.md: the guard against format drift
in a writer. Nothing reads DATA_ROOT. Every test here can fail.
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
FROZEN = REPO / "tests" / "fixtures" / "results"
LANES = list(EXTENSION_NAMES) + ["xfg_baseline"]
LIVE = [x for x in LANES if (REPORTS / ("%s.md" % x)).exists()]
TAG = re.compile(r"\s*\(fixture output, not a result\)", re.I)
# the lanes whose writer states the selected model only inside the held-out section
SELECTION_IN_HELDOUT = {"matchups", "playcalling", "load"}


def report(lane: str) -> str:
    return (FROZEN / ("%s.md" % lane)).read_text()


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


def add_line_before_heldout(text: str, line: str) -> str:
    """Insert line before the first level-2 Held-out heading (where the --skip-heldout PR puts its selection line)."""
    m = re.search(r"^## Held-out", text, re.M)
    i = m.start() if m else len(text)
    return text[:i] + line + "\n\n" + text[i:]


def add_line_after_written(text: str, line: str) -> str:
    return re.sub(r"^(Written [^\n]*\n)", lambda m: m.group(1) + "\n" + line + "\n", text, count=1, flags=re.M)


# ── the committed reports ──────────────────────────────────────────────────────────────────────────
def test_lanes_follow_the_registry():
    assert list(R.LANES)[:15] == list(EXTENSION_NAMES) and set(R.LANES) - set(EXTENSION_NAMES) == {"xfg_baseline"}


@pytest.mark.parametrize("lane", LIVE)
def test_every_live_report_parses(lane):
    """Format drift guard: every reports/<ext>.md, whatever run wrote it, parses with no unparsed field, and every
    copied value is in the file."""
    text = (REPORTS / ("%s.md" % lane)).read_text()
    r = R.parse_report(lane, text, "reports/%s.md" % lane)
    assert not r.problems, r.problems
    assert r.selected and r.oof is not None and r.written_at and r.command
    for m in [r.oof] + r.oof_extra + [c for _, c in r.candidates]:
        assert copied(m, text), m
    if r.heldout.state == R.HO_SCORED:
        for m in [r.heldout.metric] + r.heldout.extra:
            assert copied(m, text), m


# lane: (selected, out-of-fold value, interval, held-out state, held-out value, held-out interval), read by hand
# from the frozen reports. Pins which row each rule copies (a neighbouring row or bucket gives a different number).
PINNED = {
    "xfg": ("location_logistic", "0.6843", "[0.6653, 0.7095]", R.HO_SCORED, "0.6770", "[0.6554, 0.6991]"),
    "rebounds": ("prior", "1.7981", "[1.6972, 1.8927]", R.HO_SCORED, "2.1135", ""),
    "matchups": ("ridge", "1.3854", "[1.3381, 1.4578]", R.HO_SCORED, "1.2287", ""),
    "gravity": ("ridge (SHIPPED expectation (the plan's model))", "4.299", "[4.050, 4.496]", R.HO_SCORED, "4.892", ""),
    "ghosting": ("2 ridge on the lag-1 features (linear baseline)", "3.874", "[3.356, 4.304]", R.HO_SCORED, "3.623", ""),
    "embeddings": ("pca", "0.921", "[0.841, 1.000]", R.HO_ABSENT, None, None),
    "epv": ("constant", "1.3341", "[1.2365, 1.4472]", R.HO_SCORED, "1.2191", ""),
    "actions": ("hgb", "1.000", "[1.000, 1.000]", R.HO_SCORED, "1.000", ""),
    "coverage": ("hgb", "0.944", "[0.902, 0.979]", R.HO_SCORED, "0.835", ""),
    "decisions": ("location", "0.285", "[0.125, 0.421]", R.HO_SCORED, "-0.144", "[-0.167, -0.121]"),
    "playcalling": ("ridge", "1.3674", "[1.3195, 1.4340]", R.HO_SCORED, "1.2359", ""),
    "lineups": ("plus_minus", "0.624", "", R.HO_SCORED, "0.881", "[0.342, 1.000]"),
    "winprob": ("logistic", "0.1278", "", R.HO_SCORED, "0.1531", ""),
    "load": ("ridge (decline); logistic (absence)", "0.8584", "[0.5032, 1.3310]", R.HO_SCORED, "0.6271", ""),
    "betting": ("cover model market, margin model market", "0.6931", "[0.6931, 0.6931]", R.HO_SCORED, "0.6931", ""),
    "xfg_baseline": ("logistic", "0.6741", "", R.HO_NOT_RUN, None, None),
}


def test_pinned_table_covers_every_lane():
    assert sorted(PINNED) == sorted(LANES)


@pytest.mark.parametrize("lane", LANES)
def test_frozen_report_copies_the_pinned_rows(lane):
    text = report(lane)
    r = R.parse_report(lane, text, "tests/fixtures/results/%s.md" % lane)
    assert not r.problems, r.problems
    assert r.status in R.FIXTURE_LIKE and r.written_at and r.command
    sel, val, ci, hstate, hval, hci = PINNED[lane]
    assert (r.selected, r.oof.value, r.oof.interval) == (sel, val, ci)
    assert r.heldout.state == hstate
    if hval is not None:
        assert (r.heldout.metric.value, r.heldout.metric.interval) == (hval, hci)
    for m in [r.oof] + r.oof_extra + [c for _, c in r.candidates]:
        assert copied(m, text), m
    for label, sentence in r.gates:
        assert R.clean(sentence)


def test_committed_reports_render_with_fixture_banner(tmp_path):
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(FROZEN), "--out", str(out), "--strict"]) == 0
    text = out.read_text()
    block = text[text.index(R.BEGIN):text.index(R.END)]
    assert "**Fixture output, not a result.**" in block.split("### Sources")[0]
    for lane in LANES:
        assert "### %s (fixture output, not a result)" % lane in block
    assert "unparsed" not in block
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
        # today's writer names the selected model only in the held-out block: without it the summary says unparsed,
        # never guesses
        assert [f for f, _ in r.problems] == ["training"] and r.problem("training").startswith("unparsed")
    else:
        assert not r.problems, r.problems
    # the --skip-heldout PR's form: a "## Held-out ... not scored" section that keeps the prefix and states no numbers
    text2 = drop_heldout(realify(report(lane))) + "\n## Held-out games, not scored\n\nHeld-out: %s.\n" % R.SKIP_PHRASE
    r2 = R.parse_report(lane, text2)
    assert r2.heldout.state == R.HO_SKIPPED and [f for f, _ in r2.problems] == [f for f, _ in r.problems]


# the selection line the --skip-heldout PR adds outside the held-out block, before the held-out heading
SELECTION_LINES = {"matchups": ["ridge"], "playcalling": ["ridge"], "load": ["ridge on FEATURES", "logistic on FEATURES"]}


@pytest.mark.parametrize("lane", sorted(SELECTION_LINES))
def test_selected_candidate_line_is_preferred_and_survives_skip_heldout(lane):
    lines = "\n".join("Selected candidate: %s." % n for n in SELECTION_LINES[lane])
    fx = R.parse_report(lane, report(lane))
    with_line = add_line_before_heldout(report(lane), lines)
    r = R.parse_report(lane, with_line)
    assert not r.problems and r.oof == fx.oof and r.heldout.metric == fx.heldout.metric
    skipped = add_line_after_written(drop_heldout(realify(with_line)), "Held-out: %s." % R.SKIP_PHRASE)
    r = R.parse_report(lane, skipped)
    assert not r.problems, r.problems
    assert r.heldout.state == R.HO_SKIPPED and r.oof == fx.oof
    if lane == "load":
        assert r.selected == "ridge on FEATURES (decline); logistic on FEATURES (absence)"
    else:
        assert r.selected == "ridge"


def test_selected_candidate_line_wins_over_the_fallback():
    text = add_line_before_heldout(drop_heldout(report("matchups")), "Selected candidate: raw.")
    r = R.parse_report("matchups", add_line_after_written(text, R.SKIP_PHRASE))
    assert not r.problems and r.selected == "raw" and (r.oof.value, r.oof.interval) == ("1.5105", "[1.4062, 1.5897]")
    r = R.parse_report("matchups", add_line_before_heldout(report("matchups"), "Selected candidate: ridge.\nSelected candidate: raw."))
    assert "2 'Selected candidate:' lines" in r.problem("training")
    r = R.parse_report("load", add_line_after_written(drop_heldout(add_line_before_heldout(report("load"), "Selected candidate: mean.")),
                                                      R.SKIP_PHRASE))
    assert r.problem("training") and "do not map" in r.problem("training")



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


# ── review fixes: lineups pairs, decisions, epv, absent held-out, gravity levels, load absence, paths, time ─────────
def _two_pair_lineups() -> str:
    """The frozen lineups report with a second validation pair whose winner (rapm) is not pair 1's (plus_minus), and
    the mean line and held-out row rewritten for the mean winner, as analytics/lineups/train.py would write them."""
    text = report("lineups")
    i = text.index("Fit on 2020-21, predict 2021-22")
    j = text.index("Mean validation Spearman:", i)
    pair2 = ("Fit on 2020-21, 2021-22, predict 2022-23: 12 lineups present in both (130 in the fit seasons, 120 in the "
             "target), 5 gated lineups, 36 players, alpha rapm 300, alpha synergy 300.\n\n"
             "| candidate | Spearman | MSE | Spearman minus plus_minus (paired) |\n| --- | --- | --- | --- |\n"
             "| plus_minus | 0.100 [-0.500, 0.600] | 800.000 [300.000, 1300.000] | baseline |\n"
             "| rapm | 0.900 [0.700, 0.990] | 400.000 [200.000, 600.000] | 0.800 [0.400, 1.100] (excludes 0) |\n"
             "| rapm_synergy | 0.500 [0.100, 0.800] | 500.000 [250.000, 700.000] | 0.400 [0.100, 0.700] (excludes 0) |\n\n")
    text = text[:j] + pair2 + text[j:]
    text = re.sub(r"^Mean validation Spearman: .*$",
                  "Mean validation Spearman: plus_minus 0.362, rapm 0.738, rapm_synergy 0.538. Selected: **rapm**.", text, count=1, flags=re.M)
    return text.replace("| plus_minus (selected) |", "| rapm (selected) |")


def test_lineups_primary_metric_is_the_mean_over_pairs():
    r = R.parse_report("lineups", _two_pair_lineups())
    assert not r.problems, r.problems
    assert r.selected == "rapm" and (r.oof.value, r.oof.interval) == ("0.738", "")
    shown = {m.name: (m.value, m.interval) for m in r.oof_extra}
    assert shown["fit 2020-21, predict 2021-22: Spearman"] == ("0.576", "[-0.273, 0.923]")
    assert shown["fit 2020-21, 2021-22, predict 2022-23: Spearman"] == ("0.900", "[0.700, 0.990]")
    assert dict((n, m.value) for n, m in r.candidates) == {"plus_minus": "0.362", "rapm": "0.738", "rapm_synergy": "0.538"}
    assert any("predict 2022-23" in label and text.startswith("0.800") for label, text in r.gates)


def test_lineups_shapes_it_cannot_map_are_unparsed():
    text = _two_pair_lineups()
    r = R.parse_report("lineups", text.replace("Fit on 2020-21, 2021-22, predict 2022-23", "Fitted on 2020-21, 2021-22", 1))
    assert "'Fit on' lines but" in r.problem("training")
    r = R.parse_report("lineups", text.replace("| rapm_synergy | 0.500", "| rapm_extra | 0.500", 1))
    assert "differ from the mean line" in r.problem("training")
    r = R.parse_report("lineups", text.replace("rapm 0.738, ", "", 1))
    assert "no mean validation Spearman" in r.problem("training")


def test_decisions_selected_line_without_parenthetical():
    text = report("decisions").replace("Selected: **location** (no xFG table present).", "Selected: **location**.")
    assert "Selected: **location**." in text
    r = R.parse_report("decisions", text)
    assert not r.problems and r.selected == "location" and r.oof.value == "0.285"


def test_epv_accepts_the_why_line_of_a_learner_win():
    text = re.sub(r"^Why the constant: .*$", "Why: hgb has the lowest out-of-fold MSE (1.2000 against the constant's "
                  "1.3341, lift 10.0 percent).", report("epv"), count=1, flags=re.M)
    r = R.parse_report("epv", text)
    assert not r.problems and any(n.startswith("Why: hgb has the lowest") for n in r.notes)
    r = R.parse_report("epv", re.sub(r"^Why the constant: .*$", "", report("epv"), count=1, flags=re.M))
    assert "the reason for the selection" in r.problem("training")


@pytest.mark.parametrize("lane", [x for x in LANES if x not in ("embeddings", "xfg_baseline")])
def test_missing_heldout_section_is_unparsed_except_for_embeddings(lane):
    r = R.parse_report(lane, drop_heldout(realify(report(lane))))
    assert r.heldout.state == R.HO_ABSENT and "no held-out section" in r.problem("heldout")
    assert r.unparsed


def test_embeddings_and_betting_heldout_absence():
    r = R.parse_report("embeddings", realify(report("embeddings")))
    assert r.heldout.state == R.HO_ABSENT and not r.problems and r.status == R.REAL
    text = re.sub(r"^\| 2021-22 \| heldout \|.*\n", "", report("betting"), flags=re.M)
    assert "| heldout |" not in text
    r = R.parse_report("betting", text)
    assert "no held-out section" in r.problem("heldout")


def test_gravity_never_reads_the_second_run():
    text = report("gravity")
    i = text.index("## Held-out, scored once")
    j = text.index("## Second run:", i)
    r = R.parse_report("gravity", text[:i] + text[j:])
    assert "no held-out section" in r.problem("heldout")          # not the second run's "### Held-out" (MSE 2.129)
    text = text.replace("### Sign test: teammates' location-only xFG", "### Sign check: teammates' location-only xFG", 1)
    r = R.parse_report("gravity", text)
    assert "no Sign test section under Players" in r.problem("training")


def test_load_without_the_absence_model():
    text = report("load")
    i = text.index("## Candidate 3: absence from the next game")
    j = text.index("## Cross-check", i)
    text = (text[:i] + "## Candidate 3: absence from the next game (fixture output, not a result)\n\nNo player-game has a "
            "known absent_next (no loaded next game), so the absence baseline did not run.\n\n" + text[j:])
    text = re.sub(r"^\| log loss, selected model \(logistic\) \| 0.0010 \|\n", "| log loss, selected model (None) | n/a |\n",
                  text, flags=re.M)
    r = R.parse_report("load", text)
    assert not r.problems, r.problems
    assert r.selected == "ridge (decline)" and r.oof.value == "0.8584" and r.heldout.metric.value == "0.6271"
    assert r.heldout.extra == [] and any(label == "absence target" for label, _ in r.gates)
    r = R.parse_report("load", re.sub(r"\| quantity \| value \|[\s\S]*?\n\n", "No held-out player-game was loaded.\n\n", report("load")))
    assert "no held-out player-game was loaded" in r.problem("heldout")


def test_check_does_not_depend_on_path_spelling(tmp_path, monkeypatch):
    reports = _copy_reports(tmp_path / "reports")
    monkeypatch.chdir(tmp_path)
    assert R.main(["--reports", str(reports.resolve()), "--out", str((tmp_path / "results.md").resolve())]) == 0
    assert R.main(["--reports", "reports", "--out", "results.md", "--check"]) == 0
    assert R.main(["--reports", "./reports/", "--out", "./results.md", "--check"]) == 0
    assert "--reports reports --out results.md" in (tmp_path / "results.md").read_text()


def test_regeneration_keeps_the_old_time_when_nothing_changed(tmp_path):
    reports = _copy_reports(tmp_path / "reports")
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    old = re.sub(r"^Generated \S+ by", "Generated 2000-01-01T00:00:00Z by", out.read_text(), count=1, flags=re.M)
    out.write_text(old)
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    assert out.read_text() == old                                  # a no-op: same bytes, same time
    p = reports / "rebounds.md"
    p.write_text(p.read_text().replace("Selected: prior.", "Selected: logit."))
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    assert "Generated 2000-01-01T00:00:00Z" not in out.read_text()


def test_hand_text_quoting_a_marker_is_refused(tmp_path):
    reports = _copy_reports(tmp_path / "reports")
    out = tmp_path / "results.md"
    assert R.main(["--reports", str(reports), "--out", str(out)]) == 0
    text = out.read_text() + "\nThe generator writes between %s and the end marker.\n" % R.BEGIN
    out.write_text(text)
    with pytest.raises(SystemExit):
        R.main(["--reports", str(reports), "--out", str(out)])
    assert out.read_text() == text


# ── degenerate runs: the reports the writers emit when a gate leaves little or nothing to fit ──────────────────────
# Each report is written by the lane's own trainer on a tiny analytics.fixtures run (in-process, about a second each).
ONE_GAME = ["--fixtures", "--fixture-games", "2", "--fixture-heldout-every", "2", "--fixture-possessions", "12", "--bootstrap", "10"]
DEGENERATE = {
    # one training game: no grouped fold is possible, the writer labels its metrics in-sample
    **{lane: ONE_GAME for lane in ("rebounds", "matchups", "gravity", "ghosting", "epv", "actions", "coverage",
                                     "decisions", "playcalling")},
    "betting": ["--fixtures", "--fixture-games", "1"],
    # the embeddings early exit: fewer than 3 player-seasons pass the gate, nothing fitted, Inputs and Status only
    "embeddings": ["--fixtures", "--n-min", "100000", "--bootstrap", "10"],
    # one training season before the held-out one: no validation pair, plus_minus by default
    "lineups": ["--fixtures", "--fixture-generator", "local", "--n-games", "10", "--heldout-season", "2021-22", "--bootstrap", "10"],
}


@pytest.fixture(scope="module")
def degenerate_reports(tmp_path_factory):
    import importlib
    import os
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    root = tmp_path_factory.mktemp("degenerate")
    here = os.getcwd()
    os.chdir(root)                                   # nothing a fixture run writes lands in the repo
    try:
        for lane, argv in DEGENERATE.items():
            importlib.import_module("analytics.%s.train" % lane).main(argv + ["--report", str(root / ("%s.md" % lane))])
    finally:
        os.chdir(here)
    return {lane: (root / ("%s.md" % lane)).read_text() for lane in DEGENERATE}


@pytest.mark.parametrize("lane", [x for x in DEGENERATE if DEGENERATE[x] is ONE_GAME or x == "betting"])
def test_one_training_game_reports_parse_as_in_sample(degenerate_reports, lane):
    text = degenerate_reports[lane]
    r = R.parse_report(lane, text)
    assert not r.problems, r.problems
    assert r.status == R.FIXTURE and r.heldout.state == R.HO_SCORED
    assert r.oof.name.endswith("(in-sample)") and all(m.name.endswith("(in-sample)") for _, m in r.candidates)
    assert r.gates[0][0] == "metrics in-sample" and r.gates[0][1].lower().count("in-sample") >= 1
    assert copied(r.oof, text) and "n/a" not in r.oof.interval        # an n/a pair of lo and hi is no interval
    if lane in ("decisions", "playcalling"):
        assert r.oof.interval == ""                                      # these two state lo and hi as n/a here
    assert not R.parse_report(lane, report(lane)).in_sample          # the normal fixture report is out-of-fold


def test_embeddings_nothing_fitted_report_is_a_valid_record(degenerate_reports):
    text = degenerate_reports["embeddings"]
    assert "## Decision" not in text and "nothing fitted" in text
    r = R.parse_report("embeddings", text)
    assert not r.problems, r.problems
    assert r.selected == "none (fewer than 3 sufficient player-seasons: nothing fitted)"
    assert r.oof.show(primary=True) == "5-NN position accuracy n/a (nothing fitted)"
    assert r.heldout.state == R.HO_ABSENT and r.status == R.FIXTURE
    block = R.render_block([r], "T", "c")
    assert "| embeddings | fixture output, not a result | none (fewer than 3" in block
    assert "| nothing fitted | fewer than 3 sufficient player-seasons: nothing fitted |" in block
    # any other report without a Decision section stays unparsed
    other = text.replace("fewer than 3 sufficient player-seasons: nothing fitted", "something else went wrong")
    assert "no section starting 'Decision'" in R.parse_report("embeddings", other).problem("training")


def test_lineups_one_training_season_report(degenerate_reports):
    text = degenerate_reports["lineups"]
    r = R.parse_report("lineups", text)
    assert not r.problems, r.problems
    assert r.selected == "plus_minus" and r.oof.value == "n/a" and r.oof.note.startswith("One training season only")
    assert r.heldout.state == R.HO_SCORED


def test_run_all_fixture_banner_form_is_fixture():
    """PR #8 stamps run_all fixture reports: a title ending in the tag plus one banner line, nothing else."""
    text = realify(report("rebounds"))
    title, rest = text.split("\n", 1)
    stamped = title + " (fixture output, not a result)\n\n**Fixture output, not a result.**\n" + rest
    assert R.parse_report("rebounds", stamped).status == R.FIXTURE
    assert R.parse_report("rebounds", title + " (fixture output, not a result)\n" + rest).status == R.FIXTURE
    assert R.parse_report("rebounds", text).status == R.REAL_HELDOUT
