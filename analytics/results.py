"""analytics.results: one summary of every lane report, written to reports/results.md.

Reads reports/<ext>.md for the fifteen extensions of analytics.registry (and reports/xfg_baseline.md, the Phase 1
baseline of analytics.train_xfg, when present) and writes one overview table plus one summary table per lane:
the data status, the selected model, the primary out-of-fold metric with its interval, the held-out metric once a
report states one, and the gates and checks that fired, in the report's own wording.

Nothing here computes a metric. Every number is copied, as written, from the report it came from; a held-out number
is never inferred. The reports are hand-formatted markdown written by each lane's train.py, so the extraction is
one small explicit rule set per lane (`LANES`), keyed on the section headings and sentences those writers emit. When
a rule does not find what it expects the field is shown as "unparsed" with the reason; `--strict` turns any
unparsed field into a non-zero exit, which is how format drift in a writer surfaces.

Data status per lane:
  fixture output     the report carries "fixture output, not a result" (every analytics.fixtures run does)
  smoke run          the Phase 1 baseline's smoke run on the saved basketball-reference fixture pages
  real data, training folds only; held-out not scored
                     the stage 5 banner of docs/mac_runbook.md, or a report with the --skip-heldout phrase
  real data, held-out scored once
                     no fixture banner, held-out numbers stated in the report
  real data          no fixture banner, no held-out numbers (embeddings, or a held-out field that is unparsed)

Held-out states: a copied metric; "not scored (--skip-heldout)" when the report contains the exact phrase
`held-out games not scored (--skip-heldout)`; "not run yet" when the report says the held-out rows were set aside
and not scored (the baseline smoke run); "absent" when the report has no held-out section (unparsed too for every
lane but embeddings, unless the --skip-heldout phrase or the stage 5 banner says why); "unparsed" otherwise.

Regeneration rewrites only the text between `<!-- results:begin -->` and `<!-- results:end -->`; hand-written
sections outside the markers survive. Usage:

    python3 -m analytics.results [--reports reports] [--out reports/results.md] [--strict] [--check]

--check writes nothing and exits 1 when the file on disk differs from what would be generated (the generated-at
time is taken from the file on disk, so only content changes count). Contract: docs/contract/results.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from analytics.registry import EXTENSION_NAMES

BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
FIXTURE_PHRASE = "fixture output, not a result"
SKIP_PHRASE = "held-out games not scored (--skip-heldout)"
STAGE5_PHRASE = "real data, training folds only; held-out not scored"
SMOKE_PHRASE = "This is a smoke run"

FIXTURE = "fixture output, not a result"
SMOKE = "smoke run on fixture pages, not a result"
REAL_TRAINING_ONLY = STAGE5_PHRASE
REAL_HELDOUT = "real data, held-out scored once"
REAL = "real data"
MISSING = "report missing"
FIXTURE_LIKE = (FIXTURE, SMOKE)

HO_SCORED = "scored"
HO_SKIPPED = "not scored (--skip-heldout)"
HO_NOT_SCORED_BANNER = "not scored (stage 5 banner: held-out not scored)"
HO_NOT_RUN = "not run yet"
HO_ABSENT = "absent"

_TAG_RE = re.compile(r"\s*\(fixture output, not a result\)", re.I)
_WRITTEN_RE = re.compile(r"^Written (\S+) by `([^`]+)`", re.M)
_HEADING_RE = re.compile(r"^(#{1,6}) (.*?)\s*$")
_CELL_RE = re.compile(r"^\s*(\S+)\s*(\[[^\]]*\])?\s*(.*?)\s*$")


class ParseError(Exception):
    """A rule did not find what the lane's writer emits; the message is the reason shown as unparsed."""


def clean(text: str) -> str:
    """Drop the fixture tag and markdown bold from copied text (the status column carries the label instead)."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text).replace("**", "")).strip()


# ── markdown reading ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Table:
    header: list
    rows: list

    def col(self, name: str) -> int:
        if name not in self.header:
            raise ParseError("table has no column %r" % name)
        return self.header.index(name)

    def cell(self, row: list, name: str) -> str:
        i = self.col(name)
        if i >= len(row):
            raise ParseError("row %r has no %r cell" % (row[:1], name))
        return row[i]

    def where(self, name: str, pred) -> list:
        i = self.col(name)
        return [r for r in self.rows if i < len(r) and pred(r[i])]

    def one(self, name: str, pred, what: str) -> list:
        hits = self.where(name, pred)
        if len(hits) != 1:
            raise ParseError("%s: %d matching rows in column %r, expected 1" % (what, len(hits), name))
        return hits[0]


def _split_row(line: str) -> list:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


@dataclass
class Section:
    level: int
    title: str           # cleaned: no fixture tag, no bold
    lines: list          # body lines up to the next heading of the same or a higher level (subsections included)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def tables(self) -> list:
        out, i, L = [], 0, self.lines
        while i < len(L):
            if L[i].lstrip().startswith("|") and i + 1 < len(L) and re.match(r"^\s*\|\s*:?-{3,}", L[i + 1]):
                header = [clean(c) for c in _split_row(L[i])]
                rows, j = [], i + 2
                while j < len(L) and L[j].lstrip().startswith("|"):
                    rows.append(_split_row(L[j]))
                    j += 1
                out.append(Table(header, rows))
                i = j
            else:
                i += 1
        return out

    def table(self, *cols: str) -> Table:
        for t in self.tables():
            if all(c in t.header for c in cols):
                return t
        raise ParseError("section %r has no table with columns %s" % (self.title, ", ".join(cols)))

    def search(self, pattern: str, what: str, flags=re.M):
        m = re.search(pattern, self.text, flags)
        if not m:
            raise ParseError("section %r: %s not found" % (self.title, what))
        return m

    def line(self, prefix: str, what: str = None) -> str:
        """The first body line (cleaned) that starts with prefix."""
        for raw in self.lines:
            s = clean(raw.lstrip("- "))
            if s.startswith(prefix):
                return s
        raise ParseError("section %r: no line starting %r%s" % (self.title, prefix, " (%s)" % what if what else ""))

    def find(self, prefix: str, level: int = None):
        return _find(_sections(self.lines, base=self.level), prefix, level)


def _sections(lines: list, base: int = 0) -> list:
    heads = []
    for i, raw in enumerate(lines):
        m = _HEADING_RE.match(raw)
        if m and len(m.group(1)) > base:
            heads.append((i, len(m.group(1)), clean(m.group(2))))
    out = []
    for k, (i, lvl, title) in enumerate(heads):
        end = len(lines)
        for j, l2, _ in heads[k + 1:]:
            if l2 <= lvl:
                end = j
                break
        out.append(Section(lvl, title, lines[i + 1:end]))
    return out


def _find(sections: list, prefix: str, level: int = None):
    for s in sections:
        if s.title.startswith(prefix) and (level is None or s.level == level):
            return s
    return None


class Doc:
    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()
        self.sections = _sections(self.lines)

    def section(self, prefix: str, level: int = None):
        return _find(self.sections, prefix, level)

    def need(self, prefix: str, level: int = None) -> Section:
        s = self.section(prefix, level)
        if s is None:
            raise ParseError("no section starting %r" % prefix)
        return s

    def search(self, pattern: str, what: str, flags=re.M):
        m = re.search(pattern, self.text, flags)
        if not m:
            raise ParseError("%s not found" % what)
        return m


# ── the record ─────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Metric:
    name: str
    value: str
    interval: str = ""           # "[lo, hi]" exactly as written, or "" when the report states none
    note: str = ""

    def show(self, primary: bool = False) -> str:
        s = "%s %s" % (self.name, self.value)
        if self.interval:
            s += " " + self.interval
        if self.note:
            s += " (%s)" % self.note
        elif primary and not self.interval:
            s += " (no interval stated)"
        return s


def metric(name: str, cell: str, note: str = "") -> Metric:
    """A report cell such as `0.6843 [0.6653, 0.7095]` copied into value and interval, unchanged."""
    cell = clean(cell)
    m = _CELL_RE.match(cell)
    if not m or not cell:
        raise ParseError("%s: empty cell" % name)
    rest = m.group(3)
    if rest:
        raise ParseError("%s: unexpected text %r after the value" % (name, rest))
    return Metric(name, m.group(1), m.group(2) or "", note)


@dataclass
class HeldOut:
    state: str
    metric: Metric = None
    extra: list = field(default_factory=list)
    note: str = ""


@dataclass
class LaneResult:
    lane: str
    path: str
    written_at: str = ""
    command: str = ""
    status: str = ""
    selected: str = ""
    oof: Metric = None
    oof_extra: list = field(default_factory=list)
    candidates: list = field(default_factory=list)      # (name, Metric) on the primary metric
    heldout: HeldOut = None
    gates: list = field(default_factory=list)           # (label, text in the report's words)
    notes: list = field(default_factory=list)
    problems: list = field(default_factory=list)        # (field, reason): field is "report", "training" or "heldout"
    in_sample: str = ""                                 # the report's words when its training metrics are in-sample

    @property
    def unparsed(self) -> bool:
        return bool(self.problems)

    def problem(self, where: str) -> str:
        reasons = [r for f, r in self.problems if f == where or f == "report"]
        return "unparsed (%s)" % "; ".join(reasons) if reasons else ""


def _lohi(t: Table, row: list) -> str:
    """The interval of a row stated as separate lo and hi columns, as "[lo, hi]"; "" when the report gives n/a for both."""
    lo, hi = t.cell(row, "lo"), t.cell(row, "hi")
    return "" if lo == hi == "n/a" else "[%s, %s]" % (lo, hi)


def _in_sample(r, sec: Section, text: str = None):
    """Mark the record in-sample when the lane's own section says so: a heading carrying the writer's mode
    ("in-sample") or a protocol sentence saying IN-SAMPLE (one training game: no grouped fold is possible)."""
    body = sec.text if text is None else text
    if "in-sample" in sec.title.lower():
        r.in_sample = sec.title
    elif "IN-SAMPLE" in body:
        line = next(clean(x) for x in body.split("\n") if "IN-SAMPLE" in x)
        r.in_sample = re.search(r"[^.;]*IN-SAMPLE[^.;]*", line).group(0).strip()


def _sel_row(t: Table, col: str, name: str, what: str) -> list:
    return t.one(col, lambda c: clean(c).replace(" (selected)", "") == name, what)


def _candidates(t: Table, name_col: str, metric_col: str, mname: str, skip=()) -> list:
    out = []
    for r in t.rows:
        n = clean(t.cell(r, name_col)).replace(" (selected)", "")
        if n in skip:
            continue
        try:
            out.append((n, metric(mname, t.cell(r, metric_col))))
        except ParseError:
            out.append((n, Metric(mname, clean(t.cell(r, metric_col)) or "blank")))
    return out


def _gate_line(sec: Section, prefix: str, label: str, rec: LaneResult):
    rec.gates.append((label, sec.line(prefix)))


HELDOUT = "Held-out"          # every lane's held-out heading starts with this (scored once or not scored)
_SELECTED_CANDIDATE_RE = re.compile(r"^Selected candidate: (.+?)\.$")


def selected_candidates(d: Doc) -> list:
    """The names on `Selected candidate: <name>.` lines outside every held-out section, in order. The line is the
    selection statement every writer emits before its held-out heading, with or without --skip-heldout."""
    held = set()
    for i, raw in enumerate(d.lines):
        m = _HEADING_RE.match(raw)
        if m and clean(m.group(2)).startswith(HELDOUT):
            lvl, j = len(m.group(1)), i + 1
            while j < len(d.lines):
                m2 = _HEADING_RE.match(d.lines[j])
                if m2 and len(m2.group(1)) <= lvl:
                    break
                held.add(j)
                j += 1
    out = []
    for i, raw in enumerate(d.lines):
        m = _SELECTED_CANDIDATE_RE.match(clean(raw))
        if m and i not in held:
            out.append(m.group(1))
    return out


def _one_selected_candidate(d: Doc):
    names = selected_candidates(d)
    if len(names) > 1:
        raise ParseError("%d 'Selected candidate:' lines, expected 1" % len(names))
    return names[0] if names else None


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.)])\s+(?=[A-Z])", text.strip(), maxsplit=1)
    return parts[0]


# ── per-lane rules: train(doc, rec) fills the training-fold fields; heldout(doc, rec) -> HeldOut or None ─────
def xfg_train(d: Doc, r: LaneResult):
    s = d.need("Candidate table (out-of-fold")
    r.selected = s.search(r"^Selected: \*\*([^*]+)\*\*", "the Selected line").group(1)
    t = s.table("candidate", "log loss", "selected")
    row = t.one("selected", lambda c: c == "yes", "selected candidate")
    if clean(t.cell(row, "candidate")) != r.selected:
        raise ParseError("the selected column (%s) and the Selected line (%s) disagree" % (t.cell(row, "candidate"), r.selected))
    r.oof = metric("log loss", t.cell(row, "log loss"))
    r.oof_extra = [metric(c, t.cell(row, c)) for c in ("Brier", "ECE (10 bins)", "calibration slope")]
    r.candidates = _candidates(t, "candidate", "log loss", "log loss")
    r.gates.append(("ECE under 0.02 (a target, not a selection rule)", "selected candidate: %s" % t.cell(row, "ECE under 0.02")))
    pt = s.table("pair", "diff", "lo", "hi", "interval excludes 0")
    for p in pt.rows:
        r.gates.append(("paired bootstrap, log loss, %s" % pt.cell(p, "pair"), "%s [%s, %s], interval excludes 0: %s" % tuple(
            pt.cell(p, c) for c in ("diff", "lo", "hi", "interval excludes 0"))))


def xfg_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    sub = s.find("%s (held-out" % r.selected)
    if sub is not None:
        t = sub.table("metric", "point", "lo", "hi")
        row = t.one("metric", lambda c: c == "log loss", "held-out log loss")
        m = Metric("log loss", t.cell(row, "point"), _lohi(t, row))
        b = t.one("metric", lambda c: c == "Brier", "held-out Brier")
        return HeldOut(HO_SCORED, m, [Metric("Brier", t.cell(b, "point"), _lohi(t, b))])
    t = s.table("candidate", "log loss")
    row = _sel_row(t, "candidate", r.selected, "held-out row of the selected candidate")
    return HeldOut(HO_SCORED, metric("log loss", t.cell(row, "log loss"), "point estimate"))


def baseline_train(d: Doc, r: LaneResult):
    s = d.need("Metrics (")
    mode = s.title[len("Metrics ("):].split(",")[0]
    # train_xfg fits a constant floor and the logistic location model; the logistic model is the baseline (its held-out
    # section calls it "the selected (logistic) model"); there is no selection between candidates.
    r.selected = "logistic"
    t = s.table("model", "log loss", "Brier")
    row = t.one("model", lambda c: c == "logistic", "logistic row")
    r.oof = metric("log loss (%s)" % mode, t.cell(row, "log loss"))
    r.oof_extra = [metric("Brier (%s)" % mode, t.cell(row, "Brier")),
                   metric("calibration slope (%s)" % mode, t.cell(row, "calibration slope"))]
    r.candidates = _candidates(t, "model", "log loss", "log loss (%s)" % mode)
    r.notes.append("Phase 1 baseline (analytics/train_xfg.py): a constant floor and the logistic location model, no selection rule")
    if mode == "in-sample":
        r.gates.append(("metrics in-sample", d.need("Protocol").line("Metrics:")))


def baseline_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    if "NOT scored" in s.text:
        m = re.search(r"(set aside and NOT scored[^.]*)\.", clean(s.text))
        return HeldOut(HO_NOT_RUN, note=m.group(1) if m else "the report says the held-out rows were NOT scored")
    t = s.table("model", "log loss")
    row = t.one("model", lambda c: c == "logistic", "held-out logistic row")
    return HeldOut(HO_SCORED, metric("log loss", t.cell(row, "log loss"), "point estimate"),
                   [metric("Brier", t.cell(row, "Brier"), "point estimate")])


def rebounds_train(d: Doc, r: LaneResult):
    s = d.need("Candidates (", level=2)
    _in_sample(r, s)
    r.selected = s.search(r"^Selected: (\S+?)\.?$", "the Selected line").group(1)
    t = s.table("model", "per-miss log loss", "top-1 accuracy")
    row = _sel_row(t, "model", r.selected, "selected model row")
    r.oof = metric("per-miss log loss", t.cell(row, "per-miss log loss"))
    r.oof_extra = [metric("top-1 accuracy", t.cell(row, "top-1 accuracy"))]
    r.candidates = _candidates(t, "model", "per-miss log loss", "per-miss log loss")
    _gate_line(s, "Paired bootstrap by game", "paired bootstrap", r)


def rebounds_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("model", "per-miss log loss", "top-1 accuracy")
    row = _sel_row(t, "model", r.selected, "held-out row of the selected model")
    return HeldOut(HO_SCORED, metric("per-miss log loss", t.cell(row, "per-miss log loss"), "point estimate"),
                   [metric("top-1 accuracy", t.cell(row, "top-1 accuracy"), "point estimate")])


def matchups_train(d: Doc, r: LaneResult):
    s = d.need("Baseline comparison: possession points")
    _in_sample(r, s)
    t = s.table("model", "out-of-fold MSE")
    r.selected = _one_selected_candidate(d)
    if r.selected is None:
        # fallback: older writers state the selection only in the held-out section (analytics/matchups/train.py)
        r.selected = d.search(r"^Selected candidate: (\S+) \(", "the 'Selected candidate:' line").group(1)
    row = t.one("model", lambda c: c == r.selected, "selected model row")
    r.oof = metric("possession points MSE", t.cell(row, "out-of-fold MSE"))
    r.candidates = _candidates(t, "model", "out-of-fold MSE", "MSE", skip={n for n in (clean(x[0]) for x in t.rows) if " minus " in n})
    for p in t.where("model", lambda c: " minus " in c):
        r.gates.append(("paired bootstrap, %s" % clean(p[0]), clean(t.cell(p, "out-of-fold MSE"))))
    dfn = d.need("Defenders")
    r.gates.append(("defender gate", _first_sentence(_first_body(dfn))))


def _first_body(s: Section) -> str:
    for raw in s.lines:
        if raw.strip() and not raw.lstrip().startswith("|"):
            return clean(raw)
    raise ParseError("section %r is empty" % s.title)


def matchups_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("model", "MSE")
    row = t.one("model", lambda c: c == "%s (selected)" % r.selected, "held-out row of the selected model")
    return HeldOut(HO_SCORED, metric("possession points MSE", t.cell(row, "MSE"), "point estimate"))


def gravity_train(d: Doc, r: LaneResult):
    s = d.need("Candidates (", level=2)
    _in_sample(r, s)
    t = s.table("model", "MSE", "role")
    row = t.one("role", lambda c: c.startswith("SHIPPED"), "shipped expectation row")
    r.selected = "%s (%s)" % (clean(row[0]), t.cell(row, "role"))
    r.oof = metric("MSE of nearest-defender distance (ft^2)", t.cell(row, "MSE"))
    r.oof_extra = [metric("R2", t.cell(row, "R2 (1 - MSE / var y)"))]
    r.candidates = _candidates(t, "model", "MSE", "MSE")
    r.gates.append(("paired bootstrap", _first_sentence(s.line("Paired bootstrap by game"))))
    players = d.need("Players (gate:", level=2)
    r.gates.append(("player gate", players.title))
    dec = players.find("Top-decile check")
    if dec is None:
        raise ParseError("no Top-decile check section under Players")
    for raw in dec.lines:
        s2 = clean(raw)
        if "passed:" in s2 and not s2.startswith("|"):
            r.gates.append(("top-decile check", s2))
    sign = players.find("Sign test")
    if sign is None:
        raise ParseError("no Sign test section under Players")
    r.gates.append(("sign test", sign.line("Candidates")))
    if d.section("Second run:", level=2) is not None:
        r.notes.append("the fixture report carries a second run on the planted-pull generator; this summary copies the first run")


def gravity_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    m = s.search(r"refit on all \d+ training rows: MSE (\S+) \(constant floor (\S+?),", "the held-out MSE sentence")
    return HeldOut(HO_SCORED, Metric("MSE of nearest-defender distance (ft^2)", m.group(1), "", "point estimate"),
                   [Metric("constant floor MSE", m.group(2), "", "not a selection input")])


def ghosting_train(d: Doc, r: LaneResult):
    dec = d.need("Decision")
    txt = clean(" ".join(dec.lines))
    m = re.search(r"Selected ghost: (.+?)\.\s*$", txt)
    if not m:
        raise ParseError("Decision: 'Selected ghost:' not found")
    r.selected = m.group(1)
    r.gates.append(("deep candidate (MLP, GRU stand-in)", _first_sentence(txt)))
    hz = d.need("Candidates by horizon")
    _in_sample(r, hz)
    t = hz.table("candidate", "0.2 s error", "1.0 s error")
    row = t.one("candidate", lambda c: c == r.selected, "selected candidate row")
    r.oof = metric("1.0 s position error (ft)", t.cell(row, "1.0 s error"))
    r.oof_extra = [metric("0.2 s position error (ft)", t.cell(row, "0.2 s error"))]
    r.candidates = _candidates(t, "candidate", "1.0 s error", "1.0 s error")
    r.gates.append(("validation sign test", d.need("Validation").line("Sign test:")))


def ghosting_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("candidate", "0.2 s error", "1.0 s error")
    row = t.one("candidate", lambda c: c == r.selected, "held-out row of the selected candidate")
    return HeldOut(HO_SCORED, metric("1.0 s position error (ft)", t.cell(row, "1.0 s error"), "point estimate"),
                   [metric("0.2 s position error (ft)", t.cell(row, "0.2 s error"), "point estimate")])


NOTHING_FITTED = "nothing fitted"


def embeddings_train(d: Doc, r: LaneResult):
    st = d.section("Status", level=2)
    if d.section("Decision") is None and d.section("Candidates") is None and st is not None \
            and _first_body(st).endswith(NOTHING_FITTED):
        # the documented early exit: Inputs and Status only, the Status body the reason (and the Mac command)
        reason = _first_body(st)
        r.selected = "none (%s)" % reason
        r.oof = Metric("5-NN position accuracy", "n/a", note=NOTHING_FITTED)
        r.gates.append(("nothing fitted", reason))
        r.notes.append(d.need("Inputs").line("Player-seasons:"))
        return
    dec = d.need("Decision")
    m = dec.search(r"^Chosen: \*\*([^*]+)\*\*\. (.+?)\.?$", "the Chosen line")
    r.selected = m.group(1)
    r.gates.append(("autoencoder gate", clean(m.group(2))))
    t = d.need("Candidates").table("method", "5-NN position accuracy")
    row = t.one("method", lambda c: c == r.selected, "chosen method row")
    r.oof = metric("5-NN position accuracy (leave-one-out)", t.cell(row, "5-NN position accuracy"))
    r.oof_extra = [metric("majority baseline", t.cell(row, "majority baseline")),
                   metric("next-season R squared (ridge, out-of-fold)", t.cell(row, "next-season R squared (ridge)"))]
    r.candidates = _candidates(t, "method", "5-NN position accuracy", "5-NN accuracy")
    r.notes.append(d.need("Inputs").line("Player-seasons:"))


def embeddings_heldout(d: Doc, r: LaneResult):
    if d.section(HELDOUT) is not None:
        raise ParseError("an embeddings report with a held-out section: the rule for it is not written yet")
    r.notes.append("the lane scores no held-out rows: held-out material is set aside before any feature")
    return None


def epv_train(d: Doc, r: LaneResult):
    s = d.need("Candidates by elapsed-time bucket")
    _in_sample(r, s)
    r.selected = s.search(r"^Selected: \*\*([^*]+)\*\*", "the Selected line").group(1)
    t = s.table("bucket (s)", "model", "MSE expected points", "Brier P(points > 0)")
    alls = Table(t.header, t.where("bucket (s)", lambda c: c == "all"))
    row = alls.one("model", lambda c: c == r.selected, "selected model row, bucket all")
    r.oof = metric("MSE expected points", alls.cell(row, "MSE expected points"))
    r.oof_extra = [metric("Brier P(points > 0)", alls.cell(row, "Brier P(points > 0)"))]
    r.candidates = _candidates(alls, "model", "MSE expected points", "MSE")
    try:
        r.notes.append(s.line("Why the"))
    except ParseError:
        r.notes.append(s.line("Why:", "the reason for the selection"))
    g = d.need("Gradient boosting by shot-clock bucket and the gate")
    r.gates.append(("sequence model gate", g.line("Gate decision:")))
    try:
        r.gates.append(("sequence model kept or dropped", g.line("Paired bootstrap by game")))
    except ParseError:
        pass                     # the gate stayed shut: no sequence model, no paired test
    r.gates.append(("shuffled-target check", d.need("Shuffled-target check").line("Possession points permuted")))


def epv_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("bucket (s)", "model", "MSE expected points")
    alls = Table(t.header, t.where("bucket (s)", lambda c: c == "all"))
    row = alls.one("model", lambda c: c == r.selected, "held-out row of the selected model, bucket all")
    return HeldOut(HO_SCORED, metric("MSE expected points", alls.cell(row, "MSE expected points"), "point estimate"),
                   [metric("Brier P(points > 0)", alls.cell(row, "Brier P(points > 0)"), "point estimate")])


def actions_train(d: Doc, r: LaneResult):
    s = d.need("Metrics (", level=2)
    _in_sample(r, s)
    m = s.search(r"^Selected model: \*\*([^*]+)\*\* \((.+)\)\.$", "the Selected model line")
    r.selected = m.group(1)
    r.gates.append(("plan target", m.group(2)))
    t = s.table("model", "F1 ball_screen", "macro F1")
    row = t.one("model", lambda c: c == r.selected, "selected model row")
    r.oof = metric("F1 ball_screen", t.cell(row, "F1 ball_screen"))
    r.oof_extra = [metric("macro F1", t.cell(row, "macro F1"))]
    r.candidates = _candidates(t, "model", "F1 ball_screen", "F1 ball_screen")
    r.gates.append(("detections", s.line("Detected actions")))


def actions_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("model", "F1 ball_screen", "macro F1")
    row = t.one("model", lambda c: c == r.selected, "held-out row of the selected model")
    return HeldOut(HO_SCORED, metric("F1 ball_screen", t.cell(row, "F1 ball_screen"), "point estimate"),
                   [metric("macro F1", t.cell(row, "macro F1"), "point estimate")])


def coverage_train(d: Doc, r: LaneResult):
    s = d.need("Candidates (", level=2)
    _in_sample(r, s)
    m = s.search(r"^Selected: (\S+)\. (.+)$", "the Selected line")
    r.selected = m.group(1)
    r.gates.append(("target", clean(m.group(2))))
    t = s.table("model", "macro F1", "accuracy", "selected")
    row = t.one("selected", lambda c: c == "yes", "selected model row")
    if clean(row[0]) != r.selected:
        raise ParseError("the selected column (%s) and the Selected line (%s) disagree" % (row[0], r.selected))
    r.oof = metric("macro F1", t.cell(row, "macro F1"))
    r.oof_extra = [metric("accuracy", t.cell(row, "accuracy"))]
    r.candidates = _candidates(t, "model", "macro F1", "macro F1")


def coverage_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("model", "macro F1", "accuracy")
    row = t.one("model", lambda c: c == r.selected, "held-out row of the selected model")
    return HeldOut(HO_SCORED, metric("macro F1", t.cell(row, "macro F1"), "point estimate"),
                   [metric("accuracy", t.cell(row, "accuracy"), "point estimate")])


def decisions_train(d: Doc, r: LaneResult):
    v = d.need("Value-function assumptions")
    rule = v.line("Prior of the option values:")
    m = re.search(r"Selected: (\S+?)(?: \((.+?)\))?\.?$", rule)
    if not m:
        raise ParseError("Value-function assumptions: 'Selected:' not found on the prior line")
    r.selected = m.group(1)
    r.gates.append(("prior rule", rule[len("Prior of the option values: "):]))
    _in_sample(r, d.need("Inputs"))
    s = d.need("Validation: mean decision quality")
    t = s.table("prior", "games", "mean rho", "lo", "hi", "positive / signed", "sign p")
    row = t.one("prior", lambda c: c == r.selected, "selected prior row")
    r.oof = Metric("mean Spearman, decision quality vs on-court rating (by game)", t.cell(row, "mean rho"),
                   _lohi(t, row))
    r.oof_extra = [Metric("positive / signed games", t.cell(row, "positive / signed")), Metric("sign p", t.cell(row, "sign p")),
                   Metric("shuffled mean rho", t.cell(row, "shuffled mean rho"))]
    r.gates.append(("player gate", _first_body(d.need("Player table"))))


def decisions_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("prior", "mean rho", "lo", "hi")
    row = t.one("prior", lambda c: c == "%s (held out)" % r.selected, "held-out row of the selected prior")
    return HeldOut(HO_SCORED, Metric("mean Spearman, decision quality vs on-court rating (by game)", t.cell(row, "mean rho"),
                                     _lohi(t, row)))


def playcalling_train(d: Doc, r: LaneResult):
    r.selected = _one_selected_candidate(d)
    if r.selected is None:
        # fallback: older writers name the selected model only in the held-out section
        m = re.search(r"the selected model \(the (\w+)\)", d.text) or re.search(r"^\| (\w+) \(selected\) \|", d.text, re.M)
        if not m:
            raise ParseError("the report states no selected model")
        r.selected = m.group(1)
    s = d.need("3. Ridge adjustment")
    _in_sample(r, s)
    t = s.table("model", "out-of-fold MSE", "lo", "hi")
    row = t.one("model", lambda c: c == r.selected, "selected model row")
    r.oof = Metric("possession points MSE", t.cell(row, "out-of-fold MSE"), _lohi(t, row))
    r.candidates = [(clean(x[0]), Metric("MSE", t.cell(x, "out-of-fold MSE"), _lohi(t, x)))
                    for x in t.rows]
    for raw in s.lines:
        if raw.startswith("- ") and " minus " in raw:
            pair, _, rest = clean(raw[2:]).partition(": ")
            r.gates.append(("paired bootstrap, %s" % pair, rest))
    rec = d.need("4. Recommendation table")
    rt = rec.table("coverage", "action_type", "adjusted_ppp", "recommended")
    for x in rt.where("recommended", lambda c: c == "yes"):
        r.gates.append(("recommended (association)", "%s: %s, adjusted PPP %s" % (rt.cell(x, "coverage"), rt.cell(x, "action_type"),
                                                                             rt.cell(x, "adjusted_ppp"))))
    for raw in rec.lines:
        if raw.startswith("No coverage has"):
            r.gates.append(("recommendation gate", clean(raw)))
    r.notes.append("association only, not causal (the report's title)")


def playcalling_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("model", "held-out MSE")
    row = t.one("model", lambda c: c == "%s (selected)" % r.selected, "held-out row of the selected model")
    return HeldOut(HO_SCORED, metric("possession points MSE", t.cell(row, "held-out MSE"), "point estimate"))


def lineups_train(d: Doc, r: LaneResult):
    """Selection is the highest mean validation Spearman over every validation pair (analytics/lineups/train.py
    select), so the primary metric is that mean, copied from the 'Mean validation Spearman' line (no interval: the
    report states none); each pair's table is listed on its own."""
    s = d.need("Candidate table: out-of-season validation")
    m = s.search(r"^Mean validation Spearman: (.+?)\. Selected: \*\*([^*]+)\*\*\.$", "the Selected line")
    r.selected = m.group(2)
    means = {}
    for part in m.group(1).split(", "):
        name, _, val = part.rpartition(" ")
        if not name or not val:
            raise ParseError("cannot read %r in the Mean validation Spearman line" % part)
        means[name] = val
    if r.selected not in means:
        raise ParseError("the selected candidate %s has no mean validation Spearman" % r.selected)
    r.oof = Metric("mean out-of-season Spearman over validation pairs", means[r.selected])
    one = [clean(x) for x in s.lines if clean(x).startswith("One training season only")]
    if one:
        r.oof.note = one[0].rstrip(".")
    r.candidates = [(n, Metric("mean Spearman", v)) for n, v in means.items()]
    fits = [clean(x) for x in s.lines if clean(x).startswith("Fit on ")]
    tables = [t for t in s.tables() if {"candidate", "Spearman", "MSE"} <= set(t.header)]
    if len(fits) != len(tables):
        raise ParseError("%d 'Fit on' lines but %d pair tables" % (len(fits), len(tables)))
    if not tables and not any(clean(x).startswith("One training season only") for x in s.lines):
        raise ParseError("no validation pair table and no 'One training season only' line")
    r.notes.append("%d validation pair(s); the selection metric is the mean over them" % len(tables))
    pcol_name = "Spearman minus plus_minus (paired)"
    for fit, t in zip(fits, tables):
        pair = "fit " + fit.split(":")[0][len("Fit on "):]      # "fit 2020-21, predict 2021-22"
        if sorted(clean(x[0]) for x in t.rows) != sorted(means):
            raise ParseError("%s: candidates %s differ from the mean line's %s" % (pair, [clean(x[0]) for x in t.rows], sorted(means)))
        row = t.one("candidate", lambda c: c == r.selected, "selected candidate row, %s" % pair)
        r.oof_extra.append(metric("%s: Spearman" % pair, t.cell(row, "Spearman")))
        r.oof_extra.append(metric("%s: MSE" % pair, t.cell(row, "MSE")))
        if pcol_name in t.header:
            for x in t.rows:
                if t.cell(x, pcol_name) != "baseline":
                    r.gates.append(("paired Spearman, %s minus plus_minus (%s)" % (clean(x[0]), pair), t.cell(x, pcol_name)))
    r.gates.append(("selection", "Mean validation Spearman: %s" % m.group(1)))


def lineups_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    t = s.table("candidate", "Spearman", "MSE")
    row = t.one("candidate", lambda c: c == "%s (selected)" % r.selected, "held-out row of the selected candidate")
    return HeldOut(HO_SCORED, metric("Spearman", t.cell(row, "Spearman")), [metric("MSE", t.cell(row, "MSE"))])


def winprob_train(d: Doc, r: LaneResult):
    s = d.need("Candidates by time bucket")
    r.selected = s.search(r"^Selected: \*\*([^*]+)\*\*", "the Selected line").group(1)
    t = s.table("model", "selection (mean of 5)", "Brier overall", "log loss overall")
    row = _sel_row(t, "model", r.selected, "selected model row")
    r.oof = metric("mean bucket Brier (the selection metric)", t.cell(row, "selection (mean of 5)"))
    r.oof_extra = [metric("Brier overall", t.cell(row, "Brier overall")), metric("log loss overall", t.cell(row, "log loss overall"))]
    r.candidates = _candidates(t, "model", "selection (mean of 5)", "mean bucket Brier")
    m = d.need("Protocol").search(r"`prefix_invariance_violations` found [^.]+\)", "the prefix invariance check")
    r.gates.append(("prefix invariance check", clean(m.group(0))))
    b = d.section("Benchmark")
    if b is not None:
        r.notes.append(_first_body(b))


def winprob_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    m = s.search(r"^Overall: log loss (\S+) (\[[^\]]*\]), Brier (\S+) (\[[^\]]*\]), mean bucket Brier ([^\s.]+(?:\.\d+)?)\.?$",
                 "the Overall line")
    return HeldOut(HO_SCORED, Metric("mean bucket Brier", m.group(5)),
                   [Metric("Brier overall", m.group(3), m.group(4)), Metric("log loss overall", m.group(1), m.group(2))])


def load_train(d: Doc, r: LaneResult):
    c2 = d.need("Candidate 2: ridge regression of y_decline")
    _in_sample(r, c2)
    t = c2.table("model", "MSE (ft per s)^2")
    c3 = d.need("Candidate 3: absence from the next game")
    no_absence = any(clean(x).startswith("No player-game has a known absent_next") for x in c3.lines)
    t3 = None if no_absence else c3.table("model", "log loss")
    names = selected_candidates(d)
    if names:
        reg = [n for n in names if n in [clean(x[0]) for x in t.rows]]
        ab = [n for n in names if t3 is not None and n in [clean(x[0]) for x in t3.rows]]
        if len(reg) != 1 or len(ab) != (0 if no_absence else 1) or len(names) != len(reg) + len(ab):
            raise ParseError("'Selected candidate:' lines %s do not map to one decline model%s" % (
                names, "" if no_absence else " and one absence model"))
        row = t.one("model", lambda c: clean(c) == reg[0], "selected decline model row")
        row3 = None if no_absence else t3.one("model", lambda c: clean(c) == ab[0], "selected absence model row")
        reg_name, ab_name = reg[0], (ab[0] if ab else None)
    else:
        # fallback: older writers name the selected model of each pair in the held-out section only (analytics/load/train.py)
        reg_name = d.search(r"MSE, selected model \((\w+)\)", "the selected decline model").group(1)
        row = t.one("model", lambda c: c.split(" ")[0] == reg_name, "selected decline model row")
        ab_name, row3 = None, None
        if not no_absence:
            ab_name = d.search(r"log loss, selected model \((\w+)\)", "the selected absence model").group(1)
            row3 = t3.one("model", lambda c: c.split(" ")[0] == ab_name, "selected absence model row")
    r.selected = "%s (decline)" % reg_name + ("; %s (absence)" % ab_name if ab_name else "")
    r.oof = metric("y_decline MSE (ft/s)^2", t.cell(row, "MSE (ft per s)^2"))
    r.candidates = _candidates(t, "model", "MSE (ft per s)^2", "MSE")
    r.gates.append(("paired bootstrap, decline", c2.line("Paired bootstrap")))
    if no_absence:
        r.gates.append(("absence target", c3.line("No player-game has a known absent_next")))
        return
    r.oof_extra = [metric("absent_next log loss", t3.cell(row3, "log loss"))]
    r.gates.append(("paired bootstrap, absence", _first_sentence(c3.line("Paired bootstrap"))))
    for raw in c3.lines:
        if clean(raw).startswith("One class only"):
            r.gates.append(("absence target", _first_sentence(clean(raw))))


def load_heldout(d: Doc, r: LaneResult):
    s = d.section(HELDOUT, level=2)
    if s is None:
        return None
    if any(clean(x).startswith("No held-out player-game was loaded") for x in s.lines):
        raise ParseError("the held-out section says no held-out player-game was loaded")
    t = s.table("quantity", "value")
    mse = t.one("quantity", lambda c: c.startswith("MSE, selected model"), "held-out decline MSE")
    extra = []
    for ll in t.where("quantity", lambda c: c.startswith("log loss, selected model")):
        if t.cell(ll, "value") not in ("n/a", "nan", ""):
            extra.append(metric("absent_next log loss", t.cell(ll, "value"), "point estimate"))
    return HeldOut(HO_SCORED, metric("y_decline MSE (ft/s)^2", t.cell(mse, "value"), "point estimate"), extra)


def betting_train(d: Doc, r: LaneResult):
    s = d.need("Candidates (", level=2)
    _in_sample(r, s)
    m = s.search(r"^Selected: cover model \*\*([^*]+)\*\*, margin model \*\*([^*]+)\*\*\.$", "the Selected line")
    cover, margin = m.group(1), m.group(2)
    r.selected = "cover model %s, margin model %s" % (cover, margin)
    t = s.table("candidate", "cover log loss", "margin MSE", "ROI")
    row = t.one("candidate", lambda c: c == cover, "selected cover model row")
    mrow = t.one("candidate", lambda c: c == margin, "selected margin model row")
    r.oof = metric("cover log loss", t.cell(row, "cover log loss"))
    r.oof_extra = [metric("margin MSE", t.cell(mrow, "margin MSE")), Metric("bets", t.cell(row, "bets")),
                   metric("ROI", t.cell(row, "ROI"))]
    r.candidates = _candidates(t, "candidate", "cover log loss", "cover log loss")
    st = d.section("Status")
    if st is not None:
        for raw in st.lines:
            if clean(raw).startswith("Standing caveat"):
                r.notes.append(clean(raw))


def betting_heldout(d: Doc, r: LaneResult):
    s = d.section("Profit and loss per season")    # betting writes its held-out season as a row, not a section
    if s is None:
        return None
    t = s.table("season", "mode", "log loss", "margin MSE", "ROI")
    rows = t.where("mode", lambda c: c == "heldout")
    if not rows:
        return None
    if len(rows) != 1:
        raise ParseError("%d held-out season rows, expected 1" % len(rows))
    row = rows[0]
    return HeldOut(HO_SCORED, metric("cover log loss", t.cell(row, "log loss"), "point estimate, season %s" % t.cell(row, "season")),
                   [metric("margin MSE", t.cell(row, "margin MSE")), Metric("bets", t.cell(row, "bets")),
                    metric("ROI", t.cell(row, "ROI"))])


LANES = {
    "xfg": (xfg_train, xfg_heldout),
    "rebounds": (rebounds_train, rebounds_heldout),
    "matchups": (matchups_train, matchups_heldout),
    "gravity": (gravity_train, gravity_heldout),
    "ghosting": (ghosting_train, ghosting_heldout),
    "embeddings": (embeddings_train, embeddings_heldout),
    "epv": (epv_train, epv_heldout),
    "actions": (actions_train, actions_heldout),
    "coverage": (coverage_train, coverage_heldout),
    "decisions": (decisions_train, decisions_heldout),
    "playcalling": (playcalling_train, playcalling_heldout),
    "lineups": (lineups_train, lineups_heldout),
    "winprob": (winprob_train, winprob_heldout),
    "load": (load_train, load_heldout),
    "betting": (betting_train, betting_heldout),
    "xfg_baseline": (baseline_train, baseline_heldout),
}
OPTIONAL = {"xfg_baseline"}          # summarised only when the report exists
NO_HELDOUT = {"embeddings"}          # lanes that score no held-out rows: a missing held-out section is expected


# ── one report -> one record ───────────────────────────────────────────────────────────────────────
def parse_report(lane: str, text: str, path: str = "") -> LaneResult:
    rec = LaneResult(lane, path or "reports/%s.md" % lane)
    d = Doc(text)
    m = _WRITTEN_RE.search(text)
    if m:
        rec.written_at, rec.command = m.group(1), m.group(2)
    else:
        rec.problems.append(("report", "no 'Written <time> by `<command>`' line"))
    low = text.lower()
    fixture = FIXTURE_PHRASE in low
    skip = SKIP_PHRASE in text
    stage5 = STAGE5_PHRASE in low
    smoke = SMOKE_PHRASE in text and lane == "xfg_baseline"
    train_fn, held_fn = LANES[lane]
    try:
        train_fn(d, rec)
    except ParseError as e:
        rec.problems.append(("training", str(e)))
    if rec.in_sample and rec.oof is not None:
        for m in [rec.oof] + rec.oof_extra + [c for _, c in rec.candidates]:
            m.name += " (in-sample)"
        rec.gates.insert(0, ("metrics in-sample", rec.in_sample))
    try:
        ho = held_fn(d, rec)
    except ParseError as e:
        ho, held_err = None, str(e)
    else:
        held_err = None
    if skip or stage5:
        if ho is not None and ho.state == HO_SCORED:
            rec.problems.append(("heldout", "the report says held-out games were not scored but states held-out numbers"))
        rec.heldout = HeldOut(HO_SKIPPED if skip else HO_NOT_SCORED_BANNER)
    elif held_err is not None:
        rec.problems.append(("heldout", held_err))
        rec.heldout = HeldOut("unparsed")
    elif ho is None:
        rec.heldout = HeldOut(HO_ABSENT)
        if lane not in NO_HELDOUT:
            # a lane that scores held-out rows must either state them or say why not
            rec.problems.append(("heldout", "no held-out section or row, and neither the --skip-heldout phrase nor the "
                                            "stage 5 banner"))
    else:
        rec.heldout = ho
    if rec.heldout.state == HO_SCORED and (rec.oof is None or any(f == "training" for f, _ in rec.problems)):
        # never show a held-out number whose training-fold counterpart could not be read
        rec.problems.append(("heldout", "held-out numbers without training-fold numbers"))
        if rec.oof is None and not any(f == "training" for f, _ in rec.problems):
            rec.problems.append(("training", "no training-fold numbers"))
    if fixture:
        rec.status = FIXTURE
    elif smoke:
        rec.status = SMOKE
    elif skip or stage5:
        rec.status = REAL_TRAINING_ONLY
    elif rec.heldout.state == HO_SCORED:
        rec.status = REAL_HELDOUT
    else:
        rec.status = REAL
    return rec


def collect(reports: Path) -> list:
    out = []
    for lane in list(EXTENSION_NAMES) + sorted(OPTIONAL):
        p = reports / ("%s.md" % lane)
        shown = shown_path(p)
        if lane not in LANES:            # a lane added to analytics.registry without extraction rules here
            r = LaneResult(lane, shown, status=MISSING if not p.exists() else "no extraction rule", heldout=HeldOut(HO_ABSENT))
            r.problems.append(("report", "analytics/results.py has no extraction rule for lane %s" % lane))
            out.append(r)
            continue
        if not p.exists():
            if lane in OPTIONAL:
                continue
            r = LaneResult(lane, shown, status=MISSING, heldout=HeldOut(HO_ABSENT))
            r.problems.append(("report", "%s does not exist" % shown))
            out.append(r)
            continue
        out.append(parse_report(lane, p.read_text(), shown))
    return out


# ── rendering ──────────────────────────────────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _heldout_cell(r: LaneResult) -> str:
    bad = r.problem("heldout")
    if bad:
        return bad
    h = r.heldout
    if h.state == HO_SCORED:
        return h.metric.show(primary=True)
    if h.state == HO_NOT_RUN and h.note:
        return "%s (%s)" % (h.state, h.note)
    return h.state


def _training_cell(r: LaneResult, value: str) -> str:
    return r.problem("training") or value


# The overview shows each gate as its label plus the deciding words of the report's own sentence (the full sentence is
# in the lane's table). (label prefix, pattern whose first non-empty group is those words.)
SHORT = [
    ("ECE under 0.02", r"selected candidate: (\w+)"),
    ("paired bootstrap", r"(excludes (?:zero|0):? (?:True|False|yes|no))"),
    ("paired Spearman", r"\((excludes 0)\)"),
    ("defender gate", r"(\d+ pass the \S+ gate)"),
    ("player gate", r"(\d+ gated of \d+)|(nobody is ranked)"),
    ("top-decile check", r"(passed: \w+)"),
    ("sign test", r"(p = [\d.]*\d|p = n/a)"),
    ("validation sign test", r"(p = [\d.]*\d|p = n/a)"),
    ("deep candidate", r"is (kept|dropped)\b"),
    ("autoencoder gate", r"(the autoencoder is not trained|PCA stands|does not beat it|beats it)"),
    ("sequence model gate", r"was (TRAINED|NOT trained)"),
    ("sequence model kept or dropped", r"is (KEPT|DROPPED)"),
    ("target", r"(beat majority \(\w+\))"),
    ("recommendation gate", r"(no recommendation is made)"),
    ("prefix invariance check", r"(\d+ violation\(s\))"),
    ("absence target", r"(One class only)"),
    ("metrics in-sample", r"(one training game)"),
    ("plan target", r"(a fixture cannot meet or miss it)"),
]


def gate_short(label: str, text: str) -> str:
    for prefix, pat in SHORT:
        if label.startswith(prefix):
            m = re.search(pat, text)
            if m:
                return next(g for g in m.groups() if g)
            return ""
    return ""


def _gate_labels(r: LaneResult) -> str:
    seen = []
    for label, text in r.gates:
        short = gate_short(label, text)
        item = "%s: %s" % (label, short) if short else label
        if item not in seen:
            seen.append(item)
    return "; ".join(seen) if seen else "none stated"


def _status_cell(r: LaneResult) -> str:
    return r.status + ("; unparsed" if r.unparsed else "")


def render_block(results: list, generated: str, command: str) -> str:
    n_fx = sum(r.status in FIXTURE_LIKE for r in results)
    n_bad = sum(r.unparsed for r in results)
    L = [BEGIN, ""]
    if n_fx:
        L += ["**%s.** %d of %d lane reports below are fixture output (synthetic analytics.fixtures runs, or a smoke run "
              "on fixture pages); every number copied from them is fixture output, not a result." % (FIXTURE.capitalize(), n_fx, len(results)), ""]
    L += ["Generated %s by `%s` (analytics/results.py; docs/contract/results.md). Every value is copied from the report "
          "named in its row; nothing here is computed. Regeneration rewrites only the text between the results markers."
          % (generated, command), ""]
    if n_bad:
        L += ["%d lane report(s) could not be parsed cleanly; their fields say unparsed with the reason." % n_bad, ""]
    L += ["### Sources", "", "| lane | report | status | written | command |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        L.append("| %s | %s | %s | %s | %s |" % (r.lane, r.path, _esc(_status_cell(r)), r.written_at or "not stated",
                                              "`%s`" % _esc(r.command) if r.command else "not stated"))
    L += ["", "### Overview", "",
          "| lane | status | selected model | out-of-fold metric [interval] | held-out | gates and checks fired |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            r.lane, _esc(_status_cell(r)), _esc(_training_cell(r, r.selected)),
            _esc(_training_cell(r, r.oof.show(primary=True) if r.oof else "")), _esc(_heldout_cell(r)),
            _esc(_training_cell(r, _gate_labels(r)))))
    for r in results:
        tag = " (%s)" % FIXTURE if r.status in FIXTURE_LIKE else ""
        L += ["", "### %s%s" % (r.lane, tag), "", "| field | value |", "| --- | --- |",
              "| status | %s |" % _esc(_status_cell(r)),
              "| report | %s, written %s |" % (r.path, r.written_at or "not stated"),
              "| selected model | %s |" % _esc(_training_cell(r, r.selected))]
        L.append("| out-of-fold metric | %s |" % _esc(_training_cell(r, r.oof.show(primary=True) if r.oof else "")))
        if not r.problem("training"):
            for m in r.oof_extra:
                L.append("| out-of-fold, selected | %s |" % _esc(m.show()))
            if r.candidates:
                L.append("| candidates, out-of-fold | %s |" % _esc("; ".join("%s: %s" % (n, m.show()) for n, m in r.candidates)))
        L.append("| held-out | %s |" % _esc(_heldout_cell(r)))
        if r.heldout and r.heldout.state == HO_SCORED and not r.problem("heldout"):
            for m in r.heldout.extra:
                L.append("| held-out, selected | %s |" % _esc(m.show()))
        if not r.problem("training"):
            for label, text in r.gates:
                L.append("| %s | %s |" % (_esc(label), _esc(text)))
            if not r.gates:
                L.append("| gates and checks | none stated |")
        for n in r.notes:
            L.append("| note | %s |" % _esc(n))
        for f, why in r.problems:
            L.append("| unparsed (%s) | %s |" % (f, _esc(why)))
    L += ["", END]
    return "\n".join(L)


_GEN_RE = re.compile(r"^Generated (\S+) by `", re.M)


def _split_existing(text: str):
    """-> (before, block, after) around the markers, or None when the file has no markers. Raises on a broken pair."""
    nb, ne = text.count(BEGIN), text.count(END)
    if nb == 0 and ne == 0:
        return None
    if nb != 1 or ne != 1 or text.index(BEGIN) > text.index(END):
        raise SystemExit("analytics.results: %s and %s must appear once each, in that order; fix the file by hand" % (BEGIN, END))
    i, j = text.index(BEGIN), text.index(END) + len(END)
    return text[:i], text[i:j], text[j:]


DEFAULT_HEAD = "# Results\n\n"
DEFAULT_TAIL = ("\n\n<!-- Hand-written sections (stage 9 of docs/mac_runbook.md) go below this line. analytics.results rewrites "
                "only the text between the results markers. -->\n")


def _existing_generated(existing):
    """The generated-at time of the existing block, or None."""
    if existing is None:
        return None
    parts = _split_existing(existing)
    m = _GEN_RE.search(parts[1]) if parts else None
    return m.group(1) if m else None


def compose(existing, block: str) -> str:
    """The full file: the existing text with its marked block replaced, or a new file around the block."""
    if existing is None:
        return DEFAULT_HEAD + block + DEFAULT_TAIL
    parts = _split_existing(existing)
    if parts is None:
        lines = existing.split("\n", 1)
        if lines[0].startswith("# "):
            return lines[0] + "\n\n" + block + "\n" + ("\n" + lines[1].lstrip("\n") if len(lines) > 1 and lines[1].strip() else "\n")
        return block + "\n\n" + existing
    before, _, after = parts
    return before + block + after


def shown_path(p: Path) -> str:
    """p relative to the working directory when it lies under it, else absolute: the same file always prints the same
    way however --reports and --out were spelled."""
    full = Path(p).resolve()
    try:
        return full.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return full.as_posix()


def canonical_command(reports: Path, out: Path) -> str:
    return "python3 -m analytics.results --reports %s --out %s" % (shown_path(reports), shown_path(out))


def build(reports: Path, out: Path, existing, generated: str = None):
    """-> (results, full file text). With generated None the time is now, or the existing block's time for --check."""
    results = collect(reports)
    if generated is None:
        generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = render_block(results, generated, canonical_command(reports, out))
    return results, compose(existing, block)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", default="reports", help="folder of the lane reports (default reports)")
    ap.add_argument("--out", default="reports/results.md", help="the summary file (default reports/results.md)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any lane field is unparsed or a report is missing")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 when the file on disk is stale")
    a = ap.parse_args(argv)
    reports, out = Path(a.reports), Path(a.out)
    existing = out.read_text() if out.exists() else None
    if a.check:
        results, text = build(reports, out, existing, generated=_existing_generated(existing) or "unknown")
        stale = existing is None or text != existing
        print("%s: %s" % (out, "stale (rerun python3 -m analytics.results)" if stale else "up to date"))
        rc = 1 if stale else 0
    else:
        old = _existing_generated(existing)
        results, text = build(reports, out, existing, generated=old) if old else (None, None)
        if text is not None and text == existing:
            print("%s: unchanged (%d lanes)" % (out, len(results)))
        else:
            results, text = build(reports, out, existing)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
            print("wrote %s (%d lanes)" % (out, len(results)))
        rc = 0
    for r in results:
        if r.unparsed:
            print("unparsed %s: %s" % (r.lane, "; ".join("%s: %s" % p for p in r.problems)), file=sys.stderr)
    if a.strict and any(r.unparsed for r in results):
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
