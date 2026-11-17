"""export/genai.py: machine-written texts, made when the site data is built (PLAN.md P3, D76).

    python3 -m export.genai --root <data root> [--players N] [--possessions N | --all] [--live] [--cache <dir>]

Two kinds of text, one short paragraph each:
- scouting: a scouting report on one player of the prep tab, from his entry in players.json;
- possession: an account in plain words of one possession of the review, from its row in the game's possession
  index and the per-possession values of its file (result, points, the EPV curve, the shots with their xFG). The
  input is built from the possession after export.public_data.strip_possession, so it holds only what the public
  data root keeps (PLAN.md D60) and the same text is true on both variants of the site.

The input of an item is a list of facts: a field of the data root and its value exactly as the page shows it, with
the rounding of web/lib/format.ts (the functions below mirror it; web/lib/genai.ts holds the same facts and checks
each text again with the page's own formatter). The model gets the facts and a versioned prompt (export/prompts).
A text is kept only when it passes every check of `check_text`:
- numbers: every number in the text is the shown value of a fact of that item, in that fact's unit (a percent is
  never written as a share or a share as a percent, feet are never points), and no number is written as a word;
- estimates: a value with an interval is given with its interval, and a text that gives a point estimate says so;
- status: a value from a model whose lane did not work (failed, no signal, inconclusive, no result) says so;
- words: reading grade 9 or lower for the text and for each sentence, none of the words kept out of site copy, no
  claim the site never makes, no em or en dash, one paragraph of plain text.
A text that fails is dropped and its page shows nothing. Nothing here fits, tunes, scores or computes a value: the
model writes words around values the data root already holds.

Texts go to genai/scouting.json and genai/possession.json under the data root (contract model GeneratedTexts) and
the manifest is written again. export.schemas.validate_tree runs `root_problems` on every text, so a stale text
(its input changed) or a wrong number cannot pass validation.

The cache (--cache, else the GENAI_CACHE_DIR variable, else .genai-cache in the repository, which git ignores)
holds one file per call, keyed by model, prompt version, prompt hash and input hash: a rebuild calls the model only
for an item whose input or prompt changed. A text that failed its checks is cached too, with its problems, so it is
not paid for twice. The cache holds text written from the data root: it is never committed.

Without --live, fixture mode writes deterministic texts from the facts, with model "fixture". It runs only on a
synthetic data root (export.sample): fixture output is never shown as a result. --live calls the model through the
anthropic package (a venv, never the system Python; CLAUDE.md) with the key in ANTHROPIC_API_KEY, which is never
printed. A live run on many items needs a person's yes first (D76): it spends money and sends derived numbers to
the model provider. The run prints the tokens it used and what the same run would cost on every item.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from export import schemas as S
from export.public_data import strip_possession

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSIONS = PROMPT_DIR / "versions.json"
KINDS = ("scouting", "possession")
# A small current model (D76): the task is short writing from given facts.
MODEL = "claude-haiku-5-5"
# Prices in dollars per million tokens for MODEL, from the Claude API model table (2026-11-03). Used only to report
# what a run cost and what the full run would cost.
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}
# How the model is called, every request parameter but the messages. The setting is part of the cache key and of
# each cache record, and the genai file's notes say which setting wrote how many texts (call_notes).
# History: the first runs used the model's default (adaptive thinking) at low effort with 4,000 tokens
# (LEGACY_CALL). The long scouting-v3 prompt then spent the whole 4,000 on thinking for 14 of 50 players, and no
# text came back. Thinking off fixed that but broke the rules far more often (21 of 50 dropped). This model takes
# no fixed thinking budget (budget_tokens is refused), so thinking stays adaptive at low effort and max_tokens is
# 8,000: the earlier thinking-on calls averaged about 2,900 output tokens, so a text keeps well over 1,000 tokens
# after the thinking.
CALL = {"max_tokens": 8000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
LEGACY_CALL = {"max_tokens": 4000, "output_config": {"effort": "low"}}
MAX_TOKENS = CALL["max_tokens"]
MAX_ATTEMPTS = 2
FIXTURE_MODEL = "fixture"
GENERATOR = "export/genai.py"
CHECKS = (
    "Every number is a value of this page as the page rounds it, in the same unit; no number is written as a word.",
    "A value with an interval is given with its interval; a point estimate is called one.",
    "A value from a model that did not work says so; a model's value is called one in its sentence; no lane that did "
    "not work is named.",
    "Reading grade 9 or lower for the text and each sentence; plain words; no claim the site never makes.",
)

# The lane behind each player metric and each possession value, for its status (results.json).
METRIC_LANES = {"shot_making": "xfg", "gravity": "gravity", "rebounding": "rebounds", "absence_risk": "load",
                "similar": "embeddings", "xfg": "xfg", "epv": "epv"}
# What a text must say when a lane's verdict is not "works". "no_result" is a lane missing from results.json.
STATUS_WORDS = {"failed": "failed", "no_signal": "no signal", "inconclusive": "inconclusive",
                "no_result": "no result"}
RESULT_WORDS = {"shot_made": "made shot", "shot_missed": "missed shot", "free_throws": "free throws",
                "turnover": "turnover", "period_end": "end of the period", "other": "other end"}


class GenaiProblem(Exception):
    """The input, the cache or the call is not what the command needs."""


# ------------------------------------------------------------------------------- numbers as the page shows them
# Mirrors of web/lib/format.ts and of showValue(..., "as stored") in web/lib/tokens.ts. JavaScript's toFixed rounds
# the exact binary value half up, which Decimal(float) with ROUND_HALF_UP reproduces.

MINUS = "−"


def _fixed(value: float, digits: int) -> str:
    text = str(Decimal(abs(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    if digits == 0:
        text = text.split(".")[0]
    is_zero = Decimal(text) == 0
    return f"{MINUS}{text}" if value < 0 and not is_zero else text


def format_number(value, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return _fixed(float(value), digits)


def format_count(value) -> str:
    if value is None or not math.isfinite(value):
        return ""
    whole = int(Decimal(abs(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    text = f"{whole:,}"
    # formatCount writes the sign when Math.round(value) is not zero, and Math.round(-0.5) is zero.
    return f"{MINUS}{text}" if value < -0.5 else text


def format_signed(value, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = _fixed(float(value), digits)
    return f"+{text}" if value > 0 and Decimal(text) != 0 else text


def format_percent(share, digits: int = 0) -> str:
    if share is None or not math.isfinite(share):
        return ""
    return f"{_fixed(float(share) * 100, digits)}%"


def format_probability(value) -> str:
    return format_number(value, 3)


def format_clock(seconds) -> str:
    if seconds is None or not math.isfinite(seconds):
        return ""
    whole = max(0, math.floor(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def format_seconds(seconds) -> str:
    return format_number(seconds, 1)


def format_stored(value: float) -> str:
    """The digits the file stores, as JavaScript's String(number) writes them, with the true minus sign."""
    absolute = Decimal(repr(abs(float(value))))
    text = str(int(absolute)) if absolute == absolute.to_integral_value() else format(absolute, "f")
    return f"{MINUS}{text}" if value < 0 else text


def format_integer(value: int) -> str:
    """An integer placed in copy as it is ({summary.points}, {neighbour.rank}): JavaScript's String(n)."""
    return f"{MINUS}{abs(int(value))}" if value < 0 else str(int(value))


# ---------------------------------------------------------------------------------------------------- facts


@dataclass(frozen=True)
class Fact:
    """A number of the page. `shown` is the text the page prints. `unit` is percent, feet, points, seconds, clock or
    none. `kind` is value (an estimate with an interval), lo or hi (its bounds), point (a point estimate) or fact (a
    count, a rank, a period, an interval level). `group` names the estimate a value, lo or hi belongs to, and
    `metric` the view whose model status applies to it."""

    field: str
    shown: str
    unit: str
    kind: str
    group: str = ""
    metric: str = ""
    about: str = ""


def _estimate_facts(field: str, estimate, digits: int, unit: str, metric: str, about: str) -> list:
    if estimate is None:
        return []
    if estimate.point_estimate:
        facts = [Fact(field, format_number(estimate.value, digits), unit, "point", field, metric,
                      f"{about} (a point estimate)")]
    else:
        facts = [Fact(f"{field}.value", format_number(estimate.value, digits), unit, "value", field, metric, about),
                 Fact(f"{field}.lo", format_number(estimate.lo, digits), unit, "lo", field, metric,
                      f"low end of the range of {about}"),
                 Fact(f"{field}.hi", format_number(estimate.hi, digits), unit, "hi", field, metric,
                      f"high end of the range of {about}")]
        level = re.match(r"^(\d+)% ", estimate.interval or "")
        if level:
            facts.append(Fact(f"{field}.interval", f"{level.group(1)}%", "percent", "fact", "", metric,
                              "the level of that range"))
    if estimate.n is not None:
        facts.append(Fact(f"{field}.n", format_count(estimate.n), "none", "fact", "", metric,
                          f"sample of that value, in {estimate.n_unit or 'items'}"))
    return facts


def _lane_status(results, lane: str) -> str:
    if results is None:
        return "no_result"
    for row in results.lanes:
        if row.lane == lane:
            return row.verdict_class
    return "no_result"


def player_input(player, players, results) -> dict:
    """The input of a scouting text: the numbers the player page shows for `player`, as it shows them."""
    facts = [Fact("games", format_count(player.games), "none", "fact", about="tracked games he played")]
    metrics, names = {}, [player.name, *player.teams]
    shots = player.shot_making
    if shots.shown and shots.over_expected_per_100 is not None:
        facts += _estimate_facts("shot_making.over_expected_per_100", shots.over_expected_per_100, 1, "none",
                                 "shot_making", "made shots above the expected rate, per 100 attempts")
        if shots.baseline_error_per_100 is not None:
            facts.append(Fact("shot_making.baseline_error_per_100", format_signed(shots.baseline_error_per_100, 1),
                              "none", "point", "shot_making.baseline_error_per_100", "shot_making",
                              "the part of the value above that the expected-rate model's own error explains, "
                              "per 100 attempts (a point estimate)"))
        if shots.fg_pct is not None:
            facts.append(Fact("shot_making.fg_pct", format_percent(shots.fg_pct, 1), "percent", "point",
                              "shot_making.fg_pct", "shot_making",
                              "share of his attempts that went in (descriptive, a point estimate)"))
        if shots.expected_fg_pct is not None:
            facts.append(Fact("shot_making.expected_fg_pct", format_percent(shots.expected_fg_pct, 1), "percent",
                              "point", "shot_making.expected_fg_pct", "shot_making",
                              "share the expected-rate model gave his attempts (a point estimate)"))
    facts += [Fact("shot_making.attempts", format_count(shots.attempts), "none", "fact", "", "shot_making",
                   "his attempts counted"),
              Fact("shot_making.min_attempts", format_count(shots.min_attempts), "none", "fact", "", "shot_making",
                   "attempts needed before the value is shown")]
    metrics["shot_making"] = shots.shown

    gravity = player.gravity
    if gravity.shown and gravity.gravity_ft is not None:
        facts += _estimate_facts("gravity.gravity_ft", gravity.gravity_ft, 2, "feet", "gravity",
                                 "feet the nearest defender stays closer to him than his spot on the floor explains "
                                 "(descriptive; a negative value means farther away)")
    facts += [Fact("gravity.possessions", format_count(gravity.possessions), "none", "fact", "", "gravity",
                   "his possessions counted"),
              Fact("gravity.min_possessions", format_count(gravity.min_possessions), "none", "fact", "", "gravity",
                   "possessions needed before the value is shown")]
    metrics["gravity"] = gravity.shown

    rebounding = player.rebounding
    if rebounding.shown and rebounding.rate_per_100 is not None:
        facts += _estimate_facts("rebounding.rate_per_100", rebounding.rate_per_100, 1, "none", "rebounding",
                                 "rebounds per 100 chances (descriptive)")
        if rebounding.model_expected_per_100 is not None:
            facts.append(Fact("rebounding.model_expected_per_100",
                              format_number(rebounding.model_expected_per_100, 1), "none", "point",
                              "rebounding.model_expected_per_100", "rebounding",
                              "rebounds per 100 chances the model expected; it already knows his rate in other "
                              "games, so the gap is not a skill number (a point estimate)"))
    facts += [Fact("rebounding.chances", format_count(rebounding.chances), "none", "fact", "", "rebounding",
                   "his rebound chances counted"),
              Fact("rebounding.min_chances", format_count(rebounding.min_chances), "none", "fact", "", "rebounding",
                   "chances needed before the value is shown")]
    metrics["rebounding"] = rebounding.shown

    absence = player.absence_risk
    if absence.shown:
        facts += _estimate_facts("absence_risk.model_probability", absence.model_probability, 3, "none",
                                 "absence_risk", "the model's chance he misses the team's next tracked game, not an "
                                 "injury forecast")
        facts += _estimate_facts("absence_risk.observed_rate", absence.observed_rate, 3, "none", "absence_risk",
                                 "share of his games after which he missed the next one (descriptive)")
        facts.append(Fact("absence_risk.absent", format_count(absence.absent), "none", "fact", "", "absence_risk",
                          "games after which he missed the next tracked game"))
    facts += [Fact("absence_risk.games", format_count(absence.games), "none", "fact", "", "absence_risk",
                   "his games counted for absence"),
              Fact("absence_risk.min_games", format_count(absence.min_games), "none", "fact", "", "absence_risk",
                   "games needed before the value is shown")]
    metrics["absence_risk"] = absence.shown

    similar = player.similar
    neighbours = []
    if similar.available and similar.shown:
        for row in similar.neighbours:
            names.append(row.name)
            neighbours.append(row.name)
            facts += [Fact(f"similar.neighbours.{row.rank}.rank", format_integer(row.rank), "none", "fact", "",
                           "similar", f"rank of {row.name} among the most similar players"),
                      Fact(f"similar.neighbours.{row.rank}.distance", format_number(row.distance, 3), "none",
                           "point", f"similar.neighbours.{row.rank}.distance", "similar",
                           f"style distance to {row.name}, smaller is closer (descriptive, a point estimate)")]
    facts += [Fact("similar.possessions", format_count(similar.possessions), "none", "fact", "", "similar",
                   "his possessions counted for style"),
              Fact("similar.min_possessions", format_count(similar.min_possessions), "none", "fact", "", "similar",
                   "possessions needed before similar players are shown")]
    metrics["similar"] = similar.shown and similar.available

    status = {metric: _lane_status(results, METRIC_LANES[metric]) for metric in metrics}
    context = {"name": player.name, "teams": list(player.teams), "position": player.position,
               "shown": metrics, "model_status": status, "similar_players": neighbours}
    return {"kind": "scouting", "key": player.slug, "context": context, "names": names,
            "facts": [asdict(fact) for fact in facts]}


def possession_input(summary, possession, results) -> dict:
    """The input of a possession text: the per-possession values the review page shows, as it shows them, from the
    possession without its frame data (the public root's copy)."""
    public = strip_possession(possession)
    key = f"{public.game_slug}/{public.possession}"
    facts = [Fact("possession", format_integer(summary.possession), "none", "fact", about="possession number"),
             Fact("period", format_integer(summary.period), "none", "fact", about="period"),
             Fact("start_clock", format_clock(summary.start_clock_s), "clock", "fact",
                  about="game clock at the start"),
             Fact("points", format_integer(summary.points), "points", "fact", about="points the offense scored"),
             Fact("duration_s", format_seconds(summary.duration_s), "seconds", "fact",
                  about="length of the possession")]
    if summary.xfg is not None:
        facts.append(Fact("xfg", format_probability(summary.xfg), "none", "point", "xfg", "xfg",
                          "chance the last shot goes in, as the model saw it (a point estimate)"))
    if summary.epv_start is not None:
        facts.append(Fact("epv_start", format_number(summary.epv_start, 2), "points", "point", "epv_start", "epv",
                          "expected points for the offense at the start (a point estimate)"))
    if summary.epv_end is not None:
        facts.append(Fact("epv_end", format_number(summary.epv_end, 2), "points", "point", "epv_end", "epv",
                          "expected points for the offense at the end (a point estimate)"))
    curve = public.epv
    if curve is not None and curve.value:
        n = min(len(curve.t), len(curve.value))
        top = max(range(n), key=lambda k: curve.value[k])
        low = min(range(n), key=lambda k: curve.value[k])
        for k in range(n):
            label = " (the highest)" if k == top else " (the lowest)" if k == low else ""
            facts += [Fact(f"epv.t.{k}", format_seconds(curve.t[k]), "seconds", "fact", "", "epv",
                           f"time of expected-points state {k + 1}{label}"),
                      Fact(f"epv.value.{k}", format_stored(curve.value[k]), "points", "point", f"epv.value.{k}",
                           "epv", f"expected points for the offense at that time{label} (a point estimate)")]
    shots, names = [], [public.offense, public.defense]
    for j, shot in enumerate(public.shots):
        if shot.shooter:
            names.append(shot.shooter)
        shots.append({"shooter": shot.shooter, "made": shot.made,
                      "kind": None if shot.three is None else "three-point" if shot.three else "two-point"})
        if shot.t is not None:
            facts.append(Fact(f"shots.{j}.t", format_seconds(shot.t), "seconds", "fact", "", "",
                              f"time of shot {j + 1} from the start of the possession"))
        if shot.xfg is not None:
            facts.append(Fact(f"shots.{j}.xfg", format_probability(shot.xfg), "none", "point", f"shots.{j}.xfg",
                              "xfg", f"chance shot {j + 1} goes in, as the model saw it (a point estimate)"))
    used = {"xfg": summary.xfg is not None or any(s.xfg is not None for s in public.shots),
            "epv": curve is not None and bool(curve.value)}
    status = {metric: _lane_status(results, METRIC_LANES[metric]) for metric, on in used.items() if on}
    context = {"offense": public.offense, "defense": public.defense,
               "result": RESULT_WORDS.get(summary.result, summary.result), "shots": shots,
               "clock_gap": summary.gap, "model_status": status}
    return {"kind": "possession", "key": key, "context": context, "names": sorted(set(names)),
            "facts": [asdict(fact) for fact in facts]}


def input_sha256(item: dict) -> str:
    return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _facts(item: dict) -> list:
    return [Fact(**fact) for fact in item["facts"]]


# --------------------------------------------------------------------------------------------- the checks

# A number in a text: a clock (11:20), or digits with thousands separators and decimals. Every digit of a text
# belongs to one of these, so no digit can escape the check.
_CLOCK_RE = re.compile(r"\d{1,2}:\d{2}(?!\d)")
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|\d+(?:\.\d+)?")
_PERCENT_AFTER = re.compile(r"\s*(?:%|percent\b|per cent\b)", re.I)
_UNIT_AFTER = (("feet", re.compile(r"\s*(?:feet|foot|ft)\b", re.I)),
               ("points", re.compile(r"\s*(?:points?|pts)\b", re.I)),
               ("seconds", re.compile(r"\s*(?:seconds?|secs?|s)\b", re.I)))
# "per 100 attempts" names a scale, not a value.
_SCALE_RE = re.compile(r"\bper\s+(?:100|hundred)\b", re.I)
# Numbers written as words, which would escape the digit check. "three-point" and "two-point" name a kind of shot.
_KIND_OF_SHOT = re.compile(r"\b(?:three|two)[- ]point(?:ers?)?\b", re.I)
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "hundreds", "thousand", "thousands", "million",
    "half", "halves", "twice", "double", "doubled", "triple", "tripled", "dozen", "quarter", "quarters", "once",
    "first", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
)
_NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.I)
# "second" is a unit after a number ("1 second") and an ordinal anywhere else.
_SECOND_RE = re.compile(r"(?<!\d )\bsecond\b", re.I)
_POINT_ESTIMATE_RE = re.compile(r"\bpoint estimates?\b", re.I)
# Values that are a model's output, not an observation. A text gives one only with a word that says so in the same
# sentence, so a model's probability can never read as an observed rate (the fact's group names the value).
MODEL_GROUP_RE = re.compile(r"^(?:absence_risk\.model_probability|shot_making\.(?:expected_fg_pct|baseline_error_per_100|"
                            r"over_expected_per_100)|rebounding\.model_expected_per_100|xfg|shots\.\d+\.xfg|epv_start|"
                            r"epv_end|epv\.value\.\d+)$")
MODEL_WORD_RE = re.compile(r"\b(?:models?|model's|expect(?:ed|s)?|predict(?:ed|s|ion|ions)?|estimat(?:e|ed|es))\b", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
# Lanes that are not shown on the prep tab or the review and did not work: failed (decisions), no signal
# (matchups), inconclusive (lineups), or no result yet (actions, coverage, playcalling, betting). A text names
# none of them, with or without a number (PLAN.md C10: they are shown as tested with no signal, on their own pages).
LANE_WORDS = {
    "decisions": r"decisions?",
    "matchups": r"match-?ups?",
    "lineups": r"line-?ups?",
    "actions": r"pass|passes|passed|passing|drives?|drove|driving|screens?",
    "coverage": r"coverages?",
    "playcalling": r"play ?calls?|play-?calling|playcalling",
    "betting": r"betting|bets?|odds",
}
_LANE_WORD_RE = {lane: re.compile(rf"\b(?:{words})\b", re.I) for lane, words in LANE_WORDS.items()}


def _sentence_span(text: str, index: int) -> tuple:
    """(start, stop) of the sentence of `text` that holds the character at `index`, split as sentence_count splits."""
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(text)]
    start = max((end for end in ends if end <= index), default=0)
    stop = min((end for end in ends if end > index), default=len(text))
    return start, stop


def _sentence_at(text: str, index: int) -> str:
    """The sentence of `text` that holds the character at `index`."""
    start, stop = _sentence_span(text, index)
    return text[start:stop]


# Words that say a value went up or down. A text that uses one must be right about the values it links: the value
# after the word against the one before it (export/genai.py check_directions, web/lib/genai.ts checkDirections).
UP_WORDS = r"rose|rise|rises|rising|climbed|climb|climbs|climbing|grew|grow|grows|growing|increased|increases|" \
           r"increase|jumped|jump|jumps|jumping|went up|goes up|go up|going up|moved up|ticked up"
DOWN_WORDS = r"fell|fall|falls|falling|dropped|drop|drops|dropping|dipped|dip|dips|dipping|sank|sink|sinks|" \
             r"sinking|went down|goes down|go down|going down|slid|slide|slides|sliding|declined|decline|declines|" \
             r"decreased|decrease|decreases|slipped|slip|slips|moved down|ticked down"
_CHANGE_RE = re.compile(rf"\b(?:(?P<up>{UP_WORDS})|(?P<down>{DOWN_WORDS}))\b", re.I)
_FROM_BEFORE_RE = re.compile(r"\bfrom\s*$", re.I)


def series_of(field: str, unit: str):
    """The run of values a fact belongs to, for the direction check: expected points, shot chances, or one
    estimate (its value and bounds). A time, a clock or a count belongs to none."""
    if unit in ("seconds", "clock"):
        return None
    if re.fullmatch(r"epv_start|epv_end|epv\.value\.\d+", field):
        return "epv"
    if re.fullmatch(r"xfg|shots\.\d+\.xfg", field):
        return "xfg"
    match = re.fullmatch(r"(.+)\.(?:value|lo|hi)", field)
    if match:
        return match.group(1)
    if field.endswith((".n", ".interval")) or re.search(r"(attempts|possessions|chances|games|absent|rank)$", field):
        return None
    return field


def _value_of(canon: str):
    """A token's value and its count of decimals."""
    plain = canonical(canon).rstrip("%")
    decimals = len(plain.split(".")[1]) if "." in plain else 0
    return Decimal(plain), decimals


# A value the page prints as the file stores it (an EPV curve state) is exact: "1.1" is 1.100, not a rounding.
_EXACT_FIELD_RE = re.compile(r"epv\.value\.\d+")


def same_at_shown_rounding(a: str, b: str, a_exact: bool = False, b_exact: bool = False) -> bool:
    """Whether two shown values can be one value: a rounded value matches an exact one when the exact one, rounded
    to its decimals, gives it; two rounded values match when they agree at the coarser rounding; two exact values
    match only when equal."""
    (x, dx), (y, dy) = _value_of(a), _value_of(b)
    if a_exact and b_exact:
        return x == y
    if a_exact or b_exact:
        exact, rounded, digits = (x, y, dy) if a_exact else (y, x, dx)
        return exact.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP) == rounded
    step = Decimal(1).scaleb(-min(dx, dy))
    return x.quantize(step, rounding=ROUND_HALF_UP) == y.quantize(step, rounding=ROUND_HALF_UP)


def check_directions(text: str, facts, names=()) -> list:
    """A word of change (rose, fell, dipped, ...) must agree with the values it links. The value it moves to is the
    first number after it in its sentence that belongs to a run of values (series_of); the value it moves from is
    the number after "from" when the sentence says "from A to B", else the nearest earlier number of the same run,
    in this sentence or an earlier one. Two values that can be one value (same_at_shown_rounding) are no change, so
    any word of change on them fails. A word with no such pair (the shot fell short) is not checked."""
    masked = _mask_names(text, names)
    tokens = numbers_in(text, names)

    def series(token):
        found = {series_of(f.field, f.unit) for f in facts
                 if canonical(f.shown) == token.canon and _unit_fits(token.unit, f.unit)}
        return found.pop() if len(found) == 1 else None

    def exact(token) -> bool:
        fields = [f.field for f in facts if canonical(f.shown) == token.canon and _unit_fits(token.unit, f.unit)]
        return bool(fields) and all(_EXACT_FIELD_RE.fullmatch(field) for field in fields)

    runs = [(token, series(token)) for token in tokens]
    problems = []
    for word in _CHANGE_RE.finditer(masked):
        end = _sentence_span(masked, word.start())[1]
        after = [(t, r) for t, r in runs if word.end() <= t.start < end and r]
        if not after:
            continue
        target, run = after[0]
        source = None
        if _FROM_BEFORE_RE.search(masked[word.end():target.start]):
            later = [(t, r) for t, r in after[1:] if r == run]
            if not later:
                continue
            source, target = target, later[0][0]
        else:
            before = [t for t, r in runs if t.start < word.start() and r == run]
            source = before[-1] if before else None
        if source is None:
            continue
        said = "up" if word.group("up") else "down"
        if same_at_shown_rounding(source.canon, target.canon, exact(source), exact(target)):
            problems.append(f"{word.group(0)!r} links {source.raw!r} and {target.raw!r}, which are one value at two "
                            "roundings: no change")
            continue
        went = "up" if _value_of(target.canon)[0] > _value_of(source.canon)[0] else "down"
        if went != said:
            problems.append(f"{word.group(0)!r} says {said}, but {source.raw!r} to {target.raw!r} went {went}")
    return problems


def check_lanes(text: str, names=()) -> list:
    """A text never names a lane that did not work and is not on its page (LANE_WORDS)."""
    words = _mask_names(text, names)
    return [f"names the {lane} lane ({match.group(0)!r}), which did not work and is not on this page"
            for lane, pattern in _LANE_WORD_RE.items() for match in pattern.finditer(words)]


@dataclass(frozen=True)
class Token:
    """A number of a text: `canon` is its sign and digits (thousands separators dropped, a percent sign kept),
    `unit` what the words around it make it (percent, feet, points, seconds, clock or none)."""

    raw: str
    canon: str
    unit: str
    start: int


def canonical(shown: str) -> str:
    """A shown value as a comparable string: the minus sign as "-", no plus sign, no thousands separator."""
    text = shown.replace(MINUS, "-").replace(",", "")
    return text[1:] if text.startswith("+") else text


def _mask_names(text: str, names) -> str:
    """The text with each name blanked out, longest first: a digit inside a name is not a number of the text."""
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        text = re.sub(re.escape(name), lambda m: " " * len(m.group(0)), text)
    return text


def numbers_in(text: str, names=()) -> list:
    """Every number of a text, in order."""
    plain = _SCALE_RE.sub(lambda m: " " * len(m.group(0)), _mask_names(text, names))
    tokens, taken = [], [False] * len(plain)
    for match in _CLOCK_RE.finditer(plain):
        tokens.append(Token(match.group(0), match.group(0), "clock", match.start()))
        for i in range(match.start(), match.end()):
            taken[i] = True
    for match in _NUMBER_RE.finditer(plain):
        if taken[match.start()]:
            continue
        start, raw, sign = match.start(), match.group(0), ""
        before = plain[start - 1] if start >= 1 else ""
        if before in ("-", MINUS, "+"):
            # A sign, unless the mark joins two numbers ("11-17"). Read as a minus, a number can only match a
            # negative value, so a doubtful mark never lets a sign error through.
            edge = plain[start - 2] if start >= 2 else " "
            if not edge.isdigit():
                sign = "-" if before in ("-", MINUS) else ""
                raw = before + raw
        canon = sign + raw.lstrip("+-" + MINUS).replace(",", "")
        rest = plain[match.end():]
        unit = "none"
        percent = _PERCENT_AFTER.match(rest)
        if percent:
            unit, canon, raw = "percent", canon + "%", raw + percent.group(0)
        else:
            for name, pattern in _UNIT_AFTER:
                if pattern.match(rest):
                    unit = name
                    break
        tokens.append(Token(raw, canon, unit, start))
    return sorted(tokens, key=lambda token: token.start)


def _unit_fits(token_unit: str, fact_unit: str) -> bool:
    if token_unit in ("percent", "clock") or fact_unit in ("percent", "clock"):
        return token_unit == fact_unit
    return token_unit == "none" or token_unit == fact_unit


def check_numbers(text: str, facts, names=()) -> tuple:
    """Problems with the numbers of a text, and for each number the fact it equals. Every number must equal the
    shown value of a fact in a fitting unit; a value with an interval needs both bounds in the text; a point
    estimate needs the words "point estimate"; a number written as a word is refused."""
    problems, matched = [], []
    tokens = numbers_in(text, names)
    present = {token.canon for token in tokens}
    says_point = bool(_POINT_ESTIMATE_RE.search(text))
    bounds = {}
    for fact in facts:
        if fact.kind in ("lo", "hi"):
            bounds.setdefault(fact.group, {})[fact.kind] = canonical(fact.shown)
    for token in tokens:
        candidates = [fact for fact in facts if canonical(fact.shown) == token.canon and _unit_fits(token.unit, fact.unit)]
        if not candidates:
            near = [fact for fact in facts if canonical(fact.shown).rstrip("%") == token.canon.rstrip("%")]
            why = (f"is a value of the page in another unit ({near[0].unit}), not {token.unit}" if near else
                   "is not a value of this page as the page shows it")
            problems.append(f"number {token.raw!r} {why}")
            continue

        def satisfied(fact) -> bool:
            if fact.kind == "value":
                pair = bounds.get(fact.group, {})
                return pair.get("lo") in present and pair.get("hi") in present
            if fact.kind == "point":
                return says_point
            return True

        good = [fact for fact in candidates if satisfied(fact)]
        if not good:
            kind = candidates[0].kind
            problems.append(f"number {token.raw!r} is " + ("a value given without its interval" if kind == "value"
                                                         else "a point estimate the text does not call one"))
            continue
        if all(MODEL_GROUP_RE.match(fact.group) for fact in good) and not MODEL_WORD_RE.search(
                _sentence_at(text, token.start)):
            problems.append(f"number {token.raw!r} is a model's value, and its sentence does not say so (model, "
                            "expected, predicted or estimate)")
            continue
        matched.append(S.GeneratedNumber(shown=token.raw, field=good[0].field))
    words = _mask_names(text, names)
    words = _KIND_OF_SHOT.sub(" ", words)
    for match in _NUMBER_WORD_RE.finditer(words):
        problems.append(f"the number word {match.group(0)!r}: numbers are written as digits")
    for match in _SECOND_RE.finditer(words):
        problems.append(f"the ordinal {match.group(0)!r}: numbers are written as digits")
    for char in sorted(set(words)):
        if unicodedata.category(char).startswith("N") and not char.isascii():
            problems.append(f"the character {char!r}: numbers are written with the digits 0 to 9")
    return problems, matched


def check_status(text: str, item: dict) -> list:
    """A text that uses a value from a model whose lane did not work must say so, in the lane's own words."""
    problems = []
    status = item["context"].get("model_status", {})
    used = {fact.metric for fact in _facts(item)
            if fact.metric and any(canonical(fact.shown) == token.canon for token in numbers_in(text, item["names"]))}
    for metric in sorted(used):
        verdict = status.get(metric, "works")
        words = STATUS_WORDS.get(verdict)
        if words and not re.search(rf"\b{re.escape(words)}\b", text, re.I):
            problems.append(f"uses {metric} values and does not say its model's status ({words})")
    return problems


# Reading grade and plain words: a port of web/scripts/lib/readability.mjs and of the rules of
# web/scripts/check-copy.mjs, so the generator drops what `npm run check` would refuse.
MAX_GRADE = 9
_SYLLABLE_OVERRIDES = {"video": 3, "area": 3, "idea": 3, "create": 2, "science": 2, "league": 1, "leagues": 1,
                       "nba": 3, "tv": 2}
_TOKEN_RE = re.compile(r"\{([a-z][a-z_]*(?:\.[a-z0-9_]+)+)\}")
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z]+\}")
JARGON = (("homograph", False), ("reprojection", False), ("bootstrap", False), ("Hungarian", False),
          ("RANSAC", False), ("YOLO", False), ("BoT-SORT", False), (r"\bCLIP\b", True), ("embedding", False),
          ("confidence interval", False), (r"\bCI\b", True), (r"\bPBP\b", True), (r"\bxFG\b", False),
          (r"\bEPV\b", True), ("ghost", False), ("held[- ]out", False), ("out[- ]of[- ]fold", False),
          ("log loss", False), (r"\bMSE\b", True), ("Brier", False), ("calibrat", False), ("percentile", False),
          (r"\bp50\b", False), (r"\bp90\b", False), (r"\bmedian\b", False), ("ground truth", False))
FORBIDDEN = (r"\bbest\b", r"\boptimal", r"\bideal\b", r"\bshould\b", r"no model (?:saw|has seen|ever saw)",
             r"\bpasses\b", r"\bproves?\b")
# check-copy.mjs BANNED: the ghost defenders are an average defense, never the best one.
BANNED = r"\b(best|optimal|ideal|perfect|correct)\s+(defen[cs]e|defenders?|positions?|positioning)\b"
DASHES = (chr(0x2013), chr(0x2014))
MARKUP = set("*#_`<>[]{}|")
MIN_WORDS, MAX_WORDS = 30, 160


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    if w in _SYLLABLE_OVERRIDES:
        return _SYLLABLE_OVERRIDES[w]
    s = "".join("j" if c == "y" and (i == 0 or w[i - 1] in "aeiou") and (w[i + 1] if i + 1 < len(w) else "#") in "aeiou"
                else c for i, c in enumerate(w))
    n = len(re.findall(r"[aeiouy]+", s))
    if re.search(r"[^aeiou]ying$", s) or re.search(r"[aeiou]ing$", s):
        n += 1
    if re.search(r"[^aeiouy]ier$", s) and n > 1:
        n += 1
    if s.endswith("e") and n > 1:
        if s.endswith("le") and len(s) > 2 and s[-3] not in "aeiouy":
            pass
        elif s[-2] not in "aeiouy":
            n -= 1
    elif s.endswith("es") and n > 1:
        if not (re.search(r"(s|x|z|ch|sh|c|g)es$", s) or re.search(r"[^aeiouy]les$", s)):
            n -= 1
    elif s.endswith("ed") and n > 1:
        if not re.search(r"[td]ed$", s):
            n -= 1
    return max(1, n)


def words_of(text: str) -> list:
    flat = _PLACEHOLDER_RE.sub(" TOKEN ", _TOKEN_RE.sub(" TOKEN ", text))
    words = []
    for raw in flat.split():
        for part in raw.split("-"):
            core = re.sub(r"[^A-Za-z0-9']", "", part)
            if core:
                words.append(core)
    return words


def sentence_count(text: str) -> int:
    total = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        paragraph = " ".join(block.split())
        if not paragraph:
            continue
        total += len(re.findall(r"[.!?]+(?=\s|$)", paragraph))
        if not re.search(r"[.!?]$", paragraph):
            total += 1
    return max(1, total)


def fk_grade(text: str, token_syllables: int = 1) -> float:
    words = words_of(text)
    if not words:
        return 0.0
    syllables = sum(token_syllables if word == "TOKEN" else count_syllables(word) for word in words)
    return 0.39 * len(words) / sentence_count(text) + 11.8 * syllables / len(words) - 15.59


def round1(value: float) -> float:
    """JavaScript's Math.round(value * 10) / 10, as web/scripts/lib/readability.mjs rounds a grade."""
    return math.floor(value * 10 + 0.5) / 10


def sentences_of(text: str) -> list:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def check_words(text: str) -> list:
    """The rules of site copy that apply to a generated text (web/scripts/lib/generated-text.mjs holds the same)."""
    problems = []
    if "\n" in text or text != text.strip():
        problems.append("not one paragraph of plain text")
    markup = sorted(MARKUP & set(text))
    if markup:
        problems.append(f"markup characters {''.join(markup)!r}")
    if any(dash in text for dash in DASHES):
        problems.append("an em dash or an en dash")
    count = len(words_of(text))
    if not MIN_WORDS <= count <= MAX_WORDS:
        problems.append(f"{count} words, not {MIN_WORDS} to {MAX_WORDS}")
    grade = round1(fk_grade(text))
    if grade > MAX_GRADE:
        problems.append(f"reads at grade {grade}, above {MAX_GRADE}")
    for sentence in sentences_of(text):
        one = round1(fk_grade(sentence))
        if one > MAX_GRADE:
            problems.append(f"a sentence reads at grade {one}: {sentence[:60]!r}")
    for pattern, cased in JARGON:
        if re.search(pattern, text, 0 if cased else re.I):
            problems.append(f"jargon /{pattern}/")
    for pattern in FORBIDDEN:
        if re.search(pattern, text, re.I):
            problems.append(f"forbidden claim /{pattern}/")
    if re.search(BANNED, text, re.I):
        problems.append("calls a defense the best or the optimal one")
    return problems


def check_text(text: str, item: dict) -> tuple:
    """Every check of a text against its item's input. Returns (problems, numbers): empty problems means kept."""
    number_problems, numbers = check_numbers(text, _facts(item), item["names"])
    problems = (number_problems + check_directions(text, _facts(item), item["names"]) + check_status(text, item)
                + check_lanes(text, item["names"]) + check_words(text))
    return problems, numbers


# --------------------------------------------------------------------------------------------------- prompts


def prompt(kind: str) -> tuple:
    """(version, text, sha256) of the prompt in use for a kind. export/prompts/versions.json names the version and
    pins the hash of its file: a changed prompt is a new version, never an edit of an old one."""
    versions = json.loads(PROMPT_VERSIONS.read_text(encoding="utf-8"))
    version = versions["current"][kind]
    text = (PROMPT_DIR / f"{version}.md").read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if versions["sha256"].get(version) != digest:
        raise GenaiProblem(f"prompt {version} does not match the hash pinned in versions.json: a changed prompt is "
                           "a new version")
    return version, text, digest


def user_message(item: dict) -> str:
    shown = {"context": item["context"],
             "facts": [{k: fact[k] for k in ("field", "shown", "unit", "kind", "about")} for fact in item["facts"]]}
    return "The input:\n\n" + json.dumps(shown, indent=1, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ fixture mode


def fixture_text(item: dict) -> str:
    """A deterministic text built from the facts, for synthetic roots and tests. It passes every check."""
    facts = {fact.field: fact for fact in _facts(item)}
    status = item["context"].get("model_status", {})
    lines = []

    def flag(metric: str) -> str:
        words = STATUS_WORDS.get(status.get(metric, "works"))
        return f" Its model has the status {words}." if words else ""

    if item["kind"] == "scouting":
        c = item["context"]
        lines.append(f"This is a test text for {c['name']}. He played {facts['games'].shown} tracked games.")
        value = facts.get("shot_making.over_expected_per_100.value")
        if value:
            lo, hi = facts["shot_making.over_expected_per_100.lo"], facts["shot_making.over_expected_per_100.hi"]
            lines.append(f"He made {value.shown} shots per 100 above the expected rate, with a range of {lo.shown} "
                         f"to {hi.shown}." + flag("shot_making"))
        else:
            lines.append(f"His shot value is not shown. He has {facts['shot_making.attempts'].shown} attempts and "
                         f"needs {facts['shot_making.min_attempts'].shown}.")
        value = facts.get("gravity.gravity_ft.value")
        if value:
            lo, hi = facts["gravity.gravity_ft.lo"], facts["gravity.gravity_ft.hi"]
            lines.append(f"Defenders stand {value.shown} feet closer than his spot explains, with a range of "
                         f"{lo.shown} to {hi.shown} feet. This is descriptive." + flag("gravity"))
        lines.append("These are test words from a fixture, not a result.")
    else:
        c = item["context"]
        lines.append(f"This is a test text. {c['offense']} had the ball against {c['defense']} for "
                     f"{facts['duration_s'].shown} seconds. It ended in a {c['result']} worth "
                     f"{facts['points'].shown} points.")
        if "epv_start" in facts and "epv_end" in facts:
            lines.append(f"The expected points went from {facts['epv_start'].shown} to {facts['epv_end'].shown}. "
                         "Each is a point estimate." + flag("epv"))
        if "xfg" in facts:
            lines.append(f"The last shot had a {facts['xfg'].shown} chance to go in, a point estimate."
                         + flag("xfg"))
        lines.append("These are test words from a fixture, not a result. They are here to show where a text goes.")
    return " ".join(lines)


# ---------------------------------------------------------------------------------------------------- the call


def cache_dir(path=None) -> Path:
    return Path(path or os.environ.get("GENAI_CACHE_DIR") or REPO_ROOT / ".genai-cache")


def call_id(call: dict) -> str:
    return json.dumps(call, sort_keys=True, separators=(",", ":"))


def cache_key(model: str, version: str, prompt_sha: str, input_hash: str, call: dict = LEGACY_CALL) -> str:
    """The cache file of one ask: model, prompt version and hash, input hash and the call setting. A record of the
    first setting (LEGACY_CALL) keeps the key it was written under, so those records stay readable."""
    base = f"{model}\n{version}\n{prompt_sha}\n{input_hash}"
    if call_id(call) != call_id(LEGACY_CALL):
        base += f"\n{call_id(call)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise GenaiProblem("live mode needs the anthropic package: run with the venv's python, not the system "
                           "Python (CLAUDE.md)") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise GenaiProblem("live mode needs ANTHROPIC_API_KEY in the environment")
    return anthropic.Anthropic()


def _text_of(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


def call_model(client, item: dict, system: str, model: str = MODEL) -> dict:
    """Ask the model for the text of one item, at most MAX_ATTEMPTS times. A second attempt sees the problems of
    the first. Returns the record the cache keeps: the kept text or none, the problems, and every attempt's usage."""
    try:
        import anthropic
        api_errors = (anthropic.APIStatusError, anthropic.APIConnectionError)
    except ImportError:  # a test passes its own client and runs on the system Python, which has no SDK
        api_errors = ()
    messages = [{"role": "user", "content": user_message(item)}]
    attempts, text, problems, numbers = [], None, ["no attempt"], []
    for _ in range(MAX_ATTEMPTS):
        try:
            # The SDK retries a rate limit, an overload and a dropped connection itself (max_retries).
            response = client.messages.create(model=model, system=system, messages=messages, **CALL)
        except api_errors as exc:
            status = getattr(exc, "status_code", None)
            raise GenaiProblem(f"the model call failed ({type(exc).__name__}"
                               + (f", status {status})" if status else ")")) from exc
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        candidate = _text_of(response)
        if response.stop_reason != "end_turn" or not candidate:
            found = [f"the model stopped with {response.stop_reason}"]
        else:
            found, found_numbers = check_text(candidate, item)
        attempts.append({"text": candidate, "stop_reason": response.stop_reason, "problems": found, "usage": usage})
        if not found:
            text, problems, numbers = candidate, [], found_numbers
            break
        problems = found
        if response.stop_reason in ("refusal", "max_tokens"):
            break
        messages += [{"role": "assistant", "content": response.content},
                     {"role": "user", "content": "That text broke these rules:\n- " + "\n- ".join(found) +
                      "\nWrite the paragraph again and fix every one. Copy each number exactly as shown."}]
    return {"text": text, "problems": problems, "numbers": [n.model_dump() for n in numbers], "attempts": attempts}


def text_for(item: dict, *, live: bool, client=None, cache=None, model: str = MODEL, reuse_older: bool = True) -> tuple:
    """The text of one item, from the cache when the same model, prompt, input and call setting were asked before.
    Returns (record, from_cache). In fixture mode nothing is cached and no model is called.

    A text kept under an older call setting is still a text that passed every check, so it is reused (unless
    `reuse_older` is false, for a sample that tests the current setting). A failure under an older setting is asked
    once more under the current one; a failure under the current setting is final."""
    input_hash = input_sha256(item)
    if not live:
        text = fixture_text(item)
        problems, numbers = check_text(text, item)
        return ({"text": None if problems else text, "problems": problems, "numbers": [n.model_dump() for n in numbers],
                 "attempts": [], "model": FIXTURE_MODEL, "prompt_version": FIXTURE_MODEL,
                 "input_sha256": input_hash}, False)
    version, system, prompt_sha = prompt(item["kind"])
    folder = cache_dir(cache) / item["kind"]
    path = folder / f"{cache_key(model, version, prompt_sha, input_hash, CALL)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    older = folder / f"{cache_key(model, version, prompt_sha, input_hash, LEGACY_CALL)}.json"
    if reuse_older and older.exists():
        record = json.loads(older.read_text(encoding="utf-8"))
        if record.get("text"):
            return record, True
    record = call_model(client, item, system, model)
    record.update({"call": CALL, "key": item["key"], "kind": item["kind"], "model": model, "prompt_version": version,
                   "prompt_sha256": prompt_sha, "input_sha256": input_hash,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return record, False


def describe_call(call) -> str:
    """A call setting in words, for the notes of a genai file."""
    call = call or LEGACY_CALL
    thinking = (call.get("thinking") or {}).get("type", "adaptive (the model's default)")
    return (f"thinking {thinking}, effort {call.get('output_config', {}).get('effort', 'default')}, "
            f"at most {call.get('max_tokens', LEGACY_CALL['max_tokens'])} output tokens")


def call_notes(root: Path, kind: str, texts, cache=None) -> list:
    """One note per call setting: how many of the texts it wrote (read from the cache records). A text whose
    record is not in this cache is counted as not known."""
    builders = {item[0]: item[1] for item in items_of(root, kind)}
    counts: dict = {}
    for text in texts:
        if text.model == FIXTURE_MODEL or text.key not in builders:
            continue
        found = "not known (no cache record here)"
        try:
            versions = json.loads(PROMPT_VERSIONS.read_text(encoding="utf-8"))
            prompt_sha = versions["sha256"][text.prompt_version]
        except KeyError:
            prompt_sha = None
        if prompt_sha:
            for call in (CALL, LEGACY_CALL):
                path = cache_dir(cache) / kind / f"{cache_key(text.model, text.prompt_version, prompt_sha, text.input_sha256, call)}.json"
                if path.exists():
                    found = describe_call(json.loads(path.read_text(encoding="utf-8")).get("call"))
                    break
        counts[found] = counts.get(found, 0) + 1
    return [f"Call setting: {setting}, {n} text(s)." for setting, n in sorted(counts.items())]


# ---------------------------------------------------------------------------------------------- the data root


def _load(root: Path, rel: str, model):
    return model.model_validate_json((root / rel).read_bytes())


def items_of(root: Path, kind: str) -> list:
    """Every item of a kind the data root can have a text for, with what builds its input, in a fixed order."""
    root = Path(root)
    results = _load(root, "results.json", S.Results) if (root / "results.json").exists() else None
    if kind == "scouting":
        players = _load(root, "players.json", S.PlayersFile)
        return [(player.slug, lambda p=player: player_input(p, players, results)) for player in players.players]
    out = []
    for index_path in sorted((root / "games").glob("*/possessions.json")):
        slug = index_path.parent.name
        index = _load(root, f"games/{slug}/possessions.json", S.PossessionIndex)
        for row in index.possessions:
            def build(row=row, slug=slug):
                return possession_input(row, _load(root, f"games/{slug}/{row.possession}.json", S.Possession), results)
            out.append((f"{slug}/{row.possession}", build, row))
    return out


def choose(root: Path, kind: str, n, seed: int = 0) -> list:
    """The items of a sample run: all of them when n is None, else n drawn at random (seed 0). Possessions are
    drawn from the featured ones first (PLAN.md D48), which end in a shot with an xFG and have an EPV curve."""
    items = items_of(root, kind)
    if n is None or n >= len(items):
        return [(item[0], item[1]) for item in items]
    rng = random.Random(seed)
    if kind == "possession":
        featured = [item for item in items if item[2].featured]
        rest = [item for item in items if not item[2].featured]
        picked = rng.sample(featured, min(n, len(featured)))
        picked += rng.sample(rest, n - len(picked)) if len(picked) < n else []
    else:
        picked = rng.sample(items, n)
    order = {item[0]: k for k, item in enumerate(items)}
    return [(item[0], item[1]) for item in sorted(picked, key=lambda item: order[item[0]])]


def _sources(root: Path, kind: str, texts, synthetic: bool) -> list:
    """The generator, the prompt version each kept text was written with (a file may hold texts of several
    versions after a merge), and the sources of the inputs the texts were written from."""
    if synthetic:
        return [S.Source(path=S.FIXTURE_SOURCE, what="synthetic fixture values, not a result", rows="not_split")]
    keys = [t.key for t in texts]
    versions = sorted({t.prompt_version for t in texts if t.prompt_version != FIXTURE_MODEL})
    own = [S.Source(path=GENERATOR, what="machine-written text, every number checked against the page's values "
                    "(PLAN.md P3)", rows="not_split")]
    own += [S.Source(path=f"export/prompts/{version}.md", what=f"the prompt, version {version}", rows="not_split")
            for version in versions]
    found = []
    if kind == "scouting":
        found += _load(root, "players.json", S.PlayersFile).sources
    else:
        seen = set()
        for key in keys:
            slug, number = key.split("/")
            seen.add(slug)
            found += strip_possession(_load(root, f"games/{slug}/{number}.json", S.Possession)).sources
        for slug in sorted(seen):
            found += _load(root, f"games/{slug}/possessions.json", S.PossessionIndex).sources
    unique, keys_seen = [], set()
    for source in own + found:
        k = (source.path, source.what, source.rows)
        if k not in keys_seen:
            keys_seen.add(k)
            unique.append(source)
    return unique


def _dump(model: S.Strict) -> str:
    return json.dumps(model.model_dump(mode="json"), separators=(",", ":"), allow_nan=False) + "\n"


def write_texts(root: Path, kind: str, texts: list, synthetic: bool, manifest: bool = True, extra_notes=()) -> None:
    """Write the texts of one kind into the data root (or remove the file when there is none) and, unless told
    not to, write the manifest again with the genai family. Nothing else of the root changes."""
    root = Path(root)
    path = root / S.GENAI_FILES[kind]
    if texts:
        notes = ["Machine-written from the numbers of each page. A text that failed a check was dropped, so a page "
                 "may have none."]
        if synthetic:
            notes.append("Fixture texts of a synthetic data root, not a result.")
        notes = notes + list(extra_notes)
        file = S.GeneratedTexts(schema_version="1.2.0", kind=kind, label="machine-written", checks=list(CHECKS),
                                items=sorted(texts, key=lambda t: t.key), notes=notes,
                                sources=_sources(root, kind, texts, synthetic))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(file), encoding="utf-8")
    elif path.exists():
        path.unlink()
        if path.parent.exists() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    if manifest:
        current = _load(root, "manifest.json", S.Manifest)
        (root / "manifest.json").write_text(_dump(with_genai_entry(current, root)), encoding="utf-8")


def genai_entries(root: Path) -> list:
    """One manifest entry for each genai file on disk, in a fixed order."""
    entries = []
    for rel in sorted(S.GENAI_FILES.values()):
        path = Path(root) / rel
        if not path.exists():
            continue
        data = path.read_bytes()
        entries.append(S.ManifestFile(path=rel, model=S.GeneratedTexts.__name__, count=1, bytes=len(data),
                                      sha256=hashlib.sha256(data).hexdigest(),
                                      sources=S.GeneratedTexts.model_validate_json(data).sources))
    return entries


def with_genai_entry(manifest: S.Manifest, root: Path) -> S.Manifest:
    """The manifest with one entry for each genai file on disk, or none when there is none. A build id made from
    the content (export.site_data: a prefix, a dash and the first 12 characters of S.content_id) is made again, so
    a change to the texts alone gives a new build id; any other build id is kept as it is."""
    files = [entry for entry in manifest.files if entry.model != S.GeneratedTexts.__name__] + genai_entries(root)
    build_id = manifest.build_id
    derived = re.fullmatch(r"(.+-)([0-9a-f]{12})", build_id)
    if derived and derived.group(2) == S.content_id(manifest.files)[:12]:
        build_id = derived.group(1) + S.content_id(files)[:12]
    return manifest.model_copy(update={"files": files, "build_id": build_id})


def root_problems(texts: S.GeneratedTexts, parsed: dict) -> list:
    """The checks of every text of a genai file, against the data root it sits in (`parsed`: path to validated
    contract object, as export.schemas.validate_tree holds them)."""
    problems = []
    results = parsed.get("results.json")
    players = parsed.get("players.json")
    for item in texts.items:
        try:
            if texts.kind == "scouting":
                player = next((p for p in players.players if p.slug == item.key), None) if players else None
                if player is None:
                    problems.append(f"{item.key}: no such player in players.json")
                    continue
                data = player_input(player, players, results)
            else:
                slug, _, number = item.key.partition("/")
                index = parsed.get(f"games/{slug}/possessions.json")
                possession = parsed.get(f"games/{slug}/{number}.json")
                row = next((r for r in index.possessions if str(r.possession) == number), None) if index else None
                if row is None or possession is None:
                    problems.append(f"{item.key}: no such possession in the data root")
                    continue
                data = possession_input(row, possession, results)
        except (ValueError, KeyError) as exc:
            problems.append(f"{item.key}: its input could not be built: {type(exc).__name__}")
            continue
        if input_sha256(data) != item.input_sha256:
            problems.append(f"{item.key}: stale, its input changed since it was written (run python3 -m export.genai)")
            continue
        found, numbers = check_text(item.text, data)
        problems += [f"{item.key}: {line}" for line in found]
        if not found and [n.model_dump() for n in numbers] != [n.model_dump() for n in item.numbers]:
            problems.append(f"{item.key}: the numbers it lists are not the numbers of its text")
    return problems


def prune(root: Path) -> dict:
    """Drop every text whose input changed or whose item is gone, with no model call. The manifest is left to the
    caller: export.site_data runs this before it writes its manifest, so rebuilding the site data never leaves a
    stale text behind."""
    root = Path(root)
    dropped = {}
    for kind, rel in S.GENAI_FILES.items():
        if not (root / rel).exists():
            continue
        texts = _load(root, rel, S.GeneratedTexts)
        try:
            builders = {item[0]: item[1] for item in items_of(root, kind)}
        except FileNotFoundError:
            builders = {}
        kept = []
        for text in texts.items:
            try:
                if text.key in builders and input_sha256(builders[text.key]()) == text.input_sha256:
                    kept.append(text)
            except FileNotFoundError:
                pass
        dropped[kind] = len(texts.items) - len(kept)
        if dropped[kind]:
            synthetic = all(source.path == S.FIXTURE_SOURCE for source in texts.sources)
            write_texts(root, kind, kept, synthetic, manifest=False)
    return dropped


# ------------------------------------------------------------------------------------------------------- run


def earlier_texts(root: Path, kind: str, skip=()) -> tuple:
    """The texts already in the root's genai file that still hold: the item still exists, its input is unchanged
    (same hash) and the text passes every check of today. Returns (kept, dropped count); keys in `skip` are left
    out and not counted."""
    rel = S.GENAI_FILES[kind]
    if not (Path(root) / rel).exists():
        return [], 0
    builders = {item[0]: item[1] for item in items_of(root, kind)}
    kept, dropped = [], 0
    for text in _load(root, rel, S.GeneratedTexts).items:
        if text.key in skip:
            continue
        item = builders[text.key]() if text.key in builders else None
        if item is None or input_sha256(item) != text.input_sha256:
            dropped += 1
            continue
        problems, numbers = check_text(text.text, item)
        if problems:
            dropped += 1
            continue
        kept.append(text.model_copy(update={"numbers": numbers}))
    return kept, dropped


def run(root, kinds=KINDS, n_players=None, n_possessions=None, *, live=False, cache=None, client=None,
        model: str = MODEL, log=print) -> dict:
    """Generate (or take from the cache) the texts of the chosen items and merge them into the root's texts, then
    validate the root. Returns counts and token use.

    A run merges, it never replaces: a text already in the root for an item this run did not choose, or for an
    item whose new text was dropped, stays when it still holds (earlier_texts). So a sample run after a full run
    cannot throw away texts that were paid for and still pass, and a stale or failing text never survives, because
    the input hash and every check are applied to it again. Refusing a partial run instead would force a full,
    paid rerun for any top-up and still not protect a text whose page changed."""
    root = Path(root)
    manifest = _load(root, "manifest.json", S.Manifest)
    if not live and not manifest.synthetic:
        raise GenaiProblem("fixture texts go only into a synthetic data root: pass --live for real data")
    if live and client is None:
        client = _client()
    info = {"model": model if live else FIXTURE_MODEL, "kinds": {}}
    for kind in kinds:
        chosen = choose(root, kind, n_players if kind == "scouting" else n_possessions)
        kept, stats = [], {"items": len(chosen), "kept": 0, "dropped": 0, "cached": 0, "calls": 0,
                           "input_tokens": 0, "output_tokens": 0, "problems": {}}
        for key, build in chosen:
            item = build()
            record, cached = text_for(item, live=live, client=client, cache=cache, model=model)
            stats["cached"] += cached
            if not cached:
                for attempt in record["attempts"]:
                    stats["calls"] += 1
                    stats["input_tokens"] += attempt["usage"]["input_tokens"]
                    stats["output_tokens"] += attempt["usage"]["output_tokens"]
            if record["text"]:
                stats["kept"] += 1
                kept.append(S.GeneratedText(key=key, text=record["text"], input_sha256=record["input_sha256"],
                                            model=record["model"], prompt_version=record["prompt_version"],
                                            numbers=[S.GeneratedNumber(**n) for n in record["numbers"]]))
            else:
                stats["dropped"] += 1
                stats["problems"][key] = record["problems"]
            log(f"{kind} {key}: {'kept' if record['text'] else 'dropped'}{' (cache)' if cached else ''}")
        earlier, stats["earlier_dropped"] = earlier_texts(root, kind, skip={text.key for text in kept})
        stats["earlier_kept"] = len(earlier)
        write_texts(root, kind, kept + earlier, manifest.synthetic,
                    extra_notes=call_notes(root, kind, kept + earlier, cache) if live else ())
        stats["all_items"] = len(items_of(root, kind))
        info["kinds"][kind] = stats
    info["problems"] = S.validate_tree(root, allow_missing=S.STAGED_FILES)
    return info


def cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_PER_MTOK["input"] + output_tokens * PRICE_PER_MTOK["output"]) / 1e6


def report(info: dict) -> list:
    lines = []
    total_in = total_out = 0
    projected = 0.0
    for kind, stats in info["kinds"].items():
        total_in += stats["input_tokens"]
        total_out += stats["output_tokens"]
        lines.append(f"{kind}: {stats['kept']} kept, {stats['dropped']} dropped of {stats['items']} "
                     f"({stats['cached']} from the cache); earlier texts kept {stats.get('earlier_kept', 0)}, "
                     f"dropped {stats.get('earlier_dropped', 0)}; {stats['calls']} calls, {stats['input_tokens']} input and "
                     f"{stats['output_tokens']} output tokens, ${cost(stats['input_tokens'], stats['output_tokens']):.4f}")
        fresh = stats["items"] - stats["cached"]
        if fresh:
            per_item = cost(stats["input_tokens"], stats["output_tokens"]) / fresh
            projected += per_item * stats["all_items"]
            lines.append(f"{kind}: projected for all {stats['all_items']} items at this run's mean: "
                         f"${per_item * stats['all_items']:.2f}")
        for key, found in list(stats["problems"].items())[:10]:
            lines.append(f"  dropped {key}: {'; '.join(found[:3])}")
    lines.append(f"model {info['model']}: {total_in} input and {total_out} output tokens, ${cost(total_in, total_out):.4f}"
                 + (f"; projected full run ${projected:.2f}" if projected else ""))
    lines += [f"PROBLEM: {line}" for line in info["problems"][:20]]
    lines.append(f"{len(info['problems'])} problem(s)" if info["problems"] else "data root valid with its texts")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write machine-written texts into a data root (PLAN.md P3).")
    parser.add_argument("--root", required=True, help="the data root to read and write (for example web/data)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--players", type=int, default=10, help="scouting texts for a sample of this many players")
    parser.add_argument("--possessions", type=int, default=20, help="texts for a sample of this many possessions")
    parser.add_argument("--all", action="store_true", help="every player and every possession (needs a yes, D76)")
    parser.add_argument("--live", action="store_true", help="call the model (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--cache", help="the cache folder (default GENAI_CACHE_DIR, else .genai-cache)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--summary", help="also write the run's counts and token use to this json file")
    args = parser.parse_args(argv)
    try:
        info = run(args.root, args.kinds, None if args.all else args.players, None if args.all else args.possessions,
                   live=args.live, cache=args.cache, model=args.model)
    except GenaiProblem as exc:
        print(f"genai: {exc}", file=sys.stderr)
        return 1
    if args.summary:
        Path(args.summary).write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    for line in report(info):
        print(line)
    return 1 if info["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
