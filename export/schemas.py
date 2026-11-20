"""export/schemas.py: the site data contract (PLAN.md, Site structure; D31).

Python produces data, the web only reads it. The pydantic models below are the contract for every file under
web/data. `python3 -m export.schemas --out web/lib/types/contract.schema.json` writes the JSON Schema; the
TypeScript types in web/lib/types are generated from that file and never edited by hand. The same shapes are meant
to come from a live service later, so nothing here depends on how the files were produced.

Rules the models and validate_tree enforce:
- every file carries the schema version and names at least one source, and each source says which rows it read;
- an estimate either has an interval or says it is a point estimate; a single prediction (an xFG, an EPV value, a
  ghost position) is a point estimate by definition, and its docstring says so;
- a player value is shown only at or above its minimum sample, and a rebuilt object only with a passed check;
- in a real data root the review side reads held-out games only and the player side never reads a held-out row;
- no absolute local path and no email address can appear in any string (find_private).

The JSON Schema carries the shapes and the fixed labels. The cross-field rules above live in the validators here,
so a data root is checked with `python3 -m export.schemas --validate`, not with the JSON Schema alone.

Changes from 1.0.0 to 1.1.0 (after the M0 review; decisions D49 to D53 in PLAN.md):
- Source: `rows` (heldout, out_of_fold, training, not_split) replaces the `heldout` flag; every `sources` list
  needs at least one entry; a source path is never absolute.
- RebuildCheck replaces ModelCheck (fields name, compared, n, n_expected, max_abs_diff, tolerance, passed, unit).
  PlayersFile gains `checks` and `shot_making_baseline`; ShotMaking gains `baseline_error_per_100`.
- SwapSummary: an unavailable comparison has n 0, null differences and a note. DemoSwap gains `match_ft` and
  `max_gap_s` and needs a passed check for every model it shows. DemoFrames gains `scene_rule`.
- DemoBox.status is matched or unmatched (was matched or ghost).
- PossessionSummary gains `featured_by` and `featured_cell`. LaneResult gains `verdict_source`.
- Probabilities and shares must lie between 0 and 1. A scorecard cell's status must agree with `passed`.

Changes from 1.1.0 to 1.2.0 (PLAN.md D72, the methods page):
- methods.json (MethodsFile): per lane the papers it builds on, our numbers beside the published ones with the
  comparability verdict of the lane dossier, and every pre-registered experiment with its status and, once it
  has run, its result. export/research_methods.py writes it from reports/research/methods.json.
- Source.rows gains `testset`: the D71 test set (PLAN.md D77), scored once per experiment.
- A data file of an earlier minor version of the same major version still validates (the change is additive), so
  `schema_version` accepts every version in COMPATIBLE_VERSIONS; the manifest and the new file carry this one.

Also in 1.2.0 (PLAN.md P3, D76; merged with the methods page into one version, D82), an addition only:
- A new file family, genai/<kind>.json (model GeneratedTexts): machine-written texts, a scouting report per player
  of the prep tab and an account in words of each possession of the review. Each text is optional; a data root
  with no genai folder is complete. export/genai.py writes them and holds the checks every text must pass.
- A genai file is 1.2.0 only (CurrentVersion), like the manifest; the other files of its root may still carry
  1.1.0 (COMPATIBLE_VERSIONS).

Also in 1.2.0 (PLAN.md D94, player values broken by the frozen name joins), an addition only:
- ShotMaking and Rebounding gain an optional `withheld` reason: a hidden value whose cause is not its sample (the
  stored lane tables lost or mixed up some of the player's rows). A value without one is shown exactly when its
  sample reaches the minimum.
- Rebounding refuses a shown rate of zero with a zero upper end.

Units: court positions are in feet in the SportVU frame, x from 0 to 94 along the court, y from 0 to 50 across it,
rims at (5.25, 25) and (88.75, 25). Clocks are in seconds. Times `t` are seconds from the start of the possession
or of the clip.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.2.0"
# Versions a data file may carry: this one and the earlier minor versions of the same major version, whose files
# are still valid because every change since them only added a file or a value. The manifest and every file a
# version introduced carry SCHEMA_VERSION itself (CurrentVersion).
COMPATIBLE_VERSIONS = ("1.1.0", "1.2.0")
SchemaVersion = Literal["1.1.0", "1.2.0"]
CurrentVersion = Literal["1.2.0"]

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
RIM_X_FT = (5.25, 88.75)
RIM_Y_FT = 25.0

# A string in a data file must never match one of these: a local path, a cloud mount, or an email address.
PRIVATE_PATTERNS = (
    r"/Users/",
    r"/home/",
    r"/root/",
    r"/private/",
    r"/var/folders",
    r"/tmp/",
    r"/Volumes/",
    r"/content/drive",
    r"/opt/anaconda",
    r"[A-Za-z]:\\\\?Users",
    r"(?<![\w$])~/",
    r"CloudStorage",
    r"GoogleDrive",
    r"My ?Drive",
    r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
)
_PRIVATE_RE = re.compile("|".join(PRIVATE_PATTERNS))
# Web addresses are not local paths: they are removed before the scan, so a link such as a report on GitHub is fine.
_URL_RE = re.compile(r"https?://[^\s\"'<>)]+")
# The only fixed files that may be absent from a data root, and only while the demo bridge has not run yet.
STAGED_FILES = ("demo/swap.json", "demo/frames.json")
# Files the operating system drops into folders. They are ignored by name; any other unknown file is a problem.
IGNORED_NAMES = (".DS_Store",)


def find_private(text: str) -> list[str]:
    """Return every local path, cloud mount name or email address found in `text` (empty when clean)."""
    plain = _URL_RE.sub(" ", text.replace("\\/", "/"))
    return sorted({m.group(0) for m in _PRIVATE_RE.finditer(plain)})


class Strict(BaseModel):
    """Base for every contract model: unknown fields are an error, so the contract cannot drift silently."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


SourceRows = Literal["heldout", "out_of_fold", "training", "not_split", "testset"]


class Source(Strict):
    """Where a value comes from. `path` is relative to the repository (for example reports/xfg.md) or starts with
    $DATA_ROOT for a data table. `rows` says which rows were read: heldout (held-out games only), out_of_fold
    (training games, each row predicted by a model that never saw its game), training (training games, read as a
    description or fitted in sample), not_split (the source is not split by game), testset (the D71 test set of
    PLAN.md D77, scored once per experiment)."""

    path: str = Field(min_length=1)
    what: str = Field(min_length=1)
    rows: SourceRows

    @model_validator(mode="after")
    def _relative(self):
        if self.path.startswith(("/", "\\")) or self.path != self.path.strip():
            raise ValueError("a source path is relative to the repository or starts with $DATA_ROOT")
        return self


# Every file names at least one source: a value with no source cannot be shown.
Sources = Annotated[list[Source], Field(min_length=1)]
# A probability or a share.
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class Estimate(Strict):
    """A number with its uncertainty. Either `lo` and `hi` are both set and `interval` names the method, or
    `point_estimate` is true and all three are null."""

    value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    interval: Optional[str] = None
    point_estimate: bool
    n: Optional[int] = None
    n_unit: Optional[str] = None

    @model_validator(mode="after")
    def _interval_or_point(self):
        has = self.lo is not None and self.hi is not None
        if (self.lo is None) != (self.hi is None):
            raise ValueError("lo and hi must be set together")
        if has == self.point_estimate:
            raise ValueError("an estimate has an interval or is a point estimate, never both and never neither")
        if has and not self.interval:
            raise ValueError("an interval must name its method")
        if not has and self.interval:
            raise ValueError("a point estimate names no interval method")
        if has and not (self.lo <= self.hi):
            raise ValueError("lo must not exceed hi")
        return self


MODEL_KINDS = ("xfg", "epv", "ghosting")


class RebuildCheck(Strict):
    """Record that a rebuilt object is the frozen one (PLAN.md D45, D49; the tracked copy is
    reports/export/frozen_checks.json). For a model (xfg, epv, ghosting) its predictions were compared with every
    stored held-out prediction: n must equal n_expected. For an output (absence, embeddings) recomputed
    out-of-fold numbers were compared with the lane report. No held-out outcome is read by a check."""

    name: Literal["xfg", "epv", "ghosting", "absence", "embeddings"]
    compared: str = Field(min_length=1)
    n: int
    n_expected: Optional[int] = None
    max_abs_diff: float
    tolerance: float
    passed: bool
    unit: str

    @model_validator(mode="after")
    def _passed(self):
        ok = self.n > 0 and self.max_abs_diff <= self.tolerance
        if self.name in MODEL_KINDS:
            ok = ok and self.n_expected is not None and self.n == self.n_expected
        if self.passed and not ok:
            raise ValueError(f"{self.name}: passed is true but the record does not meet the rule")
        return self


def _has_passed(checks, name: str) -> bool:
    return any(c.name == name and c.passed for c in checks)


# ----------------------------------------------------------------------------------------------- manifest.json


Split = Literal["heldout_game", "heldout_arena"]


class ManifestFile(Strict):
    """One file, or one family of files, under the data root. A family (`count` above 1) is written as a path
    pattern such as games/<slug>/<possession>.json; its sha256 is the hash of the sorted member hashes."""

    path: str
    model: str
    count: int
    bytes: int
    sha256: str
    sources: Sources


class ManifestGame(Strict):
    slug: str
    game_id: str
    label: str
    date: str
    home: str
    away: str
    split: Split
    n_possessions: int
    n_featured: int
    bytes: int


class Manifest(Strict):
    """The entry point. `synthetic` is true when the data was built from fixtures; the site must then label
    every number as fixture output, never as a result."""

    schema_version: CurrentVersion
    build_id: str
    generated_at: str
    repo_commit: str
    freeze_tag: str
    synthetic: bool
    season: str
    n_games: int
    n_possessions: int
    games: list[ManifestGame]
    files: list[ManifestFile]


# ------------------------------------------------------------------------- games/<slug>/possessions.json (index)


PossessionResult = Literal["shot_made", "shot_missed", "free_throws", "turnover", "period_end", "other"]


class GameInfo(Strict):
    slug: str
    game_id: str
    label: str
    date: str
    home: str
    away: str
    split: Split


class PossessionSummary(Strict):
    """One row of the possession list. `xfg` is the frozen model's probability for the last field-goal attempt of
    the possession, when that attempt has a held-out prediction. `xfg_bucket` is its fifth (1 lowest, 5 highest)
    among all held-out attempts, with the edges given once in the index. xFG and EPV values are point estimates:
    the frozen lanes store no interval for a single prediction. `featured_by` is rule for a pick drawn by the
    written rule and hand for one swapped in by a person; `featured_cell` names the cell of the rule."""

    possession: int
    period: int
    start_clock_s: float
    end_clock_s: float
    duration_s: float
    offense: str
    defense: str
    result: PossessionResult
    points: int
    gap: bool
    n_frames: int
    xfg: Optional[Unit] = None
    xfg_bucket: Optional[Annotated[int, Field(ge=1, le=5)]] = None
    epv_start: Optional[float] = None
    epv_end: Optional[float] = None
    has_ghosts: bool
    featured: bool
    featured_by: Optional[Literal["rule", "hand"]] = None
    featured_cell: Optional[str] = None
    demo: bool

    @model_validator(mode="after")
    def _featured(self):
        if self.featured:
            if self.featured_by is None or not self.featured_cell:
                raise ValueError("a featured possession says whether the rule or a hand picked it, and its cell")
            if self.result not in ("shot_made", "shot_missed") or self.xfg is None:
                raise ValueError("a featured possession ends in a made or missed shot that has an xFG")
        elif self.featured_by is not None or self.featured_cell is not None:
            raise ValueError("only a featured possession carries featured_by and featured_cell")
        return self


class PossessionIndex(Strict):
    schema_version: SchemaVersion
    game: GameInfo
    xfg_bucket_edges: list[float]
    possessions: list[PossessionSummary]
    notes: list[str]
    sources: Sources


# ----------------------------------------------------------------------- games/<slug>/<possession>.json (detail)


class PlayerTrack(Strict):
    """One player's path, one value per frame of the possession. A value is null where the player has no tracking
    row in that frame."""

    player_id: int
    name: str
    jersey: Optional[str] = None
    team: str
    offense: bool
    x: list[Optional[float]]
    y: list[Optional[float]]


class BallTrack(Strict):
    x: list[Optional[float]]
    y: list[Optional[float]]
    z: list[Optional[float]]


class GhostTrack(Strict):
    """Where the model expects the defender `player_id` to stand, one value per frame. Null where the model has no
    output for that frame."""

    player_id: int
    x: list[Optional[float]]
    y: list[Optional[float]]


class Ghosts(Strict):
    """Ghost defenders. Each ghost is where an average defender would be `horizon_s` after his real position,
    given how the ball and the attackers really moved. The label is fixed: an average defense, never the best
    one."""

    label: Literal["average defense"]
    horizon_s: float
    tracks: list[GhostTrack]
    dropped_points: int
    mean_deviation_ft: Optional[float] = None


class EpvCurve(Strict):
    """Expected points for the offense from each state to the end of the possession, about one value a second.
    Values are the model's raw output and are not clipped."""

    t: list[float]
    value: list[float]


class Shot(Strict):
    """A field-goal attempt in the possession. `xfg` is the frozen model's probability that it goes in, a point
    estimate (the lane stores no interval for one shot). Where the release was not located in the tracking, the
    time and the location fall back to the play-by-play and the shot chart and say so."""

    event: int
    t: Optional[float] = None
    t_source: Literal["release_frame", "play_by_play", "none"]
    x: Optional[float] = None
    y: Optional[float] = None
    location_source: Literal["tracking_at_release", "shot_chart", "none"]
    shooter_id: Optional[int] = None
    shooter: Optional[str] = None
    made: bool
    three: Optional[bool] = None
    xfg: Optional[Unit] = None
    has_contest: Optional[bool] = None


class Possession(Strict):
    """One held-out possession at 5 Hz. Frame k is at `t[k]` seconds from the start; every track has one value
    per frame. `attack_x` is the x of the rim the offense attacks."""

    schema_version: SchemaVersion
    game_id: str
    game_slug: str
    possession: int
    period: int
    hz: float
    frame0: int
    t: list[float]
    game_clock_s: list[Optional[float]]
    shot_clock_s: list[Optional[float]]
    offense: str
    defense: str
    attack_x: float
    result: PossessionResult
    points: int
    gap: bool
    players: list[PlayerTrack]
    ball: BallTrack
    epv: Optional[EpvCurve] = None
    ghosts: Optional[Ghosts] = None
    shots: list[Shot]
    notes: list[str]
    sources: Sources

    @model_validator(mode="after")
    def _aligned(self):
        n = len(self.t)
        tracks = [("game_clock_s", self.game_clock_s), ("shot_clock_s", self.shot_clock_s),
                  ("ball.x", self.ball.x), ("ball.y", self.ball.y), ("ball.z", self.ball.z)]
        for p in self.players:
            tracks += [(f"player {p.player_id} x", p.x), (f"player {p.player_id} y", p.y)]
        if self.ghosts:
            for g in self.ghosts.tracks:
                tracks += [(f"ghost {g.player_id} x", g.x), (f"ghost {g.player_id} y", g.y)]
        bad = [name for name, values in tracks if len(values) != n]
        if bad:
            raise ValueError(f"tracks not aligned to {n} frames: {bad}")
        if self.epv and len(self.epv.t) != len(self.epv.value):
            raise ValueError("epv t and value differ in length")
        return self


# ------------------------------------------------------------------------------------------------ players.json


def _check_shown(metric: str, shown: bool, n: int, min_n: int, estimate) -> None:
    """A value is shown only at or above its minimum sample, and a hidden value carries no estimate."""
    if shown and n < min_n:
        raise ValueError(f"{metric}: shown with {n}, below the minimum sample of {min_n}")
    if shown and estimate is None:
        raise ValueError(f"{metric}: shown without an estimate")
    if not shown and estimate is not None:
        raise ValueError(f"{metric}: a hidden value carries no estimate")


def _check_withheld(metric: str, shown: bool, n: int, min_n: int, withheld) -> None:
    """`withheld` says why a value is hidden when the reason is not its sample: the stored lane tables lost or mixed
    up some of the player's rows (the name joins section of export/site_players.py). It then replaces the
    minimum-sample sentence, which would be wrong or would hide the cause. A value with no such reason is shown
    exactly when its sample reaches the minimum, so the page never calls a large sample too small."""
    if withheld is not None and (shown or not withheld.strip()):
        raise ValueError(f"{metric}: a withheld reason belongs only to a hidden value")
    if withheld is None and shown != (n >= min_n):
        raise ValueError(f"{metric}: hidden at or above the minimum sample without a withheld reason")


class ShotMaking(Strict):
    """Makes above expected per 100 attempts, from out-of-fold rows. The expectation is the lane's location and
    contest model, which does not know the shooter and was not the selected model. That baseline has a known
    out-of-fold calibration error (PlayersFile.shot_making_baseline): `baseline_error_per_100` is the part of this
    player's value that the baseline's own error explains for his mix of shots (his attempts in each calibration
    bin times that bin's observed minus predicted rate). The frozen xFG model includes a shooter prior (it already
    knows the shooter), so its own residual is not a measure over expected and is given only for reference."""

    shown: bool
    attempts: int
    min_attempts: int
    over_expected_per_100: Optional[Estimate] = None
    fg_pct: Optional[Unit] = None
    expected_fg_pct: Optional[Unit] = None
    baseline_error_per_100: Optional[float] = None
    frozen_model_residual_per_100: Optional[float] = None
    withheld: Optional[str] = None

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("shot_making", self.shown, self.attempts, self.min_attempts, self.over_expected_per_100)
        _check_withheld("shot_making", self.shown, self.attempts, self.min_attempts, self.withheld)
        return self


class Gravity(Strict):
    """How far defenders stay from the player beyond what location explains, in feet (the stored player table)."""

    shown: bool
    possessions: int
    min_possessions: int
    games: int
    gravity_ft: Optional[Estimate] = None

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("gravity", self.shown, self.possessions, self.min_possessions, self.gravity_ft)
        return self


class Rebounding(Strict):
    """Rebounds per 100 chances, a chance being a located miss while he was on the floor. The model expectation
    already uses his rebounding rate in other games, so the gap between the two is not a skill number."""

    shown: bool
    chances: int
    min_chances: int
    rate_per_100: Optional[Estimate] = None
    model_expected_per_100: Optional[float] = None
    withheld: Optional[str] = None

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("rebounding", self.shown, self.chances, self.min_chances, self.rate_per_100)
        _check_withheld("rebounding", self.shown, self.chances, self.min_chances, self.withheld)
        if self.shown and self.rate_per_100.value == 0 and self.rate_per_100.hi == 0:
            raise ValueError("rebounding: a shown rate of zero on the minimum sample or more points at rebounds the "
                             "lane tables lost, not at a rate")
        return self


class AbsenceRisk(Strict):
    """Chance that he does not appear in the team's next tracked game. `model_probability` is the mean
    out-of-fold probability of the frozen logistic and is null when that output could not be reproduced;
    `observed_rate` is the plain share of games after which he was absent. Neither is an injury forecast."""

    shown: bool
    games: int
    min_games: int
    absent: int
    kind: Literal["model_out_of_fold", "observed_only"]
    model_probability: Optional[Estimate] = None
    observed_rate: Optional[Estimate] = None

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("absence_risk", self.shown, self.games, self.min_games, self.observed_rate)
        if self.kind == "observed_only" and self.model_probability is not None:
            raise ValueError("absence_risk: observed_only carries no model probability")
        if not self.shown and self.model_probability is not None:
            raise ValueError("absence_risk: a hidden value carries no estimate")
        if self.shown and self.kind == "model_out_of_fold" and self.model_probability is None:
            raise ValueError("absence_risk: model_out_of_fold needs the model probability")
        return self


class SimilarPlayer(Strict):
    rank: int
    player_id: int
    name: str
    distance: float


class Similar(Strict):
    """Nearest players in the style embedding (descriptive, fitted on training games, point estimates).
    `available` is false when the embedding output could not be reproduced."""

    shown: bool
    available: bool
    possessions: int
    min_possessions: int
    neighbours: list[SimilarPlayer]

    @model_validator(mode="after")
    def _shown(self):
        if self.shown and not (self.available and self.possessions >= self.min_possessions and self.neighbours):
            raise ValueError("similar: shown needs an available embedding, the minimum sample and neighbours")
        if not self.shown and self.neighbours:
            raise ValueError("similar: a hidden value carries no neighbours")
        return self


class PlayerView(Strict):
    player_id: int
    slug: str
    name: str
    teams: list[str]
    position: Optional[str] = None
    games: int
    shot_making: ShotMaking
    gravity: Gravity
    rebounding: Rebounding
    absence_risk: AbsenceRisk
    similar: Similar


class MinSampleRule(Strict):
    metric: Literal["shot_making", "gravity", "rebounding", "absence_risk", "similar"]
    min_n: int
    unit: str
    reason: str


class MetricNote(Strict):
    metric: Literal["shot_making", "gravity", "rebounding", "absence_risk", "similar"]
    verdict: str
    notes: list[str]
    sources: Sources


class CalibrationRow(Strict):
    """One row of a calibration table as the lane report prints it: predictions between `lo` and `hi`."""

    lo: float
    hi: float
    n: int
    predicted: Unit
    observed: Unit


class Baseline(Strict):
    """The expectation behind shot-making over expected (D47, D51): the lane's location and contest model. It was
    not the selected model. `ece` and `calibration` are its out-of-fold calibration error and table, copied from
    the lane report, so a reader can size how far its own error moves a player's value."""

    name: str = Field(min_length=1)
    selected: bool
    ece: Estimate
    calibration: list[CalibrationRow]
    source: Source


METRIC_KEYS = ("shot_making", "gravity", "rebounding", "absence_risk", "similar")


class PlayersFile(Strict):
    """Player views for the prep tab. Model predictions shown here come from out-of-fold rows; descriptive tables
    come from training games; held-out rows feed none of them. Players with no metric above its minimum sample are
    left out and counted in `n_hidden`. `checks` holds the reproduction checks of the two rebuilt outputs
    (absence, embeddings): a view that rests on one appears only when its check passed."""

    schema_version: SchemaVersion
    season: str
    window: str
    n_players: int
    n_shown: int
    n_hidden: int
    rules: list[MinSampleRule]
    metrics: list[MetricNote]
    shot_making_baseline: Baseline
    checks: list[RebuildCheck]
    players: list[PlayerView]
    sources: Sources

    @model_validator(mode="after")
    def _consistent(self):
        if self.n_shown != len(self.players) or self.n_players != self.n_shown + self.n_hidden:
            raise ValueError("n_shown must equal the number of players and n_players = n_shown + n_hidden")
        minimum = {rule.metric: rule.min_n for rule in self.rules}
        if sorted(minimum) != sorted(METRIC_KEYS) or len(self.rules) != len(METRIC_KEYS):
            raise ValueError("there is exactly one minimum-sample rule for each of the five metrics")
        if sorted(note.metric for note in self.metrics) != sorted(METRIC_KEYS):
            raise ValueError("there is exactly one note for each of the five metrics")
        for player in self.players:
            used = {"shot_making": player.shot_making.min_attempts, "gravity": player.gravity.min_possessions,
                    "rebounding": player.rebounding.min_chances, "absence_risk": player.absence_risk.min_games,
                    "similar": player.similar.min_possessions}
            if used != minimum:
                raise ValueError(f"player {player.player_id}: minimum samples differ from the rules")
            shown = (player.shot_making.shown, player.gravity.shown, player.rebounding.shown,
                     player.absence_risk.shown, player.similar.shown)
            if not any(shown):
                raise ValueError(f"player {player.player_id}: listed with no value above its minimum sample")
            if player.absence_risk.kind == "model_out_of_fold" and not _has_passed(self.checks, "absence"):
                raise ValueError("a model absence probability needs a passed absence check")
            if player.similar.available and not _has_passed(self.checks, "embeddings"):
                raise ValueError("similar players need a passed embeddings check")
        return self


# ---------------------------------------------------------------------------------------------- scorecard.json


ScorecardStatus = Literal["met", "not met", "reported", "not measurable"]


class ScorecardCell(Strict):
    """One target on one game. `text` is the line as the scorecard states it; `values` holds the same numbers
    for charts. A scorecard value is a point estimate unless `interval` says otherwise."""

    game: str
    split: Split
    status: ScorecardStatus
    passed: Optional[bool] = None
    text: str
    values: dict[str, float]
    n: Optional[float] = None
    n_unit: Optional[str] = None
    interval: Optional[str] = None

    @model_validator(mode="after")
    def _status(self):
        expected = {"met": True, "not met": False}.get(self.status)
        if self.passed is not expected:
            raise ValueError(f"status {self.status!r} does not agree with passed {self.passed}")
        return self


class ScorecardRow(Strict):
    key: str
    target: str
    cells: list[ScorecardCell]


class ScorecardGame(Strict):
    game: str
    game_id: str
    split: Split
    truth_frames: int
    testable: int
    untestable_share: float


class ScorecardBuild(Strict):
    """One build of the tracking pipeline. `role` is adopted for the build the project uses and candidate for a
    build that was measured and not adopted."""

    build: str
    role: Literal["adopted", "candidate"]
    note: str
    games: list[ScorecardGame]
    rows: list[ScorecardRow]
    caveats: list[str]


class PairedDifference(Strict):
    """Paired change between two builds on the observations both score. Positive means `build` is better than
    `baseline`."""

    game: str
    metric: str
    baseline: str
    build: str
    diff: float
    lo: float
    hi: float
    excludes_zero: bool

    @model_validator(mode="after")
    def _interval(self):
        if self.lo > self.hi:
            raise ValueError("lo must not exceed hi")
        if self.excludes_zero != (self.lo > 0 or self.hi < 0):
            raise ValueError("excludes_zero must say whether the interval lies on one side of zero")
        return self


class Scorecard(Strict):
    """The tracking scorecard against the Stage 1 targets. Exactly one build is the adopted one; a candidate is a
    build that was measured and not adopted. Every value is a point estimate."""

    schema_version: SchemaVersion
    decision: str
    builds: list[ScorecardBuild]
    paired: list[PairedDifference]
    method: dict[str, str]
    sources: Sources

    @model_validator(mode="after")
    def _one_adopted(self):
        if sum(build.role == "adopted" for build in self.builds) != 1:
            raise ValueError("exactly one build is the adopted build")
        return self


# ------------------------------------------------------------------------------------------------ results.json


VerdictClass = Literal["works", "no_signal", "failed", "inconclusive"]


class MetricValue(Strict):
    """A reported metric. `text` is copied from the report; the numbers are parsed from it and are null when the
    report gives no number."""

    text: str
    name: Optional[str] = None
    value: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    point_estimate: bool

    @model_validator(mode="after")
    def _interval(self):
        if (self.lo is None) != (self.hi is None):
            raise ValueError("lo and hi must be set together")
        if self.point_estimate != (self.lo is None):
            raise ValueError("point_estimate must be true exactly when the report states no interval")
        if self.lo is not None and self.lo > self.hi:
            raise ValueError("lo must not exceed hi")
        return self


class ReportRow(Strict):
    field: str
    value: str


# Lanes the plan requires to be shown as tested with no signal (C10).
NO_SIGNAL_LANES = ("decisions", "matchups")


class LaneResult(Strict):
    """One lane of the results table. `verdict` is the sentence of the hand-written summary of the results
    write-up, copied word for word and shown as a quote from `verdict_source`; `verdict_class` and the metrics
    come from the generated tables."""

    lane: str
    question: str
    selected_model: str
    status: str
    out_of_fold: MetricValue
    heldout: Optional[MetricValue] = None
    verdict: str
    verdict_source: str = Field(min_length=1)
    verdict_class: VerdictClass
    report: str = Field(min_length=1)
    rows: list[ReportRow]

    @model_validator(mode="after")
    def _no_signal_lanes(self):
        if self.lane in NO_SIGNAL_LANES and self.verdict_class not in ("no_signal", "failed"):
            raise ValueError(f"{self.lane} is shown as tested with no signal (C10), not as {self.verdict_class}")
        return self


class PendingLane(Strict):
    """A lane with no result yet: its report is fixture output, so no number from it is shown."""

    lane: str
    reason: str


class Results(Strict):
    schema_version: SchemaVersion
    freeze_tag: str
    generated: str
    lanes: list[LaneResult]
    pending: list[PendingLane]
    notes: list[str]
    sources: Sources


# ---------------------------------------------------------------------------------------------- demo/swap.json


class SwapItem(Strict):
    """One model output computed twice: from sensor positions and from broadcast positions."""

    kind: Literal["xfg", "epv", "ghosting"]
    window: str
    possession: Optional[int] = None
    key: str
    t: Optional[float] = None
    sensor: float
    broadcast: float
    abs_diff: float
    fill_share: Unit


class SwapSummary(Strict):
    """The comparison for one model. When the model's rebuild is missing or did not pass its check, `available`
    is false, n is 0, the numbers are null and `note` says why."""

    kind: Literal["xfg", "epv", "ghosting"]
    n: int
    mean_abs_diff: Optional[float] = None
    median_abs_diff: Optional[float] = None
    unit: str
    available: bool
    note: Optional[str] = None

    @model_validator(mode="after")
    def _available(self):
        if self.available and (self.n <= 0 or self.mean_abs_diff is None or self.median_abs_diff is None):
            raise ValueError(f"{self.kind}: an available comparison states n and both differences")
        if not self.available and (self.n != 0 or self.mean_abs_diff is not None
                                   or self.median_abs_diff is not None or not self.note):
            raise ValueError(f"{self.kind}: an unavailable comparison carries no number and says why")
        return self


class SwapPossession(Strict):
    """Fill share for one possession: the share of player-frames whose position came from the sensor data because
    the broadcast tracker had no matched box there."""

    possession: int
    window: str
    n_player_frames: int
    n_filled: int
    fill_share: Unit
    matched_p50_ft: Optional[float] = None


class DemoSwap(Strict):
    """The same frozen models run on broadcast positions and on sensor positions. An illustration (D16): it
    measures the effect of position error only, because identity, team and ball come from the sensor data.
    `match_ft` is the matching distance, read from the scorecard, and `max_gap_s` the longest gap between two
    broadcast samples that is interpolated (D50); neither was tuned. A model appears only with a passed check."""

    schema_version: SchemaVersion
    label: Literal["illustration"]
    game_id: str
    game: str
    build: str
    n_windows: int
    n_frames: int
    fill_share: Unit
    match_ft: float
    max_gap_s: float
    summaries: list[SwapSummary]
    items: list[SwapItem]
    possessions: list[SwapPossession]
    checks: list[RebuildCheck]
    method: list[str]
    sources: Sources

    @model_validator(mode="after")
    def _checked(self):
        kinds = [s.kind for s in self.summaries]
        if sorted(kinds) != sorted(MODEL_KINDS):
            raise ValueError("the swap has exactly one summary for each of xfg, epv and ghosting")
        for summary in self.summaries:
            used = summary.available or any(item.kind == summary.kind for item in self.items)
            if used and not _has_passed(self.checks, summary.kind):
                raise ValueError(f"{summary.kind}: shown without a passed rebuild check")
            if not summary.available and any(item.kind == summary.kind for item in self.items):
                raise ValueError(f"{summary.kind}: items without an available summary")
        return self


# -------------------------------------------------------------------------------------------- demo/frames.json


class DemoBox(Strict):
    """A detected box in clip pixels. `status` is matched when the box was paired with a sensor player within the
    matching distance, unmatched when it was not (the scorecard calls these ghost boxes; the word ghost is kept for
    the average-defense model)."""

    x: float
    y: float
    w: float
    h: float
    track: int
    status: Literal["matched", "unmatched"]
    player_id: Optional[int] = None


class DemoPoint(Strict):
    player_id: Optional[int] = None
    team: Optional[str] = None
    x: float
    y: float


class DemoFrame(Strict):
    """One sampled frame of a scene. `broadcast` holds court positions from the broadcast build, `truth` the
    sensor positions at the synced moment (empty when the frame has no synced moment), and `error_ft` the median
    distance between matched pairs."""

    t: float
    boxes: list[DemoBox]
    broadcast: list[DemoPoint]
    truth: list[DemoPoint]
    error_ft: Optional[float] = None


class DemoScene(Strict):
    """A short stretch of one demo window. `clip` is a file name under media-private (private build only);
    `still` is a path under public/ shown by the public build."""

    id: str
    window: str
    period: int
    start_clock_s: Optional[float] = None
    clip: str
    still: str
    still_t: float
    duration_s: float
    fps: float
    width: int
    height: int
    n_matched: int
    p50_error_ft: Optional[float] = None
    p90_error_ft: Optional[float] = None
    frames: list[DemoFrame]


class DemoFrames(Strict):
    """Scenes for the cold open. `scene_rule` states how the scenes were chosen; the rule reads no position
    error (D50), and a scene's error is shown beside the whole-game scorecard value, never alone."""

    schema_version: SchemaVersion
    game_id: str
    game: str
    build: str
    total_clip_s: float
    scene_rule: str = Field(min_length=1)
    scenes: list[DemoScene]
    notes: list[str]
    sources: Sources


# ------------------------------------------------------------------------------------------------ methods.json

# A number as text: a sign, digits with thousands commas, decimals and a percent mark. A digit that is part of a
# name (p50, E1, r23, gsw_phx_2016) is not a number: the look-behind skips it. web/scripts/check-methods.mjs
# reads numbers off the built page with the same pattern.
NUMBER_TEXT_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?")
ComparabilityClass = Literal["roughly", "not_directly", "none"]
ExperimentStatus = Literal["pre_registered", "development_met_test_pending", "not_adopted_development",
                           "adopted", "not_adopted_test", "passed_test", "adopted_research",
                           "ran_result_in_writeup", "decided_by_part", "adopted_research_not_contradicted",
                           "adopted_research_confirmed"]
# The state of one part of an experiment whose rule is decided part by part.
PartStatus = Literal["pre_registered", "confirmed", "development_met_test_not_met", "not_adopted_development",
                     "development_met_test_pending", "passed_test", "not_passed_test"]
# How a rule compares the end of an interval it reads with its threshold.
Comparison = Literal["below", "at_most", "above", "at_least"]
_COMPARE = {"below": lambda v, t: v < t, "at_most": lambda v, t: v <= t, "above": lambda v, t: v > t,
            "at_least": lambda v, t: v >= t}


# A stage's main rule may also require its range to hold a number of the file (`contains`: D113's consistency check)
# `not_wholly_above` is a confirmation read (D108, D121): the stage is met unless its interval lies wholly above the
# threshold, the only reading that stops an adoption ("contradiction"); the end shown is the lower end
StageComparison = Literal["below", "at_most", "above", "at_least", "contains", "not_wholly_above"]


def _is_number_text(text: Optional[str]) -> bool:
    return text is None or bool(re.fullmatch(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?", text))


class OurValue(Strict):
    """One number of ours, as text exactly as the report prints it (so trailing zeros stay), with the interval
    the report prints beside it on the same line. An estimate without an interval is a point estimate; a setting
    (a penalty chosen by cross-validation) is not an estimate and has no interval. `source` is the report under
    reports/ it was copied from, and `rows` says which rows the report computed it on. The label names what the
    number is; any number inside it is in the report too."""

    label: str = Field(min_length=1)
    value: str
    lo: Optional[str] = None
    hi: Optional[str] = None
    kind: Literal["estimate", "setting"]
    source: str = Field(min_length=1)
    rows: SourceRows

    @model_validator(mode="after")
    def _numbers(self):
        if not all(_is_number_text(t) for t in (self.value, self.lo, self.hi)):
            raise ValueError(f"{self.label}: value, lo and hi are numbers written as text")
        if (self.lo is None) != (self.hi is None):
            raise ValueError(f"{self.label}: lo and hi must be set together")
        if self.kind == "setting" and self.lo is not None:
            raise ValueError(f"{self.label}: a setting has no interval")
        return self


class Paper(Strict):
    """A paper a lane builds on. `cite`, `take` (what the lane takes from it) and `differs` (how ours differs, when
    the dossier gives it a column of its own; otherwise `take` says both) are copied word for word from the lane
    dossier; `url` is a link the dossier names."""

    cite: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://")
    take: str = Field(min_length=1)
    differs: Optional[str] = None


class Benchmark(Strict):
    """Our number next to the published one, one row of the benchmark table of the lane dossier. `published`,
    `paper`, `differs` and `comparability` are the dossier's cells word for word, so the verdict is the
    dossier's own; `comparability_class` sorts it into three kinds for the page. `ours` holds our numbers, each
    traced to a report. A row with no number of ours has an empty `ours` and may carry `ours_text`, a phrase of
    the dossier with no number in it. `published_values` are the published numbers of `published`, `where` the
    table or section of the paper, `url` a link to the paper (null when the row names no paper)."""

    ours: list[OurValue]
    ours_text: Optional[str] = None
    published: str = Field(min_length=1)
    published_values: list[str]
    paper: str = Field(min_length=1)
    where: Optional[str] = None
    url: Optional[str] = Field(default=None, pattern=r"^https?://")
    differs: str = Field(min_length=1)
    comparability: str = Field(min_length=1)
    comparability_class: ComparabilityClass

    @model_validator(mode="after")
    def _consistent(self):
        if self.ours and self.ours_text is not None:
            raise ValueError("a row with numbers of ours carries no ours_text")
        if self.ours_text is not None and re.search(r"\d", self.ours_text):
            raise ValueError("ours_text holds no number: a number of ours is an OurValue traced to a report")
        if not all(_is_number_text(v) for v in self.published_values):
            raise ValueError("published_values are numbers written as text")
        missing = [v for v in self.published_values if v not in NUMBER_TEXT_RE.findall(self.published)]
        if missing:
            raise ValueError(f"published values {missing} are not numbers of the published cell")
        return self


class StageValue(Strict):
    """A further number of a stage result, read from the result file: a value with its interval when the file
    gives one, else a point estimate."""

    label: str = Field(min_length=1)
    value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    point_estimate: bool
    # the file's reading of this value, checked against its pre-registered rule (genai-E2 H4's number failure rate)
    reading: Optional[str] = None

    @model_validator(mode="after")
    def _interval(self):
        if (self.lo is None) != (self.hi is None) or self.point_estimate != (self.lo is None):
            raise ValueError(f"{self.label}: an interval has lo and hi, a point estimate has neither")
        if self.lo is not None and self.lo > self.hi:
            raise ValueError(f"{self.label}: lo must not exceed hi")
        return self


class Condition(Strict):
    """A further condition of a stage's rule: a statistic with its 95 percent interval, met when the end of the
    interval it reads (`bound`), or its point value (`bound` point), compares with `threshold` as `comparison` says.
    A threshold keeps its sign ("at most -0.002" where the text says "improves by at least 0.002"). `label` names
    the statistic."""

    label: str = Field(min_length=1)
    value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    bound: Literal["lo", "hi", "point"]
    comparison: Comparison
    threshold: float
    met: bool

    @model_validator(mode="after")
    def _met(self):
        if (self.lo is None) != (self.hi is None) or (self.lo is None and self.bound != "point"):
            raise ValueError(f"{self.label}: a condition has both ends or neither, and with neither reads its point")
        if self.lo is not None and not (self.lo <= self.value <= self.hi):
            raise ValueError(f"{self.label}: the value lies inside its interval")
        end = {"lo": self.lo, "hi": self.hi, "point": self.value}[self.bound]
        if self.met != _COMPARE[self.comparison](end, self.threshold):
            raise ValueError(f"{self.label}: met says whether the {self.bound} end meets the threshold")
        return self


class GuardBucket(Strict):
    """One bucket of a guard: its paired difference with its 95 percent interval, and whether it reads as worse."""

    label: str = Field(min_length=1)
    value: float
    lo: float
    hi: float
    worse: bool

    @model_validator(mode="after")
    def _inside(self):
        if not (self.lo <= self.value <= self.hi):
            raise ValueError(f"{self.label}: the value lies inside its interval")
        return self


class BucketGuard(Strict):
    """A guard of a stage's rule over every bucket: a bucket is worse when the end of its interval (`bound`)
    compares with `threshold` as `comparison` says, and the guard is met when no bucket is worse."""

    label: str = Field(min_length=1)
    bound: Literal["lo", "hi"]
    comparison: Comparison
    threshold: float
    buckets: Annotated[list[GuardBucket], Field(min_length=1)]
    met: bool

    @model_validator(mode="after")
    def _worse(self):
        for bucket in self.buckets:
            end = bucket.lo if self.bound == "lo" else bucket.hi
            if bucket.worse != _COMPARE[self.comparison](end, self.threshold):
                raise ValueError(f"{bucket.label}: worse says whether its {self.bound} end is {self.comparison} "
                                 f"{self.threshold}")
        if self.met != (not any(bucket.worse for bucket in self.buckets)):
            raise ValueError("the guard is met when no bucket is worse")
        return self


class StageResult(Strict):
    """One stage of an experiment, read from its result file under reports/research/: development (out of fold on
    the development games) or test (the D71 test set, scored once); `design` is the file's own words for how the
    stage was run (its folds or its forward chaining). `diff` is the difference the adoption rule reads, with its
    95 percent interval (`interval` names the method); `diff` is null when the file states the interval only. The
    rule is met when `upper`, the end of the interval the rule reads (the upper end for at most and below, the lower
    end for at least and above), compares with `threshold` as `comparison` says, and every further condition
    (`conditions`) and every guard (`guard`, `guards`) is met too. `reading` is the file's or the write-up's own word
    for the outcome (confirms, inconclusive, passes, hurts), checked against the rule. `also_read` names further
    files whose numbers are reported beside the rule. Numbers are rounded once, here in the export."""

    stage: Literal["development", "test"]
    source: Source
    also_read: list[Source] = []
    design: Optional[str] = None
    metric: str = Field(min_length=1)
    rows: int
    rows_unit: str = Field(min_length=1)
    games: Optional[int] = None
    diff: Optional[float] = None
    lo: float
    hi: float
    interval: str = Field(min_length=1)
    threshold: float
    upper: float
    comparison: StageComparison = "at_most"
    conditions: list["Condition"] = []
    guard: Optional[BucketGuard] = None
    guards: list[BucketGuard] = []
    met: bool
    reading: Optional[str] = None
    values: list[StageValue]
    # the heading of a test stage scored on a test set of its own (the genai test players), never the D71 test games
    test_set: Optional[str] = None

    @model_validator(mode="after")
    def _rule(self):
        if self.lo > self.hi or (self.diff is not None and not (self.lo <= self.diff <= self.hi)):
            raise ValueError("the difference lies inside its interval")
        if self.comparison == "contains":
            # the range must hold the threshold; the end shown is the one nearer it, the end that decides
            end = self.lo if abs(self.threshold - self.lo) <= abs(self.threshold - self.hi) else self.hi
            main = self.lo <= self.threshold <= self.hi
        elif self.comparison == "not_wholly_above":
            if self.stage != "test":
                raise ValueError("a confirmation read (not wholly above) belongs to a test stage only")
            end, main = self.lo, self.lo <= self.threshold
            if (self.reading or "").startswith("confirms") and not self.hi < self.threshold:
                raise ValueError("a confirmation reads confirms only when its whole interval lies below the bar")
        else:
            end = self.lo if self.comparison in ("above", "at_least") else self.hi
            main = _COMPARE[self.comparison](self.upper, self.threshold)
        if self.upper != end:
            raise ValueError("the main rule reads the upper end for at most and below, the lower end for at least "
                             "and above, and the end nearer its number for contains")
        guards = ([self.guard] if self.guard else []) + list(self.guards)
        if self.met != (main and all(c.met for c in self.conditions) and all(g.met for g in guards)):
            raise ValueError("met says whether the bound meets the threshold and every further condition is met")
        if self.test_set is not None and (self.stage != "test" or "D71" in self.source.what
                                          or "test games" in self.test_set.lower()):
            raise ValueError("a test set of its own heads a test stage and names neither D71 nor test games")
        expected = "testset" if self.stage == "test" else "out_of_fold"
        if self.source.rows != expected or any(s.rows != expected for s in self.also_read):
            raise ValueError(f"a {self.stage} result reads rows {expected}")
        return self


class PostHocInterval(Strict):
    """A valid 95 percent interval computed after the result and its review (post hoc), which no verdict reads.
    `method` names how it was drawn; `source` is the file it was read from."""

    lo: float
    hi: float
    method: str = Field(min_length=1)
    source: Source

    @model_validator(mode="after")
    def _ordered(self):
        if self.lo > self.hi:
            raise ValueError("lo must not exceed hi")
        return self


class Hypothesis(Strict):
    """One pre-registered statement of a protocol experiment, read on its point value as pre-registered. `statement`
    is copied from the result file and `threshold` is its threshold, part of it. `holds` is null when the
    statistic is not a finite number, and `result` then reads undefined, never holds or fails. A pre-registered
    range that is not a valid confidence interval (PLAN.md D86) is never carried here: `interval` is the valid post
    hoc one, or null, and then the value is a point estimate."""

    label: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    value: Optional[float] = None
    holds: Optional[bool] = None
    result: Literal["holds", "fails", "undefined"]
    interval: Optional[PostHocInterval] = None
    point_estimate: bool

    @model_validator(mode="after")
    def _reading(self):
        if self.threshold not in self.statement:
            raise ValueError(f"{self.label}: the threshold is part of the statement")
        expected = "undefined" if self.holds is None else "holds" if self.holds else "fails"
        if self.result != expected or (self.value is None) != (self.holds is None):
            raise ValueError(f"{self.label}: a finite value holds or fails; no value reads undefined")
        if self.point_estimate != (self.interval is None) or (self.value is None and self.interval is not None):
            raise ValueError(f"{self.label}: a value has its valid interval or is a point estimate")
        if self.interval is not None and not (self.interval.lo <= self.value <= self.interval.hi):
            raise ValueError(f"{self.label}: the value lies inside its valid interval")
        return self


class AdoptionValue(Strict):
    """A number the adoption rule reads, with its valid post hoc interval or as a point estimate."""

    label: str = Field(min_length=1)
    value: float
    interval: Optional[PostHocInterval] = None
    point_estimate: bool

    @model_validator(mode="after")
    def _interval(self):
        if self.point_estimate != (self.interval is None):
            raise ValueError(f"{self.label}: a value has its valid interval or is a point estimate")
        if self.interval is not None and not (self.interval.lo <= self.value <= self.interval.hi):
            raise ValueError(f"{self.label}: the value lies inside its valid interval")
        return self


class ProtocolResult(Strict):
    """The development result of an experiment whose pre-registration is a protocol of hypotheses (statements on
    reliability and stability) rather than one paired difference against a threshold. `rule`, `adopted` and
    `reading` (the protocol's own conclusion) are copied from the result file. `caveat` and `not_scored` (why the
    test set was not scored, when it was not) are sentences of the experiment's write-up, word for word."""

    stage: Literal["development"]
    source: Source
    games: int
    rows: int
    rows_unit: str = Field(min_length=1)
    hypotheses: Annotated[list[Hypothesis], Field(min_length=1)]
    rule: str = Field(min_length=1)
    adoption: list[AdoptionValue]
    adopted: bool
    reading: str = Field(min_length=1)
    caveat: Optional[str] = None
    not_scored: list[str]

    @model_validator(mode="after")
    def _rows(self):
        if self.source.rows != "training":
            raise ValueError("a protocol result is a description of the development games (rows training)")
        return self


class ExperimentPart(Strict):
    """One part of an experiment whose pre-registration decides each part on its own (who shoots and make given
    xFG; each hypothesis of a model test). A part has its own stages, read like an experiment's: development decides
    and the test set confirms, or the test set alone decides (`stages` test_only). `status` follows from them;
    `test_not_scored` marks a test deliberately not spent after the development bar was missed."""

    label: str = Field(min_length=1)
    stages: Literal["development_and_test", "test_only"] = "development_and_test"
    development: Optional[StageResult] = None
    test: Optional[StageResult] = None
    test_not_scored: bool = False
    # sentences of the write-up quoted word for word beside this part
    notes: list[str] = []
    status: PartStatus

    @model_validator(mode="after")
    def _status(self):
        dev, test = self.development, self.test
        if (dev and dev.stage != "development") or (test and test.stage != "test"):
            raise ValueError(f"{self.label}: a stage result sits under its own stage")
        if test is not None and test.comparison == "not_wholly_above":
            raise ValueError(f"{self.label}: a confirmation read (not wholly above) confirms a whole experiment's "
                             "development adoption, never a part")
        if self.stages == "test_only":
            if dev is not None or self.test_not_scored:
                raise ValueError(f"{self.label}: a test-only part has no development result")
            expected = "pre_registered" if test is None else "passed_test" if test.met else "not_passed_test"
        else:
            if test is not None and dev is None:
                raise ValueError(f"{self.label}: the test set is scored after the development result exists")
            if self.test_not_scored and (dev is None or dev.met or test is not None):
                raise ValueError(f"{self.label}: a test left unscored belongs to a missed development bar")
            expected = ("pre_registered" if dev is None else "not_adopted_development" if not dev.met
                        else "development_met_test_pending" if test is None
                        else "confirmed" if test.met else "development_met_test_not_met")
        if self.status != expected:
            raise ValueError(f"{self.label}: status is {self.status!r}, the results say {expected!r}")
        return self


class Experiment(Strict):
    """A pre-registered experiment of a lane dossier. `title`, `hypothesis` and `threshold` (the adoption rule,
    fixed before any run) are copied from the dossier; `rank`, `what`, `gain`, `effort`, `test_data` and
    `backlog_status` are its row of docs/research/BACKLOG.md. `status` follows from the result files: no file is
    pre_registered (not run); a development result that meets the rule waits for its single test score; adoption
    needs the rule met on both. `writeup` is the report of an experiment that has run, under reports/research/.
    An experiment whose pre-registration has no verdict on the development games (a dry run only) is `stages`
    test_only: its single test score decides, and a pass is passed_test, not adopted, which would say the
    development bar was met too. `adopted_as` research_result marks a pass on both stages adopted as a research
    result only (the lane and the site do not change): its status is adopted_research, never adopted; a protocol or
    a test-only experiment never carries it. `result_in_writeup` marks an experiment that has run but whose result
    has a shape this page does not show yet: its status is ran_result_in_writeup and the page links the write-up.
    `test_not_scored` marks a development result that missed its bar and whose single test score was deliberately
    not spent (the write-up says so and the ledger has no entry): it is not_adopted_development with no test result.
    `writeup_notes` are sentences of the write-up's verdict paragraph, word for word
    (the narrow claim, a data-fault note).
    An experiment pre-registered as a protocol of hypotheses carries `protocol` in place of stage results: its
    adoption is read on the development games, and when the rule fails it is not_adopted_development, the same
    meaning as a paired difference that misses its bar there (no test score follows). A protocol whose rule holds
    on a test set that cannot be scored has no status yet and is refused."""

    id: str = Field(pattern=r"^[a-z]+-E\d+$")
    title: str = Field(min_length=1)
    what: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    rank: int = Field(ge=1)
    gain: int = Field(ge=1, le=5)
    effort: int = Field(ge=1, le=5)
    test_data: str = Field(min_length=1)
    backlog_status: str = Field(min_length=1)
    status: ExperimentStatus
    writeup: Optional[str] = None
    writeup_notes: list[str] = []
    stages: Literal["development_and_test", "test_only"] = "development_and_test"
    adopted_as: Optional[Literal["research_result"]] = None
    result_in_writeup: bool = False
    test_not_scored: bool = False
    parts: list[ExperimentPart] = []
    development: Optional[StageResult] = None
    test: Optional[StageResult] = None
    protocol: Optional[ProtocolResult] = None

    @model_validator(mode="after")
    def _status(self):
        dev, test = self.development, self.test
        # Checked before any shape returns early: a test left unscored belongs only to a development result that
        # missed its bar, with no test result, never to a result kept in its write-up, a protocol, a test-only
        # experiment or one decided part by part (each part carries its own flag).
        if self.test_not_scored and (self.result_in_writeup or self.protocol is not None or self.parts
                                     or self.stages == "test_only" or dev is None or dev.met or test is not None):
            raise ValueError(f"{self.id}: a test left unscored belongs to a development result that missed its bar, "
                             "with no test result")
        if self.parts or self.status == "decided_by_part":
            if not (self.parts and self.status == "decided_by_part" and self.writeup and dev is None and test is None
                    and self.protocol is None and not self.result_in_writeup and self.adopted_as is None
                    and not self.test_not_scored and self.stages == "development_and_test"):
                raise ValueError(f"{self.id}: an experiment decided part by part has its parts, its write-up, no "
                                 "stage or protocol of its own, and the status decided_by_part")
            if len({part.label for part in self.parts}) != len(self.parts):
                raise ValueError(f"{self.id}: every part has its own label")
            return self
        if self.result_in_writeup or self.status == "ran_result_in_writeup":
            if not (self.result_in_writeup and self.status == "ran_result_in_writeup" and self.writeup
                    and dev is None and test is None and self.protocol is None and self.adopted_as is None):
                raise ValueError(f"{self.id}: a result kept in its write-up has the write-up, no result here, and the "
                                 "status ran_result_in_writeup")
            return self
        if self.status == "pre_registered" and self.writeup is not None:
            raise ValueError(f"{self.id}: a write-up belongs to an experiment that has run, not a pre-registered one")
        if self.adopted_as is not None and (self.protocol is not None or self.stages == "test_only"):
            raise ValueError(f"{self.id}: adopted as a research result belongs to a pass on both stages, not to a "
                             "protocol or a test-only experiment")
        if self.protocol is not None:
            if dev is not None or test is not None:
                raise ValueError(f"{self.id}: a protocol result replaces the stage results, never sits beside them")
            if self.protocol.adopted and self.protocol.not_scored:
                raise ValueError(f"{self.id}: an adopted protocol whose test set cannot be scored has no status")
            expected = "development_met_test_pending" if self.protocol.adopted else "not_adopted_development"
            if self.status != expected:
                raise ValueError(f"{self.id}: status is {self.status!r}, the results say {expected!r}")
            return self
        if (dev and dev.stage != "development") or (test and test.stage != "test"):
            raise ValueError(f"{self.id}: a stage result sits under its own stage")
        if self.stages == "test_only":
            if dev is not None or self.protocol is not None:
                raise ValueError(f"{self.id}: a test-only experiment has no development verdict")
            if test is not None and test.comparison == "not_wholly_above":
                raise ValueError(f"{self.id}: a confirmation read (not wholly above) needs the development adoption "
                                 "it confirms, so never a test-only experiment")
            expected = "pre_registered" if test is None else "passed_test" if test.met else "not_adopted_test"
            if self.status != expected:
                raise ValueError(f"{self.id}: status is {self.status!r}, the results say {expected!r}")
            return self
        if test is not None and dev is None:
            raise ValueError(f"{self.id}: the test set is scored after the development result exists")
        if self.test_not_scored and (dev is None or dev.met or test is not None):
            raise ValueError(f"{self.id}: a test left unscored belongs to a development result that missed its bar, "
                             "with no test result")
        expected = ("pre_registered" if dev is None else "not_adopted_development" if not dev.met
                    else "development_met_test_pending" if test is None
                    else ("adopted_research" if self.adopted_as == "research_result" else "adopted") if test.met
                    else "not_adopted_test")
        if test is not None and test.met and test.comparison == "not_wholly_above" and dev is not None and dev.met:
            # the test set did not contradict a development adoption: never shown as a pass on the test set, and
            # only ever a research result
            if self.adopted_as != "research_result":
                raise ValueError(f"{self.id}: a confirmation read on the test set adopts a research result only")
            # D108's "confirms" (the interval wholly below the bar) is told apart from a test that only did not
            # contradict, and from a test set that met the bar itself (adopted_research)
            expected = ("adopted_research_confirmed" if (test.reading or "").startswith("confirms")
                        else "adopted_research_not_contradicted")
        if self.adopted_as is not None and expected not in ("adopted_research", "adopted_research_not_contradicted",
                                                            "adopted_research_confirmed"):
            raise ValueError(f"{self.id}: adopted as a research result needs the bar met on both stages")
        if self.status != expected:
            raise ValueError(f"{self.id}: status is {self.status!r}, the results say {expected!r}")
        return self


class MethodsLane(Strict):
    """One lane: its dossier, the papers it builds on, its benchmark rows, the dossier's notes on what has no
    published counterpart (word for word), and its pre-registered experiments."""

    lane: str = Field(min_length=1)
    dossier: str = Field(min_length=1)
    report: str = Field(min_length=1)
    foundations: Annotated[list[Paper], Field(min_length=1)]
    benchmarks: list[Benchmark]
    notes: list[str]
    experiments: list[Experiment]


class MethodsFile(Strict):
    """The methods and literature page (PLAN.md D72). Every text is copied from a lane dossier or from
    docs/research/BACKLOG.md, and every number of ours from a report, as export/research_methods.py checks
    before it writes this file. `written` is the date of the dossiers, `backlog` the ranked list of every
    proposal."""

    schema_version: CurrentVersion
    written: str
    backlog: str = Field(min_length=1)
    lanes: list[MethodsLane]
    sources: Sources

    @model_validator(mode="after")
    def _unique(self):
        lanes = [lane.lane for lane in self.lanes]
        ids = [e.id for lane in self.lanes for e in lane.experiments]
        if len(set(lanes)) != len(lanes) or len(set(ids)) != len(ids):
            raise ValueError("every lane and every experiment appears once")
        ranks = sorted(e.rank for lane in self.lanes for e in lane.experiments)
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("the backlog ranks run from 1 without a gap")
        return self


# ----------------------------------------------------------------------------------------- genai/<kind>.json


GenaiKind = Literal["scouting", "possession"]
GENAI_FILES = {"scouting": "genai/scouting.json", "possession": "genai/possession.json"}


class GeneratedNumber(Strict):
    """One number of a text: as the text writes it, and the field of the item's input whose shown value it
    equals (export/genai.py, check_numbers)."""

    shown: str = Field(min_length=1)
    field: str = Field(min_length=1)


class GeneratedText(Strict):
    """One machine-written text. `key` is the player slug (scouting) or <game slug>/<possession> (possession).
    `input_sha256` is the hash of the item's input as export/genai.py builds it from this data root; a text whose
    input changed since it was written is stale and refused. `model` is the model id that wrote it ("fixture" for
    the deterministic texts of a synthetic root), `prompt_version` the version of the prompt it was given, and
    `call` how the model was called (thinking, effort, token limit), when that is known."""

    key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    input_sha256: str = Field(min_length=64, max_length=64)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    numbers: list[GeneratedNumber]
    call: Optional[str] = None


class GeneratedTexts(Strict):
    """Machine-written texts of one kind (PLAN.md P3, D76). The words are the model's; every number in them is a
    value of the page the text sits on, as the page rounds it, and each text passed every check of
    export/genai.py when it was written. `validate_tree` runs those checks again against the data root, so a text
    with a wrong or stale number cannot be in a valid root. A page shows a text only when it has one: no text is
    required."""

    schema_version: CurrentVersion
    kind: GenaiKind
    label: Literal["machine-written"]
    checks: list[str] = Field(min_length=1)
    items: list[GeneratedText]
    notes: list[str]
    sources: Sources

    @model_validator(mode="after")
    def _unique(self):
        keys = [item.key for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("a key appears twice among the texts")
        return self


# --------------------------------------------------------------------------------------- files and validation

FIXTURE_SOURCE = "export/sample.py"
FREEZE_TAG = "freeze-v1"

# Path pattern under the data root, and the model each file must validate against.
FILES: tuple[tuple[str, type[Strict]], ...] = (
    ("manifest.json", Manifest),
    ("games/<slug>/possessions.json", PossessionIndex),
    ("games/<slug>/<possession>.json", Possession),
    ("players.json", PlayersFile),
    ("scorecard.json", Scorecard),
    ("results.json", Results),
    ("demo/swap.json", DemoSwap),
    ("demo/frames.json", DemoFrames),
    ("methods.json", MethodsFile),
    ("genai/<kind>.json", GeneratedTexts),  # genai/scouting.json and genai/possession.json, each optional
)
MODELS: dict[str, type[Strict]] = {model.__name__: model for _, model in FILES}

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_POSSESSION_FILE_RE = re.compile(r"^\d+\.json$")


def model_for(relpath: str) -> Optional[type[Strict]]:
    """The model a file under the data root must validate against, or None for a path the contract does not know."""
    parts = Path(relpath).as_posix().split("/")
    fixed = {pattern: model for pattern, model in FILES if "<" not in pattern}
    if "/".join(parts) in fixed:
        return fixed["/".join(parts)]
    if len(parts) == 2 and parts[0] == "genai" and "/".join(parts) in GENAI_FILES.values():
        return GeneratedTexts
    if len(parts) == 3 and parts[0] == "games" and _SLUG_RE.match(parts[1]):
        if parts[2] == "possessions.json":
            return PossessionIndex
        if _POSSESSION_FILE_RE.match(parts[2]):
            return Possession
    return None


def json_schema() -> dict:
    """One JSON Schema document holding every contract model under $defs."""
    defs: dict = {}
    for name, model in MODELS.items():
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(schema.pop("$defs", {}))
        defs[name] = schema
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "contract.schema.json",
        "title": "SiteContract",
        "description": "Generated by export/schemas.py. Do not edit by hand.",
        "x-schema-version": SCHEMA_VERSION,
        "x-compatible-versions": list(COMPATIBLE_VERSIONS),
        "x-files": {pattern: model.__name__ for pattern, model in FILES},
        "type": "object",
        "additionalProperties": False,
        "properties": {name: {"$ref": f"#/$defs/{name}"} for name in MODELS},
        "$defs": dict(sorted(defs.items())),
    }


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2, sort_keys=False) + "\n"


def validate_file(root: Path, relpath: str) -> list[str]:
    """Validate one file under `root`. Returns a list of problems, empty when the file is valid."""
    return _validate_file(Path(root), relpath)[0]


def _validate_file(root: Path, relpath: str):
    """The problems of one file and, when it parsed, the validated object."""
    model = model_for(relpath)
    if model is None:
        return [f"{relpath}: not a file the contract knows"], None
    text = (root / relpath).read_text(encoding="utf-8")
    problems = [f"{relpath}: private string {found!r}" for found in find_private(text)]
    try:
        obj = model.model_validate_json(text)
    except Exception as exc:  # pydantic.ValidationError, or malformed JSON
        problems.append(f"{relpath}: {type(exc).__name__}: {str(exc)[:500]}")
        return problems, None
    parts = Path(relpath).as_posix().split("/")
    if isinstance(obj, Possession):
        if obj.game_slug != parts[1] or f"{obj.possession}.json" != parts[2]:
            problems.append(f"{relpath}: holds possession {obj.possession} of {obj.game_slug}")
    if isinstance(obj, PossessionIndex) and obj.game.slug != parts[1]:
        problems.append(f"{relpath}: holds the index of {obj.game.slug}")
    if isinstance(obj, GeneratedTexts) and GENAI_FILES[obj.kind] != relpath:
        problems.append(f"{relpath}: holds the {obj.kind} texts, which belong in {GENAI_FILES[obj.kind]}")
    return problems, obj


def _all_sources(obj) -> list:
    """Every Source inside a validated file."""
    found = []

    def walk(value):
        if isinstance(value, Source):
            found.append(value)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(obj)
    return found


def _content_problems(relpath: str, obj, synthetic: bool) -> list[str]:
    """Rules that depend on whether the data root is synthetic. Real data: no fixture source, held-out games only
    on the review side, no held-out row on the player side, the frozen tag, no fixture lane among the results.
    Synthetic data: every source is the fixture generator."""
    problems = []
    sources = _all_sources(obj)
    fixture = [s for s in sources if s.path == FIXTURE_SOURCE]
    if synthetic and len(fixture) != len(sources):
        problems.append(f"{relpath}: a synthetic data root names only {FIXTURE_SOURCE} as its source")
    if synthetic:
        return problems
    if fixture:
        problems.append(f"{relpath}: fixture values in a data root that is not marked synthetic")
    if isinstance(obj, (Possession, PossessionIndex)):
        bad = sorted({s.path for s in sources if s.rows in ("out_of_fold", "training")})
        if bad:
            problems.append(f"{relpath}: the possession review reads held-out rows only, not {bad}")
        game_id = obj.game_id if isinstance(obj, Possession) else obj.game.game_id
        if not _is_heldout(game_id):
            problems.append(f"{relpath}: {game_id} is not a held-out game")
    if isinstance(obj, PlayersFile) or (isinstance(obj, GeneratedTexts) and obj.kind == "scouting"):
        bad = sorted({s.path for s in sources if s.rows == "heldout"})
        if bad:
            problems.append(f"{relpath}: player values never read held-out rows: {bad}")
    if isinstance(obj, GeneratedTexts) and obj.kind == "possession":
        bad = sorted({s.path for s in sources if s.rows in ("out_of_fold", "training")})
        if bad:
            problems.append(f"{relpath}: the possession review reads held-out rows only, not {bad}")
    if isinstance(obj, GeneratedTexts):
        fixture = sorted({item.key for item in obj.items if item.model == "fixture"})
        if fixture:
            problems.append(f"{relpath}: fixture texts in a data root that is not marked synthetic: {fixture[:3]}")
    if isinstance(obj, Manifest):
        if obj.freeze_tag != FREEZE_TAG:
            problems.append(f"{relpath}: freeze_tag is {obj.freeze_tag!r}, not {FREEZE_TAG!r}")
        problems += [f"{relpath}: {g.game_id} is not a held-out game" for g in obj.games if not _is_heldout(g.game_id)]
    if isinstance(obj, Results):
        for lane in obj.lanes:
            if "fixture" in lane.status.lower():
                problems.append(f"{relpath}: lane {lane.lane} is fixture output and cannot be shown as a result")
            if not lane.report.startswith("reports/"):
                problems.append(f"{relpath}: lane {lane.lane} cites {lane.report!r}, not a file under reports/")
    if isinstance(obj, MethodsFile):
        for lane in obj.lanes:
            if not lane.dossier.startswith("docs/research/") or not lane.report.startswith("reports/"):
                problems.append(f"{relpath}: lane {lane.lane} names a dossier outside docs/research/ or a report "
                                "outside reports/")
            for row in lane.benchmarks:
                bad = sorted({v.source for v in row.ours if not v.source.startswith("reports/") or v.rows == "testset"})
                if bad:
                    problems.append(f"{relpath}: lane {lane.lane}: a number of ours comes from a report under "
                                    f"reports/ and never from the test set, not {bad}")
            for experiment in lane.experiments:
                if experiment.writeup is not None and not experiment.writeup.startswith("reports/research/"):
                    problems.append(f"{relpath}: {experiment.id} names its write-up {experiment.writeup!r}, not a "
                                    "file under reports/research/")
                protocol = experiment.protocol
                paths = [] if protocol is None else [protocol.source.path] + [
                    i.source.path for i in [h.interval for h in protocol.hypotheses] + [a.interval for a in
                                                                                       protocol.adoption] if i]
                for path in paths:
                    if not path.startswith("reports/research/"):
                        problems.append(f"{relpath}: {experiment.id} reads its protocol from {path!r}, not a file "
                                        "under reports/research/")
                for stage in (experiment.development, experiment.test):
                    if stage is not None and not stage.source.path.startswith("reports/research/"):
                        problems.append(f"{relpath}: {experiment.id} reads its result from "
                                        f"{stage.source.path!r}, not a file under reports/research/")
    return problems


def _is_heldout(game_id: str) -> bool:
    from analytics.eval import is_heldout_game  # the frozen split helper; it reads sportvu/splits.json
    return bool(is_heldout_game(game_id))


def _manifest_problems(root: Path, seen: set) -> list[str]:
    """The manifest must list every file on disk, exactly once, with its real size and hash."""
    import hashlib

    try:
        manifest = Manifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return []  # already reported by validate_file
    problems, covered = [], set()
    for entry in manifest.files:
        if entry.count == 1 and "<" not in entry.path:
            members = [entry.path]
        else:
            folder = entry.path.rsplit("/", 1)[0]
            members = sorted(rel for rel in seen if rel.startswith(folder + "/") and rel.endswith(".json")
                             and (model_for(rel).__name__ if model_for(rel) else None) == entry.model)
        if len(members) != entry.count:
            problems.append(f"manifest.json: {entry.path} lists {entry.count} file(s), {len(members)} on disk")
        missing = [rel for rel in members if rel not in seen]
        if missing:
            problems.append(f"manifest.json: {entry.path} is listed and not on disk")
            continue
        hashes = [hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in members]
        size = sum((root / rel).stat().st_size for rel in members)
        digest = hashes[0] if len(members) == 1 and entry.count == 1 and "<" not in entry.path else \
            hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
        if size != entry.bytes or digest != entry.sha256:
            problems.append(f"manifest.json: {entry.path} does not match the files on disk (size or hash)")
        covered.update(members)
    for rel in sorted(seen - covered - {"manifest.json"}):
        problems.append(f"manifest.json: {rel} is on disk and not listed")
    return problems


def content_id(files) -> str:
    """The hash a build id is made from: every manifest entry's path and sha256, in the manifest's order. A change
    to any listed file, a machine-written text included, gives a new id."""
    import hashlib

    return hashlib.sha256("\n".join(f"{f.path}:{f.sha256}" for f in files).encode()).hexdigest()


def _genai_problems(parsed: dict) -> list[str]:
    """Every machine-written text checked again against the data root it sits in (export/genai.py): its input is
    rebuilt from the root, its hash must match, and the text must pass every check, numbers first."""
    texts = {rel: obj for rel, obj in parsed.items() if isinstance(obj, GeneratedTexts)}
    if not texts:
        return []
    from export import genai  # it builds on this module, so it is imported only when a root holds texts

    problems = []
    for rel, obj in texts.items():
        problems += [f"{rel}: {line}" for line in genai.root_problems(obj, parsed)]
    return problems


def validate_tree(root, allow_missing=()) -> list[str]:
    """Validate every file under a data root: each must be known to the contract and valid against its model and
    sit at the path its content names, the fixed files must all exist (except the paths in `allow_missing`), the
    manifest must list every file with its real size and hash, and no file may hold a local path or an email
    address. Only the demo files may be allowed to be missing, and only files in IGNORED_NAMES are skipped.
    With a real (not synthetic) manifest the content rules apply too: held-out games only on the review side, no
    held-out row behind a player value, no fixture value anywhere."""
    root = Path(root)
    not_staged = sorted(set(allow_missing) - set(STAGED_FILES))
    if not_staged:
        raise ValueError(f"only {list(STAGED_FILES)} may be missing, not {not_staged}")
    problems: list[str] = []
    seen = set()
    parsed = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path.name in IGNORED_NAMES:
            continue  # dropped by the operating system; the site copies contract files only
        seen.add(rel)
        found, obj = _validate_file(root, rel)
        problems += found
        if obj is not None:
            parsed[rel] = obj
    manifest = parsed.get("manifest.json")
    if manifest is not None:
        problems += _manifest_problems(root, seen)
        for rel, obj in parsed.items():
            problems += _content_problems(rel, obj, manifest.synthetic)
        problems += _genai_problems(parsed)
    for pattern, _ in FILES:
        if "<" not in pattern and pattern not in seen and pattern not in allow_missing:
            problems.append(f"{pattern}: missing")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the JSON Schema of the site data contract, or check it.")
    parser.add_argument("--out", help="path of the schema file to write")
    parser.add_argument("--check", help="path of a schema file that must equal the current schema")
    parser.add_argument("--validate", help="a data root (web/data) to validate against the contract")
    parser.add_argument("--allow-missing", nargs="*", default=[], choices=list(STAGED_FILES),
                        help="demo files that may be absent while the demo bridge has not run")
    args = parser.parse_args(argv)
    status = 0
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(schema_text(), encoding="utf-8")
        print(f"schema {SCHEMA_VERSION} written: {len(MODELS)} file models")
    if args.check:
        current = Path(args.check).read_text(encoding="utf-8") if Path(args.check).exists() else ""
        if current != schema_text():
            print("schema file is stale: run python3 -m export.schemas --out <path>")
            status = 1
        else:
            print("schema file is current")
    if args.validate:
        problems = validate_tree(args.validate, allow_missing=tuple(args.allow_missing))
        for line in problems[:50]:
            print(line)
        print(f"{len(problems)} problem(s)" if problems else "data root is valid")
        status = status or (1 if problems else 0)
    return status


if __name__ == "__main__":
    # Run the imported module, not this __main__ copy: export/genai.py imports export.schemas, and the checks of a
    # machine-written text must meet the same contract classes the validated files were read with.
    from export import schemas as _contract

    raise SystemExit(_contract.main())
