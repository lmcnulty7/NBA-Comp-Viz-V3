"""export/schemas.py: the site data contract (PLAN.md, Site structure; D31).

Python produces data, the web only reads it. The pydantic models below are the contract for every file under
web/data. `python3 -m export.schemas --out web/lib/types/contract.schema.json` writes the JSON Schema; the
TypeScript types in web/lib/types are generated from that file and never edited by hand. The same shapes are meant
to come from a live service later, so nothing here depends on how the files were produced.

Rules the models enforce:
- every file carries the schema version;
- every value that could be read as a result carries its sources, and says whether it is held out;
- an estimate either has an interval or says it is a point estimate;
- no absolute local path and no email address can appear in any string (find_private).

Units: court positions are in feet in the SportVU frame, x from 0 to 94 along the court, y from 0 to 50 across it,
rims at (5.25, 25) and (88.75, 25). Clocks are in seconds. Times `t` are seconds from the start of the possession
or of the clip.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = "1.0.0"
SchemaVersion = Literal["1.0.0"]

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


class Source(Strict):
    """Where a value comes from. `path` is relative to the repository (for example reports/xfg.md) or starts with
    $DATA_ROOT for a data table. `heldout` is true when only held-out rows were read, false when only training-fold
    rows were read, and null when the source is not split by fold."""

    path: str
    what: str
    heldout: Optional[bool] = None


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
    sources: list[Source]


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

    schema_version: SchemaVersion
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
    among all held-out attempts, with the edges given once in the index."""

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
    xfg: Optional[float] = None
    xfg_bucket: Optional[int] = None
    epv_start: Optional[float] = None
    epv_end: Optional[float] = None
    has_ghosts: bool
    featured: bool
    demo: bool


class PossessionIndex(Strict):
    schema_version: SchemaVersion
    game: GameInfo
    xfg_bucket_edges: list[float]
    possessions: list[PossessionSummary]
    notes: list[str]
    sources: list[Source]


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
    """A field-goal attempt in the possession. `xfg` is the frozen model's probability that it goes in. Where the
    release was not located in the tracking, the time and the location fall back to the play-by-play and the shot
    chart and say so."""

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
    xfg: Optional[float] = None
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
    sources: list[Source]

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


class ShotMaking(Strict):
    """Makes above expected per 100 attempts, from out-of-fold rows. The expectation is the lane's location and
    contest model, which does not know the shooter. The frozen xFG model includes a shooter prior, so its own
    residual is near zero by construction and is given only for reference."""

    shown: bool
    attempts: int
    min_attempts: int
    over_expected_per_100: Optional[Estimate] = None
    fg_pct: Optional[float] = None
    expected_fg_pct: Optional[float] = None
    frozen_model_residual_per_100: Optional[float] = None

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("shot_making", self.shown, self.attempts, self.min_attempts, self.over_expected_per_100)
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

    @model_validator(mode="after")
    def _shown(self):
        _check_shown("rebounding", self.shown, self.chances, self.min_chances, self.rate_per_100)
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
    sources: list[Source]


class PlayersFile(Strict):
    """Player views for the prep tab. Every value comes from training-fold outputs; held-out rows feed none of
    them. Players with no metric above its minimum sample are left out and counted in `n_hidden`."""

    schema_version: SchemaVersion
    season: str
    window: str
    n_players: int
    n_shown: int
    n_hidden: int
    rules: list[MinSampleRule]
    metrics: list[MetricNote]
    players: list[PlayerView]
    sources: list[Source]


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
    schema_version: SchemaVersion
    decision: str
    builds: list[ScorecardBuild]
    paired: list[PairedDifference]
    method: dict[str, str]
    sources: list[Source]


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


class LaneResult(Strict):
    lane: str
    question: str
    selected_model: str
    status: str
    out_of_fold: MetricValue
    heldout: Optional[MetricValue] = None
    verdict: str
    verdict_class: VerdictClass
    report: str
    rows: list[ReportRow]


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
    sources: list[Source]


# ---------------------------------------------------------------------------------------------- demo/swap.json


class ModelCheck(Strict):
    """Record that a rebuilt model object is the frozen model: its predictions on sensor positions were compared
    with the stored held-out predictions. No outcome is read by this check."""

    model: Literal["xfg", "epv", "ghosting"]
    n: int
    max_abs_diff: float
    tolerance: float
    passed: bool
    unit: str


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
    fill_share: float


class SwapSummary(Strict):
    kind: Literal["xfg", "epv", "ghosting"]
    n: int
    mean_abs_diff: float
    median_abs_diff: float
    unit: str
    available: bool
    note: Optional[str] = None


class SwapPossession(Strict):
    """Fill share for one possession: the share of player-frames whose position came from the sensor data because
    the broadcast tracker had no matched box there."""

    possession: int
    window: str
    n_player_frames: int
    n_filled: int
    fill_share: float
    matched_p50_ft: Optional[float] = None


class DemoSwap(Strict):
    """The same frozen models run on broadcast positions and on sensor positions. An illustration (D16): it
    measures the effect of position error only, because identity, team and ball come from the sensor data."""

    schema_version: SchemaVersion
    label: Literal["illustration"]
    game_id: str
    game: str
    build: str
    n_windows: int
    n_frames: int
    fill_share: float
    summaries: list[SwapSummary]
    items: list[SwapItem]
    possessions: list[SwapPossession]
    checks: list[ModelCheck]
    method: list[str]
    sources: list[Source]


# -------------------------------------------------------------------------------------------- demo/frames.json


class DemoBox(Strict):
    """A detected box in clip pixels. `status` is matched when the box was paired with a sensor player within the
    matching distance, ghost when it was not."""

    x: float
    y: float
    w: float
    h: float
    track: int
    status: Literal["matched", "ghost"]
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
    schema_version: SchemaVersion
    game_id: str
    game: str
    build: str
    total_clip_s: float
    scenes: list[DemoScene]
    notes: list[str]
    sources: list[Source]


# --------------------------------------------------------------------------------------- files and validation

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
    model = model_for(relpath)
    if model is None:
        return [f"{relpath}: not a file the contract knows"]
    text = (root / relpath).read_text(encoding="utf-8")
    problems = [f"{relpath}: private string {found!r}" for found in find_private(text)]
    try:
        obj = model.model_validate_json(text)
    except Exception as exc:  # pydantic.ValidationError, or malformed JSON
        problems.append(f"{relpath}: {type(exc).__name__}: {str(exc)[:500]}")
        return problems
    parts = Path(relpath).as_posix().split("/")
    if isinstance(obj, Possession):
        if obj.game_slug != parts[1] or f"{obj.possession}.json" != parts[2]:
            problems.append(f"{relpath}: holds possession {obj.possession} of {obj.game_slug}")
    if isinstance(obj, PossessionIndex) and obj.game.slug != parts[1]:
        problems.append(f"{relpath}: holds the index of {obj.game.slug}")
    return problems


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


def validate_tree(root, allow_missing=()) -> list[str]:
    """Validate every file under a data root: each must be known to the contract and valid against its model and
    sit at the path its content names, the fixed files must all exist (except the paths in `allow_missing`), the
    manifest must list every file with its real size and hash, and no file may hold a local path or an email
    address. Only the demo files may be allowed to be missing, and only files in IGNORED_NAMES are skipped."""
    root = Path(root)
    not_staged = sorted(set(allow_missing) - set(STAGED_FILES))
    if not_staged:
        raise ValueError(f"only {list(STAGED_FILES)} may be missing, not {not_staged}")
    problems: list[str] = []
    seen = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path.name in IGNORED_NAMES:
            continue  # dropped by the operating system; the site copies contract files only
        seen.add(rel)
        problems += validate_file(root, rel)
    if "manifest.json" in seen:
        problems += _manifest_problems(root, seen)
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
    raise SystemExit(main())
