"""export/site_games.py: the possession review data (PLAN.md M1, C8, D28, D48, D52, D53).

For each held-out game this builds games/<slug>/<possession>.json for every possession and, after the featured
possessions are drawn, games/<slug>/possessions.json. Only held-out games are read: a game id that
analytics.eval.is_heldout_game does not accept is refused, and every model output (xFG, EPV, ghosts) must be a
held-out row (fold -1), so every prediction shown is out of sample. Tracking frames are read one game at a time.

What a possession file holds, and where it comes from:
  frames   every fifth frame from the possession's first frame (np.arange(start_frame, end_frame + 1, 5)): the grid
           of the stored ghost rows, so ghosts align by frame. t = (frame - start_frame) / 25. Positions to 0.1 ft.
           The ball is three columns on every row of the tracking table, not a row. A player with no row in a
           frame is null there.
  attack   the rim the offense attacks is not in the tracking data. It is inferred per team and period by the
           frozen helper analytics.decisions.features.baskets_by_team_period and checked against where the ball
           ends up.
  EPV      the stored states of the possession, value = epv_points as stored (never clipped), about one a second.
  ghosts   the stored 1.0 s rows of ghost_frames. A point more than 10 ft outside the court is dropped (a display
           rule, counted in dropped_points). The featured rule counts a frame as covered by the stored rows, before
           that drop, so it never reads how far off a ghost is.
  shots    the held-out rows of xfg_predictions whose play-by-play event lies in the possession. Time and place
           come from the release frame when the shooter's release was located; otherwise the time is the
           play-by-play frame (the logged outcome) and the place is the shot chart marker, used only in a game
           where the chart's orientation was verified against tracked release positions.
An xFG, an EPV value and a ghost position are point estimates: the frozen lanes store no interval for a single
prediction (D53).

Nothing here fits, tunes or scores a model. The only frozen code that runs is read-only helpers: the attack
direction vote, the shot chart cleaning and clock match of the xFG lane, and its rule for whether a shot's contest
features were measured.
"""

import json
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema as contract_tables
from analytics.decisions.features import baskets_by_team_period
from analytics.eval import is_heldout_game
from analytics.possessions import GAP_S, game_meta
from analytics.xfg import features as xfg_features
from analytics.xfg import validate as xfg_validate
from export import schemas as S
from export import site_common as C
from export.site_players import name_tokens
from sportvu import splits

ExportProblem = C.ExportProblem

FRAME_COLUMNS = ["frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z", "team_id",
                 "player_id", "x", "y"]
LEVEL_COLUMNS = ("game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z")   # one value per frame
GHOST_HORIZON_S = 1.0             # the stored horizon that is visibly apart from the real defender
GHOST_MARGIN_FT = 10.0            # a ghost point farther outside the court than this is not drawn
N_ON_COURT = 10
RESULTS = typing.get_args(S.PossessionResult)
SHOT_RESULTS = ("shot_made", "shot_missed")

# Featured possessions (PLAN.md D48, D52).
FEATURED_SEED = 0
FEATURED_MIN_EPV_STATES = 5
FEATURED_MIN_GHOST_SHARE = 0.8
FEATURED_MIN_S, FEATURED_MAX_S = 6.0, 24.0
N_FIFTHS = 5

# Shot chart orientation check (positions only, no outcome).
ORIENT_MIN_OFF_MIDDLE_FT = 5.0    # a shot nearer the middle of the court than this cannot tell the two sides apart
ORIENT_MIN_SHOTS = 10
ORIENT_MIN_SHARE = 0.9

# Attack direction check against the ball.
BALL_CHECK_MIN_FRAMES = 11        # two seconds of play at 5 frames a second
BALL_CHECK_MIN_VOTES = 5


# ------------------------------------------------------------------------------------------------ games


@dataclass
class Game:
    game_id: str
    slug: str
    label: str
    date: str
    home: str
    away: str
    split: str

    def info(self) -> S.GameInfo:
        return S.GameInfo(slug=self.slug, game_id=self.game_id, label=self.label, date=self.date, home=self.home,
                          away=self.away, split=self.split)


def game_of(game_id: str) -> Game:
    """A held-out game id as the site names it. Anything that is not a held-out game is refused (C8)."""
    if not is_heldout_game(game_id):
        raise ExportProblem(f"{game_id} is not a held-out game: the possession review shows held-out games only")
    try:
        meta = game_meta(game_id)
    except ValueError:
        raise ExportProblem(f"{game_id} is not a game id of the form MM.DD.YYYY.AWAY.at.HOME") from None
    sp = splits.load()
    scorecard_games = {g.get("sportvu") for g in sp["heldout_game"]["games"].values()}
    arenas = set(sp["heldout_arena"].get("arenas") or [])
    if game_id in scorecard_games:
        split = "heldout_game"
    elif meta["home"] in arenas:
        split = "heldout_arena"
    else:
        raise ExportProblem(f"{game_id} is held out but belongs to no split the site knows")
    slug = "%s-%s-at-%s" % (meta["date"], meta["away"].lower(), meta["home"].lower())
    if S.model_for(f"games/{slug}/possessions.json") is not S.PossessionIndex:
        raise ExportProblem(f"{game_id}: {slug!r} is not a usable slug")
    return Game(game_id=game_id, slug=slug, label="%s at %s" % (meta["away"], meta["home"]), date=meta["date"],
                home=meta["home"], away=meta["away"], split=split)


def heldout_games(data_root) -> list:
    """The held-out games with possessions on the data root, in date order. A file in that folder whose name is
    not a held-out game id stops the build."""
    folder = Path(data_root) / "analytics" / "possessions" / "heldout" / C.SEASON
    ids = sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []
    if not ids:
        raise ExportProblem("no held-out possessions under " + C.data_path("analytics/possessions/heldout", C.SEASON))
    games = [game_of(g) for g in ids]
    if len({g.slug for g in games}) != len(games):
        raise ExportProblem("two held-out games share one slug")
    return sorted(games, key=lambda g: (g.date, g.game_id))


# ------------------------------------------------------------------------------------------------ xFG context


@dataclass
class ShotContext:
    """The held-out xFG rows of the listed games and the cleaned held-out shot chart, read once."""

    rows: pd.DataFrame                # held-out rows of xfg_predictions for the games, in file order
    chart: pd.DataFrame               # the cleaned held-out shot chart, in file order
    edges: list                       # the 4 inner edges of the xFG fifths, over every held-out prediction
    n_bucket_rows: int                # every held-out row of xfg_predictions: the base of the fifths
    n_bucket_games: int = 0           # the held-out games those rows come from
    n_rows_shown: int = 0             # the rows of the listed games (the games with a possessions table)
    games_not_shown: list = field(default_factory=list)   # held-out games with predictions and no possessions table


def published_xfg(values) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), 3)


def xfg_bucket(xfg, edges) -> int:
    """The fifth of a published xFG value: 1 lowest, 5 highest."""
    return 1 + int(sum(xfg > e for e in edges))


def load_shot_context(data_root, game_ids) -> ShotContext:
    """Held-out xFG rows (heldout true, fold -1) of all listed games, the fifths of the predictions, and the
    held-out shot chart cleaned by the lane's own function. The fifths are over every held-out prediction of the
    table, the rows the lane report scores, also those of a held-out game that has no possessions table and so no
    review page. They describe the predictions; no outcome is read."""
    data_root = Path(data_root)
    path = data_root / "analytics" / "xfg" / C.SEASON / "xfg_predictions.parquet"
    table = C.read_table(path, ["sportvu_game", "event", "player_id", "p_selected", "made", "fold", "heldout"],
                         C.data_path("analytics/xfg", C.SEASON, "xfg_predictions.parquet"))
    held = table["heldout"].astype(bool).to_numpy()
    if ((table["fold"].to_numpy() == -1) != held).any():
        raise ExportProblem("xfg_predictions: the heldout flag and fold -1 do not name the same rows")
    held_rows = table[held]
    if held_rows["p_selected"].isna().any():
        raise ExportProblem("xfg_predictions: a held-out row has no prediction")
    held_games = sorted(held_rows["sportvu_game"].astype(str).unique())
    wrong = [g for g in held_games if not is_heldout_game(g)]
    if wrong:
        raise ExportProblem(f"xfg_predictions: rows flagged held out belong to {wrong[:3]}, which are not held-out games")
    rows = held_rows[held_rows["sportvu_game"].isin(list(game_ids)).to_numpy()].reset_index(drop=True)
    if len(rows) == 0:
        raise ExportProblem("xfg_predictions holds no held-out row for the held-out games")
    edges = np.quantile(published_xfg(held_rows["p_selected"]), [k / N_FIFTHS for k in range(1, N_FIFTHS)])
    chart_path = data_root / "collect" / "bref_shots" / "heldout" / "shotchart" / f"{C.SEASON}.parquet"
    chart = C.read_table(chart_path, what=C.data_path("collect/bref_shots/heldout/shotchart", f"{C.SEASON}.parquet"))
    chart, _summary = xfg_validate.clean_shots(chart)
    return ShotContext(rows=rows, chart=chart, edges=[round(float(e), 4) for e in edges],
                       n_bucket_rows=int(len(held_rows)), n_bucket_games=len(held_games), n_rows_shown=int(len(rows)),
                       games_not_shown=sorted(set(held_games) - {str(g) for g in game_ids}))


# ------------------------------------------------------------------------------------------------ shot chart


def chart_to_court(chart_x: float, chart_y: float, attack_x: float, mirrored: bool):
    """A shot chart marker in the tracking frame. The chart draws every shot as an attack on the left rim. With
    `mirrored` the chart's sideline axis runs against the tracking y axis."""
    x, y = float(chart_x), float(chart_y)
    if mirrored:
        y = S.COURT_WIDTH_FT - y
    if attack_x > C.HALF_X_FT:
        x, y = S.COURT_LENGTH_FT - x, S.COURT_WIDTH_FT - y
    return x, y


def check_orientation(samples: list) -> dict:
    """Which way the shot chart's sideline axis runs, from shots that have both a located release and a chart
    marker. Each sample is (tracked x, tracked y, chart x, chart y, attack_x). A shot at least
    ORIENT_MIN_OFF_MIDDLE_FT off the middle of the court votes for the reading that puts its marker nearer the
    shooter's tracked position. The orientation counts as verified when at least ORIENT_MIN_SHOTS shots vote and
    ORIENT_MIN_SHARE of them agree. Positions only: no outcome is read."""
    votes = {False: 0, True: 0}
    errors = {False: [], True: []}
    for tx, ty, cx, cy, attack_x in samples:
        dist = {}
        for mirrored in (False, True):
            x, y = chart_to_court(cx, cy, attack_x, mirrored)
            dist[mirrored] = float(np.hypot(tx - x, ty - y))
            errors[mirrored].append(dist[mirrored])
        if abs(cy - S.RIM_Y_FT) >= ORIENT_MIN_OFF_MIDDLE_FT and dist[False] != dist[True]:
            votes[dist[True] < dist[False]] += 1
    n = votes[False] + votes[True]
    winner = votes[True] > votes[False]
    verified = n >= ORIENT_MIN_SHOTS and votes[winner] >= ORIENT_MIN_SHARE * n
    return {"verified": bool(verified), "mirrored": bool(winner) if verified else None, "n_votes": n,
            "n_agree": votes[winner], "n_shots": len(samples),
            "median_ft": round(float(np.median(errors[winner])), 1) if verified and samples else None}


# ------------------------------------------------------------------------------------------------ one game's tables


class Frames:
    """The tracking frames of one game as arrays sorted by frame."""

    def __init__(self, table: pd.DataFrame):
        if not table["frame"].is_monotonic_increasing:
            table = table.sort_values("frame", kind="stable").reset_index(drop=True)
        self.table = table
        self.frame = table["frame"].to_numpy()
        self.player = table["player_id"].to_numpy()
        self.team = table["team_id"].to_numpy()
        self.x = table["x"].to_numpy(dtype=float)
        self.y = table["y"].to_numpy(dtype=float)
        self.level = {c: table[c].to_numpy(dtype=float) for c in LEVEL_COLUMNS}

    def span(self, start: int, end: int):
        lo = int(np.searchsorted(self.frame, start, side="left"))
        hi = int(np.searchsorted(self.frame, end, side="right"))
        return lo, hi

    def position(self, frame: int, player_id: int):
        """(x, y) of a player at one frame, or None when he has no finite row there."""
        lo, hi = self.span(frame, frame)
        hit = np.flatnonzero(self.player[lo:hi] == player_id)
        if not len(hit):
            return None
        x, y = self.x[lo + hit[0]], self.y[lo + hit[0]]
        return (float(x), float(y)) if np.isfinite(x) and np.isfinite(y) else None


def load_tables(data_root: Path, game_id: str) -> dict:
    """The tables of one held-out game. Model outputs that are not held-out rows stop the build."""
    analytics = data_root / "analytics"
    tracked = data_root / "sportvu_contract" / "heldout"

    def lane(table: str) -> Path:
        return contract_tables.table_path(analytics, table, C.SEASON, game_id, heldout=True)

    ghost = analytics / "ghosting" / "heldout" / C.SEASON
    out = {
        "frames": C.read_table(tracked / f"{game_id}.parquet", FRAME_COLUMNS, f"tracking frames of {game_id}"),
        "events": C.read_table(tracked / f"{game_id}_pbp.parquet", ["event", "period", "desc", "kind", "frame"],
                               f"play-by-play of {game_id}"),
        "possessions": C.read_table(lane("possessions"), what=f"possessions of {game_id}"),
        "rosters": C.read_table(lane("rosters"), ["player_id", "player", "team_id", "team", "home", "jersey", "bref_slug"],
                                f"rosters of {game_id}"),
        "shot_frames": C.read_table(lane("shot_frames"), ["sportvu_game", "event", "period", "clock_s", "made",
                                                          "shooter_id", "release_frame", "identity_match"],
                                    f"shot_frames of {game_id}"),
        "epv": C.read_table(analytics / "epv" / "heldout" / C.SEASON / f"{game_id}.parquet",
                            ["possession", "frame", "epv_points", "fold", "heldout"], f"EPV states of {game_id}"),
        "ghost_frames": C.read_table(ghost / f"ghost_frames_{game_id}.parquet",
                                     ["possession", "frame", "def_player_id", "ghost_x", "ghost_y", "horizon_s", "fold",
                                      "heldout"], f"ghost_frames of {game_id}"),
        "ghost_possessions": C.read_table(ghost / f"ghost_possessions_{game_id}.parquet",
                                          ["possession", "mean_deviation_ft"], f"ghost_possessions of {game_id}"),
    }
    for name in ("epv", "ghost_frames"):
        t = out[name]
        if len(t) and not ((t["fold"].to_numpy() == -1).all() and t["heldout"].astype(bool).all()):
            raise ExportProblem(f"{name} of {game_id} holds rows that are not held-out rows (fold -1)")
    poss = out["possessions"]
    if "sportvu_game" in poss.columns and (poss["sportvu_game"].astype(str) != game_id).any():
        raise ExportProblem(f"possessions of {game_id} name another game")
    if poss["possession"].duplicated().any():
        raise ExportProblem(f"possessions of {game_id}: a possession number is listed twice")
    if out["shot_frames"]["event"].duplicated().any():
        raise ExportProblem(f"shot_frames of {game_id}: an event is listed twice")
    return out


def game_sources(game_id: str) -> dict:
    """The sources of one game's tables, by what they feed. Every table is read for held-out rows only."""
    def lane(table: str, name: str = None) -> str:
        return C.data_path("analytics", table, "heldout", C.SEASON, name or f"{game_id}.parquet")

    return {
        "frames": [C.source(C.data_path("sportvu_contract/heldout", f"{game_id}.parquet"),
                            "tracking frames: player positions, ball, clocks", "heldout"),
                   C.source(lane("possessions"), "possession spans, offense, result and points", "heldout"),
                   C.source(lane("rosters"), "player names, jerseys and teams", "heldout"),
                   C.source("analytics/decisions/features.py",
                            "baskets_by_team_period: the rule that infers the attacked rim", "not_split")],
        "epv": [C.source(C.data_path("analytics/epv/heldout", C.SEASON, f"{game_id}.parquet"),
                         "EPV states of the frozen model, held-out rows (fold -1); point estimates", "heldout"),
                C.source("reports/epv.md", "the EPV lane's report: its held-out metric and interval", "not_split")],
        "ghosts": [C.source(lane("ghosting", f"ghost_frames_{game_id}.parquet"),
                            "ghost defender positions of the frozen model, 1.0 s rows, held-out rows (fold -1); "
                            "point estimates", "heldout"),
                   C.source(lane("ghosting", f"ghost_possessions_{game_id}.parquet"),
                            "mean deviation of the possession", "heldout"),
                   C.source("reports/ghosting.md", "the ghosting lane's report: its held-out metric", "not_split")],
        "shots": [C.source(C.data_path("analytics/xfg", C.SEASON, "xfg_predictions.parquet"),
                           "xFG of the frozen model, held-out rows (heldout true, fold -1); point estimates", "heldout"),
                  C.source(lane("shot_frames"), "release frame and shooter of each attempt", "heldout"),
                  C.source(C.data_path("sportvu_contract/heldout", f"{game_id}_pbp.parquet"),
                           "play-by-play: event frame and shot value", "heldout"),
                  C.source(C.data_path("collect/bref_shots/heldout/shotchart", f"{C.SEASON}.parquet"),
                           "basketball-reference shot chart: fallback shot location and shooter", "heldout"),
                  C.source("reports/xfg.md", "the xFG lane's report: its held-out metric and interval", "not_split")],
    }


def _teams(rosters: pd.DataFrame, game: Game) -> dict:
    """team_id -> abbreviation for the two teams of the game."""
    teams = {int(t): str(a) for t, a in zip(rosters["team_id"], rosters["team"])}
    if len(teams) != 2:
        raise ExportProblem(f"rosters of {game.game_id} name {len(teams)} teams")
    return teams


def attack_rims(game: Game, frames: Frames, possessions: pd.DataFrame) -> dict:
    """{(team_id, period): x of the attacked rim} from the frozen helper, after checking that the two teams of a
    period attack opposite rims."""
    rims = baskets_by_team_period(frames.table, frames.frame, possessions)
    out = {key: float(rim[0]) for key, rim in rims.items()}
    for (team, period), rim in out.items():
        for (other, other_period), other_rim in out.items():
            if other_period == period and other != team and other_rim == rim:
                raise ExportProblem(f"{game.game_id}: both teams are inferred to attack the same rim in period {period}")
    return out


# ------------------------------------------------------------------------------------------------ shots of a game


def _contest_flags(frames: Frames, shots: pd.DataFrame, game_id: str) -> np.ndarray:
    """Whether the xFG lane measured contest features for each attempt. This applies the lane's own rule
    (GameFrames.contest, on frames cleaned by its clean_frames) to the rows of the release frames, without fitting
    or predicting anything."""
    flags = np.zeros(len(shots), dtype=bool)
    todo = shots[(shots["release_frame"] >= 0) & shots["player_id"].notna()]
    if not len(todo):
        return flags
    wanted = np.isin(frames.frame, todo["release_frame"].to_numpy(dtype=np.int64))
    rows = frames.table[wanted].copy()
    rows.insert(0, "game", game_id)
    cleaned, _summary = xfg_validate.clean_frames(rows)
    if not len(cleaned):
        return flags
    index = xfg_features.GameFrames(cleaned)
    for position, row in zip(todo.index, todo.itertuples(index=False)):
        flags[position] = bool(index.contest(int(row.release_frame), int(row.player_id))["has_contest"])
    return flags


def game_shots(game: Game, ctx: ShotContext, tables: dict, frames: Frames) -> pd.DataFrame:
    """One row per held-out xFG prediction of the game, joined to its attempt (shot_frames), its play-by-play
    event and its shot chart row. The chart rows are tied to events by the lane's own clock match, and the result
    must equal the events stored with the predictions."""
    rows = ctx.rows[ctx.rows["sportvu_game"] == game.game_id].reset_index(drop=True)
    located = tables["shot_frames"]
    chart = ctx.chart[ctx.chart["sportvu_game"].astype(str) == game.game_id].reset_index(drop=True)
    if rows["event"].duplicated().any() or (rows["event"] < 0).any():
        raise ExportProblem(f"xfg_predictions of {game.game_id}: an event is missing or listed twice")
    matched = xfg_features.match_events(chart, located).to_numpy() if len(chart) else np.array([], dtype=np.int64)
    if len(chart) != len(rows) or (matched != rows["event"].to_numpy()).any():
        raise ExportProblem(f"{game.game_id}: the shot chart rows do not line up with the stored xFG events")
    by_event = located.set_index("event")
    missing = sorted(set(rows["event"]) - set(by_event.index))
    if missing:
        raise ExportProblem(f"{game.game_id}: xFG events {missing[:3]} are not in shot_frames")
    events = tables["events"].drop_duplicates("event").set_index("event")
    out = rows.copy()
    ev = out["event"].to_numpy()
    attempt = by_event.loc[ev]
    out["period"] = attempt["period"].to_numpy()
    out["release_frame"] = attempt["release_frame"].to_numpy()
    out["shooter_id"] = attempt["shooter_id"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    out["identity"] = attempt["identity_match"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    out["made_located"] = attempt["made"].astype(bool).to_numpy()
    out["event_frame"] = [int(events.at[e, "frame"]) if e in events.index else -1 for e in ev]
    text = [str(events.at[e, "desc"]) if e in events.index else "" for e in ev]
    out["three"] = pd.Series([True if "3-pt" in d else False if "2-pt" in d else None for d in text], dtype=object)
    out["chart_x"] = chart["court_x"].to_numpy(dtype=float)
    out["chart_y"] = chart["court_y"].to_numpy(dtype=float)
    out["chart_slug"] = chart["player_slug"].astype(object).to_numpy() if "player_slug" in chart.columns else None
    out["has_contest"] = _contest_flags(frames, out, game.game_id)
    return out


def slug_stem(name):
    """The stem basketball-reference builds a player's slug from: the first five letters of the last name and the
    first two of the first name, in plain lower-case letters ("kanteen" for Enes Kanter). None for a one-word name."""
    tokens = name_tokens(name)
    return "".join(tokens[1:])[:5] + tokens[0][:2] if len(tokens) >= 2 else None


def chart_shooter(slug: str, slug_to_player: dict, roster_names: dict):
    """The tracked player of a shot chart marker whose play-by-play shooter the frozen name match did not resolve:
    the one roster player carrying the marker's slug, else the one roster player without a slug whose tracking name
    gives the slug's stem. The second step finds a player the roster's slug join missed because the play-by-play
    lists him under a later name (Enes Kanter as "Enes Freedom", slug kanteen01). None unless exactly one player
    fits."""
    hits = slug_to_player.get(slug, [])
    if len(hits) == 1:
        return hits[0]
    if hits:
        return None
    stem = re.sub(r"\d+$", "", slug)
    slugged = {pid for ids in slug_to_player.values() for pid in ids}
    fits = [pid for pid, name in roster_names.items() if pid not in slugged and slug_stem(name) == stem]
    return fits[0] if len(fits) == 1 else None


# ------------------------------------------------------------------------------------------------ one possession


@dataclass
class Grid:
    """The frames of one possession on the ghosting grid."""

    start: int
    end: int
    n: int
    rows: np.ndarray                  # positions in the game's frame arrays of the rows that lie on the grid
    slot: np.ndarray                  # grid index of each of those rows
    per_frame: np.ndarray             # player rows in each grid frame
    level: dict                       # frame-level columns on the grid, NaN where the frame has no row


def possession_grid(frames: Frames, start: int, end: int) -> Grid:
    n = (end - start) // C.STRIDE + 1
    lo, hi = frames.span(start, end)
    rows = lo + np.flatnonzero((frames.frame[lo:hi] - start) % C.STRIDE == 0)
    slot = ((frames.frame[rows] - start) // C.STRIDE).astype(np.int64)
    level = {}
    filled, first = np.unique(slot, return_index=True)
    for name, values in frames.level.items():
        column = np.full(n, np.nan)
        column[filled] = values[rows][first]
        level[name] = column
    return Grid(start=start, end=end, n=n, rows=rows, slot=slot, per_frame=np.bincount(slot, minlength=n), level=level)


def player_tracks(frames: Frames, grid: Grid, people: dict, teams: dict, offense_id: int, where: str):
    """(tracks, every player has a position in every frame). Offense first, then by player id."""
    tracks, complete = [], True
    players, team_ids = frames.player[grid.rows], frames.team[grid.rows]
    xs_all, ys_all = frames.x[grid.rows], frames.y[grid.rows]
    for player_id in np.unique(players):
        mine = players == player_id
        xs, ys = np.full(grid.n, np.nan), np.full(grid.n, np.nan)
        xs[grid.slot[mine]], ys[grid.slot[mine]] = xs_all[mine], ys_all[mine]
        complete = complete and bool(np.isfinite(xs).all() and np.isfinite(ys).all())
        person = people.get(int(player_id))
        if person is None or not isinstance(person.player, str):
            raise ExportProblem(f"{where}: player {int(player_id)} is not in the roster")
        team_id = int(team_ids[mine][0])
        if team_id not in teams:
            raise ExportProblem(f"{where}: player {int(player_id)} is on a third team")
        tracks.append(S.PlayerTrack(player_id=int(player_id), name=person.player,
                                    jersey=person.jersey if isinstance(person.jersey, str) else None,
                                    team=teams[team_id], offense=team_id == offense_id,
                                    x=C.round_list(xs, 1), y=C.round_list(ys, 1)))
    tracks.sort(key=lambda t: (not t.offense, t.player_id))
    return tracks, complete


def epv_curve(states, grid: Grid, where: str):
    """The stored EPV states of the possession by frame time, or None when there is none."""
    if states is None or not len(states):
        return None
    f = states["frame"].to_numpy()
    if (f < grid.start).any() or (f > grid.end).any() or len(np.unique(f)) != len(f):
        raise ExportProblem(f"{where}: EPV states lie outside the possession or repeat a frame")
    return S.EpvCurve(t=C.round_list((f - grid.start) / C.FPS, 2), value=C.round_list(states["epv_points"], 3))


def ghost_tracks(rows, grid: Grid, defenders: set, mean_deviation, where: str):
    """(Ghosts or None, share of frames with a stored ghost, points dropped). The stored 1.0 s rows, aligned by
    frame. The share counts a frame when the table stores a finite ghost position for it, before the display rule
    drops far points: the featured rule (PLAN.md D48) reads it, and that rule may not read how far off a
    prediction is."""
    if rows is None or not len(rows):
        return None, 0.0, 0
    f = rows["frame"].to_numpy()
    if ((f - grid.start) % C.STRIDE != 0).any() or (f < grid.start).any() or (f > grid.end).any():
        raise ExportProblem(f"{where}: ghost rows are off the frame grid of the possession")
    slot = (f - grid.start) // C.STRIDE
    gx, gy = rows["ghost_x"].to_numpy(dtype=float), rows["ghost_y"].to_numpy(dtype=float)
    stored = np.isfinite(gx) & np.isfinite(gy)
    far = ((gx < -GHOST_MARGIN_FT) | (gx > S.COURT_LENGTH_FT + GHOST_MARGIN_FT) | (gy < -GHOST_MARGIN_FT)
           | (gy > S.COURT_WIDTH_FT + GHOST_MARGIN_FT) | ~np.isfinite(gx) | ~np.isfinite(gy))
    gx, gy = np.where(far, np.nan, gx), np.where(far, np.nan, gy)
    owners = rows["def_player_id"].to_numpy()
    tracks, covered = [], np.zeros(grid.n, dtype=bool)
    for player_id in np.unique(owners):
        if int(player_id) not in defenders:
            raise ExportProblem(f"{where}: a ghost of player {int(player_id)}, who is not a defender here")
        mine = owners == player_id
        if len(np.unique(slot[mine])) != int(mine.sum()):
            raise ExportProblem(f"{where}: two ghost rows for one defender and frame")
        xs, ys = np.full(grid.n, np.nan), np.full(grid.n, np.nan)
        xs[slot[mine]], ys[slot[mine]] = gx[mine], gy[mine]
        covered[slot[mine & stored]] = True
        tracks.append(S.GhostTrack(player_id=int(player_id), x=C.round_list(xs, 1), y=C.round_list(ys, 1)))
    ghosts = S.Ghosts(label="average defense", horizon_s=GHOST_HORIZON_S, tracks=tracks, dropped_points=int(far.sum()),
                      mean_deviation_ft=C.rnd(mean_deviation, 2))
    return ghosts, float(covered.mean()), int(far.sum())


def possession_notes(grid: Grid, row, n_players: int, epv, ghosts, dropped: int, by_time: int, by_chart: int,
                     unpredicted: int) -> list:
    """What a reader must know about this possession's data, in plain words. Empty for a clean possession."""
    missing = int((grid.per_frame == 0).sum())
    short = int(((grid.per_frame > 0) & (grid.per_frame < N_ON_COURT)).sum())
    no_ball = int((~np.isfinite(grid.level["ball_x"]) & (grid.per_frame > 0)).sum())
    notes = []
    if grid.n == 1:
        notes.append("Single frame: the tracking holds one frame for this possession, so there is nothing to play.")
    if bool(row.gap):
        notes.append(f"Clock gap: the game clock jumps by more than {GAP_S:g} s inside this possession, so part of the "
                     "play is missing.")
    if missing:
        notes.append(f"Frames missing from the tracking table: {missing} of {grid.n}.")
    if short:
        notes.append(f"Frames with fewer than {N_ON_COURT} tracked players: {short} of {grid.n}.")
    if n_players > N_ON_COURT:
        notes.append(f"Players in this possession: {n_players}. The lineup changed during it.")
    if no_ball:
        notes.append(f"Frames with no ball position: {no_ball} of {grid.n}.")
    if epv is None:
        notes.append("No EPV state is stored for this possession.")
    if ghosts is None:
        notes.append("No ghost defenders are stored for this possession.")
    if dropped:
        notes.append(f"Ghost points more than {GHOST_MARGIN_FT:g} ft outside the court, not drawn (a display rule): "
                     f"{dropped}.")
    if by_time:
        notes.append(f"Shots whose release was not located: {by_time}. Their time is the play-by-play frame, which "
                     "marks the logged outcome, not the release.")
    if by_chart:
        notes.append("Shots placed by the basketball-reference shot chart marker, not by a tracked position: "
                     f"{by_chart}.")
    if unpredicted:
        notes.append(f"Field-goal attempts with no xFG prediction: {unpredicted}.")
    return notes


# ------------------------------------------------------------------------------------------------ one game


@dataclass
class GameResult:
    game: Game
    possessions: list                         # S.Possession, in possession order
    rows: list                                # one dict per possession: the index fields and the featured facts
    notes: list
    sources: list
    counts: dict = field(default_factory=dict)


COUNT_KEYS = ("one_frame", "gap", "no_ghosts", "no_epv", "frames_missing", "frames_short", "ghost_points_dropped",
              "shots_release", "shots_event_time", "shots_no_time", "shots_chart_location", "shots_no_location",
              "shots_with_contest", "shooter_from_chart", "attempts_without_prediction", "made_flags_differ",
              "ball_agree", "ball_checked")


def build_game(data_root, game: Game, ctx: ShotContext) -> GameResult:
    """Every possession of one held-out game as contract models, plus the facts the index and the featured draw
    need. The tracking frames of the game are held only while this function runs."""
    data_root = Path(data_root)
    tables = load_tables(data_root, game.game_id)
    frames = Frames(tables["frames"])
    possessions = tables["possessions"].sort_values("possession", kind="stable").reset_index(drop=True)
    rosters = tables["rosters"]
    teams = _teams(rosters, game)
    offense = possessions["offense_team_id"]
    if offense.isna().any() or not offense.astype("int64").isin(list(teams)).all():
        raise ExportProblem(f"{game.game_id}: a possession's offense is not one of the two teams of the game")
    people = {int(r.player_id): r for r in rosters.itertuples(index=False)}
    slug_to_player = {}
    for r in rosters.itertuples(index=False):
        if isinstance(r.bref_slug, str) and r.bref_slug:
            slug_to_player.setdefault(r.bref_slug, []).append(int(r.player_id))
    roster_names = {int(r.player_id): r.player for r in rosters.itertuples(index=False)
                    if int(r.player_id) > 0 and isinstance(r.player, str)}
    rims = attack_rims(game, frames, possessions)
    shots = game_shots(game, ctx, tables, frames)
    sources = game_sources(game.game_id)
    attempts = tables["shot_frames"]

    epv_by = {int(k): g.sort_values("frame", kind="stable") for k, g in tables["epv"].groupby("possession")}
    ghost_rows = tables["ghost_frames"]
    ghost_rows = ghost_rows[ghost_rows["horizon_s"].to_numpy(dtype=float) > (GHOST_HORIZON_S / 2)]
    ghost_by = {int(k): g for k, g in ghost_rows.groupby("possession")}
    deviation = {int(p): float(v) for p, v in zip(tables["ghost_possessions"]["possession"],
                                                  tables["ghost_possessions"]["mean_deviation_ft"])}

    # Pass 1: place every prediction in a possession and find the tracked release position of located shots.
    known = possessions[["possession", "period", "start_event", "end_event", "start_frame", "end_frame",
                         "offense_team_id"]].to_dict("records")
    placed, release_xy, samples = {}, {}, []
    for i, shot in enumerate(shots.itertuples(index=False)):
        owner = [p for p in known if p["period"] == shot.period and p["start_event"] <= shot.event <= p["end_event"]]
        if len(owner) > 1:
            raise ExportProblem(f"{game.game_id}: event {shot.event} lies in {len(owner)} possessions")
        if not owner:
            continue
        p = owner[0]
        placed.setdefault(int(p["possession"]), []).append(i)
        inside = shot.release_frame >= 0 and p["start_frame"] <= shot.release_frame <= p["end_frame"]
        if np.isfinite(shot.shooter_id) and inside:
            xy = frames.position(int(shot.release_frame), int(shot.shooter_id))
            if xy is not None:
                release_xy[i] = xy
                rim = rims.get((int(p["offense_team_id"]), int(p["period"])))
                if shot.identity and rim is not None and np.isfinite(shot.chart_x) and np.isfinite(shot.chart_y):
                    samples.append((xy[0], xy[1], shot.chart_x, shot.chart_y, rim))
    orientation = check_orientation(samples)

    counts = {key: 0 for key in COUNT_KEYS}
    counts.update(possessions=int(len(possessions)), predictions=int(len(shots)),
                  predictions_unplaced=int(len(shots) - sum(len(v) for v in placed.values())),
                  predictions_with_contest=int(shots["has_contest"].sum()))
    ball_votes = {}
    out_possessions, out_rows = [], []
    for row in possessions.itertuples(index=False):
        number, start, end = int(row.possession), int(row.start_frame), int(row.end_frame)
        where = f"{game.game_id} possession {number}"
        if end < start:
            raise ExportProblem(f"{where}: end_frame is before start_frame")
        if row.end_kind not in RESULTS:
            raise ExportProblem(f"{where}: unknown result {row.end_kind!r}")
        offense_id = int(row.offense_team_id)
        defense_id = next(t for t in teams if t != offense_id)
        key = (offense_id, int(row.period))
        if key not in rims:
            raise ExportProblem(f"{where}: no attack direction for its team and period")
        attack_x = rims[key]

        grid = possession_grid(frames, start, end)
        tracks, complete = player_tracks(frames, grid, people, teams, offense_id, where)
        ten_tracked = complete and len(tracks) == N_ON_COURT and bool((grid.per_frame == N_ON_COURT).all())

        # the attack direction against the ball: where the ball is over the last third of the possession
        if grid.n >= BALL_CHECK_MIN_FRAMES:
            tail = grid.level["ball_x"][grid.n - grid.n // 3:]
            tail = tail[np.isfinite(tail)]
            if len(tail):
                agrees = bool((np.median(tail) > C.HALF_X_FT) == (attack_x > C.HALF_X_FT))
                ball_votes.setdefault(key, [0, 0])[0 if agrees else 1] += 1
                counts["ball_checked"] += 1
                counts["ball_agree"] += int(agrees)

        epv = epv_curve(epv_by.get(number), grid, where)
        defenders = {t.player_id for t in tracks if not t.offense}
        ghosts, ghost_share, dropped = ghost_tracks(ghost_by.get(number), grid, defenders, deviation.get(number), where)

        # shots: the predictions placed in this possession, in event order
        shot_models, by_time, by_chart = [], 0, 0
        for i in sorted(placed.get(number, []), key=lambda i: shots.at[i, "event"]):
            shot = shots.iloc[i]
            shooter_id = int(shot["shooter_id"]) if np.isfinite(shot["shooter_id"]) else None
            if shooter_id is None and isinstance(shot["chart_slug"], str):
                shooter_id = chart_shooter(shot["chart_slug"], slug_to_player, roster_names)
                counts["shooter_from_chart"] += int(shooter_id is not None)
            t = x = y = None
            t_source, location_source = "none", "none"
            if i in release_xy:
                t, t_source = C.rnd((int(shot["release_frame"]) - start) / C.FPS, 2), "release_frame"
                (x, y), location_source = release_xy[i], "tracking_at_release"
                counts["shots_release"] += 1
            else:
                event_frame = int(shot["event_frame"])
                if event_frame >= 0 and start <= event_frame <= end:
                    t, t_source = C.rnd((event_frame - start) / C.FPS, 2), "play_by_play"
                    counts["shots_event_time"] += 1
                    by_time += 1
                else:
                    counts["shots_no_time"] += 1
                if orientation["verified"] and np.isfinite(shot["chart_x"]) and np.isfinite(shot["chart_y"]):
                    x, y = chart_to_court(shot["chart_x"], shot["chart_y"], attack_x, orientation["mirrored"])
                    location_source = "shot_chart"
                    counts["shots_chart_location"] += 1
                    by_chart += 1
                else:
                    counts["shots_no_location"] += 1
            counts["shots_with_contest"] += int(bool(shot["has_contest"]))
            counts["made_flags_differ"] += int(bool(shot["made"]) != bool(shot["made_located"]))
            person = people.get(shooter_id) if shooter_id is not None else None
            shot_models.append(S.Shot(
                event=int(shot["event"]), t=t, t_source=t_source, x=C.rnd(x, 1), y=C.rnd(y, 1),
                location_source=location_source, shooter_id=shooter_id,
                shooter=person.player if person is not None and isinstance(person.player, str) else None,
                made=bool(shot["made"]), three=None if shot["three"] is None else bool(shot["three"]),
                xfg=float(published_xfg(shot["p_selected"])), has_contest=bool(shot["has_contest"])))

        # the last field-goal attempt of the possession, with or without a prediction
        inside = attempts[(attempts["period"] == row.period) & (attempts["event"] >= row.start_event)
                          & (attempts["event"] <= row.end_event)]
        predicted = {s.event: s for s in shot_models}
        unpredicted = int(len(inside) - len(set(inside["event"]) & set(predicted)))
        last = predicted.get(int(inside["event"].max())) if len(inside) else None

        used = list(sources["frames"])
        used += sources["epv"] if epv is not None else []
        used += sources["ghosts"] if ghosts is not None else []
        used += sources["shots"] if shot_models else []
        model = S.Possession(
            schema_version=S.SCHEMA_VERSION, game_id=game.game_id, game_slug=game.slug, possession=number,
            period=int(row.period), hz=C.HZ, frame0=start, t=C.round_list(np.arange(grid.n) * C.STRIDE / C.FPS, 2),
            game_clock_s=C.round_list(grid.level["game_clock_s"], 2),
            shot_clock_s=C.round_list(grid.level["shot_clock_s"], 2),
            offense=teams[offense_id], defense=teams[defense_id], attack_x=attack_x, result=row.end_kind,
            points=int(row.points), gap=bool(row.gap), players=tracks,
            ball=S.BallTrack(x=C.round_list(grid.level["ball_x"], 1), y=C.round_list(grid.level["ball_y"], 1),
                             z=C.round_list(grid.level["ball_z"], 1)),
            epv=epv, ghosts=ghosts, shots=shot_models,
            notes=possession_notes(grid, row, len(tracks), epv, ghosts, dropped, by_time, by_chart, unpredicted),
            sources=C.merge_sources(used))
        out_possessions.append(model)

        duration = model.t[-1]
        last_xfg = last.xfg if last is not None else None
        bucket = xfg_bucket(last_xfg, ctx.edges) if last_xfg is not None else None
        eligible = bool(
            row.end_kind in SHOT_RESULTS and last is not None and last.t_source == "release_frame"
            and last.location_source == "tracking_at_release" and last_xfg is not None
            and epv is not None and len(epv.t) >= FEATURED_MIN_EPV_STATES and ghost_share >= FEATURED_MIN_GHOST_SHARE
            and not bool(row.gap) and ten_tracked and FEATURED_MIN_S <= duration <= FEATURED_MAX_S)
        out_rows.append({
            "possession": number, "period": int(row.period), "start_clock_s": C.rnd(row.start_clock_s, 2),
            "end_clock_s": C.rnd(row.end_clock_s, 2), "duration_s": duration, "offense": teams[offense_id],
            "defense": teams[defense_id], "result": row.end_kind, "points": int(row.points), "gap": bool(row.gap),
            "n_frames": grid.n, "xfg": last_xfg, "xfg_bucket": bucket,
            "epv_start": epv.value[0] if epv is not None else None, "epv_end": epv.value[-1] if epv is not None else None,
            "has_ghosts": ghosts is not None,
            "eligible": eligible, "cell": cell_name(row.end_kind, bucket) if eligible else None,
        })
        counts["one_frame"] += int(grid.n == 1)
        counts["gap"] += int(bool(row.gap))
        counts["no_ghosts"] += int(ghosts is None)
        counts["no_epv"] += int(epv is None)
        counts["frames_missing"] += int((grid.per_frame == 0).sum())
        counts["frames_short"] += int(((grid.per_frame > 0) & (grid.per_frame < N_ON_COURT)).sum())
        counts["ghost_points_dropped"] += dropped
        counts["attempts_without_prediction"] += unpredicted

    for key, (agree, disagree) in ball_votes.items():
        if agree + disagree >= BALL_CHECK_MIN_VOTES and disagree >= agree:
            raise ExportProblem(f"{game.game_id}: the inferred attack direction of team {key[0]} in period {key[1]} "
                                f"disagrees with where the ball ends up ({disagree} of {agree + disagree} possessions)")
    counts["eligible"] = sum(r["eligible"] for r in out_rows)
    counts["orientation"] = orientation
    return GameResult(game=game, possessions=out_possessions, rows=out_rows, notes=index_notes(counts, ctx),
                      sources=C.merge_sources(*sources.values()), counts=counts)


def index_notes(counts: dict, ctx: ShotContext) -> list:
    """The notes of a game's index: how to read the fields, and what this game's data looks like."""
    o = counts["orientation"]
    if o["verified"]:
        way = "against" if o["mirrored"] else "along"
        chart = (f"Shot chart orientation: checked on {o['n_votes']} shots of this game that have a located release and "
                 f"lie at least {ORIENT_MIN_OFF_MIDDLE_FT:g} ft off the middle of the court. {o['n_agree']} of them sit "
                 f"nearer the shooter's tracked position when the chart's sideline axis runs {way} the tracking y axis. "
                 f"A chart marker sits a median of {o['median_ft']:g} ft from the tracked release position "
                 f"({o['n_shots']} shots), so a chart location is approximate. It is used only where no release was "
                 "located.")
    else:
        chart = (f"Shot chart orientation: not verified in this game ({o['n_agree']} of {o['n_votes']} shots agree; "
                 f"{ORIENT_MIN_SHOTS} are needed with {ORIENT_MIN_SHARE:.0%} agreeing), so no shot takes its location "
                 "from the shot chart.")
    not_shown = ""
    if ctx.games_not_shown:
        not_shown = (f" {ctx.n_bucket_rows - ctx.n_rows_shown} of those attempts belong to {len(ctx.games_not_shown)} "
                     f"held-out game(s) with no possessions table ({', '.join(ctx.games_not_shown)}), so the review "
                     f"cannot show them. The other {ctx.n_rows_shown} are in the games shown here.")
    notes = [
        "Held-out game: the frozen models never saw it, so every prediction shown here is out of sample.",
        f"Positions: {C.HZ:g} frames a second, on the grid of the stored ghost rows (every fifth frame from the "
        f"possession's first frame). t is frame time at {C.FPS} frames a second, so it keeps running while the game "
        "clock is stopped. n_frames counts the frames of the possession file. The last frame is the last grid frame "
        "at or before the possession's end, and duration_s is its t, so a shot time or an EPV time can lie up to "
        f"{(C.STRIDE - 1) / C.FPS:g} s after the last frame.",
        "Attack direction: the tracking data does not say which rim a team attacks. It is inferred per team and "
        "period by a majority vote of where the offense stands. The ball's side over the last third of a possession "
        f"agrees with it in {counts['ball_agree']} of {counts['ball_checked']} possessions of at least two seconds.",
        "xFG: the frozen model's probability that the shot goes in, a point estimate (the lane stores no interval for "
        "one shot). xfg in this list is the last field-goal attempt of the possession. xfg_bucket is its fifth among "
        f"the {ctx.n_bucket_rows} held-out attempts with a prediction, from {ctx.n_bucket_games} held-out games (edges "
        "in xfg_bucket_edges)." + not_shown,
        "EPV: expected points for the offense to the end of the possession, the model's raw output, about one value "
        "a second of game clock, never clipped. Each value is a point estimate.",
        f"Ghost defenders show an average defense: where an average defender would be {GHOST_HORIZON_S:g} s after his "
        "real position, given how the ball and the attackers really moved. Each of the five is computed alone, with "
        f"the other four real, and is a point estimate. A point more than {GHOST_MARGIN_FT:g} ft outside the court is "
        "not drawn.",
        f"Shots: {counts['predictions']} attempts of this game have a prediction ({counts['predictions_unplaced']} "
        f"of them fall in no possession and are not shown). {counts['shots_release']} have a located release (time and "
        f"place from tracking). {counts['shots_event_time']} take their time from the play-by-play frame, which marks "
        f"the logged outcome. {counts['shots_chart_location']} take their place from the shot chart. "
        f"{counts['shots_with_contest']} had their contest features measured by the xFG lane; for the others the "
        f"prediction rests on location, period, clock and the shooter prior. {counts['attempts_without_prediction']} "
        "attempts have no prediction.",
        chart,
        "Shooter: the play-by-play name matched to a tracked player; where that match failed, the shot chart's "
        f"player ({counts['shooter_from_chart']} shots of this game).",
        f"This game: {counts['possessions']} possessions, {counts['one_frame']} with a single frame, {counts['gap']} "
        f"with a clock gap, {counts['no_ghosts']} without ghost defenders.",
    ]
    if counts["made_flags_differ"]:
        notes.append(f"{counts['made_flags_differ']} shots are made in one of the shot chart and the play-by-play and "
                     "missed in the other. The shot chart's flag is shown: it is the one the prediction was scored on.")
    return notes


# ------------------------------------------------------------------------------------------------ featured, index


def cell_name(result: str, fifth: int) -> str:
    """The name of a cell of the featured rule: the possession's result and the fifth of its xFG."""
    return f"{result} fifth {fifth}"


def pick_featured(candidates: list) -> dict:
    """The featured possessions (PLAN.md D48, D52): one drawn at random from each of the 10 cells of result (made,
    missed) by xFG fifth. `candidates` is a list of (game_id, possession, cell name). One generator
    (numpy default_rng, seed 0) serves the whole draw. The cells are visited by result (made, then missed), then
    xFG fifth (1 to 5), with one draw per cell from its ids in sorted order. An empty cell takes no draw: it is
    reported, not refilled. No model error is read."""
    cells = {cell_name(result, fifth): [] for result in SHOT_RESULTS for fifth in range(1, N_FIFTHS + 1)}
    for game_id, possession, cell in candidates:
        if cell not in cells:
            raise ExportProblem(f"featured draw: {cell!r} is not a cell of result by xFG fifth")
        cells[cell].append((str(game_id), int(possession)))
    rng = np.random.default_rng(FEATURED_SEED)
    picks, empty = {}, []
    for cell, ids in cells.items():
        if not ids:
            empty.append(cell)
            continue
        ids = sorted(ids)
        picks[cell] = ids[int(rng.integers(len(ids)))]
    return {"picks": picks, "empty": empty, "eligible": {cell: len(ids) for cell, ids in cells.items()}}


def featured_marks(picks: dict, swaps: list, rows: dict) -> dict:
    """{(game_id, possession): (cell, "rule" or "hand")}: the featured possessions as the index marks them.

    `picks` is the rule's draw (cell -> id). Each swap names a cell and the possession a person puts there in
    place of the rule's pick (PLAN.md D48: the picks may be swapped; D52: a swapped pick is marked as hand-picked).
    A hand pick must be a possession of this run that ends in the cell's result with an xFG in the cell's fifth,
    so the cell still says what the possession is. `rows` maps (game_id, possession) to its index row."""
    marks = {tuple(pick): (cell, "rule") for cell, pick in picks.items()}
    cells = {cell_name(result, fifth) for result in SHOT_RESULTS for fifth in range(1, N_FIFTHS + 1)}
    swapped = set()
    for swap in swaps:
        cell, key = swap["cell"], (str(swap["game_id"]), int(swap["possession"]))
        if cell not in cells or cell in swapped:
            raise ExportProblem(f"featured swap: {cell!r} is not a cell of the rule, or is swapped twice")
        row = rows.get(key)
        if row is None:
            raise ExportProblem(f"featured swap: possession {key[1]} of {key[0]} is not in this run")
        if row["result"] not in SHOT_RESULTS or row["xfg_bucket"] is None \
                or cell_name(row["result"], row["xfg_bucket"]) != cell:
            raise ExportProblem(f"featured swap: possession {key[1]} of {key[0]} does not belong to the cell {cell!r}")
        marks = {k: v for k, v in marks.items() if v[0] != cell}
        marks[key] = (cell, "hand")
        swapped.add(cell)
    return marks


def demo_possessions(artifacts) -> set:
    """(game_id, possession) pairs that the demo bridge lists in <artifacts>/demo/possessions.json; empty when the
    file is absent. Accepted shapes: a list of {"game_id": ..., "possession": ...} objects, the same list under
    the key "possessions", or a mapping of game id to a list of possession numbers."""
    path = Path(artifacts) / "demo" / "possessions.json"
    if not path.exists():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("possessions"), list):
            doc = doc["possessions"]
        if isinstance(doc, dict):
            return {(str(game), int(p)) for game, numbers in doc.items() for p in numbers}
        out = set()
        for item in doc:
            game = item.get("game_id") or item.get("sportvu_game")
            if not game:
                raise ValueError("an entry names no game")
            out.add((str(game), int(item["possession"])))
        return out
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ExportProblem(f"artifacts/demo/possessions.json is not a list of game and possession: {exc}") from None


def index_of(result: GameResult, ctx: ShotContext, marks: dict, demo: set, featured_note: str) -> S.PossessionIndex:
    """The index of one game. `marks` is the output of featured_marks: (game_id, possession) -> (cell, who)."""
    rows = []
    for row in result.rows:
        key = (result.game.game_id, row["possession"])
        fields = {k: v for k, v in row.items() if k not in ("eligible", "cell")}
        cell, who = marks.get(key, (None, None))
        rows.append(S.PossessionSummary(**fields, featured=cell is not None, featured_by=who, featured_cell=cell,
                                        demo=key in demo))
    return S.PossessionIndex(schema_version=S.SCHEMA_VERSION, game=result.game.info(), xfg_bucket_edges=ctx.edges,
                             possessions=rows, notes=result.notes + [featured_note], sources=result.sources)
