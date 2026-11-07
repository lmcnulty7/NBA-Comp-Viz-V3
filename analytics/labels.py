"""analytics/labels.py: hand labels for actions and ball-screen coverages (docs/contract/labels.md).

Two tables, both append-only JSONL under DATA_ROOT/labels/:

    labels/actions/<sportvu_game>.jsonl   one line per action label (ACTIONS_SCHEMA)
    labels/screens/<sportvu_game>.jsonl   one line per coverage label (SCREENS_SCHEMA)

Every line carries ``labelled_at`` (ISO 8601, UTC) and ``labeller_pass`` (1 = first pass, 2 = the relabel a
week later). A label is never rewritten in place: a deletion is a tombstone line ``{"action_id": k,
"deleted": true, "labelled_at": ...}`` (``screen_id`` for screens) and ``read_labels`` resolves the file in
order, so the last line about an id wins. ``read_labels`` (or the parquet that ``label_tool --export``
writes from it) is the only sanctioned reader: a consumer that reads the raw lines must drop every line
carrying ``deleted`` and keep the last line per id, and training consumers read ``labeller_pass == 1``
only (``read_labels(..., labeller_pass=1)``). A line without an id (older fixture roots) takes the running
0-based position of the non-tombstone lines of its file.

A third file, ``labels/visited/<sportvu_game>.jsonl``, records every possession a labeller pass finished
(``mark_visited``); ``kappa_report`` restricts the two passes to the possessions both passes visited, so an
action one pass saw and the other did not enters the kappa as ``none``.

Sampling (``sample_possessions``): half-court possessions only (the ball crosses the half-court line and
then spends at least HALF_COURT_DWELL_S in the front court), training games only (an id carrying a held-out
token is never sampled), stratified by offensive team, with at least half of each team's draw carrying the
rule-based screen candidate flag (a non-handler offensive player within SCREEN_CANDIDATE_FT of the
handler's nearest defender for SCREEN_CANDIDATE_S).

Consistency (``kappa``): two labeller passes over the same possessions are paired by temporal overlap of
their actions and Cohen's kappa is reported for ``action_type`` and for ``coverage`` (``sure`` coverages
only); an action one pass has and the other does not counts as the category ``none``.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------

ACTION_TYPES = ("ball_screen", "handoff", "off_ball_screen", "isolation", "post_up", "drive", "cut")
COVERAGES = ("switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over")
LABEL_SOURCES = ("hand", "model", "synthetic")
CONFIDENCES = ("sure", "unsure")

# Action types that need a screener_id (the second offensive player of the action).
SCREENER_REQUIRED = ("ball_screen", "handoff", "off_ball_screen")

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------

ACTIONS_SCHEMA = pa.schema([
    pa.field("sportvu_game", pa.string(), nullable=False),
    pa.field("possession", pa.int32(), nullable=False),
    pa.field("action_id", pa.int32(), nullable=False),
    pa.field("action_type", pa.string(), nullable=False),
    pa.field("start_frame", pa.int32(), nullable=False),
    pa.field("end_frame", pa.int32(), nullable=False),
    pa.field("ball_handler_id", pa.int32(), nullable=False),
    pa.field("screener_id", pa.int32(), nullable=True),
    pa.field("label_source", pa.string(), nullable=False),
    pa.field("labeller_pass", pa.int16(), nullable=False),
    pa.field("labelled_at", pa.string(), nullable=False),
])

SCREENS_SCHEMA = pa.schema([
    pa.field("sportvu_game", pa.string(), nullable=False),
    pa.field("possession", pa.int32(), nullable=False),
    pa.field("screen_id", pa.int32(), nullable=False),
    pa.field("action_id", pa.int32(), nullable=False),
    pa.field("coverage", pa.string(), nullable=False),
    pa.field("on_ball_defender_id", pa.int32(), nullable=False),
    pa.field("screener_defender_id", pa.int32(), nullable=False),
    pa.field("label_source", pa.string(), nullable=False),
    pa.field("confidence", pa.string(), nullable=False),
    pa.field("labeller_pass", pa.int16(), nullable=False),
    pa.field("labelled_at", pa.string(), nullable=False),
])

ENUMS: dict[str, tuple[str, ...]] = {
    "action_type": ACTION_TYPES,
    "coverage": COVERAGES,
    "label_source": LABEL_SOURCES,
    "confidence": CONFIDENCES,
}

ID_FIELD = {"actions": "action_id", "screens": "screen_id"}
SCHEMAS = {"actions": ACTIONS_SCHEMA, "screens": SCREENS_SCHEMA}
LABELLER_PASSES = (1, 2)

# Sampling constants
MIDCOURT_X = 47.0
HALF_COURT_DWELL_S = 2.0
SCREEN_CANDIDATE_FT = 4.0
SCREEN_CANDIDATE_S = 1.0
DEFAULT_FPS = 25.0


# --------------------------------------------------------------------------------------
# dtypes and casting
# --------------------------------------------------------------------------------------

def _pandas_dtype(field: pa.Field):
    t = field.type
    if pa.types.is_string(t):
        return pd.StringDtype()
    if pa.types.is_integer(t):
        bits = t.bit_width
        return pd.api.types.pandas_dtype(f"Int{bits}") if field.nullable else np.dtype(f"int{bits}")
    if pa.types.is_floating(t):
        return np.dtype("float32")
    if pa.types.is_boolean(t):
        return pd.BooleanDtype() if field.nullable else np.dtype("bool")
    raise TypeError(f"unsupported type {t}")


def pandas_dtypes(kind: str) -> dict:
    return {f.name: _pandas_dtype(f) for f in SCHEMAS[kind]}


def empty(kind: str) -> pd.DataFrame:
    """An empty actions or screens frame with the contract dtypes."""
    return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in pandas_dtypes(kind).items()})


def cast(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """``df`` with the contract dtypes and column order of ``kind`` (``actions`` or ``screens``).

    Missing columns raise KeyError; a null in a non-nullable column or a non-integral value raises
    ValueError. Extra columns are dropped.
    """
    schema = SCHEMAS[kind]
    missing = [f.name for f in schema if f.name not in df.columns]
    if missing:
        raise KeyError(f"{kind}: missing column(s) {missing}")
    out = {}
    for f in schema:
        s = df[f.name]
        dt = _pandas_dtype(f)
        if isinstance(dt, pd.StringDtype):
            vals = s.astype("object").where(~s.isna(), None)
            vals = vals.map(lambda v: v if (v is None or isinstance(v, str)) else str(v))
            if not f.nullable and vals.isna().any():
                raise ValueError(f"{kind}.{f.name}: null in non-nullable column")
            out[f.name] = vals.astype(dt)
        elif isinstance(dt, np.dtype) and dt.kind == "i":
            if s.isna().any():
                raise ValueError(f"{kind}.{f.name}: null in non-nullable column")
            vals = pd.to_numeric(s)
            arr = np.asarray(vals, dtype="float64")
            if not np.all(arr == np.floor(arr)):
                raise ValueError(f"{kind}.{f.name}: non-integral value")
            out[f.name] = vals.astype(dt)
        else:  # nullable int
            out[f.name] = pd.to_numeric(s).astype(dt)
    return pd.DataFrame(out, index=df.index)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# JSONL storage
# --------------------------------------------------------------------------------------

def labels_path(data_root, kind: str, game: str) -> Path:
    """DATA_ROOT/labels/<kind>/<game>.jsonl."""
    if kind not in SCHEMAS:
        raise KeyError(f"unknown label kind {kind!r}; expected one of {sorted(SCHEMAS)}")
    return Path(data_root) / "labels" / kind / f"{game}.jsonl"


def _json_value(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else (int(v) if float(v).is_integer() else float(v))
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _records(df: pd.DataFrame, kind: str, labelled_at: str | None = None) -> list[dict]:
    """Rows of ``df`` as JSON-ready dicts in schema order, filling labelled_at where missing."""
    if df is None or len(df) == 0:
        return []
    df = df.copy()
    stamp = labelled_at or now_iso()
    if "labelled_at" not in df.columns:
        df["labelled_at"] = stamp
    else:
        df["labelled_at"] = df["labelled_at"].where(~df["labelled_at"].isna(), stamp)
    if "labeller_pass" not in df.columns:
        df["labeller_pass"] = 1
    if "label_source" not in df.columns:
        df["label_source"] = "hand"
    if kind == "actions" and "screener_id" not in df.columns:
        df["screener_id"] = pd.array([None] * len(df), dtype="Int32")
    casted = cast(df, kind)
    cols = [f.name for f in SCHEMAS[kind]]
    return [{c: _json_value(r[c]) for c in cols} for r in casted.to_dict("records")]


def _append_lines(path: Path, lines: Iterable[dict]) -> int:
    lines = list(lines)
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(lines)


def write_labels(data_root, game: str, actions_df: pd.DataFrame | None, screens_df: pd.DataFrame | None,
                 labelled_at: str | None = None) -> dict:
    """Append the rows of ``actions_df`` and ``screens_df`` to the game's JSONL files.

    Rows are cast to the contract first (a row that does not cast raises before anything is written).
    ``labelled_at`` is filled with the current UTC time where the frame lacks it; ``labeller_pass``
    defaults to 1 and ``label_source`` to ``hand``. Returns ``{"actions": n, "screens": n, "paths": {...}}``.
    """
    recs = {"actions": _records(actions_df, "actions", labelled_at),
            "screens": _records(screens_df, "screens", labelled_at)}
    out = {"paths": {}}
    for kind in ("actions", "screens"):
        p = labels_path(data_root, kind, game)
        out[kind] = _append_lines(p, recs[kind])
        out["paths"][kind] = str(p)
    return out


def append_tombstone(data_root, game: str, kind: str, label_id: int, labelled_at: str | None = None) -> Path:
    """Append a deletion line for ``label_id`` (append-only: the label stays in the file, marked deleted)."""
    p = labels_path(data_root, kind, game)
    _append_lines(p, [{ID_FIELD[kind]: int(label_id), "deleted": True, "labelled_at": labelled_at or now_iso()}])
    return p


def _read_jsonl(path: Path, kind: str) -> pd.DataFrame:
    """Resolve a JSONL file in order: the last line about an id wins, deleted ids are dropped."""
    idf = ID_FIELD[kind]
    if not path.exists():
        return empty(kind)
    live: dict = {}
    order: list = []
    n_labels = 0
    with path.open("r", encoding="utf-8") as fh:
        for ln, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ln + 1}: bad JSON ({exc})") from exc
            key = rec.get(idf)
            if rec.get("deleted"):
                if key is None:
                    raise ValueError(f"{path}:{ln + 1}: tombstone has no {idf}")
                live.pop(int(key), None)
                continue
            if key is None:
                # a root written without ids (older fixtures): the running position of the labels
                key = n_labels
                rec[idf] = key
            key = int(key)
            n_labels += 1
            if key not in live:
                order.append(key)
            live[key] = rec
    rows = [live[k] for k in order if k in live]
    if not rows:
        return empty(kind)
    df = pd.DataFrame(rows)
    for f in SCHEMAS[kind]:
        if f.name not in df.columns:
            df[f.name] = None
    return cast(df, kind).reset_index(drop=True)


def read_labels(data_root, game: str, labeller_pass: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(actions_df, screens_df) of ``game`` with tombstones applied and contract dtypes.

    ``labeller_pass`` keeps that pass only; training consumers pass 1 (the relabel pass 2 exists for the
    kappa check and must never reach a trainer as extra rows).
    """
    acts = _read_jsonl(labels_path(data_root, "actions", game), "actions")
    scrs = _read_jsonl(labels_path(data_root, "screens", game), "screens")
    if labeller_pass is not None:
        acts = acts.loc[acts["labeller_pass"] == int(labeller_pass)].reset_index(drop=True)
        scrs = scrs.loc[scrs["labeller_pass"] == int(labeller_pass)].reset_index(drop=True)
    return acts, scrs


VISITED_COLUMNS = ("sportvu_game", "possession", "labeller_pass", "visited_at")


def visited_path(data_root, game: str) -> Path:
    return Path(data_root) / "labels" / "visited" / f"{game}.jsonl"


def mark_visited(data_root, game: str, possession: int, labeller_pass: int, visited_at: str | None = None) -> Path:
    """Append the record that ``labeller_pass`` finished ``possession`` (whether or not it holds labels)."""
    p = visited_path(data_root, game)
    _append_lines(p, [{"sportvu_game": str(game), "possession": int(possession), "labeller_pass": int(labeller_pass),
                       "visited_at": visited_at or now_iso()}])
    return p


def read_visited(data_root, game: str) -> pd.DataFrame:
    """The visited records of ``game`` (one row per line; a possession may appear more than once)."""
    p = visited_path(data_root, game)
    rows = []
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
    df = pd.DataFrame(rows, columns=list(VISITED_COLUMNS))
    return df.astype({"possession": "int32", "labeller_pass": "int16"}) if len(df) else df


def next_id(df: pd.DataFrame, kind: str) -> int:
    """The next free id for a label of ``kind`` (max existing + 1, starting at 1)."""
    idf = ID_FIELD[kind]
    if df is None or len(df) == 0 or idf not in df.columns:
        return 1
    return int(pd.to_numeric(df[idf]).max()) + 1


def list_games(data_root) -> list[str]:
    """Games that have an actions or screens JSONL under DATA_ROOT/labels."""
    root = Path(data_root) / "labels"
    out = set()
    for kind in SCHEMAS:
        d = root / kind
        if d.is_dir():
            out |= {p.stem for p in d.glob("*.jsonl")}
    return sorted(out)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def _enum_problems(df: pd.DataFrame, kind: str) -> list[str]:
    probs = []
    for col, allowed in ENUMS.items():
        if col in df.columns:
            bad = sorted(set(df[col].dropna().astype(str).unique()) - set(allowed))
            if bad:
                probs.append(f"{kind}.{col}: unknown value(s) {bad}; allowed {list(allowed)}")
    return probs


def validate_labels(actions_df: pd.DataFrame | None, screens_df: pd.DataFrame | None,
                    possessions_df: pd.DataFrame | None = None, frames_df: pd.DataFrame | None = None) -> list[str]:
    """Problems with the label frames; empty when they conform.

    Checks: contract columns present and castable, unknown enum values, labeller_pass in (1, 2), start_frame
    after end_frame, screener_id missing on a screen or handoff action, frames outside the possession span
    (when ``possessions_df`` is given), overlapping actions of the same type in one possession, screens
    whose action_id is not a ball_screen action of the same game and possession, duplicate ids, and, when
    ``frames_df`` and the possessions' ``offense_team_id`` are given, that the handler and screener are
    on the offence and the two defenders on the defence of their possession.
    """
    probs: list[str] = []
    acts = actions_df if actions_df is not None else empty("actions")
    scrs = screens_df if screens_df is not None else empty("screens")
    try:
        acts = cast(acts, "actions") if len(acts) else empty("actions")
    except (KeyError, ValueError) as exc:
        return [f"actions: {exc}"]
    try:
        scrs = cast(scrs, "screens") if len(scrs) else empty("screens")
    except (KeyError, ValueError) as exc:
        return [f"screens: {exc}"]

    probs += _enum_problems(acts, "actions") + _enum_problems(scrs, "screens")

    for kind, df in (("actions", acts), ("screens", scrs)):
        idf = ID_FIELD[kind]
        if len(df) and df.duplicated([ "sportvu_game", idf]).any():
            dup = sorted(df.loc[df.duplicated(["sportvu_game", idf]), idf].unique().tolist())
            probs.append(f"{kind}: duplicate {idf}(s) {dup}")

    for kind, df in (("actions", acts), ("screens", scrs)):
        if len(df):
            bad = sorted(set(df["labeller_pass"].astype(int).tolist()) - set(LABELLER_PASSES))
            if bad:
                probs.append(f"{kind}.labeller_pass: unknown pass(es) {bad}; allowed {list(LABELLER_PASSES)}")

    for r in acts.itertuples(index=False):
        tag = f"action {int(r.action_id)} ({r.action_type}, possession {int(r.possession)})"
        if int(r.start_frame) > int(r.end_frame):
            probs.append(f"{tag}: start_frame {int(r.start_frame)} after end_frame {int(r.end_frame)}")
        if r.action_type in SCREENER_REQUIRED and pd.isna(r.screener_id):
            probs.append(f"{tag}: screener_id is required for {r.action_type}")
        if not pd.isna(r.screener_id) and int(r.screener_id) == int(r.ball_handler_id):
            probs.append(f"{tag}: screener_id equals ball_handler_id")

    if possessions_df is not None and len(acts):
        span = {(str(g), int(p)): (int(s), int(e)) for g, p, s, e in zip(
            possessions_df["sportvu_game"], possessions_df["possession"],
            possessions_df["start_frame"], possessions_df["end_frame"])}
        for r in acts.itertuples(index=False):
            key = (str(r.sportvu_game), int(r.possession))
            tag = f"action {int(r.action_id)} ({r.action_type}, possession {int(r.possession)})"
            if key not in span:
                probs.append(f"{tag}: possession not in the possessions table")
                continue
            s, e = span[key]
            if int(r.start_frame) < s or int(r.end_frame) > e:
                probs.append(f"{tag}: frames [{int(r.start_frame)}, {int(r.end_frame)}] outside the possession span [{s}, {e}]")

    if len(acts):
        for (g, p, t), grp in acts.groupby(["sportvu_game", "possession", "action_type"], sort=True):
            if len(grp) < 2:
                continue
            rows = sorted(zip(grp["start_frame"].astype(int), grp["end_frame"].astype(int), grp["action_id"].astype(int)))
            for (s1, e1, i1), (s2, e2, i2) in zip(rows[:-1], rows[1:]):
                if s2 <= e1:
                    probs.append(f"actions {i1} and {i2} ({t}, possession {int(p)}) overlap: [{s1}, {e1}] and [{s2}, {e2}]")

    if frames_df is not None and possessions_df is not None and "offense_team_id" in possessions_df.columns:
        team_of = {int(p): int(t) for p, t in zip(frames_df["player_id"], frames_df["team_id"])}
        offense = {(str(g), int(p)): (None if pd.isna(t) else int(t)) for g, p, t in zip(
            possessions_df["sportvu_game"], possessions_df["possession"], possessions_df["offense_team_id"])}

        def side_problem(tag, role, pid, g, p, want_offense):
            off = offense.get((str(g), int(p)))
            if off is None or pid is None or pd.isna(pid):
                return None
            team = team_of.get(int(pid))
            if team is None:
                return f"{tag}: {role} {int(pid)} is not in the frames"
            if (team == off) != want_offense:
                return f"{tag}: {role} {int(pid)} is on the {'defence' if want_offense else 'offence'}"
            return None

        for r in acts.itertuples(index=False):
            tag = f"action {int(r.action_id)} ({r.action_type}, possession {int(r.possession)})"
            for role, pid in (("ball_handler_id", r.ball_handler_id), ("screener_id", r.screener_id)):
                pr = side_problem(tag, role, pid, r.sportvu_game, r.possession, True)
                if pr:
                    probs.append(pr)
        for r in scrs.itertuples(index=False):
            tag = f"screen {int(r.screen_id)} (possession {int(r.possession)})"
            for role, pid in (("on_ball_defender_id", r.on_ball_defender_id), ("screener_defender_id", r.screener_defender_id)):
                pr = side_problem(tag, role, pid, r.sportvu_game, r.possession, False)
                if pr:
                    probs.append(pr)

    if len(scrs):
        ball = {(str(g), int(p), int(a)) for g, p, a, t in zip(
            acts["sportvu_game"], acts["possession"], acts["action_id"], acts["action_type"]) if t == "ball_screen"}
        for r in scrs.itertuples(index=False):
            tag = f"screen {int(r.screen_id)} (possession {int(r.possession)})"
            if (str(r.sportvu_game), int(r.possession), int(r.action_id)) not in ball:
                probs.append(f"{tag}: action_id {int(r.action_id)} is not a ball_screen action of that game and possession")
            if int(r.on_ball_defender_id) == int(r.screener_defender_id):
                probs.append(f"{tag}: on_ball_defender_id equals screener_defender_id")
    return probs


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def estimate_fps(frames_df: pd.DataFrame) -> float:
    """Frames per second from the median clock step of the frame-level rows (DEFAULT_FPS when unknown)."""
    fl = frames_df.drop_duplicates("frame")
    if len(fl) < 3 or "game_clock_s" not in fl.columns:
        return DEFAULT_FPS
    d = -np.diff(fl.sort_values("frame")["game_clock_s"].to_numpy(dtype="float64"))
    d = d[(d > 0) & (d < 1.0)]
    if len(d) == 0:
        return DEFAULT_FPS
    step = float(np.median(d))
    return DEFAULT_FPS if step <= 0 else 1.0 / step


def _span_rows(frames_df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    f = frames_df["frame"].to_numpy()
    return frames_df.loc[(f >= start) & (f <= end)]


def is_half_court(span: pd.DataFrame, fps: float) -> bool:
    """The ball crosses MIDCOURT_X inside the span and then spends at least HALF_COURT_DWELL_S (summed
    over the frames after the first crossing) in the new half."""
    fl = span.drop_duplicates("frame").sort_values("frame")
    bx = fl["ball_x"].to_numpy(dtype="float64")
    bx = bx[~np.isnan(bx)]
    if len(bx) < 2:
        return False
    side = np.sign(bx - MIDCOURT_X)
    side[side == 0] = 1
    change = np.nonzero(side[1:] != side[:-1])[0]
    if len(change) == 0:
        return False
    k = int(change[0]) + 1
    new_side = side[k]
    dwell = int(np.sum(side[k:] == new_side))
    return dwell >= int(round(HALF_COURT_DWELL_S * fps))


def is_screen_candidate(span: pd.DataFrame, offense_team_id, fps: float) -> bool:
    """A non-handler offensive player within SCREEN_CANDIDATE_FT of the handler's nearest defender for
    SCREEN_CANDIDATE_S of consecutive frames. The handler is the offensive player nearest the ball."""
    if offense_team_id is None or pd.isna(offense_team_id):
        return False
    off_id = int(offense_team_id)
    need = max(1, int(round(SCREEN_CANDIDATE_S * fps)))
    sp = span.sort_values("frame", kind="stable")
    frames = sp["frame"].to_numpy()
    x = sp["x"].to_numpy(dtype="float64")
    y = sp["y"].to_numpy(dtype="float64")
    bx = sp["ball_x"].to_numpy(dtype="float64")
    by = sp["ball_y"].to_numpy(dtype="float64")
    is_off = sp["team_id"].to_numpy() == off_id
    bounds = np.nonzero(np.diff(frames))[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(frames)]])
    run = 0
    for a, b in zip(starts, ends):
        ok = False
        oi = np.nonzero(is_off[a:b])[0] + a
        di = np.nonzero(~is_off[a:b])[0] + a
        if len(oi) >= 2 and len(di) >= 1 and not np.isnan(bx[a]) and not np.isnan(by[a]):
            d_ball = np.hypot(x[oi] - bx[a], y[oi] - by[a])
            h = oi[int(np.argmin(d_ball))]
            d_def = np.hypot(x[di] - x[h], y[di] - y[h])
            nd = di[int(np.argmin(d_def))]
            others = oi[oi != h]
            ok = bool(np.min(np.hypot(x[others] - x[nd], y[others] - y[nd])) <= SCREEN_CANDIDATE_FT)
        run = run + 1 if ok else 0
        if run >= need:
            return True
    return False


def possession_flags(possessions_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    """Per possession: ``half_court`` and ``screen_candidate`` (plus the keys and offense_team_id)."""
    fps = estimate_fps(frames_df)
    rows = []
    for r in possessions_df.itertuples(index=False):
        span = _span_rows(frames_df, int(r.start_frame), int(r.end_frame))
        hc = is_half_court(span, fps) if len(span) else False
        sc = is_screen_candidate(span, r.offense_team_id, fps) if len(span) else False
        rows.append({"sportvu_game": str(r.sportvu_game), "possession": int(r.possession),
                     "offense_team_id": None if pd.isna(r.offense_team_id) else int(r.offense_team_id),
                     "half_court": bool(hc), "screen_candidate": bool(sc)})
    out = pd.DataFrame(rows, columns=["sportvu_game", "possession", "offense_team_id", "half_court", "screen_candidate"])
    out["offense_team_id"] = out["offense_team_id"].astype("Int32")
    return out


def is_heldout(game: str, heldout_tokens: Iterable[str] | None) -> bool:
    g = str(game)
    return any(t and str(t) in g for t in (heldout_tokens or ()))


def sample_possessions(possessions_df: pd.DataFrame, frames_df: pd.DataFrame, n: int, seed: int = 0,
                       heldout_tokens: Iterable[str] | None = None, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """``n`` possessions to label, stratified as the labeling guide prescribes.

    1. training games only: a row whose sportvu_game carries any of ``heldout_tokens`` is dropped;
    2. half-court possessions only (``is_half_court``);
    3. the quota is split evenly across offensive teams (the remainder goes to the teams with more
       eligible possessions); a team with fewer eligible possessions than its quota gives the shortfall to
       the other teams, and what cannot be placed anywhere is ``attrs["n_short"]``;
    4. within each team at least half of the quota comes from screen-candidate possessions when that
       many exist, the rest from the others; a short stratum is topped up from the other one.
    ``flags`` is a ``possession_flags`` frame computed earlier (it is computed here when absent). Rows are
    returned in a seeded random order with the flag columns and ``sample_rank``; attrs hold the counts per
    stratum and the number of held-out rows refused.
    """
    toks = set(str(t) for t in (heldout_tokens or ()) if t)
    poss = possessions_df.copy()
    held = poss["sportvu_game"].astype(str).map(lambda g: is_heldout(g, toks))
    n_held = int(held.sum())
    poss = poss.loc[~held]
    if flags is None:
        flags = possession_flags(poss, frames_df) if len(poss) else possession_flags(possessions_df.iloc[:0], frames_df)
    else:
        keep = set(zip(poss["sportvu_game"].astype(str), poss["possession"].astype(int)))
        flags = flags.loc[[(str(g), int(p)) in keep for g, p in zip(flags["sportvu_game"], flags["possession"])]]
    elig = flags.loc[flags["half_court"] & flags["offense_team_id"].notna()].reset_index(drop=True)
    rng = np.random.RandomState(int(seed))
    teams = sorted(int(t) for t in elig["offense_team_id"].dropna().unique())
    picked: list[pd.DataFrame] = []
    counts: dict = {}
    n_short = 0
    if teams and n > 0:
        base, rem = divmod(int(n), len(teams))
        by_size = sorted(teams, key=lambda t: (-int((elig["offense_team_id"] == t).sum()), t))
        quota = {t: base + (1 if i < rem else 0) for i, t in enumerate(by_size)}
        avail = {t: int((elig["offense_team_id"] == t).sum()) for t in teams}
        # a team short of its quota hands the shortfall to the teams with spare eligible possessions
        spare = sum(max(0, quota[t] - avail[t]) for t in teams)
        quota = {t: min(quota[t], avail[t]) for t in teams}
        for t in sorted(teams, key=lambda t: (-(avail[t] - quota[t]), t)):
            give = min(spare, avail[t] - quota[t])
            quota[t] += give
            spare -= give
        n_short = spare
        for t in teams:
            e = elig.loc[elig["offense_team_id"] == t]
            cand = e.loc[e["screen_candidate"]]
            rest = e.loc[~e["screen_candidate"]]
            q = min(quota[t], len(e))
            k_cand = min(len(cand), max(int(math.ceil(q / 2)), q - len(rest)))
            k_rest = q - k_cand
            ci = rng.choice(cand.index.to_numpy(), size=k_cand, replace=False) if k_cand else np.array([], dtype=int)
            ri = rng.choice(rest.index.to_numpy(), size=k_rest, replace=False) if k_rest else np.array([], dtype=int)
            picked.append(elig.loc[np.concatenate([ci, ri]).astype(int)])
            counts[t] = {"eligible": int(len(e)), "candidates": int(len(cand)), "drawn": int(q),
                         "drawn_candidates": int(k_cand)}
    out = pd.concat(picked, ignore_index=True) if picked else elig.iloc[:0].copy()
    if len(out):
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out["sample_rank"] = np.arange(len(out), dtype=np.int32)
    out.attrs["strata"] = counts
    out.attrs["n_short"] = int(n_short)
    out.attrs["n_heldout_refused"] = n_held
    out.attrs["n_half_court"] = int(flags["half_court"].sum()) if len(flags) else 0
    out.attrs["seed"] = int(seed)
    return out


# --------------------------------------------------------------------------------------
# Consistency: Cohen's kappa between labeller passes
# --------------------------------------------------------------------------------------

def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa of two equal-length category sequences (nan when empty; 1.0 on perfect agreement
    with a single category, 0.0 when chance agreement is total but the raters disagree)."""
    a = [str(v) for v in a]
    b = [str(v) for v in b]
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _unpack(labels):
    if isinstance(labels, Mapping):
        return labels.get("actions"), labels.get("screens")
    if isinstance(labels, (tuple, list)) and len(labels) == 2:
        return labels[0], labels[1]
    if isinstance(labels, pd.DataFrame):
        return labels, None
    raise TypeError("labels must be (actions_df, screens_df) or {'actions': ..., 'screens': ...}")


def _overlap(s1, e1, s2, e2) -> int:
    return max(0, min(e1, e2) - max(s1, s2) + 1)


def pair_actions(actions1: pd.DataFrame, actions2: pd.DataFrame) -> list[tuple]:
    """Greedy pairing of two passes' actions by temporal overlap inside each (game, possession):
    (row1 or None, row2 or None) tuples, where rows are namedtuples of the action frames."""
    a1 = cast(actions1, "actions") if actions1 is not None and len(actions1) else empty("actions")
    a2 = cast(actions2, "actions") if actions2 is not None and len(actions2) else empty("actions")
    keys = sorted(set(zip(a1["sportvu_game"].astype(str), a1["possession"].astype(int))) |
                  set(zip(a2["sportvu_game"].astype(str), a2["possession"].astype(int))))
    pairs: list[tuple] = []
    for g, p in keys:
        r1 = [r for r in a1.itertuples(index=False) if str(r.sportvu_game) == g and int(r.possession) == p]
        r2 = [r for r in a2.itertuples(index=False) if str(r.sportvu_game) == g and int(r.possession) == p]
        cands = sorted(((_overlap(int(x.start_frame), int(x.end_frame), int(y.start_frame), int(y.end_frame)), i, j)
                        for i, x in enumerate(r1) for j, y in enumerate(r2)), key=lambda t: (-t[0], t[1], t[2]))
        used1, used2 = set(), set()
        for ov, i, j in cands:
            if ov <= 0:
                break
            if i in used1 or j in used2:
                continue
            used1.add(i)
            used2.add(j)
            pairs.append((r1[i], r2[j]))
        pairs += [(r1[i], None) for i in range(len(r1)) if i not in used1]
        pairs += [(None, r2[j]) for j in range(len(r2)) if j not in used2]
    return pairs


def kappa(labels_pass1, labels_pass2) -> dict:
    """Cohen's kappa between two labeller passes.

    ``action_type``: over paired actions (``pair_actions``), an action the other pass lacks counts as
    ``none``. ``coverage``: over paired ball_screen actions, the ``sure`` coverage each pass attached
    (``none`` when a pass attached no sure screen label; ``unsure`` labels are left out of the kappa and of
    the first training set). Returns ``{"action_type", "coverage", "n_actions", "n_screens"}`` with nan
    where a pass has no items.
    """
    acts1, scr1 = _unpack(labels_pass1)
    acts2, scr2 = _unpack(labels_pass2)
    scr1 = scr1.loc[scr1["confidence"] == "sure"] if scr1 is not None and len(scr1) else scr1
    scr2 = scr2.loc[scr2["confidence"] == "sure"] if scr2 is not None and len(scr2) else scr2
    pairs = pair_actions(acts1, acts2)
    t1 = [("none" if x is None else str(x.action_type)) for x, _ in pairs]
    t2 = [("none" if y is None else str(y.action_type)) for _, y in pairs]

    def cov_of(scr, row):
        if scr is None or row is None or len(scr) == 0:
            return "none"
        m = scr.loc[(scr["sportvu_game"].astype(str) == str(row.sportvu_game)) &
                    (scr["possession"].astype(int) == int(row.possession)) &
                    (scr["action_id"].astype(int) == int(row.action_id))]
        return "none" if len(m) == 0 else str(m["coverage"].iloc[-1])

    c1, c2 = [], []
    for x, y in pairs:
        if (x is not None and str(x.action_type) == "ball_screen") or (y is not None and str(y.action_type) == "ball_screen"):
            c1.append(cov_of(scr1, x))
            c2.append(cov_of(scr2, y))
    return {"action_type": cohen_kappa(t1, t2), "coverage": cohen_kappa(c1, c2),
            "n_actions": len(pairs), "n_screens": len(c1)}


__all__ = [
    "ACTION_TYPES", "COVERAGES", "LABEL_SOURCES", "CONFIDENCES", "SCREENER_REQUIRED", "ENUMS",
    "ACTIONS_SCHEMA", "SCREENS_SCHEMA", "SCHEMAS", "ID_FIELD",
    "MIDCOURT_X", "HALF_COURT_DWELL_S", "SCREEN_CANDIDATE_FT", "SCREEN_CANDIDATE_S", "DEFAULT_FPS",
    "LABELLER_PASSES", "VISITED_COLUMNS",
    "pandas_dtypes", "empty", "cast", "now_iso",
    "labels_path", "write_labels", "append_tombstone", "read_labels", "next_id", "list_games",
    "visited_path", "mark_visited", "read_visited",
    "validate_labels",
    "estimate_fps", "is_half_court", "is_screen_candidate", "possession_flags", "is_heldout", "sample_possessions",
    "cohen_kappa", "pair_actions", "kappa",
]
