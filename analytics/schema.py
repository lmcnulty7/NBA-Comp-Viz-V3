"""Phase 0 schema: the five foundation tables of docs/DATA_CONTRACT.md sections 16 to 20.

Every lane codes against the names and dtypes here. ``TABLES`` is the single source of truth
(column order and pyarrow type), ``NULLABLE`` marks the columns that may hold nulls, ``ENUMS`` the
closed string vocabularies, ``TARGET_COLUMNS`` the label columns a trainer must never see as a
feature, and ``ID_COLUMNS`` the identifiers that are never features either.

Layout on disk (one parquet file per game):
    analytics/<table>/<season>/<sportvu_game>.parquet
    analytics/<table>/heldout/<season>/<sportvu_game>.parquet   (sportvu/splits.json held-out games)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

# --------------------------------------------------------------------------------------
# Table definitions (section numbers refer to docs/DATA_CONTRACT.md)
# --------------------------------------------------------------------------------------

END_KINDS = ("shot_made", "shot_missed", "free_throws", "turnover", "period_end", "other")
CONFIDENCE_FLAGS = ("high", "low", "unmatched")
MATCH_METHODS = ("exact", "lastname", "fuzzy", "none")


def _states_1hz_columns() -> dict:
    cols = {
        "sportvu_game": pa.string(),
        "possession": pa.int32(),
        "frame": pa.int32(),
        "period": pa.int8(),
        "game_clock_s": pa.float32(),
        "shot_clock_s": pa.float32(),
        "elapsed_in_possession_s": pa.float32(),
        "offense_team_id": pa.int32(),
        "handler_id": pa.int32(),
        "ball_x": pa.float32(),
        "ball_y": pa.float32(),
        "ball_z": pa.float32(),
    }
    for side in ("off", "def"):
        for i in range(1, 6):
            cols[f"{side}{i}_id"] = pa.int32()
        for i in range(1, 6):
            cols[f"{side}{i}_x"] = pa.float32()
            cols[f"{side}{i}_y"] = pa.float32()
    cols["handler_nearest_def_dist_ft"] = pa.float32()
    cols["points_to_end"] = pa.int16()
    cols["possession_end_kind"] = pa.string()
    return cols


TABLES: dict[str, dict[str, pa.DataType]] = {
    # 16. rosters
    "rosters": {
        "sportvu_game": pa.string(),
        "player_id": pa.int32(),
        "player": pa.string(),
        "team_id": pa.int32(),
        "team": pa.string(),
        "home": pa.bool_(),
        "jersey": pa.string(),
        "bref_slug": pa.string(),
        "bref_name": pa.string(),
        "pbp_name": pa.string(),
        "match_method": pa.string(),
    },
    # 17. shot_frames
    "shot_frames": {
        "sportvu_game": pa.string(),
        "event": pa.int32(),
        "period": pa.int8(),
        "clock_s": pa.float32(),
        "made": pa.bool_(),
        "pbp_shooter_name": pa.string(),
        "shooter_id": pa.int32(),
        "release_frame": pa.int32(),
        "apex_frame": pa.int32(),
        "outcome_frame": pa.int32(),
        "shooter_at_release_id": pa.int32(),
        "identity_match": pa.bool_(),
        "confidence": pa.float32(),
        "confidence_flag": pa.string(),
    },
    # 18. rebound_frames
    "rebound_frames": {
        "sportvu_game": pa.string(),
        "event": pa.int32(),
        "rebound_event": pa.int32(),
        "period": pa.int8(),
        "clock_s": pa.float32(),
        "rim_frame": pa.int32(),
        "rebound_frame": pa.int32(),
        "rebounder_id": pa.int32(),
        "pbp_rebounder_name": pa.string(),
        "pbp_rebounder_id": pa.int32(),
        "identity_match": pa.bool_(),
        "confidence": pa.float32(),
        "confidence_flag": pa.string(),
    },
    # 19. possessions
    "possessions": {
        "sportvu_game": pa.string(),
        "possession": pa.int32(),
        "period": pa.int8(),
        "start_frame": pa.int32(),
        "end_frame": pa.int32(),
        "start_clock_s": pa.float32(),
        "end_clock_s": pa.float32(),
        "start_event": pa.int32(),
        "end_event": pa.int32(),
        "offense_team_id": pa.int32(),
        "end_kind": pa.string(),
        "points": pa.int16(),
        "gap": pa.bool_(),
        "n_frames": pa.int32(),
    },
    # 20. states_1hz (Phase 2 fills it)
    "states_1hz": _states_1hz_columns(),
}

# Columns that may hold nulls. Every other column is non-nullable.
NULLABLE: dict[str, frozenset] = {
    "rosters": frozenset({"jersey", "bref_slug", "bref_name", "pbp_name"}),
    "shot_frames": frozenset({"pbp_shooter_name", "shooter_id", "shooter_at_release_id", "identity_match"}),
    "rebound_frames": frozenset({"rebound_event", "rebounder_id", "pbp_rebounder_name",
                                 "pbp_rebounder_id", "identity_match"}),
    "possessions": frozenset({"offense_team_id"}),
    "states_1hz": frozenset({"shot_clock_s", "handler_id", "handler_nearest_def_dist_ft"}),
}

# Closed vocabularies of the string enum columns: (table, column) -> allowed values.
ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("rosters", "match_method"): MATCH_METHODS,
    ("shot_frames", "confidence_flag"): CONFIDENCE_FLAGS,
    ("rebound_frames", "confidence_flag"): CONFIDENCE_FLAGS,
    ("possessions", "end_kind"): END_KINDS,
    ("states_1hz", "possession_end_kind"): END_KINDS,
}

# Label columns: never a feature. assert_time_forward refuses a frame that holds any of them.
TARGET_COLUMNS: dict[str, set[str]] = {
    "rosters": set(),
    "shot_frames": {"made", "identity_match", "confidence", "confidence_flag"},
    "rebound_frames": {"rebounder_id", "pbp_rebounder_id", "pbp_rebounder_name", "identity_match", "confidence", "confidence_flag"},
    "possessions": {"end_kind", "points"},
    "states_1hz": {"points_to_end", "possession_end_kind"},
}

# Columns measured at or after the outcome: excluded from feature frames exactly like targets.
POST_OUTCOME: dict[str, set[str]] = {
    "rosters": set(),
    "shot_frames": {"apex_frame", "outcome_frame", "pbp_shooter_name"},
    "rebound_frames": {"rebound_frame"},
    "possessions": {"end_frame", "end_clock_s", "end_event", "n_frames"},
    "states_1hz": set(),
}

# Every target and post-outcome column of every table. assert_time_forward checks against this
# union, so a feature frame built for one table cannot carry another table's labels either.
ALL_EXCLUDED: frozenset = frozenset(
    set().union(*TARGET_COLUMNS.values()).union(*POST_OUTCOME.values()))

# Identifier columns: join keys, never features.
ID_COLUMNS: frozenset = frozenset(
    {"sportvu_game", "event", "rebound_event", "possession", "frame", "player_id", "shooter_id"})

# Sentinel for a frame index that was not found (release_frame, apex_frame, outcome_frame,
# rim_frame, rebound_frame).
NOT_FOUND = -1


# --------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------

def _check_table(table: str) -> None:
    if table not in TABLES:
        raise KeyError(f"unknown table {table!r}; known tables: {sorted(TABLES)}")


def columns(table: str) -> list[str]:
    """Contract column names of ``table`` in order."""
    _check_table(table)
    return list(TABLES[table])


def is_nullable(table: str, column: str) -> bool:
    _check_table(table)
    return column in NULLABLE[table]


def schema(table: str) -> pa.Schema:
    """The pyarrow schema of ``table`` (columns in contract order, nullability as the contract)."""
    _check_table(table)
    fields = [pa.field(name, typ, nullable=name in NULLABLE[table]) for name, typ in TABLES[table].items()]
    return pa.schema(fields)


def feature_allowlist(table: str) -> list[str]:
    """Columns of ``table`` that are neither targets, post-outcome columns nor ids, in contract order."""
    _check_table(table)
    excluded = TARGET_COLUMNS[table] | POST_OUTCOME[table]
    return [c for c in TABLES[table] if c not in excluded and c not in ID_COLUMNS]


def assert_time_forward(df: pd.DataFrame, table: str) -> None:
    """Leakage guard: raise ValueError naming every target or post-outcome column present in ``df``.

    The check is against ``ALL_EXCLUDED``, the union over all five tables, so a ``states_1hz``
    feature frame holding ``points`` or ``made`` raises too. Every trainer calls this on its
    feature frame before fitting.
    """
    _check_table(table)
    leaked = [c for c in df.columns if c in ALL_EXCLUDED]
    if leaked:
        raise ValueError(
            f"feature frame for {table!r} holds target or post-outcome column(s) {leaked}; "
            "these are never features (docs/DATA_CONTRACT.md section 20)")


def table_path(root, table: str, season: str, sportvu_game: str, heldout: bool = False) -> Path:
    """analytics/<table>/<season>/<sportvu_game>.parquet, with ``heldout/`` inserted for held-out games."""
    _check_table(table)
    base = Path(root) / table
    if heldout:
        base = base / "heldout"
    return base / season / f"{sportvu_game}.parquet"


# --------------------------------------------------------------------------------------
# pandas dtypes
# --------------------------------------------------------------------------------------

def _pandas_dtype(typ: pa.DataType, nullable: bool):
    """The pandas dtype that holds a contract column: numpy where non-nullable, the pandas
    nullable extension dtype where nullable (ints as Int32, bools as boolean). Strings are the
    pandas string dtype either way; floats are float32 either way (NaN is the null)."""
    if pa.types.is_string(typ):
        return pd.StringDtype()
    if pa.types.is_boolean(typ):
        return pd.BooleanDtype() if nullable else np.dtype("bool")
    if pa.types.is_floating(typ):
        return np.dtype("float32")
    if pa.types.is_integer(typ):
        bits = typ.bit_width
        return pd.api.types.pandas_dtype(f"Int{bits}") if nullable else np.dtype(f"int{bits}")
    raise TypeError(f"unsupported contract type {typ}")


def pandas_dtypes(table: str) -> dict[str, object]:
    """column -> pandas dtype of the contract for ``table``."""
    _check_table(table)
    return {c: _pandas_dtype(t, c in NULLABLE[table]) for c, t in TABLES[table].items()}


def _cast_column(s: pd.Series, dtype) -> pd.Series:
    if isinstance(dtype, pd.StringDtype):
        out = s.astype("object")
        mask = out.isna()
        out = out.where(~mask, None)
        # Non-strings (e.g. numeric jersey numbers) become their str form.
        out = out.map(lambda v: v if (v is None or isinstance(v, str)) else str(v))
        return out.astype(dtype)
    if isinstance(dtype, pd.BooleanDtype):
        return s.astype("boolean")
    if isinstance(dtype, np.dtype) and dtype.kind == "b":
        if s.isna().any():
            raise ValueError("null in non-nullable bool column")
        return s.astype(bool)
    if isinstance(dtype, np.dtype) and dtype.kind == "i":
        if s.isna().any():
            raise ValueError("null in non-nullable integer column")
        vals = pd.to_numeric(s)
        if not np.all(np.asarray(vals, dtype="float64") == np.floor(np.asarray(vals, dtype="float64"))):
            raise ValueError("non-integral value in integer column")
        return vals.astype(dtype)
    if isinstance(dtype, np.dtype) and dtype.kind == "f":
        return pd.to_numeric(s).astype(dtype)
    # pandas nullable integer
    vals = pd.to_numeric(s)
    return vals.astype(dtype)


def cast(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return ``df`` with the contract dtypes and columns in contract order.

    Nullable ints become pandas ``Int32`` (``Int8``, ``Int16`` for the smaller types), nullable bools
    ``boolean``, strings the pandas string dtype, floats ``float32``. A missing column raises KeyError;
    an impossible conversion (null in a non-nullable int, text in a number) raises ValueError.
    Extra columns are dropped.
    """
    _check_table(table)
    missing = [c for c in TABLES[table] if c not in df.columns]
    if missing:
        raise KeyError(f"{table}: missing column(s) {missing}")
    out = {}
    for col, dtype in pandas_dtypes(table).items():
        try:
            out[col] = _cast_column(df[col], dtype)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{table}.{col}: cannot cast to {dtype}: {exc}") from exc
    return pd.DataFrame(out, index=df.index)


def empty(table: str) -> pd.DataFrame:
    """An empty frame with the contract dtypes and column order."""
    return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in pandas_dtypes(table).items()})


def _dtype_ok(actual, expected) -> bool:
    if isinstance(expected, pd.StringDtype):
        return isinstance(actual, pd.StringDtype)
    return actual == expected


def validate(df: pd.DataFrame, table: str) -> list[str]:
    """Problems with ``df`` against the contract of ``table``; empty when the frame conforms.

    Checks: missing columns, dtype after a cast attempt (a column that cannot be cast, or whose
    cast dtype differs from the contract), nulls in non-nullable columns, and unknown values in
    the string enum columns. Extra columns are not a problem (collectors may carry provenance).
    """
    _check_table(table)
    problems: list[str] = []
    expected = pandas_dtypes(table)
    present = [c for c in TABLES[table] if c in df.columns]
    for c in TABLES[table]:
        if c not in df.columns:
            problems.append(f"missing column {c!r}")
    for col in present:
        dtype = expected[col]
        try:
            casted = _cast_column(df[col], dtype)
        except (ValueError, TypeError) as exc:
            problems.append(f"column {col!r} cannot be cast to {dtype}: {exc}")
            continue
        if not _dtype_ok(casted.dtype, dtype):
            problems.append(f"column {col!r} has dtype {casted.dtype} after cast, contract says {dtype}")
        if col not in NULLABLE[table] and casted.isna().any():
            problems.append(f"column {col!r} is not nullable but holds {int(casted.isna().sum())} null(s)")
        allowed = ENUMS.get((table, col))
        if allowed is not None:
            values = set(casted.dropna().unique().tolist())
            bad = sorted(values - set(allowed))
            if bad:
                problems.append(f"column {col!r} has unknown enum value(s) {bad}; allowed {list(allowed)}")
    return problems


__all__ = [
    "TABLES", "NULLABLE", "ENUMS", "TARGET_COLUMNS", "POST_OUTCOME", "ALL_EXCLUDED", "ID_COLUMNS",
    "NOT_FOUND",
    "END_KINDS", "CONFIDENCE_FLAGS", "MATCH_METHODS",
    "columns", "is_nullable", "schema", "feature_allowlist", "assert_time_forward", "table_path",
    "pandas_dtypes", "cast", "empty", "validate",
]
