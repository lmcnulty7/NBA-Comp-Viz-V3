"""analytics.schema: the Phase 0 contract tables (docs/DATA_CONTRACT.md sections 16 to 20)."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from analytics import schema as S

ALL_TABLES = sorted(S.TABLES)


def synthetic(table: str, n: int = 3) -> pd.DataFrame:
    """A conforming frame of n rows for ``table``: enum columns take their first allowed value,
    nullable columns get one null, ids count up."""
    rows = {}
    for col, typ in S.TABLES[table].items():
        allowed = S.ENUMS.get((table, col))
        if allowed is not None:
            vals = [allowed[i % len(allowed)] for i in range(n)]
        elif pa.types.is_string(typ):
            vals = [f"{col}_{i}" for i in range(n)]
        elif pa.types.is_boolean(typ):
            vals = [bool(i % 2) for i in range(n)]
        elif pa.types.is_floating(typ):
            vals = [float(i) + 0.5 for i in range(n)]
        else:
            vals = [i + 1 for i in range(n)]
        if col in S.NULLABLE[table] and n > 1:
            vals = list(vals)
            vals[-1] = None
        rows[col] = vals
    df = pd.DataFrame(rows)
    df["sportvu_game"] = "12.16.2015.PHX.at.GSW"
    return df


# ---------------------------------------------------------------- schemas build

@pytest.mark.parametrize("table", ALL_TABLES)
def test_schema_builds(table):
    sch = S.schema(table)
    assert isinstance(sch, pa.Schema)
    assert sch.names == list(S.TABLES[table])
    for field in sch:
        assert field.nullable == (field.name in S.NULLABLE[table]), field.name


def test_five_tables_and_key_columns():
    assert ALL_TABLES == ["possessions", "rebound_frames", "rosters", "shot_frames", "states_1hz"]
    assert all(next(iter(S.TABLES[t])) == "sportvu_game" for t in ALL_TABLES)
    assert S.TABLES["possessions"]["points"] == pa.int16()
    assert S.TABLES["shot_frames"]["period"] == pa.int8()
    assert S.TABLES["states_1hz"]["points_to_end"] == pa.int16()
    for side in ("off", "def"):
        for i in range(1, 6):
            assert S.TABLES["states_1hz"][f"{side}{i}_id"] == pa.int32()
            assert S.TABLES["states_1hz"][f"{side}{i}_x"] == pa.float32()
            assert S.TABLES["states_1hz"][f"{side}{i}_y"] == pa.float32()


def test_unknown_table_raises():
    with pytest.raises(KeyError):
        S.schema("nope")


# ---------------------------------------------------------------- allowlist and leakage guard

@pytest.mark.parametrize("table", ALL_TABLES)
def test_feature_allowlist_excludes_targets_and_ids(table):
    allow = S.feature_allowlist(table)
    assert not set(allow) & S.TARGET_COLUMNS[table]
    assert not set(allow) & S.ID_COLUMNS
    assert all(c in S.TABLES[table] for c in allow)
    # Everything else is allowed.
    expected = [c for c in S.TABLES[table] if c not in S.TARGET_COLUMNS[table] and c not in S.ID_COLUMNS]
    assert allow == expected


def test_feature_allowlist_examples():
    assert "made" not in S.feature_allowlist("shot_frames")
    assert "shooter_id" not in S.feature_allowlist("shot_frames")
    assert "release_frame" in S.feature_allowlist("shot_frames")
    assert "end_kind" not in S.feature_allowlist("possessions")
    assert "possession_end_kind" not in S.feature_allowlist("states_1hz")
    assert "handler_nearest_def_dist_ft" in S.feature_allowlist("states_1hz")


def test_assert_time_forward_raises_on_target():
    df = synthetic("states_1hz")
    with pytest.raises(ValueError, match="points_to_end"):
        S.assert_time_forward(df, "states_1hz")
    feats = df[S.feature_allowlist("states_1hz")]
    S.assert_time_forward(feats, "states_1hz")  # no raise
    with pytest.raises(ValueError, match="made"):
        S.assert_time_forward(pd.DataFrame({"made": [True], "release_frame": [3]}), "shot_frames")
    S.assert_time_forward(pd.DataFrame({"anything": [1]}), "rosters")


# ---------------------------------------------------------------- validate

@pytest.mark.parametrize("table", ALL_TABLES)
def test_validate_passes_conforming_frame(table):
    assert S.validate(synthetic(table), table) == []
    # And the cast output is conforming too.
    assert S.validate(S.cast(synthetic(table), table), table) == []


def test_validate_flags_bad_enum():
    df = synthetic("possessions")
    df.loc[0, "end_kind"] = "steal"
    problems = S.validate(df, "possessions")
    assert len(problems) == 1 and "end_kind" in problems[0] and "steal" in problems[0]

    df = synthetic("rosters")
    df.loc[1, "match_method"] = "guess"
    assert any("match_method" in p for p in S.validate(df, "rosters"))


def test_validate_flags_missing_column():
    df = synthetic("shot_frames").drop(columns=["confidence_flag"])
    problems = S.validate(df, "shot_frames")
    assert problems == ["missing column 'confidence_flag'"]


def test_validate_flags_uncastable_and_null_in_non_nullable():
    df = synthetic("rebound_frames")
    df["rim_frame"] = ["a", "b", "c"]
    assert any("rim_frame" in p and "cannot be cast" in p for p in S.validate(df, "rebound_frames"))
    df = synthetic("possessions")
    df["n_frames"] = [1, None, 3]
    assert any("n_frames" in p for p in S.validate(df, "possessions"))


# ---------------------------------------------------------------- cast

def test_cast_returns_contract_dtypes():
    df = synthetic("shot_frames")
    # Scramble the input: python ints, floats as float64, strings as object, columns reversed.
    df = df[list(reversed(df.columns))]
    out = S.cast(df, "shot_frames")
    assert list(out.columns) == list(S.TABLES["shot_frames"])
    assert out["event"].dtype == np.dtype("int32")
    assert out["period"].dtype == np.dtype("int8")
    assert out["clock_s"].dtype == np.dtype("float32")
    assert out["confidence"].dtype == np.dtype("float32")
    assert out["made"].dtype == np.dtype("bool")
    assert str(out["shooter_id"].dtype) == "Int32"
    assert str(out["shooter_at_release_id"].dtype) == "Int32"
    assert str(out["identity_match"].dtype) == "boolean"
    assert isinstance(out["pbp_shooter_name"].dtype, pd.StringDtype)
    assert out["shooter_id"].isna().sum() == 1
    assert out["release_frame"].tolist() == [1, 2, 3]


def test_cast_possessions_and_states():
    out = S.cast(synthetic("possessions"), "possessions")
    assert out["points"].dtype == np.dtype("int16")
    assert out["gap"].dtype == np.dtype("bool")
    assert str(out["offense_team_id"].dtype) == "Int32"
    out = S.cast(synthetic("states_1hz"), "states_1hz")
    assert out["points_to_end"].dtype == np.dtype("int16")
    assert str(out["handler_id"].dtype) == "Int32"
    assert out["off1_x"].dtype == np.dtype("float32")


def test_cast_roundtrips_through_arrow():
    for table in ALL_TABLES:
        out = S.cast(synthetic(table), table)
        tbl = pa.Table.from_pandas(out, schema=S.schema(table), preserve_index=False)
        assert tbl.schema.equals(S.schema(table))
        back = tbl.to_pandas()
        assert S.validate(back, table) == []


def test_cast_raises_on_missing_or_null_in_non_nullable():
    with pytest.raises(KeyError):
        S.cast(synthetic("rosters").drop(columns=["team"]), "rosters")
    df = synthetic("rosters")
    df["player_id"] = [1, None, 3]
    with pytest.raises(ValueError, match="player_id"):
        S.cast(df, "rosters")


def test_empty_and_path_helpers():
    e = S.empty("rebound_frames")
    assert list(e.columns) == list(S.TABLES["rebound_frames"]) and len(e) == 0
    assert S.validate(e, "rebound_frames") == []
    p = S.table_path("analytics", "possessions", "2015-16", "12.16.2015.PHX.at.GSW")
    assert str(p) == "analytics/possessions/2015-16/12.16.2015.PHX.at.GSW.parquet"
    p = S.table_path("analytics", "possessions", "2015-16", "12.16.2015.PHX.at.GSW", heldout=True)
    assert str(p) == "analytics/possessions/heldout/2015-16/12.16.2015.PHX.at.GSW.parquet"
