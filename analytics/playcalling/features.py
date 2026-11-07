"""analytics/playcalling/features.py: the action-level table (docs/contract/playcalling.md, action_outcomes).

    python -m analytics.playcalling.features --fixtures | --data DATA_ROOT [--out OUT_ROOT] [--season 2015-16]

One row per action event. Each row joins the action to its possession (docs/DATA_CONTRACT.md section 19),
its coverage (hand screen label, else the coverage lane's prediction, else "none"), the two lineups on the
floor at the possession's first frame (the ten player ids of that frame, split by team against
offense_team_id, each side sorted ascending and joined with "-"), the score margin and the shot clock at the
action's start frame, the period, and the outcomes: ppp (the possession's points) and epv_gain (the EPV
lane's epv_points at the action's end frame minus at its start frame; null when no epv_states exist or
either frame is missing).

Leakage rules. The outcomes are never features: ``features`` returns only ``FEATURES`` plus the id columns
and calls ``analytics.schema.assert_time_forward`` on that frame (its names are checked against every target
and post-outcome column of the five foundation tables). The score margin uses only possessions that ended
before this one (earlier possession numbers of the same game): their points are known at the action's start.
Lineups come from the possession's first frame, never from a later one. The shot clock is read at the
action's start frame.

Assumptions (the upstream lanes were built in parallel; see analytics/playcalling/__init__.py):
  - actions: sportvu_game, possession, action_type, start_frame, end_frame, ball_handler_id, screener_id,
    label_source; action_id is used when present, else assigned as the row's position within its game;
  - screens (hand): sportvu_game, possession, action_id, coverage; coverage_predictions: the same keys with
    ``predicted`` (or ``coverage``) and p_<class> columns;
  - epv_states: sportvu_game, possession, frame, epv_points;
  - frames: section 4 (game, frame, team_id, player_id, shot_clock_s);
  - a possession whose first frame does not hold exactly five players per side gets null lineups, counted in
    the build log and kept (the ridge treats a null lineup as its own "unknown" level).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import schema
from analytics.playcalling import SEASON, WHO

FEATURES = ("action_type", "coverage", "off_lineup", "def_lineup", "score_margin", "shot_clock_s", "period")
ID_COLUMNS = ("sportvu_game", "possession", "action_id")
OUTCOMES = ("ppp", "epv_gain")
ACTION_OUTCOMES_COLUMNS = ("sportvu_game", "possession", "action_id", "action_type", "coverage", "off_lineup", "def_lineup",
                           "ppp", "epv_gain", "heldout", "score_margin", "shot_clock_s", "period", "label_source",
                           "coverage_source", "start_frame", "end_frame")


def feature_columns() -> list[str]:
    """The feature columns of the action table, in order."""
    return list(FEATURES)


def assign_action_ids(actions: pd.DataFrame) -> pd.DataFrame:
    """actions with an action_id column: the given ids kept; a row without one (or the whole table without the column)
    takes the 0-based position of the row within its game in the given row order (the convention of
    analytics/coverage and analytics/fixtures: jsonl line order). Existing ids are never overwritten."""
    a = actions.copy()
    if "action_id" not in a.columns:
        a["action_id"] = a.groupby("sportvu_game", sort=False).cumcount()
    else:
        m = pd.to_numeric(a["action_id"], errors="coerce").isna()
        if m.any():          # only the rows without an id (e.g. detected rows beside hand labels) take positions
            a["action_id"] = pd.to_numeric(a["action_id"], errors="coerce").astype(object)
            a.loc[m, "action_id"] = a[m].groupby("sportvu_game", sort=False).cumcount().to_numpy()
    a["action_id"] = a["action_id"].astype(int)
    return a


def lineup_id(ids) -> str | None:
    """Five player ids sorted and joined with -; None unless exactly five ids are present."""
    ids = [int(i) for i in ids if pd.notna(i)]
    if len(ids) != 5:
        return None
    return "-".join(str(i) for i in sorted(ids))


def lineups_at_first_frame(frames: pd.DataFrame, possessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(sportvu_game, possession, off_lineup, def_lineup) from the ten players of each possession's start_frame.
    The log counts possessions whose first frame is missing or does not split five against five."""
    game_col = "game" if "game" in frames.columns else "sportvu_game"
    key = frames[[game_col, "frame", "team_id", "player_id"]].rename(columns={game_col: "sportvu_game"})
    need = possessions[["sportvu_game", "possession", "start_frame", "offense_team_id"]]
    merged = need.merge(key, left_on=["sportvu_game", "start_frame"], right_on=["sportvu_game", "frame"], how="left")
    rows, log = [], {"possessions": int(len(need)), "no_first_frame": 0, "not_five_v_five": 0, "no_offense_team": 0}
    for (g, p), grp in merged.groupby(["sportvu_game", "possession"], sort=False):
        off_team = grp["offense_team_id"].iloc[0]
        if grp["player_id"].isna().all():
            log["no_first_frame"] += 1
            rows.append((g, p, None, None))
            continue
        if pd.isna(off_team):
            log["no_offense_team"] += 1
            rows.append((g, p, None, None))
            continue
        off = lineup_id(grp.loc[grp["team_id"] == off_team, "player_id"])
        de = lineup_id(grp.loc[grp["team_id"] != off_team, "player_id"])
        if off is None or de is None:
            log["not_five_v_five"] += 1
        rows.append((g, p, off, de))
    out = pd.DataFrame(rows, columns=["sportvu_game", "possession", "off_lineup", "def_lineup"])
    return out, log


def score_margin_before(possessions: pd.DataFrame) -> pd.Series:
    """Offence minus defence points scored on the earlier possessions of the same game (known at the start)."""
    p = possessions.sort_values(["sportvu_game", "possession"])
    out = pd.Series(np.nan, index=p.index, dtype=float)
    for g, grp in p.groupby("sportvu_game", sort=False):
        totals: dict = {}
        for idx, row in zip(grp.index, grp.itertuples(index=False)):
            off = row.offense_team_id
            if pd.isna(off):
                out[idx] = np.nan
            else:
                mine = totals.get(int(off), 0)
                theirs = sum(v for k, v in totals.items() if k != int(off))
                out[idx] = float(mine - theirs)
            if not pd.isna(off):
                totals[int(off)] = totals.get(int(off), 0) + int(row.points)
    return out.reindex(possessions.index)


def resolve_coverage(actions: pd.DataFrame, screens: pd.DataFrame | None, coverage_predictions: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
    """(coverage, coverage_source) per action row: the hand screen label, else the prediction, else "none"."""
    keys = ["sportvu_game", "possession", "action_id"]
    cov = pd.Series(None, index=actions.index, dtype=object)
    src = pd.Series("none", index=actions.index, dtype=object)
    if coverage_predictions is not None and len(coverage_predictions):
        cp = coverage_predictions.copy()
        col = "predicted" if "predicted" in cp.columns else "coverage"
        cp = cp.drop_duplicates(keys)[keys + [col]].rename(columns={col: "_pred"})
        m = actions[keys].merge(cp, on=keys, how="left")["_pred"].to_numpy()
        has = pd.notna(m)
        cov[has] = m[has]
        src[has] = "predicted"
    if screens is not None and len(screens):
        sc = screens.drop_duplicates(keys)[keys + ["coverage"]].rename(columns={"coverage": "_hand"})
        m = actions[keys].merge(sc, on=keys, how="left")["_hand"].to_numpy()
        has = pd.notna(m)
        cov[has] = m[has]
        src[has] = "hand"
    cov = cov.fillna("none").astype(str)
    return cov, src


def epv_gain(actions: pd.DataFrame, epv_states: pd.DataFrame | None) -> pd.Series:
    """epv_points at the action's end frame minus at its start frame (nearest state frame of the same possession
    at or before each; null when either side is missing or no epv_states were given)."""
    out = pd.Series(np.nan, index=actions.index, dtype=float)
    if epv_states is None or not len(epv_states):
        return out
    e = epv_states[["sportvu_game", "possession", "frame", "epv_points"]].sort_values(["sportvu_game", "possession", "frame"])
    groups = {k: (v["frame"].to_numpy(), v["epv_points"].to_numpy(dtype=float)) for k, v in e.groupby(["sportvu_game", "possession"], sort=False)}

    def at(fr, ep, f):
        i = np.searchsorted(fr, f, side="right") - 1
        return ep[i] if i >= 0 else np.nan

    for idx, r in zip(actions.index, actions[["sportvu_game", "possession", "start_frame", "end_frame"]].itertuples(index=False)):
        g = groups.get((r.sportvu_game, r.possession))
        if g is None:
            continue
        fr, ep = g
        out[idx] = at(fr, ep, int(r.end_frame)) - at(fr, ep, int(r.start_frame))
    return out


def build_action_table(frames: pd.DataFrame, possessions: pd.DataFrame, actions: pd.DataFrame, screens: pd.DataFrame | None = None,
                       coverage_predictions: pd.DataFrame | None = None, epv_states: pd.DataFrame | None = None,
                       heldout_games=()) -> tuple[pd.DataFrame, dict]:
    """The action_outcomes table of docs/contract/playcalling.md -> (table, build log)."""
    a = assign_action_ids(actions)
    pk = ["sportvu_game", "possession"]
    poss = possessions.copy()
    poss["_margin"] = score_margin_before(poss)
    lus, log = lineups_at_first_frame(frames, poss)
    t = a.merge(poss[pk + ["period", "points", "_margin"]], on=pk, how="inner").merge(lus, on=pk, how="left")
    log["actions_in"] = int(len(a))
    log["actions_without_possession"] = int(len(a) - len(t))
    cov, src = resolve_coverage(t, screens, coverage_predictions)
    t["coverage"], t["coverage_source"] = cov.to_numpy(), src.to_numpy()
    game_col = "game" if "game" in frames.columns else "sportvu_game"
    clock = (frames[[game_col, "frame", "shot_clock_s"]].drop_duplicates([game_col, "frame"])
             .rename(columns={game_col: "sportvu_game", "frame": "start_frame"}))
    t = t.merge(clock, on=["sportvu_game", "start_frame"], how="left")
    t["ppp"] = t["points"].astype(float)
    t["epv_gain"] = epv_gain(t, epv_states).to_numpy()
    held = set(map(str, heldout_games))
    t["heldout"] = t["sportvu_game"].astype(str).isin(held)
    t = t.rename(columns={"_margin": "score_margin"})
    if "label_source" not in t.columns:
        t["label_source"] = "hand"
    t["period"] = t["period"].astype(int)
    t = t.drop(columns=["points"])
    t = t[list(ACTION_OUTCOMES_COLUMNS)].sort_values(["sportvu_game", "possession", "action_id"]).reset_index(drop=True)
    log["rows"] = int(len(t))
    log["epv_gain_rows"] = int(t["epv_gain"].notna().sum())
    log["coverage_source"] = {k: int(v) for k, v in t["coverage_source"].value_counts().items()}
    log["null_lineups"] = int(t["off_lineup"].isna().sum())
    return t, log


def features(table: pd.DataFrame) -> pd.DataFrame:
    """The feature frame of the action table: id columns plus FEATURES, nothing else. Raises ValueError (through
    schema.assert_time_forward) when a target or post-outcome column of the foundation tables is present, and
    when an outcome of this table (ppp, epv_gain) is."""
    cols = list(ID_COLUMNS) + list(FEATURES) + (["heldout"] if "heldout" in table.columns else [])
    f = table[cols].copy()
    schema.assert_time_forward(f, "possessions")
    leaked = [c for c in f.columns if c in OUTCOMES]
    if leaked:
        raise ValueError("%s feature frame holds outcome column(s) %s" % (WHO, leaked))
    return f


def out_path(out_root, season: str, name: str, heldout: bool = False) -> Path:
    """OUT_ROOT/analytics/playcalling/[heldout/]<season>/<name>.parquet."""
    base = Path(out_root) / "analytics" / "playcalling"
    return (base / "heldout" / season if heldout else base / season) / (name + ".parquet")


OUTPUT_DTYPES = {
    "action_outcomes": {"sportvu_game": "string", "possession": "int32", "action_id": "int32", "action_type": "string",
                        "coverage": "string", "off_lineup": "string", "def_lineup": "string", "ppp": "float32",
                        "epv_gain": "float32", "heldout": "bool", "score_margin": "float32", "shot_clock_s": "float32",
                        "period": "int8", "label_source": "string", "coverage_source": "string", "start_frame": "int32",
                        "end_frame": "int32"},
    "ppp_by_action": {"action_type": "string", "coverage": "string", "n": "int32", "ppp": "float32", "lo": "float32",
                      "hi": "float32", "insufficient": "bool"},
    "playcalling_adjusted": {"coverage": "string", "action_type": "string", "adjusted_ppp": "float32", "lo": "float32",
                             "hi": "float32", "n": "int32", "insufficient": "bool", "recommended": "bool"},
}


def cast_output(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """The contract dtypes of docs/contract/playcalling.md for a written table."""
    out = df.copy()
    for c, d in OUTPUT_DTYPES[name].items():
        if c in out.columns:
            out[c] = out[c].astype(d)
    return out


def write_action_outcomes(table: pd.DataFrame, out_root, season: str = SEASON) -> list[Path]:
    """Write the training rows and the held-out rows to their two files with contract dtypes; returns the paths."""
    written = []
    for held in (False, True):
        part = cast_output(table[table["heldout"] == held], "action_outcomes")
        if held and not len(part):
            continue
        p = out_path(out_root, season, "action_outcomes", heldout=held)
        p.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(p, index=False)
        written.append(p)
    return written


def main(argv=None) -> dict:
    """Build the action table from the fixture or DATA_ROOT inputs, write action_outcomes under --out, print the logs."""
    from analytics.playcalling import validate as v
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", action="store_true", help="synthetic inputs (fixture output, not a result)")
    g.add_argument("--data", help="DATA_ROOT (sportvu_contract/, analytics/possessions/, labels/)")
    ap.add_argument("--out", default=None, help="OUT_ROOT for upstream outputs and this lane's tables (default DATA_ROOT)")
    ap.add_argument("--season", default=SEASON, help="season directory (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="fixture seed (default %(default)s)")
    v.add_fixture_args(ap)
    a = ap.parse_args(argv)
    inputs = v.fixture_inputs_from_args(a) if a.fixtures else v.load_data_root(a.data, a.out or a.data, a.season)
    inputs, vlog = v.validate_inputs(inputs)
    table, log = build_action_table(inputs["frames"], inputs["possessions"], inputs["actions"], inputs.get("screens"),
                                    inputs.get("coverage_predictions"), inputs.get("epv_states"), inputs.get("heldout_games", ()))
    features(table)
    written = []
    if a.out or a.data:
        written = write_action_outcomes(table, a.out or a.data, a.season)
    print(json.dumps({"validation": vlog, "build": log, "written": [str(p) for p in written]}, indent=1, default=str))
    return {"table": table, "log": log, "validation": vlog, "written": written}


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
