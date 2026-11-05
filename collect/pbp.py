"""collect/pbp.py: play-by-play for every SportVU game (collection plan step 3), run on the Mac.

Fills the <game>_pbp.parquet files that the Step 1 Colab job skipped with --no-pbp, in the Step 1 layout
(heldout/ for the held-out games), joined to the frames exactly as sportvu.colab_contract.join_pbp does
(same period, nearest game clock within 2 s). Source order per game:
  1. stats.nba.com PlayByPlayV2 (the NBA game id read from the frames parquet)
  2. basketball-reference.com (fetch_pbp.fetch_game) at --br-rpm requests a minute (17 by default), only
     when NBA.com failed; the first HTTP 429 from basketball-reference stops every further BR request in
     the run (a jail lasts up to a day)
Every event row carries source, endpoint, season, fetched_at; NBA.com rows also carry event_id,
event_type, action_type, player ids and team_id (docs/DATA_CONTRACT.md, play-by-play table v1.1).

  python -m collect.pbp --contract-dir "~/Library/CloudStorage/GoogleDrive-.../My Drive/nba_harvest/sportvu_contract"
  python -m collect.pbp --contract-dir ... --retry-failed          # redo the games whose last attempt failed
  python -m collect.pbp --contract-dir ... --games 12.16.2015.PHX.at.GSW --no-br
Both manifests are kept: the Step 1 manifest.json (pbp_joined, pbp.{code, events, matched, error}) and
collect/pbp/manifest.json (per game: source used, rows, matched, parameters, error).
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np

from collect import common
from collect.nba_stats import NBAStats, ClientError, first_set, game_id_season, SOURCE as NBA_SOURCE
from sportvu import colab_contract as cc

FAMILY = "pbp"
BR_SOURCE = "basketball-reference.com"
EVENT_KIND = {1: "shot_made", 2: "shot_missed", 4: "rebound", 5: "turnover", 6: "foul", 7: "violation", 8: "substitution",
              9: "timeout", 10: "jump_ball", 11: "ejection", 12: "period_start", 13: "period_end", 18: "instant_replay"}
TEXT_COLS = {"homedescription", "neutraldescription", "visitordescription", "score", "pctimestring"}
EXTRA_COLS = ("source", "endpoint", "season", "fetched_at", "event_id", "event_type", "action_type",
              "player1_id", "player2_id", "player3_id", "team_id", "clock_str")


class BRJailed(Exception):
    stop_run = False


def nba_events(payload: dict, home_team_id: int) -> list[dict]:
    """PlayByPlayV2 rows -> contract events (period, clock_s, team, desc, points, score, kind + NBA fields)."""
    df = first_set(payload, "PlayByPlay")
    txt = lambda v: v if isinstance(v, str) and v else None        # pandas reads a missing string as NaN
    events = []
    for r in df.itertuples(index=False):
        d = {k: (txt(v) if isinstance(v, (str, float)) and k in TEXT_COLS else v) for k, v in r._asdict().items()}
        desc = " | ".join(s for s in (d.get("homedescription"), d.get("neutraldescription"), d.get("visitordescription")) if s)
        et = int(d.get("eventmsgtype") or 0)
        tid = d.get("player1_team_id")
        tid = int(tid) if tid not in (None, "") and not (isinstance(tid, float) and np.isnan(tid)) else None
        if et == 3:                                             # free throw: the description says MISS or not
            kind = "ft_missed" if "MISS" in (desc or "").upper() else "ft_made"
        else:
            kind = EVENT_KIND.get(et, "other")
        points = 0
        if kind == "shot_made":
            points = 3 if "3PT" in desc.upper() else 2
        elif kind == "ft_made":
            points = 1
        if tid is not None:
            team = "home" if tid == home_team_id else "away"
        elif d.get("homedescription") and not d.get("visitordescription"):
            team = "home"
        elif d.get("visitordescription") and not d.get("homedescription"):
            team = "away"
        else:
            team = None
        score = d.get("score")
        score = re.sub(r"\s*-\s*", "-", score) if score else None   # '12 - 10' -> '12-10' (away-home, as BR)
        pid = lambda k: (int(d[k]) if d.get(k) not in (None, "", 0) and not (isinstance(d.get(k), float) and np.isnan(d.get(k))) else None)
        events.append({"period": int(d["period"]), "clock_s": common.clock_to_s(d.get("pctimestring")), "team": team,
                       "desc": desc, "points": points, "score": score, "kind": kind,
                       "event_id": int(d["eventnum"]), "event_type": et, "action_type": int(d.get("eventmsgactiontype") or 0),
                       "player1_id": pid("player1_id"), "player2_id": pid("player2_id"), "player3_id": pid("player3_id"),
                       "team_id": tid, "clock_str": d.get("pctimestring")})
    events = [e for e in events if e["clock_s"] is not None]
    events.sort(key=lambda e: (e["period"], e["event_id"]))
    return events


def frames_meta(frames_path: Path) -> tuple[str, int | None]:
    """(NBA game id, home team id) from the first row group of a Step 1 frames parquet."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(frames_path)
    t = pf.read_row_group(0, columns=["gameid", "team_id", "home"]).to_pandas()
    home = t.loc[t["home"], "team_id"]
    return str(t["gameid"].iloc[0]), (int(home.iloc[0]) if len(home) else None)


def with_extras(joined, events: list[dict], source: str, endpoint: str, season: str, fetched_at: str):
    """The contract pbp table (join_pbp output) plus the v1.1 provenance and NBA columns."""
    import pyarrow as pa
    n = joined.num_rows
    cols = {"source": pa.array([source] * n, pa.string()), "endpoint": pa.array([endpoint] * n, pa.string()),
            "season": pa.array([season] * n, pa.string()), "fetched_at": pa.array([fetched_at] * n, pa.string())}
    for k, typ in (("event_id", pa.int32()), ("event_type", pa.int16()), ("action_type", pa.int16()), ("player1_id", pa.int32()),
                   ("player2_id", pa.int32()), ("player3_id", pa.int32()), ("team_id", pa.int32()), ("clock_str", pa.string())):
        cols[k] = pa.array([e.get(k) for e in events], typ)
    for k, v in cols.items():
        joined = joined.append_column(k, v)
    return joined


def fetch_br(code: str, throttle: common.Throttle, state: dict) -> list[dict]:
    if state.get("br_jailed"):
        raise BRJailed("basketball-reference skipped: a 429 was seen earlier in this run")
    import fetch_pbp
    throttle.wait()
    try:
        return fetch_pbp.fetch_game(code)
    except Exception as e:                                      # noqa: BLE001 (classify, then re-raise)
        if "429" in str(e):
            state["br_jailed"] = True
            raise BRJailed("basketball-reference HTTP 429 on %s: no more BR requests this run (jail lasts up to a day)" % code)
        raise


def do_game(game: dict, step1: dict, contract_dir: Path, stats: NBAStats, br_throttle, state: dict, use_br: bool,
            sp: dict, scratch: Path) -> tuple[int, dict]:
    """One SportVU game -> <game>_pbp.parquet in the Step 1 layout. Returns (rows, manifest fields)."""
    import pyarrow.parquet as pq
    gid = game["game"]
    g = {"game": gid, "date": game["date"], "away": game["away"], "home": game["home"]}
    frames_path, pbp_path, split = cc.game_paths(g, contract_dir, sp)
    nba_id, home_team_id = frames_meta(frames_path)
    season = game_id_season(nba_id)
    table = pq.read_table(frames_path, columns=["frame", "period", "game_clock_s"])
    fp, fc = cc.frame_clocks(table)
    source, endpoint, params, events, errors = None, None, None, None, {}
    try:
        params = {"GameID": nba_id}
        rec = stats.get_record("playbyplayv2", params)
        events = nba_events(rec["body"], home_team_id)
        if not events:
            raise ValueError("PlayByPlayV2 returned no events")
        source, endpoint, fetched_at = NBA_SOURCE, "playbyplayv2", rec["fetched_at"]
    except ClientError as e:
        errors["nba"] = str(e)[:300]
        if e.stop_run:
            raise
    except Exception as e:                                      # noqa: BLE001 (fall back to BR)
        errors["nba"] = repr(e)[:300]
    if events is None and use_br:
        code = cc.br_code(g)
        try:
            events = fetch_br(code, br_throttle, state)
            source, endpoint, params, fetched_at = BR_SOURCE, "boxscores/pbp/%s.html" % code, {"code": code}, common.utc_now()
        except Exception as e:                                  # noqa: BLE001 (recorded)
            errors["br"] = repr(e)[:300]
    if events is None:
        entry = step1["games"].get(gid)
        if entry is not None:
            entry.update(pbp_joined=False, pbp={"code": nba_id, "events": 0, "matched": 0, "error": json.dumps(errors)[:300]})
        raise RuntimeError("no play-by-play: %s" % json.dumps(errors))
    joined = cc.join_pbp(gid, events, fp, fc)
    if source == NBA_SOURCE:
        joined = with_extras(joined, events, source, endpoint, season, fetched_at)
    else:
        joined = with_extras(joined, [{} for _ in events], source, endpoint, season, fetched_at)
    matched = int(np.sum(joined.column("frame").to_numpy() >= 0))
    cc.write_parquet(joined, pbp_path, scratch)
    entry = step1["games"].get(gid)
    if entry is not None:
        entry.update(pbp_joined=True, pbp_parquet=str(pbp_path.relative_to(contract_dir)),
                     pbp={"code": nba_id if source == NBA_SOURCE else cc.br_code(g), "events": len(events), "matched": matched,
                          "error": None, "source": source})
    return len(events), {"source": source, "endpoint": endpoint, "params": params, "matched": matched, "nba_game_id": nba_id,
                         "split": split, "heldout": cc.is_heldout(split), "pbp_parquet": str(pbp_path), "errors": errors or None}


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], ["2015-16"])
    ap.add_argument("--contract-dir", required=True, help="the Step 1 sportvu_contract directory (Drive copy on the Mac)")
    ap.add_argument("--games", nargs="*", help="only these SportVU game ids")
    ap.add_argument("--no-br", action="store_true", help="never fall back to basketball-reference")
    ap.add_argument("--br-rpm", type=float, default=17.0, help="basketball-reference requests per minute (max 17)")
    a = ap.parse_args(argv)
    contract_dir = Path(a.contract_dir).expanduser()
    step1_path = contract_dir / "manifest.json"
    if not step1_path.exists():
        sys.exit("no Step 1 manifest at %s" % step1_path)
    step1 = json.loads(step1_path.read_text())
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    from sportvu import splits
    sp = splits.load()
    games = [e for e in step1["games"].values() if e.get("status") == "ok"]
    if a.games:
        games = [e for e in games if e["game"] in set(a.games)]
    games = cc.order_games([{"game": e["game"], "date": e["date"], "away": e["away"], "home": e["home"]} for e in games], sp)
    by_id = {e["game"]: e for e in step1["games"].values()}
    units = [(common.Manifest.key("pbp", "2015-16", g["game"]), {"game": g["game"], **by_id[g["game"]], "params": {"game": g["game"]}}) for g in games]
    stats = NBAStats(fam.raw, common.Throttle(a.rpm))
    br_throttle = common.Throttle(min(a.br_rpm, 17.0))
    state: dict = {}
    scratch = fam.dir / "_scratch"

    def do_unit(unit):
        rows, fields = do_game(unit, step1, contract_dir, stats, br_throttle, state, not a.no_br, sp, scratch)
        cc.write_manifest(contract_dir, step1)
        return rows, fields

    try:
        final = common.run_units(fam, units, do_unit, a, "play-by-play for SportVU games")
    finally:
        cc.write_manifest(contract_dir, step1)
    return final


if __name__ == "__main__":
    main()
