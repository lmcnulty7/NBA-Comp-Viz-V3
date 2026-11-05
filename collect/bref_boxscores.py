"""collect/bref_boxscores.py: box scores per game from basketball-reference.com (https://www.basketball-reference.com/boxscores/<code>.html).

One unit per game of the schedule (collect.bref_schedule.games_for_season), one request each. The page carries one
basic box score table per team, away first then home:

  <table id="box-PHO-game-basic">  (older pages: id="box_pho_basic")
    <thead> ... <tr><th data-stat="player">Starters</th><th data-stat="mp">MP</th><th data-stat="fg">FG</th> ... </tr></thead>
    <tbody>
      <tr><th data-stat="player" data-append-csv="blednar01"><a href="/players/b/blednar01.html">Eric Bledsoe</a></th>
          <td data-stat="mp">35:23</td><td data-stat="fg">7</td> ... <td data-stat="plus_minus">-9</td></tr>
      ... five starters, then <tr class="thead"><th data-stat="player">Reserves</th> ... </tr>, the reserves, and
      <tr><th data-stat="player" ...>Name</th><td data-stat="reason">Did Not Play</td></tr> for those who sat out
    </tbody> <tfoot> Team Totals </tfoot>
The starters are the first five rows of each team's table (the rows before the Reserves divider).

Outputs under --out/bref_boxscores/:
  starters/<season>.parquet   game_id, team, player_slug, player, starter, minutes (+ home, season_type, game_date,
                              split, heldout, provenance): every player who played, starter for the first five
  players/<season>.parquet    the same rows with every basic box score column (fg, fga, ... plus_minus) and the
                              Did Not Play rows (minutes null, reason)
After compaction the starters table is also written to --out/stints/starters/<season>.parquet (held-out rows to
--out/stints/heldout/starters/<season>.parquet), which the lineups lane reads to seed each game's opening lineup.
At most 17 requests a minute; the first HTTP 429 or 403 stops the run.

  python -m collect.bref_schedule --seasons 2023-24
  python -m collect.bref_boxscores --seasons 2023-24 [--season-types "Regular Season" Playoffs] [--limit 100]
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

from collect import common
from collect import bref_schedule as sched

FAMILY = "bref_boxscores"
STARTERS = "starters"
PLAYERS = "players"
RAW_ENDPOINT = "bref_boxscore"
SOURCE = "basketball-reference.com"
STINTS_FAMILY = "stints"
MAX_RPM = sched.MAX_RPM
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TABLE_ID_RE = re.compile(r"^box[-_]([A-Za-z]{3})[-_](?:game[-_])?basic$")
STAT_COLS = ("fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct", "ft", "fta", "ft_pct", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts", "plus_minus")
STARTER_COLUMNS = ("game_id", "team", "player_slug", "player", "starter", "minutes", "home", "season_type", "game_date", "split", "heldout",
                   "source", "endpoint", "season", "fetched_at")
PLAYER_COLUMNS = STARTER_COLUMNS[:6] + ("mp_str", "reason") + STAT_COLS + STARTER_COLUMNS[6:]


def boxscore_url(code: str) -> str:
    return "%s/boxscores/%s.html" % (sched.HOST, code)


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def parse_boxscore(html: str) -> list[dict]:
    """Every player row of the basic box score tables -> dict(team, player_slug, player, starter, minutes, mp_str, reason,
    fg ... plus_minus, order). Teams come in page order (away, then home); starter is True for the rows before the
    Reserves divider (the first five)."""
    rows = []
    for t in sched.find_tables(html, TABLE_ID_RE.pattern):
        team = TABLE_ID_RE.match(t["id"]).group(1).upper()
        body = t.find("tbody") or t
        starter, order = True, 0
        for tr in body.find_all("tr"):
            if "thead" in (tr.get("class") or []):
                if "reserve" in tr.get_text(" ", strip=True).lower():
                    starter = False
                continue
            cells = sched.row_cells(tr)
            pc = cells.get("player")
            slug = sched.slug_of(pc)
            if pc is None or slug is None:
                continue
            reason = sched.cell_text(cells, "reason")
            mp = sched.cell_text(cells, "mp")
            rec = {"team": team, "player_slug": slug, "player": pc.get_text(" ", strip=True), "starter": bool(starter and reason is None),
                   "minutes": sched.minutes_of(mp) if reason is None else None, "mp_str": mp if reason is None else None, "reason": reason, "order": order}
            for s in STAT_COLS:
                rec[s] = sched.cell_num(cells, s) if reason is None else None
            rows.append(rec)
            order += 1
    return rows


def rows_for_game(g: dict, players: list[dict], fetched_at: str):
    """Parsed rows of one game -> (starters DataFrame, players DataFrame)."""
    import pandas as pd
    code = g["game_id"]
    base = {"game_id": code, "season_type": g["season_type"], "game_date": g["date"], "split": g.get("split"), "heldout": bool(g.get("heldout")),
            "source": SOURCE, "endpoint": "boxscores/%s.html" % code, "season": g["season"], "fetched_at": fetched_at}
    recs = [{**p, **base, "home": p["team"] == g["home"]} for p in players]
    allp = pd.DataFrame(recs, columns=list(PLAYER_COLUMNS))
    for c in STAT_COLS + ("minutes",):
        allp[c] = allp[c].astype("float32")
    for c in ("starter", "home", "heldout"):
        allp[c] = allp[c].astype("bool")
    for c in ("game_id", "team", "player_slug", "player", "mp_str", "reason", "season_type", "game_date", "split", "source", "endpoint", "season", "fetched_at"):
        allp[c] = allp[c].astype("string")
    starters = allp[allp["minutes"].notna()][list(STARTER_COLUMNS)].reset_index(drop=True)
    return starters, allp


# ── one unit ───────────────────────────────────────────────────────────────────────────────────────
def do_game(unit: dict, fam: common.Family, throttle: common.Throttle) -> tuple[int, dict]:
    g, code = unit["game"], unit["game"]["game_id"]
    rec = sched.page_record(fam, RAW_ENDPOINT, unit["params"], boxscore_url(code), throttle)
    players = parse_boxscore(rec["body"])
    starters, allp = rows_for_game(g, players, rec["fetched_at"])
    teams = sorted(set(allp["team"])) if len(allp) else []
    expected = sorted({g["home"], g["away"]})
    if teams and teams != expected:
        raise ValueError("%s: box score tables for %s, schedule says %s" % (code, teams, expected))
    n_start = {t: int(n) for t, n in starters[starters["starter"]].groupby("team").size().items()} if len(starters) else {}
    if any(n != 5 for n in n_start.values()) or (teams and len(n_start) != 2):
        raise ValueError("%s: starters per team %s, want five each" % (code, n_start))
    if len(starters):
        fam.write_part(STARTERS, g["season"], code, starters)
    if len(allp):
        fam.write_part(PLAYERS, g["season"], code, allp)
    return len(starters), {"fetched_at": rec["fetched_at"], "endpoint": "boxscores/%s.html" % code, "players": int(len(allp)),
                           "dnp": int(allp["reason"].notna().sum()) if len(allp) else 0, "split": g.get("split"), "heldout": bool(g.get("heldout"))}


def units_for(fam: common.Family, seasons: list[str], season_types) -> list[tuple[str, dict]]:
    out = []
    for s in seasons:
        games = sched.games_for_season(fam, s, season_types)
        if not games:
            print("!! no schedule for %s under %s: run python -m collect.bref_schedule --seasons %s first" % (s, fam.dir.parent, s), flush=True)
        for g in games:
            out.append((common.Manifest.key(STARTERS, s, g["game_id"]), {"season": s, "game": g, "params": {"code": g["game_id"]}}))
    return out


def copy_starters_to_stints(fam: common.Family) -> list[Path]:
    """--out/bref_boxscores/starters/<season>.parquet -> --out/stints/starters/<season>.parquet (and heldout/starters/),
    the table the lineups lane reads. Returns the paths written."""
    out = []
    for sub in ("", "heldout"):
        src_dir = fam.dir / sub / STARTERS if sub else fam.dir / STARTERS
        if not src_dir.exists():
            continue
        for p in sorted(src_dir.glob("*.parquet")):
            dst = fam.dir.parent / STINTS_FAMILY / sub / STARTERS / p.name if sub else fam.dir.parent / STINTS_FAMILY / STARTERS / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(p, tmp)
            tmp.replace(dst)
            out.append(dst)
    return out


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    ap.set_defaults(rpm=MAX_RPM)
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); print("stints copies:", [str(p) for p in copy_starters_to_stints(fam)]); return "compact only"
    units = units_for(fam, a.seasons, a.season_types)
    throttle = sched.capped_throttle(a.rpm)
    sched.estimate_requests(fam, units, a, lambda u: (RAW_ENDPOINT, u["params"]))
    final = common.run_units(fam, units, lambda u: do_game(u, fam, throttle), a, "basketball-reference box scores, %d games" % len(units))
    if not a.dry_run:
        print("stints copies:", [str(p) for p in copy_starters_to_stints(fam)], flush=True)
    return final


if __name__ == "__main__":
    main()
