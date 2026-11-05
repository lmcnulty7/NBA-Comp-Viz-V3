"""collect/bref_schedule.py: the season schedule from basketball-reference.com month pages, and the page helpers the
other basketball-reference drivers share (collect.bref_seasons, bref_boxscores, bref_lineups, positions).

stats.nba.com drops non-browser clients, so basketball-reference (at most 17 requests a minute, shared with every
other driver) is the reachable source. One month page per unit:

  https://www.basketball-reference.com/leagues/NBA_<endyear>_games-<month>.html

  <table id="schedule"> ... <tr><th data-stat="date_game" csk="201510270ATL"><a ...>Tue, Oct 27, 2015</a></th>
    <td data-stat="visitor_team_name" csk="DET.201510270ATL"><a href="/teams/DET/2016.html">Detroit Pistons</a></td>
    <td data-stat="visitor_pts">106</td> <td data-stat="home_team_name" csk="ATL.201510270ATL">...</td>
    <td data-stat="home_pts">94</td> <td data-stat="box_score_text"><a href="/boxscores/201510270ATL.html">Box Score</a></td>
    ... <td data-stat="game_remarks"></td></tr>
  A row <tr class="thead"><th>Playoffs</th></tr> separates the regular season from the playoffs on the month page
  where they meet; a Notes cell saying "Play-In Game" marks the play-in games (2020-21 on).

Rows: game_id (the basketball-reference code YYYYMMDD0HOME), date, home, away (basketball-reference codes: BRK, CHO,
PHO for BKN, CHA, PHX), home_score, away_score (null until played), season, season_type (Regular Season, Play-In,
Playoffs), split (sportvu/splits.json for 2015-16 games, else null), provenance. The schedule is a listing, not
fitting data: no heldout column, so nothing is compacted apart; games_for_season tags held-out games on the way out.

Output: --out/bref_schedule/games/<season>.parquet (parts per month under parts/games/<season>/<month>.parquet); the
raw pages under --out/raw/bref_schedule/. One request per month page (nine a season; ten for 2019-20, eight for
2020-21), cached and never re-requested. The first HTTP 429 or 403 stops the run.

  python -m collect.bref_schedule --seasons 2016-17 2017-18
  python -m collect.bref_schedule --seasons 2019-20 --dry-run
games_for_season(fam, season) feeds the per-game drivers from the compacted table (or the parts when it is absent).
"""
from __future__ import annotations
import re
from pathlib import Path

from collect import common
from collect import bref_shots as bs
from sportvu import colab_contract as cc

FAMILY = "bref_schedule"
SEASON_TYPES = common.SEASON_TYPES + ("Play-In",)
RPM_HELP = "basketball-reference requests per minute (default and maximum 17)"
ENDPOINT = "games"
RAW_ENDPOINT = "bref_schedule"
SOURCE = "basketball-reference.com"
HOST = bs.HOST
MAX_RPM = bs.MAX_RPM
PageError = bs.PageError
BREF_TO_SPORTVU = bs.BREF_TO_SPORTVU
SPORTVU_TO_BREF = cc.BR_ABBR
DEFAULT_SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
# the 30 teams as basketball-reference codes them from 2014-15 on (fallback when no schedule is collected yet)
TEAMS = ("ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS")
MONTHS_DEFAULT = ("october", "november", "december", "january", "february", "march", "april", "may", "june")
MONTHS_BY_SEASON = {
    "2011-12": ("december", "january", "february", "march", "april", "may", "june"),                       # the lockout season
    "2019-20": ("october-2019", "november", "december", "january", "february", "march", "july", "august", "september", "october-2020"),  # the bubble
    "2020-21": ("december", "january", "february", "march", "april", "may", "june", "july"),
}
COLUMNS = ("game_id", "date", "home", "away", "home_score", "away_score", "season", "season_type", "month", "start_time",
           "overtimes", "attendance", "arena", "notes", "split", "source", "endpoint", "fetched_at")
MONTH_NUM = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}
_SPLITS = None


# ── shared page helpers ────────────────────────────────────────────────────────────────────────────
def end_year(season: str) -> int:
    return common.season_start_year(season) + 1


def months_of(season: str) -> tuple:
    return MONTHS_BY_SEASON.get(season, MONTHS_DEFAULT)


def schedule_url(season: str, month: str) -> str:
    return "%s/leagues/NBA_%d_games-%s.html" % (HOST, end_year(season), month)


def page_record(fam: common.Family, raw_endpoint: str, params: dict, url: str, throttle: common.Throttle) -> dict:
    """The raw-cache record of a page ({"body": html, "fetched_at", ...}), fetched through collect.bref_shots.fetch_html
    when absent (so one monkeypatch of bref_shots.fetch_html covers every driver). 429 and 403 stop the run."""
    rec = fam.raw.get(raw_endpoint, params)
    if rec is None:
        html = bs.fetch_html(url, throttle)
        rec = fam.raw.put(raw_endpoint, params, html, url, 200)
    return rec


def soup_of(html: str):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def find_tables(html: str, id_re: str) -> list:
    """Every <table> whose id matches id_re, live or inside an HTML comment (basketball-reference comments out most
    tables after the first), in page order."""
    from bs4 import BeautifulSoup, Comment
    soup = soup_of(html)
    rx = re.compile(id_re)
    out = [t for t in soup.find_all("table") if rx.match(t.get("id") or "")]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            out.extend(t for t in BeautifulSoup(c, "html.parser").find_all("table") if rx.match(t.get("id") or ""))
    return out


def row_cells(tr) -> dict:
    """{data-stat: cell} of one table row (th and td alike); cells without a data-stat are skipped."""
    return {c.get("data-stat"): c for c in tr.find_all(["th", "td"]) if c.get("data-stat")}


def cell_text(cells: dict, stat: str) -> str | None:
    c = cells.get(stat)
    if c is None:
        return None
    t = c.get_text(" ", strip=True)
    return t if t != "" else None


def cell_num(cells: dict, stat: str) -> float | None:
    t = cell_text(cells, stat)
    if t is None:
        return None
    t = t.replace(",", "").replace("+", "")
    try:
        return float(t)
    except ValueError:
        return None


def slug_of(cell) -> str | None:
    """The player slug of a cell: data-append-csv, else the /players/x/<slug>.html link."""
    if cell is None:
        return None
    if cell.get("data-append-csv"):
        return cell["data-append-csv"]
    a = cell.find("a", href=True)
    m = re.search(r"/players/\w/([^/.]+)\.html", a["href"]) if a else None
    return m.group(1) if m else None


def team_of_link(cell) -> str | None:
    """'ATL' from a /teams/ATL/2016.html link in the cell."""
    if cell is None:
        return None
    a = cell.find("a", href=True)
    m = re.search(r"/teams/([A-Z]{3})/", a["href"]) if a else None
    return m.group(1) if m else None


def minutes_of(text: str | None) -> float | None:
    """'35:23' -> 35.38, '1,065:30' -> 1065.5, '240' -> 240.0; None for blank or non-numeric (Did Not Play)."""
    if not text:
        return None
    t = text.replace(",", "").strip()
    m = re.match(r"^(\d+):(\d{1,2})(?:\.\d+)?$", t)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 60.0, 2)
    try:
        return float(t)
    except ValueError:
        return None


def parse_date(text: str) -> str | None:
    """'Tue, Oct 27, 2015' -> '2015-10-27'."""
    m = re.search(r"([A-Z][a-z]{2})\w*\s+(\d{1,2}),\s*(\d{4})", text or "")
    if not m or m.group(1).lower() not in MONTH_NUM:
        return None
    return "%s-%02d-%02d" % (m.group(3), MONTH_NUM[m.group(1).lower()], int(m.group(2)))


def code_of_game(date: str, home_br: str) -> str:
    return date.replace("-", "") + "0" + home_br


def sportvu_abbr(team_br: str) -> str:
    return BREF_TO_SPORTVU.get(team_br, team_br)


def sportvu_game_id(date: str, away_br: str, home_br: str) -> str:
    """'MM.DD.YYYY.AWAY.at.HOME' in SportVU abbreviations (the `game` column of the contract tables)."""
    y, m, d = date.split("-")
    return "%s.%s.%s.%s.at.%s" % (m, d, y, sportvu_abbr(away_br), sportvu_abbr(home_br))


def split_for(date: str, away_br: str, home_br: str) -> str | None:
    """sportvu/splits.json split of a game (train, heldout_game, heldout_arena, heldout_era) by date and teams; only
    2015-16 games can have one, every other season gets None."""
    global _SPLITS
    if not date or bs.season_of(date) != "2015-16":
        return None
    if _SPLITS is None:
        from sportvu import splits
        _SPLITS = splits.load()
    return cc.split_of({"game": sportvu_game_id(date, away_br, home_br), "date": date, "away": sportvu_abbr(away_br), "home": sportvu_abbr(home_br)}, _SPLITS)


def is_heldout(split: str | None) -> bool:
    return cc.is_heldout(split)


def bref_parser(description: str, default_seasons: list[str]):
    """common.base_parser for a basketball-reference driver: --rpm defaults to 17 with its own help, --season-types
    also accepts Play-In."""
    ap = common.base_parser(description, default_seasons, rpm_help=RPM_HELP)
    ap.set_defaults(rpm=MAX_RPM)
    for action in ap._actions:
        if action.dest == "season_types":
            action.choices = SEASON_TYPES
    return ap


def capped_throttle(rpm: float) -> common.Throttle:
    """The shared throttle: never more than MAX_RPM (17) a minute, whatever --rpm says (0 or negative: the maximum)."""
    if rpm > MAX_RPM:
        print("--rpm %.0f capped at %.0f (basketball-reference's limit)" % (rpm, MAX_RPM), flush=True)
    return common.Throttle(MAX_RPM if rpm <= 0 else min(rpm, MAX_RPM))


def estimate_requests(fam: common.Family, units: list, args, raw_of) -> int:
    """How many pages a run would fetch: pending units (after --limit) whose page is not in the raw cache. raw_of(unit)
    -> (raw_endpoint, params), or a list of them when any one cached record spares the request. Printed before every run."""
    keys = [k for k, _ in units]
    pending = fam.manifest.pending(keys, args.retry_failed)
    if args.limit:
        pending = pending[:args.limit]
    by_key = dict(units)

    def cached(unit) -> bool:
        cands = raw_of(unit)
        cands = cands if isinstance(cands, list) else [cands]
        return any(fam.raw.path(ep, params).exists() for ep, params in cands)
    n = sum(1 for k in pending if not cached(by_key[k]))
    print("estimated requests: %d (%d pending units, %d pages already cached); at %.0f a minute about %.0f min"
          % (n, len(pending), len(pending) - n, MAX_RPM, n / MAX_RPM), flush=True)
    return n


# ── parsing ────────────────────────────────────────────────────────────────────────────────────────
def require_table(html: str, id_re: str, what: str, where: str = "") -> list:
    """find_tables, or ValueError("<where>: no table <what> on the page") so the unit fails (and --retry-failed redoes
    it) instead of being marked empty for good."""
    tables = find_tables(html, id_re)
    if not tables:
        raise ValueError("%sno table %s on the page" % (where + ": " if where else "", what))
    return tables


def parse_schedule(html: str, where: str = "") -> list[dict]:
    """Every game row of a month page -> dict(game_id, date, home, away, home_score, away_score, season_type, start_time,
    overtimes, attendance, arena, notes). Rows after the 'Playoffs' divider are Playoffs; a 'Play-In' note marks Play-In.
    ValueError when the page has no schedule table."""
    tables = require_table(html, r"^schedule$", "id=schedule", where)
    games, season_type = [], "Regular Season"
    for tr in tables[0].find_all("tr"):
        if "thead" in (tr.get("class") or []):
            if "playoff" in tr.get_text(" ", strip=True).lower():
                season_type = "Playoffs"
            continue
        cells = row_cells(tr)
        if "date_game" not in cells or "home_team_name" not in cells:
            continue
        date = parse_date(cell_text(cells, "date_game") or "")
        home, away = team_of_link(cells.get("home_team_name")), team_of_link(cells.get("visitor_team_name"))
        if not date or not home or not away:
            continue
        code = None
        box = cells.get("box_score_text")
        a = box.find("a", href=True) if box is not None else None
        m = re.search(r"/boxscores/(\d{8}0[A-Z]{3})\.html", a["href"]) if a else None
        if m:
            code = m.group(1)
        elif cells["date_game"].get("csk"):
            code = cells["date_game"]["csk"]
        else:
            code = code_of_game(date, home)
        notes = cell_text(cells, "game_remarks")
        st = "Play-In" if notes and "play-in" in notes.lower() else season_type
        hs, vs = cell_num(cells, "home_pts"), cell_num(cells, "visitor_pts")
        games.append({"game_id": code, "date": date, "home": home, "away": away,
                      "home_score": int(hs) if hs is not None else None, "away_score": int(vs) if vs is not None else None,
                      "season_type": st, "start_time": cell_text(cells, "game_start_time"), "overtimes": cell_text(cells, "overtimes"),
                      "attendance": cell_num(cells, "attendance"), "arena": cell_text(cells, "arena_name"), "notes": notes})
    return games


def rows_for_month(games: list[dict], season: str, month: str, fetched_at: str):
    import pandas as pd
    recs = [{**g, "season": season, "month": month, "split": split_for(g["date"], g["away"], g["home"]),
             "source": SOURCE, "endpoint": "leagues/NBA_%d_games-%s.html" % (end_year(season), month), "fetched_at": fetched_at} for g in games]
    df = pd.DataFrame(recs, columns=list(COLUMNS))
    for c in ("home_score", "away_score"):
        df[c] = df[c].astype("Int32")
    df["attendance"] = df["attendance"].astype("float32")
    for c in ("game_id", "date", "home", "away", "season", "season_type", "month", "start_time", "overtimes", "arena", "notes", "split", "source", "endpoint", "fetched_at"):
        df[c] = df[c].astype("string")
    return df


# ── the games of a season for the per-game drivers ─────────────────────────────────────────────────
def games_for_season(fam: common.Family, season: str, season_types=("Regular Season",), played_only: bool = True) -> list[dict]:
    """The games of a season from --out/bref_schedule/games/<season>.parquet (or its parts), as dicts: game_id (the
    basketball-reference code), date, home, away (basketball-reference codes), home_score, away_score, season,
    season_type, split, heldout, game (the SportVU-style id). fam is the bref_schedule Family (or any Family under the
    same --out). Empty when the schedule has not been collected."""
    import pandas as pd
    sched = Path(fam.dir).parent / FAMILY
    path = sched / ENDPOINT / ("%s.parquet" % season)
    if path.exists():
        df = common.read_parquet(path)
    else:
        parts = sorted((sched / "parts" / ENDPOINT / season).glob("*.parquet")) if (sched / "parts" / ENDPOINT / season).exists() else []
        if not parts:
            return []
        df = pd.concat([common.read_parquet(p) for p in parts], ignore_index=True)
    if season_types:
        df = df[df["season_type"].isin(list(season_types))]
    if played_only:
        df = df[df["home_score"].notna() & df["away_score"].notna()]
    df = df.drop_duplicates("game_id").sort_values(["date", "game_id"])
    out = []
    for r in df.itertuples(index=False):
        split = split_for(r.date, r.away, r.home)
        out.append({"game_id": r.game_id, "date": r.date, "home": r.home, "away": r.away,
                    "home_score": int(r.home_score) if pd.notna(r.home_score) else None, "away_score": int(r.away_score) if pd.notna(r.away_score) else None,
                    "season": season, "season_type": r.season_type, "split": split, "heldout": is_heldout(split),
                    "game": sportvu_game_id(r.date, r.away, r.home)})
    return out


def teams_for_season(fam: common.Family, season: str) -> list[str]:
    """The team codes of a season from its schedule (home teams of the regular season), else TEAMS."""
    games = games_for_season(fam, season, ("Regular Season",), played_only=False)
    teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
    return teams or list(TEAMS)


# ── one unit ───────────────────────────────────────────────────────────────────────────────────────
def do_month(unit: dict, fam: common.Family, throttle: common.Throttle) -> tuple[int, dict]:
    season, month = unit["season"], unit["month"]
    rec = page_record(fam, RAW_ENDPOINT, unit["params"], schedule_url(season, month), throttle)
    games = parse_schedule(rec["body"], "%s %s" % (season, month))
    df = rows_for_month(games, season, month, rec["fetched_at"])
    if len(df):
        fam.write_part(ENDPOINT, season, month, df)
    by_type = {t: int(n) for t, n in df.groupby("season_type").size().items()} if len(df) else {}
    return len(df), {"fetched_at": rec["fetched_at"], "endpoint": "leagues/NBA_%d_games-%s.html" % (end_year(season), month),
                     "played": int(df["home_score"].notna().sum()) if len(df) else 0, "by_type": by_type}


def units_for(seasons: list[str]) -> list[tuple[str, dict]]:
    return [(common.Manifest.key(ENDPOINT, s, m), {"season": s, "month": m, "params": {"season": s, "month": m}}) for s in seasons for m in months_of(s)]


def main(argv=None) -> str:
    ap = bref_parser(__doc__.split("\n\n")[0], DEFAULT_SEASONS)
    a = ap.parse_args(argv)
    fam = common.Family(a.out, FAMILY)
    if a.compact_only:
        print(fam.compact()); return "compact only"
    units = units_for(a.seasons)
    throttle = capped_throttle(a.rpm)
    estimate_requests(fam, units, a, lambda u: (RAW_ENDPOINT, u["params"]))
    return common.run_units(fam, units, lambda u: do_month(u, fam, throttle), a, "basketball-reference schedule, %d month pages" % len(units))


if __name__ == "__main__":
    main()
