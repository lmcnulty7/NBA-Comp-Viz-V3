"""collect/extras.py: odds history and injury reports (collection plan step 8, only if pursued; both behind flags).

--odds      The Odds API historical snapshots (https://the-odds-api.com, key in $ODDS_API_KEY, paid tier for
            history; coverage from mid-2020, featured markets h2h, spreads, totals). For every game date of the
            seasons (leaguegamelog, cached) one snapshot at --snapshot-utc (23:30 by default, before the evening
            tip-offs), then one snapshot per distinct tip time at commence minus --close-minutes for the closing
            line. Rows: the contract odds table (game_key, commence_time, book, market, side, point, price,
            snapshot_at, is_close). Every snapshot costs credits: --limit first.
--injuries  Injury report archive CSVs (https://statsurge.substack.com/p/downloadable-nba-injury-datasets,
            downloaded by hand: the files need a browser) -> the contract injuries table. Columns are mapped by
            name (player, team, date, status, reason) and every other column is kept; the mapping is checked
            against the first file you give it and reported.

  ODDS_API_KEY=... python -m collect.extras --odds --seasons 2020-21 --limit 5
  python -m collect.extras --injuries ~/Downloads/injury_reports_2021-22.csv ~/Downloads/injury_reports_2022-23.csv
"""
from __future__ import annotations
import datetime as dt, json, os, sys, time
from pathlib import Path
from collect import common
from collect.nba_stats import NBAStats, ClientError, season_games

FAMILY = "extras"
ODDS_BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
ODDS_SOURCE = "the-odds-api.com"
INJURY_SOURCE = "statsurge.substack.com (NBA official injury reports)"
INJURY_ALIASES = {"player": ("player", "player_name", "name"), "team": ("team", "team_name", "tm"),
                  "date": ("date", "report_date", "game_date", "report date", "game date"),
                  "status": ("status", "current_status", "current status", "game_status"),
                  "reason": ("reason", "injury", "injury_reason", "injury / illness", "description")}


class OddsClient:
    """Historical snapshots through the raw cache (one request per snapshot time, ever)."""
    def __init__(self, cache: common.RawCache, throttle: common.Throttle, key: str, regions: str = "us", markets: str = "h2h,spreads,totals"):
        self.cache, self.throttle, self.key, self.regions, self.markets = cache, throttle, key, regions, markets
        self.requests_made, self.credits_left = 0, None

    def snapshot(self, at_iso: str) -> dict:
        params = {"regions": self.regions, "markets": self.markets, "oddsFormat": "american", "date": at_iso}
        rec = self.cache.get("oddsapi_historical", params)
        if rec is not None:
            return rec
        import requests
        self.throttle.wait()
        self.requests_made += 1
        r = requests.get(ODDS_BASE, params={**params, "apiKey": self.key}, timeout=30)
        self.credits_left = r.headers.get("x-requests-remaining")
        if r.status_code != 200:
            raise ClientError("odds api HTTP %d: %s" % (r.status_code, r.text[:160]), r.status_code, stop_run=r.status_code in (401, 402, 429))
        return self.cache.put("oddsapi_historical", params, r.json(), r.url.split("apiKey")[0], 200)


def odds_rows(rec: dict, is_close: bool) -> list[dict]:
    body = rec["body"]
    snap = body.get("timestamp")
    rows = []
    for ev in body.get("data", []):
        key = "%s_%s_%s" % (ev.get("commence_time", "")[:10], ev.get("away_team", "").replace(" ", ""), ev.get("home_team", "").replace(" ", ""))
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({"game_key": key, "commence_time": ev.get("commence_time"), "home_team": ev.get("home_team"),
                                 "away_team": ev.get("away_team"), "book": bk.get("key"), "market": m.get("key"),
                                 "side": o.get("name"), "point": o.get("point"), "price": o.get("price"),
                                 "snapshot_at": snap, "is_close": is_close, "source": ODDS_SOURCE, "endpoint": "historical/odds",
                                 "fetched_at": rec["fetched_at"]})
    return rows


def do_odds_date(client: OddsClient, fam: common.Family, u: dict, snapshot_utc: str, close_minutes: int) -> tuple[int, dict]:
    import pandas as pd
    date = u["id"]
    first = client.snapshot("%sT%sZ" % (date, snapshot_utc))
    rows = odds_rows(first, False)
    tips = sorted({ev["commence_time"] for ev in first["body"].get("data", []) if ev.get("commence_time", "")[:10] == date})
    for tip in tips:
        t = dt.datetime.strptime(tip, "%Y-%m-%dT%H:%M:%SZ") - dt.timedelta(minutes=close_minutes)
        rows += odds_rows(client.snapshot(t.strftime("%Y-%m-%dT%H:%M:%SZ")), True)
    df = pd.DataFrame(rows)
    if len(df):
        df["season"] = u["season"]
        fam.write_part("odds", u["season"], date, df)
    return len(df), {"snapshots": 1 + len(tips), "credits_left": client.credits_left}


def injuries_frame(path: Path):
    """One archive CSV -> the contract injuries columns + every other column; returns (df, mapping)."""
    import pandas as pd
    raw = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in raw.columns}
    mapping = {}
    for want, names in INJURY_ALIASES.items():
        for n in names:
            if n in cols:
                mapping[want] = cols[n]
                break
    df = raw.rename(columns={v: k for k, v in mapping.items()})
    for want in INJURY_ALIASES:
        if want not in df.columns:
            df[want] = None
    if df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["source"], df["endpoint"], df["fetched_at"] = INJURY_SOURCE, path.name, common.utc_now()
    df["season"] = df["date"].map(lambda d: common.season_str(int(d[:4]) - (1 if int(d[5:7]) < 8 else 0)) if isinstance(d, str) else None)
    return df, mapping


def main(argv=None) -> str:
    ap = common.base_parser(__doc__.split("\n\n")[0], ["2020-21"])
    ap.add_argument("--odds", action="store_true", help="collect The Odds API historical snapshots ($ODDS_API_KEY)")
    ap.add_argument("--snapshot-utc", default="23:30:00", help="pre-game snapshot time per game date (UTC)")
    ap.add_argument("--close-minutes", type=int, default=5, help="closing snapshot: this many minutes before tip")
    ap.add_argument("--injuries", nargs="*", help="injury report archive CSV files to load")
    a = ap.parse_args(argv)
    if not a.odds and not a.injuries:
        sys.exit("nothing to do: pass --odds and/or --injuries CSV...")
    fam = common.Family(a.out, FAMILY)
    final = "RUN OK"
    if a.odds:
        key = os.environ.get("ODDS_API_KEY")
        if not key:
            sys.exit("set ODDS_API_KEY (The Odds API key; the historical endpoint needs a paid plan)")
        stats = NBAStats(fam.raw, common.Throttle(a.rpm))
        client = OddsClient(fam.raw, common.Throttle(a.rpm), key)
        units = []
        for s in a.seasons:
            for st in a.season_types:
                dates = sorted({g["date"] for g in season_games(stats, fam, s, st)})
                units += [(common.Manifest.key("odds", s, d), {"endpoint": "odds", "season": s, "id": d, "params": {"date": d, "snapshot_utc": a.snapshot_utc}}) for d in dates]
        final = common.run_units(fam, units, lambda u: do_odds_date(client, fam, u, a.snapshot_utc, a.close_minutes), a, "odds: one game date per unit")
    if a.injuries:
        import pandas as pd
        for f in a.injuries:
            df, mapping = injuries_frame(Path(f).expanduser())
            print("%s: %d rows, columns mapped %s" % (f, len(df), mapping), flush=True)
            for season, part in df.groupby("season", dropna=False):
                fam.write_part("injuries", str(season), Path(f).stem, part)
            fam.manifest.mark(common.Manifest.key("injuries", "file", Path(f).name), "ok", rows=len(df), params={"file": str(f), "mapping": mapping})
        fam.manifest.save()
        print(fam.compact(["injuries"]))
    return final


if __name__ == "__main__":
    main()
