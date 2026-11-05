"""collect/common.py: what every collector in collect/ shares (NBA CV V3 extension data, collection plan).

One CLI pattern (collect.common.base_parser):
  python -m collect.<family> --out DIR --seasons 2015-16 2016-17 [--season-types "Regular Season" Playoffs]
                             [--rpm 30] [--limit N] [--retry-failed] [--compact-only] [--dry-run]
One output layout: --out/raw/<endpoint>/<sha1 of params>.json.gz holds every response as received, shared by
all families (never re-requested while present), and per family under --out/<family>/:
  parts/<endpoint>/<season>/<unit>.parquet   one table per unit (endpoint, season, id)
  <endpoint>/<season>.parquet                 the compacted table for fitting (compact() at the end of a run)
  heldout/<endpoint>/<season>.parquet         rows of the held-out games of sportvu/splits.json, kept apart
  manifest.json                               every unit: status, rows, the exact parameters, fetched_at, error
Every row carries source, endpoint, season, fetched_at (docs/DATA_CONTRACT.md, provenance columns).

Resumable by unit: a unit with status ok is never redone; a failed unit is redone only with --retry-failed, so a
rate-limit jail is not hammered on a rerun. Nothing here talks to the network: collect.nba_stats does.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, hashlib, json, os, re, shutil, sys, tempfile, time
from pathlib import Path

import config

DEFAULT_OUT = config.PROJECT_ROOT / "data" / "collect"
DRIVE_OUT = "/content/drive/MyDrive/nba_harvest/collect"
SEASON_TYPES = ("Regular Season", "Playoffs")
FIRST_TRACKING_SEASON = 2013          # league-wide SportVU cameras: 2013-14
TEAM_IDS = list(range(1610612737, 1610612767))   # the 30 NBA teams on stats.nba.com
PERIOD_S = {1: 720.0, 2: 720.0, 3: 720.0, 4: 720.0}
OT_S = 300.0
T0 = time.time()


# ── seasons and time ───────────────────────────────────────────────────────────────────────────────
def season_str(start_year: int) -> str:
    return "%d-%02d" % (start_year, (start_year + 1) % 100)


def season_start_year(season: str) -> int:
    m = re.match(r"^(\d{4})-\d{2}$", season)
    if not m:
        raise ValueError("season must look like 2015-16, got %r" % season)
    return int(m.group(1))


def last_completed_season(today: dt.date | None = None) -> int:
    """Start year of the last season whose playoffs are over (a season ends in June)."""
    today = today or dt.date.today()
    return today.year - 1 if today.month >= 7 else today.year - 2


def seasons_from(first_year: int, last_year: int | None = None) -> list[str]:
    last_year = last_completed_season() if last_year is None else last_year
    return [season_str(y) for y in range(first_year, last_year + 1)]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clock_to_s(mmss: str | None) -> float | None:
    """'11:23' -> 683.0, '0:04.5' -> 4.5; None when missing."""
    if not mmss:
        return None
    m, s = mmss.strip().split(":")
    return int(m) * 60 + float(s)


def elapsed_s(period: int, clock_s: float) -> float:
    """Seconds since tip-off for (period, seconds remaining in the period)."""
    before = sum(PERIOD_S.get(p, OT_S) for p in range(1, period))
    return before + PERIOD_S.get(period, OT_S) - clock_s


# ── throttle and backoff ───────────────────────────────────────────────────────────────────────────
class Throttle:
    """Never two requests closer than 60 / rpm seconds; wait(extra) adds a backoff pause."""
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self, extra: float = 0.0) -> None:
        pause = self.last + self.interval + extra - time.time()
        if pause > 0:
            time.sleep(pause)
        self.last = time.time()


BACKOFF_S = (5, 10, 20, 40, 80)


# ── raw response cache ─────────────────────────────────────────────────────────────────────────────
def params_key(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:20]


class RawCache:
    """raw/<endpoint>/<key>.json.gz: {"endpoint", "url", "params", "fetched_at", "status", "body"}."""
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, endpoint: str, params: dict) -> Path:
        return self.root / endpoint / (params_key(params) + ".json.gz")

    def get(self, endpoint: str, params: dict) -> dict | None:
        p = self.path(endpoint, params)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt") as f:
                return json.load(f)
        except Exception:                                       # noqa: BLE001 (a torn write: refetch)
            return None

    def put(self, endpoint: str, params: dict, body, url: str, status: int) -> dict:
        rec = {"endpoint": endpoint, "url": url, "params": params, "fetched_at": utc_now(), "status": status, "body": body}
        p = self.path(endpoint, params)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with gzip.open(tmp, "wt") as f:
            json.dump(rec, f)
        os.replace(tmp, p)
        return rec


# ── manifest ───────────────────────────────────────────────────────────────────────────────────────
class Manifest:
    """manifest.json of one family: {"family", "updated", "counts", "units": {key: record}}."""
    def __init__(self, path: Path, family: str):
        self.path, self.family = Path(path), family
        self.doc = {"family": family, "contract": "docs/DATA_CONTRACT.md", "units": {}}
        if self.path.exists():
            try:
                self.doc = json.loads(self.path.read_text())
            except Exception:                                   # noqa: BLE001 (torn write: start over)
                pass
        self.doc.setdefault("units", {})

    @staticmethod
    def key(endpoint: str, season: str, unit_id) -> str:
        return "%s|%s|%s" % (endpoint, season, unit_id)

    def get(self, key: str) -> dict | None:
        return self.doc["units"].get(key)

    def mark(self, key: str, status: str, **fields) -> dict:
        rec = {"status": status, "updated": utc_now(), **fields}
        self.doc["units"][key] = rec
        return rec

    def pending(self, keys: list[str], retry_failed: bool) -> list[str]:
        out = []
        for k in keys:
            st = (self.get(k) or {}).get("status")
            if st == "ok" or st == "empty" or (st == "failed" and not retry_failed):
                continue
            out.append(k)
        return out

    def save(self) -> None:
        units = self.doc["units"].values()
        self.doc["updated"] = utc_now()
        self.doc["counts"] = {s: sum(1 for u in units if u.get("status") == s) for s in ("ok", "empty", "failed")}
        self.doc["counts"]["rows"] = sum(u.get("rows", 0) or 0 for u in units)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.doc, indent=1) + "\n")
        os.replace(tmp, self.path)


# ── tables ─────────────────────────────────────────────────────────────────────────────────────────
def add_provenance(df, source: str, endpoint: str, season: str | None, fetched_at: str, **extra):
    """The provenance columns every contract table carries, plus any unit-level fields (season_type, split)."""
    df = df.copy()
    df["source"], df["endpoint"], df["season"], df["fetched_at"] = source, endpoint, season, fetched_at
    for k, v in extra.items():
        df[k] = v
    return df


def snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def write_parquet(df, path: Path) -> int:
    """pandas -> parquet via a .part rename; returns bytes."""
    import pyarrow as pa, pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)
    return path.stat().st_size


def read_parquet(path: Path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()


def to_long(df, id_cols: list[str], keep_cols: list[str] = ()):
    """Wide endpoint rows -> the contract aggregates table: one row per numeric metric.
    id_cols name the entity (player_id or team_id and their names), keep_cols ride along (split, season_type)."""
    import pandas as pd
    prov = ["source", "endpoint", "season", "fetched_at"]
    fixed = [c for c in list(id_cols) + list(keep_cols) + prov if c in df.columns]
    num = [c for c in df.columns if c not in fixed and pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_rank")]
    if not num:
        return df.iloc[0:0][fixed].assign(metric=pd.Series(dtype=str), value=pd.Series(dtype=float))
    long = df.melt(id_vars=fixed, value_vars=num, var_name="metric", value_name="value")
    long["value"] = long["value"].astype(float)
    return long


# ── held-out tagging (sportvu/splits.json through sportvu.colab_contract.split_of) ─────────────────
_SPLITS = None


def heldout_split(date: str, away: str, home: str) -> str | None:
    """Split name of a 2015-16 game given its date (YYYY-MM-DD) and SportVU abbreviations; None if free to fit on."""
    global _SPLITS
    from sportvu import splits
    from sportvu.colab_contract import split_of, is_heldout
    if _SPLITS is None:
        _SPLITS = splits.load()
    if not date:
        return None
    y, m, d = date[:10].split("-")
    s = split_of({"game": "%s.%s.%s.%s.at.%s" % (m, d, y, away, home), "date": date[:10], "away": away, "home": home}, _SPLITS)
    return s if is_heldout(s) else None


def guard_not_heldout(items, who: str) -> int:
    """For fitting code: SystemExit when any item names a held-out game (sportvu.splits.refuse_heldout)."""
    from sportvu import splits
    return splits.refuse_heldout(items, who)


# ── a family run ───────────────────────────────────────────────────────────────────────────────────
class Family:
    """The on-disk state of one collector family under out/<family>/."""
    def __init__(self, out: Path, family: str):
        self.family = family
        self.dir = Path(out) / family
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = RawCache(Path(out) / "raw")          # shared by every family: one request per response, ever
        self.manifest = Manifest(self.dir / "manifest.json", family)
        self.parts = self.dir / "parts"

    def part_path(self, endpoint: str, season: str, unit_id) -> Path:
        return self.parts / endpoint / str(season) / ("%s.parquet" % re.sub(r"[^0-9A-Za-z._-]+", "_", str(unit_id)))

    def write_part(self, endpoint: str, season: str, unit_id, df) -> Path:
        p = self.part_path(endpoint, season, unit_id)
        write_parquet(df, p)
        return p

    def compact(self, endpoints: list[str] | None = None) -> dict:
        """parts/<endpoint>/<season>/*.parquet -> <endpoint>/<season>.parquet, held-out rows -> heldout/...
        Returns {endpoint/season: rows}."""
        import pandas as pd, pyarrow.parquet as pq
        out = {}
        for ep_dir in sorted(self.parts.glob("*")) if self.parts.exists() else []:
            if endpoints and ep_dir.name not in endpoints:
                continue
            for season_dir in sorted(ep_dir.glob("*")):
                files = sorted(season_dir.glob("*.parquet"))
                if not files:
                    continue
                df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
                if "heldout" in df.columns:
                    held = df[df["heldout"].fillna(False).astype(bool)]
                    df = df[~df["heldout"].fillna(False).astype(bool)]
                    if len(held):
                        write_parquet(held, self.dir / "heldout" / ep_dir.name / (season_dir.name + ".parquet"))
                write_parquet(df, self.dir / ep_dir.name / (season_dir.name + ".parquet"))
                out["%s/%s" % (ep_dir.name, season_dir.name)] = int(len(df))
        return out


def base_parser(description: str, default_seasons: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("NBA_COLLECT_OUT") or (DRIVE_OUT if os.path.isdir("/content/drive/MyDrive") else str(DEFAULT_OUT)),
                    help="output root (default: $NBA_COLLECT_OUT, Drive nba_harvest/collect on Colab, else data/collect)")
    ap.add_argument("--seasons", nargs="*", default=default_seasons, help="seasons like 2015-16 (default: %s)" % " ".join(default_seasons[:3] + (["..."] if len(default_seasons) > 3 else [])))
    ap.add_argument("--season-types", nargs="*", default=["Regular Season"], choices=SEASON_TYPES)
    ap.add_argument("--rpm", type=float, default=30.0, help="stats.nba.com requests per minute (default 30)")
    ap.add_argument("--limit", type=int, help="stop after this many units")
    ap.add_argument("--retry-failed", action="store_true", help="redo units whose last attempt failed")
    ap.add_argument("--compact-only", action="store_true", help="only rebuild the per-season parquet from parts/")
    ap.add_argument("--dry-run", action="store_true", help="list the pending units and stop")
    return ap


def run_units(fam: Family, units: list[tuple[str, dict]], do_unit, args, label: str) -> str:
    """The shared loop: pending units in order, one try each, manifest after every unit, compact at the end.
    do_unit(unit) -> (rows, extra manifest fields) or raises. Returns the RUN OK / RUN FAILED line."""
    keys = [k for k, _ in units]
    pending = fam.manifest.pending(keys, args.retry_failed)
    if args.limit:
        pending = pending[:args.limit]
    print("%s: %d units listed, %d done, %d pending" % (label, len(keys), len(keys) - len(fam.manifest.pending(keys, True)), len(pending)), flush=True)
    if args.dry_run:
        for k in pending[:50]:
            print("  ", k)
        return "DRY RUN: %d pending" % len(pending)
    by_key = dict(units)
    failed = 0
    for i, k in enumerate(pending):
        unit = by_key[k]
        t = time.time()
        try:
            rows, fields = do_unit(unit)
            fields = {"params": unit.get("params"), **fields}       # a unit may report the parameters it really used
            fam.manifest.mark(k, "ok" if rows else "empty", rows=int(rows), seconds=round(time.time() - t, 1), **fields)
            print("%5d/%d %-60s %6d rows (t+%.0f s)" % (i + 1, len(pending), k[:60], rows, time.time() - T0), flush=True)
        except KeyboardInterrupt:
            fam.manifest.save()
            raise
        except Exception as e:                                  # noqa: BLE001 (one unit never ends the run)
            failed += 1
            fam.manifest.mark(k, "failed", params=unit.get("params"), error=repr(e)[:400], seconds=round(time.time() - t, 1))
            print("%5d/%d %-60s FAILED %s" % (i + 1, len(pending), k[:60], repr(e)[:160]), flush=True)
            if getattr(e, "stop_run", False):
                print("!! stopping the run: %s" % e, flush=True)
                break
        fam.manifest.save()
    fam.manifest.save()
    compacted = fam.compact()
    print("compacted:", json.dumps(compacted), flush=True)
    left = fam.manifest.pending(keys, True)
    c = fam.manifest.doc.get("counts", {})
    line = "%d ok, %d empty, %d failed, %d rows; %d units still pending" % (c.get("ok", 0), c.get("empty", 0), c.get("failed", 0), c.get("rows", 0), len(left))
    final = ("RUN FAILED: " if (failed or left) else "RUN OK: ") + line
    print(final, flush=True)
    if os.path.isdir("/content"):
        open("/content/run_status.txt", "w").write(final)
    return final
