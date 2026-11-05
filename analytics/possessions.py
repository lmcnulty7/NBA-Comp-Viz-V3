"""analytics/possessions.py: possession segments from SportVU frames and play-by-play (DATA_CONTRACT section 19).

Inputs are the section 4 frames table (one row per player per frame) and the section 5 play-by-play table
(one row per event, `frame` = nearest frame id or -1). Output: one row per possession with the section 19
columns.

Rules (section 19):
  - a possession ends at a made field goal (shot_made), at the last free throw of a trip when it is made
    (free_throws; a missed last free throw followed by a defensive rebound ends it as free_throws too), at a
    turnover, at a defensive rebound (end_kind shot_missed) and at period_end; an offensive rebound continues it
  - the next possession starts at the frame of the ending event (or, when that event has no frame, the next
    frame of the period with a smaller clock)
  - offense = team of the first scoring, shooting or turnover event in the span, else the team opposite the
    previous possession's offense
  - points = points of the span's events credited to the offense
  - gap = consecutive frame-level clocks inside the span differ by more than 2 s
  - possessions shorter than 1 frame are dropped and counted (`possessions.attrs["n_dropped"]`)

Assumptions taken where the contract is silent:
  - basketball-reference play-by-play carries no period_end events: when a period has none, the last open
    possession is closed at the period's last frame with end_kind period_end and end_event = the period's
    last event
  - an and-one (a made shot followed by free throws of the same team at the same clock) is one possession
    that ends at the free throws (free_throws), so the free throws never form a zero-length possession
  - the free throw position in a trip is read from the description ("1 of 2"); a free throw without it is
    the last of its trip, except a technical free throw, which never ends a possession
  - a rebound with no preceding miss is defensive when its team differs from the current offense

CLI: python -m analytics.possessions --game G --frames PATH --pbp PATH --out-dir analytics/possessions
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP_S = 2.0
END_KINDS = ("shot_made", "shot_missed", "free_throws", "turnover", "period_end", "other")
OFFENSE_KINDS = ("shot_made", "shot_missed", "ft_made", "ft_missed", "turnover")
FT_RE = re.compile(r"(\d+)\s+of\s+(\d+)")
COLUMNS = ["sportvu_game", "possession", "period", "start_frame", "end_frame", "start_clock_s", "end_clock_s",
           "start_event", "end_event", "offense_team_id", "end_kind", "points", "gap", "n_frames"]
DTYPES = {"sportvu_game": "string", "possession": "int32", "period": "int8", "start_frame": "int32",
          "end_frame": "int32", "start_clock_s": "float32", "end_clock_s": "float32", "start_event": "int32",
          "end_event": "int32", "offense_team_id": "Int32", "end_kind": "string", "points": "int16",
          "gap": "bool", "n_frames": "int32"}


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────
def frame_level(frames: pd.DataFrame) -> pd.DataFrame:
    """First row of each frame: frame, period, game_clock_s, sorted by frame."""
    fl = frames.drop_duplicates("frame", keep="first")[["frame", "period", "game_clock_s"]]
    return fl.sort_values("frame", kind="stable").reset_index(drop=True)


def team_ids(frames: pd.DataFrame) -> tuple[int | None, int | None]:
    """(home team id, away team id) from the frames rows' home flag."""
    home = frames.loc[frames["home"].astype(bool), "team_id"]
    away = frames.loc[~frames["home"].astype(bool), "team_id"]
    h = int(home.iloc[0]) if len(home) else None
    a = int(away.iloc[0]) if len(away) else None
    return h, a


def ft_position(desc) -> tuple[bool, bool]:
    """(is_last_of_trip, is_technical) for a free throw description."""
    d = str(desc or "")
    if "technical" in d.lower():
        return False, True
    m = FT_RE.search(d)
    if not m:
        return True, False
    return int(m.group(1)) >= int(m.group(2)), False


class _Period:
    """Frame-level lookups for one period."""

    def __init__(self, fl: pd.DataFrame):
        self.frames = fl["frame"].to_numpy(dtype=np.int64)
        self.clocks = fl["game_clock_s"].to_numpy(dtype=np.float64)

    @property
    def empty(self) -> bool:
        return len(self.frames) == 0

    def nearest(self, clock_s: float) -> int:
        if self.empty:
            return -1
        return int(self.frames[int(np.argmin(np.abs(self.clocks - float(clock_s))))])

    def next_smaller(self, clock_s: float) -> int:
        """First frame (in frame order) whose clock is below clock_s, else the nearest frame."""
        if self.empty:
            return -1
        hit = np.where(self.clocks < float(clock_s))[0]
        return int(self.frames[hit[0]]) if len(hit) else self.nearest(clock_s)

    def span(self, start_frame: int, end_frame: int):
        """Indices of the frames in [start_frame, end_frame]."""
        if self.empty or start_frame < 0 or end_frame < start_frame:
            return np.zeros(0, dtype=np.int64)
        return np.where((self.frames >= start_frame) & (self.frames <= end_frame))[0]

    def clock_of(self, frame: int) -> float:
        i = np.where(self.frames == frame)[0]
        return float(self.clocks[i[0]]) if len(i) else float("nan")


# ── segmentation ───────────────────────────────────────────────────────────────────────────────────
def segment(frames: pd.DataFrame, events: pd.DataFrame, sportvu_game: str) -> pd.DataFrame:
    """Possessions (DATA_CONTRACT section 19) from a frames table and its play-by-play table."""
    fl = frame_level(frames)
    home_id, away_id = team_ids(frames)
    side_id = {"home": home_id, "away": away_id}
    ev = events.sort_values("event", kind="stable").reset_index(drop=True)
    ev_idx = ev["event"].to_numpy(dtype=np.int64)
    ev_period = ev["period"].to_numpy(dtype=np.int64)
    ev_clock = ev["clock_s"].to_numpy(dtype=np.float64)
    ev_kind = ev["kind"].astype(object).to_numpy()
    ev_team = ev["team"].astype(object).to_numpy() if "team" in ev else np.array([None] * len(ev), dtype=object)
    ev_points = pd.to_numeric(ev["points"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    ev_frame = ev["frame"].to_numpy(dtype=np.int64) if "frame" in ev else np.full(len(ev), -1, dtype=np.int64)
    ev_desc = ev["desc"].astype(object).to_numpy() if "desc" in ev else np.array([None] * len(ev), dtype=object)

    def tid(i: int):
        t = ev_team[i]
        return side_id.get(t) if isinstance(t, str) else None

    periods = sorted(set(fl["period"].astype(int).tolist()) | set(int(p) for p in ev_period))
    rows, n_dropped = [], 0
    prev_offense = None

    def emit(period: int, P: _Period, start_frame: int, end_frame: int, start_event: int, end_event: int,
             span_events: list[int], end_kind: str) -> None:
        nonlocal prev_offense, n_dropped
        offense = None
        for i in span_events:
            if ev_kind[i] in OFFENSE_KINDS and tid(i) is not None:
                offense = tid(i)
                break
        if offense is None and prev_offense is not None:
            offense = away_id if prev_offense == home_id else home_id
        points = int(sum(ev_points[i] for i in span_events if offense is not None and tid(i) == offense))
        idx = P.span(start_frame, end_frame)
        if len(idx) < 1:
            n_dropped += 1
            prev_offense = offense if offense is not None else prev_offense
            return
        clocks = P.clocks[idx]
        gap = bool(len(clocks) > 1 and np.max(np.abs(np.diff(clocks))) > GAP_S)
        start_clock, end_clock = float(clocks[0]), float(clocks[-1])
        end_clock = min(end_clock, start_clock)
        rows.append({"sportvu_game": sportvu_game, "possession": len(rows), "period": period,
                     "start_frame": int(P.frames[idx[0]]), "end_frame": int(P.frames[idx[-1]]),
                     "start_clock_s": start_clock, "end_clock_s": end_clock, "start_event": int(start_event),
                     "end_event": int(end_event), "offense_team_id": offense, "end_kind": end_kind,
                     "points": points, "gap": gap, "n_frames": int(len(idx))})
        prev_offense = offense if offense is not None else prev_offense

    for period in periods:
        P = _Period(fl[fl["period"].astype(int) == period])
        pe = np.where(ev_period == period)[0]
        if P.empty and len(pe) == 0:
            continue
        start_frame = int(P.frames[0]) if not P.empty else -1
        start_event = int(ev_idx[pe[0]]) if len(pe) else -1
        span_events: list[int] = []
        pending_miss = None        # (shooter team id, end_kind if a defensive rebound follows)
        deferred_and_one = False   # a made shot whose free throws are still to come
        closed_by_period_end = False

        def event_frame(i: int) -> int:
            return int(ev_frame[i]) if ev_frame[i] >= 0 else P.nearest(ev_clock[i])

        def close(i: int, end_kind: str) -> None:
            nonlocal start_frame, start_event, span_events, pending_miss, deferred_and_one
            end_frame = event_frame(i)
            emit(period, P, start_frame, end_frame, start_event, int(ev_idx[i]), span_events, end_kind)
            start_frame = end_frame if ev_frame[i] >= 0 else P.next_smaller(ev_clock[i])
            start_event = int(ev_idx[i]) + 1
            span_events = []
            pending_miss = None
            deferred_and_one = False

        for n, i in enumerate(pe):
            span_events.append(int(i))
            kind, team = ev_kind[i], tid(i)
            if kind == "shot_made":
                # and-one: free throws of the same team at the same clock follow the made shot
                deferred_and_one = False
                for j in pe[n + 1:]:
                    if ev_kind[j] in OFFENSE_KINDS or ev_kind[j] in ("rebound", "period_end"):
                        deferred_and_one = (ev_kind[j] in ("ft_made", "ft_missed") and tid(j) == team
                                            and abs(ev_clock[j] - ev_clock[i]) <= 1.0)
                        break
                if not deferred_and_one:
                    close(i, "shot_made")
            elif kind == "shot_missed":
                pending_miss = (team, "shot_missed")
            elif kind in ("ft_made", "ft_missed"):
                last, technical = ft_position(ev_desc[i])
                if technical:
                    continue
                if last and kind == "ft_made":
                    close(i, "free_throws")
                elif last:
                    pending_miss = (team, "free_throws")
            elif kind == "rebound":
                shooter = pending_miss[0] if pending_miss else None
                end_kind = pending_miss[1] if pending_miss else "shot_missed"
                if shooter is None:
                    cur = None
                    for k in span_events:
                        if ev_kind[k] in OFFENSE_KINDS and tid(k) is not None:
                            cur = tid(k)
                            break
                    shooter = cur if cur is not None else (
                        (away_id if prev_offense == home_id else home_id) if prev_offense is not None else None)
                if shooter is not None and team is not None and team != shooter:
                    close(i, end_kind)
                else:
                    pending_miss = None
            elif kind == "turnover":
                close(i, "turnover")
            elif kind == "period_end":
                close(i, "period_end")
                closed_by_period_end = True
        if not closed_by_period_end and not P.empty:
            end_frame = int(P.frames[-1])
            end_event = int(ev_idx[pe[-1]]) if len(pe) else start_event
            if len(pe) and start_event > end_event:
                start_event = end_event
            emit(period, P, start_frame, end_frame, start_event, end_event, span_events, "period_end")

    out = pd.DataFrame(rows, columns=COLUMNS)
    out = out.astype(DTYPES)
    out.attrs["n_dropped"] = n_dropped
    return out


# ── summary ────────────────────────────────────────────────────────────────────────────────────────
def summary(possessions: pd.DataFrame) -> dict:
    n = int(len(possessions))
    dur = (possessions["start_clock_s"].astype(float) - possessions["end_clock_s"].astype(float))
    by_kind = possessions["end_kind"].astype(str).value_counts().to_dict() if n else {}
    out = {"n": n,
           "by_end_kind": {k: int(by_kind.get(k, 0)) for k in END_KINDS if by_kind.get(k, 0)},
           "points_total": int(possessions["points"].sum()) if n else 0,
           "n_gap": int(possessions["gap"].sum()) if n else 0,
           "mean_duration_s": round(float(dur.mean()), 3) if n else None,
           "share_offense_known": round(float(possessions["offense_team_id"].notna().mean()), 4) if n else None}
    if "n_dropped" in possessions.attrs:
        out["n_dropped"] = int(possessions.attrs["n_dropped"])
    return out


# ── io ─────────────────────────────────────────────────────────────────────────────────────────────
def game_meta(game: str) -> dict:
    """'12.16.2015.PHX.at.GSW' -> {'game', 'date', 'away', 'home', 'season'}."""
    mm, dd, yyyy, away, _at, home = game.split(".")
    y = int(yyyy)
    season = "%d-%02d" % (y, (y + 1) % 100) if int(mm) >= 9 else "%d-%02d" % (y - 1, y % 100)
    return {"game": game, "date": "%s-%s-%s" % (yyyy, mm, dd), "away": away, "home": home, "season": season}


def read_frames(frames_path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(frames_path, columns=["frame", "period", "game_clock_s", "team_id", "home"]).to_pandas()


def read_events(game: str, pbp_path, frames: pd.DataFrame) -> pd.DataFrame:
    """Section 5 events from a `<game>_pbp.parquet` or a fetch_pbp.py JSON joined to the frames."""
    pbp_path = Path(pbp_path)
    if pbp_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(pbp_path).to_pandas()
    import pyarrow as pa
    from sportvu.colab_contract import frame_clocks, join_pbp
    events = json.loads(pbp_path.read_text())["events"]
    fl = frame_level(frames)
    table = pa.table({"frame": fl["frame"].to_numpy(), "period": fl["period"].to_numpy(),
                      "game_clock_s": fl["game_clock_s"].to_numpy()})
    fp, fc = frame_clocks(table)
    return join_pbp(game, events, fp, fc).to_pandas()


def output_path(game: str, out_dir) -> Path:
    from sportvu import splits
    from sportvu.colab_contract import HELDOUT_SUBDIR, is_heldout, split_of
    meta = game_meta(game)
    split = split_of(meta, splits.load())
    d = Path(out_dir) / HELDOUT_SUBDIR if is_heldout(split) else Path(out_dir)
    return d / meta["season"] / (game + ".parquet")


def run(game: str, frames_path, pbp_json_or_parquet, out_dir) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    frames = read_frames(frames_path)
    events = read_events(game, pbp_json_or_parquet, frames)
    poss = segment(frames, events, game)
    dest = output_path(game, out_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(poss, preserve_index=False), dest, compression="zstd")
    s = summary(poss)
    s["path"] = str(dest)
    print(json.dumps(s, indent=1))
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True, help="SportVU game id, e.g. 12.16.2015.PHX.at.GSW")
    ap.add_argument("--frames", required=True, help="frames parquet (DATA_CONTRACT section 4)")
    ap.add_argument("--pbp", required=True, help="<game>_pbp.parquet (section 5) or a fetch_pbp.py JSON")
    ap.add_argument("--out-dir", default="analytics/possessions")
    a = ap.parse_args(argv)
    run(a.game, a.frames, a.pbp, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
