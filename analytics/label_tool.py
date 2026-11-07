"""analytics/label_tool.py: local hand-labeling tool for actions and ball-screen coverages.

    python -m analytics.label_tool --data DATA_ROOT --game G [--sample N --seed 0] [--port 8765]
    python -m analytics.label_tool --data DATA_ROOT --game G --export OUT.parquet
    python -m analytics.label_tool --data DATA_ROOT [--game G] --kappa [--passes 1 2]

Serves analytics/label_tool.html (everything inline, no network) from a standard-library http.server and
two JSON endpoints:

    GET  /possessions           the possession list (sampled when --sample is given) with flags and counts
    GET  /possession?index=i    the i-th possession: its frames (ten players and the ball at the source
                                rate, every second frame when the span is over MAX_FRAMES frames), the
                                players (name, jersey, team, side) and the existing labels
    POST /label                 append an action or screen label (JSON body, validated through labels.py;
                                400 with the problems when invalid)
    POST /delete                append a tombstone for a label ({"kind": "action"|"screen", "id": k})
    POST /visited               record that this pass finished a possession ({"possession": p}); the page
                                sends it on next/previous, and --kappa compares only possessions both
                                passes visited

POST bodies are JSON (Content-Type application/json, at most 1 MiB) from 127.0.0.1 or localhost only.
A held-out game (sportvu/splits.json) opens with a HELD-OUT notice: its labels are evaluation labels.

Inputs are read at run time and never copied anywhere: the section 4 frames parquet (``--frames``, or
DATA_ROOT/<game>.parquet, DATA_ROOT/frames/<game>.parquet, DATA_ROOT/sportvu_contract/<game>.parquet) and
the section 19 possessions parquet (``--possessions``, or any <game>.parquet under DATA_ROOT/analytics/
possessions or DATA_ROOT/possessions). Labels go to DATA_ROOT/labels/{actions,screens}/<game>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from analytics import labels as L

HTML_PATH = Path(__file__).with_name("label_tool.html")
MAX_FRAMES = 2000
DEFAULT_PORT = 8765
MAX_BODY = 1 << 20
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------

def find_frames(data_root, game: str, explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    root = Path(data_root)
    for p in (root / f"{game}.parquet", root / "frames" / f"{game}.parquet", root / "sportvu_contract" / f"{game}.parquet"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no frames parquet for {game} under {root} (pass --frames)")


def find_possessions(data_root, game: str, explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    root = Path(data_root)
    for base in (root / "analytics" / "possessions", root / "possessions"):
        if base.is_dir():
            hits = sorted(base.rglob(f"{game}.parquet"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"no possessions parquet for {game} under {root} (pass --possessions)")


def heldout_tokens() -> set:
    """sportvu/splits.json held-out tokens (game tags, SportVU ids, video ids). A missing splits file is
    a SystemExit from sportvu.splits.load: the tool never runs without knowing the held-out games."""
    from sportvu import splits
    return set(splits.heldout_tokens())


# --------------------------------------------------------------------------------------
# Session: the game in memory plus the label files
# --------------------------------------------------------------------------------------

class LabelSession:
    """Frames and possessions of one game plus the append-only label files."""

    def __init__(self, data_root, game: str, frames: pd.DataFrame, possessions: pd.DataFrame,
                 sample: int | None = None, seed: int = 0, labeller_pass: int = 1, label_source: str = "hand",
                 heldout: set | None = None):
        self.data_root = Path(data_root)
        self.game = str(game)
        self.frames = frames.sort_values("frame", kind="stable").reset_index(drop=True)
        self.possessions = possessions.sort_values("possession").reset_index(drop=True)
        self.labeller_pass = int(labeller_pass)
        self.label_source = str(label_source)
        self.fps = L.estimate_fps(self.frames)
        self.lock = threading.Lock()
        self.flags = L.possession_flags(self.possessions, self.frames)
        tokens = heldout if heldout is not None else heldout_tokens()
        self.heldout = L.is_heldout(self.game, tokens)
        if sample:
            s = L.sample_possessions(self.possessions, self.frames, int(sample), int(seed), tokens, flags=self.flags)
            self.order = [int(p) for p in s["possession"]]
            self.sample_info = {"n": int(sample), "seed": int(seed), **{k: v for k, v in s.attrs.items()}}
        else:
            self.order = [int(p) for p in self.possessions["possession"]]
            self.sample_info = None
        self._frame_vals = self.frames["frame"].to_numpy()
        self.players = self._players()

    # -- players ------------------------------------------------------------------------
    def _players(self) -> dict:
        cols = [c for c in ("player_id", "player", "jersey", "team_id", "team", "home") if c in self.frames.columns]
        d = self.frames[cols].drop_duplicates("player_id")
        out = {}
        for r in d.itertuples(index=False):
            pid = int(r.player_id)
            out[pid] = {"id": pid, "name": str(getattr(r, "player", pid)),
                        "jersey": None if pd.isna(getattr(r, "jersey", None)) else str(getattr(r, "jersey", "")),
                        "team_id": int(r.team_id), "team": str(getattr(r, "team", "")),
                        "home": bool(getattr(r, "home", False))}
        return out

    # -- possessions --------------------------------------------------------------------
    def possession_row(self, index: int) -> pd.Series:
        if index < 0 or index >= len(self.order):
            raise IndexError(index)
        p = self.order[index]
        return self.possessions.loc[self.possessions["possession"] == p].iloc[0]

    def possession_list(self) -> list[dict]:
        acts, scrs = self.read()
        na = acts.groupby("possession").size() if len(acts) else pd.Series(dtype=int)
        ns = scrs.groupby("possession").size() if len(scrs) else pd.Series(dtype=int)
        fl = self.flags.set_index("possession")
        seen = self.visited()
        out = []
        for i, p in enumerate(self.order):
            r = self.possessions.loc[self.possessions["possession"] == p].iloc[0]
            out.append({"index": i, "possession": int(p), "period": int(r["period"]),
                        "start_clock_s": _f(r.get("start_clock_s")), "end_kind": str(r.get("end_kind", "")),
                        "offense_team_id": None if pd.isna(r["offense_team_id"]) else int(r["offense_team_id"]),
                        "half_court": bool(fl.loc[p, "half_court"]) if p in fl.index else None,
                        "screen_candidate": bool(fl.loc[p, "screen_candidate"]) if p in fl.index else None,
                        "n_actions": int(na.get(p, 0)), "n_screens": int(ns.get(p, 0)), "visited": p in seen})
        return out

    def possession_payload(self, index: int) -> dict:
        r = self.possession_row(index)
        s, e = int(r["start_frame"]), int(r["end_frame"])
        lo, hi = np.searchsorted(self._frame_vals, s, "left"), np.searchsorted(self._frame_vals, e, "right")
        span = self.frames.iloc[lo:hi]
        frames_out, downsampled = frames_payload(span, MAX_FRAMES)
        off = None if pd.isna(r["offense_team_id"]) else int(r["offense_team_id"])
        players = {}
        for pid in sorted(set(int(v) for v in span["player_id"].unique())):
            p = dict(self.players.get(pid, {"id": pid, "name": str(pid), "jersey": None, "team_id": None, "team": "", "home": False}))
            p["side"] = None if off is None else ("offense" if p["team_id"] == off else "defense")
            players[str(pid)] = p
        acts, scrs = self.read()
        p = int(r["possession"])
        fl = self.flags.loc[self.flags["possession"] == p]
        return {"index": index, "n": len(self.order), "game": self.game, "fps": self.fps,
                "possession": {"possession": p, "period": int(r["period"]), "start_frame": s, "end_frame": e,
                               "offense_team_id": off, "end_kind": str(r.get("end_kind", "")),
                               "points": _i(r.get("points")), "start_clock_s": _f(r.get("start_clock_s")),
                               "end_clock_s": _f(r.get("end_clock_s")),
                               "half_court": bool(fl["half_court"].iloc[0]) if len(fl) else None,
                               "screen_candidate": bool(fl["screen_candidate"].iloc[0]) if len(fl) else None},
                "downsampled": downsampled, "frames": frames_out, "players": players,
                "labels": self.labels_of(p, acts, scrs), "visited": p in self.visited(),
                "meta": {"labeller_pass": self.labeller_pass, "label_source": self.label_source, "heldout": self.heldout,
                         "sample": self.sample_info, "action_types": list(L.ACTION_TYPES),
                         "coverages": list(L.COVERAGES), "confidences": list(L.CONFIDENCES)}}

    # -- labels -------------------------------------------------------------------------
    def read(self):
        return L.read_labels(self.data_root, self.game)

    def labels_of(self, possession: int, acts=None, scrs=None) -> dict:
        if acts is None:
            acts, scrs = self.read()
        a = acts.loc[acts["possession"] == possession] if len(acts) else acts
        s = scrs.loc[scrs["possession"] == possession] if len(scrs) else scrs
        return {"actions": _records(a), "screens": _records(s)}

    def add_label(self, body: dict) -> dict:
        kind = str(body.get("kind", ""))
        if kind not in ("action", "screen"):
            raise ValueError("kind must be 'action' or 'screen'")
        with self.lock:
            acts, scrs = self.read()
            rec = {"sportvu_game": self.game, "label_source": body.get("label_source", self.label_source),
                   "labeller_pass": int(body.get("labeller_pass", self.labeller_pass)), "labelled_at": L.now_iso()}
            if kind == "action":
                rec.update({"possession": body.get("possession"), "action_id": L.next_id(acts, "actions"),
                            "action_type": body.get("action_type"), "start_frame": body.get("start_frame"),
                            "end_frame": body.get("end_frame"), "ball_handler_id": body.get("ball_handler_id"),
                            "screener_id": body.get("screener_id")})
                new = pd.DataFrame([rec])
                try:
                    new = L.cast(new, "actions")
                except (KeyError, ValueError) as exc:
                    raise ValueError(str(exc)) from exc
                probs = L.validate_labels(pd.concat([acts, new], ignore_index=True), scrs, self.possessions, self.frames)
                if probs:
                    raise ValueError("; ".join(probs))
                L.write_labels(self.data_root, self.game, new, None, labelled_at=rec["labelled_at"])
            else:
                rec.update({"possession": body.get("possession"), "screen_id": L.next_id(scrs, "screens"),
                            "action_id": body.get("action_id"), "coverage": body.get("coverage"),
                            "on_ball_defender_id": body.get("on_ball_defender_id"),
                            "screener_defender_id": body.get("screener_defender_id"),
                            "confidence": body.get("confidence", "sure")})
                if rec["possession"] is None and rec["action_id"] is not None and len(acts):
                    m = acts.loc[acts["action_id"] == int(rec["action_id"])]
                    if len(m):
                        rec["possession"] = int(m["possession"].iloc[0])
                new = pd.DataFrame([rec])
                try:
                    new = L.cast(new, "screens")
                except (KeyError, ValueError) as exc:
                    raise ValueError(str(exc)) from exc
                probs = L.validate_labels(acts, pd.concat([scrs, new], ignore_index=True), self.possessions, self.frames)
                if probs:
                    raise ValueError("; ".join(probs))
                L.write_labels(self.data_root, self.game, None, new, labelled_at=rec["labelled_at"])
            saved = _records(new)[0]
            return {"ok": True, "label": saved, "kind": kind, "labels": self.labels_of(int(saved["possession"]))}

    def visited(self) -> set:
        v = L.read_visited(self.data_root, self.game)
        return set(v.loc[v["labeller_pass"] == self.labeller_pass, "possession"].astype(int)) if len(v) else set()

    def mark_visited(self, body: dict) -> dict:
        try:
            p = int(body["possession"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("possession must be an integer") from exc
        if p not in set(int(x) for x in self.possessions["possession"]):
            raise ValueError(f"no possession {p}")
        with self.lock:
            L.mark_visited(self.data_root, self.game, p, self.labeller_pass)
        return {"ok": True, "possession": p, "labeller_pass": self.labeller_pass, "visited": sorted(self.visited())}

    def delete_label(self, body: dict) -> dict:
        kind = str(body.get("kind", ""))
        if kind not in ("action", "screen"):
            raise ValueError("kind must be 'action' or 'screen'")
        try:
            lid = int(body["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("id must be an integer") from exc
        with self.lock:
            acts, scrs = self.read()
            if kind == "action":
                m = acts.loc[acts["action_id"] == lid] if len(acts) else acts
                if not len(m):
                    raise ValueError(f"no action {lid}")
                possession = int(m["possession"].iloc[0])
                # an action's screens go with it (append-only: tombstones)
                for sid in (scrs.loc[scrs["action_id"] == lid, "screen_id"].tolist() if len(scrs) else []):
                    L.append_tombstone(self.data_root, self.game, "screens", int(sid))
                L.append_tombstone(self.data_root, self.game, "actions", lid)
            else:
                m = scrs.loc[scrs["screen_id"] == lid] if len(scrs) else scrs
                if not len(m):
                    raise ValueError(f"no screen {lid}")
                possession = int(m["possession"].iloc[0])
                L.append_tombstone(self.data_root, self.game, "screens", lid)
            return {"ok": True, "kind": kind, "id": lid, "labels": self.labels_of(possession)}


def _f(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except TypeError:
        return None


def _i(v):
    try:
        return None if v is None or pd.isna(v) else int(v)
    except TypeError:
        return None


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if (v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA)
                        else (int(v) if isinstance(v, (np.integer,)) else (bool(v) if isinstance(v, np.bool_) else v)))
                    for k, v in r.items()})
    return out


def frames_payload(span: pd.DataFrame, max_frames: int = MAX_FRAMES) -> tuple[list[dict], bool]:
    """Per-frame dicts (frame, clock, shot_clock, ball [x, y, z], players [[id, x, y], ...]); every
    second frame when the span has more than ``max_frames`` frames."""
    sp = span.sort_values("frame", kind="stable")
    fr = sp["frame"].to_numpy()
    if len(fr) == 0:
        return [], False
    bounds = np.nonzero(np.diff(fr))[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(fr)]])
    downsampled = len(starts) > max_frames
    step = 2 if downsampled else 1
    clock = sp["game_clock_s"].to_numpy(dtype="float64") if "game_clock_s" in sp.columns else np.full(len(fr), np.nan)
    shot = sp["shot_clock_s"].to_numpy(dtype="float64") if "shot_clock_s" in sp.columns else np.full(len(fr), np.nan)
    bx = sp["ball_x"].to_numpy(dtype="float64")
    by = sp["ball_y"].to_numpy(dtype="float64")
    bz = sp["ball_z"].to_numpy(dtype="float64") if "ball_z" in sp.columns else np.full(len(fr), np.nan)
    pid = sp["player_id"].to_numpy()
    x = sp["x"].to_numpy(dtype="float64")
    y = sp["y"].to_numpy(dtype="float64")
    out = []
    for a, b in list(zip(starts, ends))[::step]:
        ball = [None if np.isnan(v) else round(float(v), 2) for v in (bx[a], by[a], bz[a])]
        out.append({"frame": int(fr[a]), "clock": None if np.isnan(clock[a]) else round(float(clock[a]), 2),
                    "shot_clock": None if np.isnan(shot[a]) else round(float(shot[a]), 2), "ball": ball,
                    "players": [[int(pid[k]), round(float(x[k]), 2), round(float(y[k]), 2)] for k in range(a, b)]})
    return out, downsampled


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------

def make_handler(session: LabelSession, html_path: Path = HTML_PATH):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/possessions":
                    self._json(200, {"game": session.game, "possessions": session.possession_list(),
                                     "sample": session.sample_info, "labeller_pass": session.labeller_pass,
                                     "heldout": session.heldout,
                                     "notice": "HELD-OUT game: evaluation labels only" if session.heldout else None})
                elif u.path == "/possession":
                    try:
                        i = int(q.get("index", ["0"])[0])
                    except ValueError:
                        return self._json(400, {"error": "index must be an integer"})
                    try:
                        self._json(200, session.possession_payload(i))
                    except IndexError:
                        self._json(404, {"error": f"no possession at index {i}; n = {len(session.order)}"})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

        def do_POST(self):
            u = urlparse(self.path)
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
                else (self.headers.get("Host") or "").split("]")[0] + "]"
            if host and host not in LOCAL_HOSTS:
                return self._json(403, {"error": f"POST accepted from 127.0.0.1 or localhost only, not {host!r}"})
            ctype = (self.headers.get("Content-Type") or "").strip().lower()
            if not ctype.startswith("application/json"):
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n < 0:
                    raise ValueError
            except (TypeError, ValueError):
                return self._json(400, {"error": "bad Content-Length"})
            if n > MAX_BODY:
                return self._json(413, {"error": f"body over {MAX_BODY} bytes"})
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": "body is not JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "body must be a JSON object"})
            try:
                if u.path == "/label":
                    self._json(200, session.add_label(body))
                elif u.path == "/delete":
                    self._json(200, session.delete_label(body))
                elif u.path == "/visited":
                    self._json(200, session.mark_visited(body))
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

    return Handler


def make_server(session: LabelSession, port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """A bound (not yet serving) server; port 0 picks a free port (``server.server_address[1]``)."""
    srv = ThreadingHTTPServer((host, port), make_handler(session))
    srv.daemon_threads = True
    return srv


# --------------------------------------------------------------------------------------
# Export and kappa
# --------------------------------------------------------------------------------------

def export_parquet(data_root, game: str, out, labeller_pass: int | None = 1) -> dict:
    """Write the game's actions to OUT.parquet and its screens next to it as OUT_screens.parquet.

    ``labeller_pass`` defaults to 1: the export is what training reads, and the relabel pass 2 exists for
    the kappa check only. ``None`` exports every pass.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    acts, scrs = L.read_labels(data_root, game, labeller_pass=labeller_pass)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_s = out.with_name(out.stem + "_screens" + out.suffix)
    pq.write_table(pa.Table.from_pandas(acts, schema=L.ACTIONS_SCHEMA, preserve_index=False), out)
    pq.write_table(pa.Table.from_pandas(scrs, schema=L.SCREENS_SCHEMA, preserve_index=False), out_s)
    return {"actions": str(out), "screens": str(out_s), "n_actions": len(acts), "n_screens": len(scrs),
            "labeller_pass": labeller_pass}


def visited_by_pass(data_root, game: str, labeller_pass: int, acts: pd.DataFrame | None = None) -> tuple[set, str]:
    """(possessions ``labeller_pass`` finished, source): the visited record when the pass has one, else
    the possessions it labelled (older roots without a visited file)."""
    v = L.read_visited(data_root, game)
    if len(v) and (v["labeller_pass"] == labeller_pass).any():
        return set(v.loc[v["labeller_pass"] == labeller_pass, "possession"].astype(int)), "visited"
    if acts is None:
        acts, _ = L.read_labels(data_root, game, labeller_pass=labeller_pass)
    return set(acts["possession"].astype(int)), "labels"


def kappa_report(data_root, games: list[str], passes=(1, 2), heldout: Iterable[str] | None = None) -> dict:
    """Cohen's kappa between two labeller passes over ``games`` (pooled) and per game.

    Only possessions both passes visited (``visited_by_pass``: the visited record, or the labelled
    possessions of a pass without one) enter, so an action one pass saw and the other did not counts as
    ``none``. Games carrying a held-out token are skipped and listed in ``skipped_heldout``: the kappa
    tightens the definitions, and held-out games tune nothing. ``heldout`` defaults to
    ``heldout_tokens()`` (sportvu/splits.json); pass an empty tuple to compare everything.
    """
    p1, p2 = int(passes[0]), int(passes[1])
    if heldout is None:
        heldout = heldout_tokens()
    toks = set(str(t) for t in heldout if t)
    skipped = [g for g in games if L.is_heldout(g, toks)]
    games = [g for g in games if g not in skipped]
    per = {}
    all1 = {"actions": [], "screens": []}
    all2 = {"actions": [], "screens": []}
    for g in games:
        a1, s1 = L.read_labels(data_root, g, labeller_pass=p1)
        a2, s2 = L.read_labels(data_root, g, labeller_pass=p2)
        v1, src1 = visited_by_pass(data_root, g, p1, a1)
        v2, src2 = visited_by_pass(data_root, g, p2, a2)
        both = v1 & v2
        a1, a2 = a1.loc[a1["possession"].isin(both)], a2.loc[a2["possession"].isin(both)]
        s1, s2 = s1.loc[s1["possession"].isin(both)], s2.loc[s2["possession"].isin(both)]
        per[g] = L.kappa((a1, s1), (a2, s2))
        per[g]["n_possessions"] = len(both)
        per[g]["visited_source"] = {str(p1): src1, str(p2): src2}
        all1["actions"].append(a1); all1["screens"].append(s1)
        all2["actions"].append(a2); all2["screens"].append(s2)
    cat = lambda d: {k: (pd.concat(v, ignore_index=True) if v else None) for k, v in d.items()}
    pooled = L.kappa(cat(all1), cat(all2)) if games else {"action_type": float("nan"), "coverage": float("nan"),
                                                           "n_actions": 0, "n_screens": 0}
    return {"passes": [p1, p2], "pooled": pooled, "per_game": per, "threshold": 0.7, "skipped_heldout": skipped}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_session(a) -> LabelSession:
    frames = pd.read_parquet(find_frames(a.data, a.game, a.frames))
    poss = pd.read_parquet(find_possessions(a.data, a.game, a.possessions))
    return LabelSession(a.data, a.game, frames, poss, sample=a.sample, seed=a.seed,
                        labeller_pass=a.labeller_pass, label_source=a.source)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="DATA_ROOT: frames, possessions and labels/ live under it")
    ap.add_argument("--game", help="SportVU game id, e.g. 11.14.2015.BKN.at.GSW")
    ap.add_argument("--frames", help="frames parquet (default: found under DATA_ROOT)")
    ap.add_argument("--possessions", help="possessions parquet (default: found under DATA_ROOT)")
    ap.add_argument("--sample", type=int, help="label only N sampled possessions (labels.sample_possessions)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the --sample draw; the same N and seed give the same list")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port to serve on (0 picks a free one)")
    ap.add_argument("--labeller-pass", type=int, default=1, choices=list(L.LABELLER_PASSES), help="1 first pass, 2 the relabel")
    ap.add_argument("--source", default="hand", choices=list(L.LABEL_SOURCES),
                    help="label_source written on every label: hand (default) for a person at this tool")
    ap.add_argument("--export", help="write the game's JSONL labels to OUT.parquet (and OUT_screens.parquet)")
    ap.add_argument("--export-pass", default="1", help="labeller pass to export: 1 (default, what training reads), 2, or all")
    ap.add_argument("--kappa", action="store_true", help="compare labeller passes and print Cohen's kappa")
    ap.add_argument("--passes", type=int, nargs=2, default=(1, 2), metavar=("P1", "P2"),
                    help="the two labeller passes --kappa compares (default 1 2)")
    a = ap.parse_args(argv)

    if a.kappa:
        games = [a.game] if a.game else L.list_games(a.data)
        rep = kappa_report(a.data, games, a.passes)
        print(json.dumps(rep, indent=1))
        if rep["skipped_heldout"]:
            print("skipped %d held-out game(s): %s" % (len(rep["skipped_heldout"]), rep["skipped_heldout"]))
        k = rep["pooled"]
        ok = all(not (isinstance(v, float) and math.isnan(v)) and v >= 0.7 for v in (k["action_type"], k["coverage"]))
        print("kappa action_type %.3f coverage %.3f over %d actions, %d screens: %s" % (
            k["action_type"], k["coverage"], k["n_actions"], k["n_screens"], "PASS (0.7 or better)" if ok else "below 0.7"))
        return 0
    if not a.game:
        ap.error("--game is required")
    if a.export:
        ep = None if str(a.export_pass).lower() == "all" else int(a.export_pass)
        rep = export_parquet(a.data, a.game, a.export, labeller_pass=ep)
        print(json.dumps(rep, indent=1))
        return 0
    session = build_session(a)
    srv = make_server(session, a.port)
    port = srv.server_address[1]
    if session.heldout:
        print("HELD-OUT game: evaluation labels only (sportvu/splits.json); nothing here may tune a model", flush=True)
    print(f"label tool: {session.game}, {len(session.order)} possessions"
          + (f" (sample n={a.sample}, seed={a.seed})" if a.sample else "")
          + f", pass {session.labeller_pass}; open http://127.0.0.1:{port}/  (Ctrl-C stops)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
