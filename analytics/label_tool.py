"""analytics/label_tool.py: local hand-labeling tool for actions and ball-screen coverages.

    python -m analytics.label_tool --data DATA_ROOT --game G [--sample N --seed 0] [--port 8765]
    python -m analytics.label_tool --data DATA_ROOT --game G --export OUT.parquet
    python -m analytics.label_tool --data DATA_ROOT [--game G] --kappa [--passes 1 2]
    python -m analytics.label_tool --data DATA_ROOT --audit SAMPLE.parquet [--port 8765] [--no-backup]

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

Audit mode (``--audit``, stage 3 of docs/mac_runbook.md) serves analytics/label_tool_audit.html instead:
the analytics.audit_shots sample in sample order across games, each game's frames loaded on demand, the
court opening at the detected release frame, and three endpoints, GET /audit/state, GET /audit/shot?index=i
and POST /audit/save, that write audited_release_frame, audited_ok and auditor_note straight back into the
sample parquet after every verdict (``AuditSession``; semantics in docs/contract/phase0_driver.md). It
never touches labels/, and it refuses a sample holding a held-out game.

Inputs are read at run time and never copied anywhere: the section 4 frames parquet (``--frames``, or
DATA_ROOT/<game>.parquet, DATA_ROOT/frames/<game>.parquet, DATA_ROOT/sportvu_contract/<game>.parquet) and
the section 19 possessions parquet (``--possessions``, or any <game>.parquet under DATA_ROOT/analytics/
possessions or DATA_ROOT/possessions). Labels go to DATA_ROOT/labels/{actions,screens}/<game>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

def request_host(handler) -> str:
    """The Host header's host part ('' when absent): '127.0.0.1', 'localhost', '[::1]', ..."""
    h = handler.headers.get("Host") or ""
    return h.split("]")[0] + "]" if h.startswith("[") else h.rsplit(":", 1)[0]


def read_post_json(handler) -> tuple[dict | None, tuple[int, dict] | None]:
    """(body, None) for a valid local JSON POST, else (None, (status, error)): the Host must be
    127.0.0.1 or localhost (403), the Content-Type application/json (415), the Content-Length a number
    (400) of at most MAX_BODY bytes (413), and the body a JSON object (400)."""
    host = request_host(handler)
    if host and host not in LOCAL_HOSTS:
        return None, (403, {"error": f"POST accepted from 127.0.0.1 or localhost only, not {host!r}"})
    ctype = (handler.headers.get("Content-Type") or "").strip().lower()
    if not ctype.startswith("application/json"):
        return None, (415, {"error": "Content-Type must be application/json"})
    try:
        n = int(handler.headers.get("Content-Length") or 0)
        if n < 0:
            raise ValueError
    except (TypeError, ValueError):
        return None, (400, {"error": "bad Content-Length"})
    if n > MAX_BODY:
        return None, (413, {"error": f"body over {MAX_BODY} bytes"})
    raw = handler.rfile.read(n) if n else b""
    try:
        body = json.loads(raw.decode("utf-8")) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, (400, {"error": "body is not JSON"})
    if not isinstance(body, dict):
        return None, (400, {"error": "body must be a JSON object"})
    return body, None


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
            body, err = read_post_json(self)
            if err is not None:
                return self._json(*err)
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
# Audit mode: walk the analytics.audit_shots sample shot by shot
# --------------------------------------------------------------------------------------

AUDIT_HTML_PATH = Path(__file__).with_name("label_tool_audit.html")
AUDIT_CONTEXT_FRAMES = 100        # frames served either side of the detected release (4 s at 25 Hz)
AUDIT_UNFOUND_CLOCK_S = (2.5, 4.0)  # no detected release: clock_s - 2.5 .. clock_s + 4.0 of the period
AUDIT_NOTE_MAX = 500
AUDIT_VERDICTS = ("accept", "release", "bad")
AUDIT_FRAME_COLUMNS = ["frame", "period", "game_clock_s", "shot_clock_s", "ball_x", "ball_y", "ball_z",
                       "player_id", "player", "jersey", "team_id", "team", "home", "x", "y"]
AUDIT_CACHE_GAMES = 2
_AUDIT_DTYPE = {"audited_release_frame": "Int32", "audited_ok": "boolean", "auditor_note": "string"}


def _tolerance() -> int:
    from analytics import audit_shots as A
    return int(A.TOLERANCE_FRAMES)


class HeldoutInSample(ValueError):
    """The sample holds a held-out game: the audit validates the localiser constants, so it refuses."""


def load_audit_frames(data_root, game: str) -> pd.DataFrame:
    """The section 4 frames of ``game`` (find_frames under DATA_ROOT), only the columns the audit draws."""
    import pyarrow.parquet as pq
    path = find_frames(data_root, game)
    have = set(pq.read_schema(str(path)).names)
    return pd.read_parquet(path, columns=[c for c in AUDIT_FRAME_COLUMNS if c in have])


def default_heldout_check(tokens: set):
    """game -> held out, by the label tool's token rule or by analytics.audit_shots' split rule (which
    also holds out every game at the held-out arena). Both read sportvu/splits.json and fail closed."""
    from analytics import audit_shots as A
    return lambda g: L.is_heldout(g, tokens) or A._is_heldout_game(g)


class AuditSession:
    """The analytics.audit_shots sample in memory, its rows served one shot at a time, every verdict
    written straight back into the sample parquet (atomic: a temporary file next to it, then os.replace).

    Semantics of the three audit columns (what ``audit_shots --summarise`` reads):
      accept   audited_release_frame = release_frame, audited_ok = True  (the detected frame is right)
      release  audited_release_frame = the frame the auditor marked, audited_ok = True  (a real shot,
               released at that frame; the localiser's frame may be off)
      bad      audited_release_frame = <NA>, audited_ok = False  (no release to compare: not a shot, the
               ball or shooter cannot be followed, or the frames are broken)
    so a row has a frame exactly when audited_ok is True, --summarise's agreement covers the confirmed
    shots and share_audited_ok the share of rows that are real, placeable shots. A row is audited when
    either column is filled; the session resumes at the first row with both empty.
    """

    def __init__(self, data_root, sample_path, frames_loader=None, heldout: set | None = None,
                 heldout_check=None, backup: bool = True):
        import pyarrow.parquet as pq
        from analytics import audit_shots as A
        self.data_root = Path(data_root)
        self.sample_path = Path(sample_path).resolve()
        self.schema = pq.read_schema(str(self.sample_path))
        df = pd.read_parquet(self.sample_path)
        missing = [c for c in A.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self.sample_path} is not an audit_shots sample: missing columns {missing}")
        for c, dt in _AUDIT_DTYPE.items():
            if str(df[c].dtype) != dt:
                df[c] = df[c].astype(dt)
        self.df = df.reset_index(drop=True)
        check = heldout_check or default_heldout_check(set(heldout) if heldout is not None else heldout_tokens())
        held = sorted({g for g in self.df["sportvu_game"].astype(str).unique() if check(g)})
        if held:
            raise HeldoutInSample(
                "the sample holds held-out game(s) %s: the shot audit validates the localiser constants, and "
                "held-out games select nothing (docs/mac_runbook.md rule 1); draw the sample with "
                "analytics.audit_shots, which takes training games only" % held)
        self.frames_loader = frames_loader or (lambda g: load_audit_frames(self.data_root, g))
        self.backup = bool(backup)
        self.backup_path = self.sample_path.with_name(self.sample_path.name + ".bak")
        self._backed_up = False
        self._cache: dict = {}
        self.lock = threading.Lock()
        self._stamp = self._file_stamp()

    # -- state ----------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.df)

    def audited_mask(self) -> pd.Series:
        return self.df["audited_release_frame"].notna() | self.df["audited_ok"].notna()

    def first_unaudited(self) -> int | None:
        idx = np.flatnonzero(~self.audited_mask().to_numpy())
        return int(idx[0]) if len(idx) else None

    def state(self) -> dict:
        return {"sample": self.sample_path.name, "n": self.n, "n_audited": int(self.audited_mask().sum()),
                "first_unaudited": self.first_unaudited(), "verdicts": list(AUDIT_VERDICTS),
                "note_max": AUDIT_NOTE_MAX, "tolerance_frames": _tolerance(),
                "heldout": "refused: the sample holds training games only"}

    def audit_of(self, i: int) -> dict:
        r = self.df.iloc[i]
        return {"audited_release_frame": _i(r["audited_release_frame"]),
                "audited_ok": None if pd.isna(r["audited_ok"]) else bool(r["audited_ok"]),
                "auditor_note": None if pd.isna(r["auditor_note"]) else str(r["auditor_note"])}

    # -- frames ---------------------------------------------------------------------------
    def game_frames(self, game: str) -> pd.DataFrame:
        if game not in self._cache:
            fr = self.frames_loader(game).sort_values("frame", kind="stable").reset_index(drop=True)
            while len(self._cache) >= AUDIT_CACHE_GAMES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[game] = fr
        return self._cache[game]

    def _row_index(self, index) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("index must be an integer")
        if index < 0 or index >= self.n:
            raise IndexError(index)
        return int(index)

    def shot_payload(self, index) -> dict:
        i = self._row_index(index)
        r = self.df.iloc[i]
        game, period = str(r["sportvu_game"]), int(r["period"])
        fr = self.game_frames(game)
        per = fr.loc[fr["period"] == period]
        rel = int(r["release_frame"])
        if rel >= 0:
            span = per.loc[(per["frame"] >= rel - AUDIT_CONTEXT_FRAMES) & (per["frame"] <= rel + AUDIT_CONTEXT_FRAMES)]
        else:
            c = float(r["clock_s"])
            lo, hi = AUDIT_UNFOUND_CLOCK_S
            span = per.loc[(per["game_clock_s"] >= c - lo) & (per["game_clock_s"] <= c + hi)]
        frames_out, downsampled = frames_payload(span, MAX_FRAMES)
        shooter = _i(r["shooter_id"])
        pl = fr.drop_duplicates("player_id")
        team_of = {int(p): int(t) for p, t in zip(pl["player_id"], pl["team_id"])}
        on_floor = set(int(v) for v in span["player_id"].unique())
        players = {}
        for p in pl.itertuples(index=False):
            pid = int(p.player_id)
            if pid not in on_floor:
                continue
            players[str(pid)] = {"id": pid, "name": str(getattr(p, "player", pid)),
                                 "jersey": None if pd.isna(getattr(p, "jersey", None)) else str(getattr(p, "jersey", "")),
                                 "team_id": int(p.team_id), "team": str(getattr(p, "team", "")),
                                 "home": bool(getattr(p, "home", False)),
                                 "side": "offense" if shooter is not None and team_of.get(shooter) == int(p.team_id) else "defense"}
        frame_ids = [f["frame"] for f in frames_out]
        open_frame = rel if rel in frame_ids else (frame_ids[len(frame_ids) // 2] if frame_ids else None)
        # the attacked basket: the ball's side at the apex (else at the last served frame)
        ref = int(r["apex_frame"]) if int(r["apex_frame"]) in frame_ids else (frame_ids[-1] if frame_ids else None)
        bx = next((f["ball"][0] for f in frames_out if f["frame"] == ref), None)
        sh = players.get(str(shooter)) if shooter is not None else None
        return {"index": i, "n": self.n, "game": game, "event": int(r["event"]), "period": period,
                "clock_s": _f(r["clock_s"]), "shooter_id": shooter,
                "shooter": sh or ({"id": shooter, "name": str(shooter)} if shooter is not None else None),
                "confidence_flag": None if pd.isna(r["confidence_flag"]) else str(r["confidence_flag"]),
                "release_frame": rel, "apex_frame": int(r["apex_frame"]), "outcome_frame": int(r["outcome_frame"]),
                "open_frame": open_frame, "attack_right": True if bx is None else bool(bx > 47.0),
                "fps": L.estimate_fps(per) if len(per) > 1 else 25.0, "downsampled": downsampled,
                "frames": frames_out, "players": players, "audit": self.audit_of(i),
                "n_audited": int(self.audited_mask().sum()), "first_unaudited": self.first_unaudited()}

    # -- verdicts -------------------------------------------------------------------------
    def _check_frame(self, r: pd.Series, frame) -> int:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError("audited_release_frame must be an integer frame")
        frame = int(frame)
        fr = self.game_frames(str(r["sportvu_game"]))
        fv = fr["frame"].to_numpy()
        if len(fv) == 0 or frame < int(fv[0]) or frame > int(fv[-1]):
            raise ValueError(f"frame {frame} is outside the game's frames {int(fv[0]) if len(fv) else None} to "
                             f"{int(fv[-1]) if len(fv) else None}")
        k = np.searchsorted(fv, frame)
        if k >= len(fv) or int(fv[k]) != frame:
            raise ValueError(f"frame {frame} is not in the game's frames")
        if int(fr["period"].iloc[k]) != int(r["period"]):
            raise ValueError(f"frame {frame} is in period {int(fr['period'].iloc[k])}, the shot in period {int(r['period'])}")
        return frame

    def save(self, body: dict) -> dict:
        """Apply one verdict ({"index", "verdict", "audited_release_frame" for release, "auditor_note"})
        and write the sample. ValueError on any invalid input, before anything is written."""
        if "index" not in body:
            raise ValueError("index is required")
        try:
            i = self._row_index(body["index"])
        except IndexError:
            raise ValueError(f"no shot at index {body['index']}; n = {self.n}")
        verdict = body.get("verdict")
        if verdict not in AUDIT_VERDICTS:
            raise ValueError(f"verdict must be one of {list(AUDIT_VERDICTS)}")
        note = body.get("auditor_note")
        if note is not None:
            if not isinstance(note, str):
                raise ValueError("auditor_note must be a string or null")
            note = " ".join(note.split())
            if len(note) > AUDIT_NOTE_MAX:
                raise ValueError(f"auditor_note is over {AUDIT_NOTE_MAX} characters")
            note = note or None
        r = self.df.iloc[i]
        frame = body.get("audited_release_frame")
        if verdict == "accept":
            if frame is not None:
                raise ValueError("accept takes the detected release frame; send verdict release to set another")
            if int(r["release_frame"]) < 0:
                raise ValueError("this shot has no detected release frame to accept: mark the frame or call it bad")
            frame, ok = self._check_frame(r, int(r["release_frame"])), True
        elif verdict == "release":
            if frame is None:
                raise ValueError("verdict release needs audited_release_frame")
            frame, ok = self._check_frame(r, frame), True
        else:
            if frame is not None:
                raise ValueError("verdict bad carries no release frame (audited_release_frame stays empty)")
            ok = False
        with self.lock:
            new = self.df.copy()
            new.loc[i, "audited_release_frame"] = pd.NA if frame is None else frame
            new.loc[i, "audited_ok"] = ok
            new.loc[i, "auditor_note"] = pd.NA if note is None else note
            self._write(new)
            self.df = new
        return {"ok": True, "index": i, "verdict": verdict, "audit": self.audit_of(i),
                "n_audited": int(self.audited_mask().sum()), "first_unaudited": self.first_unaudited()}

    # -- the file -------------------------------------------------------------------------
    def _file_stamp(self):
        st = self.sample_path.stat()
        return (st.st_mtime_ns, st.st_size)

    def _schema(self):
        """The sample's own arrow schema (so every other column keeps its type and the pandas metadata
        stays), with an audit column whose type is not the contract's (say a CSV round trip left it a
        null or double column) replaced by int32 / bool / large_string."""
        import pyarrow as pa
        allowed = {"audited_release_frame": (pa.int32(),), "audited_ok": (pa.bool_(),),
                   "auditor_note": (pa.string(), pa.large_string())}
        fields = []
        for f in self.schema:
            if f.name in allowed and f.type not in allowed[f.name]:
                f = pa.field(f.name, allowed[f.name][-1], nullable=True)
            fields.append(f)
        return pa.schema(fields, metadata=self.schema.metadata)

    def _write(self, df: pd.DataFrame) -> None:
        """Atomic: write a temporary file next to the sample, fsync it, os.replace it over the sample.
        Refuses when the sample changed on disk since this session last read or wrote it."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        if self._file_stamp() != self._stamp:
            raise RuntimeError(f"{self.sample_path} changed on disk since this session opened it; restart the "
                               "audit so nothing written elsewhere is overwritten")
        if self.backup and not self._backed_up:
            import shutil
            shutil.copy2(self.sample_path, self.backup_path)
            self._backed_up = True
        schema = self._schema()
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        tmp = self.sample_path.with_name(".%s.%d.tmp" % (self.sample_path.name, os.getpid()))
        try:
            pq.write_table(table, str(tmp))
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, self.sample_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self.schema = schema
        self._stamp = self._file_stamp()


def make_audit_handler(session: AuditSession, html_path: Path = AUDIT_HTML_PATH):
    """GET / (the audit page), /audit/state, /audit/shot?index=i; POST /audit/save. Every request must
    come with a local Host header; nothing else is served, so audit mode can never write labels."""
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
            host = request_host(self)
            if host and host not in LOCAL_HOSTS:
                return self._json(403, {"error": f"served to 127.0.0.1 or localhost only, not {host!r}"})
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/audit/state":
                    self._json(200, session.state())
                elif u.path == "/audit/shot":
                    try:
                        i = int(q.get("index", ["0"])[0])
                    except ValueError:
                        return self._json(400, {"error": "index must be an integer"})
                    try:
                        self._json(200, session.shot_payload(i))
                    except IndexError:
                        self._json(404, {"error": f"no shot at index {i}; n = {session.n}"})
                else:
                    self._json(404, {"error": "not found"})
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

        def do_POST(self):
            u = urlparse(self.path)
            body, err = read_post_json(self)
            if err is not None:
                return self._json(*err)
            try:
                if u.path == "/audit/save":
                    self._json(200, session.save(body))
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._json(404, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

    return Handler


def make_audit_server(session: AuditSession, port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_audit_handler(session))
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


def run_audit(a) -> int:
    try:
        session = AuditSession(a.data, a.audit, backup=not a.no_backup)
    except HeldoutInSample as exc:
        print("refused: %s" % exc, file=sys.stderr, flush=True)
        return 2
    srv = make_audit_server(session, a.port)
    port = srv.server_address[1]
    st = session.state()
    start = st["first_unaudited"]
    print(f"shot audit: {session.sample_path} ({st['n']} shots, {st['n_audited']} audited; "
          + (f"resuming at shot {start + 1})" if start is not None else "all audited)")
          + f"; open http://127.0.0.1:{port}/  (Ctrl-C stops)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


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
    ap.add_argument("--audit", metavar="SAMPLE",
                    help="shot audit mode: walk this analytics.audit_shots sample parquet shot by shot and write "
                         "the verdicts straight back into it (frames from DATA_ROOT; no labels are written)")
    ap.add_argument("--no-backup", action="store_true",
                    help="with --audit: skip the copy SAMPLE.bak made before the first write of a session")
    a = ap.parse_args(argv)

    if a.audit:
        return run_audit(a)

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
