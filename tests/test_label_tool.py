"""tests/test_label_tool.py: analytics.labels and analytics.label_tool on synthetic frames (tests/_fx_labels.py)."""
from __future__ import annotations

import http.client
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fx_labels as fx  # noqa: E402
from analytics import labels as L  # noqa: E402
from analytics import label_tool as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def game():
    return fx.synth_game(0, n_possessions=24)


@pytest.fixture(scope="module")
def labelset(game):
    return fx.synth_labels(0, game)


def _span(game, p):
    r = game["possessions"].iloc[p]
    return int(r.start_frame), int(r.end_frame), int(r.offense_team_id)


def _ids(game, p):
    s, e, off = _span(game, p)
    fr = game["frames"]
    first = fr.loc[fr["frame"] == s]
    return (first.loc[first["team_id"] == off, "player_id"].astype(int).tolist(),
            first.loc[first["team_id"] != off, "player_id"].astype(int).tolist())


# ---------------------------------------------------------------------------- schemas and validation

def test_schemas_and_enums():
    assert isinstance(L.ACTIONS_SCHEMA, pa.Schema) and isinstance(L.SCREENS_SCHEMA, pa.Schema)
    assert L.ACTIONS_SCHEMA.names[:2] == ["sportvu_game", "possession"]
    for col in ("action_type", "start_frame", "end_frame", "ball_handler_id", "screener_id", "label_source",
                "labeller_pass", "labelled_at"):
        assert col in L.ACTIONS_SCHEMA.names
    for col in ("action_id", "coverage", "on_ball_defender_id", "screener_defender_id", "label_source", "confidence",
                "labeller_pass", "labelled_at"):
        assert col in L.SCREENS_SCHEMA.names
    assert L.ACTIONS_SCHEMA.field("screener_id").nullable
    assert not L.ACTIONS_SCHEMA.field("ball_handler_id").nullable
    assert set(L.ACTION_TYPES) == {"ball_screen", "handoff", "off_ball_screen", "isolation", "post_up", "drive", "cut"}
    assert set(L.COVERAGES) == {"switch", "drop", "hedge_show", "blitz_trap", "ice", "under", "over"}
    assert set(L.LABEL_SOURCES) == {"hand", "model", "synthetic"}
    assert set(L.CONFIDENCES) == {"sure", "unsure"}


def test_validate_clean(game, labelset):
    assert L.validate_labels(labelset["actions"], labelset["screens"], game["possessions"]) == []


def test_shared_fixture_truth_labels_validate(game):
    shared = fx.shared_truth_labels(game)
    if shared is None:
        pytest.skip("lane I1's analytics.fixtures not available; local fallback in use")
    probs = L.validate_labels(shared["actions"], shared["screens"], game["possessions"])
    assert probs == [], probs[:5]
    k = L.kappa(shared, shared)
    assert k["action_type"] == 1.0 and k["coverage"] == 1.0


def test_validate_problems(game, labelset):
    acts, scrs = labelset["actions"].copy(), labelset["screens"].copy()
    s0, e0, _ = _span(game, 0)
    # unknown enums
    bad = acts.copy(); bad.loc[0, "action_type"] = "pick_and_pop"
    assert any("unknown value" in p and "pick_and_pop" in p for p in L.validate_labels(bad, scrs, game["possessions"]))
    bad = scrs.copy(); bad.loc[0, "coverage"] = "zone"
    assert any("zone" in p for p in L.validate_labels(acts, bad, game["possessions"]))
    # screen without a ball_screen action
    bad = scrs.copy(); bad.loc[0, "action_id"] = 2  # action 2 is a drive
    assert any("not a ball_screen" in p for p in L.validate_labels(acts, bad, game["possessions"]))
    # frames outside the possession span
    bad = acts.copy(); bad.loc[0, "end_frame"] = e0 + 50
    assert any("outside the possession span" in p for p in L.validate_labels(bad, scrs, game["possessions"]))
    # overlapping actions of the same type in one possession
    extra = acts.iloc[[0]].copy(); extra["action_id"] = 9; extra["start_frame"] = s0 + 150; extra["end_frame"] = s0 + 200
    probs = L.validate_labels(pd.concat([acts, extra], ignore_index=True), scrs, game["possessions"])
    assert any("overlap" in p for p in probs)
    # a different type may overlap
    extra["action_type"] = "drive"; extra["screener_id"] = None
    assert L.validate_labels(pd.concat([acts, extra], ignore_index=True), scrs, game["possessions"]) == []
    # a ball screen needs a screener
    bad = acts.copy(); bad.loc[0, "screener_id"] = None
    assert any("screener_id is required" in p for p in L.validate_labels(bad, scrs, game["possessions"]))


# ---------------------------------------------------------------------------- sampling

def test_sampling_stratified_and_training_only(game):
    poss, frames = game["possessions"], game["frames"].copy()
    # make the geometry explicit no matter which fixture produced the frames: possessions 0..5 carry a
    # planted screen, 6..11 do not, 12 and 13 never cross half court
    for p in range(12):
        s, e, off = _span(game, p)
        fx.plant_screen(frames, s, e, off, present=(p < 6))
    for p in (12, 13):
        s, e, _ = _span(game, p)
        fx.plant_no_crossing(frames, s, e)
    flags = L.possession_flags(poss, frames).set_index("possession")
    assert flags.loc[list(range(6)), "screen_candidate"].all()
    assert not flags.loc[list(range(6, 12)), "screen_candidate"].any()
    assert not flags.loc[[12, 13], "half_court"].any()

    # a held-out game is never sampled, even when it is the only one
    held = poss.copy(); held["sportvu_game"] = fx.HELDOUT_GAME
    out = L.sample_possessions(held, frames, 8, seed=0, heldout_tokens=fx.HELDOUT_TOKENS)
    assert len(out) == 0 and out.attrs["n_heldout_refused"] == len(poss)
    mixed = pd.concat([poss, held.assign(possession=held["possession"] + 1000)], ignore_index=True)
    out = L.sample_possessions(mixed, frames, 8, seed=0, heldout_tokens=fx.HELDOUT_TOKENS)
    assert set(out["sportvu_game"]) == {str(poss["sportvu_game"].iloc[0])}

    out = L.sample_possessions(poss, frames, 8, seed=3, heldout_tokens=fx.HELDOUT_TOKENS)
    assert len(out) == 8
    assert out["half_court"].all()
    assert not out["possession"].isin([12, 13]).any()
    per_team = out.groupby("offense_team_id").size()
    assert len(per_team) == 2 and per_team.max() - per_team.min() <= 1
    for tid, grp in out.groupby("offense_team_id"):
        assert grp["screen_candidate"].sum() >= len(grp) / 2
    # deterministic in the seed
    again = L.sample_possessions(poss, frames, 8, seed=3, heldout_tokens=fx.HELDOUT_TOKENS)
    assert out["possession"].tolist() == again["possession"].tolist()
    other = L.sample_possessions(poss, frames, 8, seed=4, heldout_tokens=fx.HELDOUT_TOKENS)
    assert set(other["possession"]) != set(out["possession"]) or other["possession"].tolist() != out["possession"].tolist()


def test_half_court_rule(game):
    frames = game["frames"].copy()
    s, e, _ = _span(game, 0)
    fps = L.estimate_fps(frames)
    assert 20 <= fps <= 30
    fx.plant_no_crossing(frames, s, e)
    assert not L.is_half_court(frames.loc[(frames["frame"] >= s) & (frames["frame"] <= e)], fps)
    # a crossing with a dwell of only 1 s is not a half-court possession
    m = (frames["frame"] >= s) & (frames["frame"] <= e)
    f = frames.loc[m, "frame"].to_numpy()
    frames.loc[m, "ball_x"] = np.where(f < e - int(fps), 30.0, 60.0).astype("float32")
    assert not L.is_half_court(frames.loc[m], fps)
    frames.loc[m, "ball_x"] = np.where(f < s + 2 * int(fps), 30.0, 60.0).astype("float32")
    assert L.is_half_court(frames.loc[m], fps)


# ---------------------------------------------------------------------------- JSONL round trip

def test_jsonl_round_trip(tmp_path, game, labelset):
    g = str(game["possessions"]["sportvu_game"].iloc[0])
    acts, scrs = labelset["actions"], labelset["screens"]
    rep = L.write_labels(tmp_path, g, acts, scrs)
    assert rep["actions"] == len(acts) and rep["screens"] == len(scrs)
    pa_ = L.labels_path(tmp_path, "actions", g)
    ps_ = L.labels_path(tmp_path, "screens", g)
    assert pa_ == tmp_path / "labels" / "actions" / f"{g}.jsonl" and ps_.exists()
    lines = [json.loads(x) for x in pa_.read_text().splitlines()]
    assert len(lines) == len(acts) and all("labelled_at" in r and "labeller_pass" in r for r in lines)
    a2, s2 = L.read_labels(tmp_path, g)
    assert a2["action_type"].tolist() == acts["action_type"].tolist()
    assert a2["start_frame"].tolist() == acts["start_frame"].tolist()
    assert str(a2["screener_id"].dtype) == "Int32" and pd.isna(a2["screener_id"].iloc[1])
    assert s2["coverage"].tolist() == scrs["coverage"].tolist()
    assert L.validate_labels(a2, s2, game["possessions"]) == []
    # append-only: a second write adds lines, a tombstone removes a label on read but not from the file
    L.write_labels(tmp_path, g, acts.assign(action_id=[11, 12], start_frame=acts["start_frame"] + 300,
                                          end_frame=acts["end_frame"] + 300, possession=acts["possession"] + 2), None)
    L.append_tombstone(tmp_path, g, "actions", 11)
    a3, _ = L.read_labels(tmp_path, g)
    assert a3["action_id"].tolist() == [1, 2, 12]
    assert len(pa_.read_text().splitlines()) == 5
    assert L.next_id(a3, "actions") == 13
    # parquet export carries the schema
    rep = T.export_parquet(tmp_path, g, tmp_path / "out" / "labels.parquet")
    import pyarrow.parquet as pq
    t = pq.read_table(rep["actions"])
    assert t.schema.equals(L.ACTIONS_SCHEMA, check_metadata=False) and t.num_rows == 3
    assert pq.read_table(rep["screens"]).num_rows == 1
    assert L.list_games(tmp_path) == [g]


# ---------------------------------------------------------------------------- kappa

def test_cohen_kappa_known_value():
    a = ["A", "A", "A", "B", "B", "B"]
    b = ["A", "A", "B", "B", "B", "B"]
    # po = 5/6, pe = (3/6)(2/6) + (3/6)(4/6) = 1/2, kappa = (5/6 - 1/2) / (1/2) = 2/3
    assert L.cohen_kappa(a, b) == pytest.approx(2 / 3)
    assert L.cohen_kappa(a, a) == 1.0
    assert L.cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == 0.0
    assert np.isnan(L.cohen_kappa([], []))


def test_kappa_between_passes(game):
    g = str(game["possessions"]["sportvu_game"].iloc[0])
    rows1, rows2, scr1, scr2 = [], [], [], []
    # six possessions, each with one action; pass 2 agrees on four of six action types (same frames) and
    # on two of three coverages
    types1 = ["ball_screen", "ball_screen", "ball_screen", "drive", "drive", "cut"]
    types2 = ["ball_screen", "ball_screen", "ball_screen", "drive", "cut", "cut"]
    covs1 = ["drop", "switch", "ice"]
    covs2 = ["drop", "switch", "over"]
    for p in range(6):
        s, e, _ = _span(game, p)
        off, de = _ids(game, p)
        for rows, types, k in ((rows1, types1, 1), (rows2, types2, 2)):
            rows.append({"sportvu_game": g, "possession": p, "action_id": p + 1, "action_type": types[p],
                         "start_frame": s + 10 + (5 if k == 2 else 0), "end_frame": s + 80, "ball_handler_id": off[0],
                         "screener_id": off[1] if types[p] == "ball_screen" else None, "label_source": "hand",
                         "labeller_pass": k, "labelled_at": "2026-01-01T00:00:00Z"})
        if p < 3:
            for scr, covs, k in ((scr1, covs1, 1), (scr2, covs2, 2)):
                scr.append({"sportvu_game": g, "possession": p, "screen_id": p + 1, "action_id": p + 1, "coverage": covs[p],
                            "on_ball_defender_id": de[0], "screener_defender_id": de[1], "label_source": "hand",
                            "confidence": "sure", "labeller_pass": k, "labelled_at": "2026-01-01T00:00:00Z"})
    a1, a2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    s1, s2 = pd.DataFrame(scr1), pd.DataFrame(scr2)
    k = L.kappa((a1, s1), (a2, s2))
    assert k["n_actions"] == 6 and k["n_screens"] == 3
    assert k["action_type"] == pytest.approx(L.cohen_kappa(types1, types2))
    assert k["coverage"] == pytest.approx(L.cohen_kappa(covs1, covs2))
    # an action only one pass has counts as "none" on the other side
    k2 = L.kappa((a1, s1), (a2.iloc[:5], s2))
    assert k2["n_actions"] == 6 and k2["action_type"] < k["action_type"]
    # the CLI report reads both passes from one JSONL file
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        L.write_labels(d, g, pd.concat([a1, a2.assign(action_id=a2["action_id"] + 100)], ignore_index=True),
                       pd.concat([s1, s2.assign(screen_id=s2["screen_id"] + 100, action_id=s2["action_id"] + 100)], ignore_index=True))
        rep = T.kappa_report(d, [g], (1, 2))
        assert rep["pooled"]["action_type"] == pytest.approx(k["action_type"])
        assert rep["pooled"]["coverage"] == pytest.approx(k["coverage"])
        assert rep["per_game"][g]["n_possessions"] == 6


# ---------------------------------------------------------------------------- server

@pytest.fixture(scope="module")
def server(tmp_path_factory, game):
    root = tmp_path_factory.mktemp("data")
    session = T.LabelSession(root, str(game["possessions"]["sportvu_game"].iloc[0]), game["frames"], game["possessions"],
                             sample=None, seed=0, labeller_pass=1, heldout=fx.HELDOUT_TOKENS)
    srv = T.make_server(session, 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield {"root": root, "session": session, "port": srv.server_address[1]}
    srv.shutdown()
    srv.server_close()


def _req(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return r.status, raw


def test_server_get_possession(server, game):
    port = server["port"]
    status, page = _req(port, "GET", "/")
    assert status == 200 and b"<svg" in page and b"Action Label Tool" in page
    status, d = _req(port, "GET", "/possession?index=0")
    assert status == 200
    s, e, off = _span(game, 0)
    assert d["possession"]["start_frame"] == s and d["possession"]["end_frame"] == e
    assert d["frames"][0]["frame"] == s and d["frames"][-1]["frame"] == e
    assert len(d["frames"]) == e - s + 1 and d["downsampled"] is False
    f0 = d["frames"][0]
    assert len(f0["players"]) == 10 and len(f0["ball"]) == 3 and f0["clock"] is not None
    assert len(d["players"]) == 10
    sides = {v["side"] for v in d["players"].values()}
    assert sides == {"offense", "defense"}
    assert all("name" in v and "jersey" in v and "team_id" in v for v in d["players"].values())
    assert d["labels"] == {"actions": [], "screens": []}
    assert d["meta"]["action_types"] == list(L.ACTION_TYPES)
    status, lst = _req(port, "GET", "/possessions")
    assert status == 200 and len(lst["possessions"]) == len(game["possessions"])
    assert _req(port, "GET", "/possession?index=999")[0] == 404
    assert _req(port, "GET", "/nothing")[0] == 404


def test_server_post_label_and_delete(server, game):
    port = server["port"]
    s, e, off = _span(game, 0)
    offs, defs = _ids(game, 0)
    # invalid: unknown action type
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "pick",
                                             "start_frame": s + 10, "end_frame": s + 60, "ball_handler_id": offs[0]})
    assert status == 400 and "pick" in r["error"]
    # invalid: outside the possession span
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "drive",
                                             "start_frame": s + 10, "end_frame": e + 60, "ball_handler_id": offs[0]})
    assert status == 400 and "outside" in r["error"]
    # invalid: a ball screen without a screener
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "ball_screen",
                                             "start_frame": s + 10, "end_frame": s + 60, "ball_handler_id": offs[0]})
    assert status == 400 and "screener_id" in r["error"]
    # invalid: not JSON
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/label", body=b"not json", headers={"Content-Type": "application/json"})
    assert c.getresponse().status == 400
    c.close()
    # valid action
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "ball_screen",
                                             "start_frame": s + 10, "end_frame": s + 60, "ball_handler_id": offs[0],
                                             "screener_id": offs[1]})
    assert status == 200 and r["ok"] and r["label"]["action_id"] == 1 and r["label"]["labeller_pass"] == 1
    assert r["label"]["label_source"] == "hand" and r["label"]["labelled_at"].endswith("Z")
    aid = r["label"]["action_id"]
    # screen on a drive is refused, on the ball screen accepted
    status, r = _req(port, "POST", "/label", {"kind": "screen", "action_id": 99, "coverage": "drop",
                                             "on_ball_defender_id": defs[0], "screener_defender_id": defs[1]})
    assert status == 400
    status, r = _req(port, "POST", "/label", {"kind": "screen", "action_id": aid, "coverage": "drop",
                                             "on_ball_defender_id": defs[0], "screener_defender_id": defs[1],
                                             "confidence": "unsure"})
    assert status == 200 and r["label"]["screen_id"] == 1 and r["label"]["possession"] == 0
    status, r = _req(port, "POST", "/label", {"kind": "screen", "action_id": aid, "coverage": "zone",
                                             "on_ball_defender_id": defs[0], "screener_defender_id": defs[1]})
    assert status == 400 and "zone" in r["error"]
    # the labels come back with the possession and sit in the JSONL files
    status, d = _req(port, "GET", "/possession?index=0")
    assert [a["action_id"] for a in d["labels"]["actions"]] == [aid]
    assert d["labels"]["screens"][0]["coverage"] == "drop" and d["labels"]["screens"][0]["confidence"] == "unsure"
    g = server["session"].game
    acts, scrs = L.read_labels(server["root"], g)
    assert len(acts) == 1 and len(scrs) == 1
    assert L.labels_path(server["root"], "actions", g).read_text().count("\n") == 1
    # a second action of the same type overlapping the first is refused; a later one is fine
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "ball_screen",
                                             "start_frame": s + 50, "end_frame": s + 90, "ball_handler_id": offs[0],
                                             "screener_id": offs[2]})
    assert status == 400 and "overlap" in r["error"]
    status, r = _req(port, "POST", "/label", {"kind": "action", "possession": 0, "action_type": "ball_screen",
                                             "start_frame": s + 61, "end_frame": s + 90, "ball_handler_id": offs[0],
                                             "screener_id": offs[2]})
    assert status == 200 and r["label"]["action_id"] == 2
    # delete the first action: its screen goes with it, append-only
    status, r = _req(port, "POST", "/delete", {"kind": "action", "id": aid})
    assert status == 200 and [a["action_id"] for a in r["labels"]["actions"]] == [2] and r["labels"]["screens"] == []
    acts, scrs = L.read_labels(server["root"], g)
    assert acts["action_id"].tolist() == [2] and len(scrs) == 0
    assert L.labels_path(server["root"], "actions", g).read_text().count("\n") == 3
    assert _req(port, "POST", "/delete", {"kind": "action", "id": 777})[0] == 400
    assert _req(port, "POST", "/delete", {"kind": "bogus", "id": 1})[0] == 400


def test_downsampling_over_2000_frames(game):
    fr = game["frames"]
    s, e, _ = _span(game, 0)
    span = fr.loc[(fr["frame"] >= s) & (fr["frame"] <= e)]
    n = e - s + 1
    reps = int(np.ceil(2001 / n))
    big = pd.concat([span.assign(frame=span["frame"] + k * n) for k in range(reps)], ignore_index=True)
    out, down = T.frames_payload(big, T.MAX_FRAMES)
    assert down is True and len(out) == int(np.ceil(reps * n / 2))
    assert out[1]["frame"] - out[0]["frame"] == 2
    out2, down2 = T.frames_payload(span, T.MAX_FRAMES)
    assert down2 is False and len(out2) == n


def test_sampled_session_lists_training_possessions_only(tmp_path, game):
    poss = game["possessions"]
    held = poss.copy(); held["sportvu_game"] = fx.HELDOUT_GAME
    sess = T.LabelSession(tmp_path, fx.HELDOUT_GAME, game["frames"], held, sample=6, seed=0, heldout=fx.HELDOUT_TOKENS)
    assert sess.order == [] and sess.sample_info["n_heldout_refused"] == len(poss)
    sess = T.LabelSession(tmp_path, str(poss["sportvu_game"].iloc[0]), game["frames"], poss, sample=6, seed=0,
                          heldout=fx.HELDOUT_TOKENS)
    assert len(sess.order) == 6 and sess.sample_info["n"] == 6
    lst = sess.possession_list()
    assert [x["possession"] for x in lst] == sess.order
    assert all(x["half_court"] for x in lst)


# ---------------------------------------------------------------------------- the page is self-contained

def test_html_has_no_external_resources():
    html = (ROOT / "analytics" / "label_tool.html").read_text(encoding="utf-8")
    stripped = html.replace("http://www.w3.org/", "")
    assert "http://" not in stripped and "https://" not in stripped
    assert re.search(r"<script[^>]*\ssrc=", html) is None
    assert "<link" not in html and "@import" not in html and "url(" not in html
    assert chr(0x2014) not in html
    for key in ("ArrowLeft", "ArrowRight", "Enter", "Escape", "coverage"):
        assert key in html


def test_no_em_dashes_in_owned_files():
    for rel in ("analytics/label_tool.py", "analytics/label_tool.html", "analytics/labels.py", "tests/test_label_tool.py",
                "tests/_fx_labels.py", "docs/contract/labels.md", "docs/labeling_guide.md"):
        p = ROOT / rel
        if p.exists():
            assert chr(0x2014) not in p.read_text(encoding="utf-8"), rel


# ---------------------------------------------------------------------------- review fixes

def test_read_labels_by_pass_and_export_default_pass_1(tmp_path, game, labelset):
    g = str(game["possessions"]["sportvu_game"].iloc[0])
    acts, scrs = labelset["actions"], labelset["screens"]
    L.write_labels(tmp_path, g, acts, scrs)
    L.write_labels(tmp_path, g, acts.assign(labeller_pass=2, action_id=acts["action_id"] + 10),
                   scrs.assign(labeller_pass=2, screen_id=scrs["screen_id"] + 10, action_id=scrs["action_id"] + 10))
    a_all, s_all = L.read_labels(tmp_path, g)
    assert len(a_all) == 4 and len(s_all) == 2
    a1, s1 = L.read_labels(tmp_path, g, labeller_pass=1)
    assert a1["labeller_pass"].tolist() == [1, 1] and s1["labeller_pass"].tolist() == [1]
    a2, _ = L.read_labels(tmp_path, g, labeller_pass=2)
    assert a2["action_id"].tolist() == [11, 12]
    import pyarrow.parquet as pq
    rep = T.export_parquet(tmp_path, g, tmp_path / "e" / "labels.parquet")
    assert rep["labeller_pass"] == 1 and rep["n_actions"] == 2 and rep["n_screens"] == 1
    assert pq.read_table(rep["actions"]).to_pandas()["labeller_pass"].tolist() == [1, 1]
    rep = T.export_parquet(tmp_path, g, tmp_path / "e" / "all.parquet", labeller_pass=None)
    assert rep["n_actions"] == 4 and rep["n_screens"] == 2
    # the CLI flag
    assert T.main(["--data", str(tmp_path), "--game", g, "--export", str(tmp_path / "e" / "p2.parquet"), "--export-pass", "2"]) == 0
    assert pq.read_table(tmp_path / "e" / "p2.parquet").num_rows == 2
    assert T.main(["--data", str(tmp_path), "--game", g, "--export", str(tmp_path / "e" / "pa.parquet"), "--export-pass", "all"]) == 0
    assert pq.read_table(tmp_path / "e" / "pa.parquet").num_rows == 4


def test_read_jsonl_without_ids(tmp_path, game, labelset):
    g = "noid.game"
    acts, scrs = labelset["actions"].copy(), labelset["screens"].copy()
    pa_ = L.labels_path(tmp_path, "actions", g)
    pa_.parent.mkdir(parents=True)
    rows = [{k: (None if pd.isna(v) else (int(v) if hasattr(v, "item") else v)) for k, v in r.items() if k != "action_id"}
            for r in acts.to_dict("records")]
    pa_.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ps_ = L.labels_path(tmp_path, "screens", g)
    ps_.parent.mkdir(parents=True)
    srows = [{k: (int(v) if hasattr(v, "item") else v) for k, v in r.items() if k != "screen_id"} for r in scrs.to_dict("records")]
    srows[0]["action_id"] = 0  # a fixture root references actions by 0-based position
    ps_.write_text(json.dumps(srows[0]) + "\n")
    a, s = L.read_labels(tmp_path, g)
    assert a["action_id"].tolist() == [0, 1] and s["screen_id"].tolist() == [0]
    assert L.validate_labels(a, s, game["possessions"]) == []
    # a tombstone still needs an id
    with pa_.open("a") as fh:
        fh.write(json.dumps({"deleted": True, "labelled_at": "2026-01-01T00:00:00Z"}) + "\n")
    with pytest.raises(ValueError, match="tombstone has no action_id"):
        L.read_labels(tmp_path, g)


def _two_pass_root(root, game, visited=True, unsure_second=False):
    """Pass 1 labels possessions 0..3 (ball_screen, ball_screen, drive, cut), pass 2 labels 0..2 the same
    and finds nothing in possession 3; both passes visit 0..3, pass 1 also visits 4 with a drive."""
    g = str(game["possessions"]["sportvu_game"].iloc[0])
    types = ["ball_screen", "ball_screen", "drive", "cut", "drive"]
    rows, scr = [], []
    for k, n in ((1, 5), (2, 3)):
        for p in range(n):
            s, e, _ = _span(game, p)
            off, de = _ids(game, p)
            rows.append({"sportvu_game": g, "possession": p, "action_id": p + 1 + 100 * k, "action_type": types[p],
                         "start_frame": s + 10, "end_frame": s + 60, "ball_handler_id": off[0],
                         "screener_id": off[1] if types[p] == "ball_screen" else None, "label_source": "hand",
                         "labeller_pass": k, "labelled_at": "2026-01-01T00:00:00Z"})
            if types[p] == "ball_screen":
                scr.append({"sportvu_game": g, "possession": p, "screen_id": p + 1 + 100 * k, "action_id": p + 1 + 100 * k,
                            "coverage": "drop" if p == 0 else "switch", "on_ball_defender_id": de[0],
                            "screener_defender_id": de[1], "label_source": "hand",
                            "confidence": "unsure" if (unsure_second and k == 2 and p == 1) else "sure",
                            "labeller_pass": k, "labelled_at": "2026-01-01T00:00:00Z"})
    L.write_labels(root, g, pd.DataFrame(rows), pd.DataFrame(scr))
    if visited:
        for p in range(5):
            L.mark_visited(root, g, p, 1)
        for p in range(4):
            L.mark_visited(root, g, p, 2)
    return g


def test_kappa_report_uses_visited_record(tmp_path, game):
    g = _two_pass_root(tmp_path, game)
    v = L.read_visited(tmp_path, g)
    assert len(v) == 9 and set(v.columns) == set(L.VISITED_COLUMNS)
    rep = T.kappa_report(tmp_path, [g], (1, 2))
    per = rep["per_game"][g]
    # possessions 0..3 are compared (4 was visited by pass 1 only); possession 3 is (cut, none)
    assert per["n_possessions"] == 4 and per["n_actions"] == 4
    assert per["visited_source"] == {"1": "visited", "2": "visited"}
    assert per["action_type"] == pytest.approx(L.cohen_kappa(["ball_screen", "ball_screen", "drive", "cut"],
                                                             ["ball_screen", "ball_screen", "drive", "none"]))
    assert per["coverage"] == 1.0 and per["n_screens"] == 2
    # without a visited record the labelled possessions stand in (and possession 3 would never enter)
    g2 = _two_pass_root(tmp_path / "novisit", game, visited=False)
    per2 = T.kappa_report(tmp_path / "novisit", [g2], (1, 2))["per_game"][g2]
    assert per2["visited_source"] == {"1": "labels", "2": "labels"} and per2["n_possessions"] == 3
    assert per2["action_type"] == 1.0


def test_kappa_skips_heldout_games(tmp_path, game):
    g = _two_pass_root(tmp_path, game)
    held = fx.HELDOUT_GAME
    acts, scrs = L.read_labels(tmp_path, g)
    L.write_labels(tmp_path, held, acts.assign(sportvu_game=held), scrs.assign(sportvu_game=held))
    for p in range(4):
        L.mark_visited(tmp_path, held, p, 1); L.mark_visited(tmp_path, held, p, 2)
    assert L.list_games(tmp_path) == sorted([held, g])
    rep = T.kappa_report(tmp_path, L.list_games(tmp_path), (1, 2), heldout=fx.HELDOUT_TOKENS)
    assert rep["skipped_heldout"] == [held] and list(rep["per_game"]) == [g]
    assert rep["pooled"]["n_actions"] == 4
    rep_all = T.kappa_report(tmp_path, L.list_games(tmp_path), (1, 2), heldout=())
    assert rep_all["pooled"]["n_actions"] == 8


def test_kappa_report_default_guard(tmp_path, game, monkeypatch):
    g = _two_pass_root(tmp_path, game)
    held = fx.HELDOUT_GAME
    acts, scrs = L.read_labels(tmp_path, g)
    L.write_labels(tmp_path, held, acts.assign(sportvu_game=held), scrs.assign(sportvu_game=held))
    for p in range(4):
        L.mark_visited(tmp_path, held, p, 1); L.mark_visited(tmp_path, held, p, 2)
    monkeypatch.setattr(T, "heldout_tokens", lambda: set(fx.HELDOUT_TOKENS))
    rep = T.kappa_report(tmp_path, L.list_games(tmp_path), (1, 2))  # no heldout argument
    assert rep["skipped_heldout"] == [held] and list(rep["per_game"]) == [g]


def test_kappa_coverage_sure_only(tmp_path, game):
    g = _two_pass_root(tmp_path, game, unsure_second=True)
    per = T.kappa_report(tmp_path, [g], (1, 2))["per_game"][g]
    # pass 2 marked possession 1's coverage unsure: that screen pairs as (switch, none), not (switch, switch)
    assert per["n_screens"] == 2
    assert per["coverage"] == pytest.approx(L.cohen_kappa(["drop", "switch"], ["drop", "none"]))
    a1, s1 = L.read_labels(tmp_path, g, labeller_pass=1)
    a2, s2 = L.read_labels(tmp_path, g, labeller_pass=2)
    assert L.kappa((a1, s1), (a2, s2))["coverage"] == per["coverage"]


def test_heldout_tokens_fail_closed(monkeypatch):
    from sportvu import splits
    monkeypatch.setattr(splits, "heldout_tokens", lambda *a, **k: (_ for _ in ()).throw(SystemExit("no splits")))
    with pytest.raises(SystemExit):
        T.heldout_tokens()
    assert fx.HELDOUT_GAME not in T.heldout_tokens.__doc__ and "SystemExit" in T.heldout_tokens.__doc__


def test_heldout_game_notice(tmp_path, game):
    poss = game["possessions"].copy(); poss["sportvu_game"] = fx.HELDOUT_GAME
    sess = T.LabelSession(tmp_path, fx.HELDOUT_GAME, game["frames"], poss, heldout=fx.HELDOUT_TOKENS)
    assert sess.heldout is True and len(sess.order) == len(poss)
    srv = T.make_server(sess, 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    try:
        status, d = _req(srv.server_address[1], "GET", "/possessions")
        assert status == 200 and d["heldout"] is True and d["notice"].startswith("HELD-OUT")
        status, d = _req(srv.server_address[1], "GET", "/possession?index=0")
        assert d["meta"]["heldout"] is True
    finally:
        srv.shutdown(); srv.server_close()
    train = T.LabelSession(tmp_path, str(game["possessions"]["sportvu_game"].iloc[0]), game["frames"],
                           game["possessions"], heldout=fx.HELDOUT_TOKENS)
    assert train.heldout is False


def test_post_guards(server):
    port = server["port"]
    # no JSON content type
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/label", body=b"{}", headers={"Content-Type": "text/plain"})
    assert c.getresponse().status == 415
    c.close()
    # bad Content-Length
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest("POST", "/label"); c.putheader("Content-Type", "application/json"); c.putheader("Content-Length", "abc")
    c.endheaders()
    assert c.getresponse().status == 400
    c.close()
    # body over 1 MiB is refused before it is read
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest("POST", "/label"); c.putheader("Content-Type", "application/json")
    c.putheader("Content-Length", str(T.MAX_BODY + 1)); c.endheaders()
    assert c.getresponse().status == 413
    c.close()
    # a foreign Host header is refused
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/delete", body=b"{}", headers={"Content-Type": "application/json", "Host": "evil.example:80"})
    assert c.getresponse().status == 403
    c.close()
    for host in ("127.0.0.1:%d" % port, "localhost", "[::1]:8765"):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/delete", body=b'{"kind": "bogus"}', headers={"Content-Type": "application/json", "Host": host})
        assert c.getresponse().status == 400, host
        c.close()


def test_visited_endpoint(server, game):
    port = server["port"]
    status, r = _req(port, "POST", "/visited", {"possession": 1})
    assert status == 200 and r["visited"] == [1] and r["labeller_pass"] == 1
    assert _req(port, "POST", "/visited", {"possession": 10 ** 6})[0] == 400
    assert _req(port, "POST", "/visited", {})[0] == 400
    status, lst = _req(port, "GET", "/possessions")
    flags = {x["possession"]: x["visited"] for x in lst["possessions"]}
    assert flags[1] is True and flags[0] is False
    status, d = _req(port, "GET", "/possession?index=1")
    assert d["visited"] is True
    v = L.read_visited(server["root"], server["session"].game)
    assert v["possession"].tolist() == [1] and v["labeller_pass"].tolist() == [1]


def test_html_key_guards():
    html = (ROOT / "analytics" / "label_tool.html").read_text(encoding="utf-8")
    assert "if (ev.ctrlKey || ev.metaKey || ev.altKey) return;" in html
    i = html.index("function enterCoverage()")
    assert 'if (!S.data)' in html[i:i + 200]
    assert '"/visited"' in html
    j = html.index("function nextPossession(k)")
    body = html[j:j + 500]
    assert body.index("markVisited()") < body.index("i >= S.n"), "the bounds check must follow the visited mark"


def test_sampling_redistributes_short_team(game):
    poss, frames = game["possessions"].copy(), game["frames"].copy()
    teams = sorted(int(t) for t in poss["offense_team_id"].dropna().unique())
    assert len(teams) == 2
    flags = L.possession_flags(poss, frames)
    elig = flags.loc[flags["half_court"]]
    per_team = elig.groupby("offense_team_id").size()
    small = int(per_team.idxmin())
    # keep only one eligible possession of the small team
    keep_small = elig.loc[elig["offense_team_id"] == small, "possession"].iloc[:1].tolist()
    drop = elig.loc[(elig["offense_team_id"] == small) & (~elig["possession"].isin(keep_small)), "possession"]
    poss2 = poss.loc[~poss["possession"].isin(drop)]
    n_big = int(per_team.max())
    out = L.sample_possessions(poss2, frames, 6, seed=0, heldout_tokens=fx.HELDOUT_TOKENS)
    assert len(out) == 6 and out.attrs["n_short"] == 0
    assert (out["offense_team_id"] == small).sum() == 1
    out = L.sample_possessions(poss2, frames, n_big + 5, seed=0, heldout_tokens=fx.HELDOUT_TOKENS)
    assert len(out) == n_big + 1 and out.attrs["n_short"] == 4
    # precomputed flags give the same draw as computing them inside
    a = L.sample_possessions(poss2, frames, 6, seed=1, heldout_tokens=fx.HELDOUT_TOKENS)
    b = L.sample_possessions(poss2, frames, 6, seed=1, heldout_tokens=fx.HELDOUT_TOKENS, flags=flags)
    assert a["possession"].tolist() == b["possession"].tolist()


def test_validate_sides_and_pass(game, labelset, server):
    acts, scrs = labelset["actions"].copy(), labelset["screens"].copy()
    poss, frames = game["possessions"], game["frames"]
    off0, def0 = _ids(game, 0)
    assert L.validate_labels(acts, scrs, poss, frames) == []
    bad = acts.copy(); bad.loc[0, "ball_handler_id"] = def0[0]
    assert any("ball_handler_id" in p and "on the defence" in p for p in L.validate_labels(bad, scrs, poss, frames))
    bad = acts.copy(); bad.loc[0, "screener_id"] = def0[1]
    assert any("screener_id" in p and "on the defence" in p for p in L.validate_labels(bad, scrs, poss, frames))
    bad = scrs.copy(); bad.loc[0, "on_ball_defender_id"] = off0[2]
    assert any("on_ball_defender_id" in p and "on the offence" in p for p in L.validate_labels(acts, bad, poss, frames))
    bad = acts.copy(); bad.loc[0, "ball_handler_id"] = 424242
    assert any("not in the frames" in p for p in L.validate_labels(bad, scrs, poss, frames))
    bad = acts.copy(); bad.loc[0, "labeller_pass"] = 3
    assert any("labeller_pass" in p for p in L.validate_labels(bad, scrs, poss))
    # the server passes its frames: a defender as handler is a 400
    s, e, _ = _span(game, 2)
    off2, def2 = _ids(game, 2)
    status, r = _req(server["port"], "POST", "/label", {"kind": "action", "possession": 2, "action_type": "drive",
                                                       "start_frame": s + 10, "end_frame": s + 40, "ball_handler_id": def2[0]})
    assert status == 400 and "defence" in r["error"]


def test_cli_help_and_docs():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        T.main(["--help"])
    h = " ".join(buf.getvalue().split())  # argparse wraps help text
    for flag, text in (("--seed", "same N and seed"), ("--port", "free one"), ("--source", "label_source"),
                       ("--passes", "compares"), ("--export-pass", "what training reads")):
        assert flag in h and text in h, flag
    md = (ROOT / "docs" / "contract" / "labels.md").read_text(encoding="utf-8")
    for text in ("read_labels", "labeller_pass == 1", "`hand`", '"deleted"', "last line per id", "visited",
                 "spends at least 2 s", "unsure"):
        assert text in md, text
    assert "exactly the columns above" not in md
    guide = (ROOT / "docs" / "labeling_guide.md").read_text(encoding="utf-8")
    assert "spends at least 2 s" in guide and "sure" in guide and "HELD-OUT" in guide
