"""tests/test_label_tool_audit.py: the shot audit mode of analytics.label_tool (``--audit SAMPLE``).

A synthetic DATA_ROOT with two training games (frames under sportvu_contract/, shot_frames tables under
analytics/shot_frames/<season>/), a sample drawn by analytics.audit_shots, then the audit server driven over
HTTP on an ephemeral 127.0.0.1 port: land on the detected release, accept, move the frame, call a shot bad,
resume after a restart, refuse bad input, keep every other column and dtype, and finally run
``audit_shots --summarise`` on the edited parquet.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import re
import shutil
import sys
import threading
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fx_labels as fx  # noqa: E402
from analytics import audit_shots as A  # noqa: E402
from analytics import label_tool as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "analytics" / "label_tool_audit.html"
N_PER_GAME = 4
TOKENS = set(fx.HELDOUT_TOKENS)


# ---------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """DATA_ROOT with two training games and the pristine sample audit_shots draws from them."""
    root = tmp_path_factory.mktemp("audit_root")
    games = []
    for seed in (0, 1):
        g = fx.synth_game(seed, n_possessions=12)
        assert "shot_frames" in g, "the shared fixture (analytics.fixtures) must provide shot_frames"
        name = str(g["frames"]["game"].iloc[0])
        (root / "sportvu_contract").mkdir(exist_ok=True)
        g["frames"].to_parquet(root / "sportvu_contract" / f"{name}.parquet", index=False)
        sf = root / "analytics" / "shot_frames" / "2015-16"
        sf.mkdir(parents=True, exist_ok=True)
        g["shot_frames"].to_parquet(sf / f"{name}.parquet", index=False)
        games.append({"name": name, "frames": g["frames"]})
    pristine = root / "pristine.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = A.main(["--data", str(root), "--n", str(2 * N_PER_GAME), "--seed", "0", "--out", str(pristine)])
    assert rc == 0
    return {"root": root, "pristine": pristine, "games": {g["name"]: g["frames"] for g in games}}


@pytest.fixture
def sample(data_root, tmp_path):
    """A fresh copy of the drawn sample per test, in its own folder (the backup lands next to it)."""
    p = tmp_path / "audit" / "shots_sample.parquet"
    p.parent.mkdir()
    shutil.copy2(data_root["pristine"], p)
    return p


class Server:
    def __init__(self, data_root, sample_path, **kw):
        kw.setdefault("heldout", TOKENS)
        self.session = T.AuditSession(data_root, sample_path, **kw)
        self.srv = T.make_audit_server(self.session, 0)
        self.port = self.srv.server_address[1]
        self.th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.th.start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def req(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Content-Type": "application/json"} if (body is not None or raw is not None) else {}
        h.update(headers or {})
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = r.read()
        c.close()
        try:
            return r.status, json.loads(out.decode("utf-8"))
        except ValueError:
            return r.status, out

    def save(self, **body):
        return self.req("POST", "/audit/save", body)


@pytest.fixture
def server(data_root, sample):
    s = Server(data_root["root"], sample)
    yield s
    s.close()


def _disk(path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------- the sample is what audit_shots draws

def test_sample_is_an_audit_shots_draw(data_root):
    s = _disk(data_root["pristine"])
    assert list(s.columns) == A.COLUMNS and len(s) == 2 * N_PER_GAME
    assert set(s["sportvu_game"]) == set(data_root["games"]) and (s["release_frame"] >= 0).all()
    assert s[A.AUDIT_COLUMNS].isna().all().all()


# ---------------------------------------------------------------------------- landing on the detected release

def test_first_shot_lands_on_detected_release(server, data_root, sample):
    status, page = server.req("GET", "/")
    assert status == 200 and b"Shot Audit" in page and b"/audit/save" in page
    status, st = server.req("GET", "/audit/state")
    assert status == 200 and st["n"] == 2 * N_PER_GAME and st["n_audited"] == 0 and st["first_unaudited"] == 0
    assert st["tolerance_frames"] == A.TOLERANCE_FRAMES == 2
    row = _disk(sample).iloc[0]
    status, d = server.req("GET", "/audit/shot?index=0")
    assert status == 200
    rel = int(row["release_frame"])
    assert d["release_frame"] == rel and d["open_frame"] == rel
    ids = [f["frame"] for f in d["frames"]]
    k = ids.index(rel)
    assert k >= 25 and len(ids) - 1 - k >= 25, "a second of context either side of the release"
    assert ids == sorted(ids) and ids[0] >= rel - T.AUDIT_CONTEXT_FRAMES and ids[-1] <= rel + T.AUDIT_CONTEXT_FRAMES
    assert d["game"] == row["sportvu_game"] and d["event"] == int(row["event"]) and d["period"] == int(row["period"])
    assert d["shooter_id"] == int(row["shooter_id"]) and d["shooter"]["name"]
    assert str(d["shooter_id"]) in d["players"] and d["players"][str(d["shooter_id"])]["side"] == "offense"
    assert d["confidence_flag"] == row["confidence_flag"]
    assert d["audit"] == {"audited_release_frame": None, "audited_ok": None, "auditor_note": None}
    # every served frame is in the shot's period
    fr = data_root["games"][d["game"]]
    assert set(fr.loc[fr["frame"].isin(ids), "period"]) == {d["period"]}


# ---------------------------------------------------------------------------- the walk, then --summarise

def test_walk_accept_move_bad_then_summarise(server, data_root, sample, tmp_path):
    before = _disk(sample)
    schema0 = pq.read_schema(sample)
    rel = before["release_frame"].astype(int).tolist()

    # 0: accept the detected frame
    status, r = server.save(index=0, verdict="accept")
    assert status == 200 and r["audit"] == {"audited_release_frame": rel[0], "audited_ok": True, "auditor_note": None}
    assert r["first_unaudited"] == 1
    on_disk = _disk(sample)   # written at once, not at exit
    assert on_disk.loc[0, "audited_release_frame"] == rel[0] and bool(on_disk.loc[0, "audited_ok"]) is True
    assert on_disk.loc[1:, A.AUDIT_COLUMNS].isna().all().all()

    # 1: the release is three frames later than detected (outside the 2-frame mark)
    status, r = server.save(index=1, verdict="release", audited_release_frame=rel[1] + 3, auditor_note="  late\nrelease ")
    assert status == 200 and r["audit"] == {"audited_release_frame": rel[1] + 3, "audited_ok": True, "auditor_note": "late release"}

    # 2: not a shot
    status, r = server.save(index=2, verdict="bad", auditor_note="tip-in, no release")
    assert status == 200 and r["audit"] == {"audited_release_frame": None, "audited_ok": False, "auditor_note": "tip-in, no release"}

    # 3: two frames early (at the mark)
    status, r = server.save(index=3, verdict="release", audited_release_frame=rel[3] - 2)
    assert status == 200 and r["n_audited"] == 4 and r["first_unaudited"] == 4

    after = _disk(sample)
    # row order, every other column and every dtype unchanged; the arrow schema (with its metadata) too
    pd.testing.assert_frame_equal(after[A.REPLAY_COLUMNS], before[A.REPLAY_COLUMNS], check_dtype=True)
    assert after.dtypes.to_dict() == before.dtypes.to_dict()
    assert pq.read_schema(sample).equals(schema0, check_metadata=True)
    assert after["audited_release_frame"].tolist()[:4] == [rel[0], rel[1] + 3, pd.NA, rel[3] - 2]
    assert after["audited_ok"].tolist()[:4] == [True, True, False, True]
    assert after["auditor_note"].tolist()[:4] == [pd.NA, "late release", "tip-in, no release", pd.NA]
    assert after.loc[4:, A.AUDIT_COLUMNS].isna().all().all()
    # nothing but the sample and its one backup in the folder; the backup is the state before the first write
    assert sorted(p.name for p in sample.parent.iterdir()) == ["shots_sample.parquet", "shots_sample.parquet.bak"]
    pd.testing.assert_frame_equal(_disk(sample.with_name("shots_sample.parquet.bak")), before)
    # audit mode wrote no labels anywhere
    assert not (data_root["root"] / "labels").exists() and not (sample.parent / "labels").exists()

    res = A.summarise(after)
    assert res["audited"] == 3 and res["within_tolerance"] == 2
    assert res["agreement_within_tolerance"] == round(2 / 3, 4)
    assert res["mean_abs_diff_frames"] == round((0 + 3 + 2) / 3, 3)
    assert res["audited_ok_filled"] == 4 and res["share_audited_ok"] == 0.75
    assert sum(v["audited"] for v in res["by_game"].values()) == 3
    report = tmp_path / "shot_audit.md"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert A.main(["--summarise", str(sample), "--report", str(report)]) == 0
    assert "agreement within 2 frames: 0.6667 over 3 audited rows" in buf.getvalue()
    assert "| within tolerance | 2 |" in report.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- resume

def test_resume_after_restart_lands_on_first_unaudited(data_root, sample):
    s1 = Server(data_root["root"], sample)
    try:
        rel0 = s1.session.shot_payload(0)["release_frame"]
        assert s1.save(index=0, verdict="accept")[0] == 200
        assert s1.save(index=1, verdict="bad", auditor_note="out of frame")[0] == 200
    finally:
        s1.close()
    s2 = Server(data_root["root"], sample)
    try:
        status, st = s2.req("GET", "/audit/state")
        assert st["first_unaudited"] == 2 and st["n_audited"] == 2
        status, d = s2.req("GET", "/audit/shot?index=1")
        assert d["audit"] == {"audited_release_frame": None, "audited_ok": False, "auditor_note": "out of frame"}
        # a verdict can be changed: shot 0 becomes bad, its frame is cleared
        status, r = s2.save(index=0, verdict="bad")
        assert status == 200 and r["audit"]["audited_release_frame"] is None and r["first_unaudited"] == 2
        assert pd.isna(_disk(sample).loc[0, "audited_release_frame"]) and rel0 >= 0
    finally:
        s2.close()
    # a row filled by hand with only the frame (the old CSV route) counts as audited
    df = _disk(sample)
    df.loc[2, "audited_release_frame"] = int(df.loc[2, "release_frame"])
    df.to_parquet(sample, index=False)
    assert T.AuditSession(data_root["root"], sample, heldout=TOKENS).first_unaudited() == 3


def test_page_resumes_from_state_not_from_a_stale_hash():
    html = HTML.read_text(encoding="utf-8")
    assert "location.hash" not in html
    assert "loadShot(st.first_unaudited === null ? 0 : st.first_unaudited)" in html


# ---------------------------------------------------------------------------- validation and hardening

def test_invalid_inputs_rejected_and_nothing_written(server, data_root, sample):
    raw0 = sample.read_bytes()
    n = server.session.n
    row0 = server.session.df.iloc[0]
    game_fr = data_root["games"][str(row0["sportvu_game"])]
    rel = int(row0["release_frame"])
    other_period = int(game_fr.loc[game_fr["period"] != int(row0["period"]), "frame"].iloc[0])
    bad_bodies = [
        {"verdict": "accept"},                                                   # no index
        {"index": -1, "verdict": "accept"}, {"index": n, "verdict": "accept"},   # out of range
        {"index": True, "verdict": "accept"}, {"index": "0", "verdict": "accept"}, {"index": 0.0, "verdict": "accept"},
        {"index": 0, "verdict": "maybe"}, {"index": 0},                          # verdict
        {"index": 0, "verdict": "release"},                                      # no frame
        {"index": 0, "verdict": "release", "audited_release_frame": True},
        {"index": 0, "verdict": "release", "audited_release_frame": str(rel)},
        {"index": 0, "verdict": "release", "audited_release_frame": rel + 0.5},
        {"index": 0, "verdict": "release", "audited_release_frame": -1},
        {"index": 0, "verdict": "release", "audited_release_frame": int(game_fr["frame"].max()) + 1000},
        {"index": 0, "verdict": "release", "audited_release_frame": other_period},
        {"index": 0, "verdict": "accept", "audited_release_frame": rel + 4},     # accept takes the detected frame
        {"index": 0, "verdict": "bad", "audited_release_frame": rel},            # bad carries no frame
        {"index": 0, "verdict": "accept", "auditor_note": "x" * (T.AUDIT_NOTE_MAX + 1)},
        {"index": 0, "verdict": "accept", "auditor_note": 123},
    ]
    for body in bad_bodies:
        status, r = server.req("POST", "/audit/save", body)
        assert status == 400 and r["error"], body
    assert sample.read_bytes() == raw0 and not sample.with_name(sample.name + ".bak").exists()
    assert server.req("GET", "/audit/shot?index=abc")[0] == 400
    assert server.req("GET", "/audit/shot?index=%d" % n)[0] == 404
    # a note at the cap is fine
    assert server.save(index=0, verdict="accept", auditor_note="y" * T.AUDIT_NOTE_MAX)[0] == 200


def test_hardening_and_no_label_endpoints(server, data_root, sample):
    raw0 = sample.read_bytes()
    assert server.req("POST", "/audit/save", raw=b'{"index": 0, "verdict": "accept"}',
                      headers={"Content-Type": "text/plain"})[0] == 415
    assert server.req("POST", "/audit/save", {"index": 0, "verdict": "accept"}, headers={"Host": "evil.example:80"})[0] == 403
    assert server.req("GET", "/audit/shot?index=0", headers={"Host": "evil.example"})[0] == 403
    assert server.req("GET", "/audit/state", headers={"Host": "localhost:8765"})[0] == 200
    assert server.req("POST", "/audit/save", raw=b"[1, 2]")[0] == 400
    assert server.req("POST", "/audit/save", raw=b"not json")[0] == 400
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    c.putrequest("POST", "/audit/save"); c.putheader("Content-Type", "application/json")
    c.putheader("Content-Length", str(T.MAX_BODY + 1)); c.endheaders()
    assert c.getresponse().status == 413
    c.close()
    # the label endpoints do not exist in audit mode: it can never write labels
    for path, body in (("/label", {"kind": "action"}), ("/delete", {"kind": "action", "id": 0}), ("/visited", {"possession": 0})):
        assert server.req("POST", path, body)[0] == 404
    for path in ("/possessions", "/possession?index=0"):
        assert server.req("GET", path)[0] == 404
    assert sample.read_bytes() == raw0
    assert not (data_root["root"] / "labels").exists()


def test_change_on_disk_is_not_overwritten(server, sample):
    df = _disk(sample)
    df.loc[5, "auditor_note"] = "written elsewhere"
    df.to_parquet(sample, index=False)
    raw = sample.read_bytes()
    status, r = server.save(index=0, verdict="accept")
    assert status == 409 and "changed on disk" in r["error"]
    assert sample.read_bytes() == raw


def test_failed_write_keeps_the_sample(data_root, sample, monkeypatch):
    sess = T.AuditSession(data_root["root"], sample, heldout=TOKENS, backup=False)
    raw = sample.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(T.os, "replace", boom)
    with pytest.raises(OSError):
        sess.save({"index": 0, "verdict": "accept"})
    assert sample.read_bytes() == raw
    assert [p.name for p in sample.parent.iterdir()] == ["shots_sample.parquet"], "no temporary file left behind"
    assert sess.first_unaudited() == 0 and sess.df["audited_ok"].isna().all()


def test_csv_typed_audit_columns_are_restored(data_root, sample):
    """A sample whose audit columns went through a CSV (all-null double and object columns) is written
    back with the contract types; every other column keeps its arrow type."""
    df = _disk(sample)
    df["audited_release_frame"] = float("nan")
    df["audited_ok"] = pd.Series([None] * len(df), dtype=object)
    df["auditor_note"] = pd.Series([None] * len(df), dtype=object)
    df.to_parquet(sample, index=False)
    s = Server(data_root["root"], sample)
    try:
        assert s.save(index=0, verdict="accept")[0] == 200
    finally:
        s.close()
    sch = pq.read_schema(sample)
    orig = pq.read_schema(data_root["pristine"])
    for name in A.COLUMNS:
        assert sch.field(name).type == orig.field(name).type, name
    out = _disk(sample)
    assert str(out["audited_release_frame"].dtype) == "Int32" and str(out["audited_ok"].dtype) == "boolean"
    assert A.summarise(out)["audited"] == 1


def test_arrow_types_of_the_sample_are_kept(data_root, sample):
    """A sample written with plain (not large) arrow strings and an int64 event keeps those types."""
    import pyarrow as pa
    t = pq.read_table(sample)
    fields = [pa.field(f.name, pa.string() if pa.types.is_large_string(f.type) else
                       (pa.int64() if f.name == "event" else f.type)) for f in t.schema]
    pq.write_table(t.cast(pa.schema(fields, metadata=t.schema.metadata)), sample)
    before = pq.read_schema(sample)
    s = Server(data_root["root"], sample)
    try:
        assert s.save(index=0, verdict="release", audited_release_frame=int(s.session.df.loc[0, "release_frame"]) + 1,
                      auditor_note="n")[0] == 200
    finally:
        s.close()
    after = pq.read_schema(sample)
    assert after.remove_metadata().equals(before.remove_metadata())
    assert after.field("sportvu_game").type == pa.string() and after.field("event").type == pa.int64()


def test_unfound_release_row(data_root, sample):
    """A row without a detected release (--all) opens on the clock window of its period; accept is refused."""
    df = _disk(sample)
    df.loc[0, "release_frame"] = -1
    df.to_parquet(sample, index=False)
    s = Server(data_root["root"], sample)
    try:
        status, d = s.req("GET", "/audit/shot?index=0")
        assert status == 200 and d["release_frame"] == -1 and d["frames"]
        ids = [f["frame"] for f in d["frames"]]
        assert d["open_frame"] in ids
        status, r = s.save(index=0, verdict="accept")
        assert status == 400 and "no detected release" in r["error"]
        assert s.save(index=0, verdict="release", audited_release_frame=d["open_frame"])[0] == 200
    finally:
        s.close()


# ---------------------------------------------------------------------------- held-out

def test_heldout_sample_refused(data_root, sample, capsys):
    df = _disk(sample)
    df["sportvu_game"] = df["sportvu_game"].astype("string")
    df.loc[0, "sportvu_game"] = "12.16.2015.PHX.at.GSW"          # in sportvu/splits.json
    df.to_parquet(sample, index=False)
    with pytest.raises(T.HeldoutInSample):
        T.AuditSession(data_root["root"], sample)                  # the default guard reads the repo splits
    assert T.main(["--data", str(data_root["root"]), "--audit", str(sample), "--port", "0"]) == 2
    assert "held-out" in capsys.readouterr().err
    df.loc[0, "sportvu_game"] = "01.01.2016.ABQ.at.CLE"          # the held-out arena rule of audit_shots
    df.to_parquet(sample, index=False)
    with pytest.raises(T.HeldoutInSample):
        T.AuditSession(data_root["root"], sample, heldout=set())
    # the training sample itself passes the same default guard
    T.AuditSession(data_root["root"], data_root["pristine"])


def test_not_a_sample_refused(tmp_path, data_root):
    p = tmp_path / "x.parquet"
    pd.DataFrame({"sportvu_game": ["a"], "event": [1]}).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="not an audit_shots sample"):
        T.AuditSession(data_root["root"], p, heldout=TOKENS)


# ---------------------------------------------------------------------------- the page and the docs

# key -> what the page's keydown handler must call for it
KEYMAP = {
    '"a"': 'save("accept")',
    '"Enter"': 'save("release")',
    '"x"': 'save("bad")',
    '"m"': "markHere()",
    '"r"': "jumpToFrame(S.data.release_frame)",
    '"t"': '$("note").focus()',
    '"Escape"': "S.mark = null",
    '"ArrowRight"': "step(ev.shiftKey ? 5 : 1)",
    '"ArrowLeft"': "step(ev.shiftKey ? -5 : -1)",
    '"n"': "goShot(1)",
    '"p"': "goShot(-1)",
    '" "': "togglePlay()",
    '"f"': "S.full = !S.full",
}


def _handler(html: str) -> str:
    i = html.index('document.addEventListener("keydown"')
    return html[i:html.index("// ---------------------------------------------------------------- buttons", i)]


def test_html_wires_the_documented_keys_to_the_api():
    html = HTML.read_text(encoding="utf-8")
    h = _handler(html)
    assert "if (ev.ctrlKey || ev.metaKey || ev.altKey) return;" in h
    for key, call in KEYMAP.items():
        m = re.search(r"if \(k === %s\) \{([^\n]*)\}" % re.escape(key), h)
        assert m and call in m.group(1), (key, call)
    # inside the note field keys type text; enter and esc only leave the field
    j = h.index('tag === "INPUT"')
    assert "ev.target.blur()" in h[j:j + 220] and "return;" in h[j:j + 220]
    # save() posts the verdict, the marked frame and the note to /audit/save
    s = html[html.index("function save(verdict)"):html.index("function goShot(k)")]
    assert 'api("POST", "/audit/save", body)' in s and "body.audited_release_frame = S.mark" in s
    assert "auditor_note:" in s and "verdict: verdict" in s
    assert 'api("GET", "/audit/shot?index=" + i)' in html and 'api("GET", "/audit/state")' in html
    # the shot opens at the detected release frame
    ls = html[html.index("function loadShot(i)"):html.index("function save(verdict)")]
    assert "indexOf(d.open_frame)" in ls
    # the on-page key list documents every verdict and step key
    for kbd in ("a", "m", "enter", "x", "r", "esc", "t", "n", "p", "left", "right", "shift"):
        assert "<kbd>%s</kbd>" % kbd in html, kbd
    # buttons go to the same calls
    for btn, call in (("btn-accept", 'save("accept")'), ("btn-save", 'save("release")'), ("btn-bad", 'save("bad")'),
                      ("btn-mark", "markHere")):
        m = re.search(r'\$\("%s"\)\.addEventListener\("click", ([^\n]*)\);' % btn, html)
        assert m and call in m.group(1), btn


def test_audit_keys_do_not_clash_with_label_tool_keys():
    label = (ROOT / "analytics" / "label_tool.html").read_text(encoding="utf-8")
    used = set(re.findall(r'k === "([^"]+)"', _handler(label)))
    assert {"s", "e", "h", "c", "v", "u", "j", "n", "p"} <= used, "the label tool's own keys are found"
    for verdict_key in ("a", "x", "m", "r", "t"):
        assert verdict_key not in used, verdict_key


def test_html_is_self_contained_and_dash_free():
    html = HTML.read_text(encoding="utf-8")
    stripped = html.replace("http://www.w3.org/", "")
    assert "http://" not in stripped and "https://" not in stripped
    assert re.search(r"<script[^>]*\ssrc=", html) is None and "<link" not in html and "url(" not in html
    for rel in ("analytics/label_tool_audit.html", "analytics/label_tool.py", "tests/test_label_tool_audit.py",
                "docs/labeling_guide.md", "docs/contract/phase0_driver.md", "docs/contract/labels.md",
                "docs/mac_runbook.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, rel


def test_cli_and_docs():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        T.main(["--help"])
    h = " ".join(buf.getvalue().split())
    assert "--audit SAMPLE" in h and "--no-backup" in h and "no labels are written" in h
    guide = (ROOT / "docs" / "labeling_guide.md").read_text(encoding="utf-8")
    assert "--audit" in guide and "audit_shots --summarise" in guide
    for row in ("| `a` |", "| `m` |", "| `enter` |", "| `x` |", "| `r` |", "| `t` |", "| `esc` |"):
        assert row in guide, row
    contract = (ROOT / "docs" / "contract" / "phase0_driver.md").read_text(encoding="utf-8")
    for text in ("audited_ok = False", "audited_release_frame = <NA>", "first row with both", "os.replace"):
        assert text in contract, text
    runbook = (ROOT / "docs" / "mac_runbook.md").read_text(encoding="utf-8")
    assert "label_tool --data \"$DATA_ROOT\" --audit" in runbook and "export the sample to CSV" not in runbook
