"""tools.p8_drive_xfer offline: a fake Drive behind the transport. Hash checks, Range resume, refusal on mismatch,
the staging folder rules, the way back (resumable upload), and that the token never reaches output, errors or files."""

import hashlib
import json
import urllib.parse

import pytest

from tools import p8_drive_xfer as X

TOKEN = "ya29.SECRET-TOKEN-never-in-output"


class FakeDrive:
    """files: id -> {name, parent, data, folder}; answers files.list, alt=media Range GETs and resumable uploads."""

    def __init__(self):
        self.files, self.n, self.calls, self.fail_next_get = {}, 0, [], 0
        self.sessions = {}

    def add(self, name, parent="root", data=b"", folder=False, app=None):
        self.n += 1
        fid = f"id{self.n}"
        self.files[fid] = {"name": name, "parent": parent, "data": data, "folder": folder, "app": app or {}}
        return fid

    def record(self, fid):
        f = self.files[fid]
        out = {"id": fid, "name": f["name"], "mimeType": X.FOLDER_MIME if f["folder"] else "application/octet-stream"}
        if not f["folder"]:
            out.update({"size": str(len(f["data"])), "md5Checksum": hashlib.md5(f["data"]).hexdigest()})
        return out

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers)))
        assert headers.get("Authorization") == f"Bearer {TOKEN}"
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        if url in self.sessions:
            s = self.sessions[url]
            rng = headers["Content-Range"]
            if body:
                s["data"] += body
            if len(s["data"]) < s["size"]:
                return 308, {"range": f"bytes=0-{len(s['data']) - 1}"}, b""
            fid = self.add(s["meta"]["name"], s["meta"]["parents"][0], s["data"])
            return 200, {}, json.dumps(self.record(fid)).encode()
        if url.startswith(X.UPLOAD):
            meta = json.loads(body)
            sid = f"{X.UPLOAD}?uploadType=resumable&upload_id=s{len(self.sessions)}"
            self.sessions[sid] = {"meta": meta, "data": b"", "size": int(headers["X-Upload-Content-Length"])}
            return 200, {"location": sid}, b""
        if method == "POST":                                       # create folder
            meta = json.loads(body)
            fid = self.add(meta["name"], (meta.get("parents") or ["root"])[0], folder=True,
                           app=meta.get("appProperties"))
            return 200, {}, json.dumps({"id": fid}).encode()
        if u.path.endswith("/files") and "q" in q:                 # files.list
            qq = q["q"][0]
            if qq.startswith("appProperties has"):
                kids = [self.record(i) for i, f in self.files.items()
                        if f["folder"] and f["app"].get(X.APP_KEY) == X.APP_VALUE]
            else:
                parent = qq.split("'")[1]
                kids = [self.record(i) for i, f in self.files.items() if f["parent"] == parent]
            return 200, {}, json.dumps({"files": kids}).encode()
        fid = u.path.rsplit("/", 1)[1]
        if "alt" not in q:                                         # files.get metadata
            return 200, {}, json.dumps(self.record(fid)).encode()
        if self.fail_next_get:
            self.fail_next_get -= 1
            return 503, {"x-echo-authorization": headers["Authorization"]}, b"busy"
        lo, hi = map(int, headers["Range"].split("=")[1].split("-"))
        return 206, {}, self.files[fid]["data"][lo:hi + 1]


def _map(fake, blobs):
    files = {}
    for name, data in blobs.items():
        fid = fake.add(name, data=data)
        files[name] = {"id": fid, "size": len(data), "md5": hashlib.md5(data).hexdigest(),
                       "sha256": hashlib.sha256(data).hexdigest()}
    return {"folder": "nba_p8_xfer/t", "files": files}


def test_fetch_checks_hashes_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "CHUNK", 7)                            # many Range requests
    fake = FakeDrive()
    blobs = {"p8_cuda_pack.tar": bytes(range(256)) * 3, "pack_manifest.json": b'{"game": "gsw_ind_2016"}'}
    idmap = _map(fake, blobs)
    dest = tmp_path / "in"
    dest.mkdir()
    (dest / "p8_cuda_pack.tar.part").write_bytes(blobs["p8_cuda_pack.tar"][:100])     # a broken earlier download
    done = X.download(X.Drive(TOKEN, fake), idmap, dest)
    assert done == {"p8_cuda_pack.tar": "fetched", "pack_manifest.json": "fetched"}
    assert (dest / "p8_cuda_pack.tar").read_bytes() == blobs["p8_cuda_pack.tar"]
    first = [c for c in fake.calls if c[2].get("Range")][0]
    assert first[2]["Range"] == "bytes=100-106"                  # resumed from the .part file
    assert X.download(X.Drive(TOKEN, fake), idmap, dest)["p8_cuda_pack.tar"] == "present"


def test_fetch_refuses_a_mismatch_and_keeps_it_apart(tmp_path):
    fake = FakeDrive()
    idmap = _map(fake, {"a.tar": b"abcdef"})
    idmap["files"]["a.tar"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        X.download(X.Drive(TOKEN, fake), idmap, tmp_path)
    assert (tmp_path / "a.tar.bad").exists() and not (tmp_path / "a.tar").exists()


def test_bad_id_maps_are_refused(tmp_path):
    for bad in ({"files": {}}, {"files": {"../x": {"id": "1", "size": 1, "md5": "", "sha256": ""}}},
                {"files": {"a.tar": {"id": "1", "size": 1}}}):
        with pytest.raises(SystemExit):
            X.download(X.Drive(TOKEN, FakeDrive()), bad, tmp_path)


def test_the_token_never_reaches_output_errors_or_files(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    fake = FakeDrive()
    idmap = _map(fake, {"a.tar": b"abcdef"})
    fake.fail_next_get = 99                                      # Drive keeps failing, echoing the header back
    with pytest.raises(SystemExit) as e:
        X.download(X.Drive(TOKEN, fake), idmap, tmp_path)
    assert TOKEN not in str(e.value)
    err = X.DriveError(503, f"{X.API}/id1?alt=media&access_token={TOKEN}")
    assert TOKEN not in str(err)
    fake.fail_next_get = 0
    X.download(X.Drive(TOKEN, fake), idmap, tmp_path)
    fid = X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out")
    X.upload_one(X.Drive(TOKEN, fake), tmp_path / "a.tar", fid, "a.tar")
    out = capsys.readouterr()
    assert TOKEN not in out.out and TOKEN not in out.err
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert TOKEN.encode() not in p.read_bytes()
    monkeypatch.delenv("DRIVE_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        X.runtime_token()


def test_make_folder_upload_and_pull_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(X, "UP_CHUNK", 5)
    fake = FakeDrive()
    fid = X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out")
    assert X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out") == fid           # reused, not made twice
    src = tmp_path / "p8_cuda_gsw_ind_2016.tar"
    src.write_bytes(b"result-bytes-" * 7)
    rec = X.upload_one(X.Drive(TOKEN, fake), src, fid, src.name)
    assert fake.files[rec["id"]]["data"] == src.read_bytes() and fake.files[rec["id"]]["parent"] == fid
    assert sum(1 for c in fake.calls if c[0] == "PUT") > 1                           # resumable, in pieces
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line) == rec                                                    # the line the runtime prints
    back = tmp_path / "back"
    X.pull(X.Drive(TOKEN, fake), rec["id"], rec["name"], rec["md5"], rec["sha256"], back, rec["size"])
    assert (back / src.name).read_bytes() == src.read_bytes()
    with pytest.raises(SystemExit):                                                   # not what the runtime printed
        X.pull(X.Drive(TOKEN, fake), rec["id"], "again.tar", rec["md5"], "0" * 64, back, rec["size"])
    assert (back / "again.tar.bad").exists()
    with pytest.raises(SystemExit):
        X.upload_one(X.Drive(TOKEN, fake), src, fid, "../escape.tar")


def test_upload_refuses_when_drive_holds_other_bytes(tmp_path, monkeypatch):
    fake = FakeDrive()
    fid = X.make_folder(X.Drive(TOKEN, fake), "out")
    src = tmp_path / "r.tar"
    src.write_bytes(b"abc")
    real = fake.__call__

    def corrupt(method, url, headers, body):
        if url in fake.sessions and body:
            body = b"xyz"
        return real(method, url, headers, body)
    with pytest.raises(SystemExit):
        X.upload_one(X.Drive(TOKEN, corrupt), src, fid, "r.tar")


def _mount(tmp_path, monkeypatch):
    mydrive = tmp_path / "drive_root"
    (mydrive / "nba_harvest").mkdir(parents=True)
    monkeypatch.setenv("DATA_ROOT", str(mydrive / "nba_harvest"))
    return mydrive


def test_stage_dir_rules(tmp_path, monkeypatch):
    mydrive = _mount(tmp_path, monkeypatch)
    assert X.check_stage_dir(mydrive / "nba_p8_xfer" / "cuda") == ["nba_p8_xfer", "cuda"]
    for bad in (mydrive / "nba_harvest" / "x",                  # DATA_ROOT
                mydrive / "elsewhere" / "cuda",                 # not nba_p8_xfer
                mydrive / "nba_p8_xfer",                        # no name
                mydrive / "nba_p8_xfer" / "a" / "b",            # too deep
                tmp_path / "not_drive" / "nba_p8_xfer" / "x",   # off the mount
                X.Path(__file__).resolve().parents[1] / "reports" / "x"):   # the repository
        with pytest.raises(SystemExit):
            X.check_stage_dir(bad)


def test_stage_copies_waits_for_sync_and_writes_the_map(tmp_path, monkeypatch):
    mydrive = _mount(tmp_path, monkeypatch)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "p8_cuda_pack.tar").write_bytes(b"tar-bytes" * 50)
    (pack / "pack_manifest.json").write_text(json.dumps({"game": "gsw_ind_2016", "windows": {"gsw_ind_2016_s02_w03": {}}}))
    (pack / "p8_cuda_pack.tar.prev-20261125T000000Z").write_bytes(b"old")      # never staged
    fake = FakeDrive()
    clock = iter(range(0, 10_000, 15))
    polls = []

    def sleep(s):                       # the Drive client "uploads" between polls: folder, then one file per poll
        polls.append(s)
        if len(polls) == 1:
            x = fake.add("nba_p8_xfer", folder=True)
            fake.add("cuda", parent=x, folder=True)
        else:
            folder = [i for i, f in fake.files.items() if f["name"] == "cuda"][0]
            have = {f["name"] for f in fake.files.values() if f["parent"] == folder}
            for p in sorted(pack.iterdir()):
                if p.name in ("p8_cuda_pack.tar", "pack_manifest.json") and p.name not in have:
                    fake.add(p.name, parent=folder, data=p.read_bytes())
                    break
    idmap = X.stage(pack, mydrive / "nba_p8_xfer" / "cuda", X.Drive(TOKEN, fake), timeout_s=600, sleep=sleep,
                    clock=lambda: next(clock))
    assert set(idmap["files"]) == {"p8_cuda_pack.tar", "pack_manifest.json"} and len(polls) == 3
    assert (mydrive / "nba_p8_xfer" / "cuda" / "p8_cuda_pack.tar").exists()
    on_disk = json.loads((pack / X.ID_MAP).read_text())
    assert on_disk == idmap and idmap["folder"] == "nba_p8_xfer/cuda"
    line = X.push_map(pack)
    assert "\n" not in line and json.loads(line) == idmap
    from export.schemas import find_private
    assert find_private(line) == [] and TOKEN not in line


def test_stage_times_out_when_drive_never_syncs(tmp_path, monkeypatch, capsys):
    mydrive = _mount(tmp_path, monkeypatch)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "a.tar").write_bytes(b"x")
    clock = iter(range(0, 10_000, 100))
    with pytest.raises(SystemExit) as e:
        X.stage(pack, mydrive / "nba_p8_xfer" / "t", X.Drive(TOKEN, FakeDrive()), timeout_s=250,
                sleep=lambda s: None, clock=lambda: next(clock))
    assert "0 of 1" in str(e.value) and not (pack / X.ID_MAP).exists()


def test_synced_needs_size_and_md5():
    local = {"a": {"size": 3, "md5": hashlib.md5(b"abc").hexdigest(), "sha256": ""}}
    assert X.synced(local, [{"name": "a", "id": "1", "size": "3", "md5Checksum": hashlib.md5(b"abc").hexdigest()}])
    assert not X.synced(local, [{"name": "a", "id": "1", "size": "2", "md5Checksum": hashlib.md5(b"abc").hexdigest()}])
    assert not X.synced(local, [{"name": "a", "id": "1", "size": "3"}])          # still uploading: no md5 yet


def test_pull_without_size_reads_drive_metadata_and_checks_its_md5(tmp_path):
    fake = FakeDrive()
    data = b"complete-json-bytes"
    fid = fake.add("p8_cuda_gsw_ind_2016.complete.json", data=data)
    md5, sha = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
    X.pull(X.Drive(TOKEN, fake), fid, "p8_cuda_gsw_ind_2016.complete.json", md5, sha, tmp_path)
    assert (tmp_path / "p8_cuda_gsw_ind_2016.complete.json").read_bytes() == data
    with pytest.raises(SystemExit):
        X.pull(X.Drive(TOKEN, fake), fid, "other.json", "0" * 32, sha, tmp_path)



# ── security review fixes (2026-11-25) ──────────────────────────────────────────────────────────────────────────────
def test_tokens_are_stripped_and_anything_else_is_refused_unseen(monkeypatch, tmp_path):
    monkeypatch.setenv("DRIVE_TOKEN", TOKEN + "\n")                       # a token ending in a newline
    assert X.runtime_token() == TOKEN
    for bad in ("ya29.abc\ndef", "ya29 abc", "ya29.abc\r\nX-Evil: 1", "   "):
        monkeypatch.setenv("DRIVE_TOKEN", bad)
        with pytest.raises(SystemExit) as e:
            X.runtime_token()
        assert bad.strip() not in str(e.value) or not bad.strip()
    helper = tmp_path / "colab_headers.py"
    helper.write_text("def token():\n    return {'access_token': 'ya29.x\\ny'}\n")
    monkeypatch.setattr(X, "HELPER", helper)
    with pytest.raises(SystemExit) as e:
        X.mac_token()
    assert "ya29" not in str(e.value)
    helper.write_text(f"def token():\n    return {{'access_token': '{TOKEN}\\n'}}\n")
    assert X.mac_token() == TOKEN


def test_a_header_http_client_rejects_never_shows_the_token(monkeypatch):
    class Boom:
        def open(self, req, timeout=None):
            raise ValueError(f"Invalid header value b'Bearer {TOKEN}\\n'")      # what http.client says
    monkeypatch.setattr(X, "opener", lambda: Boom())
    with pytest.raises(ConnectionError) as e:
        X.urllib_transport("GET", f"{X.API}/id1?alt=media", {"Authorization": f"Bearer {TOKEN}\n"}, None)
    assert TOKEN not in str(e.value) and "alt=media" not in str(e.value)
    monkeypatch.undo()
    with pytest.raises(ConnectionError) as e:                                  # the real stack, no network reached
        X.urllib_transport("GET", "http://127.0.0.1:9/x?q=1", {"Authorization": f"Bearer {TOKEN}\nX: y"}, None, timeout=2)
    assert TOKEN not in str(e.value)


def test_redirects_are_refused_so_the_header_never_follows(tmp_path):
    import http.server
    import threading
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/a"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/b")
                self.end_headers()
            else:
                seen.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"stolen")

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        status, _, body = X.urllib_transport("GET", f"http://127.0.0.1:{srv.server_port}/a",
                                             {"Authorization": f"Bearer {TOKEN}"}, None, timeout=5)
    finally:
        srv.shutdown()
    assert status == 302 and body != b"stolen" and seen == []


def test_drive_error_carries_googles_reason_only():
    body = json.dumps({"error": {"code": 403, "status": "PERMISSION_DENIED",
                                 "errors": [{"reason": "accessNotConfigured", "message": "Drive API disabled"}]}}).encode()
    e = X.DriveError(403, f"{X.API}?q=x", body=body)
    assert "accessNotConfigured" in str(e) and "Drive API disabled" not in str(e) and "q=x" not in str(e)
    assert X.error_reason(json.dumps({"error": {"status": "SERVICE_DISABLED"}}).encode()) == "SERVICE_DISABLED"
    assert X.error_reason(json.dumps({"error": {"errors": [{"reason": f"Bearer {TOKEN}"}]}}).encode()) is None
    assert X.error_reason(b"not json") is None


def test_make_folder_reuses_only_its_own_app_folder():
    fake = FakeDrive()
    stranger = fake.add("nba_p8_xfer_out", folder=True)                  # same name, not this client's
    fid = X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out")
    assert fid != stranger and fake.files[fid]["app"] == {X.APP_KEY: X.APP_VALUE}
    assert X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out") == fid
    fake.add("nba_p8_xfer_out", folder=True, app={X.APP_KEY: X.APP_VALUE})
    with pytest.raises(SystemExit):
        X.make_folder(X.Drive(TOKEN, fake), "nba_p8_xfer_out")


def test_refusals_name_the_place_not_the_path(tmp_path, monkeypatch):
    mydrive = _mount(tmp_path, monkeypatch)
    secret_parent = str(mydrive)
    for p, label in ((mydrive / "nba_harvest" / "x", "DATA_ROOT"),
                     (X.Path(__file__).resolve().parents[1] / "reports" / "y", "the repository")):
        with pytest.raises(SystemExit) as e:
            X.outside(p)
        msg = str(e.value)
        assert label in msg and p.name in msg and secret_parent not in msg and str(p.parent) not in msg
    with pytest.raises(SystemExit) as e:
        X.check_stage_dir(mydrive / "nba_harvest" / "x")
    assert secret_parent not in str(e.value)


def test_drive_ids_are_validated(tmp_path):
    fake = FakeDrive()
    for bad_id in ("../x", "id 1", "id1?alt=media", "", 5):
        with pytest.raises(SystemExit):
            X.check_map({"files": {"a.tar": {"id": bad_id, "size": 1, "md5": "m", "sha256": "s"}}})
        with pytest.raises(SystemExit):
            X.pull(X.Drive(TOKEN, fake), bad_id, "a.tar", "m", "s", tmp_path, 1)
    with pytest.raises(SystemExit):
        X.upload_one(X.Drive(TOKEN, fake), tmp_path, "id1/../x", "a.tar")


def test_symlinks_and_a_pack_in_the_repository_are_refused(tmp_path, monkeypatch):
    fake = FakeDrive()
    idmap = _map(fake, {"a.tar": b"abcdef"})
    dest = tmp_path / "d"
    dest.mkdir()
    (tmp_path / "elsewhere").write_bytes(b"")
    (dest / "a.tar.part").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(SystemExit):
        X.download(X.Drive(TOKEN, fake), idmap, dest)
    assert (tmp_path / "elsewhere").read_bytes() == b""
    mydrive = _mount(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):                                       # a pack inside the repository
        X.stage(X.Path(__file__).resolve().parents[1] / "reports", mydrive / "nba_p8_xfer" / "t",
                X.Drive(TOKEN, fake), timeout_s=1, sleep=lambda s: None)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "a.tar").write_bytes(b"x")
    stage_dir = mydrive / "nba_p8_xfer" / "t"
    stage_dir.mkdir(parents=True)
    (stage_dir / "a.tar").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(SystemExit):
        X.stage(pack, stage_dir, X.Drive(TOKEN, fake), timeout_s=1, sleep=lambda s: None)


def test_upload_gives_up_when_drive_makes_no_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "UP_CHUNK", 4)
    fake = FakeDrive()
    fid = X.make_folder(X.Drive(TOKEN, fake), "out")
    src = tmp_path / "r.tar"
    src.write_bytes(b"0123456789")
    puts = []

    def stuck(method, url, headers, body):
        if url in fake.sessions:
            puts.append(1)
            return 308, {}, b""                                            # never acknowledges a byte
        return fake(method, url, headers, body)
    with pytest.raises(X.DriveError) as e:                                 # main() turns it into an exit
        X.upload_one(X.Drive(TOKEN, stuck), src, fid, "r.tar")
    assert len(puts) <= X.MAX_STALLS + 2 and TOKEN not in str(e.value)


def test_get_range_refuses_an_ignored_range_or_a_wrong_length():
    def full(method, url, headers, body):
        return 200, {}, b"whole-file"
    with pytest.raises(X.DriveError):
        X.Drive(TOKEN, full).get_range("id1", 5, 9)                       # 200 for a Range not at 0
    def short(method, url, headers, body):
        return 206, {}, b"abc"
    with pytest.raises(X.DriveError):
        X.Drive(TOKEN, short).get_range("id1", 0, 9)
    assert X.Drive(TOKEN, lambda *a: (200, {}, b"0123456789")).get_range("id1", 0, 9) == b"0123456789"


def test_cuda_check_no_longer_mentions_collect():
    src = (X.Path(__file__).resolve().parents[1] / "tools" / "p8_cuda_check.py").read_text()
    assert "collect" not in src
