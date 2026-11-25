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

    def add(self, name, parent="root", data=b"", folder=False):
        self.n += 1
        fid = f"id{self.n}"
        self.files[fid] = {"name": name, "parent": parent, "data": data, "folder": folder}
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
        if url.startswith(X.UPLOAD):
            meta = json.loads(body)
            sid = f"https://upload.example/session{len(self.sessions)}"
            self.sessions[sid] = {"meta": meta, "data": b"", "size": int(headers["X-Upload-Content-Length"])}
            return 200, {"location": sid}, b""
        if url in self.sessions:
            s = self.sessions[url]
            rng = headers["Content-Range"]
            if body:
                s["data"] += body
            if len(s["data"]) < s["size"]:
                return 308, {"range": f"bytes=0-{len(s['data']) - 1}"}, b""
            fid = self.add(s["meta"]["name"], s["meta"]["parents"][0], s["data"])
            return 200, {}, json.dumps(self.record(fid)).encode()
        if method == "POST":                                       # create folder
            meta = json.loads(body)
            fid = self.add(meta["name"], (meta.get("parents") or ["root"])[0], folder=True)
            return 200, {}, json.dumps({"id": fid}).encode()
        if u.path.endswith("/files") and "q" in q:                 # files.list
            parent = q["q"][0].split("'")[1]
            kids = [self.record(i) for i, f in self.files.items() if f["parent"] == parent]
            return 200, {}, json.dumps({"files": kids}).encode()
        fid = u.path.rsplit("/", 1)[1]                             # files.get alt=media
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
    X.push(X.Drive(TOKEN, fake), [tmp_path / "a.tar"], "nba_p8_xfer_out")
    out = capsys.readouterr()
    assert TOKEN not in out.out and TOKEN not in out.err
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert TOKEN.encode() not in p.read_bytes()
    monkeypatch.delenv("DRIVE_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        X.runtime_token()


def test_push_uploads_in_pieces_and_maps_the_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(X, "UP_CHUNK", 5)
    fake = FakeDrive()
    src = tmp_path / "p8_cuda_gsw_ind_2016.tar"
    src.write_bytes(b"result-bytes-" * 7)
    m = X.push(X.Drive(TOKEN, fake), [src], "nba_p8_xfer_out")
    rec = m["files"][src.name]
    assert rec["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert fake.files[rec["id"]]["data"] == src.read_bytes()
    assert sum(1 for c in fake.calls if c[0] == "PUT") > 1
    m2 = X.push(X.Drive(TOKEN, fake), [src], "nba_p8_xfer_out")   # the folder is found, not made twice
    assert sum(1 for f in fake.files.values() if f["folder"]) == 1
    back = tmp_path / "back"
    X.download(X.Drive(TOKEN, fake), m2, back)
    assert (back / src.name).read_bytes() == src.read_bytes()


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
