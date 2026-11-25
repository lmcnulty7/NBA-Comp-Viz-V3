"""tools/p8_drive_xfer.py: move P8 files between the Mac and a Colab API runtime through Google Drive, because a local
web filter blocks the Mac's HTTP route to the runtime (docs/research/P8_cuda_check.md, amendment A14). Stdlib only.

    python3 -m tools.p8_drive_xfer stage --pack <folder> --drive-dir <Drive root>/nba_p8_xfer/<name>       Mac
    python3 -m tools.p8_drive_xfer push-map --pack <folder>                                                Mac
    python -m tools.p8_drive_xfer fetch --ids <id map json> --dest <dir>                                   runtime
    python3 -m tools.p8_drive_xfer make-folder --name <name>                                              Mac
    python -m tools.p8_drive_xfer upload --src <file> --drive-dir-id <folder id> --name <name>              runtime
    python3 -m tools.p8_drive_xfer pull --file-id <id> --name <name> --md5 M --sha256 S [--size N] --dest <dir>   Mac

stage: copies the pack's files into a folder of the Drive mount (its root is the parent of DATA_ROOT; under
nba_p8_xfer/ only; never under DATA_ROOT, reports/ or a checkout), then polls the Drive API (files.list by parent,
fields id, name, size, md5Checksum) until every file is there with the local size and md5, and writes drive_ids.json
next to the pack manifest: per file its name, Drive id, size, md5 and sha256. push-map prints that map on one line;
it holds no secret and goes into the execute_code call text.

fetch: on the runtime, downloads each file of the map (files.get alt=media, Range requests, resumed from a .part
file), checks size, md5 and sha256 against the map and refuses on any mismatch (the file is renamed .bad).

The way back (drive.file scope, approved by Lucien beside drive.readonly): drive.file lets this OAuth client create
files and touch only files it created. make-folder creates the result folder from the Mac with the same client, so
the app owns it, and prints its id. upload (runtime) puts one file into that folder with a resumable upload, checks
Drive's size and md5 against the local file and prints one line: name, Drive id, size, md5, sha256. pull (Mac)
downloads that file id and checks size, md5 and sha256 against the values the runtime printed.

Tokens: the Mac takes its access token from the Colab sign-in helper (nba-colab/colab_headers.py in the user config folder, its token());
the runtime takes it from the environment variable DRIVE_TOKEN only. A token is never printed, logged, written to a
file or put in an error message.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WHO = "tools.p8_drive_xfer"
API = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
XFER_ROOT = "nba_p8_xfer"
ID_MAP = "drive_ids.json"
CHUNK = 16 << 20                       # bytes per Range request
UP_CHUNK = 8 << 20                     # bytes per upload request (a multiple of 256 KiB)
RETRIES = 4
HELPER = Path.home() / ".config" / "nba-colab" / "colab_headers.py"


class DriveError(Exception):
    """An HTTP failure, named by status and URL only (never the headers, so never the token)."""

    def __init__(self, status, url: str, what: str = ""):
        self.status = status
        super().__init__(f"{WHO}: HTTP {status} on {url.split('?')[0]} {what}".rstrip())


# ── tokens ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def mac_token() -> str:
    """The Mac's access token from the Colab sign-in helper (it renews from the stored refresh token)."""
    if not HELPER.exists():
        raise SystemExit(f"{WHO}: the Colab sign-in helper is missing; sign in again")
    spec = importlib.util.spec_from_file_location("nba_colab_headers", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = (mod.token() or {}).get("access_token")
    if not tok:
        raise SystemExit(f"{WHO}: the sign-in helper gave no access token; sign in again")
    return tok


def runtime_token() -> str:
    tok = os.environ.get("DRIVE_TOKEN")
    if not tok:
        raise SystemExit(f"{WHO}: set DRIVE_TOKEN in this process's environment")
    return tok


# ── HTTP ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _ssl_context():
    import ssl
    try:
        import certifi                                 # python.org builds ship no CA bundle
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: int = 600):
    """One request -> (status, response headers, body). An HTTP error status is returned, not raised; a network
    failure raises ConnectionError with no header in its message."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read() or b""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ConnectionError(f"{WHO}: network error on {url.split('?')[0]}: {type(e).__name__}") from None


class Drive:
    def __init__(self, token: str, transport=urllib_transport):
        self._token = token
        self._send = transport

    def _req(self, method, url, headers=None, body=None, ok=(200,)):
        h = {"Authorization": f"Bearer {self._token}", **(headers or {})}
        status, rh, data = self._send(method, url, h, body)
        if status not in ok:
            raise DriveError(status, url)
        return status, rh, data

    def list_children(self, parent: str) -> list:
        q = f"'{parent}' in parents and trashed = false"
        out, page = [], None
        while True:
            params = {"q": q, "fields": "nextPageToken,files(id,name,size,md5Checksum,mimeType)", "pageSize": 1000,
                      "spaces": "drive"}
            if page:
                params["pageToken"] = page
            _, _, data = self._req("GET", f"{API}?{urllib.parse.urlencode(params)}")
            d = json.loads(data)
            out += d.get("files", [])
            page = d.get("nextPageToken")
            if not page:
                return out

    def find(self, parent: str, name: str, folder: bool | None = None):
        hits = [f for f in self.list_children(parent) if f["name"] == name
                and (folder is None or (f.get("mimeType") == FOLDER_MIME) == folder)]
        return hits[0] if len(hits) == 1 else (None if not hits else _ambiguous(name))

    def folder_id(self, parts: list):
        """The id of the folder <parts...> under the Drive root, or None while it is not on Drive yet."""
        fid = "root"
        for p in parts:
            hit = self.find(fid, p, folder=True)
            if hit is None:
                return None
            fid = hit["id"]
        return fid

    def get_range(self, file_id: str, start: int, end: int) -> bytes:
        _, _, data = self._req("GET", f"{API}/{file_id}?alt=media", {"Range": f"bytes={start}-{end}"}, ok=(200, 206))
        return data

    def create_folder(self, name: str, parent: str | None = None) -> str:
        meta = {"name": name, "mimeType": FOLDER_MIME, **({"parents": [parent]} if parent else {})}
        _, _, data = self._req("POST", f"{API}?fields=id", {"Content-Type": "application/json"},
                               json.dumps(meta).encode())
        return json.loads(data)["id"]

    def upload(self, path: Path, parent: str, name: str | None = None) -> dict:
        """Resumable upload of one file into parent -> Drive's {id, name, size, md5Checksum}."""
        size = path.stat().st_size
        meta = json.dumps({"name": name or path.name, "parents": [parent]}).encode()
        _, rh, _ = self._req("POST", f"{UPLOAD}?uploadType=resumable&fields=id,name,size,md5Checksum",
                             {"Content-Type": "application/json; charset=UTF-8",
                              "X-Upload-Content-Length": str(size)}, meta)
        session = rh.get("location")
        if not session:
            raise DriveError("no session", UPLOAD)
        pos = 0
        with open(path, "rb") as f:
            while True:
                f.seek(pos)
                chunk = f.read(UP_CHUNK)
                end = pos + len(chunk) - 1
                rng = f"bytes {pos}-{end}/{size}" if chunk else f"bytes */{size}"
                status, rh, data = self._send("PUT", session, {"Content-Range": rng,
                                                               "Authorization": f"Bearer {self._token}"}, chunk)
                if status in (200, 201):
                    return json.loads(data)
                if status == 308:                          # resume from what Drive says it holds
                    got = rh.get("range")
                    pos = int(got.split("-")[1]) + 1 if got else 0
                    continue
                raise DriveError(status, UPLOAD, "during upload")


def _ambiguous(name):
    raise SystemExit(f"{WHO}: more than one Drive item named {name!r} in that folder; clear it by hand")


# ── hashes and the id map (pure) ────────────────────────────────────────────────────────────────────────────────────
def hashes(p: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            md5.update(b)
            sha.update(b)
    return {"size": Path(p).stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..") and not name.startswith(".")


def check_map(idmap: dict) -> dict:
    files = idmap.get("files") or {}
    if not files:
        raise SystemExit(f"{WHO}: the id map lists no file")
    for name, r in files.items():
        if not safe_name(name) or not all(k in r for k in ("id", "size", "md5", "sha256")):
            raise SystemExit(f"{WHO}: bad id map entry {name!r}")
    return files


def synced(local: dict, remote: list) -> dict:
    """{name: local hashes}, Drive's list of the folder -> {name: Drive record} for files whose size and md5 match;
    a file still missing or still uploading is left out."""
    by = {f["name"]: f for f in remote}
    out = {}
    for n, h in local.items():
        f = by.get(n)
        if f and int(f.get("size", -1)) == h["size"] and f.get("md5Checksum") == h["md5"]:
            out[n] = f
    return out


# ── where staged files may go ───────────────────────────────────────────────────────────────────────────────────────
def drive_mount() -> Path:
    if not os.environ.get("DATA_ROOT"):
        raise SystemExit(f"{WHO}: DATA_ROOT is not set; the Drive mount is its parent")
    return Path(os.environ["DATA_ROOT"]).expanduser().resolve().parent


def check_stage_dir(d: Path) -> list:
    """A staging folder must be nba_p8_xfer/<name> at the root of the Drive mount: never under DATA_ROOT, the repository,
    the main checkout or reports/ -> the folder's parts relative to the Drive root."""
    from tools.p8_common import check_outside
    p = check_outside(Path(d).expanduser())             # DATA_ROOT, this repository, the main checkout
    root = drive_mount()
    try:
        rel = p.relative_to(root)
    except ValueError:
        raise SystemExit(f"{WHO}: {p.name} is not on the Drive mount; stage under {XFER_ROOT}/<name> at the Drive root") from None
    parts = list(rel.parts)
    if len(parts) != 2 or parts[0] != XFER_ROOT or not safe_name(parts[1]) or "reports" in parts:
        raise SystemExit(f"{WHO}: stage only into {XFER_ROOT}/<name> at the Drive root, not {'/'.join(parts)}")
    return parts


def guard_pack(pack: Path, names: list) -> None:
    """The project guard on every staged name and, for a P8 pack manifest, its game and windows."""
    from tools.p8_common import guard
    from tools.research_testset import refuse_cv_test
    items = [n.rsplit(".", 1)[0] for n in names]
    man = pack / "pack_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        items += [m.get("game") or "", *m.get("windows", {}), *m.get("report_only_windows", [])]
    items = [i for i in items if i]
    refuse_cv_test(items, WHO)
    guard(items, WHO)


# ── commands ────────────────────────────────────────────────────────────────────────────────────────────────────────
def pack_files(pack: Path) -> dict:
    """The files to stage: every regular file of the pack folder but the id map and kept-aside .prev copies."""
    return {p.name: p for p in sorted(Path(pack).iterdir())
            if p.is_file() and safe_name(p.name) and p.name != ID_MAP and ".prev-" not in p.name and not p.name.endswith(".part")}


def stage(pack: Path, drive_dir: Path, drive: Drive, timeout_s: float = 3600, poll_s: float = 15,
          sleep=time.sleep, clock=time.time) -> dict:
    parts = check_stage_dir(drive_dir)
    files = pack_files(pack)
    guard_pack(pack, list(files))
    local = {n: hashes(p) for n, p in files.items()}
    dest = Path(drive_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    for n, p in files.items():
        q = dest / n
        if not (q.exists() and hashes(q) == local[n]):
            shutil.copyfile(p, q)
    t0, got = clock(), {}
    while True:
        fid = drive.folder_id(parts)
        if fid is not None:
            got = synced(local, drive.list_children(fid))
        if len(got) == len(local):
            break
        if clock() - t0 > timeout_s:
            raise SystemExit(f"{WHO}: Drive has {len(got)} of {len(local)} files after {int(timeout_s)} s; "
                             f"waiting for {sorted(set(local) - set(got))}")
        print(json.dumps({"synced": len(got), "of": len(local), "seconds": round(clock() - t0)}), flush=True)
        sleep(poll_s)
    idmap = {"folder": "/".join(parts), "files": {n: {"id": got[n]["id"], **local[n]} for n in local}}
    from tools.p8_common import atomic_write
    atomic_write(Path(pack) / ID_MAP, json.dumps(idmap, indent=1))
    print(json.dumps({"staged": len(local), "folder": idmap["folder"], "id_map": ID_MAP}), flush=True)
    return idmap


def push_map(pack: Path) -> str:
    """The id map on one line, for the execute_code call text (no secret, no local path)."""
    idmap = json.loads((Path(pack) / ID_MAP).read_text())
    check_map(idmap)
    return json.dumps(idmap, separators=(",", ":"))


def download(drive: Drive, idmap: dict, dest: Path) -> dict:
    """Every file of the map into dest, resumed from <name>.part, then size, md5 and sha256 checked; a mismatch is
    renamed <name>.bad and stops the command."""
    files = check_map(idmap)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    done = {}
    for name, r in files.items():
        final, part = dest / name, dest / (name + ".part")
        if final.exists() and hashes(final) == {k: r[k] for k in ("size", "md5", "sha256")}:
            done[name] = "present"
            continue
        pos = part.stat().st_size if part.exists() else 0
        if pos > r["size"]:
            part.unlink()
            pos = 0
        with open(part, "ab") as f:
            while pos < r["size"]:
                end = min(pos + CHUNK, r["size"]) - 1
                for attempt in range(RETRIES):
                    try:
                        b = drive.get_range(r["id"], pos, end)
                        break
                    except (ConnectionError, DriveError) as e:
                        if attempt == RETRIES - 1 or (isinstance(e, DriveError) and e.status not in (429, 500, 502, 503, 504)):
                            raise SystemExit(str(e)) from None
                        time.sleep(5 * (attempt + 1))
                if not b:
                    raise SystemExit(f"{WHO}: {name}: Drive returned no bytes at {pos}")
                f.write(b)
                pos += len(b)
        got = hashes(part)
        if got != {k: r[k] for k in ("size", "md5", "sha256")}:
            part.replace(dest / (name + ".bad"))
            raise SystemExit(f"{WHO}: {name} does not match the id map (size {got['size']}); kept as {name}.bad")
        part.replace(final)
        done[name] = "fetched"
    print(json.dumps({"files": done}), flush=True)
    return done


def make_folder(drive: Drive, name: str) -> str:
    """The result folder at the Drive root, created by this OAuth client so drive.file covers it; an existing folder of
    that name that the client already owns is reused -> its id."""
    if not safe_name(name):
        raise SystemExit(f"{WHO}: bad folder name {name!r}")
    hit = drive.find("root", name, folder=True)
    fid = hit["id"] if hit else drive.create_folder(name)
    print(json.dumps({"folder": name, "id": fid}), flush=True)
    return fid


def upload_one(drive: Drive, src: Path, folder_id: str, name: str) -> dict:
    """One file into the app's folder (runtime) -> {name, id, size, md5, sha256}, printed on one line; Drive's size and
    md5 must equal the local file's."""
    if not safe_name(name) or not safe_name(str(folder_id)):
        raise SystemExit(f"{WHO}: bad name {name!r} or folder id")
    src = Path(src)
    h = hashes(src)
    meta = drive.upload(src, folder_id, name)
    if int(meta.get("size", -1)) != h["size"] or meta.get("md5Checksum") != h["md5"]:
        raise SystemExit(f"{WHO}: Drive holds another {name} than the one sent")
    rec = {"name": name, "id": meta["id"], **h}
    print(json.dumps(rec, separators=(",", ":")), flush=True)
    return rec


def pull(drive: Drive, file_id: str, name: str, md5: str, sha256: str, dest: Path, size: int | None = None) -> dict:
    """One file the runtime uploaded (Mac) -> dest/name, checked against the size, md5 and sha256 the runtime printed.
    Without --size, Drive's reported size is used for the Range requests and still checked by both hashes."""
    if size is None:
        _, _, data = drive._req("GET", f"{API}/{file_id}?fields=size,md5Checksum")
        meta = json.loads(data)
        if meta.get("md5Checksum") != md5:
            raise SystemExit(f"{WHO}: Drive's md5 of {name} is not the one the runtime printed")
        size = int(meta["size"])
    return download(drive, {"files": {name: {"id": file_id, "size": size, "md5": md5, "sha256": sha256}}}, dest)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--pack", required=True)
    s.add_argument("--drive-dir", required=True)
    s.add_argument("--timeout-s", type=float, default=3600)
    m = sub.add_parser("push-map")
    m.add_argument("--pack", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--ids", required=True)
    f.add_argument("--dest", required=True)
    k = sub.add_parser("make-folder")
    k.add_argument("--name", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--src", required=True)
    u.add_argument("--drive-dir-id", required=True)
    u.add_argument("--name", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--file-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--md5", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        stage(Path(a.pack), Path(a.drive_dir), Drive(mac_token()), timeout_s=a.timeout_s)
    elif a.cmd == "push-map":
        print(push_map(Path(a.pack)))
    elif a.cmd == "fetch":
        download(Drive(runtime_token()), json.loads(Path(a.ids).read_text()), Path(a.dest))
    elif a.cmd == "make-folder":
        make_folder(Drive(mac_token()), a.name)
    elif a.cmd == "upload":
        upload_one(Drive(runtime_token()), Path(a.src), a.drive_dir_id, a.name)
    else:
        from tools.p8_common import check_outside
        pull(Drive(mac_token()), a.file_id, a.name, a.md5, a.sha256, check_outside(Path(a.dest).expanduser()), a.size)


if __name__ == "__main__":
    try:
        main()
    except DriveError as e:
        sys.exit(str(e))
