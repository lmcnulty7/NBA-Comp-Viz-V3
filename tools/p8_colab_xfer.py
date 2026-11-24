"""tools/p8_colab_xfer.py: move files between the Mac and a Colab API runtime (D164) over the runtime's HTTP endpoints,
because drive.mount fails on runtimes the Colab API creates ("mount failed", smoke test of 2026-11-24).

The runtime URL and its proxy token come from the environment, COLAB_URL and COLAB_TOKEN (from the Colab MCP tool
get_runtime; the token lasts about an hour). Neither is printed or written anywhere.

    python3 -m tools.p8_colab_xfer put-corpus --fold gsw_bkn_2015 --pack <e1_pack folder>
    python3 -m tools.p8_colab_xfer get-run --fold gsw_bkn_2015 --seed 0 --dest <folder>

put-corpus: guard on the fold's games; the fold's parts (common and its two training games) and pack_manifest.json,
each tar's sha256 checked against the manifest on the Mac first, uploaded to REMOTE_CORPUS in pieces under 50 MB
(the runtime's contents API: "chunk" 1, 2, ..., -1 on the last). On the runtime, tools.p8_e1_colab colab
--corpus-dir REMOTE_CORPUS checks members and bytes again before extracting.

get-run: downloads <run>.tar and <run>.complete.json from REMOTE_RUNS with Range requests, then checks that the
best.pt and last.pt inside the tar have the sha256 values complete.json states. A mismatch is an error.

Endpoints (Colab MCP get_runtime description): upload PUT {url}/api/contents/{path} with JSON type file, format
base64; download GET {url}/files/{path} with Range. Paths have no leading slash.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.p8_common import guard
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_colab_xfer"
REMOTE_CORPUS = "content/p8_e1_in/corpus"
REMOTE_RUNS = "content/p8_e1_out"
CHUNK = 32 << 20                                       # raw bytes per request; base64 makes it about 43 MB
RETRIES = 4


def _ssl_context():
    import ssl
    try:
        import certifi                                 # python.org builds ship no CA bundle
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class Runtime:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.environ.get("COLAB_URL", "")).rstrip("/")
        self.token = token or os.environ.get("COLAB_TOKEN", "")
        if not self.url or not self.token:
            raise SystemExit(f"{WHO}: set COLAB_URL and COLAB_TOKEN (from the Colab get_runtime tool)")
        self.ctx = _ssl_context()

    def request(self, method: str, path: str, body=None, headers=None, timeout: int = 600) -> bytes:
        h = {"X-Colab-Runtime-Proxy-Token": self.token, **(headers or {})}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        r = urllib.request.Request(f"{self.url}/{path}", data=data, method=method, headers=h)
        with urllib.request.urlopen(r, timeout=timeout, context=self.ctx) as resp:
            return resp.read()


def chunk_bodies(data_iter, n: int):
    """Contents API bodies for n pieces: no chunk key for a single piece, else 1, 2, ..., and -1 for the last."""
    for i, b in enumerate(data_iter):
        body = {"type": "file", "format": "base64", "content": base64.b64encode(b).decode()}
        if n > 1:
            body["chunk"] = -1 if i == n - 1 else i + 1
        yield body


def _retry(fn, what: str):
    for attempt in range(RETRIES):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == RETRIES - 1:
                raise
            print(f"{WHO}: retry {what}: {repr(e)[:120]}", flush=True)
            time.sleep(5 * (attempt + 1))


def mkdirs(rt: Runtime, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        try:
            rt.request("PUT", f"api/contents/{'/'.join(parts[:i])}", {"type": "directory"})
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise


def put(rt: Runtime, local: Path, remote: str) -> dict:
    size = local.stat().st_size
    n = max(1, -(-size // CHUNK))
    t0 = time.time()
    with open(local, "rb") as f:
        pieces = (f.read(CHUNK) for _ in range(n))
        for i, body in enumerate(chunk_bodies(pieces, n)):
            _retry(lambda: rt.request("PUT", f"api/contents/{remote}", body), f"{local.name} piece {i + 1}/{n}")
    return {"file": local.name, "bytes": size, "seconds": round(time.time() - t0, 1)}


def get(rt: Runtime, remote: str, local: Path) -> dict:
    t0, pos = time.time(), 0
    tmp = local.with_name(local.name + ".part")
    with open(tmp, "wb") as f:
        while True:
            try:
                b = _retry(lambda: rt.request("GET", f"files/{remote}",
                                              headers={"Range": f"bytes={pos}-{pos + CHUNK - 1}"}), f"{remote} at {pos}")
            except urllib.error.HTTPError as e:
                if e.code == 416 and pos > 0:          # the size was an exact multiple of CHUNK
                    break
                raise
            f.write(b)
            pos += len(b)
            if len(b) < CHUNK:
                break
    tmp.replace(local)
    return {"file": local.name, "bytes": pos, "seconds": round(time.time() - t0, 1)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fold_parts(fold: str) -> list:
    from tools.p8_e1_colab import fold_games
    return ["common", *fold_games(fold)]


def put_corpus(rt: Runtime, fold: str, pack: Path) -> list:
    from tools.p8_e1_colab import GAMES
    refuse_cv_test(list(GAMES), WHO)
    guard(list(GAMES), WHO)
    man = json.loads((pack / "pack_manifest.json").read_text())
    parts = fold_parts(fold)
    for part in parts:                                 # a stale or damaged local tar never leaves the Mac
        rec = man["parts"][part]
        if sha256(pack / rec["tar"]) != rec["sha256"]:
            raise SystemExit(f"{WHO}: {rec['tar']} does not match pack_manifest.json")
    mkdirs(rt, REMOTE_CORPUS)
    done = [put(rt, pack / "pack_manifest.json", f"{REMOTE_CORPUS}/pack_manifest.json")]
    for part in parts:
        done.append(put(rt, pack / man["parts"][part]["tar"], f"{REMOTE_CORPUS}/{man['parts'][part]['tar']}"))
        print(json.dumps(done[-1]), flush=True)
    return done


def check_run_tar(tar_path: Path, complete: dict) -> dict:
    """sha256 of each weights file inside the result tar against complete.json -> {name: sha}; SystemExit if any
    stated file is missing or differs."""
    found = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in complete["weights"] and Path(m.name).parent.name == "weights":
                h = hashlib.sha256()
                src = tf.extractfile(m)
                for b in iter(lambda: src.read(1 << 20), b""):
                    h.update(b)
                found[name] = h.hexdigest()
    bad = [k for k, v in complete["weights"].items() if found.get(k) != v]
    if bad:
        raise SystemExit(f"{WHO}: weights in {tar_path.name} differ from complete.json: {bad}")
    return found


def get_run(rt: Runtime, fold: str, seed: int, dest: Path) -> dict:
    from tools.p8_e1_colab import run_name
    name = run_name(fold, seed)
    dest.mkdir(parents=True, exist_ok=True)
    for p in (dest / f"{name}.tar", dest / f"{name}.complete.json"):
        if p.exists():                                 # never overwrite a result already on the Mac
            p.rename(p.with_name(f"{p.name}.prev-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    get(rt, f"{REMOTE_RUNS}/{name}.complete.json", dest / f"{name}.complete.json")
    info = get(rt, f"{REMOTE_RUNS}/{name}.tar", dest / f"{name}.tar")
    complete = json.loads((dest / f"{name}.complete.json").read_text())
    info["weights_checked"] = check_run_tar(dest / f"{name}.tar", complete)
    return info


def main(argv=None) -> None:
    from tools.p8_e1_colab import GAMES
    ap = argparse.ArgumentParser(prog=f"python3 -m {WHO}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("put-corpus")
    p.add_argument("--fold", required=True, choices=GAMES)
    p.add_argument("--pack", required=True)
    g = sub.add_parser("get-run")
    g.add_argument("--fold", required=True, choices=GAMES)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    rt = Runtime()
    if a.cmd == "put-corpus":
        put_corpus(rt, a.fold, Path(a.pack))
    else:
        print(json.dumps(get_run(rt, a.fold, a.seed, Path(a.dest)), indent=1))


if __name__ == "__main__":
    main()
