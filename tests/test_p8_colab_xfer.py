import base64
import hashlib
import io
import json
import tarfile

import pytest

from tools import p8_colab_xfer as X


def test_chunk_bodies_single_and_multi():
    one = list(X.chunk_bodies([b"abc"], 1))
    assert one == [{"type": "file", "format": "base64", "content": base64.b64encode(b"abc").decode()}]
    many = list(X.chunk_bodies([b"a", b"b", b"c"], 3))
    assert [m["chunk"] for m in many] == [1, 2, -1]
    assert b"".join(base64.b64decode(m["content"]) for m in many) == b"abc"


def test_fold_parts_leave_out_the_held_out_game():
    assert X.fold_parts("gsw_bkn_2015") == ["common", "gsw_cha_2016", "gsw_sac_2015"]


class FakeRuntime:
    def __init__(self, files):
        self.files, self.puts = files, []

    def request(self, method, path, body=None, headers=None, timeout=600):
        if method == "PUT":
            self.puts.append((path, body))
            return b"{}"
        lo, hi = headers["Range"].split("=")[1].split("-")
        return self.files[path[len("files/"):]][int(lo):int(hi) + 1]


def _run_tar(tmp_path, weights):
    p = tmp_path / "src.tar"
    with tarfile.open(p, "w") as tf:
        for name, data in weights.items():
            ti = tarfile.TarInfo(f"e1_gsw_bkn_2015_seed0/train/weights/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return p.read_bytes()


def test_get_run_checks_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "CHUNK", 100)               # force several Range reads
    w = {"best.pt": b"B" * 250, "last.pt": b"L" * 30}
    tar = _run_tar(tmp_path, w)
    comp = {"fold": "gsw_bkn_2015", "seed": 0, "weights": {k: hashlib.sha256(v).hexdigest() for k, v in w.items()}}
    rt = FakeRuntime({f"{X.REMOTE_RUNS}/e1_gsw_bkn_2015_seed0.tar": tar,
                      f"{X.REMOTE_RUNS}/e1_gsw_bkn_2015_seed0.complete.json": json.dumps(comp).encode()})
    info = X.get_run(rt, "gsw_bkn_2015", 0, tmp_path / "out")
    assert info["bytes"] == len(tar) and set(info["weights_checked"]) == {"best.pt", "last.pt"}
    comp["weights"]["best.pt"] = "0" * 64
    rt.files[f"{X.REMOTE_RUNS}/e1_gsw_bkn_2015_seed0.complete.json"] = json.dumps(comp).encode()
    with pytest.raises(SystemExit):
        X.get_run(rt, "gsw_bkn_2015", 0, tmp_path / "out")
    assert any(p.name.startswith("e1_gsw_bkn_2015_seed0.tar.prev-") for p in (tmp_path / "out").iterdir())


def test_put_corpus_refuses_a_stale_tar(tmp_path):
    for part in ("common", "gsw_cha_2016", "gsw_sac_2015"):
        (tmp_path / f"{part}.tar").write_bytes(part.encode())
    man = {"parts": {p: {"tar": f"{p}.tar", "sha256": hashlib.sha256(p.encode()).hexdigest()}
                     for p in ("common", "gsw_cha_2016", "gsw_sac_2015")}}
    (tmp_path / "pack_manifest.json").write_text(json.dumps(man))
    rt = FakeRuntime({})
    X.put_corpus(rt, "gsw_bkn_2015", tmp_path)
    sent = [p for p, b in rt.puts if b.get("type") == "file"]
    assert sent[0].endswith("pack_manifest.json") and len(sent) == 4
    assert not any("gsw_bkn_2015" in p for p in sent)
    (tmp_path / "common.tar").write_bytes(b"changed")
    with pytest.raises(SystemExit):
        X.put_corpus(FakeRuntime({}), "gsw_bkn_2015", tmp_path)
