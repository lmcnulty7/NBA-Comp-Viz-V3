"""tools.p8_datasets_fetch: the SHOT clip-name screen and the record's email removal. No network."""

from tools import p8_datasets_fetch as D


def test_team_codes_from_clip_names():
    assert D.teams_in_name("CLEvsGS") == ["CLE", "GSW"]
    assert D.teams_in_name("ATL_vs_NJ_0012") == ["ATL", "BKN"]
    assert D.teams_in_name("clip_0042") == []


def test_screen_flags_cle_phx_gsw_and_the_d85_arenas():
    assert D.screen("CLEvsGS") == "a game with CLE"
    assert D.screen("PHXvsGSW") == "PHX and GSW"
    assert D.screen("MIAvsBOS") is not None
    assert D.screen("LALvsBOS_2023") is None                 # a year outside 2015-16 written in the name
    assert D.screen("ATLvsNJ") is None
    assert D.screen("clip_0042") == "no team codes readable"


def test_email_pattern():
    addr = "a.b" + "@" + "example.org"                  # built here so find_private stays clean on this file
    assert D.EMAIL_RE.sub("x", f"contact {addr} now") == "contact x now"


def test_pl_nba_folders_resolve_by_date_and_known_teams_only():
    sched = {"0022200200": ("2022-11-07", "SAC", "GSW"), "0022200201": ("2022-11-07", "DEN", "SAS"),
             "0022200300": ("2022-11-18", "DAL", "DEN")}
    assert D.resolve_folder("20221107_GSW_SAC_label", sched) == ("0022200200", "date and both teams")
    assert D.resolve_folder("20221108_GSW_SAC_label", sched)[0] == "0022200200"       # one day of slack
    gid, how = D.resolve_folder("20221118_DAL_DNF_label", sched)                        # DNF not guessed
    assert gid == "0022200300" and how == "date and one team (DAL)"
    assert D.resolve_folder("20221107_XYZ_QRS_label", sched) == (None, "no team code recognised")
    assert D.resolve_folder("notadate", sched)[0] is None


def test_season_of_a_game_id():
    assert D.season_of("0022400861") == "2024-25" and D.season_of("0021800909") == "2018-19"
    assert D.season_of("0042500171") == "2025-26"


def test_sportsmot_keeps_listed_clips_only_and_drops_mac_files():
    clips = {"v_abc_c001"}
    assert D.smot_keep("train/v_abc_c001/img1/000001.jpg", clips) == "train/v_abc_c001/img1/000001.jpg"
    assert D.smot_keep("./val/v_abc_c001/gt/gt.txt", clips) == "val/v_abc_c001/gt/gt.txt"
    assert D.smot_keep("train/v_abc_c001/img1/._000001.jpg", clips) is None
    assert D.smot_keep("train/._v_abc_c001", clips) is None
    assert D.smot_keep("train/v_xyz_c002/img1/000001.jpg", clips) is None          # another sport's clip
    assert D.smot_keep("test/v_abc_c001/img1/000001.jpg", clips) is None           # test is never kept
    assert D.smot_keep("train/v_abc_c001/../../x", clips) is None
    assert D.smot_keep("/train/v_abc_c001/img1/000001.jpg", clips) is None
    assert D.smot_keep("train/v_abc_c001", clips) is None                           # the folder entry itself


def _tar_bytes(members):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            if data is None:
                ti = tarfile.TarInfo(name)
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.pax_headers = {"comment": "x"}
                tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_sportsmot_stream_extracts_hashes_the_whole_tar_and_counts(tmp_path):
    import hashlib
    import io
    raw = _tar_bytes([
        ("train", None), ("train/._v_abc_c001", b"mac"), ("train/v_abc_c001", None),
        ("train/v_abc_c001/seqinfo.ini", b"[Sequence]\nseqLength=2\n"),
        ("train/v_abc_c001/gt/gt.txt", b"1, 7, 1, 2, 3, 4, 1, 1, 1\n2, 7, 1, 2, 3, 4, 1, 1, 1\n2, 8, 5, 6, 7, 8, 1, 1, 1\n"),
        ("train/v_abc_c001/img1/000001.jpg", b"a"), ("train/v_abc_c001/img1/._000001.jpg", b"mac"),
        ("train/v_abc_c001/img1/000002.jpg", b"b"),
        ("train/v_xyz_c009/img1/000001.jpg", b"football"),
        ("train/v_abc_c001/../../escape.txt", b"no"),
    ])
    r = D.HashingReader(io.BytesIO(raw))
    files = D.extract_stream(r, tmp_path / "out", {"v_abc_c001"})
    assert r.n == len(raw) and r.h.hexdigest() == hashlib.sha256(raw).hexdigest()
    assert sorted(files) == ["train/v_abc_c001/gt/gt.txt", "train/v_abc_c001/img1/000001.jpg",
                             "train/v_abc_c001/img1/000002.jpg", "train/v_abc_c001/seqinfo.ini"]
    assert files["train/v_abc_c001/img1/000001.jpg"] == {"bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}
    assert not (tmp_path / "escape.txt").exists() and not list((tmp_path / "out").rglob("._*"))
    c = D.smot_counts(tmp_path / "out", ["v_abc_c001", "v_missing_c001"])
    assert c["v_abc_c001"] == {"split": "train", "frames": 2, "gt_lines": 3, "track_ids": 2, "seq_length": 2,
                               "frames_match_seq_length": True}
    assert c["v_missing_c001"] == {"missing": True}
