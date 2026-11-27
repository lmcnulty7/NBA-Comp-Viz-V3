"""tools.p8_recallcause_aligndry: the two alignment rules and the per-game comparison, on synthetic records."""

from tools import p8_recallcause_aligndry as A


def test_rules_and_ties():
    assert A.strict_ok(0.8, 1.0, 0.7) and not A.strict_ok(1.0, 1.0, 0.7)          # a tie reads not aligned
    assert A.proposed_ok(1.0, 1.0, 0.7) and A.proposed_ok(1.0, 0.96, 1.0)           # ties and small gaps aligned
    assert not A.proposed_ok(0.98, 0.72, 0.6) and not A.proposed_ok(0.7, 0.74, 0.6)  # beaten by a margin; under floor
    for t in ((0.5, 0.95, 0.6), (0.2, 0.9, 0.89)):                                    # strict implies proposed
        assert A.strict_ok(*t) and A.proposed_ok(*t)
    assert A.fails_on(1.0, 1.0, 0.7) == "tie" and A.fails_on(0.5, 0.85, 0.5) == "at_under_bar"
    assert A.fails_on(1.0, 0.95, 0.5) == "neighbour_above" and A.fails_on(0.5, 0.95, 0.5) is None


def _a3(byo, mis):
    return {"by_offset": {k: ({"Sk": sk, "S0": s0, "n": 9} if sk is not None else None) for k, (sk, s0) in byo.items()},
            "misaligned": mis}


def test_measure_counts():
    a3 = {"games": {"gsw_cha_2016": {"windows": {
        "gsw_cha_2016_s00_w00": _a3({"-1": (1.0, 1.0), "0": (1.0, 1.0), "1": (0.9, 1.0)}, False),   # tie: strict fails
        "gsw_cha_2016_s00_w01": _a3({"-1": (0.8, 0.95), "0": (0.95, 0.95), "1": (0.7, 0.95)}, False),
        "gsw_cha_2016_s00_w02": _a3({"-1": (0.95, 0.6), "0": (0.6, 0.6), "1": (0.5, 0.6)}, True)}},
        "gsw_bkn_2015": {"windows": {}}, "gsw_sac_2015": {"windows": {"gsw_sac_2015_s00_w00": _a3({}, False)}}}}
    rc = {"windows_dropped": {"gsw_cha_2016_s00_w00": 7},
          "alignment": {"gsw_cha_2016_s00_w00": {"shares": [(1.0, 1.0, 0.9)] * 5, "frames": 5},
                        "gsw_cha_2016_s00_w01": {"shares": [(0.8, 0.95, 0.7)] * 5, "frames": 5}},
          "units": {"both__pooled": {"misses": 13, "groups": {"C1": {"count": 10}}},
                    "both__gsw_cha_2016": {"misses": 13}}}
    out = A.measure(rc, a3)
    g = out["games"]["gsw_cha_2016"]
    assert (g["dropped"], g["dropped_misses"], g["dropped_a3align_aligned"], g["kept_a3align_misaligned"]) == (1, 7, 1, 0)
    assert g["own_shares_proposed"] == {"dropped_now_kept": 1, "their_misses": 7, "kept_now_dropped": 0}
    d = out["a3align_dry_run"]["gsw_cha_2016"]
    assert (d["strict_fails_on_aligned"], d["ties_on_aligned"], d["proposed_fails_on_aligned"]) == (1, 1, 0)
    assert (d["strict_flags_misaligned"], d["proposed_flags_misaligned"]) == (1, 1)
    assert out["games"]["gsw_bkn_2015"]["dropped"] == 0
    assert out["dropped_by_reason"]["tie"] == {"windows": 1, "misses": 7, "share_of_dropped_misses": 1.0}
    assert out["dropped_alignment_frames"] == {"5": 1} and g["dropped_share_of_misses"] == 7 / 20
    assert out["c1_worst_case_both"]["c1_if_no_dropped_miss_is_c1"] == 0.5
    assert set(out["a3align_dry_run"]) == {"gsw_cha_2016", "gsw_bkn_2015"}                # sac stays apart
