"""Auto-label plumbing: IoU matching, teacher/pipeline diff, qualification P/R,
and the disagreement-band triage rule (agreements pre-accepted, disagreements
always adjudicated — LABEL_SCHEMA anti-inbreeding rule 2)."""
from conftest import N_FRAMES  # noqa: F401  (repo-root sys.path side effect)
from adjudicate_labels import disagreement_band
from colab_autolabel import iou, match_sets, pr_vs_gt


def box(x, y, w=10, h=20):
    return [x, y, x + w, y + h]


def test_iou_basics():
    assert iou(box(0, 0), box(0, 0)) == 1.0
    assert iou(box(0, 0), box(100, 100)) == 0.0
    assert 0.3 < iou(box(0, 0), box(5, 0)) < 0.4       # half-overlap in x


def det(x, y, cls="player", score=0.9):
    return {"cls": cls, "box": box(x, y), "score": score}


def test_match_sets_counts():
    teacher = [det(0, 0), det(100, 0), det(200, 0)]     # 3 proposals
    pipeline = [det(1, 0), det(300, 0)]                 # 1 agrees, 1 pipeline-only
    m = match_sets(teacher, pipeline)
    assert m == {"agree": 1, "teacher_only": 2, "pipeline_only": 1}


def test_pr_vs_gt():
    gts = [box(0, 0), box(100, 0)]
    preds = [box(1, 0), box(500, 0)]                    # 1 TP, 1 FP, 1 FN
    m = pr_vs_gt(preds, gts)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_disagreement_band_triage():
    rec = {
        "teacher": [det(0, 0),                          # agrees with pipeline
                    det(100, 0),                        # teacher-only (the FN class!)
                    det(200, 0, cls="rim")],            # non-overlap class → adjudicate
        "pipeline": [det(1, 0),
                     det(300, 0)],                      # pipeline-only → adjudicate
    }
    pre, adj = disagreement_band(rec)
    assert len(pre) == 1 and pre[0]["box"] == box(0, 0)
    kinds = {(d["cls"], d.get("src", "teacher")) for d in adj}
    assert ("player", "teacher") in kinds               # teacher-only kept for Claude
    assert ("rim", "teacher") in kinds                  # non-player classes always judged
    assert ("player", "pipeline_only") in kinds         # pipeline-only kept too
    assert len(adj) == 3                                # nothing contested is dropped


# ── court candidates: geometry proposes, VLM only names ───────────────────────
def test_line_intersections_cross_and_parallel():
    from adjudicate_labels import line_intersections
    cross = [(0, 50, 100, 50), (50, 0, 50, 100)]        # + shape → one point
    pts = line_intersections(cross, 200, 200)
    assert len(pts) == 1 and pts[0] == (50.0, 50.0)
    parallel = [(0, 10, 100, 10), (0, 60, 100, 60)]
    assert line_intersections(parallel, 200, 200) == []


def test_line_intersections_rejects_far_extrapolation():
    from adjudicate_labels import line_intersections
    # segments whose infinite lines cross far beyond both spans → rejected
    segs = [(0, 0, 10, 0), (200, 100, 210, 90)]
    assert line_intersections(segs, 1000, 1000) == []


def test_cluster_points_merges_and_ranks_by_support():
    from adjudicate_labels import cluster_points
    pts = [(10, 10), (12, 11), (11, 9), (500, 500)]
    out = cluster_points(pts)
    assert len(out) == 2
    assert abs(out[0][0] - 11) < 1.5 and abs(out[0][1] - 10) < 1.5  # 3-support first
    assert out[1] == (500.0, 500.0)


# ── v3 free filters: fragments die before the API, absurd classes die after ──
def test_prefilter_drops_contained_fragment_keeps_real_overlap():
    from adjudicate_labels import prefilter
    big = {"cls": "player", "box": [100, 100, 180, 300]}          # full player
    frag = {"cls": "player", "box": [125, 140, 155, 175]}         # chest number
    partial = {"cls": "player", "box": [160, 120, 240, 310]}      # occluding player
    tiny = {"cls": "player", "box": [0, 0, 12, 20]}               # sub-min-area
    ball = {"cls": "ball", "box": [300, 300, 318, 318]}           # small is fine
    kept, rej = prefilter([big, frag, partial, tiny, ball])
    assert big in kept and partial in kept and ball in kept
    reasons = {r["reject"] for r in rej}
    assert reasons == {"prefilter_fragment", "prefilter_min_area"}


def test_sanity_filter_flips_object_class_on_person_shape():
    from adjudicate_labels import sanity_filter
    person_box = [100, 100, 160, 300]                             # tall
    v = sanity_filter(person_box, {"keep": True, "cls": "rim"})
    assert v["keep"] is False and v["sanity"]
    ok = sanity_filter(person_box, {"keep": True, "cls": "player"})
    assert ok["keep"] is True
    wide_box = [100, 100, 300, 160]                               # backboard-ish
    bb = sanity_filter(wide_box, {"keep": True, "cls": "backboard"})
    assert bb["keep"] is True


def test_containment_pass_flips_relabeled_fragment_keeps_held_ball():
    from adjudicate_labels import containment_pass
    pre = [{"cls": "player", "box": [100, 100, 180, 300]}]
    adj = [{"cls": "ball", "box": [125, 140, 155, 172]},   # chest box GDINO called ball
           {"cls": "ball", "box": [130, 200, 158, 228]}]   # actual held ball
    verdicts = {"0": {"keep": True, "cls": "player"},      # judge relabeled → fragment
                "1": {"keep": True, "cls": "ball"}}        # judge kept as ball → stays
    out = containment_pass(adj, verdicts, pre)
    assert out["0"]["keep"] is False and out["0"]["sanity"] == "contained_fragment"
    assert out["1"]["keep"] is True


# ── audit tool: report math (labeling itself is human-driven) ────────────────
def test_audit_report_math():
    from label_audit import make_report
    sample = [{"key": k, "stratum": st} for k, st in
              [("a", "rim"), ("b", "rim"), ("c", "adj_player"),
               ("d", "adj_player"), ("e", "adj_player"), ("f", "scorebug")]]
    labels = {"a": {"verdict": "y"}, "b": {"verdict": "b"},
              "c": {"verdict": "y"}, "d": {"verdict": "a"},
              "e": {"verdict": "u"}}                     # f unaudited
    rep = make_report(sample, labels)
    assert rep["audited"] == 5 and rep["sampled"] == 6
    ov = rep["overall"]
    assert (ov["n_judged"], ov["correct"], ov["attribute_wrong"], ov["unsure"]) == (4, 2, 1, 1)
    assert rep["by_stratum"]["rim"]["accuracy"] == 0.5
    assert rep["by_stratum"]["adj_player"]["attribute_wrong"] == 1
