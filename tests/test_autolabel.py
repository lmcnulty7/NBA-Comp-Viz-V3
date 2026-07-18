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
