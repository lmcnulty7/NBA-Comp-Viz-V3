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
