#!/usr/bin/env python
"""
build_numbers.py — Court Ledger portfolio page, build step 1 (brief §E.1).

Reads the committed report / data artifacts and emits numbers.json with every
headline number the page consumes, so nothing on the page is hand-typed unless
it is listed under "hand_typed_sources" with its source.

READ-ONLY with respect to the repo. Never imports a pipeline entry point that
writes. Run:
  cd "$R" && PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/anaconda3/bin/python \
      /tmp/portfolio_scout/build_numbers.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parents[1]        # repo root
OUT = Path(__file__).resolve().parent / "numbers.json"


def load(rel: str):
    p = R / rel
    with open(p) as f:
        return json.load(f)


def game_for_clip_factory():
    """Prefer the repo's own fetch_pbp.game_for_clip; fall back to an identical
    re-implementation if the import fails (it pulls requests + bs4)."""
    sys.path.insert(0, str(R))
    os.chdir(R)
    try:
        import fetch_pbp  # noqa: WPS433 — module-level code has no side effects beyond logging config
        return fetch_pbp.game_for_clip, "fetch_pbp.game_for_clip"
    except Exception as e:  # pragma: no cover
        reg = load("data/harvest/games.json")
        clip_game = {
            "curry_q1_clip": "201602270OKC",
            "curry_classic_clip": "201602270OKC",
            "clip_10m00_18m00": "201206070BOS",
            "clip_26m00_34m00": "201206070BOS",
            "clip_40m00_48m00": "201206070BOS",
            "clip_55m00_63m00": "201206070BOS",
            "clip_70m00_78m00": "201206070BOS",
        }

        def g(stem: str) -> str:
            if stem in clip_game:
                return clip_game[stem]
            tag = re.sub(r"_s\d+$", "", stem)
            return reg[tag]["game_code"]

        return g, f"fallback re-implementation (import failed: {e!r})"


def rate(records) -> dict:
    """Exact copy of tier2_crossval_corpus.rate."""
    agree = sum(1 for r in records if r.get("offense_agrees") is True)
    disagree = sum(1 for r in records if r.get("offense_agrees") is False)
    unknown = sum(1 for r in records if r.get("offense_agrees") is None)
    checked = agree + disagree
    return {"agree": agree, "disagree": disagree, "unknown": unknown,
            "n_checked": checked,
            "agreement_rate": round(agree / checked, 3) if checked else None}


def main() -> None:
    n: dict = OrderedDict()
    n["_meta"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "portfolio/build_numbers.py",
        "repo_root": str(R),
        "rule": "every value under a top-level key other than hand_typed_sources was read from the "
                "named artifact at build time; nothing here is typed by hand",
    }

    # ------------------------------------------------------------------ 1. cross-val (corpus)
    cv = load("reports/tier2_crossval_corpus.json")
    occ = cv["offense_cross_validation"]
    n["crossval"] = {
        "source": "reports/tier2_crossval_corpus.json",
        "scope": cv["scope"],
        "clips": cv["clips"],
        "possessions_aligned": cv["possessions_aligned"],
        "agree": occ["agree"],
        "disagree": occ["disagree"],
        "unknown": occ["unknown"],
        "n": occ["n_checked"],
        "rate": occ["agreement_rate"],
        "rate_pct": round(occ["agreement_rate"] * 100, 1),
        "non_duplicate": cv["non_duplicate"],
        "overlap_ge_080": cv["overlap_ge_080"],
        "anchor_failures": cv["anchor_failures"],
        "failure_reasons": cv["failure_reasons"],
    }
    # per-run artifact (the 86.0% ledger step) — read, never promoted
    cvr = load("reports/tier2_crossval.json")
    n["crossval_per_run_artifact"] = {
        "source": "reports/tier2_crossval.json (PER-RUN; ledger step 2 only — never a corpus figure)",
        "possessions_aligned": cvr["possessions_aligned"],
        "agree": cvr["offense_cross_validation"]["agree"],
        "disagree": cvr["offense_cross_validation"]["disagree"],
        "n": cvr["offense_cross_validation"]["agree"] + cvr["offense_cross_validation"]["disagree"],
        "rate": cvr["offense_cross_validation"]["agreement_rate"],
        "rate_pct": round(cvr["offense_cross_validation"]["agreement_rate"] * 100, 1),
    }

    # ------------------------------------------------------------------ 1b. per-game cross-val (recomputed)
    game_for_clip, gfc_src = game_for_clip_factory()
    files = sorted(glob.glob(str(R / "data/pbp/*_outcomes.json")))
    by_game: dict[str, list] = {}
    aligned_all = []
    for f in files:
        stem = Path(f).name[: -len("_outcomes.json")]
        game = game_for_clip(stem)
        with open(f) as fh:
            rows = json.load(fh)
        al = [r for r in rows if r.get("status") == "aligned"]
        aligned_all.extend(al)
        by_game.setdefault(game, []).extend(al)
    recomputed = rate(aligned_all)
    assert recomputed == occ, f"recompute mismatch: {recomputed} vs {occ}"
    per_game = {g: rate(v) for g, v in sorted(by_game.items())}
    rates = {g: v["agreement_rate"] for g, v in per_game.items() if v["agreement_rate"] is not None}
    gmin = min(rates, key=rates.get)
    gmax = max(rates, key=rates.get)
    n["crossval_per_game"] = {
        "source": f"recomputed from data/pbp/*_outcomes.json ({len(files)} files) grouped by {gfc_src}; "
                  "same rate() definition as tier2_crossval_corpus.py; corpus total re-derived and asserted equal",
        "n_games": len(per_game),
        "min": {"game": gmin, **per_game[gmin]},
        "max": {"game": gmax, **per_game[gmax]},
        "range_str": f"{rates[gmin]:.3f}–{rates[gmax]:.3f}",
        "per_game": per_game,
    }

    # ------------------------------------------------------------------ 2. bias audit
    ba = load("reports/tier2_bias_audit.json")

    def pop(k):
        p = ba[k]
        return {"n": p["n"], "ppp": p["ppp"], "points_mix": p["points_mix"],
                "start_type": p["start_type"]}

    n["bias_audit"] = {
        "source": "reports/tier2_bias_audit.json",
        "games": ba["games"],
        "population_all": pop("population_all"),
        "population_halfcourt": pop("population_halfcourt"),
        "population_live": pop("population_live"),
        "sampled_joined": pop("sampled_joined"),
        "selection_offset_per_100": ba["selection_offset_per_100"],
        "derived": {
            "joined_ppp_vs_halfcourt_ppp": f"{ba['sampled_joined']['ppp']:.3f} vs {ba['population_halfcourt']['ppp']:.3f}",
            "three_plus_share_joined_vs_halfcourt_pct": [round(ba["sampled_joined"]["points_mix"]["3+"] * 100, 1),
                                                         round(ba["population_halfcourt"]["points_mix"]["3+"] * 100, 1)],
            "one_pt_share_joined_vs_halfcourt_pct": [round(ba["sampled_joined"]["points_mix"]["1"] * 100, 1),
                                                     round(ba["population_halfcourt"]["points_mix"]["1"] * 100, 1)],
            "joined_live_start_share_pct": round(ba["sampled_joined"]["start_type"]["live"] * 100, 1),
        },
        "note": ba["note"],
    }

    # ------------------------------------------------------------------ 3. label audit (F5 chart data)
    la = load("reports/label_audit.json")
    strata = []
    for name, s in la["by_stratum"].items():
        strata.append({"stratum": name, "n_judged": s["n_judged"], "correct": s["correct"],
                       "accuracy": s["accuracy"], "wilson95": s["wilson95"],
                       "band": "agreement" if name == "agreement_player" else "judge_adjudicated"})
    strata.sort(key=lambda s: -s["accuracy"])
    adj = [s for s in strata if s["band"] == "judge_adjudicated"]
    n["label_audit"] = {
        "source": "reports/label_audit.json",
        "sampled": la["sampled"], "audited": la["audited"],
        "overall": {k: la["overall"][k] for k in ("n_judged", "correct", "accuracy", "wilson95")},
        "overall_note": "aggregate mixes bands — not a meaningful number",
        "agreement_band": next(s for s in strata if s["band"] == "agreement"),
        "judge_adjudicated_bands": {
            "n_total": sum(s["n_judged"] for s in adj),
            "accuracy_min": min(s["accuracy"] for s in adj),
            "accuracy_max": max(s["accuracy"] for s in adj),
            "range_pct_str": f"{min(s['accuracy'] for s in adj)*100:.1f}–{max(s['accuracy'] for s in adj)*100:.1f}%",
        },
        "f5_chart_rows": strata,
    }
    # audit crops S9/S10 — verify strata + verdict from the untracked sidecars
    try:
        sample = load("data/label_audit/sample.json")
        labels = load("data/label_audit/labels.json")
        crops = {}
        for idx in (92, 6):
            row = sample[idx]
            crops[f"{idx:04d}.jpg"] = {"key": row["key"], "cls": row["cls"], "stratum": row["stratum"],
                                       "tag": row["tag"], "frame": row["frame"],
                                       "verdict": labels.get(row["key"], {}).get("verdict"),
                                       "verdict_gloss": {"b": "box_or_class_wrong", "a": "attribute_wrong",
                                                         "c": "correct", "u": "unsure"}}
        n["label_audit"]["crops"] = {"source": "data/label_audit/sample.json + labels.json (untracked, read-only)",
                                     **crops}
    except Exception as e:  # pragma: no cover
        n["label_audit"]["crops"] = {"error": repr(e)}

    # ------------------------------------------------------------------ 4. homography
    hm = load("reports/homography_metrics.json")
    n["homography"] = {
        "source": "reports/homography_metrics.json",
        "field_names": ["val_frames", "homography_success_rate", "reproj_error_ft.{mean,median,p90}",
                        "keypoint_pixel_error.{mean,median}", "per_keypoint.<name>.{gt_count,detected_count,mean_px_err}", "note"],
        "val_frames": hm["val_frames"],
        "homography_success_rate": hm["homography_success_rate"],
        "reproj_error_ft": hm["reproj_error_ft"],
        "median_ft_str": f"{hm['reproj_error_ft']['median']:.2f}",
        "median_in": round(hm["reproj_error_ft"]["median"] * 12, 1),
        "p90_ft_str": f"{hm['reproj_error_ft']['p90']:.2f}",
        "keypoint_pixel_error": hm["keypoint_pixel_error"],
        "note": hm["note"],
        "warning": "the 'p50 1.7 px on 279 validation frames' benchmark is NOT in this file — see hand_typed_sources",
    }

    # ------------------------------------------------------------------ 5. gate
    m = load("reports/metrics.json")
    th = m["operating_points"]["trained_head"]
    zs = m["operating_points"]["zero_shot_clip"]
    hsv = m["operating_points"]["hsv_baseline"]
    n["gate"] = {
        "source": "reports/metrics.json",
        "field_names": ["test_size", "test_live", "test_dead", "operating_points.{trained_head,zero_shot_clip,hsv_baseline}."
                        "{n,accuracy,macro_f1,confusion.{tp,tn,fp,fn},false_negatives_live_skipped,"
                        "false_positives_deadball_processed,threshold}", "config_record.counts.{live,dead}",
                        "config_record.split_sizes", "config_record.clip_model"],
        "n_test": th["n"],
        "test_live": m["test_live"], "test_dead": m["test_dead"],
        "accuracy": th["accuracy"], "accuracy_pct": round(th["accuracy"] * 100, 1),
        "macro_f1": th["macro_f1"], "macro_f1_str": f"{th['macro_f1']:.3f}",
        "fn": th["confusion"]["fn"], "fp": th["confusion"]["fp"],
        "threshold_validated": th["threshold"], "threshold_validated_str": f"{th['threshold']:.2f}",
        "zero_shot_clip_accuracy_pct": round(zs["accuracy"] * 100, 1),
        "hsv_baseline_accuracy_pct": round(hsv["accuracy"] * 100, 1),
        "labeled_total": m["config_record"]["counts"]["live"] + m["config_record"]["counts"]["dead"],
        "labeled_live": m["config_record"]["counts"]["live"],
        "labeled_dead": m["config_record"]["counts"]["dead"],
        "split_sizes": m["config_record"]["split_sizes"],
        "backbone": m["config_record"]["clip_model"],
        "head": m["config_record"]["head"],
        "warning": "the harvest operating point (0.70 = threshold_validated; provenance verified DEVLOG 10-02e) and the in-domain gate-sheet metrics are NOT in this file — see hand_typed_sources",
    }

    # ------------------------------------------------------------------ 6. possessions / clock / detection
    pe = load("reports/poss_eval.json")
    n["possessions"] = {
        "source": "reports/poss_eval.json",
        "field_names": ["n_labeled", "attacked_basket.overall.{n,accuracy}", "attacked_basket.by_kind.{span_mid,span_start,pre_start}",
                        "offense.overall.{n,accuracy}", "offense.by_kind", "transition_predictions", "unclear_rates"],
        "n_labeled": pe["n_labeled"],
        "basket_accuracy": pe["attacked_basket"]["overall"]["accuracy"],
        "basket_accuracy_pct": round(pe["attacked_basket"]["overall"]["accuracy"] * 100, 1),
        "basket_n": pe["attacked_basket"]["overall"]["n"],
        "offense_accuracy": pe["offense"]["overall"]["accuracy"],
        "offense_accuracy_pct": round(pe["offense"]["overall"]["accuracy"] * 100, 1),
        "offense_n": pe["offense"]["overall"]["n"],
        "basket_by_kind": pe["attacked_basket"]["by_kind"],
        "offense_by_kind": pe["offense"]["by_kind"],
        "unclear_rates": pe["unclear_rates"],
        "warning": "brief's '100% / 100% more than 2 s from span boundaries' is not a field here: by_kind span_start/pre_start "
                   "are 1.0/1.0 for basket but 0.917/0.833 for offense — the '2 s from boundary' slice comes from the paper; "
                   "see hand_typed_sources",
    }
    ce = load("reports/clock_eval.json")
    n["clock"] = {
        "source": "reports/clock_eval.json",
        "field_names": list(ce.keys()),
        **ce,
        "accuracy_on_readable_pct": round(ce["accuracy_on_readable"] * 100, 1),
        "n_readable": ce["n"] - ce["human_unreadable"],
    }
    tm = load("reports/tracking_metrics.json")
    i5 = tm["metrics_by_iou"]["iou_0.5"]
    n["detection"] = {
        "source": "reports/tracking_metrics.json",
        "field_names": ["frames_evaluated", "total_gt_boxes", "total_pred_boxes", "detector.{weights,conf,iou_nms}",
                        "metrics_by_iou.iou_{0.3,0.5,0.7}.{precision,recall,f1,tp,fp,fn}", "note"],
        "frames_evaluated": tm["frames_evaluated"],
        "total_gt_boxes": tm["total_gt_boxes"],
        "iou_0.5": i5,
        "f1_str": f"{i5['f1']:.2f}", "precision_str": f"{i5['precision']:.2f}", "recall_str": f"{i5['recall']:.2f}",
        "detector_conf": tm["detector"]["conf"], "detector_iou_nms": tm["detector"]["iou_nms"],
        "note": tm["note"],
        "warning": "the pre-fine-tune baseline (F1 0.71, P 0.68, R 0.74) is NOT in this file — see hand_typed_sources",
    }

    # ------------------------------------------------------------------ 7. team assignment (curry_q1_clip eval)
    te = load("reports/teams_eval_curry_q1_clip.json")
    n["teams"] = {
        "source": "reports/teams_eval_curry_q1_clip.json",
        "field_names": ["n_labeled", "track_level.{accuracy,n_tracks}", "crop_level.{accuracy_identifiable,n_identifiable}", "abstained.n"],
        "n_labeled": te["n_labeled"],
        "track_level_accuracy": te["track_level"]["accuracy"],
        "track_level_accuracy_pct": round(te["track_level"]["accuracy"] * 100, 1),
        "track_level_n_tracks": te["track_level"]["n_tracks"],
        "crop_level_accuracy": te["crop_level"]["accuracy_identifiable"],
        "crop_level_accuracy_pct": round(te["crop_level"]["accuracy_identifiable"] * 100, 1),
        "crop_level_n_identifiable": te["crop_level"]["n_identifiable"],
        "abstained_n": te["abstained"]["n"],
    }

    # ------------------------------------------------------------------ 8. credit
    cr = load("data/pbp/tier2_credit.json")
    gsw = next(r for r in cr["rows"] if r["defense"] == "GSW")
    assert gsw["meaningful"] is True
    n_meaningful = sum(1 for r in cr["rows"] if r["meaningful"])
    n["credit"] = {
        "source": "data/pbp/tier2_credit.json",
        "min_bucket": cr["min_bucket"],
        "joined_total": cr["joined_total"],
        "n_rows": len(cr["rows"]),
        "n_meaningful_rows": n_meaningful,
        "gsw": {k: gsw[k] for k in ("defense", "n_possessions", "n_games", "ppp_allowed", "baseline_ppp",
                                     "credit_per_100", "ci95", "ci_excludes_zero", "credit_rel_per_100",
                                     "baseline_naive_ppp", "meaningful")},
        "gsw_str": f"{gsw['credit_per_100']:+.1f} [{gsw['ci95'][0]:+.1f}, {gsw['ci95'][1]:+.1f}]".replace("+", "+").replace("-", "−"),
        "baseline_def": cr["baseline"],
        "ci_def": cr["ci"],
        "credit_rel_def": cr["credit_rel"],
        "note": cr["note"],
        "policy": "only the meaningful=true row is drawn; sub-gate rows are never rendered (brief §H.5)",
    }
    assert n_meaningful == 1, "exactly one bucket should clear the n>=300 gate"

    # ------------------------------------------------------------------ 9. record card + disagreement card
    oc = load("data/pbp/gsw_cle_2017f_g5_s03_outcomes.json")

    def card(frame):
        r = next(x for x in oc if x["set_start_frame"] == frame)
        return OrderedDict(
            clip="gsw_cle_2017f_g5_s03",
            set_start_frame=r["set_start_frame"],
            core_end_frame=r["core_end_frame"],
            period=r["period"],
            anchors_raw=[a["raw"] for a in r["anchors"]],
            anchors_frame=[a["frame"] for a in r["anchors"]],
            anchors_clock_s=[a["clock_s"] for a in r["anchors"]],
            anchors_str=" → ".join(a["raw"] for a in r["anchors"]),
            clock_s_str=" → ".join(str(a["clock_s"]) for a in r["anchors"]),
            offense_real=r["offense_real"],
            defense_real=r["defense_real"],
            pbp_offense_real=r["pbp_offense_real"],
            offense_agrees=r["offense_agrees"],
            pbp_possession=r["pbp_possession"],
            offense_points=r["offense_points"],
            terminating_event=r["terminating_event"],
            status=r["status"],
            confidence=r["confidence"],
            overlap_frac_of_span=r["overlap_frac_of_span"],
            attacked_basket=r["attacked_basket"],
            offense_team=r["offense_team"],
            orientation_flipped=r.get("orientation_flipped"),
            orientation_conf=r.get("orientation_conf"),
        )

    n["record_card_f33816"] = {"source": "data/pbp/gsw_cle_2017f_g5_s03_outcomes.json", **card(33816)}
    n["disagreement_card_f21990"] = {"source": "data/pbp/gsw_cle_2017f_g5_s03_outcomes.json", **card(21990)}
    assert n["record_card_f33816"]["offense_agrees"] is True
    assert n["disagreement_card_f21990"]["offense_agrees"] is False

    # ------------------------------------------------------------------ 10. matchups span 729 (C5) + curry_q1 span 0 (C6)
    mj = load("data/tracking/gsw_hou_duel_s01_matchups.json")
    sp = next(p for p in mj["possessions"] if p["set_start_frame"] == 729)
    n["matchup_span_729"] = {
        "source": "data/tracking/gsw_hou_duel_s01_matchups.json → possessions[set_start_frame==729]",
        "field_names": ["set_start_frame", "core_start_frame", "core_end_frame", "core_s", "attacked_basket", "offense_team",
                        "defense_team", "confidence", "coverage_pairs_per_frame", "frames_used", "frames_excluded_team_gt5",
                        "pct_excluded", "degraded", "team_count_distribution", "spacing_conceded_ft2", "defenders[]"],
        "clip": mj["clip"],
        "set_start_frame": sp["set_start_frame"],
        "core_start_frame": sp["core_start_frame"], "core_end_frame": sp["core_end_frame"], "core_s": sp["core_s"],
        "offense_team": sp["offense_team"], "defense_team": sp["defense_team"],
        "attacked_basket": sp["attacked_basket"],
        "confidence": sp["confidence"],
        "coverage_pairs_per_frame": sp["coverage_pairs_per_frame"],
        "frames_used": sp["frames_used"],
        "frames_excluded_team_gt5": sp["frames_excluded_team_gt5"],
        "pct_excluded": sp["pct_excluded"],
        "degraded": sp["degraded"],
        "n_defenders": len(sp["defenders"]),
        "team_count_distribution": sp["team_count_distribution"],
    }
    # the period / pbp string for this possession live in the outcomes file
    try:
        hou_oc = load("data/pbp/gsw_hou_duel_s01_outcomes.json")
        hr = next(x for x in hou_oc if x["set_start_frame"] == 729)
        n["matchup_span_729"]["outcomes_row"] = {
            "source": "data/pbp/gsw_hou_duel_s01_outcomes.json",
            "period": hr.get("period"), "offense_real": hr.get("offense_real"), "status": hr.get("status"),
            "terminating_event_desc": (hr.get("terminating_event") or {}).get("desc"),
            "offense_agrees": hr.get("offense_agrees"), "confidence": hr.get("confidence"),
        }
    except Exception as e:  # pragma: no cover
        n["matchup_span_729"]["outcomes_row"] = {"error": repr(e)}

    cq = load("data/tracking/curry_q1_clip_matchups.json")
    s0 = next(p for p in cq["possessions"] if p["set_start_frame"] == 0)
    n["matchup_curry_q1_span_0"] = {
        "source": "data/tracking/curry_q1_clip_matchups.json → possessions[set_start_frame==0] (C6 banner-count target)",
        "core_start_frame": s0["core_start_frame"], "core_end_frame": s0["core_end_frame"],
        "frames_used": s0["frames_used"], "frames_excluded_team_gt5": s0["frames_excluded_team_gt5"],
        "pct_excluded": s0["pct_excluded"], "coverage_pairs_per_frame": s0["coverage_pairs_per_frame"],
    }

    # corpus coverage range, recomputed over every *_matchups.json (for comparison with the DEVLOG '2.5–4.0')
    cov = []
    for f in sorted(glob.glob(str(R / "data/tracking/*_matchups.json"))):
        with open(f) as fh:
            for p in json.load(fh)["possessions"]:
                cov.append((p["coverage_pairs_per_frame"], bool(p.get("degraded"))))
    nd = [c for c, d in cov if not d]
    nd_sorted = sorted(nd)

    def pct(v, q):
        return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]

    n["matchup_coverage_corpus"] = {
        "source": f"recomputed from data/tracking/*_matchups.json ({len(set(glob.glob(str(R / 'data/tracking/*_matchups.json'))))} files)",
        "n_possessions": len(cov),
        "n_non_degraded": len(nd),
        "non_degraded_min": min(nd), "non_degraded_max": max(nd),
        "non_degraded_p10": pct(nd_sorted, 0.10), "non_degraded_median": pct(nd_sorted, 0.5), "non_degraded_p90": pct(nd_sorted, 0.90),
        "warning": "the DEVLOG's '2.5–4.0 pairs/frame' is a typical-band statement, not min/max — the raw range is wider; "
                   "quote 2.5–4.0 only with its DEVLOG source",
    }

    # ------------------------------------------------------------------ 11. matchup labeling attempts (untracked sidecars)
    try:
        l1 = load("data/matchup_eval/labels_discarded_20260716_2343.json")
        l2 = load("data/matchup_eval/labels.json")
        c1 = Counter(v["verdict"] for v in l1.values())
        c2 = Counter(v["verdict"] for v in l2.values())
        n["matchup_labeling"] = {
            "source": "data/matchup_eval/labels_discarded_20260716_2343.json + labels.json (untracked, read-only)",
            "attempt_1_discarded_n": len(l1), "attempt_1_verdicts": dict(c1),
            "attempt_2_n": len(l2), "attempt_2_verdicts": dict(c2),
            "attempt_2_str": f"{len(l2)}: {c2.get('n', 0)} negative, {c2.get('y', 0)} positive, {c2.get('u', 0)} unsure",
        }
    except Exception as e:  # pragma: no cover
        n["matchup_labeling"] = {"error": repr(e)}

    # ------------------------------------------------------------------ 12. constants read from source
    hold = None
    src = (R / "render_overlay_video.py").read_text()
    mm = re.search(r"^HOLD_S\s*=\s*([0-9.]+)", src, re.M)
    if mm:
        hold = float(mm.group(1))
    n["constants"] = {"HOLD_S": hold, "HOLD_S_source": "render_overlay_video.py"}

    # ------------------------------------------------------------------ 13. hand-typed sources
    n["hand_typed_sources"] = OrderedDict([
        ("_rule", "figures below are NOT in any JSON artifact on disk; the page may show them only with the listed "
                  "source in the evidence ledger (brief §E.1, §H)"),
        ("corpus_games_16", "16 games — reports/tier2_bias_audit.json has games:16 (so this one IS in JSON); listed for clarity"),
        ("hero_eyebrow_years_2012_2017", "fetch_pbp.py GAMES dates run 2012-06-07 (201206070BOS) … 2017-06-12; the page derives the span from crossval_per_game keys at build time. PROJECT_SUMMARY.md and paper §2/§5/§9 say 2013–2017 — wrong by a season, flagged to the owner"),
        ("gate_harvest_in_domain_0_70_recall_0_859_precision_0_988", "reports/gate_sheet.txt FULL CORPUS @0.70; harvest ran at the validated 0.70 (DEVLOG 10-02e corrected the earlier 0.35 claim)"),
        ("detection_baseline_f1_0_71_p_0_68_r_0_74", "paper §4 / DEVLOG (pre-fine-tune YOLOv8m baseline)"),
        ("detection_external_dataset_654_images_10_to_5_classes", "paper §4 / DEVLOG"),
        ("detection_train_time_30min_T4_vs_22h_local", "DEVLOG"),
        ("detection_negative_results_conf_0_40_to_0_25_recall_0_744_to_0_764", "paper §10 / DEVLOG"),
        ("detection_negative_results_nms_0_45_0_70_no_change", "paper §10 / DEVLOG"),
        ("detection_negative_results_yolov8x_recall_0_695", "paper §10 / DEVLOG"),
        ("identity_impossible_steps_15_4_to_14_3_pct", "DEVLOG (label-free diagnostic)"),
        ("identity_churn_6_57_to_5_0", "DEVLOG (label-free diagnostic)"),
        ("homography_pixel_benchmark_p50_1_7px_279_val_frames", "paper §4 (snapped-mapping targets, not human labels) — MUST stay a separate sentence"),
        ("homography_hand_labeled_400_frames_20_point_scheme", "paper §4"),
        ("homography_unseen_games_66_89_pct_sane_mapping", "data/court_review / DEVLOG"),
        ("c7_render_track_79_pct", "HUD-color scan of 1998_CHIatUTA render (scout notes; re-verify on the cut)"),
        ("line_seg_dice_0_587", "paper §4 (train_line_seg.py)"),
        ("team_misassignment_13_pct_and_gt5_hard_exclusion", "paper §5 (derived from teams_eval 87.1%)"),
        ("team_cluster_crops_gsw_cle_2017f_g5_s03_1596_1979_84", "reports/viz/team_clusters_gsw_cle_2017f_g5_s03.png title / DEVLOG — no JSON found"),
        ("possessions_100_100_more_than_2s_from_boundaries", "paper §5 (poss_eval.json only has by_kind: basket span_start 1.0 / pre_start 1.0; offense 0.917 / 0.833)"),
        ("possessions_basket_93_0_before_ghost_fix", "DEVLOG 2026-07-07"),
        ("possessions_boundary_fuzz_1_2s", "paper §5"),
        ("clock_27_crops_7_layouts", "reports/viz/anchor_truth_sheet.png / DEVLOG"),
        ("alignment_9_of_9_offense_crossval", "DEVLOG"),
        ("orientation_crossval_69_4_to_93_7", "DEVLOG"),
        ("anchor_failure_10_pct_per_possession", "paper §6 (248/1239 ≈ 20% of spans by the corpus JSON; the '~10% per possession, retryable' phrasing is the paper's)"),
        ("matchup_coverage_2_5_4_0_pairs_per_frame", "DEVLOG 2026-07-07 (ghost audit); raw corpus range is wider — see matchup_coverage_corpus"),
        ("matchup_coverage_inflated_4_5_before_ghost_fix", "DEVLOG 2026-07-07"),
        ("ghost_points_20_30_pct_interpolated", "DEVLOG 2026-07-07"),
        ("banner_counts_104_43_50_60", "DEVLOG 2026-07-07"),
        ("credit_first_figure_minus_4_5", "DEVLOG 2026-07-16c"),
        ("credit_bootstrap_2000_resamples", "data/pbp/tier2_credit.json 'ci' string (IS in JSON as prose)"),
        ("funnel_10_12_9_10_7_9_58_pct_7_7", "paper §6 (audited sample)"),
        ("reproducibility_151_151_byte_identical_colab", "DEVLOG 2026-07-17"),
        ("reproducibility_51_of_151_changed_75_49_13_86_6_pct_92_min", "DEVLOG 2026-09-11 / paper §6.4"),
        ("ledger_91_1_n_642_retracted_2026_09_11", "CLAIMS.md A1 history / DEVLOG 2026-09-11 (run-11 alignment report)"),
        ("ledger_86_0_n_264", "reports/tier2_crossval.json (IS in JSON — see crossval_per_run_artifact)"),
        ("audit_6197_frames_21_sources", "DEVLOG 2026-09-12 / CLAIMS A6"),
        ("audit_6159_judged_10_80_usd_41_failures", "DEVLOG 2026-09-12 (Batches API run)"),
        ("audit_37916_box_agreement_corpus", "DEVLOG 2026-09-12"),
        ("audit_4694_jersey_reads_withdrawn", "DEVLOG 2026-09-12"),
        ("audit_rule_4_300_stratified", "LABEL_SCHEMA.md / label protocol (label_audit.json sampled:300 confirms n)"),
        ("gdino_p_0_70_r_0_94_vs_0_956_0_967_27_vs_15_fn_197_min", "DEVLOG 2026-07-18 (different matcher than tracking_metrics.json)"),
        ("keypoint_mps_loss_11_4_11_5_to_7_7_11_epochs", "DEVLOG 2026-06-24"),
        ("kp_review_worst_7_28_ft", "reports/viz/kp_review_worst.png title"),
        ("stale_join_guard_day_one", "DEVLOG 2026-07-16"),
        ("ocr_1080p_30_8_to_31_0_abstention_53_to_11", "DEVLOG 2026-07-12"),
        ("mlops_52_tests_15_colab_runs", "pytest --collect-only -q tests → 52 collected (2026-09-12); DEVLOG / PROJECT_SUMMARY still say 45 (stale)"),
        ("paper_18_pages", "paper/main.pdf has 19 pages (pypdf; main.log '19 pages'); the commit message said 18. The page reads the count at build time"),
        ("matchup_labels_38_then_33_23_6_4", "DEVLOG 2026-07-17b / paper §10.4 (ALSO recomputed from untracked data/matchup_eval — see matchup_labeling)"),
        ("hold_s_2s", "render_overlay_video.py HOLD_S (IS read from source — see constants)"),
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    OUT.write_text(json.dumps(n, indent=1, ensure_ascii=False))
if __name__ == "__main__":
    main()
