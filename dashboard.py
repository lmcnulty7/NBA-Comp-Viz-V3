#!/usr/bin/env python
"""
dashboard.py — Streamlit skeleton (project item 7: the "Comp Viz" deliverable).

Pure JSON/coordinate rendering — no video decode, no models — safe to run
alongside a harvest queue. Everything reads the pipeline's existing outputs:

  tabs: Possession replay (minimap from trajectory coordinates, frame slider)
        Team defense (tier2_credit table — PLACEHOLDER banner until n≥300)
        Defender comparison (fragment-level scaffold; real names await jersey OCR)
        Pipeline health (cross-val, join checks, exclusion funnel)

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import streamlit as st

import config
from court.court33 import court_ft_to_px, draw_court_topdown

PBP_DIR = config.PROJECT_ROOT / "data" / "pbp"

st.set_page_config(page_title="NBA Comp Viz — defense", layout="wide", page_icon="🏀")


@st.cache_data
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def clips_with_data() -> list[str]:
    return sorted(p.name.replace("_trajectories.json", "")
                  for p in config.TRACKING_DIR.glob("*_trajectories.json")
                  if (config.TRACKING_DIR / f"{p.name.replace('_trajectories.json','')}_possessions.json").exists())


def render_minimap(traj: dict, ident: dict, frame: int, hull_team=None):
    """Court-top-down frame from trajectory coordinates (raw-supported only)."""
    import cv2
    scale, margin = 9.0, 15
    img, _, _ = draw_court_topdown(scale, margin)
    cols = {int(k): tuple(int(x) for x in v) for k, v in (ident or {}).get("team_colors_bgr", {}).items()}
    teams = {k: v for k, v in (ident or {}).get("team_by_track", {}).items()}
    pts_by_team = {0: [], 1: []}
    for tid, rec in traj.items():
        raw_f = {p[0] for p in rec.get("raw", [])}
        for f, x, y, _ in rec["cleaned"]:
            if f == frame and f in raw_f and np.isfinite([x, y]).all():
                team = rec.get("team")
                col = cols.get(team, (160, 160, 160))
                p = court_ft_to_px(np.array((x, y), np.float32), scale, margin)[0].astype(int)
                cv2.circle(img, tuple(p), 7, col, -1, cv2.LINE_AA)
                cv2.putText(img, str(tid), (p[0] + 7, p[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
                if team in (0, 1):
                    pts_by_team[team].append((x, y))
    if hull_team in (0, 1) and len(pts_by_team[hull_team]) >= 3:
        px = court_ft_to_px(np.array(pts_by_team[hull_team], np.float32), scale, margin)
        overlay = img.copy()
        cv2.fillPoly(overlay, [cv2.convexHull(px)], cols.get(hull_team, (120, 120, 120)))
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    return img[:, :, ::-1]   # BGR → RGB for st.image


st.title("NBA Comp Viz — defensive pipeline")
st.caption("gate → tracking → identity → teams → possessions → matchups → PBP outcomes")

tab_replay, tab_credit, tab_defenders, tab_health = st.tabs(
    ["🎬 Possession replay", "🛡 Team defense", "⚔️ Defender comparison", "🩺 Pipeline health"])

# ── possession replay ─────────────────────────────────────────────────────────
with tab_replay:
    clips = clips_with_data()
    if not clips:
        st.info("No processed clips found.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            clip = st.selectbox("Clip / section", clips)
            traj = load_json(str(config.TRACKING_DIR / f"{clip}_trajectories.json"))
            poss = load_json(str(config.TRACKING_DIR / f"{clip}_possessions.json"))
            ident = load_json(str(config.TRACKING_DIR / f"{clip}_identity.json"))
            spans = [s for s in (poss or {}).get("spans", []) if s.get("kind") == "halfcourt"]
            label = lambda s: (f"@{s['set_start_frame']} {s['attacked_basket']} "
                               f"({s.get('set_s', '?')}s set, conf {s.get('confidence', 0)})")
            span = st.selectbox("Possession", spans, format_func=label) if spans else None
            show_hull = st.checkbox("Offense hull", value=True)
        with c2:
            if span and traj:
                f0, f1 = span["set_start_frame"], span["end_frame"]
                frames = sorted({p[0] for r in traj.values() for p in r["raw"]
                                 if f0 <= p[0] <= f1})
                if frames:
                    idx = st.slider("Frame", 0, len(frames) - 1, 0)
                    st.image(render_minimap(traj, ident, frames[idx],
                                            hull_team=span.get("offense_team") if show_hull else None),
                             caption=f"frame {frames[idx]} — offense team {span.get('offense_team')} "
                                     f"@ {span['attacked_basket']} basket",
                             use_column_width=True)
                else:
                    st.warning("No observed frames in this span.")

# ── team defense (placeholder-gated) ─────────────────────────────────────────
with tab_credit:
    credit = load_json(str(PBP_DIR / "tier2_credit.json"))
    if not credit:
        st.info("Run tier2_credit.py first.")
    else:
        if not any(r["meaningful"] for r in credit["rows"]):
            st.warning(f"**PLACEHOLDER** — no defense bucket has reached "
                       f"n≥{credit['min_bucket']} possessions yet. Numbers below verify the "
                       f"pipeline, not defensive quality. ({credit['joined_total']} joined so far.)")
        st.dataframe(credit["rows"], use_container_width=True)
        st.caption(f"baseline: {credit['baseline']} · credit/100 > 0 ⇒ offenses did worse "
                   f"than their own norm on these defended possessions")

# ── defender comparison scaffold (fragment-level until jersey OCR) ────────────
with tab_defenders:
    st.caption("Fragment-level defender profiles — real player names arrive with jersey OCR. "
               "Layout scaffold for the eventual player comparison.")
    join = load_json(str(PBP_DIR / "tier2_join.json"))
    if not join:
        st.info("Run tier2_join.py first.")
    else:
        recs = join["joined"]
        clips_j = sorted({r["clip"] for r in recs})
        colA, colB = st.columns(2)
        for col, side in ((colA, "A"), (colB, "B")):
            with col:
                st.subheader(f"Defender {side}")
                clip = st.selectbox("Section", clips_j, key=f"clip{side}",
                                    index=0 if side == "A" else min(1, len(clips_j) - 1))
                rows = [r for r in recs if r["clip"] == clip]
                frags = sorted({d["fragment"] for r in rows for d in r["defenders"]})
                frag = st.selectbox("Fragment id", frags, key=f"frag{side}")
                entries = [(r, d) for r in rows for d in r["defenders"] if d["fragment"] == frag]
                if entries:
                    tot_t = sum(d["time_assigned_s"] for _, d in entries)
                    med_d = float(np.median([d["matchup_dist_median_ft"] for _, d in entries]))
                    pts = sum(r["outcome"]["points"] for r, _ in entries)
                    st.metric("possessions on floor", len(entries))
                    st.metric("time assigned", f"{tot_t:.1f}s")
                    st.metric("median matchup distance", f"{med_d:.1f} ft")
                    st.metric("points allowed (team, those possessions)", pts)

# ── pipeline health ───────────────────────────────────────────────────────────
with tab_health:
    cv = load_json(str(config.REPORTS_DIR / "tier2_crossval.json"))
    join = load_json(str(PBP_DIR / "tier2_join.json"))
    c1, c2, c3 = st.columns(3)
    if cv:
        c1.metric("possessions aligned", cv["possessions_aligned"])
        rate = cv["offense_cross_validation"]["agreement_rate"]
        c2.metric("PBP cross-validation", f"{rate:.1%}" if rate else "—",
                  help="canary ≥ 90% — offense assignment vs independent play-by-play")
    if join:
        c3.metric("joined possessions", join["checks"]["joined"])
        st.subheader("Exclusion funnel (every drop has a visible reason)")
        from collections import Counter
        reasons = Counter(e["reason"].split(":")[0] for e in join["excluded"])
        st.bar_chart(dict(reasons))
        st.json(join["checks"])
