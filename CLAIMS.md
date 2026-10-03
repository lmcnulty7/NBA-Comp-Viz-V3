# CLAIMS.md — what this project will and will not claim

The definition of done (Phase 0 of the completion roadmap, 2026-07-16). Every
remaining task must serve a claim below; anything that serves no claim is out
of scope. A claim ships only with its evidence artifact and its caveats — the
project's differentiator is epistemic hygiene, and this file is its contract.

## Tier A — system & validation claims (evidence exists today)

| # | Claim | Evidence | Status |
|---|---|---|---|
| A1 | Vision-derived possession attributions agree with independent play-by-play ground truth **87.8% (n=990)**, with disagreements hard-excluded from all downstream stats by construction | reports/tier2_crossval_corpus.*; tier2_join checks | SHIPPED — **corrected 2026-09-11**: the prior 91.1% (n=642) was run-11's PER-RUN align report promoted to a corpus claim; recomputed corpus-wide it is 87.8% (869/990). Non-duplicate 89.6% (n=904); overlap≥0.80 88.0% (n=933) |
| A2 | Every pipeline stage carries a held-out or human-labeled validation number (gate 98.7%; detection P.89/R.87; homography 0.30 ft median; teams 87.1%; possessions 96.5/91.2%; clock 100%-on-readable) | PIPELINE.md table + eval artifacts | SHIPPED (one gap: A2a) |
| A2a | …except **matchup assignment**, which has structural checks only | — | OPEN → Phase 2 closes it |
| A3 | Stale-artifact consumption is structurally impossible in the credit chain (content-fingerprint guard; refused a real stale join on first deployment) | tier2_credit guard + tests; DEVLOG 07-16 | SHIPPED |
| A4 | OCR read-rate saturates at 720p (30.8%→31.0% at 1080p, controlled A/B); added resolution buys tracking length (~3×) and team-call abstention (53%→11%) instead | DEVLOG 07-12; probe artifacts | SHIPPED |
| A5 | Auto-label accuracy is **measured, not assumed** (LABEL_SCHEMA rule 4, 300 stratified accepted labels, human-judged): the **agreement band is 95.0% [88.8, 97.8] (n=100)** and is the usable dataset; the **Claude-adjudicated band measures 6.7–20.0% (n=200) and is NOT used for training**, along with the attributes riding on it | reports/label_audit.*; data/label_audit/labels.json | SHIPPED 2026-09-12 |
| A6 | The qualification gate protected only classes with an in-house baseline; classes exempted as "nothing to compare against" (rim/backboard/scorebug/ball + attributes) are exactly the ones that failed the audit — recorded as a design defect, not a footnote | reports/label_audit.*; DEVLOG 09-12 | SHIPPED 2026-09-12 |
| A7 | The judge's frame-level shot_type labels were audited before use (200 class-stratified frames, human): closeup-claimed 98.0% [89.5, 99.6] but **wide-claimed only 68.8% [57.9, 77.8]** — ~3 in 10 "wide" frames are closeups — so they are NOT used as gate training data for the wide class | reports/shot_type_audit.*; data/shot_type_audit/labels.json | SHIPPED 2026-10-02 |
| A8 (proposed 2026-10-03, FIX_PLAN B5) | On the held-out game gsw_phx_2016 against SportVU tracking (12.16.2015 PHX at GSW), on the frames where a truth homography could be fitted (1,447 of 9,500 rebuilt window frames, 15%): player position error **p50 7.3 ft, p90 15.4 ft** under the pipeline H, compressed toward mid-court (far players pulled 1.7 ft toward the camera, near players pushed 1.0 ft away); **25.1% of in-frame players have no box within 2.5 ft; 15.3% of boxes are more than 3 ft from any player (referees included)**; team label accuracy **81.8%** (n=11,240); pipeline primary man = SportVU nearest opponent **48.1%** (n=160, a proxy for C1, not the human-labeled matchup) | reports/sportvu_phx.*; reports/sportvu_truth_gsw_phx_2016.json; sportvu/ | PROPOSED, not a shipped claim. Caveats: testable frames only (the untestable 85% are where the pipeline H was too far off to match 6 players, so these numbers are optimistic); truth H derived through the pipeline's own boxes; local windowed rebuild, not the production artifacts (same code, models, h264 source); gate v2 saw phx frames (court, detector, teams, identity did not) |

## Tier B — team-level measurement claims (Phase 1 gates these)

| # | Claim | Requires | Status |
|---|---|---|---|
| B1 | Team-defense credit/100 reported **with cluster-bootstrapped 95% CIs** (by game), never as a point estimate | bootstrap in tier2_credit | SHIPPED 07-16 |
| B2 | Baselines are **leave-sample-out** and **context-matched** (each possession vs its own start-type norm — the audit showed 35% of the sample is live-start) | baseline rework | SHIPPED 07-16 |
| B3 | Funnel selection bias is **quantified**: +8.8/100 residual outcome-shape selection (FT-trip exclusion, made-shot enrichment); period/clock distributions clean. Cross-team comparisons use credit_rel (leave-bucket-out), which differences the common-mode offset out | reports/tier2_bias_audit.* | SHIPPED 07-16 |
| B4 | NEVER claim: "team X's defense is above/below league average." The sample is halfcourt set-cores from a nonrandom slice of games; the claim is always "vs. these opponents' own context-matched norms, on our surviving sample, with this CI" | discipline | STANDING |

## Tier C — player-level claims (capability demo ONLY; Phases 2+3 gate these)

| # | Claim | Requires | Status |
|---|---|---|---|
| C1 | Matchup (primary defender) assignment accuracy **X% vs. human labels (n≥150)** | Phase 2 labeling | OPEN |
| C2 | Jersey-OCR names **Y%** of joined possessions' primary-defender fragments at native 720p | Phase 3 batch + guards (min-reads ≥4, crop-color veto) | OPEN |
| C3 | Player-level credit tables are a CAPABILITY DEMONSTRATION: only GSW players appear in every game, so per-player n ≈ 20–50 — shipped with CIs and an explicit "not conclusions" banner, or not shipped | C1 + C2 + B-machinery | OPEN |

## Kill list (explicitly out of scope — deferred with reasons on record)

- Transition defense (sample is halfcourt set-cores; segmentation treats only set spans)
- Help-position / off-ball attentiveness metrics (unverifiable assumptions; no ball tracking — DEVLOG 07-07 scope decision)
- Ball tracking, shot-quality models
- Closeout tendency as a headline metric (directional only; inherits far-court homography tail)
- League-wide or era-normalized conclusions (2012–2017 pooled, GSW-centric sample)

## Standing caveats that ship with the report

1. **Gate operating point:** the possession harvest ran gate v1 at **0.70** — `harvest_driver.py` calls
   `build_trajectories.py --pregate`, which gates every frame at `thresholds.json["trained"]` (0.70 in every
   commit incl. the harvest commit b751d18). **Provenance corrected 2026-10-02:** docs from 07-05 to 10-02
   said the harvest ran at 0.35; that figure belongs to `label_factory.py` (the court LABEL factory), never to
   the possession harvest. **Measured in-domain (2026-10-02):** on all 6,159 harvest frames, human-verified
   (contact-sheet passes; 3,544 wide): at 0.70 the gate keeps 98.8% precision (37 FP, FP-rate 0.014) but
   drops **14.1% of truly wide frames** (recall 0.859, 500 FN) — the production error is lost yield, not
   contamination. At 0.35 it would have admitted 13.1% of non-wide frames (343; precision 0.912). The original
   objective (min FN s.t. FP-rate <= 0.10) picks 0.42 (recall 0.998, precision 0.933); max accuracy 0.975
   at 0.55. **Retrain-vs-recalibrate decided 2026-10-02** (leave-one-source-out, reports/gate_retrain_experiment.*):
   a head refit on the harvest frames + prototype train/val wins by the adoption rule (acc 0.970 vs 0.957,
   FP-rate 0.068 vs 0.098 at recall 0.998, worst source 0.948 vs 0.920, prototype test 0.975 within v1's
   interval) and ships as **gate v2** (`models/trained_head_v2.*`, threshold 0.33) for all FUTURE harvesting.
   Every published number was produced with v1 at 0.70 and stays so until the games are re-harvested
   under v2, after which the PBP canary and bias audit are re-run.
2. **Funnel yield ~58% span→join** and losses are not random (F3) — B3's audit
   quantifies this; until then no representativeness language.
3. **Identity = track fragment**, not player, everywhere upstream of jersey OCR.
4. **Data provenance:** broadcast video fetched from public YouTube uploads for
   research/education; no video redistributed; repo and report carry derived
   data and fair-use stills only.
