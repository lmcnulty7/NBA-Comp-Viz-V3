# Loop instructions: tracking fixes

You are fixing the court projection and player boxes of the NBA broadcast pipeline and
measuring the result against SportVU tracking on the held-out game gsw_phx_2016, so that
Lucien never has to label frames by hand. The plan is `FIX_PLAN.md`; the record is `FIX_LOG.md`.
Work on branch `track-fix`.

## Each pass
1. Read `FIX_PLAN.md` (including the diagnosis table) and `FIX_LOG.md`.
2. Take the FIRST unchecked item not tagged [HUMAN] or [BLOCKED]. Do only that item.
   An unticked [HUMAN] item is a gate: nothing below it starts until Lucien ticks it
   (the "Waiting on Lucien" section is the only exception). Reaching a gate = stop the loop.
3. Do the item with the repo's tools: `/opt/anaconda3/bin/python`, `PYTORCH_ENABLE_MPS_FALLBACK=1`,
   ffmpeg at `/opt/homebrew/bin/ffmpeg`. Anything that needs a GPU or a full-game rebuild is
   written as a repo script and queued for Colab (one-cell notebook pattern, see `colab_run.py`);
   log it as "queued for Colab" and tick the item only when its artifact exists.
4. Verify (all must pass):
   - `/opt/anaconda3/bin/python -m pytest -q` passes.
   - The item's own measurable from the plan is met and written to the named artifact under
     `reports/` (json + txt).
   - Rejection counts per rule per game are in the artifact for any item that rejects anything.
   - No number changes in CLAIMS.md, PIPELINE.md or the paper without the artifact path and a
     caveat line in the same commit.
   - For Phase C items: SportVU p50 and p90 on gsw_phx_2016 did not get worse, frames with
     positions did not drop more than 25% in any game without SportVU showing they were wrong,
     and `align_outcomes` on gsw_phx_2016 did not lose aligned possessions.
5. If verification passes: tick the item, append 2 to 3 lines to `FIX_LOG.md` (what changed,
   the key number, where to look), commit on `track-fix`.
6. If it fails: fix and re-verify. After 3 failed attempts on one item, tag it
   [BLOCKED: reason], log it, and move on.

## Stop the loop when
- every item is ticked, [HUMAN], or [BLOCKED]; or
- 3 items in a row end [BLOCKED]; or
- a [HUMAN] gate is reached.
On stop, write a summary at the top of FIX_LOG.md: done, blocked, waiting on Lucien, and
the current SportVU numbers if Phase B has run.

## Hard rules
- Never train, tune, calibrate, or choose a threshold on gsw_phx_2016 or on the extra held-out
  game named in D. Thresholds come from the other games or the prototype clips; phx is reported once.
- A fix ships behind a config flag, default off, until Lucien ticks the adoption gate.
- Every acceptance rule logs how many frames it rejected, per rule, per game. A rule that silently
  drops most frames is a failure, not a fix.
- Nothing becomes a claim until measured on gsw_phx_2016. Before that, call it a label-free
  indicator. Every number in a doc carries its artifact path and its caveat (CLAIMS.md discipline).
- Third-party data is never committed: SportVU, Roboflow, SportsMOT, DeepSportRadar live under
  `data/sportvu/` or `data/external/` (gitignored). Download scripts, manifests, and license notes
  are committed.
- Production frame numbers index the h264 harvest sections. Local section files are AV1 copies
  split at different keyframes; use `data/triage/offsets.json` (or re-solve with
  `triage_sheet.solve_offset`) before cutting local frames by production frame number.
- Do not change what an existing number means. The 87.8% canary, P .89 / R .87, 0.30 ft median
  and the gate numbers stay as they are until a re-measure replaces them with its own artifact.
- Do not launch desktop apps or a visible browser from the loop. Contact sheets are jpg files.
- No em dashes in any written output (code comments, docs, logs, commit messages).
