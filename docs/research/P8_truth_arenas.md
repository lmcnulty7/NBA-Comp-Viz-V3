# P8: Mac camera truth for NYK, OKC and POR, design before any build

Written 2026-11-25, before any build, for queue item 5 of docs/research/TRACKING_LEVELS.md. Version 2, the same day
and still before any build, applies the lead's decisions (section 0); version 3, also before any build, applies the
ml-review of 06d7e60b (section 12). Where versions differ, the latest holds. Design only: nothing in
this document has been built, and nothing here is a result. The three games are development games (D85 (3)); no
held-out game is read and no held-out number is computed (CLAUDE.md rule 4).

## 0. Version 2 (2026-11-25, after the lead's decisions; still before any build)

**Layouts (rule as worded in version 3).** A game runs only on a layout that is either in the frozen pipeline code
(clock_reader.LAYOUTS, from before P8) or found by the D152 finder, and in either case accepted through the frozen
automatic check at the registered bar (MIN_LAYOUT_SHARE 0.25). New hand calibrations made as P8 work (msg15_720 and
csn_nw15_720, 2026-11-17) are not used, under D128 and D152 (a box measured by eye is a hand step). Decision (Lucien,
2026-11-25): msg15_720 and csn_nw15_720 stay out, so nyk_bos_2016 and por_okc_2016 are not used. The plan is
okc_sac_2015 only.

The finder's answer for both games is already on record, from the D152 validation run (P8 workspace
layout_find/validation.json, P8_baseline.md):

| Game | Finder proposal | Frozen check of the proposal | Under version 2 |
| --- | --- | --- | --- |
| nyk_bos_2016 | no_period: a clock box (seen in 21 of 36 frames) but no period token on the clock's row (MSG puts the period under the clock) | not reached | dropped |
| por_okc_2016 | no_clock: the best clock group was seen in 9 frames, under the 25 percent share | not reached | dropped |

A rerun would not change this. The finder is deterministic: fixed frame positions ((k + 0.25) / per of each middle
section) and easyocr on CPU, with no random step. And `tools/p8_layout_find.py find` refuses every game while the
validation has not passed (3 of 8 passed, D152: "no new game is looked at"). A rerun would need that guard lifted,
which would change a D152 rule, and it would give the same two answers. Its lock time, if the lead still wants it, is
about 2 to 4 minutes a video: the validation job held the lock 1,846 s for 8 videos, each with a proposal and a
control check. So nyk_bos_2016 and por_okc_2016 are dropped on the existing record, with no lock time. nyk_bos_2016's
time maps made with msg15_720 are not used.

okc_sac_2015 runs on fso15_720 (registry), which passes the frozen automatic check on this video (plausible 0.475,
consistent 0.35, the D152 control run). fso15_720 was also measured by hand (clock_reader.py, on pixel-ruler crops,
2026-10-04, ROADMAP R1.2), but it is frozen pipeline code from before P8, and it is accepted for okc_sac_2015 through
the frozen automatic check, the route hou_cle_2016's layout took. For the record: the finder's own proposal for okc_sac_2015 also passed (plausible 0.70,
consistent 0.65) and is not used, per the decision.

**What runs.** okc_sac_2015 alone: about 3.5 to 5.2 lock hours for truth and 5.9 to 8.5 with its r23 baseline
(section 6). After it, development truth covers three non-sealed arenas (GSW, HOU, OKC), not five. Sections 3, 6 and 7
below are kept as written; their nyk_bos_2016 and por_okc_2016 parts no longer run.

**Platform mix: a separate queue item, ahead of using cross-arena truth for any tuning.** The truth of the GSW games
and hou_cle_2016 came from MPS label builds. Before any tuning reads truth across arenas, measure how much truth moves
when it is rebuilt from CPU-detector builds, on one game first:

- Game: gsw_cha_2016, the cheapest: 18,083 processed frames on its label windows (M9's count of the same windows),
  about 2.2 lock hours of label builds plus 0.15 h for the fit, in the fresh reference root (its sections, moments,
  time maps and mirror linked as for the R1.3 games). M9's CPU builds of this game used the r23 court model, so they
  cannot stand in for the V3 label builds that truth comes from.
- Compared: the existing MPS truth against the CPU truth of the same game, from each fit's own files.
- "Moved", fixed now, any one of: the median reprojection after refinement differs by 1.0 px or more (good fits span
  5.47 to 7.65 px across games, so 1.0 px is a large share of the spread between games); accepted camera-truth frames
  differ by 10 percent or more of the MPS count; the median absolute difference of the per-window frozen offsets
  (labels/<game>_offsets.json, windows with an offset in both) is 0.2 s or more. Why 0.2 s for this median (version
  3): the truth v2 measures use 0.2 s as a per-window threshold (the share of windows moving by more), not as a bar
  on a median, so it is borrowed here as a round size of a material shift, not a measured noise floor. The only
  evidence on device noise alone is the D99 detector-only pair, 0.04 s on one window (s02_w03: MPS 0.90 s against
  detector on CPU 0.94 s); the court-model pairs of P8_truth_v2.md (median 0.52 s) differ by more than the device.
- Moved: rebuild the truth of gsw_bkn_2015, gsw_sac_2015 and hou_cle_2016 on the CPU detector too (about 2.5, 2.3 and
  2.9 lock hours of label builds at M6 and M9 frame counts; gsw_ind_2016 stays apart, D98), before any cross-arena
  tuning. Not moved: the MPS truth stays in use beside the CPU truth, with the platform recorded per game.
- The comparison needs a small report-only tool, written and reviewed before it runs. Nothing registered is rescored.

**Check C code (written, not run).** `tools/p8_truth_v2.py measure-dev TAG` runs the same per-window measures as the
registered `measure` on any development game outside the registration, after the project guard. It refuses the four
registered games and writes truth_v2/dev/<tag>.json, which the registered `report` never reads. `tools/p8_truth_accept.py
TAG` applies checks A, B and C with the section 5 bars, gives the use verdict (use, apart, unused), and records and
enforces the truth platform from the device record of every label window and every r23 rebuild window (version 3). Tests: tests/test_p8_truth_accept.py.

## 1. Why

Mac camera truth exists today for the four GSW games and hou_cle_2016 only. Level 1 step 7 (cross-arena accuracy on
development games) therefore has one non-GSW arena to tune and check on. nyk_bos_2016, okc_sac_2015 and por_okc_2016
would add three arenas (New York, Oklahoma City, Portland), five non-sealed arenas in all.

The method is the one the GSW games and hou_cle_2016 used: the frozen Phase A steps run through `tools/p8_frozen.py`
(split, moments, sync, direction, mirror, label, fit, rebuild), driven one job at a time by `tools/p8_night.py` under
the heavy-job lock (D85 (4), D140, D146). The only change is the platform: every build runs with the player detector
on CPU (`P8_DEVICE_DETECTOR=cpu`, court and gate unchanged), the reference platform of D175.

## 2. What exists for each video today

Checked 2026-11-25 from reports/research/p8_registry.json and the P8 workspace (read only).

| | nyk_bos_2016 | okc_sac_2015 | por_okc_2016 |
| --- | --- | --- | --- |
| SportVU game | 01.12.2016.BOS.at.NYK | 12.06.2015.SAC.at.OKC | 01.10.2016.OKC.at.POR |
| Identity | confirmed by automatic OCR (D130, D138 calibration run, P8_step0.md) | same | same (after the D138 parser fix for a spelled period) |
| sportvu/splits.json | not listed (the file lists only the R1.3 broadcast item set) | not listed | not listed |
| Guard | `refuse_cv_test` and `tools.p8_common.guard` pass on tag, SportVU id, video id and home team (run for this design) | pass | pass |
| D85 test set | not in it (home NYK, not MIA, TOR, LAC or LAL) | not in it | not in it |
| Whole video | 6,180 s | 6,811 s | 5,679 s |
| Sections | 11, h264 1280x720, 30000/1001 fps | 12, same format | 10, same format |
| SportVU moments | fetched | fetched | fetched |
| Scorebug layout | msg15_720, measured by hand on a frame on 2026-11-17 (before D128) | fso15_720, a registered layout (Fox Sports Ohio, same geometry); passes the frozen automatic check and the D152 layout finder | csn_nw15_720, measured by hand on 2026-11-17; period written as a word, read by `tools.p8_frozen.make_reader`; fails the frozen check (plausible share under 0.25) |
| OCR time maps | 11 of 11, done 2026-11-18 with msg15_720; 1,856 s of running clock mapped; s00 and s01 have 0 good reads (the layout misses the period there), so the first period has no time map | none | none |
| Direction builds | 4 of 11 sections, built on MPS 2026-11-18 | none | none |
| Label builds, fit, r23 rebuild | none (51 label windows planned, 1,787 s of video) | none | none |

"Training game" here means a development game under D85 (3): not a spent held-out game of sportvu/splits.json (the
guard's `is_heldout_game` check), not a D85 test game, video or arena. None of the three is in splits.json at all.

## 3. Two decisions for the lead before any build (resolved in version 2, section 0)

1. **Hand-measured layouts (nyk_bos_2016, por_okc_2016).** D128 rules out hand labelling of any kind, and D152 ruled
   out measuring a new layout by hand for nop_gsw_2015. msg15_720 and csn_nw15_720 were measured on 2026-11-17, before
   D128, and nyk_bos_2016's time maps were made with msg15_720 the next day. Whether those two existing layouts may
   feed camera truth is not settled by any D line. Options: (a) accept the two pre-D128 layouts, recorded as such
   (all three games run); (b) refuse them (okc_sac_2015 runs alone, and the other two wait for a layout finder that
   passes its own validation, which D152's did not on these two broadcasts). This design runs under either; the order
   in section 7 says what each choice runs.
2. **Platform mix in development truth.** The existing Mac truth of the GSW games and hou_cle_2016 was built from MPS
   label builds, which carry the clip_boxes fault (D174) and fewer player boxes (P8_PLATFORM.md). Truth for the three
   new games would come from CPU-detector label builds. Cross-arena comparisons would then mix truth platforms. This
   design records the platform per game; rebuilding the existing games' truth on the reference platform (about 2 to 3
   lock hours a game for the label builds, by the rates below) is a separate queue item, not part of this one.

## 4. What is built, and how

Workspace: a fresh P8 root for reference-platform truth (`V3_P8_ROOT`, for example the P8 data folder's work_ref), as
`tools/p8_device.check_cached` requires ("use a fresh V3_P8_ROOT for another device setting"). It holds symbolic links
to the existing section videos and SportVU moments (read only, no copy). nyk_bos_2016's 11 time maps are linked too,
with their sync done markers copied, because the OCR ran on CPU (clock_reader keeps easyocr off MPS), so they do not
depend on the device. Its 4 MPS direction builds stay in the old workspace untouched and are built again in the new
one. The old workspace and every number scored in it stay as recorded.

Per video, in this order (each step a job of `tools/p8_night.py run --games <tag> --b4 none`, with
`P8_DEVICE_DETECTOR=cpu` and the fresh `V3_P8_ROOT` in the environment; each heavy job holds the lock for its own
duration only, and the device flag reaches every build through `tools.p8_device`'s proxy):

1. sweep, split (done or linked), moments (linked)
2. layout: skipped, the registry fixes a layout for all three (subject to decision 1)
3. sync, one job a section (okc_sac_2015 and por_okc_2016; nyk_bos_2016's are linked)
4. direction, one job a section: the 40 s direction windows built on the reference platform; then mirror (the frozen
   vote; it must resolve)
5. label, one job a section: every running span cut and built with gate v2 (V3 court model)
6. fit: the frozen camera position, per-frame camera truth and acceptance (autolabel phases B and C)
7. rebuild r23, one job a section: the scoring baseline build on the same windows (also needed for check C below)
8. scores: r23 and V3 on camera truth (the runner's SCORES); no B4 score (`--b4 none`, it is only for the gate games)

Every window writes its device record, which must show the detector on CPU; a window without it does not count.

## 5. Acceptance checks, fixed now

A game's truth is accepted for development use only when all three hold. The bars come from games already built, and
none of the three new games has been seen.

| Check | Bar | Why this bar |
| --- | --- | --- |
| A. Registration | a complete fit (result "partial" False), camera fit status ok and median reprojection after refinement (`camera.reproj_px_median_after`) at most 10.0 px | accepted Mac fits: gsw_sac_2015 5.47, hou_cle_2016 6.40, gsw_bkn_2015 6.52, gsw_cha_2016 7.65 px; the fit D98 found broken, gsw_ind_2016 on MPS, 18.35 px; 10.0 sits above the worst good fit and well below the broken one |
| B. Coverage | a complete fit, at least 1,000 accepted camera-truth frames, and at least 1,500 s of running clock mapped by the time maps | accepted frames on good fits: 1,366 (gsw_cha_2016) to 4,427 (hou_cle_2016); the broken fit had 970. Mapped clock (sum of mapped_video_s over each game's sync time maps): 2,308 (gsw_cha_2016) to 2,635 s (hou_cle_2016) on complete games; nyk_bos_2016 has 1,856 s with a lost quarter |
| C. Offset stability (D166, the frozen offset rule) | between the V3 label build and the r23 rebuild of each window (windows with at least 20 usable frames), median absolute offset difference at most 0.54 s and share over 0.2 s at most 0.7945, unrounded, over at least 0.8 of the windows measured (version 3) | the worst in-use pooled game of P8_truth_v2.md: median 0.54 s (gsw_cha_2016), share 0.7945 (gsw_sac_2015); the apart game gsw_ind_2016 (0.60 s, 0.7978) is not a bar. Registered records use 87 percent or more of their measured windows. hou_cle_2016 has no truth_v2 record. The frozen rule moves 73 to 80 percent of windows by more than 0.2 s even on good games, so C catches only gross instability |

A game that fails A or C is reported apart and flagged, as gsw_ind_2016 is (D98), and is not used to tune anything. A
game that fails B is not used. Version 3: a game is also not used unless every label window and every r23 rebuild
window has a device record whose detector resolved to cpu (D175); `tools/p8_truth_accept.py` enforces it. No bar, constant or fit setting changes after a game's numbers are seen (D98); a later
change would be a new registration.

Check C needs code: `tools/p8_truth_v2.py measure` accepts only its four registered games. Version 2 adds the
report-only `measure-dev` and `tools/p8_truth_accept.py` (section 0). Checks A and B read the fit's own result file.

## 6. Expected lock hours (estimates from measured rates)

Rates measured on this Mac: CPU-detector builds 0.434 to 0.446 s a processed frame (M6, M9; 0.44 used); 8.62 to 8.87
processed frames a second of label-window video (gsw_bkn_2015, hou_cle_2016; 8.75 used); OCR 0.34 to 0.80 s a 1 Hz
read (nyk_bos_2016 0.34, hou_cle_2016 0.46, GSW games 0.72 to 0.88); fit about 0.15 h; scores about 0.2 h a game.
Label-window video for okc_sac_2015 and por_okc_2016 is estimated as 30 to 42 percent of the whole video (nyk_bos_2016
0.29 with a lost quarter, hou_cle_2016 0.41); direction windows as about 38 s a section (hou_cle_2016 457 s over 12).

| | nyk_bos_2016 | por_okc_2016 | okc_sac_2015 |
| --- | --- | --- | --- |
| Label-window video | 1,787 s (planned, 51 windows) | 1,700 to 2,390 s (estimate) | 2,040 to 2,860 s (estimate) |
| Sync (OCR) | 0 (linked) | 0.5 to 1.3 h | 0.6 to 1.5 h |
| Direction builds | 0.35 h | 0.4 h | 0.5 h |
| Label builds | 1.9 h | 1.8 to 2.6 h | 2.2 to 3.1 h |
| Fit | 0.15 h | 0.15 h | 0.15 h |
| Truth subtotal | about 2.4 h | about 2.9 to 4.4 h | about 3.5 to 5.2 h |
| r23 rebuild and scores | about 2.1 h | about 2.0 to 2.8 h | about 2.4 to 3.3 h |
| Total | about 4.5 h | about 4.9 to 7.2 h | about 5.9 to 8.5 h |

All three: about 8.8 to 12.0 lock hours for truth, about 15 to 20 with the r23 baselines. No Colab units.

## 7. Order (version 1; under version 2 only okc_sac_2015 runs)

Smallest first, among the games decision 1 allows: nyk_bos_2016, then por_okc_2016, then okc_sac_2015. Under option
(b) of decision 1, only okc_sac_2015 runs. por_okc_2016 starts with a probe: the sync of its first two sections (about
0.2 h), because its word-period reader has never mapped a whole section; it goes on only if the probe passes the first
kill criterion's per-section form below. One game at a time; the next starts after the previous game's acceptance
checks are written.

## 8. Kill criteria (each checked at the earliest step that can show it)

- Before any job: the guard and refuse_cv_test fail, or the registry status is not confirmed: the game does not run.
- After sync: under 1,500 s of running clock mapped for the game (check B's bar), stop before any model build. For the
  por_okc_2016 probe: if neither of its first two sections maps any running clock, stop por_okc_2016.
- After direction: the frozen mirror vote does not resolve (status not ok): stop the game.
- During label: more than 10 percent of a game's windows fail to build (gate-empty windows are not failures), or a
  game's lock time passes 1.5 times its upper estimate in section 6: stop and diagnose before going on.
- After fit and rebuild: checks A, B and C of section 5 decide how the truth is used; a failure is recorded and nothing
  is retuned.

## 9. What this enables

- Level 1 step 7: cross-arena tuning and checking on development truth in five non-sealed arenas (GSW, HOU, NYK, OKC,
  POR) instead of two, with the D85 test arenas (MIA, TOR, LAC and LAL home games) untouched.
- Per-arena r23 baseline rows on the reference platform, beside hou_cle_2016's, so a change (E1R, E1b, a court model)
  can be read across arenas before the single D85 score.
- More arenas for the court model's SportVU-derived labels (Level 1 step 7's first item), from games it never trained
  on.

## 10. What the CUDA check's verdict would change

If CUDA matches CPU (docs/research/P8_cuda_check.md; its verdict covers the production player detector with the
frozen V3 court and gate models on the two D99 windows, on a T4, and it reports one window built with the r23 court
model beside the verdict), the label and rebuild builds, about 75 to 85 percent of the lock hours above, could run on
Colab instead of the Mac. The r23 rebuild is covered only by that one report-only window, so moving it would rest on
weaker evidence than moving the label builds. The OCR sync, the cuts, the fit and the scores would stay on the Mac (the fit reads
the window clips; OCR runs on CPU either way). The cost moves from lock hours to units and transfer: a game's label
clips are about 1.6 GB (hou_cle_2016's 93 clips), staged through Drive with `tools/p8_drive_xfer.py`. If CUDA matches
neither or is not repeatable, these builds stay on the Mac as planned here.

## 11. What is not done

No held-out game is read or scored, nothing is tuned on these games before they pass section 5, and the D85 test set is
not touched. No D line is written by this design; decisions 1 and 2 are the lead's.

## 12. Version 3 (2026-11-25, after the ml-review of 06d7e60b, PASS; before any build)

No okc_sac_2015 build has run. Each item is applied above and in `tools/p8_truth_accept.py`:

1. Check C's bars are the worst in-use pooled game, not the apart game: median at most 0.54 s (gsw_cha_2016), share
   over 0.2 s at most 0.7945 (gsw_sac_2015), unrounded. The frozen rule moves 73 to 80 percent of windows by more than
   0.2 s even on good games, so C catches only gross instability. hou_cle_2016 has no truth_v2 record.
2. C needs n at least 0.8 of the windows measured (registered records: 87 percent or more).
3. A and B need a complete fit (the result's "partial" False).
4. The verdict is "unused" unless every label window and every r23 rebuild window resolves the detector to cpu.
5. The layout rule is worded as: in the frozen pipeline code or found by the D152 finder, each accepted through the
   frozen check; P8's own hand calibrations of 2026-11-17 are not used. Decision (Lucien, 2026-11-25): they stay out,
   and nyk_bos_2016 and por_okc_2016 are not used; the plan is okc_sac_2015 only. fso15_720 was measured by hand in
   2026-10 as pipeline code before P8.
6. Check B's mapped-clock range is 2,308 to 2,635 s, from the sync time maps.
7. The "moved" offset bar's justification is reworded (0.2 s is a per-window threshold in truth v2, borrowed here as a
   round size; the only device-noise evidence is the D99 detector-only pair, 0.04 s on one window).
8. tools/p8_truth_v2.py's docstring says measure-dev runs under the heavy-job lock.

## 13. Version 4 (2026-11-26, PLAN.md D180; committed before the second OCR run)

**What happened.** okc_sac_2015 ran on fso15_720 in the fresh reference root and stopped at the first kill criterion:
445 s of running clock mapped against the 1,500 s bar (2,933 of 6,589 reads ok, 0.445; about 152 s mapped per 1,000 ok
reads, against 545 to 555 on the games that ran). No model build ran; the 12 sync jobs held the lock 2,294 s. The
frozen check had already shown fso15_720 weak on this video (consistent 0.35).

**The exception (D180, the lead's, with this disclosure).** The D152 finder's own proposal for okc_sac_2015 is tried
once. It is automatic (no hand step), and it passed the frozen automatic check at the registered bar in the D152
validation record before this kill (plausible 0.70, consistent 0.65, 40 pairs). Disclosed: the finder failed its
overall validation (3 of 8), so this is not a general clearance for it; the choice was made after fso15_720 failed the
sync kill; nothing scored depends on it yet.

**Conditions, fixed now.**

1. The layout is registered from the validation record exactly as recorded, with its provenance, by
   `tools/p8_layout_find.py register-validated okc_sac_2015 --decision D180`: registry layout auto_okc_sac_2015, clock
   box [942, 586, 1008, 624], period box [828, 586, 879, 624], period as digits; games.okc_sac_2015.layout set to it.
   The command accepts only this game under D180, and only a proposal whose recorded frozen check chose it; the `find`
   guard (no new game while the validation has not passed) stays in place.
2. The fso15_720 time maps and their sync done markers are moved aside in the reference root (kept, not deleted), and
   the sync runs again from scratch with the new layout. Same platform, same kill criteria, same acceptance bars A, B
   and C (sections 5 and 12), unchanged.
3. If this layout also maps under 1,500 s, okc_sac_2015 is unused and the matter is closed: no third layout.

Lock cost of the second sync: about 0.65 h (the first took 2,294 s).

## 14. Result (2026-11-26): okc_sac_2015 closed at the mirror vote

**What ran.** The D180 retry of section 13: registry layout auto_okc_sac_2015, the fresh reference root, the detector
on CPU, one heavy-job lock per step, kill criteria checked between steps by the driver. Log: the reference root's
logs/okc_truth.log; mirror record: the reference root's reports/sportvu_direction_okc_sac_2015.json.

**Sync passed check B's clock bar.** 12 sections, 6,589 clock reads, 3,860 ok (0.586), 105 running spans, 1,792 s of
running clock mapped (1,791.6 unrounded) against the 1,500 s bar. Per section: s00 and s01 map 0 s (6 and 9 good
reads of about 580), s02 to s10 map 55.0 to 280.0 s, s11 (a short tail) 3.0 s. The fso15_720 attempt of section 13
had mapped 445 s.

**Direction did not resolve, so the game stops (section 8).** 13 direction windows built in s02 to s10 (40 s of running
clock a section, the frozen budget); 12 could vote. The frozen vote (sportvu/direction.py, VOTE_SHARE 0.8, on the far
statistic) returned status "ambiguous", leading mirror identity, vote share 0.67 (8 of 12 windows: identity 8, flip_y
3, flip_xy 1, flip_x 0). Median distance by mirror: identity 6.42 ft, flip_y 6.94, flip_xy 22.68, flip_x 22.91; far
statistic: identity 8.93 ft, flip_y 11.49, flip_x 32.41, flip_xy 35.44; offsets from -0.26 to 1.50 s. As at the R1.2
games, the y-flip is the weak axis: identity and flip_y sit within about 0.5 ft on the median. Section 8 fixed in
advance: "After direction: the frozen mirror vote does not resolve (status not ok): stop the game." No label build,
fit or rebuild ran. Nothing is retuned and there is no third try: no new layout, no second direction budget, no
change to VOTE_SHARE, and the leading identity vote is not adopted by hand.

**Lock time.** This retry held the lock 4,306 s (1.20 h): sync 2,584 s in 12 jobs, direction 1,700 s in 12 jobs, mirror
22 s. The first sync on fso15_720 (section 13) held 2,294 s, so okc_sac_2015 used 6,600 s (1.83 h) of lock time in
all, inside the section 6 estimate for its sync and direction (1.1 to 2.0 h). The driver waited for the lock behind
another job before its first step; waiting time is not counted.

**The arena fallback (ok_by_arena) was not possible.** The rule in sportvu/prepare.py (do_report, near line 158) turns
"ambiguous" into "ok_by_arena" only when at least one other game with the same home team resolved "ok" on its own
windows, all such games agree on one mirror, and the ambiguous game's leading vote is that mirror. cle_gsw_2016 took
it from cle_nyk_2015. okc_sac_2015 has no peer: it is the only game with home OKC in reports/research/p8_registry.json
and in the R1.2 direction reports, and the one other OKC home video in data/harvest/games.json (gsw_okc_2016wcf_g6, a
2016 playoff game) has no SportVU game, so it has no direction result. The fallback is also a step of the R1.2 report,
not of tools/p8_frozen.py's mirror step, so using it here would have been a new rule after the result; section 8 names
no fallback. Report only, not a resolution: the leading mirror, identity, is the one every other resolved game took.

**What this leaves for development truth.** okc_sac_2015 is unused. With nyk_bos_2016 and por_okc_2016 out (section 0,
Lucien's decision of 2026-11-25), Mac development camera truth covers two non-sealed arenas, as before P8's arena
item: GSW (gsw_bkn_2015, gsw_cha_2016 and gsw_sac_2015 in use; gsw_ind_2016 apart under D98) and HOU (hou_cle_2016).
All of it comes from MPS label builds; the platform-mix test of section 0 (gsw_cha_2016 rebuilt on the CPU detector)
is the next item. Cross-arena work at Level 1 step 7 has one non-GSW arena, HOU, until another arena gets truth by
the registered route.
