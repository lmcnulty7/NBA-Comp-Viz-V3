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
is the next item.
(Update after section 15: that test read MOVED; see there.) Cross-arena work at Level 1 step 7 has one non-GSW arena, HOU, until another arena gets truth by
the registered route.

## 15. Result (2026-11-26): the gsw_cha_2016 platform-mix test of section 0, MOVED

**What ran.** gsw_cha_2016's camera truth rebuilt in the reference root with the detector on CPU (court and gate
unchanged, no clip fix), by the steps of section 4: link (the main checkout's sections, moments and sync maps, the
same files the MPS truth used), label one job a section (V3 court model, gate v2), then fit. Every model step held the
heavy-job lock alone; no step failed or was retried. Label: 74 windows, 73 built, 1 gate-empty (the MPS root had the
same 73 and 1). All 73 built windows carry a device record that resolves the detector to cpu (court mps, gate mps).
All 73 window time maps equal the MPS root's, so the windows cover the same frames.

**Lock time.** 9,745 s (2.71 h): label 9,567 s in 10 jobs, fit 178 s. Section 0's estimate was about 2.35 h (2.2 h of
label builds and 0.15 h of fit); the run used 1.15 times that, under the driver's cap of 1.5 times. The estimate left
out the label step's own clip cuts and window clock reads (about 1,000 s, judging by the MPS label and rebuild logs).

**Measures against the bars fixed in section 0** (tools/p8_truth_moved.py; record reports/research/p8_truth_moved_gsw_cha_2016.json):

| Measure | MPS truth | CPU truth | Difference | Bar for "moved" | Moved |
| --- | --- | --- | --- | --- | --- |
| Median reprojection after refinement | 7.65 px | 7.06 px | 0.59 px | 1.0 px or more | no |
| Accepted camera-truth frames | 1,366 | 2,459 | +1,093, 0.800 of the MPS count | 10 percent or more | yes |
| Per-window frozen offsets (73 windows in both) | | | median abs difference 0.08 s; share over 0.2 s 0.356; max 2.6 s | median 0.2 s or more | no |

**Verdict: MOVED**, on accepted frames alone. Section 0 fixed the consequence: rebuild the truth of gsw_bkn_2015,
gsw_sac_2015 and hou_cle_2016 on the CPU detector (about 2.5, 2.3 and 2.9 lock hours of label builds by section 0's
rates; this run suggests adding about 15 percent) before any cross-arena tuning. gsw_ind_2016 stays apart (D98).
Nothing registered is rescored.

**What this means for earlier results.** Every analysis scored on the Mac MPS camera truth of the pool games
(gsw_bkn_2015, gsw_cha_2016 and gsw_sac_2015; hou_cle_2016 as a check where used), all from MPS label builds,
among them: the registered r23 baseline rows (P8_baseline.md), P8_E1.md, P8_E1B.md, E1R's scores (P8_E1R.md),
P8_E7a.md, P8_truth_v2.md, P8_court_truth.md, P8_teachers.md source 2 (gsw_bkn_2015 and the apart gsw_ind_2016), P8_step0_trackB.md, P8_placement.md steps 1
and 1b, P8_footbias.md, P8_footpos.md, P8_outpos.md, P8_detrow.md, P8_buildloss.md, P8_trackreset.md and
P8_a3align.md. Their numbers stand as recorded and nothing is rescored. Any
conclusion that depends on the GSW or HOU camera truth is provisional until those games have CPU-detector truth; a
rescore needs its own registration. The gsw_cha_2016 truth in use stays the MPS truth for now: the CPU truth has
not passed section 5's acceptance (check C needs an r23 rebuild in the reference root, not yet run).

Descriptive, not part of the bars. The fit's camera position moved about 10 ft (C from (42.6, 118.7, -31.1) to
(46.9, 127.8, -31.4) ft; 400 frames pooled in both). The extra accepted frames come mostly before the line checks:
"untestable_boxes" fell from 5,181 to 3,770, as expected from the CPU detector's extra player boxes
(P8_PLATFORM.md), while more candidates then failed the line checks (line_matches_low 653 to 985, support_low 131 to
208, feet_resid_high 429 to 467).

**Report only, post hoc, not ruled** (asked by the lead after P8_placement.md step 1b; tools/p8_truth_moved_dh.py,
stored files, no lock). |d_H| = |B(p) - T(p)| in ft at every box's foot_stab pixel p of the scored reference-platform
r23 sidecar (builds/m9_r23_detcpu__gsw_cha_2016, B its per-frame H), on truth-accepted frames whose sidecar row is
accepted with an H. All boxes, not step 1b's nearest-box player population, so the values are not step 1b's 2.09 ft.

| | Frames | Boxes | Median abs d_H |
| --- | --- | --- | --- |
| MPS truth, all its frames | 1,366 | 12,835 | 1.53 ft |
| CPU truth, all its frames | 2,459 | 23,068 | 1.14 ft |
| MPS truth, frames accepted by both | 904 | 8,606 | 1.48 ft |
| CPU truth, frames accepted by both | 904 | 8,606 | 1.05 ft |

On the frames both accept, the truth itself moved a median 1.10 ft at the same foot pixels. Read only as a
description: the CPU truth sits closer to the r23 build's H than the MPS truth did. That is partly expected from a
shared input: B comes from a CPU-detector r23 build, and the new truth was fitted from CPU-detector label builds with
player feet among its inputs. So it is not evidence that the CPU truth is better. It does not show which truth is closer
to SportVU, and no rule was fixed for it.

## 16. Plan (2026-11-26, registered before any run): CPU-detector truth for the in-use games, through acceptance

Version 2 of this plan, written the same day and still before any run. Version 1 (commit c2698c7f) built the three
truths and compared them with the MPS truth, but had no r23 builds, so no CPU truth could pass section 5's check C
and come into use. Version 2 adds check C's r23 side and the acceptance checks for all four in-use games.

**Why.** Section 15 read MOVED on gsw_cha_2016. Section 0 fixed the consequence: the truth of the other in-use games is
rebuilt on the CPU detector before any cross-arena tuning. gsw_ind_2016 stays apart (D98). The gsw_cha_2016 truth in
use stays the MPS truth until its CPU truth passes section 5 (section 15). This plan builds the three truths, runs
section 5's checks A, B and C on all four CPU truths, and reports, per game, whether its truth moved. It runs only on
the lead's go, after the a3run job's report.

**Games and order.** gsw_cha_2016 first (its truth is built; only check C's r23 side and the acceptance run, minutes
of lock), then hou_cle_2016 (the only non-GSW arena with truth), gsw_bkn_2015 and gsw_sac_2015. All four are
development games: the guard and refuse_cv_test run at every step; no held-out game and no D85 test game.

**Truth steps, the same as gsw_cha_2016 (section 15).** Driver: tools/p8_truth_cpu.sh, in the reference root
(work_ref), the detector on CPU (`P8_DEVICE_DETECTOR=cpu`), court and gate unchanged, no clip fix. Per game: link;
label one job a section (V3 court model, gate v2); fit. Every model step takes the heavy-job lock alone.

- Links, read only. The GSW games use the frozen link step (the main checkout's sections, moments and sync maps, the
  files their MPS truth used). hou_cle_2016 is a channel game, so its inputs are linked from the P8 workspace (work):
  its 12 sections, its SportVU moments, its 12 sync time maps (the OCR runs on CPU, so they do not depend on the
  device), its layout result (abc_finals16_720, a frozen layout) and its mirror record (status ok, identity, vote
  share 1.0). The mirror is not voted again: as for the GSW games, whose mirror comes from the R1.2 record, it is a
  resolved property of the game, and this test is about the label builds and the fit. Nothing is written to work.

**Check C's r23 side.** Check C compares, per window, the V3 label build with an r23 build of the same clip in the same
root. The reference-platform r23 builds of all four games already exist in work (tools.p8_placement.BUILDS: M9's for
gsw_cha_2016, gsw_sac_2015 and hou_cle_2016, M6's r23 arm for gsw_bkn_2015, 73, 82, 93 and 83 windows), made from
work's label clips with the same device request (detector cpu, court and gate auto, no clip fix); they are the
r23 builds the reference-platform scores already use. The reference root's label clips are cut again from the same
sections; on gsw_cha_2016 all 73 are byte-identical to work's (SHA-256, checked before this plan without writing
anything). So:

- Default (R23_SOURCE=link): tools/p8_truth_cpu_r23.py copies a window's work build into work_ref/builds/r23 only
  when its clip is byte-identical in both roots and its device record asked for exactly the reference setting; it
  records each refused window and why. Then the frozen rebuild step runs one job a section (`rebuild --build r23`,
  the r23 court checked against its recorded SHA-256). It builds every window without an r23 build (refused windows,
  and gate-empty label windows) and refuses any cached build whose device record differs.
- Alternative (R23_SOURCE=rebuild): every window built again under the lock, about 9.8 more lock hours. Chosen by
  the lead before the go if the copied builds are not wanted; the bars and the checks are the same either way.
- Then `tools/p8_truth_v2.py measure-dev` (under the lock, about 2 to 3 minutes a game) writes
  truth_v2/dev/<game>.json in the reference root, and `tools/p8_truth_accept.py` applies A, B and C with the section 5
  bars unchanged and the platform rule (every label and r23 window resolving the detector to cpu): use, apart or
  unused. Code change for this, stated here: measure-dev refused the three pooled truth v2 games anywhere; it now
  refuses them only in the registered workspace (work), where `measure` and the registered report hold them, and
  still refuses gsw_ind_2016 everywhere (tools/p8_truth_v2.refuse_dev, with a test). The registered truth v2 record
  and report are not touched: report reads only truth_v2/<game>.json of its own root.

**Lock estimates and caps.** Label lock time on gsw_cha_2016 was 1.21 times its M9 CPU r23 build (9,567 s against
7,905 s on the same windows: the label step also cuts the clips and reads each window's clock). The same ratio is
applied to each game's CPU r23 build time; the fit is the MPS fit's time times 1.22 (gsw_cha_2016: 146 s on MPS,
178 s here). Check C with copied builds: about 400 s a game (ten to twelve rebuild jobs that find cached windows, the
gate-empty windows, measure-dev); gsw_cha_2016 is given 600 s. With R23_SOURCE=rebuild, the game's CPU r23 build time
is added. A game whose lock time in this driver passes 1.5 times its estimate stops the driver before its next step.

| Game | CPU r23 build on record | Truth (label and fit) | Estimate, copied r23 | Cap | Estimate, r23 rebuilt | Cap |
| --- | --- | --- | --- | --- | --- | --- |
| gsw_cha_2016 | 7,905 s (M9) | done (section 15, 9,745 s) | 600 s | 900 s | 8,205 s | 12,300 s |
| hou_cle_2016 | 10,339 s (M9) | 12,795 s | 13,195 s (3.67 h) | 19,790 s | 23,434 s (6.51 h) | 35,150 s |
| gsw_bkn_2015 | 8,906 s (M6, r23 arm) | 10,980 s | 11,380 s (3.16 h) | 17,070 s | 20,286 s (5.64 h) | 30,430 s |
| gsw_sac_2015 | 8,249 s (M9) | 10,200 s | 10,600 s (2.94 h) | 15,900 s | 18,849 s (5.24 h) | 28,270 s |
| All four | | | about 9.9 h | | about 19.7 h | |

**Stops.** Resumable: a step with a done marker (or, for measure-dev, its record) is skipped and built windows are
cached. A failed step is retried at most twice, then the driver stops and the failure is reported. A STOP file in the
reference root stops it before the next step. One heavy job at a time.

**Reports, per game (stored files, no lock).**

- Acceptance: tools/p8_truth_accept.py, A, B and C with section 5's bars and the platform rule, giving use, apart or
  unused.
- Moved: tools/p8_truth_moved.py with section 0's bars unchanged (reprojection 1.0 px, accepted frames 10 percent of
  the MPS count, median per-window offset difference 0.2 s), beside each game's MPS truth (gsw_cha_2016's is on
  record already, section 15).
- Report only, post hoc, not ruled: tools/p8_truth_moved_dh.py, the median |d_H| at the scored r23 sidecar's foot
  pixels for each truth, as in section 15, with section 15's confound (the sidecar and the CPU truth share the CPU
  detector).
- Records: reports/research/p8_truth_moved_<game>.json and reports/research/p8_truth_accept_<game>.json, committed
  with a dated result section here.

**What this plan does not decide.** Which truth is used afterwards (a CPU truth that reads "use" in place of the MPS
truth, or both with the platform recorded) is the lead's decision once all four acceptances are in. Nothing registered
is rescored and no bar changes after a game is seen.

**Version 3 (2026-11-26, the lead's decisions; before any run). Where it differs from version 2 above, version 3
holds.**

1. hou_cle_2016's mirror: the recorded vote is reused (status ok, identity, vote share 1.0) and not voted again (the
   lead's decision). A unanimous vote is not a close call, and the GSW games reuse their recorded mirrors too.
2. Check C needs no new r23 build, and the driver builds none. An r23 build does not depend on the truth: it is a
   build of the label clip with the r23 court. tools.p8_truth_v2's measure-dev reads the r23 builds from builds/r23 of
   its own root (V3_P8_ROOT), and tools.p8_truth_accept reads their device records there, so the workspace's
   reference-platform r23 builds (builds/m9_r23_detcpu__<game>, and e1m6_r23_detcpu for gsw_bkn_2015) serve once
   they are in work_ref/builds/r23. They are copied, not linked, by tools/p8_truth_cpu_r23.py, for each window whose
   build records the workspace's label clip as its source, whose clip is byte-identical in both roots, and whose
   device request is the reference one. Checked before any run, without writing: every build on record names the
   workspace clip and the reference request (gsw_bkn_2015 83 windows, gsw_cha_2016 73, gsw_sac_2015 82,
   hou_cle_2016 93), and gsw_cha_2016's 73 clips are byte-identical in both roots; the other three games' clips are
   compared when their label step has cut them. Copying keeps the workspace untouchable and leaves nothing in the reference root pointing into it.
   The rebuild step and R23_SOURCE=rebuild are removed from the driver.
3. Coverage: measure-dev measures only windows with both builds, so a window without an r23 build would drop out
   unseen. Check C and the acceptance therefore run only when every V3 label window of the game has a verified r23
   build (the copy record's "complete"). Otherwise the driver writes "CHECK C NOT RUN", the game's acceptance stays
   pending, and the missing windows are reported: building them would be a fresh r23 build in this root, about 0.44 s
   a processed frame (a whole game is 7,905 to 10,339 s, M9 and M6), left out of this driver.
4. Check C runs per game as a report step after the fit: the copy (no lock), measure-dev (under the lock, about 2 to 3
   minutes), then tools.p8_truth_accept (no lock), with section 5's bars and the platform rule unchanged. Then
   p8_truth_moved and the d_H report as above.
5. Base: the branch holds v4-research at 136b742c (D191), which includes the reviewed sections 14 and 15.

Lock estimates and caps under version 3 (truth as in version 2's table, plus about 250 s a game for measure-dev; for
gsw_cha_2016 only measure-dev):

| Game | Estimate | Cap (1.5 times) |
| --- | --- | --- |
| gsw_cha_2016 | 300 s | 450 s |
| hou_cle_2016 | 13,045 s (3.62 h) | 19,570 s |
| gsw_bkn_2015 | 11,230 s (3.12 h) | 16,850 s |
| gsw_sac_2015 | 10,450 s (2.90 h) | 15,680 s |
| All four | about 9.7 h | |

Review fixes (ml-review of 5c1bafd3, GO; applied before any run). A copied r23 build must also carry a detector record
naming the r23 player detector (model "r23", weights sha256 4c886d5f..., the repository's models/player_detector.pt;
every build on record does). A build already in work_ref/builds/r23 passes the same checks on its own files or stops
the copy, and one there with no build on record stops it too. The driver exits at start unless the reference root and
the P8 workspace are different folders (real paths) and the repository's models/player_detector.pt has the r23
sha256. The registered workspace path of measure-dev's guard stays written out in tools/p8_truth_v2.py (moving it into
tools/p8_common.py would change a file under the scorer's pin); a test checks it equals p8_common's default root.

**Version 4 (2026-11-26, PLAN.md D192 and the lead's decisions; registered before any fixed build). Where it differs
from versions 2 and 3 above, version 4 holds.**

*Why.* D192: the tracker double-update bug after a reset (detect/tracker.py PlayerTracker.reset; the tracker keeps
about 0.3 of the detector's boxes after a reset) also thins camera truth, since truth frames come from the label
build's tracker boxes. Truth built with the bug under-represents post-reset frames. The lead stopped the version 3 run
after hou_cle_2016 label s03 (lock spent on unfixed builds: gsw_cha_2016 9,745 s in section 15 plus 130 s, and
hou_cle_2016 4,658 s for s00 to s03).

*Decisions (the lead's).*

1. Every label build of all four games (gsw_cha_2016 included) is rebuilt with the reset fix:
   `P8_TRACKER_RESET_FIX=1` (tools/p8_device.py and tools/p8_resetfix.py of the reset-fix arm, branch r-p8-resetfix,
   design docs/research/P8_resetfix.md section 14), with the detector on CPU, court and gate auto, no clip fix,
   TRACKER_REID unset. Each label window's device record must carry "reset_fix": true.
2. Every earlier label window in the reference root is invalid. tools.p8_device.check_cached refuses a cached window
   whose reset_fix differs, so they cannot mix. They are moved aside, kept and not deleted (tools/p8_truth_fixprep.py
   supersede, into work_ref/superseded_nofix/ at the same relative paths): every unfixed label window's outputs, the
   truth fitted from them (labels/<game>/ and its result, offsets and paint files), the records read from that truth
   (truth_v2/dev, truth_accept, truth_moved), the label and fit done markers, and version 3's unfixed r23 copies
   (builds/r23/<window>_*, builds/r23_source_<game>.json). The label clips (<window>.mp4) and window time maps stay in
   place and are reused, never cut again, so they stay byte-identical to the P8 workspace's.
3. The unfixed gsw_cha_2016 CPU truth stays on record as it is: section 15, and its acceptance ("use"; A 7.06 px, B
   2,459 frames and 2,308 s, C n 63 of 73, median 0.48 s, share 0.7619) in
   reports/research/p8_truth_accept_gsw_cha_2016_nofix.json. It is not used for anything further.
4. Check C pairs fixed with fixed, so it stays a court-model comparison and section 5's bars keep their meaning. In
   the reference root, builds/r23 becomes check C's composite, filled per game by tools/p8_truth_cpu_r23.py compose
   only after the reset-fix arm's equivalence record for that game (resetfix/equiv__<game>.json) shows "passes": true:
   - the windows listed in resetfix/plan__<game>.json "reset_windows" (a reset in the baseline build, identity
     camera_cut_resets above 0): the arm's fixed build, builds/resetfix_r23_detcpu__<game> of the reference root;
   - every other window: the workspace's reference-platform r23 build (tools.p8_placement.BUILDS). With no reset, the
     fix changes nothing, which the arm's equivalence windows test byte for byte.
   A build is copied only when its sidecar's source is the window's label clip (the reference root's for an arm build,
   the workspace's for a baseline build) and that clip is byte-identical in both roots; its device record asks for the
   reference setting with reset_fix true (arm) or absent (baseline) and no clip fix; and its detector record is the
   r23 detector (model r23, sha256 4c886d5f...). Check C and the acceptance run only when every fixed V3 label window
   has such an r23 build ("complete"); otherwise "CHECK C NOT RUN" and acceptance waits. tools/p8_truth_v2.py
   measure-dev and tools/p8_truth_accept.py are unchanged; the folder r23_resetfix_checkc of an earlier message is
   dropped.
5. The arm's unfixed baseline is the arm's, in builds/r23_nofix of the reference root (made by the arm's own command
   with this tool's checks). The arm writes only builds/resetfix_r23_detcpu__<game>, builds/r23_nofix, resetfix/,
   cache/scorecard copies and tmp/resetfix_common_labels; this driver writes only label_build/, labels/, truth_v2/dev/,
   truth_accept/, truth_moved/, builds/r23/ and its records, superseded_nofix/ and meta/done/.
6. hou_cle_2016's reset-window arm builds run before the arm's pool verdict (the lead lifted that gate for builds,
   not for scoring), so hou_cle_2016's check C waits only for them; no r23 build is made by this plan.

*Order, one heavy job at a time.* (1) The arm's stage 0 and smoke, reviewed. (2) This driver's truth phase, all four
games: supersede, link, label with the fix, label-check, fit, then truth_moved and the d_H report against the MPS truth.
(3) The arm's builds and equivalence for the three pool games and hou_cle_2016 (build only, no score). (4) This
driver's accept phase per game: compose, measure-dev, p8_truth_accept. (5) The arm's pin-truth (it requires a final
use verdict and every label device record with reset_fix true and detector cpu) and pool scoring. (6) hou_cle_2016
scoring only if the pool passes. Steps 2 and 3 do not overlap.

*Driver.* tools/p8_truth_cpu.sh, two phases: `truth` and `accept`, a new log (logs/truth_fix.log) and lock names
("p8 fixtruth <phase> <game> ..."), so version 3's lock time does not count against version 4's caps. It exits at
start unless the reset-fix code is on the branch (tools.p8_resetfix importable, RESET_FIX_FLAG in tools.p8_device;
r-p8-resetfix must be merged first), TRACKER_REID is unset, the two roots differ and models/player_detector.pt is
r23's. The label-check runs before every fit and stops the driver if any label window lacks the fix or the reference
devices. Resumable, two retries at most, STOP file, as before.

*Estimates and caps (lock seconds).* Truth as in version 2's rates (label 1.21 times the CPU r23 build, fit 1.22 times
the MPS fit; gsw_cha_2016 from its section 15 run). The fix is not expected to change build time materially (it removes
a second tracker update per frame after a reset). Accept: measure-dev, 130 s on gsw_cha_2016.

| Game | Truth estimate | Truth cap | Accept estimate | Accept cap |
| --- | --- | --- | --- | --- |
| gsw_cha_2016 | 9,750 s (2.71 h) | 14,630 s | 130 to 250 s | 450 s |
| hou_cle_2016 | 12,795 s (3.55 h) | 19,190 s | 130 to 250 s | 450 s |
| gsw_bkn_2015 | 10,980 s (3.05 h) | 16,470 s | 130 to 250 s | 450 s |
| gsw_sac_2015 | 10,200 s (2.83 h) | 15,300 s | 130 to 250 s | 450 s |
| All four | 43,725 s (12.1 h) | | under 0.3 h | |

*Reading the reports.* p8_truth_moved now compares MPS truth without the fix with CPU truth with the fix, so a MOVED
reading carries both the platform and the fix; section 15 isolates the platform part on gsw_cha_2016. The bars of
section 0 and section 5 are unchanged. Records: reports/research/p8_truth_moved_<game>.json and
reports/research/p8_truth_accept_<game>.json, committed with a dated result section.

**Version 4.1 (2026-11-26, the lead's decisions after the reset-fix design review; before any fixed build).** (Commit
f0948913 briefly wrote the opposite composer choice; the lead reversed it the same day, before any review or run, and
this text replaces it.)

1. One check C composer, this plan's: `tools/p8_truth_cpu_r23.py compose` is the only writer of builds/r23. The
   reset-fix arm composes nothing; it writes its unfixed baseline to builds/r23_nofix (with its own
   builds/r23_nofix_source_<game>.json). compose gates on resetfix/equiv__<game>.json "passes": true; takes the reset
   windows from resetfix/plan__<game>.json "reset_windows" (a sorted list of window stems) and ignores the arm
   folder's other windows (equivalence and smoke windows); accepts an arm build only when its sidecar source resolves
   to the reference root's clip and that clip's SHA-256 equals the workspace clip's; takes every other window from the
   workspace's baseline build; and stops if builds/r23 already holds a file that fails these checks or a window with
   no build on record. The accept phase runs compose, then measure-dev and p8_truth_accept (unchanged) only when its
   record builds/r23_checkc_<game>.json shows "complete": true.
2. Before any composition, builds/r23 is cleared of unfixed copies: the truth phase's supersede moves version 3's
   unfixed copies (gsw_cha_2016's 73 windows, and any hou_cle_2016 ones; there are none) and
   builds/r23_source_<game>.json into superseded_nofix/builds/. supersede runs per game at the start of the truth phase;
   compose runs only in the accept phase, after the truth phase.
3. Before any fixed label build, supersede also moves aside every artefact of the bug-carrying truth that the arm's
   pin-truth could capture: labels/<game>/ (the jsonl files), labels/<game>_result.json, _offsets.json and
   _paint.jsonl, truth_v2/dev/<game>.json, truth_accept/<game>.json, truth_moved/<game>*.json, and the label and fit
   done markers. The fixed truth's files are then all written after every fixed label build of the game (the fit
   follows the label-check of every window), which the pin checks.
4. Clips (label_build/<window>.mp4) and window time maps are never moved, cut again or rewritten; the arm checks clip
   SHA-256 at plan, build, equivalence and score, and compose checks it per window.

Order: the arm's design v2.2 and its re-review, its merge to v4-research, this branch's merge of v4-research, an
ml-review of version 4.1, the arm's smoke, then the lead's go for the truth phase. Estimates, caps and bars as in
version 4.

Composer, final (the lead's definitive instruction, 2026-11-26, overriding every earlier message on it): this plan's
`tools.p8_truth_cpu_r23 compose` is present and is the only writer of builds/r23, as in the corrected version 4.1
above (29b97e62). Why: the reset-fix arm's design v2.1 has no composer, and there must be exactly one. Commits
f0948913 and 208c846a, which left the composing to the arm after crossed messages, are superseded; the code equals
29b97e62's.
