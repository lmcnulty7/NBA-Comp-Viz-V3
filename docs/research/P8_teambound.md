# P8: the upper bound on the team row for label-level fixes (design, version 1, registered before any number)

Version 1, 2026-11-28. Report only, from stored files only: nothing is fitted, tuned, trained or built, no model runs
and no video is decoded. No Colab unit is used. No held-out game, no D85 test game and no test arena is read
(CLAUDE.md rule 4). hou_cle_2016 is not read; it stays held back for a team arm's single test (D203). No lock is taken
until this design is reviewed.

## 1. Question

The Level 2 team row is 0.834 pooled on gsw_cha_2016 and gsw_bkn_2015 (D202: 0.810 and 0.857), against a target of
0.95. D203 found 26 percent of team errors where another SportVU player is within 3 ft, with the rest spread out, and
did not separate tracking from colour assignment.

From stored files only: **what is the most the team row could rise through fixes to the team labels that need no new
model**, and which kind of fix carries that headroom?

## 2. How the build assigns team labels (read from the code)

- **Crops.** detect/reid.py TorsoCropCollector keeps a torso crop of each tracked box on every 3rd processed frame
  (config.REID_CROP_EVERY), at most 40 per fragment track (config.REID_MAX_CROPS).
- **Features.** detect/teams.py TeamClassifier builds one feature per crop: the CIELAB L quartiles and the median a
  and b of the floor-masked jersey patch. A crop that is mostly floor is dropped. A track's feature is the median of its
  crops' features, needing at least 2 valid crops (config.TEAM_MIN_CROPS).
- **Clusters.** k-means with k = 2 (config.TEAM_N_CLUSTERS) is fitted on the window's fragment tracks
  (build_trajectories.py, team_clf.fit(collector.crops)).
- **Labels.** After re-ID linking, each **canonical track** is assigned the nearest centroid from the pooled crops of
  its fragments (build_trajectories.py, team_by_canon = team_clf.assign(canon_crops)). A track abstains (no label) when
  its nearest-to-farthest centroid distance ratio exceeds 0.75 (config.TEAM_ABSTAIN_RATIO), or it has too few crops.
  The fragment labels also serve as a hard veto in linking (team_of).
- **Consequences:**
  - The label is **per canonical track**, constant over the window. There are no per-detection or per-frame labels.
  - So per-frame flicker within a track cannot occur, and that error kind is zero by construction. The tool asserts
    it: every row of a track carries the same label.
  - A per-track majority vote over the build's own labels is the current scheme, since each track has one label.
- **Scoring (D202, tools.p8_e3.window_team):** a team row is (canonical track, its label, the paired SportVU player's
  team) for every box paired one to one within 5 ft under the truth (sportvu.truth._match, B.MATCH_FT). Within a
  window, each label is mapped to its majority SportVU team, and a row is correct when its label's mapped team is the
  player's team. Abstained tracks give no rows.

## 3. Inputs, selection and the gate

- **The rows** are D203's: tools.p8_teamcause.window_rows. That is the scorer's own pairing recomputed with each row's
  features, on each game's current truth (multipool v1 in work_ms) and D202's composite build. It is called on every
  window of gsw_cha_2016 and gsw_bkn_2015.
- **The gate, per window:** the recomputed rows' count and majority score equal D202's score record (team n and
  majority), as D203's gate did. D202's input hashes and a complete cache (tools.p8_placecause.cache_gaps) are checked
  too. Any difference stops the run.
- **The scorer's identity pairing** (ID_MATCH_FT, 2.5 ft) gives each row's SportVU id where the pair is that close,
  as D203's switch flag used it.

## 4. Per track: purity and the split of errors (registered now)

For each canonical track c in a window:
- n_c is its rows;
- m_c is its rows on its majority SportVU team;
- e_c is its errors under the current labels and the scorer's majority mapping.

The purity is m_c / n_c. Each track's current errors split exactly, with no threshold:
- **Tracking part:** min(e_c, n_c - m_c). These are rows of the track's minority team. No single label per track can
  make them correct: they come from a switch or merge (or a pairing to the wrong player), which is tracking.
- **Colour part:** e_c - (tracking part). These are rows that relabelling the track to its majority team would fix,
  which is colour assignment.
- **Flicker part:** zero by construction (section 2), asserted.

Reported per game and pooled, as shares of team errors and as rates over team rows:
- each part;
- the tracks by purity (1.0, 0.9 to 1.0, 0.5 to 0.9), with their rows and errors;
- each part by D203's crowding bin (another SportVU player within 3 ft, 3 to 6 ft, over 6 ft).

## 5. Oracle bounds (registered now; each is a ceiling, not a forecast)

Each oracle replaces labels only. It keeps the scorer's pairing, the rows and the majority mapping, so the denominator
(team rows) is unchanged. Each is scored with tools.p8_e3.window_team, with the new labels taking the place of the old.

- **O1, a perfect per-track colour assignment:** each canonical track is labelled with its own majority SportVU team.
  This is the most any per-track colour fix can reach. It equals a track-level majority vote with the truth as the
  voter. In this build, a majority vote over the build's own labels changes nothing (section 2), so O1 is the
  track-level bound.
- **O2, split at identity switches, then perfect colour per segment:** each track is cut where its 2.5 ft identity
  pairing moves to another SportVU id and stays there for at least 1.0 s of the track's scored frames (a sustained
  switch). Each segment is then labelled with its own majority team.
  - A brief change shorter than 1.0 s (pairing noise in a crowd) does not cut. Cutting at every change of the paired
    id would make every segment pure by construction, since a row's team follows its id, and the bound would be
    trivially 1.0.
  - O2 - O1 is the most that splitting tracks at switches could add on top of perfect colour.
- **O3, colour with the current errors kept on mixed tracks (report only):** pure tracks (purity 1.0) relabelled to
  their team, mixed tracks left as they are. This isolates fixes that need no tracking change and no handling of mixed
  tracks.
- **Abstained tracks, report only:** O1 with abstained tracks also labelled (their majority team). This adds rows, so
  the denominator changes. It is reported apart and never compared with the rule.

The current row (the reproduction of D202) is reported beside each.

## 6. Registered reading table

A fix **deserves an arm** when its bound adds at least +0.03 to the team row on both games and pooled. The added
amounts are:
- the colour fix: O1 - current;
- the tracking fix: O2 - O1.

| pattern | condition | reading |
|---|---|---|
| colour headroom | O1 - current at least +0.03 on both games and pooled | a per-track colour assignment fix could carry at least 0.03; an arm on the team classifier (features, k, the abstain rule) deserves a design |
| tracking headroom | O2 - O1 at least +0.03 on both games and pooled | splitting tracks at switches could add at least 0.03 beyond perfect colour; an arm on track splitting or re-ID deserves a design |
| both | both hold | both are read, and the larger bound goes first |
| neither | neither holds | label-level fixes cannot reach +0.03; the team gap lies in what O2 leaves: the pairing itself (rows paired to the wrong player), brief swaps, or the truth |

Also reported against the 0.95 target: O2 and the gap left under it, 0.95 - O2.

Why +0.03: it is about a fifth of the gap to the target (0.95 - 0.834 = 0.116), and well above the window-level noise
of the team row (D202's pooled interval 0.818 to 0.849). The bounds are point values of oracles on the same rows; their
window-cluster intervals (2,000 resamples, seed 0) are reported beside them. The rule reads the points, since an
oracle has no sampling error beyond the rows themselves.

## 7. What it can and cannot tell

- The oracles use the truth (the SportVU team and id of each pair). They bound what a fix could reach on these rows;
  they do not show that any model reaches it.
- O2's 1.0 s rule is a convention, fixed now. Its sensitivity at 0.5 s and 2.0 s is reported, not ruled.
- Rows come from the scorer's pairing within 5 ft. A row paired to the wrong player in a crowd counts as tracking here,
  and no label fix can correct it.
- The truth frames are selected by the same r23 boxes (D202), and the team rows lean toward r23.
- Two games, one arena, one broadcaster.

## 8. Cost and guards

- One collect per game under the lock ("p8 teambound <game>", frozen CV python, V3_P8_ROOT set to work_ref). D203's
  stage B took about 2 lock minutes for both games, and this is the same row recomputation. So about 3 lock minutes
  with the manifests, cap 15 minutes per game. The report runs in system python.
- Guards on every game, SportVU id, video id and window: tools.p8_common.guard and tools.research_testset.refuse_cv_test.
  The tool refuses hou_cle_2016 before any file is read, and a test checks it.
- SHA-256 manifests of work_ref's and work_ms's labels, builds and cache before and after each step.
- Writes go only to work_ref/teambound/<game>.json and reports/research/p8_teambound.json. A test checks that these
  are the only writes.
- One chain, reported the moment it ends.
