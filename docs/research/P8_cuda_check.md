# P8: the CUDA check of D161 (1), design fixed before any run

Written 2026-11-25 (UTC; the evening of 2026-11-24 local time), before any CUDA build, amended before the run (section 11), under PLAN.md D161 (1) (and D98, D99, D100, D164, D169). The code is
`tools/p8_cuda_check.py`. Nothing here is a result. The game is gsw_ind_2016, an R1.3 training game (development
data, D98); no held-out game is read and no held-out metric is computed (CLAUDE.md rule 4).

## 1. The question

The Mac's frozen build differs by device (docs/research/P8_PLATFORM.md, reports/research/p8_platform.json): on the
two D99 windows, MPS finds about one player box per frame fewer than CPU, and the court mapping moves about 4 px. The
Colab R1.3 run (A100, CUDA) found far more camera-truth candidates than the Mac's MPS night, and a Mac CPU build at
Colab's clock offset nearly matched it (88 against 89 on s02_w03). So: **does a CUDA build of the same windows, with
the same frozen code, weights and video, match the Mac CPU build, the Mac MPS build, or neither?**

Two side questions are measured on the same runtime, so a mismatch can be placed:

- Is CUDA repeatable? (the same CUDA build twice)
- Do the libraries matter? (a CPU build on the Colab runtime, against the Mac CPU build)

## 2. What is built

| Item | Value |
| --- | --- |
| Windows | gsw_ind_2016_s02_w03 (145 built frames on the Mac), gsw_ind_2016_s02_w04 (125); and, report only (section 11, A12), gsw_bkn_2015_s04_w00 (1102) with the r23 court model |
| Video | the two window clips the Mac builds read (P8 workspace label_build/), shipped as files with their sha256; they are not cut again on the runtime, so ffmpeg versions cannot differ. The section they were cut from (gsw_ind_2016_s02) is recorded with its sha256 and checked against DATA_ROOT/sportvu/MANIFEST_r13.json when the Drive mount is present |
| Window plan | the frozen `sportvu.rebuild.plan_windows(section, autolabel.EVERY_SPAN)` from the section's local time map, computed on the Mac and stored in the manifest; the runtime computes it again from the shipped time map and stops if it differs |
| Build path | the frozen path the Mac used: `tools.p8_frozen.install_paths`, then `safe_build` (that is `sportvu.rebuild.build`, `build_trajectories.py --pregate --no-video` at the harvest stride, with `autolabel.gate_args()`: gate head v2 at 0.33) |
| Devices | `P8_DEVICE=cuda` for every model (detector and tracker, court models, gate CLIP) through `tools/p8_device.py`, which writes each window's device record; `P8_DEVICE=cpu` for the Colab CPU arm |
| Precision | fp32: no code in the build path asks for half precision; image size 640, confidence 0.40, NMS IoU 0.45 (config) |
| Weights | player_detector.pt, court_grid_snapped.pt, court_line_seg.pt, thresholds.json (tracked in git; sha256 in the manifest, checked on the clone), trained_head_v2.joblib and, for the report-only window, court_grid_r23.pt (gitignored; shipped, sha256 checked, r23 also against reports/r23_verification.json). CLIP ViT-B/32 is checked by a fingerprint, the sha256 of its parameter tensors, because the Mac may load it from pytorch_model.bin or model.safetensors (same tensors, different files) |
| Libraries | ultralytics 8.4.48 and transformers 4.49.0 pinned (the Mac's versions; a newer transformers may preprocess CLIP's images another way, which would move the gate's frames for a reason that is not the device) and lap; torch, OpenCV and NumPy are the runtime's stock versions, recorded. The Mac builds used Python 3.12.2, torch 2.2.2, ultralytics 8.4.48, transformers 4.49.0, OpenCV 4.13.0, NumPy 1.26.4. torch is not pinned: torch 2.2.2 needs NumPy 1 and would replace the runtime's stack; the Colab CPU arm measures what the remaining library differences do. If the transformers pin cannot install, the run uses the stock version and the result says so |
| GPU | a T4 (the R1.3 run used an A100-SXM4-40GB; its ultralytics version was not recorded). The comparison is against the Mac builds, not against R1.3; the R1.3 numbers are reported beside, with no verdict. The verdict holds for a T4 only: an A100 or an L4 computes with TF32 by default (a T4 has none), so it does not cover them; the run records torch's TF32 flags (A3) |

Builds, in this order (each writes the result tar when it ends, so a stop keeps the arms already done):

1. `cuda_a`: both windows on CUDA (the build that is judged), then the report-only window
2. `cuda_b`: the same again (repeatability)
3. `colab_cpu`: both windows on the runtime's CPU (library effect; optional, it is the first to drop if time runs short)

The runtime needs no SportVU or play-by-play table: the build reads only video and weights. Every comparison runs on
the Mac afterwards with the Mac's own time maps, moments and mirror, so each build differs from the others only in
what the build produced.

## 3. What is compared

Builds on the Mac, read only: MPS (P8 workspace label_build/ for s02_w03, the identical rerun `_p8-diag/s02_w04_mps`
for s02_w04), CPU (`_p8-diag/s02_w0X_cpu`), and the detector-CPU build of s02_w03 (reported only). These are
`tools.p8_truth_v2.DEVICE_BUILDS`.

For each window and each pair of builds (X against a reference R):

| Measure | Definition |
| --- | --- |
| Frame-set agreement | common frames over the union of the two builds' frames (the gate picks the frames) |
| Boxes per frame | mean player boxes per built frame, and the difference X minus R |
| Same box count | share of common frames where X and R have the same number of boxes |
| Identical boxes | share of common frames with exactly the same box coordinates (reported, not in the rule) |
| Court mapping gap | per common frame, the median pixel distance between the two court homographies at three court points (as `tools.p8_platform_summary.h_gap_px`); the median and p90 over frames |
| Candidates at fixed offsets | camera-truth candidates (frames with 6 or more matched players, frozen `truth_for_frame` and `camera.frames_from_rows`, through `tools.p8_truth_v2.candidates`) at each offset of the window's fixed set |
| Frozen solved offset | the frozen `solve_window_offset` on the build (reported, not in the rule: it is fragile, D99) |
| Truth v2 offset | `tools.p8_truth_v2.solve_v2` on the build, read only (reported; v2 failed its bar, D166, and is not a truth rule) |

The fixed offset set of a window is fixed now, from recorded numbers only: 0.30, 0.62, 0.90 and 1.20 s (the sweep of
P8_PLATFORM.md), plus the window's Colab R1.3 offset (reports/research/p8_platform.json: 0.62 s for s02_w03, -0.06 s
for s02_w04), plus the frozen solved offset of each Mac build (recorded: s02_w03 MPS 0.90, CPU 1.42, detector-CPU
0.94; s02_w04 MPS -1.34, CPU -0.78). The compare command recomputes the Mac offsets and flags any that differ from
these.

Beside the verdict, for the record only: the CUDA build's candidates at Colab's R1.3 offset against the R1.3 log's
candidates (89 on s02_w03, 73 on s02_w04; A100, another date, unrecorded ultralytics).

## 4. The reading rule (fixed now)

X **matches** a reference R when all five tests hold on **both** windows:

| Test | Tolerance | Why this number (Mac numbers, p8_platform.json) |
| --- | --- | --- |
| (e) frame-set agreement | at least 0.95 | every Mac device pair built exactly the same frames (125 of 125, 145 of 145) |
| (a) boxes per frame | difference at most 0.25 | MPS against CPU differs by 0.90 (s02_w04) and 1.04 (s02_w03); 0.25 is about a quarter of the smaller |
| (b) same box count | share at least 0.80 | MPS against CPU agrees on 0.31 and 0.20 of frames; the same detector device agrees on 1.00 (145 of 145) |
| (c) court mapping gap | median at most 1.0 px | MPS against CPU: 4.65 and 3.84 px; the same court device: 0.0 px; 1.0 is about a quarter of 3.84 |
| (d) candidates | at every offset of the fixed set, within max(3, 10 percent of R) | MPS against CPU on s02_w03 differs by 33, 43, 25 and 19 at the four sweep offsets (34 to 54 percent); detector-CPU against CPU by 1, 0, 0, 0. The floor of 3 keeps small counts (s02_w04) from failing on one frame |

The tolerances sit far from the MPS-CPU gap and allow float-level differences between devices (a box near the 0.40
confidence line can flip). (a) and (b) together, at these values, cannot hold against both MPS and CPU at once
(agreeing with each on 80 percent of frames would force MPS and CPU to agree on 60 percent; they agree on 20 to 31).
The compare command checks this: it applies the rule to Mac CPU against Mac MPS, which must not match; if it did, the
verdict is "rule cannot separate" and nothing is read.

The verdict, in this order:

1. **CUDA not repeatable**: `cuda_b` does not match `cuda_a` under the same five tests. Nothing else is read as a
   platform verdict (the differences found below would be within CUDA's own noise).
2. **CUDA matches CPU**: `cuda_a` matches Mac CPU.
3. **CUDA matches MPS**: `cuda_a` matches Mac MPS.
4. **Matches neither**: otherwise. The result names the tests that failed, per window and per reference.

If `cuda_b` is missing (the runtime was lost after `cuda_a`), the verdict is read from `cuda_a` and marked
"repeatability not measured". The Colab CPU arm does not change the verdict; it says where a difference comes from:
`colab_cpu` matching Mac CPU means the runtime's libraries do not change the build, so a CUDA difference is the
device; `colab_cpu` not matching Mac CPU means libraries or platform change it too. `cuda_a` against `colab_cpu` is
reported, not in the verdict. If `colab_cpu` is dropped, a "matches neither" cannot be traced to the device or to the
libraries, and the result says so (A5).

The tests read unrounded values; the written json rounds to 4 decimals for reading only (A9).

No tolerance, offset or test is changed after a CUDA build is seen. A second reading with other tolerances would be a
new registration.

## 5. What each outcome means for Level 1 (docs/research/TRACKING_LEVELS.md)

Level 1 asks for the same numbers on the Mac and on Colab. The reference platform is the one whose builds define
camera truth and the scorecard.

The verdict covers r23 with the frozen court and gate models on a T4; any other detector needs its own device check
or is built and scored on the reference platform (A2: the E1 post-mortem's M6 found the CPU against MPS effect is
model-dependent, E1 minus r23 missed on gsw_bkn_2015 -0.0147 with the detector on CPU against +0.0330 on MPS).

- **CUDA matches CPU**: Colab (CUDA) and the Mac CPU agree within the section 4 tolerances on the two windows. Colab CUDA becomes the
  reference platform for heavy builds, with the Mac's detector-on-CPU setting (`P8_DEVICE_DETECTOR=cpu`, D100) as an
  equivalent local build; MPS stays out of every truth build. The R1.3 truth gap on the Mac is then MPS plus the
  offset rule (D99), as P8_PLATFORM.md proposed. D161 (1) is answered for builds; TRACKING_LEVELS.md step 2's held-out
  scorecard wording is not done by this check and changes only by a D line.
The Mac MPS reference in this check is the unfixed MPS path: the E1 post-mortem (M7, M8) found that MPS on this
Mac corrupts box x coordinates through ultralytics clip_boxes' in-place clamp branch, and the tools-side switch
`P8_MPS_CLIP_FIX` (branch r-p8-mps-fix) is not used here. So "CUDA matches MPS" would mean matching a known-faulty
path. The CUDA arms do not use that switch and are unchanged (A13).

- **CUDA matches MPS**: unexpected (R1.3 on CUDA found 89 candidates where MPS finds 45 at the same offset), and it
  would mean the R1.3 truth came from something other than the device (library versions, the A100). Colab CUDA can
  still be the reference platform, but the R1.3 truth is not reproduced and its cause must be found before Level 1
  step 7.
- **Matches neither**: every platform builds its own boxes. One platform must be fixed as the reference (Colab CUDA on
  a T4 with pinned libraries, recorded here), every truth build and every scorecard row runs there, and Mac builds
  never feed camera truth. The Colab CPU arm says whether the libraries (colab_cpu differs from Mac CPU) or the
  device (colab_cpu equals Mac CPU, CUDA differs) drive it.
- **CUDA not repeatable**: CUDA cannot be the reference platform for exact truth builds without deterministic
  settings, which would be a new registration. CPU builds (Mac or Colab, whichever the Colab CPU arm shows to match)
  become the reference, at their cost.

TRACKING_LEVELS.md words Level 1 step 2 as reproducing the stored scorecard on the held-out game. This check does not
do that: a new held-out number is forbidden (CLAUDE.md rule 4). It compares builds of two training-game windows,
which is what D161 (1) asks.

## 6. Compute budget

T4 time, estimated from the Mac timings and the R1.3 run (estimates, not measured):

| Step | Expected | Basis |
| --- | --- | --- |
| Runtime start, partial clone at the pin, pip install, upload (about 162 MB with the report-only window) | about 6 min | E1 runs (D171); 3.0 MB/s upload (P8_E1_colab.md section 5) |
| Hash checks, plan check, CLIP fingerprint | about 1 min | |
| cuda_a and cuda_b, 2 windows each | about 4 to 8 min | R1.3 on an A100 spent 52 s and 60 s on these windows including the cut and the clock reading; a T4 is slower on the models |
| the report-only window in cuda_a and cuda_b (1102 frames each) | about 4 to 10 min | the Mac MPS build runs about 0.2 s a frame (26 s for 125 frames) |
| colab_cpu, 2 windows | about 10 to 30 min | Mac CPU took 142 s and 122 s on 10 cores; a T4 runtime may have 2 vCPUs |
| Download, delete | about 2 min | |

Expected about 30 to 55 T4-minutes, about 0.6 to 1.05 units at about 1.15 units a T4-hour (D171's measured rate,
CPU hours counted in). Stop rule (A4): at 75 minutes from runtime creation, the lead goes to step 5 at once (get, then
delete_runtime) and does not wait for colab_cpu; the result tar holds every finished arm (cuda_a first). The run also
skips any arm not started by `--deadline-s` (2400 s in the runbook). The ceiling is then about 1.5 units, inside the D169 cap (17.30 of 60
spent).

## 7. Guards

- gsw_ind_2016 and gsw_bkn_2015 are training games of sportvu/splits.json (D98: gsw_ind_2016 is development data,
  reported apart, never pooled). Every command runs `tools.research_testset.refuse_cv_test` and
  `tools.p8_common.guard` on both games, their sections and the three windows before reading anything.
- The pack, the result and the comparison live outside the repository, DATA_ROOT and reports/
  (`tools.p8_common.check_outside`): the pack and result under the P8 data folder, the comparison in the P8
  workspace (cuda_check/). The comparison json holds no local path.
- No frozen file changes (`analytics/`, `sportvu/` untouched); the frozen code is imported.
- The Mac commands that load models (pack: the CLIP fingerprint; compare: none, but it reads the moments) run under
  the heavy-job lock. The runtime run happens on Colab, not on the Mac.
- The runtime refuses to start if any hash, the window plan or the CLIP fingerprint differs, or if a cuda arm finds
  no CUDA device; each window's device record must say cuda for every model in a cuda arm, or compare refuses it.

## 8. Commands

Mac (pack and compare need the frozen CV Python, which has ultralytics and transformers):

    <frozen CV python> -m tools.p8_cuda_check pack --dest <pack folder outside the repository>
    python3 -m tools.p8_cuda_check put --pack <pack folder>              (COLAB_URL and COLAB_TOKEN set)
    python3 -m tools.p8_cuda_check get --dest <result folder>            (COLAB_URL and COLAB_TOKEN set)
    <frozen CV python> -m tools.p8_cuda_check compare --result <result folder> --pack <pack folder> --pin <commit>

Runtime:

    python -m tools.p8_cuda_check run --pack-dir /content/p8_cuda_in --out-dir /content/p8_cuda_out --work /content/p8_cuda_work --install-models --deadline-s 2400

The runbook for the lead is section 10.

## 9. Plumbing dry run on the Mac

Plumbing only, not a result (2026-11-25 05:17 to 05:20 UTC, the evening of 2026-11-24 local time, under the
heavy-job lock; with the pack and code of aa14ea1e, before the amendments of section 11). The pack was built on the Mac (21.1 MB tar; the
gsw_ind_2016_s02 section's sha256 equals the R1.3 manifest's; CLIP revision 3d74acf9, 398 parameter tensors). The
runtime command then ran on the Mac with `--arms colab_cpu --windows gsw_ind_2016_s02_w04 --plumbing` into a fresh
work folder: every hash, the frozen window plan and the CLIP fingerprint checked; the build went through
`safe_build` with `P8_DEVICE=cpu` (device record: cpu for detector, court and gate) in 131 s (the D99 CPU build took
122 s); the result tar and complete.json were written, and its 5 members checked against complete.json and extracted
as `get` does. `compare --plumbing` on that result (written to the P8 workspace as cuda_check_plumbing.json):

- The runtime path against the D99 Mac CPU build of s02_w04: same 125 frames, identical boxes on every frame,
  court mapping gap 0.0 px, the same frozen offset (-0.78 s) and the same candidates at all 7 offsets. The path
  reproduces the Mac build exactly.
- The Mac numbers recomputed by compare equal P8_PLATFORM.md: MPS against CPU 1.04 and 0.90 boxes a frame, same
  count 0.20 and 0.312, court gap 3.84 and 4.65 px; s02_w03 candidates 65, 45, 30, 16 (MPS), 98, 88, 55, 35 (CPU)
  and 97, 88, 55, 35 (detector CPU) at 0.30, 0.62, 0.90, 1.20 s; every recorded frozen offset matches. The rule
  separates Mac CPU from Mac MPS on both windows (four of five tests fail; frame sets agree).
- No CUDA was involved and no verdict was read. The reading stays "plumbing only, not a result".

## 10. Runbook for the lead

While a local web filter blocks the Mac's route to the runtime, steps 2 and 5 go through Google Drive instead
(section 12, A14). Every other step stays as written.

Before: merge this branch into v4-research and push it; PIN is that pushed commit (the runtime clones from GitHub).
The pack is already built on the Mac (pack folder: the P8 data folder, cuda_check/pack; tar sha256 in its
pack_manifest.json); rebuild it with the pack command only if a pinned file changed. Log the runtime's start time.

1. Create a T4 runtime (Colab MCP create_runtime). Setup, one short execute_code call that starts a detached job and
   returns at once:

   ```python
   import subprocess
   PIN = "<pushed merge commit>"
   subprocess.Popen(["bash", "-c", f"""set -e
   [ -d /content/repo ] || git clone -q --filter=blob:none https://github.com/lmcnulty7/NBA-Comp-Viz-V3.git /content/repo
   cd /content/repo && git fetch -q origin && git checkout -q --detach {PIN} && git log --oneline -1
   pip -q install ultralytics==8.4.48 transformers==4.49.0 lap
   python -c 'import torch, ultralytics, transformers; print(torch.__version__, ultralytics.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
   echo SETUP DONE"""], stdout=open("/content/p8_setup.log", "w"), stderr=subprocess.STDOUT)
   ```

   Poll with a new execute_code call (a new executionId each time) until the log ends in SETUP DONE:
   `print(open("/content/p8_setup.log").read()[-1500:])`. If the transformers pin fails to install, rerun the setup
   without it; the run records the version it used.
2. On the Mac, with COLAB_URL and COLAB_TOKEN from get_runtime in the environment:
   `python3 -m tools.p8_cuda_check put --pack <pack folder>` (checks the tar against its manifest, then uploads about
   162 MB to /content/p8_cuda_in, about a minute).
3. Start the run from one execute_code call that waits for it, behind an O_EXCL lock so a repeated call never starts
   a second run. Do not repeat this call with the same executionId; the client returns after about a minute while
   the cell keeps waiting:

   ```python
   import os, subprocess, sys
   try:
       fd = os.open("/content/p8_cuda.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
   except FileExistsError:
       raise SystemExit("already started: poll the log instead")
   os.write(fd, b"started"); os.close(fd)
   p = subprocess.Popen([sys.executable, "-m", "tools.p8_cuda_check", "run", "--pack-dir", "/content/p8_cuda_in",
                         "--out-dir", "/content/p8_cuda_out", "--work", "/content/p8_cuda_work", "--install-models",
                         "--deadline-s", "2400"],
                        cwd="/content/repo", stdout=open("/content/p8_cuda_run.log", "w"), stderr=subprocess.STDOUT)
   print("pid", p.pid, flush=True)
   p.wait()
   print("exit", p.returncode)
   ```

4. Poll at least every 8 minutes for the whole run (D171: a waiting cell alone does not keep a T4), from a separate
   session (a new executionId each time), with no call longer than about a minute (as P8_E1_colab.md section 5):
   `print(open("/content/p8_cuda_run.log").read()[-2000:])`. The log holds one json line per step: setup ok (with
   the GPU name and whether the CLIP revision matches the pack's), one line per arm and window (ok, seconds, court
   weights, devices), a "written" line after each arm, a "skipped" line for an arm past the deadline, and "done" at
   the end. A SystemExit in the log before "setup ok" means a hash, the plan or the CLIP fingerprint differs: stop
   and report, do not work around it. Stop rule: at 75 minutes from runtime creation, go to step 5 at once (get, then
   delete_runtime); do not wait for colab_cpu. The result tar already holds every finished arm (cuda_a first).
5. On the Mac: `python3 -m tools.p8_cuda_check get --dest <result folder outside the repository>` (checks the tar
   against complete.json and every member's sha256, then extracts it). Then delete the runtime (Colab MCP
   delete_runtime) and log its end time and hours.
6. On the Mac, under the lock:
   `python3 -m export.heavy --name "p8 cuda compare" -- <frozen CV python> -m tools.p8_cuda_check compare --result <result folder> --pack <pack folder> --pin <PIN>`.
   It writes cuda_check/cuda_check_result.json in the P8 workspace and prints the flags (pin mismatch, CLIP
   revision, transformers version against the Mac's) and the reading (section 4).

## 11. Amendments before the run (2026-11-25, after the pre-run ML review of aa14ea1e)

No tolerance, offset or rule test changed. Each item below was made before any CUDA build.

- A1. Runbook step 4: polls come from a separate session, a new executionId each time, no call longer than about a
  minute.
- A2. Section 5: "agree within the section 4 tolerances on the two windows" replaces "build the same boxes"; the
  verdict's scope (r23 with the frozen court and gate models on a T4) is stated; step 2's held-out wording is not done
  by this check.
- A3. The run records `torch.backends.cudnn.allow_tf32` and `torch.backends.cuda.matmul.allow_tf32`; the verdict holds
  for a T4 and does not cover an A100 or an L4.
- A4. Stop rule: at 75 minutes from creation, get and delete at once. `run --deadline-s` skips any arm not started by
  then; skipped arms are recorded in run_record.json and the log.
- A5. Section 4: cuda_a against colab_cpu is reported, not in the verdict; without colab_cpu a "matches neither" is
  not traced (the result carries an attribution note).
- A6. compare refuses a result without `--pack` and `--pin`, prints the run's commit beside the pin and flags a
  mismatch.
- A7. run_record.json and the compare flags say whether the CLIP revision equals the manifest's (the fingerprint must
  still match).
- A8. Frames with a homography are counted per build and per pair (reported only).
- A9. The tests read unrounded values; only the written json is rounded.
- A10. compare flags a runtime transformers version that differs from the Mac's.
- A11. Section 9's date is in UTC with the local date beside it.
- A12. A report-only window, gsw_bkn_2015_s04_w00 (a training game; guarded), built in cuda_a and cuda_b with the r23
  court model (`COURT_GRID_WEIGHTS`, set for that build only), as the registered P8 builds are. The E1 post-mortem's
  M6 found it the most device-sensitive window known (r23 missed 362 on MPS against 784 with the detector on CPU).
  Its Mac builds exist in the P8 workspace, so nothing new is downloaded: `builds/e1m6_r23_detcpu` (detector on CPU,
  court and gate on MPS) and the registered MPS build `builds/r23`; its clip and section time map come from the
  workspace. compare reports it with the five tests against both, at the fixed offsets plus the Mac builds' frozen
  offsets and the stored label offset; it never enters the verdict. Note that its "CPU" reference has the court on
  MPS, so its court test against CUDA compares two court devices. The pack grows to about 162 MB.
- A13 (2026-11-25, after A12). Section 5 states that the Mac MPS reference is the unfixed MPS path (clip_boxes'
  in-place clamp corrupts box x coordinates on this Mac, E1 post-mortem M7 and M8), so matching it means matching a
  known-faulty path. No arm, test or reference changes.
- A14 (2026-11-25). Transfer through Google Drive (section 12): the Mac's HTTP route to the runtime is blocked by a
  local web filter, so the pack goes up through the Drive mount and the Drive API, and the result comes back the same
  way. Nothing in the builds, the arms, the tests or the reading changes; the run is still checked by every hash.

## 12. Transfer through Google Drive (A14)

Tool: `tools/p8_drive_xfer.py` (stdlib only). The Mac's Drive mount and the Colab runtime use the same Google account.
The Mac takes its access token from the Colab sign-in helper; the runtime reads it only from the environment variable
DRIVE_TOKEN of the one subprocess that needs it. The tool never prints, logs or writes a token.

Up (replaces step 2):

1. On the Mac: `python3 -m tools.p8_drive_xfer stage --pack <pack folder> --drive-dir "<Drive root>/nba_p8_xfer/cuda_check"`,
   where the Drive root is the parent of DATA_ROOT. It copies the pack's files there (never under DATA_ROOT, reports/
   or a checkout), polls the Drive API until every file shows the local size and md5, and writes drive_ids.json
   beside the pack manifest (per file: name, Drive id, size, md5, sha256). The 162 MB pack uploads at the Drive
   client's speed; the command waits up to an hour.
2. `python3 -m tools.p8_drive_xfer push-map --pack <pack folder>` prints the id map on one line. It holds no secret
   and no local path.
3. One short execute_code call writes that line to /content/p8_drive_ids.json and starts the fetch detached, with
   the token in that subprocess's environment only:

   ```python
   import os, subprocess
   open("/content/p8_drive_ids.json", "w").write('<the push-map line>')
   env = {**os.environ, "DRIVE_TOKEN": "<access token with drive.readonly>"}
   subprocess.Popen(["python", "-m", "tools.p8_drive_xfer", "fetch", "--ids", "/content/p8_drive_ids.json",
                     "--dest", "/content/p8_cuda_in"], cwd="/content/repo", env=env,
                    stdout=open("/content/p8_fetch.log", "w"), stderr=subprocess.STDOUT)
   ```

   Poll /content/p8_fetch.log with new short calls until it shows {"files": ...}. fetch reads each file in Range
   requests, resumes from a .part file, and refuses any file whose size, md5 or sha256 differs from the map. Then
   start the run as in step 3. The run's own checks (the pack tar against its manifest, every member) still apply.

Down (replaces step 5). The result tar is about 10.7 MB for all three arms (cuda_a and cuda_b about 4.9 MB each,
4.0 MB of it the report-only window; colab_cpu about 0.9 MB; the one-arm, one-window plumbing tar was 450 KB). The way
back is Drive with the drive.file scope, which Lucien approved on 2026-11-25 beside drive.readonly. drive.file lets
this OAuth client create files and touch only files it created, so the folder is made by the same client first.

1. Before the run, on the Mac: `python3 -m tools.p8_drive_xfer make-folder --name nba_p8_xfer_out` prints the
   folder's id (not a secret). It reuses a folder of that name the client already owns.
2. After "done", one short execute_code call uploads the two result files, with the token in each subprocess's
   environment only:

   ```python
   import os, subprocess
   env = {**os.environ, "DRIVE_TOKEN": "<access token with drive.file>"}
   for name in ("p8_cuda_gsw_ind_2016.complete.json", "p8_cuda_gsw_ind_2016.tar"):
       r = subprocess.run(["python", "-m", "tools.p8_drive_xfer", "upload", "--src", f"/content/p8_cuda_out/{name}",
                           "--drive-dir-id", "<folder id>", "--name", name], cwd="/content/repo", env=env,
                          capture_output=True, text=True)
       print(r.stdout.strip() or r.stderr[-500:])
   ```

   Each upload is resumable, checks Drive's size and md5 against the local file and prints one line: name, Drive id,
   size, md5, sha256. The tar (about 10.7 MB) goes up in seconds, so the call stays short.
3. On the Mac, for each printed line:
   `python3 -m tools.p8_drive_xfer pull --file-id <id> --name <name> --size <size> --md5 <md5> --sha256 <sha256> --dest <result folder>`.
   pull checks size, md5 and sha256 against what the runtime printed (a channel Drive does not touch) and refuses on
   any mismatch. Then `python3 -m tools.p8_cuda_check unpack --dest <result folder> --expect-sha256 <the tar's
   sha256>` checks the tar against complete.json and every member, and extracts it. Then delete the runtime and
   compare as in step 6.

Base64 through execute_code output is not used: the output enters the lead's context, and even trimmed to the frames
files and gzipped, the three arms are about 1.7 MB (hundreds of thousands of tokens).

Token exposure: the token typed into an execute_code call is in that session's transcript for its lifetime (about an
hour). It is the whole OAuth token, so it carries the Colab scopes too. A token minted from the refresh token with only
the Drive scopes, by the sign-in helper outside the repository, would narrow this; that is Lucien's or the lead's
change, not this tool's.
