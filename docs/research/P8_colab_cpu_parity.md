# P8: does a Colab CPU runtime reproduce the Mac's builds? Design fixed before any run

Written 2026-11-26 under PLAN.md D172 (the research loop), approved by Lucien as a Colab CPU parity test, before any
Colab or Mac build of this test. The code is `tools/p8_colab_cpu.py` (tests: `tests/test_p8_colab_cpu.py`). Sections
1 to 9 hold no result. The game is gsw_cha_2016, a pool development game of the reset-fix arm (P8_resetfix.md); no
held-out game and no D85 test game is read, and no held-out metric is computed (CLAUDE.md rule 4).

## 1. The question

If a Colab CPU runtime builds a window exactly as this Mac does, later reference rebuilds could be split between the
Mac and Colab. The CUDA check (P8_cuda_check.md, section 13) found that a T4 build matched neither Mac build: the
detector agreed with the Mac CPU detector within tolerance, but the court mapping sat 8 to 11 px away. Its Colab CPU
arm was not built, so that gap could not be traced to the device or to the libraries (torch, OpenCV and NumPy all
differed from the Mac's). So: **with the Mac's library versions pinned, does a Colab CPU build of the same windows,
from the same clips, code and weights, equal the Mac's all-CPU build byte for byte?**

## 2. A platform fact found while designing (before any build)

The Mac's frozen CV Python is an x86_64 build that runs under Rosetta on Apple silicon. Its probe (the `probe`
command, recorded in the pack manifest):

| Library | What the Mac process uses |
| --- | --- |
| torch 2.2.2 | `get_cpu_capability()` = "NO AVX" (the default kernels), 10 threads |
| oneDNN 3.3.2 (torch's) | ISA "Intel SSE4.1" |
| MKL (torch's) | "oneMKL 2022.0 Update 2 ... Intel SSE4.2 enabled processors" |
| OpenCV 4.13.0 (opencv-python-headless 4.13.0.92) | SSE4.2 at most; AVX, FP16, AVX2, AVX512-SKX not available |
| NumPy 1.26.4 | SSSE3, SSE41, POPCNT, SSE42 found; no AVX |

A Colab Xeon offers AVX2 or AVX-512, and each library picks its kernels at run time, so the same versions can compute
different bits there. The test therefore builds two Colab arms:

- **plain**: the pinned stack as it installs (what a plain rebuild on Colab would get);
- **isa**: the same stack with every library capped at the Mac's level, through environment variables set for the
  build subprocess only: `ATEN_CPU_CAPABILITY=default`, `DNNL_MAX_CPU_ISA=SSE41` (and `ONEDNN_MAX_CPU_ISA`),
  `MKL_ENABLE_INSTRUCTIONS=SSE4_2`, `OPENCV_CPU_DISABLE` (each OpenCV feature available on the runtime but not on the
  Mac) and `NPY_DISABLE_CPU_FEATURES` (each dispatchable NumPy feature found on the runtime but not on the Mac). The
  values are computed by `isa_env` from the two probes, so nothing is typed by hand; the run probes again under the
  caps and records what each library then reports.

Thread counts are not matched (the Mac uses 10, the runtime has 2 vCPUs); the result says so if it matters.

## 3. What is built

| Item | Value |
| --- | --- |
| Build path | the reset-fix build path with every model on CPU: `tools.p8_frozen.safe_build` (frozen `sportvu.rebuild.build`, `build_trajectories.py --pregate --no-video`, `autolabel.gate_args()`: gate head v2), `P8_DEVICE=cpu`, `P8_TRACKER_RESET_FIX=1`, the r23 detector (models/player_detector.pt, sha256 4c886d5f...) and the r23 court model (`COURT_GRID_WEIGHTS`, checked against reports/r23_verification.json). Nothing else is set; TRACKER_REID and the clip fix are unset |
| Why all CPU | the reference platform (D175) has the court and gate on MPS, which Colab cannot run. The like-for-like reference is the same windows built on this Mac with every model on CPU, by the same `run` command from the same pack |
| Core windows | gsw_cha_2016_s02_w03 (152 built frames in the baseline, 1 reset) and gsw_cha_2016_s00_w01 (169 frames, no reset) |
| Tail windows | gsw_cha_2016_s00_w00 (435 frames, no reset) and gsw_cha_2016_s05_w00 (1,053 frames, 2 resets with the fix), the windows the task named; built on Colab only if time allows, and then on the Mac too |
| Why not the named windows first | the named pair is 1,488 frames per arm. The Mac's all-CPU build runs about 0.8 to 1 s a frame (P8_PLATFORM.md: 142 s and 122 s on 145 and 125 frames); a 2-vCPU runtime is expected to be several times slower, so two arms on the named pair cannot fit the 1-hour cap. The core windows are the shortest pool windows of the game with and without a reset |
| Clips | the reference root's window clips (work_ref/label_build), shipped in the pack with their sha256; never cut again |
| Plan | the frozen window plan (`tools.p8_frozen.label_windows` from the sections' local time maps, shipped); the runtime recomputes it and stops if it differs |
| Code | the runtime clones the pushed base commit (6261476..., v4-research) and overlays `tools/p8_colab_cpu.py` from the pack (sha256 checked by the Drive fetch and again against the pack manifest); the run records both |
| CLIP | the gate's CLIP is checked by its parameter fingerprint (the CUDA check pack's Mac value); every run recomputes it and stops on a mismatch |

Colab build order, fixed now: plain s02_w03, plain s00_w01, isa s02_w03, isa s00_w01, plain_b s02_w03 (a repeat of
plain: repeatability), plain s00_w00, plain s05_w00.

**Start rule.** A step starts only if its predicted time fits before the end time (47 minutes after the runtime job
starts). The prediction is the measured seconds per planned frame of the same arm family (plain and plain_b share
one) times the window's planned frames; the first isa step uses the plain rate times 3; the first step always starts.
A skipped step is recorded with its prediction; later steps are still considered. A build that fails is tried once
more; a second failure stops the run (the steps done stay in the result).

## 4. Environment

Colab: one CPU runtime (SHAPE_STANDARD, accelerator NONE). The image is Python 3.13, and torch 2.2.2 has no 3.13
wheels, so the job builds a venv with uv: Python 3.12.2 (the Mac's version; python-build-standalone, where the Mac
has conda-forge's build), then:

- torch 2.2.2 and torchvision 0.17.2, CPU wheels;
- numpy 1.26.4, scipy 1.13.1, opencv-python-headless 4.13.0.92 (the wheel the Mac's cv2 comes from),
  transformers 4.49.0, tokenizers 0.21.4, safetensors 0.7.0, huggingface_hub 0.36.2, lap 0.5.13, pillow 12.2.0,
  scikit-learn 1.8.0, joblib 1.4.2, threadpoolctl 3.5.0, matplotlib 3.10.9, all held by a constraints file;
- ultralytics 8.4.48 and ultralytics-thop 2.0.19 without their dependencies (ultralytics asks for opencv-python,
  which would put a second cv2 beside the headless one);
- the rest of what the build path imports at the Mac's versions (pandas 3.0.3, pyarrow 22.0.0 and others; the list
  is `SOFT_PINS`). A pin that cannot install prints PIN_FAILED and falls back to the latest version.

Every version on both sides is recorded by the probe (each distribution's version, torch's build configuration, the
OpenCV build's CPU lines, the CPU model). If a core pin cannot install, the job stops; if a soft pin falls back, the
result reads "versions differ" beside its reading. The probe also hashes a few fixed computations per library
(a convolution, a matrix product and a softmax in torch; a resize and a blur in OpenCV; a sum and a product in NumPy),
which shows which library first computes different bits.

## 5. Measures and the reading rule (fixed now)

Per window, each Colab arm against the Mac's all-CPU build (and plain_b against plain):

| Measure | Definition |
| --- | --- |
| Equality | frames.json without "source" (the absolute clip path), as canonical JSON; trajectories.json and identity.json byte for byte |
| Frame sets | common frames over the union |
| Frames within 1 px | share of common frames whose boxes match one to one with every bbox coordinate within 1 px (bboxes are integers): same count and every box paired (an assignment that minimises the summed L-inf distance) |
| Box offset | median and max L-inf bbox distance, in px, over the assigned pairs within 32 px; unmatched boxes counted; the foot_stab distance of the same pairs reported |
| Track ids | among pairs within 1 px, the share with the same tid, and the share consistent with the most frequent tid mapping |
| Resets | identity's camera_cut_resets on each side; the reset log must hold one entry per reset, each stripping 2 callbacks |
| Court mapping | median and p90 of the per-frame gap between the two court homographies at three court points (as the CUDA check) |

Checks on every build before it is compared: the device record says cpu for detector, court and gate, reset_fix true,
no clip fix; the reset log matches the identity file. A build that fails a check is not compared and is listed.

Reading of a window:

- **identical**: frames (without source), trajectories and identity equal;
- **near**: not identical, but the frame sets are equal, at least 0.99 of common frames match within 1 px, the reset
  counts are equal and the court mapping gap median is at most 1.0 px (the CUDA check's tolerance; the Mac's CPU
  against MPS gap is 3.8 to 4.7 px);
- **different**: otherwise.

Reading of an arm: "not read" unless it was compared on at least one window with a reset (by the Mac build) and one
without; else the weakest of its window readings. The answer reports the plain arm's reading (does the pinned stack
reproduce?) and the isa arm's (does it with the Mac's instruction ceiling?), and whether plain_b equals plain. If a
soft pin fell back, the reading carries "versions differ".

Report only, never in the reading: the Mac's all-CPU build and the plain build against the reference-platform build
of the same window (detector on CPU, court and gate on MPS), where one exists with the same reset behaviour (the
reset-fix smoke's builds of s00_w00 and s05_w00, or the unfixed baseline for a window with no reset).

No tolerance or test changes after a Colab build is seen.

## 6. What each outcome means

- **plain or isa identical** on both core windows: a Colab CPU runtime with that setting reproduces the Mac's all-CPU
  build exactly on these windows. It does not reproduce the reference platform as fixed by D175 (court and gate on
  MPS). A split of reference rebuilds would need the reference to become all-CPU on both machines, or a hybrid
  check of its own; either is a new D line for the lead and Lucien. The cost of all-CPU on the Mac is about twice the
  detector-only CPU build (P8_PLATFORM.md).
- **near**: the platforms agree closely but not exactly; mixing their builds in one truth build or scorecard would
  mix two build sources. Not usable for byte-exact work without a decision.
- **different**: Colab CPU builds cannot stand in for Mac builds; the probe hashes and the per-file equality say
  which library departs first.
- **plain_b differs from plain**: the Colab CPU build is not repeatable; nothing else is read as a platform verdict.

## 7. Budget and runtime rules

- One CPU runtime, capped at 1 hour from creation, deleted when done (and at the cap whatever is running). Created
  and deleted times go in the lead's scratchpad ledger (colab_ledger.tsv).
- Polls at most every 5 minutes, never a gap above 8 minutes, each a short execute_code call with a new executionId.
- Inputs and outputs move only through Google Drive (tools/p8_drive_xfer.py, D177), with size, md5 and sha256 checks.
  The result upload is chained into the runtime job: after every build the run rewrites its result tar and uploads
  it with its complete.json under names numbered by the steps done.
- The Drive-only token (drive.file and drive.readonly, about 59 minutes) is minted by drive_token.py just before the
  job starts; it sits in an unexported shell variable of the job and reaches only the fetch and the run, and the run
  removes it from its own environment and passes it only to each upload subprocess. Where it otherwise sits is as
  P8_cuda_check.md section 12 says (the call text, the kernel history until the runtime is deleted).
- Expected: setup about 5 to 8 minutes; the core plain steps perhaps 10 to 25 minutes; the isa steps slower. CPU
  runtimes cost far less than a T4; the units are read by Lucien.
- Mac: the all-CPU builds of the core windows (about 5 minutes) under the heavy-job lock, while the truth rebuild is
  paused by the lead with its STOP file; the tail windows on the Mac only if Colab built them.
- If Colab sign-in has expired the test stops and the lead is told. A step that fails twice stops the test.

## 8. Guards

- gsw_cha_2016, its sections and its windows pass `refuse_cv_test` and `tools.p8_common.guard` in every command.
- Pack, work and result folders lie outside the repository, DATA_ROOT and reports/ (`check_outside`).
- No frozen file changes; the frozen build code is imported and run as a subprocess.
- The committed record holds no local path, email address or account name (`export.schemas.find_private`).

## 9. Commands

Mac (frozen CV Python for pack and run; the reference root as V3_P8_ROOT for pack):

    V3_P8_ROOT=<reference root> <frozen CV python> -m tools.p8_colab_cpu pack --dest <P8 data>/colab_cpu/pack
    python3 -m tools.p8_drive_xfer stage --pack <pack> --drive-dir "<Drive root>/nba_p8_xfer/colab_cpu"
    python3 -m tools.p8_drive_xfer push-map --pack <pack>
    python3 -m tools.p8_drive_xfer make-folder --name nba_p8_xfer_out
    python3 -m tools.p8_colab_cpu job --base <base commit> --folder-id <result folder id>
    python3 -m export.heavy --name "p8 colab cpu mac" -- <frozen CV python> -m tools.p8_colab_cpu run --platform mac \
        --arms mac --windows gsw_cha_2016_s02_w03,gsw_cha_2016_s00_w01 --pack-dir <pack> --work <P8 data>/colab_cpu/mac_work \
        --out-dir <P8 data>/colab_cpu/mac_out
    python3 -m tools.p8_colab_cpu unpack --dest <P8 data>/colab_cpu/mac_out --name p8_colabcpu_mac
    python3 -m tools.p8_drive_xfer pull ... --name p8_colabcpu_colab.tar (and .complete.json)
    python3 -m tools.p8_colab_cpu unpack --dest <dir> --name p8_colabcpu_colab --expect-sha256 <printed sha256>
    python3 -m tools.p8_colab_cpu compare --mac <P8 data>/colab_cpu/mac_out --colab <dir> --pack <pack> --report

Runtime: one execute_code call writes the push-map line to /content/p8_drive_ids.json and starts the job script
detached (`bash -c <script>`, log /content/p8_job.log), with the Drive-only token in that process's environment only.

## 10. Amendments before any Colab build

No window, arm, measure, tolerance or reading changed. Each item below was made before any Colab build existed.

- A1 (2026-11-26, approved by the lead before any run). The lead approved the isa arm, the plain_b repeat and the
  shorter core windows.
- A2 (2026-11-27). The first runtime (p8-colab-cpu-1, CPU standard, created 23:36:10Z, deleted 00:29:35Z) built
  nothing. The job stopped about a minute in, at the core install: the Linux opencv-python-headless 4.13.0.92 wheel
  declares numpy>=2, against the numpy 1.26.4 pin. The Mac's own install has the same metadata (that wheel, with
  numpy 1.26.4), so the faithful fix is to install it without dependencies, as the Mac has it. It moves from the core
  pins to the no-dependency pins, and the job prints the cv2 and numpy versions right after. Before the second
  runtime, every install stage was resolved on the Mac for Linux x86_64 and Python 3.12 (`uv pip compile`, uv 0.13.10,
  nothing installed). All resolve. Only psutil 5.9.0 and kiwisolver 1.4.4 have no wheel and are built from source on
  the runtime; if a build fails, the soft-pin fallback applies and the result reads "versions differ". The runtime then
  sat idle for about 50 minutes, because the job's failure was not seen until a poll at 00:29Z (a polling lapse; the
  polls are every 5 minutes from now on). The Mac builds were made on 2026-11-26 from 22:16Z to 22:22Z under the lock,
  from this same pack's clips, plan and weights. The pack is not rebuilt. The runtime's run uses the tool from the
  pack (sha256 19ec7511...), the same code the Mac builds ran. Only the job text, which is pasted into the
  execute_code call and never runs from the pack, comes from this amendment's commit. On the Mac, `pip show` in
  the frozen CV Python lists numpy 1.26.4 and opencv-python-headless 4.13.0.92 (Requires: numpy, unversioned in
  the installed metadata's summary). The job now fails fast. An ERR trap prints "JOB FAILED" with the exit code and line;
  after the core and no-dependency installs it imports cv2, numpy and torch and prints their versions (CHECK1); after
  the soft pins it imports cv2, numpy, torch and ultralytics and prints their versions (CHECK2). The first poll comes
  3 minutes after the job starts.
