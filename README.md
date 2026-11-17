# From broadcast pixels to possession value, measured against ground truth at every step

A basketball analytics product built on player tracking. The models run on real tracking data. A second
pipeline tracks players from broadcast video, as a stand-in for the tracking that teams buy from vendors. It is
scored against that real data, and today it feeds only a demo.

**Site:** https://lmcnulty7.github.io/NBA-Comp-Viz-V3/ (the public copy: possession review, prep tab, results).
Court playback and the talk slides are on a private copy, available on request (`PLAN.md` D60).

## The short version

Two to three minutes to read.

**What it is.** Models that answer coaching questions from player tracking data. How good was this shot (xFG)?
How many points is this possession worth right now (EPV)? Where would an average defense have stood (ghosting)?
They run on public SportVU tracking (the league's in-arena camera tracking of that time) of the 2015-16 season
(`reports/results.md`). The site shows them as a possession review and a prep tab, after a short demo of the
broadcast pipeline (`PLAN.md`).

No model result comes from the broadcast pipeline. The tracking models run on SportVU positions, and two lanes
(lineups, win probability) run on play-by-play. The broadcast pipeline has no ball, no stable player identity
and no reliable team label yet, so it feeds only the demo, and the demo is labelled an illustration.

**What the models found.** Each model with a held-out score was scored on it exactly once
(`reports/results.md`). Brackets are 95 percent bootstrap intervals by game.

- **Shot quality (xFG) works.** Held-out log loss 0.6232 [0.6132, 0.6327], against 0.6735 for a model that knows
  only where the shot was taken.
- **Possession value (EPV) works, narrowly.** Held-out MSE 1.2362 [1.1899, 1.2812], against 1.3319 for a
  constant.
- **Decision quality failed.** It does not track a player's on-court offensive rating: Spearman
  -0.014 [-0.116, 0.100] held out. It stays on the site as tested, no signal.

**How well the tracking works.** The adopted build, `r23`, is scored against SportVU on footage held out from
training. In one line: on the held-out game, for the players it finds, the median position error is 1.92 ft,
and it misses 25.1% of the players in view. Every value is copied from the scorecard of its game,
`reports/scorecard/<game>__r23.txt`.

| Row and target | Held-out game `gsw_phx_2016` | Held-out arena `cle_nyk_2015` | Held-out arena `cle_gsw_2016` |
| --- | --- | --- | --- |
| Position error, median / ninetieth percentile (p50 / p90) <= 2.0 / 5.0 ft | **met**: 1.92 / 4.98 ft | not met: 2.25 / 5.46 ft | not met: 2.22 / 5.5 ft |
| Near-field bias (median error toward the camera), near / far third of the court, within +-0.5 ft | not met: -0.67 / +1.52 ft | not met: -0.63 / +1.66 ft | not met: -0.12 / +2.00 ft |
| Coverage (share of live wide-angle seconds with accepted positions) >= 50% | **met**: 99.3% | **met**: 92.3% | not met: 39.7% |
| Missed players (players in view with no box) <= 5% | not met: 25.1% | not met: 24.4% | not met: 26.2% |
| Ghost boxes (boxes on no player), referees excluded, <= 5% | not met: 15.1% | not met: 14.8% | not met: 14.8% |
| Team labels >= 95% | not met: 81.9% | not met: 73.7% | not met: 78.2% |

- **Every row except coverage is scored only on frames with usable truth**: 1447 of 9500 window frames on the
  held-out game (stated at the top of its scorecard). The windows are stretches of running clock, not the whole
  game, and the scored frames lean toward frames the pipeline already handled, so those rows are optimistic.
  Coverage counts all live wide-angle seconds of the same windows.
- **Position error is met on the point estimate only.** Its p90 is 4.98 ft against 5.0 ft, with a 95% interval
  of [4.42, 6.02] (`reports/r23_verification.txt`).
- **One model did see the held-out game.** The frame gate, which picks the wide-angle frames, trained on 294 of
  its frames (`reports/sportvu_splits.txt`), so the choice of scored frames and the coverage row are kinder
  there than on a new game. The court and player models did not train on it.
- **Missed players, ghost boxes and team labels fail on every game.** Fixing detection, teams and identity is
  out of scope for this build (`PLAN.md`).

**Where to look next.** The results with what failed: [`reports/results.md`](reports/results.md). The
scorecards with their caveats: [`reports/scorecard/`](reports/scorecard/). What the tracking pipeline will and
will not claim: [`CLAIMS.md`](CLAIMS.md). That file was written before the model lanes existed and has no row
for them, so the model claims are in `reports/results.md`. Its out-of-scope list dates from then and still
names shot-quality models, and its row A2 quotes prototype stage numbers (see the broadcast pipeline section
below). The demo script: [`docs/SCRIPT.md`](docs/SCRIPT.md). The plan and every decision: [`PLAN.md`](PLAN.md).

## Run it

All commands run from the repository root, with a `python3` that has the packages in `requirements.txt`.

```bash
python3 -m pytest -q tests -m "not slow"         # the fast tests
python3 -m export.sample --out web/data          # synthetic site data for web work, never a result
python3 -m export.site_data                      # the real site data, one command, end to end
python3 -m export.schemas --validate web/data    # check the site data against the contract
```

The site is a static export. Inside `web/`:

```bash
npm ci
npm run check            # types, lint, contract check of the data, copy checks, browser tests
npm run build:public     # stills and derived data only
npm run build:private    # adds the short clips, for the access-controlled copy
```

Notes:

- `python3 -m export.site_data` needs `DATA_ROOT`, the folder that holds the SportVU contract tables and the
  model outputs (`docs/DATA_CONTRACT.md`). Without it, the synthetic data is enough to build and test the site.
  A site built from synthetic data labels every number as fixture output.
- Long jobs (the whole test suite, the data export, the site build) take a machine-wide lock so that only one
  runs at a time: `python3 -m export.heavy --name "what it is" -- <command>`.
- On the development Mac, `python3` must be the system interpreter. Another distribution's packages shadow the
  repository's `tests/` folder (`CLAUDE.md`).
- `make help` lists the older shortcuts for the analytics lanes. `docs/run_order.md` has the full order from data
  collection to every model. Never run a lane with its default paths against the real data: it would overwrite
  the frozen outputs and reports (`CLAUDE.md`, hard rules).

## How the pieces fit

Python produces data, the web only reads it, through one JSON contract.

| Path | What it holds |
| --- | --- |
| `analytics/` | The model lanes. The eleven with results (xFG, rebounds, matchups, gravity, ghosting, embeddings, EPV, decisions, lineups, win probability, load) are frozen at tag `freeze-v1`. |
| `sportvu/` | The SportVU scoreboard for the tracking pipeline: fetch, sync, truth, scorecard. Frozen with the lanes. |
| `export/schemas.py` | The site data contract. Every file under `web/data` validates against it. |
| `export/site_data.py` | Reads the reports and the model outputs, writes `web/data`. It fits nothing. |
| `export/sample.py`, `export/heavy.py` | Synthetic site data for web work, never a result. The lock that lets one long job run at a time. |
| `export/demo_bridge.py` | Turns the broadcast build of the demo game into contract frames and runs the frozen models on them, predict only. |
| `tools/frozen_models.py` | Rebuilds the frozen model objects by rerunning frozen code on training games, and keeps a result only when it reproduces the frozen record (`PLAN.md` D45, D49). |
| `web/` | The site: a static export in TypeScript. Pages read data only through one data source module. |
| `reports/` | Generated reports: one per lane, `reports/results.md`, and `reports/scorecard/`. |
| `gate/`, `court/`, `detect/` and the scripts at the root | The broadcast pipeline, described below. |

Two builds of the site come from the same code. The public build carries stills and derived data and fails if a
video file is in its output. The private build adds short clips and sits behind sign-in (`PLAN.md` D21, D29).

## The tracking measurement in detail

**What is held out.** The split is `sportvu/splits.json`. The court model's SportVU labels come from 4 games
at one arena (`reports/sportvu_splits.txt`). The held-out game was played at that same arena. The held-out
arena is Cleveland. No frame of the two scored Cleveland games is in what any training script reads, and no
Cleveland game is in the court model's SportVU labels (`reports/sportvu_splits.txt`, which matches games by
name; `reports/r23_verification.txt`). Two models did see the building: the frame gate trained on frames of
another Cleveland home game, and the first court model's training set holds frames of later Cleveland home
games (both below). Every training script refuses the held-out ids.

**The rest of the scorecard.** The table at the top holds the rows that are judged on every game. The other
rows, from the same files:

| Row | Target | Held-out game `gsw_phx_2016` | Held-out arena `cle_nyk_2015` | Held-out arena `cle_gsw_2016` |
| --- | --- | --- | --- | --- |
| Generalisation | held-out arena within 1.5x of the held-out game | the reference | **met**: p50 1.17x, p90 1.10x | **met**: p50 1.16x, p90 1.10x |
| Identity | reported, no target yet | 21.39 id switches / possession | 31.07 | 14.29 |
| Court line error | reported | not measurable | not measurable | not measurable |
| Frames with usable truth | | 1447 of 9500 | 485 of 6393 | 158 of 3380 |

The held-out game's scorecard states the frames without usable truth as a share: untestable 84.8%. The arena
passes the generalisation row and still misses the position target itself.

**Three builds, one set of boxes.** Each of these builds scores the same detected boxes on the same frames.
Only the court model changes.

- `v3` is the pipeline as tagged (`v3-frozen`). On the held-out game its position error is p50 7.33 ft,
  p90 15.35 ft (`reports/scorecard/gsw_phx_2016__v3.txt`).
- `r23` is the same court model fine-tuned on labels derived from SportVU on the training games. It was adopted
  on 2026-10-04 (`CLAIMS.md`). Position error on the held-out game: p50 1.92 ft, p90 4.98 ft
  (`reports/scorecard/gsw_phx_2016__r23.txt`).
- `r23_cam` adds a camera model. It was measured and not adopted: not every row improves on the held-out arena,
  and the two builds differ only on held-out footage, so choosing the better one there would be a selection on
  held-out data (`PLAN.md` D42). Its scorecards sit beside the others in `reports/scorecard/`.

The folder also holds the scorecards of a public court model that was tried as a baseline, in two settings
(`public`, `public_conf005`; `ROADMAP_LOG.md`), and the files that compare builds side by side
(`compare__<builds>`).

**Caveats that ship with every scorecard.** Each scorecard file prints these in full.

- The frames with usable truth are the ones where the first version of the pipeline already matched enough
  players, so they lean toward frames it handled.
- The truth mapping is fitted through the pipeline's own boxes and is valid only near those feet. The court line
  row is therefore not measurable, and the counts of missed players and ghost boxes carry that error. Position
  error and near-field bias compare only matched players near the feet.
- The scores are on a local rebuild of windows of the harvested video, not on the production artifacts.
- Wide frames are chosen by the frame gate, which trained on frames of the held-out game (above) and of
  another game at the held-out arena.
- The referee class is the detector's own. Its recall was never measured against human labels.
- Identity switches are counted on sparse samples, so the count is a lower bound.
- Team accuracy uses the best mapping for each window, so it is an upper bound.

**More limits, from `CLAIMS.md` and `ROADMAP_LOG.md`.**

- The held-out game shares its arena and season with the training games, so the arena games are the
  cross-arena evidence.
- The held-out arena means unseen games and an unseen season, not an unseen building: the first court model's
  training set holds 56 frames of 2023-24 Cleveland home games by the verification's count (`CLAIMS.md`,
  `reports/r23_verification.txt`). A later tagging of that set by arena counts Cleveland as 9 clips / 88 images
  and calls its list a lower bound (`ROADMAP_LOG.md`, `sportvu/old_set_groups.json`). The two counts use
  different units, and neither file reconciles them.
- The adopted court model is the best of three training seeds of its recipe. With the arena games pooled, the
  recipe's expected position error, p50 / p90, is about 2.35 / 5.9 ft, not this model's 2.24 / 5.47
  (`ROADMAP_LOG.md`).
- One broadcast camera gives about 1.5 ft median at best. An arena system with 14 cameras reaches about
  1 inch. The comparison this project claims is on output format and on honesty about error, not on accuracy.

**Where the scoreboard lives.** Each module runs as `python -m sportvu.<module>`. SportVU data stays under
`data/sportvu/`, which git ignores. Do not run `sportvu.bench` with its default paths: it rewrites the tracked
scorecards.

| Step | Module | Output |
| --- | --- | --- |
| Fetch and parse a SportVU game log | `sportvu/fetch.py` | `data/sportvu/<game>_moments.json`, `reports/sportvu_fetch_<game>.json` |
| Production frame to game clock, mirror and offset | `sportvu/sync.py` | `data/sportvu/sync/`, `reports/sportvu_sync_<game>.json` |
| Time maps for the local video files | `sportvu/local_sync.py` | `data/sportvu/sync/<section>_local_timemap.json` |
| Windowed local rebuild with a per-frame record | `sportvu/rebuild.py` | `data/sportvu/build/` |
| Truth mapping for each frame, fitted against SportVU | `sportvu/truth.py` | `data/sportvu/truth/`, `reports/sportvu_truth_<game>.json` |
| One scorecard with every target | `sportvu/bench.py` | `reports/scorecard/<game>__<build>.{json,txt}` |
| Two builds side by side, with paired differences | `sportvu/compare.py` | `reports/scorecard/compare__<builds>.{json,txt}` |
| The camera model candidate | `sportvu/camera_track.py` | `data/sportvu/replay/<name>/` |

## The models in detail

Eleven lanes and a location-only xFG baseline, frozen at tag `freeze-v1`. The tracking data is split by game:
604 train and 17 held out. Lineups and win probability read play-by-play of later seasons instead, and hold out
the 2024-25 season. Embeddings have no held-out score by design. Each lane chose its model on training data
only, by a rule fixed before it ran. Then each lane with a held-out score scored its held-out data once. None of
the failures was rescued after the held-out numbers were seen. The held-out and verdict cells below are copied
from the summary table of [`reports/results.md`](reports/results.md), which also has the out-of-fold values and
the reasoning.

| Lane | Question | Metric | Held-out | Verdict |
| --- | --- | --- | --- | --- |
| xfg | will this shot go in? | log loss | 0.6232 [0.6132, 0.6327] | works; beats the location-only baseline by 0.050 on held-out; held-out calibration worse (ECE 0.019) |
| rebounds | who gets this rebound? | per-miss log loss | 1.7115 | works, far above the 2.39 floor; held-out worse by 0.11 |
| matchups | who guards whom, and what do they allow? | possession-points MSE | 1.3195 vs constant 1.3251 | assignment is stable (0.82); no defender signal beats the constant out of fold |
| gravity | how far do defenders stay from a player, beyond what location explains? | MSE | 29.91 (skill 0.434) | expectation holds on held-out; teammates of top-gravity players shoot slightly better (+0.004, training games) |
| ghosting | where would an average defender have stood? | error one second ahead | 3.417 ft | works, best on held-out; a deviation sign test agrees on 13 of 17 held-out games |
| embeddings | which players move and shoot alike? | nearest-neighbour position accuracy | none by design | passes its 0.60 gate; no held-out score exists |
| epv | how many points will this possession score, from here? | MSE | 1.2362 [1.1899, 1.2812] | works; 5 percent better than the constant out of fold, 7 percent on held-out |
| decisions | was this pass, drive or shot the best option? | Spearman vs on-court rating | -0.014 [-0.116, 0.100] | **failed**: no validated signal in either sample |
| lineups | which lineups and players are good? | season-ahead Spearman | 0.065 [-0.488, 0.482] | **inconclusive**: about 20 recurring lineups a season, every interval spans zero |
| winprob | who wins from this game state? | mean bucket Brier | 0.1486; log loss 0.4879 [0.4700, 0.5077] | works; held-out slightly worse and slightly over-confident |
| load | does a player slow down, and will they sit next game? | decline MSE; absence log loss | 0.5164; 0.2776 | absence works; decline barely beats the mean on held-out |

A value with no interval is a point estimate. Ghosting is an average defense, never the best one. Three more
lanes (actions, coverage, playcalling) wait for hand labels; their reports are fixture output, not results.

## The broadcast pipeline, stage by stage

This part describes the pipeline as tagged `v3-frozen`. Its court model, detector training set, team classifier
and stats layer are frozen. The history of every stage, with the dead ends, is in `DEVLOG.md`.

The stage numbers in `PIPELINE.md` and `PROJECT_SUMMARY.md` are prototype numbers. They were measured on the
clips the models were tuned on, and each one carries the tag `[prototype]` there. They are not tracking
accuracy. The scorecard at the top of this file is.

```
broadcast video
  -> frame gate               live play or dead ball
  -> detection and tracking   boxes and track fragments
  -> court model              pixels to court feet
  -> identity                 foot-point stability, fragment linking
  -> teams                    two jersey clusters per clip
  -> possessions              spans, offense and defense
  -> matchups                 defender and attacker pairs
  -> clock and play-by-play   real teams, points, how the possession ended
```

### Frame gate (`gate/`)

A per-frame classifier that separates live game footage from dead ball: replays, close-ups, ads, crowd shots and
timeouts. It is the entry filter of the pipeline. The old project threw live frames away at this step, so the
gate was rebuilt as a component with its own measurement.

- The gate that shipped in the old project was a floor-colour rule on HSV pixels, not a learned model. It is
  reimplemented in `gate/hsv_baseline.py` as the baseline to beat.
- `gate/zero_shot.py` is a CLIP zero-shot baseline. `gate/trained_head.py` is a light head on frozen CLIP
  embeddings, the one the pipeline uses.
- The threshold of the first gate was tuned on the prototype clips, and it drops wide frames on other broadcasts
  (`CLAIMS.md`, standing caveat 1). A second gate was refitted on human-verified harvest frames and is the one
  the scorecard uses to pick wide frames.

Train and evaluate it in this order. The seed is fixed at 42 (`config.py`).

```
extract_frames.py -> presort_frames.py -> you verify into truth/ -> train_gate.py -> evaluate_gate.py
```

```bash
python3 extract_frames.py                 # frames from the clips into data/visibility/_unsorted/
python3 presort_frames.py                 # CLIP guesses into data/visibility/predicted/, a labelling aid only
python3 train_gate.py                     # logistic head on CLIP embeddings; --head mlp, --backbone dinov2
python3 evaluate_gate.py                  # the test split: trained head, zero-shot baseline, HSV baseline
```

Labelling integrity, enforced in code (`gate/labels.py`):

- `truth/` is the only source of labels for training, validation and test. A frame lands there only because a
  person confirmed it.
- `predicted/` holds CLIP's guesses. Training and evaluation never read it. Grading a model on its own guesses
  would be circular.
- If `truth/` is empty, training and evaluation stop with an error.

`evaluate_gate.py` writes the test metrics to `reports/metrics.json` and `reports/metrics.txt` (both ignored by
git): accuracy, precision, recall, the confusion matrix and a threshold sweep. A false negative is a live frame
that was skipped. A false positive is a dead frame that went through. The threshold is tuned on the validation
split, never on the test split. Contact sheets of right and wrong frames go to `reports/viz/`.

The gates expose the same interface:

```python
gate.score(frame_bgr)                        # P(live), from zero to one
gate.is_court_visible(frame_bgr, threshold)  # True or False
```

### Detection and tracking (`detect/`)

A YOLO detector with BoT-SORT tracking, in one streaming pass over contiguous video. Only the player class is
tracked.

```
detect/tracker.py      PlayerTracker: streaming update, one list of tracks per frame
detect/detector.py     single-frame detector, used for evaluation
detect/camera_cut.py   resets ids when the broadcast cuts
detect/types.py        the Track record
run_tracking.py        a clip in, an overlay video, tracks and diagnostics out
label_boxes.py         draw truth boxes on live frames
evaluate_tracking.py   detection precision and recall against those boxes
```

`run_tracking.py` prints label-free health figures: players per frame, id churn, camera-cut resets and the
distribution of track lengths. Missed players, ghost boxes and identity switches against real tracking are in
the scorecard at the top of this file.

### Court model (`court/`)

A grid-keypoint detector with line snap and tracking across frames (`court/snap_track.py`). It solves the
mapping from pixels to court feet for each frame and follows the camera through pans. `COURT_TRACKER=0` runs the
older per-frame detector instead, for comparison. `COURT_GRID_WEIGHTS` selects the weights: the tagged model by
default, the `r23` weights for the build the scorecard reports.

The loop for improving the court model without hand labelling (the label factory):

```
harvest      python3 label_factory.py harvest <video urls>     auto-labels frames the model and the line evidence agree on
spot-check   python3 verify_projections.py <rank file> <review folder> <verdict file>     worst first; bad frames are left out
build        python3 label_factory.py build                    harvest goes to training only; validation stays frozen
train        train_pose_snapped_colab.ipynb on a GPU
benchmark    python3 benchmark_court_models.py                  adopt only if the error falls on the same validation frames
deploy       point COURT_GRID_WEIGHTS at the winning weights
```

`unseen_benchmark.py` and `render_overlay_video.py` measure the court model on other footage without labels and
render overlay clips.

### Identity (`detect/footpoint.py`, `detect/reid.py`)

After the court model, most jumps in a player's path come from track fragments (one player, many track ids
across occlusions and cuts) and from foot-point jitter. Three levers, all wired into `build_trajectories.py`:

```
botsort_reid.yaml      appearance re-identification inside BoT-SORT; off by default, TRACKER_REID=1 turns it on
detect/footpoint.py    foot-point stabiliser in pixel space, before the court mapping; --no-stab turns it off
detect/reid.py         offline fragment linker: appearance, motion in court feet, a margin for ambiguity; --no-reid
```

Every run writes `data/tracking/<clip>_identity.json`, the audit trail for every accepted merge and every
refusal. The linker is conservative by design, because a false merge poisons the stats of two players. Identity
is a track fragment, not a player (`CLAIMS.md`, standing caveat 3).

### Teams (`detect/teams.py`)

Unsupervised: two clusters on pooled jersey colour, one feature per track, fitted per clip. A track that sits
between the clusters gets no team. Cluster ids are team A and team B, not home and away. The linker refuses
merges across teams (`--no-teams` turns the classifier off). `evaluate_teams.py` exports crops, takes keypress
labels without showing predictions, and reports accuracy.

### Possessions (`segment_possessions.py`)

Reads the trajectories of a clip and cuts it into halfcourt possession spans: the attacked basket from where
the players stand, offense and defense from which team sits closer to that basket. Pure geometry on existing
outputs. A span is an approach plus a set, and the stats use only the core of the set, because the cuts are
fuzzy at each end. Known limits: no change of possession inside one halfcourt stay, and free-throw groups read
as sets. `evaluate_possessions.py` is the human evaluation, blind to predictions.

### Matchups (`matchup_metrics.py`)

The first defensive figures, computed only on span cores: a defender for each attacker by the Hungarian method
per frame, matchup distance, spacing conceded and closeout tendency. The matchup assignment has no validated
accuracy number (`CLAIMS.md` C1, open), so no player-level conclusion follows from it. Help defense and
off-ball attention need the ball and are not attempted from broadcast video.

### Clock, play-by-play and outcomes

`clock_reader.py` reads the game clock from the score graphic. `fetch_pbp.py` fetches play-by-play.
`align_outcomes.py` maps video spans to real possessions, with real teams, points and the event that ended the
possession. `tier2_join.py` and `tier2_credit.py` join outcomes to matchup records and report team-level credit
with intervals. What these may claim is in `CLAIMS.md`.

## Methods and literature

Each lane builds on published work. Its dossier under `docs/research/` names the papers it follows, puts our
number next to the published one with what differs and whether the two can be compared, and pre-registers
experiments whose adoption bar was fixed before they ran. Failed and negative results stay listed. The Methods
page of the site shows the same content in full; `export/research_methods.py` checks every quote against its
dossier and every number of ours against its report before the page is built. Every proposal, ranked by expected
gain over effort, is in [docs/research/BACKLOG.md](docs/research/BACKLOG.md). An experiment is first run on the
training games, then scored once on a test set fixed and hashed before any experiment ran
(`reports/research/testset.json`), and is adopted only if it clears its bar on both; the held-out games of the
frozen models are never scored again.

| Lane | Built on | Our number next to the published one | Experiments |
| --- | --- | --- | --- |
| [xfg](docs/research/xfg.md) | Chang et al. (shot quality from SportVU), Zadrozny and Elkan (isotonic calibration), Vovk et al. (Venn-Abers predictors) | Roughly comparable: out-of-fold log loss 0.6294 against 0.6522 for the GNN of Schmid et al. on the same season, with a different split and sample | `xfg-E1` cross Venn-Abers calibration: done, not adopted; it met its bar on the training games and not on the test games, which with ten games is inconclusive rather than worse ([write-up](reports/research/xfg-E1.md)); `xfg-E2` second defender: pre-registered, not run |
| [rebounds](docs/research/rebounds.md) | Maheswaran et al. (two rebounding papers on optical tracking), Guo et al. (temperature scaling), Cao et al. (listwise ranking), Chen and Guestrin (XGBoost) | Not directly comparable: the published accuracy is for which team gets the ball, ours for which player; no player-level number could be read | `rebounds-E1`, `rebounds-E2`: pre-registered, not run |
| [matchups](docs/research/matchups.md) | Franks et al. (defensive skill, and meta-analytics), Keshri et al. (defensive assignment) | Not directly comparable: in their split of the outcome, defense improved out-of-sample prediction; in ours it did not | `matchups-E1`, `matchups-E2`: pre-registered, not run |
| [gravity](docs/research/gravity.md) | Franks et al., Keshri et al., Barron et al. (gravity from defensive density) | No published comparable number | `gravity-E1`, `gravity-E2`: pre-registered, not run |
| [ghosting](docs/research/ghosting.md) | Le et al. (data-driven ghosting, and multi-agent imitation learning), Seidl et al. (synthesized NBA defenses), Felsen et al. (forecasting defenders) | Not directly comparable: the published errors average over the horizon, roll out every defender, or take the closest of many samples | `ghosting-E1`, `ghosting-E2`: pre-registered, not run |
| [embeddings](docs/research/embeddings.md) | Miller et al. (factorized shot intensities), Franks et al. (meta-analytics), Yamada and Fujii, Golovkine and Gunning (player typologies) | Not directly comparable: no paper reports a nearest-neighbour position accuracy or a held-out style score | `embeddings-E1`, `embeddings-E2`: pre-registered, not run |
| [epv](docs/research/epv.md) | Cervone et al. (POINTWISE, and the multiresolution model), Sicilia et al. (DeepHoops), Fernandez et al. (soccer possession value), Alcorn and Nguyen (baller2vec) | Not directly comparable: no paper we could check reports an expected-points error for basketball possessions | `epv-E1`, `epv-E2`: pre-registered, not run |
| [decisions](docs/research/decisions.md) | Cervone et al., Jutamulia and Hosoi, Decroos et al. (VAEP), Fernandez et al., Franks et al., Liu and Schulte, Sandholtz and Bornn, Fujii et al. | Not directly comparable: no basketball paper reports a validated link between a decision metric and an outcome | `decisions-E1` reliability before relevance: done, not adopted; the target is reliable over half a season, so the failed validation counts against the metric, and kind shares explain about a third of the player spread ([write-up](reports/research/decisions-E1.md)); the test games were not scored, since no player has two games on their single date; `decisions-E2`: pre-registered, not run |
| [lineups](docs/research/lineups.md) | Sill (regularized adjusted plus-minus), Deshpande and Jensen, Grassetti et al., Hvattum (a review of plus-minus) | Not directly comparable: published work scores game margins inside a season, ours season-ahead lineup ranks | `lineups-E1` would score game margins as Sill did; `lineups-E2`: both pre-registered, not run |
| [winprob](docs/research/winprob.md) | Stern (a Brownian motion model of the score), Deshpande and Jensen, Yeh et al. (evaluating live forecasts) | Roughly comparable: overall Brier 0.1645 out of fold against 0.1663 for the model of Maddox et al. without pregame strength | `winprob-E1`, `winprob-E2`: pre-registered, not run |
| [load](docs/research/load.md) | Lewis (game load and injury), Esteves et al. (back-to-backs), Scanlan et al. (activity across quarters), Stephen et al. (player fatigue) | Not directly comparable; like Stephen et al., we find that load explains very little | `load-E1`, `load-E2`: pre-registered, not run |
| [tracking](docs/research/tracking.md) | Ultralytics YOLO, BoT-SORT, ByteTrack, Nie et al. (court registration), Lu et al. (broadcast player tracking), Koshkina et al., SportsMOT, SoccerNet game state, DeepSportradar | Not directly comparable: no paper reports a number on our metric, data and split | `tracking-E1`, `tracking-E2`: pre-registered, not run |

## Documents

| File | What it is |
| --- | --- |
| [`PLAN.md`](PLAN.md) | The locked plan of this build: spec, checklist, milestones and every decision. |
| [`CLAIMS.md`](CLAIMS.md) | What the tracking pipeline will and will not claim, with the evidence for each claim. It predates the model lanes, whose claims are in `reports/results.md`. Its row A2 quotes prototype numbers. |
| [`reports/results.md`](reports/results.md) | The model results, with what failed. |
| [`STATUS.md`](STATUS.md) | The closing record of the frozen analytics lanes. |
| [`docs/SCRIPT.md`](docs/SCRIPT.md) | The script of the demo and of the slides. |
| [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | The tables the models read. The site data contract is `export/schemas.py`. |
| [`docs/STRATEGY.md`](docs/STRATEGY.md), [`docs/STATE_AND_EXTENSIONS.md`](docs/STATE_AND_EXTENSIONS.md), [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | The strategy, a dated audit of the repository, and the public data sources with their terms. Each states its own date. |
| [`ROADMAP.md`](ROADMAP.md), [`ROADMAP_LOG.md`](ROADMAP_LOG.md) | The tracking rebuild: targets, phases and the record of each pass. |
| [`PIPELINE.md`](PIPELINE.md), [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | The earlier broadcast pipeline, stage by stage. Their stage numbers are prototype numbers and are tagged. |
| [`DEVLOG.md`](DEVLOG.md) | The narrative history, with the dead ends. |
| [`CLAUDE.md`](CLAUDE.md) | The rules for Claude Code sessions that work in this repository. |
| `portfolio/` | Earlier single-page write-ups, kept for the record. |

## Licence and third-party content

The code, docs and labels in this repository are released under the MIT licence (`LICENSE`). Not covered by it:

- **Broadcast-derived stills** (NBA and broadcaster footage, mostly with analysis overlays), included at low
  resolution for research and commentary. Broadcast video is never committed.
- **Short broadcast clips** are shown only on a private, access-controlled copy of the site. The public site and
  this repository carry stills and derived data only (`CLAIMS.md`, standing caveat 4, amended 2026-11-14).
- **Roboflow-export frames** under `data/court_review/`: their dataset's own terms apply.
- **Tracking data.** This repository holds counts, metrics and reports derived from the SportVU logs. It holds
  no raw tracking log (`data/sportvu/` is ignored by git) and no generated site data (`web/data/` is ignored and
  rebuilt by the export).

The public site carries no SportVU frame data: no player positions, no playback and no clip. Those appear only
on the private copy (`PLAN.md` D60, `CLAIMS.md`). `python3 -m tools.check_no_frame_data` proves git
tracks none, and the `frames` step of `npm run check` proves the public build holds none.

The paper draft lives in a separate private repository. Rights holders can request removal by opening an issue.
