# portfolio/ — the Court Ledger page, regenerable from this directory

Recruiter-facing walkthrough of the pipeline, published as a private artifact
(2026-09-12). Everything needed to rebuild it lives here so the page is an
artifact of the repo, not of one session.

## Rebuild
```
cd portfolio
/opt/anaconda3/bin/python build_numbers.py   # reads the JSON artifacts → numbers.json
/opt/anaconda3/bin/python assemble.py        # template + clips + stills → build/court-ledger.html (+ -stills.html)
/opt/anaconda3/bin/python gate.py            # grep gate: must print 0 hits
```
`build/` is gitignored — the two HTML outputs (~5 MB / ~2 MB, all media as
data: URIs) are regenerable in under a minute. Needs ffmpeg/ffprobe at
/opt/homebrew/bin, Pillow, pypdf.

## What is here
- `template.html` — the page with `{{VIDEO:…}}`, `{{IMG:…}}`, `{{POSTER:…}}`,
  `{{N:key}}` and `{{CHART:NAME}}` placeholders. No number is typed into the template.
  Two layers: every figure shows one line + one number, with a collapsed `receipt`
  (n, CI, caveats, artifact chips) beneath; the nav's **receipts** button opens all.
- `style.css` — the visual system (tokens for light/dark/toggle, the court-motif
  watermark, the two-layer exhibit anatomy, chart styles). Injected at `{{STYLE}}`.
- `build_numbers.py` — pulls every headline figure from `reports/*.json`,
  `data/pbp/*.json`, `data/tracking/*.json` into `numbers.json`; the few
  paper/DEVLOG-sourced figures are listed under `hand_typed_sources`.
- `assemble.py` — fills placeholders, encodes media as base64, extracts
  poster frames with ffmpeg, renders 13 charts from numbers.json through the
  `CHARTS` registry (hero tile minis, the stage rail, gate baselines, detection
  dumbbell, matchup coverage, the interval, per-game cross-validation, the real
  funnel + anchor-failure breakdown, points-mix, the audit Wilson chart, the
  number line), writes both outputs and a size report. Chart text lives in HTML;
  SVGs carry marks only, so labels never scale with a viewBox.
- `gate.py` — the honesty grep gate (banned phrasings, 91.1 only struck
  through, no player name within 80 chars of a credit figure, …).
- `clips/` — the seven final H.264 overlay renders (≤ 6 s, ≤ 480p, no audio;
  2.7 MB total). Raw renders were not kept. Renderers: `run_tracking.py`,
  `build_trajectories.py`, `visualize_masking.py` (repo root) and the two
  JSON-only wrappers here, `render_matchup.py` (c5) and
  `render_matchup_c6.py` + `scan_c6.py` (c6, banner relocated and
  pixel-verified against the JSON's 36 excluded frames).
- `img/` — stills as WebP q80 + `stills_manifest.json`.

## Provenance
Every moving image is a pipeline-derived overlay render of 6 s or less; the
matchup clip (c5) contains no broadcast pixels; no raw broadcast frame appears
without an overlay. See CLAIMS.md standing caveat 4.
