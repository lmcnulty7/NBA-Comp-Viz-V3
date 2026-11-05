# tests/fixtures/bref: saved basketball-reference shot chart pages for collect/bref_shots.py

`tests/test_bref_shots.py` parses every `*.html` here.

- `<code>.html`: real pages saved with `--save-pages` (2026-11-05): `201512160GSW.html` (PHX at GSW, the SportVU
  held-out game, 179 markers) and `201602270OKC.html` (GSW at OKC, the overtime game, 191 markers). The test checks
  each for a full game of placed, timed markers, the basket within 10 px of `HOOP_PX` by the distances in the tips
  (mean residual under a foot), and the count against `data/pbp/<code>.json` within 5%. These two pages fixed the
  pixel frame: markers are anchored by their top-left corner, so the basket sits at (239.5, 35.5) in marker
  coordinates, 10 px per foot. Pages are about 190 KB each; keep to a few. Save more with

      python -m collect.bref_shots --games <SportVU ids> --save-pages tests/fixtures/bref

- `<code>.synthetic.html` (written by `make_synthetic.py`): the real field-goal attempts of `data/pbp/<code>.json`
  (player, period, clock, make or miss, value, distance) placed at generated positions in the marker layout, so the
  recovered set must equal the play-by-play exactly (179 and 191 attempts). They were written before the real pages
  could be fetched and stay as the exact-recovery and run-loop fixtures. Regenerate after any change to `HOOP_PX`.
