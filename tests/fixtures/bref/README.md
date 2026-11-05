# tests/fixtures/bref: saved basketball-reference shot chart pages for collect/bref_shots.py

`tests/test_bref_shots.py` parses every `*.html` here.

- `<code>.synthetic.html` (two pages, written by `make_synthetic.py`): the real field-goal attempts of
  `data/pbp/<code>.json` (player, period, clock, make or miss, value, distance) placed at generated positions on the
  500 x 472 px chart, in the marker layout the collector parses. They were written because
  www.basketball-reference.com was not reachable from the cloud session (network policy), so they test the parser,
  the court mapping and the run loop, not the live site's pixel frame. The recovered set must equal the play-by-play
  exactly (179 attempts for 201512160GSW, 191 for 201602270OKC).
- `<code>.html` (none yet): real pages. Save them on a machine that reaches basketball-reference:

      python -m collect.bref_shots --games 12.16.2015.PHX.at.GSW 02.27.2016.GSW.at.OKC --save-pages tests/fixtures/bref

  The test then checks each real page for a full game of placed, timed markers, the hoop within 10 px of
  `HOOP_PX` by the distances in the tips (residual under a foot), and the count against `data/pbp/<code>.json` within
  5% when that file exists. Pages are 200 to 400 KB each; keep to a few.
