# tests/fixtures/bref/pages: hand-built basketball-reference pages for tests/test_collect_bref.py

Each page is a few rows in the real markup of its page type (table ids, `data-stat` names, `csk` and
`data-append-csv` attributes, the tables basketball-reference wraps in HTML comments). None was fetched: they were
written by hand (2026-11-05), so a parser that passes here still gets a look at its first real unit. They sit in
this subdirectory because tests/test_bref_shots.py treats every `*.html` one level up as a full shot chart page.

- `schedule_2016_december.html`, `schedule_2016_february.html`, `schedule_2016_april.html`: month pages
  (`leagues/NBA_2016_games-<month>.html`) with the SportVU held-out game PHO at GSW 2015-12-16, the held-out arena
  (NYK at CLE), a training game (SAC at GSW, OT), GSW at OKC 2016-02-27, a Play-In note, the Playoffs divider row
  and an unplayed row without scores.
- `boxscore_201512160GSW.html`: `box-PHO-game-basic` and `box-GSW-game-basic` with five starters, the Reserves
  divider, Did Not Play / Did Not Dress / Inactive rows, team totals in `tfoot`, and a commented-out advanced table.
- `lineups_GSW_2016.html`: `lineups_5-man_` inside an HTML comment (three rows, a repeated header row, `diff_*`
  net columns, `mp` as `MMM:SS`) and a commented-out `lineups_4-man_` table the parser must ignore.
- `per_game_2016.html`: `per_game_stats` with a traded player's TOT row plus two team rows, a repeated header row
  and a two-position player.
- `pbp_201512160GSW.html`: the `pbp` table in the layout fetch_pbp.fetch_game reads (quarter header rows, six-cell
  event rows, `+N` point cells, colspan rows for jump ball and period ends).
