# basketball-reference shot charts vs play-by-play: shots recovered per game

`python -m collect.bref_shots --fetch-pbp --games ...` on 2026-11-05 (cloud session, 17 requests a minute), ten games:
eight of the SportVU 2015-16 games (the four train games, the held-out game, the two CLE arena games, two more) and
the 2016-02-27 overtime game used as a test fixture. `bref` is the markers on the shot chart page, `pbp` the
field-goal attempts (`shot_made` + `shot_missed`) in the play-by-play page of the same site, `resid` the mean
absolute difference between the pixel distance from the basket and the distance the tip states (tips are whole feet).

| game | code | bref | pbp | diff | resid | split |
| --- | --- | --- | --- | --- | --- | --- |
| 10.27.2015.NOP.at.GSW | 201510270GSW | 177 | 179 | -2 | 0.27 ft | |
| 11.12.2015.GSW.at.MIN | 201511120MIN | 175 | 176 | -1 | 0.29 ft | |
| 11.14.2015.BKN.at.GSW | 201511140GSW | 194 | 195 | -1 | 0.29 ft | train |
| 12.16.2015.PHX.at.GSW | 201512160GSW | 179 | 179 | +0 | 0.27 ft | heldout_game |
| 12.23.2015.NYK.at.CLE | 201512230CLE | 164 | 165 | -1 | 0.34 ft | heldout_arena |
| 12.28.2015.SAC.at.GSW | 201512280GSW | 183 | 184 | -1 | 0.29 ft | train |
| 01.04.2016.CHA.at.GSW | 201601040GSW | 173 | 173 | +0 | 0.29 ft | train |
| 01.18.2016.GSW.at.CLE | 201601180CLE | 161 | 162 | -1 | 0.32 ft | heldout_arena |
| 01.22.2016.IND.at.GSW | 201601220GSW | 184 | 185 | -1 | 0.26 ft | train |
| 02.27.2016.GSW.at.OKC | 201602270OKC | 191 | 191 | +0 | 0.30 ft | (not SportVU) |
| total | | 1781 | 1789 | -8 | | |

## The eight missing attempts

Matching markers to play-by-play attempts by period, clock (within 1 s) and make or miss leaves no marker without
an attempt, and exactly these attempts without a marker:

| game | period | clock | attempt |
| --- | --- | --- | --- |
| 10.27.2015.NOP.at.GSW | 2 | 0:00 | D. Cunningham misses 3-pt jump shot from 56 ft |
| 10.27.2015.NOP.at.GSW | 3 | 0:00 | S. Livingston misses 3-pt jump shot from 51 ft |
| 11.12.2015.GSW.at.MIN | 3 | 0:01 | A. Iguodala misses 3-pt jump shot from 57 ft |
| 11.14.2015.BKN.at.GSW | 3 | 3:48 | T. Young misses 3-pt jump shot from 49 ft |
| 12.23.2015.NYK.at.CLE | 4 | 0:00 | K. Porzingis misses 3-pt jump shot from 57 ft |
| 12.28.2015.SAC.at.GSW | 3 | 0:00 | S. Curry misses 3-pt jump shot from 46 ft |
| 01.18.2016.GSW.at.CLE | 2 | 0:00 | K. Thompson misses 3-pt jump shot from 63 ft |
| 01.22.2016.IND.at.GSW | 2 | 0:01 | S. Curry makes 3-pt jump shot from 48 ft |

Every one is a heave from beyond the half court the chart draws (47 ft from the baseline). The chart recovers every
attempt inside the half court; the rows of such heaves have to come from the play-by-play (no coordinates).

## The pixel frame

Markers are positioned by their top-left corner. Fitting the basket from the stated distances on every page puts it
at (239.5, 35.5) in marker coordinates, 10 px per foot, which is the basket centre on the court image (250, 52.5)
less a glyph offset of (10.5, 17) px; `collect.bref_shots.HOOP_PX` encodes that and `calibrate` fails a unit whose
fit drifts more than 10 px from it. The 0.26 to 0.34 ft residuals are the whole-foot rounding of the tips.
