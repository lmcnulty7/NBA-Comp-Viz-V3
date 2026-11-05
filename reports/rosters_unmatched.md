# Rosters: name matching per game

Written by `python -m analytics.rosters` (docs/DATA_CONTRACT.md section 16). Counts are distinct names
of each source; a source that was not given for a game shows 0 / 0. The method columns count roster
players by the best match_method over both sources.

| game | roster | bref matched / total | pbp matched / total | exact | lastname | fuzzy | none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11.14.2015.BKN.at.GSW | 19 | 0 / 0 | 0 / 0 | 0 | 0 | 0 | 19 |
| 12.16.2015.PHX.at.GSW | 25 | 22 / 22 | 25 / 25 | 25 | 0 | 0 | 0 |

## 11.14.2015.BKN.at.GSW

- roster size: 19
- basketball-reference shot chart names matched: 0 / 0
- play-by-play names matched: 0 / 0
- unmatched names: (none)
- roster players no source named: Shaun Livingston, Andre Iguodala, Andrew Bogut, Brandon Rush, Stephen Curry, Harrison Barnes, Festus Ezeli, Draymond Green, Ian Clark, Joe Johnson, Jarrett Jack, Andrea Bargnani, Thaddeus Young, Brook Lopez, Wayne Ellington, Bojan Bogdanovic, Thomas Robinson, Shane Larkin, Rondae Hollis-Jefferson

## 12.16.2015.PHX.at.GSW

- roster size: 25
- basketball-reference shot chart names matched: 22 / 22
- play-by-play names matched: 25 / 25
- unmatched names: (none)
- roster players no source named: (none)
