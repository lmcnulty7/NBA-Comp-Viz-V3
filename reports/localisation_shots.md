# Shot localisation (analytics/shot_frames.py)

One line per game: the validate() numbers of analytics/shot_frames/<season>/<game>.parquet. Rules and
thresholds are the fixed values written in analytics/shot_frames.py; nothing is tuned on any game. A line
marked held-out comes from a held-out game of sportvu/splits.json, used as a smoke test only, with no
parameter tuned on it.

| game | split | n_shots | n_release_found | share_release_found | n_identity_known | identity_agreement | share_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12.16.2015.PHX.at.GSW | held-out (smoke test only, no parameter tuned on it) | 179 | 125 | 0.6983 | 112 | 0.9732 | 0.3966 |
