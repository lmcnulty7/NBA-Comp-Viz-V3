# Research refresh: how new papers enter the backlog

Written 2026-11-16 (PLAN.md D70, D73). The refresh keeps the dossiers current. It reads, it does not run
experiments: a paper reaches the site only as a row of a dossier and, if it attacks a measured weakness, as a
pre-registered proposal in BACKLOG.md.

## Sources (each link checked on 2026-11-16)

| Source | Link | What to read |
| --- | --- | --- |
| arXiv, computer vision | https://arxiv.org/list/cs.CV/recent | broadcast tracking, detection, re-identification, sports video |
| arXiv, machine learning (statistics) | https://arxiv.org/list/stat.ML/recent | calibration, uncertainty, sequence and multi-agent models |
| arXiv query interface (the queries below) | https://export.arxiv.org/api/query | sorted by submission date, newest first |
| MIT Sloan Sports Analytics Conference, research papers | https://sloansportsconference.com/research-paper-competition and the per-year pages (for example https://www.sloansportsconference.com/conference/2026-conference) | basketball track papers |
| CVsports, the computer vision in sports workshop at CVPR (12th edition at CVPR 2026) | https://vap.aau.dk/cvsports/ and the CVF open access proceedings https://openaccess.thecvf.com/menu | tracking, ball, identity, game state |
| MLSA, machine learning and data mining for sports analytics, at ECML PKDD (12th edition 2025, 13th 2026) | https://dtai.cs.kuleuven.be/events/MLSA25/ and the proceedings https://link.springer.com/book/10.1007/978-3-032-15165-0 | valuation, tactics, player models |

No basketball workshop at NeurIPS, ICML, KDD or AAAI was found for 2025 or 2026 in the 2026-11-16 search; the refresh
searches for one each time and adds it here when it exists.

## Queries (arXiv API, last 45 days, newest first)

Each query runs as `search_query=<q>&sortBy=submittedDate&sortOrder=descending&max_results=50`.

| Lane | Query |
| --- | --- |
| xfg | `all:basketball AND (all:"shot quality" OR all:"expected field goal" OR all:"shot probability")` |
| xfg, all lanes | `all:sports AND (all:calibration OR all:conformal OR all:"Venn-Abers")` |
| epv, decisions | `all:basketball AND (all:"possession value" OR all:EPV OR all:"decision making")` |
| epv, decisions | `(all:"action valuation" OR all:"off-policy evaluation") AND all:sports` |
| ghosting, matchups, gravity | `all:basketball AND (all:defense OR all:ghosting OR all:"trajectory prediction" OR all:imputation)` |
| embeddings | `all:basketball AND (all:embedding OR all:"player representation" OR all:"playing style")` |
| lineups | `all:basketball AND (all:"plus-minus" OR all:lineup OR all:"player rating")` |
| winprob | `all:basketball AND all:"win probability"` |
| load | `all:basketball AND (all:fatigue OR all:workload OR all:injury)` |
| tracking | `all:basketball AND (all:tracking OR all:"ball detection" OR all:"jersey number" OR all:"re-identification")` |
| tracking | `(all:"game state reconstruction" OR all:SportsMOT) AND all:tracking` |

## Triage, in order

1. Is it a real paper with a stable link (arXiv id, DOI or proceedings page)? Fetch it. If not, drop it.
2. Does it target a lane we have? If not, drop it.
3. Does it report a number we can set next to ours? Add a benchmark row to the lane's dossier with what differs
   (data, season, league, metric, split) and the comparability verdict.
4. Does it attack a weakness our scorecard measured (docs/research/<lane>.md, section "Measured weaknesses")? Only
   then write a pre-registered proposal (the template of every dossier: hypothesis, frozen baseline, metric,
   threshold, paired bootstrap by game, ablation, cost, risks, coach gain) and rank it in BACKLOG.md.
5. Nothing is proposed only because it is new. A proposal never changes a test set or a threshold after it is written.

## Schedule (proposed, not created)

Monthly, on the first Monday, as a scheduled job that runs the queries, triages, edits the dossiers and BACKLOG.md
on a branch, and leaves a summary for Lucien. Creating the job needs Lucien's yes (PLAN.md D73).

## Log

| Date | What ran | New rows | New proposals |
| --- | --- | --- | --- |
| 2026-11-16 | First survey: the dossiers of every lane and docs/research/DATA.md, written by the research team (PLAN.md D74) | see each dossier | see BACKLOG.md |
