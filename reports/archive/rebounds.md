# rebounds improvement-search archive (training folds only; held-out not scored)

Rules: docs/mac_runbook.md, "Improvement search in stage 5". Budget 5 variants; stop after 2 rejections in a row.
Metric: the selection metric of reports/rebounds.md (out-of-fold per-miss multinomial log loss), 5 grouped folds by
sportvu_game (seed 0) over the 604 training games. Acceptance: the paired bootstrap by game (1000 resamples, seed 0) of
mlogloss(variant) minus mlogloss(parent) has its 95 percent interval below 0. The 17 held-out games are set aside by
`split_train_heldout` before any fit and never read.

| id | parent | change | why (evidence) | per-miss log loss | paired vs parent | decision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | none | incumbent: hgb on the candidate rows, probabilities normalised within the miss | stage 5 selection | 1.6234 | n/a | incumbent |
| 1 | 0 | temperature on the within-miss normalisation of hgb: p proportional to q^tau, tau fitted (bounded 0.25 to 8) by minimising per-miss log loss on hgb's inner out-of-fold predictions over 5 inner grouped folds of each outer fold's training games | the top candidate is under-confident at every level (calibration slope 1.69 [1.66, 1.74], ECE 0.109; misses at mean p 0.547 are secured 0.786 of the time): a binary model normalised within the miss flattens the distribution, and one exponent fitted on inner folds can sharpen it without touching the ranking (top-1 unchanged) | pending | pending | pending |
