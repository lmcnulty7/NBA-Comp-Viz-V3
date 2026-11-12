# winprob improvement-search archive (training seasons only; held-out not scored)

Rules: docs/mac_runbook.md, "Improvement search in stage 5". Budget 5 variants; stop after 2 rejections in a row.
Metric: the selection metric of reports/winprob.md (unweighted mean of the five regulation bucket Brier scores),
out-of-fold under 5 grouped folds by game within 2020-21 to 2023-24 (seed 0). Acceptance: the paired bootstrap by game
(1000 resamples, seed 0) of selection(parent) minus selection(variant) has its 95 percent interval above 0. The
2024-25 season is set aside by `split_seasons` before anything is built and never read.

| id | parent | change | why (evidence) | selection | paired vs parent | decision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | none | incumbent: logistic on margin, seconds_remaining, possession_home, margin_x_sqrt_time | stage 5 selection | 0.1483 | n/a | incumbent |
| 1 | 0 | add margin_over_sqrt_time = margin / sqrt(seconds_remaining + 1) | incumbent is under-confident in the last 4 minutes (slope 1.69 [1.60, 1.78], ECE 0.050): its only interaction, margin x sqrt(time), vanishes as time runs out, so a late lead can never reach the certainty it carries; a lead scaled by the square root of the time left grows instead | 0.1468 | 0.00149 [0.00132, 0.00169] | **accepted** (branch exp/winprob-1, merged): regulation bucket slopes 0.97 to 1.07 (last 4 minutes 1.69 to 1.07, 48-36 1.21 to 1.01); OT Brier 0.1626 to 0.1538 |

Variant 1 was evaluated by a scratch script that refit only the logistic candidate on the same folds (the incumbent
reproduced the report's 0.14831 exactly). Merged, the feature is in features.FEATURES, so the gradient-boosting
candidate sees it too; the lane rerun reports every candidate under the unchanged selection rule.

Variants tried: 1 of 5.
