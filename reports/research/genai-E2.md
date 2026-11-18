# genai-E2: a small open model, LoRA on the Mac, trained against the checker (0.5B)

Written 2026-11-18. Pre-registration: docs/research/genai.md, section "genai-E2", with its dated amendments (the
checker, the compact input, the ml-review and pre-score amendments), all committed before the test score. Dev
choices: reports/research/genai-E2-0.5B-dev.json, committed at f3cbc2a7 before any test item was read. Scoring code:
tools/genai_e2.py at c76551f0, the HEAD recorded in the final json beside the dev json's sha256 (1816d0a6...). The
single test score is the entry "genai-E2-0.5B" of reports/research/genai-ledger.json; the resume entry was not used.
Every test number below is copied from reports/research/genai-E2-0.5B-final.json and every dev number from the dev
json; a number computed here says so. Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of
players (scouting) or games (possessions), seed 0. This is the 0.5B model only; the 1.5B model is a later decision.

**Verdict: H1 passes on both kinds; H2, H3 and H4 are inconclusive.** LoRA training lifts Qwen2.5-0.5B-Instruct
from a few-shot pass rate of 0.111 to 0.852 on the 81 test players, and from 0.665 to 1.000 on the 480 test
possessions, under the primary checker. Rejection sampling adds nothing measurable to distillation. H3 is
inconclusive: the trained model's pass rate (0.852) is above Haiku's (0.802), but the interval [-0.062, 0.160] does
not rule out a loss larger than 0.05 (no spend per text, 0.76 s per text on the Mac). Constrained decoding changes
no pass rate: it removed the number failures on scouting (0.025 to 0), while interval failures rose from 0.062 to
0.074. Read beside the pass rate: on possessions the trained model copies the template (99.2
percent of its texts are identical to the template's), and on scouting it writes with far less variety than Haiku.

## What was tested

- Model: Qwen2.5-0.5B-Instruct (Apache 2.0, revision 7ae55760), float32 on the M5 GPU (MPS), under the heavy-job
  lock in stretches of at most 3 hours.
- Arms: A0, the base model with the rules and 2 worked examples in the compact input; A0-P3, the base model on the
  full P3 prompt (scouting only, read by no hypothesis); A1, LoRA distillation on train targets (Haiku texts of train
  players, template texts of train possessions, each kept only if it passes the primary checker); A2, one round of
  rejection sampling from A1; A3, A1 with the number guard; A1 with one repair attempt (the H3 arm).
- Checkers: primary, export/genai.py of v4-research at db993008 (blob 41ba991b); freeze, the checker pinned at the
  split freeze (blob c10b8a4b), reported beside it and deciding nothing.
- Items: the 81 test players and 480 test possessions of 40 training games (reports/research/genai-splits.json).

## Development (dev json)

| Stage | Scouting (40) | Possession (240) | Mean over kinds |
| --- | --- | --- | --- |
| A0 few-shot | 0.125 | 0.629 | 0.377 |
| A1 epoch 1 | 0.800 | 1.000 | 0.900 |
| A1 epoch 2 | 0.500 | 1.000 | 0.750 |
| A1 epoch 3 | 0.700 | 1.000 | 0.850 |
| A2 round 1 | 0.675 | 1.000 | 0.838 |

- The rule chose A1 epoch 1 (highest mean over kinds). RFT round 1 changed the mean by -0.0625, under the +0.03 the
  rule needs, so round 2 did not run. The dev-best arm of both kinds, and the arm A3 decodes, is A1 epoch 1.
- Training: 3 epochs of about 2,411 to 2,445 s each; 3 examples over 1,024 tokens left out. Targets kept after the
  primary checker: 14 of 234 scouting targets dropped, 0 of 3,000 possession targets (dev json, targets).
- RFT round 1: 4 candidates per item, 2.2 s per pool item; 193 of 220 scouting items and 800 of 800 possession
  items had at least one passing candidate; 311 and 814 texts kept.

## Test (scored once)

### Pass rates

| Arm | Scouting, primary (freeze) | Possession, primary (freeze) |
| --- | --- | --- |
| A0 few-shot | 0.111 (0.333) | 0.665 (0.869) |
| A0-P3 | 0.037 (0.037) | not run |
| A1 epoch 1 | 0.852 (0.877) | 1.000 (1.000) |
| A1 epoch 1 with repair | 0.852 (0.877) | 1.000 (1.000) |
| A2 round 1 | 0.852 (0.852) | 1.000 (1.000) |
| A3 (A1 with the number guard) | 0.852 (0.877) | 1.000 (1.000) |
| Haiku, cache arm | 0.802 (0.963) | not run |
| Haiku, snapshot arm | 0.802 (0.840) | not run |

A pass rate of 1.000 over 480 possessions has the exact one-sided Clopper-Pearson 95 percent lower bound 0.994
(0.05^(1/480), computed here, as in genai-E1).

Failure rates of A1 on scouting, by rule (share of players): interval 0.062, grade 0.037, subject 0.025,
point_estimate 0.025, number_value 0.025. A0 on scouting: number_value 0.654, subject 0.580, point_word 0.123. The
Haiku cache arm fails the primary checker on subject 0.123, attribution 0.037, and has no text for 0.037.

### Hypotheses (primary checker)

| Hypothesis | Difference | 95 percent interval | Reading |
| --- | --- | --- | --- |
| H1 scouting: A1 epoch 1 minus A0 | +0.741 | [0.642, 0.840] | passes |
| H1 possession: A1 epoch 1 minus A0 | +0.335 | [0.296, 0.375] | passes |
| H2 scouting: A2 round 1 minus A1 epoch 1 | 0.000 | [-0.086, 0.099] | inconclusive |
| H2 possession: A2 round 1 minus A1 epoch 1 | 0.000 | [0.000, 0.000] | inconclusive (both pass every item) |
| H3 scouting: A1 epoch 1 with repair minus Haiku cache arm (decides) | +0.049 | [-0.062, 0.160] | inconclusive |
| H3 scouting: the same minus Haiku snapshot arm (beside it) | +0.049 | [-0.062, 0.160] | inconclusive |
| H4 scouting: A3 minus A1 epoch 1, pass rate | 0.000 | [0.000, 0.000] | inconclusive |
| H4 scouting: A3 minus A1 epoch 1, number_value failure rate | -0.025 | [-0.062, 0.000] | inconclusive (the upper end is 0, not below it) |
| H4 possession: A3 minus A1 epoch 1, pass rate and number_value failure rate | 0.000 | [0.000, 0.000] | inconclusive |

The two H3 rows are identical because the per-player pass results of the two Haiku arms are identical (65 of 81
players pass in both; the snapshot arm's missing texts belong to players whose cache text fails the primary
checker), not because of a copy error.

H1 needs a difference of at least +0.10 with its lower bound above 0: both kinds pass by a wide margin. H3 reads
non-inferior only with a lower bound at or above -0.05; at -0.062 it is inconclusive, as the pre-registration said
was likely at 81 players.

### Variety and template copying (no hypothesis reads these)

| Arm, kind | Frame variety | distinct-2 | Share of frames found in the template's | Possession texts identical to the template's |
| --- | --- | --- | --- | --- |
| A0, scouting | 0.260 [0.172, 0.220] | 0.060 [0.044, 0.054] | 0.000 | |
| A0, possession | 0.015 [0.010, 0.014] | 0.004 [0.003, 0.004] | 0.926 | 0.065 |
| A1 epoch 1, scouting | 0.106 [0.075, 0.100] | 0.032 [0.026, 0.031] | 0.119 | |
| A1 epoch 1, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 0.999 | 0.992 |
| A2 round 1, scouting | 0.114 [0.072, 0.107] | 0.033 [0.023, 0.033] | 0.115 | |
| A2 round 1, possession | 0.009 [0.007, 0.009] | 0.003 [0.003, 0.003] | 0.997 | 0.981 |
| Haiku cache arm, scouting | 0.612 [0.363, 0.461] | 0.119 [0.085, 0.103] | 0.027 | |
| Haiku snapshot arm, scouting | 0.596 [0.352, 0.460] | 0.120 [0.084, 0.106] | 0.023 | |
| A0-P3, scouting | 0.735 [0.415, 0.553] | 0.198 [0.128, 0.169] | 0.000 | |
| A3, scouting | 0.106 [0.075, 0.100] | 0.032 [0.026, 0.031] | 0.119 | |
| A3, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 0.999 | 0.992 |
| A1 epoch 1 with repair, scouting | 0.109 [0.076, 0.101] | 0.033 [0.026, 0.032] | 0.117 | |
| A1 epoch 1 with repair, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 0.999 | 0.992 |

Distinct counts are biased downward under resampling (a resampled unit's sentences repeat), so these intervals sit
below their point estimates; read them as ranges of plausible values, as in genai-E1.

- Possession: the trained model's pass rate of 1.000 is a template copy. 99.2 percent of its possession texts are
  identical to the template's text of the same item, and 99.9 percent of its sentence frames are template frames.
  It learned the possession targets it was given, which were template texts (no Haiku possession text exists on a
  training game). This pass rate says the model can reproduce a deterministic writer, not that it writes well.
- Scouting: A1 passes at 0.852 against Haiku's 0.802 (H3 inconclusive), with about a sixth of Haiku's frame variety
  (0.106 against 0.612, computed ratio 0.17). 88 percent of its frames are not template frames (computed: 1 minus
  0.119), so it does not copy the template there; it settles on fewer sentence shapes than its teacher.

### Cost and time

| Arm | API spend | Seconds per text on the M5 |
| --- | --- | --- |
| A1 epoch 1 | $0 | 0.76 (batches of 8, greedy) |
| A3 | $0 | 0.80 |
| A0 few-shot | $0 | 1.27 |
| A0-P3 | $0 | 5.21 (one at a time) |

For comparison, genai-E1 measured the Haiku texts at $2.44 per 1,000 players asked and estimated about 3.0 s per
item from record timestamps (reports/research/genai-E1.md). Training cost lock time, not money: about 2.0 hours for
the 3 A1 epochs (computed from the dev json: 2,445 + 2,423 + 2,411 s), and RFT round 1 sampled at 2.2 s per pool
item.

## What this means

- A 0.5B open model, trained for about 41 minutes on this Mac (epoch 1; about 2 hours for the 3 epochs the choice
  was made from), meets the site's faithfulness checker on 85 percent of
  unseen scouting players, against 11 percent before training, at no spend per text. That is the measured answer to
  the lane's question for the 0.5B size.
- It is not a better writer than the teacher or the template: on possessions it is the template, and on scouting it
  has far less variety than Haiku. Whether that matters to a reader is the human protocol's question (P7); the
  hand-label protocol (docs/research/genai.md) is written and not run.
- Rejection sampling against the checker did not help here. A likely reason, not measured: the distillation targets
  had already been filtered by the same checker, so the candidates it kept were close to the targets. Constrained
  decoding had almost nothing to fix after training (number_value failures 0.025 on scouting).
- Nothing reaches the site (D92). The possession texts of the site stay the Haiku texts; the scouting texts stay
  withheld (D93).

## What would change the verdict

More test players: at 81, H3's interval is 0.22 wide (computed: 0.160 minus -0.062). A possession target that is not
a template (Haiku texts on
training-game possessions, a spend that waits for a yes) would make the possession result a test of writing, not of
copying. The 1.5B model is a separate decision, to be made from these dev numbers.
