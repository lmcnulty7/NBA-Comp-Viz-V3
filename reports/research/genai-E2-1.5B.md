# genai-E2: a small open model, LoRA on the Mac, trained against the checker (1.5B)

Written 2026-11-18. Pre-registration: docs/research/genai.md, section "genai-E2", with its dated amendments, the 1.5B
amendment (PLAN.md D101) among them, all committed before the test score. Dev choices:
reports/research/genai-E2-1.5B-dev.json, committed at 1c326223 before any test item was read. Scoring code:
tools/genai_e2.py at 1c326223, the HEAD recorded in the final json beside the dev json's sha256 (a0ca9d84...). The
single test score is the entry "genai-E2-1.5B" of reports/research/genai-ledger.json; the resume entry was not used.
Every test number below is copied from reports/research/genai-E2-1.5B-final.json and every dev number from the dev
json; a number computed here says so. Intervals are 95 percent percentile intervals of a 2000-resample bootstrap of
players (scouting) or games (possessions), seed 0. The 0.5B result is reports/research/genai-E2.md; this report is
read on its own, and no hypothesis compares the two sizes.

**Verdict: H1 passes on both kinds; H2 hurts on scouting; H3 reads non-inferior; H4 is inconclusive.** LoRA
training lifts Qwen2.5-1.5B-Instruct from a few-shot pass rate of 0.012 to 0.914 on the 81 test players, and from
0.821 to 1.000 on the 480 test possessions, under the primary checker. Rejection sampling lowers the scouting pass
rate (0.642, mostly on reading grade). With one repair attempt the trained model passes on 0.914 of players against
0.802 for the Haiku texts, at no spend per text and 1.20 s per text single attempt on the Mac (the repair arm's time
is not recorded). Constrained decoding has nothing to fix. Read beside the pass rate: every possession text of the
trained model is identical to the template's, and on scouting it writes with far less variety than Haiku.

## What was tested

- Model: Qwen2.5-1.5B-Instruct (Apache 2.0, revision 989aa798), bfloat16 on the M5 GPU (MPS), under the heavy-job
  lock in stretches of at most 3 hours.
- Arms, checkers and items as for the 0.5B model (reports/research/genai-E2.md, "What was tested"): A0, A0-P3, A1
  (distillation), A2 (one round of rejection sampling), A3 (A1 with the number guard), A1 with one repair attempt;
  primary checker export/genai.py at db993008 (blob 41ba991b), freeze checker beside it; 81 test players and 480
  test possessions of 40 training games.
- Memory changes during the run, none changing a result (docs/research/genai.md, amendment o): the training loss
  keeps logits only for the answer tokens, and rejection sampling runs 2 items of 4 candidates at a time and saves
  in chunks. The equality of the answer-token loss with the model's own loss was checked on the 0.5B model in
  float32 (within 1e-6 on two train examples, commit 101f3a3a), not on the 1.5B model in bfloat16.

## Development (dev json)

| Stage | Scouting (40) | Possession (240) | Mean over kinds |
| --- | --- | --- | --- |
| A0 few-shot | 0.050 | 0.821 | 0.435 |
| A1 epoch 1 | 0.900 | 1.000 | 0.950 |
| A1 epoch 2 | 0.625 | 1.000 | 0.812 |
| A1 epoch 3 | 0.850 | 1.000 | 0.925 |
| A2 round 1 | 0.550 | 1.000 | 0.775 |

- The rule chose A1 epoch 1. RFT round 1 changed the mean by -0.175 (dev json: -0.17499999999999993), under the
  +0.03 the rule needs, so round 2 did not run. The dev-best arm of both kinds, and the arm A3 decodes, is A1
  epoch 1.
- Training times (dev json a1_train_s: 2,680, 3,144 and 398 s): only epoch 2's 3,144 s is a full epoch. Epoch 1's
  figure covers the examples after the resume from the checkpoint at example 400 (the first stretch stopped under
  the memory cap), and epoch 3's covers the last examples after a stretch deadline. They are not summed here.
- RFT round 1: 185 of 220 scouting items and 800 of 800 possession items had at least one passing candidate; 304
  and 801 texts kept; 4.8 s per pool item, measured at 2 items (8 sequences) per batch, so it is not comparable with
  the 0.5B model's 2.2 s, measured at 8 items per batch.
- Targets kept after the primary checker: 14 of 234 scouting dropped, 0 of 3,000 possession; 3 examples over 1,024
  tokens left out.

## Test (scored once)

### Pass rates

| Arm | Scouting, primary (freeze) | Possession, primary (freeze) |
| --- | --- | --- |
| A0 few-shot | 0.012 (0.062) | 0.821 (0.823) |
| A0-P3 | 0.000 (0.000) | not run |
| A1 epoch 1 | 0.914 (0.914) | 1.000 (1.000) |
| A1 epoch 1 with repair | 0.914 (0.914) | 1.000 (1.000) |
| A2 round 1 | 0.642 (0.654) | 1.000 (1.000) |
| A3 (A1 with the number guard) | 0.914 (0.914) | 1.000 (1.000) |
| Haiku, cache arm | 0.802 (0.963) | not run |
| Haiku, snapshot arm | 0.802 (0.840) | not run |

A pass rate of 1.000 over 480 possessions has the exact one-sided Clopper-Pearson 95 percent lower bound 0.994
(0.05^(1/480), computed here, as in genai-E1).

Failure rates of A1 on scouting, by rule (share of players): number_word 0.062, subject 0.049, grade 0.037, interval
0.012. A2 on scouting: grade 0.346, subject 0.025, number_value, length and style 0.012 each. A0 on scouting:
number_value 0.642, subject 0.556, grade 0.444, point_word 0.432. The repair attempt changed no outcome.

### Hypotheses (primary checker)

| Hypothesis | Difference | 95 percent interval | Reading |
| --- | --- | --- | --- |
| H1 scouting: A1 epoch 1 minus A0 | +0.901 | [0.840, 0.963] | passes |
| H1 possession: A1 epoch 1 minus A0 | +0.179 | [0.150, 0.208] | passes |
| H2 scouting: A2 round 1 minus A1 epoch 1 | -0.272 | [-0.370, -0.173] | hurts |
| H2 possession: A2 round 1 minus A1 epoch 1 | 0.000 | [0.000, 0.000] | inconclusive (both pass every item) |
| H3 scouting: A1 epoch 1 with repair minus Haiku cache arm (decides) | +0.111 | [0.012, 0.210] | non-inferior |
| H3 scouting: the same minus Haiku snapshot arm (beside it) | +0.111 | [0.012, 0.210] | non-inferior |
| H4 scouting: A3 minus A1 epoch 1, pass rate and number_value failure rate | 0.000 | [0.000, 0.000] | inconclusive |
| H4 possession: A3 minus A1 epoch 1, pass rate and number_value failure rate | 0.000 | [0.000, 0.000] | inconclusive |

H3 reads non-inferior when its lower bound is at or above -0.05. Here the whole interval is above 0, but the
pre-registration has no "superior" reading, so none is claimed. The two H3 rows are identical because the per-player
pass results of the two Haiku arms are identical (65 of 81 in both, as in the 0.5B report), not because of a copy
error. H4 has nothing to measure: neither A1 nor A3 has a number_value failure on the test items.

### Variety and template copying (no hypothesis reads these)

| Arm, kind | Frame variety | distinct-2 | Share of frames found in the template's | Possession texts identical to the template's |
| --- | --- | --- | --- | --- |
| A0, scouting | 0.337 [0.220, 0.271] | 0.077 [0.055, 0.068] | 0.000 | |
| A0, possession | 0.026 [0.017, 0.024] | 0.008 [0.005, 0.008] | 0.957 | 0.600 |
| A0-P3, scouting | 0.723 [0.433, 0.546] | 0.135 [0.096, 0.117] | 0.008 | |
| A1 epoch 1, scouting | 0.106 [0.071, 0.100] | 0.031 [0.023, 0.031] | 0.114 | |
| A1 epoch 1, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 1.000 | 1.000 |
| A1 epoch 1 with repair, scouting | 0.106 [0.071, 0.100] | 0.031 [0.023, 0.031] | 0.114 | |
| A1 epoch 1 with repair, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 1.000 | 1.000 |
| A3, scouting | 0.106 [0.071, 0.100] | 0.031 [0.023, 0.031] | 0.114 | |
| A3, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 1.000 | 1.000 |
| A2 round 1, scouting | 0.220 [0.133, 0.192] | 0.052 [0.037, 0.050] | 0.122 | |
| A2 round 1, possession | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] | 1.000 | 1.000 |
| Haiku cache arm, scouting | 0.612 [0.363, 0.461] | 0.119 [0.085, 0.103] | 0.027 | |
| Haiku snapshot arm, scouting | 0.596 [0.352, 0.460] | 0.120 [0.084, 0.106] | 0.023 | |

Distinct counts are biased downward under resampling (a resampled unit's sentences repeat), so these intervals sit
below their point estimates; read them as ranges of plausible values, as in genai-E1.

- Possession: the trained model's pass rate of 1.000 is a pure template copy: every one of its 480 possession texts
  is identical to the template's text of the same item. Its targets were template texts (no Haiku possession text
  exists on a training game), as D101 expected.
- Scouting: A1's pass rate is 0.914 against Haiku's 0.802 (H3 reads non-inferior; no superiority reading is
  pre-registered), with about a sixth of Haiku's frame variety (0.106 against 0.612, computed ratio 0.17). 89 percent
  of its frames are not template frames (computed: 1 minus 0.114). Rejection sampling doubled the scouting frame
  variety (0.220, computed ratio 2.1) and cost readability (grade failures 0.346).

### Cost and time

| Arm | API spend | Seconds per text on the M5 |
| --- | --- | --- |
| A1 epoch 1 | $0 | 1.20 (batches of 8, greedy) |
| A3 | $0 | 1.31 |
| A2 round 1 | $0 | 1.17 |
| A0 few-shot | $0 | 1.43 |
| A0-P3 | $0 | 7.78 (one at a time) |

For comparison, genai-E1 measured the Haiku texts at $2.44 per 1,000 players asked and estimated about 3.0 s per
item from record timestamps (reports/research/genai-E1.md).

### The two sizes side by side (descriptive)

This table puts numbers of the two reports next to each other. It is descriptive: no interval is computed on a
difference between sizes and no verdict is read from it.

| | 0.5B (genai-E2.md) | 1.5B (this report) |
| --- | --- | --- |
| A1 epoch 1, scouting pass rate | 0.852 | 0.914 |
| A1 epoch 1, possession pass rate | 1.000 | 1.000 |
| A0 few-shot, scouting pass rate | 0.111 | 0.012 |
| A1 epoch 1, scouting frame variety | 0.106 | 0.106 |
| H3 reading | inconclusive | non-inferior |
| A1 seconds per text | 0.76 | 1.20 |

## What this means

- A 1.5B open model, fine-tuned on this Mac, meets the site's faithfulness checker on 91 percent of unseen scouting
  players, against 1 percent before training, and on this test set its pass rate is not worse than the Haiku texts
  by more than 0.05 (H3), at no spend per text.
- It is not a richer writer: its scouting variety is the 0.5B model's (0.106) and about a sixth of Haiku's, and its
  possession texts are the template. On the question D101 asked of the larger model, variety at the same pass rate,
  the numbers side by side (descriptive, no test between sizes) show a higher pass rate and the same variety.
- Rejection sampling against the checker hurt on scouting. A likely reason, not measured: the kept candidates were
  more varied but harder to read, and the reading-grade rule then failed more often (0.346).
- Nothing reaches the site (D92). The possession texts of the site stay the Haiku texts; the scouting texts stay
  withheld (D93).

## What would change the verdict

More test players: at 81, H3's interval is 0.20 wide (computed: 0.210 minus 0.012, 0.198). A possession target that is
not a template (Haiku texts on training-game possessions, a spend that waits for a yes) would make the possession
result a test of writing, not of copying. Variety beyond the teacher's frames would need a target or a reward that
asks for it; the checker does not.
