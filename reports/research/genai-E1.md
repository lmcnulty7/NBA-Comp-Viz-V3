# genai-E1: a template writer against the Haiku texts

Written 2026-11-17. Pre-registration: docs/research/genai.md, section "genai-E1", committed before any run (0ce3b76a);
splits frozen and hashed before any text was scored (reports/research/genai-splits.json, 4061d2ed); template settled
on train items only (a5eefb1d); amendments after the dev stage and before the test score in the same section
(8b00705f). Scoring code: tools/genai_eval.py and tools/genai_templates.py at 8b00705f. Every number below is copied
from reports/research/genai-E1-final.json (test, scored once: the one entry of reports/research/genai-ledger.json) or
reports/research/genai-E1-dev.json; a number computed here from those files says so. Intervals are 95 percent
percentile intervals of a 2000-resample bootstrap of players (scouting) or games (possessions), seed 0.

**Verdict: H1 holds and H2 holds, on both Haiku arms.** On the pinned checker a template with no model passes every
test text (81 of 81 players, 480 of 480 possessions), at least as often as claude-haiku-5-5, and the Haiku texts have
far more variety. Read as pre-committed: the pinned checker alone cannot show what a model adds. A stricter checker
from the P3 branch, applied as a sensitivity check that no verdict reads, turns the order round on scouting: it
passes 36 of 81 template texts and 65 of 81 Haiku texts. So genai-E2 must report variety beside pass rate, and the
checker version is part of every result.

## What was tested

- Template (tools/genai_templates.py): one fixed sentence per slot, filled from the item's facts; no model, no random
  choice. Written and checked on 284 train players and 3,000 train possessions only.
- Haiku, as pre-registered: the text the pinned snapshot of the v4 site data root holds for the player, or none.
- Haiku, cache arm (amendment 1): the kept text of the latest cache record whose input is the player's input, or
  none. The snapshot is a moment in P3's run that later checks had already pruned; the cache records are what the
  calls produced.
- Checker: export/genai.py check_text at git blob c10b8a4b (the pin in genai-splits.json). Sensitivity checker
  (amendment 2): export/genai.py of the p3-v3 branch at commit 7f2c08fc.
- Items: 81 test players; 480 test possessions of 40 training games (template only: the Haiku possession texts are
  about held-out games and were not read, and a Haiku arm on training games needs a yes to spend).

## Scouting, 81 test players

| Arm | Pass rate | Texts | Reading grade, mean (largest) | Words per text, mean (range) | Frames / sentences | Frame variety | distinct-2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | 1.000 [1.000, 1.000] | 81 | 3.70 [3.64, 3.76] (4.3) | 117.3 [114.7, 119.9] (82 to 126) | 16 / 752 | 0.021 [0.020, 0.022] | 0.014 [0.013, 0.014] |
| Haiku, snapshot (pre-registered) | 0.840 [0.753, 0.914] | 68 | 4.66 [4.51, 4.81] (6.9) | 107.3 [105.3, 109.4] (88 to 134) | 339 / 569 | 0.596 [0.352, 0.460] | 0.120 [0.084, 0.106] |
| Haiku, cache arm | 0.963 [0.914, 1.000] | 78 | 4.65 [4.53, 4.79] (6.9) | 107.5 [105.4, 109.7] (87 to 144) | 407 / 665 | 0.612 [0.363, 0.461] | 0.119 [0.085, 0.103] |

- Every failure of a Haiku arm is an item with no text (rule no_text): 0.160 of players for the snapshot arm (13 of
  81) and 0.037 for the cache arm (3 of 81). Every text either arm holds passes the pinned checker. The template has
  no failure under any rule.
- H1 (template minus Haiku pass rate, lower bound at or above -0.05). Haiku snapshot minus template: [-0.247, -0.086],
  so template minus Haiku is [+0.086, +0.247]: holds. Cache arm minus template: [-0.086, 0.000]: holds.
- H2 (Haiku minus template frame variety, lower bound above 0). Snapshot: [0.331, 0.439]; cache arm: [0.343, 0.439]:
  holds on both. distinct-2: [0.071, 0.093] and [0.072, 0.089].
- Frame variety and distinct-2 are biased downward under resampling (amendment 3): their intervals sit below the
  point estimates. Read them as ranges of plausible values; the paired differences are read as pre-registered.
- Grade and words (descriptive): Haiku minus template, grade [0.820, 1.110] (snapshot) and [0.830, 1.089] (cache);
  words [-13.106, -6.658] and [-12.881, -6.535]. Both arms read far under the grade 9 limit.

### The sensitivity checker (no verdict reads it)

| Arm | Passes | Texts | Most common problems (start of the message, count) |
| --- | --- | --- | --- |
| template | 36 of 81 | 81 | "number _ sits in a sentence that does not say what it is" 60; "number _ is a negative defender distance (farther), called closer" 30 |
| Haiku, snapshot | 65 of 81 | 68 | "number _ is his made shots above expected, given to the model" 8 |
| Haiku, cache arm | 65 of 81 | 78 | "sits in a sentence that does not say what it is" 20; "is his made shots above expected, given to the model" 8 |

The template writes "Defenders stay -0.57 feet closer to him" for a negative value (dev item; the page means farther
away), with a second sentence saying a minus sign means more room. The pinned checker accepts this because every
digit is right; the newer checker refuses it. This is the kind of error section 1 of the dossier says the pinned
checker cannot see: the right digits with the wrong meaning in words. The template was not changed after the dev
stage showed it (amendment 2).

## Possessions, 480 test possessions of 40 training games (template only)

| Arm | Pass rate | Reading grade, mean (largest) | Words per text, mean (range) | Frames / sentences | Frame variety | distinct-2 |
| --- | --- | --- | --- | --- | --- | --- |
| template | 1.000 [1.000, 1.000] | 3.38 [3.34, 3.43] (4.6) | 68.0 [67.2, 68.8] (49 to 110) | 21 / 2,799 | 0.008 [0.007, 0.008] | 0.003 [0.002, 0.003] |

The sensitivity checker passes 480 of 480 template possession texts.

## Cost and time per 1,000 texts

| Arm | API spend | Time |
| --- | --- | --- |
| template | $0 | 1,000 texts written in 0.004 s (scouting and possession), written and checked in 3.04 s (scouting) and 1.23 s (possession), on the M5 CPU |
| Haiku (scouting) | $0.1979 for the 81 test players: 122 calls, 693,543 input and 257,001 output tokens (thinking included) at export.genai.PRICE_PER_MTOK; $2.44 per 1,000 players asked | an estimate of 3.0 s per item (median of 374 gaps between consecutive cache records of one setting; mean 4.2 s), so about 50 minutes per 1,000 items one at a time (computed: 3.0 s times 1,000) |

The Haiku time is an estimate from record timestamps: no call was made to time it, and calls can run in parallel.

### Haiku first attempts, test players (each checked again with the pinned checker)

| Prompt and call setting | Records | First attempt passes | Kept after at most 2 attempts |
| --- | --- | --- | --- |
| scouting-v3, adaptive thinking, 8,000 tokens (the last setting, D90) | 57 | 45 | 54 |
| scouting-v3, no setting stored (4,000 tokens) | 7 | 5 | 7 |
| scouting-v3, thinking off | 14 | 0 | 9 |
| scouting-v2, no setting stored | 10 | 4 | 8 |

The records of one player can span several settings (a failure was asked again under the next), so the rows are not
independent samples. Under the last setting, 45 of 57 first attempts pass the pinned checker.

## Dev stage (40 players, 240 possessions)

Template 1.000 on both kinds; Haiku snapshot 0.825 [0.700, 0.925], cache arm 1.000 [1.000, 1.000]; Haiku minus
template frame variety [0.338, 0.498]. The dev stage led to the three amendments; no threshold changed.

## What this means

- A deterministic template meets the pinned faithfulness checker on every test item, at no spend and in milliseconds,
  as RotoWire's template beat neural systems on extraction precision (Wiseman, Shieber and Rush 2017, Table 2). Pass
  rate on this checker is not enough to argue for a model.
- What the model buys is variety (frame variety 0.596 and 0.612 against 0.021) and, on the newer checker,
  fewer wrong meanings with right digits. Whether readers value the variety is the human protocol's question (P7).
- For genai-E2: the checker that trains (rejection sampling) and scores must be named and pinned, a newer checker
  should be applied beside it, and variety is reported beside every pass rate. A trained model that copies a
  template would pass the pinned checker and say nothing new; frame variety shows it.
- Nothing here reaches the site (D92).

## What would change the verdict

A Haiku arm that loses texts to a later checker (the snapshot) and one that does not (the cache) give the same
readings, so the verdict does not hang on that choice. It would change under the newer checker as the primary
evaluation: there the template is not a substitute (36 of 81). That checker is still changing on its branch, so it
is reported, not adopted.
