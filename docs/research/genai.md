# Research dossier: genai (machine-written texts checked number by number)

- Lane: genai, the P3 generator (export/genai.py, PLAN.md D76, D90) and its checker. Not a frozen V3 lane: it has no
  row in reports/results.md, and nothing from this lane reaches the site without a D line (PLAN.md D92).
- Date: 2026-11-17. Pre-registrations below were written before any genai-E1 or genai-E2 run; the splits were fixed
  and hashed before any text was scored (reports/research/genai-splits.json, tools/genai_data.py).
- Verdict before any experiment: the checker is a strict, deterministic evaluation of faithfulness to numbers, and
  claude-haiku-5-5 meets it after retries on 398 of 405 players. What is not known: whether a text with no model can
  meet it too (genai-E1), and whether a small open model run on this Mac can be trained to meet it (genai-E2).

## 1. What we built (P3)

- Task: one short paragraph per item, from a list of facts. Two kinds. Scouting: a note on one player of the prep
  tab, from his entry in players.json (descriptive training-game tables, model values out of fold; PLAN.md rule 4).
  Possession: an account of one possession of the review, from its index row and the per-possession values of its
  public file (result, points, the EPV curve, the shots with their xFG).
- Input: export.genai.player_input and export.genai.possession_input build the facts. Each fact is a field of the
  page and the value as the page prints it (`shown`), with a unit, a kind (value, lo, hi, point, fact) and a plain
  description (`about`). The model gets a versioned prompt (export/prompts, each version pinned by sha256) and the
  facts as JSON.
- Generator: claude-haiku-5-5, at most 2 attempts per item; the second attempt sees the checker's problems. The call
  setting and the history of the scouting runs are in PLAN.md D90.
- Output so far: the snapshot this lane reads (genai/scouting.json of the v4 site data root at the freeze) holds
  335 scouting texts of 405 players, the count D93's re-check kept; the Haiku cache records hold a kept text for 398
  (genai-E1, amendment 1). The possession texts are about held-out games; this lane never reads them.

### The checker as an evaluation

export.genai.check_text is the evaluation of this lane. It is deterministic, it needs no reference text, and it reads
only the item's facts. A text passes when it finds no problem. Its rules, as the problems it reports group them
(tools/genai_eval.py RULES):

| rule | what fails |
| --- | --- |
| number_value | a number that is not the shown value of a fact of the item, or is in another unit (a share written as a percent, feet as points) |
| number_word | a number written as a word or an ordinal ("two", "half", "first"), or a non-ASCII digit |
| interval | a value with an interval given without both bounds |
| point_estimate | a point estimate given without the words "point estimate" anywhere in the text |
| model_word | a model's value whose sentence has none of "model", "expected", "predicted", "estimate" |
| status | a value from a lane whose verdict is not "works" without that verdict in the text |
| lane | a word that names a lane that did not work and is not on the page (decisions, matchups, lineups, passes, drives, screens, coverage, play calls, betting) |
| grade | Flesch-Kincaid grade above 9 for the text or for any one sentence (the site's own readability port) |
| length | fewer than 30 or more than 160 words |
| style | jargon of the site's copy rules, forbidden claims ("best", "should", "proves"), markup, dashes, more than one paragraph |

What it cannot see: a claim in words that the numbers do not support ("a strong shooter" next to a negative value),
a fact given the wrong meaning with the right digits (a rebound rate called a shot rate), the wrong player named from
the input's own list, and whether the text is any good to read. A branch not yet merged (p3-v3) adds a check of words
of change ("rose", "fell") against the values they link; genai-E1 and genai-E2 use the checker as pinned in the
splits file (git blob of export/genai.py at the freeze), and the report says which.

## 2. Foundations

| paper | link | what we take, and how ours differs |
| --- | --- | --- |
| Wiseman, Shieber, Rush. Challenges in Data-to-Document Generation. EMNLP 2017. | https://arxiv.org/abs/1707.08052 | RotoWire: NBA box scores to game summaries, scored by extractive metrics (an information extraction model run on the output). Their template system "scores very poorly on BLEU, but does quite well on the extractive metrics" (Section 6.1): relation generation precision 99.30 percent on test against 71.82 for the best neural row (Table 2). genai-E1 asks the same question of our checker. Ours checks every number by rule against the item's own facts, not by a learned extractor, and needs no reference text. |
| Thomson, Reiter. A Gold Standard Methodology for Evaluating Accuracy in Data-To-Text Systems. INLG 2020. | https://arxiv.org/abs/2011.03992 | Human accuracy annotation: three annotators per text mark each error as "Incorrect number", "Incorrect named entity", "Incorrect word", "Context error", "Not checkable" or "Other", merged by majority (Sections 3.2 to 3.5). Our human faithfulness protocol (genai-E2) uses their categories for what the checker cannot see; our readers are not paid crowd workers, and our texts are about 100 words, not 300. |
| Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv 2106.09685, 2021. | https://arxiv.org/abs/2106.09685 | Freeze the weights and train low-rank matrices in each layer; "on-par or better than fine-tuning" on the models they test, with "no additional inference latency" (abstract). We use it because a 1.5B model cannot be fully fine-tuned in 16 GB. |
| Kim, Rush. Sequence-Level Knowledge Distillation. EMNLP 2016. | https://arxiv.org/abs/1606.07947 | Train the student on the teacher's output sequences; their student "runs 10 times faster than its state-of-the-art teacher with little loss in performance" (abstract, translation). Our A1 arm trains on the Haiku texts, which are also filtered by the checker, so it learns only from passing outputs. |
| Yuan et al. Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv 2308.01825, 2023. | https://arxiv.org/abs/2308.01825 | Rejection sampling fine-tuning: k sampled paths per question, wrong ones filtered by answer and by a Python check, duplicates removed (Section 3); on GSM8K, LLaMA-7B rises from 35.9 (SFT) to 41.7 (RFT, k = 100), and "RFT brings more improvement for less performant LLMs" (abstract, Table 1), while for 33B "RFT does not improve performance compared to SFT". Our A2 is the same loop with our checker as the filter, k = 4. |
| Qwen team. Qwen2.5 Technical Report. arXiv 2412.15115, 2024. | https://arxiv.org/abs/2412.15115 | The 0.5B and 1.5B open-weight models are Apache 2.0, 32K context with 8K generation (Table 1); the 3B is under the "Qwen Research" license, so it is not used. |

## 3. Benchmark

There is no published number on this checker: it is ours. The closest published task is RotoWire (NBA box score to
game summary). Its numbers (relation generation precision of an information extraction model run on the output) are
not comparable with a rule checker's pass rate, so no row is filled with a number here; what Wiseman et al. found
(a template far ahead on extraction precision and far behind on BLEU, Table 2) is the pattern genai-E1 tests on our
checker and our variety measure.

## 4. Measured weaknesses (from our records only)

1. A pass needs retries and re-asks. The scouting runs dropped items under every setting tried before the last
   (D90: 14 of 50 truncated by thinking on the first setting; 21 of 50 dropped with thinking off), and the delivered
   texts come from three prompt versions and three call settings. The rate at which one call passes is not on
   record; genai-E1 reads it from the cache records.
2. Cost and dependence. Every text is a paid call to an outside provider, and each page's numbers and player names
   are sent to it (D83). The possession texts of training games (for any comparison off the held-out games) wait for
   a yes to spend (D92).
3. The checker passes texts it cannot read for sense (section 1). No human has labelled any text.
4. Variety and value to a reader are not measured at all.

## 5. Recent work that targets these weaknesses

| weakness | work | what it offers | fit for us |
| --- | --- | --- | --- |
| 1, retries | Zelikman et al. STaR. arXiv 2203.14465, 2022. https://arxiv.org/abs/2203.14465 | Generate, keep the outputs that reach a correct answer, fine-tune on them, repeat (abstract) | The loop of A2, with the checker in place of the answer key; one or two rounds only. |
| 1, retries | Gulcehre et al. Reinforced Self-Training (ReST). arXiv 2308.08998, 2023. https://arxiv.org/abs/2308.08998 | Samples from the policy form an offline dataset that improves it; "more efficient than typical online RLHF methods" since the data is reused (abstract); applied to translation | The same family as A2. We filter by a pass or fail rule, not a learned reward, so no reward model can be gamed, only the checker. |
| 1, retries | Dong et al. RAFT: Reward rAnked FineTuning. TMLR. arXiv 2304.06767. https://arxiv.org/abs/2304.06767 | Sample many, keep the high-reward samples, fine-tune on them (abstract) | Another form of A2; ours keeps every pass, at most 2 per item, rather than a top fraction. |
| 1, numbers | Geng et al. Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. EMNLP 2023. https://arxiv.org/abs/2305.13971 | "input-dependent grammars" let the allowed output depend on the input; constrained models "substantially outperform unconstrained LMs" (abstract) | A3 is an input-dependent constraint: the digits allowed are those of the item's own facts. |
| 1, numbers | Willard, Louf. Efficient Guided Generation for Large Language Models. arXiv 2307.09702, 2023. https://arxiv.org/abs/2307.09702 | Constrained generation as transitions of a finite-state machine, with an index over the vocabulary; "adds little overhead" (abstract), implemented in the Outlines library | The efficient way to build A3 if a simple logits processor is too slow; we start with the simple one. |
| 2, cost | Kim, Rush 2016 (section 2) and Hu et al. 2021 (section 2) | A smaller student from a teacher's outputs; training that fits small memory | genai-E2 as a whole. |
| 3, evaluation | Dhingra et al. Handling Divergent Reference Texts when Evaluating Table-to-Text Generation (PARENT). ACL 2019. https://arxiv.org/abs/1906.01081 | Reference-based metrics such as BLEU "show poor correlation with human judgments when those references diverge"; PARENT aligns n-grams to the table (abstract) | Our checker needs no reference text, so this problem does not arise; PARENT is not used. Kept as the reason BLEU against the Haiku texts is not a metric here. |
| 3, evaluation | Thomson, Reiter 2020 (section 2) | Error categories for a human accuracy check | The human protocol of genai-E2. |
| 4, variety | Li et al. A Diversity-Promoting Objective Function for Neural Conversation Models. NAACL 2016. https://arxiv.org/abs/1510.03055 | distinct-1 and distinct-2: "the number of distinct unigrams and bigrams divided by total number of generated words" (Table 2); models "tend to generate safe, commonplace responses" (abstract) | distinct-2 on sentence frames, next to our own frame variety, in E1 and E2. |
| 1 to 4 | Puduppully, Dong, Lapata. Data-to-Text Generation with Content Selection and Planning. arXiv 1809.00582, 2019. https://arxiv.org/abs/1809.00582 | Plan which facts to say and in what order, then write (abstract); state of the art on RotoWire at the time | Not used: our items are short and the prompt names what to say. A reference for why a template's fixed order is a plan by hand. |

## 6. Pre-registered experiments

### Data and splits (fixed before any run)

- Scouting items: export.genai.player_input on every player of players.json, from a snapshot of the v4 site data
  root (players.json, results.json and the Haiku texts in genai/scouting.json, each pinned by sha256), split by
  player: players in sha256("genai-v1:" + slug) order, the first 20 percent test (81), the next 10 percent dev (40),
  the rest train (284).
- Possession items: the possession review shows held-out games only, so no review text can be a training or test
  item. Items are built from TRAINING games with the same code path: tools/genai_data.py builds the contract models
  export.site_games builds (PossessionSummary, Possession) from the same lane tables, with out-of-fold model rows only
  (EPV states and xFG predictions with fold >= 0), and calls export.genai.possession_input on them. It reads no
  tracking frame; the two places where a site build would read one (the shot time and the shooter fallback) are
  written in the module and in the splits file. Games are the development games of tools.research_testset.dev_games
  (no D71 test game, no held-out game), in sha256("genai-v1:" + game id) order: 40 test, 20 dev, 120 train, the
  rest unused. Possessions per game, drawn by a seeded generator: 12 for test and dev, 25 for train.
- reports/research/genai-splits.json holds counts and hashes only: the sha256 of each split's keys and of each
  split's input hashes, the snapshot pins and the checker's git blob. The items live outside the repository.
- The test items are read only by tools.genai_data.score_test_once, which writes one entry per experiment to
  reports/research/genai-ledger.json before its scorer runs and refuses a second call. Tests hold this
  (tests/test_genai_research.py).

- The players.json snapshot was frozen at 21:54 UTC on 2026-11-17, before D94 (committed at 23:07 UTC), so the
  scouting items hold values that D94 now withholds as broken by the frozen name joins: 5 rebound rates of 0.0 (4
  train players, 1 test player) and Jeff Green's shot making, which mixes in another player's attempts (a train
  player); D94's other withheld rebound rates are not identified here. A faithfulness pass rate is not affected (a
  text is checked against the item it was written from), but Haiku teacher texts state those numbers, so a trained
  arm learns from them. No text of this lane reaches the site (D92).

### genai-E1: a template writer against the Haiku texts

**Result (2026-11-17, reports/research/genai-E1.md): H1 and H2 hold.** On the pinned checker the template passes
81 of 81 test players and 480 of 480 test possessions; Haiku 0.840 [0.753, 0.914] (snapshot) and 0.963 [0.914,
1.000] (cache arm), every Haiku failure an item with no text. Haiku minus template frame variety [0.331, 0.439]. On
the sensitivity checker (p3-v3 7f2c08fc) the template passes 36 of 81 scouting texts and Haiku 65 of 81. The
pre-registration below is unchanged; the amendments follow it.

- Question. Can a writer with no model meet the checker as well as claude-haiku-5-5, at what cost in variety?
- Arms. Template (tools/genai_templates.py): fixed sentences filled from the facts, one frame per slot, no random
  choice; written and checked on train items only, before E1 reads a dev or test item. Haiku: the text the site
  snapshot holds for each player (as delivered: up to 2 attempts per call, re-asked across settings), or none when
  every attempt was dropped.
- Items. Scouting: the 81 test players (both arms). Possession: the test possessions (template only; the Haiku
  possession texts are about held-out games and are never read, and a Haiku arm on training games needs a yes to
  spend, D92). A dev stage on the 40 dev players and the dev possessions runs first and reads no test item.
- Metrics. (1) Pass rate: share of items whose arm gives a text that passes export.genai.check_text (an item with no
  text fails, rule no_text). (2) Failure rate of each rule (share of items). (3) Reading grade of each text (the
  checker's Flesch-Kincaid port, rounded as the site rounds it): mean and largest. (4) Words per text: mean, smallest,
  largest. (5) Variety: a sentence frame is a sentence with the item's names replaced by one token and every number
  the checker reads replaced by another; frame variety is distinct frames over sentences, and distinct-2 is distinct
  word pairs of the frames over their words (Li et al. 2016, Table 2). (6) Cost per 1,000 texts: template, $0 and the measured
  seconds to write and to write and check 1,000 texts on the M5; Haiku, the tokens of every cached call for the test
  players (same input hash) priced at export.genai.PRICE_PER_MTOK, per 1,000 items asked. (7) Time per text: template
  measured; Haiku estimated from the creation times of consecutive cache records (no call is made to time it), and
  labelled an estimate. (8) For Haiku also the first-attempt pass rate of each prompt version and call setting, each
  first attempt checked again with the pinned checker.
- Intervals. 2000-resample bootstrap of players (scouting) or games (possession), seed 0, 95 percent percentile
  interval; paired differences on the same draws.
- Hypotheses and readings (fixed now). H1, the checker does not separate a template from the model: template minus
  Haiku pass rate has a 95 percent lower bound at or above -0.05. H2, the model writes with more variety: Haiku minus
  template frame variety has a 95 percent lower bound above 0. If H1 and H2 hold, the checker alone cannot show
  what a model adds; genai-E2 must report variety beside pass rate, and the value of a model is a question for the
  human protocol below. If H1 fails, the template is not a free substitute, and the failing rules say why. Grade,
  words, cost and time are descriptive and carry no threshold.
- Human readability protocol (written now, run later in P7, not part of E1's verdict). 30 test players drawn by a
  seeded generator. For each, the template text and the Haiku text, in random order, with no arm named, shown to at
  least 3 readers who know basketball. Each answers: which of the two is easier to read; which would you rather get
  before a game; and marks any sentence that says something the numbers do not support. Read as the share of pairs
  where the Haiku text is preferred, with a Wilson 95 percent interval over pairs and the per-reader shares beside
  it; no threshold is set before the readers are known.
- Data. No held-out game. Scouting values are the site's descriptive training-game tables. No model is called.
- Compute. CPU, seconds: the checker on about 600 texts and a bootstrap of 2000 draws. The freeze of the splits reads
  the small per-game tables of 180 training games under the lock (about a minute).
- Risks. The template was written to the checker's rules, so its pass rate measures how well the rules are met by
  construction, not writing skill; that is the point of the baseline. A template written on train items can still
  fail on a test item with a rare shape (a missing view, a long name); that failure is reported, not fixed.
- Backlog scores. Gain 3: it decides how E2 must be read (pass rate alone or with variety) and costs nothing. Effort
  1: no model, existing checker.

#### genai-E1 amendments after the dev stage, before the test score (2026-11-17)

The dev stage (reports/research/genai-E1-dev.json) ran on 40 dev players and 240 dev possessions. Three things it
showed are fixed here, before the one test score, and change no threshold or reading above:

1. The Haiku arm as pre-registered (the texts the site snapshot holds) is a moment in P3's run, not what its calls
   produced. The v4 site data root held 398 texts at 13:29 local time, 335 at the freeze (21:54 UTC, the pinned
   snapshot) and fewer after: P3's branch is pruning it with checks newer than the pinned checker (words of change,
   subjects, attribution, signs; commits d7542f90 to 7f2c08fc on p3-v3). All 398 kept texts of the cache records
   still pass the pinned checker. So a second Haiku arm is added and reported beside the first: the kept text of the
   latest cache record whose input is the item's input (haiku_cache). The pre-registered arm stays the one H1 and H2
   are read on; the report gives both readings and says if they differ.
2. A sensitivity checker is added: export/genai.py of p3-v3 at commit 7f2c08fc, read from git, applied to every arm.
   No verdict reads it. On dev it fails most template scouting texts (a number in a sentence that does not say what
   it is; a negative defender distance written as "closer"), which the pinned checker does not see. The template is
   not changed after reading dev: it stays as settled on train items (commit a5eefb1d), and the report shows this
   as a finding about what each checker can see.
3. Distinct counts are biased downward under resampling (a resampled player's sentences repeat), so the bootstrap
   intervals of frame variety and distinct-2 sit below their point estimates. As for binned ECE in xfg-E1, they are
   read as ranges of plausible values, not centred intervals. Both arms share the bias; the paired difference is
   read as pre-registered.

### genai-E2: a small open model, LoRA on the Mac, trained against the checker

- Question. Can a 0.5B or 1.5B open instruct model, fine-tuned on this Mac, write texts that pass the checker at a
  rate near the API model's, at no spend per text?
- Models. Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (Apache 2.0 on their Hub pages, not gated: no account or
  terms to accept), each pinned by its Hub revision (0.5B 7ae55760, 1.5B 989aa798). Weights and caches live outside
  the repository (a _genai-models folder beside the worktrees, D44); the venv outside the repository and outside iCloud.
- Input format (fixed before any E2 run; changed from the first draft after the first pilot, see below). The
  trained arms (A1, A2, A3) get the compact input of tools/genai_format.py: a one-sentence fixed instruction as the
  system turn, and as the user turn the item's context on a few short lines and one line per fact or estimate
  ("field: shown value", a range as "(lo to hi, 95%)", a point estimate marked "point", the EPV curve and the
  similar players folded into one line each, a sample size that repeats a count left out). It holds the same numbers
  as the item and nothing else. On the train examples (reports/research/genai-E2-format-tokens.json, after the
  target filter of amendment g) it is 642 tokens per scouting example with its target on average (largest 757, 220
  examples) and 469 per possession example (largest 1,267; 29 of 3,000 over 800, 3 over 1,024), against 3,812 and
  3,021 on average in the P3 prompt and JSON form (first 200 of each kind; Qwen2.5 tokenizer). Training examples
  over 1,024 tokens are left out of training and counted; no evaluation item is ever left out. The first pilot, on the P3 form, ran out of memory on its first
  step (about 3,800 tokens, 19 GiB of activations), which is why the format changed.
- Arms, for each model size.
  - A0 few-shot: the base model with the rules block of tools/genai_format.py (RULES, the P3 prompt's rules cut to
    the compact format) as the system turn and 2 worked examples as earlier chat turns, in the compact format, drawn
    by a seeded generator from train items (scouting: train players and their Haiku texts; possession: train
    possessions and their template texts). A0 and the trained arms see the same input format, so H1 measures the
    training alone. Beside it, A0-P3: the base model zero-shot on the full P3 prompt and JSON message, exactly the
    setup the Haiku texts had, reported and read by no hypothesis.
  - A1 SFT (distillation): LoRA on train items. Scouting targets are the Haiku texts of train players (sequence-level
    distillation in the sense of Kim and Rush 2016). Possession targets are the template texts of train possessions,
    since there is no Haiku possession text on a training game; if Lucien says yes to the Haiku spend on training-game
    possessions, a second possession A1 on Haiku targets is added and reported apart.
  - A2 rejection-sampling fine-tuning: from A1, the model writes k = 4 candidates per train item (temperature 0.8,
    top-p 0.95), the checker keeps the passing ones (at most 2 distinct per item), and LoRA continues on them for
    one epoch. A second round runs only if round 1 raises the dev pass rate by at least 0.03 over A1.
  - A3 constrained decoding (ablation): the dev-best arm, decoded with a logits processor that lets a run of digits
    form only a prefix of a fact's shown value of that item (its digits, decimal point and comma; the sign and the
    percent sign as the fact writes them). It cannot fix a unit, a missing bound or a missing word.
- Training settings (fixed now; the pilot only checks they fit). LoRA rank 16, alpha 32, dropout 0.05, on the
  attention and MLP projections; AdamW, learning rate 2e-4 with linear decay and 10 warm-up steps; effective batch 8
  by gradient accumulation; loss on the answer tokens only; up to 3 epochs, the checkpoint kept at the epoch with the
  best dev pass rate. Seed 0. The pilot may change only the number format (float32 or bfloat16) and the batch layout
  if memory needs it; both are recorded. Gradient checkpointing is on (it changes memory and time, not the
  result), and every run caps the MPS pool (PYTORCH_MPS_HIGH_WATERMARK_RATIO 0.6, LOW 0.5) so a run that needs more
  memory stops instead of swapping.
- Checker. Superseded by the amendment of 2026-11-17 below (primary checker for E2 is the merged one).
  First draft: the checker pinned at the split freeze (git blob c10b8a4b) scores every arm and filters A2, with
  E1's sensitivity checker beside it.
- Amendment, 2026-11-17, before any E2 run (lead's decision): E2's primary checker is export/genai.py of
  v4-research at db993008, git blob 41ba991b, which holds the merged p3-v3 checks (merge 2021695d: words of change,
  negation, subjects, attribution, signs; D93). Its check_text is the same code as E1's sensitivity checker
  (export/genai.py at 7f2c08fc and at 2021695d are identical; db993008 adds only a hand-read sheet writer). It scores
  every arm, decides every hypothesis and is the filter of A2's rejection sampling. The freeze checker (blob
  c10b8a4b) is reported beside every score and decides nothing. The evaluation loads each checker from git by its
  pinned commit and verifies the blob hash before it scores, so a later change to export/genai.py cannot move an E2
  number. The frozen items are unchanged: both checkers run on them as built. Before this amendment, on the dev
  items, the template passes 6 of 40 scouting and 240 of 240 possession texts under the primary checker (dev only;
  no test item read).
- Amendments after the ml-review of the E2 code (2026-11-17, before any training run; lead's decisions with the
  reasons here):
  - (a) Why the primary checker changed, and when. It was chosen after genai-E1 had shown, on it, Haiku 65 of 81
    and the template 36 of 81 on the test players. The reason is that it is the checker the site's texts now pass
    (D93) and it sees wrong meanings with right digits that the freeze checker misses (E1). The choice was made
    before any E2 arm produced a text, and it does not favour one E2 arm over another; it does make every E2 pass
    rate harder to reach than under the freeze checker, which is reported beside it.
  - (b) H3 is read against the Haiku cache arm (the kept text of the latest cache record for the item's input), the
    fairer of E1's two Haiku arms: the snapshot arm also lost texts to checks added after the calls. The snapshot
    arm is reported beside it and decides nothing.
  - (c) Repair-loop asymmetry, stated, not removed. The small model's second attempt sees the primary checker's
    problems. Haiku's texts came from up to 2 attempts per call whose feedback came from the checker of that day,
    and some players were asked again under later prompt versions and call settings (E1, first-attempt table), so
    Haiku as delivered had more tries than 2. H3 compares delivered outputs, not equal effort.
  - (d) H4 is read on two numbers, both in code: the pass rate (as before) and the number_value failure rate (A3
    minus its arm). Wording corrected on 2026-11-18 to the code's rule: the change in failure rate reads "fewer
    wrong numbers" when its whole 95 percent interval is below 0, "more wrong numbers" when the whole interval is
    above 0, and "inconclusive" otherwise.
  - (e) Epoch choice weighs the two kinds equally: the dev pass rate of an epoch is the mean of the scouting and the
    possession dev pass rates (pooled, the 240 dev possessions would outweigh the 40 dev players 6 to 1). The same
    mean decides the dev-best arm for A3 and the round-2 rule.
  - (f) A3 takes its arm from the rule: the arm with the highest mean dev pass rate among the chosen A1 epoch and
    the last A2 round. No flag chooses it.
  - (g) Training targets and few-shot examples are kept only when they pass the primary checker; the number dropped
    per kind is recorded in the run state and in the dev json.
  - (h) RFT round 2 runs only when the mean over kinds of (A2 round 1 minus the chosen A1 epoch) dev pass rate is at
    least 0.03; the code refuses round 2 otherwise.
  - (i) The test stage refuses to run unless reports/research/genai-E2-0.5B-dev.json is tracked by git and has no
    uncommitted change; the final json records its sha256 and the HEAD commit. Each arm's test texts are saved
    outside the repository as each arm finishes. The exact test path is rehearsed once on the dev items (the same
    function, without the ledger) before the one test call.
- Amendments after the pre-score review (2026-11-18, after the dev choices of f3cbc2a7 and before the test score;
  lead's decisions):
  - (j) Crash safety. Each arm is scored under both checkers as soon as its texts exist, and its full rows (with
    sentence frames) are saved outside the repository. A checker that raises on a text counts that text as a fail
    under rule "other", not as a crash. Memory is freed between arms. One named resume entry is pre-registered:
    "genai-E2-0.5B-resume", allowed only after a crash of the "genai-E2-0.5B" scoring call, with the same code
    commit and the same dev choices, generating only the arms that have no saved rows (greedy decoding, so nothing is
    chosen again); the report says whether it was used.
  - (k) Variety with intervals. Beside every pass rate: frame variety and distinct-2 with bootstrap intervals by
    unit, and for each arm the share of possession texts identical to the template's text of that item and the
    share of the arm's sentence frames found among the template's frames. No hypothesis reads them.
  - (l) The rehearsal on the dev items records the dev json's sha256 and the git HEAD; the test stage requires both
    to equal the current ones, and refuses when tools/ or export/ has an uncommitted change.
  - (m) H1 and H3 use the dev-best arm of each kind (best_by_kind in the dev json). In this run the dev-best arm of
    both kinds is A1 epoch 1, so H1, H3 and the A3 arm all use it.
  - (n) The snapshot Haiku arm checks each text's input_sha256 against its item, as genai-E1's haiku_rows does.
  - (o) Two memory changes made during the runs, neither changing a result: A0-P3 generates one item at a time (a
    batch of 8 full P3 prompts needed about 7 GB for attention and stopped under the cap, 2026-11-17); training
    empties the MPS cache at every optimizer step and saves a checkpoint every 400 examples (the first sft stretch
    stopped under the cap after about 200 examples, 2026-11-18). After them, training memory held at 2.9 to 3.0 GB.
- Decoding for every score: greedy, at most 320 new tokens, one attempt (primary). Secondary: a repair loop as P3's
  (one more attempt with the checker's problems in the prompt, at most 2 in all), which is the fair comparison with
  the Haiku texts as delivered.
- Selection. All choices (epoch, RFT round 2, the dev-best arm for A3) are made on dev items. The test items are
  scored once per model size, all arms in one score_test_once call ("genai-E2-0.5B", "genai-E2-1.5B"), after the
  dev choices are written to the dev json.
- Primary metric. Single-attempt checker pass rate on test items, each kind apart: scouting over 81 players,
  possession over the test possessions of 40 games.
- Hypotheses and readings (fixed now). Paired bootstrap by player (scouting) or by game (possession), 2000 resamples,
  95 percent percentile interval of the difference.
  - H1, training helps: the dev-best fine-tuned arm minus A0, per kind and size. Passes when the difference is at
    least +0.10 and its lower bound is above 0; fails when the upper bound is below +0.10; otherwise inconclusive.
  - H2, the checker as a teacher: A2 minus A1. Lower bound above 0: rejection sampling adds to distillation; upper
    bound below 0: it hurts; otherwise inconclusive.
  - H3, near the API model (scouting only): the dev-best arm with the repair loop minus Haiku as delivered. Lower bound
    at or above -0.05: non-inferior; upper bound below -0.05: inferior; otherwise inconclusive. At 81 players the
    interval half-width of a pass-rate difference is about 0.1 near rates of 0.8, so inconclusive is a likely reading
    and is stated as such.
  - H4, constrained decoding: A3 minus its arm, on the number_value failure rate and on the pass rate. The pass
    rate is read by the H2 rule; the failure rate as amendment (d) says.
- Secondary. Failure rate per rule; reading grade; words per text; frame variety and distinct-2 (as E1); latency per
  text and tokens per second on the M5 (MPS); peak memory (MPS driver and process); training wall time; texts
  produced; cost per 1,000 texts as electricity is not measured, so the cost is stated as $0 in API spend and as
  lock time.
- Human-labelled faithfulness sample (a protocol, run later; tightened after the ml-review). Kind: scouting, where
  D93's hand-read found errors the checks cannot see (possession: 0 of 15). Texts: 50 test texts of the 0.5B
  dev-best arm that pass the primary checker, drawn by a seeded generator, mixed blind with 25 Haiku cache-arm texts
  and 25 template texts that also pass it (if fewer pass, all that pass, and the count is reported). Readers: at
  least 2 who did not write or tune the template, the prompts or the checker. Each reader marks every sentence
  with the error categories of Thomson and Reiter (section 2), and counts wrong attributions exactly as
  docs/GENAI_HAND_READ.md defines them (D93), so the count stands next to D93's 3 of 15. Agreement on "the text
  has at least one error" is reported (raw agreement and Cohen's kappa) before any disagreement is settled; a third
  reader then settles them. Read as the share of passing texts with at least one error, and the share with a wrong
  attribution, each with a Wilson 95 percent interval, per arm.
- Data. No held-out game is used to train, tune or evaluate (D92). Scouting rows are descriptive training-game
  tables; possession rows are training games with out-of-fold model values. No API call.
- Compute. 16 GB Mac (Apple M5, MPS), one job at a time under the heavy-job lock, weights read from the local cache.
  No training run starts before the lead's go. Pilot (2026-11-17, tools/genai_pilot.py, 96 s under the lock; a
  text-free summary in reports/research/genai-E2-pilot.json): 0.5B in float32, 20 LoRA steps on compact examples of 545 tokens on average, 0.94 s
  per step after the first, 8.8 million trainable parameters, peak MPS memory 5.2 GB, generation 41 to 50 tokens a
  second; 1.5B in bfloat16, 6 steps of 581 tokens on average, 1.32 s per step after the first, 18.5 million
  trainable, peak MPS memory 7.3 GB, generation 25 to 33 tokens a second. The base models on the full P3 prompt
  (3,889 prompt tokens): 3.8 s (0.5B, 114 new tokens) and 8.0 s (1.5B, 177 new tokens). Pilot texts were checked
  only to show the loop runs; no pilot pass rate is a result.
- Limits fixed from the pilot, before any E2 run: generation runs in batches of 8 items (left padding, greedy, the
  same outputs as one at a time up to floating point); the A2 candidate pool is every train scouting item with a
  target and 800 train possessions drawn by a seeded generator; A0-P3 runs on scouting items only (the P3 JSON
  form of a possession reaches 6,600 tokens).
- Risks and failure modes. The checker can be gamed: a model can learn to copy numbers and a stock "point estimate"
  sentence and pass while saying little; frame variety and the human sample watch for it. The scouting teacher texts
  are survivors of the checker, so distillation learns from passes only. 284 train players is a small SFT set and
  can overfit; the epoch is chosen on dev. Small test sets give wide intervals (H3). MPS kernels can differ from CPU
  numerically; every number comes from the MPS runs as they ran. A possession model trained on template targets can
  at best copy the template, so its variety is bounded by it; the report says so.
- What a coach gains. Nothing on the site until a D line; as evidence, a measured answer to whether a free local
  model can replace a paid API call for checked short texts.
- Backlog scores. Gain 4: a full fine-tune and evaluate loop on our own checker, with a cost answer. Effort 4: new
  training code, a venv, many lock hours.

## 7. Sources checked

All fetched 2026-11-17 at the source: the arXiv abstract page, and the arXiv PDF where a number or a section is
cited.

| source | link | what was read |
| --- | --- | --- |
| Wiseman, Shieber, Rush 2017 | https://arxiv.org/abs/1707.08052, https://arxiv.org/pdf/1707.08052 | abstract; Section 6.1; Table 2 |
| Thomson, Reiter 2020 | https://arxiv.org/abs/2011.03992, https://arxiv.org/pdf/2011.03992 | abstract; Sections 3.2 to 3.5 |
| Hu et al. 2021 | https://arxiv.org/abs/2106.09685 | abstract |
| Kim, Rush 2016 | https://arxiv.org/abs/1606.07947 | abstract (venue from the arXiv comment) |
| Yuan et al. 2023 | https://arxiv.org/abs/2308.01825, https://arxiv.org/pdf/2308.01825 | abstract; Section 3; Table 1 |
| Zelikman et al. 2022 | https://arxiv.org/abs/2203.14465 | abstract |
| Gulcehre et al. 2023 | https://arxiv.org/abs/2308.08998 | abstract |
| Dong et al. 2023 | https://arxiv.org/abs/2304.06767 | abstract (venue from the arXiv comment) |
| Geng et al. 2023 | https://arxiv.org/abs/2305.13971 | abstract (venue from the arXiv comment) |
| Willard, Louf 2023 | https://arxiv.org/abs/2307.09702 | abstract |
| Li et al. 2016 | https://arxiv.org/abs/1510.03055, https://arxiv.org/pdf/1510.03055 | abstract; Table 2 caption |
| Dhingra et al. 2019 | https://arxiv.org/abs/1906.01081 | abstract (venue from the arXiv comment) |
| Puduppully, Dong, Lapata 2019 | https://arxiv.org/abs/1809.00582 | abstract |
| Qwen team 2024 | https://arxiv.org/abs/2412.15115, https://arxiv.org/pdf/2412.15115 | abstract; Table 1 |
| Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct on the Hugging Face Hub | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct, and the /api/models/ pages of both | model cards (0.49B and 1.54B parameters, 32,768 context); license apache-2.0, gated false, revisions 7ae55760 and 989aa798 |
| SmolLM2-360M-Instruct and SmolLM2-1.7B-Instruct (fallback only) | https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct, https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct | model cards and API: Apache 2.0, not gated; not used unless a Qwen model fails the pilot |

Not used: no venue is printed on the arXiv pages of STaR, RFT, ReST, Outlines, Puduppully et al. or the Qwen2.5
report, so they are cited as arXiv preprints. Nothing above is cited from memory.
