# Hand-read of machine-written scouting texts (PLAN.md D93)

The checks of `export/genai.py` hold every number of a text to its page: the value, the rounding, the unit, the
range, the sign word, the words of what it is, and whose number it is in the three exact cases they can see. A
hand-read found what they cannot: 3 of 15 scouting texts gave the player's made-above-expected value to the model.
D93 lets the scouting texts ship only when a fresh hand-read sample of 30 has at most 1 wrong attribution. This is
how that sample is drawn and read.

## 1. The texts to read

A live run of the current scouting prompt (`export/prompts/versions.json`, `current.scouting`) on every player,
after a person says yes to the spend:

    python3 -m export.genai --root <data root> --kinds scouting --all --live --cache <cache>

Only texts that passed every check are in the root, so the sample reads what would ship.

## 2. The sample

Thirty texts, drawn with the fixed seed 93, leaving out every player read in an earlier hand-read (one key a line in
a file kept with the earlier sheets):

    python3 -m export.genai --root <data root> --kinds scouting --hand-read 30 --seed 93 \
        --exclude <keys read before> --out <folder outside the repository>/scouting-hand-read.csv

The draw does not depend on the reader, and the same root, seed and exclusions give the same 30 players. The sheet
holds text written from the data root, so it is written outside the repository and never committed. Each row has
the key, the name, the prompt version, the call setting, the text, and the facts of the page as the page shows them
(value, field and what it means), then five empty columns for the reader.

## 3. How to read a text

Read the text sentence by sentence beside its facts. For every number, ask two things: is it this fact, and does
the sentence say what this fact is, and whose it is. Then read the sentences with no number.

**A wrong attribution** is a number or a claim given the wrong owner or the wrong meaning. Count each one:

- the player's value given to the model, or the model's value given to the player (his made shots above
  expected written as the model's; the model's own error written as his shot-making);
- a value of one view written as another (chances called rebounds; the model's error called the gap);
- a rank or a sample size written as a count of something else (rank 4 written as 4 views);
- a sign word against the sign of the value (a value with a minus sign called closer);
- a value of a similar player given to this player, or the reverse.

Write the count in `wrong_attributions` (0 when there is none) and quote each case in `attribution_notes`.

**An unsupported claim** is a statement that no fact of the page supports: a cause, a play type, a comparison or a
trend the page does not give, a status word that is not the model's status, advice, or a caveat turned around (the
absence chance called an injury forecast). Write the count in `unsupported_claims` and quote each case in
`faithfulness_notes`. Write your name or initials in `reader`.

Read all thirty before counting the total, and do not change a mark after you have seen the total.

## 4. The result

- **Pass:** at most 1 wrong attribution across the 30 texts. The scouting texts may ship (D93).
- **Fail:** 2 or more. The scouting texts do not ship; the cases go to the next prompt or to a new exact check, and
  a fresh sample of 30 (the 30 read here added to the exclusions) is read after the next run.
- Unsupported claims are counted and reported with the result, beside the wrong attributions; each one is a case
  for a check or a prompt rule, and the lead decides whether one blocks the deploy.

The record of a hand-read (in PLAN.md, by the lead) gives the prompt version, the call setting, the seed, the count
of exclusions, the two totals and the keys of the texts with a wrong attribution. It quotes no text.

## 5. Leaving the scouting texts out of a deploy

Until a sample passes, a deploy is built from a copy of the data root without them:

    python3 -m export.genai --root <data root> --withhold scouting --out <deploy copy>

then `SITE_DATA_DIR=<deploy copy>` for the site build. The prep pages then show no machine text; the data root keeps
the texts for the next hand-read.
