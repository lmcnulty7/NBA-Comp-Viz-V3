// The rules of site copy, applied to machine-written texts (PLAN.md P3). A generated text lives in the data root,
// not in web/content, so check-copy.mjs never sees it; check-reading.mjs runs these rules on every text of the
// data root. export/genai.py applies the same rules before it keeps a text (check_words), so a text that reaches
// the data root has passed them once already; this is the second, independent pass.
//
// A text holds digits (the numbers of its page, checked by lib/genai.ts and export/genai.py), so the no-digit rule
// of copy does not apply to it. Everything else does: reading grade 9 or lower for the text and for each sentence,
// none of the words kept out of copy, no claim the site never makes, no em or en dash, the ghost defenders never
// called the best defense, and one paragraph of plain text.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { BANNED } from "../check-copy.mjs";
import { loadRules } from "./contract.mjs";
import { fkGrade, plainWordProblems, round1, wordsOf } from "./readability.mjs";

export const MAX_GRADE = 9;
export const MIN_WORDS = 30;
export const MAX_WORDS = 160;
const DASH = new RegExp(`[${String.fromCodePoint(0x2013, 0x2014)}]`);
const MARKUP = /[*#_`<>[\]{}|]/;

/** Sentences of a text: split after closing punctuation, as export/genai.py sentences_of does. */
export function sentencesOf(text) {
  return text.trim().split(/(?<=[.!?])\s+/).filter(Boolean);
}

/** The problems of one generated text under the rules of site copy. */
export function generatedTextProblems(text) {
  const problems = [];
  if (text.includes("\n") || text !== text.trim()) problems.push("not one paragraph of plain text");
  if (MARKUP.test(text)) problems.push("markup characters");
  if (DASH.test(text)) problems.push("an em dash or an en dash");
  const count = wordsOf(text).length;
  if (count < MIN_WORDS || count > MAX_WORDS) problems.push(`${count} words, not ${MIN_WORDS} to ${MAX_WORDS}`);
  const grade = round1(fkGrade(text));
  if (grade > MAX_GRADE) problems.push(`reads at grade ${grade}, above ${MAX_GRADE}`);
  for (const sentence of sentencesOf(text)) {
    const one = round1(fkGrade(sentence));
    if (one > MAX_GRADE) problems.push(`a sentence reads at grade ${one}: "${sentence.slice(0, 60)}"`);
  }
  problems.push(...plainWordProblems(text));
  for (const [pattern, why] of BANNED) if (pattern.test(text)) problems.push(why);
  return problems;
}

/** Every text of the genai files of a data root, checked. A root with no genai file has nothing to check. */
export function checkGeneratedTexts(root) {
  const problems = [];
  let texts = 0;
  for (const rel of Object.values(loadRules().genai_files ?? {})) {
    const file = path.join(root, rel);
    if (!existsSync(file)) continue;
    const document = JSON.parse(readFileSync(file, "utf8"));
    for (const item of document.items ?? []) {
      texts += 1;
      for (const found of generatedTextProblems(item.text)) problems.push(`${rel} ${item.key}: ${found}`);
    }
  }
  return { problems, texts };
}
