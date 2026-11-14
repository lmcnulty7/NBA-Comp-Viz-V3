// Reading grade and plain-words checks for site copy, with no outside package (CLAUDE.md: textstat is not
// installed). The rules are the ones tests/test_public_docs.py applies to docs/SCRIPT.md, so the script and the
// copy on the page are held to the same count:
//
//   Flesch-Kincaid grade = 0.39 x (words per sentence) + 11.8 x (syllables per word) - 15.59
//
// A token ({file.field}) counts as one word of one syllable, the way a counter treats a numeral; the upper
// count reads it as four syllables.

/** A token in copy: {file.part.part}. A {letters} placeholder is not a token. */
export const TOKEN_RE = /\{([a-z][a-z_]*(?:\.[a-z0-9_]+)+)\}/g;
const PLACEHOLDER_RE = /\{[a-zA-Z]+\}/g;

// Syllable counts the rule below gets wrong, each checked by hand (the same list as tests/test_public_docs.py).
const SYLLABLE_OVERRIDES = new Map(Object.entries({
  video: 3, area: 3, idea: 3, create: 2, science: 2, league: 1, leagues: 1, nba: 3, tv: 2,
}));

/** Syllables of one word: groups of vowels, less a silent final e, es or ed. A word with no letter is one. */
export function countSyllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!w) return 1;
  if (SYLLABLE_OVERRIDES.has(w)) return SYLLABLE_OVERRIDES.get(w);
  // y is a consonant at the start of a word and between vowels (player, beyond), a vowel elsewhere
  const s = [...w]
    .map((c, i) => (c === "y" && (i === 0 || "aeiou".includes(w[i - 1])) && "aeiou".includes(w[i + 1] ?? "#") ? "j" : c))
    .join("");
  let n = (s.match(/[aeiouy]+/g) ?? []).length;
  if (/[^aeiou]ying$/.test(s) || /[aeiou]ing$/.test(s)) n += 1; // try-ing, be-ing, go-ing
  if (/[^aeiouy]ier$/.test(s) && n > 1) n += 1; // eas-i-er
  if (s.endsWith("e") && n > 1) {
    if (s.endsWith("le") && s.length > 2 && !"aeiouy".includes(s[s.length - 3])) {
      // ta-ble, sin-gle
    } else if (!"aeiouy".includes(s[s.length - 2])) {
      n -= 1; // silent e after a consonant: frame, these, whole
    }
  } else if (s.endsWith("es") && n > 1) {
    if (!(/(s|x|z|ch|sh|c|g)es$/.test(s) || /[^aeiouy]les$/.test(s))) n -= 1; // frames, games; not box-es
  } else if (s.endsWith("ed") && n > 1) {
    if (!/[td]ed$/.test(s)) n -= 1; // moved, trained; not want-ed
  }
  return Math.max(1, n);
}

/** The words of a piece of copy. A token or a placeholder is one word; a hyphenated word is one per part. */
export function wordsOf(text) {
  const words = [];
  const flat = text.replace(TOKEN_RE, " TOKEN ").replace(PLACEHOLDER_RE, " TOKEN ");
  for (const raw of flat.split(/\s+/)) {
    for (const part of raw.split("-")) {
      const core = part.replace(/[^A-Za-z0-9']/g, "");
      if (core) words.push(core);
    }
  }
  return words;
}

/** Sentences: runs of closing punctuation, plus a paragraph that ends without one. */
export function sentenceCount(text) {
  let total = 0;
  for (const block of text.trim().split(/\n\s*\n/)) {
    const paragraph = block.split(/\s+/).filter(Boolean).join(" ");
    if (!paragraph) continue;
    total += (paragraph.match(/[.!?]+(?=\s|$)/g) ?? []).length;
    if (!/[.!?]$/.test(paragraph)) total += 1;
  }
  return Math.max(1, total);
}

/** Flesch-Kincaid grade of a piece of copy; `tokenSyllables` is what one token counts as. */
export function fkGrade(text, tokenSyllables = 1) {
  const words = wordsOf(text);
  if (words.length === 0) return 0;
  const syllables = words.reduce((sum, word) => sum + (word === "TOKEN" ? tokenSyllables : countSyllables(word)), 0);
  return 0.39 * (words.length / sentenceCount(text)) + 11.8 * (syllables / words.length) - 15.59;
}

/** Python's round() to one decimal (half to even on the binary value), so a grade reads the same in both. */
export function round1(value) {
  return Math.round(value * 10) / 10;
}

// Words that belong in the evidence layer and the presenter notes, never in the on-screen copy of the cold
// open (docs/SCRIPT.md, "Words kept out of the copy"). The second item says whether the match is case-sensitive.
export const JARGON = [
  ["homograph", false], ["reprojection", false], ["bootstrap", false], ["Hungarian", false], ["RANSAC", false],
  ["YOLO", false], ["BoT-SORT", false], ["\\bCLIP\\b", true], ["embedding", false], ["confidence interval", false],
  ["\\bCI\\b", true], ["\\bPBP\\b", true], ["\\bxFG\\b", false], ["\\bEPV\\b", true], ["ghost", false],
  ["held[- ]out", false], ["out[- ]of[- ]fold", false], ["log loss", false], ["\\bMSE\\b", true], ["Brier", false],
  ["calibrat", false], ["percentile", false], ["\\bp50\\b", false], ["\\bp90\\b", false], ["\\bmedian\\b", false],
  ["ground truth", false],
].map(([pattern, cased]) => ({ pattern, re: new RegExp(pattern, cased ? "" : "i") }));

// Claims the copy never makes (PLAN.md C10, D16; CLAUDE.md, Words).
export const FORBIDDEN = [
  "\\bbest\\b", "\\boptimal", "\\bideal\\b", "\\bshould\\b", "no model (?:saw|has seen|ever saw)", "\\bpasses\\b",
  "\\bproves?\\b",
].map((pattern) => ({ pattern, re: new RegExp(pattern, "i") }));

/** The jargon and the forbidden claims in one line of copy, as messages. Tokens are left out of the scan. */
export function plainWordProblems(line, { allowGroundTruth = false } = {}) {
  const plain = line.replace(TOKEN_RE, " ");
  const problems = [];
  for (const { pattern, re } of JARGON) {
    if (allowGroundTruth && pattern === "ground truth") continue;
    if (re.test(plain)) problems.push(`jargon /${pattern}/`);
  }
  for (const { pattern, re } of FORBIDDEN) {
    if (re.test(plain)) problems.push(`forbidden claim /${pattern}/`);
  }
  return problems;
}
