/**
 * Machine-written texts (PLAN.md P3, D76): the page's side of the number check.
 *
 * export/genai.py writes each text from the numbers of one page and drops any text with a number that is not on
 * that page. A page checks the text again before it shows it: it builds the same facts from its own data with
 * its own formatters (lib/format.ts), so "after the page's own rounding" is literal, and a text with any number
 * the page does not show, in another unit, a value without its interval, a point estimate not called one, a model's
 * value not called the model's in its sentence, or a lane that did not work, is not shown at all. The page then shows nothing rather than a wrong number.
 *
 * The facts and the tokenizer mirror export/genai.py. tests/fixtures/genai_cases.json holds the cases both must
 * agree on, and a unit test runs this check on every text of the data root.
 */
import { MINUS, formatClock, formatCount, formatNumber, formatPercent, formatProbability, formatSeconds, formatSigned } from "@/lib/format";
import { showValue } from "@/lib/tokens";
import type {
  Estimate,
  GeneratedText,
  GeneratedTexts,
  PlayerView,
  Possession,
  PossessionSummary,
} from "@/lib/types/contract";

export type FactUnit = "percent" | "feet" | "points" | "seconds" | "clock" | "none";
export type FactKind = "value" | "lo" | "hi" | "point" | "fact";

/** A number the page shows: the text it prints, its unit, and what kind of estimate it is. */
export interface Fact {
  field: string;
  shown: string;
  unit: FactUnit;
  kind: FactKind;
  group: string;
}

const fact = (field: string, shown: string, unit: FactUnit, kind: FactKind, group = ""): Fact => ({
  field,
  shown,
  unit,
  kind,
  group,
});

function estimateFacts(field: string, estimate: Estimate | null | undefined, digits: number, unit: FactUnit): Fact[] {
  if (!estimate) return [];
  const out: Fact[] = [];
  if (estimate.point_estimate) {
    out.push(fact(field, formatNumber(estimate.value, digits), unit, "point", field));
  } else {
    out.push(
      fact(`${field}.value`, formatNumber(estimate.value, digits), unit, "value", field),
      fact(`${field}.lo`, formatNumber(estimate.lo, digits), unit, "lo", field),
      fact(`${field}.hi`, formatNumber(estimate.hi, digits), unit, "hi", field),
    );
    const level = /^(\d+)% /.exec(estimate.interval ?? "");
    if (level) out.push(fact(`${field}.interval`, `${level[1]}%`, "percent", "fact"));
  }
  if (estimate.n != null) out.push(fact(`${field}.n`, formatCount(estimate.n), "none", "fact"));
  return out;
}

/** The numbers the player page shows, as it shows them (export/genai.py, player_input). */
export function playerFacts(player: PlayerView): Fact[] {
  const facts: Fact[] = [fact("games", formatCount(player.games), "none", "fact")];
  const { shot_making: shots, gravity, rebounding, absence_risk: absence, similar } = player;
  if (shots.shown && shots.over_expected_per_100) {
    facts.push(...estimateFacts("shot_making.over_expected_per_100", shots.over_expected_per_100, 1, "none"));
    if (shots.baseline_error_per_100 != null) {
      const field = "shot_making.baseline_error_per_100";
      facts.push(fact(field, formatSigned(shots.baseline_error_per_100, 1), "none", "point", field));
    }
    if (shots.fg_pct != null) {
      facts.push(fact("shot_making.fg_pct", formatPercent(shots.fg_pct, 1), "percent", "point", "shot_making.fg_pct"));
    }
    if (shots.expected_fg_pct != null) {
      const field = "shot_making.expected_fg_pct";
      facts.push(fact(field, formatPercent(shots.expected_fg_pct, 1), "percent", "point", field));
    }
  }
  facts.push(
    fact("shot_making.attempts", formatCount(shots.attempts), "none", "fact"),
    fact("shot_making.min_attempts", formatCount(shots.min_attempts), "none", "fact"),
  );
  if (gravity.shown && gravity.gravity_ft) facts.push(...estimateFacts("gravity.gravity_ft", gravity.gravity_ft, 2, "feet"));
  facts.push(
    fact("gravity.possessions", formatCount(gravity.possessions), "none", "fact"),
    fact("gravity.min_possessions", formatCount(gravity.min_possessions), "none", "fact"),
  );
  if (rebounding.shown && rebounding.rate_per_100) {
    facts.push(...estimateFacts("rebounding.rate_per_100", rebounding.rate_per_100, 1, "none"));
    if (rebounding.model_expected_per_100 != null) {
      const field = "rebounding.model_expected_per_100";
      facts.push(fact(field, formatNumber(rebounding.model_expected_per_100, 1), "none", "point", field));
    }
  }
  facts.push(
    fact("rebounding.chances", formatCount(rebounding.chances), "none", "fact"),
    fact("rebounding.min_chances", formatCount(rebounding.min_chances), "none", "fact"),
  );
  if (absence.shown) {
    facts.push(...estimateFacts("absence_risk.model_probability", absence.model_probability, 3, "none"));
    facts.push(...estimateFacts("absence_risk.observed_rate", absence.observed_rate, 3, "none"));
    facts.push(fact("absence_risk.absent", formatCount(absence.absent), "none", "fact"));
  }
  facts.push(
    fact("absence_risk.games", formatCount(absence.games), "none", "fact"),
    fact("absence_risk.min_games", formatCount(absence.min_games), "none", "fact"),
  );
  if (similar.available && similar.shown) {
    for (const row of similar.neighbours) {
      const field = `similar.neighbours.${row.rank}`;
      facts.push(
        fact(`${field}.rank`, String(row.rank), "none", "fact"),
        fact(`${field}.distance`, formatNumber(row.distance, 3), "none", "point", `${field}.distance`),
      );
    }
  }
  facts.push(
    fact("similar.possessions", formatCount(similar.possessions), "none", "fact"),
    fact("similar.min_possessions", formatCount(similar.min_possessions), "none", "fact"),
  );
  return facts;
}

/** The names of a player page, whose digits are not numbers of a text. */
export function playerNames(player: PlayerView): string[] {
  const names = [player.name, ...player.teams];
  if (player.similar.available && player.similar.shown) names.push(...player.similar.neighbours.map((row) => row.name));
  return names;
}

/** The per-possession numbers the review page shows, as it shows them (export/genai.py, possession_input). */
export function possessionFacts(summary: PossessionSummary, detail: Possession): Fact[] {
  const facts: Fact[] = [
    fact("possession", String(summary.possession), "none", "fact"),
    fact("period", String(summary.period), "none", "fact"),
    fact("start_clock", formatClock(summary.start_clock_s), "clock", "fact"),
    fact("points", String(summary.points), "points", "fact"),
    fact("duration_s", formatSeconds(summary.duration_s), "seconds", "fact"),
  ];
  if (summary.xfg != null) facts.push(fact("xfg", formatProbability(summary.xfg), "none", "point", "xfg"));
  if (summary.epv_start != null) {
    facts.push(fact("epv_start", formatNumber(summary.epv_start, 2), "points", "point", "epv_start"));
  }
  if (summary.epv_end != null) facts.push(fact("epv_end", formatNumber(summary.epv_end, 2), "points", "point", "epv_end"));
  const curve = detail.epv;
  if (curve && curve.value.length > 0) {
    const n = Math.min(curve.t.length, curve.value.length);
    for (let k = 0; k < n; k += 1) {
      facts.push(
        fact(`epv.t.${k}`, formatSeconds(curve.t[k]), "seconds", "fact"),
        fact(`epv.value.${k}`, showValue(curve.value[k] as number, "points, as stored"), "points", "point", `epv.value.${k}`),
      );
    }
  }
  detail.shots.forEach((shot, j) => {
    if (shot.t != null) facts.push(fact(`shots.${j}.t`, formatSeconds(shot.t), "seconds", "fact"));
    if (shot.xfg != null) facts.push(fact(`shots.${j}.xfg`, formatProbability(shot.xfg), "none", "point", `shots.${j}.xfg`));
  });
  return facts;
}

/** The names of a possession page, whose digits are not numbers of a text. */
export function possessionNames(detail: Possession): string[] {
  const names = [detail.offense, detail.defense];
  for (const shot of detail.shots) if (shot.shooter) names.push(shot.shooter);
  return names;
}

// ------------------------------------------------------------------------------------------------ the check

export interface Token {
  raw: string;
  canon: string;
  unit: FactUnit;
  start: number;
}

const CLOCK_RE = /\d{1,2}:\d{2}(?!\d)/g;
const NUMBER_RE = /\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|\d+(?:\.\d+)?/g;
const PERCENT_AFTER = /^\s*(?:%|percent\b|per cent\b)/i;
const UNIT_AFTER: [FactUnit, RegExp][] = [
  ["feet", /^\s*(?:feet|foot|ft)\b/i],
  ["points", /^\s*(?:points?|pts)\b/i],
  ["seconds", /^\s*(?:seconds?|secs?|s)\b/i],
];
const SCALE_RE = /\bper\s+(?:100|hundred)\b/gi;
const KIND_OF_SHOT = /\b(?:three|two)[- ]point(?:ers?)?\b/gi;
const NUMBER_WORDS = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
  "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
  "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "hundreds", "thousand", "thousands", "million",
  "half", "halves", "twice", "double", "doubled", "triple", "tripled", "dozen", "quarter", "quarters", "once",
  "first", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
];
const NUMBER_WORD_RE = new RegExp(`\\b(?:${NUMBER_WORDS.join("|")})\\b`, "gi");
const SECOND_RE = /(?<!\d )\bsecond\b/gi;
const POINT_ESTIMATE_RE = /\bpoint estimates?\b/i;
// Values that are a model's output (export/genai.py, MODEL_GROUP_RE): a text gives one only with a word that says
// so in the same sentence, so a model's probability never reads as an observed rate.
const MODEL_GROUP_RE =
  /^(?:absence_risk\.model_probability|shot_making\.(?:expected_fg_pct|baseline_error_per_100|over_expected_per_100)|rebounding\.model_expected_per_100|xfg|shots\.\d+\.xfg|epv_start|epv_end|epv\.value\.\d+)$/;
const MODEL_WORD_RE = /\b(?:models?|model's|expect(?:ed|s)?|predict(?:ed|s|ion|ions)?|estimat(?:e|ed|es))\b/i;
const SENTENCE_END_RE = /[.!?]+(?=\s|$)/g;
// Lanes that are not on the prep tab or the review and did not work (export/genai.py, LANE_WORDS).
const LANE_WORDS: [string, RegExp][] = [
  ["decisions", /\b(?:decisions?)\b/gi],
  ["matchups", /\b(?:match-?ups?)\b/gi],
  ["lineups", /\b(?:line-?ups?)\b/gi],
  ["actions", /\b(?:pass|passes|passed|passing|drives?|drove|driving|screens?)\b/gi],
  ["coverage", /\b(?:coverages?)\b/gi],
  ["playcalling", /\b(?:play ?calls?|play-?calling|playcalling)\b/gi],
  ["betting", /\b(?:betting|bets?|odds)\b/gi],
];

/** [start, stop) of the sentence of a text that holds the character at `index`, split as the reading grade splits. */
function sentenceSpan(text: string, index: number): [number, number] {
  const ends = [...text.matchAll(SENTENCE_END_RE)].map((match) => (match.index ?? 0) + match[0].length);
  const start = Math.max(0, ...ends.filter((end) => end <= index));
  const stop = Math.min(text.length, ...ends.filter((end) => end > index));
  return [start, stop];
}

/** The sentence of a text that holds the character at `index`. */
function sentenceAt(text: string, index: number): string {
  const [start, stop] = sentenceSpan(text, index);
  return text.slice(start, stop);
}

// Words that say a value went up or down (export/genai.py, UP_WORDS and DOWN_WORDS).
const UP_WORDS =
  "rose|rise|rises|rising|climbed|climb|climbs|climbing|grew|grow|grows|growing|increased|increases|increase|jumped|jump|jumps|jumping|went up|goes up|go up|going up|moved up|ticked up";
const DOWN_WORDS =
  "fell|fall|falls|falling|dropped|drop|drops|dropping|dipped|dip|dips|dipping|sank|sink|sinks|sinking|went down|goes down|go down|going down|slid|slide|slides|sliding|declined|decline|declines|decreased|decrease|decreases|slipped|slip|slips|moved down|ticked down";
const CHANGE_RE = new RegExp(`\\b(?:(?<up>${UP_WORDS})|(?<down>${DOWN_WORDS}))\\b`, "gi");
const FROM_BEFORE_RE = /\bfrom\s*$/i;

/** The run of values a fact belongs to, for the direction check (export/genai.py, series_of). */
export function seriesOf(field: string, unit: FactUnit): string | null {
  if (unit === "seconds" || unit === "clock") return null;
  if (/^(?:epv_start|epv_end|epv\.value\.\d+)$/.test(field)) return "epv";
  if (/^(?:xfg|shots\.\d+\.xfg)$/.test(field)) return "xfg";
  const estimate = /^(.+)\.(?:value|lo|hi)$/.exec(field);
  if (estimate) return estimate[1] as string;
  if (/\.(?:n|interval)$/.test(field) || /(?:attempts|possessions|chances|games|absent|rank)$/.test(field)) return null;
  return field;
}

/** A canonical value as a whole number of its last decimal place, and that count of decimals. */
function scaled(canon: string): [bigint, number] {
  const plain = canonical(canon).replace(/%$/, "");
  const negative = plain.startsWith("-");
  const [whole = "0", fraction = ""] = plain.replace(/^-/, "").split(".");
  const value = BigInt(`${whole}${fraction}` || "0");
  return [negative ? -value : value, fraction.length];
}

// A value the page prints as the file stores it (an EPV curve state) is exact: "1.1" is 1.100, not a rounding.
const EXACT_FIELD_RE = /^epv\.value\.\d+$/;

/** Half-up rounding of a scaled value from `from` decimals to `to` decimals. */
function roundScaled(value: bigint, from: number, to: number): bigint {
  if (from <= to) return value * BigInt(10) ** BigInt(to - from);
  const zero = BigInt(0);
  const factor = BigInt(10) ** BigInt(from - to);
  const magnitude = value < zero ? -value : value;
  const rounded = (magnitude + factor / BigInt(2)) / factor;
  return value < zero ? -rounded : rounded;
}

/**
 * Whether two shown values can be one value (export/genai.py, same_at_shown_rounding): a rounded value matches an
 * exact one when the exact one rounded to its decimals gives it; two rounded values match at the coarser rounding;
 * two exact values only when equal.
 */
export function sameAtShownRounding(a: string, b: string, aExact = false, bExact = false): boolean {
  const [x, dx] = scaled(a);
  const [y, dy] = scaled(b);
  if (aExact && bExact) {
    const digits = Math.max(dx, dy);
    return roundScaled(x, dx, digits) === roundScaled(y, dy, digits);
  }
  if (aExact || bExact) {
    const [exact, ex, rounded, rx] = aExact ? [x, dx, y, dy] : [y, dy, x, dx];
    return roundScaled(exact, ex, rx) === rounded;
  }
  const digits = Math.min(dx, dy);
  return roundScaled(x, dx, digits) === roundScaled(y, dy, digits);
}

/**
 * A word of change (rose, fell, dipped, ...) must agree with the values it links (export/genai.py,
 * check_directions): the first number of a run after it in its sentence, against the number after "from" in that
 * sentence, else the nearest earlier number of the same run. One value at two roundings is no change.
 */
export function checkDirections(text: string, facts: readonly Fact[], names: readonly string[] = []): string[] {
  const masked = maskNames(text, names);
  const runs = numbersIn(text, names).map((token) => {
    const found = new Set(
      facts
        .filter((f) => canonical(f.shown) === token.canon && unitFits(token.unit, f.unit))
        .map((f) => seriesOf(f.field, f.unit)),
    );
    const fields = facts
      .filter((f) => canonical(f.shown) === token.canon && unitFits(token.unit, f.unit))
      .map((f) => f.field);
    const exact = fields.length > 0 && fields.every((field) => EXACT_FIELD_RE.test(field));
    return { token, run: found.size === 1 ? ([...found][0] ?? null) : null, exact };
  });
  const problems: string[] = [];
  for (const word of masked.matchAll(CHANGE_RE)) {
    const at = word.index ?? 0;
    const after = at + word[0].length;
    const [, end] = sentenceSpan(masked, at);
    const ahead = runs.filter(({ token, run }) => run && token.start >= after && token.start < end);
    if (ahead.length === 0) continue;
    let targetEntry = ahead[0]!;
    const run = targetEntry.run;
    let sourceEntry: (typeof runs)[number] | null = null;
    if (FROM_BEFORE_RE.test(masked.slice(after, targetEntry.token.start))) {
      const later = ahead.slice(1).find((entry) => entry.run === run);
      if (!later) continue;
      sourceEntry = targetEntry;
      targetEntry = later;
    } else {
      const before = runs.filter((entry) => entry.token.start < at && entry.run === run);
      sourceEntry = before.length > 0 ? before[before.length - 1]! : null;
    }
    if (!sourceEntry) continue;
    const source = sourceEntry.token;
    const target = targetEntry.token;
    const said = word.groups?.up ? "up" : "down";
    if (sameAtShownRounding(source.canon, target.canon, sourceEntry.exact, targetEntry.exact)) {
      problems.push(`"${word[0]}" links "${source.raw}" and "${target.raw}", which are one value at two roundings: no change`);
      continue;
    }
    const went = Number(target.canon.replace(/%$/, "")) > Number(source.canon.replace(/%$/, "")) ? "up" : "down";
    if (went !== said) problems.push(`"${word[0]}" says ${said}, but "${source.raw}" to "${target.raw}" went ${went}`);
  }
  return problems;
}

/** A text never names a lane that did not work and is not on its page. */
export function checkLanes(text: string, names: readonly string[] = []): string[] {
  const words = maskNames(text, names);
  const problems: string[] = [];
  for (const [lane, pattern] of LANE_WORDS) {
    for (const match of words.matchAll(pattern)) {
      problems.push(`names the ${lane} lane ("${match[0]}"), which did not work and is not on this page`);
    }
  }
  return problems;
}
const OTHER_NUMERAL = /\p{N}/gu;

const blank = (match: string) => " ".repeat(match.length);

/** A shown value as a comparable string: the minus sign as "-", no plus sign, no thousands separator. */
export function canonical(shown: string): string {
  const text = shown.replaceAll(MINUS, "-").replaceAll(",", "");
  return text.startsWith("+") ? text.slice(1) : text;
}

function maskNames(text: string, names: readonly string[]): string {
  let out = text;
  for (const name of [...new Set(names.filter(Boolean))].sort((a, b) => b.length - a.length)) {
    out = out.split(name).join(" ".repeat(name.length));
  }
  return out;
}

/** Every number of a text, in order. Every digit of the text belongs to one. */
export function numbersIn(text: string, names: readonly string[] = []): Token[] {
  const plain = maskNames(text, names).replace(SCALE_RE, blank);
  const tokens: Token[] = [];
  const taken = new Array<boolean>(plain.length).fill(false);
  for (const match of plain.matchAll(CLOCK_RE)) {
    const start = match.index ?? 0;
    tokens.push({ raw: match[0], canon: match[0], unit: "clock", start });
    for (let i = start; i < start + match[0].length; i += 1) taken[i] = true;
  }
  for (const match of plain.matchAll(NUMBER_RE)) {
    const start = match.index ?? 0;
    if (taken[start]) continue;
    let raw = match[0];
    let sign = "";
    const before = start >= 1 ? plain[start - 1] : "";
    if (before === "-" || before === MINUS || before === "+") {
      const edge = start >= 2 ? (plain[start - 2] as string) : " ";
      if (!/\d/.test(edge)) {
        sign = before === "+" ? "" : "-";
        raw = before + raw;
      }
    }
    let canon = sign + match[0].replaceAll(",", "");
    const rest = plain.slice(start + match[0].length);
    let unit: FactUnit = "none";
    const percent = PERCENT_AFTER.exec(rest);
    if (percent) {
      unit = "percent";
      canon += "%";
      raw += percent[0];
    } else {
      for (const [name, pattern] of UNIT_AFTER) {
        if (pattern.test(rest)) {
          unit = name;
          break;
        }
      }
    }
    tokens.push({ raw, canon, unit, start });
  }
  return tokens.sort((a, b) => a.start - b.start);
}

function unitFits(tokenUnit: FactUnit, factUnit: FactUnit): boolean {
  if (tokenUnit === "percent" || tokenUnit === "clock" || factUnit === "percent" || factUnit === "clock") {
    return tokenUnit === factUnit;
  }
  return tokenUnit === "none" || tokenUnit === factUnit;
}

/**
 * The problems with the numbers of a text: every number must be a value the page shows, in a fitting unit; a value
 * with an interval needs both bounds in the text; a point estimate needs the words "point estimate"; a number
 * written as a word is refused. An empty list means the text may be shown.
 */
export function checkNumbers(text: string, facts: readonly Fact[], names: readonly string[] = []): string[] {
  const problems: string[] = [];
  const tokens = numbersIn(text, names);
  const present = new Set(tokens.map((token) => token.canon));
  const saysPoint = POINT_ESTIMATE_RE.test(text);
  const bounds = new Map<string, { lo?: string; hi?: string }>();
  for (const f of facts) {
    if (f.kind === "lo" || f.kind === "hi") {
      const pair = bounds.get(f.group) ?? {};
      pair[f.kind] = canonical(f.shown);
      bounds.set(f.group, pair);
    }
  }
  const satisfied = (f: Fact) => {
    if (f.kind === "value") {
      const pair = bounds.get(f.group) ?? {};
      return pair.lo !== undefined && pair.hi !== undefined && present.has(pair.lo) && present.has(pair.hi);
    }
    if (f.kind === "point") return saysPoint;
    return true;
  };
  for (const token of tokens) {
    const candidates = facts.filter((f) => canonical(f.shown) === token.canon && unitFits(token.unit, f.unit));
    if (candidates.length === 0) {
      const near = facts.some((f) => canonical(f.shown).replace(/%$/, "") === token.canon.replace(/%$/, ""));
      problems.push(`number "${token.raw}" ${near ? "is a value of the page in another unit" : "is not a value of this page as the page shows it"}`);
      continue;
    }
    const good = candidates.filter(satisfied);
    if (good.length === 0) {
      const kind = candidates[0]?.kind;
      problems.push(
        `number "${token.raw}" is ${kind === "value" ? "a value given without its interval" : "a point estimate the text does not call one"}`,
      );
      continue;
    }
    if (good.every((f) => MODEL_GROUP_RE.test(f.group)) && !MODEL_WORD_RE.test(sentenceAt(text, token.start))) {
      problems.push(`number "${token.raw}" is a model's value, and its sentence does not say so (model, expected, predicted or estimate)`);
    }
  }
  const words = maskNames(text, names).replace(KIND_OF_SHOT, " ");
  for (const match of words.matchAll(NUMBER_WORD_RE)) problems.push(`the number word "${match[0]}": numbers are written as digits`);
  for (const match of words.matchAll(SECOND_RE)) problems.push(`the ordinal "${match[0]}": numbers are written as digits`);
  for (const match of words.matchAll(OTHER_NUMERAL)) {
    if (!/[0-9]/.test(match[0])) problems.push(`the character "${match[0]}": numbers are written with the digits 0 to 9`);
  }
  return problems;
}

/** The text of one page's item, or null when there is none or it does not pass the page's own check. */
export function checkedText(
  texts: GeneratedTexts | null,
  key: string,
  facts: readonly Fact[],
  names: readonly string[],
): GeneratedText | null {
  const item = texts?.items.find((candidate) => candidate.key === key) ?? null;
  if (!item) return null;
  const problems = [...checkNumbers(item.text, facts, names), ...checkDirections(item.text, facts, names), ...checkLanes(item.text, names)];
  return problems.length === 0 ? item : null;
}
