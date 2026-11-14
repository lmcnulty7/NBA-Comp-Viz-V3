/**
 * The tokens of docs/SCRIPT.md. A token such as {scorecard.adopted.heldout_game.position_error.p50_ft} names one
 * field of the site data; copy holds the token, never the number. This module resolves a token to the digits
 * the data file holds, changed only as the token table of docs/SCRIPT.md allows (a share moved two decimal
 * places to read as a percentage). It never multiplies and never rounds.
 *
 * TOKENS is the token table of docs/SCRIPT.md, field path for field path; a unit test holds the two equal. A
 * token that is not in the table is an error, so a page that uses one fails the build.
 */
import { MINUS } from "@/lib/format";
import type { DemoFrames, DemoSwap, Manifest, PlayersFile, Scorecard } from "@/lib/types/contract";

export type TokenFile = "scorecard.json" | "demo/swap.json" | "demo/frames.json" | "manifest.json" | "players.json";

/** "Shown as" of the token table: the only change the page may make to a stored value. */
export type ShownAs =
  | "share as a percentage"
  | "probability as percentage points"
  | "feet, as stored"
  | "points, as stored"
  | "count, as stored"
  | "as stored";

export interface TokenSpec {
  file: TokenFile;
  /** The field path of the token table: name[selector] segments joined by dots. */
  field: string;
  shownAs: ShownAs;
}

const card = (role: string, row: string, value: string): string =>
  `builds[role=${role}].rows[key=${row}].cells[split=heldout_game].values[${value}]`;

export const TOKENS: Readonly<Record<string, TokenSpec>> = {
  "scorecard.adopted.heldout_game.missed_players.rate": {
    file: "scorecard.json",
    field: card("adopted", "missed_players", "rate"),
    shownAs: "share as a percentage",
  },
  "scorecard.adopted.heldout_game.ghost_boxes.rate": {
    file: "scorecard.json",
    field: card("adopted", "ghost_boxes", "rate"),
    shownAs: "share as a percentage",
  },
  "scorecard.adopted.heldout_game.team_labels.accuracy": {
    file: "scorecard.json",
    field: card("adopted", "team_labels", "accuracy"),
    shownAs: "share as a percentage",
  },
  "scorecard.adopted.heldout_game.identity.id_switches_per_possession": {
    file: "scorecard.json",
    field: card("adopted", "identity", "id_switches_per_possession"),
    shownAs: "as stored",
  },
  "scorecard.adopted.heldout_game.position_error.p50_ft": {
    file: "scorecard.json",
    field: card("adopted", "position_error", "p50_ft"),
    shownAs: "feet, as stored",
  },
  "scorecard.adopted.heldout_game.position_error.p90_ft": {
    file: "scorecard.json",
    field: card("adopted", "position_error", "p90_ft"),
    shownAs: "feet, as stored",
  },
  "scorecard.adopted.heldout_game.near_field_bias.far_third_ft": {
    file: "scorecard.json",
    field: card("adopted", "near_field_bias", "far_third_ft"),
    shownAs: "feet, as stored",
  },
  "scorecard.candidate.heldout_game.position_error.p50_ft": {
    file: "scorecard.json",
    field: card("candidate", "position_error", "p50_ft"),
    shownAs: "feet, as stored",
  },
  "scorecard.adopted.heldout_game.testable": {
    file: "scorecard.json",
    field: "builds[role=adopted].games[split=heldout_game].testable",
    shownAs: "count, as stored",
  },
  "scorecard.adopted.heldout_game.truth_frames": {
    file: "scorecard.json",
    field: "builds[role=adopted].games[split=heldout_game].truth_frames",
    shownAs: "count, as stored",
  },
  "scorecard.adopted.heldout_game.untestable_share": {
    file: "scorecard.json",
    field: "builds[role=adopted].games[split=heldout_game].untestable_share",
    shownAs: "share as a percentage",
  },
  "frames.scene.p50_error_ft": { file: "demo/frames.json", field: "scenes[0].p50_error_ft", shownAs: "feet, as stored" },
  "frames.scene.n_matched": { file: "demo/frames.json", field: "scenes[0].n_matched", shownAs: "count, as stored" },
  "frames.scene.frame.error_ft": {
    file: "demo/frames.json",
    field: "scenes[0].frames[k].error_ft",
    shownAs: "feet, as stored",
  },
  "swap.n_windows": { file: "demo/swap.json", field: "n_windows", shownAs: "count, as stored" },
  "swap.fill_share": { file: "demo/swap.json", field: "fill_share", shownAs: "share as a percentage" },
  "swap.xfg.mean_abs_diff": {
    file: "demo/swap.json",
    field: "summaries[kind=xfg].mean_abs_diff",
    shownAs: "probability as percentage points",
  },
  "swap.xfg.n": { file: "demo/swap.json", field: "summaries[kind=xfg].n", shownAs: "count, as stored" },
  "swap.epv.mean_abs_diff": {
    file: "demo/swap.json",
    field: "summaries[kind=epv].mean_abs_diff",
    shownAs: "points, as stored",
  },
  "swap.epv.n": { file: "demo/swap.json", field: "summaries[kind=epv].n", shownAs: "count, as stored" },
  "swap.ghosting.mean_abs_diff": {
    file: "demo/swap.json",
    field: "summaries[kind=ghosting].mean_abs_diff",
    shownAs: "feet, as stored",
  },
  "swap.ghosting.n": { file: "demo/swap.json", field: "summaries[kind=ghosting].n", shownAs: "count, as stored" },
  "manifest.n_possessions": { file: "manifest.json", field: "n_possessions", shownAs: "count, as stored" },
  "manifest.n_games": { file: "manifest.json", field: "n_games", shownAs: "count, as stored" },
  "players.n_shown": { file: "players.json", field: "n_shown", shownAs: "count, as stored" },
  "players.n_hidden": { file: "players.json", field: "n_hidden", shownAs: "count, as stored" },
};

/** A token in copy: {file.part.part}, lower case, at least one dot. A {letters} placeholder is not a token. */
export const TOKEN_PATTERN = /\{([a-z][a-z_]*(?:\.[a-z0-9_]+)+)\}/g;

/** The site data a token can name. A demo file is null when the data build has none. */
export interface TokenData {
  manifest: Manifest;
  scorecard: Scorecard;
  players: PlayersFile;
  swap: DemoSwap | null;
  frames: DemoFrames | null;
}

/** What "the frame on screen" is, for a token whose field path selects [k]. */
export interface TokenContext {
  frame?: number;
}

export class TokenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TokenError";
  }
}

/** Every token of a piece of copy, in order. */
export function tokensIn(template: string): string[] {
  return [...template.matchAll(TOKEN_PATTERN)].map((match) => match[1] as string);
}

function documentOf(spec: TokenSpec, data: Partial<TokenData>): unknown {
  switch (spec.file) {
    case "scorecard.json":
      return data.scorecard;
    case "demo/swap.json":
      return data.swap;
    case "demo/frames.json":
      return data.frames;
    case "manifest.json":
      return data.manifest;
    case "players.json":
      return data.players;
  }
}

const SEGMENT = /^([a-z_0-9]+)(?:\[([^\]]+)\])?$/;

/**
 * Walk a field path of the token table through one data document and return what it selects. A selector in
 * brackets picks a list item by the value of one of its fields (it must pick exactly one), by position, or by k
 * (the frame on screen); on a dictionary it picks a key. Returns undefined when an optional field is absent
 * or null; throws when the path does not fit the data, so a wrong row key or value name fails.
 */
export function walkField(document: unknown, field: string, context: TokenContext = {}): unknown {
  let current: unknown = document;
  for (const segment of field.split(".")) {
    const parsed = SEGMENT.exec(segment);
    if (!parsed) throw new TokenError(`${field}: cannot read ${JSON.stringify(segment)}`);
    const name = parsed[1] as string;
    const selector = parsed[2];
    if (typeof current !== "object" || current === null || Array.isArray(current)) {
      throw new TokenError(`${field}: nothing to read ${name} from`);
    }
    if (!(name in current)) {
      // A field the contract makes optional may be left out of a document.
      if (selector === undefined) return undefined;
      throw new TokenError(`${field}: the data has no ${name}`);
    }
    current = (current as Record<string, unknown>)[name];
    if (selector === undefined) continue;
    if (Array.isArray(current) && selector.includes("=")) {
      const [key, value] = selector.split("=", 2) as [string, string];
      const found = current.filter(
        (item) => typeof item === "object" && item !== null && String((item as Record<string, unknown>)[key]) === value,
      );
      if (found.length !== 1) throw new TokenError(`${field}: [${selector}] selects ${found.length} items, not one`);
      current = found[0];
    } else if (Array.isArray(current)) {
      const index = selector === "k" ? context.frame : Number(selector);
      if (index === undefined || !Number.isInteger(index) || index < 0) {
        throw new TokenError(`${field}: [${selector}] needs a frame on screen`);
      }
      if (index >= current.length) throw new TokenError(`${field}: no item [${index}]`);
      current = current[index];
    } else if (typeof current === "object" && current !== null) {
      if (!(selector in current)) throw new TokenError(`${field}: no key ${selector}`);
      current = (current as Record<string, unknown>)[selector];
    } else {
      throw new TokenError(`${field}: ${name} is neither a list nor a dictionary`);
    }
  }
  return current === null ? undefined : current;
}

/**
 * The digits of a stored number as plain decimal text, the way the data file writes them: 0.251, 1.84, 1198.
 * JSON keeps no trailing zero (1.50 is read as 1.5), and an exponent is written out.
 */
export function storedDigits(value: number): string {
  if (!Number.isFinite(value)) throw new TokenError(`${value} is not a stored number`);
  const text = String(Math.abs(value));
  const sign = value < 0 ? "-" : "";
  const exponent = /^(\d+)(?:\.(\d+))?e([+-]\d+)$/.exec(text);
  if (!exponent) return sign + text;
  const whole = exponent[1] as string;
  const fraction = exponent[2] ?? "";
  return sign + shiftPoint(`${whole}.${fraction}`, Number(exponent[3]));
}

/** Move the decimal point of plain decimal text by `places` (right when positive). Digits are never changed. */
export function shiftPoint(text: string, places: number): string {
  const negative = text.startsWith("-");
  const body = negative ? text.slice(1) : text;
  const [whole = "", fraction = ""] = body.split(".");
  let digits = whole + fraction;
  let point = whole.length + places;
  if (point < 0) {
    digits = "0".repeat(-point) + digits;
    point = 0;
  }
  if (point > digits.length) digits += "0".repeat(point - digits.length);
  let left = digits.slice(0, point).replace(/^0+(?=\d)/, "");
  const right = digits.slice(point).replace(/0+$/, "");
  if (left === "") left = "0";
  const out = right ? `${left}.${right}` : left;
  return negative && /[1-9]/.test(out) ? `-${out}` : out;
}

function withMinus(text: string): string {
  return text.startsWith("-") ? `${MINUS}${text.slice(1)}` : text;
}

function withSeparators(text: string): string {
  const [whole = "", fraction] = text.split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return fraction === undefined ? grouped : `${grouped}.${fraction}`;
}

/** A stored value as the page shows it, by the "Shown as" rule of its token. */
export function showValue(value: number, shownAs: ShownAs): string {
  const digits = storedDigits(value);
  switch (shownAs) {
    case "share as a percentage":
    case "probability as percentage points":
      return withMinus(shiftPoint(digits, 2));
    case "count, as stored":
      return withMinus(withSeparators(digits));
    case "feet, as stored":
    case "points, as stored":
    case "as stored":
      return withMinus(digits);
  }
}

export function tokenSpec(name: string): TokenSpec {
  const spec = TOKENS[name];
  if (!spec) throw new TokenError(`{${name}} is not a token of docs/SCRIPT.md`);
  return spec;
}

/**
 * The value of a token as text, or null when the data holds no value there (a null field, or a demo file this
 * data build lacks). An unknown token, or a path that does not fit the data, throws.
 */
export function resolveToken(name: string, data: Partial<TokenData>, context: TokenContext = {}): string | null {
  const spec = tokenSpec(name);
  const document = documentOf(spec, data);
  if (document === null || document === undefined) {
    if (spec.file === "demo/swap.json" || spec.file === "demo/frames.json") return null;
    throw new TokenError(`{${name}}: the data has no ${spec.file}`);
  }
  // A model of the demo comparison whose rebuild did not pass its check has no number to show (C6).
  const summary = /^summaries\[kind=([a-z]+)\]\./.exec(spec.field);
  if (summary && walkField(document, `summaries[kind=${summary[1]}].available`) !== true) return null;
  const value = walkField(document, spec.field, context);
  if (value === undefined) return null;
  if (typeof value !== "number") throw new TokenError(`{${name}}: ${spec.field} holds ${JSON.stringify(value)}, not a number`);
  return showValue(value, spec.shownAs);
}

/** True when every token of the copy has a value in this data build. Unknown tokens still throw. */
export function canRender(template: string, data: Partial<TokenData>, context: TokenContext = {}): boolean {
  return tokensIn(template).every((name) => resolveToken(name, data, context) !== null);
}

/**
 * A line of copy with its tokens filled, or its fallback line when a token of the first has no value, or null
 * when neither can be shown. Unknown tokens throw either way, so a typo fails the build.
 */
export function renderLine(
  template: string,
  fallback: string | null | undefined,
  data: Partial<TokenData>,
  context: TokenContext = {},
): string | null {
  if (canRender(template, data, context)) return renderTokens(template, data, context);
  if (fallback) {
    tokensIn(template).forEach((name) => tokenSpec(name));
    return canRender(fallback, data, context) ? renderTokens(fallback, data, context) : null;
  }
  return null;
}

/** Copy with every token replaced by its value. A token with no value is an error: use canRender first. */
export function renderTokens(template: string, data: Partial<TokenData>, context: TokenContext = {}): string {
  return template.replace(TOKEN_PATTERN, (_match, name: string) => {
    const value = resolveToken(name, data, context);
    if (value === null) throw new TokenError(`{${name}} has no value in this data build`);
    return value;
  });
}
