// Reads the design tokens (app/styles/tokens.css) and does the WCAG contrast arithmetic.
import { readFileSync } from "node:fs";
import { TOKENS_FILE } from "./paths.mjs";

const DECLARATION = /(--[a-z0-9-]+)\s*:\s*([^;]+);/gi;

/** The text between the braces that follow `selector`, counting nested braces. */
function blockAfter(css, selectorIndex) {
  const open = css.indexOf("{", selectorIndex);
  if (open < 0) return null;
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === "{") depth += 1;
    if (css[index] === "}") {
      depth -= 1;
      if (depth === 0) return css.slice(open + 1, index);
    }
  }
  return null;
}

function declarations(block) {
  const out = new Map();
  for (const match of block.matchAll(DECLARATION)) out.set(match[1], match[2].trim());
  return out;
}

/**
 * The three token blocks of the stylesheet: the light theme, the dark theme for a system that asks for dark,
 * and the dark theme for data-theme="dark".
 */
export function parseTokenBlocks(css) {
  const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const lightAt = withoutComments.search(/(^|\n):root\s*\{/);
  const mediaAt = withoutComments.search(/@media\s*\(prefers-color-scheme:\s*dark\)/);
  const attrAt = withoutComments.search(/:root\[data-theme="dark"\]\s*\{/);
  if (lightAt < 0 || mediaAt < 0 || attrAt < 0) {
    throw new Error('tokens.css must hold :root, @media (prefers-color-scheme: dark) and :root[data-theme="dark"]');
  }
  const mediaBlock = blockAfter(withoutComments, mediaAt);
  const inner = mediaBlock.search(/:root:not\(\[data-theme="light"\]\)\s*\{/);
  if (inner < 0) throw new Error('the dark media block must hold :root:not([data-theme="light"])');
  return {
    light: declarations(blockAfter(withoutComments, lightAt)),
    darkMedia: declarations(blockAfter(mediaBlock, inner)),
    darkAttr: declarations(blockAfter(withoutComments, attrAt)),
  };
}

/** Follow var(--name) references until a plain value is left. */
export function resolve(tokens, name, seen = new Set()) {
  if (seen.has(name)) throw new Error(`token ${name} refers to itself`);
  seen.add(name);
  const raw = tokens.get(name);
  if (raw === undefined) throw new Error(`token ${name} is not defined`);
  const reference = raw.match(/^var\(\s*(--[a-z0-9-]+)\s*\)$/i);
  return reference ? resolve(tokens, reference[1], seen) : raw;
}

/** The token values of both themes. The dark theme is the light block with the dark block laid over it. */
export function themes(css = readFileSync(TOKENS_FILE, "utf8")) {
  const blocks = parseTokenBlocks(css);
  const differences = [];
  const names = new Set([...blocks.darkMedia.keys(), ...blocks.darkAttr.keys()]);
  for (const name of names) {
    if (blocks.darkMedia.get(name) !== blocks.darkAttr.get(name)) differences.push(name);
  }
  return {
    light: blocks.light,
    dark: new Map([...blocks.light, ...blocks.darkAttr]),
    darkBlocksDiffer: differences.sort(),
  };
}

/** #rgb or #rrggbb to [r, g, b] in 0..255. Anything else is an error: checked tokens are plain hex. */
export function parseHex(value) {
  const hex = value.trim().toLowerCase();
  const short = hex.match(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/);
  if (short) return short.slice(1).map((digit) => parseInt(digit + digit, 16));
  const long = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/);
  if (long) return long.slice(1).map((pair) => parseInt(pair, 16));
  throw new Error(`"${value}" is not a hex colour`);
}

/** WCAG 2 relative luminance of an sRGB colour. */
export function luminance([r, g, b]) {
  const linear = [r, g, b].map((channel) => {
    const s = channel / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

/** WCAG 2 contrast ratio of two hex colours, from 1 to 21. */
export function contrast(first, second) {
  const a = luminance(parseHex(first));
  const b = luminance(parseHex(second));
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}
