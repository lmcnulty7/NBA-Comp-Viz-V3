/**
 * Turning data values into text. Every number the site shows comes from the data root and passes through
 * here; no number is typed into copy (web/content) or into a component.
 */
/** The true minus sign (U+2212), so a negative number does not start with a hyphen. */
export const MINUS = String.fromCodePoint(0x2212);

function fixed(value: number, digits: number): string {
  const text = Math.abs(value).toFixed(digits);
  const isZero = Number(text) === 0;
  return value < 0 && !isZero ? `${MINUS}${text}` : text;
}

/**
 * A number a data file stores as text (a value copied from a report, so its trailing zeros stay), shown as
 * stored with the true minus sign in place of a leading hyphen. Nothing else changes.
 */
export function numberText(text: string): string {
  return text.startsWith("-") ? `${MINUS}${text.slice(1)}` : text;
}

/** A number with a fixed count of decimals. Null and undefined become an empty string. */
export function formatNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  return fixed(value, digits);
}

/** A whole number with thousands separators. */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  const text = Math.round(Math.abs(value))
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return value < 0 && Math.round(value) !== 0 ? `${MINUS}${text}` : text;
}

/** A number that always shows its sign, for differences. */
export function formatSigned(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  const text = fixed(value, digits);
  return value > 0 && Number(text) !== 0 ? `+${text}` : text;
}

/** A share between 0 and 1 as a percentage. */
export function formatPercent(share: number | null | undefined, digits = 0): string {
  if (share === null || share === undefined || !Number.isFinite(share)) return "";
  return `${fixed(share * 100, digits)}%`;
}

/** A probability between 0 and 1 with three decimals, the way the model reports state it. */
export function formatProbability(value: number | null | undefined): string {
  return formatNumber(value, 3);
}

/** A game clock in seconds as minutes and seconds, for example 11:20. */
export function formatClock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "";
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = (whole % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

/** Seconds with one decimal, for a moment inside a possession. */
export function formatSeconds(seconds: number | null | undefined): string {
  return formatNumber(seconds, 1);
}
