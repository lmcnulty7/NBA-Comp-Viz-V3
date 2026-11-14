/** Scales and ticks for the small SVG charts. Pure. */

export interface Linear {
  (value: number): number;
  domain: [number, number];
  range: [number, number];
}

/** A linear map from a domain to a range. A domain of zero width maps everything to the middle of the range. */
export function linear(domain: [number, number], range: [number, number]): Linear {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const width = d1 - d0;
  const map = ((value: number) => (width === 0 ? (r0 + r1) / 2 : r0 + ((value - d0) / width) * (r1 - r0))) as Linear;
  map.domain = domain;
  map.range = range;
  return map;
}

/** A step of 1, 2 or 5 times a power of ten that gives about `count` ticks over the span. */
export function niceStep(span: number, count: number): number {
  if (!(span > 0) || !(count > 0)) return 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const factor = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10;
  return factor * power;
}

/** Round tick values from the floor to the ceiling of the domain, at a nice step. */
export function ticks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (max < min) [min, max] = [max, min];
  if (max === min) return [min];
  const step = niceStep(max - min, count);
  const start = Math.ceil(min / step - 1e-9) * step;
  const out: number[] = [];
  for (let value = start; value <= max + step * 1e-9; value += step) out.push(Number(value.toFixed(10)));
  return out;
}

/** The domain widened to whole steps, so the first and last ticks sit at its ends. */
export function niceDomain(min: number, max: number, count = 4): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (max < min) [min, max] = [max, min];
  if (max === min) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return niceDomain(min - pad, max + pad, count);
  }
  const step = niceStep(max - min, count);
  return [Math.floor(min / step + 1e-9) * step, Math.ceil(max / step - 1e-9) * step];
}

/** How many decimals a tick step needs. */
export function stepDecimals(step: number): number {
  if (!(step > 0) || step >= 1) return 0;
  return Math.min(6, Math.ceil(-Math.log10(step) - 1e-9));
}
