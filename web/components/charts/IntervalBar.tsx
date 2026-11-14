import { linear } from "./scale";
import styles from "./charts.module.css";

export interface IntervalBarProps {
  /** The axis, from the data (for example the smallest and largest interval ends among shown players). */
  domain: [number, number];
  value: number;
  lo?: number | null;
  hi?: number | null;
  /** Where to draw a reference line, such as zero. Left out when it falls outside the domain. */
  reference?: number | null;
}

/**
 * A value as a dot with its interval as a line, on an axis set by the caller. Purely visual: the numbers are
 * printed next to it by EstimateValue, so the picture is hidden from screen readers.
 */
export function IntervalBar({ domain, value, lo, hi, reference = null }: IntervalBarProps) {
  const x = linear(domain, [2, 98]);
  const clamp = (v: number) => Math.min(98, Math.max(2, x(v)));
  const hasInterval = lo != null && hi != null;
  const showReference = reference != null && reference >= domain[0] && reference <= domain[1];
  return (
    <svg className={styles.interval} viewBox="0 0 100 12" preserveAspectRatio="none" aria-hidden="true" data-testid="interval-bar">
      <line x1={2} x2={98} y1={6} y2={6} className={styles.track} vectorEffect="non-scaling-stroke" />
      {showReference ? (
        <line x1={x(reference)} x2={x(reference)} y1={1} y2={11} className={styles.reference} vectorEffect="non-scaling-stroke" />
      ) : null}
      {hasInterval ? (
        <line x1={clamp(lo)} x2={clamp(hi)} y1={6} y2={6} className={styles.whisker} vectorEffect="non-scaling-stroke" />
      ) : null}
      <line x1={clamp(value)} x2={clamp(value)} y1={2} y2={10} className={styles.mark} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** The axis that holds every interval in a list, with a little room at each end. */
export function domainOf(items: readonly { value: number; lo?: number | null; hi?: number | null }[], include: number[] = []): [number, number] {
  const numbers = [
    ...include,
    ...items.flatMap((item) => [item.value, item.lo, item.hi].filter((v): v is number => v != null && Number.isFinite(v))),
  ];
  if (numbers.length === 0) return [0, 1];
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const pad = max > min ? (max - min) * 0.05 : Math.abs(max) * 0.1 || 1;
  return [min - pad, max + pad];
}
