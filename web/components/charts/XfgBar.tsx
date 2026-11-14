import { linear } from "./scale";
import styles from "./charts.module.css";

/**
 * A shot's xFG on a scale from zero to one: a filled bar to the value. Purely visual; the value and the words
 * "point estimate" are printed beside it, so the picture is hidden from screen readers.
 */
export function XfgBar({ xfg }: { xfg: number }) {
  const x = linear([0, 1], [0, 100]);
  const end = Math.min(100, Math.max(0, x(xfg)));
  return (
    <svg className={styles.xfg} viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true" data-testid="xfg-bar">
      <rect x={0} y={2} width={100} height={6} className={styles.xfgTrack} />
      <rect x={0} y={2} width={end} height={6} className={styles.xfgFill} />
    </svg>
  );
}
