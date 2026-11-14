import Link from "next/link";
import { copy, fill, label } from "@/lib/copy";
import { routes } from "@/lib/site";
import type { TourStop } from "./tour";
import styles from "./tour.module.css";

const text = copy.review.tour;

/** How a stop was chosen, in words: its cell of the featured rule, and whether a person picked it by hand. */
export function stopCell(stop: TourStop): string {
  if (stop.xfgBucket != null && (stop.result === "shot_made" || stop.result === "shot_missed")) {
    return fill(text.cell, { result: label(copy.review.result, stop.result), fifth: stop.xfgBucket });
  }
  return stop.cell ? fill(text.cellRaw, { cell: stop.cell }) : text.noCell;
}

export function stopPicker(stop: TourStop): string {
  return stop.featuredBy === "hand" ? text.byHand : text.byRule;
}

/** The tour bar of a featured possession: where it is in the tour, why it was picked, and the way on. */
export function TourBar({ stops, index }: { stops: readonly TourStop[]; index: number }) {
  const stop = stops[index];
  if (!stop) return null;
  const previous = stops[index - 1] ?? null;
  const next = stops[index + 1] ?? null;
  return (
    <nav className={styles.bar} aria-label={text.label} data-testid="tour-bar" data-stop={index + 1} data-stops={stops.length}>
      <p className={styles.where}>
        <span className={styles.count}>{fill(text.stop, { k: index + 1, n: stops.length })}</span>{" "}
        <span data-featured-by={stop.featuredBy}>
          {stopPicker(stop)} {stopCell(stop)}
        </span>
      </p>
      <p className={styles.look}>{text.look}</p>
      <div className={styles.links}>
        {previous ? (
          <Link href={routes.possession(previous.gameSlug, previous.possession)} rel="prev" prefetch={false}>
            {text.previous}
          </Link>
        ) : null}
        {next ? (
          <Link
            href={routes.possession(next.gameSlug, next.possession)}
            rel="next"
            prefetch={false}
            className={styles.next}
            data-testid="tour-next"
          >
            {text.next}
          </Link>
        ) : (
          <Link href={routes.review} className={styles.next} data-testid="tour-end">
            {text.end}
          </Link>
        )}
      </div>
    </nav>
  );
}
