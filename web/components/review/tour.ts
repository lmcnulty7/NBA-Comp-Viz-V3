/**
 * The guided tour (PLAN.md M5): the featured possessions of every game, in the order the featured rule visits
 * its cells (D52): made shots, then missed shots, each from the lowest xFG fifth to the highest. A possession
 * outside those cells (a pick made by hand) comes after them. Pure, so the review page, every possession page
 * and the tests build the same tour.
 */
import type { PossessionSummary } from "@/lib/types/contract";

export interface TourGame {
  slug: string;
  label: string;
  possessions: readonly PossessionSummary[];
}

export interface TourStop {
  gameSlug: string;
  gameLabel: string;
  possession: number;
  result: PossessionSummary["result"];
  xfgBucket: number | null;
  featuredBy: "rule" | "hand";
  cell: string | null;
}

const RESULT_RANK: Partial<Record<PossessionSummary["result"], number>> = { shot_made: 0, shot_missed: 1 };

export function buildTour(games: readonly TourGame[]): TourStop[] {
  const stops: (TourStop & { gameRank: number })[] = [];
  games.forEach((game, gameRank) => {
    for (const row of game.possessions) {
      if (!row.featured) continue;
      stops.push({
        gameSlug: game.slug,
        gameLabel: game.label,
        possession: row.possession,
        result: row.result,
        xfgBucket: row.xfg_bucket ?? null,
        featuredBy: row.featured_by === "hand" ? "hand" : "rule",
        cell: row.featured_cell ?? null,
        gameRank,
      });
    }
  });
  const rank = (stop: TourStop) => RESULT_RANK[stop.result] ?? 2;
  stops.sort(
    (a, b) =>
      rank(a) - rank(b) ||
      (a.xfgBucket ?? Number.MAX_SAFE_INTEGER) - (b.xfgBucket ?? Number.MAX_SAFE_INTEGER) ||
      a.gameRank - b.gameRank ||
      a.possession - b.possession,
  );
  return stops.map((stop) => ({
    gameSlug: stop.gameSlug,
    gameLabel: stop.gameLabel,
    possession: stop.possession,
    result: stop.result,
    xfgBucket: stop.xfgBucket,
    featuredBy: stop.featuredBy,
    cell: stop.cell,
  }));
}

/** Where a possession sits in the tour: its index, or -1 when it is not a stop. */
export function stopIndex(stops: readonly TourStop[], gameSlug: string, possession: number): number {
  return stops.findIndex((stop) => stop.gameSlug === gameSlug && stop.possession === possession);
}
