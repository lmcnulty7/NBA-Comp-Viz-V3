/**
 * Build-time loading shared by the review pages: the guided tour over every game's featured possessions, and
 * the model lanes whose held-out metric sits beside the values a possession shows.
 */
import { findLane } from "@/components/charts/LaneMetric";
import type { StageLanes } from "@/components/review/PossessionStage";
import { buildTour, type TourStop } from "@/components/review/tour";
import type { DataSource } from "@/lib/data/source";

let tour: Promise<TourStop[]> | null = null;

/** The tour of the data build, read once per build. */
export function loadTour(source: DataSource): Promise<TourStop[]> {
  tour ??= (async () => {
    const manifest = await source.manifest();
    const games = [];
    for (const game of manifest.games) {
      const index = await source.possessionIndex(game.slug);
      games.push({ slug: game.slug, label: game.label, possessions: index.possessions });
    }
    return buildTour(games);
  })();
  return tour;
}

/** The lanes behind xFG, EPV and ghosting, from results.json; null for a lane this data build does not have. */
export async function loadLanes(source: DataSource): Promise<StageLanes> {
  const results = await source.results();
  // The stage needs the held-out metric and the report name only; the report rows stay out of every page.
  const slim = (key: string) => {
    const lane = findLane(results, key);
    return lane ? { ...lane, rows: [] } : null;
  };
  return { xfg: slim("xfg"), epv: slim("epv"), ghosting: slim("ghosting") };
}

/** The lanes the review does not show because they were tested and found no signal (PLAN.md C10). */
export const NOT_SHOWN_LANES = ["decisions", "matchups"] as const;
