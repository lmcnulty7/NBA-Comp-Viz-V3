/**
 * Search and sort of the player index (PLAN.md M6). Pure. A player whose value for the sort key is not shown
 * (below its minimum sample) goes after every player who has one, whatever the order.
 */
import type { Estimate, PlayerView } from "@/lib/types/contract";

export type SortKey = "name" | "games" | "shot_making" | "gravity" | "rebounding" | "absence_risk";

export const SORT_KEYS: readonly SortKey[] = ["name", "games", "shot_making", "gravity", "rebounding", "absence_risk"];

/** The estimate a metric key shows for a player, or null when it is below its minimum sample. */
export function metricEstimate(player: PlayerView, key: SortKey): Estimate | null {
  switch (key) {
    case "shot_making":
      return player.shot_making.shown ? (player.shot_making.over_expected_per_100 ?? null) : null;
    case "gravity":
      return player.gravity.shown ? (player.gravity.gravity_ft ?? null) : null;
    case "rebounding":
      return player.rebounding.shown ? (player.rebounding.rate_per_100 ?? null) : null;
    case "absence_risk":
      // The model value only: the observed share is a different quantity and is not mixed into one order.
      return player.absence_risk.shown ? (player.absence_risk.model_probability ?? null) : null;
    default:
      return null;
  }
}

export function isMetric(key: SortKey): boolean {
  return key !== "name" && key !== "games";
}

/** Lower case, accents removed, so "Doncic" finds "Dončić". */
export function normalise(text: string): string {
  return text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .trim();
}

/** Players whose name or team holds every word of the query. An empty query keeps everyone. */
export function searchPlayers(players: readonly PlayerView[], query: string): PlayerView[] {
  const words = normalise(query).split(/\s+/).filter(Boolean);
  if (words.length === 0) return [...players];
  return players.filter((player) => {
    const hay = normalise(`${player.name} ${player.teams.join(" ")} ${player.position ?? ""}`);
    return words.every((word) => hay.includes(word));
  });
}

/**
 * Sorted copy. Names go A to Z, everything else largest first; `reverse` flips that. Ties keep name order.
 */
export function sortPlayers(players: readonly PlayerView[], key: SortKey, reverse = false): PlayerView[] {
  const byName = (a: PlayerView, b: PlayerView) => a.name.localeCompare(b.name, "en");
  const direction = reverse ? -1 : 1;
  const value = (player: PlayerView): number | null => {
    if (key === "games") return player.games;
    return metricEstimate(player, key)?.value ?? null;
  };
  return [...players].sort((a, b) => {
    if (key === "name") return direction * byName(a, b);
    const va = value(a);
    const vb = value(b);
    if (va === null && vb === null) return byName(a, b);
    if (va === null) return 1;
    if (vb === null) return -1;
    return direction * (vb - va) || byName(a, b);
  });
}
