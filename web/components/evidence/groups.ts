/**
 * Which sources feed which value. A possession lists its sources once (export/site_games.py); this sorts them
 * by the value they feed, by the table or report each one names, so the evidence toggle can show each value's
 * own sources. A source no rule claims goes to "other", so none is ever dropped.
 */
import type { Source } from "@/lib/types/contract";

export type EvidenceKey = "positions" | "epv" | "ghosts" | "shots" | "other";

export type EvidenceGroups = Record<EvidenceKey, Source[]>;

/** Words in a source path that tie it to a value. The first rule that matches wins. */
const RULES: readonly [EvidenceKey, RegExp][] = [
  ["epv", /(^|[/_.])epv([/_.]|$)/i],
  ["ghosts", /ghost/i],
  ["shots", /(xfg|shot_frames|shotchart|bref_shots|_pbp\b|_pbp\.)/i],
  ["positions", /(sportvu_contract|possessions|rosters|decisions\/features)/i],
];

export function groupSources(sources: readonly Source[]): EvidenceGroups {
  const groups: EvidenceGroups = { positions: [], epv: [], ghosts: [], shots: [], other: [] };
  for (const source of sources) {
    const rule = RULES.find(([, pattern]) => pattern.test(source.path));
    groups[rule ? rule[0] : "other"].push(source);
  }
  return groups;
}

/** The rows a list of sources reads, without repeats, in a fixed order. */
export function rowsRead(sources: readonly Source[]): Source["rows"][] {
  const order: Source["rows"][] = ["heldout", "out_of_fold", "training", "not_split"];
  const seen = new Set(sources.map((source) => source.rows));
  return order.filter((rows) => seen.has(rows));
}
