/**
 * The filters of a game's possession list (PLAN.md M5): result, xFG fifth and featured. Pure, so the list and
 * its tests agree on what each filter keeps.
 */
import type { PossessionSummary } from "@/lib/types/contract";

export type Result = PossessionSummary["result"];

/** Results in the order the list offers them. */
export const RESULT_ORDER: readonly Result[] = [
  "shot_made",
  "shot_missed",
  "free_throws",
  "turnover",
  "period_end",
  "other",
];

/** "all", a fifth from 1 to 5 as text, or "none" for a possession with no xFG. */
export type FifthFilter = "all" | "none" | `${number}`;

export interface PossessionFilter {
  result: Result | "all";
  fifth: FifthFilter;
  featuredOnly: boolean;
}

export const NO_FILTER: PossessionFilter = { result: "all", fifth: "all", featuredOnly: false };

export function matches(row: PossessionSummary, filter: PossessionFilter): boolean {
  if (filter.result !== "all" && row.result !== filter.result) return false;
  if (filter.featuredOnly && !row.featured) return false;
  if (filter.fifth === "none") return row.xfg_bucket == null;
  if (filter.fifth !== "all") return row.xfg_bucket != null && String(row.xfg_bucket) === filter.fifth;
  return true;
}

export function filterPossessions(rows: readonly PossessionSummary[], filter: PossessionFilter): PossessionSummary[] {
  return rows.filter((row) => matches(row, filter));
}

/** The results that occur in this list, in the fixed order. */
export function resultsPresent(rows: readonly PossessionSummary[]): Result[] {
  const seen = new Set(rows.map((row) => row.result));
  return RESULT_ORDER.filter((result) => seen.has(result));
}

/** The xFG fifths, 1 to the number of edges plus one, each with its range of xFG from the index. */
export function fifths(edges: readonly number[]): { fifth: number; lo: number | null; hi: number | null }[] {
  const out: { fifth: number; lo: number | null; hi: number | null }[] = [];
  for (let index = 0; index <= edges.length; index += 1) {
    out.push({ fifth: index + 1, lo: edges[index - 1] ?? null, hi: edges[index] ?? null });
  }
  return out;
}

/** Read a filter from the address (?result=&fifth=&featured=1), ignoring anything it does not know. */
export function filterFromQuery(query: string): PossessionFilter {
  const params = new URLSearchParams(query);
  const result = params.get("result");
  const fifth = params.get("fifth");
  return {
    result: result && (RESULT_ORDER as readonly string[]).includes(result) ? (result as Result) : "all",
    fifth: fifth === "none" || (fifth !== null && /^[1-9]$/.test(fifth)) ? (fifth as FifthFilter) : "all",
    featuredOnly: params.get("featured") === "1",
  };
}

/** The filter as a query string, empty when nothing is filtered. */
export function filterToQuery(filter: PossessionFilter): string {
  const params = new URLSearchParams();
  if (filter.result !== "all") params.set("result", filter.result);
  if (filter.fifth !== "all") params.set("fifth", filter.fifth);
  if (filter.featuredOnly) params.set("featured", "1");
  const text = params.toString();
  return text ? `?${text}` : "";
}
