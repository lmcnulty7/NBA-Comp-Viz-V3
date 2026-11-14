"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useState } from "react";
import { copy, fill, label } from "@/lib/copy";
import { formatClock, formatCount, formatNumber, formatProbability } from "@/lib/format";
import { routes } from "@/lib/site";
import type { PossessionSummary } from "@/lib/types/contract";
import {
  NO_FILTER,
  fifths,
  filterFromQuery,
  filterPossessions,
  filterToQuery,
  resultsPresent,
  type FifthFilter,
  type PossessionFilter,
  type Result,
} from "./filters";
import styles from "./list.module.css";

const text = copy.review;

function flags(row: PossessionSummary): string {
  const parts: string[] = [];
  if (row.featured) parts.push(row.featured_by === "hand" ? text.flags.featuredByHand : text.flags.featured);
  if (row.demo) parts.push(text.flags.demo);
  if (row.gap) parts.push(text.flags.gap);
  if (!row.has_ghosts) parts.push(text.flags.noGhosts);
  if (row.n_frames <= 1) parts.push(text.flags.oneFrame);
  return parts.join(", ");
}

export interface PossessionListProps {
  gameSlug: string;
  gameLabel: string;
  rows: PossessionSummary[];
  edges: number[];
}

/**
 * A game's possessions with filters by result, xFG fifth and featured (PLAN.md M5). The filter is kept in the
 * address, so a filtered list can be shared and the back button returns to it.
 */
export function PossessionList({ gameSlug, gameLabel, rows, edges }: PossessionListProps) {
  const [filter, setFilter] = useState<PossessionFilter>(NO_FILTER);
  const resultId = useId();
  const fifthId = useId();
  const featuredId = useId();

  // The filter in the address wins once, after the page loads.
  useEffect(() => {
    const fromAddress = filterFromQuery(window.location.search);
    if (filterToQuery(fromAddress) !== "") {
      // Reading the address can only happen in the browser, after the static page has rendered.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFilter(fromAddress);
    }
  }, []);

  const update = (next: PossessionFilter) => {
    setFilter(next);
    const url = `${window.location.pathname}${filterToQuery(next)}${window.location.hash}`;
    window.history.replaceState(window.history.state, "", url);
  };

  const shown = useMemo(() => filterPossessions(rows, filter), [rows, filter]);
  const results = useMemo(() => resultsPresent(rows), [rows]);
  const buckets = useMemo(() => fifths(edges), [edges]);
  const anyWithoutXfg = rows.some((row) => row.xfg_bucket == null);
  const caption = fill(text.game.caption, { game: gameLabel });

  const fifthName = (bucket: { fifth: number; lo: number | null; hi: number | null }) => {
    if (bucket.lo === null && bucket.hi !== null) {
      return fill(text.filters.fifthLow, { k: bucket.fifth, hi: formatNumber(bucket.hi, 2) });
    }
    if (bucket.hi === null && bucket.lo !== null) {
      return fill(text.filters.fifthHigh, { k: bucket.fifth, lo: formatNumber(bucket.lo, 2) });
    }
    if (bucket.lo !== null && bucket.hi !== null) {
      return fill(text.filters.fifthMid, {
        k: bucket.fifth,
        lo: formatNumber(bucket.lo, 2),
        hi: formatNumber(bucket.hi, 2),
      });
    }
    return fill(text.filters.fifth, { k: bucket.fifth });
  };

  return (
    <div className="stack" data-testid="possession-list">
      <form className={styles.filters} onSubmit={(event) => event.preventDefault()} aria-label={text.filters.label}>
        <label htmlFor={resultId}>
          <span>{text.filters.result}</span>
          <select
            id={resultId}
            value={filter.result}
            onChange={(event) => update({ ...filter, result: event.target.value as Result | "all" })}
            data-testid="filter-result"
          >
            <option value="all">{text.filters.allResults}</option>
            {results.map((result) => (
              <option key={result} value={result}>
                {label(text.result, result)}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor={fifthId}>
          <span>{text.filters.xfg}</span>
          <select
            id={fifthId}
            value={filter.fifth}
            onChange={(event) => update({ ...filter, fifth: event.target.value as FifthFilter })}
            data-testid="filter-fifth"
          >
            <option value="all">{text.filters.allFifths}</option>
            {buckets.map((bucket) => (
              <option key={bucket.fifth} value={String(bucket.fifth)}>
                {fifthName(bucket)}
              </option>
            ))}
            {anyWithoutXfg ? <option value="none">{text.filters.noXfg}</option> : null}
          </select>
        </label>
        <div className={styles.check}>
          <input
            id={featuredId}
            type="checkbox"
            checked={filter.featuredOnly}
            onChange={(event) => update({ ...filter, featuredOnly: event.target.checked })}
            data-testid="filter-featured"
          />
          <label htmlFor={featuredId}>{text.filters.featured}</label>
        </div>
        {filterToQuery(filter) !== "" ? (
          <button type="button" className={styles.clear} onClick={() => update(NO_FILTER)}>
            {text.filters.clear}
          </button>
        ) : null}
      </form>
      <p className="small" role="status" data-testid="filter-count">
        {fill(text.filters.count, { n: formatCount(shown.length), total: formatCount(rows.length) })}
      </p>
      <p className="small muted">{text.filters.xfgNote}</p>

      <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>
              <th scope="col">{text.game.possession}</th>
              <th scope="col">{text.game.period}</th>
              <th scope="col">{text.game.clock}</th>
              <th scope="col">{text.game.offense}</th>
              <th scope="col">{text.game.result}</th>
              <th scope="col" className="wide-only">
                {text.game.points}
              </th>
              <th scope="col">{text.game.xfg}</th>
              <th scope="col" className="wide-only">
                {text.game.epv}
              </th>
              <th scope="col" className="wide-only">
                {text.game.flags}
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((row) => (
              <tr key={row.possession} data-result={row.result} data-fifth={row.xfg_bucket ?? "none"} data-featured={row.featured}>
                <th scope="row">
                  {/* A game has a couple of hundred possessions: none is fetched before it is asked for. */}
                  <Link
                    href={routes.possession(gameSlug, row.possession)}
                    aria-label={fill(text.game.open, { n: row.possession })}
                    className="num"
                    prefetch={false}
                  >
                    {row.possession}
                  </Link>
                </th>
                <td className="num">{row.period}</td>
                <td className="num">{formatClock(row.start_clock_s)}</td>
                <td>{row.offense}</td>
                <td>{label(text.result, row.result)}</td>
                <td className="num wide-only">{row.points}</td>
                <td className="num">{row.xfg != null ? formatProbability(row.xfg) : null}</td>
                <td className="num wide-only">
                  {row.epv_start != null && row.epv_end != null
                    ? fill(text.game.epvRange, {
                        start: formatNumber(row.epv_start, 2),
                        end: formatNumber(row.epv_end, 2),
                      })
                    : null}
                </td>
                <td className="wide-only">{flags(row)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
