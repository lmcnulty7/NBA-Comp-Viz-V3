"use client";

import Link from "next/link";
import { useId, useMemo, useState } from "react";
import listStyles from "@/components/review/list.module.css";
import { EstimateValue } from "@/components/ui/EstimateValue";
import { copy, fill, label } from "@/lib/copy";
import { formatCount, formatSigned } from "@/lib/format";
import { routes } from "@/lib/site";
import type { PlayerView } from "@/lib/types/contract";
import { SORT_KEYS, isMetric, metricEstimate, searchPlayers, sortPlayers, type SortKey } from "./sort";

const text = copy.prep.list;

/** Decimals each metric is shown with, the same as on the player page. */
const DIGITS: Record<SortKey, number> = { name: 0, games: 0, shot_making: 1, gravity: 2, rebounding: 1, absence_risk: 3 };

/** How many of a player's five views clear their minimum sample. */
function shownViews(player: PlayerView): number {
  return [
    player.shot_making.shown,
    player.gravity.shown,
    player.rebounding.shown,
    player.absence_risk.shown,
    player.similar.shown,
  ].filter(Boolean).length;
}

/** The player index: search by name or team, sort by name, games or a metric (PLAN.md M6). */
export function PlayerIndex({ players, views }: { players: PlayerView[]; views: number }) {
  const [query, setQuery] = useState("");
  const [key, setKey] = useState<SortKey>("name");
  const [reverse, setReverse] = useState(false);
  const searchId = useId();
  const sortId = useId();

  const shown = useMemo(() => sortPlayers(searchPlayers(players, query), key, reverse), [players, query, key, reverse]);
  const metric = isMetric(key);
  const caption = text.caption;

  return (
    <div className="stack" data-testid="player-index">
      <form className={listStyles.filters} role="search" onSubmit={(event) => event.preventDefault()}>
        <label htmlFor={searchId}>
          <span>{text.search}</span>
          <input
            id={searchId}
            type="search"
            value={query}
            autoComplete="off"
            spellCheck={false}
            onChange={(event) => setQuery(event.target.value)}
            data-testid="player-search"
          />
        </label>
        <label htmlFor={sortId}>
          <span>{text.sort}</span>
          <select id={sortId} value={key} onChange={(event) => setKey(event.target.value as SortKey)} data-testid="player-sort">
            {SORT_KEYS.map((option) => (
              <option key={option} value={option}>
                {label(text.sortBy, option)}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className={listStyles.clear}
          aria-pressed={reverse}
          onClick={() => setReverse((value) => !value)}
          data-testid="player-reverse"
        >
          {text.reverse}
        </button>
      </form>
      <p className="small" role="status" data-testid="player-count">
        {fill(text.count, { n: formatCount(shown.length), total: formatCount(players.length) })}
      </p>
      {metric ? <p className="small muted">{text.belowNote}</p> : null}
      {/* D51: shot-making is read beside the baseline's own error; absence risk is not an injury forecast. */}
      {key === "shot_making" ? (
        <p className="small" data-testid="sort-note">
          {text.baselineNote}
        </p>
      ) : null}
      {key === "absence_risk" ? (
        <p className="small" data-testid="sort-note">
          {text.absenceNote}
        </p>
      ) : null}

      <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>
              <th scope="col">{text.player}</th>
              <th scope="col">{text.teams}</th>
              <th scope="col" className="wide-only">
                {text.position}
              </th>
              <th scope="col">{text.games}</th>
              {metric ? <th scope="col">{label(text.column, key)}</th> : <th scope="col">{text.views}</th>}
            </tr>
          </thead>
          <tbody>
            {shown.map((player) => {
              const estimate = metric ? metricEstimate(player, key) : null;
              return (
                <tr key={player.slug} data-player={player.slug}>
                  <th scope="row">
                    <Link href={routes.player(player.slug)} prefetch={false}>
                      {player.name}
                    </Link>
                  </th>
                  <td>{player.teams.join(", ")}</td>
                  <td className="wide-only">{player.position ?? null}</td>
                  <td className="num">{formatCount(player.games)}</td>
                  {metric ? (
                    <td data-metric-cell={key}>
                      {estimate ? (
                        <>
                          <EstimateValue estimate={estimate} digits={DIGITS[key]} />
                          {key === "shot_making" && player.shot_making.baseline_error_per_100 != null ? (
                            <span className="note small" data-testid="baseline-error">
                              {fill(text.baselineError, { value: formatSigned(player.shot_making.baseline_error_per_100, 1) })}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className="muted">{text.below}</span>
                      )}
                    </td>
                  ) : (
                    <td className="num">{fill(text.viewCount, { n: shownViews(player), total: views })}</td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {shown.length === 0 ? <p>{text.empty}</p> : null}
    </div>
  );
}
