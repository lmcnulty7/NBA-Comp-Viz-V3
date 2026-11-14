import type { Metadata } from "next";
import Link from "next/link";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";
import { routes } from "@/lib/site";
import type { PlayerView } from "@/lib/types/contract";

const text = copy.prep;

export const metadata: Metadata = { title: text.title };

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

export default async function PrepPage() {
  const players = await getServerSource().players();
  return (
    <div className="column stack stack-loose">
      <header className="page-head">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
        <p className="small">{fill(text.window, { season: players.season, window: players.window })}</p>
      </header>

      <section className="section" aria-labelledby="rules">
        <h2 id="rules">{text.rules.title}</h2>
        <p data-testid="player-counts">
          {fill(text.counts, {
            shown: formatCount(players.n_shown),
            total: formatCount(players.n_players),
            hidden: formatCount(players.n_hidden),
          })}
        </p>
        <ul className="bullets label">
          {players.rules.map((rule) => (
            <li key={rule.metric}>
              {fill(text.rules.row, {
                metric: label(text.metrics, rule.metric),
                n: formatCount(rule.min_n),
                unit: rule.unit,
              })}{" "}
              <span className="muted">{rule.reason}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="section" aria-labelledby="players">
        <h2 id="players">{text.list.title}</h2>
        <div className="table-wrap" role="region" aria-label={text.list.caption} tabIndex={0}>
          <table>
            <caption className="visually-hidden">{text.list.caption}</caption>
            <thead>
              <tr>
                <th scope="col">{text.list.player}</th>
                <th scope="col">{text.list.teams}</th>
                <th scope="col" className="wide-only">
                  {text.list.position}
                </th>
                <th scope="col">{text.list.games}</th>
                <th scope="col">{text.list.views}</th>
              </tr>
            </thead>
            <tbody>
              {players.players.map((player) => (
                <tr key={player.slug}>
                  <th scope="row">
                    <Link href={routes.player(player.slug)} prefetch={false}>
                      {player.name}
                    </Link>
                  </th>
                  <td>{player.teams.join(", ")}</td>
                  <td className="wide-only">{player.position ?? null}</td>
                  <td className="num">{formatCount(player.games)}</td>
                  <td className="num">
                    {fill(text.list.viewCount, { n: shownViews(player), total: players.rules.length })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <SourceList sources={players.sources} />
      </section>
    </div>
  );
}
