import type { Metadata } from "next";
import { PlayerIndex } from "@/components/prep/PlayerIndex";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";

const text = copy.prep;

export const metadata: Metadata = { title: text.title };

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
        <p className="small">{text.rules.intervals}</p>
      </section>

      <section className="section" aria-labelledby="players">
        <h2 id="players">{text.list.title}</h2>
        {/* The index needs no neighbour lists; leaving them out keeps the page light. */}
        <PlayerIndex
          players={players.players.map((player) => ({ ...player, similar: { ...player.similar, neighbours: [] } }))}
          views={players.rules.length}
        />
        <SourceList sources={players.sources} />
      </section>
    </div>
  );
}
