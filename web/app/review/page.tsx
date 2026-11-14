import type { Metadata } from "next";
import Link from "next/link";
import { findLane } from "@/components/charts/LaneMetric";
import { stopCell, stopPicker } from "@/components/review/TourBar";
import tourStyles from "@/components/review/tour.module.css";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";
import { routes } from "@/lib/site";
import { NOT_SHOWN_LANES, loadTour } from "./load";

const text = copy.review;

export const metadata: Metadata = { title: text.title };

export default async function ReviewPage() {
  const source = getServerSource();
  const [manifest, stops, results] = await Promise.all([source.manifest(), loadTour(source), source.results()]);
  const notShown = NOT_SHOWN_LANES.map((key) => findLane(results, key)).filter((lane) => lane !== null);
  const first = stops[0];
  const byHand = stops.filter((stop) => stop.featuredBy === "hand").length;
  return (
    <div className="column stack stack-loose">
      <header className="page-head">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
      </header>

      <section className="section" aria-labelledby="games">
        <h2 id="games">{text.games.title}</h2>
        <p className="small" data-testid="heldout-note">
          {fill(text.games.heldout, { n: formatCount(manifest.games.length) })}
        </p>
        <div className="table-wrap" role="region" aria-label={text.games.caption} tabIndex={0}>
          <table>
            <caption className="visually-hidden">{text.games.caption}</caption>
            <thead>
              <tr>
                <th scope="col">{text.games.game}</th>
                <th scope="col">{text.games.date}</th>
                <th scope="col" className="wide-only">
                  {text.games.split}
                </th>
                <th scope="col">{text.games.possessions}</th>
                <th scope="col" className="wide-only">
                  {text.games.featured}
                </th>
              </tr>
            </thead>
            <tbody>
              {manifest.games.map((game) => (
                <tr key={game.slug} data-split={game.split}>
                  <th scope="row">
                    <Link href={routes.game(game.slug)} aria-label={fill(text.games.open, { game: game.label, date: game.date })}>
                      {game.label}
                    </Link>
                  </th>
                  <td className="num">{game.date}</td>
                  <td className="wide-only">{label(text.split, game.split)}</td>
                  <td className="num">{formatCount(game.n_possessions)}</td>
                  <td className="num wide-only">{formatCount(game.n_featured)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="tour" data-testid="tour-intro">
        <h2 id="tour">{text.tour.title}</h2>
        {first ? (
          <>
            <p>{fill(text.tour.intro, { n: formatCount(stops.length) })}</p>
            <p className="small">
              {byHand > 0 ? fill(text.tour.handCount, { n: formatCount(byHand) }) : text.tour.allByRule}
            </p>
            <Link
              href={routes.possession(first.gameSlug, first.possession)}
              className={tourStyles.start}
              data-testid="tour-start"
            >
              {text.tour.start}
            </Link>
            <ol className={tourStyles.stops}>
              {stops.map((stop) => (
                <li key={`${stop.gameSlug}/${stop.possession}`}>
                  <Link href={routes.possession(stop.gameSlug, stop.possession)} prefetch={false}>
                    {fill(text.tour.stopName, { game: stop.gameLabel, date: stop.gameDate, n: stop.possession })}
                  </Link>{" "}
                  <span className="muted">
                    {stopCell(stop)} {stopPicker(stop)}
                  </span>
                </li>
              ))}
            </ol>
          </>
        ) : (
          <p>{text.tour.none}</p>
        )}
      </section>

      {notShown.length > 0 ? (
        <section className="section" aria-labelledby="not-shown" data-testid="not-shown">
          <h2 id="not-shown">{text.notShown.title}</h2>
          <p>{text.notShown.intro}</p>
          <ul className="bullets">
            {notShown.map((lane) => (
              <li key={lane.lane} data-lane={lane.lane} data-verdict={lane.verdict_class}>
                <strong>{lane.question}</strong>{" "}
                {fill(text.notShown.row, {
                  verdict: label(copy.results.models.verdictClass, lane.verdict_class),
                })}{" "}
                <q>{lane.verdict}</q>
              </li>
            ))}
          </ul>
          <p className="small">
            <Link href={routes.results}>{text.notShown.link}</Link>
          </p>
        </section>
      ) : null}
    </div>
  );
}
