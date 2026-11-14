import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { PossessionStage } from "@/components/review/PossessionStage";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatClock, formatCount, formatNumber, formatProbability, formatSeconds } from "@/lib/format";
import { routes } from "@/lib/site";

const text = copy.review.possession;

export const dynamicParams = false;

/** Every possession of every game, from the indexes. The route uses the game slug, never the dotted id. */
export async function generateStaticParams() {
  const source = getServerSource();
  const manifest = await source.manifest();
  const params: { game: string; possession: string }[] = [];
  for (const game of manifest.games) {
    const index = await source.possessionIndex(game.slug);
    for (const row of index.possessions) {
      params.push({ game: game.slug, possession: String(row.possession) });
    }
  }
  return params;
}

async function load(slug: string, possession: string) {
  const source = getServerSource();
  const manifest = await source.manifest();
  if (!manifest.games.some((game) => game.slug === slug)) notFound();
  const index = await source.possessionIndex(slug);
  const position = index.possessions.findIndex((row) => String(row.possession) === possession);
  const summary = index.possessions[position];
  if (!summary) notFound();
  // The detail file is read at build time too, so the shots and the sources are in the page itself.
  const detail = await source.possession(slug, summary.possession);
  return {
    index,
    summary,
    detail,
    previous: index.possessions[position - 1] ?? null,
    next: index.possessions[position + 1] ?? null,
  };
}

export async function generateMetadata({
  params,
}: PageProps<"/review/[game]/[possession]">): Promise<Metadata> {
  const { game, possession } = await params;
  const { index, summary } = await load(game, possession);
  return { title: fill(text.pageTitle, { n: summary.possession, game: index.game.label }) };
}

export default async function PossessionPage({ params }: PageProps<"/review/[game]/[possession]">) {
  const { game, possession } = await params;
  const { index, summary, detail, previous, next } = await load(game, possession);
  const none = <span className="muted">{copy.site.none}</span>;
  return (
    <div className="wide stack stack-loose">
      <header className="page-head column">
        <nav className="crumbs" aria-label={copy.review.title}>
          <Link href={routes.review}>{copy.review.game.back}</Link>
          <span aria-hidden="true">/</span>
          <Link href={routes.game(index.game.slug)}>{index.game.label}</Link>
        </nav>
        <h1>{fill(text.title, { n: summary.possession })}</h1>
        <p className="lede">
          {fill(text.meta, {
            period: summary.period,
            clock: formatClock(summary.start_clock_s),
            offense: summary.offense,
            defense: summary.defense,
          })}
        </p>
      </header>

      <dl className="facts column">
        <div>
          <dt>{text.facts.result}</dt>
          <dd>{label(copy.review.result, summary.result)}</dd>
        </div>
        <div>
          <dt>{text.facts.points}</dt>
          <dd className="num">{summary.points}</dd>
        </div>
        <div>
          <dt>{text.facts.xfg}</dt>
          <dd>
            {summary.xfg != null ? <span className="num">{formatProbability(summary.xfg)}</span> : none}
            <span className="note">{text.facts.xfgNote}</span>
          </dd>
        </div>
        <div>
          <dt>{text.facts.epvStart}</dt>
          <dd>
            {summary.epv_start != null ? <span className="num">{formatNumber(summary.epv_start, 2)}</span> : none}
            <span className="note">{text.facts.epvNote}</span>
          </dd>
        </div>
        <div>
          <dt>{text.facts.epvEnd}</dt>
          <dd>
            {summary.epv_end != null ? <span className="num">{formatNumber(summary.epv_end, 2)}</span> : none}
            <span className="note">{text.facts.epvNote}</span>
          </dd>
        </div>
        <div>
          <dt>{text.facts.duration}</dt>
          <dd className="num">{fill(text.facts.seconds, { value: formatSeconds(summary.duration_s) })}</dd>
        </div>
        <div>
          <dt>{text.facts.frames}</dt>
          <dd>
            <span className="num">{formatCount(summary.n_frames)}</span>
            <span className="note">{fill(text.facts.framesNote, { hz: formatNumber(detail.hz, 0) })}</span>
          </dd>
        </div>
      </dl>

      <section className="section" aria-labelledby="court">
        <h2 id="court" className="column">
          {text.court.title}
        </h2>
        {/* The key gives each possession its own stage, so one never opens on the frame chosen for another. */}
        <PossessionStage
          key={`${index.game.slug}/${summary.possession}`}
          gameSlug={index.game.slug}
          possession={summary.possession}
        />
      </section>

      <section className="column section" aria-labelledby="shots">
        <h2 id="shots">{text.shots.title}</h2>
        {detail.shots.length === 0 ? (
          <p>{text.shots.none}</p>
        ) : (
          <div className="table-wrap" role="region" aria-label={text.shots.caption} tabIndex={0}>
            <table>
              <caption className="visually-hidden">{text.shots.caption}</caption>
              <thead>
                <tr>
                  <th scope="col">{text.shots.shooter}</th>
                  <th scope="col">{text.shots.outcome}</th>
                  <th scope="col" className="wide-only">
                    {text.shots.kind}
                  </th>
                  <th scope="col">{text.shots.time}</th>
                  <th scope="col">{text.shots.xfg}</th>
                </tr>
              </thead>
              <tbody>
                {detail.shots.map((shot) => (
                  <tr key={shot.event}>
                    <th scope="row">{shot.shooter ?? copy.site.none}</th>
                    <td>{shot.made ? text.shots.made : text.shots.missed}</td>
                    <td className="wide-only">
                      {shot.three == null ? null : shot.three ? text.shots.three : text.shots.two}
                    </td>
                    <td>
                      {shot.t != null ? (
                        <span className="num">{fill(text.shots.seconds, { value: formatSeconds(shot.t) })}</span>
                      ) : null}
                      {shot.t_source === "play_by_play" ? (
                        <span className="note small">{text.shots.timeFromPlayByPlay}</span>
                      ) : null}
                      {shot.location_source === "shot_chart" ? (
                        <span className="note small">{text.shots.locationFromShotChart}</span>
                      ) : null}
                    </td>
                    <td className="num">{shot.xfg != null ? formatProbability(shot.xfg) : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {detail.notes.length > 0 ? (
          <div className="stack stack-tight">
            <h3>{text.notes}</h3>
            <ul className="bullets small">
              {detail.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <SourceList sources={detail.sources} />
        <nav className="pager" aria-label={copy.review.game.possessions}>
          {previous ? (
            <Link href={routes.possession(index.game.slug, previous.possession)} rel="prev">
              {text.previous}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={routes.possession(index.game.slug, next.possession)} rel="next">
              {text.next}
            </Link>
          ) : null}
        </nav>
      </section>
    </div>
  );
}
