import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Evidence } from "@/components/evidence/Evidence";
import { EvidenceScope } from "@/components/evidence/EvidenceScope";
import { groupSources } from "@/components/evidence/groups";
import { PossessionStage } from "@/components/review/PossessionStage";
import { PossessionStill } from "@/components/review/PossessionStill";
import { TourBar } from "@/components/review/TourBar";
import { stopIndex } from "@/components/review/tour";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import {
  formatClock,
  formatCount,
  formatNumber,
  formatPercent,
  formatProbability,
  formatSeconds,
} from "@/lib/format";
import { VARIANT, routes } from "@/lib/site";
import { loadLanes, loadTour } from "../../load";

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
  const source = getServerSource();
  const [stops, lanes, swap] = await Promise.all([loadTour(source), loadLanes(source), source.demoSwap()]);
  const stop = stopIndex(stops, index.game.slug, summary.possession);
  const evidence = groupSources(detail.sources);
  // The broadcast demo's fill share for this possession, when the demo covers it (D27).
  const demo =
    swap && swap.game_id === index.game.game_id
      ? (swap.possessions.find((row) => row.possession === summary.possession) ?? null)
      : null;
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
        <p className="small" data-testid="heldout-flag" data-split={index.game.split}>
          {fill(text.heldout, { split: label(copy.review.split, index.game.split) })}
        </p>
      </header>

      {stop >= 0 ? (
        <div className="column">
          <TourBar stops={stops} index={stop} />
        </div>
      ) : null}

      <EvidenceScope>
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
              <Evidence what={text.facts.xfg} sources={evidence.shots} />
            </dd>
          </div>
          <div>
            <dt>{text.facts.epvStart}</dt>
            <dd>
              {summary.epv_start != null ? <span className="num">{formatNumber(summary.epv_start, 2)}</span> : none}
              <span className="note">{text.facts.epvNote}</span>
              <Evidence what={text.facts.epvStart} sources={evidence.epv} />
            </dd>
          </div>
          <div>
            <dt>{text.facts.epvEnd}</dt>
            <dd>
              {summary.epv_end != null ? <span className="num">{formatNumber(summary.epv_end, 2)}</span> : none}
              <span className="note">{text.facts.epvNote}</span>
              <Evidence what={text.facts.epvEnd} sources={evidence.epv} />
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
              <Evidence what={text.facts.frames} sources={evidence.positions} />
            </dd>
          </div>
          {summary.demo || demo ? (
            <div data-testid="demo-flag">
              <dt>{text.facts.demo}</dt>
              <dd>
                {demo ? (
                  <>
                    <span className="num">{formatPercent(demo.fill_share, 0)}</span>
                    <span className="note">
                      {fill(text.facts.fillShare, {
                        filled: formatCount(demo.n_filled),
                        total: formatCount(demo.n_player_frames),
                      })}
                    </span>
                  </>
                ) : (
                  <span>{text.facts.demoYes}</span>
                )}
                <span className="note">{text.facts.demoNote}</span>
                {swap ? <Evidence what={text.facts.demo} sources={swap.sources} /> : null}
              </dd>
            </div>
          ) : null}
          {summary.gap ? (
            <div data-testid="gap-flag">
              <dt>{text.facts.gap}</dt>
              <dd>{text.facts.gapNote}</dd>
            </div>
          ) : null}
        </dl>

        <section className="section" aria-labelledby="court">
          <h2 id="court" className="column">
            {text.court.title}
          </h2>
          {/* The key gives each possession its own stage, so one never opens on the frame chosen for another.
              The public site holds no player position (PLAN.md D60), so it shows a still court instead. */}
          {VARIANT === "public" ? (
            <PossessionStill
              key={`${index.game.slug}/${summary.possession}`}
              gameSlug={index.game.slug}
              possession={summary.possession}
              duration={summary.duration_s}
              lanes={lanes}
              evidence={evidence}
            />
          ) : (
            <PossessionStage
              key={`${index.game.slug}/${summary.possession}`}
              gameSlug={index.game.slug}
              possession={summary.possession}
              lanes={lanes}
              evidence={evidence}
            />
          )}
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
              <Link href={routes.possession(index.game.slug, previous.possession)} rel="prev" prefetch={false}>
                {text.previous}
              </Link>
            ) : (
              <span />
            )}
            {next ? (
              <Link href={routes.possession(index.game.slug, next.possession)} rel="next" prefetch={false}>
                {text.next}
              </Link>
            ) : null}
          </nav>
        </section>
      </EvidenceScope>
    </div>
  );
}
