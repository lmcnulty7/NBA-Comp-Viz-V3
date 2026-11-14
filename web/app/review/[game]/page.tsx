import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatClock, formatCount, formatNumber, formatProbability } from "@/lib/format";
import { routes } from "@/lib/site";
import type { PossessionSummary } from "@/lib/types/contract";

const text = copy.review;

export const dynamicParams = false;

export async function generateStaticParams() {
  const manifest = await getServerSource().manifest();
  return manifest.games.map((game) => ({ game: game.slug }));
}

async function load(slug: string) {
  const source = getServerSource();
  const manifest = await source.manifest();
  if (!manifest.games.some((game) => game.slug === slug)) notFound();
  return source.possessionIndex(slug);
}

export async function generateMetadata({ params }: PageProps<"/review/[game]">): Promise<Metadata> {
  const { game } = await params;
  const index = await load(game);
  return { title: index.game.label };
}

function flags(row: PossessionSummary): string {
  const parts: string[] = [];
  if (row.featured) parts.push(row.featured_by === "hand" ? text.flags.featuredByHand : text.flags.featured);
  if (row.demo) parts.push(text.flags.demo);
  if (row.gap) parts.push(text.flags.gap);
  if (!row.has_ghosts) parts.push(text.flags.noGhosts);
  return parts.join(", ");
}

export default async function GamePage({ params }: PageProps<"/review/[game]">) {
  const { game } = await params;
  const index = await load(game);
  const featured = index.possessions.filter((row) => row.featured).length;
  const caption = fill(text.game.caption, { game: index.game.label });
  return (
    <div className="wide stack stack-loose">
      <header className="page-head column">
        <nav className="crumbs" aria-label={text.title}>
          <Link href={routes.review}>{text.game.back}</Link>
        </nav>
        <h1>{index.game.label}</h1>
        <p className="lede">
          {fill(text.game.meta, { date: index.game.date, split: label(text.split, index.game.split) })}{" "}
          {fill(text.game.count, { n: formatCount(index.possessions.length), featured: formatCount(featured) })}
        </p>
      </header>

      <section className="section" aria-labelledby="possessions">
        <h2 id="possessions">{text.game.possessions}</h2>
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
                <th scope="col" className="wide-only">
                  {text.game.xfg}
                </th>
                <th scope="col" className="wide-only">
                  {text.game.epv}
                </th>
                <th scope="col" className="wide-only">
                  {text.game.flags}
                </th>
              </tr>
            </thead>
            <tbody>
              {index.possessions.map((row) => (
                <tr key={row.possession}>
                  <th scope="row">
                    {/* A game has a couple of hundred possessions: none is fetched before it is asked for. */}
                    <Link
                      href={routes.possession(index.game.slug, row.possession)}
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
                  <td className="num wide-only">{row.xfg != null ? formatProbability(row.xfg) : null}</td>
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
        {index.notes.length > 0 ? (
          <div className="column stack stack-tight">
            <h3>{text.game.notes}</h3>
            <ul className="bullets small">
              {index.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <div className="column">
          <SourceList sources={index.sources} />
        </div>
      </section>
    </div>
  );
}
