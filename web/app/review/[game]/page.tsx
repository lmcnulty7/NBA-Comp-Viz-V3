import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { PossessionList } from "@/components/review/PossessionList";
import { SourceList } from "@/components/ui/SourceList";
import { copy, fill, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";
import { routes } from "@/lib/site";

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

export default async function GamePage({ params }: PageProps<"/review/[game]">) {
  const { game } = await params;
  const index = await load(game);
  const featured = index.possessions.filter((row) => row.featured).length;
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
        <p className="small">{text.game.heldout}</p>
      </header>

      <section className="section" aria-labelledby="possessions">
        <h2 id="possessions">{text.game.possessions}</h2>
        <PossessionList
          gameSlug={index.game.slug}
          gameLabel={index.game.label}
          rows={index.possessions}
          edges={index.xfg_bucket_edges}
        />
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
