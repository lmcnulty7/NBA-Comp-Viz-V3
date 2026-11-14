import Link from "next/link";
import { CourtMotif } from "@/components/court/CourtMotif";
import { copy, fill } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";
import { routes } from "@/lib/site";
import styles from "./home.module.css";

const ENTRIES = [
  { href: routes.review, text: copy.home.entries.review },
  { href: routes.prep, text: copy.home.entries.prep },
  { href: routes.results, text: copy.home.entries.results },
] as const;

export default async function HomePage() {
  const manifest = await getServerSource().manifest();
  return (
    <div className="stack stack-loose">
      <header className={styles.hero}>
        <CourtMotif half className={styles.motif} />
        <div className={styles.head}>
          <p className="eyebrow">{copy.home.eyebrow}</p>
          <h1 className={styles.headline}>{copy.home.headline}</h1>
          <p className="lede">{copy.home.dek}</p>
        </div>
      </header>

      <section className="column section" aria-labelledby="entries">
        <h2 id="entries">{copy.home.entries.title}</h2>
        <ul className="rows">
          {ENTRIES.map((entry) => (
            <li key={entry.href}>
              <h3 className="row-title">
                <Link href={entry.href}>{entry.text.title}</Link>
              </h3>
              <p>{entry.text.body}</p>
            </li>
          ))}
        </ul>
        <p className="small" data-testid="coverage">
          {fill(copy.home.coverage, {
            possessions: formatCount(manifest.n_possessions),
            games: formatCount(manifest.n_games),
            season: manifest.season,
          })}
        </p>
        <p className="small">{copy.home.note}</p>
      </section>
    </div>
  );
}
