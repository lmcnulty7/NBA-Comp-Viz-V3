import type { Metadata } from "next";
import Link from "next/link";
import { copy, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import { formatCount } from "@/lib/format";
import { routes } from "@/lib/site";

const text = copy.review;

export const metadata: Metadata = { title: text.title };

export default async function ReviewPage() {
  const manifest = await getServerSource().manifest();
  return (
    <div className="column stack stack-loose">
      <header className="page-head">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
      </header>

      <section className="section" aria-labelledby="games">
        <h2 id="games">{text.games.title}</h2>
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
                <tr key={game.slug}>
                  <th scope="row">
                    <Link href={routes.game(game.slug)}>{game.label}</Link>
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
    </div>
  );
}
