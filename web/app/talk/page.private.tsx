import type { Metadata } from "next";
import Link from "next/link";
import { getServerSource } from "@/lib/data/server";
import { loadSlide, slideIds, talkCopy } from "@/lib/talk";
import type { TokenData } from "@/lib/tokens";
import styles from "./_components/talk.module.css";

export const metadata: Metadata = { title: talkCopy.title };

/** The talk (PLAN.md M10, D35): the private build only. The list of slides, and the way in. */
export default async function TalkPage() {
  const source = getServerSource();
  const [manifest, scorecard, players, swap, frames] = await Promise.all([
    source.manifest(),
    source.scorecard(),
    source.players(),
    source.demoSwap(),
    source.demoFrames(),
  ]);
  const data: TokenData = { manifest, scorecard, players, swap, frames };
  const slides = slideIds().map((id) => loadSlide(id, data));
  const first = slides[0];
  return (
    <div className={styles.index} data-testid="talk-index">
      <h1>{talkCopy.title}</h1>
      <p className="lede">{talkCopy.intro}</p>
      {first ? (
        <p>
          <Link href={`/talk/${first.id}/`} className={styles.navLink}>
            {talkCopy.start}
          </Link>
        </p>
      ) : null}
      <h2>{talkCopy.list}</h2>
      <ol>
        {slides.map((slide) => (
          <li key={slide.id}>
            <Link href={`/talk/${slide.id}/`}>{slide.title}</Link>
          </li>
        ))}
      </ol>
      <p className="small">{talkCopy.ui.keys}</p>
    </div>
  );
}
