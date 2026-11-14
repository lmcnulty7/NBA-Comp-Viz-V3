import { CourtMotif } from "@/components/court/CourtMotif";
import type { Variant } from "@/lib/site";
import { MediaFrame, type StoryMedia } from "./MediaFrame";
import styles from "./coldopen.module.css";

/**
 * Scene 0 of docs/SCRIPT.md: the headline (D17) over the first picture of the story, dimmed: the paused clip on
 * the private build, the still on the public one. No number is on this view.
 */
export function Hero({ title, intro, media, variant }: { title: string; intro: string[]; media: StoryMedia | null; variant: Variant }) {
  const picture = media && (media.still || (variant === "private" && media.clip)) ? media : null;
  return (
    <header className={styles.hero} data-scene="headline">
      <div className={styles.heroPicture} aria-hidden="true">
        {picture ? (
          <MediaFrame media={picture} boxes={null} moving={variant === "private"} dim />
        ) : (
          <CourtMotif half className={styles.motif} />
        )}
      </div>
      <div className={styles.heroText}>
        <h1 className={styles.headline}>{title}</h1>
        {intro.map((line) => (
          <p key={line} className="lede">
            {line}
          </p>
        ))}
        <span className={styles.scrollCue} aria-hidden="true" />
      </div>
    </header>
  );
}
