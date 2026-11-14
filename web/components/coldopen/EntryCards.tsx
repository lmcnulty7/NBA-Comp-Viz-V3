import Link from "next/link";
import { Status, type StatusTone } from "@/components/ui/Status";
import { copy, label } from "@/lib/copy";
import { routes } from "@/lib/site";
import type { LaneResult, PlayerView } from "@/lib/types/contract";
import { PossessionPreview } from "./PossessionPreview";
import styles from "./coldopen.module.css";

const open = copy.home.coldOpen.next;

const VERDICT_TONE: Record<LaneResult["verdict_class"], StatusTone> = {
  works: "pass",
  no_signal: "plain",
  failed: "fail",
  inconclusive: "warn",
};

export interface EntryCardsProps {
  /** Steps 6.1 to 6.3, tokens filled. */
  steps: [string, string, string];
  preview: { gameSlug: string; possession: number } | null;
  player: PlayerView | null;
  /** The player counts line, tokens filled. */
  players: string;
  lanes: LaneResult[];
  /** Leave the step lines out (the slide shows them as its own copy). */
  compact?: boolean;
}

/** Scene 6: the three ways in, each a card with a small live preview (docs/SCRIPT.md, scene 6). */
export function EntryCards({ steps, preview, player, players, lanes, compact = false }: EntryCardsProps) {
  return (
    <section className={`wide ${styles.entries}`} aria-labelledby="entries" data-scene="next">
      <h2 id="entries" className="visually-hidden">
        {copy.home.title}
      </h2>
      <ul className={styles.cards}>
        <li className={styles.card} data-card="review">
          <h3 className={styles.cardTitle}>
            <Link href={routes.review}>{open.labels.review}</Link>
          </h3>
          {compact ? null : <p data-testid="coverage">{steps[0]}</p>}
          {preview ? <PossessionPreview gameSlug={preview.gameSlug} possession={preview.possession} /> : null}
        </li>
        <li className={styles.card} data-card="prep">
          <h3 className={styles.cardTitle}>
            <Link href={routes.prep}>{open.labels.prep}</Link>
          </h3>
          {compact ? null : <p>{steps[1]}</p>}
          {player ? (
            <div className={styles.playerCard}>
              <p className={styles.playerName}>{player.name}</p>
              <p className="small muted">{player.teams.join(", ")}</p>
            </div>
          ) : null}
          <p className="small" data-testid="player-counts">
            {players}
          </p>
        </li>
        <li className={styles.card} data-card="results">
          <h3 className={styles.cardTitle}>
            <Link href={routes.results}>{open.labels.results}</Link>
          </h3>
          {compact ? null : <p>{steps[2]}</p>}
          <ul className={styles.lanes}>
            {lanes.map((lane) => (
              <li key={lane.lane} data-verdict={lane.verdict_class}>
                <span>{lane.lane.replace(/_/g, " ")}</span>
                <Status tone={VERDICT_TONE[lane.verdict_class]} text={label(copy.results.models.verdictClass, lane.verdict_class)} />
              </li>
            ))}
          </ul>
        </li>
      </ul>
    </section>
  );
}
