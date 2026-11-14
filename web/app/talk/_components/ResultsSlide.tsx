import { Metric, VERDICT_TONE } from "@/components/results/ModelTable";
import { words } from "@/components/results/ScorecardSection";
import { Status } from "@/components/ui/Status";
import { copy, label } from "@/lib/copy";
import type { Results } from "@/lib/types/contract";
import styles from "./talk.module.css";

/** Every model with its held-out value (or the words point estimate) and its verdict, failed ones included. */
export function ResultsSlide({ results }: { results: Results }) {
  const text = copy.results.models;
  return (
    <ul className={styles.lanes} data-testid="results-slide">
      {results.lanes.map((lane) => (
        <li key={lane.lane} data-verdict={lane.verdict_class}>
          <span className={styles.laneHead}>
            <span>{words(lane.lane)}</span>
            <Status tone={VERDICT_TONE[lane.verdict_class]} text={label(text.verdictClass, lane.verdict_class)} />
          </span>
          {lane.heldout ? (
            <Metric metric={lane.heldout} path={`lanes[lane=${lane.lane}].heldout`} />
          ) : (
            <span className="muted">{text.noHeldout}</span>
          )}
        </li>
      ))}
      {results.pending.map((pending) => (
        <li key={pending.lane} data-pending={pending.lane}>
          <span className={styles.laneHead}>
            <span>{words(pending.lane)}</span>
            <span className="muted">{text.pending.title}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
