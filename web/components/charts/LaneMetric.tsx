import { DataValue } from "@/components/results/DataValue";
import { copy, fill } from "@/lib/copy";
import type { LaneResult, Results } from "@/lib/types/contract";
import styles from "./charts.module.css";

/** The lane of the results table with this key, or null when this data build has none. */
export function findLane(results: Results | null | undefined, lane: string): LaneResult | null {
  return results?.lanes.find((candidate) => candidate.lane === lane) ?? null;
}

/**
 * A model lane's held-out metric with its interval, or the words "point estimate" when the report gives none
 * (PLAN.md D53). The numbers are read from results.json, which copies them from the lane report, and printed as
 * stored, the way the results page prints them (DataValue): no decimals are guessed from the report's text, whose
 * first number can sit in the metric's name ("1.0 s position error (ft) 3.417"). Nothing is typed here. When the
 * data build has no such lane, it says so.
 */
export function LaneMetric({ lane, model }: { lane: LaneResult | null; model: string }) {
  const text = copy.review.metric;
  const held = lane?.heldout ?? null;
  if (!lane || !held || held.value == null) {
    return (
      <p className={styles.metric} data-testid="lane-metric" data-lane-missing="true">
        {fill(text.missing, { model })}
      </p>
    );
  }
  const path = `lanes[lane=${lane.lane}].heldout`;
  const hasInterval = !held.point_estimate && held.lo != null && held.hi != null;
  return (
    <p className={styles.metric} data-testid="lane-metric" data-lane={lane.lane}>
      {fill(text.heldout, { model, name: held.name ?? text.unnamed })}{" "}
      <DataValue file="results.json" field={`${path}.value`} value={held.value} />
      {", "}
      {hasInterval ? (
        <span className="num">
          <DataValue file="results.json" field={`${path}.lo`} value={held.lo as number} />
          {copy.results.models.to}
          <DataValue file="results.json" field={`${path}.hi`} value={held.hi as number} />
        </span>
      ) : (
        <span>{copy.site.estimate.point}</span>
      )}
      {". "}
      <span className={styles.metricSource}>{fill(text.source, { report: lane.report })}</span>
    </p>
  );
}
