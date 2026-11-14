import { Status, type StatusTone } from "@/components/ui/Status";
import { copy, fill, label } from "@/lib/copy";
import type { LaneResult, MetricValue, Results } from "@/lib/types/contract";
import { DataText, DataValue } from "./DataValue";
import { words } from "./ScorecardSection";
import { RepoPath, Sources } from "./Sources";

const text = copy.results.models;
const FILE = "results.json";

export const VERDICT_TONE: Record<LaneResult["verdict_class"], StatusTone> = {
  works: "pass",
  no_signal: "plain",
  failed: "fail",
  inconclusive: "warn",
};

/**
 * A reported metric: its name and value with the interval, or the words "point estimate". A metric the report
 * gives no number for shows the report's own text.
 */
export function Metric({ metric, path }: { metric: MetricValue; path: string }) {
  if (metric.value == null) {
    return <DataText file={FILE} field={`${path}.text`} text={metric.text} />;
  }
  const interval = !metric.point_estimate && metric.lo != null && metric.hi != null;
  return (
    <span className="estimate">
      <span>
        {metric.name ? (
          <>
            <DataText file={FILE} field={`${path}.name`} text={metric.name} />{" "}
          </>
        ) : null}
        <DataValue file={FILE} field={`${path}.value`} value={metric.value} />
      </span>
      <span className="note" data-interval={interval ? "true" : "false"}>
        {interval ? (
          <>
            <DataValue file={FILE} field={`${path}.lo`} value={metric.lo as number} />
            {text.to}
            <DataValue file={FILE} field={`${path}.hi`} value={metric.hi as number} />
          </>
        ) : (
          copy.site.estimate.point
        )}
      </span>
    </span>
  );
}

/**
 * The model results table (C7): each model with its held-out value and interval (or the words point
 * estimate), its verdict class from the generated tables and its verdict quoted from the write-up with the
 * source named. Failed and inconclusive models stay in the table; a model with no result yet is listed with no
 * number.
 */
export function ModelTable({ results }: { results: Results }) {
  return (
    <section className="section" aria-labelledby="models" data-testid="models">
      <div className="column stack stack-tight">
        <h2 id="models">{text.title}</h2>
        <p className="small">{text.intro}</p>
        <p className="small">
          {fill(text.frozen, { tag: results.freeze_tag, when: results.generated })}
        </p>
      </div>
      <div className="table-wrap" role="region" aria-label={text.caption} tabIndex={0}>
        <table>
          <caption className="visually-hidden">{text.caption}</caption>
          <thead>
            <tr>
              <th scope="col">{text.lane}</th>
              <th scope="col">{text.question}</th>
              <th scope="col">{text.heldout}</th>
              <th scope="col">{text.oof}</th>
              <th scope="col">{text.verdict}</th>
            </tr>
          </thead>
          <tbody>
            {results.lanes.map((lane) => {
              const path = `lanes[lane=${lane.lane}]`;
              return (
                <tr key={lane.lane} data-verdict={lane.verdict_class} data-lane={lane.lane}>
                  <th scope="row" className="wrap">
                    {words(lane.lane)}
                    <span className="note">
                      <DataText file={FILE} field={`${path}.selected_model`} text={lane.selected_model} />
                    </span>
                    <span className="note">
                      {text.report} <RepoPath path={lane.report} />
                    </span>
                  </th>
                  <td className="wrap">
                    <DataText file={FILE} field={`${path}.question`} text={lane.question} />
                  </td>
                  <td className="wrap" data-heldout={lane.heldout ? "true" : "false"}>
                    {lane.heldout ? (
                      <Metric metric={lane.heldout} path={`${path}.heldout`} />
                    ) : (
                      <span className="muted">{text.noHeldout}</span>
                    )}
                  </td>
                  <td className="wrap">
                    <Metric metric={lane.out_of_fold} path={`${path}.out_of_fold`} />
                  </td>
                  <td className="wrap">
                    <Status tone={VERDICT_TONE[lane.verdict_class]} text={label(text.verdictClass, lane.verdict_class)} />
                    <q className="note">
                      <DataText file={FILE} field={`${path}.verdict`} text={lane.verdict} />
                    </q>
                    <span className="note">
                      {text.quoted} <RepoPath path={lane.verdict_source} />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="column stack">
        {results.pending.length > 0 ? (
          <div className="stack stack-tight" data-testid="pending">
            <h3>{text.pending.title}</h3>
            <p className="small">{text.pending.intro}</p>
            <ul className="bullets small">
              {results.pending.map((pending, index) => (
                <li key={pending.lane} data-pending={pending.lane}>
                  <strong>{words(pending.lane)}</strong>
                  {": "}
                  <DataText file={FILE} field={`pending[${index}].reason`} text={pending.reason} />
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {results.notes.length > 0 ? (
          <div className="stack stack-tight">
            <h3>{text.notes}</h3>
            <ul className="bullets small">
              {results.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <Sources sources={results.sources} />
      </div>
    </section>
  );
}
