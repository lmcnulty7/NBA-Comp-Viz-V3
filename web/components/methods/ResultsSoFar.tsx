import { DataText } from "@/components/results/DataValue";
import { RepoPath } from "@/components/results/Sources";
import { copy, fillNodes } from "@/lib/copy";
import type { MethodsFile } from "@/lib/types/contract";
import styles from "./methods.module.css";
import { ExperimentStatus, FILE, PartFacts, ProtocolFacts, StageFacts, laneName } from "./parts";

const text = copy.methods.results;

/**
 * Every test that has run, with each stage's paired difference against the bar fixed before it ran. A test that
 * cleared its bar on the training games and waits for its single score on the test games says so.
 */
export function ResultsSoFar({ methods }: { methods: MethodsFile }) {
  const ran = methods.lanes.flatMap((lane) =>
    lane.experiments.filter((experiment) => experiment.status !== "pre_registered").map((experiment) => ({ lane, experiment })),
  );
  return (
    <section className={`section column ${styles.anchor}`} id="results-so-far" aria-labelledby="results-so-far-title">
      <div className="stack stack-tight">
        <h2 id="results-so-far-title">{text.title}</h2>
        <p className="small">{text.intro}</p>
        <p className="small">{text.bootstraps}</p>
      </div>
      {ran.length === 0 ? (
        <p className="small" data-testid="no-results">
          {text.none}
        </p>
      ) : (
        <ul className="rows">
          {ran.map(({ lane, experiment }) => {
            const field = `lanes[lane=${lane.lane}].experiments[id=${experiment.id}]`;
            return (
              <li
                key={experiment.id}
                id={`result-${experiment.id}`}
                className={`stack ${styles.anchor}`}
                data-result={experiment.id}
                data-status={experiment.status}
              >
                <p className="eyebrow">
                  {laneName(lane.lane)}
                  {" / "}
                  <DataText file={FILE} field={`${field}.id`} text={experiment.id} />
                </p>
                <p className="row-title">
                  <a href={`#${experiment.id}`}>
                    <DataText file={FILE} field={`${field}.title`} text={experiment.title} />
                  </a>
                </p>
                <p>
                  <ExperimentStatus experiment={experiment} />
                </p>
                <p className="small">
                  {fillNodes(text.listed, {
                    status: (
                      <DataText file={FILE} field={`${field}.backlog_status`} text={experiment.backlog_status} />
                    ),
                  })}
                  {experiment.writeup ? (
                    <span className="note">
                      {fillNodes(text.writeup, { path: <RepoPath path={experiment.writeup} /> })}
                    </span>
                  ) : null}
                </p>
                {experiment.writeup_notes && experiment.writeup_notes.length > 0 ? (
                  <div className="stack stack-tight" data-testid="writeup-notes">
                    <p className="small">{text.notes}</p>
                    {experiment.writeup_notes.map((note, index) => (
                      <q key={note} className={styles.quote}>
                        <DataText file={FILE} field={`${field}.writeup_notes[${index}]`} text={note} />
                      </q>
                    ))}
                  </div>
                ) : null}
                {experiment.protocol ? <ProtocolFacts protocol={experiment.protocol} field={`${field}.protocol`} /> : null}
                {(experiment.parts ?? []).length > 0 ? <p className="small">{text.parts}</p> : null}
                {(experiment.parts ?? []).map((part, index) => (
                  <PartFacts key={part.label} part={part} field={`${field}.parts[${index}]`} />
                ))}
                {experiment.development ? (
                  <StageFacts stage={experiment.development} field={`${field}.development`} />
                ) : null}
                {experiment.test ? (
                  <StageFacts stage={experiment.test} field={`${field}.test`} />
                ) : experiment.test_not_scored ? (
                  <p className="small" data-testid="test-not-scored">
                    {text.notScored}
                  </p>
                ) : experiment.status === "development_met_test_pending" ? (
                  <p className="small" data-testid="test-pending">
                    {text.pending}
                  </p>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
