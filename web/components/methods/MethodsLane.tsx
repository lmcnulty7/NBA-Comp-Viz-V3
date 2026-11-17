import { DataText, DataValue } from "@/components/results/DataValue";
import { RepoPath } from "@/components/results/Sources";
import { Status } from "@/components/ui/Status";
import { copy, fillNodes, label } from "@/lib/copy";
import type { Benchmark, Experiment, MethodsLane as Lane, Paper } from "@/lib/types/contract";
import styles from "./methods.module.css";
import { ExperimentStatus, FILE, KIND_TONE, Ours, laneName } from "./parts";

const text = copy.methods.lane;

function PaperRow({ paper, field }: { paper: Paper; field: string }) {
  return (
    <li>
      <a href={paper.url} className={styles.cite} data-file={FILE} data-field={`${field}.cite`} data-shown="text">
        {paper.cite}
      </a>
      <details className="sources">
        <summary>{text.take}</summary>
        <div className={`${styles.prose} stack stack-tight`}>
          <DataText file={FILE} field={`${field}.take`} text={paper.take} />
          {paper.differs ? (
            <span>
              <strong className={styles.lead}>{text.differs}</strong>
              {": "}
              <DataText file={FILE} field={`${field}.differs`} text={paper.differs} />
            </span>
          ) : null}
        </div>
      </details>
    </li>
  );
}

function BenchmarkRow({ row, field }: { row: Benchmark; field: string }) {
  return (
    <li data-kind={row.comparability_class} className={styles.bench}>
      <div className={styles.pair}>
        <div className="stack stack-tight">
          <h4 className={styles.side}>{text.ours}</h4>
          {row.ours.length > 0 ? (
            <ul className={styles.values}>
              {row.ours.map((value, index) => (
                <Ours key={`${value.label}-${index}`} value={value} field={`${field}.ours[${index}]`} />
              ))}
            </ul>
          ) : (
            <p className={styles.prose}>
              {row.ours_text ? <DataText file={FILE} field={`${field}.ours_text`} text={row.ours_text} /> : text.noOurs}
            </p>
          )}
        </div>
        <div className="stack stack-tight">
          <h4 className={styles.side}>{text.published}</h4>
          <p className={styles.prose}>
            <DataText file={FILE} field={`${field}.published`} text={row.published} />
          </p>
          <p className="small">
            {text.paper}
            {": "}
            {row.url ? (
              <a href={row.url} data-file={FILE} data-field={`${field}.paper`} data-shown="text">
                {row.paper}
              </a>
            ) : (
              <DataText file={FILE} field={`${field}.paper`} text={row.paper} />
            )}
            {row.where ? (
              <span className="note">
                {text.where}
                {": "}
                <DataText file={FILE} field={`${field}.where`} text={row.where} />
              </span>
            ) : null}
          </p>
        </div>
      </div>
      <div className="stack stack-tight">
        <p className="small">
          <span className={styles.lead}>{text.verdict}</span>{" "}
          <Status tone={KIND_TONE[row.comparability_class]} text={label(copy.methods.kind, row.comparability_class)} />
        </p>
        <q className={styles.quote}>
          <DataText file={FILE} field={`${field}.comparability`} text={row.comparability} />
        </q>
        <details className="sources">
          <summary>{text.rowDiffers}</summary>
          <p className={`${styles.prose} small`}>
            <DataText file={FILE} field={`${field}.differs`} text={row.differs} />
          </p>
        </details>
      </div>
    </li>
  );
}

function ExperimentRow({ experiment, field }: { experiment: Experiment; field: string }) {
  const ran = experiment.status !== "pre_registered";
  return (
    <li id={experiment.id} data-experiment={experiment.id} data-status={experiment.status} className={styles.anchor}>
      <p className="eyebrow">
        <DataText file={FILE} field={`${field}.id`} text={experiment.id} />
      </p>
      <p className="row-title">
        <DataText file={FILE} field={`${field}.title`} text={experiment.title} />
      </p>
      <p>
        <ExperimentStatus experiment={experiment} />
      </p>
      {ran ? (
        <p className="small">
          <a href={`#result-${experiment.id}`}>{copy.methods.results.open}</a>
        </p>
      ) : null}
      <p className="small">
        {fillNodes(text.rank, { rank: <DataValue file={FILE} field={`${field}.rank`} value={experiment.rank} /> })}
        <span className="note">
          {fillNodes(text.score, {
            gain: <DataValue file={FILE} field={`${field}.gain`} value={experiment.gain} />,
            effort: <DataValue file={FILE} field={`${field}.effort`} value={experiment.effort} />,
          })}
        </span>
        <span className="note">
          {fillNodes(text.testData, {
            data: <DataText file={FILE} field={`${field}.test_data`} text={experiment.test_data} />,
          })}
        </span>
      </p>
      <details className="sources">
        <summary>{text.hypothesis}</summary>
        <p className={`${styles.prose} small`}>
          <DataText file={FILE} field={`${field}.hypothesis`} text={experiment.hypothesis} />
        </p>
      </details>
      <details className="sources">
        <summary>{text.threshold}</summary>
        <p className={`${styles.prose} small`}>
          <DataText file={FILE} field={`${field}.threshold`} text={experiment.threshold} />
        </p>
      </details>
    </li>
  );
}

/**
 * One lane of the methods page (PLAN.md D72): the papers it builds on, our numbers next to the published ones
 * with the dossier's own comparability verdict, the dossier's notes, and its pre-registered tests.
 */
export function MethodsLane({ lane, index }: { lane: Lane; index: number }) {
  const path = `lanes[lane=${lane.lane}]`;
  const id = `lane-${lane.lane}`;
  return (
    <section className={`section ${styles.anchor}`} id={id} aria-labelledby={`${id}-title`} data-lane={lane.lane}>
      <div className="column stack stack-tight">
        <h2 id={`${id}-title`}>{laneName(lane.lane)}</h2>
        <p className="small">
          {fillNodes(text.dossier, { path: <RepoPath path={lane.dossier} /> })}
          <span className="note">{fillNodes(text.report, { path: <RepoPath path={lane.report} /> })}</span>
        </p>
      </div>

      <div className="column stack stack-tight">
        <h3>{text.foundations}</h3>
        <ol className={`rows ${styles.papers}`}>
          {lane.foundations.map((paper, at) => (
            <PaperRow key={`${index}-${at}`} paper={paper} field={`${path}.foundations[${at}]`} />
          ))}
        </ol>
      </div>

      <div className="stack stack-tight">
        <h3 className="column">{text.benchmarks}</h3>
        {lane.benchmarks.length > 0 ? (
          <ul className="rows">
            {lane.benchmarks.map((row, at) => (
              <BenchmarkRow key={`${index}-${at}`} row={row} field={`${path}.benchmarks[${at}]`} />
            ))}
          </ul>
        ) : (
          <p className="small column">{text.noBenchmarks}</p>
        )}
        {lane.notes.length > 0 ? (
          <div className="column stack stack-tight">
            <h4 className={styles.side}>{text.notes}</h4>
            <ul className="bullets small">
              {lane.notes.map((note, at) => (
                <li key={`${index}-${at}`}>
                  <DataText file={FILE} field={`${path}.notes[${at}]`} text={note} />
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      <div className="column stack stack-tight">
        <h3>{text.experiments}</h3>
        <ul className="rows">
          {lane.experiments.map((experiment) => (
            <ExperimentRow
              key={experiment.id}
              experiment={experiment}
              field={`${path}.experiments[id=${experiment.id}]`}
            />
          ))}
        </ul>
      </div>
    </section>
  );
}
