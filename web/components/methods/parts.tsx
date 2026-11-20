import { DataText, DataValue } from "@/components/results/DataValue";
import { RepoPath } from "@/components/results/Sources";
import type { StatusTone } from "@/components/ui/Status";
import { copy, fillNodes, label } from "@/lib/copy";
import { numberText } from "@/lib/format";
import type {
  Benchmark,
  BucketGuard,
  Experiment,
  ExperimentPart,
  OurValue,
  PostHocInterval,
  ProtocolResult,
  StageResult,
  StageValue,
} from "@/lib/types/contract";
import { Status } from "@/components/ui/Status";
import styles from "./methods.module.css";

export const FILE = "methods.json";
const text = copy.methods;

export const KIND_TONE: Record<Benchmark["comparability_class"], StatusTone> = {
  roughly: "pass",
  not_directly: "warn",
  none: "plain",
};

export const STATUS_TONE: Record<Experiment["status"], StatusTone> = {
  pre_registered: "plain",
  development_met_test_pending: "warn",
  not_adopted_development: "fail",
  adopted: "pass",
  not_adopted_test: "fail",
  passed_test: "pass",
  adopted_research: "pass",
  ran_result_in_writeup: "plain",
  decided_by_part: "plain",
  adopted_research_not_contradicted: "pass",
  adopted_research_confirmed: "pass",
};

export const PART_TONE: Record<ExperimentPart["status"], StatusTone> = {
  pre_registered: "plain",
  confirmed: "pass",
  development_met_test_not_met: "warn",
  not_adopted_development: "fail",
  development_met_test_pending: "warn",
  passed_test: "pass",
  not_passed_test: "fail",
};

/** The end of the range a comparison reads: the lower end for a bar to clear from below, else the upper end. */
function endOf(comparison: StageResult["comparison"]): "lo" | "hi" {
  return comparison === "above" || comparison === "at_least" || comparison === "not_wholly_above" ? "lo" : "hi";
}

/** The plain name of a lane, or the lane key with its underscores as spaces when copy names no such lane. */
export function laneName(lane: string): string {
  return label(text.lanes, lane).replace(/_/g, " ");
}

/** A number the data file stores as text, printed as stored (a leading hyphen becomes the minus sign). */
export function NumText({ field, value }: { field: string; value: string }) {
  return (
    <span className="num" data-file={FILE} data-field={field} data-shown="number text">
      {numberText(value)}
    </span>
  );
}

/** One number of ours: its label, its value, its interval or the words point estimate, and its report. */
export function Ours({ value, field }: { value: OurValue; field: string }) {
  return (
    <li className={styles.value} data-rows={value.rows}>
      <DataText file={FILE} field={`${field}.label`} text={value.label} />
      <span className="estimate">
        <NumText field={`${field}.value`} value={value.value} />
        <span className="note" data-interval={value.lo != null ? "true" : "false"}>
          {value.lo != null && value.hi != null ? (
            fillNodes(copy.site.estimate.range, {
              lo: <NumText field={`${field}.lo`} value={value.lo} />,
              hi: <NumText field={`${field}.hi`} value={value.hi} />,
            })
          ) : value.kind === "setting" ? (
            text.lane.setting
          ) : (
            text.lane.point
          )}
        </span>
      </span>
      <span className="note">{fillNodes(text.lane.from, { path: <RepoPath path={value.source} /> })}</span>
    </li>
  );
}

function StageNumber({ value, field }: { value: StageValue; field: string }) {
  return (
    <li className={styles.value}>
      <DataText file={FILE} field={`${field}.label`} text={value.label} />
      <span className="estimate">
        <DataValue file={FILE} field={`${field}.value`} value={value.value} signed={!value.point_estimate} />
        <span className="note" data-interval={value.point_estimate ? "false" : "true"}>
          {value.lo != null && value.hi != null
            ? fillNodes(copy.site.estimate.range, {
                lo: <DataValue file={FILE} field={`${field}.lo`} value={value.lo} />,
                hi: <DataValue file={FILE} field={`${field}.hi`} value={value.hi} />,
              })
            : text.lane.point}
        </span>
        {value.reading ? (
          <span className="note" data-testid="value-reading">
            {text.results.reading}
            {": "}
            <DataText file={FILE} field={`${field}.reading`} text={value.reading} />
          </span>
        ) : null}
      </span>
    </li>
  );
}

function GuardFacts({ guard, field, depth }: { guard: BucketGuard; field: string; depth: 3 | 4 }) {
  const r = text.results;
  const Side = depth === 3 ? "h4" : "h5";
  return (
    <div className="stack stack-tight" data-testid="guard" data-guard-met={guard.met ? "true" : "false"}>
      <Side className={styles.side}>{r.guard.title}</Side>
      <p className="small">
        <DataText file={FILE} field={`${field}.label`} text={guard.label} />
        <span className="note">
          {fillNodes(r.guard.line, {
            end: label(r.guard.end, guard.bound),
            comparison: label(r.condition.comparison, guard.comparison),
            threshold: <DataValue file={FILE} field={`${field}.threshold`} value={guard.threshold} />,
          })}
        </span>
      </p>
      <ul className={styles.values}>
        {guard.buckets.map((bucket, index) => {
          const at = `${field}.buckets[${index}]`;
          return (
            <li key={bucket.label} className={styles.value} data-bucket-worse={bucket.worse ? "true" : "false"}>
              <DataText file={FILE} field={`${at}.label`} text={bucket.label} />
              <span className="estimate">
                <DataValue file={FILE} field={`${at}.value`} value={bucket.value} signed />
                <span className="note" data-interval="true">
                  {fillNodes(copy.site.estimate.range, {
                    lo: <DataValue file={FILE} field={`${at}.lo`} value={bucket.lo} />,
                    hi: <DataValue file={FILE} field={`${at}.hi`} value={bucket.hi} />,
                  })}
                </span>
              </span>
              <span className={styles.state}>
                <Status tone={bucket.worse ? "fail" : "plain"} text={bucket.worse ? r.guard.worse : r.guard.notWorse} />
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * One stage of a test that has run: the paired difference (or its range alone, when the file states no point)
 * against its bar, the reading word the file states, the sample, and the files it reads. `depth` is the heading
 * level of the stage: 3 under an experiment, 4 under one part of it.
 */
export function StageFacts({ stage, field, depth = 3 }: { stage: StageResult; field: string; depth?: 3 | 4 }) {
  const r = text.results;
  const Title = depth === 3 ? "h3" : "h4";
  const Side = depth === 3 ? "h4" : "h5";
  const comparison = stage.comparison ?? "at_most";
  const guards = [
    ...(stage.guard ? [{ guard: stage.guard, at: `${field}.guard` }] : []),
    ...(stage.guards ?? []).map((guard, index) => ({ guard, at: `${field}.guards[${index}]` })),
  ];
  const range = fillNodes(copy.site.estimate.range, {
    lo: <DataValue file={FILE} field={`${field}.lo`} value={stage.lo} />,
    hi: <DataValue file={FILE} field={`${field}.hi`} value={stage.hi} />,
  });
  return (
    <div className="stack stack-tight" data-stage={stage.stage} data-met={stage.met ? "true" : "false"}>
      <Title className={styles.stage}>
        {stage.test_set ? (
          <DataText file={FILE} field={`${field}.test_set`} text={stage.test_set} />
        ) : (
          label(r.stage, stage.stage)
        )}
      </Title>
      {stage.design ? (
        <p className="small">
          {r.design}
          {": "}
          <DataText file={FILE} field={`${field}.design`} text={stage.design} />
        </p>
      ) : null}
      <dl className="facts">
        <div>
          <dt>{stage.diff != null ? r.difference : r.rangeOnly}</dt>
          <dd>
            {stage.diff != null ? (
              <span className="estimate">
                <DataValue file={FILE} field={`${field}.diff`} value={stage.diff} signed />
                <span className="note" data-interval="true">
                  {range}
                </span>
              </span>
            ) : (
              <span className="estimate" data-interval="true">
                {range}
              </span>
            )}
          </dd>
        </div>
        <div>
          <dt>{label(r.end, comparison === "contains" ? (stage.upper === stage.lo ? "lo" : "hi") : endOf(comparison))}</dt>
          <dd>
            <DataValue file={FILE} field={`${field}.upper`} value={stage.upper} signed />
          </dd>
        </div>
        <div>
          <dt>{label(r.bar, comparison)}</dt>
          <dd>
            <DataValue file={FILE} field={`${field}.threshold`} value={stage.threshold} signed />
          </dd>
        </div>
        <div>
          <dt>{r.verdict}</dt>
          <dd className={styles.state}>
            <Status
              tone={
                // a confirmation that is not contradicted but not confirmed either is not shown as a pass
                comparison === "not_wholly_above" && stage.met && !(stage.reading ?? "").startsWith("confirms")
                  ? "warn"
                  : stage.met
                    ? "pass"
                    : "fail"
              }
              text={
                comparison === "contains"
                  ? stage.met
                    ? r.holds
                    : r.notHolds
                  : comparison === "not_wholly_above"
                    ? stage.met
                      ? r.notContradicted
                      : r.contradicted
                    : stage.met
                      ? r.met
                      : r.missed
              }
            />
          </dd>
        </div>
        {stage.reading ? (
          <div data-testid="reading">
            <dt>{r.reading}</dt>
            <dd>
              <DataText file={FILE} field={`${field}.reading`} text={stage.reading} />
            </dd>
          </div>
        ) : null}
        <div>
          <dt>{r.sample}</dt>
          <dd>
            <span className={styles.sample}>
              {fillNodes(copy.site.estimate.sample, {
                n: <DataValue file={FILE} field={`${field}.rows`} value={stage.rows} shown="count, as stored" />,
                unit: <DataText file={FILE} field={`${field}.rows_unit`} text={stage.rows_unit} />,
              })}
            </span>
            {stage.games != null ? (
              <span className="note">
                {fillNodes(r.games, {
                  n: <DataValue file={FILE} field={`${field}.games`} value={stage.games} shown="count, as stored" />,
                })}
              </span>
            ) : null}
          </dd>
        </div>
      </dl>
      {(stage.conditions ?? []).length > 0 ? (
        <div className="stack stack-tight" data-testid="conditions">
          <Side className={styles.side}>{r.condition.title}</Side>
          <ul className={styles.values}>
            {(stage.conditions ?? []).map((condition, index) => {
              const at = `${field}.conditions[${index}]`;
              return (
                <li key={condition.label} className={styles.value} data-condition-met={condition.met ? "true" : "false"}>
                  <DataText file={FILE} field={`${at}.label`} text={condition.label} />
                  <span className="estimate">
                    <DataValue file={FILE} field={`${at}.value`} value={condition.value} />
                    {condition.lo != null && condition.hi != null ? (
                      <span className="note" data-interval="true">
                        {fillNodes(copy.site.estimate.range, {
                          lo: <DataValue file={FILE} field={`${at}.lo`} value={condition.lo} />,
                          hi: <DataValue file={FILE} field={`${at}.hi`} value={condition.hi} />,
                        })}
                      </span>
                    ) : null}
                  </span>
                  <span className="note">
                    {fillNodes(r.condition.line, {
                      end: label(r.condition.end, condition.bound),
                      comparison: label(r.condition.comparison, condition.comparison),
                      threshold: <DataValue file={FILE} field={`${at}.threshold`} value={condition.threshold} />,
                    })}
                  </span>
                  <span className={styles.state}>
                    <Status tone={condition.met ? "pass" : "fail"} text={condition.met ? r.met : r.missed} />
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
      {guards.map(({ guard, at }) => (
        <GuardFacts key={at} guard={guard} field={at} depth={depth} />
      ))}
      <p className="small">
        {r.metric}
        {": "}
        <DataText file={FILE} field={`${field}.metric`} text={stage.metric} />
      </p>
      <p className="small">
        {r.method}
        {": "}
        <DataText file={FILE} field={`${field}.interval`} text={stage.interval} />
      </p>
      <p className="small">{fillNodes(r.file, { path: <RepoPath path={stage.source.path} /> })}</p>
      {(stage.also_read ?? []).map((source) => (
        <p key={source.path} className="small" data-testid="also-read">
          {fillNodes(r.alsoRead, { path: <RepoPath path={source.path} /> })}
        </p>
      ))}
      {stage.values.length > 0 ? (
        <details className="sources">
          <summary>{r.more}</summary>
          <ul className={styles.values}>
            {stage.values.map((value, index) => (
              <StageNumber key={value.label} value={value} field={`${field}.values[${index}]`} />
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

/** One part of an experiment decided part by part: its label, its state, and its own stages. */
export function PartFacts({ part, field }: { part: ExperimentPart; field: string }) {
  const r = text.results;
  // a part that did not clear its bar with an inconclusive reading is not shown as a fail
  const inconclusive = part.status === "not_passed_test" && (part.test?.reading ?? "").startsWith("inconclusive");
  return (
    <div className={`stack stack-tight ${styles.part}`} data-part={part.label} data-part-status={part.status}>
      <h3 className={styles.stage}>
        <DataText file={FILE} field={`${field}.label`} text={part.label} />
      </h3>
      <p>
        <span className={styles.state}>
          <Status tone={inconclusive ? "warn" : PART_TONE[part.status]} text={label(r.part, part.status)} />
        </span>
      </p>
      {(part.notes ?? []).length > 0 ? (
        <div className="stack stack-tight" data-testid="part-notes">
          {(part.notes ?? []).map((note, index) => (
            <q key={note} className={styles.quote}>
              <DataText file={FILE} field={`${field}.notes[${index}]`} text={note} />
            </q>
          ))}
        </div>
      ) : null}
      {part.development ? <StageFacts stage={part.development} field={`${field}.development`} depth={4} /> : null}
      {part.test ? (
        <StageFacts stage={part.test} field={`${field}.test`} depth={4} />
      ) : part.test_not_scored ? (
        <p className="small" data-testid="test-not-scored">
          {r.notScored}
        </p>
      ) : part.status === "development_met_test_pending" ? (
        <p className="small" data-testid="test-pending">
          {r.pending}
        </p>
      ) : null}
    </div>
  );
}

/** The plain label of an experiment's state, as a status mark. */
export function ExperimentStatus({ experiment }: { experiment: Experiment }) {
  return (
    <span className={styles.state}>
      <Status tone={STATUS_TONE[experiment.status]} text={label(text.status, experiment.status)} />
    </span>
  );
}

function PostHocNote({ point, interval, field }: { point: boolean; interval: PostHocInterval | null | undefined; field: string }) {
  const p = text.results.protocol;
  return (
    <span className="note" data-interval={interval ? "true" : "false"} data-post-hoc={interval ? "true" : undefined}>
      {interval ? (
        <>
          {fillNodes(copy.site.estimate.range, {
            lo: <DataValue file={FILE} field={`${field}.interval.lo`} value={interval.lo} />,
            hi: <DataValue file={FILE} field={`${field}.interval.hi`} value={interval.hi} />,
          })}
          {", "}
          <em>{p.postHoc}</em>
        </>
      ) : point ? (
        text.lane.point
      ) : null}
    </span>
  );
}

/**
 * The result of a test set up as a protocol of statements (PLAN.md D86): each statement read on its value, with
 * its valid post hoc range or the words point estimate, then the rule to adopt and the plan's own reading. The
 * ranges set up in advance are not valid confidence intervals and are never shown.
 */
export function ProtocolFacts({ protocol, field }: { protocol: ProtocolResult; field: string }) {
  const r = text.results;
  const p = r.protocol;
  const intervalFile = [...protocol.hypotheses.map((h) => h.interval), ...protocol.adoption.map((a) => a.interval)].find(
    (interval) => interval,
  )?.source.path;
  return (
    <div className="stack stack-tight" data-protocol={protocol.stage} data-adopted={protocol.adopted ? "true" : "false"}>
      <h3 className={styles.stage}>{p.stage}</h3>
      <p className="small">{p.intro}</p>
      <p className="small">
        {fillNodes(p.sample, {
          n: <DataValue file={FILE} field={`${field}.rows`} value={protocol.rows} shown="count, as stored" />,
          unit: <DataText file={FILE} field={`${field}.rows_unit`} text={protocol.rows_unit} />,
          games: <DataValue file={FILE} field={`${field}.games`} value={protocol.games} shown="count, as stored" />,
        })}
      </p>
      <h4 className={styles.side}>{p.statements}</h4>
      <ul className={styles.values}>
        {protocol.hypotheses.map((h, index) => {
          const at = `${field}.hypotheses[${index}]`;
          return (
            <li key={h.label} className={styles.value} data-hypothesis={h.label} data-reading={h.result}>
              <span>
                <strong className={styles.lead}>
                  <DataText file={FILE} field={`${at}.label`} text={h.label} />
                </strong>
                {": "}
                <DataText file={FILE} field={`${at}.statement`} text={h.statement} />
              </span>
              <span className="estimate">
                {h.value != null ? <DataValue file={FILE} field={`${at}.value`} value={h.value} /> : null}
                <PostHocNote point={h.point_estimate && h.value != null} interval={h.interval} field={at} />
              </span>
              <span className={styles.state}>
                <Status tone="plain" text={label(p.result, h.result)} />
              </span>
            </li>
          );
        })}
      </ul>
      <h4 className={styles.side}>{p.rule}</h4>
      <p className={`${styles.prose} small`}>
        <DataText file={FILE} field={`${field}.rule`} text={protocol.rule} />
      </p>
      <ul className={styles.values}>
        {protocol.adoption.map((a, index) => {
          const at = `${field}.adoption[${index}]`;
          return (
            <li key={a.label} className={styles.value}>
              <DataText file={FILE} field={`${at}.label`} text={a.label} />
              <span className="estimate">
                <DataValue file={FILE} field={`${at}.value`} value={a.value} />
                <PostHocNote point={a.point_estimate} interval={a.interval} field={at} />
              </span>
            </li>
          );
        })}
      </ul>
      <p>
        <Status tone={protocol.adopted ? "pass" : "fail"} text={protocol.adopted ? p.adopted : p.notAdopted} />
      </p>
      <p className="small">
        <span className={styles.lead}>{p.reading}</span>
        {": "}
        <DataText file={FILE} field={`${field}.reading`} text={protocol.reading} />
      </p>
      {protocol.caveat ? (
        <div className="stack stack-tight">
          <h4 className={styles.side}>{p.caveat}</h4>
          <q className={styles.quote} data-testid="protocol-caveat">
            <DataText file={FILE} field={`${field}.caveat`} text={protocol.caveat} />
          </q>
        </div>
      ) : null}
      {protocol.not_scored.length > 0 ? (
        <div className="stack stack-tight" data-testid="not-scored">
          <h4 className={styles.side}>{p.notScored}</h4>
          <ul className="bullets small">
            {protocol.not_scored.map((sentence, index) => (
              <li key={sentence}>
                <DataText file={FILE} field={`${field}.not_scored[${index}]`} text={sentence} />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="small">{p.postHocNote}</p>
      <p className="small">
        {fillNodes(r.file, { path: <RepoPath path={protocol.source.path} /> })}
        {intervalFile ? <span className="note">{fillNodes(p.intervals, { path: <RepoPath path={intervalFile} /> })}</span> : null}
      </p>
    </div>
  );
}
