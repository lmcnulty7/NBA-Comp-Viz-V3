import type { Metadata } from "next";
import { MethodsLane } from "@/components/methods/MethodsLane";
import { ResultsSoFar } from "@/components/methods/ResultsSoFar";
import { KIND_TONE, laneName } from "@/components/methods/parts";
import styles from "@/components/methods/methods.module.css";
import { RepoPath, Sources } from "@/components/results/Sources";
import { Status } from "@/components/ui/Status";
import { copy, fillNodes, label } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";
import type { Benchmark } from "@/lib/types/contract";

const text = copy.methods;
const KINDS: Benchmark["comparability_class"][] = ["roughly", "not_directly", "none"];

export const metadata: Metadata = { title: text.title };

/**
 * The methods and literature page (PLAN.md D72): per lane, the papers it builds on, our numbers next to the
 * published ones with the comparability verdict of the lane dossier, and every pre-registered test with its
 * state. Every number is read from methods.json; `npm run check` holds that no other number reaches the page.
 * The page holds no position of any kind, so it is the same in both variants.
 */
export default async function MethodsPage() {
  const methods = await getServerSource().methods();
  return (
    <div className="wide stack stack-loose" data-testid="methods">
      <header className="page-head column">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
      </header>

      <nav className="column stack stack-tight" aria-labelledby="contents-title">
        <h2 id="contents-title" className="eyebrow">
          {text.contents}
        </h2>
        <ul className={styles.contents}>
          <li>
            <a href="#results-so-far">{text.results.title}</a>
          </li>
          {methods.lanes.map((lane) => (
            <li key={lane.lane}>
              <a href={`#lane-${lane.lane}`}>{laneName(lane.lane)}</a>
            </li>
          ))}
        </ul>
      </nav>

      <section className="section column" aria-labelledby="read-title">
        <h2 id="read-title">{text.read.title}</h2>
        <div className="stack stack-tight">
          {text.read.lines.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
        <div className="stack stack-tight">
          <p className="small">{text.read.kinds}</p>
          <ul className={styles.kinds}>
            {KINDS.map((kind) => (
              <li key={kind}>
                <Status tone={KIND_TONE[kind]} text={label(text.kind, kind)} />
              </li>
            ))}
          </ul>
        </div>
      </section>

      <ResultsSoFar methods={methods} />

      {methods.lanes.map((lane, index) => (
        <MethodsLane key={lane.lane} lane={lane} index={index} />
      ))}

      <div className="column stack">
        <p className="small">{fillNodes(text.backlog, { path: <RepoPath path={methods.backlog} /> })}</p>
        <Sources sources={methods.sources} />
      </div>
    </div>
  );
}
