import type { Metadata } from "next";
import { ModelTable } from "@/components/results/ModelTable";
import { ScorecardSection } from "@/components/results/ScorecardSection";
import { copy } from "@/lib/copy";
import { getServerSource } from "@/lib/data/server";

const text = copy.results;

export const metadata: Metadata = { title: text.title };

/**
 * The results page (PLAN.md M7): the tracking scorecard against the Stage 1 targets (C5) and every model result
 * (C7), each value printed as its data file stores it, each report linked on GitHub.
 */
export default async function ResultsPage() {
  const source = getServerSource();
  const [scorecard, results] = await Promise.all([source.scorecard(), source.results()]);
  return (
    <div className="wide stack stack-loose">
      <header className="page-head column">
        <p className="eyebrow">{text.eyebrow}</p>
        <h1>{text.title}</h1>
        <p className="lede">{text.intro}</p>
      </header>
      <ScorecardSection scorecard={scorecard} />
      <ModelTable results={results} />
    </div>
  );
}
